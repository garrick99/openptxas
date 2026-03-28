"""
sass/pipeline.py — End-to-end PTX → cubin pipeline.

Orchestrates: parse → regalloc → isel → cubin emit.

Usage:
    from sass.pipeline import compile_ptx
    cubin_bytes = compile_ptx("kernel.ptx")
    # or
    cubin_bytes = compile_ptx_source(ptx_string)
"""

from __future__ import annotations
import struct
from pathlib import Path

from ptx.parser import parse, parse_file
from ptx.ir import Module, Function
from ptx.passes.rotate import run as rotate_run
from sass.regalloc import allocate
from sass.isel import ISelContext, select_function, SassInstr
from sass.encoding.sm_120_opcodes import encode_bra, encode_ldcu_64
from sass.schedule import schedule
from sass.scoreboard import assign_ctrl
from cubin.emitter import emit_cubin, KernelDesc
from ptx.ir import RegOp, Instruction


def _sink_param_loads(fn: Function) -> None:
    """Sink ld.param instructions from the entry block to first-use blocks.

    When a frontend (like OpenCUDA) loads all params eagerly in the entry block,
    the register allocator assigns unique GPRs for all of them simultaneously,
    causing high register pressure. Sinking each ld.param to the block where
    its dest is first used reduces the peak live register count.

    Only sinks 64-bit param loads (u64) since those consume register PAIRS.
    32-bit params (used for bounds checks) stay in the entry block.
    """
    if len(fn.blocks) < 2:
        return  # single block — nothing to sink

    entry = fn.blocks[0]

    # Find ld.param.u64 instructions in the entry block whose dests
    # are ONLY used in non-entry blocks (safe to sink).
    # Collect all register names used in the entry block (excluding ld.param dests)
    entry_uses = set()
    for inst in entry.instructions:
        if inst.op == 'ld' and 'param' in inst.types:
            continue  # skip ld.param when collecting uses
        for src in inst.srcs:
            if isinstance(src, RegOp):
                entry_uses.add(src.name)
            elif hasattr(src, 'base') and isinstance(src.base, str):
                entry_uses.add(src.base)

    to_sink = []  # (instruction, dest_name)
    keep = []
    for inst in entry.instructions:
        if (inst.op == 'ld' and 'param' in inst.types and 'u64' in inst.types
                and isinstance(inst.dest, RegOp)
                and inst.dest.name not in entry_uses):  # NOT used in entry block
            to_sink.append((inst, inst.dest.name))
        else:
            keep.append(inst)

    if not to_sink:
        return

    # For each sinkable ld.param, find the first block that uses the dest register
    for inst, dest_name in to_sink:
        sunk = False
        for bb in fn.blocks[1:]:  # skip entry
            for other_inst in bb.instructions:
                # Check if any source operand references this dest
                for src in other_inst.srcs:
                    src_name = None
                    if isinstance(src, RegOp):
                        src_name = src.name
                    elif hasattr(src, 'base'):  # MemOp
                        src_name = src.base if isinstance(src.base, str) else None
                    if src_name == dest_name:
                        # Found first use — insert at the start of this block
                        bb.instructions.insert(0, inst)
                        sunk = True
                        break
                if sunk:
                    break
            if sunk:
                break

        if not sunk:
            # Dest never used in other blocks — keep in entry
            keep.append(inst)

    entry.instructions = keep

    # Second pass: within each block, move sunk ld.param to just before
    # its first consumer (not at the block start). Exception: if the param
    # is used for a store address (the store target), sink it to just before
    # the store instruction to minimize live range of the address register.
    for bb in fn.blocks[1:]:
        ld_params = []
        other = []
        for inst in bb.instructions:
            if inst.op == 'ld' and 'param' in inst.types:
                ld_params.append(inst)
            else:
                other.append(inst)

        if not ld_params:
            continue

        # Rebuild the block: for each non-param instruction, check if any
        # pending ld.param's dest is needed, and insert it just before.
        rebuilt = []
        pending = list(ld_params)
        for inst in other:
            # Check if any pending param's dest is used by this instruction
            needed = []
            still_pending = []
            for lp in pending:
                lp_dest = lp.dest.name if isinstance(lp.dest, RegOp) else None
                used_here = False
                for src in inst.srcs:
                    src_name = src.name if isinstance(src, RegOp) else (
                        src.base if hasattr(src, 'base') and isinstance(src.base, str) else None)
                    if src_name == lp_dest:
                        used_here = True
                        break
                if used_here:
                    needed.append(lp)
                else:
                    still_pending.append(lp)
            rebuilt.extend(needed)
            pending = still_pending
            rebuilt.append(inst)
        # Any remaining pending params go at the end (shouldn't happen)
        rebuilt.extend(pending)
        bb.instructions = rebuilt


def _sink_ldc64_params(instrs: list) -> list:
    """Sink LDC.64 param loads from the top of the block to just before
    their first consumer. This prevents scoreboard slot overcommit when
    multiple LDC.64 instructions are emitted consecutively.

    Only sinks LDC.64 (opcode 0xb82 with b9=0x0a) that are at the start
    of the instruction list (consecutive param loads).
    """
    # Identify leading LDC.64 instructions
    ldc64_instrs = []
    rest_start = 0
    for i, si in enumerate(instrs):
        lo = struct.unpack_from('<Q', si.raw, 0)[0]
        opc = lo & 0xFFF
        if opc == 0xb82 and si.raw[9] == 0x0a:  # LDC.64
            ldc64_instrs.append((i, si, si.raw[2]))  # (index, instr, dest_reg)
        elif opc == 0xb82:  # LDC.32 — also a param load, leave in place
            ldc64_instrs.append((i, si, si.raw[2]))
        else:
            rest_start = i
            break
    else:
        return instrs  # all LDC, nothing to sink

    if len(ldc64_instrs) <= 1:
        return instrs  # 0 or 1 LDC, no overcommit risk

    # Keep the first LDC in place, sink the rest
    result = list(instrs[:rest_start])  # non-sunk LDCs stay (will be reordered below)
    remaining = list(instrs[rest_start:])

    # Actually: just keep all instructions in original order.
    # The key insight: if params are loaded early, just add NOP gaps between them.
    # One NOP between each LDC.64 lets the scoreboard slot clear.
    from sass.encoding.sm_120_opcodes import encode_nop
    from sass.isel import SassInstr

    result = []
    ldc_count = 0
    for si in instrs:
        lo = struct.unpack_from('<Q', si.raw, 0)[0]
        opc = lo & 0xFFF
        if opc == 0xb82:  # LDC or LDC.64
            if ldc_count > 0:
                # Insert NOP gap between consecutive LDC instructions
                result.append(SassInstr(encode_nop(), 'NOP  // LDC slot gap'))
            ldc_count += 1
        else:
            ldc_count = 0  # reset when non-LDC encountered
        result.append(si)

    return result


def _if_convert(fn: Function) -> None:
    """Convert short if-else diamonds to predicated instructions.

    Handles two patterns:

    Pattern A — conditional BRA embedded mid-block (common when front-end emits
    the then-path inline before an unconditional jump to merge):
        block B: ... ; @Px BRA label_else ; then_instrs... ; BRA label_merge
        block E (label_else): else_instrs...  (falls through to merge)
        block M (label_merge): ...
    Converts to:
        block B: ... ; @!Px then_instrs... ; @Px else_instrs...
        block M: ...
    (block E is removed; fall-through from B now goes directly to M)

    Pattern B — separate then/else blocks (traditional diamond):
        block B: ... ; @Px BRA label_else
        block T (fall-through): then_instrs... ; BRA label_merge
        block E (label_else): else_instrs...
        block M (label_merge): ...
    Converts to:
        block B: ... ; @!Px then_instrs... ; @Px else_instrs...
        block M: ...
    (blocks T and E are removed)

    This matches ptxas's behaviour for short divergent branches on SM_120
    where predicated execution is preferred over actual warp divergence.
    """
    from ptx.ir import LabelOp
    import copy

    def _bra_target(instr):
        for s in instr.srcs:
            if isinstance(s, LabelOp):
                return s.name
            if isinstance(s, str):
                return s
        return None

    def _guard(inst_list, pred_n, negated):
        result = []
        for inst in inst_list:
            new_inst = copy.copy(inst)
            new_inst.pred = pred_n
            new_inst.neg = negated
            result.append(new_inst)
        return result

    changed = True
    while changed:
        changed = False
        blocks = fn.blocks
        for i, bb in enumerate(blocks):
            instrs = bb.instructions
            if not instrs:
                continue

            # --- Pattern A: block ends with unconditional BRA (merge jump),
            #     and somewhere inside has a conditional BRA to the else-block ---
            last = instrs[-1]
            if last.op == 'bra' and not last.pred:
                label_merge = _bra_target(last)
                if label_merge is not None:
                    bb_merge = next((b for b in blocks if b.label == label_merge), None)
                    if bb_merge is not None:
                        idx_merge = blocks.index(bb_merge)
                        # Scan for embedded conditional BRA
                        for bra_idx in range(len(instrs) - 1):
                            cond = instrs[bra_idx]
                            if cond.op != 'bra' or not cond.pred:
                                continue
                            label_else = _bra_target(cond)
                            if label_else is None or label_else == label_merge:
                                continue
                            bb_else = next((b for b in blocks if b.label == label_else), None)
                            if bb_else is None:
                                continue
                            idx_else = blocks.index(bb_else)
                            # Else block must immediately precede merge block
                            if idx_else != idx_merge - 1:
                                continue
                            # Else block must fall through (no unconditional BRA at end)
                            if bb_else.instructions:
                                el = bb_else.instructions[-1]
                                if el.op == 'bra' and not el.pred:
                                    continue
                            # Found Pattern A diamond
                            pred_name = cond.pred
                            neg_bra = cond.neg
                            then_instrs = instrs[bra_idx + 1 : -1]
                            else_instrs = list(bb_else.instructions)
                            guarded_then = _guard(then_instrs, pred_name, not neg_bra)
                            guarded_else = _guard(else_instrs, pred_name, neg_bra)
                            bb.instructions = instrs[:bra_idx] + guarded_then + guarded_else
                            fn.blocks = [b for b in blocks if b is not bb_else]
                            changed = True
                            break

            if changed:
                break

            # --- Pattern B: block ends with conditional BRA (fall-through = then-block) ---
            if last.op == 'bra' and last.pred:
                label_else = _bra_target(last)
                if label_else is None:
                    continue
                if i + 1 >= len(blocks):
                    continue
                bb_then = blocks[i + 1]
                if not bb_then.instructions:
                    continue
                last_then = bb_then.instructions[-1]
                if last_then.op != 'bra' or last_then.pred:
                    continue
                label_merge = _bra_target(last_then)
                if label_merge is None:
                    continue
                bb_else = next((b for b in blocks if b.label == label_else), None)
                bb_merge = next((b for b in blocks if b.label == label_merge), None)
                if bb_else is None or bb_merge is None:
                    continue
                idx_else = blocks.index(bb_else)
                idx_merge = blocks.index(bb_merge)
                idx_then = i + 1
                if idx_else != idx_then + 1:
                    continue
                pred_name = last.pred
                neg_bra = last.neg
                then_instrs = bb_then.instructions[:-1]
                else_instrs = list(bb_else.instructions)
                guarded_then = _guard(then_instrs, pred_name, not neg_bra)
                guarded_else = _guard(else_instrs, pred_name, neg_bra)
                bb.instructions = bb.instructions[:-1] + guarded_then + guarded_else
                fn.blocks = [b for b in blocks if b is not bb_then and b is not bb_else]
                changed = True
                break


def compile_function(fn: Function, verbose: bool = False) -> bytes:
    """
    Compile a single PTX function/kernel to a cubin.

    Returns raw cubin bytes ready for cuModuleLoad.
    """
    # 0a. If-conversion: convert short if-else diamonds to predicated instructions,
    # matching ptxas behaviour for divergent branches on SM_120.
    _if_convert(fn)

    # 0b. Sink ld.param from entry block to first-use block (reduces GPR pressure)
    _sink_param_loads(fn)

    # 1. Register allocation
    alloc = allocate(fn)

    if verbose:
        print(f"[pipeline] {fn.name}: {alloc.num_gprs} GPRs, "
              f"{len(fn.params)} params")
        for p in fn.params:
            off = alloc.param_offsets.get(p.name, -1)
            print(f"  param {p.name}: c[0][0x{off:x}]")

    # 2. Emit kernel preamble — use EXACT bytes from ptxas for the setup
    # instructions that configure frame pointer and memory descriptors.
    # These ctrl/barrier values are critical for correct GPU execution.
    preamble = [
        # LDC R1, c[0][0x37c] — frame pointer (first instruction)
        SassInstr(bytes.fromhex('827b01ff00df00000008000000e20f00'),
                  'LDC R1, c[0][0x37c]  // frame ptr'),
    ]

    ur4_desc_instr = SassInstr(
        encode_ldcu_64(4, 0, 0x358, ctrl=0x717),
        'LDCU.64 UR4, c[0][0x358]  // mem desc')

    # 3. Instruction selection
    # Compute literal pool base: after the last param in c[0], 4-byte aligned.
    if alloc.param_offsets:
        # Build a name→size map from fn.params, then find the highest param end.
        param_size_map = {p.name: (8 if p.type.width >= 64 else 4) for p in fn.params}
        last_param_end = max(
            off + param_size_map.get(name, 4)
            for name, off in alloc.param_offsets.items()
        )
    else:
        last_param_end = 0x380
    lit_pool_base = (last_param_end + 3) & ~3  # 4-byte align

    ctx = ISelContext(
        ra=alloc.ra,
        param_offsets=alloc.param_offsets,
        ur_desc=4,  # UR4 for memory descriptors (ptxas convention)
        _const_pool_base=lit_pool_base,
        _next_gpr=alloc.num_gprs,
    )
    body_instrs = select_function(fn, ctx)

    # SM_120 requires at least one S2R before LDCU param loads.
    # If the body has no S2R, insert a dummy S2R R0, SR_TID.X.
    from sass.encoding.sm_120_opcodes import encode_s2r, SR_TID_X
    has_s2r = any(
        struct.unpack_from('<Q', si.raw, 0)[0] & 0xFFF in (0x919, 0x9c3)
        for si in body_instrs
    )
    if not has_s2r:
        body_instrs.insert(0, SassInstr(encode_s2r(0, SR_TID_X),
                                         'S2R R0, SR_TID.X  // required for LDCU init'))

    # Insert LDCU.64 UR4 (mem descriptor) right after the bounds-check predicated EXIT.
    # SM_120 requires ≥3-instruction gap between LDCU.64 and its first UR consumer
    # (LDG/STG/IADD.64-UR/ISETP-RU that read UR at byte[4]).  Placing UR4 immediately
    # after the EXIT maximises the gap.  For kernels without a predicated EXIT,
    # fall back to just after the last S2R/S2UR in the preamble.
    #
    # Predicated EXIT: opcode=0x94d with guard nibble ≠ 0x7 (not @PT).
    insert_idx = 0
    for idx, si in enumerate(body_instrs):
        opcode = struct.unpack_from('<Q', si.raw, 0)[0] & 0xFFF
        guard_nibble = (si.raw[1] >> 4) & 0xF
        is_predicated = guard_nibble != 0x7

        if opcode == 0x94d and is_predicated:  # predicated EXIT (bounds check)
            # Insert UR4 right after the EXIT so it has maximum instruction gap
            # before any LDG/STG that uses UR4 as a descriptor.  SM_120 requires
            # ≥3-instruction gap between LDCU.64 and its first UR consumer; placing
            # UR4 after the bounds-check EXIT guarantees this gap for all kernels.
            insert_idx = idx + 1
            break
        elif opcode in (0x919, 0x9c3):  # S2R/S2UR fallback (no predicated EXIT)
            insert_idx = idx + 1
    body_instrs.insert(insert_idx, ur4_desc_instr)

    # Update ctx.label_map and _bra_fixups to reflect the UR4 insertion.
    # Any label or BRA index that was at or after insert_idx has shifted by 1.
    for label in list(ctx.label_map):
        body_byte = ctx.label_map[label]
        body_idx = body_byte // 16
        if body_idx >= insert_idx:
            ctx.label_map[label] = body_byte + 16
    if hasattr(ctx, '_bra_fixups'):
        ctx._bra_fixups = [
            (bra_idx + 1 if bra_idx >= insert_idx else bra_idx, target_label)
            for bra_idx, target_label in ctx._bra_fixups
        ]

    # 4. Schedule: reorder for LDG latency, then assign ctrl via scoreboard
    raw_instrs = preamble + body_instrs
    reordered = schedule(raw_instrs)
    # The preamble (LDC R1) has hardcoded ctrl from ptxas.
    # Only assign scoreboard ctrl to body instructions (after preamble).
    n_preamble = len(preamble)
    preamble_instrs = reordered[:n_preamble]
    body_scheduled = assign_ctrl(reordered[n_preamble:])
    sass_instrs = preamble_instrs + body_scheduled

    # 5. BRA offset fixup: resolve branch targets AFTER scheduling.
    # ctx.label_map and ctx._bra_fixups have been updated for UR4 insertion
    # (step above). Here we also account for latency NOPs inserted by schedule().
    #
    # Strategy: map body instruction indices to final sass_instrs positions by
    # iterating the stream and counting non-latency-NOP instructions. Latency
    # NOPs have 'latency' in their comment and are transparent to label positions.
    if hasattr(ctx, '_bra_fixups'):

        def _body_idx_to_abs(sass_instrs, n_preamble, body_idx):
            """Return the abs sass_instrs index for the n-th body instruction
            (0-based, after preamble), skipping scheduler-inserted latency NOPs."""
            count = 0
            for j in range(n_preamble, len(sass_instrs)):
                if 'latency' in sass_instrs[j].comment.lower():
                    continue
                if count == body_idx:
                    return j
                count += 1
            return n_preamble + body_idx  # fallback

        for bra_idx, target_label in ctx._bra_fixups:
            # bra_idx is body-relative (after preamble, after UR4 update).
            actual_bra_idx = _body_idx_to_abs(sass_instrs, n_preamble, bra_idx)

            if target_label not in ctx.label_map:
                continue

            # Check if target block is a ret-only block (maps to EXIT).
            target_is_exit = any(
                tbb.label == target_label
                and len(tbb.instructions) == 1
                and tbb.instructions[0].op == 'ret'
                for tbb in fn.blocks
            )

            if target_is_exit:
                # Scan backward for the last unconditional EXIT.
                actual_target_byte = None
                for si_idx in range(len(sass_instrs) - 1, -1, -1):
                    if sass_instrs[si_idx].raw[:2] == bytes([0x4d, 0x79]):
                        actual_target_byte = si_idx * 16
                        break
            else:
                # body_idx of target (post-UR4, label_map is updated).
                target_body_idx = ctx.label_map[target_label] // 16
                actual_target_abs = _body_idx_to_abs(
                    sass_instrs, n_preamble, target_body_idx)
                actual_target_byte = actual_target_abs * 16

            if actual_target_byte is None:
                continue

            actual_bra_byte = (actual_bra_idx + 1) * 16
            rel_offset = actual_target_byte - actual_bra_byte

            if actual_bra_idx < len(sass_instrs):
                # Patch BRA offset in-place, preserving predicate and ctrl.
                old_raw = bytearray(sass_instrs[actual_bra_idx].raw)
                signed_insns = rel_offset // 16
                offset18 = signed_insns & 0x3FFFF
                old_raw[8]  = offset18 & 0xFF
                old_raw[9]  = (offset18 >> 8) & 0xFF
                old_raw[10] = 0x80 | ((offset18 >> 16) & 0x03)
                old_raw[11] = 0x03
                old_comment = sass_instrs[actual_bra_idx].comment
                pred_prefix = old_comment.split('BRA')[0] if 'BRA' in old_comment else ''
                sass_instrs[actual_bra_idx] = SassInstr(
                    bytes(old_raw),
                    f'{pred_prefix}BRA {target_label} (offset={rel_offset})')

    if verbose:
        print(f"[pipeline] {len(sass_instrs)} SASS instructions:")
        for i, si in enumerate(sass_instrs):
            print(f"  +{i*16:4d}: {si.hex()}  // {si.comment}")

    # 3. Concatenate SASS bytes
    sass_bytes = b''.join(si.raw for si in sass_instrs)

    # 4. Build param size list
    param_sizes = []
    for p in fn.params:
        if p.type.width >= 64:
            param_sizes.append(8)
        elif p.type.width >= 32:
            param_sizes.append(4)
        else:
            param_sizes.append(p.type.width // 8)

    # 5. Emit cubin
    # Compute constant bank size: param area + literal pool
    total_param_bytes = sum(param_sizes)
    param_area_end = ((0x380 + total_param_bytes + 3) // 4) * 4
    # Add literal pool entries (4 bytes each)
    lit_pool_bytes = len(ctx._const_pool) * 4
    const0_size = ((param_area_end + lit_pool_bytes + 3) // 4) * 4

    # Build const0 init data: zeros throughout, literal values at pool offsets
    const0_init: bytearray | None = None
    if ctx._const_pool:
        import struct as _struct
        const0_init = bytearray(const0_size)
        for value, offset in ctx._const_pool.items():
            _struct.pack_into('<I', const0_init, offset, value & 0xFFFFFFFF)
        const0_init = bytes(const0_init)

    # Find EXIT and S2R instruction offsets in the SASS byte stream
    exit_offset = 0
    s2r_offset = 0x10  # default
    for i in range(0, len(sass_bytes), 16):
        if sass_bytes[i:i+2] == bytes([0x4d, 0x79]):  # EXIT opcode
            exit_offset = i
            break
    # Find S2R that reads CTAID.X (SR code 0x25 in byte 9)
    # Falls back to any S2R if no CTAID S2R is found
    for i in range(0, len(sass_bytes), 16):
        if sass_bytes[i:i+2] == bytes([0x19, 0x79]) and sass_bytes[i+9] == 0x25:
            s2r_offset = i
            break
    else:
        # No CTAID S2R — find first S2R of any type
        for i in range(0, len(sass_bytes), 16):
            if sass_bytes[i:i+2] == bytes([0x19, 0x79]):
                s2r_offset = i
                break

    desc = KernelDesc(
        name=fn.name,
        sass_bytes=sass_bytes,
        num_gprs=max(alloc.num_gprs, ctx._next_gpr),
        num_params=len(fn.params),
        param_sizes=param_sizes,
        param_offsets=alloc.param_offsets,
        const0_size=const0_size,
        const0_init_data=const0_init,
        exit_offset=exit_offset,
        s2r_offset=s2r_offset,
    )
    return emit_cubin(desc)


def compile_ptx_source(ptx_src: str, verbose: bool = False) -> dict[str, bytes]:
    """
    Compile PTX source text. Returns {kernel_name: cubin_bytes} for each kernel.
    """
    mod = parse(ptx_src)
    mod, rotate_groups = rotate_run(mod)

    results = {}
    for fn in mod.functions:
        if fn.is_kernel:
            results[fn.name] = compile_function(fn, verbose=verbose)

    return results


def compile_ptx(ptx_path: str, verbose: bool = False) -> dict[str, bytes]:
    """
    Compile a PTX file. Returns {kernel_name: cubin_bytes} for each kernel.
    """
    src = Path(ptx_path).read_text(encoding='utf-8')
    return compile_ptx_source(src, verbose=verbose)
