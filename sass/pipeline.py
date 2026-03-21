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


def compile_function(fn: Function, verbose: bool = False) -> bytes:
    """
    Compile a single PTX function/kernel to a cubin.

    Returns raw cubin bytes ready for cuModuleLoad.
    """
    # 0. Sink ld.param from entry block to first-use block (reduces GPR pressure)
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
    ctx = ISelContext(
        ra=alloc.ra,
        param_offsets=alloc.param_offsets,
        ur_desc=4,  # UR4 for memory descriptors (ptxas convention)
    )
    body_instrs = select_function(fn, ctx)

    # Insert LDCU UR4 AFTER all S2R instructions in the body.
    # SM_120 requires S2R to be among the first instructions.
    insert_idx = 0
    for idx, si in enumerate(body_instrs):
        opcode = struct.unpack_from('<Q', si.raw, 0)[0] & 0xFFF
        if opcode in (0x919, 0x9c3):  # S2R, S2UR
            insert_idx = idx + 1
    body_instrs.insert(insert_idx, ur4_desc_instr)

    # 4. Schedule: reorder for LDG latency, then assign ctrl via scoreboard
    raw_instrs = preamble + body_instrs
    reordered = schedule(raw_instrs)
    # The preamble instructions (LDC R1, LDCU UR4) have hardcoded ctrl from ptxas.
    # Only assign scoreboard ctrl to body instructions (after preamble).
    n_preamble = len(preamble)
    preamble_instrs = reordered[:n_preamble]
    body_scheduled = assign_ctrl(reordered[n_preamble:])
    sass_instrs = preamble_instrs + body_scheduled

    # 5. BRA offset fixup: resolve branch targets AFTER scheduling
    # (scheduler may insert NOPs that shift instruction positions)
    if hasattr(ctx, '_bra_fixups'):
        # Rebuild label map: scan for BRA placeholder targets in comments
        # and find the actual instruction index after scheduling
        n_total = len(sass_instrs)
        for i in range(n_total):
            comment = sass_instrs[i].comment
            # Labels are emitted as comments like "LABEL: <label_name>"
            # But we don't have label markers in the output. Instead, recalculate
            # offsets by mapping from old body indices to new post-schedule indices.
        # Simple approach: find BRA instructions and recalculate based on scanning
        # for the EXIT or target label pattern
        for bra_idx, target_label in ctx._bra_fixups:
            # Find the BRA instruction in the post-schedule output
            # The bra_idx is body-relative, add preamble offset
            abs_bra_idx = n_preamble + bra_idx
            # Account for scheduler-inserted NOPs before this BRA
            # Count NOPs inserted before bra_idx in the body
            nops_before_bra = 0
            for j in range(n_preamble, abs_bra_idx + nops_before_bra + 1):
                if j < len(sass_instrs) and 'latency' in sass_instrs[j].comment.lower():
                    if j <= abs_bra_idx + nops_before_bra:
                        nops_before_bra += 1
            actual_bra_idx = abs_bra_idx + nops_before_bra

            # Find target: count NOPs before the target label's original position
            if target_label in ctx.label_map:
                orig_target_body_byte = ctx.label_map[target_label]
                orig_target_body_idx = orig_target_body_byte // 16
                abs_target_idx = n_preamble + orig_target_body_idx
                nops_before_target = 0
                for j in range(n_preamble, abs_target_idx + nops_before_target + 1):
                    if j < len(sass_instrs) and 'latency' in sass_instrs[j].comment.lower():
                        if j <= abs_target_idx + nops_before_target:
                            nops_before_target += 1
                actual_target_idx = abs_target_idx + nops_before_target
                actual_target_byte = actual_target_idx * 16
                actual_bra_byte = (actual_bra_idx + 1) * 16
                rel_offset = actual_target_byte - actual_bra_byte
                if actual_bra_idx < len(sass_instrs):
                    # Preserve predicate from original BRA encoding
                    old_raw = sass_instrs[actual_bra_idx].raw
                    old_pred_byte = old_raw[1] & 0xF0
                    new_raw = bytearray(encode_bra(rel_offset))
                    new_raw[1] = (new_raw[1] & 0x0F) | old_pred_byte
                    old_comment = sass_instrs[actual_bra_idx].comment
                    pred_prefix = old_comment.split('BRA')[0] if 'BRA' in old_comment else ''
                    sass_instrs[actual_bra_idx] = SassInstr(
                        bytes(new_raw),
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
    # Compute constant bank size: param_base + sum of all param sizes, rounded up to 4
    total_param_bytes = sum(param_sizes)
    const0_size = ((0x380 + total_param_bytes + 3) // 4) * 4  # 4-byte aligned

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
        num_gprs=alloc.num_gprs,
        num_params=len(fn.params),
        param_sizes=param_sizes,
        param_offsets=alloc.param_offsets,
        const0_size=const0_size,
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
