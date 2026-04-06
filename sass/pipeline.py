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
from sass.encoding.sm_120_opcodes import encode_bra, encode_ldcu_64, encode_exit, encode_nop
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

    # For each sinkable ld.param, find the first block that uses the dest register.
    # Skip labeled blocks (loop headers / branch targets): sinking into them causes
    # the param load to re-execute on every loop iteration.  When a LDCU.64 becomes
    # the first instruction of a labeled block, _hoist_ldcu64 moves it (and the
    # block label) to before all entry-block initializations, creating an infinite
    # loop because the loop back-edge targets the hoisted position instead of the
    # true loop start.
    for inst, dest_name in to_sink:
        sunk = False
        for bb in fn.blocks[1:]:  # skip entry
            if bb.label is not None:
                continue  # never sink into labeled (loop-header / branch-target) blocks
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
            # Dest never used in other blocks — keep in entry.
            # Insert at the START so it stays above any conditional BRA
            # that the entry block may contain.  Appending to the END
            # would place it AFTER the BRA, making it unreachable dead code.
            keep.insert(0, inst)

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

    def _has_inner_predicates(inst_list):
        """Check if any instruction already has a predicate (from inner if-conversion)."""
        return any(inst.pred for inst in inst_list)

    def _overwrites_pred(inst_list, pred_name):
        """Check if any instruction overwrites the guard predicate register.

        If a setp inside the guarded block writes to the same predicate
        that the guard uses, if-conversion is unsafe: the predicated
        instructions after the setp would use the NEW predicate value
        (from the comparison result) instead of the ORIGINAL guard value.
        """
        for inst in inst_list:
            if inst.op == 'setp' and isinstance(inst.dest, RegOp):
                if inst.dest.name == pred_name:
                    return True
        return False

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
                            # Skip if body already has inner predicates (nested if-conversion)
                            if (_has_inner_predicates(then_instrs) or _has_inner_predicates(else_instrs)
                                    or _overwrites_pred(then_instrs, pred_name)
                                    or _overwrites_pred(else_instrs, pred_name)):
                                continue
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
                # Skip if body already has inner predicates (nested if-conversion)
                if (_has_inner_predicates(then_instrs) or _has_inner_predicates(else_instrs)
                                    or _overwrites_pred(then_instrs, pred_name)
                                    or _overwrites_pred(else_instrs, pred_name)):
                    continue
                guarded_then = _guard(then_instrs, pred_name, not neg_bra)
                guarded_else = _guard(else_instrs, pred_name, neg_bra)
                bb.instructions = bb.instructions[:-1] + guarded_then + guarded_else
                fn.blocks = [b for b in blocks if b is not bb_then and b is not bb_else]
                changed = True
                break

            if changed:
                break

            # --- Pattern D: early-exit for ret-only false path ---
            # Block ends with: @Px BRA label_true ; BRA label_false
            # true_block = body ; false_block = ret-only
            # Converts to: @!Px ret (early exit) ; body (unpredicated)
            # This avoids predicated body instructions that conflict with
            # IADD3.cb carry output clobbering the execution predicate.
            if (len(instrs) >= 2
                    and last.op == 'bra' and not last.pred
                    and instrs[-2].op == 'bra' and instrs[-2].pred):
                cond_bra_d = instrs[-2]
                label_true_d = _bra_target(cond_bra_d)
                label_false_d = _bra_target(last)
                if label_true_d and label_false_d:
                    bb_true_d = next((b for b in blocks if b.label == label_true_d), None)
                    bb_false_d = next((b for b in blocks if b.label == label_false_d), None)
                    if (bb_true_d and bb_false_d
                            and len(bb_false_d.instructions) <= 1
                            and (not bb_false_d.instructions
                                 or bb_false_d.instructions[0].op == 'ret')):
                        # False path is ret-only → emit @!pred ret + unpredicated body
                        pred_name_d = cond_bra_d.pred
                        neg_d = cond_bra_d.neg
                        # @!pred ret (early exit for inactive threads)
                        exit_inst = copy.copy(bb_false_d.instructions[0] if bb_false_d.instructions
                                              else Instruction(op='ret', dest=None, srcs=[]))
                        exit_inst.pred = pred_name_d
                        exit_inst.neg = not neg_d  # invert: @pred bra true → @!pred ret
                        # Unpredicated body from true block (strip trailing bra)
                        true_body_d = bb_true_d.instructions[:-1] if (
                            bb_true_d.instructions and bb_true_d.instructions[-1].op == 'bra'
                            and not bb_true_d.instructions[-1].pred) else list(bb_true_d.instructions)
                        bb.instructions = instrs[:-2] + [exit_inst] + true_body_d
                        # Append merge block body if the true block's bra targets a merge
                        if (bb_true_d.instructions and bb_true_d.instructions[-1].op == 'bra'):
                            merge_label = _bra_target(bb_true_d.instructions[-1])
                            bb_merge_d = next((b for b in blocks if b.label == merge_label), None)
                            if bb_merge_d:
                                bb.instructions += list(bb_merge_d.instructions)
                                # If the merge block originally fell through (no
                                # terminator), its successor was the next block in
                                # the original layout.  When that successor is the
                                # ret-only false block (already absorbed as early
                                # exit), append an explicit ret so threads don't
                                # fall through into whatever block follows.
                                _merge_last = bb_merge_d.instructions[-1] if bb_merge_d.instructions else None
                                if _merge_last and _merge_last.op not in ('bra', 'ret'):
                                    _merge_idx = blocks.index(bb_merge_d)
                                    _merge_succ = blocks[_merge_idx + 1] if _merge_idx + 1 < len(blocks) else None
                                    if _merge_succ is bb_false_d or _merge_succ is None:
                                        bb.instructions.append(Instruction(op='ret', dest=None, srcs=[]))
                                fn.blocks = [b for b in blocks
                                             if b not in (bb_true_d, bb_false_d, bb_merge_d)]
                            else:
                                fn.blocks = [b for b in blocks
                                             if b not in (bb_true_d, bb_false_d)]
                        else:
                            fn.blocks = [b for b in blocks
                                         if b not in (bb_true_d, bb_false_d)]
                        # Replace any remaining BRA <label_false_d> in surviving
                        # blocks with ret, since the ret-only false block was removed.
                        for _rb in fn.blocks:
                            for _ri, _rinst in enumerate(_rb.instructions):
                                if (_rinst.op == 'bra' and not _rinst.pred
                                        and _bra_target(_rinst) == label_false_d):
                                    _rb.instructions[_ri] = Instruction(
                                        op='ret', dest=None, srcs=[])
                        changed = True
                        break

            # --- Pattern C: two-way branch where both targets jump to same merge ---
            # Block ends with: @Px BRA label_true ; BRA label_false
            # true_block:  instrs... ; BRA label_merge
            # false_block: instrs... ; BRA label_merge  (or falls through to merge)
            # Converts to: instrs_before ; @Px true_instrs ; @!Px false_instrs
            if (len(instrs) >= 2
                    and last.op == 'bra' and not last.pred
                    and instrs[-2].op == 'bra' and instrs[-2].pred):
                cond_bra = instrs[-2]
                label_true = _bra_target(cond_bra)
                label_false = _bra_target(last)
                if label_true is not None and label_false is not None:
                    bb_true = next((b for b in blocks if b.label == label_true), None)
                    bb_false = next((b for b in blocks if b.label == label_false), None)
                    if bb_true is not None and bb_false is not None:
                        # Both blocks must end with BRA to the same merge label
                        if (bb_true.instructions and bb_false.instructions
                                and bb_true.instructions[-1].op == 'bra'
                                and not bb_true.instructions[-1].pred):
                            label_merge_t = _bra_target(bb_true.instructions[-1])
                            # False block can end with BRA merge or fall through
                            if (bb_false.instructions[-1].op == 'bra'
                                    and not bb_false.instructions[-1].pred):
                                label_merge_f = _bra_target(bb_false.instructions[-1])
                                false_body = bb_false.instructions[:-1]
                            else:
                                # Fall-through: merge is the block after false
                                idx_false = blocks.index(bb_false)
                                if idx_false + 1 < len(blocks):
                                    label_merge_f = blocks[idx_false + 1].label
                                else:
                                    label_merge_f = None
                                false_body = list(bb_false.instructions)
                            if (label_merge_t is not None
                                    and label_merge_t == label_merge_f):
                                pred_name = cond_bra.pred
                                neg_bra = cond_bra.neg
                                true_body = bb_true.instructions[:-1]
                                # Skip if body already has inner predicates
                                if (_has_inner_predicates(true_body)
                                        or _has_inner_predicates(false_body)
                                        or _overwrites_pred(true_body, pred_name)
                                        or _overwrites_pred(false_body, pred_name)):
                                    continue
                                guarded_true = _guard(true_body, pred_name, neg_bra)
                                guarded_false = _guard(false_body, pred_name, not neg_bra)
                                # Merge the merge block's body into this block
                                # (avoids a separate block that causes scheduler
                                # reordering issues with LDCU/LDG/FSEL ordering).
                                bb_merge = next((b for b in blocks
                                                 if b.label == label_merge_t), None)
                                if bb_merge is not None:
                                    # Include merge block body (everything except terminal BRA)
                                    merge_body = list(bb_merge.instructions)
                                    bb.instructions = (instrs[:-2]
                                                       + guarded_true + guarded_false
                                                       + merge_body)
                                    fn.blocks = [b for b in blocks
                                                 if b not in (bb_true, bb_false, bb_merge)]
                                else:
                                    merge_bra = Instruction(
                                        op='bra', dest=None,
                                        srcs=[LabelOp(name=label_merge_t)])
                                    bb.instructions = (instrs[:-2]
                                                       + guarded_true + guarded_false
                                                       + [merge_bra])
                                    fn.blocks = [b for b in blocks
                                                 if b is not bb_true and b is not bb_false]
                                changed = True
                                break


def compile_function(fn: Function, verbose: bool = False,
                     ptxas_meta: dict = None, sm_version: int = 120) -> bytes:
    """
    Compile a single PTX function/kernel to a cubin.

    Returns raw cubin bytes ready for cuModuleLoad.
    ptxas_meta: optional {'capmerc': bytes, 'merc_info': bytes} from ptxas.
    """
    # 0a. If-conversion: convert short if-else diamonds to predicated instructions,
    # matching ptxas behaviour for divergent branches on SM_120.
    _if_convert(fn)

    # 0b. Sink ld.param from entry block to first-use block (reduces GPR pressure)
    _sink_param_loads(fn)

    # 1. Register allocation
    from sass.regalloc import PARAM_BASE_SM120, PARAM_BASE_SM89
    param_base = PARAM_BASE_SM89 if sm_version == 89 else PARAM_BASE_SM120
    has_capmerc = ptxas_meta is not None and 'capmerc' in (ptxas_meta or {})
    alloc = allocate(fn, param_base=param_base, has_capmerc=has_capmerc,
                     sm_version=sm_version)

    if verbose:
        print(f"[pipeline] {fn.name}: {alloc.num_gprs} GPRs, "
              f"{len(fn.params)} params")
        for p in fn.params:
            off = alloc.param_offsets.get(p.name, -1)
            print(f"  param {p.name}: c[0][0x{off:x}]")

    # 2. Emit kernel preamble — architecture-specific
    if sm_version == 89:
        # SM_89: IMAD.MOV.U32 R1, RZ, RZ, c[0][0x28] — frame pointer
        from sass.encoding.sm_89_opcodes import (
            encode_imad_mov_u32_cbuf as sm89_imad_mov,
            encode_uldc_64 as sm89_uldc64,
        )
        preamble = [
            SassInstr(sm89_imad_mov(1, 0, 0x28),
                      'IMAD.MOV.U32 R1, RZ, RZ, c[0][0x28]  // frame ptr'),
        ]
        # SM_89 descriptor: ULDC.64 UR4, c[0][0x118]
        ur4_desc_instr = SassInstr(
            sm89_uldc64(4, 0, 0x118),
            'ULDC.64 UR4, c[0][0x118]  // mem desc')
    else:
        # SM_120: LDC R1, c[0][0x37c] — frame pointer (first instruction)
        preamble = [
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
        last_param_end = param_base
    lit_pool_base = (last_param_end + 3) & ~3  # 4-byte align

    # Reserve a dedicated even-aligned scratch pair for UR→GPR address
    # materializations (ld.global / st.global with pointer-only params).
    # This pair is reused across all address materializations without going
    # through the scratch pool, preventing register pressure from growing when
    # the pool contains only odd-indexed registers.
    _addr_scratch_base = alloc.num_gprs
    if _addr_scratch_base % 2 != 0:
        _addr_scratch_base += 1  # align to even

    ctx = ISelContext(
        ra=alloc.ra,
        param_offsets=alloc.param_offsets,
        ur_desc=4,  # UR4 for memory descriptors (ptxas convention)
        _const_pool_base=lit_pool_base,
        _next_gpr=_addr_scratch_base + 2,  # pool-based scratch starts above addr pair
        _next_pred=alloc.num_pred,
        sm_version=sm_version,
    )
    ctx._addr_scratch_lo = _addr_scratch_base  # dedicated addr pair: R(base):R(base+1)
    # Both SM_89 and SM_120 now have full register range available.
    # SM_89: no capmerc DRM.
    # SM_120: capmerc byte[10] fix (0x81→0x01, 0xc1→0x01) unlocks R12+.
    # Verified 2026-04-01 (commit 8d516ca).
    ctx._gpr_limit = 255

    # Compute shared memory variable offsets for isel
    ctx._smem_offsets = {}
    if hasattr(fn, 'shared_decls') and fn.shared_decls:
        offset = 0
        for sd in fn.shared_decls:
            offset = (offset + sd.align - 1) & ~(sd.align - 1)
            ctx._smem_offsets[sd.name] = offset
            offset += sd.size

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

    # Insert ULDC/LDCU descriptor load after the bounds-check predicated EXIT.
    # Both SM_89 and SM_120 use UR4 for the memory descriptor; only the
    # encoding and constant bank offset differ (handled above).
    #
    # Predicated EXIT: opcode=0x94d with guard nibble ≠ 0x7 (not @PT).
    insert_idx = 0
    for idx, si in enumerate(body_instrs):
        opcode = struct.unpack_from('<Q', si.raw, 0)[0] & 0xFFF
        guard_nibble = (si.raw[1] >> 4) & 0xF
        is_predicated = guard_nibble != 0x7

        if opcode == 0x94d and is_predicated:  # predicated EXIT (bounds check)
            insert_idx = idx + 1
            break
        elif opcode in (0x919, 0x9c3):  # S2R/S2UR fallback (no predicated EXIT)
            insert_idx = idx + 1
    body_instrs.insert(insert_idx, ur4_desc_instr)

    # Update ctx.label_map and _bra_fixups to reflect the UR4 insertion.
    # Labels at or after insert_idx shift by 1 instruction (16 bytes).
    # EXCEPTION: the first body label (BRA target from bounds check) should
    # NOT be shifted — the BRA should land ON the LDCU.64 so active threads
    # execute the descriptor load. Without this, BRA jumps past LDCU.64 and
    # all LDG/STG use an uninitialized descriptor → crash.
    first_body_label = None
    if hasattr(ctx, '_bra_fixups') and ctx._bra_fixups:
        # Find the first BRA that goes to a post-boundary label
        for _, target in ctx._bra_fixups:
            tgt_byte = ctx.label_map.get(target, 0)
            if tgt_byte // 16 >= insert_idx:
                first_body_label = target
                break

    for label in list(ctx.label_map):
        if label == first_body_label:
            continue  # Don't shift — BRA should land on LDCU.64
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
    # BRA fixup: resolve branch targets in the FINAL instruction stream.
    # The scheduler may reorder instructions, so body-relative indices are unreliable.
    # Instead, scan the final stream for BRA instructions and match by comment,
    # then find target labels by scanning for label-bearing instructions.
    #
    # Step 1: Find where each label lands in the final stream.
    # Labels were set as body-relative bytes. Map them to absolute positions
    # by scanning the stream for the instruction at that body position.
    if hasattr(ctx, '_bra_fixups') and ctx._bra_fixups:
        # Find label positions in the final stream by scanning instruction comments.
        # Labels are embedded as "LABEL:xxx" in the comment of the first instruction
        # of each block (set by the isel when it records the label).
        # Since the scheduler may reorder and insert NOPs, body-relative byte
        # offsets from label_map are unreliable. Instead, search for label markers.
        label_abs_byte = {}
        for j, si in enumerate(sass_instrs):
            for lbl in ctx.label_map:
                # Match only the label tag at the START of the comment (not in BRA comments)
                if si.comment.startswith(f'// {lbl}:'):
                    label_abs_byte[lbl] = j * 16
        # Fallback: for labels not found by comment, use preamble + body offset
        for lbl, body_byte in ctx.label_map.items():
            if lbl not in label_abs_byte:
                label_abs_byte[lbl] = n_preamble * 16 + body_byte

        # Step 2: Find BRA instructions in the final stream by opcode
        for i, si in enumerate(sass_instrs):
            opc = (si.raw[0] | (si.raw[1] << 8)) & 0xFFF
            if opc != 0x947:  # BRA opcode
                continue
            # Extract target label from comment — match "BRA <label>" specifically
            # to avoid matching label tags at the start of the comment
            target_label = None
            for _, tl in ctx._bra_fixups:
                if f'BRA {tl}' in si.comment:
                    target_label = tl
                    break
            if target_label is None:
                continue

            # Check if target is a ret-only block
            target_is_exit = any(
                tbb.label == target_label
                and len(tbb.instructions) == 1
                and tbb.instructions[0].op == 'ret'
                for tbb in fn.blocks
            )

            if target_is_exit:
                # Target is a ret-only block — replace BRA with EXIT directly.
                # Previous approach jumped to the last EXIT instruction, but
                # that fails when the only EXIT is predicated (bounds check).
                exit_raw = encode_exit()
                # Preserve the predicate from the BRA instruction if present
                pred_nibble = (si.raw[1] >> 4) & 0xF
                if pred_nibble != 0x7:  # not PT — has predicate guard
                    exit_raw = bytearray(exit_raw)
                    exit_raw[1] = (exit_raw[1] & 0x0F) | (si.raw[1] & 0xF0)
                    exit_raw = bytes(exit_raw)
                pred_prefix = si.comment.split('BRA')[0] if 'BRA' in si.comment else ''
                sass_instrs[i] = SassInstr(exit_raw,
                    f'{pred_prefix}EXIT  // replaces BRA {target_label} (ret-only)')
                continue
            elif target_label in label_abs_byte:
                actual_target_byte = label_abs_byte[target_label]
            else:
                continue

            if actual_target_byte is None:
                continue

            bra_next_byte = (i + 1) * 16
            rel_offset = actual_target_byte - bra_next_byte

            if rel_offset == 0:
                sass_instrs[i] = SassInstr(encode_nop(),
                    f'NOP  // eliminated fall-through BRA {target_label}')
                continue

            old_raw = bytearray(si.raw)
            if sm_version == 89:
                # SM_89: signed 32-bit byte offset in bytes 4-7
                off32 = rel_offset & 0xFFFFFFFF
                old_raw[4] = off32 & 0xFF
                old_raw[5] = (off32 >> 8) & 0xFF
                old_raw[6] = (off32 >> 16) & 0xFF
                old_raw[7] = (off32 >> 24) & 0xFF
            else:
                # SM_120 BRA encoding.
                # b1 upper nibble is set by patch_pred in isel.py:
                #   0x79 = PT (unconditional)  0x09 = @P0  0x89 = @!P0  etc.
                # Unconditional forward BRA uses BRA.U !UP0 (UP0 initialises to 0
                #   so !UP0=1 = always-fire). Predicated BRAs and unconditional
                #   backward BRAs use the @P/@!P encoding with formula:
                #     total = delta_instrs_from_next * 4  (signed int)
                #     b2 = total & 0xFF
                #     b4 = ((total >> 8) << 2) & 0xFF
                #     b5-b9 = sign-extension byte (0xFF if b4 >= 0x80 else 0x00)
                #     b10 = 0x80 | ((total >> 16) & 0x03)
                #     b11 = 0x03
                pred_nibble = (old_raw[1] >> 4) & 0xF
                is_pt = (pred_nibble == 0x7)  # PT = unconditional
                if is_pt and rel_offset > 0:
                    # Unconditional forward BRA: BRA.U !UP0.
                    # offset_instrs counts from the *current* instruction.
                    offset_instrs = rel_offset // 16 + 1
                    old_raw[1]  = 0x75
                    old_raw[2]  = (offset_instrs * 4) & 0xFF
                    old_raw[3]  = 0x08
                    old_raw[4]  = (1 + 4 * (offset_instrs >> 6)) & 0xFF
                    old_raw[5]  = 0x00
                    old_raw[6]  = 0x00
                    old_raw[7]  = 0x00
                    old_raw[8]  = 0x00
                    old_raw[9]  = 0x00
                    old_raw[10] = 0x80
                    old_raw[11] = 0x0b
                    old_raw[12] = 0x00
                    old_raw[13] = 0xea
                    old_raw[14] = 0x0f
                    old_raw[15] = 0x00
                else:
                    # Predicated BRA (any direction) or unconditional backward BRA.
                    # Convert PT backward → @!P0: loop-back is only reached when
                    # the loop-exit @P0 BRA was not taken, guaranteeing P0=0.
                    if is_pt:
                        old_raw[1] = 0x89  # @!P0
                    total = (rel_offset // 16) * 4
                    b2  = total & 0xFF
                    b4  = ((total >> 8) << 2) & 0xFF
                    se  = 0xFF if b4 >= 0x80 else 0x00
                    b10 = 0x80 | ((total >> 16) & 0x03)
                    old_raw[2]  = b2
                    old_raw[3]  = 0x00
                    old_raw[4]  = b4
                    old_raw[5]  = se
                    old_raw[6]  = se
                    old_raw[7]  = se
                    old_raw[8]  = se
                    old_raw[9]  = se
                    old_raw[10] = b10
                    old_raw[11] = 0x03
                    old_raw[12] = 0x00
                    old_raw[13] = 0xea
                    old_raw[14] = 0x0f
                    old_raw[15] = 0x00
            pred_prefix = si.comment.split('BRA')[0] if 'BRA' in si.comment else ''
            sass_instrs[i] = SassInstr(
                bytes(old_raw),
                f'{pred_prefix}BRA {target_label} (offset={rel_offset})')

    # Append BRA trap loop after the final EXIT (required by NVIDIA hardware).
    # This catches warps that somehow continue past EXIT and prevents
    # execution of uninitialized memory.
    if sm_version == 89:
        from sass.encoding.sm_89_opcodes import encode_bra as sm89_bra
        sass_instrs.append(SassInstr(sm89_bra(-16), 'BRA $ // trap loop'))
    else:
        sass_instrs.append(SassInstr(encode_bra(-16), 'BRA $ // trap loop'))

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
    param_area_end = ((param_base + total_param_bytes + 3) // 4) * 4
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

    # Find ALL EXIT and S2R instruction offsets in the SASS byte stream
    exit_offsets_all = []
    exit_offset = 0
    s2r_offset = 0x10  # default
    for i in range(0, len(sass_bytes), 16):
        if sass_bytes[i:i+2] == bytes([0x4d, 0x79]):  # EXIT opcode
            exit_offsets_all.append(i)
            if not exit_offset:
                exit_offset = i
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

    _final_gprs = max(alloc.num_gprs, ctx._next_gpr,
                      getattr(ctx, '_scratch_highwater', 0))
    if verbose:
        print(f"[pipeline] final num_gprs: alloc={alloc.num_gprs} ctx._next_gpr={ctx._next_gpr} "
              f"highwater={getattr(ctx, '_scratch_highwater', 0)} -> {_final_gprs}")
    # Compute shared memory size from declarations
    smem_size = 0
    if hasattr(fn, 'shared_decls') and fn.shared_decls:
        offset = 0
        for sd in fn.shared_decls:
            # Align offset to declaration alignment
            offset = (offset + sd.align - 1) & ~(sd.align - 1)
            offset += sd.size
        smem_size = offset

    desc = KernelDesc(
        name=fn.name,
        sass_bytes=sass_bytes,
        num_gprs=_final_gprs,
        num_params=len(fn.params),
        param_sizes=param_sizes,
        param_offsets=alloc.param_offsets,
        const0_size=const0_size,
        const0_init_data=const0_init,
        exit_offset=exit_offset,
        s2r_offset=s2r_offset,
        smem_size=smem_size,
        # Pass ptxas capmerc to emitter for ELF section sizing — UNLESS our kernel
        # needs more GPRs than ptxas's (high-register kernel). In that case use
        # None so the emitter calls build_capmerc_from_sass with 0x2000 capability
        # bit and full-range type-02 barrier records (byte[10]=0x01), which is
        # required to enable R14+ access on SM_120 Mercury.
        ptxas_capmerc=_select_capmerc(ptxas_meta, _final_gprs),
        ptxas_merc_info=None,
    )
    if sm_version == 89:
        # SM_89: use simplified emitter (no capmerc/merc)
        from cubin.emitter_sm89 import emit_cubin_sm89
        # Convert param_offsets dict → list of relative offsets within param area
        # alloc.param_offsets has absolute cbank offsets (e.g. 0x160, 0x168);
        # the SM_89 emitter expects offsets relative to param base (0, 8, 16...).
        param_off_list = [
            alloc.param_offsets.get(p.name, 0) - param_base
            for p in fn.params
        ]
        cubin_bytes = emit_cubin_sm89(
            kernel_name=desc.name,
            sass_bytes=desc.sass_bytes,
            num_gprs=desc.num_gprs,
            num_params=desc.num_params,
            param_sizes=desc.param_sizes,
            param_offsets=param_off_list,
            const0_size=desc.const0_size,
            const0_init_data=desc.const0_init_data,
            exit_offsets=exit_offsets_all,
            s2r_offset=s2r_offset,
        )
    else:
        # SM_120: full emitter with capmerc/merc
        cubin_bytes = emit_cubin(desc)
        # Post-process: patch merc.nv.info with ptxas metadata.
        # capmerc is already embedded (passed to emitter above).
        if ptxas_meta:
            cubin_bytes = _patch_ptxas_metadata(cubin_bytes, ptxas_meta,
                                                min_gprs=desc.num_gprs)

    return cubin_bytes


def _select_capmerc(ptxas_meta: dict | None, kernel_gprs: int) -> bytes | None:
    """Return ptxas's capmerc for the emitter's ELF section sizing, or None.

    Returns None (→ use generated capmerc) when our kernel needs R14+ access.
    SM_120 Mercury restricts access to R0-R13 unless the capmerc capability mask
    has bit 0x2000 (high-register bit) and type-02 barrier byte[10]=0x01.
    ptxas only sets these flags when it allocates >14 GPRs. If our kernel uses
    more than 14 GPRs, we must use our generated capmerc (not ptxas's).
    """
    if not ptxas_meta or 'capmerc' not in ptxas_meta:
        return None
    ptxas_cap = ptxas_meta['capmerc']
    # Check if ptxas's capmerc already has the 0x2000 high-register capability bit.
    # If not, and our kernel needs R14+, use generated capmerc instead.
    if len(ptxas_cap) >= 16:
        cap_mask = int.from_bytes(ptxas_cap[12:16], 'little')
        ptxas_has_highreg = bool(cap_mask & 0x2000)
    else:
        ptxas_has_highreg = False
    if kernel_gprs > 14 and not ptxas_has_highreg:
        return None  # use generated capmerc with 0x2000 + full-range barrier records
    return ptxas_cap


def _patch_ptxas_metadata(cubin_bytes: bytes, ptxas_meta: dict,
                          min_gprs: int = 0) -> bytes:
    """Overwrite capmerc and merc.nv.info section data with ptxas-generated values.

    min_gprs: if > 0, enforce that the patched capmerc byte[8] (GPR count) is
    at least this value. Prevents ptxas's compact capmerc from under-reporting
    the register count when our isel uses more scratch registers than ptxas does.
    """
    import struct
    data = bytearray(cubin_bytes)
    e_shoff = struct.unpack_from('<Q', data, 40)[0]
    e_shnum = struct.unpack_from('<H', data, 60)[0]
    e_shentsize = struct.unpack_from('<H', data, 58)[0]
    e_shstrndx = struct.unpack_from('<H', data, 62)[0]
    sh_off = e_shoff + e_shstrndx * e_shentsize
    str_offset = struct.unpack_from('<Q', data, sh_off + 24)[0]
    strtab = data[str_offset:str_offset +
                  struct.unpack_from('<Q', data, sh_off + 32)[0]]
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_name_idx = struct.unpack_from('<I', data, off)[0]
        sh_offset_val = struct.unpack_from('<Q', data, off + 24)[0]
        sh_size = struct.unpack_from('<Q', data, off + 32)[0]
        name_end = strtab.index(0, sh_name_idx)
        sname = strtab[sh_name_idx:name_end].decode('ascii', errors='replace')
        if 'capmerc' in sname and 'text' in sname and 'capmerc' in ptxas_meta:
            # When our kernel uses more registers than ptxas's kernel (min_gprs > ptxas_gprs),
            # use our generated capmerc so the capability mask includes 0x2000
            # (high-register bit) and type-02 barrier byte[10]=0x01 (full range).
            # ptxas's capmerc is for ≤14 GPR kernels and lacks these flags,
            # causing Mercury to restrict access to R0-R13 at runtime (ERR715).
            ptxas_cap = ptxas_meta['capmerc']
            cap_mask = int.from_bytes(ptxas_cap[12:16], 'little') if len(ptxas_cap) >= 16 else 0
            ptxas_has_highreg = bool(cap_mask & 0x2000)
            if min_gprs > 14 and not ptxas_has_highreg:
                # Our kernel needs R14+ but ptxas's capmerc lacks 0x2000 bit.
                # Skip overwriting so the emitter's generated capmerc stays in place.
                pass
            else:
                # ptxas capmerc has required capability bits (or we don't need R14+)
                patch = bytearray(ptxas_cap[:sh_size])
                if len(patch) < sh_size:
                    patch.extend(bytearray(sh_size - len(patch)))
                if min_gprs > 0 and len(patch) > 8:
                    patch[8] = max(patch[8], min_gprs)
                data[sh_offset_val:sh_offset_val + sh_size] = patch
        if 'merc.nv.info.' in sname and 'merc_info' in ptxas_meta:
            # Only use ptxas merc.nv.info when ptxas and our isel agree on
            # register count. If we use more registers (e.g. UR→GPR scratch
            # spills), ptxas's 0x5a per-instruction blob may restrict R14+
            # access and cause ERR715. Fall back to our generated merc.nv.info
            # (which uses the permissive vector_add 0x5a blob).
            ptxas_cap = ptxas_meta.get('capmerc', b'')
            ptxas_gprs = ptxas_cap[8] if len(ptxas_cap) > 8 else 0
            if min_gprs <= ptxas_gprs:
                ptxas_mi = ptxas_meta['merc_info']
                patch = bytearray(ptxas_mi[:sh_size])
                if len(patch) < sh_size:
                    patch.extend(bytearray(sh_size - len(patch)))
                data[sh_offset_val:sh_offset_val + sh_size] = patch
    return bytes(data)


def _extract_ptxas_metadata(ptx_src: str) -> dict[str, dict]:
    """Run ptxas on the PTX source and extract capmerc + merc.nv.info per kernel.

    Returns {kernel_name: {'capmerc': bytes, 'merc_info': bytes}} or empty dict
    if ptxas is not available.
    """
    import tempfile, subprocess, struct
    result = {}
    try:
        # ptxas requires ASCII-only input — strip non-ASCII characters
        # (e.g. em-dash U+2014 in Forge-generated comments) before passing to ptxas.
        ptx_ascii = ptx_src.encode('ascii', errors='replace').decode('ascii')
        with tempfile.NamedTemporaryFile(suffix='.ptx', delete=False, mode='w',
                                         encoding='ascii') as f:
            f.write(ptx_ascii)
            ptx_path = f.name
        cubin_path = ptx_path.replace('.ptx', '.cubin')
        r = subprocess.run(['ptxas', '-arch', 'sm_120', '-o', cubin_path, ptx_path],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return result
        data = open(cubin_path, 'rb').read()
        e_shoff = struct.unpack_from('<Q', data, 40)[0]
        e_shnum = struct.unpack_from('<H', data, 60)[0]
        e_shentsize = struct.unpack_from('<H', data, 58)[0]
        e_shstrndx = struct.unpack_from('<H', data, 62)[0]
        sh_off = e_shoff + e_shstrndx * e_shentsize
        str_offset = struct.unpack_from('<Q', data, sh_off + 24)[0]
        strtab = data[str_offset:str_offset +
                      struct.unpack_from('<Q', data, sh_off + 32)[0]]
        sections = {}
        for i in range(e_shnum):
            off = e_shoff + i * e_shentsize
            sh_name_idx = struct.unpack_from('<I', data, off)[0]
            sh_offset_val = struct.unpack_from('<Q', data, off + 24)[0]
            sh_size = struct.unpack_from('<Q', data, off + 32)[0]
            name_end = strtab.index(0, sh_name_idx)
            sname = strtab[sh_name_idx:name_end].decode('ascii', errors='replace')
            sections[sname] = data[sh_offset_val:sh_offset_val + sh_size]
        for sname, sec_data in sections.items():
            if sname.startswith('.nv.capmerc.text.'):
                kname = sname[len('.nv.capmerc.text.'):]
                if kname not in result:
                    result[kname] = {}
                result[kname]['capmerc'] = sec_data
            if sname.startswith('.nv.merc.nv.info.'):
                kname = sname[len('.nv.merc.nv.info.'):]
                if kname not in result:
                    result[kname] = {}
                result[kname]['merc_info'] = sec_data
        import os
        os.unlink(ptx_path)
        os.unlink(cubin_path)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return result


def compile_ptx_source(ptx_src: str, verbose: bool = False) -> dict[str, bytes]:
    """
    Compile PTX source text. Returns {kernel_name: cubin_bytes} for each kernel.
    """
    mod = parse(ptx_src)
    mod, rotate_groups = rotate_run(mod)

    # Detect target architecture from PTX
    sm_version = 120  # default
    if hasattr(mod, 'target') and mod.target:
        import re
        m = re.search(r'sm_(\d+)', mod.target)
        if m:
            sm_version = int(m.group(1))

    # Extract capmerc/merc metadata from ptxas (SM_120 only — SM_89 has no capmerc)
    ptxas_meta = _extract_ptxas_metadata(ptx_src) if sm_version >= 120 else {}

    results = {}
    for fn in mod.functions:
        if fn.is_kernel:
            results[fn.name] = compile_function(
                fn, verbose=verbose, ptxas_meta=ptxas_meta.get(fn.name),
                sm_version=sm_version)

    return results


def compile_ptx(ptx_path: str, verbose: bool = False) -> dict[str, bytes]:
    """
    Compile a PTX file. Returns {kernel_name: cubin_bytes} for each kernel.
    """
    src = Path(ptx_path).read_text(encoding='utf-8')
    return compile_ptx_source(src, verbose=verbose)
