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
from sass.encoding.sm_120_opcodes import encode_bra, encode_ldcu_64, encode_exit, encode_nop, encode_ldc
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

    def _pred_from_float_setp(block_instrs, pred_name):
        """Check if the guard predicate is set by a float setp in the same block.

        Float setps on SM_120 use FSEL.step (not ISETP inversion), so the
        _negated_preds convention doesn't apply correctly.  If-conversion
        of diamonds guarded by float predicates produces wrong guard sense.
        Block if-conversion in this case and let it fall through to branches.
        """
        for inst in block_instrs:
            if (inst.op == 'setp'
                    and isinstance(inst.dest, RegOp)
                    and inst.dest.name == pred_name
                    and inst.types
                    and any(t in ('f32', 'f64') for t in inst.types)):
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
                                    or _overwrites_pred(else_instrs, pred_name)
                                    or _pred_from_float_setp(instrs, pred_name)):
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
                                        or _overwrites_pred(false_body, pred_name)
                                        or _pred_from_float_setp(bb.instructions, pred_name)):
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

    # WB-7: pre-allocator address-fold analysis.  Detects
    # `add.u64 %A, %B, IMM` patterns whose only consumer folds the
    # offset into a load/store.  The dead-add's dest vreg is then
    # excluded from regalloc so its phys pair is never reserved.
    from sass.isel import analyze_addr_offset_fold
    _addr_fold_map, _addr_fold_dead_adds = analyze_addr_offset_fold(fn)
    _addr_fold_dead_vregs: set[str] = set()
    if _addr_fold_dead_adds:
        from ptx.ir import RegOp
        for bb in fn.blocks:
            for inst in bb.instructions:
                if (id(inst) in _addr_fold_dead_adds
                        and isinstance(inst.dest, RegOp)):
                    _addr_fold_dead_vregs.add(inst.dest.name)

    # 1. Register allocation
    from sass.regalloc import PARAM_BASE_SM120, PARAM_BASE_SM89
    param_base = PARAM_BASE_SM89 if sm_version == 89 else PARAM_BASE_SM120
    has_capmerc = ptxas_meta is not None and 'capmerc' in (ptxas_meta or {})
    alloc = allocate(fn, param_base=param_base, has_capmerc=has_capmerc,
                     sm_version=sm_version, skip_vregs=_addr_fold_dead_vregs)

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
    # FG-4.4 Bug 2 investigation found that the CUDA driver zeroes a
    # region of cbuf[0] past the last declared param (observed up to
    # at least 32 bytes past param_base).  Any 32-bit literal placed
    # adjacent to the params is overwritten at launch and the LDCU.32
    # that loads it reads 0.  The fix at the isel site is to prefer
    # encode_imad_r_imm (opcode 0x824, inline 16-bit imm) over the
    # LDCU.32 + IMAD R-UR path for small immediates — that sidesteps
    # the literal pool entirely.  Large immediates (> 0xffff) still
    # need a literal; 32-byte alignment is enough to keep them past
    # the driver's pad region.
    lit_pool_base = (last_param_end + 31) & ~31

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
        _next_gpr=_addr_scratch_base,  # scratch allocated on demand, not pre-reserved
        _next_pred=alloc.num_pred,
        sm_version=sm_version,
    )
    ctx._addr_scratch_lo = _addr_scratch_base  # dedicated addr pair: R(base):R(base+1)
    # WB-7: aliased-base address-chain folding (analysis ran above
    # before allocate() so the dead vregs are excluded from int_regs).
    # Pass the maps to isel via ctx.
    ctx._addr_fold_map = _addr_fold_map
    ctx._addr_fold_dead_adds = _addr_fold_dead_adds
    # WB-5.0: tiny-kernel direct LDC.64 path.  When a kernel has a
    # single-use u64 pointer param, the allocator gives it a real GPR
    # pair and tags it for _select_ld_param to emit LDC.64 directly
    # (no LDCU.64 + IADD.64 R-UR materialization).
    ctx._direct_ldc_params = alloc.direct_ldc_params
    # WB-2: HMMA "all-zero inputs" RZ-substitution analysis.
    # Restricted to HMMA shapes; IMMA / DMMA / QMMA paths are unaffected.
    from sass.isel import analyze_mma_zero_subst
    _hmma_rz_subst, _hmma_dead_movs = analyze_mma_zero_subst(fn)
    ctx._hmma_rz_subst = _hmma_rz_subst
    ctx._hmma_dead_movs = _hmma_dead_movs

    # WB-2 / WB-4: When mma RZ-substitution leaves the address scratch
    # above the kernel's true register high-water, rebase it down into
    # the lowest even-aligned pair whose phys slots are NOT live at
    # the address-materialization point.
    #
    # A phys is live-at-mat iff its vreg has any source use AFTER the
    # last mma in the function.  Vregs whose last use is the mma itself
    # are dead by the time isel needs the address scratch (which is
    # always after every mma in the kernel — there are no LDG/STG
    # operations interleaved with mma in the kernels we currently
    # support, and the scratch only matters at store time).
    #
    # Earlier WB-2 logic checked vreg membership in `dead_movs`, which
    # was wrong: a vreg can have its INIT mov be dead (eliminated by
    # RZ-substitution) and STILL be alive post-mma because the mma's
    # dst tuple writes the same vreg name and a downstream store reads
    # it.  HMMA happened to work by accident because R2:R3 was empty.
    # QMMA has live B operands at R2:R3, so the search would skip past
    # them and pick R4:R5 — which collides with the dst quad.
    if _hmma_dead_movs:
        from ptx.ir import RegOp, VectorRegOp, MemOp, ScalarKind
        _all_instrs = []
        for bb in fn.blocks:
            _all_instrs.extend(bb.instructions)

        # Build reg_last_use (source-use index per vreg, including all
        # vector tuple components and MemOp bases).
        _last_use: dict[str, int] = {}
        for idx, inst in enumerate(_all_instrs):
            for src in (inst.srcs or []):
                if isinstance(src, VectorRegOp):
                    for v in src.regs:
                        _last_use[v] = idx
                elif isinstance(src, RegOp):
                    _last_use[src.name] = idx
                elif isinstance(src, MemOp) and isinstance(src.base, str):
                    bn = src.base if src.base.startswith('%') else f'%{src.base}'
                    _last_use[bn] = idx

        _last_mma_idx = max(
            (i for i, inst in enumerate(_all_instrs)
             if inst.op == 'mma' and 'sync' in inst.types),
            default=-1,
        )

        _u64_vreg_names: set[str] = set()
        for rd in fn.reg_decls:
            if rd.type.kind == ScalarKind.PRED:
                continue
            if rd.type.width >= 64:
                _u64_vreg_names.update(rd.names)

        _live_phys: set[int] = {0, 1}
        for vreg, phys in alloc.ra.int_regs.items():
            last = _last_use.get(vreg, -1)
            # A phys is live at the address-materialization point iff
            # its vreg has any source use strictly after the last mma.
            if last > _last_mma_idx:
                _live_phys.add(phys)
                if vreg in _u64_vreg_names:
                    _live_phys.add(phys + 1)

        for _r in range(2, _addr_scratch_base + 1, 2):
            if _r not in _live_phys and (_r + 1) not in _live_phys:
                if _r < _addr_scratch_base:
                    _addr_scratch_base = _r
                    ctx._addr_scratch_lo = _r
                    ctx._next_gpr = _r
                break
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

    # SM_120 rule #25: detect kernels needing ptxas fallback.
    # Complex kernels with LDG produce instruction streams that differ from
    # ptxas in ways that cause 700/715 for large text sizes.
    # Use ptxas fallback for LDG kernels with sync (BAR/VOTE/REDUX) or
    # complex control flow (if-else chains producing >512B text).
    _has_vote_shfl = any(
        inst.op in ('vote', 'redux', 'shfl')
        for bb in fn.blocks
        for inst in bb.instructions
    )
    _has_sync = _has_vote_shfl  # BAR alone works natively now
    _has_ldg = any(
        inst.op == 'ld' and 'global' in inst.types
        for bb in fn.blocks
        for inst in bb.instructions
    )
    _has_stg = any(
        (inst.op == 'st' and 'global' in inst.types) or inst.op == 'atom'
        for bb in fn.blocks
        for inst in bb.instructions
    )
    _n_params = len(fn.params)
    _has_complex_cf = len(fn.blocks) > 4
    _has_atom = any(
        inst.op == 'atom'
        for bb in fn.blocks
        for inst in bb.instructions
    )
    _has_bar = any(
        inst.op == 'bar'
        for bb in fn.blocks
        for inst in bb.instructions
    )
    _has_smem = any(
        (inst.op == 'st' and 'shared' in inst.types)
        or (inst.op == 'ld' and 'shared' in inst.types)
        for bb in fn.blocks
        for inst in bb.instructions
    )
    # All instruction classes now compile natively.
    # Complex CF+STG (>4 blocks with stores) was the last gate;
    # if-conversion + deferred params handle this correctly.
    ctx._has_vote = _has_vote_shfl

    # Pre-scan: identify u32 params consumed ONLY by setp.
    # These don't need a GPR LDC — setp handler emits LDCU.32 directly.
    # Only safe for non-divergent kernels (no if-converted branches).
    _setp_only_params = set()
    _has_if_converted = any(
        inst.pred and inst.op in ('ld', 'st')
        for bb in fn.blocks for inst in bb.instructions
    )
    if sm_version >= 120 and not _has_if_converted:
        _ld_param_dests = {}
        for bb in fn.blocks:
            for inst in bb.instructions:
                if inst.op == 'ld' and 'param' in inst.types and inst.dest:
                    typ = inst.types[-1] if inst.types else ''
                    if typ in ('u32', 's32', 'b32'):
                        _ld_param_dests[inst.dest.name] = True
        for pname in list(_ld_param_dests):
            uses = []
            for bb in fn.blocks:
                for inst in bb.instructions:
                    if inst.op == 'ld' and inst.dest and inst.dest.name == pname:
                        continue
                    if any(getattr(s, 'name', None) == pname for s in (inst.srcs or [])):
                        uses.append(inst.op)
            if uses and all(op == 'setp' for op in uses):
                _setp_only_params.add(pname)
        if _setp_only_params:
            ctx._setp_only_params = _setp_only_params

    # Pre-scan: detect if kernel uses bar.sync (shared memory synchronization).
    # Kernels with bar.sync need preamble-only constant loads to avoid
    # LDC/LDCU poisoning of IADD.64-UR.
    ctx._has_bar_sync = any(
        inst.op == 'bar' and 'sync' in inst.types
        for bb in fn.blocks for inst in bb.instructions
    )

    body_instrs = select_function(fn, ctx)

    # SM_120 requires at least one S2R before LDCU param loads.
    # If the body has no S2R, insert a dummy S2R R0, SR_TID.X.
    #
    # WB-5.0: skip the dummy S2R for tiny-kernel direct-LDC mode.  When
    # the only param is loaded via LDC.64 (not LDCU.64), the body has
    # no LDCU.64 param load and no S2R is needed before the descriptor
    # LDCU.64.  ptxas's DMMA confirms this pattern works.
    from sass.encoding.sm_120_opcodes import encode_s2r, SR_TID_X
    has_s2r = any(
        struct.unpack_from('<Q', si.raw, 0)[0] & 0xFFF in (0x919, 0x9c3)
        for si in body_instrs
    )
    body_has_ldcu_param = any(
        (struct.unpack_from('<Q', si.raw, 0)[0] & 0xFFF) == 0x7ac
        and si.raw[9] == 0x0a
        for si in body_instrs
    )
    if not has_s2r and (body_has_ldcu_param or not ctx._direct_ldc_params):
        body_instrs.insert(0, SassInstr(encode_s2r(0, SR_TID_X),
                                         'S2R R0, SR_TID.X  // required for LDCU init'))

    # SM_120 preamble window: insert ALL LDCUs right after S2R, before
    # any body instruction that uses UR values (including bounds-check ISETP).
    # Then insert UR4 descriptor after the predicated EXIT (if any).
    #
    # SM_120 preamble construction:
    # Phase 1: classify each instruction in original body_instrs BEFORE
    #          any insertions. Tag body LDC/LDCU that need hoisting.
    # Phase 2: build the final instruction list in one pass.
    #
    # SM_120 rule: ANY constant-bank load (LDC 0xb82, LDCU 0x7ac) in the
    # body region of a BAR.SYNC kernel poisons IADD.64-UR → ERR715.
    # All constant loads must be in the preamble window.

    # Find S2R position
    s2r_pos = 0
    for idx, si in enumerate(body_instrs):
        opcode = struct.unpack_from('<Q', si.raw, 0)[0] & 0xFFF
        if opcode in (0x919, 0x9c3):
            s2r_pos = idx + 1
            break

    # Detect BAR.SYNC in original body_instrs
    has_bar_in_body = any(
        (si.raw[0] | (si.raw[1] << 8)) & 0xFFF == 0xb1d
        for si in body_instrs)

    # Tag body constant loads for hoisting (only in BAR kernels).
    # Hoist LDCU (UR bank, safe) and LDC that appear BEFORE BAR.
    # Do NOT hoist LDC that appears AFTER BAR (post-BAR LDC is intentional
    # and its register may be reused between preamble and BAR).
    hoist_indices = set()
    if has_bar_in_body:
        bar_pos_orig = None
        for idx, si in enumerate(body_instrs):
            opc = (si.raw[0] | (si.raw[1] << 8)) & 0xFFF
            if opc == 0xb1d:
                bar_pos_orig = idx
                break
        for idx, si in enumerate(body_instrs):
            opc = (si.raw[0] | (si.raw[1] << 8)) & 0xFFF
            is_ldcu = opc == 0x7ac
            is_pre_bar_ldc = opc == 0xb82 and si.raw[2] != 1 and bar_pos_orig and idx < bar_pos_orig
            if idx >= s2r_pos and (is_ldcu or is_pre_bar_ldc):
                hoist_indices.add(idx)

    # Build the final instruction list:
    # Preamble (hardcoded ctrl, scheduler-immune):
    #   [LDC R1] [preamble LDCUs] [hoisted body LDCs] [UR4 desc]
    # Body (scoreboard ctrl, scheduled):
    #   [S2R] [rest of body without hoisted loads]
    preamble_ldcus = getattr(ctx, '_preamble_ldcus', [])

    # P3-7: UR activation is now handled AFTER scheduling (see line ~1050).
    # The old P3-5 preamble_ldcus injection is removed.
    hoisted_loads = [body_instrs[i] for i in sorted(hoist_indices)]
    body_without_hoisted = [si for i, si in enumerate(body_instrs) if i not in hoist_indices]

    if has_bar_in_body:
        # BAR kernel: add ALL constant loads + UR4 to preamble (scheduler-immune).
        for item in preamble_ldcus + hoisted_loads + [ur4_desc_instr]:
            preamble.append(item)
        body_instrs = body_without_hoisted
        _n_removed = len(hoist_indices)
    else:
        # Non-BAR kernel: insert preamble LDCUs + UR4 into body_instrs
        # at s2r_pos (the original pattern that works for these kernels).
        # P3-7: skip ur4_desc_instr if already emitted in activation sequence
        _skip_ur4 = getattr(ctx, '_ur_activation_sr', None) is not None
        all_inserted = preamble_ldcus + ([] if _skip_ur4 else [ur4_desc_instr])
        for pi, item in enumerate(all_inserted):
            body_without_hoisted.insert(s2r_pos + pi, item)
        body_instrs = body_without_hoisted
        _n_removed = 0

    if has_bar_in_body:
        # BAR kernel: loads removed from body, nothing inserted.
        _net_shift = -_n_removed
    else:
        # Non-BAR kernel: preamble LDCUs + UR4 inserted at s2r_pos.
        _net_shift = len(preamble_ldcus) + 1  # +1 for UR4
    first_body_label = None
    if hasattr(ctx, '_bra_fixups') and ctx._bra_fixups:
        for _, target in ctx._bra_fixups:
            tgt_byte = ctx.label_map.get(target, 0)
            if tgt_byte // 16 >= s2r_pos:
                first_body_label = target
                break

    for label in list(ctx.label_map):
        if label == first_body_label:
            continue
        body_byte = ctx.label_map[label]
        body_idx = body_byte // 16
        if body_idx >= s2r_pos:
            ctx.label_map[label] = body_byte + _net_shift * 16
    if hasattr(ctx, '_bra_fixups'):
        ctx._bra_fixups = [
            (bra_idx + _net_shift if bra_idx >= s2r_pos else bra_idx, target_label)
            for bra_idx, target_label in ctx._bra_fixups
        ]

    # WB-8: LDCU.128 param packing.
    #
    # Replace pairs of LDCU.64 instructions whose:
    #   - dest URs are 4-aligned and consecutive (URa, URa+2)
    #   - cbuf byte offsets are X and X+8
    #   - X is 16-byte aligned (i.e. qword index even)
    # ...with one LDCU.128 (lower UR, lower offset) + a NOP placeholder
    # for the partner.
    #
    # The replacement is done in-place by flipping byte[9] from 0x0a
    # (64-bit) to 0x0c (128-bit) on the lower-offset LDCU and NOP-ing
    # the higher-offset partner.  This preserves the scoreboard ctrl
    # bytes the existing assign_ctrl pass set up.
    def _wb8_pack_ldcu_128(instrs):
        from sass.encoding.sm_120_opcodes import encode_nop
        # Collect all LDCU.64 (b9=0x0a) param-load instructions.
        ldcu64 = []
        for j, si in enumerate(instrs):
            opc = (si.raw[0] | (si.raw[1] << 8)) & 0xFFF
            if opc == 0x7ac and si.raw[9] == 0x0a:
                ldcu64.append((j, si.raw[2], si.raw[5] * 8))
        by_key: dict[tuple, int] = {(d, b): j for j, d, b in ldcu64}
        replacements: dict[int, SassInstr] = {}
        paired: set[int] = set()
        for j, d, b in ldcu64:
            if j in paired:
                continue
            if d % 4 != 0 or b % 16 != 0:
                continue
            partner_j = by_key.get((d + 2, b + 8))
            if partner_j is None or partner_j in paired:
                continue
            # Patch: flip byte[9] 0x0a -> 0x0c on the lower-offset LDCU.
            packed_raw = bytearray(instrs[j].raw)
            packed_raw[9] = 0x0c
            replacements[j] = SassInstr(
                bytes(packed_raw),
                f'LDCU.128 UR{d}, c[0][0x{b:x}]  // WB-8 packed')
            replacements[partner_j] = SassInstr(
                encode_nop(), 'NOP  // WB-8: LDCU.128 absorbed')
            paired.add(j)
            paired.add(partner_j)
        if not replacements:
            return instrs
        return [replacements.get(k, si) for k, si in enumerate(instrs)]

    body_instrs = _wb8_pack_ldcu_128(body_instrs)

    # 4. Schedule: reorder for LDG latency, then assign ctrl via scoreboard
    raw_instrs = preamble + body_instrs
    reordered = schedule(raw_instrs)
    # The preamble (LDC R1) has hardcoded ctrl from ptxas.
    # Only assign scoreboard ctrl to body instructions (after preamble).
    n_preamble = len(preamble)
    preamble_instrs = list(reordered[:n_preamble])
    body_scheduled = assign_ctrl(reordered[n_preamble:])

    # SM_120 rule #25: add LDCU.32 prelude for vote-kernel s32 params.
    # Body LDC (0xb82) is forbidden in vote kernels. Use LDCU.32 in the
    # prelude region instead. The value stays in UR and is consumed by
    # ISETP.UR or MOV UR→GPR at point of use.
    # Insert BEFORE the S2R instruction (which must come before LDCU params).
    # P3-7: inject UR activation sequence AFTER scheduling, BEFORE body.
    # This ensures the scheduler NEVER sees or reorders these instructions.
    # They go between the preamble (S2R R1) and the scheduled body.
    # -----------------------------------------------------------------------
    # PHASE-4.3: PTXAS-faithful template for atom.global.xor.b32
    # -----------------------------------------------------------------------
    # Phase-4.2 proved that the UR pipeline activation state depends on the
    # exact surrounding instruction context, not just the activation opcodes.
    # Generic post-scheduling injection cannot reproduce the PTXAS context.
    #
    # Solution: emit the exact PTXAS instruction byte sequence for atom.xor
    # kernels, parameterized only by the UIADD constant K.  The template
    # replaces BOTH the activation AND the body for the atom.xor block.
    #
    # Supported: atom.global.xor.b32 with uniform SR-derived data (direct
    # or tid+constant), kernel signature (.u64 p_out, .u32 n).
    # -----------------------------------------------------------------------
    _ur_activation = []
    _ur_act_sr = getattr(ctx, '_ur_activation_sr', None)
    if _ur_act_sr is not None:
        _ur_act_add = getattr(ctx, '_ur_activation_add', 0)

        # ---------------------------------------------------------------
        # TEMPLATE-ENGINE-5D: spec-driven rendering from auto-generated
        # templates.  For Variant A (direct SR, no UIADD), uses the
        # generalized atom_ur spec that also covers MIN/MAX via
        # operation-specific parameter fields.  Falls back to inline
        # bytes if spec loading fails.
        # ---------------------------------------------------------------
        _spec_ok = False
        try:
            import json as _json
            from pathlib import Path as _Path
            _spec_dir = _Path(__file__).resolve().parent.parent / 'tools' / 'template_engine' / 'generated'
            if _ur_act_add != 0:
                _spec_file = _spec_dir / 'atom_xor_uniform_tid_plus_constant.json'
            else:
                # TE5-D: use generalized spec (covers XOR/MIN/MAX via params)
                _spec_file = _spec_dir / 'atom_ur_generalized_xor_min_max.json'

            if _spec_file.exists():
                _spec_data = _json.loads(_spec_file.read_text(encoding='utf-8'))
                _spec_instrs = _spec_data['instructions']
                _T = []
                for _si in _spec_instrs[1:]:  # skip preamble S2R
                    _raw = bytearray(bytes.fromhex(_si['bytes']))
                    for _p in _si.get('params', []):
                        if _p['name'] == 'add_imm_K':
                            for _bi in range(_p['byte_length']):
                                _raw[_p['byte_offset'] + _bi] = (_ur_act_add >> (8 * _bi)) & 0xFF
                        # TE5-D: generalized spec params use representative
                        # values (XOR operation).  No patching needed for XOR
                        # since the representative IS atom_xor.
                    _T.append(SassInstr(bytes(_raw), f"{_si['role']}  // TE5"))
                body_scheduled = []
                _ur_activation = _T
                _spec_ok = True
        except Exception:
            pass  # fall through to inline fallback

        if not _spec_ok:
            # Inline fallback (original P4.3 hardcoded bytes)
            _T = []
            _T.append(SassInstr(bytes.fromhex('19790000000000000021000000220e00'), 'S2UR UR0, TID.X  // P4.3'))
            _T.append(SassInstr(bytes.fromhex('ac7704ff007100000008000800240e00'), 'LDCU UR4, c[0x71]  // P4.3'))
            _T.append(SassInstr(bytes.fromhex('0c7c0000040000007060f00b00da1f00'), 'ISETP.RUR bounds  // P4.3'))
            _T.append(SassInstr(bytes.fromhex('4d090000000000000000800300ea0f00'), 'EXIT @P0  // P4.3'))
            _T.append(SassInstr(bytes.fromhex('19790200000000000000000000220e00'), 'S2UR UR2, LANEID  // P4.3'))
            if _ur_act_add != 0:
                _uiadd_b = bytearray(bytes.fromhex('357800000100000000008e0700e20f00'))
                _uiadd_b[4] = _ur_act_add & 0xFF
                _uiadd_b[5] = (_ur_act_add >> 8) & 0xFF
                _uiadd_b[6] = (_ur_act_add >> 16) & 0xFF
                _T.append(SassInstr(bytes(_uiadd_b), f'UIADD UR0 += {_ur_act_add}  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('867804000000000000018e0300e20f00'), '0x886  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('ac7706ff006b0000000a000800640e00'), 'LDCU UR6, desc  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('c4730500000000000080000000a20e00'), 'UMOV UR5, UR0  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('bd7204000400000000000e0800e20f00'), '0x2bd  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('027c050005000000000f000800ca4f00'), 'MOV.UR R5, UR5  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('0c7c0002040000007020f00b00e41f00'), 'ISETP.RUR flush  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('827b02ff00e00000000a000000760e00'), 'S2R R2, addr  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('8e0900020500000006e1920f00e22f00'), 'ATOMG.XOR  // P4.3'))
            else:
                _T.append(SassInstr(bytes.fromhex('c4730500000000000080000000620e00'), 'UMOV UR5, UR0  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('867804000000000000018e0300e20f00'), '0x886  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('ac7706ff006b0000000a000800a20e00'), 'LDCU UR6, desc  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('bd7204000400000000000e0800e20f00'), '0x2bd  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('027c050005000000000f000800ca2f00'), 'MOV.UR R5, UR5  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('0c7c0002040000007020f00b00e41f00'), 'ISETP.RUR flush  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('827b02ff00e00000000a000000b60e00'), 'S2R R2, addr  // P4.3'))
                _T.append(SassInstr(bytes.fromhex('8e0900020500000006e1920f00e24f00'), 'ATOMG.XOR  // P4.3'))
            body_scheduled = []
            _ur_activation = _T
        _ur_activation = _T

    sass_instrs = preamble_instrs + _ur_activation + body_scheduled

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

    # Ensure every code path ends with EXIT before the trap loop.
    # Blocks that fall through without a terminator (BRA/EXIT/RET) need
    # an explicit EXIT appended.  This handles cases where block
    # reordering places a non-terminating block at the end of the stream.
    if sass_instrs:
        last = sass_instrs[-1]
        last_opc = (last.raw[0] | (last.raw[1] << 8)) & 0xFFF
        _TERMINATORS = {0x947, 0x94d}  # BRA, EXIT
        if last_opc not in _TERMINATORS:
            # Use EXIT ctrl matching ptxas convention (wdep=0x3f, misc=5)
            exit_ctrl = (0x01 << 10) | (0x3f << 4) | 5  # rbar=1, wdep=0x3f, misc=5
            sass_instrs.append(SassInstr(encode_exit(ctrl=exit_ctrl), '// fall-through EXIT'))

    # Append BRA trap loop after the final EXIT (required by NVIDIA hardware).
    # This catches warps that somehow continue past EXIT and prevents
    # execution of uninitialized memory.
    _last_opc = (sass_instrs[-1].raw[0] | (sass_instrs[-1].raw[1] << 8)) & 0xFFF if sass_instrs else 0
    if _last_opc != 0x947:  # don't add if already ends with BRA
        if sm_version == 89:
            from sass.encoding.sm_89_opcodes import encode_bra as sm89_bra
            sass_instrs.append(SassInstr(sm89_bra(-16), 'BRA $ // trap loop'))
        else:
            sass_instrs.append(SassInstr(encode_bra(-16), 'BRA $ // trap loop'))

    # Deduplicate trailing EXIT+BRA pairs. Multiple ret-path blocks can
    # each generate their own EXIT, creating redundant trap loops that
    # inflate the text section and cause EXIT count mismatches with the
    # capmerc structure. ptxas emits exactly one EXIT+BRA trap loop.
    while len(sass_instrs) >= 4:
        tail_ops = [(si.raw[0] | (si.raw[1] << 8)) & 0xFFF for si in sass_instrs[-4:]]
        # Pattern: ..., EXIT, BRA, EXIT, BRA → remove second-to-last EXIT+BRA
        if tail_ops == [0x94d, 0x947, 0x94d, 0x947]:
            del sass_instrs[-4:-2]  # remove the first EXIT+BRA, keep last
        else:
            break

    if verbose:
        print(f"[pipeline] {len(sass_instrs)} SASS instructions:")
        for i, si in enumerate(sass_instrs):
            print(f"  +{i*16:4d}: {si.hex()}  // {si.comment}")

    # SM_120 rule #29: the first LDCU.64 param load after @P0 EXIT must use
    # b9=0x0c (not 0x0a). ptxas uses this encoding. Without it, post-EXIT
    # param loads cause 715.
    if sm_version >= 120:
        found_exit = False
        for i, si in enumerate(sass_instrs):
            opc = (si.raw[0] | (si.raw[1] << 8)) & 0xFFF
            guard = (si.raw[1] >> 4) & 0xF
            if opc == 0x94d and guard != 7:  # predicated EXIT
                found_exit = True
            if found_exit and opc == 0x7ac and si.raw[9] == 0x0a:
                if si.raw[5] >= 0x70 and 'deferred' not in si.comment:
                    # Only patch non-deferred post-EXIT LDCU.64 (preamble loads).
                    # Deferred LDCU.64 deep in the body use standard b9=0x0a.
                    patched = bytearray(si.raw)
                    patched[9] = 0x0c
                    sass_instrs[i] = SassInstr(bytes(patched), si.comment + ' [b9=0x0c]')
                    break

    # 3. Concatenate SASS bytes
    # FB-4.2: field-safe register compaction. Only runs if all opcodes have
    # GPR field metadata. Skips entirely on coverage gating failure.
    # FB-4.3: per-kernel compaction report (kernel name + counters).
    from sass.compact import compact as _fb42_compact
    sass_instrs, _compact_count = _fb42_compact(
        sass_instrs, verbose=verbose, kernel_name=fn.name)

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
    # Add literal pool entries (4 bytes each) — the pool starts at
    # lit_pool_base, which is 16-byte aligned past the params (FG-4.4
    # Bug 2 fix).  const0_size must therefore extend to lit_pool_base
    # + lit_pool_bytes, not just param_area_end + lit_pool_bytes,
    # otherwise the literal slots at the aligned offset overflow the
    # const section buffer.
    lit_pool_bytes = len(ctx._const_pool) * 4
    if lit_pool_bytes > 0:
        const0_size = ((lit_pool_base + lit_pool_bytes + 3) // 4) * 4
    else:
        const0_size = param_area_end

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

    # FB-4.4 + PERF-5.1: post-SASS register accounting.
    # First compute the allocator's estimate (high-water from allocation).
    _allocator_count = max(alloc.num_gprs, ctx._next_gpr,
                           getattr(ctx, '_scratch_highwater', 0))
    if _compact_count > 0:
        _final_gprs = max(_compact_count, 2)
    else:
        _final_gprs = max(_allocator_count, 2)
    # PERF-5.1: scan the actual emitted SASS to find the highest GPR
    # index that appears in any instruction's dest or src fields.  The
    # allocator can over-estimate when isel folds virtual regs to RZ
    # (e.g., DMMA zero-init patterns where %fd inputs are allocated
    # real pairs but the encoder uses RZ operands).  Declaring fewer
    # registers improves occupancy without affecting correctness.
    from sass.scoreboard import _get_dest_regs, _get_src_regs
    _sass_max_gpr = 0
    for _si in range(0, len(sass_bytes), 16):
        _raw = sass_bytes[_si:_si + 16]
        if len(_raw) < 16:
            break
        for _r in _get_dest_regs(_raw) | _get_src_regs(_raw):
            if _r < 255 and _r > _sass_max_gpr:
                _sass_max_gpr = _r
    _sass_gpr_count = _sass_max_gpr + 2  # +1 for index→count, +1 for pair safety
    if _sass_gpr_count < _final_gprs:
        _final_gprs = max(_sass_gpr_count, 2)
    if verbose and _final_gprs != _allocator_count:
        print(f"[pipeline] FB-4.4: final regs={_final_gprs} "
              f"(allocator reported {_allocator_count})")
    # For deferred-param kernels, the actual UR usage is lower than regalloc's
    # estimate (deferred params use post-EXIT URs, not pre-allocated ones).
    # Cap num_uniform to ctx._next_ur to avoid LAUNCH_OUT_OF_RESOURCES (716).
    # Only reduce num_uniform for kernels that actually used deferred params.
    # _deferred_ur_params is cleared after flush, so check if it WAS populated.
    _had_deferred = getattr(ctx, '_had_deferred_params', False)
    if _had_deferred:
        alloc.num_uniform = -1  # signal emitter to use ptxas UR count (14)
    else:
        alloc.num_uniform = max(alloc.num_uniform, ctx._next_ur)
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
        num_uniform=alloc.num_uniform,
        # Use ptxas capmerc when available — it has correct instruction class
        # records that our simpler generator may miss. Patch GPR count to match
        # our actual allocation since ptxas may allocate differently.
        ptxas_capmerc=_patch_ptxas_capmerc_gprs(
            ptxas_meta.get('capmerc'), _final_gprs) if ptxas_meta else None,
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
        # SM_120 rule #25: complex kernels with LDG + sync require
        # ptxas fallback. Our native backend's instruction stream differs
        # from ptxas in ways that cause 700/715 for these patterns.
        # Fallback: vote/shfl, global atoms, complex CF (>4 blocks),
        # scalar LDG from raw param pointer (SASS structural issue, not capmerc).
        _has_ldg_atom = _has_ldg and _has_atom
        # Scalar LDG: ld.global from raw param pointer (no add.u64 offset).
        _param_regs = set()
        _offset_regs = set()
        for bb in fn.blocks:
            for inst in bb.instructions:
                if not hasattr(inst, 'op'):
                    continue
                if inst.op == 'ld' and 'param' in inst.types and inst.dest:
                    _param_regs.add(inst.dest.name)
                if inst.op in ('add', 'mul', 'mad', 'cvt', 'shl') and inst.dest:
                    _offset_regs.add(inst.dest.name)
        _has_scalar_ldg = _has_ldg and any(
            inst.op == 'ld' and 'global' in inst.types and inst.srcs
            and any(hasattr(s, 'base') and s.base in _param_regs
                    and s.base not in _offset_regs
                    for s in inst.srcs if hasattr(s, 'base'))
            for bb in fn.blocks for inst in bb.instructions
            if hasattr(inst, 'op'))
        _needs_full_fallback = (ctx._has_vote or _has_ldg_atom
                                or _has_complex_cf or _has_scalar_ldg
                                or _has_bar)
        if _needs_full_fallback and ptxas_meta:
            ptxas_cubin = ptxas_meta.get('cubin_bytes')
            if ptxas_cubin:
                cubin_bytes = ptxas_cubin
            else:
                cubin_bytes = emit_cubin(desc)
        else:
            cubin_bytes = emit_cubin(desc)

    return cubin_bytes


def _select_capmerc(ptxas_meta: dict | None, kernel_gprs: int) -> bytes | None:
    """Return ptxas's capmerc for the emitter's ELF section sizing, or None.

    Always returns ptxas's capmerc when available. The capmerc section size
    must match ptxas's output — our 146-byte generated capmerc is too small
    for multi-page kernels (ptxas may generate 300+ bytes). Using ptxas's
    capmerc ensures correct section sizing; _patch_ptxas_metadata then patches
    byte[8] for GPR count. Hardware does not enforce the 0x2000 capability
    bit for register access (verified: ptxas kernels with 28 GPRs lack 0x2000).
    """
    # Use ptxas's capmerc when available — it has the correct instruction
    # class encoding and multi-page structure. Our SASS uses the same
    # opcodes as ptxas (just different scheduling/NOP count), so ptxas's
    # capmerc class descriptors are a valid superset of ours.
    # Patch byte[8] (GPR count) to match our actual register usage.
    if ptxas_meta and 'capmerc' in ptxas_meta:
        cm = bytearray(ptxas_meta['capmerc'])
        if len(cm) >= 9:
            cm[8] = max(cm[8], kernel_gprs)
        return bytes(cm)
    return None


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
                # Use ptxas capmerc anyway (hardware doesn't enforce reg_count)
                # but patch byte[8] to our actual GPR count.
                patch = bytearray(ptxas_cap[:sh_size])
                if len(patch) < sh_size:
                    patch.extend(bytearray(sh_size - len(patch)))
                if len(patch) > 8:
                    patch[8] = max(patch[8], min_gprs)
                data[sh_offset_val:sh_offset_val + sh_size] = patch
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


def _patch_ptxas_capmerc_gprs(capmerc: bytes | None, num_gprs: int) -> bytes | None:
    """Patch ptxas capmerc GPR count to match our native allocation."""
    if capmerc is None:
        return None
    cm = bytearray(capmerc)
    # ALLOC-1: the previous floor of 8 was conservative — PTXAS
    # declares as few as 7 for small tensor kernels (dmma_zero).
    # Use the actual num_gprs without an artificial floor.  SM_120
    # hardware has no documented minimum GPR declaration beyond 2.
    cm[8] = max(num_gprs, 2)
    return bytes(cm)


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
        # Fix Forge-generated PTX: mov.f32 %fN, 0 → mov.f32 %fN, 0f00000000
        # (ptxas rejects bare '0' as a float immediate)
        import re as _ptx_re
        ptx_ascii = _ptx_re.sub(
            r'mov\.f32\s+(%f\d+),\s*0\b',
            r'mov.f32 \1, 0f00000000', ptx_ascii)
        ptx_ascii = _ptx_re.sub(
            r'mov\.f64\s+(%fd?\d+),\s*0\b',
            r'mov.f64 \1, 0d0000000000000000', ptx_ascii)
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
        # Store full ptxas cubin for fallback
        for kname in result:
            result[kname]['cubin_bytes'] = data
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
