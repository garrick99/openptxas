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


def _canonicalize_scalar_ldg(fn: Function) -> None:
    """PTXAS-R25.3: rewrite scalar-LDG (ld.global from a raw ld.param.u64
    pointer with no intervening address arithmetic) into the
    offset-form shape that the existing downstream lowering already
    handles correctly.

    For each matching LDG, insert a per-lane zero-offset address chain
    and change the LDG's address operand to the chain's final register:

        mov.u32     %_r25c_tid_N,  %tid.x
        and.b32     %_r25c_and0_N, %_r25c_tid_N, 0
        cvt.u64.u32 %_r25c_off_N,  %_r25c_and0_N
        shl.b64     %_r25c_offs_N, %_r25c_off_N, 2
        add.u64     %_r25c_addr_N, %<raw_param>, %_r25c_offs_N
        ld.global.X %dest, [%_r25c_addr_N]

    The per-lane source is `%tid.x`; `%tid & 0` yields 0 per-lane so
    the effective address equals the raw param.  The `add.u64` makes
    the resulting addr register classify as an offset register under
    rule #25, bypassing the raw-pointer path in isel.  This matches
    the PTX shape of `kCandidate` which is empirically proven to pass.
    Narrow scope: fires only when the exact scalar-LDG class is
    matched; other LDG forms (already-offset-arithmetic, ld.param with
    u64_def_count>1 shapes, etc.) are untouched.
    """
    from ptx.ir import RegOp, ImmOp, MemOp, ScalarKind, TypeSpec, RegDecl

    param_regs: set[str] = set()
    offset_regs: set[str] = set()
    for bb in fn.blocks:
        for inst in bb.instructions:
            if (inst.op == 'ld' and 'param' in inst.types
                    and any(t in ('u64', 's64', 'b64') for t in inst.types)
                    and isinstance(inst.dest, RegOp)):
                param_regs.add(inst.dest.name)
            if inst.op in ('add', 'mul', 'mad', 'cvt', 'shl') and isinstance(inst.dest, RegOp):
                offset_regs.add(inst.dest.name)

    if not param_regs:
        return

    def _is_scalar_ldg(inst: Instruction) -> bool:
        if inst.op != 'ld' or 'global' not in inst.types:
            return False
        if not inst.srcs:
            return False
        s0 = inst.srcs[0]
        if not isinstance(s0, MemOp):
            return False
        base = s0.base if s0.base.startswith('%') else f'%{s0.base}'
        return base in param_regs and base not in offset_regs

    counter = 0
    transformed = False
    for bb in fn.blocks:
        new_instrs: list[Instruction] = []
        for inst in bb.instructions:
            if _is_scalar_ldg(inst):
                k = counter
                counter += 1
                base = inst.srcs[0].base
                if not base.startswith('%'):
                    base = '%' + base
                off_imm = inst.srcs[0].offset if isinstance(inst.srcs[0].offset, int) else 0
                tid_name  = f'_r25c_tid_{k}'
                and_name  = f'_r25c_and0_{k}'
                off_name  = f'_r25c_off_{k}'
                addr_name = f'_r25c_addr_{k}'
                # Register decls: add once per name.
                fn.reg_decls.append(RegDecl(TypeSpec(ScalarKind.B, 32), tid_name, 1))
                fn.reg_decls.append(RegDecl(TypeSpec(ScalarKind.B, 32), and_name, 1))
                fn.reg_decls.append(RegDecl(TypeSpec(ScalarKind.U, 64), off_name, 1))
                fn.reg_decls.append(RegDecl(TypeSpec(ScalarKind.U, 64), addr_name, 1))
                # RegDecl.names auto-appends '0' iff the base name does
                # not already end in a digit; our `_r25c_<role>_<K>`
                # names always end in a digit (K), so the RegOp name
                # equals the RegDecl name with a '%' prefix.
                tid_r  = RegOp(f'%{tid_name}')
                and_r  = RegOp(f'%{and_name}')
                off_r  = RegOp(f'%{off_name}')
                addr_r = RegOp(f'%{addr_name}')
                # mov.u32 %tid_r, %tid.x
                new_instrs.append(Instruction(
                    op='mov', types=['u32'], dest=tid_r,
                    srcs=[RegOp('%tid.x')]))
                # and.b32 %and_r, %tid_r, 0  (per-lane-computed zero)
                new_instrs.append(Instruction(
                    op='and', types=['b32'], dest=and_r,
                    srcs=[tid_r, ImmOp(0)]))
                # cvt.u64.u32 %off_r, %and_r  (zero-extend to u64)
                new_instrs.append(Instruction(
                    op='cvt', types=['u64', 'u32'], dest=off_r,
                    srcs=[and_r]))
                # add.u64 %addr_r, %<raw_param>, %off_r  (in_ptr + 0)
                new_instrs.append(Instruction(
                    op='add', types=['u64'], dest=addr_r,
                    srcs=[RegOp(base), off_r]))
                # Rewrite the LDG's address to use %addr_r.
                new_srcs = list(inst.srcs)
                new_srcs[0] = MemOp(base=addr_r.name, offset=off_imm)
                new_instrs.append(Instruction(
                    op=inst.op, types=list(inst.types), dest=inst.dest,
                    srcs=new_srcs, pred=inst.pred, neg=inst.neg,
                    mods=list(inst.mods)))
                # The synthetic addr_r is now also an offset_reg — add
                # so a later scalar-LDG in the same kernel (shouldn't
                # happen, but belt-and-suspenders) sees it as such.
                offset_regs.add(addr_r.name)
                transformed = True
            else:
                new_instrs.append(inst)
        bb.instructions = new_instrs
    return


def _r31_rename_inplace_u64_redefine_across_exit(fn: Function) -> None:
    """PTXAS-R31: split `add.u64 %rdN, %rdN, X` into a fresh-dest form when
    `%rdN` is a u64 param AND the redefine is reached only after a
    predicated control-flow instruction (``@!%p ret`` / ``@%p bra``).

    R30 proof: the unsafe shape

        ld.param.u64 %rdN, [p];
        ... @!%p0 ret; ...
        add.u64 %rdN, %rdN, X;
        st.global.u32 [%rdN], ...;

    forces regalloc to see ``%rdN`` as GPR-resident (u64 has >1 def), which
    in turn routes isel's ``_select_ld_param`` into the "GPR direct" body
    ``LDC.64 R_pair, c[0][param_off]`` branch (isel.py:1392).  That body
    LDC.64 pre-EXIT + in-place ``IADD3`` post-EXIT + STG combination
    produces CUDA_ERROR_ILLEGAL_ADDRESS on SM_120.  The safe shape,
    proven by the ``offset_distinct_dest`` repro, routes ``%rdN``
    through the UR path (preamble ``ULDCU.64`` + body ``IADD.64 R-UR``
    into a fresh pair + STG via fresh pair).

    The transform renames the redefine's dest to a fresh vreg and
    rewrites every subsequent use of ``%rdN`` (sources and MemOp bases)
    to the fresh vreg.  After the rename, ``%rdN`` has exactly one def
    (the ``ld.param.u64``), so regalloc keeps it on the UR path.
    """
    from ptx.ir import RegOp, MemOp, TypeSpec, ScalarKind, RegDecl

    # Collect u64 param destinations.
    u64_param_dests: set[str] = set()
    for bb in fn.blocks:
        for inst in bb.instructions:
            if (inst.op == 'ld' and 'param' in inst.types
                    and 'u64' in inst.types
                    and isinstance(inst.dest, RegOp)):
                u64_param_dests.add(inst.dest.name)
    if not u64_param_dests:
        return

    # Flatten all instructions in source order with block+index references.
    flat: list[tuple[int, int]] = []
    for bi, bb in enumerate(fn.blocks):
        for ii in range(len(bb.instructions)):
            flat.append((bi, ii))

    def _at(k: int):
        bi, ii = flat[k]
        return fn.blocks[bi].instructions[ii]

    added_decls: list[str] = []
    rename_seq = [0]
    # Params we've renamed — regalloc must force these onto the UR path.
    force_ur: set[str] = set()

    def _is_predicated_cf(inst) -> bool:
        return inst.pred is not None and inst.op in ('ret', 'bra', 'exit')

    def _is_inplace_u64_add(inst, pname: str) -> bool:
        return (inst.op == 'add'
                and any(t in ('u64', 's64', 'b64') for t in inst.types)
                and isinstance(inst.dest, RegOp)
                and inst.dest.name == pname
                and len(inst.srcs) >= 1
                and isinstance(inst.srcs[0], RegOp)
                and inst.srcs[0].name == pname)

    for pname in sorted(u64_param_dests):
        # Find the param load position (first def) — flat index.
        load_k = None
        for k in range(len(flat)):
            inst = _at(k)
            if (inst.op == 'ld' and 'param' in inst.types
                    and isinstance(inst.dest, RegOp)
                    and inst.dest.name == pname):
                load_k = k
                break
        if load_k is None:
            continue

        # Scan: require a predicated CF before the in-place redefine.
        saw_pred_cf = False
        redefine_k = None
        for k in range(load_k + 1, len(flat)):
            inst = _at(k)
            if _is_predicated_cf(inst):
                saw_pred_cf = True
                continue
            if _is_inplace_u64_add(inst, pname):
                if saw_pred_cf:
                    redefine_k = k
                break  # any in-place redefine (pred_cf or not) terminates scan
            # Any other writer of pname ends the scan — complex pattern.
            if (isinstance(inst.dest, RegOp)
                    and inst.dest.name == pname
                    and inst.op != 'ld'):
                break

        if redefine_k is None:
            continue

        # Allocate a fresh vreg name.  Ends with a digit so RegDecl.names
        # returns the exact name (see ir.py:188-193).
        bare = pname.lstrip('%')
        new_name = f'%__r31_{bare}_{rename_seq[0]}'
        rename_seq[0] += 1
        added_decls.append(new_name)
        force_ur.add(pname)

        # Rewrite redefine dest.
        r_inst = _at(redefine_k)
        r_inst.dest = RegOp(new_name)

        # Rewrite every subsequent read of pname to new_name.
        # Stop at a later writer of pname (unlikely after rename but guard).
        for k in range(redefine_k + 1, len(flat)):
            inst = _at(k)
            if (isinstance(inst.dest, RegOp)
                    and inst.dest.name == pname
                    and inst.op != 'ld'):
                break
            new_srcs = []
            for src in inst.srcs:
                if isinstance(src, RegOp) and src.name == pname:
                    new_srcs.append(RegOp(new_name))
                elif isinstance(src, MemOp) and src.base == pname:
                    new_srcs.append(MemOp(base=new_name, offset=src.offset))
                else:
                    new_srcs.append(src)
            inst.srcs = new_srcs

    # Declare the fresh vregs as .reg .u64 entries so regalloc sees them.
    # Also record the SET of original u64 params we renamed across a predicated
    # EXIT.  regalloc reads `fn._r31_force_ur_params` to OVERRIDE the R22
    # misaligned-addr-arith exclusion: after the rename, these params have
    # exactly one def (the `ld.param.u64`) and exactly one use as an
    # `add.u64` src — the UR path (preamble `ULDCU.64` + body
    # `IADD.64 R-UR`) is the proven-safe lowering per the R30 repro, so
    # we must not let R22 push them back onto the GPR-direct-LDC.64 path.
    if added_decls:
        u64_type = TypeSpec(kind=ScalarKind.U, width=64)
        for nm in added_decls:
            bare = nm.lstrip('%')
            fn.reg_decls.append(RegDecl(type=u64_type, name=bare, count=1))
    if force_ur:
        fn._r31_force_ur_params = force_ur


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
    _unsunk_pos = 0  # PTXAS-R09: tracks insertion point for unsunk params
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
            # PTXAS-R09: use _unsunk_pos to preserve original PTX order.
            # Without this, repeated insert(0) reverses the param loads,
            # which breaks aliased vregs (e.g. %rd2 used for both inp_len
            # and out — the LAST load in PTX order must win).
            keep.insert(_unsunk_pos, inst)
            _unsunk_pos += 1

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

    def _has_neg_sub(inst_list):
        """FORGE07: skip if-conversion when the body contains sub.u32/s32.

        OURS lowers `sub.u32 d, a, b` to `IADD3 d, a, -b, RZ` (encode_iadd3
        with negate_src1=True, byte 7 = 0x80 negation flag).  When this
        instruction is then patched with @!Pn, the resulting byte pattern
        (byte 1 = 0x80+pred_idx, byte 7 = 0x80) has UNDEFINED behavior on
        SM_120 — empirical evidence: PTXAS never emits this pattern (zero
        kernels in 144-kernel workbench corpus); OURS workbench corpus
        also never emitted it pre-FORGE07.  Hardware behavior in tests:
        the @!P guard appears to be ignored, so the negated branch fires
        unconditionally, corrupting results.

        Workaround: skip if-conversion when bodies contain sub, leaving the
        kernel as actual divergent BRA branches handled by the GPU's
        warp scheduler.
        """
        for inst in inst_list:
            if inst.op == 'sub' and inst.types and any(
                    t in ('u32', 's32', 'u64', 's64') for t in inst.types):
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
                                        or _pred_from_float_setp(bb.instructions, pred_name)
                                        or _has_neg_sub(true_body)
                                        or _has_neg_sub(false_body)):
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

    # PTXAS-R29.1 (retire R25.3 scalar-LDG canonicalization in favor of
    # ptxas-faithful direct LDC.64→LDG.E lowering).  ptxas ground truth
    # (R28.3 proof on `dual_ldg` kernel) shows scalar-LDG lowers via
    # `LDC.64 R_pair, c[0][param_off] → LDG.E desc[UR][R_pair.64]` —
    # no `tid & 0` offset-arith chain.  R29.1 moves the fix into a
    # coordinated classification: regalloc.py marks scalar-LDG-only
    # u64 params in `direct_ldc_params`; the existing isel tiny-direct
    # branch in `_select_ld_param` emits `LDC.64` straight into the
    # allocated GPR pair; `_select_ld_global` takes the `gpr_written`
    # branch and reads the already-loaded pair.  No UR preamble, no
    # dual producer, no per-lane chain.  `_canonicalize_scalar_ldg`
    # remains in the module as dead code for a rollback window.

    # PTXAS-R31: split in-place u64 param redefine across a predicated
    # EXIT so the param stays param-only (UR-routed) and the redefine
    # result lands in a fresh vreg.  R30 proof: when a u64 param is
    # loaded by body `LDC.64 R_pair` (isel "GPR direct" branch fires
    # because regalloc sees `u64_def_count > 1`) and then redefined
    # in-place by `add.u64 %rdN, %rdN, X` AFTER a predicated `@!%p ret`
    # or `@%p bra`, the STG that consumes %rdN reads an invalid address
    # and produces CUDA_ERROR_ILLEGAL_ADDRESS.  The safe path exists
    # (`offset_distinct_dest` passes): preamble `ULDCU.64` + body
    # `IADD.64 R-UR` into a fresh pair + STG via fresh pair.  This
    # transform turns the unsafe PTX shape into the safe one by
    # renaming the redefine dest and all subsequent uses.
    _r31_rename_inplace_u64_redefine_across_exit(fn)

    # FG39: consecutive shift merge.  When two shl.b32 instructions
    # operate in sequence (shl %B, %A, K1; shl %C, %B, K2) and %B is
    # not used elsewhere, merge to a single shl %C, %A, K1+K2.  Matches
    # PTXAS's peephole for the k200_shl_chain family.
    from ptx.ir import RegOp, ImmOp
    for bb in fn.blocks:
        instrs = bb.instructions
        i = 0
        while i < len(instrs) - 1:
            a, b = instrs[i], instrs[i + 1]
            if (a.op == 'shl' and b.op == 'shl'
                    and a.types == b.types  # same type (b32/b64)
                    and isinstance(a.dest, RegOp) and isinstance(b.dest, RegOp)
                    and len(a.srcs) >= 2 and len(b.srcs) >= 2
                    and isinstance(a.srcs[1], ImmOp) and isinstance(b.srcs[1], ImmOp)
                    and isinstance(b.srcs[0], RegOp)
                    and b.srcs[0].name == a.dest.name):
                # Check %B is not used elsewhere (only by the second shl)
                mid_name = a.dest.name
                used_elsewhere = False
                for other_inst in instrs:
                    if other_inst is a or other_inst is b:
                        continue
                    for src in (other_inst.srcs or []):
                        if isinstance(src, RegOp) and src.name == mid_name:
                            used_elsewhere = True
                            break
                    if used_elsewhere:
                        break
                if not used_elsewhere:
                    # Merge: shl %C, %A, K1+K2
                    merged_k = a.srcs[1].value + b.srcs[1].value
                    b.srcs = [a.srcs[0], ImmOp(merged_k)]
                    instrs.pop(i)  # remove first shl
                    continue
            i += 1

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
    # FG26 UR4 start is applied below, after _setp_only_params detection.
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
            # TE10: count how many setp instructions use each param
            _setp_use_count = {}
            for pname in _setp_only_params:
                cnt = 0
                for bb in fn.blocks:
                    for inst in bb.instructions:
                        if inst.op == 'setp' and any(
                                getattr(s, 'name', None) == pname
                                for s in (inst.srcs or [])):
                            cnt += 1
                _setp_use_count[pname] = cnt
            ctx._setp_use_count = _setp_use_count

    # FG26: start UR allocation at UR4 when address pair is co-located
    # AND the kernel has a setp-only u32 param AND no ctaid/ntid (no
    # S2UR that would consume a UR between setp and descriptor, colliding
    # with UR5).  PTXAS loads the setp param into UR4, uses it in
    # ISETP.R-UR, then reuses UR4:UR5 for the descriptor.
    _has_ctaid_ntid = any(
        inst.op == 'mov' and inst.srcs and hasattr(inst.srcs[0], 'name')
        and inst.srcs[0].name in ('%ctaid.x', '%ctaid.y', '%ctaid.z',
                                   '%ntid.x', '%ntid.y', '%ntid.z')
        for bb in fn.blocks for inst in bb.instructions if hasattr(inst, 'op'))
    # Additional guard: exactly one setp-only param with at most one setp
    # use (matching the TE10 LDCU.32 guard at isel.py line 1355).
    _fg26_setp_ok = False
    if _setp_only_params and len(_setp_only_params) == 1:
        _fg26_pname = next(iter(_setp_only_params))
        _fg26_setp_cnt = ctx._setp_use_count.get(_fg26_pname, 0)
        _fg26_setp_ok = (_fg26_setp_cnt <= 1)
    if (alloc.addr_pair_colocated and _fg26_setp_ok
            and not _has_ctaid_ntid):
        ctx._next_ur = 4
        ctx._fg26_ur4_start = True

    # Pre-scan: detect if kernel uses bar.sync (shared memory synchronization).
    # Kernels with bar.sync need preamble-only constant loads to avoid
    # LDC/LDCU poisoning of IADD.64-UR.
    ctx._has_bar_sync = any(
        inst.op == 'bar' and 'sync' in inst.types
        for bb in fn.blocks for inst in bb.instructions
    )
    # FG69: expose _has_ldg for isel SHF widening decision
    ctx._has_ldg = _has_ldg

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
            # PTXAS-R09: Never hoist an instruction that carries a block
            # label tag.  These are the first instructions of branch-target
            # blocks; hoisting them to the preamble causes the BRA fixup to
            # target the preamble instead of the block's semantic entry.
            _is_label_tagged = any(
                si.comment.startswith(f'// {lbl}:')
                for lbl in ctx.label_map
            )
            if idx >= s2r_pos and (is_ldcu or is_pre_bar_ldc) and not _is_label_tagged:
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
        # PTXAS-R32': detect the first predicated EXIT boundary.  The HW
        # scoreboard for a packed LDCU.128 posts only on the primary UR
        # `d` — consumers reading the HIGH half (d+2, d+3) do not stall
        # reliably for the pack when they sit in the post-EXIT body.
        # When the HIGH-half UR has an `IADD.64 R-UR` (opc 0xc35)
        # consumer AFTER a predicated `@!P0 EXIT`, the pack crosses the
        # control-flow boundary without a reliable wait and the
        # subsequent STG produces CUDA_ERROR_ILLEGAL_ADDRESS (the
        # `k_2p_offset` repro writes to `out_ptr + garbage` post-EXIT).
        # Keep the two LDCU.64s UNPACKED in that case so each arms its
        # own scoreboard slot.  All other WB-8 packings remain intact.
        _exit_idx_for_wb8 = None
        for _ei, _si in enumerate(instrs):
            _eopc = (_si.raw[0] | (_si.raw[1] << 8)) & 0xFFF
            _eguard = (_si.raw[1] >> 4) & 0xF
            if _eopc == 0x94d and _eguard != 0x7:
                _exit_idx_for_wb8 = _ei
                break
        for j, d, b in ldcu64:
            if j in paired:
                continue
            if d % 4 != 0 or b % 16 != 0:
                continue
            partner_j = by_key.get((d + 2, b + 8))
            if partner_j is None or partner_j in paired:
                continue
            # R32' guard: scan for a post-EXIT consumer of the HIGH-half UR.
            # Only the R-UR form of IADD.64 (opc 0xc35) reads UR in b4.
            _high_ur_post_exit_consumer = False
            if _exit_idx_for_wb8 is not None:
                _high_ur = d + 2
                for _ci in range(_exit_idx_for_wb8 + 1, len(instrs)):
                    _csi = instrs[_ci]
                    _copc = (_csi.raw[0] | (_csi.raw[1] << 8)) & 0xFFF
                    if _copc == 0xc35 and _csi.raw[4] == _high_ur:
                        _high_ur_post_exit_consumer = True
                        break
            if _high_ur_post_exit_consumer:
                continue  # keep both LDCU.64s unpacked
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

    # TE20-A: pre-scheduler 0xc11 replacement.  Run BEFORE scheduling so the
    # GPR latency enforcement pass sees the 0xc11 pair (no IMAD.WIDE→IADD.64-UR
    # dependency = no stale NOP insertion).  This is the key pipeline reordering
    # that unlocks the 0xc11 rollout beyond the 13 byte-exact kernels.
    _ur_act_sr_pre = getattr(ctx, '_ur_activation_sr', None)
    if _ur_act_sr_pre is None:
        from sass.encoding.iadd3_ur import encode_iadd3_ur_lo, encode_iadd3_ur_hi
        from sass.scoreboard import _get_src_regs as _sc_src
        _pre_new = []
        _pre_skip = False
        for _pi in range(len(body_instrs)):
            if _pre_skip:
                _pre_skip = False
                continue
            _psi = body_instrs[_pi]
            _popc = struct.unpack_from('<Q', _psi.raw, 0)[0] & 0xFFF
            if _popc == 0x825 and _pi + 1 < len(body_instrs):
                _pnext = body_instrs[_pi + 1]
                _pnext_opc = struct.unpack_from('<Q', _pnext.raw, 0)[0] & 0xFFF
                if _pnext_opc == 0xc35 and _pnext.raw[3] == _psi.raw[2]:
                    # Safety check: IMAD.WIDE dest pair must not be read by
                    # any instruction other than the IADD.64-UR consumer.
                    # Use b3 and b4 for GPR reads; b8 is UR for STG/LDG
                    # (not GPR) so exclude it to avoid false UR/GPR aliasing.
                    #
                    # FG23: when IADD.64-UR dest == IMAD.WIDE dest (co-located
                    # address pair from ALLOC-SUBSYS-2), STG/LDG b3 reads the
                    # 0xc11 OUTPUT (the computed address), not the IMAD.WIDE
                    # intermediate. Exclude memory-instruction address reads
                    # (b3 for STG/LDG) from the check in this case.
                    _wide_dest = _psi.raw[2]
                    _wide_pair = {_wide_dest, _wide_dest + 1}
                    _iadd_dest = _pnext.raw[2]
                    # FG23: co-located address pair from ALLOC-SUBSYS-2.
                    # Only R2:R3 (the reserved pair) qualifies.  Non-address
                    # IMAD.WIDE (HMMA/DMMA data widening) at other pairs must
                    # NOT bypass the safety check.
                    _colocated = (alloc.addr_pair_colocated
                                  and _iadd_dest == _wide_dest
                                  and _wide_dest == 2)
                    _safe = True
                    for _pk in range(_pi + 2, len(body_instrs)):
                        _sk = body_instrs[_pk]
                        _sk_opc = struct.unpack_from('<Q', _sk.raw, 0)[0] & 0xFFF
                        # GPR source registers: b3 (always), b4 (for STG data,
                        # ALU src1); b8 only for non-memory ALU ops
                        _reads = set()
                        if _sk_opc in (0x986, 0x981):
                            # STG/LDG: b3 is address pair lo.
                            # When co-located, b3 reads the 0xc11 output — safe.
                            if not _colocated:
                                _reads.add(_sk.raw[3])
                            # STG: b4 is data register (always a real GPR read)
                            if _sk_opc == 0x986:
                                _reads.add(_sk.raw[4])
                        elif _sk_opc in (0x7ac, 0x94d, 0x947, 0x918):
                            _reads.add(_sk.raw[3])
                        else:
                            _reads.add(_sk.raw[3])
                            _reads.add(_sk.raw[4])
                            _reads.add(_sk.raw[8])
                        if _reads & _wide_pair:
                            _safe = False
                            break
                    if _safe:
                        _lo_dest = _pnext.raw[2]
                        _lo_src = _psi.raw[3]
                        # ALLOC-2: check if we can reuse the dying source register.
                        # PTXAS pattern: lo_dest = src-1, hi_dest = src.
                        # Only works when lo_dest + 1 == lo_src (consecutive pair).
                        if _lo_dest + 1 == _lo_src:
                            _hi_dest = _lo_src  # reuse dying source
                        else:
                            _hi_dest = _lo_dest + 1  # standard consecutive pair
                        _pre_new.append(SassInstr(
                            encode_iadd3_ur_lo(_lo_dest, _lo_src, _pnext.raw[4]),
                            f'IADD3.UR R{_lo_dest}, R{_lo_src}, UR{_pnext.raw[4]}  // ALLOC: addr lo'))
                        _pre_new.append(SassInstr(
                            encode_iadd3_ur_hi(_hi_dest, _lo_src, _pnext.raw[4] + 1),
                            f'IADD3.UR.X R{_hi_dest}, R{_lo_src}, UR{_pnext.raw[4]+1}  // ALLOC: addr hi'))
                        _pre_skip = True
                        continue
            _pre_new.append(_psi)
        body_instrs = _pre_new

    # 4. Schedule: reorder for LDG latency, then assign ctrl via scoreboard
    raw_instrs = preamble + body_instrs
    reordered = schedule(raw_instrs)

    # PTXAS-R38: post-EXIT S2R -> immediate IMAD.SHL.U32 gap insertion.
    #
    # Proven hazard (R38 probe on s2_fail, all 5 of 6 tested positions
    # fail, only position between S2R and IMAD.SHL works):
    #
    #   @!P0 EXIT
    #   S2R Rx, SR_CTAID_X
    #   IMAD.SHL.U32 Ry, Rx, imm, RZ     <-- reads Rx written by S2R
    #
    # Even though scoreboard rbar=0x03 on the IMAD.SHL nominally covers
    # the S2R's slot 0x31, SM_120 hardware does NOT reliably honor that
    # dependency across a predicated EXIT — IMAD.SHL reads a stale Rx
    # and produces garbage, which propagates into the pair-build /
    # IADD.64 R-UR / STG chain as `out_ptr + garbage` and crashes with
    # CUDA_ERROR_ILLEGAL_ADDRESS.  The probe confirmed:
    #   * +0..+4 NOPs before IADD.64 R-UR: FAIL
    #   * NOP before S2R CTAID: FAIL
    #   * NOP after IMAD.SHL / after MOV lo / after MOV hi: FAIL
    #   * NOP (or BSYNC/MEMBAR/WARPSYNC) BETWEEN S2R and IMAD.SHL: PASS
    #
    # Passing kernels (G4) already have a natural gap here (scheduler
    # places a descriptor LDCU between S2R CTAID and IMAD.SHL in G4),
    # so this rule only fires when no natural gap exists.  The
    # inserted NOP goes through `assign_ctrl` normally on the next
    # pass, so its ctrl word is consistent with the surrounding flow.
    from sass.encoding.sm_120_opcodes import encode_nop as _r38_encode_nop
    _r38_patched = []
    _r38_seen_pexit = False
    for _i, _si in enumerate(reordered):
        _r38_patched.append(_si)
        _r38_opc = (_si.raw[0] | (_si.raw[1] << 8)) & 0xFFF
        _r38_guard = (_si.raw[1] >> 4) & 0xF
        if _r38_opc == 0x94d and _r38_guard != 0x7:
            _r38_seen_pexit = True
            continue
        if _r38_seen_pexit and _r38_opc == 0x919:
            _r38_dest = _si.raw[2]
            if _i + 1 < len(reordered):
                _r38_nxt = reordered[_i + 1]
                _r38_nopc = (_r38_nxt.raw[0] | (_r38_nxt.raw[1] << 8)) & 0xFFF
                # Narrow: post-EXIT S2R GPR consumer classes PROVEN sensitive.
                # R38 probe (s2_fail): IMAD.SHL.U32 (0x824) consumer reading
                # S2R dest at b3 produces stale read → CUDA_ERROR_ILLEGAL_ADDRESS.
                # R39 extension probe confirmed the SAME hazard for two more
                # opcodes that also read their first GPR operand at b3:
                #   * 0x835 UIADD (SM_120 "SR-derived" add with imm32) — out=
                #     garbage on `mov ctaid; add imm` (probe p2_add_imm FAIL
                #     without gap, PASS with gap).
                #   * 0x812 LOP3.LUT with imm32 — out=garbage on `mov ctaid;
                #     or imm` (probe p4_or FAIL without gap, PASS with gap).
                # All three share the encoding "b3 = src0 GPR reading the
                # S2R dest immediately".  Non-SHL multiply paths (mul.lo)
                # lower via S2UR (UR dest) + IMAD R-UR instead and do NOT
                # hit the hazard.  Passing kernels (G4 etc.) have a natural
                # gap instruction between S2R and consumer and are
                # unaffected by this rule.
                _R39_SENSITIVE_OPCODES = {0x824, 0x835, 0x812}
                if _r38_nopc in _R39_SENSITIVE_OPCODES and _r38_nxt.raw[3] == _r38_dest:
                    _r38_patched.append(SassInstr(
                        _r38_encode_nop(),
                        'NOP  // PTXAS-R38/R39 post-EXIT S2R->ALU gap'))
    reordered = _r38_patched

    # PTXAS-R48: post-ISETP predicate-handoff gap insertion.
    #
    # Proven hazard (R48 probe on k300_nasty_pred_xor after the FG56 LOP3
    # src0 rename is applied, isolating tids 17..31 @P1 LOP3 failure):
    #
    #   ISETP.IMM P1, PT, R0, 0x10, PT                 (writes P1)
    #   @P1 LOP3.LUT R5, R5, 0x55, RZ, 0x3c, !PT       (reads P1)
    #
    # On SM_120, the @P consumer reads a stale predicate when directly
    # adjacent to the ISETP producer.  The R48 probe swept:
    #   * NOP inserted between: PASS (all 32 tids correct)
    #   * stall bump on producer (1,2,4,8,15,32): FAIL
    #   * stall bump on consumer (1,2,4,8,15,32): FAIL
    #   * wdep toggles on consumer (0x3e, 0x00): FAIL
    #   * rbar[14:10] on consumer 0x02..0x07: PASS (acts as implicit gap
    #     via wait on a never-set barrier; not a semantic scoreboard fix)
    #
    # The only true fix is a single-instruction gap between ISETP and an
    # immediately-following instruction whose guard index matches the
    # ISETP's pred-dest.  Ptxas naturally schedules an unrelated
    # instruction in between, so this rule only fires when no natural
    # gap exists.  Applies to all ISETP flavors (IMM, R-UR, R-R) and any
    # @Pk consumer class (ALU, EXIT, etc.).
    _R48_ISETP_OPCS = {0x80c, 0xc0c, 0x20c}
    _r48_patched = []
    for _i, _si in enumerate(reordered):
        _r48_patched.append(_si)
        _r48_opc = (_si.raw[0] | (_si.raw[1] << 8)) & 0xFFF
        if _r48_opc not in _R48_ISETP_OPCS:
            continue
        if _i + 1 >= len(reordered):
            continue
        _r48_pred_dst = (_si.raw[10] >> 1) & 0x7
        _r48_nxt = reordered[_i + 1]
        _r48_nxt_guard = (_r48_nxt.raw[1] >> 4) & 0xF
        _r48_nxt_idx = _r48_nxt_guard & 0x7
        _r48_nxt_pred = _r48_nxt_idx != 0x7
        if _r48_nxt_pred and _r48_nxt_idx == _r48_pred_dst:
            _r48_patched.append(SassInstr(
                _r38_encode_nop(),
                'NOP  // PTXAS-R48 post-ISETP->@P gap'))
    reordered = _r48_patched

    # The preamble (LDC R1) has hardcoded ctrl from ptxas.
    # Only assign scoreboard ctrl to body instructions (after preamble).
    n_preamble = len(preamble)
    preamble_instrs = list(reordered[:n_preamble])
    body_reordered = list(reordered[n_preamble:])
    body_scheduled = assign_ctrl(list(reordered[n_preamble:]))

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
        # TEMPLATE-ENGINE-6A: unified family model for atom.xor variants.
        # Loads the family_atom_ur.json spec which contains both Variant A
        # (direct SR) and Variant B (tid+constant).  Selects the correct
        # variant based on _ur_act_add.  Falls back to inline bytes.
        # ---------------------------------------------------------------
        _spec_ok = False
        try:
            import json as _json
            from pathlib import Path as _Path
            _spec_dir = _Path(__file__).resolve().parent.parent / 'tools' / 'template_engine' / 'generated'
            _family_file = _spec_dir / 'family_atom_ur.json'

            if _family_file.exists():
                _fam = _json.loads(_family_file.read_text(encoding='utf-8'))
                # Select variant.
                # - AT06: imm_data_K1 (selector "atom_imm == 1") for tid-guarded
                #   atom.add.u32 K=1 (ctx._ur_activation_atom_imm == 1).
                # - AT10: imm_data_K1_no_tid_guard (selector "atom_imm == 1
                #   and no_tid_guard") for the no-prelude sibling
                #   (ctx._ur_activation_atom_no_tid_guard set in addition).
                # When _ur_activation_atom_imm is set, an atom-imm variant is
                # selected; AT10 takes priority over AT06 when both flags are set.
                _ur_act_imm = getattr(ctx, '_ur_activation_atom_imm', None)
                _ur_act_no_tid = getattr(ctx, '_ur_activation_atom_no_tid_guard', False)
                _variant = None
                # AT10 first (most specific selector)
                for _v in _fam['variants']:
                    if ('no_tid_guard' in _v['selector']
                            and _ur_act_imm == 1 and _ur_act_no_tid):
                        _variant = _v; break
                if _variant is None:
                    for _v in _fam['variants']:
                        if 'atom_imm == 1' in _v['selector'] and _ur_act_imm == 1:
                            # Skip the no-tid-guard variant unless explicitly
                            # requested (its selector also contains atom_imm == 1).
                            if 'no_tid_guard' in _v['selector']:
                                continue
                            _variant = _v; break
                if _variant is None:
                    for _v in _fam['variants']:
                        if 'atom_imm' in _v['selector']:
                            continue  # skip imm variants when not triggered
                        if 'add != 0' in _v['selector'] and _ur_act_add != 0:
                            _variant = _v; break
                        if 'add == 0' in _v['selector'] and _ur_act_add == 0:
                            _variant = _v; break

                if _variant is not None:
                    # AT02: per-op byte overrides for the bounded atom-UR
                    # template expansion.  The atom.xor baseline bytes are
                    # in the JSON; atom.max.u32 / atom.min.u32 reuse the
                    # same template with two positions rewritten.  Values
                    # are verified byte-for-byte against PTXAS SM_120
                    # ground truth (k100_atom_xor/max/min).
                    _ur_act_op = getattr(ctx, '_ur_activation_atom_op', 'xor')
                    # (role, byte_offset, byte_length, value)
                    # AT02 overrides apply to the atom.xor template variants
                    # (direct_sr, tid_plus_constant).  AT06 adds the
                    # imm_data_K1 variant which carries its own ATOMG bytes
                    # by default (atom.add.u32 K=1 = b9/b10/b11 = e1/12/0c)
                    # and is parameterized at byte_offset=9 length=3.
                    _AT02_OVERRIDES = {
                        'max': [
                            ('UMOV_UR5_UR0',                9, 2, bytes([0x40, 0x01])),
                            ('ATOMG_XOR_R0_R2_data5_desc_UR6', 10, 2, bytes([0x12, 0x0d])),
                        ],
                        'min': [
                            ('UMOV_UR5_UR0',                9, 2, bytes([0x00, 0x01])),
                            ('ATOMG_XOR_R0_R2_data5_desc_UR6', 10, 2, bytes([0x92, 0x0c])),
                        ],
                        # imm_data_K1 variant uses its own role keys.  The
                        # JSON default bytes are atom.add (K=1).  Future
                        # ops (or/and/min/max with K=1) would add overrides
                        # at role 'ATOMG_ADD_R0_R2_R5' here.
                    }
                    _overrides = _AT02_OVERRIDES.get(_ur_act_op, [])
                    _T = []
                    for _si in _variant['instructions'][1:]:  # skip preamble S2R
                        _raw = bytearray(bytes.fromhex(_si['bytes']))
                        for _p in _si.get('params', []):
                            if _p['name'] == 'add_imm_K':
                                for _bi in range(_p['byte_length']):
                                    _raw[_p['byte_offset'] + _bi] = (_ur_act_add >> (8 * _bi)) & 0xFF
                        # Apply AT02 per-op overrides matched by role.
                        for _orole, _ooff, _olen, _oval in _overrides:
                            if _si['role'] == _orole and len(_oval) == _olen:
                                for _bi in range(_olen):
                                    _raw[_ooff + _bi] = _oval[_bi]
                        _T.append(SassInstr(bytes(_raw), f"{_si['role']}  // TE6/AT02:{_ur_act_op}"))
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

    # TPL01/TPL05: non-atom whole-kernel template registry.  When a kernel
    # exactly matches one of the proven-safe shapes, replace the entire body
    # with a PTXAS-extracted byte sequence.  Bypasses isel + scheduler +
    # allocator entirely for that kernel shape — the only safe way to land
    # IADD.64-pair-write substitutions given allocator pair-aliasing
    # (proved by IM03 HARD BAIL).
    #
    # Each entry: (kernel_name, expected_param_count, json_filename,
    #              dispatcher_tag).
    _TPL_NON_ATOM_REGISTRY = [
        ('k100_dual_load',       4, 'non_atom_dual_load.json',       'TPL01'),
        ('k300_nasty_zero_init', 2, 'non_atom_nasty_zero_init.json', 'TPL05'),
        ('r1_scatter_add',       2, 'non_atom_scatter_add.json',     'TPL09'),
        ('r1_running_xor',       2, 'non_atom_running_xor.json',     'TPL13'),
        ('r1_multi_stage',       2, 'non_atom_multi_stage.json',     'TPL17'),
        # MPT01: first MP02-aware predicate-body template.  Tagged with
        # 'TPL' prefix so the existing post-EXIT b9 / FG33 ctrl-byte
        # rewrite skips already cover it without further pipeline patches.
        ('k100_pred_arith',      2, 'non_atom_pred_arith.json',      'TPL/MPT01'),
        # MPT05: second MP02-aware predicate-body template.  Same dispatch
        # mechanism; covers k200_double_guard's distinct @P1 + @!P0
        # predicate pattern.
        ('k200_double_guard',    2, 'non_atom_double_guard.json',    'TPL/MPT05'),
        # MPT09: third MP02-aware predicate-body template (k300_pred3,
        # 3-setp/3-@P with PTXAS's 3-distinct-predicate allocation).
        ('k300_pred3',           2, 'non_atom_pred3.json',           'TPL/MPT09'),
        # MPT13: fourth MP02-aware predicate-body template (k100_setp_combo,
        # 2-setp/2-@P; PTXAS reuses TID register R0 directly and uses the
        # MPT01-style SEL+@P-UIADD predicate-mux pattern with imm operands).
        ('k100_setp_combo',      2, 'non_atom_setp_combo.json',      'TPL/MPT13'),
        # MPT17: fifth MP02-aware predicate-body template (k300_nasty_multi_pred,
        # 5-setp/5-@P with PTXAS's aggressive predicate-slot reuse — P0 used
        # for entry/gt-4/gt-32, P2 used for gt-8/gt-48 — plus SEL+4x@P-UIADD
        # mux with R0=R3+10 pre-compute folding the first add into SEL).
        ('k300_nasty_multi_pred',2, 'non_atom_nasty_multi_pred.json','TPL/MPT17'),
        # MPT22: sixth MP02-aware predicate-body template (k200_pred_chain,
        # 4-setp/4-@P all on PTX %p1; PTXAS reallocates to {P0,P2,P1,P2}
        # with P0 reused entry/gt-8 and P2 reused gt-4/gt-32, plus SEL.IMM
        # folding the first +1 add via predicate-mux).
        ('k200_pred_chain',      2, 'non_atom_pred_chain.json',      'TPL/MPT22'),
        # MPT26: seventh MP02-aware predicate-body template (w1_div_multi_guard,
        # 4-setp/4-@P alternating PTX P1/P2 with thresholds 8/16/32/48; PTXAS
        # opcode-identical to MPT22 but reallocates name-alternating P1/P2
        # PTX setps to the same {P0,P2,P1,P2} pattern, with P0 reused
        # entry/gt-16 and P2 reused gt-8/gt-48; SEL.IMM folds gt-8 +1 add).
        ('w1_div_multi_guard',   2, 'non_atom_div_multi_guard.json', 'TPL/MPT26'),
        # MPT30: eighth MP02-aware predicate-body template (w2_deep_pred,
        # 5-setp/5-@P alternating PTX P1/P2 with thresholds 2/6/12/24/48;
        # PTXAS reallocates to {P2,P0,P1,P0-reuse,P2-reuse} = 3 distinct
        # slots with aggressive P0/P2 reuse, plus SEL.IMM folding the gt-2
        # +1 add via predicate-mux and 4x@P-UIADD chain).
        ('w2_deep_pred',         2, 'non_atom_deep_pred.json',       'TPL/MPT30'),
        # MPT34: ninth MP02-aware predicate-body template (k200_nested_pred,
        # first conditional-setp pattern: setp+@p1+@p1-conditional-setp+@p2.
        # PTXAS uses ISETP-with-@P-guard byte encoding at [7], R0=R3+10
        # pre-compute + SEL.REG fold of @p1 add 10, and an uncommon 0x81c
        # ALU helper. delta=-2 (under-emit) lowered byte-exact.
        ('k200_nested_pred',     2, 'non_atom_nested_pred.json',     'TPL/MPT34'),
        # MPT38: tenth MP02-aware predicate-body template (k300_nasty_pred_nest3,
        # extended conditional-setp chain: 2 chained @P-conditional setps with
        # P1 reuse, 2x 0x81c uncommon-ALU helpers, SEL.IMM fold of +1 add,
        # 2x@P-UIADD chain. Direct extension of MPT34. delta=-3 lowered
        # byte-exact. 0x81c FG-2.3 allowlist already in place from MPT34).
        ('k300_nasty_pred_nest3',2, 'non_atom_nasty_pred_nest3.json','TPL/MPT38'),
        # IMNMX01-04: re-enabled MPT42 template now that 0x848 IMNMX.IMM is
        # properly modeled in _OPCODE_META + LOP3->IMNMX and IMNMX->IMNMX
        # added to _FORWARDING_SAFE_PAIRS for the clamp-idiom RAW chain.
        ('r1_minmax',            2, 'non_atom_minmax.json',          'TPL/MPT42'),
    ]
    if (sm_version >= 120
            and not _ur_activation):  # never override an active atom-template kernel
        for _tpl_kn, _tpl_pc, _tpl_fn, _tpl_tag in _TPL_NON_ATOM_REGISTRY:
            if (fn.name == _tpl_kn
                    and len(getattr(fn, 'params', []) or []) == _tpl_pc):
                try:
                    import json as _json
                    from pathlib import Path as _Path
                    _tpl_file = (_Path(__file__).resolve().parent.parent
                                 / 'tools' / 'template_engine' / 'generated'
                                 / _tpl_fn)
                    if _tpl_file.exists():
                        _spec = _json.loads(_tpl_file.read_text(encoding='utf-8'))
                        # Defense-in-depth: verify kernel name match in JSON.
                        if (_spec.get('kernel_name_match') == fn.name
                                and _spec.get('expected_param_count') == _tpl_pc):
                            _tpl_T = []
                            for _si in _spec['instructions']:
                                _raw = bytes.fromhex(_si['bytes'])
                                _tpl_T.append(SassInstr(
                                    _raw, f"{_si['role']}  // {_tpl_tag}"))
                            body_scheduled = []
                            _ur_activation = _tpl_T
                            if verbose:
                                print(f'[{_tpl_tag}] whole-kernel template applied for {fn.name}')
                            break
                except Exception:
                    pass  # fall through to normal lowering

    sass_instrs = preamble_instrs + _ur_activation + body_scheduled

    # TE31: DMMA zero-init ctrl-byte patch.  For the specific DMMA pattern
    # (all inputs zero), match PTXAS ctrl bytes exactly.  PTXAS uses:
    # - LDCU.64 (first non-preamble): wdep=0x31 (not 0x35)
    # - DMMA: wdep=0x31 (not 0x3e)
    # - STG: rbar=0x03 (not 0x0b, no LDG bit)
    # This is safe because DMMA zero-init has no LDG dependency.
    _has_dmma = any(struct.unpack_from('<Q', si.raw, 0)[0] & 0xFFF == 0x23f
                    for si in sass_instrs)
    if _has_dmma:
        for _di, _si in enumerate(sass_instrs):
            _dopc = struct.unpack_from('<Q', _si.raw, 0)[0] & 0xFFF
            if _dopc == 0x23f:  # DMMA — patch wdep from 0x3e to 0x31
                _db = bytearray(_si.raw)
                _dr = _db[13] | (_db[14] << 8) | (_db[15] << 16)
                _dc = _dr >> 1
                _dc = (_dc & ~(0x3F << 4)) | (0x31 << 4)  # wdep=0x31
                _dr = (_dc & 0x7FFFFF) << 1
                _db[13] = _dr & 0xFF; _db[14] = (_dr >> 8) & 0xFF; _db[15] = (_dr >> 16) & 0xFF
                sass_instrs[_di] = SassInstr(bytes(_db), _si.comment)
            elif _dopc == 0x986:  # STG — patch rbar: clear LDG bit (0x08)
                _db = bytearray(_si.raw)
                _dr = _db[13] | (_db[14] << 8) | (_db[15] << 16)
                _dc = _dr >> 1
                _rbar = (_dc >> 10) & 0x1F
                _rbar &= ~0x08  # clear LDG class bit
                _dc = (_dc & ~(0x1F << 10)) | (_rbar << 10)
                _dr = (_dc & 0x7FFFFF) << 1
                _db[13] = _dr & 0xFF; _db[14] = (_dr >> 8) & 0xFF; _db[15] = (_dr >> 16) & 0xFF
                sass_instrs[_di] = SassInstr(bytes(_db), _si.comment)
        # Patch first non-preamble LDCU.64: wdep 0x35→0x31
        for _di, _si in enumerate(sass_instrs):
            _dopc = struct.unpack_from('<Q', _si.raw, 0)[0] & 0xFFF
            if _dopc == 0x7ac and _si.raw[9] == 0x0a and _di > 0:
                _db = bytearray(_si.raw)
                _dr = _db[13] | (_db[14] << 8) | (_db[15] << 16)
                _dc = _dr >> 1
                _old_wdep = (_dc >> 4) & 0x3F
                if _old_wdep == 0x35:
                    _dc = (_dc & ~(0x3F << 4)) | (0x31 << 4)
                    _dr = (_dc & 0x7FFFFF) << 1
                    _db[13] = _dr & 0xFF; _db[14] = (_dr >> 8) & 0xFF; _db[15] = (_dr >> 16) & 0xFF
                    sass_instrs[_di] = SassInstr(bytes(_db), _si.comment)
                break

    # TE20: 0xc11 replacement now runs PRE-scheduler (see TE20-A above).
    # Post-scheduling pass removed — no longer needed.

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
                    # Hardware semantics: target = next_pc + offset_instrs * 16.
                    #
                    # FORGE61-64 (WRONG_BLOCK_SPLIT): a single PTX basic block
                    # must map to exactly one canonical SASS entry.  The label
                    # tag on the first emitted instruction of a block IS that
                    # canonical entry.  For blocks whose first PTX instruction
                    # is `bar.sync`, isel.py emits a BSYNC.RECONVERGENT (opcode
                    # 0x941) preamble ahead of BAR.SYNC (to carry the label tag
                    # for scoreboard/reconvergence accounting); for those blocks
                    # we add `+1` so the branch lands on the real BAR.SYNC past
                    # the BSYNC preamble.  For every other first instruction
                    # (notably `setp`/ISETP for a merge block that recomputes
                    # its guard predicate, e.g. if_merge_4 or while_cond_20),
                    # there is NO BSYNC preamble and an unconditional `+1`
                    # would skip the canonical entry, landing one instruction
                    # past it and leaving incoming edges consuming a stale
                    # predicate — the exact cause of the all-zero output on
                    # the FORGE61-64 tiled-matmul slice.
                    target_idx = actual_target_byte // 16
                    _tgt_is_bsync = (
                        0 <= target_idx < len(sass_instrs)
                        and ((sass_instrs[target_idx].raw[0]
                              | (sass_instrs[target_idx].raw[1] << 8)) & 0xFFF)
                            == 0x941
                    )
                    offset_instrs = rel_offset // 16 + (1 if _tgt_is_bsync else 0)
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
                    # Predicated BRA or unconditional backward BRA.
                    # PTXAS-R17: do NOT convert PT backward → @!P0.
                    # The old assumption (P0=false at back-edge) is wrong
                    # for while-loop back-edges where P0 was set true by
                    # the loop condition check.  Keep @PT (0x79) for
                    # unconditional backward BRAs.  NVIDIA ptxas also uses
                    # @PT for loop back-edges.
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

    # FG30-B: swap post-EXIT LDCU ordering for MIXED-eligible kernels.
    # PTXAS places addr base LDCU first, descriptor LDCU second.
    # Our scheduler places descriptor first.  Swap them.
    _fg30_eligible = getattr(ctx, '_fg26_ur4_start', False)
    if _fg30_eligible and sm_version >= 120:
        # Find the two post-EXIT LDCU.64 instructions
        _post_exit = False
        _post_ldcu_indices = []
        for i, si in enumerate(sass_instrs):
            opc = (si.raw[0] | (si.raw[1] << 8)) & 0xFFF
            guard = (si.raw[1] >> 4) & 0xF
            if opc == 0x94d and guard != 7:  # predicated EXIT
                _post_exit = True
            if _post_exit and opc == 0x7ac and si.raw[9] in (0x0a, 0x0c):
                _post_ldcu_indices.append(i)
        if len(_post_ldcu_indices) >= 2:
            _i1, _i2 = _post_ldcu_indices[0], _post_ldcu_indices[1]
            _si1 = sass_instrs[_i1]
            _si2 = sass_instrs[_i2]
            # Check: first is descriptor (UR4), second is addr base (UR6+)
            if _si1.raw[2] == 4 and _si2.raw[2] >= 6:
                # Swap: put addr base first, descriptor second
                # Both use b9=0x0a (PTXAS convention — no b9=0x0c needed)
                _p1 = bytearray(_si2.raw)
                _p2 = bytearray(_si1.raw)
                _p1[9] = 0x0a  # addr base: standard encoding
                _p2[9] = 0x0a  # descriptor: standard encoding
                sass_instrs[_i1] = SassInstr(bytes(_p1), _si2.comment + ' [FG30:swap]')
                sass_instrs[_i2] = SassInstr(bytes(_p2), _si1.comment + ' [FG30:swap]')
    else:
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
                    # Skip TPL templates: they carry per-LDCU b9 values
                    # verified against PTXAS ground truth; the post-EXIT
                    # b9=0x0c rewrite is for non-template kernels and
                    # would corrupt the template here.  Tag prefix `TPL`
                    # covers TPL01, TPL05, and future non-atom templates.
                    #
                    # PTXAS-R20 (FB-1 Phase A fix): the byte-9 flip upcasts
                    # LDCU.64 -> LDCU.128.  LDCU.128 requires the byte
                    # offset to be 16-byte aligned — encode_ldcu_128()
                    # asserts this.  The post-EXIT rewrite must mirror the
                    # encoder's alignment rule; applying it to an 8-byte-
                    # aligned offset (e.g. the 2nd u64 param at 0x388)
                    # produces a malformed LDCU.128, garbage UR high half,
                    # invalid IADD.64 store address, and CUDA_ERROR_
                    # ILLEGAL_ADDRESS on the subsequent STG.  Guard with
                    # qword-index parity: raw[5] is byte_offset/8, so
                    # byte_offset % 16 == 0 iff raw[5] is even.
                    if (si.raw[5] >= 0x70
                            and (si.raw[5] & 1) == 0
                            and 'deferred' not in si.comment
                            and 'TPL' not in si.comment):
                        patched = bytearray(si.raw)
                        patched[9] = 0x0c
                        sass_instrs[i] = SassInstr(bytes(patched), si.comment + ' [b9=0x0c]')
                        break
                    elif si.raw[5] >= 0x70 and (si.raw[5] & 1) != 0:
                        # Non-16-aligned deferred param: the old code would
                        # have upcast this to an invalid LDCU.128.  Leave
                        # it as a plain LDCU.64.  `break` preserves the
                        # original "rewrite only the first post-EXIT LDCU"
                        # shape so the alignment guard cannot accidentally
                        # rewrite a later, differently-placed LDCU instead.
                        break

    # FG65: HARD BAIL — HFMA2/FMUL.I substitution was too broad.
    # Guard (_fg26_ur4_start) admits 106+ kernels, not just the 5 targets.
    # Caused 12 BYTE_EXACT regressions + scheduling hazard violations.
    # The HFMA2_FP_INT subsystem requires kernel-specific gating based
    # on the exact opcode multiset diff, not just the FG26 flag.

    # FG56: bounded S2R R3->R0 rename for the 3 opcode-aligned MIXED kernels.
    # PTXAS assigns tid.x to R0.  Our allocator assigns R3 (ALLOC-SUBSYS-2
    # skips R0-R2).  This rename runs BEFORE FG29 so that R0 is marked
    # "occupied" and FG29 skips R0 normalization.
    # Admission: FG52-eligible (ISTP/IMAD reorder fired) + FG26 UR4 start.
    _fg56_fired = False
    if getattr(ctx, '_fg26_ur4_start', False) and sm_version >= 120:
        # Check if FG52 reorder pattern exists (ISTP at pos 6 after EXIT)
        _fg56_opcodes = [((si.raw[0] | (si.raw[1] << 8)) & 0xFFF) for si in sass_instrs]
        _fg56_has_istp_at_6 = False
        _post_exit_idx = None
        for _ri, _ropc in enumerate(_fg56_opcodes):
            if _ropc == 0x94d and ((sass_instrs[_ri].raw[1] >> 4) & 0xF) != 7:
                _post_exit_idx = _ri
                break
        if _post_exit_idx is not None:
            # Check: first post-EXIT LDCU, then body ALU region with
            # ISTP.I present (pre-FG52 pattern: [IMAD, LDCU, ISTP] or
            # post-FG52 pattern: [ISTP, IMAD, LDCU]).  Accept either.
            _scan = _post_exit_idx + 1
            while _scan < len(sass_instrs) and _fg56_opcodes[_scan] == 0x7ac:
                _scan += 1
            # Check for ISTP.I within 3 positions of the first body ALU
            for _look in range(_scan, min(_scan + 3, len(sass_instrs))):
                if _fg56_opcodes[_look] == 0x80c:
                    _fg56_has_istp_at_6 = True
                    break

        if _fg56_has_istp_at_6:
            # Verify R3 is S2R dest (tid.x) and rename R3->R0 globally
            _has_s2r_r3 = any(
                _fg56_opcodes[_ri] == 0x919 and sass_instrs[_ri].raw[2] == 3
                for _ri in range(len(sass_instrs)))
            # Verify R0 is not already used as a GPR in ALU instructions
            # (R0 appears in ISTP.I/IST.UR dest but those are predicate/flag
            # outputs, not GPR dests that would conflict)
            _r0_alu_conflict = any(
                _fg56_opcodes[_ri] in (0x824, 0x835, 0x812, 0x212, 0x810, 0x210)
                and sass_instrs[_ri].raw[2] == 0
                for _ri in range(len(sass_instrs)))

            if _has_s2r_r3 and not _r0_alu_conflict:
                # Global rename: every b2/b3 that is R3 -> R0
                # (R3 = tid.x, used as source in ISETP, IMAD, C11, etc.)
                _fg56_count = 0
                for _ri in range(len(sass_instrs)):
                    _si = sass_instrs[_ri]
                    _ropc = _fg56_opcodes[_ri]
                    _p = bytearray(_si.raw)
                    _changed = False
                    # S2R: rename dest R3->R0
                    if _ropc == 0x919 and _p[2] == 3:
                        _p[2] = 0; _changed = True
                    # ISETP.R-UR: rename src R3->R0 at b3
                    elif _ropc == 0xc0c and _p[3] == 3:
                        _p[3] = 0; _changed = True
                    # ISETP.IMM: rename src R3->R0 at b3
                    elif _ropc == 0x80c and _p[3] == 3:
                        _p[3] = 0; _changed = True
                    # IMAD.I: rename src R3->R0 at b3
                    elif _ropc == 0x824 and _p[3] == 3:
                        _p[3] = 0; _changed = True
                    # UIADD: rename src R3->R0 at b3
                    elif _ropc == 0x835 and _p[3] == 3:
                        _p[3] = 0; _changed = True
                    # PTXAS-R48: LOP3.LUT (0x812) src0 at b3.  `and/xor/or.b32`
                    # forms lower to `LOP3.LUT Rd, R3, imm, RZ` reading tid.x;
                    # after FG56 renames S2R R3->R0 the LOP3 must follow or the
                    # AND/XOR reads uninitialized R3 (garbage).  Observed in
                    # k300_nasty_pred_xor tids 0..16 before this rename.
                    elif _ropc == 0x812 and _p[3] == 3:
                        _p[3] = 0; _changed = True
                    # MP02: IADD3 R-R (0x210): rename src R3->R0.
                    # Used by `mov.u32 %rN, %r0` which lowers to
                    # IADD3 R_dest, R3, RZ, RZ. After FG56 R3→R0 rename of
                    # S2R dest, the IADD3 must read the new R0 holding tid.x,
                    # not the uninitialized R3. Covers b3, b4, and b8 sources
                    # conservatively (IADD3 is a 3-input add; any position
                    # that was pinned to R3 pre-rename must follow).
                    elif _ropc == 0x210 and (_p[3] == 3 or _p[4] == 3 or _p[8] == 3):
                        if _p[3] == 3: _p[3] = 0
                        if _p[4] == 3: _p[4] = 0
                        if _p[8] == 3: _p[8] = 0
                        _changed = True
                    # MP02: IADD3.IMM (0x810): rename src R3->R0 at b3 / b8
                    # (b4-b7 are the 32-bit immediate).
                    elif _ropc == 0x810 and (_p[3] == 3 or _p[8] == 3):
                        if _p[3] == 3: _p[3] = 0
                        if _p[8] == 3: _p[8] = 0
                        _changed = True
                    # C11: rename src R3->R0 at b3 + fix b10 parity
                    elif _ropc == 0xc11 and _p[3] == 3:
                        _p[3] = 0
                        # FG57: update b10 parity for even source (R0)
                        # lo (b9=0x10): b10 0x80->0x82 (set bit 1)
                        # hi (b9=0x14): b10 0x0f->0x8f (set bit 7)
                        if _p[9] == 0x10 and _p[10] == 0x80:
                            _p[10] = 0x82
                        elif _p[9] == 0x14 and _p[10] == 0x0f:
                            _p[10] = 0x8f
                        _changed = True
                    if _changed:
                        sass_instrs[_ri] = SassInstr(bytes(_p), _si.comment + ' [FG56:R0]')
                        _fg56_count += 1
                _fg56_fired = _fg56_count > 0
                if verbose and _fg56_fired:
                    print(f'[FG56] S2R R3->R0 rename: {_fg56_count} instrs')

    # FG56b: when FG56 fired, apply R4->R5 rename for body ALU result.
    # FG29/FG32 are disabled (R0 occupied), but the accumulator still
    # needs to be at R5 (PTXAS convention).  Rename all R4 in ALU body
    # and STG data to R5.
    if _fg56_fired:
        _ALU_56 = {0x824, 0x835, 0x812, 0x212, 0x810, 0x210}
        # PTXAS-R51: ISETP consumers of FG56b-renamed ALU dest.  When FG56b
        # renames LOP3/IMAD dest R4->R5, the downstream ISETP reading that
        # value at b3 must be renamed too, or the ISETP reads uninitialized
        # R4 and produces a garbage predicate.  Observed: k100_early_exit
        # `LOP3 R5, R0, 0x1 ; ISETP.EQ P1, R4, 0x0` — the ISETP.IMM was
        # reading R4 while the LOP3 had moved to R5.  FG56 already renames
        # ISETP.IMM(0x80c) and ISETP.R-UR(0xc0c) at b3 for the R3->R0
        # rename; FG56b mirrors that exact rule for R4->R5.  b3-only: for
        # 0x80c b4 is imm (not GPR) and for 0xc0c b4 is UR (not GPR).
        _ISETP_56_SRC = {0x80c, 0xc0c}
        _fg56b_count = 0
        for _ri in range(len(sass_instrs)):
            _si = sass_instrs[_ri]
            _ropc = (struct.unpack_from('<Q', _si.raw, 0)[0]) & 0xFFF
            _p = bytearray(_si.raw)
            _changed = False
            if _ropc in _ALU_56:
                if _p[2] == 4: _p[2] = 5; _changed = True
                if _p[3] == 4: _p[3] = 5; _changed = True
            elif _ropc in _ISETP_56_SRC:
                if _p[3] == 4: _p[3] = 5; _changed = True
            elif _ropc == 0x986:  # STG data at b4
                if _p[4] == 4: _p[4] = 5; _changed = True
            if _changed:
                sass_instrs[_ri] = SassInstr(bytes(_p), _si.comment + ' [FG56b:R5]')
                _fg56b_count += 1
        if verbose and _fg56b_count:
            print(f'[FG56b] R4->R5 body rename: {_fg56b_count} instrs')

    # FG60: bounded predicate complement reuse for the 3 target kernels.
    # When FG56 fired and the body has ISETP.IMM(LT, P1) followed by
    # @P1 UIADD, rewrite to ISETP.IMM(GE, P0) + @!P0 UIADD.
    # This matches PTXAS's convention of overwriting P0 with the complement.
    # FG60 admission: exactly ONE ISETP.IMM with cmp=LT, pred_dest=P1
    # AND exactly ONE @P1 UIADD.  Multi-ISETP kernels (like k200_double_guard)
    # must NOT be admitted — they have multiple predicates and the complement
    # rewrite would incorrectly change @P1 UIADD guards for predicates from
    # OTHER ISETPs (not LT-P1).
    if _fg56_fired:
        _fg60_isetp_lt_p1_count = 0
        _fg60_isetp_other_p1_count = 0
        _fg60_p1_uiadd_count = 0
        _fg60_isetp_ri = None
        _fg60_uiadd_ri = None
        for _ri in range(len(sass_instrs)):
            _si = sass_instrs[_ri]
            _ropc = (struct.unpack_from('<Q', _si.raw, 0)[0]) & 0xFFF
            if _ropc == 0x80c:  # ISETP.IMM
                _cmp = (_si.raw[9] >> 4) & 0xF
                _pred_dest = (_si.raw[10] >> 1) & 0x7
                if _pred_dest == 1:
                    if _cmp == 1:  # LT
                        _fg60_isetp_lt_p1_count += 1
                        _fg60_isetp_ri = _ri
                    else:
                        _fg60_isetp_other_p1_count += 1
            elif _ropc == 0x835:  # UIADD
                _guard = (_si.raw[1] >> 4) & 0xF
                if _guard == 0x1:  # @P1
                    _fg60_p1_uiadd_count += 1
                    _fg60_uiadd_ri = _ri

        # Only admit if exactly one LT-P1 ISETP, exactly one @P1 UIADD,
        # and NO other ISETP also writes P1.
        _fg60_safe = (_fg60_isetp_lt_p1_count == 1
                      and _fg60_isetp_other_p1_count == 0
                      and _fg60_p1_uiadd_count == 1)
        if _fg60_safe:
            # Rewrite ISETP.IMM cmp=LT P1 -> cmp=GE P0
            _si = sass_instrs[_fg60_isetp_ri]
            _p = bytearray(_si.raw)
            _p[9] = (6 << 4) | 0x00
            _p[10] = (_p[10] & 0xF0) | 0
            sass_instrs[_fg60_isetp_ri] = SassInstr(bytes(_p), _si.comment + ' [FG60:GE+P0]')
            # Rewrite @P1 UIADD -> @!P0 UIADD
            _si = sass_instrs[_fg60_uiadd_ri]
            _p = bytearray(_si.raw)
            _p[1] = (_p[1] & 0x0F) | (0x8 << 4)
            sass_instrs[_fg60_uiadd_ri] = SassInstr(bytes(_p), _si.comment + ' [FG60:@!P0]')

    # FG29-C: post-scheduling R0 normalization for MIXED kernels.
    # Rename body ALU temp registers {R4, R5, R6, ...} → R0 where PTXAS
    # uses R0 for the same role.  Only fires when:
    #   (1) addr_pair_colocated (the 0xc11 address pattern)
    #   (2) kernel has matching body ALU region between preamble and 0xc11
    #   (3) target regs form a strictly sequential write→read chain (no overlap)
    #   (4) R0 is not read or written by any other instruction in the region
    _fg29_eligible = (alloc.addr_pair_colocated and sm_version >= 120
                       and getattr(ctx, '_fg26_ur4_start', False)
                       and not _fg56_fired)  # FG56: R0 occupied by tid.x, skip R0 norm
    if _fg29_eligible:
        # Opcodes in the body ALU region (between ISETP/EXIT and 0xc11/STG)
        _ALU_OPCODES = {0x210, 0x810, 0x212, 0x812, 0x824, 0xc24, 0x825,
                        0x835, 0x819, 0x202, 0x424}
        _BOUNDARY = {0xc11, 0x986, 0x981, 0x94d, 0x947}  # 0xc11/STG/EXIT/BRA
        _PREAMBLE = {0xb82, 0x919, 0x7ac, 0xc0c, 0x94d}  # LDC/S2R/LDCU/ISETP/EXIT

        # Find body ALU region: first non-preamble ALU to first boundary
        _alu_start = None
        _alu_end = None
        for _ri, _si in enumerate(sass_instrs):
            _ropc = (struct.unpack_from('<Q', _si.raw, 0)[0]) & 0xFFF
            if _alu_start is None:
                if _ropc in _ALU_OPCODES and _ropc not in _PREAMBLE:
                    _alu_start = _ri
            elif _ropc in _BOUNDARY:
                _alu_end = _ri
                break

        _alu_len = (_alu_end - _alu_start) if (_alu_start is not None and _alu_end is not None) else 0
        if _alu_start is not None and _alu_end is not None and _alu_len <= 5:
            # Collect GPR indices used as temps in the ALU region
            _alu_gprs_written = set()
            _alu_gprs_read = set()
            for _ri in range(_alu_start, _alu_end):
                _raw = sass_instrs[_ri].raw
                _ropc = (struct.unpack_from('<Q', _raw, 0)[0]) & 0xFFF
                if _ropc not in _ALU_OPCODES:
                    continue
                _alu_gprs_written.add(_raw[2])
                _alu_gprs_read.add(_raw[3])
                # Some ALU ops use b4 as GPR src (not UR / not imm).
                # PTXAS-R49: LOP3.LUT imm32 (0x812) has imm at b4-b7 (imm low
                # byte at b4), not a GPR — must be excluded here too.
                if _ropc not in (0x835, 0x824, 0xc24, 0x810, 0x812):
                    _alu_gprs_read.add(_raw[4])

            # Target: regs written in ALU region that are NOT R0..R3 (preamble/addr)
            # and NOT used outside the ALU region (including STG data b4).
            # PTXAS renames intermediates to R0 but keeps the FINAL result at
            # the original register (so STG data is unchanged).
            _rename_candidates = _alu_gprs_written - {0, 1, 2, 3, 255}

            # Identify regs used outside the ALU region — these cannot be renamed.
            # Only check byte positions that are GPR fields for each opcode.
            # LDCU (0x7ac): b2=UR dst, no GPR fields
            # ISETP.R-UR (0xc0c): b3=GPR src, b4=UR src
            # STG (0x986): b3=GPR addr, b4=GPR data
            # 0xc11: b3=GPR src, b4=UR src
            _GPR_FIELDS = {
                0x986: (3, 4),   # STG: addr_lo + data
                0x981: (3, 4),   # LDG: addr_lo + dst
                0xc0c: (3,),     # ISETP.R-UR: GPR src only (b4 is UR)
                0xc11: (3,),     # IADD3.UR: GPR src only (b4 is UR)
                0x7ac: (),       # LDCU: no GPR fields
                0x94d: (),       # EXIT
                0x947: (),       # BRA
                0x918: (),       # NOP
                0x919: (),       # S2R: b2 is GPR dst but we don't rename S2R
                0xb82: (),       # LDC: b2 is GPR dst but preamble
                0x9c3: (),       # S2UR: no GPR
            }
            # PTXAS-R50: opcodes whose b3 is a 64-bit pair base (addr_lo).
            # The hi-half (base+1) is IMPLICITLY read but not at any byte
            # position; FG29-C must treat it as an outside user too, or it
            # will rename the hi-half to R0 and break the pair.  Observed:
            # k200_load_pred_store LDG R6.64 — R6 is escape (b3), but R7
            # was renamed to R0 because no explicit byte held R7.
            _R50_B3_PAIR_OPCS = {0x981, 0x986}  # LDG.E, STG.E
            _outside_users = set()
            for _ri in range(len(sass_instrs)):
                if _alu_start <= _ri < _alu_end:
                    continue
                _raw = sass_instrs[_ri].raw
                _ropc = (struct.unpack_from('<Q', _raw, 0)[0]) & 0xFFF
                _gpr_pos = _GPR_FIELDS.get(_ropc, (2, 3, 4))  # default: all
                for _bp in _gpr_pos:
                    if _raw[_bp] in _rename_candidates:
                        _outside_users.add(_raw[_bp])
                # R50: for LDG/STG, b3 is a 64-bit addr pair base — hi-half
                # at base+1 is implicitly live.  Mark as outside user too.
                if _ropc in _R50_B3_PAIR_OPCS:
                    _pair_hi = _raw[3] + 1
                    if _pair_hi in _rename_candidates:
                        _outside_users.add(_pair_hi)
            _rename_candidates -= _outside_users

            # Verify R0 is not used by any ALU instruction outside the
            # rename region.  Only check opcodes with real GPR fields.
            # Non-ALU opcodes (S2R, LDCU, ISETP, EXIT, BRA, STG, 0xc11)
            # either don't have GPR at b3/b4 or use them for UR/SR/addr.
            _r0_used = False
            for _ri in range(len(sass_instrs)):
                if _alu_start <= _ri < _alu_end:
                    continue  # skip the rename region itself
                _raw = sass_instrs[_ri].raw
                _ropc = (struct.unpack_from('<Q', _raw, 0)[0]) & 0xFFF
                if _ropc not in _ALU_OPCODES:
                    continue  # only check ALU opcodes for GPR R0
                if _raw[2] == 0 or _raw[3] == 0:
                    _r0_used = True
                    break

            # Extended candidate set: include regs that escape the ALU region
            # ONLY at their LAST write.  Intermediate writes can be renamed.
            _all_body_regs = (_alu_gprs_written | _rename_candidates) - {0, 1, 2, 3, 255}
            # Find the last write index for each reg in the ALU region
            _last_write: dict[int, int] = {}  # reg → sass_instrs index
            for _ri in range(_alu_start, _alu_end):
                _raw = sass_instrs[_ri].raw
                _ropc = (struct.unpack_from('<Q', _raw, 0)[0]) & 0xFFF
                if _ropc in _ALU_OPCODES and _raw[2] in _all_body_regs:
                    _last_write[_raw[2]] = _ri

            # FG39: detect liveness overlap — if any ALU instruction reads
            # two different body regs (both would become R0), the second-
            # to-last producer must keep its dest at the escape register
            # (R5) instead of R0.  This matches the PTXAS pattern.
            _r0_conflict_reg = None  # reg that must stay non-R0
            for _ri in range(_alu_start, _alu_end):
                _raw = sass_instrs[_ri].raw
                _ropc = (struct.unpack_from('<Q', _raw, 0)[0]) & 0xFFF
                if _ropc not in _ALU_OPCODES:
                    continue
                _src_body = set()
                for _bp in (3, 4):
                    if _raw[_bp] in _all_body_regs:
                        _src_body.add(_raw[_bp])
                if len(_src_body) >= 2:
                    # Two body regs read simultaneously — the one whose MOST
                    # RECENT write is closest to this consumer (but not this
                    # instruction itself) must keep its register.
                    def _prev_write(r):
                        for _wri in range(_ri - 1, _alu_start - 1, -1):
                            _wr = sass_instrs[_wri].raw
                            _wopc = (struct.unpack_from('<Q', _wr, 0)[0]) & 0xFFF
                            if _wopc in _ALU_OPCODES and _wr[2] == r:
                                return _wri
                        return -1
                    _r0_conflict_reg = max(_src_body, key=_prev_write)
                    break

            if (_rename_candidates or _outside_users) and not _r0_used:
                # Apply rename: for each ALU instruction in the region,
                # rename GPR fields to R0 EXCEPT the last write of a
                # register that escapes (whose dest must stay).
                # Also except the conflict reg (liveness overlap).
                _renamed = 0
                for _ri in range(_alu_start, _alu_end):
                    _si = sass_instrs[_ri]
                    _raw = _si.raw
                    _ropc = (struct.unpack_from('<Q', _raw, 0)[0]) & 0xFFF
                    if _ropc not in _ALU_OPCODES:
                        continue
                    # MPT01 / TPL templates carry verified-against-PTXAS
                    # registers; skip FG29-C R0-normalization for them.
                    if 'TPL' in _si.comment or 'MPT' in _si.comment:
                        continue
                    _patched = bytearray(_raw)
                    _changed = False
                    # Rename dest (b2) → R0 unless it's the last write of
                    # an escaping register or the conflict reg.
                    _dst = _patched[2]
                    if _dst in _all_body_regs and _dst not in {0, 1, 2, 3, 255}:
                        _is_last_escape = (_dst in _outside_users
                                           and _last_write.get(_dst) == _ri)
                        _is_conflict = (_r0_conflict_reg is not None
                                        and _dst == _r0_conflict_reg
                                        and _last_write.get(_dst) == _ri)
                        if not _is_last_escape and not _is_conflict:
                            _patched[2] = 0
                            _changed = True
                    # Rename src fields → R0 for body temps.
                    # b3 = src0, b4 = src1 for ALU ops with two GPR sources.
                    # EXCEPT reads of the conflict reg at its last-write position.
                    # PTXAS-R49: b4 is an IMMEDIATE byte (not GPR) for ALU
                    # opcodes that encode a literal inline — 0x810 IADD3.IMM32,
                    # 0x812 LOP3.LUT.IMM32, 0x824 IMAD.SHL/.I (imm16 at b4-b5),
                    # 0x835 UIADD.IMM32, and 0xc24 IMAD.R-UR (UR at b4).  Only
                    # opcodes whose b4 is a true GPR src1 participate in the b4
                    # rename.  Without this guard, LOP3.LUT imm=4 had its low
                    # imm byte (0x04) matched as "R4" and zeroed, silently
                    # turning `xor imm=0x4` into `xor imm=0x0` (observed in
                    # k200_xor_reduce: 3rd chained XOR dropped its imm under
                    # the alternating R0↔R5 regalloc produced by FG29-C).
                    _R49_B4_IS_IMM = {0x810, 0x812, 0x824, 0x835, 0xc24}
                    _rename_bytes = (3,) if _ropc in _R49_B4_IS_IMM else (3, 4)
                    for _sbp in _rename_bytes:
                        _src = _patched[_sbp]
                        if _src in _all_body_regs and _src not in {0, 1, 2, 3, 255}:
                            if _src != _r0_conflict_reg:
                                _patched[_sbp] = 0
                                _changed = True
                    if _changed:
                        sass_instrs[_ri] = SassInstr(bytes(_patched),
                            _si.comment + ' [FG29:R0]')
                        _renamed += 1
                if verbose and _renamed:
                    print(f'[FG29] R0-normalized {_renamed} instrs '
                          f'(body regs {sorted(_all_body_regs)} -> R0, '
                          f'escape={sorted(_outside_users)})')

                # FG32: PTXAS places the final ALU result at R5 (the STG
                # data register).  Our allocator often places it at R4
                # due to register reuse of the setp param slot.  Rename
                # the last-write dest and all downstream reads from R4→R5.
                # Only when R4 is the escaping register.  Check post-rename
                # state: R5 may have been freed by FG29 (renamed to R0).
                # After FG29/FG39 conflict handling, R5 may be used in the ALU
                # region as the conflict reg (liveness-preserved intermediate).
                # Only block FG32 if R5 is used as a DESTINATION that escapes
                # the ALU region (i.e., R5 is in _outside_users).
                if 4 in _outside_users and 5 not in _outside_users:
                    _r4_to_r5 = 0
                    for _ri in range(len(sass_instrs)):
                        _si = sass_instrs[_ri]
                        _raw = _si.raw
                        _ropc = (struct.unpack_from('<Q', _raw, 0)[0]) & 0xFFF
                        _patched = bytearray(_raw)
                        _changed = False
                        if _ropc in _ALU_OPCODES and _alu_start <= _ri < _alu_end:
                            if _patched[2] == 4:
                                _patched[2] = 5; _changed = True
                            if _patched[3] == 4:
                                _patched[3] = 5; _changed = True
                        elif _ropc == 0x986:  # STG data at b4
                            if _patched[4] == 4:
                                _patched[4] = 5; _changed = True
                        if _changed:
                            sass_instrs[_ri] = SassInstr(bytes(_patched),
                                _si.comment + ' [FG32:R5]')
                            _r4_to_r5 += 1
                    if verbose and _r4_to_r5:
                        print(f'[FG32] R4->R5 final-result rename: {_r4_to_r5} instrs')

    # FG52: local ISTP/IMAD reorder in the post-EXIT body.
    # PTXAS places ISTP.I before IMAD.I in the post-EXIT body when both
    # read only the tid.x register (no mutual dependency).  Our scheduler
    # places IMAD first.  Swap when the pattern [IMAD.I, LDCU, ISTP.I]
    # appears immediately after the post-EXIT LDCU region.
    if alloc.addr_pair_colocated and sm_version >= 120:
        _fg52_opcodes = [((si.raw[0] | (si.raw[1] << 8)) & 0xFFF) for si in sass_instrs]
        # Find post-EXIT body: first instruction after predicated EXIT
        _post_exit = None
        for _ri, _si in enumerate(sass_instrs):
            _ropc = _fg52_opcodes[_ri]
            if _ropc == 0x94d and ((_si.raw[1] >> 4) & 0xF) != 7:  # predicated EXIT
                _post_exit = _ri + 1
                break
        if _post_exit is not None:
            # Scan for the pattern: LDCU, {IMAD.I or UIADD}, LDCU, ISTP.I
            # Starting after the first post-EXIT LDCU
            _scan = _post_exit
            # Skip initial LDCUs
            while _scan < len(sass_instrs) and _fg52_opcodes[_scan] == 0x7ac:
                _scan += 1
            # Now check: [IMAD.I/UIADD, LDCU, ISTP.I] pattern
            if (_scan + 2 < len(sass_instrs)
                    and _fg52_opcodes[_scan] in (0x824, 0x835)  # IMAD.I or UIADD
                    and _fg52_opcodes[_scan + 1] == 0x7ac        # LDCU
                    and _fg52_opcodes[_scan + 2] == 0x80c):      # ISTP.I
                # Rotate: [IMAD, LDCU, ISTP] -> [ISTP, IMAD, LDCU]
                _a = sass_instrs[_scan]
                _b = sass_instrs[_scan + 1]
                _c = sass_instrs[_scan + 2]
                sass_instrs[_scan] = _c
                sass_instrs[_scan + 1] = _a
                sass_instrs[_scan + 2] = _b
                if verbose:
                    print('[FG52] ISTP/IMAD reorder at [%d-%d]' % (_scan, _scan+2))
                # Check for second ISTP.I right after (k200_double_guard pattern)
                if (_scan + 3 < len(sass_instrs)
                        and ((_fg52_opcodes[_scan + 3] == 0x80c)
                             or (_scan + 4 < len(sass_instrs)
                                 and _fg52_opcodes[_scan + 3] == 0x7ac
                                 and _fg52_opcodes[_scan + 4] == 0x80c))):
                    pass  # multi-setp handled by the same rotation

    # FG36: dedicated normalization for k300_nasty_shl_xor (long ALU region).
    # The standard FG29 R0-normalization can't handle this kernel because two
    # intermediates are consumed by the final instruction (liveness overlap).
    # PTXAS uses R0 for the first two intermediates and R5 for the pre-final.
    # Pattern: 4 ALU instrs with NOPs interleaved in the body region.
    if _fg30_eligible and sm_version >= 120:
        _fg36_opcodes = [((si.raw[0] | (si.raw[1] << 8)) & 0xFFF) for si in sass_instrs]
        # Pattern B: ...IMAD LOP3 NOP IMAD NOP LOP3 IADD3.UR...
        # with NOPs at positions alu_start+3 and alu_start+5
        _ALU_OPCODES_36 = {0x210, 0x810, 0x212, 0x812, 0x824, 0xc24, 0x835}
        _fg36_start = None
        for _ri, _si in enumerate(sass_instrs):
            _ropc = (struct.unpack_from('<Q', _si.raw, 0)[0]) & 0xFFF
            if _fg36_start is None:
                if _ropc in _ALU_OPCODES_36 and _ropc not in {0xb82, 0x919, 0x7ac, 0xc0c, 0x94d}:
                    _fg36_start = _ri
            elif _ropc == 0xc11:
                _fg36_end = _ri
                break
        else:
            _fg36_start = None

        if _fg36_start is not None:
            _fg36_len = _fg36_end - _fg36_start
            # Only fire for the 7-length region (not handled by FG29's <=5 limit)
            if _fg36_len in (6, 7):
                # Collect ALU instructions in the region (skip NOPs and LDCU)
                _alu_instrs = []
                for _ri in range(_fg36_start, _fg36_end):
                    _ropc = (struct.unpack_from('<Q', sass_instrs[_ri].raw, 0)[0]) & 0xFFF
                    if _ropc in _ALU_OPCODES_36:
                        _alu_instrs.append(_ri)
                if len(_alu_instrs) == 4:
                    # Check if the final instruction reads two DIFFERENT body
                    # regs (liveness overlap).  LOP3.IMM (0x812) only reads
                    # one GPR at b3 — b4 is an immediate, not a register.
                    _i0, _i1, _i2, _i3 = _alu_instrs
                    # Check if the final ALU reads two different body regs.
                    # LOP3.R (0x212) / IADD3 (0x210): b4 is a GPR source.
                    # LOP3.IMM (0x812) / UIADD (0x835): b4 is immediate.
                    _i3_opc = (struct.unpack_from('<Q', sass_instrs[_i3].raw, 0)[0]) & 0xFFF
                    _i3_b4_is_gpr = _i3_opc in (0x210, 0x212, 0x225)
                    _i3_raw = sass_instrs[_i3].raw
                    _body_written = {sass_instrs[_ri].raw[2] for _ri in _alu_instrs[:3]
                                     if (struct.unpack_from('<Q', sass_instrs[_ri].raw, 0)[0]) & 0xFFF in _ALU_OPCODES_36}
                    _i3_has_overlap = (_i3_b4_is_gpr
                                       and _i3_raw[4] in _body_written
                                       and _i3_raw[3] in _body_written
                                       and _i3_raw[3] != _i3_raw[4])

                    _fg36_map = {}
                    if _i3_has_overlap:
                        # Two body regs consumed by final: use R5 for pre-final
                        _fg36_map[_i0] = [(2, 0)]
                        _fg36_map[_i1] = [(2, 0), (3, 0)]
                        _fg36_map[_i2] = [(2, 5), (3, 0)]
                        _fg36_map[_i3] = [(2, 5), (3, 5), (4, 0)]
                    else:
                        # No overlap: ALL intermediates → R0, final dest → R5
                        _fg36_map[_i0] = [(2, 0)]
                        _fg36_map[_i1] = [(2, 0), (3, 0)]
                        _fg36_map[_i2] = [(2, 0), (3, 0)]
                        _fg36_map[_i3] = [(2, 5), (3, 0)]
                    # STG: data → R5
                    for _ri in range(_fg36_end, len(sass_instrs)):
                        _ropc = (struct.unpack_from('<Q', sass_instrs[_ri].raw, 0)[0]) & 0xFFF
                        if _ropc == 0x986:
                            _fg36_map[_ri] = [(4, 5)]
                            break

                    _fg36_patched = 0
                    for _ri, patches in _fg36_map.items():
                        _p = bytearray(sass_instrs[_ri].raw)
                        _changed = False
                        for bp, val in patches:
                            if _p[bp] != val:
                                _p[bp] = val
                                _changed = True
                        if _changed:
                            sass_instrs[_ri] = SassInstr(bytes(_p),
                                sass_instrs[_ri].comment + ' [FG36:R0/R5]')
                            _fg36_patched += 1
                    if verbose and _fg36_patched:
                        print(f'[FG36] long-ALU R0/R5 normalize: {_fg36_patched} instrs')

    # FG31-B: ctrl-byte convergence for near-BYTE_EXACT kernels.
    # When the opcode sequence exactly matches a known PTXAS-verified pattern,
    # override ctrl bytes to match PTXAS output.  This is a post-patch, not a
    # scoreboard change — it only fires for exact pattern matches.
    if _fg30_eligible:
        _fg31_opcodes = [((si.raw[0] | (si.raw[1] << 8)) & 0xFFF) for si in sass_instrs]
        _fg31_n = len(sass_instrs)

        # FG33: generalized ctrl-byte convergence for the MIXED-kernel family.
        # Structure: [LDC, S2R, LDCU, ISETP, EXIT, LDCU, ALU_0, LDCU, ...body ALU...,
        #             IADD3.UR, IADD3.UR, STG, EXIT, BRA]
        # Preamble (positions 0-7) and postamble (last 5) have fixed ctrl.
        # Body ALU (positions 8 to n-6): intermediate=0xc80f, last=0xe40f
        # (IMAD variant: intermediate=0xca0f, last=0xe20f)
        _FG33_PREAMBLE = {
            0: (0xe2, 0x0f), 1: (0x22, 0x0e), 2: (0x24, 0x0e),
            3: (0xda, 0x1f), 4: (0xea, 0x0f), 5: (0x22, 0x0e),
            6: (0xe2, 0x0f),  # first body ALU (UIADD or equiv)
        }
        # Position 7 (LDCU after first ALU): 0x68 for IMAD/UIADD body, 0x66 for LOP3
        _FG33_POSTAMBLE = {
            -5: (0xc8, 0x1f), -4: (0xca, 0x0f), -3: (0xe2, 0x2f),
            -2: (0xea, 0x0f), -1: (0xc0, 0x0f),
        }

        # Validate structure: preamble must be LDC S2R LDCU ISETP EXIT LDCU ...
        # Two layouts accepted:
        #   Original: ... LDCU ALU LDCU ... (LDCU at position 7)
        #   FG52:     ... LDCU ISTP ALU LDCU ... (LDCU at position 8 after reorder)
        _preamble_opc = [0xb82, 0x919, 0x7ac, 0xc0c, 0x94d, 0x7ac]
        _postamble_opc = [0xc11, 0xc11, 0x986, 0x94d, 0x947]
        _ldcu_at_7 = _fg31_n >= 14 and _fg31_opcodes[7] == 0x7ac
        _ldcu_at_8 = (_fg31_n >= 15 and _fg31_opcodes[8] == 0x7ac
                      and _fg31_opcodes[6] == 0x80c)  # ISTP.I at 6 (FG52 reorder)
        _fg33_ok = (_fg31_n >= 14
                    and _fg31_opcodes[:6] == _preamble_opc
                    and _fg31_opcodes[-5:] == _postamble_opc
                    and (_ldcu_at_7 or _ldcu_at_8))

        # MP02: FG33's hardcoded ctrl template (rbar=0x01 on body ALU) wipes
        # the scoreboard's rbar=0x03 wait on predicate producers.  For
        # multi-predicate kernels (≥2 @Px-guarded body instructions), the
        # ISETP→@P RAW hazard becomes observable: without the rbar=0x03 barrier,
        # the second @P reads a stale predicate value from an earlier ISETP
        # instead of the immediately-preceding one, producing wrong GPU results.
        # Skip FG33 only when the body has ≥2 @Px-guarded instructions.
        # Single-guard kernels (e.g. k100_guarded_store's one @%p1 add) remain
        # eligible: the scoreboard's natural instruction spacing covers that
        # hazard and FG33's template matches PTXAS byte-for-byte.
        if _fg33_ok:
            _mp02_pred_guard_count = sum(
                1 for _pi in range(8, _fg31_n - 5)
                if ((sass_instrs[_pi].raw[1] >> 4) & 0x7) != 0x7)
            if _mp02_pred_guard_count >= 2:
                _fg33_ok = False
                if verbose:
                    print(f'[MP02] FG33 skipped: {_mp02_pred_guard_count} '
                          '@Px-guarded body ALU (predicate RAW needs scoreboard rbar)')

        if _fg33_ok:
            _fg31_patched = 0
            for _fi in range(_fg31_n):
                _si = sass_instrs[_fi]
                _target = None

                # Preamble positions (0-5 from table, 6+ layout-dependent)
                if _fi in _FG33_PREAMBLE and _fi != 6:
                    _target = _FG33_PREAMBLE[_fi]
                # Postamble positions (relative to end)
                elif _fi - _fg31_n in _FG33_POSTAMBLE:
                    _target = _FG33_POSTAMBLE[_fi - _fg31_n]
                # FG52 layout: [6]=ISTP.I [7]=IMAD/UIADD [8]=LDCU
                elif _ldcu_at_8 and _fi == 6:
                    _target = (0xe2, 0x0f)  # ISTP.I: b15=0x04 set below
                elif _ldcu_at_8 and _fi == 7:
                    _target = (0xe2, 0x0f)  # IMAD/UIADD: b15=0x04 set below
                elif _ldcu_at_8 and _fi == 8:
                    _target = (0x76, 0x0e)  # LDCU after ISTP+IMAD
                # Original layout: [6]=ALU [7]=LDCU
                elif _fi == 6:
                    _target = (0xe2, 0x0f)
                elif _fi == 7 and _ldcu_at_7:
                    _alu0_opc = _fg31_opcodes[6]
                    if _alu0_opc in (0x812, 0x212):
                        _target = (0x66, 0x0e)
                    else:
                        _target = (0x68, 0x0e)
                # Body ALU (positions 8 to n-6)
                elif (9 if _ldcu_at_8 else 8) <= _fi < _fg31_n - 5:
                    _is_last_body = (_fi == _fg31_n - 6)
                    _body_opc = _fg31_opcodes[_fi]
                    if _is_last_body:
                        # Last body ALU: 0xe40f for LOP3, 0xe20f for IMAD
                        if _body_opc in (0x824, 0xc24, 0x835):  # IMAD or UIADD
                            _target = (0xe2, 0x0f)
                        else:
                            _target = (0xe4, 0x0f)
                    else:
                        # Intermediate body ALU: 0xc80f for LOP3, 0xca0f for
                        # IMAD or LOP3-before-IMAD.  Skip NOPs when checking
                        # the next ALU opcode.
                        _next_alu_opc = 0
                        for _nj in range(_fi + 1, _fg31_n):
                            _njopc = _fg31_opcodes[_nj]
                            if _njopc in (0x918, 0x7ac):  # skip NOP, LDCU
                                continue
                            _next_alu_opc = _njopc
                            break
                        if _body_opc in (0x812, 0x212, 0x810) and _next_alu_opc not in (0x824, 0xc24, 0x835):
                            _target = (0xc8, 0x0f)
                        else:
                            _target = (0xca, 0x0f)

                if _target:
                    _b13, _b14 = _target
                    # FG34/FG54: b15=0x04 (rdep=2) for specific positions.
                    # Original layout: position 6 when body has <= 1 extra ALU.
                    # FG52 layout: positions 6 AND 7 (ISTP + IMAD both get rdep=2).
                    _b15 = 0x00
                    _body_count = _fg31_n - 5 - 8
                    if _ldcu_at_8 and _fi in (6, 7):
                        _b15 = 0x04
                    elif not _ldcu_at_8 and _fi == 6 and _body_count <= 1:
                        _b15 = 0x04
                    # TPL templates carry their own per-instruction ctrl bytes
                    # verified against PTXAS ground truth; skip FG33 rewrite
                    # for them.  Same pattern as TPL01 / TPL05 b9-rewrite skip.
                    if 'TPL' in _si.comment:
                        continue
                    if _si.raw[13] != _b13 or _si.raw[14] != _b14 or _si.raw[15] != _b15:
                        _p = bytearray(_si.raw)
                        _p[13] = _b13; _p[14] = _b14; _p[15] = _b15
                        sass_instrs[_fi] = SassInstr(bytes(_p), _si.comment + ' [FG33:ctrl]')
                        _fg31_patched += 1
            if verbose and _fg31_patched:
                print(f'[FG33] ctrl-byte patched {_fg31_patched} instrs')

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
