"""
sass/regalloc.py — Register allocator for PTX → SASS.

Simple sequential allocator: walks register declarations in a PTX function
and assigns physical SASS registers linearly.  No spilling, no liveness
analysis — sufficient for the small kernels OpenPTXas targets.

Physical register layout (SM_120 / Blackwell):
    R0..R254   — 32-bit general-purpose registers (R255 = RZ = zero)
    P0..P6     — predicate registers (P7 = PT = always-true)
    UR0..UR62  — uniform registers (UR63 = URZ = zero)

64-bit PTX registers (%rd0, %rd1, ...) map to register pairs:
    %rd0 → (R0, R1), %rd1 → (R2, R3), etc.

32-bit PTX registers (%r0, %r1, ...) start after the 64-bit pairs.

Predicate registers (%p0, %p1, ...) map to P0..P6.

Kernel parameter ABI (SM_120):
    Parameters are passed via constant bank c[0][...].
    Base offset: 0x380 (confirmed from ptxas output for sm_120).
    Parameters are laid out in declaration order, 8-byte aligned for u64.
    c[0][0x37c] holds a 32-bit "frame size" value loaded into R1 by ptxas.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from ptx.ir import Function, RegDecl, ParamDecl, TypeSpec, ScalarKind


# ---------------------------------------------------------------------------
# RegAlloc dataclass (canonical definition — imported by isel)
# ---------------------------------------------------------------------------

@dataclass
class RegAlloc:
    """
    Maps PTX virtual register names to SASS physical register indices.

    - 64-bit PTX registers (%rd0, %rd1, ...) map to register pairs:
        %rd0 → (lo=R0, hi=R1), %rd1 → (lo=R2, hi=R3), etc.
    - 32-bit PTX registers (%r0, %r1, ...) map to single registers.
    - Predicate registers (%p0, %p1, ...) map to P0..P5.
    - Uniform registers (%ur0, ...) map to UR0..UR253.
    """
    int_regs: dict[str, int] = field(default_factory=dict)
    pred_regs: dict[str, int] = field(default_factory=dict)
    unif_regs: dict[str, int] = field(default_factory=dict)

    def lo(self, ptx_name: str) -> int:
        return self.int_regs[ptx_name]

    def hi(self, ptx_name: str) -> int:
        return self.int_regs[ptx_name] + 1

    def r32(self, ptx_name: str) -> int:
        return self.int_regs[ptx_name]

    def pred(self, ptx_name: str) -> int:
        return self.pred_regs.get(ptx_name, 7)  # default PT

    def ur(self, ptx_name: str) -> int:
        return self.unif_regs.get(ptx_name, 0)


# Parameter ABI base offsets per architecture
PARAM_BASE_SM120 = 0x380   # Blackwell
PARAM_BASE_SM89  = 0x160   # Ada Lovelace


def _type_size(t: TypeSpec) -> int:
    """Return the size in bytes of a PTX type."""
    if t.kind == ScalarKind.PRED:
        return 1
    return t.width // 8


def _align_up(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


@dataclass
class AllocResult:
    """Result of register allocation for one function."""
    ra: RegAlloc
    param_offsets: dict[str, int]     # PTX param name → c[0][byte_offset]
    num_gprs: int                     # total GPRs used (for .nv.info)
    num_pred: int                     # predicate regs used
    num_uniform: int                  # uniform regs used
    direct_ldc_params: set = None     # WB-5.0: u64 params eligible for LDC.64

    def __post_init__(self):
        if self.direct_ldc_params is None:
            self.direct_ldc_params = set()


def _find_ldg_coalesces(fn: Function) -> dict[str, tuple[str, int]]:
    """
    Find ld.global patterns where dest and addr can share registers.

    Returns a dict mapping dest_reg_name → (addr_reg_name, load_instr_idx).
    e.g. {'%rd1': ('%rd0', 4)} means %rd1 can share %rd0's phys reg at instr 4.

    The caller must still verify liveness: addr's last use must be the load
    itself, and dest's first def must be the load itself.
    """
    from ptx.ir import RegOp, MemOp
    coalesces: dict[str, tuple[str, int]] = {}
    idx = 0
    for bb in fn.blocks:
        for inst in bb.instructions:
            if inst.op == 'ld' and 'global' in inst.types:
                if inst.dest and inst.srcs:
                    dest = inst.dest
                    src = inst.srcs[0]
                    if isinstance(dest, RegOp) and isinstance(src, MemOp):
                        base = f'%{src.base}' if not src.base.startswith('%') else src.base
                        coalesces[dest.name] = (base, idx)
            idx += 1
    return coalesces


def allocate(fn: Function, param_base: int = PARAM_BASE_SM120,
             has_capmerc: bool = False, sm_version: int = 120) -> AllocResult:
    """
    Allocate physical registers for a PTX function.

    Walks fn.reg_decls to assign GPR indices, then fn.params to compute
    constant-bank offsets.  Returns an AllocResult with a filled RegAlloc
    and param_offsets dict ready for the instruction selector.
    """
    int_regs: dict[str, int] = {}
    pred_regs: dict[str, int] = {}
    unif_regs: dict[str, int] = {}

    # Find LDG coalescing opportunities (dest shares addr register)
    coalesces = _find_ldg_coalesces(fn)

    # Liveness analysis: compute live ranges (first def, last use) per register
    from ptx.ir import RegOp, MemOp, LabelOp
    used_regs: set[str] = set()
    reg_first_def: dict[str, int] = {}  # name → instruction index of first write
    reg_last_use: dict[str, int] = {}   # name → instruction index of last read

    all_instrs = []
    for bb in fn.blocks:
        all_instrs.extend(bb.instructions)

    for idx, inst in enumerate(all_instrs):
        if inst.dest and isinstance(inst.dest, RegOp):
            name = inst.dest.name
            used_regs.add(name)
            if name not in reg_first_def:
                reg_first_def[name] = idx
        for src in inst.srcs:
            if isinstance(src, RegOp):
                name = src.name
                used_regs.add(name)
                reg_last_use[name] = idx
            if isinstance(src, MemOp) and src.base:
                bname = src.base if src.base.startswith('%') else f'%{src.base}'
                used_regs.add(bname)
                reg_last_use[bname] = idx

    # Extend live ranges across loop back-edges.  The linear scan above records
    # only the last *forward* use; a register read every iteration of a loop is
    # live until the back-branch, not just until its last textual appearance.
    # Without this, the allocator wrongly reuses a param register (e.g. fd3/b)
    # as workspace for an intra-loop temp (e.g. fd24) from the 2nd iteration on.
    # Labels are on BasicBlock objects; build a map from label → first instr index.
    label_to_idx: dict[str, int] = {}
    running_idx = 0
    for bb in fn.blocks:
        if bb.label:
            label_to_idx[bb.label] = running_idx
        running_idx += len(bb.instructions)
    for bra_idx, inst in enumerate(all_instrs):
        if inst.op != 'bra' or not inst.srcs or not isinstance(inst.srcs[0], LabelOp):
            continue
        loop_start = label_to_idx.get(inst.srcs[0].name, bra_idx + 1)
        if loop_start > bra_idx:
            continue  # forward branch — not a back-edge
        # Any register defined before the loop and used within the loop body is
        # loop-carried: its live range must reach at least to the back-branch.
        for name, last in list(reg_last_use.items()):
            first = reg_first_def.get(name, 0)
            if first < loop_start and loop_start <= last <= bra_idx:
                reg_last_use[name] = bra_idx

    next_pred = 0
    next_ur = 4

    # Predicate allocation (simple sequential)
    for rd in fn.reg_decls:
        if rd.type.kind == ScalarKind.PRED:
            for name in rd.names:
                if name in used_regs:
                    pred_regs[name] = next_pred
                    next_pred += 1

    # SM_120: identify u64 param registers that will be loaded into UR via LDCU.64
    # and never require a GPR pair.  Two consumer classes qualify (FB-5.1):
    #   1. Pointer arithmetic: add / sub / mul / shl on the 64-bit value.
    #      add.u64 has an R-UR form (IADD.64 R-UR), so the param can be sourced
    #      directly from UR with no GPR mapping.
    #   2. Direct global memory address: ld / st / atom with 'global' in types
    #      where this vreg is the MemOp base.  isel materializes via the
    #      reserved _addr_scratch_lo pair (UR -> GPR move), no static GPR
    #      reservation required.
    # Predicated ld.param.u64 (divergent fallback path) is excluded — that
    # path materializes the result into ra.lo(dest) and needs the GPR mapping.
    ur_param_regs: set[str] = set()
    if sm_version == 120:
        # Collect u64 ld.param dests, tracking whether the load is predicated.
        # Also count def sites — a vreg redefined after the param load (e.g.
        # `add.u64 %rd0, %rd0, 4`) needs a real GPR because the increment
        # writes a new value into the same name.  Such vregs are NOT pure
        # param values and cannot be UR-bound.
        param_u64_names: set[str] = set()
        param_u64_predicated: set[str] = set()
        u64_def_count: dict[str, int] = {}
        for inst in all_instrs:
            if inst.dest and isinstance(inst.dest, RegOp):
                u64_def_count[inst.dest.name] = u64_def_count.get(inst.dest.name, 0) + 1
            if (inst.op == 'ld' and 'param' in inst.types
                    and any(t in ('u64', 's64', 'b64') for t in inst.types)
                    and isinstance(inst.dest, RegOp)):
                param_u64_names.add(inst.dest.name)
                if inst.pred:
                    param_u64_predicated.add(inst.dest.name)
        for pname in param_u64_names:
            if pname in param_u64_predicated:
                continue  # divergent fallback needs GPR
            if u64_def_count.get(pname, 0) > 1:
                continue  # rewritten after param load — needs real GPR
            only_safe = True
            for inst in all_instrs:
                for src in inst.srcs:
                    src_name = (src.name if isinstance(src, RegOp) else
                                (src.base if isinstance(src, MemOp) and
                                 isinstance(src.base, str) else None))
                    if src_name == pname:
                        # Class 1: add.u64 has IADD.64 R-UR form (only add — sub
                        # falls through to GPR IADD3, mul.lo uses IMAD.WIDE
                        # which is GPR-only, shl is also GPR-only).
                        arith_ok = (inst.op == 'add'
                                    and any(t in ('u64', 's64', 'b64') for t in inst.types))
                        # Class 2: global memory address base (FB-5.1 broadening)
                        base_ok = (isinstance(src, MemOp)
                                   and inst.op in ('ld', 'st', 'atom')
                                   and 'global' in inst.types)
                        if not (arith_ok or base_ok):
                            only_safe = False
                            break
                if not only_safe:
                    break
            if only_safe:
                ur_param_regs.add(pname)

        # FB-5.1 second pass: IADD.64-UR has only ONE UR source slot.  If a
        # single add.u64 has BOTH source operands in ur_param_regs, one must
        # fall back to a GPR allocation.  Exclude the second-listed source.
        for inst in all_instrs:
            if not (inst.op == 'add'
                    and any(t in ('u64', 's64', 'b64') for t in inst.types)):
                continue
            ur_srcs = [s.name for s in inst.srcs
                       if isinstance(s, RegOp) and s.name in ur_param_regs]
            if len(ur_srcs) >= 2:
                for extra in ur_srcs[1:]:
                    ur_param_regs.discard(extra)

        # FB-5.2: DMMA opt-out is no longer needed.  The 64-bit branch of
        # the linear scan now enforces 4-alignment for f64 vregs that are
        # mma dst tuple bases (and places the lone follower at base+2),
        # so honoring ur_param_regs in DMMA kernels is safe.

    # WB-5.0: tiny-kernel direct LDC.64 path.
    #
    # If the kernel has exactly one u64 ld.param dest AND that vreg's
    # only use is as a single MemOp.base in a global memory op, AND
    # the kernel has at least one mma whose inputs will be RZ-
    # substituted by analyze_mma_zero_subst (so the dst quad has no
    # competing live operands at low GPR slots), use LDC.64 to load
    # the param directly into a GPR pair.
    #
    # Saves:
    #   - LDCU.64 UR8 for the param
    #   - IADD.64 R-UR materialization
    #   - the unconditional S2R R0 (becomes unnecessary because no
    #     LDCU param load is left in the body)
    #
    # Excluded (proven to regress on regs):
    #   - kernels with a QMMA (its B operand cannot be RZ-substituted
    #     due to the hardware base<8 constraint, so it claims R4..R5
    #     and forces the dst quad up by +4)
    #   - kernels with atom ops (have their own scratch register
    #     interaction with the addr-mat path)
    direct_ldc_params: set[str] = set()
    if sm_version == 120:
        _u64_param_dests = [
            inst.dest.name for inst in all_instrs
            if (inst.op == 'ld' and 'param' in inst.types
                and any(t in ('u64', 's64', 'b64') for t in inst.types)
                and isinstance(inst.dest, RegOp)
                and not inst.pred)
        ]
        _has_hmma_imma_dmma = any(
            inst.op == 'mma' and 'sync' in inst.types
            and 'e4m3' not in inst.types and 'e5m2' not in inst.types
            for inst in all_instrs
        )
        _has_qmma = any(
            inst.op == 'mma' and 'sync' in inst.types
            and ('e4m3' in inst.types or 'e5m2' in inst.types)
            for inst in all_instrs
        )
        _has_atom = any(inst.op == 'atom' for inst in all_instrs)
        if (len(_u64_param_dests) == 1
                and _has_hmma_imma_dmma
                and not _has_qmma
                and not _has_atom):
            _pname = _u64_param_dests[0]
            _n_uses = 0
            _n_base_uses = 0
            for inst in all_instrs:
                for src in inst.srcs:
                    if (isinstance(src, MemOp)
                            and isinstance(src.base, str)
                            and inst.op in ('ld', 'st', 'atom')
                            and 'global' in inst.types):
                        bn = src.base if src.base.startswith('%') else f'%{src.base}'
                        if bn == _pname:
                            _n_uses += 1
                            _n_base_uses += 1
                    elif isinstance(src, RegOp) and src.name == _pname:
                        _n_uses += 1
            if _n_uses == 1 and _n_base_uses == 1:
                direct_ldc_params.add(_pname)
                ur_param_regs.discard(_pname)

    # SM_120: identify f64 param registers that will be loaded into UR via LDCU.64.
    # DFMA R-R-UR-UR handles them directly; no GPR needed.  Keeping these in UR
    # frees GPR slots so all accumulators fit in R0-R13 (avoiding R14+ restriction).
    ur_only_f64_regs: set[str] = set()
    if sm_version == 120:
        f64_param_names: set[str] = set()
        for inst in all_instrs:
            if (inst.op == 'ld' and 'param' in inst.types
                    and 'f64' in inst.types
                    and isinstance(inst.dest, RegOp)):
                f64_param_names.add(inst.dest.name)
        # Only UR-ify when the register is exclusively used as a source in f64
        # arithmetic (fma / mul / add / mov with f64 type).
        for pname in f64_param_names:
            safe = True
            for inst in all_instrs:
                for src in inst.srcs:
                    src_name = (src.name if isinstance(src, RegOp) else
                                (src.base if isinstance(src, MemOp) and
                                 isinstance(src.base, str) else None))
                    if src_name == pname:
                        if not (inst.op in ('mul', 'add', 'fma', 'mov')
                                and any(t == 'f64' for t in inst.types)):
                            safe = False
                            break
                if not safe:
                    break
            if safe:
                ur_only_f64_regs.add(pname)


    # SM_89 cbuf optimization: identify 64-bit param registers that are ONLY
    # used as sources in add.u64 (where IADD3.cb reads from cbuf directly).
    # These don't need GPR allocation, saving 2 GPRs per pointer param.
    cbuf_only_regs: set[str] = set()
    if sm_version == 89:
        # Find all regs defined by ld.param.u64
        param_load_regs = set()
        for inst in all_instrs:
            if (inst.op == 'ld' and 'param' in inst.types
                    and any(t in ('u64', 's64', 'b64') for t in inst.types)
                    and isinstance(inst.dest, RegOp)):
                param_load_regs.add(inst.dest.name)
        # Check if each param reg is ONLY used as source in add.u64
        for pname in param_load_regs:
            only_add64 = True
            for inst in all_instrs:
                for src in inst.srcs:
                    src_name = src.name if isinstance(src, RegOp) else (
                        src.base if isinstance(src, MemOp) and isinstance(src.base, str) else None)
                    if src_name == pname:
                        if not (inst.op == 'add' and any(t in ('u64', 's64') for t in inst.types)):
                            only_add64 = False
                            break
                if not only_add64:
                    break
            if only_add64:
                cbuf_only_regs.add(pname)

    # Collect registers that require 4-register alignment (HMMA/IMMA/QMMA/DMMA
    # accumulator).  mma.sync.aligned instructions write a 4-GPR accumulator;
    # hardware requires the base % 4 == 0.
    #
    # The dst is a VectorRegOp whose .regs holds the actual tuple, e.g.
    #   HMMA/IMMA/QMMA: {%f0, %f1, %f2, %f3}  → 4 vregs × 32-bit = 4 GPRs
    #   DMMA:           {%fd0, %fd1}          → 2 vregs × 64-bit = 4 GPRs
    # Only the FIRST element is the quad base; the rest are followers.
    #
    # FB-5.2: read followers from VectorRegOp.regs directly (not from the
    # ambient .reg declaration's positional neighbors), so DMMA's f64 dst
    # tuple is correctly modeled with one follower instead of three.
    quad_align_regs: set[str] = set()
    quad_follow_regs: set[str] = set()
    from ptx.ir import RegOp as _RegOp, VectorRegOp as _VectorRegOp
    for inst in all_instrs:
        if inst.op != 'mma' or 'sync' not in inst.types or inst.dest is None:
            continue
        if isinstance(inst.dest, _VectorRegOp) and inst.dest.regs:
            quad_align_regs.add(inst.dest.regs[0])
            for follower in inst.dest.regs[1:]:
                quad_follow_regs.add(follower)
        elif isinstance(inst.dest, _RegOp):
            quad_align_regs.add(inst.dest.name)
            # Fallback: positional neighbors (legacy behavior for non-vector dst)
            for rd in fn.reg_decls:
                if rd.type.kind == ScalarKind.PRED:
                    continue
                for i, nm in enumerate(rd.names):
                    if nm == inst.dest.name:
                        for j in range(1, 4):
                            if i + j < len(rd.names):
                                quad_follow_regs.add(rd.names[i + j])

    # Co-location preferences: cvt.u64.u32 dest should overlap with 32-bit source.
    # If the source is dead after the cvt, the 64-bit dest can start at the
    # source's GPR (eliminating the MOV copy in isel).
    _ENABLE_CVT_COLOCATE = True
    cvt_colocate: dict[str, str] = {}
    if _ENABLE_CVT_COLOCATE:
        for inst in all_instrs:
            if (inst.op == 'cvt' and isinstance(inst.dest, RegOp) and inst.srcs
                    and isinstance(inst.srcs[0], RegOp)):
                types = inst.types
                is_u64_dst = any(t in ('u64', 'b64') for t in types[:1])
                is_u32_src = any(t in ('u32', 'b32') for t in types[1:])
                if is_u64_dst and is_u32_src:
                    src_name = inst.srcs[0].name
                    dst_name = inst.dest.name
                    src_last = reg_last_use.get(src_name, 0)
                    cvt_def = reg_first_def.get(dst_name, len(all_instrs))
                    if src_last <= cvt_def:
                        cvt_colocate[dst_name] = src_name

    # Identify 32-bit registers that are cvt.u64.u32 sources — prefer even alignment
    cvt_sources = set(cvt_colocate.values()) if cvt_colocate else set()

    # Linear scan register allocation for GPRs
    # Sort registers by first definition order
    reg_info = []  # (name, is_64, first_def, last_use)
    for rd in fn.reg_decls:
        if rd.type.kind == ScalarKind.PRED:
            continue
        is_64 = rd.type.width >= 64
        for name in rd.names:
            if name not in used_regs:
                continue
            if name in cbuf_only_regs:
                continue  # SM_89: skip GPR for cbuf-inline params
            # FB-5.1: u64 params resolved by ur_param_regs analysis are
            # consumed entirely via UR (LDCU.64 -> R-UR add / _addr_scratch_lo
            # materialization).  No phys GPR is ever written to those slots,
            # so reserving a pair just inflates next_gpr.  Skip them here.
            if name in ur_param_regs:
                continue
            first = reg_first_def.get(name, 0)
            # Dead registers (defined but never read) get last_use = first_def,
            # so their pair is freed immediately for reuse.
            last = reg_last_use.get(name, first)
            reg_info.append((name, is_64, first, last))

    # Sort by first definition
    reg_info.sort(key=lambda x: x[2])

    # FB-5.1: u64 vregs used as memory address bases (ld/st/atom MemOp.base).
    # When such a vreg expires, its pair must NOT be immediately reused as the
    # destination of the next 64-bit allocation: the next IADD.64-UR would
    # write the same R4:R5 the in-flight LDG.E is still consuming as address,
    # producing a WAR hazard the existing single-NOP wait-state pass cannot
    # cover.  We defer such pair into a one-step "quarantine" so it is only
    # available to the allocation AFTER the immediate next one.
    addr_vregs: set[str] = set()
    for inst in all_instrs:
        if inst.op in ('ld', 'st', 'atom') and 'global' in inst.types:
            for src in inst.srcs:
                if isinstance(src, MemOp) and isinstance(src.base, str):
                    bn = src.base if src.base.startswith('%') else f'%{src.base}'
                    addr_vregs.add(bn)

    # Linear scan: assign physical registers, reclaiming dead ones
    # active = [(name, phys_reg, last_use, is_64)]
    active: list[tuple[str, int, int, bool]] = []
    free_regs_64: list[int] = []  # available even-aligned register pairs
    free_regs_32: list[int] = []  # available single registers
    quarantine_64: list[int] = []  # FB-5.1: address pairs in 1-step cooldown
    next_gpr = 2  # R0-R1 reserved

    for name, is_64, first_def, last_use in reg_info:
        # FB-5.1: previous step's quarantine becomes available now.
        if quarantine_64:
            free_regs_64.extend(quarantine_64)
            quarantine_64 = []

        # Expire old intervals: free registers whose last use is before this def
        new_active = []
        for aname, areg, alast, a64 in active:
            if alast < first_def:
                # This register is dead — reclaim it
                if a64:
                    if aname in addr_vregs:
                        quarantine_64.append(areg)  # FB-5.1: 1-step cooldown
                    else:
                        free_regs_64.append(areg)
                else:
                    free_regs_32.append(areg)
            else:
                new_active.append((aname, areg, alast, a64))
        active = new_active

        # Allocate: prefer reusing a free register, else allocate new.
        # SM_120 HARDWARE LIMIT: capmerc byte[10] fix (0x81→0x01, 0xc1→0x01)
        # unlocks R12+ at load time. With correct capmerc generation, the full
        # register range is available. Verified 2026-04-01 (commit 8d516ca).
        _MAX_GPR = 255
        if is_64:
            # FB-5.2: f64 quad alignment for DMMA accumulator.
            #
            # DMMA writes a 4-GPR accumulator as two consecutive f64 pairs.
            # The first pair (quad base) must land at a 4-aligned GPR; the
            # second pair (the lone follower) must land at base+2 so the
            # full quad is contiguous.
            #
            # We handle this BEFORE colocation/free-list/fresh allocation
            # so the alignment constraint is respected unconditionally.
            need_quad = name in quad_align_regs
            is_quad_follower = name in quad_follow_regs
            if need_quad:
                # Stash any skipped slots into free_regs_32 for later use.
                aligned_base = (next_gpr + 3) & ~3
                if aligned_base != next_gpr:
                    for r in range(next_gpr, aligned_base):
                        free_regs_32.append(r)
                    next_gpr = aligned_base
                phys = next_gpr
                next_gpr += 2
                int_regs[name] = phys
                active.append((name, phys, last_use, True))
                continue
            if is_quad_follower:
                # Place at next_gpr; the quad base just consumed next_gpr..
                # next_gpr+1, so this lands at base+2 (still 2-aligned).
                if next_gpr % 2 != 0:
                    free_regs_32.append(next_gpr)
                    next_gpr += 1
                phys = next_gpr
                next_gpr += 2
                int_regs[name] = phys
                active.append((name, phys, last_use, True))
                continue

            # Co-location: if this is a cvt.u64.u32 dest, try to place it
            # at the source register's GPR (so the MOV copy becomes a no-op).
            colocated = False
            if name in cvt_colocate:
                src_name = cvt_colocate[name]
                if src_name in int_regs:
                    src_phys = int_regs[src_name]
                    if src_phys % 2 == 0:
                        # Source is even-aligned — can form a pair at src_phys.
                        # Check no OTHER register occupies src_phys or src_phys+1.
                        # The source itself is allowed (that's the co-location).
                        pair_ok = True
                        for aname, areg, alast, a64 in active:
                            if aname == src_name:
                                continue  # allow overlap with co-location source
                            if a64 and areg == src_phys:
                                pair_ok = False; break
                            if not a64 and areg in (src_phys, src_phys + 1) and alast >= first_def:
                                pair_ok = False; break
                        if pair_ok:
                            phys = src_phys
                            # Remove from free lists if present
                            if phys in free_regs_64:
                                free_regs_64.remove(phys)
                            if phys in free_regs_32:
                                free_regs_32.remove(phys)
                            if phys + 1 in free_regs_32:
                                free_regs_32.remove(phys + 1)
                            # FB-5.1 prerequisite: if phys+1 is the next fresh
                            # GPR (or beyond), advance next_gpr past the hi
                            # half so a later 32-bit allocation cannot reuse
                            # it.  Without this, the colocated pair's hi half
                            # collides with the next-allocated 32-bit reg.
                            # The bug was latent until ur_param_regs skipping
                            # pulled allocations down to low addresses where
                            # next_gpr == phys+1 became common.
                            if next_gpr <= phys + 1:
                                next_gpr = phys + 2
                            # Remove source from active — its GPR slot is now
                            # owned by the 64-bit dest. Without this, the
                            # source's expiry would free the GPR while the
                            # 64-bit register is still alive.
                            active = [(n, r, l, w) for n, r, l, w in active
                                      if n != src_name]
                            colocated = True
            if not colocated and free_regs_64:
                phys = free_regs_64.pop(0)
            elif not colocated:
                # Try to form a 64-bit pair from freed 32-bit slots.
                # When u32 regs free up even-aligned slots, we can pair them with
                # their odd neighbour (either also freed or the next fresh register),
                # recovering low GPR slots that the 64-bit allocator would otherwise
                # skip past due to alignment.
                formed = False
                for even_r in sorted(r for r in free_regs_32 if r % 2 == 0):
                    odd_r = even_r + 1
                    if odd_r in free_regs_32:
                        # Both halves available from freed 32-bit slots
                        free_regs_32.remove(even_r)
                        free_regs_32.remove(odd_r)
                        phys = even_r
                        formed = True
                        break
                    elif odd_r == next_gpr:
                        # Even half freed; odd half is the next fresh GPR
                        free_regs_32.remove(even_r)
                        next_gpr += 1  # consume the odd half
                        phys = even_r
                        formed = True
                        break
                if not formed:
                    if next_gpr % 2 != 0:
                        next_gpr += 1
                    if next_gpr + 1 >= _MAX_GPR:
                        # Only evict intervals that DON'T overlap the new one.
                        # Evicting a still-live register causes silent miscompilation.
                        safe = [(a, i) for i, (a, ar, al, a64) in enumerate(active)
                                if a64 and al < first_def]
                        if safe:
                            _, idx = min(safe, key=lambda x: active[x[1]][2])
                            evicted = active.pop(idx)
                            phys = evicted[1]
                        else:
                            # No safe eviction — exceed limit (correct code > ERR715 risk)
                            phys = next_gpr
                            next_gpr += 2
                    else:
                        phys = next_gpr
                        next_gpr += 2
            int_regs[name] = phys
            active.append((name, phys, last_use, True))
        else:
            need_quad = name in quad_align_regs
            if need_quad:
                # HMMA/DMMA/IMMA dest: must be 4-register aligned.
                # Align next_gpr to 4; stash skipped slots for later 32-bit use.
                # Do NOT pull from free_regs_32 (freed slots may not be 4-aligned,
                # and the next 3 followers must be consecutive after this base).
                if next_gpr % 4 != 0:
                    for r in range(next_gpr, _align_up(next_gpr, 4)):
                        free_regs_32.append(r)
                    next_gpr = _align_up(next_gpr, 4)
                phys = next_gpr
                next_gpr += 1
            elif name in quad_follow_regs:
                # Consecutive slot following a quad-aligned base: must use next_gpr
                # directly so all 4 accumulator registers are contiguous in GPR space.
                phys = next_gpr
                next_gpr += 1
            elif name in cvt_sources and any(r % 2 == 0 for r in free_regs_32):
                # cvt.u64.u32 source: prefer even-aligned for co-location
                even = next(r for r in sorted(free_regs_32) if r % 2 == 0)
                free_regs_32.remove(even)
                phys = even
            elif free_regs_32:
                phys = free_regs_32.pop(0)
            elif free_regs_64:
                # Borrow the lower half of a freed 64-bit pair.
                # Put the upper half back into free_regs_32 for future 32-bit use.
                pair_base = free_regs_64.pop(0)
                phys = pair_base
                free_regs_32.append(pair_base + 1)
            elif next_gpr >= _MAX_GPR:
                # Only evict non-overlapping intervals
                safe = [(a, i) for i, (a, ar, al, a64) in enumerate(active)
                        if not a64 and al < first_def]
                if safe:
                    _, idx = min(safe, key=lambda x: active[x[1]][2])
                    evicted = active.pop(idx)
                    phys = evicted[1]
                else:
                    phys = next_gpr
                    next_gpr += 1
            else:
                if name in cvt_sources and next_gpr % 2 != 0:
                    # cvt source: bump to even for co-location
                    free_regs_32.append(next_gpr)
                    next_gpr += 1
                phys = next_gpr
                next_gpr += 1
            int_regs[name] = phys
            active.append((name, phys, last_use, False))

    # LDG coalescing: when `ld.global %dest, [%addr]` is the *last* use of %addr
    # and the *first* def of %dest, the two regs can share a physical slot.
    # The load instruction reads %addr then writes %dest; hardware already
    # supports same-reg src/dst. This saves 1 GPR per such pair.
    #
    # Safety: require that %addr's final use is exactly the load, %dest's first
    # def is exactly the load, and the types/widths match. If any of these fail
    # (e.g. %addr is alive past the load, %dest was already written), live
    # ranges interfere and we skip this pair.
    _ordered = sorted(coalesces.items(), key=lambda kv: kv[1][1])
    for dest_name, (addr_name, load_idx) in _ordered:
        if dest_name not in int_regs or addr_name not in int_regs:
            continue
        if int_regs[dest_name] == int_regs[addr_name]:
            continue  # already shared
        addr_last = reg_last_use.get(addr_name, -1)
        dest_first = reg_first_def.get(dest_name, -1)
        if addr_last != load_idx or dest_first != load_idx:
            continue  # interference — live ranges overlap
        # Width check: both must match so we don't stomp a 64-bit pair with a
        # 32-bit reg (or vice versa).
        def _width(nm: str) -> int:
            for rd in fn.reg_decls:
                if nm in rd.names:
                    return rd.type.width
            return 0
        if _width(dest_name) != _width(addr_name):
            continue
        # Ensure no other live register currently maps to addr's phys reg that
        # would be displaced — conservative check: no other name shares the
        # addr phys AND has first_def > load_idx.
        addr_phys = int_regs[addr_name]
        dest_phys = int_regs[dest_name]
        conflict = False
        dest_last = reg_last_use.get(dest_name, load_idx)
        for other, p in int_regs.items():
            if other in (dest_name, addr_name):
                continue
            if p == addr_phys:
                other_first = reg_first_def.get(other, -1)
                other_last  = reg_last_use.get(other, -1)
                # `other` conflicts iff its range strictly overlaps the dest
                # range (load_idx, dest_last]. We allow `other_last == load_idx`
                # (other dies at the load — same cycle as addr) and
                # `other_first > dest_last` (starts after dest ends).
                if not (other_last <= load_idx or other_first > dest_last):
                    conflict = True
                    break
        if conflict:
            continue
        # Safe: point dest_name at addr's phys reg. Note: for 64-bit regs the
        # phys slot already represents the pair base, so nothing extra needed.
        int_regs[dest_name] = addr_phys
        # Free dest's original phys reg for future use by shrinking next_gpr
        # when dest was the last allocation. This only trims the trivially
        # freeable tail; the more general case is handled implicitly by the
        # fact that dest_phys won't be looked up anymore.
        if dest_phys == next_gpr - 1:
            next_gpr -= 1
        elif _width(dest_name) >= 64 and dest_phys == next_gpr - 2:
            next_gpr -= 2

    # Note: nv.info EIATTR_MAX_REG_COUNT limits available registers.
    # Default template uses 0x80 (8 GPR groups). For > 8 GPRs, the emitter
    # uses the 0x90 template. This is handled in cubin/emitter.py.

    # Parameter offsets in c[0][...]
    param_offsets: dict[str, int] = {}
    param_offset = param_base

    for p in fn.params:
        size = _type_size(p.type)
        align = p.align or max(size, 4)
        param_offset = _align_up(param_offset, align)
        param_offsets[p.name] = param_offset
        param_offset += size

    return AllocResult(
        ra=RegAlloc(
            int_regs=int_regs,
            pred_regs=pred_regs,
            unif_regs=unif_regs,
        ),
        param_offsets=param_offsets,
        num_gprs=next_gpr,
        num_pred=max(next_pred, 1),
        num_uniform=max(next_ur, 5),
        direct_ldc_params=direct_ldc_params if sm_version == 120 else set(),
    )
