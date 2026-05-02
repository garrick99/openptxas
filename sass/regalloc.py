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
    addr_pair_colocated: bool = False  # ALLOC-SUBSYS-2: add.u64 dest co-located with cvt pair

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
             has_capmerc: bool = False, sm_version: int = 120,
             skip_vregs: set | None = None,
             imad_wide_fuse_bases: set | None = None) -> AllocResult:
    """
    Allocate physical registers for a PTX function.

    Walks fn.reg_decls to assign GPR indices, then fn.params to compute
    constant-bank offsets.  Returns an AllocResult with a filled RegAlloc
    and param_offsets dict ready for the instruction selector.

    `skip_vregs` (WB-7): vreg names whose producing instruction will be
    elided by isel (e.g. address-fold dead `add.u64`).  These vregs are
    excluded from reg_info so no phys pair is reserved.

    `imad_wide_fuse_bases` (Phase 20): u64 param vreg names that appear
    as the addend of a Phase 19v2 IMAD.WIDE.U32-imm-with-addend fusion.
    Eligible params are promoted from LDCU.64 (UR) to LDC.64 (GPR) so
    the fused IMAD.WIDE.U32-imm consumer reads them directly without a
    MOV R, UR materialization.
    """
    if skip_vregs is None:
        skip_vregs = set()
    if imad_wide_fuse_bases is None:
        imad_wide_fuse_bases = set()
    int_regs: dict[str, int] = {}
    pred_regs: dict[str, int] = {}
    unif_regs: dict[str, int] = {}

    # Auto-declare any vregs referenced in the function body that lack a
    # `.reg` declaration.  Forge-emitted PTX occasionally references names
    # like `%rd_unknown_col_len` without declaring them; without a RegDecl
    # the linear scan never assigns a slot and isel's int_regs lookup
    # KeyErrors.  Infer the type from the standard PTX prefix and append a
    # synthetic RegDecl so the rest of the allocator behaves normally.
    from ptx.ir import RegOp as _RO0, MemOp as _MO0, VectorRegOp as _VRO0
    _declared_names: set[str] = set()
    for _rd in fn.reg_decls:
        _declared_names.update(_rd.names)
    _SPECIAL_SR = {
        '%tid.x', '%tid.y', '%tid.z',
        '%ctaid.x', '%ctaid.y', '%ctaid.z',
        '%ntid.x', '%ntid.y', '%ntid.z',
        '%nctaid.x', '%nctaid.y', '%nctaid.z',
        '%laneid', '%lanemask_lt',
        '%warpid', '%smid', '%gridid', '%clock', '%clock64',
    }
    _used_in_body: set[str] = set()
    for _bb in fn.blocks:
        for _inst in _bb.instructions:
            if isinstance(_inst.dest, _RO0):
                _used_in_body.add(_inst.dest.name)
            if isinstance(_inst.dest, _VRO0) and _inst.dest.regs:
                _used_in_body.update(_inst.dest.regs)
            for _src in _inst.srcs:
                if isinstance(_src, _RO0):
                    _used_in_body.add(_src.name)
                if isinstance(_src, _VRO0) and _src.regs:
                    _used_in_body.update(_src.regs)
                if (isinstance(_src, _MO0) and isinstance(_src.base, str)
                        and _src.base.startswith('%')):
                    _used_in_body.add(_src.base)
            if _inst.pred:
                _pn = _inst.pred.lstrip('@').lstrip('!')
                if _pn.startswith('%'):
                    _used_in_body.add(_pn)
    _PREFIX_TYPES = (
        ('rd', TypeSpec(ScalarKind.U, 64)),
        ('fd', TypeSpec(ScalarKind.F, 64)),
        ('r',  TypeSpec(ScalarKind.U, 32)),
        ('f',  TypeSpec(ScalarKind.F, 32)),
        ('p',  TypeSpec(ScalarKind.PRED, 1)),
    )
    for _name in sorted(_used_in_body - _declared_names):
        if _name in _SPECIAL_SR or '.' in _name:
            continue
        _bare = _name.lstrip('%')
        for _prefix, _ts in _PREFIX_TYPES:
            if _bare.startswith(_prefix):
                fn.reg_decls.append(RegDecl(type=_ts, name=_bare, count=1))
                _declared_names.add(_name)
                break

    # Find LDG coalescing opportunities (dest shares addr register)
    coalesces = _find_ldg_coalesces(fn)

    # Liveness analysis: compute live ranges (first def, last use) per register
    from ptx.ir import RegOp, MemOp, LabelOp, ImmOp
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
        # Predicate-guard uses: `@%pN` or `@!%pN` on an instruction's
        # pred field. These aren't in inst.srcs, so the above loop
        # misses them — but they ARE real uses of the predicate vreg
        # and must participate in liveness tracking so the pred
        # allocator can reuse dead slots post-EXIT.
        if inst.pred:
            _pn = inst.pred.lstrip('@').lstrip('!')
            if _pn.startswith('%'):
                used_regs.add(_pn)
                reg_last_use[_pn] = idx

    # === Dead-write redirection (narrow fix for the 0827506f-class clobber) ===
    # When a PTX vreg is written multiple times and a particular write has
    # no later read of that vreg (i.e., the write is dead), the emitted SASS
    # instruction still writes to a physical register.  The linear-scan
    # allocator freed the register at the last READ — between that point
    # and this dead write, the slot may have been reassigned to another
    # live vreg.  The dead write then clobbers that live vreg.
    #
    # Fix: redirect such writes to R255 (RZ) by rewriting the instruction's
    # dest to a reserved sentinel vreg.  RZ discards writes, so the emitted
    # instruction becomes a harmless no-op.
    #
    # Restrictions (why this is NARROW):
    #   1. Only rewrite when the name has >= 2 writes.  A single-write dead
    #      dest (e.g. QMMA tensor padding mov.b32 %r6,0 used to force a
    #      4-aligned accumulator base) is intentional and must be left alone.
    #   2. Only 32-bit scalar names.  64-bit pair / predicate / uniform
    #      dests have different encoding paths.
    #   3. Only rewrite writes whose index is strictly after the vreg's
    #      last READ (reg_last_use[name] < idx).  Writes before or at the
    #      last read are producing a live value.
    _DEAD = '_DEAD_'
    write_count: dict[str, int] = {}
    for idx, inst in enumerate(all_instrs):
        if inst.dest and isinstance(inst.dest, RegOp):
            nm = inst.dest.name
            if nm.startswith('%rd') or nm.startswith('%p') or nm.startswith('%ur'):
                continue
            write_count[nm] = write_count.get(nm, 0) + 1

    dead_write_count = 0
    for idx, inst in enumerate(all_instrs):
        if not (inst.dest and isinstance(inst.dest, RegOp)):
            continue
        nm = inst.dest.name
        if nm.startswith('%rd') or nm.startswith('%p') or nm.startswith('%ur'):
            continue
        if write_count.get(nm, 0) < 2:
            continue
        # Redirect only if no read of this name strictly after this write.
        if reg_last_use.get(nm, -1) > idx:
            continue
        # At this point: multi-write vreg, this write has no later read → dead.
        inst.dest = RegOp(name=_DEAD)
        dead_write_count += 1

    if dead_write_count > 0:
        # Rebuild liveness tables after the rewrite so the linear scan does
        # not reserve GPRs for names whose last writes were redirected away.
        used_regs = set()
        reg_first_def = {}
        reg_last_use = {}
        for idx, inst in enumerate(all_instrs):
            if inst.dest and isinstance(inst.dest, RegOp):
                nm = inst.dest.name
                used_regs.add(nm)
                if nm not in reg_first_def:
                    reg_first_def[nm] = idx
            for src in inst.srcs:
                if isinstance(src, RegOp):
                    nm = src.name
                    used_regs.add(nm)
                    reg_last_use[nm] = idx
                if isinstance(src, MemOp) and src.base:
                    bnm = src.base if src.base.startswith('%') else f'%{src.base}'
                    used_regs.add(bnm)
                    reg_last_use[bnm] = idx
        # Sentinel → RZ.  The linear scan won't see _DEAD_ in reg_decls.
        int_regs[_DEAD] = 255

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

    # Predicate allocation — linear scan over liveness ranges so that
    # predicates whose last use ends before a new predicate's first
    # definition can share a physical slot.  Matches ptxas's pattern
    # where `@Pn EXIT` (or `@Pn ret`) ends Pn's live range and the
    # next setp reuses the same slot.
    pred_names = [
        name
        for rd in fn.reg_decls
        if rd.type.kind == ScalarKind.PRED
        for name in rd.names
        if name in used_regs
    ]
    # Sort by first definition (predicates with no def are kept in
    # declaration order at the end so they don't steal low slots).
    pred_names.sort(key=lambda n: reg_first_def.get(n, 10**9))
    pred_free: list[int] = []  # free-list, lowest first
    pred_active: list[tuple[int, str, int]] = []  # (last_use, name, phys_id)
    for name in pred_names:
        first = reg_first_def.get(name, 0)
        # Release any active predicate whose last use ended strictly
        # before this name's first definition.
        still_active: list[tuple[int, str, int]] = []
        for last, aname, phys in pred_active:
            if last < first:
                pred_free.append(phys)
            else:
                still_active.append((last, aname, phys))
        pred_active = still_active
        if pred_free:
            pred_free.sort()
            phys = pred_free.pop(0)
        else:
            phys = next_pred
            next_pred += 1
        pred_regs[name] = phys
        pred_active.append((reg_last_use.get(name, first), name, phys))

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
    # PTXAS-R22 (FB-1 Phase A fix): pre-compute per-param byte offsets so
    # the ur_param_regs analysis can discriminate UR-safe u64 params from
    # those that must be direct-LDC-loaded.  The only shape that matters
    # for R22 is "u64 param used as base of add.u64 whose dest is itself
    # the base of a global mem op" — which is an address-arithmetic chain.
    # For those, IADD.64 R-UR is safe only when the LDCU.64 offset is
    # 16-byte aligned; on a non-16-byte-aligned offset (canonical failing
    # case: 2nd u64 param at c[0][0x388]) the mixed-domain address
    # computation yields CUDA_ERROR_ILLEGAL_ADDRESS.  Mirror the matmul
    # path (16-byte aligned param) by forcing direct LDC.64 for the
    # unsafe subset, leaving the aligned/unrelated UR paths untouched.
    _r22_param_offsets: dict[str, int] = {}
    if sm_version == 120:
        _r22_param_offset = param_base
        for _p in fn.params:
            _r22_size = _type_size(_p.type)
            _r22_align = _p.align or max(_r22_size, 4)
            _r22_param_offset = _align_up(_r22_param_offset, _r22_align)
            _r22_param_offsets[_p.name] = _r22_param_offset
            _r22_param_offset += _r22_size
    _r22_misaligned_addr_arith_params: set[str] = set()
    if sm_version == 120:
        # Map u64 param vreg → PTX param name so we can look up offsets.
        _r22_vreg_to_param: dict[str, str] = {}
        for inst in all_instrs:
            if (inst.op == 'ld' and 'param' in inst.types
                    and any(t in ('u64', 's64', 'b64') for t in inst.types)
                    and isinstance(inst.dest, RegOp)
                    and inst.srcs and isinstance(inst.srcs[0], MemOp)
                    and isinstance(inst.srcs[0].base, str)):
                _r22_vreg_to_param[inst.dest.name] = inst.srcs[0].base
        # For each add.u64 whose dest is consumed as a global MemOp base,
        # flag the u64-param sources whose param offset is not 16-byte
        # aligned.  Those params must take the direct-LDC.64 path.
        for inst in all_instrs:
            if not (inst.op == 'add'
                    and any(t in ('u64', 's64', 'b64') for t in inst.types)):
                continue
            if not isinstance(inst.dest, RegOp):
                continue
            _r22_add_dest = inst.dest.name
            _r22_dest_feeds_memop = False
            for inst2 in all_instrs:
                if inst2 is inst:
                    continue
                if inst2.op not in ('ld', 'st', 'atom'):
                    continue
                if 'global' not in inst2.types:
                    continue
                for _s in inst2.srcs:
                    if (isinstance(_s, MemOp)
                            and isinstance(_s.base, str)):
                        _sn = (_s.base if _s.base.startswith('%')
                               else f'%{_s.base}')
                        if _sn == _r22_add_dest:
                            _r22_dest_feeds_memop = True
                            break
                if _r22_dest_feeds_memop:
                    break
            if not _r22_dest_feeds_memop:
                continue
            for _src in inst.srcs:
                if not isinstance(_src, RegOp):
                    continue
                _pname = _r22_vreg_to_param.get(_src.name)
                if _pname is None:
                    continue
                _poff = _r22_param_offsets.get(_pname)
                if _poff is None:
                    continue
                if _poff % 16 != 0:
                    # PTXAS-R22 WB-8 exemption: if this misaligned param has
                    # an aligned u64 partner at (_poff - 8) (i.e. they form
                    # an adjacent pair with the lower offset 16-byte aligned),
                    # WB-8 packing will coalesce their LDCU.64s into a single
                    # LDCU.128 at the aligned base offset.  The hardware load
                    # then comes from an aligned address and R22's
                    # ILLEGAL_ADDRESS concern does not apply.  Leaving the
                    # param UR-bound is therefore safe AND avoids the LDC.64
                    # scoreboard race that fired with the LDC direct fallback
                    # (observed: _fuzz_bugs/add_shr_add_with_tid_guard —
                    # STG address read R4 before LDC.64 completed, producing
                    # p_in + tid*8 stride-8 writes instead of p_out + tid*4).
                    _partner_off = _poff - 8
                    _has_aligned_partner = (
                        _partner_off >= 0
                        and _partner_off % 16 == 0
                        and any(
                            _off == _partner_off
                            for _off in _r22_param_offsets.values()
                        )
                    )
                    if not _has_aligned_partner:
                        _r22_misaligned_addr_arith_params.add(_src.name)

    ur_param_regs: set[str] = set()
    # PTXAS-R23C: default empty dead-load set for non-SM_120 paths (see below).
    _r23c_dead_ldparam_ids: set[int] = set()
    if sm_version == 120:
        # Collect u64 ld.param dests, tracking whether the load is predicated.
        # Also count def sites — a vreg redefined after the param load (e.g.
        # `add.u64 %rd0, %rd0, 4`) needs a real GPR because the increment
        # writes a new value into the same name.  Such vregs are NOT pure
        # param values and cannot be UR-bound.
        param_u64_names: set[str] = set()
        param_u64_predicated: set[str] = set()
        u64_def_count: dict[str, int] = {}
        # PTXAS-R23C: track per-name last ld.param.u64 writer.  When a u64
        # PTX vreg is written by two or more ld.param.u64 (e.g. `ld.param.u64
        # %rd0, [in]; ld.param.u64 %rd0, [out]` in 2-u64-param kernels), all
        # earlier writes are dead — the final write clobbers them before
        # any consumer can read.  R23C proof: emitting the dead LDC.64s
        # produces a pre-EXIT sequence whose structural shape violates an
        # SM_120 driver invariant tied to the post-EXIT descriptor-priming
        # ULDCU.  Dropping dead writes restores the correct pre-/post-EXIT
        # balance (exactly one LDC.64 pre-EXIT per live u64 param; rule
        # #29 then upcasts the post-EXIT LDCU.64 to ULDCU.128).  Leaving
        # them in caused CUDA_ERROR_ILLEGAL_ADDRESS at STG.E on G8-shape.
        _r23c_last_ldparam_id: dict[str, int] = {}
        _r23c_all_ldparam_ids: list[tuple[str, int]] = []
        for inst in all_instrs:
            if inst.dest and isinstance(inst.dest, RegOp):
                u64_def_count[inst.dest.name] = u64_def_count.get(inst.dest.name, 0) + 1
            if (inst.op == 'ld' and 'param' in inst.types
                    and any(t in ('u64', 's64', 'b64') for t in inst.types)
                    and isinstance(inst.dest, RegOp)):
                param_u64_names.add(inst.dest.name)
                if inst.pred:
                    param_u64_predicated.add(inst.dest.name)
                _r23c_last_ldparam_id[inst.dest.name] = id(inst)
                _r23c_all_ldparam_ids.append((inst.dest.name, id(inst)))
        _r23c_dead_ldparam_ids: set[int] = set()
        for pname, iid in _r23c_all_ldparam_ids:
            if _r23c_last_ldparam_id.get(pname) != iid:
                _r23c_dead_ldparam_ids.add(iid)
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
            # PTXAS-R31: force UR-route for params we renamed across a
            # predicated EXIT (pipeline.py::_r31_rename_inplace_u64_redefine_
            # across_exit).  These params have exactly one def (the
            # `ld.param.u64`) and exactly one use as the src of an
            # `add.u64` whose dest is a fresh vreg, so the UR-only path is
            # safe and R22's misaligned-addr-arith exclusion does not apply.
            _r31_force = getattr(fn, '_r31_force_ur_params', set())
            if only_safe and (pname not in _r22_misaligned_addr_arith_params
                              or pname in _r31_force):
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

        # WB-10: multi-param atom kernels (atom_cas64-style).
        # When the kernel is dominated by a single atomic op consuming
        # multiple u64 params (each as either an atom operand or as a
        # store base), each param is single-use after stripping the
        # PTX `add.u64 %rd, %rd, 0` "materialize-to-GPR" hints.
        # Direct LDC.64 loads them straight into the GPR pairs the
        # ATOMG.CAS path needs, eliminating the LDCU.64 + IADD.64-UR
        # materialization chain (saves one instruction per param).
        #
        # Restricted to multi-param (>= 2) so this doesn't fire on
        # single-u64-param atom kernels (atom_or, atomg_add,
        # multi_block_atomic) which are excluded for reg-pressure
        # reasons.
        if not direct_ldc_params and _has_atom and len(_u64_param_dests) >= 2:
            _eligible: set[str] = set()
            for _pn in _u64_param_dests:
                ok = True
                for inst in all_instrs:
                    if not ok:
                        break
                    # Self-add-zero hint: ignore (we'll skip emission too).
                    is_self_add_zero = (
                        inst.op == 'add'
                        and any(t in ('u64', 's64', 'b64') for t in inst.types)
                        and isinstance(inst.dest, RegOp)
                        and inst.dest.name == _pn
                        and len(inst.srcs) >= 2
                        and isinstance(inst.srcs[0], RegOp)
                        and inst.srcs[0].name == _pn
                        and isinstance(inst.srcs[1], ImmOp)
                        and inst.srcs[1].value == 0
                    )
                    if is_self_add_zero:
                        continue
                    for src in inst.srcs:
                        sn = (src.name if isinstance(src, RegOp) else
                              (src.base if isinstance(src, MemOp)
                               and isinstance(src.base, str) else None))
                        if sn != _pn:
                            continue
                        # Allowed: MemOp base in global ld/st/atom
                        if (isinstance(src, MemOp)
                                and inst.op in ('ld', 'st', 'atom')
                                and 'global' in inst.types):
                            continue
                        # Allowed: RegOp source in atom global (cmp/new
                        # operands of atom.global.cas)
                        if (isinstance(src, RegOp)
                                and inst.op == 'atom'
                                and 'global' in inst.types):
                            continue
                        # Anything else disqualifies the param.
                        ok = False
                        break
                if ok:
                    _eligible.add(_pn)
            # Only switch ALL params at once — partial conversion
            # would mix LDC and UR paths in unpredictable ways.
            if len(_eligible) == len(_u64_param_dests):
                direct_ldc_params.update(_eligible)
                for _p in _eligible:
                    ur_param_regs.discard(_p)

        # PTXAS-R29.1: scalar-LDG-only u64 params → direct_ldc_params.
        # A u64 param whose ONLY consumer is a scalar `ld.global` (MemOp
        # base with no offset arithmetic and no redefinition) gets the
        # ptxas-faithful lowering: one `LDC.64 R_pair, c[0][param_off]`
        # in the body that feeds `LDG.E desc[UR][R_pair.64]` directly.
        # Without this classification, `_select_ld_param` queues a UR
        # preamble `ULDCU.64` AND `_select_ld_global` attempts a body
        # `IADD.64 R-RZ-UR` materialization — the latter crashes the
        # SM_120 descriptor-LDG path (R23A.4 proof).  R25.3's PTX
        # canonicalization workaround is retired in favor of this
        # classification (see `_canonicalize_scalar_ldg` in pipeline.py,
        # kept for backup but not invoked).  Mixed kernels (one scalar +
        # one offset-form) only classify the scalar-LDG-base param,
        # leaving offset-form vregs on their working path untouched.
        for _pn in _u64_param_dests:
            if _pn in direct_ldc_params:
                continue  # already marked by tiny-kernel / atom path
            only_scalar_ldg_base = True
            has_any_use = False
            for inst in all_instrs:
                if (isinstance(inst.dest, RegOp)
                        and inst.dest.name == _pn
                        and inst.op != 'ld'):
                    # Redefined by a non-ld op (e.g. add.u64, mul)
                    only_scalar_ldg_base = False
                    break
                for src in inst.srcs:
                    sn = (src.name if isinstance(src, RegOp) else
                          (src.base if isinstance(src, MemOp)
                           and isinstance(src.base, str) else None))
                    if sn != _pn:
                        continue
                    has_any_use = True
                    # Allowed: scalar ld.global with zero offset.
                    if (isinstance(src, MemOp)
                            and inst.op == 'ld'
                            and 'global' in inst.types
                            and src.offset == 0):
                        continue
                    only_scalar_ldg_base = False
                    break
                if not only_scalar_ldg_base:
                    break
            if has_any_use and only_scalar_ldg_base:
                direct_ldc_params.add(_pn)
                ur_param_regs.discard(_pn)

        # Phase 20: u64 params used as the addend of a fused IMAD.WIDE.U32-imm
        # (Phase 19v2 lowering) need their base value in a GPR pair so the
        # IMAD.WIDE.U32 R, R, IMM, R form can read it directly.  When the
        # param is left in UR space (default LDCU.64 path), the consumer
        # has to emit a 2-MOV R,UR materialization sequence per fusion
        # site — one of those per IMAD.WIDE-imm.  Promoting the param to
        # LDC.64 GPR-direct eliminates the MOVs and matches ptxas's
        # natural codegen for `(param_base + idx*K_const)` address
        # patterns (e.g. forge gather/merkle).
        #
        # Eligibility:
        #   - param vreg appears as the addend of >=1 IMAD.WIDE-fuse pair
        #   - param is NOT redefined elsewhere (single ld.param.u64 def)
        #   - param is NOT predicated (predicated load defeats the
        #     unconditional-LDC.64 emission path)
        #
        # Backward-compat: kernels with no IMAD.WIDE-fuse hits leave
        # `imad_wide_fuse_bases` empty → no promotion happens → existing
        # LDCU.64 paths are preserved.
        for _pn in _u64_param_dests:
            if _pn in direct_ldc_params:
                continue
            if _pn not in imad_wide_fuse_bases:
                continue
            # Verify single ld.param def, not redefined.
            n_def = 0
            redefined = False
            ld_param_pred = False
            for inst in all_instrs:
                if (isinstance(inst.dest, RegOp)
                        and inst.dest.name == _pn):
                    if inst.op == 'ld' and 'param' in inst.types:
                        n_def += 1
                        if inst.pred:
                            ld_param_pred = True
                    else:
                        redefined = True
                        break
            if redefined or n_def != 1 or ld_param_pred:
                continue
            direct_ldc_params.add(_pn)
            ur_param_regs.discard(_pn)

        # PTXAS-R29.3: S2R live-range extension for direct-LDC.64 safety.
        #
        # R29.1 lowers scalar-LDG params via body `LDC.64 R_pair` that writes
        # both halves of a GPR pair.  The SASS scheduler hoists `S2R R, %tid.x`
        # (and peer special-reg S2Rs) to the top of the body to hide latency,
        # so S2R's physical write EXECUTES BEFORE the LDC.64 even though PTX
        # source order has the S2R defined LATER than the LDC.64 input vreg.
        #
        # Consequence: if the linear-scan allocator reuses the high half of
        # an already-dead direct-LDC.64 pair as the S2R destination, then:
        #   pos 1:  S2R  R3, SR_TID.X        // writes R3 = tid.x
        #   pos 4:  LDC.64 R2:R3, c[0][...]  // OVERWRITES R3 with ptr_hi
        #   pos 9:  ISETP.IMM R3, 0          // reads ptr_hi, not tid.x
        # which is the G5 Family-B residual (see R29.2 bail notes).
        #
        # Fix: when direct-LDC.64 lowering is active for any pair in this
        # kernel, extend `reg_first_def` of every S2R-destined single reg
        # back to 0.  This accurately reflects the scheduler's hoisting and
        # forces the linear scan to allocate S2R-dest single regs BEFORE any
        # direct-LDC.64 pair, guaranteeing no overlap with a pair half.
        # Scoped to direct_ldc_params-active kernels so pre-R29.1 register
        # layouts are unchanged for all other kernels.
        if direct_ldc_params:
            _SPECIAL_SRC_NAMES = {
                '%tid.x', '%tid.y', '%tid.z',
                '%ctaid.x', '%ctaid.y', '%ctaid.z',
                '%ntid.x', '%ntid.y', '%ntid.z',
                '%nctaid.x', '%nctaid.y', '%nctaid.z',
                '%laneid',
            }
            for inst in all_instrs:
                if (inst.op == 'mov'
                        and isinstance(inst.dest, RegOp)
                        and inst.srcs
                        and isinstance(inst.srcs[0], RegOp)
                        and inst.srcs[0].name in _SPECIAL_SRC_NAMES):
                    dname = inst.dest.name
                    if dname in reg_first_def and reg_first_def[dname] > 0:
                        reg_first_def[dname] = 0

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
        if inst.op != 'mma' or 'sync' not in inst.types:
            continue
        # Destination tuple: 4 consecutive GPRs (or 2 for f64 DMMA), base 4-aligned.
        if inst.dest is not None:
            if isinstance(inst.dest, _VectorRegOp) and inst.dest.regs:
                quad_align_regs.add(inst.dest.regs[0])
                for follower in inst.dest.regs[1:]:
                    quad_follow_regs.add(follower)
            elif isinstance(inst.dest, _RegOp):
                quad_align_regs.add(inst.dest.name)
                for rd in fn.reg_decls:
                    if rd.type.kind == ScalarKind.PRED:
                        continue
                    for i, nm in enumerate(rd.names):
                        if nm == inst.dest.name:
                            for j in range(1, 4):
                                if i + j < len(rd.names):
                                    quad_follow_regs.add(rd.names[i + j])
        # Source tuples (A/B/C): each VectorRegOp is a contiguous GPR run.
        # The HMMA hardware reads N consecutive registers starting at the
        # encoded base — without this constraint, the linear-scan allocator
        # coalesces the source vregs into a single physreg and the tensor
        # core reads garbage from the gap registers.  Surfaced by mower
        # probe hmma/m16n8k16/all_ones (2026-04-29).
        for src in (inst.srcs or ()):
            if isinstance(src, _VectorRegOp) and src.regs:
                quad_align_regs.add(src.regs[0])
                for follower in src.regs[1:]:
                    quad_follow_regs.add(follower)

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

    # FB-5.1 (moved up for ALLOC-SUBSYS-2): u64 vregs used as memory address
    # bases (ld/st/atom MemOp.base).
    addr_vregs: set[str] = set()
    for inst in all_instrs:
        if inst.op in ('ld', 'st', 'atom') and 'global' in inst.types:
            for src in inst.srcs:
                if isinstance(src, MemOp) and isinstance(src.base, str):
                    bn = src.base if src.base.startswith('%') else f'%{src.base}'
                    addr_vregs.add(bn)

    # ALLOC-SUBSYS-2: Reserve R2 for 0xc11 address pair lo when the kernel
    # has cvt.u64.u32 AND global memory operations.  PTXAS assigns the
    # first PTX virtual register (element offset from mad.lo) to R3, keeping
    # R2 free for the carry-chain lo dest so the pair {R2,R3} forms naturally.
    # Without this, our allocator puts the offset at R2, making the constrained
    # pair {R1,R2} which collides with the preamble-reserved R1.
    #
    # The cvt.u64.u32 widens an element index to 64-bit, which feeds (via shl +
    # add.u64) into the global memory address.  The condition is: cvt_colocate
    # is non-empty (has cvt.u64.u32) AND the kernel has global ld/st/atom
    # (addr_vregs non-empty, meaning some vreg is a global memory base).
    _has_global_mem = bool(addr_vregs)
    _reserve_r2_for_addr = bool(cvt_colocate) and _has_global_mem

    # ALLOC-SUBSYS-2b: Address-chain co-location.  When add.u64 %rd2, %X, %rd1
    # where %rd1 is from cvt.u64.u32 and %rd2 is used as a global address base,
    # co-locate %rd2 at %rd1's pair.  PTXAS does this: the 0xc11 pair overwrites
    # the widened offset pair with the final address in-place.
    add_colocate: dict[str, str] = {}  # add dest → cvt dest (= pair to reuse)
    if _reserve_r2_for_addr:
        cvt_dests = set(cvt_colocate.keys())
        for inst in all_instrs:
            if (inst.op == 'add'
                    and any(t in ('u64', 's64', 'b64') for t in inst.types)
                    and isinstance(inst.dest, RegOp)
                    and len(inst.srcs) >= 2):
                dest_name = inst.dest.name
                if dest_name not in addr_vregs:
                    continue
                # Check if either source is a cvt.u64.u32 result
                for src in inst.srcs:
                    if isinstance(src, RegOp) and src.name in cvt_dests:
                        # Verify the cvt result dies at this add (last_use == add's index)
                        cvt_dest_name = src.name
                        cvt_last = reg_last_use.get(cvt_dest_name, -1)
                        add_def = reg_first_def.get(dest_name, -1)
                        if cvt_last <= add_def:
                            add_colocate[dest_name] = cvt_dest_name
                        break

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
            # WB-7: vregs whose producing instruction is elided by isel
            # (e.g. address-fold dead add.u64).  No SASS reads or writes
            # the slot, so don't reserve one.
            if name in skip_vregs:
                continue
            first = reg_first_def.get(name, 0)
            # Dead registers (defined but never read) get last_use = first_def,
            # so their pair is freed immediately for reuse.
            last = reg_last_use.get(name, first)
            reg_info.append((name, is_64, first, last))

    # Sort by first definition
    reg_info.sort(key=lambda x: x[2])

    # FB-5.1: addr_vregs (computed above) drives quarantine — when such a vreg
    # expires, its pair enters a one-step cooldown to prevent WAR hazards where
    # the next IADD.64-UR would write the same pair an in-flight LDG.E still
    # consumes as address.

    # Linear scan: assign physical registers, reclaiming dead ones
    # active = [(name, phys_reg, last_use, is_64)]
    active: list[tuple[str, int, int, bool]] = []
    free_regs_64: list[int] = []  # available even-aligned register pairs
    free_regs_32: list[int] = []  # available single registers
    quarantine_64: list[int] = []  # FB-5.1: address pairs in 1-step cooldown
    next_gpr = 3 if _reserve_r2_for_addr else 2  # R0-R1 reserved; R2 for 0xc11 addr pair
    # FG27 HARD BAIL: R0 body-temp reuse causes 90 GPU failures.
    # Register shift propagates through scheduling/scoreboard.
    # R0 remains reserved (not allocated).

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
                    elif src_phys % 2 == 1 and _reserve_r2_for_addr and src_phys >= 1:
                        # ALLOC-SUBSYS-2: Odd-aligned co-location for 0xc11
                        # address pair.  Place pair at (src-1):(src) so
                        # the carry-chain lo dest = src-1 and hi = src
                        # (reusing the dying element offset register).
                        lo_slot = src_phys - 1
                        pair_ok = True
                        for aname, areg, alast, a64 in active:
                            if aname == src_name:
                                continue
                            if a64 and areg == lo_slot:
                                pair_ok = False; break
                            if not a64 and areg == lo_slot and alast >= first_def:
                                pair_ok = False; break
                        if pair_ok:
                            phys = lo_slot  # even-aligned pair base
                            if phys in free_regs_64:
                                free_regs_64.remove(phys)
                            if phys in free_regs_32:
                                free_regs_32.remove(phys)
                            if phys + 1 in free_regs_32:
                                free_regs_32.remove(phys + 1)
                            if next_gpr <= phys + 1:
                                next_gpr = phys + 2
                            active = [(n, r, l, w) for n, r, l, w in active
                                      if n != src_name]
                            colocated = True
            # ALLOC-SUBSYS-2b: address-chain co-location (add.u64 dest → cvt pair).
            if not colocated and name in add_colocate:
                cvt_pair_name = add_colocate[name]
                if cvt_pair_name in int_regs:
                    pair_phys = int_regs[cvt_pair_name]
                    # Reuse the cvt result's pair.  The cvt result dies at this
                    # add instruction, so we can safely take over its slot.
                    phys = pair_phys
                    # Remove the cvt result from active (its slot is ours now)
                    active = [(n, r, l, w) for n, r, l, w in active
                              if n != cvt_pair_name]
                    if phys in free_regs_64:
                        free_regs_64.remove(phys)
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
            elif (name in cvt_sources and not _reserve_r2_for_addr
                  and any(r % 2 == 0 for r in free_regs_32)):
                # cvt.u64.u32 source: prefer even-aligned for co-location.
                # Skip when R2 is reserved — odd co-location handles it.
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
                if (name in cvt_sources and next_gpr % 2 != 0
                        and not _reserve_r2_for_addr):
                    # cvt source: bump to even for co-location.
                    # Skip when R2 is reserved — odd co-location handles
                    # the pair at (src-1):(src) instead.
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

    _ra = RegAlloc(
        int_regs=int_regs,
        pred_regs=pred_regs,
        unif_regs=unif_regs,
    )
    # PTXAS-R23C: piggy-back the dead-ld.param.u64 set onto the shared
    # RegAlloc instance so isel._select_ld_param can consult it without
    # a new pipeline-level plumbing change.  Empty for SM_89 and for
    # SM_120 kernels without any redefined u64 param.
    _ra._r23c_dead_ldparam_ids = _r23c_dead_ldparam_ids
    return AllocResult(
        ra=_ra,
        param_offsets=param_offsets,
        num_gprs=next_gpr,
        num_pred=max(next_pred, 1),
        num_uniform=max(next_ur, 5),
        direct_ldc_params=direct_ldc_params if sm_version == 120 else set(),
        addr_pair_colocated=bool(add_colocate),
    )
