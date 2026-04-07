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

    # Collect registers that require 4-register alignment (HMMA/DMMA/IMMA dest).
    # mma.sync.aligned instructions write a 4-register accumulator; hardware
    # requires dest % 4 == 0.  We note the first register of the dest tuple.
    # We also mark the 3 following registers in the same .reg declaration as
    # "quad followers" — they must be allocated consecutively after the base.
    quad_align_regs: set[str] = set()
    quad_follow_regs: set[str] = set()
    from ptx.ir import RegOp as _RegOp
    for inst in all_instrs:
        if inst.op == 'mma' and 'sync' in inst.types and inst.dest is not None:
            if isinstance(inst.dest, _RegOp):
                quad_align_regs.add(inst.dest.name)
    # Find the following 3 registers in the same RegDecl for each quad base.
    for rd in fn.reg_decls:
        if rd.type.kind == ScalarKind.PRED:
            continue
        for i, nm in enumerate(rd.names):
            if nm in quad_align_regs:
                for j in range(1, 4):
                    if i + j < len(rd.names):
                        quad_follow_regs.add(rd.names[i + j])

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
            # Don't skip GPR for UR-loaded params — the isel's LDC fallback
            # for 3rd+ params needs GPRs even for UR-declared params.
            # if name in ur_only_f64_regs:
            #     continue
            first = reg_first_def.get(name, 0)
            last = reg_last_use.get(name, len(all_instrs))
            reg_info.append((name, is_64, first, last))

    # Sort by first definition
    reg_info.sort(key=lambda x: x[2])

    # Linear scan: assign physical registers, reclaiming dead ones
    # active = [(name, phys_reg, last_use, is_64)]
    active: list[tuple[str, int, int, bool]] = []
    free_regs_64: list[int] = []  # available even-aligned register pairs
    free_regs_32: list[int] = []  # available single registers
    next_gpr = 2  # R0-R1 reserved

    for name, is_64, first_def, last_use in reg_info:
        # Expire old intervals: free registers whose last use is before this def
        new_active = []
        for aname, areg, alast, a64 in active:
            if alast < first_def:
                # This register is dead — reclaim it
                if a64:
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
            if free_regs_64:
                phys = free_regs_64.pop(0)
            else:
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
    )
