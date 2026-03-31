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


def _find_ldg_coalesces(fn: Function) -> dict[str, str]:
    """
    Find ld.global patterns where dest and addr can share registers.

    Returns a dict mapping dest_reg_name → addr_reg_name for coalescing.
    e.g. {'%rd1': '%rd0'} means %rd1 should share %rd0's physical register.
    """
    from ptx.ir import RegOp, MemOp
    coalesces: dict[str, str] = {}
    for bb in fn.blocks:
        for inst in bb.instructions:
            if inst.op == 'ld' and 'global' in inst.types:
                if inst.dest and inst.srcs:
                    dest = inst.dest
                    src = inst.srcs[0]
                    if isinstance(dest, RegOp) and isinstance(src, MemOp):
                        coalesces[dest.name] = f'%{src.base}' if not src.base.startswith('%') else src.base
    return coalesces


def allocate(fn: Function, param_base: int = PARAM_BASE_SM120,
             has_capmerc: bool = False) -> AllocResult:
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
    from ptx.ir import RegOp, MemOp
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

    next_pred = 0
    next_ur = 4

    # Predicate allocation (simple sequential)
    for rd in fn.reg_decls:
        if rd.type.kind == ScalarKind.PRED:
            for name in rd.names:
                if name in used_regs:
                    pred_regs[name] = next_pred
                    next_pred += 1

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
        # SM_120 HARDWARE LIMIT: Without correct per-instruction capmerc metadata,
        # the GPU only reliably supports R0-R13. The driver validates capmerc against
        # .text at load time, so ptxas-generated capmerc only works when our instruction
        # schedule matches. Until we can generate our own capmerc, cap at R14.
        # NOTE: The EIATTR 0x5a blob (52 bytes) is a universal ptxas 13.0 signature
        # needed to authenticate capmerc. See memory/project_openptxas_sm120_rules.md.
        # SM_120: without correct capmerc, R12+ is unreliable for some instruction
        # patterns (FSEL at R7+ crashes). Cap at 12 for safety.
        _MAX_GPR = 12
        if is_64:
            if free_regs_64:
                phys = free_regs_64.pop(0)
            else:
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
            if free_regs_32:
                phys = free_regs_32.pop(0)
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

    # LDG coalescing disabled — causes live range conflicts when multiple loads
    # are active simultaneously. The linear scan allocator handles register reuse.
    # TODO: re-enable with interference graph check

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
