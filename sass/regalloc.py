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


def allocate(fn: Function, param_base: int = PARAM_BASE_SM120) -> AllocResult:
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

    # Find which registers are actually referenced in instructions
    from ptx.ir import RegOp
    used_regs: set[str] = set()
    for bb in fn.blocks:
        for inst in bb.instructions:
            if inst.dest and isinstance(inst.dest, RegOp):
                used_regs.add(inst.dest.name)
            for src in inst.srcs:
                if isinstance(src, RegOp):
                    used_regs.add(src.name)

    next_gpr = 2       # R0-R1 reserved (R0=return, R1=stack frame ptr)
    next_pred = 0       # P0..P6
    next_ur = 4         # UR0-UR3 reserved by driver; UR4+ for user

    for rd in fn.reg_decls:
        is_64 = rd.type.width >= 64 and rd.type.kind != ScalarKind.PRED
        is_pred = rd.type.kind == ScalarKind.PRED

        if is_pred:
            for name in rd.names:
                if name in used_regs:
                    pred_regs[name] = next_pred
                    next_pred += 1
        elif is_64:
            for i in range(rd.count):
                name = f"%{rd.name}{i}" if rd.count > 1 else f"%{rd.name}"
                if name not in used_regs:
                    continue  # skip unused registers
                if next_gpr % 2 != 0:
                    next_gpr += 1
                int_regs[name] = next_gpr
                next_gpr += 2
        else:
            for i in range(rd.count):
                name = f"%{rd.name}{i}" if rd.count > 1 else f"%{rd.name}"
                if name not in used_regs:
                    continue
                int_regs[name] = next_gpr
                next_gpr += 1

    # Apply LDG coalescing: dest regs share the addr reg's physical register.
    for dest_name, addr_name in coalesces.items():
        if addr_name in int_regs and dest_name in int_regs:
            int_regs[dest_name] = int_regs[addr_name]

    # Compact allocation: pack register pairs tightly starting at R2
    seen_regs = set()
    remap = {}
    next_r = 2
    for name, reg in sorted(int_regs.items(), key=lambda x: x[1]):
        if reg in remap:
            int_regs[name] = remap[reg]
        elif reg not in seen_regs:
            if next_r % 2 != 0:
                next_r += 1
            remap[reg] = next_r
            int_regs[name] = next_r
            seen_regs.add(reg)
            next_r += 2
        else:
            int_regs[name] = remap.get(reg, reg)
    next_gpr = next_r if int_regs else next_gpr

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
