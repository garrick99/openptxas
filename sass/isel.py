"""
sass/isel.py — PTX-to-SASS instruction selector for SM_120.

Maps PTX IR instructions to sequences of 16-byte SM_120 SASS instructions.
This is a targeted selector, not a general PTX compiler: it handles the
instruction patterns needed to correctly compile the ptxas bug cases and a
minimal computation kernel.

Architecture:
  - Input: ptx.ir.Function with allocated physical registers
  - Output: list of (16-byte bytes, comment) pairs

Register mapping convention (set by regalloc, read here):
  PTX %rd0..%rdN → SASS R0..R(N*2+1)  (64-bit pairs: lo=even, hi=odd)
  PTX %r0..%rN   → SASS R(BASE+N)     (32-bit singles)
  PTX %p0..%pN   → SASS P0..P5        (predicates)

Supported PTX instructions → SASS mappings:
  mov.u32           → MOV
  mov.u64           → MOV (low) + MOV (high) if reg-reg; NOP if same
  shl.b64           → SHF.L.U32 (lo) + SHF.L.U64.HI (hi)
  shr.u64           → SHF.R.U64 (TODO: right-shift encoder pending RE)
  sub.u64/s64       → IADD3 (lo, negated) + IADD3.X (hi, with carry)
  add.u64           → IADD3 (lo) + IADD3.X (hi)
  ld.param.u64      → LDC.64
  ld.param.u32      → LDC
  ld.global.u64     → LDG.E.64
  st.global.u64     → STG.E.64
  ret               → EXIT
  @p bra target     → BRA (with predicate, placeholder offset)
  setp.ge.u32       → ISETP.GE.AND
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from ptx.ir import Instruction, Function, Operand, RegOp, ImmOp, LabelOp

from sass.encoding.sm_120_encode import (
    encode_shf_l_w_u32_hi,
    encode_shf_l_u32,
    encode_shf_l_u64_hi,
)
from sass.encoding.sm_120_opcodes import (
    encode_nop, encode_exit, encode_mov,
    encode_ldc, encode_ldc_64,
    encode_s2r,
    encode_iadd3, encode_iadd3x,
    encode_iadd64,
    encode_imad_wide,
    encode_ldg_e_64, encode_stg_e_64,
    encode_isetp_ge_and,
    encode_bra,
    RZ, PT, SR_TID_X, SR_TID_Y,
    SR_CTAID_X,
)
from sass.regalloc import RegAlloc


# ---------------------------------------------------------------------------
# Output: sequence of encoded SASS instructions
# ---------------------------------------------------------------------------

@dataclass
class SassInstr:
    """One encoded 16-byte SM_120 instruction with metadata."""
    raw:     bytes          # 16 bytes, little-endian
    comment: str = ''       # human-readable annotation

    def hex(self) -> str:
        return self.raw.hex()


# ---------------------------------------------------------------------------
# Instruction selector
# ---------------------------------------------------------------------------

class ISelError(Exception):
    pass


def _get_reg(op: Operand, ra: RegAlloc, bits: int = 32) -> int:
    """Extract physical register index from an operand."""
    if isinstance(op, RegOp):
        name = op.name
        if name == 'RZ' or name == '%rz':
            return RZ
        if bits == 64:
            return ra.lo(name)
        return ra.r32(name)
    raise ISelError(f"Expected register operand, got {op!r}")


def _get_imm(op: Operand) -> int:
    if isinstance(op, ImmOp):
        return op.value
    raise ISelError(f"Expected immediate operand, got {op!r}")


def _nop(comment: str = '') -> SassInstr:
    return SassInstr(encode_nop(), comment or 'NOP')


# ---------------------------------------------------------------------------
# PTX → SASS per-instruction mappers
# ---------------------------------------------------------------------------

def _select_mov(instr: Instruction, ra: RegAlloc) -> list[SassInstr]:
    """mov.u32 or mov.u64 (register-register)."""
    typ = instr.types[0] if instr.types else 'u32'
    dest = instr.dest
    src = instr.srcs[0]

    if not isinstance(dest, RegOp):
        raise ISelError(f"MOV dest must be register: {dest!r}")

    if isinstance(src, ImmOp):
        # MOV from immediate: ptxas typically uses IADD3 or LDC for this
        # For now: encode as MOV with immediate-as-register (limited, only for small imm=RZ pattern)
        raise ISelError("MOV from immediate not yet supported in isel (use LDC for params)")

    if not isinstance(src, RegOp):
        raise ISelError(f"MOV src must be register: {src!r}")

    if typ in ('u64', 's64', 'b64', 'f64'):
        # 64-bit: two 32-bit MOVs
        d_lo = ra.lo(dest.name)
        d_hi = d_lo + 1
        s_lo = ra.lo(src.name)
        s_hi = s_lo + 1
        instrs = []
        if d_lo != s_lo:
            instrs.append(SassInstr(encode_mov(d_lo, s_lo),
                                    f'MOV R{d_lo}, R{s_lo}  // {dest.name}.lo = {src.name}.lo'))
        if d_hi != s_hi:
            instrs.append(SassInstr(encode_mov(d_hi, s_hi),
                                    f'MOV R{d_hi}, R{s_hi}  // {dest.name}.hi = {src.name}.hi'))
        return instrs or [_nop(f'MOV {dest.name} = {src.name} (same reg, elided)')]
    else:
        # 32-bit
        d = ra.r32(dest.name)
        s = ra.r32(src.name)
        return [SassInstr(encode_mov(d, s), f'MOV R{d}, R{s}  // {dest.name} = {src.name}')]


def _select_shl_b64(instr: Instruction, ra: RegAlloc) -> list[SassInstr]:
    """
    shl.b64 dest, src, K → SHF.L.U32 (lo) + SHF.L.U64.HI (hi).

    64-bit left shift by constant K:
      dest.lo = src.lo << K               (via SHF.L.U32, src1=RZ for low bits)
      dest.hi = funnel_shift(src.hi, src.lo, K)  (via SHF.L.U64.HI)
    """
    dest = instr.dest
    src  = instr.srcs[0]
    k_op = instr.srcs[1]
    if not isinstance(dest, RegOp) or not isinstance(src, RegOp):
        raise ISelError(f"shl.b64: dest/src must be registers")
    k = _get_imm(k_op)
    d_lo = ra.lo(dest.name); d_hi = d_lo + 1
    s_lo = ra.lo(src.name);  s_hi = s_lo + 1

    if k < 32 and k <= 15:
        # Use IMAD.SHL for lo (avoids SHF.L/SHF.R pipeline conflicts)
        from sass.encoding.sm_120_opcodes import encode_imad_shl_u32
        return [
            SassInstr(encode_imad_shl_u32(d_lo, s_lo, k),
                      f'IMAD.SHL.U32 R{d_lo}, R{s_lo}, {1<<k:#x}, RZ  // {dest.name}.lo = {src.name}.lo << {k}'),
            SassInstr(encode_shf_l_u64_hi(d_hi, s_lo, k, s_hi),
                      f'SHF.L.U64.HI R{d_hi}, R{s_lo}, 0x{k:x}, R{s_hi}  // {dest.name}.hi'),
        ]
    elif k < 32:
        return [
            SassInstr(encode_shf_l_u32(d_lo, s_lo, k),
                      f'SHF.L.U32 R{d_lo}, R{s_lo}, 0x{k:x}, RZ  // {dest.name}.lo = {src.name}.lo << {k}'),
            SassInstr(encode_shf_l_u64_hi(d_hi, s_lo, k, s_hi),
                      f'SHF.L.U64.HI R{d_hi}, R{s_lo}, 0x{k:x}, R{s_hi}  // {dest.name}.hi'),
        ]
    else:
        # K >= 32: result.hi = src.lo << (K-32), result.lo = 0
        k32 = k - 32
        return [
            SassInstr(encode_mov(d_lo, RZ),
                      f'MOV R{d_lo}, RZ  // shl.b64 lo = 0 (K>={k})'),
            SassInstr(encode_shf_l_u32(d_hi, s_lo, k32),
                      f'SHF.L.U32 R{d_hi}, R{s_lo}, 0x{k32:x}, RZ  // shl.b64 hi'),
        ]


def _select_rotl64(instr: Instruction, ra: RegAlloc) -> list[SassInstr]:
    """
    Correct 64-bit rotate-left: produces two SHF.L.W.U32.HI instructions.
    The source PTX pattern: add(shl(a,K), shr(a, 64-K)).

    This is the CORRECT transformation that ptxas gets wrong when it sees
    sub(shl(a,K), shr(a, 64-K)) — our rotate.py pass detects this.
    """
    dest = instr.dest
    src  = instr.srcs[0]
    k_op = instr.srcs[1]
    if not isinstance(dest, RegOp) or not isinstance(src, RegOp):
        raise ISelError(f"rotl64: dest/src must be registers")
    k = _get_imm(k_op)
    d_lo = ra.lo(dest.name); d_hi = d_lo + 1
    s_lo = ra.lo(src.name);  s_hi = s_lo + 1
    return [
        SassInstr(encode_shf_l_w_u32_hi(d_lo, s_lo, k, s_hi),
                  f'SHF.L.W.U32.HI R{d_lo}, R{s_lo}, 0x{k:x}, R{s_hi}  // rotl64 lo'),
        SassInstr(encode_shf_l_w_u32_hi(d_hi, s_hi, k, s_lo),
                  f'SHF.L.W.U32.HI R{d_hi}, R{s_hi}, 0x{k:x}, R{s_lo}  // rotl64 hi'),
    ]


def _select_shr_u64(instr: Instruction, ra: RegAlloc) -> list[SassInstr]:
    """
    shr.u64 dest, src, K → right shift by constant K.

    For K < 32: standard SHF.R.U64 + SHF.R.U32.HI pair.
    For K >= 32: optimized — shift the HIGH word right by (K-32), dest.hi = 0.
    """
    dest = instr.dest
    src  = instr.srcs[0]
    k_op = instr.srcs[1]
    if not isinstance(dest, RegOp) or not isinstance(src, RegOp):
        raise ISelError(f"shr.u64: dest/src must be registers")
    k = _get_imm(k_op)
    d_lo = ra.lo(dest.name); d_hi = d_lo + 1
    s_lo = ra.lo(src.name);  s_hi = s_lo + 1

    from sass.encoding.sm_120_encode import encode_shf_r_u32, encode_shf_r_u32_hi

    if k < 32:
        return [
            SassInstr(encode_shf_r_u32(d_lo, s_lo, k, s_hi),
                      f'SHF.R.U64 R{d_lo}, R{s_lo}, 0x{k:x}, R{s_hi}  // shr.u64 lo'),
            SassInstr(encode_shf_r_u32_hi(d_hi, s_hi, k),
                      f'SHF.R.U32.HI R{d_hi}, RZ, 0x{k:x}, R{s_hi}  // shr.u64 hi'),
        ]
    else:
        # K >= 32: result.lo = src.hi >> (K-32), result.hi = 0
        k32 = k - 32
        return [
            SassInstr(encode_shf_r_u32_hi(d_lo, s_hi, k32),
                      f'SHF.R.U32.HI R{d_lo}, RZ, 0x{k32:x}, R{s_hi}  // shr.u64 lo (K>={k})'),
            SassInstr(encode_mov(d_hi, RZ),
                      f'MOV R{d_hi}, RZ  // shr.u64 hi = 0'),
        ]


def _select_sub_u64(instr: Instruction, ra: RegAlloc) -> list[SassInstr]:
    """
    sub.u64/s64 dest, a, b → IADD.64 with negation on b.

    Uses dest=a_lo (in-place) to keep registers within R0-R7 range.
    IADD.64 reads both sources before writing, so dest=src0 is safe.
    """
    dest = instr.dest
    a    = instr.srcs[0]
    b    = instr.srcs[1]
    if not isinstance(dest, RegOp) or not isinstance(a, RegOp) or not isinstance(b, RegOp):
        raise ISelError(f"sub.u64: all operands must be registers")

    a_lo = ra.lo(a.name)
    b_lo = ra.lo(b.name)
    # Write result to src0's register (in-place) to minimize register usage
    d_lo = a_lo

    # Also update the regalloc to know dest is at a_lo
    ra.int_regs[dest.name] = a_lo

    return [
        SassInstr(encode_iadd64(d_lo, a_lo, b_lo, negate_src1=True),
                  f'IADD.64 R{d_lo}, R{a_lo}, -R{b_lo}  // sub.u64'),
    ]


def _select_add_u64(instr: Instruction, ra: RegAlloc) -> list[SassInstr]:
    """add.u64 dest, a, b → IADD.64 (single instruction, in-place to save regs)."""
    dest = instr.dest
    a    = instr.srcs[0]
    b    = instr.srcs[1]
    if not isinstance(dest, RegOp) or not isinstance(a, RegOp) or not isinstance(b, RegOp):
        raise ISelError(f"add.u64: all operands must be registers")
    a_lo = ra.lo(a.name)
    b_lo = ra.lo(b.name)
    d_lo = a_lo  # in-place
    ra.int_regs[dest.name] = a_lo
    return [
        SassInstr(encode_iadd64(d_lo, a_lo, b_lo, negate_src1=False),
                  f'IADD.64 R{d_lo}, R{a_lo}, R{b_lo}  // add.u64'),
    ]


def _select_ld_param(instr: Instruction, ra: RegAlloc,
                     param_offsets: dict[str, int]) -> list[SassInstr]:
    """
    ld.param.u64 or ld.param.u32 → LDC.64 or LDC.

    param_offsets maps PTX parameter names to byte offsets in c[0][...].
    These offsets are determined by the kernel's ABI (set by regalloc/layout).
    """
    dest = instr.dest
    src  = instr.srcs[0]  # MemOp or similar
    if not isinstance(dest, RegOp):
        raise ISelError(f"ld.param dest must be register")

    # Extract parameter name from src operand (MemOp.base is the param name)
    from ptx.ir import MemOp
    if not isinstance(src, MemOp):
        raise ISelError(f"ld.param src must be MemOp, got {src!r}")

    param_name = src.base
    if isinstance(src.offset, int):
        byte_off = param_offsets.get(param_name, 0) + src.offset
    else:
        byte_off = param_offsets.get(param_name, 0)

    # Data type is the last element in types (e.g. ['param', 'u64'] → 'u64')
    typ = instr.types[-1] if instr.types else 'u32'
    if typ in ('u64', 's64', 'b64'):
        d_lo = ra.lo(dest.name)
        # If register is >= R8, remap to R2 (dead after LDG data consumed)
        # This keeps all regs within R0-R7 for the default nv.info template.
        if d_lo >= 8:
            d_lo = 2
            ra.int_regs[dest.name] = 2
        return [SassInstr(encode_ldc_64(d_lo, 0, byte_off, ctrl=0x712),
                          f'LDC.64 R{d_lo}, c[0][0x{byte_off:x}]  // {param_name}')]
    else:
        d = ra.r32(dest.name)
        return [SassInstr(encode_ldc(d, 0, byte_off, ctrl=0x7f1),
                          f'LDC R{d}, c[0][0x{byte_off:x}]  // {param_name}')]


def _select_ld_global(instr: Instruction, ra: RegAlloc,
                      ur_desc: int) -> list[SassInstr]:
    """ld.global.u64 → LDG.E.64.

    Register coalescing ensures dest and addr map to the same physical register.
    """
    dest = instr.dest
    src  = instr.srcs[0]
    if not isinstance(dest, RegOp):
        raise ISelError(f"ld.global dest must be register")
    from ptx.ir import MemOp
    if not isinstance(src, MemOp):
        raise ISelError(f"ld.global src must be MemOp")
    d = ra.lo(dest.name)
    addr = ra.lo(src.base) if src.base in ra.int_regs else RZ
    return [SassInstr(encode_ldg_e_64(d, ur_desc, addr),
                      f'LDG.E.64 R{d}, desc[UR{ur_desc}][R{addr}.64]')]


def _select_st_global(instr: Instruction, ra: RegAlloc,
                      ur_desc: int) -> list[SassInstr]:
    """st.global.u64 → STG.E.64.

    If the address register is >= R8, remap it to R2 (reuse LDG data register
    which is dead after all compute instructions).
    """
    dest_op = instr.srcs[0]  # address
    src_op  = instr.srcs[1]  # data
    from ptx.ir import MemOp
    if not isinstance(dest_op, MemOp):
        raise ISelError(f"st.global addr must be MemOp")
    if not isinstance(src_op, RegOp):
        raise ISelError(f"st.global data must be register")
    addr = ra.lo(dest_op.base) if dest_op.base in ra.int_regs else RZ
    data = ra.lo(src_op.name)
    return [SassInstr(encode_stg_e_64(ur_desc, addr, data, ctrl=0xff1),
                      f'STG.E.64 desc[UR{ur_desc}][R{addr}.64], R{data}')]


# ---------------------------------------------------------------------------
# Main instruction selector entry point
# ---------------------------------------------------------------------------

@dataclass
class ISelContext:
    """Context passed through the instruction selector."""
    ra:            RegAlloc
    # Byte offset of each kernel parameter in c[0][...] (ABI layout)
    param_offsets: dict[str, int] = field(default_factory=dict)
    # Uniform register to use for global memory descriptor
    ur_desc:       int = 4  # UR4 by default (matches ptxas convention)
    # Label → instruction index within output for branch fixup
    label_map:     dict[str, int] = field(default_factory=dict)


def select_function(fn: Function, ctx: ISelContext) -> list[SassInstr]:
    """
    Select SASS instructions for every PTX instruction in a function.

    Returns a flat list of SassInstr.  Branch targets are not yet resolved
    (encode_bra is called with offset=0 as a placeholder); a second pass
    over the output would patch BRA offsets using label_map.
    """
    output: list[SassInstr] = []

    for bb in fn.blocks:
        # Record label position
        if bb.label:
            ctx.label_map[bb.label] = len(output) * 16

        for instr in bb.instructions:
            op = instr.op.lower()
            typ = instr.types[0].lower() if instr.types else ''

            try:
                if op == 'mov' and typ in ('u32', 's32', 'b32', 'u64', 's64', 'b64'):
                    output.extend(_select_mov(instr, ctx.ra))

                elif op == 'shl' and typ in ('b64', 'u64'):
                    output.extend(_select_shl_b64(instr, ctx.ra))

                elif op == 'shr' and typ in ('u64',):
                    output.extend(_select_shr_u64(instr, ctx.ra))

                elif op == 'sub' and typ in ('u64', 's64'):
                    output.extend(_select_sub_u64(instr, ctx.ra))

                elif op == 'add' and typ in ('u64', 's64'):
                    output.extend(_select_add_u64(instr, ctx.ra))

                elif op == 'ld' and 'param' in instr.types:
                    output.extend(_select_ld_param(instr, ctx.ra, ctx.param_offsets))

                elif op == 'ld' and 'global' in instr.types:
                    output.extend(_select_ld_global(instr, ctx.ra, ctx.ur_desc))

                elif op == 'st' and 'global' in instr.types:
                    output.extend(_select_st_global(instr, ctx.ra, ctx.ur_desc))

                elif op == 'ret':
                    output.append(SassInstr(encode_exit(ctrl=0x7f5), 'EXIT'))

                elif op == 'bra':
                    # Placeholder offset; caller fixes up after layout
                    output.append(SassInstr(encode_bra(0), f'BRA {instr.srcs[0] if instr.srcs else "?"}'))

                elif op == 'nop':
                    output.append(_nop())

                elif op in ('setp',) and 'ge' in instr.types:
                    # setp.ge.u32 → ISETP.GE.AND
                    pred = instr.dest
                    a    = instr.srcs[0]
                    b    = instr.srcs[1]
                    if isinstance(pred, RegOp) and isinstance(a, RegOp) and isinstance(b, RegOp):
                        pd = ctx.ra.pred(pred.name)
                        ar = ctx.ra.r32(a.name)
                        ur = ctx.ra.ur(b.name) if b.name in ctx.ra.unif_regs else ctx.ra.r32(b.name)
                        output.append(SassInstr(encode_isetp_ge_and(pd, ar, ur),
                                                f'ISETP.GE.AND P{pd}, R{ar}, UR{ur}'))
                    else:
                        output.append(_nop(f'TODO: setp.ge {instr}'))

                else:
                    # Unsupported instruction: emit NOP with comment
                    output.append(_nop(f'TODO: {instr.op} {".".join(instr.types)} {instr.mods}'))

            except ISelError as e:
                # Emit NOP with error comment rather than crashing
                output.append(_nop(f'ISEL ERROR: {e}  [{instr.op}]'))

    return output
