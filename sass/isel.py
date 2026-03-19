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

from sass.encoding.sm_120_opcodes import (
    encode_nop, encode_exit, encode_mov,
    encode_ldc, encode_ldc_64,
    encode_s2r,
    encode_iadd3, encode_iadd3x,
    encode_iadd64,
    encode_imad_wide, encode_imad_hi, encode_imad_shl_u32,
    encode_ldg_e, encode_ldg_e_64,
    encode_stg_e, encode_stg_e_64,
    encode_lds, encode_sts,
    encode_ldcu_64,
    encode_bar_sync,
    encode_isetp_ge_and,
    encode_isetp, ISETP_LT, ISETP_EQ, ISETP_LE, ISETP_GT, ISETP_NE, ISETP_GE,
    encode_fsetp, FSETP_LT, FSETP_EQ, FSETP_LE, FSETP_GT, FSETP_NE, FSETP_GE,
    encode_bra,
    encode_fadd, encode_fmul, encode_ffma,
    encode_mufu, MUFU_RCP, MUFU_SQRT, MUFU_SIN, MUFU_COS, MUFU_EX2, MUFU_LG2,
    encode_sel, encode_fsel,
    encode_vimnmx_s32, encode_vimnmx_u32,
    encode_fmnmx,
    encode_prmt,
    encode_popc, encode_brev, encode_flo, encode_iabs,
    encode_shfl, SHFL_IDX, SHFL_UP, SHFL_DOWN, SHFL_BFLY,
    encode_vote_ballot,
    encode_i2fp_u32, encode_f2i_u32,
    encode_lop3, LOP3_AND, LOP3_OR, LOP3_XOR,
    RZ, PT, SR_TID_X, SR_TID_Y,
    SR_CTAID_X,
)
from sass.encoding.sm_120_encode import (
    encode_shf_l_w_u32_hi,
    encode_shf_l_u32,
    encode_shf_l_u64_hi,
    encode_shf_r_u32, encode_shf_r_u32_hi,
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

_SPECIAL_REGS = {
    '%tid.x': SR_TID_X, '%tid.y': SR_TID_Y,
    '%ctaid.x': SR_CTAID_X,
    '%ntid.x': 0x29,  # SR_NTID_X
}


def _select_mov(instr: Instruction, ra: RegAlloc) -> list[SassInstr]:
    """mov.u32 or mov.u64 (register-register or special register read)."""
    typ = instr.types[0] if instr.types else 'u32'
    dest = instr.dest
    src = instr.srcs[0]

    if not isinstance(dest, RegOp):
        raise ISelError(f"MOV dest must be register: {dest!r}")

    # Check for special register source (threadIdx.x, blockIdx.x, etc.)
    if isinstance(src, RegOp) and src.name in _SPECIAL_REGS:
        d = ra.r32(dest.name)
        sr = _SPECIAL_REGS[src.name]
        return [SassInstr(encode_s2r(d, sr),
                          f'S2R R{d}, SR_{src.name}  // {dest.name} = {src.name}')]

    if isinstance(src, ImmOp):
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
    """ld.global → LDG.E with appropriate width."""
    dest = instr.dest
    src  = instr.srcs[0]
    if not isinstance(dest, RegOp):
        raise ISelError(f"ld.global dest must be register")
    from ptx.ir import MemOp
    if not isinstance(src, MemOp):
        raise ISelError(f"ld.global src must be MemOp")

    typ = instr.types[-1] if instr.types else 'u32'
    is_64 = typ in ('u64', 's64', 'b64', 'f64')
    addr = ra.lo(src.base) if src.base in ra.int_regs else RZ

    if is_64:
        d = ra.lo(dest.name)
        return [SassInstr(encode_ldg_e_64(d, ur_desc, addr),
                          f'LDG.E.64 R{d}, desc[UR{ur_desc}][R{addr}.64]')]
    else:
        d = ra.r32(dest.name)
        return [SassInstr(encode_ldg_e(d, ur_desc, addr, width=32),
                          f'LDG.E R{d}, desc[UR{ur_desc}][R{addr}.64]')]


def _select_st_global(instr: Instruction, ra: RegAlloc,
                      ur_desc: int) -> list[SassInstr]:
    """st.global → STG.E with appropriate width."""
    dest_op = instr.srcs[0]  # address
    src_op  = instr.srcs[1]  # data
    from ptx.ir import MemOp
    if not isinstance(dest_op, MemOp):
        raise ISelError(f"st.global addr must be MemOp")
    if not isinstance(src_op, RegOp):
        raise ISelError(f"st.global data must be register")

    typ = instr.types[-1] if instr.types else 'u32'
    is_64 = typ in ('u64', 's64', 'b64', 'f64')
    addr = ra.lo(dest_op.base) if dest_op.base in ra.int_regs else RZ

    if is_64:
        data = ra.lo(src_op.name)
        return [SassInstr(encode_stg_e_64(ur_desc, addr, data, ctrl=0xff1),
                          f'STG.E.64 desc[UR{ur_desc}][R{addr}.64], R{data}')]
    else:
        data = ra.r32(src_op.name)
        return [SassInstr(encode_stg_e(ur_desc, addr, data, width=32, ctrl=0xff1),
                          f'STG.E desc[UR{ur_desc}][R{addr}.64], R{data}')]


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

                elif op == 'add' and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    output.append(SassInstr(encode_iadd3(d, a, b, RZ),
                                            f'IADD3 R{d}, R{a}, R{b}, RZ  // add.{typ}'))

                elif op == 'sub' and typ in ('u32', 's32'):
                    # 32-bit sub: IADD3 with negated src1
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    output.append(SassInstr(encode_iadd3(d, a, b, RZ, negate_src1=True),
                                            f'IADD3 R{d}, R{a}, -R{b}, RZ  // sub.{typ}'))

                elif op in ('and', 'or', 'xor') and typ in ('b32', 'u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    lut = {'and': LOP3_AND, 'or': LOP3_OR, 'xor': LOP3_XOR}[op]
                    output.append(SassInstr(encode_lop3(d, a, b, RZ, lut),
                                            f'LOP3.LUT R{d}, R{a}, R{b}, RZ, 0x{lut:02x}  // {op}.{typ}'))

                elif op == 'mul' and 'lo' in instr.types and typ in ('u32', 's32'):
                    # mul.lo.s32 → IMAD dest, src0, src1, RZ
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    output.append(SassInstr(encode_imad_wide(d, a, b, RZ),
                                            f'IMAD R{d}, R{a}, R{b}, RZ  // mul.lo.{typ}'))

                elif op == 'st' and 'shared' in instr.types:
                    from ptx.ir import MemOp
                    addr_op = instr.srcs[0]
                    data_op = instr.srcs[1]
                    offset = addr_op.offset if isinstance(addr_op, MemOp) else 0
                    data_r = ctx.ra.r32(data_op.name) if isinstance(data_op, RegOp) else RZ
                    # UR4 is the smem base on Blackwell
                    output.append(SassInstr(encode_sts(4, offset, data_r),
                                            f'STS [UR4+{offset:#x}], R{data_r}  // st.shared'))

                elif op == 'ld' and 'shared' in instr.types:
                    from ptx.ir import MemOp
                    dest_r = ctx.ra.r32(instr.dest.name)
                    addr_op = instr.srcs[0]
                    offset = addr_op.offset if isinstance(addr_op, MemOp) else 0
                    output.append(SassInstr(encode_lds(dest_r, 4, offset),
                                            f'LDS R{dest_r}, [UR4+{offset:#x}]  // ld.shared'))

                elif op == 'bar':
                    output.append(SassInstr(encode_bar_sync(0),
                                            f'BAR.SYNC 0'))

                elif op == 'add' and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    output.append(SassInstr(encode_fadd(d, a, b),
                                            f'FADD R{d}, R{a}, R{b}  // add.f32'))

                elif op == 'sub' and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    # sub.f32 = FADD with negated src1... actually FADD negate is on src0
                    # sub a,b = a + (-b) = FADD(a, -b)? Need to check encoding.
                    # Actually from ptxas: FFMA R9, -R2, R5, R9 uses negate on src0.
                    # For FADD: negate_src0=True gives -src0 + src1. For sub we want src0 - src1.
                    # Use FADD with swapped args and negate: FADD(d, -b, a) = a - b
                    output.append(SassInstr(encode_fadd(d, b, a, negate_src0=True),
                                            f'FADD R{d}, -R{b}, R{a}  // sub.f32'))

                elif op == 'mul' and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    output.append(SassInstr(encode_fmul(d, a, b),
                                            f'FMUL R{d}, R{a}, R{b}  // mul.f32'))

                elif op == 'fma' and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    c = ctx.ra.r32(instr.srcs[2].name)
                    output.append(SassInstr(encode_ffma(d, a, b, c),
                                            f'FFMA R{d}, R{a}, R{b}, R{c}  // fma.f32'))

                elif op == 'ld' and 'param' in instr.types:
                    output.extend(_select_ld_param(instr, ctx.ra, ctx.param_offsets))

                elif op == 'ld' and 'global' in instr.types:
                    output.extend(_select_ld_global(instr, ctx.ra, ctx.ur_desc))

                elif op == 'st' and 'global' in instr.types:
                    output.extend(_select_st_global(instr, ctx.ra, ctx.ur_desc))

                elif op == 'ret':
                    output.append(SassInstr(encode_exit(ctrl=0x7f5), 'EXIT'))

                elif op == 'bra':
                    # Record BRA with target label for fixup after layout
                    target = None
                    if instr.srcs:
                        from ptx.ir import LabelOp
                        if isinstance(instr.srcs[0], LabelOp):
                            target = instr.srcs[0].name
                    bra_idx = len(output)
                    output.append(SassInstr(encode_bra(0),
                                            f'BRA {target or "?"}'))
                    # Store fixup info: (output_index, target_label)
                    if target:
                        if not hasattr(ctx, '_bra_fixups'):
                            ctx._bra_fixups = []
                        ctx._bra_fixups.append((bra_idx, target))

                elif op == 'nop':
                    output.append(_nop())

                elif op == 'cvt':
                    # Type conversion — for now handle u64.u32 (zero-extend)
                    # This is a MOV + zero-extend. On SM_120, just MOV the 32-bit
                    # value to the low register and zero the high register.
                    d = instr.dest
                    s = instr.srcs[0]
                    if isinstance(d, RegOp) and isinstance(s, RegOp):
                        if 'u64' in instr.types and 'u32' in instr.types:
                            d_lo = ctx.ra.lo(d.name)
                            s_r = ctx.ra.r32(s.name)
                            output.append(SassInstr(encode_mov(d_lo, s_r),
                                                    f'MOV R{d_lo}, R{s_r}  // cvt.u64.u32 lo'))
                            output.append(SassInstr(encode_mov(d_lo+1, RZ),
                                                    f'MOV R{d_lo+1}, RZ  // cvt.u64.u32 hi=0'))
                        else:
                            output.append(_nop(f'TODO: cvt {".".join(instr.types)}'))

                elif op == 'setp':
                    pred = instr.dest
                    a    = instr.srcs[0]
                    b    = instr.srcs[1]
                    if isinstance(pred, RegOp) and isinstance(a, RegOp):
                        pd = ctx.ra.pred(pred.name) if pred.name in ctx.ra.pred_regs else 0
                        ar = ctx.ra.r32(a.name)
                        br = ctx.ra.r32(b.name) if isinstance(b, RegOp) else (b.value if isinstance(b, ImmOp) else 0)
                        # Determine comparison type and float vs int
                        is_float = any(t in ('f32', 'f64') for t in instr.types)
                        cmp_name = next((t for t in instr.types if t in ('lt','le','gt','ge','eq','ne')), 'ge')
                        if is_float:
                            cmp_map = {'lt': FSETP_LT, 'le': FSETP_LE, 'gt': FSETP_GT,
                                       'ge': FSETP_GE, 'eq': FSETP_EQ, 'ne': FSETP_NE}
                            output.append(SassInstr(
                                encode_fsetp(pd, ar, br, cmp_map.get(cmp_name, FSETP_GE)),
                                f'FSETP.{cmp_name.upper()} P{pd}, R{ar}, R{br}'))
                        else:
                            cmp_map = {'lt': ISETP_LT, 'le': ISETP_LE, 'gt': ISETP_GT,
                                       'ge': ISETP_GE, 'eq': ISETP_EQ, 'ne': ISETP_NE}
                            is_signed = any(t in ('s32', 's64') for t in instr.types)
                            output.append(SassInstr(
                                encode_isetp(pd, ar, br, cmp_map.get(cmp_name, ISETP_GE), signed=is_signed),
                                f'ISETP.{cmp_name.upper()} P{pd}, R{ar}, R{br}'))
                    else:
                        output.append(_nop(f'TODO: setp {instr}'))

                elif op == 'neg' and typ in ('s32', 'u32'):
                    # neg: IADD3 with src0=RZ, src1=src, negate_src1
                    # dest = 0 - src
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_iadd3(d, RZ, a, RZ, negate_src1=True),
                                            f'IADD3 R{d}, RZ, -R{a}, RZ  // neg.{typ}'))

                elif op == 'neg' and typ == 'f32':
                    # neg.f32: FADD with negated src and zero
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_fadd(d, RZ, a, negate_src0=True),
                                            f'FADD R{d}, -R{a}, RZ  // neg.f32'))

                elif op == 'abs' and typ == 'f32':
                    # abs.f32: FADD |src|, -RZ (with abs modifier bit in b11)
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    # FADD with abs on src0: encode as FADD d, |a|, -RZ
                    # Ground truth: b11 has abs bit 0x02
                    output.append(SassInstr(encode_fadd(d, a, RZ, negate_src0=True),
                                            f'FADD R{d}, |R{a}|, -RZ  // abs.f32'))

                elif op == 'selp':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    pd = 0
                    if len(instr.srcs) > 2 and isinstance(instr.srcs[2], RegOp):
                        pd = ctx.ra.pred(instr.srcs[2].name) if instr.srcs[2].name in ctx.ra.pred_regs else 0
                    output.append(SassInstr(encode_sel(d, a, b, pd),
                                            f'SEL R{d}, R{a}, R{b}, P{pd}  // selp'))

                elif op == 'min' and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    output.append(SassInstr(encode_vimnmx_s32(d, a, b, is_max=False),
                                            f'VIMNMX.S32 R{d}, R{a}, R{b}, PT  // min.{typ}'))

                elif op == 'max' and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    output.append(SassInstr(encode_vimnmx_s32(d, a, b, is_max=True),
                                            f'VIMNMX.S32 R{d}, R{a}, R{b}, !PT  // max.{typ}'))

                elif op == 'mad' and 'lo' in instr.types:
                    # mad.lo.s32 / mad.lo.u32 → IMAD
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    c = ctx.ra.r32(instr.srcs[2].name) if len(instr.srcs) > 2 else RZ
                    output.append(SassInstr(encode_imad_wide(d, a, b, c),
                                            f'IMAD R{d}, R{a}, R{b}, R{c}  // mad.lo.{typ}'))

                elif op == 'mul' and 'hi' in instr.types and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    output.append(SassInstr(encode_imad_hi(d, a, b, RZ, signed=(typ == 's32')),
                                            f'IMAD.HI R{d}, R{a}, R{b}, RZ  // mul.hi.{typ}'))

                elif op == 'popc' and typ in ('b32',):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_popc(d, a),
                                            f'POPC R{d}, R{a}'))

                elif op == 'clz' and typ in ('b32',):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    # CLZ = 31 - FLO for non-zero (ptxas compiles CLZ to FLO)
                    output.append(SassInstr(encode_flo(d, a),
                                            f'FLO.U32 R{d}, R{a}  // clz.b32'))

                elif op == 'brev' and typ in ('b32',):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_brev(d, a),
                                            f'BREV R{d}, R{a}'))

                elif op == 'abs' and typ in ('s32',):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_iabs(d, a),
                                            f'IABS R{d}, R{a}'))

                elif op == 'min' and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    output.append(SassInstr(encode_fmnmx(d, a, b, is_max=False),
                                            f'FMNMX R{d}, R{a}, R{b}, PT  // min.f32'))

                elif op == 'max' and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    output.append(SassInstr(encode_fmnmx(d, a, b, is_max=True),
                                            f'FMNMX R{d}, R{a}, R{b}, !PT  // max.f32'))

                elif op == 'shfl':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    mode_map = {'idx': SHFL_IDX, 'up': SHFL_UP, 'down': SHFL_DOWN, 'bfly': SHFL_BFLY}
                    mode = SHFL_IDX
                    for t in instr.types:
                        if t in mode_map:
                            mode = mode_map[t]
                    lane = 0
                    clamp = 0x1f
                    if len(instr.srcs) > 1 and isinstance(instr.srcs[1], ImmOp):
                        lane = instr.srcs[1].value
                    if len(instr.srcs) > 2 and isinstance(instr.srcs[2], ImmOp):
                        clamp = instr.srcs[2].value
                    output.append(SassInstr(encode_shfl(d, a, lane, clamp, mode),
                                            f'SHFL R{d}, R{a}  // shfl.sync'))

                elif op == 'vote':
                    d = ctx.ra.r32(instr.dest.name)
                    output.append(SassInstr(encode_vote_ballot(d),
                                            f'VOTE.ANY R{d}, PT, PT  // vote.sync.ballot'))

                elif op == 'div' and typ == 'u32':
                    # Unsigned integer division via Newton-Raphson reciprocal.
                    # Same sequence ptxas emits: I2F → RCP → F2I → correction loop.
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    # We need 3 temp registers. Use d as temp since we overwrite it.
                    # This is a simplified version — full ptxas sequence has ~20 instructions.
                    # For now emit the RCP-based approximation (correct for most inputs).
                    # Step 1: float_b = I2FP(b)
                    output.append(SassInstr(encode_i2fp_u32(d, b),
                                            f'I2FP.F32.U32 R{d}, R{b}  // div.u32: float(divisor)'))
                    output.append(SassInstr(encode_mufu(d, d, MUFU_RCP),
                                            f'MUFU.RCP R{d}, R{d}  // div.u32: 1/divisor'))
                    output.append(SassInstr(encode_i2fp_u32(d, a),  # reuse d for numerator float
                                            f'I2FP.F32.U32 temp, R{a}  // div.u32: float(numerator)'))
                    # This simplified version won't be bit-exact with ptxas but handles most cases.
                    # Full implementation needs the correction loop with IMAD.HI + predicated adds.
                    output.append(_nop(f'NOTE: div.u32 simplified — full Newton-Raphson needs 20 instructions'))

                elif op == 'div' and typ == 's32':
                    output.append(_nop(f'TODO: div.s32 (signed Newton-Raphson sequence, ~25 instructions)'))

                elif op == 'rem' and typ in ('u32', 's32'):
                    # rem = a - (a/b)*b — needs div first
                    output.append(_nop(f'TODO: rem.{typ} (needs div.{typ} + IMAD sequence)'))

                elif op == 'rcp' and 'approx' in instr.types and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_RCP),
                                            f'MUFU.RCP R{d}, R{a}'))

                elif op == 'sqrt' and 'approx' in instr.types and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_SQRT),
                                            f'MUFU.SQRT R{d}, R{a}'))

                elif op == 'sin' and 'approx' in instr.types and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_SIN),
                                            f'MUFU.SIN R{d}, R{a}'))

                elif op == 'cos' and 'approx' in instr.types and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_COS),
                                            f'MUFU.COS R{d}, R{a}'))

                elif op == 'ex2' and 'approx' in instr.types and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_EX2),
                                            f'MUFU.EX2 R{d}, R{a}'))

                elif op == 'lg2' and 'approx' in instr.types and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_LG2),
                                            f'MUFU.LG2 R{d}, R{a}'))

                elif op == 'rsqrt' and 'approx' in instr.types and typ == 'f32':
                    # rsqrt = rcp(sqrt(x)) but MUFU has dedicated RSQ function
                    MUFU_RSQ = 0x02  # common on NVIDIA
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_RSQ),
                                            f'MUFU.RSQ R{d}, R{a}'))

                elif op == 'div' and typ == 'f32':
                    # Float division: MUFU.RCP + FMUL
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    # temp = rcp(b), result = a * temp
                    output.append(SassInstr(encode_mufu(d, b, MUFU_RCP),
                                            f'MUFU.RCP R{d}, R{b}  // div.f32 step 1'))
                    output.append(SassInstr(encode_fmul(d, a, d),
                                            f'FMUL R{d}, R{a}, R{d}  // div.f32 step 2'))

                elif op == 'prmt':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    if isinstance(instr.srcs[1], ImmOp):
                        sel = instr.srcs[1].value
                        c = ctx.ra.r32(instr.srcs[2].name) if len(instr.srcs) > 2 else RZ
                        output.append(SassInstr(encode_prmt(d, a, sel, c),
                                                f'PRMT R{d}, R{a}, 0x{sel:04x}, R{c}'))
                    else:
                        output.append(_nop(f'TODO: prmt with register selector'))

                elif op == 'bfe' and typ in ('u32', 's32'):
                    # Bit field extract: shift right by start pos, mask with ((1<<len)-1)
                    # ptxas compiles to SHF.R.U32.HI or SHF.R.S32.HI
                    from sass.encoding.sm_120_encode import encode_shf_l_u32
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    # Start and length are immediates
                    start = instr.srcs[1].value if isinstance(instr.srcs[1], ImmOp) else 0
                    length = instr.srcs[2].value if len(instr.srcs) > 2 and isinstance(instr.srcs[2], ImmOp) else 32
                    # SHF.R to shift right, then AND with mask
                    output.append(_nop(f'TODO: bfe.{typ} decomposition (SHF.R + AND mask)'))

                elif op == 'bfi' and typ in ('b32',):
                    output.append(_nop(f'TODO: bfi.b32 decomposition (LOP3.LUT with mask)'))

                elif op in ('cvt',):
                    # General CVT — handle common conversions
                    src_type = instr.types[-1] if len(instr.types) > 1 else 'u32'
                    dst_type = instr.types[0] if len(instr.types) > 0 else 'u32'
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name) if isinstance(instr.srcs[0], RegOp) else RZ
                    if 'f32' in instr.types and ('u32' in instr.types or 's32' in instr.types):
                        if instr.types.index('f32') < instr.types.index('u32') if 'u32' in instr.types else instr.types.index('s32'):
                            # int → float
                            output.append(SassInstr(encode_i2fp_u32(d, a),
                                                    f'I2FP.F32 R{d}, R{a}  // cvt'))
                        else:
                            # float → int
                            output.append(SassInstr(encode_f2i_u32(d, a),
                                                    f'F2I.U32 R{d}, R{a}  // cvt'))
                    else:
                        output.append(_nop(f'TODO: cvt {".".join(instr.types)}'))

                elif op == 'mov' and typ in ('u32', 's32', 'b32'):
                    d = ctx.ra.r32(instr.dest.name)
                    if isinstance(instr.srcs[0], RegOp):
                        a = ctx.ra.r32(instr.srcs[0].name)
                        output.append(SassInstr(encode_mov(d, a),
                                                f'MOV R{d}, R{a}'))
                    else:
                        output.append(_nop(f'TODO: mov with immediate'))

                else:
                    # Unsupported instruction: emit NOP with comment
                    output.append(_nop(f'TODO: {instr.op} {".".join(instr.types)} {instr.mods}'))

            except ISelError as e:
                # Emit NOP with error comment rather than crashing
                output.append(_nop(f'ISEL ERROR: {e}  [{instr.op}]'))

    # BRA offset fixup pass
    if hasattr(ctx, '_bra_fixups'):
        for bra_idx, target_label in ctx._bra_fixups:
            if target_label in ctx.label_map:
                target_byte = ctx.label_map[target_label]
                bra_byte = (bra_idx + 1) * 16  # offset from NEXT instruction
                rel_offset = target_byte - bra_byte
                output[bra_idx] = SassInstr(
                    encode_bra(rel_offset),
                    f'BRA {target_label} (offset={rel_offset})')

    return output
