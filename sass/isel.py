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
    encode_imad_wide, encode_imad_wide_rr, encode_imad_wide_u32, encode_imad_wide_u32_carry, encode_imad_wide_u32x,
    encode_imad, encode_imad_rr, encode_imad_ur, encode_imad_hi, encode_imad_shl_u32,
    encode_s2ur,
    encode_ldg_e, encode_ldg_e_64,
    encode_stg_e, encode_stg_e_64,
    encode_lds, encode_sts,
    encode_ldcu_64, encode_ldcu_32,
    encode_iadd64_ur,
    encode_bar_sync,
    encode_isetp_ge_and, encode_isetp_ur,
    encode_isetp, ISETP_LT, ISETP_EQ, ISETP_LE, ISETP_GT, ISETP_NE, ISETP_GE,
    encode_fsetp, FSETP_LT, FSETP_EQ, FSETP_LE, FSETP_GT, FSETP_NE, FSETP_GE,
    encode_bra, patch_pred,
    encode_fadd, encode_fmul, encode_ffma,
    encode_mufu, MUFU_RCP, MUFU_SQRT, MUFU_SIN, MUFU_COS, MUFU_EX2, MUFU_LG2,
    encode_sel, encode_fsel,
    encode_vimnmx_s32, encode_vimnmx_u32,
    encode_fmnmx,
    encode_prmt, encode_prmt_reg,
    encode_popc, encode_brev, encode_flo, encode_iabs, encode_bfe_sext,
    encode_shfl, SHFL_IDX, SHFL_UP, SHFL_DOWN, SHFL_BFLY,
    encode_vote_ballot,
    encode_atomg_cas_b32,
    encode_dadd, encode_dmul, encode_dfma,
    encode_i2fp_u32, encode_f2i_u32,
    encode_f2f_f32_f64, encode_f2f_f64_f32,
    encode_f2i_s32_f64, encode_f2i_u32_f64, encode_i2f_f64_s32,
    encode_i2f_u32_rp, encode_i2f_s32_rp, encode_f2i_ftz_u32_trunc, encode_hfma2_zero,
    encode_iadd3_imm32, encode_iadd3_neg_b4, encode_iadd3_neg_b3,
    encode_iadd3_pred_neg_b4, encode_iadd3_pred_small_imm,
    encode_iadd3_pred_neg_b3, encode_lop3_pred,
    encode_lop3, LOP3_AND, LOP3_OR, LOP3_XOR,
    RZ, PT, SR_TID_X, SR_TID_Y,
    SR_CTAID_X,
)
from sass.encoding.sm_120_encode import (
    encode_shf_l_w_u32_hi,
    encode_shf_l_u32,
    encode_shf_l_u64_hi,
    encode_shf_r_u32, encode_shf_r_u32_hi,
    encode_shf_l_u32_var, encode_shf_r_u32_hi_var,
    encode_shf_r_s32_hi, encode_shf_r_s32_hi_var,
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

# SM_120 constant bank offsets for system values (driver-populated)
_CBANK_NTID_X = 0x360  # blockDim.x lives at c[0][0x360]


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

        # ntid.x: load from constant bank c[0][0x360] instead of S2R.
        # The driver populates this offset with blockDim.x. This avoids
        # the SR bus and gives us a constant-bank source that the mad.lo
        # handler can use with LDCU.32 + IMAD R-UR.
        if src.name == '%ntid.x':
            return [SassInstr(encode_ldc(d, 0, _CBANK_NTID_X),
                              f'LDC R{d}, c[0][0x{_CBANK_NTID_X:x}]  // ntid.x')]

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
            instrs.append(SassInstr(encode_iadd3(d_lo, s_lo, RZ, RZ),
                                    f'MOV R{d_lo}, R{s_lo}  // {dest.name}.lo = {src.name}.lo'))
        if d_hi != s_hi:
            instrs.append(SassInstr(encode_iadd3(d_hi, s_hi, RZ, RZ),
                                    f'MOV R{d_hi}, R{s_hi}  // {dest.name}.hi = {src.name}.hi'))
        return instrs or [_nop(f'MOV {dest.name} = {src.name} (same reg, elided)')]
    else:
        # 32-bit
        d = ra.r32(dest.name)
        s = ra.r32(src.name)
        return [SassInstr(encode_iadd3(d, s, RZ, RZ), f'MOV R{d}, R{s}  // {dest.name} = {src.name}')]


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
            SassInstr(encode_iadd3(d_lo, RZ, RZ, RZ),
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
            SassInstr(encode_iadd3(d_hi, RZ, RZ, RZ),
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
    d_lo = ra.lo(dest.name)  # use allocator's assignment

    return [
        SassInstr(encode_iadd64(d_lo, a_lo, b_lo, negate_src1=True),
                  f'IADD.64 R{d_lo}, R{a_lo}, -R{b_lo}  // sub.u64'),
    ]


def _select_add_u64(instr: Instruction, ra: RegAlloc,
                    ctx: 'ISelContext' = None) -> list[SassInstr]:
    """add.u64 dest, a, b → IADD.64 with UR source if one operand is in UR bank."""
    from sass.encoding.sm_120_opcodes import encode_iadd64_ur
    dest = instr.dest
    a    = instr.srcs[0]
    b    = instr.srcs[1]
    if not isinstance(dest, RegOp) or not isinstance(a, RegOp) or not isinstance(b, RegOp):
        raise ISelError(f"add.u64: all operands must be registers")

    # Check if either operand is a UR param (loaded via LDCU)
    a_in_ur = ctx and a.name in ctx._ur_params
    b_in_ur = ctx and b.name in ctx._ur_params

    if a_in_ur or b_in_ur:
        # Use IADD.64 R-UR variant: dest(R) = src_r(R) + src_ur(UR)
        if a_in_ur:
            ur_idx = ctx._ur_params[a.name]
            r_lo = ra.lo(b.name)
        else:
            ur_idx = ctx._ur_params[b.name]
            r_lo = ra.lo(a.name)
        d_lo = ra.lo(dest.name) if dest.name in ra.int_regs else r_lo
        return [
            SassInstr(encode_iadd64_ur(d_lo, r_lo, ur_idx),
                      f'IADD.64 R{d_lo}, R{r_lo}, UR{ur_idx}  // add.u64 (UR base)'),
        ]
    else:
        # Both operands in R bank: use IADD.64 single instruction
        a_lo = ra.lo(a.name)
        b_lo = ra.lo(b.name)
        d_lo = ra.lo(dest.name)  # use allocator's assignment
        return [
            SassInstr(encode_iadd64(d_lo, a_lo, b_lo),
                      f'IADD.64 R{d_lo}, R{a_lo}, R{b_lo}  // add.u64'),
        ]


def _select_ld_param(instr: Instruction, ra: RegAlloc,
                     param_offsets: dict[str, int],
                     ctx: 'ISelContext' = None) -> list[SassInstr]:
    """
    ld.param.u64 → LDCU.64 (into uniform register for descriptor-based addressing).
    ld.param.u32 → LDC (into general register).

    SM_120 descriptor-based memory model requires pointer params in UR bank.
    """
    dest = instr.dest
    src  = instr.srcs[0]
    if not isinstance(dest, RegOp):
        raise ISelError(f"ld.param dest must be register")

    from ptx.ir import MemOp
    if not isinstance(src, MemOp):
        raise ISelError(f"ld.param src must be MemOp, got {src!r}")

    param_name = src.base
    if isinstance(src.offset, int):
        byte_off = param_offsets.get(param_name, 0) + src.offset
    else:
        byte_off = param_offsets.get(param_name, 0)

    typ = instr.types[-1] if instr.types else 'u32'
    if typ in ('u64', 's64', 'b64'):
        # Load 64-bit param into UR via LDCU.64, materialize to GPR via IADD.64-UR.
        # Avoids LDC.64 single-slot scoreboard collision for 3+ pointer params.
        ur_idx = ctx._next_ur if ctx else 6
        if ctx:
            if ur_idx % 2 != 0:  # LDCU.64 requires even-aligned UR
                ur_idx += 1
                ctx._next_ur = ur_idx
            ctx._next_ur += 2
            ctx._ur_params[dest.name] = ur_idx
        # Just load into UR — address computation (IADD.64 Roffset + UR)
        # is done at the point of use (ld.global / st.global), matching ptxas.
        # Do NOT materialize into GPR here — that clobbers registers used
        # by other parameters.
        return [
            SassInstr(encode_ldcu_64(ur_idx, 0, byte_off),
                      f'LDCU.64 UR{ur_idx}, c[0][0x{byte_off:x}]  // {param_name}'),
        ]
    else:
        # u32 param: load directly into a GPR via LDC.
        # Using LDCU.32 would consume an LDCU counter slot before the descriptor,
        # forcing the descriptor to a higher counter than 2 (wdep≠0x35), which
        # breaks LDG's rbar=0x09 requirement on SM_120.
        if dest.name not in ra.int_regs:
            # Dead u32 parameter — skip
            return []
        d = ra.r32(dest.name)
        if ctx:
            ctx._reg_param_off[dest.name] = byte_off
        return [SassInstr(encode_ldc(d, 0, byte_off, ctrl=0x7f1),
                          f'LDC R{d}, c[0][0x{byte_off:x}]  // {param_name}')]


def _select_ld_global(instr: Instruction, ra: RegAlloc,
                      ur_desc: int, ctx: 'ISelContext' = None) -> list[SassInstr]:
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

    base_name = src.base if src.base.startswith('%') else f'%{src.base}'
    addr = ra.lo(src.base) if src.base in ra.int_regs else RZ

    if is_64:
        d = ra.lo(dest.name)
        return [SassInstr(encode_ldg_e_64(d, ur_desc, addr),
                          f'LDG.E.64 R{d}, desc[UR{ur_desc}][R{addr}.64]')]
    else:
        d = ra.r32(dest.name)
        return [SassInstr(encode_ldg_e(d, ur_desc, addr, width=32),
                          f'LDG.E R{d}, desc[UR{ur_desc}][R{addr}.64]')]


def _select_atom_cas(instr: Instruction, ra: RegAlloc,
                     ctx: 'ISelContext' = None) -> list[SassInstr]:
    """atom.cas.b32 → ATOMG.E.CAS.b32."""
    from ptx.ir import MemOp
    dest_op = instr.dest
    addr_op = instr.srcs[0]
    cmp_op  = instr.srcs[1]
    new_op  = instr.srcs[2]
    if not isinstance(addr_op, MemOp):
        raise ISelError("atom.cas addr must be MemOp")
    d   = ra.r32(dest_op.name)
    addr = ra.lo(addr_op.base) if addr_op.base in ra.int_regs else RZ
    cmp = ra.r32(cmp_op.name)
    nv  = ra.r32(new_op.name)
    return [SassInstr(encode_atomg_cas_b32(d, addr, cmp, nv),
                      f'ATOMG.E.CAS.b32 R{d}, [R{addr}], R{cmp}, R{nv}')]


def _select_st_global(instr: Instruction, ra: RegAlloc,
                      ur_desc: int, ctx: 'ISelContext' = None) -> list[SassInstr]:
    """st.global → STG.E with appropriate width."""
    dest_op = instr.srcs[0]  # address
    src_op  = instr.srcs[1]  # data
    from ptx.ir import MemOp
    if not isinstance(dest_op, MemOp):
        raise ISelError(f"st.global addr must be MemOp")
    if not isinstance(src_op, RegOp):
        # Immediate data: materialize into a temporary register first.
        if isinstance(src_op, ImmOp):
            t = ctx._next_gpr; ctx._next_gpr += 1
            lit_off = ctx._alloc_literal(src_op.value & 0xFFFFFFFF)
            from sass.encoding.sm_120_opcodes import encode_ldc
            preamble = [SassInstr(encode_ldc(t, 4, lit_off),
                                  f'LDC R{t}, c[0x4][{lit_off:#x}]  // materialize imm for st')]
        else:
            raise ISelError(f"st.global data must be register or immediate")

    typ = instr.types[-1] if instr.types else 'u32'
    is_64 = typ in ('u64', 's64', 'b64', 'f64')

    base_name = dest_op.base if dest_op.base.startswith('%') else f'%{dest_op.base}'
    addr = ra.lo(dest_op.base) if dest_op.base in ra.int_regs else RZ

    # Handle materialized immediate
    if not isinstance(src_op, RegOp):
        data = t  # from materialized temp above
        result = preamble + [SassInstr(encode_stg_e(ur_desc, addr, data, width=32, ctrl=0xff1),
                                       f'STG.E desc[UR{ur_desc}][R{addr}.64], R{data}')]
        return result

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
    # Next available uniform register for LDCU param loading (UR6+)
    _next_ur:      int = 6  # UR4 = mem desc, UR6+ for params
    # Map PTX register name → UR index (for params loaded via LDCU)
    _ur_params:    dict[str, int] = field(default_factory=dict)
    # Map PTX register name → param byte offset (for setp LDCU fallback)
    _reg_param_off: dict[str, int] = field(default_factory=dict)
    # Map PTX register name → SR code (for S2UR in mad.lo)
    _reg_sr_source: dict[str, int] = field(default_factory=dict)
    # Map PTX register name → UR index (u32 params loaded via LDCU.32 for ISETP R-UR)
    _ur_for_param:  dict[str, int] = field(default_factory=dict)
    # Literal constant pool: value → c[0] byte offset (baked into .nv.constant0)
    # Base offset is set by the pipeline after regalloc (after the param area ends).
    _const_pool_base: int = 0
    _const_pool:      dict[int, int] = field(default_factory=dict)
    # Next available scratch GPR (for isel-internal temporaries, e.g. bfe mask)
    # Initialized from alloc.num_gprs by the pipeline; may grow during isel.
    _next_gpr: int = 0
    # Next available scratch predicate register (for isel-internal use, e.g. div.u32)
    # Initialized from alloc.num_pred by the pipeline; may grow during isel.
    _next_pred: int = 0

    def _alloc_literal(self, value: int) -> int:
        """Return the c[0] byte offset for a 32-bit literal constant.

        Allocates a new slot in the literal pool if the value has not been
        seen before.  Slots are 4 bytes each.
        """
        value = value & 0xFFFFFFFF  # normalise to u32 bit pattern
        if value not in self._const_pool:
            offset = self._const_pool_base + len(self._const_pool) * 4
            self._const_pool[value] = offset
        return self._const_pool[value]


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
            # typ = last type qualifier (the data type). Earlier elements are modifiers (lo, hi, approx, etc.)
            typ = instr.types[-1].lower() if instr.types else ''

            # Track output length before this instruction so we can apply
            # predicates to all newly-generated SASS after the handler runs.
            _pre_len = len(output)

            try:
                if op == 'mov' and typ in ('u32', 's32', 'b32', 'u64', 's64', 'b64'):
                    # Immediate source: load from literal pool in constant bank
                    if isinstance(instr.srcs[0], ImmOp) and typ in ('u32', 's32', 'b32'):
                        d = ctx.ra.r32(instr.dest.name)
                        imm = instr.srcs[0].value & 0xFFFFFFFF
                        lit_off = ctx._alloc_literal(imm)
                        output.append(SassInstr(encode_ldc(d, 0, lit_off),
                                                f'LDC R{d}, c[0][0x{lit_off:x}]  // mov imm={imm:#x}'))
                        continue
                    # Track special register sources
                    if (isinstance(instr.srcs[0], RegOp) and
                        instr.srcs[0].name in _SPECIAL_REGS and
                        isinstance(instr.dest, RegOp)):
                        ctx._reg_sr_source[instr.dest.name] = _SPECIAL_REGS[instr.srcs[0].name]
                        # ntid.x loaded from constant bank — track as param-like source
                        # so mad.lo can use LDCU.32 + IMAD R-UR
                        if instr.srcs[0].name == '%ntid.x':
                            ctx._reg_param_off[instr.dest.name] = _CBANK_NTID_X
                        elif instr.srcs[0].name == '%ctaid.x':
                            # Put ctaid into a fresh UR via S2UR so mad.lo can use IMAD R-UR.
                            # IMAD R-R (0x224) is not validated on SM_120; IMAD R-UR (0xc24)
                            # is confirmed by ptxas. Must NOT use UR4 (reserved for mem
                            # descriptor by pipeline.py) to avoid a WAR hazard where the
                            # descriptor LDCU overwrites UR4 before IMAD finishes reading it.
                            ur_ctaid = ctx._next_ur; ctx._next_ur += 1
                            ctx._ur_for_param[instr.dest.name] = ur_ctaid
                            output.append(SassInstr(encode_s2ur(ur_ctaid, SR_CTAID_X),
                                                    f'S2UR UR{ur_ctaid}, SR_CTAID_X  // {instr.dest.name} = ctaid.x'))
                            continue
                    output.extend(_select_mov(instr, ctx.ra))

                elif op == 'shl' and typ in ('b64', 'u64', 's64'):
                    output.extend(_select_shl_b64(instr, ctx.ra))

                elif op == 'shl' and typ in ('b32', 'u32', 's32'):
                    # 32-bit shift left: IMAD.SHL or SHF.L.U32 for constants,
                    # SHF.L.U32.VAR (opcode 0x7299) for runtime register shifts.
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    if isinstance(instr.srcs[1], ImmOp):
                        k = instr.srcs[1].value
                        if k <= 15:
                            output.append(SassInstr(encode_imad_shl_u32(d, a, k),
                                                    f'IMAD.SHL.U32 R{d}, R{a}, {1<<k:#x}, RZ  // shl.{typ} {k}'))
                        else:
                            output.append(SassInstr(encode_shf_l_u32(d, a, k, RZ),
                                                    f'SHF.L.U32 R{d}, R{a}, 0x{k:x}, RZ  // shl.{typ} {k}'))
                    else:
                        k_reg = ctx.ra.r32(instr.srcs[1].name)
                        output.append(SassInstr(encode_shf_l_u32_var(d, a, k_reg),
                                                f'SHF.L.U32 R{d}, R{a}, R{k_reg}, RZ  // shl.{typ} (var)'))

                elif op == 'shr' and typ in ('b32', 'u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    is_signed = (typ == 's32')
                    if isinstance(instr.srcs[1], ImmOp):
                        k = instr.srcs[1].value
                        if is_signed:
                            output.append(SassInstr(encode_shf_r_s32_hi(d, a, k),
                                                    f'SHF.R.S32.HI R{d}, RZ, 0x{k:x}, R{a}  // shr.s32 {k}'))
                        else:
                            output.append(SassInstr(encode_shf_r_u32_hi(d, a, k),
                                                    f'SHF.R.U32.HI R{d}, RZ, 0x{k:x}, R{a}  // shr.{typ} {k}'))
                    else:
                        k_reg = ctx.ra.r32(instr.srcs[1].name)
                        if is_signed:
                            output.append(SassInstr(encode_shf_r_s32_hi_var(d, a, k_reg),
                                                    f'SHF.R.S32.HI R{d}, RZ, R{k_reg}, R{a}  // shr.s32 (var)'))
                        else:
                            output.append(SassInstr(encode_shf_r_u32_hi_var(d, a, k_reg),
                                                    f'SHF.R.U32.HI R{d}, RZ, R{k_reg}, R{a}  // shr.{typ} (var)'))

                elif op == 'shr' and typ in ('u64',):
                    output.extend(_select_shr_u64(instr, ctx.ra))

                elif op == 'shr' and typ in ('s64',):
                    # shr.s64: arithmetic 64-bit right shift (sign-extends).
                    # K < 32: lo = SHF.R.U64(s_lo, k, s_hi) [pull in hi bits]
                    #          hi = SHF.R.S32.HI(s_hi, k)   [arithmetic shift of hi]
                    # K >= 32: lo = SHF.R.S32.HI(s_hi, k-32) [lo gets shifted hi]
                    #           hi = SHF.R.S32.HI(s_hi, 31)  [hi = all sign bits]
                    d_lo = ctx.ra.lo(instr.dest.name); d_hi = d_lo + 1
                    s_lo = ctx.ra.lo(instr.srcs[0].name); s_hi = s_lo + 1
                    k = _get_imm(instr.srcs[1])
                    if k < 32:
                        output.append(SassInstr(encode_shf_r_u32(d_lo, s_lo, k, s_hi),
                            f'SHF.R.U64 R{d_lo}, R{s_lo}, 0x{k:x}, R{s_hi}  // shr.s64 lo'))
                        output.append(SassInstr(encode_shf_r_s32_hi(d_hi, s_hi, k),
                            f'SHF.R.S32.HI R{d_hi}, RZ, 0x{k:x}, R{s_hi}  // shr.s64 hi'))
                    else:
                        k32 = k - 32
                        output.append(SassInstr(encode_shf_r_s32_hi(d_lo, s_hi, k32),
                            f'SHF.R.S32.HI R{d_lo}, RZ, 0x{k32:x}, R{s_hi}  // shr.s64 lo (K>={k})'))
                        output.append(SassInstr(encode_shf_r_s32_hi(d_hi, s_hi, 31),
                            f'SHF.R.S32.HI R{d_hi}, RZ, 0x1f, R{s_hi}  // shr.s64 hi=sign'))

                elif op == 'sub' and typ in ('u64', 's64'):
                    output.extend(_select_sub_u64(instr, ctx.ra))

                elif op == 'add' and typ in ('u64', 's64'):
                    output.extend(_select_add_u64(instr, ctx.ra, ctx))

                elif op == 'add' and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    if isinstance(instr.srcs[1], ImmOp):
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        lit_off = ctx._alloc_literal(imm)
                        output.append(SassInstr(encode_ldc(d, 0, lit_off),
                                                f'LDC R{d}, c[0][0x{lit_off:x}]  // add imm={imm:#x}'))
                        output.append(SassInstr(encode_iadd3(d, a, d, RZ),
                                                f'IADD3 R{d}, R{a}, R{d}, RZ  // add.{typ} imm'))
                    else:
                        b = ctx.ra.r32(instr.srcs[1].name)
                        output.append(SassInstr(encode_iadd3(d, a, b, RZ),
                                                f'IADD3 R{d}, R{a}, R{b}, RZ  // add.{typ}'))

                elif op == 'sub' and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    if isinstance(instr.srcs[1], ImmOp):
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        lit_off = ctx._alloc_literal(imm)
                        output.append(SassInstr(encode_ldc(d, 0, lit_off),
                                                f'LDC R{d}, c[0][0x{lit_off:x}]  // sub imm={imm:#x}'))
                        output.append(SassInstr(encode_iadd3(d, a, d, RZ, negate_src1=True),
                                                f'IADD3 R{d}, R{a}, -R{d}, RZ  // sub.{typ} imm'))
                    else:
                        b = ctx.ra.r32(instr.srcs[1].name)
                        output.append(SassInstr(encode_iadd3(d, a, b, RZ, negate_src1=True),
                                                f'IADD3 R{d}, R{a}, -R{b}, RZ  // sub.{typ}'))

                elif op in ('and', 'or', 'xor') and typ in ('b32', 'u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    lut = {'and': LOP3_AND, 'or': LOP3_OR, 'xor': LOP3_XOR}[op]
                    if isinstance(instr.srcs[1], ImmOp):
                        # Immediate src1: load from literal pool into dest, then LOP3.LUT.
                        # LOP3.LUT reads old dest (= mask) and src as inputs → result correct.
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        lit_off = ctx._alloc_literal(imm)
                        output.append(SassInstr(encode_ldc(d, 0, lit_off),
                                                f'LDC R{d}, c[0][0x{lit_off:x}]  // {op} imm={imm:#x}'))
                        output.append(SassInstr(encode_lop3(d, a, d, RZ, lut),
                                                f'LOP3.LUT R{d}, R{a}, R{d}, RZ, 0x{lut:02x}  // {op}.{typ} imm'))
                    else:
                        b = ctx.ra.r32(instr.srcs[1].name)
                        output.append(SassInstr(encode_lop3(d, a, b, RZ, lut),
                                                f'LOP3.LUT R{d}, R{a}, R{b}, RZ, 0x{lut:02x}  // {op}.{typ}'))

                elif op in ('and', 'or', 'xor') and typ in ('b64', 'u64', 's64'):
                    # 64-bit logic: apply LOP3 to lo and hi words separately.
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = ctx.ra.lo(instr.srcs[0].name)
                    lut = {'and': LOP3_AND, 'or': LOP3_OR, 'xor': LOP3_XOR}[op]
                    if isinstance(instr.srcs[1], ImmOp):
                        imm = instr.srcs[1].value & 0xFFFF_FFFF_FFFF_FFFF
                        imm_lo = imm & 0xFFFFFFFF
                        imm_hi = (imm >> 32) & 0xFFFFFFFF
                        t = ctx._next_gpr; ctx._next_gpr += 1
                        lit_lo = ctx._alloc_literal(imm_lo)
                        output.append(SassInstr(encode_ldc(t, 0, lit_lo),
                                                f'LDC R{t}, c[0][0x{lit_lo:x}]  // {op}.b64 imm_lo'))
                        output.append(SassInstr(encode_lop3(d_lo, a_lo, t, RZ, lut),
                                                f'LOP3.LUT R{d_lo}, R{a_lo}, R{t}, RZ, 0x{lut:02x}  // {op}.b64 lo'))
                        lit_hi = ctx._alloc_literal(imm_hi)
                        output.append(SassInstr(encode_ldc(t, 0, lit_hi),
                                                f'LDC R{t}, c[0][0x{lit_hi:x}]  // {op}.b64 imm_hi'))
                        output.append(SassInstr(encode_lop3(d_lo+1, a_lo+1, t, RZ, lut),
                                                f'LOP3.LUT R{d_lo+1}, R{a_lo+1}, R{t}, RZ, 0x{lut:02x}  // {op}.b64 hi'))
                    else:
                        b_lo = ctx.ra.lo(instr.srcs[1].name)
                        output.append(SassInstr(encode_lop3(d_lo, a_lo, b_lo, RZ, lut),
                                                f'LOP3.LUT R{d_lo}, R{a_lo}, R{b_lo}, RZ, 0x{lut:02x}  // {op}.b64 lo'))
                        output.append(SassInstr(encode_lop3(d_lo+1, a_lo+1, b_lo+1, RZ, lut),
                                                f'LOP3.LUT R{d_lo+1}, R{a_lo+1}, R{b_lo+1}, RZ, 0x{lut:02x}  // {op}.b64 hi'))

                elif op == 'not' and typ in ('b32', 'u32', 's32'):
                    # not.b32 d, a  →  LOP3.LUT d, a, RZ, RZ, 0x0F  (~a)
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_lop3(d, a, RZ, RZ, 0x0F),
                                            f'LOP3.LUT R{d}, R{a}, RZ, RZ, 0x0f  // not.{typ}'))

                elif op == 'not' and typ in ('b64', 'u64', 's64'):
                    # not.b64 d, a  →  two LOP3.LUT on lo and hi words
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = ctx.ra.lo(instr.srcs[0].name)
                    output.append(SassInstr(encode_lop3(d_lo, a_lo, RZ, RZ, 0x0F),
                                            f'LOP3.LUT R{d_lo}, R{a_lo}, RZ, RZ, 0x0f  // not.{typ} lo'))
                    output.append(SassInstr(encode_lop3(d_lo+1, a_lo+1, RZ, RZ, 0x0F),
                                            f'LOP3.LUT R{d_lo+1}, R{a_lo+1}, RZ, RZ, 0x0f  // not.{typ} hi'))

                elif op == 'mul' and 'lo' in instr.types and typ in ('u32', 's32'):
                    # mul.lo.s32 → IMAD R-UR or IMAD.WIDE with immediate
                    # NOTE: IMAD R-R (0x224) is NOT valid on SM_120!
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    if isinstance(instr.srcs[1], ImmOp):
                        # Immediate multiplier: use IMAD.SHL.U32 if power-of-2 and ≤15,
                        # else load into UR via literal pool and use IMAD R-UR.
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        if imm > 0 and (imm & (imm - 1)) == 0:
                            # Power of two: IMAD.SHL.U32 dest, src, imm, RZ
                            shift = imm.bit_length() - 1
                            if shift <= 15:
                                output.append(SassInstr(encode_imad_shl_u32(d, a, shift),
                                    f'IMAD.SHL.U32 R{d}, R{a}, 0x{imm:x}, RZ  // mul.lo imm={imm}'))
                            else:
                                lit_off = ctx._alloc_literal(imm)
                                ur_tmp = ctx._next_ur; ctx._next_ur += 1
                                output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, lit_off),
                                    f'LDCU.32 UR{ur_tmp}, c[0][0x{lit_off:x}]  // mul.lo imm'))
                                output.append(SassInstr(encode_imad_ur(d, a, ur_tmp, RZ),
                                    f'IMAD R{d}, R{a}, UR{ur_tmp}, RZ  // mul.lo imm'))
                        else:
                            lit_off = ctx._alloc_literal(imm)
                            ur_tmp = ctx._next_ur; ctx._next_ur += 1
                            output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, lit_off),
                                f'LDCU.32 UR{ur_tmp}, c[0][0x{lit_off:x}]  // mul.lo imm'))
                            output.append(SassInstr(encode_imad_ur(d, a, ur_tmp, RZ),
                                f'IMAD R{d}, R{a}, UR{ur_tmp}, RZ  // mul.lo imm'))
                        continue
                    b = ctx.ra.r32(instr.srcs[1].name)
                    # Check if either source lives in a UR (ctaid.x via S2UR)
                    a_ur = ctx._ur_for_param.get(
                        instr.srcs[0].name if isinstance(instr.srcs[0], RegOp) else None)
                    b_ur = ctx._ur_for_param.get(
                        instr.srcs[1].name if isinstance(instr.srcs[1], RegOp) else None)
                    if a_ur is not None:
                        # src0 is in UR (e.g., ctaid.x) — use IMAD R{b}, UR{a_ur}, RZ
                        output.append(SassInstr(encode_imad_ur(d, b, a_ur, RZ),
                            f'IMAD R{d}, R{b}, UR{a_ur}, RZ  // mul.lo.{typ} (src0 in UR)'))
                        continue
                    if b_ur is not None:
                        # src1 is in UR — use IMAD R{a}, UR{b_ur}, RZ
                        output.append(SassInstr(encode_imad_ur(d, a, b_ur, RZ),
                            f'IMAD R{d}, R{a}, UR{b_ur}, RZ  // mul.lo.{typ} (src1 in UR)'))
                        continue
                    # Check if either source is a param → use IMAD R-UR
                    b_param = ctx._reg_param_off.get(
                        instr.srcs[1].name if isinstance(instr.srcs[1], RegOp) else None)
                    a_param = ctx._reg_param_off.get(
                        instr.srcs[0].name if isinstance(instr.srcs[0], RegOp) else None)
                    if b_param is not None:
                        ur_tmp = ctx._next_ur; ctx._next_ur += 1
                        output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, b_param),
                            f'LDCU.32 UR{ur_tmp}, c[0][0x{b_param:x}]'))
                        output.append(SassInstr(encode_imad_ur(d, a, ur_tmp, RZ),
                            f'IMAD R{d}, R{a}, UR{ur_tmp}, RZ  // mul.lo.{typ}'))
                    elif a_param is not None:
                        ur_tmp = ctx._next_ur; ctx._next_ur += 1
                        output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, a_param),
                            f'LDCU.32 UR{ur_tmp}, c[0][0x{a_param:x}]'))
                        output.append(SassInstr(encode_imad_ur(d, b, ur_tmp, RZ),
                            f'IMAD R{d}, R{b}, UR{ur_tmp}, RZ  // mul.lo.{typ}'))
                    else:
                        # Both sources are computed GPRs — use R-R IMAD (opcode 0x2a4,
                        # validated against ptxas 13.0 on SM_120).
                        output.append(SassInstr(encode_imad_rr(d, a, b, RZ),
                            f'IMAD R{d}, R{a}, R{b}, RZ  // mul.lo.{typ} R-R'))

                elif op == 'mul' and 'lo' in instr.types and typ in ('u64', 's64', 'b64'):
                    # mul.lo.u64 d, a, b = lower 64 bits of a * b
                    # Decomposed into three IMAD operations:
                    #   IMAD.WIDE d_lo, a_lo, b_lo, RZ  → d_lo:d_hi = a_lo × b_lo
                    #   IMAD.RR   d_hi, a_lo, b_hi, d_hi → d_hi += a_lo × b_hi (lo bits only)
                    #   IMAD.RR   d_hi, a_hi, b_lo, d_hi → d_hi += a_hi × b_lo
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = ctx.ra.lo(instr.srcs[0].name)
                    b_lo = ctx.ra.lo(instr.srcs[1].name)
                    output.append(SassInstr(encode_imad_wide_rr(d_lo, a_lo, b_lo, RZ),
                        f'IMAD.WIDE R{d_lo}, R{a_lo}, R{b_lo}, RZ  // mul.lo.{typ} wide'))
                    output.append(SassInstr(encode_imad_rr(d_lo+1, a_lo, b_lo+1, d_lo+1),
                        f'IMAD R{d_lo+1}, R{a_lo}, R{b_lo+1}, R{d_lo+1}  // mul.lo.{typ} cross a_lo*b_hi'))
                    output.append(SassInstr(encode_imad_rr(d_lo+1, a_lo+1, b_lo, d_lo+1),
                        f'IMAD R{d_lo+1}, R{a_lo+1}, R{b_lo}, R{d_lo+1}  // mul.lo.{typ} cross a_hi*b_lo'))

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

                elif op == 'add' and typ == 'f64':
                    d = ctx.ra.lo(instr.dest.name)
                    a = ctx.ra.lo(instr.srcs[0].name)
                    b = ctx.ra.lo(instr.srcs[1].name)
                    output.append(SassInstr(encode_dadd(d, a, b),
                                            f'DADD R{d}, R{a}, R{b}  // add.f64'))

                elif op == 'mul' and typ == 'f64':
                    d = ctx.ra.lo(instr.dest.name)
                    a = ctx.ra.lo(instr.srcs[0].name)
                    b = ctx.ra.lo(instr.srcs[1].name)
                    output.append(SassInstr(encode_dmul(d, a, b),
                                            f'DMUL R{d}, R{a}, R{b}  // mul.f64'))

                elif op == 'fma' and typ == 'f64':
                    d = ctx.ra.lo(instr.dest.name)
                    a = ctx.ra.lo(instr.srcs[0].name)
                    b = ctx.ra.lo(instr.srcs[1].name)
                    c = ctx.ra.lo(instr.srcs[2].name)
                    output.append(SassInstr(encode_dfma(d, a, b, c),
                                            f'DFMA R{d}, R{a}, R{b}, R{c}  // fma.f64'))

                elif op == 'ld' and 'param' in instr.types:
                    output.extend(_select_ld_param(instr, ctx.ra, ctx.param_offsets, ctx))

                elif op == 'ld' and 'global' in instr.types:
                    output.extend(_select_ld_global(instr, ctx.ra, ctx.ur_desc, ctx))

                elif op == 'st' and 'global' in instr.types:
                    output.extend(_select_st_global(instr, ctx.ra, ctx.ur_desc, ctx))

                elif op == 'atom' and 'cas' in instr.types and 'b32' in instr.types:
                    output.extend(_select_atom_cas(instr, ctx.ra, ctx))

                elif op == 'ret':
                    output.append(SassInstr(encode_exit(ctrl=0x7f5), 'EXIT'))

                elif op == 'bra':
                    from ptx.ir import LabelOp
                    target = None
                    if instr.srcs:
                        if isinstance(instr.srcs[0], LabelOp):
                            target = instr.srcs[0].name

                    # Optimization: if @Px bra TARGET and TARGET is a ret-only block,
                    # emit @Px EXIT instead of @Px BRA TARGET. On SM_120 (Blackwell),
                    # this is what ptxas does — predicated EXIT correctly exits
                    # idle threads without requiring reconvergence management.
                    if instr.pred and target:
                        target_is_ret = False
                        for tbb in fn.blocks:
                            if tbb.label == target:
                                if (len(tbb.instructions) == 1
                                        and tbb.instructions[0].op == 'ret'):
                                    target_is_ret = True
                                break
                        if target_is_ret:
                            pd = ctx.ra.pred(instr.pred) if instr.pred in ctx.ra.pred_regs else 0
                            neg = instr.neg
                            if hasattr(ctx, '_negated_preds') and pd in ctx._negated_preds:
                                neg = not neg
                            exit_raw = patch_pred(encode_exit(), pred=pd, neg=neg)
                            pred_str = f'@{"!" if neg else ""}P{pd} '
                            output.append(SassInstr(exit_raw,
                                                    f'{pred_str}EXIT  // early exit (idle threads)'))
                            continue

                    # Unconditional BRA to ret-only block → EXIT
                    if not instr.pred and target:
                        _tgt_is_ret = False
                        for tbb in fn.blocks:
                            if tbb.label == target:
                                if (len(tbb.instructions) == 1
                                        and tbb.instructions[0].op == 'ret'):
                                    _tgt_is_ret = True
                                break
                        if _tgt_is_ret:
                            output.append(SassInstr(encode_exit(),
                                                    f'EXIT  // unconditional return'))
                            continue

                    # General BRA with offset fixup
                    bra_idx = len(output)
                    bra_raw = encode_bra(0)
                    if instr.pred:
                        pd = ctx.ra.pred(instr.pred) if instr.pred in ctx.ra.pred_regs else 0
                        # Check if the predicate was negated by the setp handler
                        # (e.g., setp.lt emits GE + negate)
                        neg = instr.neg
                        if hasattr(ctx, '_negated_preds') and pd in ctx._negated_preds:
                            neg = not neg  # flip the negation
                        bra_raw = patch_pred(bra_raw, pred=pd, neg=neg)
                        pred_str = f'@{"!" if neg else ""}P{pd} '
                    else:
                        pred_str = ''
                    output.append(SassInstr(bra_raw,
                                            f'{pred_str}BRA {target or "?"}'))
                    if target:
                        if not hasattr(ctx, '_bra_fixups'):
                            ctx._bra_fixups = []
                        ctx._bra_fixups.append((bra_idx, target))

                elif op == 'nop':
                    output.append(_nop())

                elif op == 'cvt':
                    # Type conversion — handle widening to 64-bit
                    # CSE: if same source was already converted, reuse the result
                    d = instr.dest
                    s = instr.srcs[0]
                    if isinstance(d, RegOp) and isinstance(s, RegOp):
                        _types_set = set(instr.types)
                        _is_64_dst = any(t in ('u64','s64','b64') for t in instr.types[:1])
                        # Only zero-extend from unsigned 32-bit; signed widening needs
                        # SHF.R.S32.HI (no encoder yet) for correct sign extension.
                        _is_32_src = any(t in ('u32','b32') for t in instr.types[1:])
                        if _is_64_dst and _is_32_src:
                            s_r = ctx.ra.r32(s.name)
                            # CSE: check if we already converted this source register
                            if not hasattr(ctx, '_cvt_cache'):
                                ctx._cvt_cache = {}
                            if s.name in ctx._cvt_cache:
                                # Reuse previous conversion result
                                prev_lo = ctx._cvt_cache[s.name]
                                ctx.ra.int_regs[d.name] = prev_lo
                                output.append(_nop(f'cvt.64.32 {d.name}={s.name} (CSE reuse R{prev_lo})'))
                                continue
                            d_lo = ctx.ra.lo(d.name)
                            ctx._cvt_cache[s.name] = d_lo
                            output.append(SassInstr(encode_iadd3(d_lo, s_r, RZ, RZ),
                                                    f'MOV R{d_lo}, R{s_r}  // cvt.64.32 lo'))
                            output.append(SassInstr(encode_iadd3(d_lo+1, RZ, RZ, RZ),
                                                    f'MOV R{d_lo+1}, RZ  // cvt.64.32 hi=0'))
                        elif _is_64_dst and any(t == 's32' for t in instr.types[1:]):
                            # Sign-extend s32 → s64/u64/b64
                            # SHF.R.U32.HI d_hi, RZ, 31, s_r → d_hi = 0 or 1
                            # INEG d_hi, d_hi               → d_hi = 0 or 0xFFFFFFFF
                            # MOV  d_lo, s_r                → lo word
                            s_r = ctx.ra.r32(s.name)
                            d_lo = ctx.ra.lo(d.name)
                            d_hi = d_lo + 1
                            # Use SHF.R.S32.HI directly (signed shift-right fills
                            # with sign bit, producing 0x00000000 or 0xFFFFFFFF)
                            output.append(SassInstr(
                                encode_shf_r_s32_hi(d_hi, s_r, 31),
                                f'SHF.R.S32.HI R{d_hi}, RZ, 0x1f, R{s_r}  // cvt.s64.s32 sign'))
                            if d_lo != s_r:
                                output.append(SassInstr(
                                    encode_iadd3(d_lo, s_r, RZ, RZ),
                                    f'MOV R{d_lo}, R{s_r}  // cvt.s64.s32 lo'))
                        else:
                            # General 32-bit and float conversions
                            _ROUNDING = {'rn','rz','rm','rp','rni','rzi','rmi','rpi'}
                            _core = [t for t in instr.types if t not in _ROUNDING]
                            _dst_t = _core[0] if _core else 'u32'
                            _src_t = _core[1] if len(_core) > 1 else 'u32'
                            _32B = {'u32', 's32', 'b32', 'f32'}
                            _64B = {'u64', 's64', 'b64', 'f64'}
                            if _dst_t == 'f32' and _src_t == 'f64':
                                # cvt.rn.f32.f64: double-precision → single-precision
                                d_r  = ctx.ra.r32(d.name)
                                a_lo = ctx.ra.lo(s.name)
                                output.append(SassInstr(encode_f2f_f32_f64(d_r, a_lo),
                                                        f'F2F.F32.F64 R{d_r}, R{a_lo}'))
                            elif _dst_t == 'f64' and _src_t == 'f32':
                                # cvt.f64.f32: single-precision → double-precision
                                d_lo = ctx.ra.lo(d.name)
                                a_r  = ctx.ra.r32(s.name)
                                output.append(SassInstr(encode_f2f_f64_f32(d_lo, a_r),
                                                        f'F2F.F64.F32 R{d_lo}, R{a_r}'))
                            elif _dst_t == 's32' and _src_t == 'f64':
                                # cvt.rzi.s32.f64: double → signed int32
                                d_r  = ctx.ra.r32(d.name)
                                a_lo = ctx.ra.lo(s.name)
                                output.append(SassInstr(encode_f2i_s32_f64(d_r, a_lo),
                                                        f'F2I.S32.F64 R{d_r}, R{a_lo}'))
                            elif _dst_t == 'u32' and _src_t == 'f64':
                                # cvt.rzi.u32.f64: double → unsigned int32
                                d_r  = ctx.ra.r32(d.name)
                                a_lo = ctx.ra.lo(s.name)
                                output.append(SassInstr(encode_f2i_u32_f64(d_r, a_lo),
                                                        f'F2I.U32.F64 R{d_r}, R{a_lo}'))
                            elif _dst_t == 'f64' and _src_t == 's32':
                                # cvt.rn.f64.s32: signed int32 → double
                                d_lo = ctx.ra.lo(d.name)
                                a_r  = ctx.ra.r32(s.name)
                                output.append(SassInstr(encode_i2f_f64_s32(d_lo, a_r),
                                                        f'I2F.F64.S32 R{d_lo}, R{a_r}'))
                            elif 'f32' in _types_set and ('u32' in _types_set or 's32' in _types_set):
                                d_r = ctx.ra.r32(d.name)
                                a_r = ctx.ra.r32(s.name)
                                _fi = instr.types.index('f32')
                                _ii = (instr.types.index('u32') if 'u32' in instr.types
                                       else instr.types.index('s32'))
                                if _fi < _ii:
                                    output.append(SassInstr(encode_i2fp_u32(d_r, a_r),
                                                            f'I2FP.F32 R{d_r}, R{a_r}  // cvt.f32.{_src_t}'))
                                else:
                                    output.append(SassInstr(encode_f2i_u32(d_r, a_r),
                                                            f'F2I.U32 R{d_r}, R{a_r}  // cvt.{_dst_t}.f32'))
                            elif _dst_t in ('u8', 's8', 'b8') and _src_t in _32B:
                                # Truncate to 8 bits: AND with 0xFF
                                d_r = ctx.ra.r32(d.name)
                                a_r = ctx.ra.r32(s.name)
                                lit_off = ctx._alloc_literal(0xFF)
                                t = ctx._next_gpr; ctx._next_gpr += 1
                                output.append(SassInstr(encode_ldc(t, 0, lit_off),
                                                        f'LDC R{t}, c[0][0x{lit_off:x}]  // 0xFF mask'))
                                output.append(SassInstr(encode_lop3(d_r, a_r, t, RZ, LOP3_AND),
                                                        f'LOP3.AND R{d_r}, R{a_r}, R{t}, RZ  // cvt.{_dst_t}.{_src_t}'))
                            elif _dst_t in ('u16', 's16', 'b16') and _src_t in _32B:
                                # Truncate to 16 bits: AND with 0xFFFF
                                d_r = ctx.ra.r32(d.name)
                                a_r = ctx.ra.r32(s.name)
                                lit_off = ctx._alloc_literal(0xFFFF)
                                t = ctx._next_gpr; ctx._next_gpr += 1
                                output.append(SassInstr(encode_ldc(t, 0, lit_off),
                                                        f'LDC R{t}, c[0][0x{lit_off:x}]  // 0xFFFF mask'))
                                output.append(SassInstr(encode_lop3(d_r, a_r, t, RZ, LOP3_AND),
                                                        f'LOP3.AND R{d_r}, R{a_r}, R{t}, RZ  // cvt.{_dst_t}.{_src_t}'))
                            elif _dst_t in _32B and _src_t in ('u8', 's8', 'b8', 'u16', 's16', 'b16'):
                                # Widening from narrow: just copy (narrow stored as u32, already zero-extended)
                                d_r = ctx.ra.r32(d.name)
                                a_r = ctx.ra.r32(s.name)
                                if d_r != a_r:
                                    output.append(SassInstr(encode_iadd3(d_r, a_r, RZ, RZ),
                                                            f'MOV R{d_r}, R{a_r}  // cvt.{_dst_t}.{_src_t}'))
                            elif _dst_t in _32B and _src_t in _32B:
                                d_r = ctx.ra.r32(d.name)
                                a_r = ctx.ra.r32(s.name)
                                if d_r != a_r:
                                    output.append(SassInstr(encode_iadd3(d_r, a_r, RZ, RZ),
                                                            f'MOV R{d_r}, R{a_r}  // cvt.{_dst_t}.{_src_t}'))
                                else:
                                    output.append(_nop(f'cvt.{_dst_t}.{_src_t} nop (d==a)'))
                            elif _dst_t in _32B and _src_t in _64B:
                                d_r = ctx.ra.r32(d.name)
                                a_lo = ctx.ra.lo(s.name)
                                if d_r != a_lo:
                                    output.append(SassInstr(encode_iadd3(d_r, a_lo, RZ, RZ),
                                                            f'MOV R{d_r}, R{a_lo}  // cvt.{_dst_t}.{_src_t} trunc'))
                                else:
                                    output.append(_nop(f'cvt.{_dst_t}.{_src_t} nop (d==a_lo)'))
                            elif _dst_t in _64B and _src_t in _64B:
                                # 64-bit reinterpret (u64↔s64, b64↔u64, etc.) — identity copy
                                d_lo = ctx.ra.lo(d.name)
                                a_lo = ctx.ra.lo(s.name)
                                if d_lo != a_lo:
                                    output.append(SassInstr(encode_iadd3(d_lo, a_lo, RZ, RZ),
                                                            f'MOV R{d_lo}, R{a_lo}  // cvt.{_dst_t}.{_src_t} lo'))
                                    output.append(SassInstr(encode_iadd3(d_lo+1, a_lo+1, RZ, RZ),
                                                            f'MOV R{d_lo+1}, R{a_lo+1}  // cvt.{_dst_t}.{_src_t} hi'))
                                # else: same register, nothing to do (NOP omitted)
                            elif _dst_t in _64B and _src_t in _32B:
                                # 32→64 widening: zero-extend (u64.u32/b64.b32) or sign-extend (s64.s32)
                                d_lo = ctx.ra.lo(d.name)
                                a_r  = ctx.ra.r32(s.name)
                                # lo = src
                                if d_lo != a_r:
                                    output.append(SassInstr(encode_iadd3(d_lo, a_r, RZ, RZ),
                                                            f'MOV R{d_lo}, R{a_r}  // cvt.{_dst_t}.{_src_t} lo'))
                                # hi = sign extension (s32) or 0 (u32/b32)
                                if _src_t == 's32' and _dst_t == 's64':
                                    # encode_shf_r_s32_hi already imported at module level
                                    output.append(SassInstr(
                                        encode_shf_r_s32_hi(d_lo+1, a_r, 31),
                                        f'SHF.R.S32.HI R{d_lo+1}, RZ, 31, R{a_r}  // cvt.s64.s32 sign'))
                                else:
                                    output.append(SassInstr(encode_iadd3(d_lo+1, RZ, RZ, RZ),
                                                            f'MOV R{d_lo+1}, RZ  // cvt.{_dst_t}.{_src_t} zero-ext'))
                            else:
                                output.append(_nop(f'TODO: cvt {".".join(instr.types)}'))

                elif op == 'setp':
                    pred = instr.dest
                    a    = instr.srcs[0]
                    b    = instr.srcs[1]
                    if isinstance(pred, RegOp) and isinstance(a, RegOp):
                        pd = ctx.ra.pred(pred.name) if pred.name in ctx.ra.pred_regs else 0
                        ar = ctx.ra.r32(a.name)
                        is_float = any(t in ('f32', 'f64') for t in instr.types)
                        cmp_name = next((t for t in instr.types if t in ('lt','le','gt','ge','eq','ne')), 'ge')
                        if is_float:
                            br = ctx.ra.r32(b.name) if isinstance(b, RegOp) else (b.value if isinstance(b, ImmOp) else 0)
                            cmp_map = {'lt': FSETP_LT, 'le': FSETP_LE, 'gt': FSETP_GT,
                                       'ge': FSETP_GE, 'eq': FSETP_EQ, 'ne': FSETP_NE}
                            output.append(SassInstr(
                                encode_fsetp(pd, ar, br, cmp_map.get(cmp_name, FSETP_GE)),
                                f'FSETP.{cmp_name.upper()} P{pd}, R{ar}, R{br}'))
                        else:
                            # Integer comparison: use ISETP R-UR (opcode 0xc0c) when src1
                            # is a u32 param backed GPR. The R-R variant (0x20c) silently
                            # produces P=FALSE on SM_120 hardware.
                            # SM_120: ISETP.LT encoding (b8=0x10) doesn't work on hardware.
                            # Invert LT→GE and GT→LE, negate the predicate on branches.
                            _INVERT = {'lt': 'ge', 'gt': 'le'}
                            if cmp_name in _INVERT:
                                cmp_name = _INVERT[cmp_name]
                                if not hasattr(ctx, '_negated_preds'):
                                    ctx._negated_preds = set()
                                ctx._negated_preds.add(pd)
                            cmp_map = {'lt': ISETP_LT, 'le': ISETP_LE, 'gt': ISETP_GT,
                                       'ge': ISETP_GE, 'eq': ISETP_EQ, 'ne': ISETP_NE}
                            isetp_cmp = cmp_map.get(cmp_name, ISETP_GE)
                            if isinstance(b, RegOp):
                                b_param_off = ctx._reg_param_off.get(b.name) if ctx else None
                                if b_param_off is not None:
                                    # src1 came from ld.param.u32 — load via LDCU.32 into UR,
                                    # then compare R vs UR. This is the ptxas-verified path.
                                    #
                                    # SM_120 hardware: ISETP R-UR (0xc0c) silently produces
                                    # P=FALSE when pred_dest > 0. Force pred_dest=0 (P0) and
                                    # remap the PTX predicate so consumers use P0 with their
                                    # original sign — no comparison inversion needed.
                                    emit_pd = pd
                                    if pd > 0 and ctx:
                                        emit_pd = 0
                                        ctx.ra.pred_regs[pred.name] = 0
                                    ur_tmp = ctx._next_ur
                                    ctx._next_ur += 1
                                    output.append(SassInstr(
                                        encode_ldcu_32(ur_tmp, 0, b_param_off),
                                        f'LDCU.32 UR{ur_tmp}, c[0][0x{b_param_off:x}]  // setp src'))
                                    output.append(SassInstr(
                                        encode_isetp_ur(emit_pd, ar, ur_tmp, cmp=isetp_cmp),
                                        f'ISETP.{cmp_name.upper()}.U32.AND P{emit_pd}, PT, R{ar}, UR{ur_tmp}, PT'))
                                else:
                                    br = ctx.ra.r32(b.name)
                                    # R-R fallback; only works correctly if both operands are GPRs
                                    # not tracked as param-backed. Note: R-R (0x20c) is broken on
                                    # SM_120 for the predicated-exit pattern — TODO: fix when needed.
                                    output.append(SassInstr(
                                        encode_isetp(pd, ar, br, cmp=isetp_cmp),
                                        f'ISETP.{cmp_name.upper()}.U32.AND P{pd}, PT, R{ar}, R{br}, PT'))
                            elif isinstance(b, ImmOp):
                                # Immediate src1: load via literal pool into UR,
                                # then use ISETP R-UR (verified path on SM_120).
                                imm_val = b.value & 0xFFFFFFFF
                                lit_off = ctx._alloc_literal(imm_val)
                                emit_pd = pd
                                if pd > 0 and ctx:
                                    emit_pd = 0
                                    ctx.ra.pred_regs[pred.name] = 0
                                ur_tmp = ctx._next_ur
                                ctx._next_ur += 1
                                output.append(SassInstr(
                                    encode_ldcu_32(ur_tmp, 0, lit_off),
                                    f'LDCU.32 UR{ur_tmp}, c[0][0x{lit_off:x}]  // setp imm={imm_val:#x}'))
                                output.append(SassInstr(
                                    encode_isetp_ur(emit_pd, ar, ur_tmp, cmp=isetp_cmp),
                                    f'ISETP.{cmp_name.upper()}.U32.AND P{emit_pd}, PT, R{ar}, UR{ur_tmp}, PT'))
                            else:
                                output.append(_nop(f'TODO: setp with non-register src1'))
                    else:
                        output.append(_nop(f'TODO: setp {instr}'))

                elif op == 'testp' and 'finite' in instr.types and 'f32' in instr.types:
                    # testp.finite.f32 p, f:
                    #   p = isfinite(f) = (f_bits & 0x7F800000) < 0x7F800000
                    # Lowering (4 instructions):
                    #   R_mask = 0x7F800000         (IADD3_IMM)
                    #   R_abs  = f_bits & R_mask    (LOP3.AND)
                    #   UR_thr = 0x7F800000         (LDCU.32 from literal pool)
                    #   p      = (R_abs < UR_thr)   (ISETP.LT.U32.AND R-UR)
                    pred   = instr.dest
                    f_op   = instr.srcs[0]
                    pd = ctx.ra.pred(pred.name) if pred.name in ctx.ra.pred_regs else 0
                    emit_pd = pd
                    if pd > 0 and ctx:
                        emit_pd = 0
                        ctx.ra.pred_regs[pred.name] = 0
                    f_reg = ctx.ra.r32(f_op.name)
                    R_mask = ctx._next_gpr; ctx._next_gpr += 1
                    R_abs  = ctx._next_gpr; ctx._next_gpr += 1
                    FINITE_MASK = 0x7F800000
                    output.append(SassInstr(
                        encode_iadd3_imm32(R_mask, RZ, FINITE_MASK, RZ),
                        f'MOV R{R_mask}, 0x7f800000  // testp.finite mask'))
                    output.append(SassInstr(
                        encode_lop3(R_abs, f_reg, R_mask, RZ, LOP3_AND),
                        f'LOP3.AND R{R_abs}, R{f_reg}, R{R_mask}, RZ  // testp.finite & exp mask'))
                    lit_off = ctx._alloc_literal(FINITE_MASK)
                    ur_thr  = ctx._next_ur; ctx._next_ur += 1
                    output.append(SassInstr(
                        encode_ldcu_32(ur_thr, 0, lit_off),
                        f'LDCU.32 UR{ur_thr}, c[0][0x{lit_off:x}]  // testp.finite threshold'))
                    output.append(SassInstr(
                        encode_isetp_ur(emit_pd, R_abs, ur_thr, cmp=ISETP_LT),
                        f'ISETP.LT.U32.AND P{emit_pd}, PT, R{R_abs}, UR{ur_thr}, PT  // testp.finite'))

                elif op == 'neg' and typ in ('s32', 'u32'):
                    # neg: IADD3 with src0=RZ, src1=src, negate_src1
                    # dest = 0 - src
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_iadd3(d, RZ, a, RZ, negate_src1=True),
                                            f'IADD3 R{d}, RZ, -R{a}, RZ  // neg.{typ}'))

                elif op == 'neg' and typ in ('s64', 'u64', 'b64'):
                    # neg.s64: IADD.64 d, RZ, -a  (two's complement of 64-bit value)
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = ctx.ra.lo(instr.srcs[0].name)
                    output.append(SassInstr(encode_iadd64(d_lo, RZ, a_lo, negate_src1=True),
                                            f'IADD.64 R{d_lo}, RZ, -R{a_lo}  // neg.{typ}'))

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
                    pd = 0
                    if len(instr.srcs) > 2 and isinstance(instr.srcs[2], RegOp):
                        pd = ctx.ra.pred(instr.srcs[2].name) if instr.srcs[2].name in ctx.ra.pred_regs else 0
                    def _sel_src(src_op, out):
                        if isinstance(src_op, RegOp):
                            return ctx.ra.r32(src_op.name)
                        elif isinstance(src_op, ImmOp):
                            t = ctx._next_gpr; ctx._next_gpr += 1
                            out.append(SassInstr(encode_iadd3_imm32(t, RZ, src_op.value & 0xFFFFFFFF, RZ),
                                                 f'MOV R{t}, {src_op.value}  // selp imm'))
                            return t
                        return RZ
                    a = _sel_src(instr.srcs[0], output)
                    b = _sel_src(instr.srcs[1], output)
                    output.append(SassInstr(encode_sel(d, a, b, pd),
                                            f'SEL R{d}, R{a}, R{b}, P{pd}  // selp'))

                elif op == 'min' and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    if isinstance(instr.srcs[1], ImmOp):
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        output.append(SassInstr(encode_vimnmx_u32(d, a, imm, is_max=False),
                            f'VIMNMX.U32 R{d}, R{a}, 0x{imm:x}, PT  // min.{typ} imm'))
                    else:
                        b = ctx.ra.r32(instr.srcs[1].name)
                        output.append(SassInstr(encode_vimnmx_s32(d, a, b, is_max=False),
                            f'VIMNMX.S32 R{d}, R{a}, R{b}, PT  // min.{typ}'))

                elif op == 'max' and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    if isinstance(instr.srcs[1], ImmOp):
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        output.append(SassInstr(encode_vimnmx_u32(d, a, imm, is_max=True),
                            f'VIMNMX.U32 R{d}, R{a}, 0x{imm:x}, !PT  // max.{typ} imm'))
                    else:
                        b = ctx.ra.r32(instr.srcs[1].name)
                        output.append(SassInstr(encode_vimnmx_s32(d, a, b, is_max=True),
                            f'VIMNMX.S32 R{d}, R{a}, R{b}, !PT  // max.{typ}'))

                elif op == 'sad' and typ in ('u32', 's32'):
                    # sad.u32 d, a, b, c  →  d = |a - b| + c
                    # VIMNMX.MAX t0, a, b
                    # VIMNMX.MIN t1, a, b
                    # IADD3 d, t0, -t1, c  (d = max - min + c = |a-b| + c)
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    c = ctx.ra.r32(instr.srcs[2].name) if len(instr.srcs) > 2 and isinstance(instr.srcs[2], RegOp) else RZ
                    t_max = ctx._next_gpr; ctx._next_gpr += 1
                    t_min = ctx._next_gpr; ctx._next_gpr += 1
                    is_signed = typ == 's32'
                    output.append(SassInstr(encode_vimnmx_s32(t_max, a, b, is_max=True) if is_signed else encode_vimnmx_u32(t_max, a, b, is_max=True),
                                            f'VIMNMX.{"S" if is_signed else "U"}32 R{t_max}, R{a}, R{b}  // sad max'))
                    output.append(SassInstr(encode_vimnmx_s32(t_min, a, b, is_max=False) if is_signed else encode_vimnmx_u32(t_min, a, b, is_max=False),
                                            f'VIMNMX.{"S" if is_signed else "U"}32 R{t_min}, R{a}, R{b}  // sad min'))
                    output.append(SassInstr(encode_iadd3_neg_b4(d, t_max, t_min, c),
                                            f'IADD3 R{d}, R{t_max}, -R{t_min}, R{c}  // sad |a-b|+c'))

                elif op == 'mad' and 'lo' in instr.types:
                    # mad.lo.s32 → dest = src0 * src1 + src2
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    c_op = instr.srcs[2] if len(instr.srcs) > 2 else None
                    c = ctx.ra.r32(c_op.name) if isinstance(c_op, RegOp) else RZ
                    if isinstance(instr.srcs[1], ImmOp):
                        # Immediate multiplier: IMAD.SHL if power-of-2, else LDCU+IMAD R-UR
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        if imm > 0 and (imm & (imm - 1)) == 0:
                            shift = imm.bit_length() - 1
                            if shift <= 15:
                                t = ctx._next_gpr; ctx._next_gpr += 1
                                output.append(SassInstr(encode_imad_shl_u32(t, a, shift),
                                    f'IMAD.SHL.U32 R{t}, R{a}, 0x{imm:x}, RZ  // mad.lo shift'))
                                output.append(SassInstr(encode_iadd3(d, t, c, RZ),
                                    f'IADD3 R{d}, R{t}, R{c}, RZ  // mad.lo add'))
                            else:
                                lit_off = ctx._alloc_literal(imm)
                                ur_tmp = ctx._next_ur; ctx._next_ur += 1
                                output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, lit_off),
                                    f'LDCU.32 UR{ur_tmp}, c[0][0x{lit_off:x}]'))
                                output.append(SassInstr(encode_imad_ur(d, a, ur_tmp, c),
                                    f'IMAD R{d}, R{a}, UR{ur_tmp}, R{c}  // mad.lo imm'))
                        else:
                            lit_off = ctx._alloc_literal(imm)
                            ur_tmp = ctx._next_ur; ctx._next_ur += 1
                            output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, lit_off),
                                f'LDCU.32 UR{ur_tmp}, c[0][0x{lit_off:x}]'))
                            output.append(SassInstr(encode_imad_ur(d, a, ur_tmp, c),
                                f'IMAD R{d}, R{a}, UR{ur_tmp}, R{c}  // mad.lo imm'))
                        continue
                    b = ctx.ra.r32(instr.srcs[1].name)
                    src0_name = instr.srcs[0].name if instr.srcs else ''
                    src1_name = instr.srcs[1].name if len(instr.srcs) > 1 else ''
                    ur_map = getattr(ctx, '_ur_for_param', {})
                    if src0_name in ur_map:
                        # src0 is in a UR (e.g. ctaid via S2UR) — use IMAD R-UR.
                        # IMAD R-UR: dest = src0_gpr * ur + src2. Multiplication is
                        # commutative so we put the GPR operand (src1) in src0 position.
                        ur_src = ur_map[src0_name]
                        output.append(SassInstr(encode_imad_ur(d, b, ur_src, c),
                            f'IMAD R{d}, R{b}, UR{ur_src}, R{c}  // mad.lo.{typ}'))
                    elif src1_name in ur_map:
                        ur_src = ur_map[src1_name]
                        output.append(SassInstr(encode_imad_ur(d, a, ur_src, c),
                            f'IMAD R{d}, R{a}, UR{ur_src}, R{c}  // mad.lo.{typ}'))
                    else:
                        # Both src0 and src1 are computed GPRs — use R-R IMAD (0x2a4).
                        output.append(SassInstr(encode_imad_rr(d, a, b, c),
                            f'IMAD R{d}, R{a}, R{b}, R{c}  // mad.lo.{typ} R-R'))

                elif op == 'mad' and 'wide' in instr.types and typ in ('u32', 's32'):
                    # mad.wide.u32/s32 d64, a32, b32_or_imm, c64
                    # Result pair: (dest_lo, dest_hi) = a * b + c64
                    # IMAD.WIDE writes dest and dest+1 atomically.
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a    = ctx.ra.r32(instr.srcs[0].name)
                    c_lo = ctx.ra.lo(instr.srcs[2].name) if len(instr.srcs) > 2 else RZ
                    if isinstance(instr.srcs[1], ImmOp):
                        imm = instr.srcs[1].value & 0xFFFF_FFFF
                        if imm <= 0xFF:
                            output.append(SassInstr(
                                encode_imad_wide(d_lo, a, imm, c_lo),
                                f'IMAD.WIDE R{d_lo}, R{a}, 0x{imm:x}, R{c_lo}  // mad.wide.{typ}'))
                        else:
                            # Large immediate: load via literal pool into UR, then R-UR IMAD.WIDE
                            lit_off = ctx._alloc_literal(imm)
                            ur_tmp = ctx._next_ur; ctx._next_ur += 1
                            output.append(SassInstr(
                                encode_ldcu_32(ur_tmp, 0, lit_off),
                                f'LDCU.32 UR{ur_tmp}, c[0][0x{lit_off:x}]  // mad.wide imm={imm:#x}'))
                            # Use R-imm form with UR treated as immediate slot
                            output.append(SassInstr(
                                encode_imad_wide(d_lo, a, ur_tmp, c_lo),
                                f'IMAD.WIDE R{d_lo}, R{a}, UR{ur_tmp}, R{c_lo}  // mad.wide large imm'))
                    else:
                        b = ctx.ra.r32(instr.srcs[1].name)
                        output.append(SassInstr(
                            encode_imad_wide_rr(d_lo, a, b, c_lo),
                            f'IMAD.WIDE R{d_lo}, R{a}, R{b}, R{c_lo}  // mad.wide.{typ} R-R'))

                elif op == 'mul' and 'hi' in instr.types and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    b = ctx.ra.r32(instr.srcs[1].name)
                    output.append(SassInstr(encode_imad_hi(d, a, b, RZ, signed=(typ == 's32')),
                                            f'IMAD.HI R{d}, R{a}, R{b}, RZ  // mul.hi.{typ}'))

                elif op == 'mul' and 'hi' in instr.types and typ in ('u64', 's64'):
                    # mul.hi.u64: upper 64 bits of 128-bit unsigned product.
                    # Algorithm (schoolbook using IMAD.WIDE.U32):
                    #   a_lo, a_hi = src0 pair; b_lo, b_hi = src1 pair
                    #   t0 = IMAD.WIDE.U32(a_hi, b_lo, 0)          → a_hi*b_lo [64-bit]
                    #   t1 = IMAD.WIDE.U32(a_lo, b_hi, t0, P0)     → a_lo*b_hi + t0 [64-bit, sets P0]
                    #   t2 = IMAD.WIDE.U32(a_lo, b_lo, 0)          → a_lo*b_lo [64-bit]
                    #   carry = 0 + P0 (IADD3.X)                   → capture carry from step 2
                    #   sum_hi = t1_hi (=R9 in ground truth)
                    #   IADD3 RZ, P0, t2_hi, t1_lo, RZ             → detect carry from bit-32 sum
                    #   d = IMAD.WIDE.U32.X(a_hi, b_hi, sum_hi, P0) → a_hi*b_hi + sum_hi + carry
                    # Ground truth verified from ptxas mul.hi.u64 on SM_120.
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = ctx.ra.lo(instr.srcs[0].name)
                    b_lo = ctx.ra.lo(instr.srcs[1].name)
                    a_hi = a_lo + 1;  b_hi = b_lo + 1
                    t0_lo = ctx._next_gpr; ctx._next_gpr += 2  # t0 pair
                    t1_lo = ctx._next_gpr; ctx._next_gpr += 2  # t1 pair
                    t2_lo = ctx._next_gpr; ctx._next_gpr += 2  # t2 pair
                    carry = ctx._next_gpr; ctx._next_gpr += 1
                    # Step 1: t0 = a_hi * b_lo
                    output.append(SassInstr(encode_imad_wide_u32(t0_lo, a_hi, b_lo, RZ),
                        f'IMAD.WIDE.U32 R{t0_lo}, R{a_hi}, R{b_lo}, RZ  // mul.hi.u64 step1'))
                    # Step 2: t1 = a_lo * b_hi + t0 (sets P0 carry)
                    output.append(SassInstr(encode_imad_wide_u32_carry(t1_lo, a_lo, b_hi, t0_lo),
                        f'IMAD.WIDE.U32 R{t1_lo}, P0, R{a_lo}, R{b_hi}, R{t0_lo}  // mul.hi.u64 step2'))
                    # Step 3: t2 = a_lo * b_lo
                    output.append(SassInstr(encode_imad_wide_u32(t2_lo, a_lo, b_lo, RZ),
                        f'IMAD.WIDE.U32 R{t2_lo}, R{a_lo}, R{b_lo}, RZ  // mul.hi.u64 step3'))
                    # Step 4: carry = 0 + P0 carry from step 2
                    output.append(SassInstr(encode_iadd3x(carry, RZ, RZ, RZ),
                        f'IADD3.X R{carry}, PT, PT, RZ, RZ, RZ, P0, !PT  // mul.hi.u64 carry'))
                    # Step 5: save t1_hi (hi word of a_lo*b_hi + t0) for final product
                    sum_hi = ctx._next_gpr; ctx._next_gpr += 1
                    output.append(SassInstr(encode_mov(sum_hi, t1_lo + 1),
                        f'MOV R{sum_hi}, R{t1_lo+1}  // mul.hi.u64 save t1_hi'))
                    # Step 6: IADD3 to detect carry from t2_hi + t1_lo into P0
                    output.append(SassInstr(encode_iadd3(RZ, t2_lo + 1, t1_lo, RZ),
                        f'IADD3 RZ, P0, PT, R{t2_lo+1}, R{t1_lo}, RZ  // mul.hi.u64 carry detect'))
                    # Step 7: d = a_hi * b_hi + sum_hi + P0 carry
                    output.append(SassInstr(encode_imad_wide_u32x(d_lo, a_hi, b_hi, sum_hi),
                        f'IMAD.WIDE.U32.X R{d_lo}, R{a_hi}, R{b_hi}, R{sum_hi}, P0  // mul.hi.u64 final'))

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

                elif op == 'abs' and typ in ('s64',):
                    # abs.s64 d, a  — branchless sign-bit trick:
                    #   sign = arithmetic-right-shift(a_hi, 31) = 0 or 0xFFFFFFFF
                    #   d    = (a XOR sign) + (-sign)   where -sign = 0 or 1
                    # This avoids predicated 64-bit instructions.
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = ctx.ra.lo(instr.srcs[0].name)
                    sign = ctx._next_gpr; ctx._next_gpr += 1
                    t_lo = ctx._next_gpr; ctx._next_gpr += 1  # addend lo (0 or 1)
                    t_hi = ctx._next_gpr; ctx._next_gpr += 1  # addend hi (always 0)
                    output.append(SassInstr(encode_shf_r_s32_hi(sign, a_lo+1, 31),
                        f'SHF.R.S32.HI R{sign}, RZ, 0x1f, R{a_lo+1}  // abs.s64 sign'))
                    output.append(SassInstr(encode_lop3(d_lo,   a_lo,   sign, RZ, LOP3_XOR),
                        f'LOP3.XOR R{d_lo}, R{a_lo}, R{sign}, RZ  // abs.s64 lo XOR'))
                    output.append(SassInstr(encode_lop3(d_lo+1, a_lo+1, sign, RZ, LOP3_XOR),
                        f'LOP3.XOR R{d_lo+1}, R{a_lo+1}, R{sign}, RZ  // abs.s64 hi XOR'))
                    output.append(SassInstr(encode_iadd3(t_hi, RZ, RZ, RZ),
                        f'MOV R{t_hi}, RZ  // abs.s64 addend hi=0'))
                    output.append(SassInstr(encode_iadd3(t_lo, RZ, sign, RZ, negate_src1=True),
                        f'IADD3 R{t_lo}, RZ, -R{sign}, RZ  // abs.s64 addend=-sign'))
                    output.append(SassInstr(encode_iadd64(d_lo, d_lo, t_lo),
                        f'IADD.64 R{d_lo}, R{d_lo}, R{t_lo}  // abs.s64 add'))

                elif op == 'min' and typ in ('u64', 's64'):
                    # min.u64 branchless: min(a,b) = b + ((a-b) & sign_mask(a-b))
                    #   diff = a - b; mask = sign_fill(diff_hi); d = b + (diff & mask)
                    # Works for unsigned because a < b → diff wraps to large value with sign=1.
                    # For signed min (s64), the same bit trick applies (signed subtraction).
                    d_lo  = ctx.ra.lo(instr.dest.name)
                    a_lo  = ctx.ra.lo(instr.srcs[0].name)
                    b_lo  = ctx.ra.lo(instr.srcs[1].name)
                    t_lo  = ctx._next_gpr; ctx._next_gpr += 2   # diff pair (t_lo, t_lo+1)
                    mask  = ctx._next_gpr; ctx._next_gpr += 1
                    output.append(SassInstr(encode_iadd64(t_lo, a_lo, b_lo, negate_src1=True),
                        f'IADD.64 R{t_lo}, R{a_lo}, -R{b_lo}  // min.{typ} diff'))
                    output.append(SassInstr(encode_shf_r_s32_hi(mask, t_lo+1, 31),
                        f'SHF.R.S32.HI R{mask}, RZ, 0x1f, R{t_lo+1}  // min.{typ} mask'))
                    output.append(SassInstr(encode_lop3(t_lo,   t_lo,   mask, RZ, LOP3_AND),
                        f'LOP3.AND R{t_lo}, R{t_lo}, R{mask}, RZ  // min.{typ} lo'))
                    output.append(SassInstr(encode_lop3(t_lo+1, t_lo+1, mask, RZ, LOP3_AND),
                        f'LOP3.AND R{t_lo+1}, R{t_lo+1}, R{mask}, RZ  // min.{typ} hi'))
                    output.append(SassInstr(encode_iadd64(d_lo, b_lo, t_lo),
                        f'IADD.64 R{d_lo}, R{b_lo}, R{t_lo}  // min.{typ} result'))

                elif op == 'max' and typ in ('u64', 's64'):
                    # max.u64 branchless: max(a,b) = b + ((a-b) & ~sign_mask(a-b))
                    #   diff = a - b; mask = ~sign_fill(diff_hi); d = b + (diff & ~mask)
                    d_lo  = ctx.ra.lo(instr.dest.name)
                    a_lo  = ctx.ra.lo(instr.srcs[0].name)
                    b_lo  = ctx.ra.lo(instr.srcs[1].name)
                    t_lo  = ctx._next_gpr; ctx._next_gpr += 2   # diff pair
                    mask  = ctx._next_gpr; ctx._next_gpr += 1   # inverted sign mask
                    output.append(SassInstr(encode_iadd64(t_lo, a_lo, b_lo, negate_src1=True),
                        f'IADD.64 R{t_lo}, R{a_lo}, -R{b_lo}  // max.{typ} diff'))
                    output.append(SassInstr(encode_shf_r_s32_hi(mask, t_lo+1, 31),
                        f'SHF.R.S32.HI R{mask}, RZ, 0x1f, R{t_lo+1}  // max.{typ} sign'))
                    output.append(SassInstr(encode_lop3(mask, mask, RZ, RZ, 0x0F),
                        f'LOP3.NOT R{mask}, R{mask}, RZ, RZ  // max.{typ} ~sign'))
                    output.append(SassInstr(encode_lop3(t_lo,   t_lo,   mask, RZ, LOP3_AND),
                        f'LOP3.AND R{t_lo}, R{t_lo}, R{mask}, RZ  // max.{typ} lo'))
                    output.append(SassInstr(encode_lop3(t_lo+1, t_lo+1, mask, RZ, LOP3_AND),
                        f'LOP3.AND R{t_lo+1}, R{t_lo+1}, R{mask}, RZ  // max.{typ} hi'))
                    output.append(SassInstr(encode_iadd64(d_lo, b_lo, t_lo),
                        f'IADD.64 R{d_lo}, R{b_lo}, R{t_lo}  // max.{typ} result'))

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
                    # Full Newton-Raphson unsigned 32-bit division.
                    # Matches the exact sequence ptxas emits for div.u32 (sm_120).
                    # Ground truth: cuobjdump verified against ptxas 13.0 output.
                    d  = ctx.ra.r32(instr.dest.name)
                    a  = ctx.ra.r32(instr.srcs[0].name)   # dividend
                    b  = ctx.ra.r32(instr.srcs[1].name)   # divisor
                    # Allocate 4 scratch GPRs and 3 scratch predicate registers
                    t0 = ctx._next_gpr; ctx._next_gpr += 1
                    t1 = ctx._next_gpr; ctx._next_gpr += 1
                    t2 = ctx._next_gpr; ctx._next_gpr += 1
                    t3 = ctx._next_gpr; ctx._next_gpr += 1
                    pnz  = ctx._next_pred; ctx._next_pred += 1  # divisor != 0
                    pge1 = ctx._next_pred; ctx._next_pred += 1  # first correction
                    pge2 = ctx._next_pred; ctx._next_pred += 1  # second correction
                    # Step 1: float approximation of reciprocal (round-up for conservative estimate)
                    output.append(SassInstr(encode_i2f_u32_rp(t0, b),
                        f'I2F.U32.RP R{t0}, R{b}  // div.u32 step 1: float(divisor)'))
                    output.append(SassInstr(encode_isetp(pnz, b, RZ, ISETP_NE),
                        f'ISETP.NE.U32 P{pnz}, PT, R{b}, RZ, PT  // P{pnz}=(divisor!=0)'))
                    output.append(SassInstr(encode_mufu(t0, t0, MUFU_RCP),
                        f'MUFU.RCP R{t0}, R{t0}  // t0 = 1/float(divisor)'))
                    # Step 2: bias the reciprocal approximation toward the correct int
                    output.append(SassInstr(encode_iadd3_imm32(t1, t0, 0x0ffffffe, RZ),
                        f'IADD3 R{t1}, R{t0}, 0xffffffe, RZ  // bias rcp approx'))
                    output.append(SassInstr(encode_f2i_ftz_u32_trunc(t2, t1),
                        f'F2I.FTZ.U32.TRUNC R{t2}, R{t1}  // int approx of rcp'))
                    output.append(SassInstr(encode_hfma2_zero(t1),
                        f'HFMA2 R{t1}, -RZ, RZ, 0, 0  // zero R{t1}'))
                    # Step 3: Newton-Raphson refinement via multiply-high
                    output.append(SassInstr(encode_iadd3_neg_b4(t3, RZ, t2, RZ),
                        f'IADD3 R{t3}, RZ, -R{t2}, RZ  // negate approx'))
                    output.append(SassInstr(encode_imad(t3, t3, b, RZ),
                        f'IMAD R{t3}, R{t3}, R{b}, RZ  // error term'))
                    output.append(SassInstr(encode_imad_hi(t2, t2, t3, t1),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{t3}, R{t1}  // refine estimate'))
                    # Step 4: compute quotient approximation
                    output.append(SassInstr(encode_imad_hi(d, t2, a, RZ),
                        f'IMAD.HI.U32 R{d}, R{t2}, R{a}, RZ  // quotient approx'))
                    # Step 5: compute remainder and apply correction(s)
                    output.append(SassInstr(encode_iadd3_neg_b3(t3, d, RZ, RZ),
                        f'IADD3 R{t3}, -R{d}, RZ, RZ  // negate quotient'))
                    output.append(SassInstr(encode_imad(t3, b, t3, a),
                        f'IMAD R{t3}, R{b}, R{t3}, R{a}  // remainder = a - d*b'))
                    output.append(SassInstr(encode_isetp(pge1, t3, b, ISETP_GE),
                        f'ISETP.GE.U32 P{pge1}, PT, R{t3}, R{b}, PT'))
                    output.append(SassInstr(encode_iadd3_pred_neg_b4(t3, t3, b, RZ, pge1),
                        f'@P{pge1} IADD3 R{t3}, R{t3}, -R{b}, RZ  // correction 1'))
                    output.append(SassInstr(encode_iadd3_pred_small_imm(d, d, 1, RZ, pge1),
                        f'@P{pge1} IADD3 R{d}, R{d}, 0x1, RZ  // correction 1'))
                    output.append(SassInstr(encode_isetp(pge2, t3, b, ISETP_GE),
                        f'ISETP.GE.U32 P{pge2}, PT, R{t3}, R{b}, PT'))
                    output.append(SassInstr(encode_iadd3_pred_small_imm(d, d, 1, RZ, pge2),
                        f'@P{pge2} IADD3 R{d}, R{d}, 0x1, RZ  // correction 2'))
                    # Step 6: handle division by zero (result = 0xFFFFFFFF per CUDA spec)
                    output.append(SassInstr(encode_lop3_pred(d, RZ, b, RZ, 0x33, pnz, inverted=True),
                        f'@!P{pnz} LOP3.LUT R{d}, RZ, R{b}, RZ, 0x33  // div-by-zero: result=0xFFFFFFFF'))

                elif op == 'div' and typ == 's32':
                    # Signed 32-bit division via Newton-Raphson on absolute values.
                    # Matches ptxas sm_120 div.s32 sequence: IABS both operands,
                    # LOP3.XOR to capture sign, NR on |a|/|b|, then sign-correct.
                    # Ground truth: cuobjdump verified against ptxas 13.0 output.
                    d  = ctx.ra.r32(instr.dest.name)
                    a  = ctx.ra.r32(instr.srcs[0].name)   # dividend
                    b  = ctx.ra.r32(instr.srcs[1].name)   # divisor
                    t0 = ctx._next_gpr; ctx._next_gpr += 1
                    t1 = ctx._next_gpr; ctx._next_gpr += 1
                    t2 = ctx._next_gpr; ctx._next_gpr += 1
                    t3 = ctx._next_gpr; ctx._next_gpr += 1
                    ab_s = ctx._next_gpr; ctx._next_gpr += 1  # |a| temp / saved |a|
                    sign = ctx._next_gpr; ctx._next_gpr += 1  # sign = a ^ b (bit 31)
                    ppos  = ctx._next_pred; ctx._next_pred += 1  # result is positive
                    pge1  = ctx._next_pred; ctx._next_pred += 1
                    pge2  = ctx._next_pred; ctx._next_pred += 1
                    pnz   = ctx._next_pred; ctx._next_pred += 1  # divisor != 0
                    # Compute |b| in t2 (reuse t2 for NR), |a| saved in ab_s
                    abs_b = ctx._next_gpr; ctx._next_gpr += 1  # |b| for NR
                    output.append(SassInstr(encode_iabs(abs_b, b),
                        f'IABS R{abs_b}, R{b}  // div.s32: |b|'))
                    output.append(SassInstr(encode_iabs(ab_s, a),
                        f'IABS R{ab_s}, R{a}  // div.s32: |a|'))
                    output.append(SassInstr(encode_i2f_s32_rp(t0, abs_b),
                        f'I2F.S32.RP R{t0}, R{abs_b}  // float(|b|) round-up'))
                    output.append(SassInstr(encode_lop3(sign, a, b, RZ, LOP3_XOR),
                        f'LOP3.XOR R{sign}, R{a}, R{b}, RZ  // sign = a^b'))
                    output.append(SassInstr(encode_mufu(t0, t0, MUFU_RCP),
                        f'MUFU.RCP R{t0}, R{t0}'))
                    output.append(SassInstr(encode_iadd3_imm32(t1, t0, 0x0ffffffe, RZ),
                        f'IADD3 R{t1}, R{t0}, 0xffffffe, RZ'))
                    output.append(SassInstr(encode_f2i_ftz_u32_trunc(t2, t1),
                        f'F2I.FTZ.U32.TRUNC R{t2}, R{t1}'))
                    output.append(SassInstr(encode_hfma2_zero(t1),
                        f'HFMA2 R{t1}, -RZ, RZ, 0, 0'))
                    output.append(SassInstr(encode_iadd3_neg_b4(t3, RZ, t2, RZ),
                        f'IADD3 R{t3}, RZ, -R{t2}, RZ'))
                    output.append(SassInstr(encode_imad(t3, t3, abs_b, RZ),
                        f'IMAD R{t3}, R{t3}, R{abs_b}, RZ'))
                    # ab_s = |a| (saved), use as dividend for NR
                    output.append(SassInstr(encode_imad_hi(t2, t2, t3, t1),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{t3}, R{t1}'))
                    output.append(SassInstr(encode_imad_hi(t2, t2, ab_s, RZ),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{ab_s}, RZ  // quotient approx'))
                    output.append(SassInstr(encode_iadd3_neg_b3(t3, t2, RZ, RZ),
                        f'IADD3 R{t3}, -R{t2}, RZ, RZ  // negate q'))
                    output.append(SassInstr(encode_imad(t3, abs_b, t3, ab_s),
                        f'IMAD R{t3}, R{abs_b}, R{t3}, R{ab_s}  // remainder'))
                    # Correction: if |b| > remainder, no correction needed
                    output.append(SassInstr(encode_isetp(pge1, abs_b, t3, ISETP_GT),
                        f'ISETP.GT.U32 P{pge1}, PT, R{abs_b}, R{t3}, PT'))
                    output.append(SassInstr(
                        encode_iadd3_pred_neg_b4(t3, t3, abs_b, RZ, pge1, inverted=True),
                        f'@!P{pge1} IADD3 R{t3}, R{t3}, -R{abs_b}, RZ'))
                    output.append(SassInstr(
                        encode_iadd3_pred_small_imm(t2, t2, 1, RZ, pge1, inverted=True),
                        f'@!P{pge1} IADD3 R{t2}, R{t2}, 0x1, RZ'))
                    # Sign check: if sign_bit >= 0 (positive), keep quotient as-is
                    output.append(SassInstr(encode_isetp(ppos, sign, RZ, ISETP_GE, signed=True),
                        f'ISETP.GE.S32 P{ppos}, PT, R{sign}, RZ, PT'))
                    output.append(SassInstr(encode_isetp(pge2, t3, abs_b, ISETP_GE),
                        f'ISETP.GE.U32 P{pge2}, PT, R{t3}, R{abs_b}, PT'))
                    output.append(SassInstr(
                        encode_iadd3_pred_small_imm(t2, t2, 1, RZ, pge2),
                        f'@P{pge2} IADD3 R{t2}, R{t2}, 0x1, RZ'))
                    # Check if divisor is zero
                    output.append(SassInstr(encode_isetp(pnz, b, RZ, ISETP_NE, signed=True),
                        f'ISETP.NE.S32 P{pnz}, PT, R{b}, RZ, PT'))
                    output.append(SassInstr(encode_mov(d, t2),
                        f'MOV R{d}, R{t2}'))
                    # Negate quotient if sign bit indicates negative result
                    output.append(SassInstr(
                        encode_iadd3_pred_neg_b3(d, d, RZ, RZ, ppos, inverted=True),
                        f'@!P{ppos} IADD3 R{d}, -R{d}, RZ, RZ'))
                    # Div-by-zero: result = 0xFFFFFFFF (CUDA signed div-by-zero behavior)
                    output.append(SassInstr(
                        encode_lop3_pred(d, RZ, b, RZ, 0x33, pnz, inverted=True),
                        f'@!P{pnz} LOP3.LUT R{d}, RZ, R{b}, RZ, 0x33  // div-by-zero'))

                elif op == 'rem' and typ == 'u32':
                    # rem.u32 d, a, b = a - (a/b)*b
                    # Uses same Newton-Raphson setup as div.u32 but outputs remainder.
                    # Ground truth: cuobjdump verified against ptxas 13.0 rem.u32 output.
                    d  = ctx.ra.r32(instr.dest.name)
                    a  = ctx.ra.r32(instr.srcs[0].name)
                    b  = ctx.ra.r32(instr.srcs[1].name)
                    t0 = ctx._next_gpr; ctx._next_gpr += 1
                    t1 = ctx._next_gpr; ctx._next_gpr += 1
                    t2 = ctx._next_gpr; ctx._next_gpr += 1
                    t3 = ctx._next_gpr; ctx._next_gpr += 1
                    pnz  = ctx._next_pred; ctx._next_pred += 1
                    pge1 = ctx._next_pred; ctx._next_pred += 1
                    pge2 = ctx._next_pred; ctx._next_pred += 1
                    # NR setup (same as div.u32)
                    output.append(SassInstr(encode_i2f_u32_rp(t0, b),
                        f'I2F.U32.RP R{t0}, R{b}  // rem.u32: float(divisor)'))
                    output.append(SassInstr(encode_isetp(pnz, b, RZ, ISETP_NE),
                        f'ISETP.NE.U32 P{pnz}, PT, R{b}, RZ, PT'))
                    output.append(SassInstr(encode_mufu(t0, t0, MUFU_RCP),
                        f'MUFU.RCP R{t0}, R{t0}'))
                    output.append(SassInstr(encode_iadd3_imm32(t1, t0, 0x0ffffffe, RZ),
                        f'IADD3 R{t1}, R{t0}, 0xffffffe, RZ'))
                    output.append(SassInstr(encode_f2i_ftz_u32_trunc(t2, t1),
                        f'F2I.FTZ.U32.TRUNC R{t2}, R{t1}'))
                    output.append(SassInstr(encode_hfma2_zero(t1),
                        f'HFMA2 R{t1}, -RZ, RZ, 0, 0'))
                    output.append(SassInstr(encode_iadd3_neg_b4(t3, RZ, t2, RZ),
                        f'IADD3 R{t3}, RZ, -R{t2}, RZ'))
                    output.append(SassInstr(encode_imad(t3, t3, b, RZ),
                        f'IMAD R{t3}, R{t3}, R{b}, RZ'))
                    output.append(SassInstr(encode_imad_hi(t2, t2, t3, t1),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{t3}, R{t1}'))
                    output.append(SassInstr(encode_imad_hi(t2, t2, a, RZ),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{a}, RZ  // quotient approx in t2'))
                    # Compute remainder: negate quotient in-place, then IMAD
                    output.append(SassInstr(encode_iadd3_neg_b3(t2, t2, RZ, RZ),
                        f'IADD3 R{t2}, -R{t2}, RZ, RZ  // negate quotient'))
                    output.append(SassInstr(encode_imad(d, b, t2, a),
                        f'IMAD R{d}, R{b}, R{t2}, R{a}  // d = a - q*b = remainder'))
                    # Two correction loops (subtract divisor, not increment quotient)
                    output.append(SassInstr(encode_isetp(pge1, d, b, ISETP_GE),
                        f'ISETP.GE.U32 P{pge1}, PT, R{d}, R{b}, PT'))
                    output.append(SassInstr(encode_iadd3_pred_neg_b4(d, d, b, RZ, pge1),
                        f'@P{pge1} IADD3 R{d}, R{d}, -R{b}, RZ'))
                    output.append(SassInstr(encode_isetp(pge2, d, b, ISETP_GE),
                        f'ISETP.GE.U32 P{pge2}, PT, R{d}, R{b}, PT'))
                    output.append(SassInstr(encode_iadd3_pred_neg_b4(d, d, b, RZ, pge2),
                        f'@P{pge2} IADD3 R{d}, R{d}, -R{b}, RZ'))
                    output.append(SassInstr(encode_lop3_pred(d, RZ, b, RZ, 0x33, pnz, inverted=True),
                        f'@!P{pnz} LOP3.LUT R{d}, RZ, R{b}, RZ, 0x33  // rem of div-by-zero=0xFFFFFFFF'))

                elif op == 'rem' and typ == 's32':
                    # Signed 32-bit remainder via Newton-Raphson on absolute values.
                    # Sign of remainder = sign of dividend (C semantics: a = (a/b)*b + rem).
                    # Ground truth: cuobjdump verified against ptxas 13.0 rem.s32 output.
                    d     = ctx.ra.r32(instr.dest.name)
                    a     = ctx.ra.r32(instr.srcs[0].name)   # dividend (original, for sign)
                    b     = ctx.ra.r32(instr.srcs[1].name)   # divisor (original, for NE check)
                    abs_b = ctx._next_gpr; ctx._next_gpr += 1
                    abs_a = ctx._next_gpr; ctx._next_gpr += 1
                    t0    = ctx._next_gpr; ctx._next_gpr += 1
                    t1    = ctx._next_gpr; ctx._next_gpr += 1
                    t2    = ctx._next_gpr; ctx._next_gpr += 1
                    t3    = ctx._next_gpr; ctx._next_gpr += 1
                    pgt1  = ctx._next_pred; ctx._next_pred += 1  # |b| > rem (no correction)
                    psign = ctx._next_pred; ctx._next_pred += 1  # a >= 0
                    pgt2  = ctx._next_pred; ctx._next_pred += 1  # second correction check
                    pnz   = ctx._next_pred; ctx._next_pred += 1  # b != 0
                    output.append(SassInstr(encode_iabs(abs_b, b),
                        f'IABS R{abs_b}, R{b}  // rem.s32: |b|'))
                    output.append(SassInstr(encode_iabs(abs_a, a),
                        f'IABS R{abs_a}, R{a}  // rem.s32: |a|'))
                    output.append(SassInstr(encode_i2f_s32_rp(t0, abs_b),
                        f'I2F.S32.RP R{t0}, R{abs_b}  // float(|b|) round-up'))
                    output.append(SassInstr(encode_mufu(t0, t0, MUFU_RCP),
                        f'MUFU.RCP R{t0}, R{t0}'))
                    output.append(SassInstr(encode_iadd3_imm32(t1, t0, 0x0ffffffe, RZ),
                        f'IADD3 R{t1}, R{t0}, 0xffffffe, RZ'))
                    output.append(SassInstr(encode_f2i_ftz_u32_trunc(t2, t1),
                        f'F2I.FTZ.U32.TRUNC R{t2}, R{t1}'))
                    output.append(SassInstr(encode_hfma2_zero(t1),
                        f'HFMA2 R{t1}, -RZ, RZ, 0, 0'))
                    output.append(SassInstr(encode_iadd3_neg_b4(t3, RZ, t2, RZ),
                        f'IADD3 R{t3}, RZ, -R{t2}, RZ'))
                    output.append(SassInstr(encode_imad(t3, t3, abs_b, RZ),
                        f'IMAD R{t3}, R{t3}, R{abs_b}, RZ'))
                    output.append(SassInstr(encode_imad_hi(t2, t2, t3, t1),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{t3}, R{t1}'))
                    output.append(SassInstr(encode_imad_hi(t2, t2, abs_a, RZ),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{abs_a}, RZ  // quotient approx'))
                    output.append(SassInstr(encode_iadd3_neg_b3(t2, t2, RZ, RZ),
                        f'IADD3 R{t2}, -R{t2}, RZ, RZ  // negate quotient'))
                    output.append(SassInstr(encode_imad(d, abs_b, t2, abs_a),
                        f'IMAD R{d}, R{abs_b}, R{t2}, R{abs_a}  // rem = |a| + |b|*(-q)'))
                    # Correction 1: if |b| > rem, no subtract needed; else subtract |b|
                    output.append(SassInstr(encode_isetp(pgt1, abs_b, d, ISETP_GT),
                        f'ISETP.GT.U32 P{pgt1}, PT, R{abs_b}, R{d}, PT'))
                    output.append(SassInstr(
                        encode_iadd3_pred_neg_b4(d, d, abs_b, RZ, pgt1, inverted=True),
                        f'@!P{pgt1} IADD3 R{d}, R{d}, -R{abs_b}, RZ'))
                    # Sign check: P=1 if original dividend was non-negative
                    output.append(SassInstr(encode_isetp(psign, a, RZ, ISETP_GE, signed=True),
                        f'ISETP.GE.S32 P{psign}, PT, R{a}, RZ, PT'))
                    # Correction 2: second overshoot check
                    output.append(SassInstr(encode_isetp(pgt2, abs_b, d, ISETP_GT),
                        f'ISETP.GT.U32 P{pgt2}, PT, R{abs_b}, R{d}, PT'))
                    output.append(SassInstr(
                        encode_iadd3_pred_neg_b4(d, d, abs_b, RZ, pgt2, inverted=True),
                        f'@!P{pgt2} IADD3 R{d}, R{d}, -R{abs_b}, RZ'))
                    # Div-by-zero predicate
                    output.append(SassInstr(encode_isetp(pnz, b, RZ, ISETP_NE, signed=True),
                        f'ISETP.NE.S32 P{pnz}, PT, R{b}, RZ, PT'))
                    # Negate remainder if dividend was negative
                    output.append(SassInstr(
                        encode_iadd3_pred_neg_b3(d, d, RZ, RZ, psign, inverted=True),
                        f'@!P{psign} IADD3 R{d}, -R{d}, RZ, RZ'))
                    # Div-by-zero: result = 0xFFFFFFFF
                    output.append(SassInstr(
                        encode_lop3_pred(d, RZ, b, RZ, 0x33, pnz, inverted=True),
                        f'@!P{pnz} LOP3.LUT R{d}, RZ, R{b}, RZ, 0x33  // rem-by-zero'))

                elif op == 'rcp' and any(m in instr.types for m in ('approx','rn','rz','rm','rp')) and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_RCP),
                                            f'MUFU.RCP R{d}, R{a}'))

                elif op == 'sqrt' and any(m in instr.types for m in ('approx','rn','rz','rm','rp')) and typ == 'f32':
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
                        # prmt d, a, sel_imm, c  (selector is 2nd arg, immediate)
                        sel = instr.srcs[1].value
                        c = ctx.ra.r32(instr.srcs[2].name) if len(instr.srcs) > 2 else RZ
                        output.append(SassInstr(encode_prmt(d, a, sel, c),
                                                f'PRMT R{d}, R{a}, 0x{sel:04x}, R{c}'))
                    elif len(instr.srcs) >= 3 and isinstance(instr.srcs[2], ImmOp):
                        # prmt d, a, b, sel_imm  (selector is last arg, immediate)
                        b = ctx.ra.r32(instr.srcs[1].name)
                        sel = instr.srcs[2].value
                        output.append(SassInstr(encode_prmt(d, a, sel, b),
                                                f'PRMT R{d}, R{a}, 0x{sel:04x}, R{b}'))
                    elif len(instr.srcs) >= 3:
                        # prmt d, a, b, sel_reg  (all register operands)
                        b = ctx.ra.r32(instr.srcs[1].name)
                        sel_r = ctx.ra.r32(instr.srcs[2].name)
                        output.append(SassInstr(encode_prmt_reg(d, a, b, sel_r),
                                                f'PRMT.REG R{d}, R{a}, R{b}, R{sel_r}'))
                    else:
                        output.append(_nop(f'TODO: prmt with unsupported operands'))

                elif op == 'bfe' and typ == 'u32':
                    # Bit field extract: dest = (src >> start) & ((1<<length)-1)
                    # Decomposed as: SHF.R.U32.HI + (optional LDC + LOP3 for masking)
                    d = ctx.ra.r32(instr.dest.name)
                    a = ctx.ra.r32(instr.srcs[0].name)
                    start  = instr.srcs[1].value if isinstance(instr.srcs[1], ImmOp) else 0
                    length = instr.srcs[2].value if (len(instr.srcs) > 2 and isinstance(instr.srcs[2], ImmOp)) else 32
                    mask = (1 << length) - 1 if length < 32 else 0xFFFFFFFF
                    if length >= 32:
                        # No masking needed — just shift
                        output.append(SassInstr(
                            encode_shf_r_u32_hi(d, a, start),
                            f'SHF.R.U32.HI R{d}, RZ, 0x{start:x}, R{a}  // bfe.u32 len={length}'))
                    elif start == 0:
                        # Shift is 0 — just mask (LDC d=mask, LOP3 d=src&d)
                        lit_off = ctx._alloc_literal(mask)
                        output.append(SassInstr(
                            encode_ldc(d, 0, lit_off),
                            f'LDC R{d}, c[0][0x{lit_off:x}]  // bfe.u32 mask=0x{mask:x}'))
                        output.append(SassInstr(
                            encode_lop3(d, a, d, RZ, LOP3_AND),
                            f'LOP3.LUT R{d}, R{a}, R{d}, RZ, 0xC0  // bfe.u32 &mask'))
                    else:
                        # General: SHF shift into d, load mask into temp, AND
                        output.append(SassInstr(
                            encode_shf_r_u32_hi(d, a, start),
                            f'SHF.R.U32.HI R{d}, RZ, 0x{start:x}, R{a}  // bfe.u32 >>start'))
                        lit_off = ctx._alloc_literal(mask)
                        t = ctx._next_gpr; ctx._next_gpr += 1
                        output.append(SassInstr(
                            encode_ldc(t, 0, lit_off),
                            f'LDC R{t}, c[0][0x{lit_off:x}]  // bfe.u32 mask=0x{mask:x}'))
                        output.append(SassInstr(
                            encode_lop3(d, d, t, RZ, LOP3_AND),
                            f'LOP3.LUT R{d}, R{d}, R{t}, RZ, 0xC0  // bfe.u32 &mask'))

                elif op == 'bfe' and typ == 's32':
                    # bfe.s32 dest, src, pos, len: sign-extend bits [pos+len-1:pos]
                    # Two-instruction sequence (ptxas ground truth):
                    #   If pos > 0: SHF.R.S32.HI dest, RZ, pos, src
                    #   Then:       BFE_SEXT dest, src_or_dest, len
                    # encode_shf_r_s32_hi already imported at module level
                    d   = ctx.ra.r32(instr.dest.name)
                    a   = ctx.ra.r32(instr.srcs[0].name)
                    pos = instr.srcs[1].value if isinstance(instr.srcs[1], ImmOp) else 0
                    length = instr.srcs[2].value if (len(instr.srcs) > 2 and isinstance(instr.srcs[2], ImmOp)) else 32
                    if pos > 0:
                        output.append(SassInstr(
                            encode_shf_r_s32_hi(d, a, pos),
                            f'SHF.R.S32.HI R{d}, RZ, {pos}, R{a}  // bfe.s32 pos={pos}'))
                        output.append(SassInstr(
                            encode_bfe_sext(d, d, length),
                            f'BFE_SEXT R{d}, R{d}, {length}  // bfe.s32 len={length}'))
                    else:
                        output.append(SassInstr(
                            encode_bfe_sext(d, a, length),
                            f'BFE_SEXT R{d}, R{a}, {length}  // bfe.s32 len={length}'))

                elif op == 'bfi' and typ in ('b32',):
                    # bfi.b32 d, a, b, start, len
                    #   d = (b & ~shifted_mask) | ((a << start) & shifted_mask)
                    # Requires constant start and len (most common case).
                    d  = ctx.ra.r32(instr.dest.name)
                    a  = ctx.ra.r32(instr.srcs[0].name) if isinstance(instr.srcs[0], RegOp) else RZ
                    b  = ctx.ra.r32(instr.srcs[1].name) if isinstance(instr.srcs[1], RegOp) else RZ
                    start = instr.srcs[2].value if len(instr.srcs) > 2 and isinstance(instr.srcs[2], ImmOp) else 0
                    count = instr.srcs[3].value if len(instr.srcs) > 3 and isinstance(instr.srcs[3], ImmOp) else 32
                    raw_mask  = (1 << count) - 1 if count < 32 else 0xFFFFFFFF
                    shifted_mask     = (raw_mask << start) & 0xFFFFFFFF
                    not_shifted_mask = (~shifted_mask) & 0xFFFFFFFF
                    t1 = ctx._next_gpr; ctx._next_gpr += 1
                    t2 = ctx._next_gpr; ctx._next_gpr += 1
                    # t1 = (a << start) & shifted_mask
                    output.append(SassInstr(
                        encode_shf_l_u32(t1, a, start),
                        f'SHF.L.U32 R{t1}, R{a}, 0x{start:x}, RZ  // bfi shift'))
                    lit_sm = ctx._alloc_literal(shifted_mask)
                    output.append(SassInstr(
                        encode_ldc(t2, 0, lit_sm),
                        f'LDC R{t2}, c[0][0x{lit_sm:x}]  // bfi shifted_mask'))
                    output.append(SassInstr(
                        encode_lop3(t1, t1, t2, RZ, LOP3_AND),
                        f'LOP3.LUT R{t1}, R{t1}, R{t2}, RZ, 0xC0  // bfi a&mask'))
                    # t2 = b & ~shifted_mask
                    lit_nsm = ctx._alloc_literal(not_shifted_mask)
                    output.append(SassInstr(
                        encode_ldc(t2, 0, lit_nsm),
                        f'LDC R{t2}, c[0][0x{lit_nsm:x}]  // bfi ~shifted_mask'))
                    output.append(SassInstr(
                        encode_lop3(t2, b, t2, RZ, LOP3_AND),
                        f'LOP3.LUT R{t2}, R{b}, R{t2}, RZ, 0xC0  // bfi b&~mask'))
                    # d = t1 | t2
                    output.append(SassInstr(
                        encode_lop3(d, t1, t2, RZ, LOP3_OR),
                        f'LOP3.LUT R{d}, R{t1}, R{t2}, RZ, 0xFC  // bfi insert'))

                else:
                    # Unsupported instruction: emit NOP with comment
                    output.append(_nop(f'TODO: {instr.op} {".".join(instr.types)} {instr.mods}'))

            except ISelError as e:
                # Emit NOP with error comment rather than crashing
                output.append(_nop(f'ISEL ERROR: {e}  [{instr.op}]'))

            # Apply predicate guard to all SASS instructions generated for
            # this PTX instruction (except bra/ret which handle it themselves).
            # LDCU (0x7ac) and S2UR (0x9c3) write to warp-uniform UR registers
            # and MUST NOT be predicated with divergent thread predicates —
            # the hardware ignores or mishandles divergent predicates on UR writes.
            _UR_WRITE_OPCODES = frozenset({0x7ac, 0x9c3})
            if instr.pred and op not in ('bra', 'ret'):
                pd = ctx.ra.pred(instr.pred) if instr.pred in ctx.ra.pred_regs else 0
                neg = instr.neg
                if hasattr(ctx, '_negated_preds') and pd in ctx._negated_preds:
                    neg = not neg
                pred_str = f'@{"!" if neg else ""}P{pd} '
                for si_idx in range(_pre_len, len(output)):
                    old = output[si_idx]
                    opcode = (old.raw[0] | (old.raw[1] << 8)) & 0xFFF
                    if opcode in _UR_WRITE_OPCODES:
                        continue  # UR-write instrs must be unconditional
                    new_raw = patch_pred(old.raw, pred=pd, neg=neg)
                    output[si_idx] = SassInstr(new_raw, pred_str + old.comment)

    # BRA offset fixup pass
    if hasattr(ctx, '_bra_fixups'):
        for bra_idx, target_label in ctx._bra_fixups:
            if target_label in ctx.label_map:
                target_byte = ctx.label_map[target_label]
                bra_byte = (bra_idx + 1) * 16  # offset from NEXT instruction
                rel_offset = target_byte - bra_byte
                # Preserve predicate from original BRA encoding
                old_raw = output[bra_idx].raw
                old_pred_byte = old_raw[1] & 0xF0  # predicate bits
                new_raw = encode_bra(rel_offset)
                new_raw = bytearray(new_raw)
                new_raw[1] = (new_raw[1] & 0x0F) | old_pred_byte
                output[bra_idx] = SassInstr(
                    bytes(new_raw),
                    f'{output[bra_idx].comment.split("BRA")[0]}BRA {target_label} (offset={rel_offset})')

    return output
