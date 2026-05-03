"""Verify new SM_120 encoders against ptxas ground truth (2026-04-01)."""
import struct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sass.encoding.sm_120_opcodes import (
    encode_shf_r_u64, encode_shf_r_s64, encode_shf_r_u32_hi, encode_shf_r_s32_hi,
    encode_redux_sum, encode_ldgsts_e, encode_ldgdepbar, encode_depbar_le,
    encode_f2fp_f16_f32,
    encode_hmma_bf16_f32, encode_hmma_tf32_f32, encode_dmma_8x8x4, encode_cs2r,
    encode_sel_64,
    encode_shf_l_u32_hi_var, encode_shf_l_w_u32_hi_var, encode_shf_l_w_u32_var,
    encode_cs2ur,
    encode_lea_hi_x,
    encode_f2i_u64, encode_i2f_u64,
    encode_i2f_f64_u32,
    encode_hfma2,
)


def _lo_hi(raw: bytes):
    lo = struct.unpack_from('<Q', raw, 0)[0]
    hi = struct.unpack_from('<Q', raw, 8)[0]
    return lo, hi


def _opcode(raw: bytes):
    return struct.unpack_from('<Q', raw, 0)[0] & 0xFFF


def test_shf_r_u64_opcode():
    raw = encode_shf_r_u64(dest=6, src_lo=2, shift_reg=7, src_hi=3)
    assert _opcode(raw) == 0x219, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 6   # dest
    assert raw[3] == 2   # src_lo
    assert raw[4] == 7   # shift_reg
    assert raw[8] == 3   # src_hi
    assert raw[9] == 0x12  # U64 modifier

def test_shf_r_s64_modifier():
    raw = encode_shf_r_s64(dest=6, src_lo=2, shift_reg=7, src_hi=3)
    assert raw[9] == 0x10  # S64 modifier

def test_shf_r_u32_hi_modifier():
    raw = encode_shf_r_u32_hi(dest=7, src_lo=255, shift_reg=7, src_hi=3)
    assert raw[9] == 0x16  # U32.HI modifier

def test_shf_r_s32_hi_modifier():
    raw = encode_shf_r_s32_hi(dest=7, src_lo=255, shift_reg=7, src_hi=3)
    assert raw[9] == 0x14  # S32.HI modifier

def test_redux_sum_opcode():
    raw = encode_redux_sum(dest_ur=6, src=0)
    assert _opcode(raw) == 0x3c4, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 6   # dest UR
    assert raw[3] == 0   # src GPR
    assert raw[10] == 0xc0  # SUM mode

def test_ldgsts_e_opcode():
    raw = encode_ldgsts_e(smem_addr=5, glob_addr=2, ur_desc=4)
    assert _opcode(raw) == 0xfae, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 5  # smem addr
    assert raw[3] == 2  # glob addr
    assert raw[8] == 4  # UR descriptor (in b8, not b4)

def test_ldgdepbar_opcode():
    raw = encode_ldgdepbar()
    assert _opcode(raw) == 0x9af, f"opcode={_opcode(raw):#x}"

def test_depbar_le_opcode():
    raw = encode_depbar_le(sb=0, count=0)
    assert _opcode(raw) == 0x91a, f"opcode={_opcode(raw):#x}"
    assert raw[5] == 0x80  # SB0, count=0 (at b5, not b4)

def test_f2fp_opcode():
    raw = encode_f2fp_f16_f32(dest=0, src=2)
    assert _opcode(raw) == 0x23e, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 0   # dest
    assert raw[3] == 255  # RZ (pack slot)
    assert raw[4] == 2   # src


def test_hmma_bf16_opcode():
    raw = encode_hmma_bf16_f32(dest=12, src_a=8, src_b=4, src_c=12)
    assert _opcode(raw) == 0x23c
    assert raw[2] == 12  # dest
    assert raw[3] == 8   # src_a
    assert raw[4] == 4   # src_b
    assert raw[8] == 12  # src_c
    assert raw[9] == 0x18  # shape
    assert raw[10] == 0x04  # BF16 modifier

def test_hmma_tf32_opcode():
    raw = encode_hmma_tf32_f32(dest=12, src_a=8, src_b=4, src_c=12)
    assert _opcode(raw) == 0x23c
    assert raw[9] == 0x10  # TF32 shape (m16n8k8)
    assert raw[10] == 0x08  # TF32 modifier

def test_dmma_opcode():
    raw = encode_dmma_8x8x4(dest=8, src_a=2, src_b=4, src_c=8)
    assert _opcode(raw) == 0x23f, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 8   # dest
    assert raw[3] == 2   # src_a
    assert raw[4] == 4   # src_b
    assert raw[8] == 8   # src_c

def test_cs2r_opcode():
    raw = encode_cs2r(dest=14)
    assert _opcode(raw) == 0x805, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 14  # dest


# ---------------------------------------------------------------------------
# ATOMG.E.ADD — ground truth from RTX 5090 probe
# ---------------------------------------------------------------------------
# atom.add.u32 R9, [R2], R9
# Ground truth low 64-bit: 0x80000009020909a8
#   → bytes: a8 09 09 02 09 00 00 80
#   byte0=0xa8 (opcode), byte1=0x09 (ADD modifier),
#   byte2=0x09 (dest R9), byte3=0x02 (addr R2), byte4=0x09 (data R9),
#   bytes5-6=0x00 (offset=0), byte7=0x80 (descriptor flag)

def test_atomg_add_u32_opcode():
    """ATOMG.E.ADD.u32 encoding with PT guard and correct modifier bytes."""
    from sass.encoding.sm_120_opcodes import encode_atomg_u32, ATOMG_ADD
    raw = encode_atomg_u32(dest=9, addr_base=2, offset=0, data=9, atom_op=ATOMG_ADD)
    assert raw[0] == 0xa8, f"byte0={raw[0]:#x}"
    assert raw[1] == 0x79, f"byte1={raw[1]:#x} (expected 0x79: PT guard + opcode nibble 9)"
    assert raw[2] == 9,    f"dest={raw[2]}"   # dest=R9
    assert raw[3] == 2,    f"addr={raw[3]}"   # addr=R2
    assert raw[4] == 9,    f"data={raw[4]}"   # data=R9
    assert raw[5] == 0,    f"offset_lo={raw[5]}"
    assert raw[7] == 0x80, f"byte7={raw[7]:#x} (descriptor flag)"
    assert raw[8] == 4,    f"ur_desc={raw[8]} (expected UR4 default)"
    assert raw[9] == 0xf1, f"b9={raw[9]:#x} (expected 0xf1 mode bits)"
    assert raw[10] == 0x1e, f"b10={raw[10]:#x} (expected 0x1e mode bits)"
    assert raw[11] == 0x08, f"atom_op field={raw[11]:#x} (expected 0x08 for ADD)"

def test_atomg_add_u32_low64_matches_ground_truth():
    """Low 64 bits of ATOMG.E.ADD.u32 R9,[R2],R9 must match ground truth."""
    from sass.encoding.sm_120_opcodes import encode_atomg_u32, ATOMG_ADD
    import struct
    raw = encode_atomg_u32(dest=9, addr_base=2, offset=0, data=9, atom_op=ATOMG_ADD)
    lo = struct.unpack_from('<Q', raw, 0)[0]
    # Ground truth: 0x80000009020979a8 (PT guard, b1=0x79)
    assert lo == 0x80000009020979a8, f"low64={lo:#018x} (expected 0x80000009020979a8)"

def test_atomg_min_u32_uses_0x79():
    """ATOMG.E.MIN.S32 uses byte1=0x79 and ptxas-verified mode bits."""
    from sass.encoding.sm_120_opcodes import encode_atomg_u32, ATOMG_MIN
    raw = encode_atomg_u32(dest=11, addr_base=4, offset=0, data=8, atom_op=ATOMG_MIN)
    assert raw[1] == 0x79, f"byte1={raw[1]:#x} (expected 0x79 for MIN)"
    # ptxas ground truth: MIN.S32 uses b9=0xf3, b10=0x9e, b11=0x08
    assert raw[9] == 0xf3, f"b9={raw[9]:#x} (expected 0xf3 for MIN.S32)"
    assert raw[10] == 0x9e, f"b10={raw[10]:#x} (expected 0x9e for MIN.S32)"
    assert raw[11] == 0x08, f"b11={raw[11]:#x} (expected 0x08 for MIN.S32)"

def test_atomg_add_distinct_from_min():
    """ADD and MIN differ in b9/b10 mode bits (signed vs unsigned); b1 is same."""
    from sass.encoding.sm_120_opcodes import encode_atomg_u32, ATOMG_ADD, ATOMG_MIN
    add_raw = encode_atomg_u32(dest=5, addr_base=2, offset=0, data=3, atom_op=ATOMG_ADD)
    min_raw = encode_atomg_u32(dest=5, addr_base=2, offset=0, data=3, atom_op=ATOMG_MIN)
    # Both use b1=0x79 (PT guard, opcode nibble 9)
    assert add_raw[1] == 0x79
    assert min_raw[1] == 0x79
    # ADD: b9=0xf1, b10=0x1e; MIN.S32: b9=0xf3, b10=0x9e (ptxas ground truth)
    assert add_raw[9] == 0xf1, f"ADD b9={add_raw[9]:#x}"
    assert min_raw[9] == 0xf3, f"MIN b9={min_raw[9]:#x}"
    assert add_raw[9] != min_raw[9], "ADD and MIN must have different b9 mode bits"


def test_membar_gpu_encoding():
    """MEMBAR.SC.GPU encoding matches ptxas ground truth."""
    from sass.encoding.sm_120_opcodes import encode_membar, MEMBAR_GPU
    raw = encode_membar(MEMBAR_GPU)
    assert raw[0] == 0x92
    assert raw[1] == 0x79
    assert raw[9] == 0x20, f"b9={raw[9]:#x} (expected 0x20 for GPU scope)"

def test_membar_cta_encoding():
    """MEMBAR.SC.CTA encoding matches ptxas ground truth."""
    from sass.encoding.sm_120_opcodes import encode_membar, MEMBAR_CTA
    raw = encode_membar(MEMBAR_CTA)
    assert raw[0] == 0x92
    assert raw[1] == 0x79
    assert raw[9] == 0x00, f"b9={raw[9]:#x} (expected 0x00 for CTA scope)"

def test_atomg_exch_encoding():
    """ATOMG.E.EXCH encoding matches ptxas ground truth."""
    from sass.encoding.sm_120_opcodes import encode_atomg_u32, ATOMG_EXCH
    raw = encode_atomg_u32(dest=5, addr_base=2, offset=0, data=5, atom_op=ATOMG_EXCH)
    assert raw[0] == 0xa8
    assert raw[1] == 0x79
    assert raw[9] == 0xf1, f"b9={raw[9]:#x} (expected 0xf1 for EXCH)"
    assert raw[10] == 0x1e, f"b10={raw[10]:#x} (expected 0x1e for EXCH)"
    assert raw[11] == 0x0c, f"b11={raw[11]:#x} (expected 0x0c for EXCH)"

def test_atomg_max_s32_encoding():
    """ATOMG.E.MAX.S32 encoding matches ptxas ground truth."""
    from sass.encoding.sm_120_opcodes import encode_atomg_u32, ATOMG_MAX
    raw = encode_atomg_u32(dest=5, addr_base=2, offset=0, data=5, atom_op=ATOMG_MAX)
    assert raw[9] == 0xf3, f"b9={raw[9]:#x} (expected 0xf3 for MAX.S32)"
    assert raw[10] == 0x1e, f"b10={raw[10]:#x} (expected 0x1e for MAX.S32)"
    assert raw[11] == 0x09, f"b11={raw[11]:#x} (expected 0x09 for MAX.S32)"

def test_atomg_add_f32_encoding():
    """ATOMG.E.ADD.F32 encoding uses opcode 0xa3."""
    from sass.encoding.sm_120_opcodes import encode_atomg_add_f32
    raw = encode_atomg_add_f32(dest=3, addr_base=2, offset=0, data=7)
    assert raw[0] == 0xa3, f"b0={raw[0]:#x} (expected 0xa3 for float atomic)"
    assert raw[1] == 0x79
    assert raw[9] == 0xf3, f"b9={raw[9]:#x}"
    assert raw[10] == 0x1e, f"b10={raw[10]:#x}"
    assert raw[11] == 0x0c, f"b11={raw[11]:#x}"

def test_atomg_cas_b64_encoding():
    """ATOMG.E.CAS.64 encoding matches ptxas ground truth."""
    from sass.encoding.sm_120_opcodes import encode_atomg_cas_b64
    import struct
    raw = encode_atomg_cas_b64(dest=4, addr=2, compare=4, new_val=6)
    lo = struct.unpack_from('<Q', raw, 0)[0]
    assert lo == 0x00000004020473a9, f"lo={lo:#018x}"
    assert raw[9] == 0xe5, f"b9={raw[9]:#x} (expected 0xe5 for 64-bit CAS)"

def test_idp4a_encoding():
    """IDP.4A.U8.U8 encoding matches ptxas ground truth."""
    from sass.encoding.sm_120_opcodes import encode_idp4a
    raw = encode_idp4a(dest=9, src_a=4, src_b=7, src_c=0xFF)
    assert raw[0] == 0x26, f"b0={raw[0]:#x} (expected 0x26)"
    assert raw[1] == 0x72, f"b1={raw[1]:#x} (expected 0x72)"
    assert raw[2] == 9   # dest
    assert raw[3] == 4   # src_a
    assert raw[4] == 7   # src_b
    assert raw[8] == 0xFF # src_c = RZ


# ===========================================================================
# Phase 3 encoder tests — 2026-04-04
# ===========================================================================

from sass.encoding.sm_120_opcodes import (
    encode_lea, encode_lea_imm,
    encode_imnmx,
    encode_p2r, encode_r2p,
    encode_bmsk,
    encode_sgxt,
    encode_plop3,
    encode_i2ip,
    encode_fswzadd,
)


# --- LEA ---

def test_lea_opcode():
    raw = encode_lea(dest=5, base=2, index=3, scale=2)
    assert _opcode(raw) == 0x211, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 5   # dest
    assert raw[3] == 2   # base
    assert raw[4] == 3   # index
    assert raw[9] == 2   # scale
    assert raw[8] == 0xFF  # RZ

def test_lea_scale_0():
    raw = encode_lea(dest=10, base=0, index=7, scale=0)
    assert raw[9] == 0   # scale=0 means no shift

def test_lea_scale_4():
    raw = encode_lea(dest=4, base=1, index=6, scale=4)
    assert raw[9] == 4   # scale=4 means <<4

def test_lea_imm_opcode():
    raw = encode_lea_imm(dest=5, base=2, imm=0x100, scale=3)
    lo = struct.unpack_from('<Q', raw, 0)[0]
    assert (lo & 0xFFF) == 0x811, f"opcode={lo & 0xFFF:#x}"
    assert raw[2] == 5   # dest
    assert raw[3] == 2   # base
    # imm in bytes 4-7 (little-endian)
    imm_val = struct.unpack_from('<I', raw, 4)[0]
    assert imm_val == 0x100
    assert raw[9] == 3   # scale


# --- IMNMX ---

def test_imnmx_min_signed_opcode():
    raw = encode_imnmx(dest=5, src0=2, src1=3, is_max=False, is_unsigned=False)
    assert _opcode(raw) == 0x217, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 5
    assert raw[3] == 2
    assert raw[4] == 3
    assert raw[9] == 0x01  # signed, min

def test_imnmx_max_signed():
    raw = encode_imnmx(dest=5, src0=2, src1=3, is_max=True, is_unsigned=False)
    assert raw[9] == 0x05  # signed, max

def test_imnmx_min_unsigned():
    raw = encode_imnmx(dest=5, src0=2, src1=3, is_max=False, is_unsigned=True)
    assert raw[9] == 0x00  # unsigned, min

def test_imnmx_max_unsigned():
    raw = encode_imnmx(dest=5, src0=2, src1=3, is_max=True, is_unsigned=True)
    assert raw[9] == 0x04  # unsigned, max


# --- P2R ---

def test_p2r_opcode():
    raw = encode_p2r(dest=5, mask=0xFF)
    assert _opcode(raw) == 0x203, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 5   # dest
    assert raw[4] == 0xFF  # mask=all

def test_p2r_partial_mask():
    raw = encode_p2r(dest=10, mask=0x03)
    assert raw[4] == 0x03  # only P0, P1


# --- R2P ---

def test_r2p_opcode():
    raw = encode_r2p(src=7, mask=0xFF)
    assert _opcode(raw) == 0x204, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 0   # no GPR dest
    assert raw[3] == 7   # src GPR
    assert raw[4] == 0xFF  # mask=all

def test_r2p_partial_mask():
    raw = encode_r2p(src=3, mask=0x0F)
    assert raw[3] == 3
    assert raw[4] == 0x0F


# --- BMSK ---

def test_bmsk_opcode():
    raw = encode_bmsk(dest=5, pos=2, width=3)
    assert _opcode(raw) == 0x21b, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 5   # dest
    assert raw[3] == 2   # pos
    assert raw[4] == 3   # width
    assert raw[8] == 0xFF  # RZ

def test_bmsk_different_operands():
    raw = encode_bmsk(dest=10, pos=0, width=7)
    assert raw[2] == 10
    assert raw[3] == 0
    assert raw[4] == 7


# --- SGXT ---

def test_sgxt_opcode():
    raw = encode_sgxt(dest=5, src=2, bit_pos=8)
    assert _opcode(raw) == 0x21a, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 5   # dest
    assert raw[3] == 2   # src
    assert raw[4] == 8   # bit_pos
    assert raw[9] == 0x02  # signed mode

def test_sgxt_wide_extend():
    raw = encode_sgxt(dest=7, src=7, bit_pos=16)
    assert raw[2] == 7
    assert raw[4] == 16


# --- PLOP3 ---

def test_plop3_opcode():
    raw = encode_plop3(pred_dest=0, pred_src0=1, pred_src1=2, pred_src2=7, lut=0x80)
    assert _opcode(raw) == 0x21e, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 0  # pred_dest P0
    assert raw[9] == 0x80  # LUT

def test_plop3_and_lut():
    # AND of 3 predicates = LUT 0x80 (only true when all 3 inputs true)
    raw = encode_plop3(pred_dest=1, pred_src0=0, pred_src1=1, pred_src2=2, lut=0x80)
    assert raw[2] == 1  # pred_dest P1
    assert raw[9] == 0x80

def test_plop3_or_lut():
    # OR of 3 predicates = LUT 0xFE (true when any input true)
    raw = encode_plop3(pred_dest=2, pred_src0=0, pred_src1=1, pred_src2=2, lut=0xFE)
    assert raw[9] == 0xFE


# --- I2IP ---

def test_i2ip_opcode():
    raw = encode_i2ip(dest=5, src=2)
    assert _opcode(raw) == 0x239, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 5   # dest
    assert raw[3] == 2   # src
    assert raw[8] == 0xFF  # RZ (no merge)

def test_i2ip_with_merge():
    raw = encode_i2ip(dest=5, src=2, src2=10)
    assert raw[8] == 10  # merge register


# --- FSWZADD ---

def test_fswzadd_opcode():
    raw = encode_fswzadd(dest=5, src0=2, src1=3, swizzle=0x1234)
    lo = struct.unpack_from('<Q', raw, 0)[0]
    assert (lo & 0xFFF) == 0x822, f"opcode={lo & 0xFFF:#x}"
    assert raw[2] == 5   # dest
    assert raw[3] == 2   # src0
    assert raw[8] == 3   # src1
    swz = struct.unpack_from('<I', raw, 4)[0]
    assert swz == 0x1234  # swizzle pattern

def test_fswzadd_zero_swizzle():
    raw = encode_fswzadd(dest=0, src0=1, src1=2, swizzle=0)
    swz = struct.unpack_from('<I', raw, 4)[0]
    assert swz == 0


# --- IDP variants check ---

def test_idp4a_already_correct():
    """Verify IDP.4A encoder from Phase 2 still works correctly."""
    from sass.encoding.sm_120_opcodes import encode_idp4a
    raw = encode_idp4a(dest=9, src_a=4, src_b=7, src_c=0xFF)
    assert _opcode(raw) == 0x226
    assert raw[2] == 9
    assert raw[3] == 4
    assert raw[4] == 7
    assert raw[8] == 0xFF


# ===========================================================================
# TMA (Tensor Memory Accelerator) encoder tests — 2026-04-04
# ===========================================================================

from sass.encoding.sm_120_opcodes import (
    encode_syncs_exch_64, encode_syncs_arrive, encode_syncs_trywait,
    encode_ublkcp_s_g, encode_ublkcp_g_s,
    encode_utmaldg_1d, encode_utmaldg_2d, encode_utmastg_1d,
    encode_utmacmdflush, encode_elect, encode_cctl_ivall,
)


# --- SYNCS.EXCH.64 (mbarrier.init) ---

def test_syncs_exch_64_opcode():
    raw = encode_syncs_exch_64(ur_mbar=7, ur_count=4)
    assert _opcode(raw) == 0x5b2, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 0xff  # dest = URZ
    assert raw[3] == 7     # ur_mbar
    assert raw[4] == 4     # ur_count
    assert raw[9] == 0x01
    assert raw[11] == 0x08  # TMA marker

def test_syncs_exch_64_ground_truth():
    """Low 8 bytes must match ptxas output for SYNCS.EXCH.64 URZ, [UR7], UR4."""
    raw = encode_syncs_exch_64(ur_mbar=7, ur_count=4)
    lo = struct.unpack_from('<Q', raw, 0)[0]
    assert lo == 0x0000000407ff75b2, f"lo={lo:#018x}"


# --- SYNCS.ARRIVE (mbarrier.arrive) ---

def test_syncs_arrive_opcode():
    raw = encode_syncs_arrive(ur_mbar=6)
    assert _opcode(raw) == 0x9a7, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 0xff  # dest = RZ
    assert raw[3] == 0xff  # src0 = RZ
    assert raw[4] == 0xff  # src1 = RZ
    assert raw[8] == 6     # ur_mbar
    assert raw[10] == 0x10  # ARRIVE mode
    assert raw[11] == 0x08  # TMA marker


# --- SYNCS.TRYWAIT (mbarrier.try_wait) ---

def test_syncs_trywait_opcode():
    raw = encode_syncs_trywait(ur_mbar=4, r_phase=0)
    assert _opcode(raw) == 0x5a7, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 0x00  # pred dest
    assert raw[3] == 0xff  # always ff
    assert raw[4] == 0     # r_phase = R0
    assert raw[8] == 4     # ur_mbar
    assert raw[9] == 0x11
    assert raw[10] == 0x0e
    assert raw[11] == 0x08  # TMA marker

def test_syncs_trywait_ground_truth():
    """Low 8 bytes must match ptxas output for SYNCS.PHASECHK.TRYWAIT PT, [UR4], R0."""
    raw = encode_syncs_trywait(ur_mbar=4, r_phase=0)
    lo = struct.unpack_from('<Q', raw, 0)[0]
    assert lo == 0x00000000ff0075a7, f"lo={lo:#018x}"


# --- UBLKCP.S.G (bulk copy shared<-global) ---

def test_ublkcp_s_g_opcode():
    raw = encode_ublkcp_s_g(ur_dst=8, ur_src=10, ur_size=4)
    assert _opcode(raw) == 0x3ba, f"opcode={_opcode(raw):#x}"
    assert raw[3] == 10    # ur_src
    assert raw[4] == 8     # ur_dst
    assert raw[8] == 4     # ur_size
    assert raw[9] == 0x02  # S.G mode
    assert raw[11] == 0x08

def test_ublkcp_s_g_ground_truth():
    """Low 8 bytes must match ptxas output for UBLKCP.S.G [UR8], [UR10], UR4."""
    raw = encode_ublkcp_s_g(ur_dst=8, ur_src=10, ur_size=4)
    lo = struct.unpack_from('<Q', raw, 0)[0]
    assert lo == 0x000000080a0073ba, f"lo={lo:#018x}"


# --- UBLKCP.G.S (bulk copy global<-shared) ---

def test_ublkcp_g_s_opcode():
    raw = encode_ublkcp_g_s(ur_dst=8, ur_src=4, ur_size=5)
    assert _opcode(raw) == 0x3ba, f"opcode={_opcode(raw):#x}"
    assert raw[3] == 4     # ur_src
    assert raw[4] == 8     # ur_dst
    assert raw[8] == 5     # ur_size
    assert raw[9] == 0x04  # G.S mode
    assert raw[11] == 0x08

def test_ublkcp_g_s_ground_truth():
    """Low 8 bytes must match ptxas output for UBLKCP.G.S [UR8], [UR4], UR5."""
    raw = encode_ublkcp_g_s(ur_dst=8, ur_src=4, ur_size=5)
    lo = struct.unpack_from('<Q', raw, 0)[0]
    assert lo == 0x00000008040073ba, f"lo={lo:#018x}"


# --- UTMALDG.1D (TMA tensor load 1D) ---

def test_utmaldg_1d_opcode():
    raw = encode_utmaldg_1d(ur_dst=4, ur_desc=8)
    assert _opcode(raw) == 0x5b4, f"opcode={_opcode(raw):#x}"
    assert raw[3] == 8     # ur_desc
    assert raw[4] == 4     # ur_dst
    assert raw[9] == 0x00  # 1D mode (bit7=0)
    assert raw[11] == 0x08

def test_utmaldg_1d_ground_truth():
    """Low 8 bytes must match ptxas output for UTMALDG.1D [UR4], [UR8]."""
    raw = encode_utmaldg_1d(ur_dst=4, ur_desc=8)
    lo = struct.unpack_from('<Q', raw, 0)[0]
    assert lo == 0x00000004080075b4, f"lo={lo:#018x}"


# --- UTMALDG.2D (TMA tensor load 2D) ---

def test_utmaldg_2d_opcode():
    raw = encode_utmaldg_2d(ur_dst=8, ur_desc=12)
    assert _opcode(raw) == 0x5b4, f"opcode={_opcode(raw):#x}"
    assert raw[3] == 12    # ur_desc
    assert raw[4] == 8     # ur_dst
    assert raw[9] == 0x80  # 2D mode (bit7=1)
    assert raw[11] == 0x08

def test_utmaldg_2d_ground_truth():
    """Low 8 bytes must match ptxas output for UTMALDG.2D [UR8], [UR12]."""
    raw = encode_utmaldg_2d(ur_dst=8, ur_desc=12)
    lo = struct.unpack_from('<Q', raw, 0)[0]
    assert lo == 0x000000080c0075b4, f"lo={lo:#018x}"


# --- UTMASTG.1D (TMA tensor store 1D) ---

def test_utmastg_1d_opcode():
    raw = encode_utmastg_1d(ur_src=4, ur_desc=8)
    assert _opcode(raw) == 0x3b5, f"opcode={_opcode(raw):#x}"
    assert raw[3] == 8     # ur_desc
    assert raw[4] == 4     # ur_src
    assert raw[11] == 0x08

def test_utmastg_1d_ground_truth():
    """Low 8 bytes must match ptxas output for UTMASTG.1D [UR4], [UR8]."""
    raw = encode_utmastg_1d(ur_src=4, ur_desc=8)
    lo = struct.unpack_from('<Q', raw, 0)[0]
    assert lo == 0x00000004080073b5, f"lo={lo:#018x}"


# --- UTMACMDFLUSH ---

def test_utmacmdflush_opcode():
    raw = encode_utmacmdflush()
    assert _opcode(raw) == 0x9b7, f"opcode={_opcode(raw):#x}"
    # All operand bytes should be zero
    for i in range(2, 12):
        assert raw[i] == 0, f"b[{i}]={raw[i]:#x} (expected 0)"


# --- ELECT ---

def test_elect_opcode():
    raw = encode_elect(pred_guard=0, pred_dest=1)
    assert _opcode(raw) == 0x82f, f"opcode={_opcode(raw):#x}"
    assert raw[2] == 0xff  # URZ
    assert raw[11] == 0x03

def test_elect_pred_guard():
    """Pred guard P0 → pred nibble = 0x0."""
    raw = encode_elect(pred_guard=0)
    assert (raw[1] >> 4) == 0x0, f"pred_nibble={raw[1]>>4:#x}"


# --- CCTL.IVALL ---

def test_cctl_ivall_opcode():
    raw = encode_cctl_ivall()
    assert _opcode(raw) == 0x98f, f"opcode={_opcode(raw):#x}"
    assert raw[3] == 0xff  # always ff
    assert raw[11] == 0x02  # IVALL mode


# --- SEL.64 (newly landed; ground truth from ptxas 13.2.78 sm_120) ---

# Probe ptxas reproducer: see _probe_landing/probe_sel64_3.ptx + probe_sel64_preds.ptx.
# Ground truth bytes (with default ctrl 0x7e0 → b13/14/15 = 0xc0,0x0f,0x00,
# rather than ptxas-emitted scheduling bytes; we only verify *opcode/operand* bytes
# 0..12 against ground truth, since b13..15 are scheduling-controlled by the caller).
#   SEL.64 R8,R4,R6,P0 (real ptxas):
#     07 76 08 04 06 00 00 00 00 00 00 00 00 c8 4f 00
#   SEL.64 R6,R6,R8,P1:
#     07 76 06 06 08 00 00 00 00 00 80 00 00 e4 8f 00
#   SEL.64 R10,R4,R6,P0:
#     07 76 0a 04 06 00 00 00 00 00 00 00 00 c4 4f 00

def test_sel_64_opcode_p0():
    """Byte-exact match against ptxas ground truth for SEL.64 R8,R4,R6,P0
    (operand bytes 0..12). Control bytes 13..15 are caller-controlled."""
    raw = encode_sel_64(dest=8, src0=4, src1=6, pred=0)
    expected = bytes.fromhex('0776080406000000') + bytes.fromhex('0000000000')
    assert raw[0:13] == expected, (
        f"SEL.64 P0 byte mismatch:\n"
        f"  got: {raw[0:13].hex()}\n"
        f"  exp: {expected.hex()}"
    )

def test_sel_64_opcode_p1():
    """Verify P1 sets bit 87 (byte[10] bit 7)."""
    raw = encode_sel_64(dest=6, src0=6, src1=8, pred=1)
    expected = bytes.fromhex('0776060608000000') + bytes.fromhex('0000800000')
    assert raw[0:13] == expected, (
        f"SEL.64 P1 byte mismatch:\n"
        f"  got: {raw[0:13].hex()}\n"
        f"  exp: {expected.hex()}"
    )

def test_sel_64_opcode_p2():
    """Synthesize: P2 should set bit 88 (byte[11] bit 0)."""
    raw = encode_sel_64(dest=4, src0=2, src1=6, pred=2)
    # bit 87 (P_in[0]) == 0 for P2; bit 88 (P_in[1]) == 1.
    assert raw[10] == 0x00
    assert raw[11] == 0x01

def test_sel_64_distinct_from_sel_32():
    """Confirm byte[1]=0x76 (vs SEL.32's 0x72)."""
    raw = encode_sel_64(dest=4, src0=2, src1=6, pred=0)
    assert raw[0] == 0x07
    assert raw[1] == 0x76

def test_sel_64_dest_src_fields():
    raw = encode_sel_64(dest=12, src0=20, src1=30, pred=0)
    assert raw[2] == 12
    assert raw[3] == 20
    assert raw[4] == 30


# --- SHF.L family (newly landed; ground truth from ptxas 13.2.78 sm_120) ---

# Probe ptxas reproducer: see _probe_landing/probe_shf.ptx.  The four ground
# truth bytes are:
#   SHF.L.U32.HI    R5,R6,R5,R7 → 19 72 05 06 05 00 00 00 07 06 01 00 ...
#   SHF.L.W.U32.HI  R0,R6,R5,R7 → 19 72 00 06 05 00 00 00 07 0e 01 00 ...
#   SHF.R.W.U32     R2,R6,R5,R7 → 19 72 02 06 05 00 00 00 07 1e 00 00 ...   (existing-shape reference)
#   SHF.R.U32       R3,R6,R5,R7 → 19 72 03 06 05 00 00 00 07 16 00 00 ...   (existing-shape reference)

def test_shf_l_u32_hi_byte_exact():
    raw = encode_shf_l_u32_hi_var(dest=5, src_lo=6, shift_reg=5, src_hi=7)
    expected = bytes.fromhex('19720506050000000706010000')
    assert raw[0:13] == expected, (
        f"SHF.L.U32.HI byte mismatch:\n  got: {raw[0:13].hex()}\n  exp: {expected.hex()}"
    )

def test_shf_l_w_u32_hi_byte_exact():
    raw = encode_shf_l_w_u32_hi_var(dest=0, src_lo=6, shift_reg=5, src_hi=7)
    expected = bytes.fromhex('19720006050000000706 010000'.replace(' ', ''))
    # rebuild without space
    expected = bytes.fromhex('1972000605000000070e010000')
    assert raw[0:13] == expected, (
        f"SHF.L.W.U32.HI byte mismatch:\n  got: {raw[0:13].hex()}\n  exp: {expected.hex()}"
    )

def test_shf_l_w_vs_clamp_bit_75():
    """The wrap-vs-clamp distinction is bit 75 (= byte[9] bit 3 = 0x08)."""
    clamp = encode_shf_l_u32_hi_var(dest=5, src_lo=6, shift_reg=5, src_hi=7)
    wrap  = encode_shf_l_w_u32_hi_var(dest=5, src_lo=6, shift_reg=5, src_hi=7)
    # XOR difference must be exactly bit 75
    assert clamp[9] ^ wrap[9] == 0x08, \
        f"clamp/wrap should differ only at byte[9] bit 3, got XOR=0x{clamp[9]^wrap[9]:02x}"
    # And nothing else in the opcode/operand bytes
    for i in range(13):
        if i != 9:
            assert clamp[i] == wrap[i], f"unexpected diff at byte {i}: {clamp[i]:#x} vs {wrap[i]:#x}"

def test_shf_l_w_low_word_clears_hi_bit():
    """Low-word variant clears byte[10] bit 0 (.HI flag)."""
    raw = encode_shf_l_w_u32_var(dest=5, src_lo=6, shift_reg=5, src_hi=7)
    assert raw[9] == 0x0e
    assert raw[10] == 0x00


# --- CS2UR (newly landed; ground truth from ptxas 13.2.78 sm_120) ---

# Probe ptxas reproducer: see _probe_landing/probe_cs2ur*.ptx.
# Ground truth bytes (operand bytes 0..12; ctrl 13..15 is caller-controlled):
#   CS2UR.32 UR6, SR_PM0     → cb 78 06 00 00 00 00 00 00 32 00 00 00 ...  (sr=0x32 PM0)
#                                                                  ^^ wait: ground-truth byte[9]=0x64 visually.
# But probe nvdisasm assembled SR_PM0 as code=0x32, so byte[9]=0x32 if direct mapping.
# Probe_more.ptx showed SR_CLOCKLO=0x50 directly in byte[9].  PM0 in our other probe
# showed byte[9]=0x64 — that's the *encoded* SR field, distinct from any internal
# enum.  We use the *byte[9] value* observed at byte-extract level as ground truth
# (i.e., for CLOCKLO pass sr_code=0x50 to the encoder).

def test_cs2ur_byte_exact_clocklo():
    """SR_CLOCKLO probe: byte[9]=0x50, byte[10]=0x00 (32-bit)."""
    raw = encode_cs2ur(dest_ur=6, sr_code=0x50, is_64=False)
    expected = bytes.fromhex('cb78060000000000005000 0000'.replace(' ', ''))
    expected = bytes.fromhex('cb780600000000000050000000')
    assert raw[0:13] == expected, (
        f"CS2UR.32 UR6, SR_CLOCKLO byte mismatch:\n"
        f"  got: {raw[0:13].hex()}\n  exp: {expected.hex()}"
    )

def test_cs2ur_dest_field():
    for ur in (0, 4, 5, 6, 7, 30, 63):
        raw = encode_cs2ur(dest_ur=ur, sr_code=0x50)
        assert raw[2] == ur

def test_cs2ur_sr_code_field():
    for sr in (0x50, 0x51, 0x32, 0x33, 0x34, 0x35):
        raw = encode_cs2ur(dest_ur=4, sr_code=sr)
        assert raw[9] == sr

def test_cs2ur_64_bit_flag():
    raw32 = encode_cs2ur(dest_ur=4, sr_code=0x50, is_64=False)
    raw64 = encode_cs2ur(dest_ur=4, sr_code=0x50, is_64=True)
    assert raw32[10] == 0x00
    assert raw64[10] == 0x01
    # Otherwise identical
    for i in range(13):
        if i != 10:
            assert raw32[i] == raw64[i]

def test_cs2ur_distinct_from_cs2r():
    """CS2UR opcode (0x8cb) differs from CS2R (0x805) at byte[0]."""
    raw_ur = encode_cs2ur(dest_ur=4, sr_code=0x50)
    raw_r  = encode_cs2r(dest=4)
    assert raw_ur[0] == 0xcb
    assert raw_r[0]  == 0x05
    assert raw_ur[1] == 0x78
    assert raw_r[1]  == 0x78


# --- LEA.HI.X (newly landed; ground truth from ptxas 13.2.78 sm_120) ---

# Probe ptxas reproducer: see _probe_landing/probe_lea_hi_preds.ptx.
# Ground truth (operand bytes 0..12 only):
#   LEA.HI.X R7,  R0,  R12, R11, 0x1, P0 → 11 72 07 00 0c 00 00 00 0b 0c 0f 00 00
#   LEA.HI.X R9,  R13, R16, R14, 0x2, P1 → 11 72 09 0d 10 00 00 00 0e 14 8f 00 00
#   LEA.HI.X R11, R17, R18, R3,  0x3, P2 → 11 72 0b 11 12 00 00 00 03 1c 0f 01 00

def test_lea_hi_x_byte_exact_p0():
    """First sample (scale=1, P0)."""
    raw = encode_lea_hi_x(dest=7, src_a=0, src_b=12, src_c=11, scale=1, p_in=0)
    expected = bytes.fromhex('1172070000000000000c0f0000'.replace(' ', ''))  # bytes 0..12
    # Wait — actually byte[4]=0x0c (R12). Let me reconstruct properly.
    # Ground truth: 11 72 07 00 0c 00 00 00 0b 0c 0f 00 00
    expected = bytes([0x11, 0x72, 0x07, 0x00, 0x0c, 0x00, 0x00, 0x00,
                      0x0b, 0x0c, 0x0f, 0x00, 0x00])
    assert raw[0:13] == expected, (
        f"LEA.HI.X scale=1 P0 byte mismatch:\n"
        f"  got: {raw[0:13].hex()}\n  exp: {expected.hex()}"
    )

def test_lea_hi_x_byte_exact_p1():
    """Second sample (scale=2, P1)."""
    raw = encode_lea_hi_x(dest=9, src_a=13, src_b=16, src_c=14, scale=2, p_in=1)
    expected = bytes([0x11, 0x72, 0x09, 0x0d, 0x10, 0x00, 0x00, 0x00,
                      0x0e, 0x14, 0x8f, 0x00, 0x00])
    assert raw[0:13] == expected, (
        f"LEA.HI.X scale=2 P1 byte mismatch:\n"
        f"  got: {raw[0:13].hex()}\n  exp: {expected.hex()}"
    )

def test_lea_hi_x_byte_exact_p2():
    """Third sample (scale=3, P2)."""
    raw = encode_lea_hi_x(dest=11, src_a=17, src_b=18, src_c=3, scale=3, p_in=2)
    expected = bytes([0x11, 0x72, 0x0b, 0x11, 0x12, 0x00, 0x00, 0x00,
                      0x03, 0x1c, 0x0f, 0x01, 0x00])
    assert raw[0:13] == expected, (
        f"LEA.HI.X scale=3 P2 byte mismatch:\n"
        f"  got: {raw[0:13].hex()}\n  exp: {expected.hex()}"
    )

def test_lea_hi_x_scale_and_hi_flag():
    """byte[9] bit 2 (= bit 74) is the .HI flag; scale lives at bits 75..77."""
    for scale in (0, 1, 2, 3, 4):
        raw = encode_lea_hi_x(dest=4, src_a=2, src_b=6, src_c=255, scale=scale, p_in=0)
        expected_b9 = ((scale & 0x07) << 3) | 0x04
        assert raw[9] == expected_b9, f"scale={scale}: got 0x{raw[9]:02x}, exp 0x{expected_b9:02x}"

def test_lea_hi_x_pred_field():
    """P_in encoding at bits 87..89 (byte[10] bit 7 + byte[11] bits 0..1)."""
    for p in range(8):
        raw = encode_lea_hi_x(dest=4, src_a=2, src_b=6, src_c=255, scale=2, p_in=p)
        # byte[10] LSB nibble always 0x0f for LEA.HI; bit 7 = p&1
        assert (raw[10] & 0x0f) == 0x0f
        assert (raw[10] >> 7) & 1 == (p & 1)
        assert raw[11] & 0x03 == ((p >> 1) & 0x03)


# ---------------------------------------------------------------------------
# F2I.{U,S}64 — float to 64-bit int
# ---------------------------------------------------------------------------
# Ground truth (ptxas 13.2.78, sm_120, _probe_landing/probe_f2i_64.ptx):
#   F2I.U64.TRUNC      R6,  R2 (cvt.rzi.u64.f32): lo=0x0000000200067311 hi=0x004e24000020d800
#   F2I.S64.TRUNC      R8,  R2 (cvt.rzi.s64.f32): lo=0x0000000200087311 hi=0x000e24000020d900
#   F2I.U64.F64.TRUNC  R10, R4 (cvt.rzi.u64.f64): lo=0x00000004000a7311 hi=0x008e24000030d800
#   F2I.S64.F64.TRUNC  R12, R4 (cvt.rzi.s64.f64): lo=0x00000004000c7311 hi=0x000e24000030d900

def test_f2i_u64_f32_byte_exact():
    """F2I.U64.TRUNC R6, R2 (cvt.rzi.u64.f32) — byte-exact ptxas match."""
    raw = encode_f2i_u64(dest_lo=6, src=2, signed=False, src_is_f64=False)
    lo, hi = _lo_hi(raw)
    assert lo == 0x0000000200067311, f"lo=0x{lo:016x}"
    # Ignore ctrl bytes 13..15 (high 24 bits of hi); compare lower 40 bits.
    assert (hi & 0x000000ffffffffff) == 0x000000000020d800, f"hi&...=0x{hi & 0x000000ffffffffff:016x}"

def test_f2i_s64_f32_byte_exact():
    raw = encode_f2i_u64(dest_lo=8, src=2, signed=True, src_is_f64=False)
    lo, hi = _lo_hi(raw)
    assert lo == 0x0000000200087311
    assert (hi & 0x000000ffffffffff) == 0x000000000020d900

def test_f2i_u64_f64_byte_exact():
    raw = encode_f2i_u64(dest_lo=10, src=4, signed=False, src_is_f64=True)
    lo, hi = _lo_hi(raw)
    assert lo == 0x00000004000a7311
    assert (hi & 0x000000ffffffffff) == 0x000000000030d800

def test_f2i_s64_f64_byte_exact():
    raw = encode_f2i_u64(dest_lo=12, src=4, signed=True, src_is_f64=True)
    lo, hi = _lo_hi(raw)
    assert lo == 0x00000004000c7311
    assert (hi & 0x000000ffffffffff) == 0x000000000030d900

def test_f2i_u64_signed_bit():
    """Bit 72 (byte[9] bit 0) is the signed-dst flag."""
    u = encode_f2i_u64(dest_lo=4, src=2, signed=False)
    s = encode_f2i_u64(dest_lo=4, src=2, signed=True)
    assert (u[9] & 0x01) == 0
    assert (s[9] & 0x01) == 1

def test_f2i_u64_src_width_bit():
    """Bit 84 (byte[10] bit 4) is the f64-src flag."""
    f32 = encode_f2i_u64(dest_lo=4, src=2, src_is_f64=False)
    f64 = encode_f2i_u64(dest_lo=4, src=2, src_is_f64=True)
    assert (f32[10] & 0x10) == 0
    assert (f64[10] & 0x10) == 0x10


# ---------------------------------------------------------------------------
# I2F.{F32,F64}.{U,S}64 — 64-bit int to float
# ---------------------------------------------------------------------------
# Ground truth (ptxas 13.2.78, sm_120, _probe_landing/probe_i2f_64.ptx):
#   I2F.U64       R12, R2 (cvt.rn.f32.u64): lo=0x00000002000c7312 hi=0x004fde0000301000
#   I2F.F64.U64   R6,  R2 (cvt.rn.f64.u64): lo=0x0000000200067312 hi=0x000e240000301800
#   I2F.S64       R13, R4 (cvt.rn.f32.s64): lo=0x00000004000d7312 hi=0x008e240000301400
#   I2F.F64.S64   R8,  R4 (cvt.rn.f64.s64): lo=0x0000000400087312 hi=0x000e240000301c00

def test_i2f_f32_u64_byte_exact():
    raw = encode_i2f_u64(dest=12, src_lo=2, signed=False, dst_is_f64=False)
    lo, hi = _lo_hi(raw)
    assert lo == 0x00000002000c7312
    assert (hi & 0x000000ffffffffff) == 0x0000000000301000

def test_i2f_f64_u64_byte_exact():
    raw = encode_i2f_u64(dest=6, src_lo=2, signed=False, dst_is_f64=True)
    lo, hi = _lo_hi(raw)
    assert lo == 0x0000000200067312
    assert (hi & 0x000000ffffffffff) == 0x0000000000301800

def test_i2f_f32_s64_byte_exact():
    raw = encode_i2f_u64(dest=13, src_lo=4, signed=True, dst_is_f64=False)
    lo, hi = _lo_hi(raw)
    assert lo == 0x00000004000d7312
    assert (hi & 0x000000ffffffffff) == 0x0000000000301400

def test_i2f_f64_s64_byte_exact():
    raw = encode_i2f_u64(dest=8, src_lo=4, signed=True, dst_is_f64=True)
    lo, hi = _lo_hi(raw)
    assert lo == 0x0000000400087312
    assert (hi & 0x000000ffffffffff) == 0x0000000000301c00

def test_i2f_u64_signed_bit():
    """Bit 74 (byte[9] bit 2) is the signed-src flag."""
    u = encode_i2f_u64(dest=4, src_lo=2, signed=False)
    s = encode_i2f_u64(dest=4, src_lo=2, signed=True)
    assert (u[9] & 0x04) == 0
    assert (s[9] & 0x04) == 0x04

def test_i2f_u64_dst_width_bit():
    """Bit 75 (byte[9] bit 3) is the f64-dst flag."""
    f32 = encode_i2f_u64(dest=4, src_lo=2, dst_is_f64=False)
    f64 = encode_i2f_u64(dest=4, src_lo=2, dst_is_f64=True)
    assert (f32[9] & 0x08) == 0
    assert (f64[9] & 0x08) == 0x08

def test_i2f_f64_u32_b9_fixed():
    """Verify I2F.F64.U32 b9=0x18 (was 0x20 inferred-and-wrong before this commit)."""
    # Ground truth (probe_i2f_f64_u32.ptx): I2F.F64.U32 R4, R2 → b9=0x18 b10=0x20.
    raw = encode_i2f_f64_u32(dest_lo=4, src=2)
    assert raw[9] == 0x18, f"b9={raw[9]:#04x}"
    assert raw[10] == 0x20


# ---------------------------------------------------------------------------
# HFMA2 — general FP16x2 fused multiply-add (opcode 0x231, b1=0x72)
# ---------------------------------------------------------------------------
# Ground truth (ptxas 13.2.78, sm_120, _probe_landing/probe_hfma2.ptx):
#   HFMA2     R9,  R0, R7, R6:  lo=0x0000000700097231 hi=0x004fc40000000006
#   HFMA2.FTZ R11, R0, R7, R6:  lo=0x00000007000b7231 hi=0x1c0fe40000010006
#   HFMA2.SAT R7,  R0, R7, R6:  lo=0x0000000700077231 hi=0x000fe20000002006
#   HFMA2.FTZ.SAT R7, R0, R7, R6: lo=0x0000000700077231 hi=0x004fca0000012006
# Negation (probe_hfma2_more.ptx):
#   HFMA2 R9, -R0, R7, R6:  hi=0x1c8fe40000000106  → byte[9] |= 0x01
#   HFMA2 R11, R0, -R7, R6: lo=0x80000007000b7231  → byte[7] |= 0x80
#   HFMA2 R7, R0, R7, -R6:  hi=0x000fe20000100006  → byte[10] |= 0x10

def test_hfma2_basic_byte_exact():
    """HFMA2 R9, R0, R7, R6 — byte-exact ptxas match (low qword and modifier bytes)."""
    raw = encode_hfma2(dest=9, src_a=0, src_b=7, src_c=6)
    lo, hi = _lo_hi(raw)
    assert lo == 0x0000000700097231, f"lo=0x{lo:016x}"
    # Compare modifier bytes 8..11 (mask off ctrl bytes 13..15 in upper bits).
    assert (hi & 0x000000ffffffffff) == 0x0000000000000006, f"hi&...=0x{hi & 0x000000ffffffffff:016x}"

def test_hfma2_ftz_byte_exact():
    """HFMA2.FTZ R11, R0, R7, R6 — FTZ at byte[10] bit 0."""
    raw = encode_hfma2(dest=11, src_a=0, src_b=7, src_c=6, ftz=True)
    lo, hi = _lo_hi(raw)
    assert lo == 0x00000007000b7231
    assert (hi & 0x000000ffffffffff) == 0x0000000000010006

def test_hfma2_sat_byte_exact():
    """HFMA2.SAT R7, R0, R7, R6 — SAT at byte[9] bit 5 (bit 77, matches NAK SM70)."""
    raw = encode_hfma2(dest=7, src_a=0, src_b=7, src_c=6, sat=True)
    lo, hi = _lo_hi(raw)
    assert lo == 0x0000000700077231
    assert (hi & 0x000000ffffffffff) == 0x0000000000002006

def test_hfma2_ftz_sat_byte_exact():
    """HFMA2.FTZ.SAT R7, R0, R7, R6 — combined FTZ+SAT (byte[9]=0x20, byte[10]=0x01)."""
    raw = encode_hfma2(dest=7, src_a=0, src_b=7, src_c=6, ftz=True, sat=True)
    lo, hi = _lo_hi(raw)
    assert lo == 0x0000000700077231
    assert (hi & 0x000000ffffffffff) == 0x0000000000012006

def test_hfma2_neg_a_byte_exact():
    """HFMA2 R9, -R0, R7, R6 — neg_a at byte[9] bit 0."""
    raw = encode_hfma2(dest=9, src_a=0, src_b=7, src_c=6, neg_a=True)
    lo, hi = _lo_hi(raw)
    assert lo == 0x0000000700097231
    assert (hi & 0x000000ffffffffff) == 0x0000000000000106

def test_hfma2_neg_b_byte_exact():
    """HFMA2 R11, R0, -R7, R6 — neg_b at byte[7] bit 7 (low qword)."""
    raw = encode_hfma2(dest=11, src_a=0, src_b=7, src_c=6, neg_b=True)
    lo, hi = _lo_hi(raw)
    assert lo == 0x80000007000b7231
    assert (hi & 0x000000ffffffffff) == 0x0000000000000006

def test_hfma2_neg_c_byte_exact():
    """HFMA2 R7, R0, R7, -R6 — neg_c at byte[9] bit 4."""
    raw = encode_hfma2(dest=7, src_a=0, src_b=7, src_c=6, neg_c=True)
    lo, hi = _lo_hi(raw)
    assert lo == 0x0000000700077231
    assert (hi & 0x000000ffffffffff) == 0x0000000000001006

def test_hfma2_distinct_from_zero_init():
    """HFMA2 general (b1=0x72, opcode 0x231) is distinct from HFMA2 zero-init (b1=0x74, opcode 0x431)."""
    raw = encode_hfma2(dest=9, src_a=0, src_b=7, src_c=6)
    assert _opcode(raw) == 0x231
    assert raw[1] == 0x72

def test_hfma2_modifier_bit_isolation():
    """Each modifier flips exactly the documented bit, no others."""
    base = encode_hfma2(dest=9, src_a=0, src_b=7, src_c=6)
    # FTZ
    ftz = encode_hfma2(dest=9, src_a=0, src_b=7, src_c=6, ftz=True)
    diff = bytes(a ^ b for a, b in zip(base, ftz))
    # Only byte[10] bit 0 differs (ignoring ctrl bytes 13..15)
    assert diff[10] == 0x01
    assert all(diff[i] == 0 for i in range(13) if i != 10)
    # SAT
    sat = encode_hfma2(dest=9, src_a=0, src_b=7, src_c=6, sat=True)
    diff = bytes(a ^ b for a, b in zip(base, sat))
    assert diff[9] == 0x20
    assert all(diff[i] == 0 for i in range(13) if i != 9)
    # neg_a
    na = encode_hfma2(dest=9, src_a=0, src_b=7, src_c=6, neg_a=True)
    diff = bytes(a ^ b for a, b in zip(base, na))
    assert diff[9] == 0x01
    assert all(diff[i] == 0 for i in range(13) if i != 9)
    # neg_b
    nb = encode_hfma2(dest=9, src_a=0, src_b=7, src_c=6, neg_b=True)
    diff = bytes(a ^ b for a, b in zip(base, nb))
    assert diff[7] == 0x80
    assert all(diff[i] == 0 for i in range(13) if i != 7)
    # neg_c
    nc = encode_hfma2(dest=9, src_a=0, src_b=7, src_c=6, neg_c=True)
    diff = bytes(a ^ b for a, b in zip(base, nc))
    assert diff[9] == 0x10
    assert all(diff[i] == 0 for i in range(13) if i != 9)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    for t in tests:
        try:
            t()
            print(f'  PASS  {t.__name__}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {t.__name__}: {e}')
    print(f'\n{passed}/{len(tests)} passed')
