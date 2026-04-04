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
    assert raw[4] == 4  # UR descriptor

def test_ldgdepbar_opcode():
    raw = encode_ldgdepbar()
    assert _opcode(raw) == 0x9af, f"opcode={_opcode(raw):#x}"

def test_depbar_le_opcode():
    raw = encode_depbar_le(sb=0, count=0)
    assert _opcode(raw) == 0x91a, f"opcode={_opcode(raw):#x}"
    assert raw[4] == 0x80  # SB0, count=0

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
