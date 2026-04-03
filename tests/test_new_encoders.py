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
