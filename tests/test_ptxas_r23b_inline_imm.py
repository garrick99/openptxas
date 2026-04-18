"""PTXAS-R23B.A regression — u32 immediate store-payloads must be
materialized inline via MOV32I (opcode 0x431), not via a literal-pool
LDC into constant bank 0.

R23A.4 proof established that `.nv.constant0.*` bytes past the declared
param area are not driver-visible on SM_120: LDC from such offsets
returns zero, breaking any `st.global.u32 [addr], imm` lowered through
the old `ctx._alloc_literal` + `encode_ldc` path.  NVCC's own cubin for
the same shape materializes the immediate inline with opcode 0x431
(`MOV32I R, imm32`), embedding the 32-bit value in bytes 4-7 of the
16-byte SASS instruction.  These tests prove:

  1. No LDC in the emitted text reads from beyond the raw param area
     (the literal-pool region inside c[0]).
  2. A MOV32I with the expected imm32 appears in the text.
  3. The immediate bytes are bit-exact in b4..b7 of that MOV32I.
  4. No c[0][lit_pool_offset] LDC remains in the immediate-store path.
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.parser import parse
from sass.pipeline import compile_function


_OPC_LDC     = 0xb82
_OPC_MOV32I  = 0x431   # opcode 0x431 with b4..b7 = imm32, b3=b8=0xff
_PARAM_BASE  = 0x380
_PARAM_AREA_END_G4 = _PARAM_BASE + 8    # single u64 param
_PARAM_AREA_END_G8 = _PARAM_BASE + 16   # two u64 params


def _text(cubin: bytes, kernel: str) -> bytes:
    e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]

    def sh(i):
        return struct.unpack_from('<IIQQQQIIQQ', cubin, e_shoff + i * 64)

    _, _, _, _, so, ss, *_ = sh(e_shstrndx)
    shs = cubin[so:so + ss]
    target = f'.text.{kernel}'.encode()
    for i in range(e_shnum):
        nm, ty, _, _, off, sz, *_ = sh(i)
        end = shs.index(b'\x00', nm)
        if shs[nm:end] == target and ty == 1:
            return cubin[off:off + sz]
    raise AssertionError(f'no .text.{kernel}')


def _compile(ptx_src: str) -> bytes:
    return compile_function(parse(ptx_src).functions[0],
                            verbose=False, sm_version=120)


def _find_mov32i_with_imm(text: bytes, imm32: int):
    """Scan the text section for MOV32I (opcode 0x431, b3=b8=0xff) with
    the given 32-bit immediate packed into b4..b7.  Returns the byte
    offset of the first match, or None."""
    target = struct.pack('<I', imm32 & 0xFFFFFFFF)
    for a in range(0, len(text), 16):
        raw = text[a:a + 16]
        opc = (raw[0] | (raw[1] << 8)) & 0xFFF
        if opc != _OPC_MOV32I:
            continue
        if raw[3] != 0xff or raw[8] != 0xff:
            continue
        if raw[4:8] == target:
            return a
    return None


def _ldc_cbank_offset(raw: bytes) -> int:
    """Decode LDC's c[0][word_offset] into byte offset.  From
    encode_ldc: b5 = dword_offset (bits 0-7).  LDC also uses b4 for the
    bank; we trust caller to have already checked opc==LDC."""
    # For SM_120 LDC, the byte offset = (b5 | ((b6 & 0x03) << 8)) * 4
    word = raw[5] | ((raw[6] & 0x03) << 8)
    return word * 4


_PTX_G4 = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_g4(.param .u64 out)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [out];
    mov.u32 %r0, %tid.x;
    setp.eq.u32 %p0, %r0, 0;
    @!%p0 ret;
    mov.u32 %r1, %ctaid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd2, %r2;
    add.u64 %rd1, %rd0, %rd2;
    st.global.u32 [%rd1], 2864434397;           // 0xAABBCCDD
    ret;
}
"""

_PTX_G8 = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_g8(.param .u64 in, .param .u64 out)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [in];
    ld.param.u64 %rd0, [out];
    mov.u32 %r0, %tid.x;
    setp.eq.u32 %p0, %r0, 0;
    @!%p0 ret;
    mov.u32 %r1, %ctaid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd2, %r2;
    add.u64 %rd1, %rd0, %rd2;
    st.global.u32 [%rd1], 305419896;            // 0x12345678
    ret;
}
"""


# ---------------------------------------------------------------------------
# Test 1 — MOV32I with the exact immediate must appear in the text
# ---------------------------------------------------------------------------

def test_g4_emits_mov32i_with_aabbccdd():
    text = _text(_compile(_PTX_G4), 'k_g4')
    off = _find_mov32i_with_imm(text, 0xAABBCCDD)
    assert off is not None, (
        "G4 text contains no MOV32I (opc 0x431) with imm32 0xAABBCCDD "
        "embedded in b4..b7; PTXAS-R23B.A requires inline materialization.")


def test_g8_emits_mov32i_with_12345678():
    text = _text(_compile(_PTX_G8), 'k_g8')
    off = _find_mov32i_with_imm(text, 0x12345678)
    assert off is not None, (
        "G8 text contains no MOV32I (opc 0x431) with imm32 0x12345678 "
        "embedded in b4..b7; PTXAS-R23B.A requires inline materialization.")


# ---------------------------------------------------------------------------
# Test 2 — no LDC reads from the literal-pool region past the param area
# ---------------------------------------------------------------------------

def test_g4_no_ldc_past_param_area():
    text = _text(_compile(_PTX_G4), 'k_g4')
    for a in range(0, len(text), 16):
        raw = text[a:a + 16]
        opc = (raw[0] | (raw[1] << 8)) & 0xFFF
        if opc != _OPC_LDC:
            continue
        bank = raw[4]
        off = _ldc_cbank_offset(raw)
        assert not (bank == 0 and off >= _PARAM_AREA_END_G4), (
            f"G4: LDC at 0x{a:x} reads c[0][0x{off:x}] which lies in the "
            f"literal-pool region (>= 0x{_PARAM_AREA_END_G4:x}).  R23B.A "
            f"must have eliminated this path for u32 store-payload imms.")


def test_g8_no_ldc_past_param_area():
    text = _text(_compile(_PTX_G8), 'k_g8')
    for a in range(0, len(text), 16):
        raw = text[a:a + 16]
        opc = (raw[0] | (raw[1] << 8)) & 0xFFF
        if opc != _OPC_LDC:
            continue
        bank = raw[4]
        off = _ldc_cbank_offset(raw)
        assert not (bank == 0 and off >= _PARAM_AREA_END_G8), (
            f"G8: LDC at 0x{a:x} reads c[0][0x{off:x}] which lies in the "
            f"literal-pool region (>= 0x{_PARAM_AREA_END_G8:x}).  R23B.A "
            f"must have eliminated this path for u32 store-payload imms.")


# ---------------------------------------------------------------------------
# Test 3 — MOV32I encoder byte-exact round-trip
# ---------------------------------------------------------------------------

def test_mov32i_encoder_bytes_match_nvcc_ground_truth():
    """Reproduce NVCC's reproG4 MOV32I bytes exactly (ignoring ctrl)."""
    from sass.encoding.sm_120_opcodes import encode_mov32i
    raw = encode_mov32i(dest=7, imm32=0xAABBCCDD)
    assert raw[0:12] == bytes.fromhex('317407ffddccbbaaff010000'), (
        f"encode_mov32i bytes[0..12] diverge from NVCC ground truth: "
        f"got {raw[0:12].hex()}")


def test_mov32i_encoder_packs_imm_little_endian():
    from sass.encoding.sm_120_opcodes import encode_mov32i
    raw = encode_mov32i(dest=0, imm32=0x12345678)
    assert raw[4:8] == b'\x78\x56\x34\x12', (
        f"imm32 not packed little-endian in b4..b7: got {raw[4:8].hex()}")
