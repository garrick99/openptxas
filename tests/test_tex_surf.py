"""
test_tex_surf.py — Ground truth encoding tests for SM_120 texture/surface instructions.

Tests encode_tex, encode_tld_lz, encode_tld4, encode_txq, encode_suld, encode_sust
against actual ptxas 13.0 SM_120 cubin output.

Run:
    python -m pytest tests/test_tex_surf.py -v
"""

import sys
import os
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sass.encoding.sm_120_opcodes import (
    encode_tex, encode_tld_lz, encode_tld4, encode_txq,
    encode_suld, encode_sust,
    TEX_DIM_1D, TEX_DIM_2D, TEX_DIM_3D,
    SURF_DIM_1D, SURF_DIM_2D,
    SURF_MODE_B32, SURF_MODE_B64,
    TXQ_WIDTH, TXQ_HEIGHT, TXQ_DEPTH,
    RZ,
)


def _check(got: bytes, expected_lo: int, expected_hi: int):
    """Assert 16-byte encoding matches expected lo/hi qwords."""
    lo, hi = struct.unpack('<QQ', got)
    assert lo == expected_lo, f"lo mismatch: got 0x{lo:016x}, expected 0x{expected_lo:016x}"
    assert hi == expected_hi, f"hi mismatch: got 0x{hi:016x}, expected 0x{expected_hi:016x}"


# ---------------------------------------------------------------------------
# TEX (0xf60) — texture fetch, 2D and 3D
# ---------------------------------------------------------------------------

def test_tex_2d():
    """TEX RZ, R4, R4, UR4, 2D, 0x3 — ptxas ground truth."""
    _check(encode_tex(4, 4, 4, TEX_DIM_2D, mask=0x03, ctrl=0x0008b6),
           0x20ff04ff04047f60, 0x00116c00081e03ff)


def test_tex_3d():
    """TEX RZ, R5, R4, UR4, 3D, 0x1 — ptxas ground truth."""
    _check(encode_tex(5, 4, 4, TEX_DIM_3D, mask=0x01, ctrl=0x0008b6),
           0x40ff04ff04057f60, 0x00116c00081e01ff)


# ---------------------------------------------------------------------------
# TLD.LZ (0xf66) — texture load, level zero (1D integer coords)
# ---------------------------------------------------------------------------

def test_tld_lz_1comp():
    """TLD.LZ RZ, R5, R5, UR4, 1D, 0x1 — single component."""
    _check(encode_tld_lz(5, 5, 4, mask=0x01, dest_hi=RZ, ctrl=0x0008b4),
           0x00ff04ff05057f66, 0x00116800081e01ff)


def test_tld_lz_4comp():
    """TLD.LZ R6, R4, R4, UR4, 1D — all 4 components."""
    _check(encode_tld_lz(4, 4, 4, mask=0x0f, dest_hi=6, ctrl=0x0008b4),
           0x00ff04ff04047f66, 0x00116800081e0f06)


# ---------------------------------------------------------------------------
# TLD4.R (0xf63) — texture gather (4 texels, 2D)
# ---------------------------------------------------------------------------

def test_tld4_r_2d():
    """TLD4.R R6, R4, R4, UR4, 2D — ptxas ground truth."""
    _check(encode_tld4(4, 4, 4, dest_hi=6, ctrl=0x0008b6),
           0x20ff04ff04047f63, 0x00116c00081e0f06)


# ---------------------------------------------------------------------------
# TXQ (0xf6f) — texture query
# ---------------------------------------------------------------------------

def test_txq_width():
    """TXQ RZ, R5, R5, TEX_HEADER_DIMENSION, UR4, 0x0, 0x1 — width."""
    _check(encode_txq(5, 4, TXQ_WIDTH, ctrl=0x000fb1),
           0x0000040005057f6f, 0x001f6200080001ff)


def test_txq_height():
    """TXQ RZ, R7, R7, TEX_HEADER_DIMENSION, UR4, 0x0, 0x2 — height."""
    _check(encode_txq(7, 4, TXQ_HEIGHT, ctrl=0x0007b1),
           0x0000040007077f6f, 0x000f6200080002ff)


# ---------------------------------------------------------------------------
# SULD (0xf99) — surface load
# ---------------------------------------------------------------------------

def test_suld_1d_b32():
    """SULD.D.BA.1D.STRONG.SM.TRAP R5, [R5], UR4."""
    _check(encode_suld(5, 5, 4, SURF_DIM_1D, SURF_MODE_B32, ctrl=0x0008b5),
           0x10ff040005057f99, 0x00116a00081ea900)


def test_suld_1d_v2():
    """SULD.D.BA.1D.64.STRONG.SM.TRAP R4, [R4], UR4."""
    _check(encode_suld(4, 4, 4, SURF_DIM_1D, SURF_MODE_B64, ctrl=0x0008b5),
           0x10ff040004047f99, 0x00116a00081eab00)


# ---------------------------------------------------------------------------
# SUST (0xf9d) — surface store
# ---------------------------------------------------------------------------

def test_sust_1d_b32():
    """SUST.D.BA.1D.STRONG.SM.TRAP [R3], R0, UR4."""
    _check(encode_sust(0, 3, 4, SURF_DIM_1D, SURF_MODE_B32, ctrl=0x0008f2),
           0x10ff040003007f9d, 0x0011e4000810a900)


def test_sust_1d_v2():
    """SUST.D.BA.1D.64.STRONG.SM.TRAP [R0], R2, UR4."""
    _check(encode_sust(2, 0, 4, SURF_DIM_1D, SURF_MODE_B64, ctrl=0x0008f2),
           0x10ff040200007f9d, 0x0011e4000810ab00)


def test_sust_2d_b32():
    """SUST.D.BA.2D.STRONG.SM.TRAP [R2], R0, UR4."""
    _check(encode_sust(0, 2, 4, SURF_DIM_2D, SURF_MODE_B32, ctrl=0x0008f2),
           0x70ff040002007f9d, 0x0011e4000810a900)


# ---------------------------------------------------------------------------
# Opcode correctness
# ---------------------------------------------------------------------------

def test_opcode_values():
    """Verify opcode bytes match expected values."""
    assert encode_tex(0, 0, 0, TEX_DIM_2D)[0] == 0x60
    assert (encode_tex(0, 0, 0, TEX_DIM_2D)[1] & 0x0f) == 0x0f  # lo nibble of b1
    assert encode_tld_lz(0, 0, 0)[0] == 0x66
    assert encode_tld4(0, 0, 0)[0] == 0x63
    assert encode_txq(0, 0, TXQ_WIDTH)[0] == 0x6f
    assert encode_suld(0, 0, 0)[0] == 0x99
    assert encode_sust(0, 0, 0)[0] == 0x9d
