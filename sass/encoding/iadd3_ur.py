"""IADD3.R-UR (0xc11) encoder for SM_120 carry-chain 64-bit address computation.

TEMPLATE-ENGINE-11B: this instruction adds a 32-bit GPR offset to a 64-bit
UR pair, producing a 64-bit GPR address pair.  Used in pairs:

  Low:  R_lo = R_offset + UR_param_lo  (generates carry)
  High: R_hi = R_src2 + UR_param_hi    (consumes carry)

Ground truth from PTXAS SM_120 (226 instances across 144-kernel corpus):
  Low:  b9=0x10, b10=0x80, b11=0x0f
  High: b9=0x14, b10=0x0f, b11=0x08
"""
from __future__ import annotations


def _ctrl_to_bytes(ctrl: int) -> tuple[int, int, int]:
    raw24 = (ctrl & 0x7FFFFF) << 1
    return raw24 & 0xFF, (raw24 >> 8) & 0xFF, (raw24 >> 16) & 0xFF


_CTRL_DEFAULT = 0x7e0


def encode_iadd3_ur_lo(dest: int, src_gpr: int, src_ur: int,
                       ctrl: int = 0) -> bytes:
    """IADD3.R-UR low half: R_dest = R_src + UR_src (carry out).

    Ground truth: R2 = R3 + UR6 → 117c020306000000ff10800f00c81f00
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x11
    raw[1] = 0x7c
    raw[2] = dest & 0xFF
    raw[3] = src_gpr & 0xFF
    raw[4] = src_ur & 0xFF
    # b5-b7 = 0
    raw[8] = 0xFF  # RZ (no third source)
    raw[9] = 0x10
    raw[10] = 0x80
    raw[11] = 0x0f
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


def encode_iadd3_ur_hi(dest: int, src_gpr: int, src_ur: int,
                       ctrl: int = 0) -> bytes:
    """IADD3.R-UR high half: R_dest = R_src + UR_src + carry (carry in).

    Ground truth: R3 = R3 + UR7 → 117c030307000000ff140f0800ca0f00
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x11
    raw[1] = 0x7c
    raw[2] = dest & 0xFF
    raw[3] = src_gpr & 0xFF
    raw[4] = src_ur & 0xFF
    # b5-b7 = 0
    raw[8] = 0xFF  # RZ
    raw[9] = 0x14
    raw[10] = 0x0f
    raw[11] = 0x08
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)
