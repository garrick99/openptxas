"""
sass/encoding/sm_89_opcodes.py — SM_89 (Ada Lovelace / RTX 4090) instruction encoders.

SM_89 SASS encoding for Ada Lovelace GPUs. Key differences from SM_120:
- No capmerc/merc DRM system (register allocation is simpler)
- Many ALU opcodes differ (byte[1] = 0x7a on SM_89 vs 0x72 on SM_120)
- Some opcodes shared: S2R (0x919), EXIT (0x94d), NOP (0x918), BRA (0x947),
  LDG (0x981), STG (0x986), SHF (0x819)
- ELF uses different e_machine and e_flags values

Opcode mapping populated from ptxas 12.4 reference cubins on RTX 4090.
"""

from __future__ import annotations
import struct

# SM_89 ELF constants
SM_89_VERSION = 89
SM_89_E_FLAGS = 0x00060059  # SM_89 = 0x59 = 89

# Register constants (same as SM_120)
RZ = 0xFF  # Zero register
PT = 0x07  # Always-true predicate

# Default ctrl word (same format as SM_120)
_CTRL_DEFAULT = 0x7e0  # wdep=0x3e, rbar=0x01, misc=0, stall=0

# Special register codes (same as SM_120)
SR_TID_X = 0x21
SR_TID_Y = 0x22
SR_TID_Z = 0x23
SR_CTAID_X = 0x25
SR_CTAID_Y = 0x26
SR_CTAID_Z = 0x27


def _ctrl_to_bytes(ctrl: int) -> tuple[int, int, int]:
    """Convert 23-bit ctrl to bytes 13, 14, 15."""
    raw24 = (ctrl & 0x7FFFFF) << 1
    return raw24 & 0xFF, (raw24 >> 8) & 0xFF, (raw24 >> 16) & 0xFF


def _build(b0: int, b1: int, *, b2=0, b3=0, b4=0, b5=0, b6=0, b7=0,
           b8=0, b9=0, b10=0, b11=0, ctrl: int = 0) -> bytes:
    """Build a 16-byte SM_89 SASS instruction."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = b0 & 0xFF
    raw[1] = b1 & 0xFF
    raw[2] = b2 & 0xFF
    raw[3] = b3 & 0xFF
    raw[4] = b4 & 0xFF
    raw[5] = b5 & 0xFF
    raw[6] = b6 & 0xFF
    raw[7] = b7 & 0xFF
    raw[8] = b8 & 0xFF
    raw[9] = b9 & 0xFF
    raw[10] = b10 & 0xFF
    raw[11] = b11 & 0xFF
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# Instructions shared with SM_120 (same encoding)
# ---------------------------------------------------------------------------

def encode_nop(comment: str = '') -> bytes:
    """NOP instruction (same on SM_89 and SM_120)."""
    return _build(0x18, 0x79, ctrl=_CTRL_DEFAULT)


def encode_exit(ctrl: int = 0) -> bytes:
    """EXIT instruction (same on SM_89 and SM_120)."""
    if ctrl == 0:
        ctrl = 0x7f5  # wdep=0x3f, misc=5
    return _build(0x4d, 0x79, ctrl=ctrl)


def encode_s2r(dest: int, sr_code: int, ctrl: int = 0) -> bytes:
    """S2R dest, SR_code (same on SM_89 and SM_120)."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x19, 0x79, b2=dest, b9=sr_code, ctrl=ctrl)


def encode_bra(offset: int, ctrl: int = 0) -> bytes:
    """BRA with signed 18-bit offset (same on SM_89 and SM_120)."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    signed_insns = offset // 16
    offset18 = signed_insns & 0x3FFFF
    return _build(0x47, 0x79,
                  b2=0xFC, b3=0x00,
                  b4=0xFC & 0xFF,
                  b8=offset18 & 0xFF,
                  b9=(offset18 >> 8) & 0xFF,
                  b10=0x80 | ((offset18 >> 16) & 0x03),
                  b11=0x03,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# SM_89-specific opcodes (populated from ptxas RE)
# TODO: Fill in after opcode mapping agent completes
# ---------------------------------------------------------------------------

# Placeholder — will be populated with actual SM_89 encodings
# SM_89 opcodes identified so far:
# 0x624: LDC (param load) — SM_120 = 0xb82
# 0xa24: IMAD R-UR — SM_120 = 0xc24
# 0xa0c: ISETP R-UR — SM_120 = 0xc0c
# 0xab9: LDCU? — SM_120 = 0x7ac
# 0xa10: IADD3? — SM_120 = 0x210
