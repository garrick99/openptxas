"""
sass/encoding/sm_89_opcodes.py — SM_89 (Ada Lovelace / RTX 4090) instruction encoders.

Opcode mapping from ptxas 12.4 reference cubins (cuobjdump verified).
SM_89 differences from SM_120:
- No capmerc/merc DRM — full register range available
- Param ABI base: c[0x0][0x160] (SM_120 = c[0][0x380])
- Frame pointer: c[0x0][0x28] (SM_120 = c[0][0x37c])
- Ctrl format: Turing-era (SM_75+), different bit layout from Blackwell
- Many ALU opcodes differ (byte[1] = 0x7a vs 0x72/0x7c on SM_120)

Shared opcodes (identical encoding): NOP, EXIT, BRA, S2R, LDG.E, STG.E
"""

from __future__ import annotations
import struct

# Architecture constants
SM_VERSION = 89
PARAM_BASE = 0x160   # c[0x0][0x160] for first param (SM_120 = 0x380)
FRAME_PTR_OFFSET = 0x28  # c[0x0][0x28] (SM_120 = 0x37c)
E_FLAGS = 0x00590559

# Register constants
RZ = 0xFF
PT = 0x07

# Default ctrl word for SM_89 (Turing-era format)
# Encoding: bits[3:0]=stall, [7:4]=yield/sched, [12:8]=rbar, [16:13]=wbar
_CTRL_DEFAULT = 0x000fe4  # no stall, no barriers
_CTRL_EXIT    = 0x000fea  # EXIT ctrl
_CTRL_NOP     = 0x000fc0  # NOP/BRA ctrl

# Special register codes (same as SM_120)
SR_TID_X = 0x21
SR_TID_Y = 0x22
SR_TID_Z = 0x23
SR_CTAID_X = 0x25
SR_CTAID_Y = 0x26
SR_CTAID_Z = 0x27


def _ctrl_to_bytes(ctrl: int) -> tuple[int, int, int]:
    """Convert SM_89 ctrl to bytes 13, 14, 15."""
    return ctrl & 0xFF, (ctrl >> 8) & 0xFF, (ctrl >> 16) & 0xFF


def _build(b0: int, b1: int, *, b2=0, b3=0, b4=0, b5=0, b6=0, b7=0,
           b8=0, b9=0, b10=0, b11=0, ctrl: int = 0) -> bytes:
    """Build a 16-byte SM_89 SASS instruction."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = b0 & 0xFF; raw[1] = b1 & 0xFF
    raw[2] = b2 & 0xFF; raw[3] = b3 & 0xFF
    raw[4] = b4 & 0xFF; raw[5] = b5 & 0xFF
    raw[6] = b6 & 0xFF; raw[7] = b7 & 0xFF
    raw[8] = b8 & 0xFF; raw[9] = b9 & 0xFF
    raw[10] = b10 & 0xFF; raw[11] = b11 & 0xFF
    raw[13] = b13; raw[14] = b14; raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# SHARED opcodes (identical encoding on SM_89 and SM_120)
# ---------------------------------------------------------------------------

def encode_nop(comment: str = '') -> bytes:
    return _build(0x18, 0x79, ctrl=_CTRL_NOP)

def encode_exit(ctrl: int = 0) -> bytes:
    return _build(0x4d, 0x79, ctrl=ctrl or _CTRL_EXIT)

def encode_s2r(dest: int, sr_code: int, ctrl: int = 0) -> bytes:
    """S2R dest, SR_code — read special register."""
    return _build(0x19, 0x79, b2=dest, b9=sr_code, ctrl=ctrl or _CTRL_DEFAULT)

def encode_bra(offset: int, ctrl: int = 0) -> bytes:
    """BRA with signed offset (PC-relative from next instruction).

    SM_89 BRA encoding (from ptxas ground truth):
    - bytes 0-1: opcode (0x47, 0x79)
    - bytes 4-7: signed 32-bit byte offset (LE)
    - bytes 8-11: flags (0xff, 0xff, 0x83, 0x03)
    - bytes 13-15: ctrl

    Ground truth: BRA self-loop (offset=-16) at 0x140:
        47 79 00 00 f0 ff ff ff  ff ff 83 03  00 c0 0f 00
    """
    # Encode offset as signed 32-bit byte offset in bytes 4-7
    off32 = offset & 0xFFFFFFFF
    return _build(0x47, 0x79,
                  b4=off32 & 0xFF,
                  b5=(off32 >> 8) & 0xFF,
                  b6=(off32 >> 16) & 0xFF,
                  b7=(off32 >> 24) & 0xFF,
                  b8=0xFF, b9=0xFF,
                  b10=0x83, b11=0x03,
                  ctrl=ctrl or _CTRL_NOP)

def encode_ldg_e(dest: int, addr: int, width: int = 32, ctrl: int = 0) -> bytes:
    """LDG.E dest, [addr.64] — 0x981, same encoding as SM_120.

    Ground truth: LDG.E R5, [R4.64]
        81790504 04000000 00191e0c 00a80e00
    """
    b9_map = {32: 0x19, 64: 0x1b, 128: 0x1d}
    return _build(0x81, 0x79,
                  b2=dest, b3=addr, b4=0x04,
                  b9=b9_map.get(width, 0x19), b10=0x1e, b11=0x0c,
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_stg_e(addr: int, data: int, width: int = 32, ctrl: int = 0) -> bytes:
    """STG.E [addr.64], data — 0x986, same encoding as SM_120.

    Ground truth: STG.E [R2.64], R5
        86790002 05000000 0419100c 00e20f00
    """
    b9_map = {32: 0x19, 64: 0x1b}
    return _build(0x86, 0x79,
                  b2=0x00, b3=addr, b4=data,
                  b8=0x04,  # addressing mode (SM_89, from ptxas ground truth)
                  b9=b9_map.get(width, 0x19), b10=0x10, b11=0x0c,
                  ctrl=ctrl or 0x000fe2)


# ---------------------------------------------------------------------------
# SM_89-specific ALU opcodes
# ---------------------------------------------------------------------------

def encode_imad_mov_u32_cbuf(dest: int, bank: int, offset_bytes: int,
                              ctrl: int = 0) -> bytes:
    """IMAD.MOV.U32 dest, RZ, RZ, c[bank][offset] — param/const load (0x624).

    Replaces LDC on SM_120. Loads 32-bit value from constant bank.

    Ground truth: IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]
        247601ff 000a0000 ff008e07 00e40f00
    Ground truth: IMAD.MOV.U32 R5, RZ, RZ, c[0x0][0x160]
        247605ff 00580000 ff008e07 00e20f00
    """
    # Offset encoding: byte[4:5] = offset >> 2 (dword offset in 14-bit field)
    dword_off = (offset_bytes >> 2) & 0x3FFF
    return _build(0x24, 0x76,
                  b2=dest, b3=RZ,
                  b4=(dword_off >> 8) & 0xFF, b5=dword_off & 0xFF,
                  b6=(bank & 0xF) << 4,
                  b8=RZ, b9=0x00, b10=0x8e, b11=0x07,
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_mov_cbuf(dest: int, bank: int, offset_bytes: int,
                    ctrl: int = 0) -> bytes:
    """MOV dest, c[bank][offset] — constant bank load (0xa02).

    Ground truth: MOV R2, c[0x0][0x168]
        027a0200 005a0000 000f0000 00e20f00
    """
    dword_off = (offset_bytes >> 2) & 0x3FFF
    return _build(0x02, 0x7a,
                  b2=dest,
                  b4=(dword_off >> 8) & 0xFF, b5=dword_off & 0xFF,
                  b9=0x0f,
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_uldc_64(dest_ur: int, bank: int, offset_bytes: int,
                   ctrl: int = 0) -> bytes:
    """ULDC.64 URdest, c[bank][offset] — uniform const load (0xab9).

    Ground truth: ULDC.64 UR4, c[0x0][0x118]
        b97a0400 00460000 000a0000 00e20f00
    """
    dword_off = (offset_bytes >> 2) & 0x3FFF
    return _build(0xb9, 0x7a,
                  b2=dest_ur,
                  b4=(dword_off >> 8) & 0xFF, b5=dword_off & 0xFF,
                  b9=0x0a,
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_iadd3_cbuf(dest: int, src0: int, bank: int, offset_bytes: int,
                      src2: int = RZ, pred_out: int = 0, ctrl: int = 0) -> bytes:
    """IADD3 dest, P{pred_out}, src0, c[bank][offset], src2 — add with cbuf + carry out.

    Ground truth: IADD3 R2, P0, R6, c[0x0][0x168], RZ
        107a0206 005a0000 ffe0f107 00e40f00
    Ground truth: IADD3 R4, P1, R6, c[0x0][0x170], RZ
        107a0406 005c0000 ffe0f307 00e40f04
    """
    dword_off = (offset_bytes >> 2) & 0x3FFF
    b10 = 0xF0 | ((pred_out & 0x07) << 1) | 1  # P0→0xF1, P1→0xF3
    return _build(0x10, 0x7a,
                  b2=dest, b3=src0,
                  b4=(dword_off >> 8) & 0xFF, b5=dword_off & 0xFF,
                  b8=src2, b9=0xe0, b10=b10, b11=0x07,
                  ctrl=ctrl or _CTRL_DEFAULT)


def encode_iadd3x_cbuf(dest: int, src0: int, bank: int, offset_bytes: int,
                       src2: int = RZ, pred_in: int = 0, ctrl: int = 0) -> bytes:
    """IADD3.X dest, src0, c[bank][offset], src2, P{pred_in}, !PT — carry-extended cbuf add.

    Ground truth: IADD3.X R3, R0, c[0x0][0x16c], RZ, P0, !PT
        107a0300 005b0000 ffe47f00 00e40f00
    Ground truth: IADD3.X R5, R0, c[0x0][0x174], RZ, P1, !PT
        107a0500 005d0000 ffe4ff00 00e40f04
    """
    dword_off = (offset_bytes >> 2) & 0x3FFF
    b10 = 0x7F if pred_in == 0 else 0xFF  # P0→0x7F, P1→0xFF
    return _build(0x10, 0x7a,
                  b2=dest, b3=src0,
                  b4=(dword_off >> 8) & 0xFF, b5=dword_off & 0xFF,
                  b8=src2, b9=0xe4, b10=b10, b11=0x00,
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_iadd3(dest: int, src0: int, src1: int, src2: int = RZ,
                 ctrl: int = 0) -> bytes:
    """IADD3 dest, src0, src1, src2 — register-register add.

    Uses SM_89 R-R form (byte[1]=0x72, opcode 0x210 — same as SM_120!).
    Ground truth: IADD3 R5, R4, R5, RZ
        1072050405000000ffe0ff0700ca2f00
    """
    return _build(0x10, 0x72,
                  b2=dest, b3=src0, b4=src1,
                  b8=src2, b9=0xe0, b10=0xff, b11=0x07,
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_imad_shl_u32(dest: int, src: int, shift: int,
                        ctrl: int = 0) -> bytes:
    """IMAD.SHL.U32 dest, src, 1<<shift, RZ — shift via multiply (0x824, same as SM_120).

    Ground truth: IMAD.SHL.U32 R6, R0, 0x4, RZ
        2478060004000000ff008e0700e20f00
    """
    return _build(0x24, 0x78,
                  b2=dest, b3=src, b4=shift,
                  b8=RZ, b9=0x00, b10=0x8e, b11=0x07,
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_shf(dest: int, src_lo: int, shift: int, src_hi: int,
               left: bool = True, ctrl: int = 0) -> bytes:
    """SHF dest, src_lo, shift, src_hi — funnel shift (0xa19 on SM_89).

    Ground truth: SHF.L.U64.HI R0, RZ, 0x1e, R0
        197a00ff1e0000000016010000e20f00
    """
    return _build(0x19, 0x7a,
                  b2=dest, b3=src_lo, b4=shift,
                  b8=src_hi, b9=0x16 if left else 0x14, b10=0x01,
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_isetp_cbuf(pred_dest: int, src0: int, bank: int, offset_bytes: int,
                      cmp: int = 0x50, ctrl: int = 0) -> bytes:
    """ISETP.*.U32.AND Ppred, PT, src0, c[bank][offset], PT — comparison (0xa0c).

    Ground truth: ISETP.NE.U32.AND P0, PT, RZ, c[0x0][0x160], PT
        0c7aff00 00580000 f0030070 00da0f00
    """
    dword_off = (offset_bytes >> 2) & 0x3FFF
    return _build(0x0c, 0x7a,
                  b2=src0, b3=0x00,
                  b4=(dword_off >> 8) & 0xFF, b5=dword_off & 0xFF,
                  b8=0xf0 | (pred_dest & 0x07),
                  b9=0x03, b10=0x00, b11=cmp,
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_lop3_lut(dest: int, src0: int, src1: int, src2: int = RZ,
                    lut: int = 0xC0, ctrl: int = 0) -> bytes:
    """LOP3.LUT dest, src0, src1, src2, lut — 3-input logic (0xa12).

    LUT values: AND=0xC0, OR=0xFC, XOR=0x3C, NOT(src0)=0x0F
    Ground truth: LOP3.LUT R5, R5, c[0x0][0x164], RZ, 0xc0, !PT
        127a050500590000ffc08e0700ca0f00
    """
    return _build(0x12, 0x7a,
                  b2=dest, b3=src0, b4=src1,
                  b8=src2, b9=lut, b10=0x8e, b11=0x07,
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_mov(dest: int, src: int, ctrl: int = 0) -> bytes:
    """MOV dest, src — register move.

    Uses IADD3 dest, src, RZ, RZ pattern (same as SM_120).
    """
    return encode_iadd3(dest, src, RZ, RZ, ctrl=ctrl)

def encode_sel_imm(dest: int, src: int, imm: int, pred: int = 0,
                   neg: bool = False, ctrl: int = 0) -> bytes:
    """SEL dest, src, imm, [!]Ppred — conditional select (0x807).

    Ground truth: SEL R5, RZ, 0x1, !P0
        07780001ff057807  → byte[0]=0x07, byte[1]=0x78
        000fca0004000000
    """
    return _build(0x07, 0x78,
                  b2=dest, b3=src,
                  b4=imm & 0xFF, b5=(imm >> 8) & 0xFF,
                  b9=(0x04 if neg else 0x00) | (pred & 0x07),
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_fadd(dest: int, src0: int, src1: int, ctrl: int = 0) -> bytes:
    """FADD dest, src0, src1 — float add (0x621 cbuf form / 0xa21 R-R).

    Ground truth: FADD R5, R5, c[0x0][0x164]
        217605050059000000000000 00ca0f00
    """
    return _build(0x21, 0x7a,
                  b2=dest, b3=src0, b4=src1,
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_fmul(dest: int, src0: int, src1: int, ctrl: int = 0) -> bytes:
    """FMUL dest, src0, src1 — float multiply."""
    return _build(0x20, 0x7a,
                  b2=dest, b3=src0, b4=src1,
                  ctrl=ctrl or _CTRL_DEFAULT)

def encode_ffma(dest: int, src0: int, src1: int, src2: int,
                ctrl: int = 0) -> bytes:
    """FFMA dest, src0, src1, src2 — fused multiply-add."""
    return _build(0x23, 0x7a,
                  b2=dest, b3=src0, b4=src1,
                  b8=src2,
                  ctrl=ctrl or _CTRL_DEFAULT)


# ---------------------------------------------------------------------------
# Predicate patching (same mechanism as SM_120)
# ---------------------------------------------------------------------------

def patch_pred(raw: bytes, pred: int = 0, neg: bool = False) -> bytes:
    """Patch predicate guard into an instruction's byte[1] bits[7:4]."""
    buf = bytearray(raw)
    pred_field = pred & 0x7
    if neg:
        pred_field |= 0x8
    buf[1] = (buf[1] & 0x0F) | (pred_field << 4)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------
encode_ldc = encode_imad_mov_u32_cbuf  # SM_89 uses IMAD.MOV for param loads
encode_ldcu_64 = encode_uldc_64

# Comparison constants
ISETP_LT = 0x10
ISETP_EQ = 0x20
ISETP_LE = 0x30
ISETP_GT = 0x40
ISETP_NE = 0x50
ISETP_GE = 0x60

# LOP3 truth table constants
LOP3_AND = 0xC0
LOP3_OR  = 0xFC
LOP3_XOR = 0x3C
LOP3_NOT = 0x0F
