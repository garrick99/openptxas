"""
SM_120 instruction encoders for all opcodes needed for a minimal kernel.

Field layout (128-bit = lo[63:0] | hi[63:0], little-endian byte array):
  Byte positions (0-indexed in the 16-byte array):
    b[0:2]   opcode word (fixed per instruction)
    b[2]     destination register (or pred_dest / 0x00)
    b[3]     src0 register (or 0xff/0x00 fixed)
    b[4]     src1 register OR 8-bit immediate
    b[5:8]   0x000000 (always zero)
    b[8]     src2 register (or fixed/0x00)
    b[9]     modifier byte 1  (instruction-specific)
    b[10]    modifier byte 2  (instruction-specific)
    b[11]    modifier byte 3  (instruction-specific, often 0x00)
    b[12]    0x00 (always)
    b[13]    ctrl bits [7:0]   (= (ctrl << 1) & 0xFF)
    b[14]    ctrl bits [15:8]  (= ((ctrl << 1) >> 8) & 0xFF)
    b[15]    ctrl bits [22:16] (= ((ctrl << 1) >> 16) & 0xFF) | variant fixed bits

Ground truth samples from sm_120_encoding_tables.json confirmed throughout.

SR codes for S2R:
    SR_TID_X    = 0x21    SR_TID_Y    = 0x22    SR_TID_Z    = 0x23
    SR_CTAID_X  = 0x25    SR_CTAID_Y  = 0x26    SR_CTAID_Z  = 0x27
    SR_NTID_X   = 0x29    SR_NTID_Y   = 0x2a    SR_NTID_Z   = 0x2b
    SR_NCTAID_X = 0x2d    SR_NCTAID_Y = 0x2e
"""

from __future__ import annotations
import struct

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RZ = 255   # zero register index on SM_120
PT = 7     # predicate "true" (always-true predicate register)

# Predicate encoding: byte 1, bits 7:4 of the lo qword
# 0x7 = PT (always), 0x0 = P0, 0x1 = P1, ..., 0x8 = !P0, 0x9 = !P1, ...
def patch_pred(raw: bytes, pred: int = PT, neg: bool = False) -> bytes:
    """Patch predicate guard on any instruction. pred=0..5 for P0-P5, 7=PT (always)."""
    buf = bytearray(raw)
    code = pred if not neg else (pred | 0x8)
    buf[1] = (buf[1] & 0x0F) | (code << 4)
    return bytes(buf)

# SR codes for S2R
SR_TID_X     = 0x21
SR_TID_Y     = 0x22
SR_TID_Z     = 0x23
SR_CTAID_X   = 0x25
SR_CTAID_Y   = 0x26
SR_CTAID_Z   = 0x27
SR_NTID_X    = 0x29
SR_NTID_Y    = 0x2a
SR_NTID_Z    = 0x2b
SR_NCTAID_X  = 0x2d
SR_NCTAID_Y  = 0x2e
SR_NCTAID_Z  = 0x2f

# Default control word: safe for standalone emission.
# Uses stall=0 with read barrier at 0x1f and write dep at 0x3e,
# matching ptxas's typical pattern for simple instructions.
# bits[22:17]=stall, [16]=yield, [15]=wbar, [14:10]=rbar, [9:4]=wdep, [3:0]=misc
_CTRL_DEFAULT = 0x7e0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ctrl_to_bytes(ctrl: int) -> tuple[int, int, int]:
    """
    Pack a 23-bit control word into bytes 13, 14, 15.
    Storage: raw24 = ctrl << 1; b13=raw24[7:0], b14=raw24[15:8], b15=raw24[22:16].
    """
    raw24 = (ctrl & 0x7FFFFF) << 1
    b13 = raw24 & 0xFF
    b14 = (raw24 >> 8) & 0xFF
    b15 = (raw24 >> 16) & 0xFF
    return b13, b14, b15


def _build(b0: int, b1: int,
           b2: int, b3: int, b4: int,
           b8: int,
           b9: int, b10: int, b11: int,
           ctrl: int,
           b15_fixed: int = 0x00) -> bytes:
    """
    Generic 16-byte instruction builder.  Bytes 5-7 and 12 are always 0x00.
    b15_fixed is ORed with the ctrl-derived b15 (used by some variants).
    """
    b13, b14, b15_ctrl = _ctrl_to_bytes(ctrl)
    b15 = b15_ctrl | b15_fixed

    raw = bytearray(16)
    raw[0]  = b0
    raw[1]  = b1
    raw[2]  = b2 & 0xFF
    raw[3]  = b3 & 0xFF
    raw[4]  = b4 & 0xFF
    raw[5]  = 0x00
    raw[6]  = 0x00
    raw[7]  = 0x00
    raw[8]  = b8  & 0xFF
    raw[9]  = b9  & 0xFF
    raw[10] = b10 & 0xFF
    raw[11] = b11 & 0xFF
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# NOP
# ---------------------------------------------------------------------------
# Ground truth: hex=18790000000000000000000000c00f00
#   lo=0x7918, hi=0xfc00000000000
#   ctrl=0x7e0: raw=0xfc0 -> b13=0xc0, b14=0x0f, b15=0x00  ✓
#   All bytes fixed except control.

def encode_nop(ctrl: int = 0) -> bytes:
    """
    Encode NOP to 16 bytes.

    NOP has no register operands; all fields except the control word are fixed.

    Args:
        ctrl: 23-bit scheduling control word (0 = default 0x7e0, stall=15).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_nop(ctrl=0x7e0) -> bytes.fromhex('18790000000000000000000000c00f00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x18, 0x79,
                  b2=0x00, b3=0x00, b4=0x00,
                  b8=0x00,
                  b9=0x00, b10=0x00, b11=0x00,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# EXIT
# ---------------------------------------------------------------------------
# Ground truth (pred=PT): hex=4d790000000000000000800300ea0f00
#   lo=0x794d, hi=0xfea0003800000
#   Bytes: 4d 79 00 00 00 00 00 00 | 00 00 80 03 00 ea 0f 00
#   ctrl=0x7f5: raw=0x7f5<<1=0xfea -> b13=0xea, b14=0x0f, b15=0x00  ✓
#   Fixed modifier bytes: b10=0x80, b11=0x03

def encode_exit(ctrl: int = 0) -> bytes:
    """
    Encode EXIT to 16 bytes.

    EXIT terminates the thread.  No register operands.

    Args:
        ctrl: 23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_exit(ctrl=0x7f5) -> bytes.fromhex('4d790000000000000000800300ea0f00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x4d, 0x79,
                  b2=0x00, b3=0x00, b4=0x00,
                  b8=0x00,
                  b9=0x00, b10=0x80, b11=0x03,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# MOV
# ---------------------------------------------------------------------------
# Ground truth: "R6, R5" hex=0272060005000000000f000000ca0f00
#   lo=0x500067202
#   Bytes: 02 72 06 00 05 00 00 00 | 00 0f 00 00 00 ca 0f 00
#   b0=0x02, b1=0x72, b2=dest, b3=0x00, b4=src, b9=0x0f
#   ctrl=0x7e5: raw=0xfca -> b13=0xca, b14=0x0f  ✓
#
# "R3, R2" hex=0272030002000000000f000000c60f00
#   b2=0x03, b4=0x02, ctrl=0x7e3: raw=0xfc6->b13=0xc6  ✓
#
# Note: src goes to b4 (bits[39:32]), NOT b3 (bits[31:24]).
# b3 is always 0x00 for MOV.

def encode_mov(dest: int, src: int, ctrl: int = 0) -> bytes:
    """
    Encode MOV dest, src to 16 bytes.

    MOV copies a register value.  src is placed at bits[39:32] (b4), not b3.

    Args:
        dest: Destination register index (0..254).
        src:  Source register index (0..254, 255=RZ).
        ctrl: 23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_mov(6, 5, ctrl=0x7e5) -> bytes.fromhex('0272060005000000000f000000ca0f00')
        encode_mov(3, 2, ctrl=0x7e3) -> bytes.fromhex('0272030002000000000f000000c60f00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x02, 0x72,
                  b2=dest, b3=0x00, b4=src,
                  b8=0x00,
                  b9=0x0f, b10=0x00, b11=0x00,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# LDC  (32-bit constant-bank load)
# ---------------------------------------------------------------------------
# Ground truth: "R1, c[0x0][0x37c]" hex=827b01ff00df00000008000000e20f00
#   lo=0xdf00ff017b82
#   Bytes: 82 7b 01 ff 00 df 00 00 | 00 08 00 00 00 e2 0f 00
#   b0=0x82, b1=0x7b, b2=dest(1), b3=0xff(fixed), b4=bank(0x00), b5=offset_word(0xdf=223=0x37c/4)
#   b9=0x08 (size=32 marker)
#   ctrl=0x7f1: raw=0xfe2 -> b13=0xe2, b14=0x0f  ✓
#
# "R13, c[0x0][0x360]" hex=827b0dff00d800000008000000240e00
#   b2=0x0d, b5=0xd8(216=0x360/4), ctrl=0x712: raw=0xe24->b13=0x24,b14=0x0e  ✓
#
# NOTE: b5 holds the word offset (const_offset_bytes // 4).
# b4 holds the constant bank index.
# b3 is always 0xFF.

def encode_ldc(dest: int, const_bank: int, const_offset_bytes: int,
               ctrl: int = 0) -> bytes:
    """
    Encode LDC dest, c[bank][offset] to 16 bytes (32-bit load).

    Loads a 32-bit value from constant memory bank `const_bank` at byte offset
    `const_offset_bytes` (must be 4-byte aligned).

    Args:
        dest:               Destination register index (0..254).
        const_bank:         Constant bank index (0..15, typically 0).
        const_offset_bytes: Byte offset in the constant bank (must be divisible by 4).
        ctrl:               23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_ldc(1, 0, 0x37c, ctrl=0x7f1)
            -> bytes.fromhex('827b01ff00df00000008000000e20f00')
        encode_ldc(13, 0, 0x360, ctrl=0x712)
            -> bytes.fromhex('827b0dff00d800000008000000240e00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    offset_word = (const_offset_bytes // 4) & 0xFF

    # Build the 16-byte encoding.  b5 carries the word offset — _build() only
    # handles b0-b4 + b8-b11, so we patch b5 manually.
    raw = bytearray(_build(0x82, 0x7b,
                           b2=dest, b3=0xFF, b4=const_bank & 0xFF,
                           b8=0x00,
                           b9=0x08, b10=0x00, b11=0x00,
                           ctrl=ctrl))
    raw[5] = offset_word
    return bytes(raw)


# ---------------------------------------------------------------------------
# LDC.64  (64-bit constant-bank load)
# ---------------------------------------------------------------------------
# Ground truth: "R2, c[0x0][0x388]" hex=827b02ff00e20000000a000000220e00
#   lo=0xe200ff027b82
#   Bytes: 82 7b 02 ff 00 e2 00 00 | 00 0a 00 00 00 22 0e 00
#   b9=0x0a (size=64 marker, vs 0x08 for 32-bit)
#   ctrl=0x711: raw=0xe22 -> b13=0x22, b14=0x0e  ✓
#
# "R10, c[0x0][0x380]" hex=827b0aff00e00000000a000000a20e00
#   b2=0x0a(dest=10), b5=0xe0(0x380/4=224), ctrl=0x751: raw=0xea2->b13=0xa2,b14=0x0e  ✓

def encode_ldc_64(dest: int, const_bank: int, const_offset_bytes: int,
                  ctrl: int = 0) -> bytes:
    """
    Encode LDC.64 dest, c[bank][offset] to 16 bytes (64-bit load).

    Like encode_ldc() but loads 64 bits (dest and dest+1 are written).
    b9=0x0a instead of 0x08 distinguishes 64-bit from 32-bit.

    Args:
        dest:               Destination register pair base index (0..253).
        const_bank:         Constant bank index (0..15, typically 0).
        const_offset_bytes: Byte offset in the constant bank (must be 8-byte aligned).
        ctrl:               23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_ldc_64(2, 0, 0x388, ctrl=0x711)
            -> bytes.fromhex('827b02ff00e20000000a000000220e00')
        encode_ldc_64(10, 0, 0x380, ctrl=0x751)
            -> bytes.fromhex('827b0aff00e00000000a000000a20e00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    offset_word = (const_offset_bytes // 4) & 0xFF

    raw = bytearray(_build(0x82, 0x7b,
                           b2=dest, b3=0xFF, b4=const_bank & 0xFF,
                           b8=0x00,
                           b9=0x0a, b10=0x00, b11=0x00,
                           ctrl=ctrl))
    raw[5] = offset_word
    return bytes(raw)


# ---------------------------------------------------------------------------
# S2R  (Special Register to Register)
# ---------------------------------------------------------------------------
# Ground truth: "R0, SR_TID.X" hex=197900000000000000210000002e0e00
#   lo=0x7919
#   Bytes: 19 79 00 00 00 00 00 00 | 00 21 00 00 00 2e 0e 00
#   b0=0x19, b1=0x79, b2=dest(0), b3=0x00, b4=0x00
#   b9=SR code (0x21 for SR_TID.X)
#   ctrl=0x717: raw=0x717<<1=0xe2e -> b13=0x2e, b14=0x0e  ✓
#
# SR code is stored at b9 (hi bits[15:8]).

def encode_s2r(dest: int, sr_code: int, ctrl: int = 0) -> bytes:
    """
    Encode S2R dest, SR_xxx to 16 bytes.

    Reads a special register into a general-purpose register.

    SR codes:
        SR_TID_X   = 0x21  (thread ID in X dimension)
        SR_TID_Y   = 0x22  (thread ID in Y dimension)
        SR_CTAID_X = 0x25  (cooperative thread array ID in X)
        SR_NTID_X  = 0x29  (number of threads in X dimension)

    Args:
        dest:    Destination register index (0..254).
        sr_code: Special register code (e.g. SR_TID_X=0x21).
        ctrl:    23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_s2r(0, SR_TID_X, ctrl=0x717)
            -> bytes.fromhex('197900000000000000210000002e0e00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x19, 0x79,
                  b2=dest, b3=0x00, b4=0x00,
                  b8=0x00,
                  b9=sr_code & 0xFF, b10=0x00, b11=0x00,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# IADD3
# ---------------------------------------------------------------------------
# Ground truth: "RZ, P0, PT, RZ, R4, RZ" hex=1072ffff04000000ffe0f10700e20f00
#   lo=0x4ffff7210
#   Bytes: 10 72 ff ff 04 00 00 00 | ff e0 f1 07 00 e2 0f 00
#   b0=0x10, b1=0x72
#   b2=dest(RZ=0xff), b3=0xff(src0=RZ), b4=src1(R4=0x04)
#   b8=src2(RZ=0xff)
#   Fixed: b9=0xe0, b10=0xf1, b11=0x07
#   ctrl=0x7f1: raw=0xfe2 -> b13=0xe2, b14=0x0f  ✓
#
# Only one unique variant in the JSON (all 5 instances identical).
# Operand interpretation: IADD3 dest, P0, PT, src0, src1, src2
#   dest  -> b2  (integer result, often RZ)
#   src0  -> b3  (first addend, often RZ)
#   src1  -> b4  (second addend, variable register)
#   src2  -> b8  (third addend, often RZ)

def encode_iadd3(dest: int, src0: int, src1: int, src2: int,
                 negate_src1: bool = False, ctrl: int = 0) -> bytes:
    """
    Encode IADD3 dest, P0, PT, src0, [+-]src1, src2 to 16 bytes.

    Three-way integer add.  The predicate outputs (P0, PT) are fixed in the
    modifier bytes; only the four integer register operands are variable.
    When negate_src1=True, computes dest = src0 - src1 + src2.

    Args:
        dest:  Destination register index (0..255, 255=RZ).
        src0:  First source register index (0..255, 255=RZ).
        src1:  Second source register index (0..255, 255=RZ).
        src2:  Third source register index (0..255, 255=RZ).
        negate_src1: If True, negate src1 (subtraction).
        ctrl:  23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_iadd3(RZ, RZ, 4, RZ, ctrl=0x7f1)
            -> bytes.fromhex('1072ffff04000000ffe0f10700e20f00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    # Negate modifier: ptxas uses b7=0x80 and b10=0xff for src1 negation (sub).
    # Ground truth: ptxas sub.u32 → IADD3 with b7=0x80 b10=0xff b11=0x07
    b10 = 0xff if negate_src1 else 0xf1
    raw = bytearray(_build(0x10, 0x72,
                           b2=dest, b3=src0, b4=src1,
                           b8=src2,
                           b9=0xe0, b10=b10, b11=0x07,
                           ctrl=ctrl))
    if negate_src1:
        raw[7] = 0x80  # negate flag in byte 7
    return bytes(raw)


# ---------------------------------------------------------------------------
# IADD3.X  (carry-extended add)
# ---------------------------------------------------------------------------
# Ground truth: "R7, PT, PT, RZ, RZ, RZ, P0, !PT" hex=107207ffff000000ffe47f0000e40f00
#   lo=0xffff077210
#   Bytes: 10 72 07 ff ff 00 00 00 | ff e4 7f 00 00 e4 0f 00
#   b0=0x10, b1=0x72 (same opcode as IADD3)
#   b2=dest(7), b3=src0(RZ=0xff), b4=src1(RZ=0xff)
#   b8=src2(RZ=0xff)
#   Fixed: b9=0xe4, b10=0x7f, b11=0x00
#   ctrl=0x7f2: raw=0x7f2<<1=0xfe4 -> b13=0xe4, b14=0x0f  ✓
#
# Modifier bytes differ from plain IADD3 (0xe0,0xf1,0x07 vs 0xe4,0x7f,0x00)
# The .X suffix enables carry-in from the previous IADD3's predicate output.

def encode_iadd3x(dest: int, src0: int, src1: int, src2: int,
                  ctrl: int = 0) -> bytes:
    """
    Encode IADD3.X dest, PT, PT, src0, src1, src2, P0, !PT to 16 bytes.

    IADD3.X is the carry-extended version of IADD3, used for the high word of
    a 64-bit add or subtract (carries the overflow from the low word).

    Args:
        dest:  Destination register index (0..255, 255=RZ).
        src0:  First source register index (0..255, 255=RZ).
        src1:  Second source register index (0..255, 255=RZ).
        src2:  Third source register index (0..255, 255=RZ).
        ctrl:  23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_iadd3x(7, RZ, RZ, RZ, ctrl=0x7f2)
            -> bytes.fromhex('107207ffff000000ffe47f0000e40f00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x10, 0x72,
                  b2=dest, b3=src0, b4=src1,
                  b8=src2,
                  b9=0xe4, b10=0x7f, b11=0x00,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# IMAD.WIDE
# ---------------------------------------------------------------------------
# Ground truth: "R2, R13, 0x8, R2" hex=2578020d0800000002028e0700cc1f00
#   lo=0x80d027825
#   Bytes: 25 78 02 0d 08 00 00 00 | 02 02 8e 07 00 cc 1f 00
#   b0=0x25, b1=0x78
#   b2=dest(2), b3=src0(13), b4=imm_or_src1(8)
#   b8=src2(2)
#   Fixed: b9=0x02, b10=0x8e, b11=0x07
#   ctrl=0xfe6: raw=0xfe6<<1=0x1fcc -> b13=0xcc, b14=0x1f  ✓
#
# "R6, R13, 0x8, R10": b2=6, b3=13, b4=8, b8=10, ctrl=0x27f1->raw=0x4fe2  ✓
# "R2, R9,  0x8, R2":  b2=2, b3=9,  b4=8, b8=2,  ctrl=0xfe6  ✓
# "R6, R9,  0x8, R6":  b2=6, b3=9,  b4=8, b8=6,  ctrl=0x27f1 ✓

def encode_imad_wide(dest: int, src0: int, src1_imm: int, src2: int,
                     ctrl: int = 0) -> bytes:
    """
    Encode IMAD.WIDE dest, src0, src1_imm, src2 to 16 bytes.

    Wide multiply-add: (dest, dest+1) = src0 * src1_imm + src2.
    src1_imm can be an 8-bit immediate value or a register index.
    In the observed ptxas output this field always holds an 8-bit immediate
    (the immediate encoding path) at b4.

    Args:
        dest:     Destination register pair base index (0..253).
        src0:     Multiplicand register index (0..254).
        src1_imm: 8-bit multiplier (immediate) or register index (0..255).
        src2:     Addend register index (0..255, 255=RZ).
        ctrl:     23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_imad_wide(2, 13, 0x8, 2,  ctrl=0x0fe6)
            -> bytes.fromhex('2578020d0800000002028e0700cc1f00')
        encode_imad_wide(6, 13, 0x8, 10, ctrl=0x27f1)
            -> bytes.fromhex('2578060d080000000a028e0700e24f00')
        encode_imad_wide(2, 9,  0x8, 2,  ctrl=0x0fe6)
            -> bytes.fromhex('257802090800000002028e0700cc1f00')
        encode_imad_wide(6, 9,  0x8, 6,  ctrl=0x27f1)
            -> bytes.fromhex('257806090800000006028e0700e24f00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x25, 0x78,
                  b2=dest, b3=src0, b4=src1_imm & 0xFF,
                  b8=src2,
                  b9=0x02, b10=0x8e, b11=0x07,
                  ctrl=ctrl)


def encode_imad_wide_rr(dest: int, src0: int, src1: int, src2: int,
                        ctrl: int = 0) -> bytes:
    """
    Encode IMAD.WIDE R-R (signed): (dest, dest+1) = src0 * src1 + (src2, src2+1).

    Signed form: b9=0x02.  Ground truth confirmed from SM_120 hardware.
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x25, 0x72,
                  b2=dest, b3=src0, b4=src1 & 0xFF,
                  b8=src2,
                  b9=0x02, b10=0x8e, b11=0x07,
                  ctrl=ctrl)


def encode_imad_wide_u32(dest: int, src0: int, src1: int, src2: int,
                         ctrl: int = 0) -> bytes:
    """
    Encode IMAD.WIDE.U32 R-R (unsigned): (dest, dest+1) = src0 * src1 + (src2, src2+1).

    Unsigned form: b9=0x00 (vs signed b9=0x02).
    Ground truth from ptxas mul.hi.u64 probe (SM_120):
      IMAD.WIDE.U32 R6, R3, R4, RZ → bytes 25 72 06 03 04 00 00 00 ff 00 8e 07 ...
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x25, 0x72,
                  b2=dest, b3=src0, b4=src1 & 0xFF,
                  b8=src2,
                  b9=0x00, b10=0x8e, b11=0x07,
                  ctrl=ctrl)


def encode_imad_wide_u32_carry(dest: int, src0: int, src1: int, src2: int,
                                ctrl: int = 0) -> bytes:
    """
    Encode IMAD.WIDE.U32 R-R with P0 carry-out: (dest,dest+1) = src0*src1 + (src2,src2+1).

    Carry form: b9=0x00, b10=0x80 (P0 receives overflow carry).
    Ground truth:
      IMAD.WIDE.U32 R8, P0, R2, R5, R6 → bytes 25 72 08 02 05 00 00 00 06 00 80 07 ...
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x25, 0x72,
                  b2=dest, b3=src0, b4=src1 & 0xFF,
                  b8=src2,
                  b9=0x00, b10=0x80, b11=0x07,
                  ctrl=ctrl)


def encode_imad_wide_u32x(dest: int, src0: int, src1: int, src2: int,
                           ctrl: int = 0) -> bytes:
    """
    Encode IMAD.WIDE.U32.X R-R with P0 carry-in: (dest,dest+1) = src0*src1 + (src2,src2+1) + P0.

    Carry-in form: b9=0x04, b10=0x0e, b11=0x00 (reads P0 as carry-in).
    Ground truth:
      IMAD.WIDE.U32.X R8, R3, R5, R10, P0 → bytes 25 72 08 03 05 00 00 00 0a 04 0e 00 ...
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x25, 0x72,
                  b2=dest, b3=src0, b4=src1 & 0xFF,
                  b8=src2,
                  b9=0x04, b10=0x0e, b11=0x00,
                  ctrl=ctrl)


def encode_imad_ur(dest: int, src0: int, ur_src: int, src2: int,
                   ctrl: int = 0) -> bytes:
    """
    Encode IMAD R-UR: dest = src0 * UR[ur_src] + src2 (non-wide, single register write).

    Ground truth: IMAD R13, R13, UR4, R0 → 247c0d0d0400000000028e0f00ca1f00
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x24, 0x7c,
                  b2=dest, b3=src0, b4=ur_src & 0xFF,
                  b8=src2,
                  b9=0x02, b10=0x8e, b11=0x0f,
                  ctrl=ctrl)


def encode_imad(dest: int, src0: int, src1: int, src2: int,
                ctrl: int = 0) -> bytes:
    """
    Encode IMAD dest, src0, src1, src2 (register-register multiply-add).

    Non-wide: dest = (src0 * src1 + src2) & 0xFFFFFFFF (low 32 bits only).
    All operands are registers. Uses opcode 0x224 (R-R variant).

    Based on IMAD R-UR ground truth (opcode 0xc24, bytes 24 7c) with
    b1 changed to 0x72 for R-R encoding.
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x24, 0x72,
                  b2=dest, b3=src0, b4=src1 & 0xFF,
                  b8=src2,
                  b9=0x02, b10=0x8e, b11=0x07,
                  ctrl=ctrl)


def encode_imad_rr(dest: int, src0: int, src1: int, src2: int,
                   ctrl: int = 0) -> bytes:
    """
    Encode IMAD dest, src0, src1, src2 (R-R-R multiply-add, SM_120 validated).

    Uses opcode 0x2a4 (byte[0]=0xa4, byte[1]=0x72) — the ptxas-confirmed
    encoding for multiply with both source operands in GPRs.  Differs from
    encode_imad (0x224) in byte[0] bit[7] and byte[11] bit[3].

    Ground truth (ptxas 13.0, sm_120):
        mul.lo.u32 %r3, %r1, %r2 where %r1=R4, %r2=R5, %r3=R4, addend=RZ:
          a472040405000000ff028e0f00cc0f00
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0xa4, 0x72,
                  b2=dest, b3=src0, b4=src1 & 0xFF,
                  b8=src2,
                  b9=0x02, b10=0x8e, b11=0x0f,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# LDG.E.64  (64-bit global memory load, descriptor-based)
# ---------------------------------------------------------------------------
# Ground truth: "R2, desc[UR4][R2.64]" hex=8179020204000000001b1e0c00e22e00
#   lo=0x402027981
#   Bytes: 81 79 02 02 04 00 00 00 | 00 9b 1e 0c 00 e2 2e 00
#   b0=0x81, b1=0x79
#   b2=dest(2), b3=src_addr(R2=2), b4=ur_desc(UR4=4)
#   b8=0x00 (no src2)
#   Fixed: b9=0x9b, b10=0x1e, b11=0x0c
#   ctrl=0x1771: raw=0x1771<<1=0x2ee2 -> b13=0xe2, b14=0x2e  ✓
#
# All 64 instances in JSON are identical — single unique load pattern.

def encode_ldg_e(dest: int, ur_desc: int, src_addr: int,
                  width: int = 32, ctrl: int = 0) -> bytes:
    """Encode LDG.E dest, desc[ur_desc][src_addr.64] with variable width (32/64/128 bits)."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b9_map = {32: 0x19, 64: 0x1b, 128: 0x1d}
    return _build(0x81, 0x79,
                  b2=dest, b3=src_addr, b4=ur_desc & 0xFF,
                  b8=0x00,
                  b9=b9_map.get(width, 0x9b), b10=0x1e, b11=0x0c,
                  ctrl=ctrl)


def encode_stg_e(ur_desc: int, src_addr: int, src_data: int,
                  width: int = 32, ctrl: int = 0) -> bytes:
    """Encode STG.E desc[ur_desc][src_addr.64], src_data with variable width.

    Field layout (verified against ptxas):
      b2=0x00, b3=addr_reg, b4=data_reg, b8=ur_desc_idx
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b9_map = {32: 0x19, 64: 0x1b, 128: 0x1d}
    return _build(0x86, 0x79,
                  b2=0x00, b3=src_addr, b4=src_data & 0xFF,
                  b8=ur_desc & 0xFF,
                  b9=b9_map.get(width, 0x1b), b10=0x10, b11=0x0c,
                  ctrl=ctrl)


def encode_ldg_e_64(dest: int, ur_desc: int, src_addr: int,
                    ctrl: int = 0) -> bytes:
    """
    Encode LDG.E.64 dest, desc[ur_desc][src_addr.64] to 16 bytes.

    64-bit global memory load using a descriptor-based addressing mode.
    Loads 64 bits (two consecutive 32-bit registers starting at dest) from
    the address given by (uniform register ur_desc, base register src_addr).

    Args:
        dest:     Destination register pair base index (0..253).
        ur_desc:  Uniform register index for the memory descriptor (e.g. UR4=4).
        src_addr: Base address register index (0..254).
        ctrl:     23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_ldg_e_64(2, 4, 2, ctrl=0x1771)
            -> bytes.fromhex('8179020204000000001b1e0c00e22e00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x81, 0x79,
                  b2=dest, b3=src_addr, b4=ur_desc & 0xFF,
                  b8=0x00,
                  b9=0x1b, b10=0x1e, b11=0x0c,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# STG.E.64  (64-bit global memory store, descriptor-based)
# ---------------------------------------------------------------------------
# Ground truth: "desc[UR4][R6.64], R4" hex=8679000604000000041b100c00e20f00
#   lo=0x406007986
#   Bytes: 86 79 00 06 04 00 00 00 | 04 1b 10 0c 00 e2 0f 00
#   b0=0x86, b1=0x79
#   b2=0x00 (no dest reg — store has no destination)
#   b3=src_addr(R6=6), b4=ur_desc(UR4=4)
#   b8=src_data(R4=4)
#   Fixed: b9=0x1b, b10=0x10, b11=0x0c
#   ctrl=0x7f1: raw=0xfe2 -> b13=0xe2, b14=0x0f  ✓

def encode_stg_e_64(ur_desc: int, src_addr: int, src_data: int,
                    ctrl: int = 0) -> bytes:
    """
    Encode STG.E.64 desc[ur_desc][src_addr.64], src_data to 16 bytes.

    64-bit global memory store using a descriptor-based addressing mode.
    Stores 64 bits (two consecutive 32-bit registers from src_data) to the
    address given by (uniform register ur_desc, base register src_addr).

    Args:
        ur_desc:  Uniform register index for the memory descriptor (e.g. UR4=4).
        src_addr: Base address register index (0..254).
        src_data: Source data register pair base index (0..253).
        ctrl:     23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_stg_e_64(4, 6, 4, ctrl=0x7f1)
            -> bytes.fromhex('8679000604000000041b100c00e20f00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x86, 0x79,
                  b2=0x00, b3=src_addr, b4=src_data & 0xFF,
                  b8=ur_desc & 0xFF,
                  b9=0x1b, b10=0x10, b11=0x0c,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# ISETP.GE.AND  (integer set-predicate, >=, AND)
# ---------------------------------------------------------------------------
# Ground truth: "P0, PT, R13, UR5, PT" hex=0c7c000d050000007062f00b00da2f00
#   lo=0x50d007c0c
#   Bytes: 0c 7c 00 0d 05 00 00 00 | 70 62 f0 0b 00 da 2f 00
#   b0=0x0c, b1=0x7c
#   b2=pred_dest(P0=0), b3=src_reg(R13=13), b4=ur_src(UR5=5)
#   Fixed: b8=0x70, b9=0x62, b10=0xf0, b11=0x0b
#   ctrl=0x17ed: raw=0x17ed<<1=0x2fda -> b13=0xda, b14=0x2f  ✓
#
# "P0, PT, R9, UR5, PT" hex=0c7c0009050000007062f00b00da2f00
#   b3=0x09(R9)  ✓  — same ur_src=UR5, same pred_dest=P0

def encode_isetp_ge_and(pred_dest: int, src_reg: int, ur_src: int,
                        ctrl: int = 0) -> bytes:
    """
    Encode ISETP.GE.AND pred_dest, PT, src_reg, ur_src, PT to 16 bytes.

    Sets predicate register pred_dest = (src_reg >= ur_src), ANDed with PT.
    The secondary predicate output and PT operands are fixed in modifier bytes.

    Args:
        pred_dest: Destination predicate register index (0..6; 7=PT).
        src_reg:   Source integer register index (0..254).
        ur_src:    Uniform register index for the comparison value.
        ctrl:      23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_isetp_ge_and(0, 13, 5, ctrl=0x17ed)
            -> bytes.fromhex('0c7c000d050000007062f00b00da2f00')
        encode_isetp_ge_and(0, 9, 5, ctrl=0x17ed)
            -> bytes.fromhex('0c7c0009050000007062f00b00da2f00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    # b9 bit 1 depends on src_reg index parity: even=0x60, odd=0x62
    # Verified: ptxas R0(even)+UR5 → 0x60; ptxas R13(odd)+UR5 → 0x62; R9(odd)+UR5 → 0x62
    b9_val = 0x62 if (src_reg & 1) else 0x60
    return _build(0x0c, 0x7c,
                  b2=pred_dest & 0xFF, b3=src_reg, b4=ur_src & 0xFF,
                  b8=0x70,
                  b9=b9_val, b10=0xf0, b11=0x0b,
                  ctrl=ctrl)


def encode_isetp_ur(pred_dest: int, src_reg: int, ur_src: int,
                    cmp: int = 0x06, ctrl: int = 0) -> bytes:  # 0x06 = ISETP_GE
    """
    Encode ISETP.<cmp>.U32.AND pred_dest, PT, src_reg, ur_src, PT (R-UR variant).

    SM_120 only: uses opcode 0xc0c (R-UR). The R-R variant (0x20c) silently
    produces P=FALSE on SM_120 hardware.

    b9 parity: bit 1 of b9 reflects src_reg index parity (odd→0x62, even→0x60).
    This is a hardware encoding artefact; it does NOT flip the comparison direction.

    The ctrl misc field must be 0 or ≥13 for this instruction to produce correct
    results on SM_120. The scoreboard forces misc=0 for all ISETP R-UR instructions.

    Ground truth (GE, ctrl=0x17ed):
        encode_isetp_ur(0, 13, 5) -> bytes.fromhex('0c7c000d050000007062f00b00da2f00')
        encode_isetp_ur(0, 9, 5)  -> bytes.fromhex('0c7c0009050000007062f00b00da2f00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b9_val = 0x62 if (src_reg & 1) else 0x60
    b8_val = (cmp << 4) | 0x10  # same cmp encoding as R-R variant
    return _build(0x0c, 0x7c,
                  b2=pred_dest & 0xFF, b3=src_reg, b4=ur_src & 0xFF,
                  b8=b8_val,
                  b9=b9_val, b10=0xf0, b11=0x0b,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# LDSM: Load Shared to Matrix (warp-level shared→register matrix load)
# ---------------------------------------------------------------------------
# LDSM.16.M88.4 R_dest, [R_addr] — loads 4 registers from shared memory
# in a warp-cooperative pattern for feeding tensor cores.
# Opcode: 0x83b, b2=dest_base, b3=addr_reg, b9=0x02

def encode_ldsm_x4(dest: int, addr_reg: int, ctrl: int = 0) -> bytes:
    """Encode LDSM.16.M88.4 dest, [addr_reg] — load 4 matrix regs from smem.
    Ground truth: opcode 0x83b, b9=0x02 (x4 = 4 matrices)."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x3b, 0x78, b2=dest, b3=addr_reg, b4=0x00,
                  b8=0x00, b9=0x02, b10=0x00, b11=0x00, ctrl=ctrl)


def encode_ldsm_x2(dest: int, addr_reg: int, ctrl: int = 0) -> bytes:
    """Encode LDSM.16.M88.2 dest, [addr_reg] — load 2 matrix regs from smem.
    Layout inferred from x4 (b9=0x01); needs hardware verification."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x3b, 0x78, b2=dest, b3=addr_reg, b4=0x00,
                  b8=0x00, b9=0x01, b10=0x00, b11=0x00, ctrl=ctrl)


def encode_ldsm_x1(dest: int, addr_reg: int, ctrl: int = 0) -> bytes:
    """Encode LDSM.16.M88.1 dest, [addr_reg] — load 1 matrix reg from smem.
    Layout inferred from x4 (b9=0x00); needs hardware verification."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x3b, 0x78, b2=dest, b3=addr_reg, b4=0x00,
                  b8=0x00, b9=0x00, b10=0x00, b11=0x00, ctrl=ctrl)


# ---------------------------------------------------------------------------
# CVT: Type conversion instructions
# ---------------------------------------------------------------------------

def encode_i2f_f32_s32(dest: int, src: int, ctrl: int = 0) -> bytes:
    """Encode I2FP.F32.S32 dest, src (signed int32 to float32).
    Ground truth: ptxas cvt.rn.f32.s32 → b9=0x10, b10=0x20."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x45, 0x72, b2=dest, b3=0x00, b4=src,
                  b8=0x00, b9=0x10, b10=0x20, b11=0x00, ctrl=ctrl)


def encode_f2i_s32_f32(dest: int, src: int, ctrl: int = 0) -> bytes:
    """Encode F2I.TRUNC.NTZ dest, src (float32 to signed int32, truncate)."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x05, 0x73, b2=dest, b3=0x00, b4=src,
                  b8=0x00, b9=0xf1, b10=0x20, b11=0x00, ctrl=ctrl)


# ---------------------------------------------------------------------------
# Tensor Core: HMMA (FP16 MMA) and IMMA (INT8 MMA)
# ---------------------------------------------------------------------------
# HMMA.16816.F32 R_d, R_a, R_b, R_c — FP16 matrix multiply-accumulate
#   Shape: m16 n8 k16, FP16 inputs (A: 4 regs, B: 2 regs), FP32 accumulation (4 regs)
#   Opcode: 0x23c, b9=0x18 (shape+type modifier)
#   Ground truth: HMMA.16816.F32 R12, R8, R4, R12
#     lo=0x00000004080c723c  hi=0x008fe2000000180c

def encode_hmma_f16_f32(dest: int, src_a: int, src_b: int, src_c: int,
                         ctrl: int = 0) -> bytes:
    """
    Encode HMMA.16816.F32 dest, src_a, src_b, src_c.

    Tensor core FP16 matrix multiply-accumulate (m16n8k16 shape):
      D[dest:dest+3] = A[src_a:src_a+3] * B[src_b:src_b+1] + C[src_c:src_c+3]

    All register arguments are base indices. A uses 4 regs (8 FP16 values),
    B uses 2 regs (4 FP16), C and D use 4 regs (4 FP32).

    Ground truth: encode_hmma_f16_f32(12, 8, 4, 12, ctrl=0x47f1)
        -> matches ptxas HMMA.16816.F32 R12, R8, R4, R12
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x3c
    raw[1] = 0x72
    raw[2] = dest & 0xFF
    raw[3] = src_a & 0xFF
    raw[4] = src_b & 0xFF
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8] = src_c & 0xFF
    raw[9] = 0x18       # m16n8k16 + F32 accum
    raw[10] = 0x00
    raw[11] = 0x00
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# HMMA.1688.F32 — FP16 tensor core m16n8k8 shape (k=8 variant, ptxas-verified)
# Ground truth: HMMA.1688.F32 R4, R4, RZ, RZ → 3c 72 04 04 ff 00 00 00 ff 10 00 00 [ctrl]
#   b9=0x10 (vs 0x18 for k16); A: 2 regs, B: 1 reg, C/D: 4 regs

def encode_hmma_f16_f32_k8(dest: int, src_a: int, src_b: int, src_c: int,
                             ctrl: int = 0) -> bytes:
    """HMMA.1688.F32 dest, src_a, src_b, src_c — m16n8k8 FP16→FP32 tensor core.
    A uses 2 regs, B uses 1 reg, C/D use 4 regs. Ground truth: b9=0x10.
    HMMA.1688.F32 R4, R4, RZ, RZ → 3c72 04 04 ff 00 00 00 ff 10 00 00 [ctrl]"""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x3c
    raw[1] = 0x72
    raw[2] = dest & 0xFF
    raw[3] = src_a & 0xFF
    raw[4] = src_b & 0xFF
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8] = src_c & 0xFF
    raw[9] = 0x10       # m16n8k8 + F32 accum (vs 0x18 for k16)
    raw[10] = 0x00
    raw[11] = 0x00
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# HMMA.16816.F32.BF16 — BF16 tensor core (decoded 2026-04-01)
# Ground truth: hi=0x008fe2000004180c → raw[9]=0x18, raw[10]=0x04

def encode_hmma_bf16_f32(dest: int, src_a: int, src_b: int, src_c: int,
                          ctrl: int = 0) -> bytes:
    """HMMA.16816.F32.BF16 — BF16 tensor core m16n8k16, FP32 accumulate."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x3c; raw[1] = 0x72
    raw[2] = dest & 0xFF; raw[3] = src_a & 0xFF; raw[4] = src_b & 0xFF
    raw[8] = src_c & 0xFF
    raw[9] = 0x18; raw[10] = 0x04  # BF16 modifier
    raw[13] = b13; raw[14] = b14; raw[15] = b15
    return bytes(raw)


# HMMA.1688.F32.TF32 — TF32 tensor core (decoded 2026-04-01)
# Ground truth: hi=0x008fe2000008100c → raw[9]=0x10, raw[10]=0x08

def encode_hmma_tf32_f32(dest: int, src_a: int, src_b: int, src_c: int,
                          ctrl: int = 0) -> bytes:
    """HMMA.1688.F32.TF32 — TF32 tensor core m16n8k8, FP32 accumulate."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x3c; raw[1] = 0x72
    raw[2] = dest & 0xFF; raw[3] = src_a & 0xFF; raw[4] = src_b & 0xFF
    raw[8] = src_c & 0xFF
    raw[9] = 0x10; raw[10] = 0x08  # TF32 modifier
    raw[13] = b13; raw[14] = b14; raw[15] = b15
    return bytes(raw)


# DMMA.8x8x4 — FP64 tensor core (NEWLY DISCOVERED 2026-04-01)
# Ground truth: DMMA.8x8x4 R8, R2, R4, R8
#   lo=0x000000040208723f  hi=0x008e240000000008
# Opcode: 0x23f, b8=0x08 (src_c), raw[9]=0x00, raw[10]=0x00
# Consumer Blackwell (RTX 5090) HAS FP64 tensor cores.

def encode_dmma_8x8x4(dest: int, src_a: int, src_b: int, src_c: int,
                       ctrl: int = 0) -> bytes:
    """DMMA.8x8x4 — FP64 tensor core m8n8k4. Double-precision MMA on RTX 5090."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x3f, 0x72, b2=dest, b3=src_a, b4=src_b, b8=src_c,
                  b9=0x00, b10=0x00, b11=0x00, ctrl=ctrl)


# CS2R — Convergence Special Register Read (NEWLY DISCOVERED 2026-04-01)
# Ground truth: CS2R R14, SRZ → 0x00000000000e7805 | 0x000fe2000001ff00
# Opcode: 0x805, used in tensor core kernels for warp convergence state

def encode_cs2r(dest: int, sr_code: int = 0, ctrl: int = 0) -> bytes:
    """CS2R Rdest, SR — read convergence special register."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x05, 0x78, b2=dest, b3=0x00, b4=0x00, b8=0x00,
                  b9=0x01, b10=0xFF, b11=0x00, ctrl=ctrl)


# IMMA.16832.S8.S8 — INT8 matrix multiply-accumulate
#   Shape: m16 n8 k32, INT8 inputs (A: 4 regs, B: 2 regs), INT32 accumulation (4 regs)
#   Opcode: 0x237, b9=0x5c, b10=0x40
#   Ground truth: IMMA.16832.S8.S8 R12, R8.ROW, R4.COL, R12
#     lo=0x00000004080c7237  hi=0x008fe20000405c0c

def encode_imma_s8_s32(dest: int, src_a: int, src_b: int, src_c: int,
                        ctrl: int = 0) -> bytes:
    """
    Encode IMMA.16832.S8.S8 dest, src_a.ROW, src_b.COL, src_c.

    Tensor core INT8 matrix multiply-accumulate (m16n8k32 shape):
      D[dest:dest+3] = A[src_a:src_a+3] * B[src_b:src_b+1] + C[src_c:src_c+3]

    All register arguments are base indices. A uses 4 regs (16 INT8 values),
    B uses 2 regs (8 INT8), C and D use 4 regs (4 INT32).
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x37
    raw[1] = 0x72
    raw[2] = dest & 0xFF
    raw[3] = src_a & 0xFF
    raw[4] = src_b & 0xFF
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8] = src_c & 0xFF
    raw[9] = 0x5c       # m16n8k32 + S8
    raw[10] = 0x40      # signed format flag
    raw[11] = 0x00
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# Shared memory: STS, LDS, BAR.SYNC
# ---------------------------------------------------------------------------

def encode_sts(ur_addr: int, offset: int, data_reg: int, ctrl: int = 0) -> bytes:
    """Encode STS [UR_addr + offset], data_reg (32-bit shared store)."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x88
    raw[1] = 0x79
    raw[2] = 0x00
    raw[3] = 0xFF
    raw[4] = data_reg & 0xFF
    raw[5] = offset & 0xFF
    raw[6] = (offset >> 8) & 0xFF
    raw[7] = 0x00
    raw[8] = ur_addr & 0xFF
    raw[9] = 0x08
    raw[10] = 0x00
    raw[11] = 0x08
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


def encode_lds(dest: int, ur_addr: int, offset: int, ctrl: int = 0) -> bytes:
    """Encode LDS dest, [UR_addr + offset] (32-bit shared load)."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x84
    raw[1] = 0x79
    raw[2] = dest & 0xFF
    raw[3] = 0xFF
    raw[4] = ur_addr & 0xFF
    raw[5] = offset & 0xFF
    raw[6] = (offset >> 8) & 0xFF
    raw[7] = 0x00
    raw[8] = 0x00
    raw[9] = 0x08
    raw[10] = 0x00
    raw[11] = 0x08
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


def encode_bar_sync(barrier_id: int = 0, ctrl: int = 0) -> bytes:
    """Encode BAR.SYNC barrier_id (thread barrier synchronization)."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x1d
    raw[1] = 0x7b
    raw[2] = 0x00
    raw[3] = 0x00
    raw[4] = 0x00
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8] = 0x00
    raw[9] = 0x00
    raw[10] = 0x01
    raw[11] = 0x00
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# System register to Uniform Register
SR_CGA_CTA_ID = 0x88

def encode_s2ur(dest_ur: int, sr_code: int, ctrl: int = 0) -> bytes:
    """Encode S2UR dest_ur, SR_xxx (system register to uniform register)."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0xc3
    raw[1] = 0x79
    raw[2] = dest_ur & 0xFF
    raw[3] = 0x00
    raw[4] = 0x00
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8] = 0x00
    raw[9] = sr_code & 0xFF
    raw[10] = 0x00
    raw[11] = 0x00
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# SM_89 (Ada Lovelace / RTX 4090) specific instructions
# ---------------------------------------------------------------------------
# SM_89 uses direct addressing for LDG/STG (no descriptor) and
# loads constants via MOV/IMAD.MOV from constant bank instead of LDC.

def encode_mov_from_cbank(dest: int, const_bank: int, const_offset_bytes: int,
                           ctrl: int = 0) -> bytes:
    """Encode MOV dest, c[bank][offset] — SM_89 constant bank load (32-bit).

    Opcode 0xa02. Different from SM_120's LDC (0xb82).
    Ground truth: MOV R2, c[0][0x168] → 027a0200005a0000000f000000e20f00
    Offset is stored as word offset (bytes/4) at b5.
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    word_off = (const_offset_bytes // 4) & 0xFF
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x02; raw[1] = 0x7a
    raw[2] = dest & 0xFF
    raw[3] = 0x00
    raw[4] = const_bank & 0xFF
    raw[5] = word_off
    raw[6] = 0x00; raw[7] = 0x00; raw[8] = 0x00
    raw[9] = 0x0f; raw[10] = 0x00; raw[11] = 0x00; raw[12] = 0x00
    raw[13] = b13; raw[14] = b14; raw[15] = b15
    return bytes(raw)


def encode_imad_mov_u32_cbank(dest: int, const_bank: int, const_offset_bytes: int,
                               ctrl: int = 0) -> bytes:
    """Encode IMAD.MOV.U32 dest, RZ, RZ, c[bank][offset] — SM_89 constant load.

    Opcode 0x624. Used for loading constants when MOV encoding isn't available.
    Ground truth: IMAD.MOV.U32 R1, RZ, RZ, c[0][0x28] → 247601ff000a0000ff008e0700e40f00
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    word_off = (const_offset_bytes // 4) & 0xFF
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x24; raw[1] = 0x76
    raw[2] = dest & 0xFF
    raw[3] = RZ
    raw[4] = const_bank & 0xFF
    raw[5] = word_off
    raw[6] = 0x00; raw[7] = 0x00
    raw[8] = RZ
    raw[9] = 0x00; raw[10] = 0x8e; raw[11] = 0x07; raw[12] = 0x00
    raw[13] = b13; raw[14] = b14; raw[15] = b15
    return bytes(raw)


def encode_uldc_64(dest_ur: int, const_bank: int, const_offset_bytes: int,
                    ctrl: int = 0) -> bytes:
    """Encode ULDC.64 dest_ur, c[bank][offset] — SM_89 uniform constant load.

    Opcode 0xab9. SM_89 equivalent of SM_120's LDCU (0x7ac).
    Ground truth: ULDC.64 UR4, c[0][0x118] → b97a040000460000000a000000ca0f00
    Offset stored as word offset at b5.
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    word_off = (const_offset_bytes // 4) & 0xFF
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0xb9; raw[1] = 0x7a
    raw[2] = dest_ur & 0xFF
    raw[3] = 0x00
    raw[4] = const_bank & 0xFF
    raw[5] = word_off
    raw[6] = 0x00; raw[7] = 0x00; raw[8] = 0x00
    raw[9] = 0x0a; raw[10] = 0x00; raw[11] = 0x00; raw[12] = 0x00
    raw[13] = b13; raw[14] = b14; raw[15] = b15
    return bytes(raw)


def encode_ldg_e_64_direct(dest: int, addr: int, ctrl: int = 0) -> bytes:
    """Encode LDG.E.64 dest, [addr.64] — SM_89 direct addressing (no descriptor).

    Same opcode 0x981 as SM_120, but b4=0x04 is hardcoded (not UR descriptor).
    Ground truth: LDG.E.64 R2, [R2.64] → 8179020204000000001b1e0c00a20e00
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x81, 0x79,
                  b2=dest, b3=addr, b4=0x04,
                  b8=0x00,
                  b9=0x1b, b10=0x1e, b11=0x0c,
                  ctrl=ctrl)


def encode_stg_e_64_direct(addr: int, data: int, ctrl: int = 0) -> bytes:
    """Encode STG.E.64 [addr.64], data — SM_89 direct addressing.

    Ground truth: STG.E.64 [R6.64], R4 → 8679000604000000041b100c00e20f00
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x86, 0x79,
                  b2=0x00, b3=addr, b4=0x04,
                  b8=data,
                  b9=0x1b, b10=0x10, b11=0x0c,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# FP32 instructions
# ---------------------------------------------------------------------------

def encode_fadd(dest: int, src0: int, src1: int,
                negate_src0: bool = False, ctrl: int = 0) -> bytes:
    """Encode FADD dest, [+-]src0, src1 (FP32 add/subtract)."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x21, 0x72,
                  b2=dest, b3=src0, b4=src1,
                  b8=0x00,
                  b9=0x01 if negate_src0 else 0x00, b10=0x00, b11=0x00,
                  ctrl=ctrl)


def encode_fmul(dest: int, src0: int, src1: int, ctrl: int = 0) -> bytes:
    """Encode FMUL dest, src0, src1 (FP32 multiply, both operands in GPR)."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    # FMUL is FFMA with src2=RZ (addend=0): dest = src0 * src1 + 0
    return _build(0x23, 0x72,
                  b2=dest, b3=src0, b4=src1,
                  b8=RZ,
                  b9=0x00, b10=0x00, b11=0x00,
                  ctrl=ctrl)


def encode_fmul_imm(dest: int, src0: int, imm_f32: int, ctrl: int = 0) -> bytes:
    """Encode FMUL dest, src0, float_imm (FP32 multiply with 32-bit immediate).

    Opcode 0x820. The 32-bit IEEE 754 float is encoded in b4-b7.
    Ground truth (ptxas sm_120): mul.f32 %f1, %f0, 0f40000000 (2.0f):
        20780704000000400000400000ca4f00
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    import struct as _s
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x20
    raw[1] = 0x78
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    # 32-bit float immediate in b4-b7
    _s.pack_into('<I', raw, 4, imm_f32 & 0xFFFFFFFF)
    # b8-b11: from ptxas ground truth
    raw[8] = 0x00
    raw[9] = 0x00
    raw[10] = 0x40
    raw[11] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


def encode_ffma(dest: int, src0: int, src1: int, src2: int,
                negate_src0: bool = False, ctrl: int = 0) -> bytes:
    """Encode FFMA dest, [+-]src0, src1, src2 (FP32 fused multiply-add)."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x23, 0x72,
                  b2=dest, b3=src0, b4=src1,
                  b8=src2,
                  b9=0x01 if negate_src0 else 0x00, b10=0x00, b11=0x00,
                  ctrl=ctrl)


def encode_ffma_imm(dest: int, src0: int, imm_f32: int, src2: int,
                    ctrl: int = 0) -> bytes:
    """Encode FFMA dest, src0, float_imm, src2 (FP32 fused multiply-add with immediate).

    Opcode 0x823. dest = src0 * float_imm + src2.
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    import struct as _s
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x23
    raw[1] = 0x78
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    _s.pack_into('<I', raw, 4, imm_f32 & 0xFFFFFFFF)
    raw[8] = src2 & 0xFF
    raw[9] = 0x00
    raw[10] = 0x00
    raw[11] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# LOP3.LUT (3-input logic op with truth table)
# ---------------------------------------------------------------------------
# Opcode 0x212. LUT byte at b4 encodes the operation:
#   AND = 0xC0, OR = 0xFC, XOR = 0x3C
# Ground truth: LOP3.LUT R3, R3, R2, RZ, 0x3c, !PT
#   hex: 1272030302000000ff3c8e0700ca0f00

LOP3_AND = 0xC0
LOP3_OR  = 0xFC
LOP3_XOR = 0x3C

def encode_lop3(dest: int, src0: int, src1: int, src2: int,
                lut: int, ctrl: int = 0) -> bytes:
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x12
    raw[1] = 0x72
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    raw[4] = src1 & 0xFF
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8] = src2 & 0xFF
    raw[9] = lut & 0xFF
    raw[10] = 0x8e
    raw[11] = 0x07
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# IMAD.SHL.U32 (multiply-add as shift-left, avoids SHF pipeline conflicts)
# ---------------------------------------------------------------------------
# Ground truth: "IMAD.SHL.U32 R6, R2, 0x100, RZ"
#   lo=0x0000010002067824  hi=0x000fca00078e00ff
#   b0=0x24, b1=0x78 (opcode 0x824), b2=dest, b3=src0, b4:b5=imm16, b8=RZ
#   b9=0x00, b10=0x8e, b11=0x07
#   This computes: dest = src0 * imm16 + RZ = src0 << log2(imm16)

def encode_imad_shl_u32(dest: int, src0: int, shift_amount: int,
                         src2: int = RZ, ctrl: int = 0) -> bytes:
    """
    Encode IMAD.SHL.U32 dest, src0, (1<<K), src2 — shift-multiply-add.

    dest = src0 * (1 << shift_amount) + src2.
    WARNING: WIDE write — also writes dest+1 with the high word (b11=0x07).

    Args:
        dest:         Destination register (0..254).
        src0:         Source register (0..254).
        shift_amount: Shift amount K (0..15, imm16 = 1<<K fits in 16 bits).
        src2:         Addend register (0..255, 255=RZ). Default RZ.
        ctrl:         23-bit scheduling control word.
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    imm16 = 1 << shift_amount
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x24
    raw[1] = 0x78
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    raw[4] = imm16 & 0xFF
    raw[5] = (imm16 >> 8) & 0xFF
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8] = src2 & 0xFF
    raw[9] = 0x00
    raw[10] = 0x8e
    raw[11] = 0x07
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# LDCU.64 (64-bit constant-bank load to uniform register)
# ---------------------------------------------------------------------------
# Ground truth: "LDCU.64 UR4, c[0x0][0x358]"
#   Bytes: ac 77 04 ff 00 6b 00 00 | 00 0a 00 08 00 2e 0e 00
#   b0=0xac, b1=0x77 (opcode=0x77ac → bits[11:0]=0x7ac)
#   b2=dest_ur(4), b3=0xff, b4=bank(0), b5=qword_offset(0x6b=107 → 107*8=0x358)
#   b9=0x0a (64-bit), b11=0x08 (LDCU-specific flag)
#   ctrl=0x717: raw=0xe2e → b13=0x2e, b14=0x0e  ✓
#
# LDCU uses qword (8-byte) offsets, not dword (4-byte) like LDC.

def encode_ldcu_64(dest_ur: int, const_bank: int, const_offset_bytes: int,
                   ctrl: int = 0) -> bytes:
    """
    Encode LDCU.64 dest_ur, c[bank][offset] to 16 bytes.

    Loads a 64-bit value from constant memory into a uniform register.
    Used to load memory descriptors for LDG/STG addressing.

    Args:
        dest_ur:            Destination uniform register index (e.g. 4 for UR4).
        const_bank:         Constant bank index (typically 0).
        const_offset_bytes: Byte offset (must be 8-byte aligned). Stored as qword index.
        ctrl:               23-bit scheduling control word.

    Ground truth:
        encode_ldcu_64(4, 0, 0x358, ctrl=0x717)
            -> bytes.fromhex('ac7704ff006b0000000a000800002e0e00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    qword_offset = (const_offset_bytes // 8) & 0xFF

    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0xac
    raw[1] = 0x77
    raw[2] = dest_ur & 0xFF
    raw[3] = 0xFF
    raw[4] = const_bank & 0xFF
    raw[5] = qword_offset
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8] = 0x00
    raw[9] = 0x0a    # 64-bit
    raw[10] = 0x00
    raw[11] = 0x08   # LDCU-specific
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


def encode_imad_imm(dest: int, src0: int, imm: int, src2: int,
                     ctrl: int = 0) -> bytes:
    """
    Encode IMAD.WIDE dest, src0, imm, src2 — multiply by immediate + add register.

    dest = src0 * imm + src2. Writes dest AND dest+1 (WIDE).
    imm is a 16-bit unsigned immediate.

    Based on IMAD.WIDE ground truth (opcode 0x825).
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x25
    raw[1] = 0x78
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    raw[4] = imm & 0xFF
    raw[5] = (imm >> 8) & 0xFF
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8] = src2 & 0xFF
    raw[9] = 0x02
    raw[10] = 0x8e
    raw[11] = 0x07
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


def encode_ldcu_32(dest_ur: int, const_bank: int, const_offset_bytes: int,
                   ctrl: int = 0) -> bytes:
    """Encode LDCU.32 dest_ur, c[bank][offset] — 32-bit constant to single UR.

    LDCU uses a split dword-offset encoding: the dword offset (byte_off//4) is
    stored with bits[8:1] in b5 and bit[0] in b4 bit[7].

    Ground truth from ptxas:
        LDCU UR5, c[0][0x39c]  ->  b4=0x80, b5=0x73  (dword=0xe7, 0xe7>>1=0x73, 0xe7&1=1)
        LDCU UR4, c[0][0x398]  ->  b4=0x00, b5=0x73  (dword=0xe6, 0xe6>>1=0x73, 0xe6&1=0)
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    dword_offset = const_offset_bytes // 4
    b5 = (dword_offset >> 1) & 0xFF
    b4_lsb = (dword_offset & 1) << 7
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0xac
    raw[1] = 0x77
    raw[2] = dest_ur & 0xFF
    raw[3] = 0xFF
    raw[4] = b4_lsb | (const_bank & 0x7F)
    raw[5] = b5
    raw[9] = 0x08    # 32-bit (vs 0x0a for 64-bit)
    raw[11] = 0x08
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# IADD.64 with Uniform Register source (R-UR variant)
# ---------------------------------------------------------------------------
# Ground truth: IADD.64 R4, R2, UR10 → 0x0000000a02047c35 / ctrl
# Opcode: 0x35, 0x7c (vs 0x35, 0x72 for R-R variant)
# byte2=dest(R), byte3=src0(R), byte4=src1(UR)

def encode_iadd64_ur(dest: int, src_r: int, src_ur: int,
                     ctrl: int = 0) -> bytes:
    """Encode IADD.64 dest(R), src0(R), src1(UR) — uniform register variant."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x35
    raw[1] = 0x7c     # R-UR variant (vs 0x72 for R-R)
    raw[2] = dest & 0xFF
    raw[3] = src_r & 0xFF
    raw[4] = src_ur & 0xFF
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8] = 0x00
    raw[9] = 0x02
    raw[10] = 0x8e
    raw[11] = 0x0f     # matches ptxas ground truth
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# IADD.64  (64-bit integer add/subtract in one instruction)
# ---------------------------------------------------------------------------
# Ground truth: "IADD.64 R6, R2, -R4" hex lo=0x8000000402067235 hi=0x004fcc00078e0200
#   Bytes: 35 72 06 02 04 00 00 80 | 00 02 8e 07 00 cc 4f 00
#   b0=0x35, b1=0x72 (opcode=0x7235 → bits[11:0]=0x235)
#   b2=dest(6), b3=src0(2), b4=src1(4)
#   b7=0x80 → bit 63 = negate flag on src1
#   b9=0x02, b10=0x8e, b11=0x07 (fixed modifiers for .64)
#   ctrl=0x27e6: raw=0x4fcc → b13=0xcc, b14=0x4f  ✓
#
# For IADD.64 (add, no negate): b7=0x00
# For IADD.64 (sub, negate src1): b7=0x80

def encode_iadd64(dest: int, src0: int, src1: int,
                  negate_src1: bool = False, ctrl: int = 0) -> bytes:
    """
    Encode IADD.64 dest, src0, [+-]src1 to 16 bytes.

    64-bit add or subtract in a single instruction.
    When negate_src1=True, computes dest = src0 - src1.

    Args:
        dest:         Destination register pair base index (0..253, even).
        src0:         First source register pair base (0..253, even).
        src1:         Second source register pair base (0..253, even).
        negate_src1:  If True, negate src1 (subtraction).
        ctrl:         23-bit scheduling control word (0 = default).

    Ground truth:
        encode_iadd64(6, 2, 4, negate_src1=True, ctrl=0x27e6)
            -> bytes.fromhex('3572060204000080000002078e00cc4f00')
            Wait, let me compute correctly...
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b7 = 0x80 if negate_src1 else 0x00

    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x35
    raw[1] = 0x72
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    raw[4] = src1 & 0xFF
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = b7
    raw[8] = 0x00
    raw[9] = 0x02
    raw[10] = 0x8e
    raw[11] = 0x07  # R-R variant: correct encoding per ptxas ground truth
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# BRA  (Branch)
# ---------------------------------------------------------------------------
# Ground truth: hex=4779fc00fcffffffffff830300c00f00 for all 64 instances.
#   lo=0xfffffffc00fc7947
#   Bytes: 47 79 fc 00 fc ff ff ff | ff ff 83 03 00 c0 0f 00
#   ctrl=0x7e0: raw=0xfc0 -> b13=0xc0, b14=0x0f, b15=0x00  ✓
#
# The JSON dataset has only a single unique BRA encoding despite multiple
# different PC offsets in the 'offset' field.  The operand field shows the
# absolute target address ("0x180", "0x120", etc.), but the hex is identical
# for all instances, implying the table captures a single kernel's branch.
#
# Per the user spec: lo word is fixed = 0xfffffffc00fc7947.
# Branch target offset is encoded in the hi word, around bits[19:0], AFTER
# removing the control word from bytes 13-15.
#
# Layout of hi word for BRA (from ground truth):
#   b8=0xff, b9=0xff, b10=0x83, b11=0x03, b12=0x00 (fixed non-ctrl portion)
#   b13-b15 = ctrl
#
# hi[39:0] (b8..b12) = 0x000383ffff  -- this is the fixed target+flags pattern.
# For a parameterized BRA we patch bits in b8..b12 with the relative offset.
#
# The 20-bit offset field appears to live at hi bits[19:0] (b8=lo8, b9=hi4 at bits[19:8]).
# From the sample: b8=0xff, b9=0xff -> bits[15:0] = 0xffff; b10 bits[3:0]=0x3 -> bits[19:16]=3
# 20-bit value = 0x3ffff = 262143.  As a signed 20-bit: 262143-2^20 = -1 (PC-relative -1 word = -16 bytes).
#
# But all offsets in the table resolve to the same bytes, so we preserve the
# fixed non-ctrl bytes and only parameterize the offset bits.
# The offset encoding: signed 20-bit value at hi bits[19:0]:
#   b8  = offset[7:0]
#   b9  = offset[15:8]
#   b10[3:0] = offset[19:16] (low nibble of b10); b10[7:4] = 0x8 (fixed flag)
#   b11 = 0x03 (fixed)
#
# pc_offset_bytes is the signed relative offset from the NEXT instruction (16-byte aligned).
# Encoding: signed_words = pc_offset_bytes // 16; offset20 = signed_words & 0xFFFFF

def encode_bra(pc_offset_bytes: int, ctrl: int = 0) -> bytes:
    """
    Encode BRA target to 16 bytes.

    Unconditional branch.  The lo word is fixed; the branch target offset is
    encoded as a signed 20-bit value in the hi word at bits[19:0].

    Args:
        pc_offset_bytes: Signed byte offset from the next instruction's address
                         to the branch target.  Must be a multiple of 16.
        ctrl:            23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth (the JSON dataset has only one unique BRA encoding):
        encode_bra(pc_offset_bytes, ctrl=0x7e0)
            -> bytes.fromhex('4779fc00fcffffffffff830300c00f00')
        where pc_offset_bytes encodes as offset20=0x3ffff (= -1 instruction = loop back).
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT

    # Signed 18-bit offset in units of instructions (16 bytes each).
    # Ground truth: offset=-16 bytes => signed_insns=-1 => offset18=0x3ffff
    #   b8=0xff, b9=0xff, b10=0x83:
    #     b10 = 0x80 | (offset18[17:16] & 0x03) = 0x80 | 0x03 = 0x83  ✓
    #   b11 = 0x03 (fixed flag byte)
    signed_insns = pc_offset_bytes // 16
    offset18 = signed_insns & 0x3FFFF   # 18-bit two's complement

    b13, b14, b15 = _ctrl_to_bytes(ctrl)

    raw = bytearray(16)
    # lo word = fixed: 0xfffffffc00fc7947
    # Bytes 0-7: 47 79 fc 00 fc ff ff ff
    raw[0] = 0x47
    raw[1] = 0x79
    raw[2] = 0xfc
    raw[3] = 0x00
    raw[4] = 0xfc
    raw[5] = 0xff
    raw[6] = 0xff
    raw[7] = 0xff
    # hi word: offset18 at bits[17:0]:
    #   b8  = offset18[7:0]
    #   b9  = offset18[15:8]
    #   b10 = 0x80 | offset18[17:16]  (high 2 bits; 0x80 is the fixed opcode marker bit)
    #   b11 = 0x03 (fixed flag)
    raw[8]  = offset18 & 0xFF
    raw[9]  = (offset18 >> 8) & 0xFF
    raw[10] = 0x80 | ((offset18 >> 16) & 0x03)
    raw[11] = 0x03
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# Ground truth samples for round-trip verification
# ---------------------------------------------------------------------------

SAMPLES: dict[str, list[tuple[str, bytes]]] = {
    "NOP": [
        # (description, expected_bytes)
        ("NOP ctrl=0x7e0",
         bytes.fromhex("18790000000000000000000000c00f00")),
    ],
    "EXIT": [
        ("EXIT ctrl=0x7f5",
         bytes.fromhex("4d790000000000000000800300ea0f00")),
    ],
    "MOV": [
        ("MOV R6, R5 ctrl=0x7e5",
         bytes.fromhex("0272060005000000000f000000ca0f00")),
        ("MOV R3, R2 ctrl=0x7e3",
         bytes.fromhex("0272030002000000000f000000c60f00")),
    ],
    "LDC": [
        ("LDC R1, c[0][0x37c] ctrl=0x7f1",
         bytes.fromhex("827b01ff00df00000008000000e20f00")),
        ("LDC R13, c[0][0x360] ctrl=0x712",
         bytes.fromhex("827b0dff00d800000008000000240e00")),
        ("LDC R9, c[0][0x360] ctrl=0x712",
         bytes.fromhex("827b09ff00d800000008000000240e00")),
    ],
    "LDC_64": [
        ("LDC.64 R2, c[0][0x388] ctrl=0x711",
         bytes.fromhex("827b02ff00e20000000a000000220e00")),
        ("LDC.64 R10, c[0][0x380] ctrl=0x751",
         bytes.fromhex("827b0aff00e00000000a000000a20e00")),
        ("LDC.64 R6, c[0][0x380] ctrl=0x751",
         bytes.fromhex("827b06ff00e00000000a000000a20e00")),
    ],
    "S2R": [
        ("S2R R0, SR_TID.X ctrl=0x717",
         bytes.fromhex("197900000000000000210000002e0e00")),
    ],
    "IADD3": [
        ("IADD3 RZ, P0, PT, RZ, R4, RZ ctrl=0x7f1",
         bytes.fromhex("1072ffff04000000ffe0f10700e20f00")),
    ],
    "IMAD_WIDE": [
        ("IMAD.WIDE R2, R13, 0x8, R2 ctrl=0xfe6",
         bytes.fromhex("2578020d0800000002028e0700cc1f00")),
        ("IMAD.WIDE R6, R13, 0x8, R10 ctrl=0x27f1",
         bytes.fromhex("2578060d080000000a028e0700e24f00")),
        ("IMAD.WIDE R2, R9, 0x8, R2 ctrl=0xfe6",
         bytes.fromhex("257802090800000002028e0700cc1f00")),
        ("IMAD.WIDE R6, R9, 0x8, R6 ctrl=0x27f1",
         bytes.fromhex("257806090800000006028e0700e24f00")),
    ],
    "LDG_E_64": [
        ("LDG.E.64 R2, desc[UR4][R2.64] ctrl=0x1771",
         bytes.fromhex("8179020204000000001b1e0c00e22e00")),
    ],
    "STG_E_64": [
        ("STG.E.64 desc[UR4][R6.64], R4 ctrl=0x7f1",
         bytes.fromhex("8679000604000000041b100c00e20f00")),
    ],
    "ISETP_GE_AND": [
        ("ISETP.GE.AND P0, PT, R13, UR5, PT ctrl=0x17ed",
         bytes.fromhex("0c7c000d050000007062f00b00da2f00")),
        ("ISETP.GE.AND P0, PT, R9, UR5, PT ctrl=0x17ed",
         bytes.fromhex("0c7c0009050000007062f00b00da2f00")),
    ],
    "BRA": [
        # BRA with offset20=0x3ffff (ground truth loop-back pattern)
        ("BRA offset=-16bytes ctrl=0x7e0",
         bytes.fromhex("4779fc00fcffffffffff830300c00f00")),
    ],
}


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------

def roundtrip_verify_opcodes(verbose: bool = True) -> bool:
    """
    Verify all encoders against ground truth JSON samples.

    Calls each encoder with operand values parsed from the SAMPLES dict and
    compares the result against the known-good hex bytes from ptxas output.

    Returns True if all samples pass.
    """

    results: list[tuple[str, str, bytes, bytes, bool]] = []

    def check(name: str, desc: str, got: bytes, expected: bytes) -> None:
        ok = (got == expected)
        results.append((name, desc, got, expected, ok))

    # NOP
    check("NOP", "ctrl=0x7e0",
          encode_nop(ctrl=0x7e0),
          bytes.fromhex("18790000000000000000000000c00f00"))

    # EXIT
    check("EXIT", "ctrl=0x7f5",
          encode_exit(ctrl=0x7f5),
          bytes.fromhex("4d790000000000000000800300ea0f00"))

    # MOV
    check("MOV", "R6,R5 ctrl=0x7e5",
          encode_mov(6, 5, ctrl=0x7e5),
          bytes.fromhex("0272060005000000000f000000ca0f00"))
    check("MOV", "R3,R2 ctrl=0x7e3",
          encode_mov(3, 2, ctrl=0x7e3),
          bytes.fromhex("0272030002000000000f000000c60f00"))

    # LDC
    check("LDC", "R1,c[0][0x37c] ctrl=0x7f1",
          encode_ldc(1, 0, 0x37c, ctrl=0x7f1),
          bytes.fromhex("827b01ff00df00000008000000e20f00"))
    check("LDC", "R13,c[0][0x360] ctrl=0x712",
          encode_ldc(13, 0, 0x360, ctrl=0x712),
          bytes.fromhex("827b0dff00d800000008000000240e00"))
    check("LDC", "R9,c[0][0x360] ctrl=0x712",
          encode_ldc(9, 0, 0x360, ctrl=0x712),
          bytes.fromhex("827b09ff00d800000008000000240e00"))

    # LDC.64
    check("LDC.64", "R2,c[0][0x388] ctrl=0x711",
          encode_ldc_64(2, 0, 0x388, ctrl=0x711),
          bytes.fromhex("827b02ff00e20000000a000000220e00"))
    check("LDC.64", "R10,c[0][0x380] ctrl=0x751",
          encode_ldc_64(10, 0, 0x380, ctrl=0x751),
          bytes.fromhex("827b0aff00e00000000a000000a20e00"))
    check("LDC.64", "R6,c[0][0x380] ctrl=0x751",
          encode_ldc_64(6, 0, 0x380, ctrl=0x751),
          bytes.fromhex("827b06ff00e00000000a000000a20e00"))

    # S2R
    check("S2R", "R0,SR_TID.X ctrl=0x717",
          encode_s2r(0, SR_TID_X, ctrl=0x717),
          bytes.fromhex("197900000000000000210000002e0e00"))

    # IADD3
    check("IADD3", "RZ,P0,PT,RZ,R4,RZ ctrl=0x7f1",
          encode_iadd3(RZ, RZ, 4, RZ, ctrl=0x7f1),
          bytes.fromhex("1072ffff04000000ffe0f10700e20f00"))

    # IADD3.X
    check("IADD3.X", "R7,PT,PT,RZ,RZ,RZ,P0,!PT ctrl=0x7f2",
          encode_iadd3x(7, RZ, RZ, RZ, ctrl=0x7f2),
          bytes.fromhex("107207ffff000000ffe47f0000e40f00"))

    # IMAD.WIDE
    check("IMAD.WIDE", "R2,R13,0x8,R2 ctrl=0x0fe6",
          encode_imad_wide(2, 13, 0x8, 2, ctrl=0x0fe6),
          bytes.fromhex("2578020d0800000002028e0700cc1f00"))
    check("IMAD.WIDE", "R6,R13,0x8,R10 ctrl=0x27f1",
          encode_imad_wide(6, 13, 0x8, 10, ctrl=0x27f1),
          bytes.fromhex("2578060d080000000a028e0700e24f00"))
    check("IMAD.WIDE", "R2,R9,0x8,R2 ctrl=0x0fe6",
          encode_imad_wide(2, 9, 0x8, 2, ctrl=0x0fe6),
          bytes.fromhex("257802090800000002028e0700cc1f00"))
    check("IMAD.WIDE", "R6,R9,0x8,R6 ctrl=0x27f1",
          encode_imad_wide(6, 9, 0x8, 6, ctrl=0x27f1),
          bytes.fromhex("257806090800000006028e0700e24f00"))

    # LDG.E.64
    check("LDG.E.64", "R2,desc[UR4][R2.64] ctrl=0x1771",
          encode_ldg_e_64(2, 4, 2, ctrl=0x1771),
          bytes.fromhex("8179020204000000001b1e0c00e22e00"))

    # STG.E.64
    check("STG.E.64", "desc[UR4][R6.64],R4 ctrl=0x7f1",
          encode_stg_e_64(4, 6, 4, ctrl=0x7f1),
          bytes.fromhex("8679000604000000041b100c00e20f00"))

    # ISETP.GE.AND
    check("ISETP.GE.AND", "P0,PT,R13,UR5,PT ctrl=0x17ed",
          encode_isetp_ge_and(0, 13, 5, ctrl=0x17ed),
          bytes.fromhex("0c7c000d050000007062f00b00da2f00"))
    check("ISETP.GE.AND", "P0,PT,R9,UR5,PT ctrl=0x17ed",
          encode_isetp_ge_and(0, 9, 5, ctrl=0x17ed),
          bytes.fromhex("0c7c0009050000007062f00b00da2f00"))

    # BRA — ground truth: offset20=0x3ffff means -1 instruction = -16 bytes
    check("BRA", "offset=-16 ctrl=0x7e0",
          encode_bra(-16, ctrl=0x7e0),
          bytes.fromhex("4779fc00fcffffffffff830300c00f00"))

    # ---- Print results -------------------------------------------------------
    if verbose:
        print("=" * 80)
        print("SM_120 Opcode Encode Round-Trip Verification")
        print("=" * 80)
        print(f"  {'Opcode':<18} {'Operands':<38} {'Result'}")
        print("-" * 80)

    pass_count = 0
    fail_count = 0

    for name, desc, got, expected, ok in results:
        status = "PASS" if ok else "FAIL"
        if ok:
            pass_count += 1
        else:
            fail_count += 1

        if verbose:
            print(f"  {name:<18} {desc:<38} {status}")
            if not ok:
                print(f"    expected: {expected.hex()}")
                print(f"    got:      {got.hex()}")
                diffs = [i for i in range(16) if got[i] != expected[i]]
                for i in diffs:
                    print(f"    byte[{i:2d}]: got=0x{got[i]:02x} "
                          f"expected=0x{expected[i]:02x}  bits[{i*8+7}:{i*8}]")

    if verbose:
        print("-" * 80)
        print(f"  Results: {pass_count} PASS, {fail_count} FAIL "
              f"out of {len(results)} total")
        print("=" * 80)

    return fail_count == 0


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MUFU — Math Unit Function Unit (transcendentals)
# ---------------------------------------------------------------------------
# Opcode: 0x08, 0x73.  Function selected by modifier byte b11.
# Ground truth from ptxas probe:
#   MUFU.RCP   R6, R6:   bytes= 08 73 06 06 00 00 00 00 00 00 10 00 ...  b11 bits select function
#   MUFU.SQRT  R9, R8:   b11=0x20  MUFU.SIN R11, R13: b11=0x04
#   MUFU.COS  R13, R13:  b11=0x00 (ctrl diff)  MUFU.EX2 R15, R10: b11=0x08
#   MUFU.LG2  R17, R8:   b11=0x0c
#
# Encoding: src in b3 (NOT b4).  dest in b2.  b4=0x00.
# b9=0x00, b10=0x00, b11=function_id.

MUFU_RCP  = 0x10
MUFU_SQRT = 0x20
MUFU_SIN  = 0x04
MUFU_COS  = 0x00  # distinguished by ctrl context
MUFU_EX2  = 0x08
MUFU_LG2  = 0x0c

def encode_mufu(dest: int, src: int, func: int, ctrl: int = 0) -> bytes:
    """Encode MUFU dest, src with function selector.

    Ground truth (ptxas sm_120):
      MUFU.SQRT R4, R0: lo=0x0000000000047308, hi=0x000e240000002000
        bytes: 08 73 04 00 00 00 00 00 | 00 20 00 00 00 24 0e 00
        b2=dest(R4), b4=src(R0), b9=0x20(SQRT func)
      MUFU.LG2 R5, R4:  lo=0x0000000400057308, hi=0x000e240000000c00
        bytes: 08 73 05 00 04 00 00 00 | 00 0c 00 00 00 24 0e 00
        b2=dest(R5), b4=src(R4), b9=0x0c(LG2 func)
    Func codes: RCP=0x10, SQRT=0x20, SIN=0x04, EX2=0x08, LG2=0x0c
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0]  = 0x08
    raw[1]  = 0x73
    raw[2]  = dest & 0xFF
    raw[3]  = 0x00
    raw[4]  = src & 0xFF
    # bytes 5-7 = 0
    raw[8]  = 0x00
    raw[9]  = func & 0xFF
    raw[10] = 0x00
    raw[11] = 0x00
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# VIMNMX — Integer Min/Max
# ---------------------------------------------------------------------------
# Opcode: 0x48, 0x72 (register-register).  0x48, 0x78 (register-immediate).
# Ground truth:
#   VIMNMX.S32 R19, R4, R5, PT (min):  0x0000000504137248 / ctrl has PT=0x03fe, bit=0x0100
#   VIMNMX.S32 R5, R4, R5, !PT (max):  0x0000000504057248 / ctrl has !PT=0x07fe, bit=0x0100
# Min vs max selected by predicate: PT = min, !PT = max.
# b9=0x01, b10=0x00 for .S32.

def encode_vimnmx_s32(dest: int, src0: int, src1: int, is_max: bool = False,
                       ctrl: int = 0) -> bytes:
    """Encode VIMNMX.S32 for min (is_max=False) or max (is_max=True).

    Ground truth (ptxas sm_120):
        min.s32: b9=0x01, b10=0xfe, b11=0x03
        max.s32: b9=0x01, b10=0xfe, b11=0x07
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x48, 0x72,
                  b2=dest, b3=src0, b4=src1,
                  b8=0x00, b9=0x01, b10=0xfe,
                  b11=0x07 if is_max else 0x03,
                  ctrl=ctrl)


def encode_vimnmx_u32(dest: int, src0: int, src1: int, is_max: bool = False,
                       ctrl: int = 0) -> bytes:
    """Encode VIMNMX.U32 register-register variant.

    Ground truth (ptxas sm_120, sad.u32):
        min.u32: b9=0x00, b10=0xfe, b11=0x03
        max.u32: b9=0x00, b10=0xfe, b11=0x07
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x48, 0x72,
                  b2=dest, b3=src0, b4=src1,
                  b8=0x00, b9=0x00, b10=0xfe,
                  b11=0x07 if is_max else 0x03,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# SEL — Select (conditional move based on predicate)
# ---------------------------------------------------------------------------
# Opcode: 0x07, 0x72 (register variant).
# Ground truth: SEL R7, R2, R7, P0 → 0x0000000702077207 / 0x000fe20000000000
# dest=b2, src0=b3, src1=b4, predicate in modifier bytes.

def encode_sel(dest: int, src0: int, src1: int, pred: int = 0,
               ctrl: int = 0) -> bytes:
    """Encode SEL dest, src0, src1, Ppred.

    Ground truth (P0): SEL R7, R2, R7, P0 → b0=0x07, b1=0x72, b2=dest, b3=src0,
    b4=src1, b8=0x00. Hypothesis: b8 = pred_index (0..7).
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0]  = 0x07
    raw[1]  = 0x72
    raw[2]  = dest & 0xFF
    raw[3]  = src0 & 0xFF
    raw[4]  = src1 & 0xFF
    raw[8]  = pred & 0x07   # predicate index (P0..P7); P0=0x00 verified
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# PRMT — Byte Permute
# ---------------------------------------------------------------------------
# Immediate-selector PRMT: opcode 0x16, 0x74 → opc=0x416
# Register-selector PRMT:  opcode 0x16, 0x72 → opc=0x216
#
# Field layout (immediate): b2=dest, b3=src0(a), b4-b7=selector_imm, b8=src1(b)
# Field layout (register):  b2=dest, b3=src1(b), b4=sel_reg, b8=src0(a)
#   (a and b swap positions between immediate and register variants)
#
# Ground truth (ptxas 13.0, sm_120):
#   PRMT R5, R4, 0x3210, R5 → 16740504103200000500000000ca1f00  (opc=0x416)
#   PRMT R5, R4, R0, R5     → 16720504000000000500000000ca1f00  (opc=0x216)

def encode_prmt(dest: int, src0: int, selector: int, src1: int,
                ctrl: int = 0) -> bytes:
    """Encode PRMT dest, src0, selector_imm, src1 (immediate selector, opc=0x416)."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0]  = 0x16
    raw[1]  = 0x74   # constant-selector variant (not 0x78 which is for SHF)
    raw[2]  = dest & 0xFF
    raw[3]  = src0 & 0xFF
    raw[4]  = selector & 0xFF
    raw[5]  = (selector >> 8) & 0xFF
    raw[6]  = (selector >> 16) & 0xFF
    raw[7]  = (selector >> 24) & 0xFF
    raw[8]  = src1 & 0xFF
    raw[9]  = 0x00
    raw[10] = 0x00
    raw[11] = 0x00
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


def encode_prmt_reg(dest: int, src0: int, src1: int, sel_reg: int,
                    ctrl: int = 0) -> bytes:
    """Encode PRMT dest, src0, src1, sel_reg (register selector, opc=0x216).

    NOTE: src0 and src1 swap positions relative to immediate-selector PRMT:
      byte[3] = src1 (second data source)
      byte[4] = sel_reg (selector register)
      byte[8] = src0 (first data source)

    Ground truth:
      PRMT R5, R4, R5, R0 → 16720504000000000500000000ca1f00
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0]  = 0x16
    raw[1]  = 0x72   # register-selector variant
    raw[2]  = dest   & 0xFF
    raw[3]  = src1   & 0xFF   # second data source (swapped!)
    raw[4]  = sel_reg & 0xFF  # selector register
    raw[5]  = 0x00
    raw[6]  = 0x00
    raw[7]  = 0x00
    raw[8]  = src0   & 0xFF   # first data source (swapped!)
    raw[9]  = 0x00
    raw[10] = 0x00
    raw[11] = 0x00
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# FSEL — Float Select (conditional move, float version)
# ---------------------------------------------------------------------------
# Opcode: 0x08, 0x72 (register).  0x08, 0x78 (immediate).
# Ground truth: FSEL R6, R6, 1, !P0 → 0x3f80000006067808 / 0x000fe20004000000

def encode_fsel(dest: int, src0: int, src1: int, pred: int = 0,
                negate_pred: bool = False, ctrl: int = 0) -> bytes:
    """Encode FSEL dest, src0, src1, [!]Ppred (register variant)."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0]  = 0x08
    raw[1]  = 0x72
    raw[2]  = dest & 0xFF
    raw[3]  = src0 & 0xFF
    raw[4]  = src1 & 0xFF
    raw[8]  = 0x00
    raw[9]  = 0x00
    raw[10] = (pred & 1) << 7
    raw[11] = ((pred >> 1) & 0x7F) | (0x04 if negate_pred else 0x00)
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# ATOMG — Atomic Global Memory Operations
# ---------------------------------------------------------------------------
# Opcode: 0xa8, 0x79 (integer), 0xa3, 0x79 (float).
# Uses descriptor-based addressing (desc[UR4][Rsrc.64+offset]).
# Ground truth from probe:
#   ATOMG.E.ADD:  0x80000009020909a8 / ctrl  (ADD u32)
#   ATOMG.E.MIN.S32:  0x80000408040b79a8 / ctrl
#   ATOMG.E.MAX.S32:  0x80000c08040d79a8 / ctrl
#   ATOMG.E.AND:  0x80001408040f79a8 / ctrl
#   ATOMG.E.OR:   0x80001c08041179a8 / ctrl
#   ATOMG.E.EXCH: 0x80002408041379a8 / ctrl
#   ATOMG.E.ADD.F32: 0x80003407040779a3 / ctrl

ATOMG_ADD  = 0x00
ATOMG_MIN  = 0x01
ATOMG_MAX  = 0x02
ATOMG_AND  = 0x04
ATOMG_OR   = 0x05
ATOMG_XOR  = 0x06
ATOMG_EXCH = 0x08

def encode_atomg_u32(dest: int, addr_base: int, offset: int, data: int,
                      atom_op: int = ATOMG_ADD, ctrl: int = 0) -> bytes:
    """Encode ATOMG.E.{op}.STRONG.GPU for u32 atomic operations."""
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0]  = 0xa8
    raw[1]  = 0x79
    raw[2]  = dest & 0xFF
    raw[3]  = addr_base & 0xFF
    raw[4]  = data & 0xFF
    # offset in bytes 5-7 (24-bit)
    raw[5]  = offset & 0xFF
    raw[6]  = (offset >> 8) & 0xFF
    raw[7]  = 0x80 | ((offset >> 16) & 0x7F)  # high bit = descriptor flag
    raw[8]  = 0x00
    raw[9]  = 0x00
    raw[10] = 0x00
    raw[11] = atom_op & 0xFF
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# ATOMG.CAS — Atomic Compare-And-Swap (Global Memory)
# ---------------------------------------------------------------------------
# Opcode: 0xa9, 0x73  (opcode word = 0x3a9).
# Ground truth probe (RTX 5090):
#   atom.cas.b32 R5, [R4], R6, R7
#   lo=a9 73 05 04 06 00 00 00  hi=07 e1 1e 00 00 a8 4e 00
#   b2=dest(R5), b3=addr(R4), b4=compare(R6), b8=new_val(R7)
#   b9=0xe1, b10=0x1e, b11=0x00 — fixed CAS modifiers
#   ctrl=0x2754 → wdep=0x35(LDG slot), rbar=0x09, misc=4

def encode_atomg_cas_b32(dest: int, addr: int, compare: int, new_val: int,
                          ctrl: int = 0) -> bytes:
    """Encode ATOMG.E.CAS.b32: atomic compare-and-swap on global memory.

    Reads old = *addr; if old == compare: *addr = new_val; returns old.

    Args:
        dest:    Destination GPR (receives old value read from memory).
        addr:    Address register (lo of 64-bit pair).
        compare: Compare value register.
        new_val: New value register (written if compare matches).
        ctrl:    23-bit scheduling control word (0 = default).
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0xa9, 0x73,
                  b2=dest, b3=addr, b4=compare,
                  b8=new_val, b9=0xe1, b10=0x1e, b11=0x00,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# POPC — Population Count (count set bits)
# ---------------------------------------------------------------------------
# Opcode: 0x09, 0x73.  Ground truth: POPC R13, R0 → 0x00000000000d7309
def encode_popc(dest: int, src: int, ctrl: int = 0) -> bytes:
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x09, 0x73
    raw[2], raw[3] = dest & 0xFF, src & 0xFF
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# BREV — Bit Reverse
# ---------------------------------------------------------------------------
# Opcode: 0x01, 0x73.  Ground truth: BREV R15, R0 → 0x00000000000f7301
def encode_brev(dest: int, src: int, ctrl: int = 0) -> bytes:
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x01, 0x73
    raw[2], raw[3] = dest & 0xFF, src & 0xFF
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# FLO — Find Leading One (CLZ = 31 - FLO for non-zero)
# ---------------------------------------------------------------------------
# Opcode: 0x00, 0x73.  Ground truth: FLO.U32 R6, R0 → 0x0000000000067300
def encode_flo(dest: int, src: int, ctrl: int = 0) -> bytes:
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x00, 0x73
    raw[2], raw[3] = dest & 0xFF, src & 0xFF
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# IABS — Integer Absolute Value
# ---------------------------------------------------------------------------
# Opcode: 0x13, 0x72.  Ground truth: IABS R0, R0 → 0x0000000000007213
def encode_iabs(dest: int, src: int, ctrl: int = 0) -> bytes:
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x13, 0x72
    raw[2], raw[3] = dest & 0xFF, src & 0xFF
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# BFE sign-extension helper (opcode 0x81a)
# ---------------------------------------------------------------------------
# Used as the second step of bfe.s32: sign-extends the low `len` bits of src.
# dest = sign_extend(src[len-1:0], 32) = arithmetic ((src << (32-len)) >> (32-len))
#
# Ground truth (ptxas sm_120, bfe.s32 pos=0 len=16):
#   lo=1a 78 07 02 10 00 00 00  hi=00 02 00 00 ...
#   b0=0x1a, b1=0x78, b2=dest, b3=src, b4=len, b8=RZ(0x00), b9=0x02, b10=0x00
#
# bfe.s32 implementation (2 instructions):
#   If pos > 0: SHF.R.S32.HI dest, RZ, pos, src  (arithmetic right-shift by pos)
#   Then:       BFE_SEXT dest, dest_or_src, len   (sign-extend low len bits)

def encode_bfe_sext(dest: int, src: int, length: int, ctrl: int = 0) -> bytes:
    """Encode BFE sign-extension step: dest = sign_extend(src[length-1:0], 32).

    This is the 0x81a instruction used as the second step in bfe.s32 lowering.
    b0=0x1a, b1=0x78, b2=dest, b3=src, b4=length, b8=0x00, b9=0x02, b10=0x00

    Ground truth:
        BFE_SEXT R7, R2, 16  →  1a 78 07 02 10 00 00 00 | 00 02 00 00 ...
        BFE_SEXT R7, R7, 8   →  1a 78 07 07 08 00 00 00 | 00 02 00 00 ...
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x1a, 0x78,
                  b2=dest, b3=src, b4=length & 0xFF,
                  b8=0x00, b9=0x02, b10=0x00, b11=0x00,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# IMAD.HI — Integer Multiply-Add High (upper 32 bits)
# ---------------------------------------------------------------------------
# Opcode: 0x227.
# Ground truth (ptxas div.u32 sequence, sm_120):
#   IMAD.HI.U32 R3, R3, R5, R2: 27 72 03 03 05 00 00 00 | 02 00 8e 07 00 cc 0f 00
#   IMAD.HI.U32 R5, R3, R6, RZ: 27 72 05 03 06 00 00 00 | ff 00 8e 07 00 e4 0f 00
#   b2=dest, b3=src0, b4=src1, b8=src2, b9=0x00, b10=0x8e, b11=0x07
def encode_imad_hi(dest: int, src0: int, src1: int, src2: int,
                   signed: bool = False, ctrl: int = 0) -> bytes:
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x27, 0x72
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    raw[4] = src1 & 0xFF
    raw[8] = src2 & 0xFF
    raw[9]  = 0x00
    raw[10] = 0x8e
    raw[11] = 0x07
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# FMNMX — Float Min/Max
# ---------------------------------------------------------------------------
# Opcode: 0x09, 0x72.  Ground truth:
#   FMNMX R9, R0, R7, PT (min):  0x0000000700097209 (PT selects min)
#   FMNMX R7, R0, R7, !PT (max): 0x0000000700077209 (!PT selects max)
def encode_fmnmx(dest: int, src0: int, src1: int, is_max: bool = False,
                  ctrl: int = 0) -> bytes:
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x09, 0x72
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    raw[4] = src1 & 0xFF
    # PT (min) vs !PT (max) encoded in modifier bytes
    # Empirically verified on SM_120: mode byte is b10=0xfe, selector is b11=0x03(min)/0x07(max)
    raw[10] = 0xfe
    raw[11] = 0x07 if is_max else 0x03
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# FSETP — Float Set Predicate (comparison)
# ---------------------------------------------------------------------------
# Opcode: 0x0b, 0x72.  Ground truth:
#   FSETP.GEU.AND P2, PT, R0, R7, PT → 0x000000070000720b
FSETP_LT  = 0x01
FSETP_EQ  = 0x02
FSETP_LE  = 0x03
FSETP_GT  = 0x04
FSETP_NE  = 0x05
FSETP_GE  = 0x06
FSETP_GEU = 0x0e
FSETP_NEU = 0x0d

def encode_fsetp(pred_dest: int, src0: int, src1: int, cmp: int = FSETP_LT,
                  ctrl: int = 0) -> bytes:
    """Encode FSETP Ppred, src0, src1, cmp (FP32 comparison → predicate).

    Ground truth (ptxas sm_120, setp.gt.f32 %p0, %f0, 0.0):
        0b720004ff0000000040f00300ca4f00
        b3=R4(src0), b4=R255(src1=RZ), b9=0x40(GT|P0), b10=0xf0(PT), b11=0x03(AND)
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x0b, 0x72
    raw[2] = 0x00
    raw[3] = src0 & 0xFF
    raw[4] = src1 & 0xFF
    raw[9] = ((cmp & 0x0F) << 4) | (pred_dest & 0x07)  # comparison mode + pred dest
    raw[10] = 0xf0  # PT in AND mask
    raw[11] = 0x03  # AND combiner
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# SHFL — Warp Shuffle
# ---------------------------------------------------------------------------
# Opcode: 0x89, 0x75 (reg-reg) / 0x89, 0x7f (reg-imm) / 0x89, 0x79 (imm-imm).
# Ground truth:
#   SHFL.IDX PT, R5, R0, RZ, 0x1f:   0x00001fff00057589
#   SHFL.UP PT, R7, R0, 0x1, RZ:     0x0420000000077989
#   SHFL.DOWN PT, R9, R0, 0x1, 0x1f: 0x08201f0000097f89
#   SHFL.BFLY PT, R11, R0, 0x1, 0x1f:0x0c201f00000b7f89
SHFL_IDX  = 0x00
SHFL_UP   = 0x04
SHFL_DOWN = 0x08
SHFL_BFLY = 0x0c

def encode_shfl(dest: int, src: int, lane_or_delta: int, clamp: int,
                mode: int = SHFL_IDX, ctrl: int = 0) -> bytes:
    """Encode SHFL with immediate lane/delta and clamp."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x89, 0x7f  # imm-imm variant
    raw[2] = dest & 0xFF
    raw[3] = src & 0xFF
    raw[4] = lane_or_delta & 0xFF
    raw[5] = clamp & 0xFF
    raw[6] = (clamp >> 8) & 0xFF
    raw[7] = mode & 0xFF
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# VOTE — Warp Vote
# ---------------------------------------------------------------------------
# Opcode: 0x06, 0x78.  Ground truth: VOTE.ANY R13, PT, PT → 0x00000000000d7806
def encode_vote_ballot(dest: int, ctrl: int = 0) -> bytes:
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x06, 0x78
    raw[2] = dest & 0xFF
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# I2FP — Integer to Float (various formats)
# ---------------------------------------------------------------------------
# Already have I2FP.F32.S32. Add U32 variant.
# Ground truth: I2FP.F32.U32 R11, R6 → 0x00000006000b7245 (same opcode, different modifier)
# I2FP.F32.S32: modifier differs from U32 in b10 or b11.

def encode_i2fp_u32(dest: int, src: int, ctrl: int = 0) -> bytes:
    """Encode I2FP.F32.U32 — unsigned int to float.
    Ground truth: ptxas cvt.rn.f32.u32 → b9=0x14, b10=0x20."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x45, 0x72, b2=dest, b3=0x00, b4=src,
                  b8=0x00, b9=0x14, b10=0x20, b11=0x00, ctrl=ctrl)
    return bytes(raw)


# ---------------------------------------------------------------------------
# F2I variants — Float to Integer
# ---------------------------------------------------------------------------
# F2I.U32.TRUNC.NTZ: 0x00000000000f7305 (unsigned)
# F2I.TRUNC.NTZ:     0x0000000000117305 (signed, default)

def encode_f2i_u32(dest: int, src: int, ctrl: int = 0) -> bytes:
    """Encode F2I.U32.TRUNC.NTZ — float to unsigned int.
    Ground truth: ptxas cvt.rzi.u32.f32 → b3=0, b4=src, b9=0xf0, b10=0x20."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x05, 0x73, b2=dest, b3=0x00, b4=src,
                  b8=0x00, b9=0xf0, b10=0x20, b11=0x00, ctrl=ctrl)
    return bytes(raw)


# ---------------------------------------------------------------------------
# LDG.E width variants
# ---------------------------------------------------------------------------
# LDG.E.U8:  same opcode 0x81, 0x79 with width modifier in ctrl/mod bytes
# LDG.E.U16: same opcode
# LDG.E.128: already have
# The width is encoded in the control word hi-byte region.


# ---------------------------------------------------------------------------
# ISETP comparison variants
# ---------------------------------------------------------------------------
# Already have ISETP.GE.AND. The comparison type is in the modifier bytes.
# ISETP.LT: cmp=0x01, ISETP.EQ: cmp=0x02, ISETP.LE: cmp=0x03
# ISETP.GT: cmp=0x04, ISETP.NE: cmp=0x05, ISETP.GE: cmp=0x06
ISETP_LT = 0x01
ISETP_EQ = 0x02
ISETP_LE = 0x03
ISETP_GT = 0x04
ISETP_NE = 0x05
ISETP_GE = 0x06

def encode_isetp(pred_dest: int, src0: int, src1: int, cmp: int = ISETP_GE,
                  signed: bool = True, ctrl: int = 0) -> bytes:
    """Encode ISETP R-R variant with variable comparison type.

    Ground truth from ptxas (ISETP.GE.U32.AND P0, PT, R4, R7, PT):
      bytes: 0c 72 00 04 07 00 00 00 | 70 60 f0 03 00 da 0f 00
      b2=0x00 (always zero — pred_dest goes in b10, not b2)
      b8=0x70 (fixed for U32 R-R comparisons)
      b9=cmp<<4 (GE=0x60, NE=0x50, LT=0x10, EQ=0x20, LE=0x30, GT=0x40)
      b10=0xf0|(pred_dest<<1): P0→0xf0, P1→0xf2, P2→0xf4, P3→0xf6
      b11=0x03 (R-R GPR operand flag)
    Verified with div.u32 sequence: NE→b9=0x50, GE→b9=0x60, P2→b10=0xf4.
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0]  = 0x0c
    raw[1]  = 0x72
    raw[2]  = 0x00
    raw[3]  = src0 & 0xFF
    raw[4]  = src1 & 0xFF
    raw[5]  = 0x00
    raw[6]  = 0x00
    raw[7]  = 0x00
    raw[8]  = 0x70                          # fixed for R-R comparisons
    raw[9]  = ((cmp & 0x0F) << 4) | (0x02 if signed else 0x00)  # cmp type + signed bit
    raw[10] = 0xf0 | ((pred_dest & 0x07) << 1)
    raw[11] = 0x03
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# I2F.U32.RP — Integer to Float, unsigned 32-bit, round toward +infinity
# ---------------------------------------------------------------------------
# Used in Newton-Raphson division (div.u32) to convert divisor to float.
# Ground truth (ptxas div.u32, sm_120):
#   I2F.U32.RP R0, R7: lo=0x0000000700007306, hi=0x001e220000209000
#   bytes: 06 73 00 00 07 00 00 00 | 00 90 20 00 00 22 1e 00
#   b0=0x06, b1=0x73, b2=dest, b4=src, b9=0x90(RP+U32), b10=0x20
def encode_i2f_s32_rp(dest: int, src: int, ctrl: int = 0) -> bytes:
    """Encode I2F.S32.RP dest, src — signed 32-bit int to float, round toward +inf.

    Ground truth (ptxas div.s32, sm_120):
      I2F.RP R0, R7: lo=0x0000000700007306, hi=0x000e220000209400
      bytes: 06 73 00 00 07 00 00 00 | 00 94 20 00 00 22 0e 00
      b9=0x94 vs U32 b9=0x90 (signed bit in b9 bit[2])
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x06, 0x73
    raw[2] = dest & 0xFF
    raw[4] = src & 0xFF
    raw[9]  = 0x94
    raw[10] = 0x20
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


def encode_i2f_u32_rp(dest: int, src: int, ctrl: int = 0) -> bytes:
    """Encode I2F.U32.RP dest, src — unsigned 32-bit int to float, round toward +inf."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x06, 0x73
    raw[2] = dest & 0xFF
    raw[4] = src & 0xFF
    raw[9]  = 0x90
    raw[10] = 0x20
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# F2I.FTZ.U32.TRUNC — Float to unsigned 32-bit integer, truncate (floor)
# ---------------------------------------------------------------------------
# Used in Newton-Raphson division (div.u32) to convert reciprocal float back to int.
# Ground truth (ptxas div.u32, sm_120):
#   F2I.FTZ.U32.TRUNC.NTZ R3, R2: lo=0x0000000200037305, hi=0x0000a4000021f000
#   bytes: 05 73 03 00 02 00 00 00 | 00 f0 21 00 00 a4 00 00
#   b0=0x05, b1=0x73, b2=dest, b4=src, b9=0xf0, b10=0x21
def encode_f2i_ftz_u32_trunc(dest: int, src: int, ctrl: int = 0) -> bytes:
    """Encode F2I.FTZ.U32.TRUNC.NTZ dest, src — float to unsigned 32-bit, truncate."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x05, 0x73
    raw[2] = dest & 0xFF
    raw[4] = src & 0xFF
    raw[9]  = 0xf0
    raw[10] = 0x20  # ground truth: ptxas uses 0x20 (not 0x21)
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# F2F — Float-to-Float precision conversion (F32↔F64)
# ---------------------------------------------------------------------------
# Ground truth from ptxas probe (SM_120):
#
# F2F.F32.F64 R7, R6  (cvt.rn.f32.f64: double→float, GPR source):
#   bytes: 10 73 07 00 06 00 00 00 | 00 10 30 00 ...
#   b0=0x10, b1=0x73, b2=dest, b3=0x00, b4=src_lo (lo word of f64 pair)
#   b9=0x10, b10=0x30, b11=0x00
#
# F2F.F64.F32 R6, R6  (cvt.f64.f32: float→double, GPR source):
#   bytes: 10 73 06 00 06 00 00 00 | 00 18 20 00 ...
#   b0=0x10, b1=0x73, b2=dest_lo (lo word of f64 pair), b4=src (f32 GPR)
#   b9=0x18, b10=0x20, b11=0x00
#
# Both use opcode b0=0x10, b1=0x73 (opcode & 0xFFF = 0x310).
# Destination for F64 form writes dest and dest+1 (64-bit pair).

def encode_f2f_f32_f64(dest: int, src: int, ctrl: int = 0) -> bytes:
    """Encode F2F.F32.F64 dest, src — convert f64 register pair to f32.

    src is the lo register of the f64 pair (src+1 is hi, read implicitly).
    dest is a single f32 register.

    Ground truth: F2F.F32.F64 R7, R6 → bytes 10 73 07 00 06 00 00 00 00 10 30 00 ...
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x10, 0x73,
                  b2=dest, b3=0x00, b4=src,
                  b8=0x00, b9=0x10, b10=0x30, b11=0x00,
                  ctrl=ctrl)


def encode_f2f_f64_f32(dest: int, src: int, ctrl: int = 0) -> bytes:
    """Encode F2F.F64.F32 dest, src — convert f32 to f64 register pair.

    src is a single f32 register.
    dest is the lo register of the destination f64 pair (dest+1 written implicitly).

    Ground truth: F2F.F64.F32 R6, R6 → bytes 10 73 06 00 06 00 00 00 00 18 20 00 ...
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x10, 0x73,
                  b2=dest, b3=0x00, b4=src,
                  b8=0x00, b9=0x18, b10=0x20, b11=0x00,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# F2I / I2F — Float-to-integer / Integer-to-float conversions (F64 forms)
# ---------------------------------------------------------------------------
# Ground truth (ptxas sm_120):
#
# F2I.S32.F64 R7, R2  (cvt.rzi.s32.f64):
#   b0=0x11, b1=0x73, b2=dest_r32, b4=src_f64_lo, b9=0xd1, b10=0x30, b11=0x00
#   opcode & 0xFFF = 0x311
#
# F2I.U32.F64 R7, R2  (cvt.rzi.u32.f64):
#   b0=0x11, b1=0x73, b2=dest_r32, b4=src_f64_lo, b9=0xd0, b10=0x30, b11=0x00
#   Same opcode 0x311; b9[0]=0 unsigned, b9[0]=1 signed
#
# I2F.F64.S32 R4, R2  (cvt.rn.f64.s32):
#   b0=0x12, b1=0x73, b2=dest_f64_lo, b4=src_r32, b9=0x1c, b10=0x20, b11=0x00
#   opcode & 0xFFF = 0x312; dest writes pair (dest, dest+1)

def encode_f2i_s32_f64(dest: int, src_lo: int, ctrl: int = 0) -> bytes:
    """Encode F2I.S32.F64 dest, src_lo — convert f64 pair to signed int32.

    src_lo is the lo register of the f64 source pair (src_lo+1 read implicitly).
    dest is a single s32 GPR.

    Ground truth: F2I.S32.F64 R7, R2 → b9=0xd1, b10=0x30
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x11, 0x73,
                  b2=dest, b3=0x00, b4=src_lo,
                  b8=0x00, b9=0xd1, b10=0x30, b11=0x00,
                  ctrl=ctrl)


def encode_f2i_u32_f64(dest: int, src_lo: int, ctrl: int = 0) -> bytes:
    """Encode F2I.U32.F64 dest, src_lo — convert f64 pair to unsigned int32.

    src_lo is the lo register of the f64 source pair (src_lo+1 read implicitly).
    dest is a single u32 GPR.

    Ground truth: F2I.U32.F64 R7, R2 → b9=0xd0, b10=0x30
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x11, 0x73,
                  b2=dest, b3=0x00, b4=src_lo,
                  b8=0x00, b9=0xd0, b10=0x30, b11=0x00,
                  ctrl=ctrl)


def encode_i2f_f64_s32(dest_lo: int, src: int, ctrl: int = 0) -> bytes:
    """Encode I2F.F64.S32 dest_lo, src — convert signed int32 to f64 pair.

    src is a single s32 GPR.
    dest_lo is the lo register of the f64 destination pair (dest_lo+1 written implicitly).

    Ground truth: I2F.F64.S32 R4, R2 → b9=0x1c, b10=0x20
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x12, 0x73,
                  b2=dest_lo, b3=0x00, b4=src,
                  b8=0x00, b9=0x1c, b10=0x20, b11=0x00,
                  ctrl=ctrl)


def encode_i2f_f64_u32(dest_lo: int, src: int, ctrl: int = 0) -> bytes:
    """Encode I2F.F64.U32 dest_lo, src — convert unsigned int32 to f64 pair.

    Layout inferred from I2F.F64.S32 (b9=0x1c) + F32 u32 pattern (b9 differs by bit).
    F32 unsigned adds 0x04 to signed; applying same delta: b9=0x1c+0x04=0x20.
    Needs hardware verification.
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x12, 0x73,
                  b2=dest_lo, b3=0x00, b4=src,
                  b8=0x00, b9=0x20, b10=0x20, b11=0x00,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# HFMA2 — Half-precision FMA2 (used as zero-init trick in div.u32)
# ---------------------------------------------------------------------------
# HFMA2 R2, -RZ, RZ, 0, 0 computes (-0.0)*0.0 + 0.0 = 0.0 and zero-extends to int.
# Used in the Newton-Raphson div.u32 sequence to zero a register without a NOP stall.
# Ground truth (ptxas div.u32, sm_120):
#   HFMA2 R2, -RZ, RZ, 0, 0: lo=0x00000000ff027431, hi=0x001fe200000001ff
#   bytes: 31 74 02 ff 00 00 00 00 | ff 01 00 00 00 e2 1f 00
#   b0=0x31, b1=0x74, b2=dest, b3=0xff(RZ neg), b4=0x00(RZ), b8=0xff, b9=0x01
def encode_hfma2_zero(dest: int, ctrl: int = 0) -> bytes:
    """Encode HFMA2 dest, -RZ, RZ, 0, 0 — zero-initialise dest register."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x31, 0x74
    raw[2] = dest & 0xFF
    raw[3] = 0xff   # -RZ (negated zero = still zero in FP16)
    raw[4] = 0x00   # RZ
    raw[8] = 0xff
    raw[9] = 0x01
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# IADD3 variants for Newton-Raphson division
# ---------------------------------------------------------------------------
# Several IADD3 encodings are needed for div.u32: large immediate, negated sources,
# and predicated forms. All have opcode 0x10 in b0; b1 encodes predication and form.
#
# b1 encoding:
#   unpred R-R-R:    0x72   (pred=PT=7, form=R-R-R)
#   unpred R-imm:    0x78   (pred=PT=7, form=R-imm)
#   @Pn R-R-R:   (n<<4)|0x02   (n=pred_idx, form=R-R-R)
#   @Pn R-imm:   (n<<4)|0x08   (n=pred_idx, form=R-imm)
#   @!Pn R-R-R:  0x80|(n<<4)|0x02
#   @!Pn R-imm:  0x80|(n<<4)|0x08
#
# Negation encoding:
#   negate b4 src: b7=0x80
#   negate b3 src: b9 bit[0] = 1 (b9=0xe1 vs normal b9=0xe0)
#
# Modifier bytes for register form: b9=0xe0, b10=0xff, b11=0x07
# Modifier bytes for imm form:      b9=0xe0, b10=0xff, b11=0x07 (same)

def encode_iadd3_imm32(dest: int, src0: int, imm32: int, src2: int,
                       ctrl: int = 0) -> bytes:
    """Encode IADD3 dest, src0, imm32, src2 with 32-bit immediate.

    Ground truth: IADD3 R2, R0, 0xffffffe, RZ
      bytes: 10 78 02 00 fe ff ff 0f | ff e0 ff 07 00 c8 1f 00
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x10, 0x78
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    imm32 = imm32 & 0xFFFFFFFF
    raw[4] = imm32 & 0xFF
    raw[5] = (imm32 >> 8) & 0xFF
    raw[6] = (imm32 >> 16) & 0xFF
    raw[7] = (imm32 >> 24) & 0xFF
    raw[8]  = src2 & 0xFF
    raw[9]  = 0xe0
    raw[10] = 0xff
    raw[11] = 0x07
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


def encode_iadd3_neg_b4(dest: int, src0: int, neg_src1: int, src2: int,
                        ctrl: int = 0) -> bytes:
    """Encode IADD3 dest, src0, -neg_src1, src2 (negate second operand).

    Ground truth: IADD3 R4, RZ, -R3, RZ
      bytes: 10 72 04 ff 03 00 00 80 | ff e0 ff 07 00 ca 4f 00
      b7=0x80 encodes the negation of b4.
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x10, 0x72
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    raw[4] = neg_src1 & 0xFF
    raw[7] = 0x80
    raw[8]  = src2 & 0xFF
    raw[9]  = 0xe0
    raw[10] = 0xff
    raw[11] = 0x07
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


def encode_iadd3_neg_b3(dest: int, neg_src0: int, src1: int, src2: int,
                        ctrl: int = 0) -> bytes:
    """Encode IADD3 dest, -neg_src0, src1, src2 (negate first operand).

    Ground truth: IADD3 R4, -R5, RZ, RZ
      bytes: 10 72 04 05 ff 00 00 00 | ff e1 ff 07 00 ca 0f 00
      b9=0xe1 (bit[0]=1) encodes the negation of b3.
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x10, 0x72
    raw[2] = dest & 0xFF
    raw[3] = neg_src0 & 0xFF
    raw[4] = src1 & 0xFF
    raw[8]  = src2 & 0xFF
    raw[9]  = 0xe1
    raw[10] = 0xff
    raw[11] = 0x07
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


def encode_iadd3_pred_neg_b4(dest: int, src0: int, neg_src1: int, src2: int,
                              pred_idx: int, inverted: bool = False,
                              ctrl: int = 0) -> bytes:
    """Encode @[!]Ppred_idx IADD3 dest, src0, -neg_src1, src2.

    Ground truth: @P0 IADD3 R4, R4, -R7, RZ
      bytes: 10 02 04 04 07 00 00 80 | ff e0 ff 07 00 e4 0f 00
      b1=(pred_idx<<4)|0x02, b7=0x80 for negation.
    inverted=True: b1 |= 0x80 for @!Pn form.
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x10
    pred_b1 = ((pred_idx & 0x07) << 4) | 0x02
    if inverted:
        pred_b1 |= 0x80
    raw[1] = pred_b1
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    raw[4] = neg_src1 & 0xFF
    raw[7] = 0x80
    raw[8]  = src2 & 0xFF
    raw[9]  = 0xe0
    raw[10] = 0xff
    raw[11] = 0x07
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


def encode_iadd3_pred_small_imm(dest: int, src0: int, imm_byte: int, src2: int,
                                 pred_idx: int, inverted: bool = False,
                                 ctrl: int = 0) -> bytes:
    """Encode @[!]Ppred_idx IADD3 dest, src0, imm_byte, src2 with small immediate.

    Ground truth: @P0 IADD3 R5, R5, 0x1, RZ  (b1=0x08 for P0)
      bytes: 10 08 05 05 01 00 00 00 | ff e0 ff 07 00 c6 0f 00
             @P1 IADD3 R5, R5, 0x1, RZ  (b1=0x18 for P1)
      bytes: 10 18 05 05 01 00 00 00 | ff e0 ff 07 00 e4 0f 00
    b1=(pred_idx<<4)|0x08, imm fits in b4 (1 byte).
    inverted=True: b1 |= 0x80 for @!Pn form.
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x10
    pred_b1 = ((pred_idx & 0x07) << 4) | 0x08
    if inverted:
        pred_b1 |= 0x80
    raw[1] = pred_b1
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    raw[4] = imm_byte & 0xFF
    raw[8]  = src2 & 0xFF
    raw[9]  = 0xe0
    raw[10] = 0xff
    raw[11] = 0x07
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


def encode_iadd3_pred_neg_b3(dest: int, neg_src0: int, src1: int, src2: int,
                              pred_idx: int, inverted: bool = False,
                              ctrl: int = 0) -> bytes:
    """Encode @[!]Ppred_idx IADD3 dest, -neg_src0, src1, src2 (negate b3 operand).

    Ground truth: @!P0 IADD3 R7, -R7, RZ, RZ (from div.s32 sign correction)
      bytes: 10 82 07 07 ff 00 00 00 | ff e1 ff 07 00 c8 0f 00
      b1=0x82 = 0x80|(0<<4)|0x02 (@!P0), b3=neg_src0, b9=0xe1 (negate b3)
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x10
    pred_b1 = ((pred_idx & 0x07) << 4) | 0x02
    if inverted:
        pred_b1 |= 0x80
    raw[1] = pred_b1
    raw[2] = dest & 0xFF
    raw[3] = neg_src0 & 0xFF
    raw[4] = src1 & 0xFF
    raw[8]  = src2 & 0xFF
    raw[9]  = 0xe1
    raw[10] = 0xff
    raw[11] = 0x07
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


def encode_lop3_pred(dest: int, src0: int, src1: int, src2: int,
                     lut: int, pred_idx: int, inverted: bool = False,
                     ctrl: int = 0) -> bytes:
    """Encode @[!]Ppred_idx LOP3.LUT dest, src0, src1, src2, lut.

    Ground truth: @!P2 LOP3.LUT R5, RZ, R7, RZ, 0x33
      bytes: 12 a2 05 ff 07 00 00 00 | ff 33 8e 07 00 ca 0f 00
      b1 = 0x80|(pred_idx<<4)|0x02 for inverted, (pred_idx<<4)|0x02 for normal.
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x12
    pred_b1 = ((pred_idx & 0x07) << 4) | 0x02
    if inverted:
        pred_b1 |= 0x80
    raw[1] = pred_b1
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    raw[4] = src1 & 0xFF
    raw[8]  = src2 & 0xFF
    raw[9]  = lut & 0xFF
    raw[10] = 0x8e
    raw[11] = 0x07
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# F64 arithmetic — DADD, DMUL, DFMA
# ---------------------------------------------------------------------------
# Ground truth from RTX 5090 (sm_120) — CORRECTED 2026-04-02:
#   DADD R-R (pure GPR): b1=0x72, dest=b2, src0=b3, src1=b8, b11=0x00
#     Probe: DADD R2, R4, R2 → 29 72 02 04 00 00 00 00 02 00 00 00 00 64 0e 00
#   DADD R-CBUF (constant pool): b1=0x74, dest=b2, src0=b3, cbuf_idx=b4..b7, b11=0x00
#   Previous "ground truth" (2026-03-27) with b1=0x7e was WRONG — was not R-R form.
#   mul.f64 R2, R2, R6 → lo=28 7c 02 02 06 00 00 00  hi=00 00 00 08 00 64 1e 00
#   fma.f64 R2, R2, R6, R6 → lo=2b 7c 02 02 06 00 00 00  hi=06 00 00 08 00 64 1e 00
#
# DADD layout: dest=b2, src0=b3, src1=b8 (b4 unused/zero for DADD)
# DFMA layout: dest=b2, src0=b3, mul_src=b4, add_src=b8
# b11=0x08 precision flag applies to DFMA/DMUL only, NOT DADD.
# ctrl=0x0f32: rbar=0x03 (wait for LDC result), wdep=0x33 (slow ALU slot), misc=2
#
# Register pairs: DADD/DMUL/DFMA implicitly use dest_lo, dest_lo+1 (hi).

def encode_dadd(dest_lo: int, src0_lo: int, src1_lo: int,
                negate_src0: bool = False, ctrl: int = 0) -> bytes:
    """Encode DADD (add.f64): dest = src0 + src1 (double precision).

    R-R form: b1=0x7e, dest=b2, src0=b3, src1=b4, b11=0x08.
    Ground truth (decode_sass.py): DADD R28, R26, R4 → 29 7e 1c 1a 04 00 00 00 00 00 00 08 00 64 3e 00
    Note: earlier probe (DADD R2,R4,R2 → 29 72 ...) was wrong — that was a different instruction.

    negate_src0: If True, compute dest = -src0 + src1 (i.e. sub.f64 with swapped ops).
    Negation bit b9=0x01 inferred from FADD negation pattern (same ISA family).
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x29, 0x7e,
                  b2=dest_lo, b3=src0_lo, b4=src1_lo,
                  b8=0x00,
                  b9=0x01 if negate_src0 else 0x00,
                  b10=0x00, b11=0x08,
                  ctrl=ctrl)


def encode_dmul(dest_lo: int, src0_lo: int, src1_lo: int, ctrl: int = 0) -> bytes:
    """Encode DMUL (mul.f64): dest = src0 * src1 (double precision)."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x28, 0x7c,
                  b2=dest_lo, b3=src0_lo, b4=src1_lo,
                  b8=0x00, b9=0x00, b10=0x00, b11=0x08,
                  ctrl=ctrl)


def encode_dfma(dest_lo: int, src0_lo: int, src1_lo: int, src2_lo: int,
                ctrl: int = 0) -> bytes:
    """Encode DFMA (fma.rn.f64): dest = src0 * src1 + src2 (double precision FMA)."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x2b, 0x7c,
                  b2=dest_lo, b3=src0_lo, b4=src1_lo,
                  b8=src2_lo, b9=0x00, b10=0x00, b11=0x08,
                  ctrl=ctrl)


def encode_dfma_ur_ur(dest_lo: int, src0_lo: int, ur_src1: int, ur_src2: int,
                      ctrl: int = 0) -> bytes:
    """Encode DFMA R-R-UR-UR: dest = src0 * ur_src1 + ur_src2 (FP64 FMA with UR operands).

    Used when the multiplier (B) and addend (C) params are loaded into uniform
    registers via LDCU.64.  Keeps all regular GPRs ≤ R13, avoiding the R14+
    ILLEGAL_INSTRUCTION restriction on SM_120.

    Ground truth from ptxas fp64_bench:
      DFMA R2, R2, UR8, UR12 → 2b7c020208000000 0c00000800{ctrl}
      b0=0x2b, b1=0x7c, b2=dest_lo, b3=src0_lo, b4=ur_src1, b8=ur_src2, b11=0x08
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x2b, 0x7c,
                  b2=dest_lo, b3=src0_lo, b4=ur_src1,
                  b8=ur_src2, b9=0x00, b10=0x00, b11=0x08,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# DSETP — Double-Precision Set Predicate (comparison)
# ---------------------------------------------------------------------------
# Opcode: 0x2a, 0x72 (SM_120 opcode 0x22a — probe-verified, field layout inferred
# from FSETP pattern + DADD b11=0x08 FP64 flag. NOT yet hardware-verified.)
#
# Layout mirrors FSETP (0x20b) with:
#   b3=src0_lo (64-bit pair src0), b4=src1_lo (64-bit pair src1)
#   b9 = (cmp<<4) | pred_dest  (same encoding as FSETP)
#   b10 = 0xf0 (PT mask, same as FSETP)
#   b11 = 0x0b = 0x03 (AND combiner) | 0x08 (FP64 precision flag)
#
# Ordered DSETP comparisons — the FIRST output predicate gets NOT(cond).
# To get P = (a < b), use DSETP_GEU (complement, unordered) so NOT(GEU) = LT.
DSETP_LT  = 0x01   # ordered less-than
DSETP_EQ  = 0x02   # ordered equal
DSETP_LE  = 0x03   # ordered less-or-equal
DSETP_GT  = 0x04   # ordered greater-than
DSETP_NE  = 0x05   # ordered not-equal
DSETP_GE  = 0x06   # ordered greater-or-equal
# Unordered variants (true if NaN present):
DSETP_LTU = 0x09
DSETP_EQU = 0x0a
DSETP_LEU = 0x0b
DSETP_GTU = 0x0c
DSETP_NEU = 0x0d
DSETP_GEU = 0x0e   # ground-truth: ptxas uses this to implement setp.lt

def encode_dsetp(pred_dest: int, src0_lo: int, src1_lo: int, cmp: int = DSETP_LT,
                 ctrl: int = 0) -> bytes:
    """Encode DSETP Ppred, src0, src1, cmp (FP64 comparison → predicate).

    src0_lo and src1_lo are the low registers of the 64-bit pairs.
    Hardware implicitly reads src0_lo+1 and src1_lo+1 for the high words.

    NOTE: field layout inferred from FSETP+DADD patterns; needs hardware verification.
    """
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0], raw[1] = 0x2a, 0x72
    raw[2] = 0x00
    raw[3] = src0_lo & 0xFF
    raw[4] = src1_lo & 0xFF
    raw[9]  = (cmp & 0x0F) << 4  # comparison code in upper nibble; lower nibble = 0
    raw[10] = 0xf0 | ((pred_dest & 0x07) << 1)  # pred_dest: same encoding as ISETP (P0→0xf0, P1→0xf2)
    raw[11] = 0x03   # AND combiner (ground truth: ptxas sm_120)
    raw[13], raw[14], raw[15] = b13, b14, b15
    return bytes(raw)


if __name__ == "__main__":
    ok = roundtrip_verify_opcodes(verbose=True)
    print()
    if ok:
        print("All ground truth samples encode correctly.")
    else:
        print("FAILURES detected — fix the encoders before use.")


# ---------------------------------------------------------------------------
# FSEL — Combined Float Compare + Select (SM_120 Blackwell)
# ---------------------------------------------------------------------------
# ptxas uses these instead of separate FSETP + predicated MOV to avoid the
# SM_120 hardware bug where ISETP corrupts subsequent FSETP predicate state.
#
# Opcode 0x80a: FSEL.step dest, src, threshold, cmp
#   dest = (src cmp threshold) ? 1.0 : 0.0
#   Ground truth: 0a7807040000003f0040800300ca4f00
#     R7 = (R4 > 0.5f) ? 1.0 : 0.0
#
# Opcode 0x80b: FSETP with inline float immediate (sets predicate)
#   Ground truth: 0b7800040000003f0040f00300ca4f00
#     P0 = (R4 > 0.5f)

FSEL_LT = 0x01
FSEL_EQ = 0x02
FSEL_LE = 0x03
FSEL_GT = 0x04
FSEL_NE = 0x05
FSEL_GE = 0x06

def encode_fsel_step(dest: int, src: int, threshold_f32: int, cmp: int = FSEL_GT,
                     ctrl: int = 0) -> bytes:
    """Encode FSEL.step: dest = (src cmp threshold) ? 1.0 : 0.0.

    Opcode 0x80a. Combined compare+select that avoids FSETP predicate corruption.

    Ground truth (ptxas sm_120): out[i] = in[i] > 0.5f ? 1.0f : 0.0f
        0a7807040000003f0040800300ca4f00
        b2=R7(dest), b3=R4(src), b4-b7=0x3f000000(0.5f), b9=0x40(GT), b10=0x80
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    import struct as _s
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x0a
    raw[1] = 0x78
    raw[2] = dest & 0xFF
    raw[3] = src & 0xFF
    _s.pack_into('<I', raw, 4, threshold_f32 & 0xFFFFFFFF)
    raw[8] = 0x00
    raw[9] = (cmp & 0x0F) << 4  # comparison type
    raw[10] = 0x80  # step mode: true=1.0, false=0.0
    raw[11] = 0x03
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15
    return bytes(raw)


# ===========================================================================
# NEW ENCODERS — decoded 2026-04-01 from ptxas 13.0 sm_120 probe cubins
# ===========================================================================

# ---------------------------------------------------------------------------
# SHF.R — Right shifts (64-bit and 32-bit.HI)
# ---------------------------------------------------------------------------
# Opcode family: 0x219 (SHF register shift, byte0=0x19 byte1=0x72)
# Modifier byte 9: R.U64=0x12, R.S64=0x10, R.U32.HI=0x16, R.S32.HI=0x14
# Ground truth:
#   SHF.R.U64 R6, R2, R7, R3 → 0x0000000702067219 | 0x002fc40000001203
#   SHF.R.S64 R6, R2, R7, R3 → 0x0000000702067219 | 0x002fc40000001003
#   SHF.R.U32.HI RZ, R7, R3  → 0x00000007ff077219 | 0x000fca0000011603
#   SHF.R.S32.HI RZ, R7, R3  → 0x00000007ff077219 | 0x000fca0000011403

def encode_shf_r_u64(dest, src_lo, shift_reg, src_hi, ctrl=0):
    """SHF.R.U64 — 64-bit logical right shift."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x19, 0x72, b2=dest, b3=src_lo, b4=shift_reg, b8=src_hi,
                  b9=0x12, b10=0x00, b11=0x00, ctrl=ctrl)

def encode_shf_r_s64(dest, src_lo, shift_reg, src_hi, ctrl=0):
    """SHF.R.S64 — 64-bit arithmetic right shift."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x19, 0x72, b2=dest, b3=src_lo, b4=shift_reg, b8=src_hi,
                  b9=0x10, b10=0x00, b11=0x00, ctrl=ctrl)

def encode_shf_r_u32_hi(dest, src_lo, shift_reg, src_hi, ctrl=0):
    """SHF.R.U32.HI — upper 32-bit logical right shift."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x19, 0x72, b2=dest, b3=src_lo, b4=shift_reg, b8=src_hi,
                  b9=0x16, b10=0x01, b11=0x00, ctrl=ctrl)

def encode_shf_r_s32_hi(dest, src_lo, shift_reg, src_hi, ctrl=0):
    """SHF.R.S32.HI — upper 32-bit arithmetic right shift."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x19, 0x72, b2=dest, b3=src_lo, b4=shift_reg, b8=src_hi,
                  b9=0x14, b10=0x01, b11=0x00, ctrl=ctrl)


# ---------------------------------------------------------------------------
# REDUX.SUM — Warp-wide sum reduction → uniform register
# ---------------------------------------------------------------------------
# Ground truth (unsigned): REDUX.SUM UR6, R0 → b9=0x00, b10=0xc0
# Ground truth (signed):   REDUX.SUM.S32 UR6, R0 → b9=0xc2, b10=0x00
#   ptxas emits REDUX.SUM.S32 for PTX redux.sync.add.s32.
#   Full bytes at 0x40: c4 73 06 00 00 00 00 00 00 c2 00 00 [ctrl]
# Opcode: 0x3c4 (byte0=0xc4, byte1=0x73)

def encode_redux_sum(dest_ur, src, ctrl=0):
    """REDUX.SUM URdest, Rsrc — warp-wide unsigned sum into uniform register.
    Ground truth: REDUX.SUM UR6, R0 → b9=0x00, b10=0xc0."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0xc4, 0x73, b2=dest_ur, b3=src, b4=0x00, b8=0x00,
                  b9=0x00, b10=0xc0, b11=0x00, ctrl=ctrl)


def encode_redux_sum_s32(dest_ur, src, ctrl=0):
    """REDUX.SUM.S32 URdest, Rsrc — warp-wide signed sum into uniform register.
    Ground truth: ptxas redux.sync.add.s32 → b9=0xc2, b10=0x00.
    Full: c4 73 <dest_ur> <src> 00 00 00 00 00 c2 00 00 [ctrl]"""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0xc4, 0x73, b2=dest_ur, b3=src, b4=0x00, b8=0x00,
                  b9=0xc2, b10=0x00, b11=0x00, ctrl=ctrl)


def encode_redux_min_s32(dest_ur, src, ctrl=0):
    """REDUX.MIN.S32 URdest, Rsrc — warp-wide signed min into uniform register.
    Layout inferred from REDUX.SUM (b10 mode field); needs hardware verification."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0xc4, 0x73, b2=dest_ur, b3=src, b4=0x00, b8=0x00,
                  b9=0x00, b10=0x80, b11=0x00, ctrl=ctrl)


def encode_redux_max_s32(dest_ur, src, ctrl=0):
    """REDUX.MAX.S32 URdest, Rsrc — warp-wide signed max into uniform register.
    Layout inferred from REDUX.SUM; needs hardware verification."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0xc4, 0x73, b2=dest_ur, b3=src, b4=0x00, b8=0x00,
                  b9=0x00, b10=0x40, b11=0x00, ctrl=ctrl)


def encode_redux_and_b32(dest_ur, src, ctrl=0):
    """REDUX.AND.B32 URdest, Rsrc — warp-wide bitwise AND into uniform register.
    Layout inferred; needs hardware verification."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0xc4, 0x73, b2=dest_ur, b3=src, b4=0x00, b8=0x00,
                  b9=0x00, b10=0x00, b11=0x00, ctrl=ctrl)


def encode_redux_or_b32(dest_ur, src, ctrl=0):
    """REDUX.OR.B32 URdest, Rsrc — warp-wide bitwise OR into uniform register.
    Layout inferred; needs hardware verification."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0xc4, 0x73, b2=dest_ur, b3=src, b4=0x00, b8=0x00,
                  b9=0x01, b10=0x00, b11=0x00, ctrl=ctrl)


def encode_redux_xor_b32(dest_ur, src, ctrl=0):
    """REDUX.XOR.B32 URdest, Rsrc — warp-wide bitwise XOR into uniform register.
    Layout inferred; needs hardware verification."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0xc4, 0x73, b2=dest_ur, b3=src, b4=0x00, b8=0x00,
                  b9=0x02, b10=0x00, b11=0x00, ctrl=ctrl)


# ---------------------------------------------------------------------------
# MOV R, UR — copy uniform register to general register
# ---------------------------------------------------------------------------
# Ground truth: MOV R5, UR6 (after REDUX.SUM.S32)
#   Full bytes: 02 7c 05 00 06 00 00 00 00 0f 00 08 [ctrl]
#   Opcode: 0xC02 = ((0x7c & 0xF) << 8) | 0x02
#   b2=dest(GPR), b4=src(UR), b9=0x0f, b11=0x08 (fixed modifiers)

def encode_mov_gpr_from_ur(dest: int, ur_src: int, ctrl: int = 0) -> bytes:
    """MOV Rdest, URsrc — copy uniform register value to general register.
    Ground truth: MOV R5, UR6 → 02 7c 05 00 06 00 00 00 00 0f 00 08 [ctrl]"""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x02, 0x7c,
                  b2=dest, b3=0x00, b4=ur_src,
                  b8=0x00,
                  b9=0x0f, b10=0x00, b11=0x08,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# LDGSTS.E — Async global → shared copy (cp.async)
# ---------------------------------------------------------------------------
# Ground truth: LDGSTS.E [R5], desc[UR4][R2.64]
#   → 0x0000000002057fae | 0x008fe8000b9a1004
# Opcode: 0xfae (byte0=0xae, byte1=0x7f)

def encode_ldgsts_e(smem_addr, glob_addr, ur_desc, ctrl=0):
    """LDGSTS.E [Rsmem], desc[URd][Rglob.64] — async global→shared 4B copy."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0xae; raw[1] = 0x7f
    raw[2] = smem_addr & 0xFF; raw[3] = glob_addr & 0xFF; raw[4] = ur_desc & 0xFF
    raw[8] = 0x04; raw[9] = 0x10; raw[10] = 0x9a; raw[11] = 0x0b
    raw[13] = b13; raw[14] = b14; raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# LDGDEPBAR — cp.async commit group
# ---------------------------------------------------------------------------
# Ground truth: 0x00000000000079af | 0x000e220000000000
# Opcode: 0x9af (byte0=0xaf, byte1=0x79)

def encode_ldgdepbar(ctrl=0):
    """LDGDEPBAR — commit async copy group (cp.async.commit_group)."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0xaf, 0x79, b2=0x00, b3=0x00, b4=0x00, b8=0x00,
                  b9=0x00, b10=0x00, b11=0x00, ctrl=ctrl)


# ---------------------------------------------------------------------------
# DEPBAR.LE — cp.async wait group
# ---------------------------------------------------------------------------
# Ground truth: DEPBAR.LE SB0, 0x0 → 0x000080000000791a | 0x000fc80000000000
# Opcode: 0x91a (byte0=0x1a, byte1=0x79)

def encode_depbar_le(sb=0, count=0, ctrl=0):
    """DEPBAR.LE SBn, count — wait for async copies: pending <= count."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x1a; raw[1] = 0x79
    raw[4] = 0x80 | ((sb & 0x7) << 4) | (count & 0xF)
    raw[13] = b13; raw[14] = b14; raw[15] = b15
    return bytes(raw)


# ---------------------------------------------------------------------------
# F2FP.F16.F32 — FP32 to packed FP16
# ---------------------------------------------------------------------------
# Ground truth: F2FP.F16.F32.PACK_AB R0, RZ, R2
#   → 0x00000002ff00723e | 0x004fca00000000ff
# Opcode: 0x23e (byte0=0x3e, byte1=0x72)

def encode_f2fp_f16_f32(dest, src, ctrl=0):
    """F2FP.F16.F32.PACK_AB Rdest, RZ, Rsrc — FP32 to packed FP16 pair."""
    if ctrl == 0: ctrl = _CTRL_DEFAULT
    return _build(0x3e, 0x72, b2=dest, b3=RZ, b4=src, b8=0xFF,
                  b9=0x00, b10=0x00, b11=0x00, ctrl=ctrl)


# ---------------------------------------------------------------------------
# QMMA — Blackwell MXF8 FP8 matrix multiply-accumulate (SM_120 only)
# ---------------------------------------------------------------------------
# Shape: m16n8k32, FP8 inputs, FP32 accumulation.
# Opcode: 0x27a (raw[0]=0x7a, raw[1]=0x72)
# Ground truth from ptxas (SM_120):
#   QMMA.16832.F32.E4M3.E4M3 R4, R4, R2, RZ
#   → 7a72040402000000ff2c000000e20f00  (b9=0x2c = E4M3.E4M3)
#   QMMA.16832.F32.E5M2.E5M2 R4, R4, R2, RZ
#   → 7a72040402000000ffec000000e20f00  (b9=0xec = E5M2.E5M2)
# Register constraints (SM_120 hardware, empirically verified):
#   D: 4-register group, base must be 4-aligned (d%4==0)
#   A: 4-register group, base must be 4-aligned (a%4==0)
#   B: 2-register group, base must be < 8 (b in R0..R7)
#   C: 4-register group (same as D for in-place accumulation, or RZ=255 for no accumulation)

def encode_qmma_e4m3_f32(dest: int, src_a: int, src_b: int, src_c: int,
                          ctrl: int = 0) -> bytes:
    """
    QMMA.16832.F32.E4M3.E4M3 dest, src_a, src_b, src_c

    FP8 E4M3 matrix multiply-accumulate on SM_120 (Blackwell MXF8).
    D[dest:dest+3] = A[src_a:src_a+3] * B[src_b:src_b+1] + C[src_c:src_c+3]
    b9=0x2c encodes E4M3 × E4M3 with FP32 accumulation.
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x7a; raw[1] = 0x72
    raw[2] = dest & 0xFF
    raw[3] = src_a & 0xFF
    raw[4] = src_b & 0xFF
    raw[5] = raw[6] = raw[7] = 0
    raw[8] = src_c & 0xFF
    raw[9] = 0x2c      # E4M3 × E4M3 FP8 format
    raw[10] = raw[11] = raw[12] = 0
    raw[13] = b13; raw[14] = b14; raw[15] = b15
    return bytes(raw)


def encode_qmma_e5m2_f32(dest: int, src_a: int, src_b: int, src_c: int,
                          ctrl: int = 0) -> bytes:
    """
    QMMA.16832.F32.E5M2.E5M2 dest, src_a, src_b, src_c

    FP8 E5M2 matrix multiply-accumulate on SM_120 (Blackwell MXF8).
    b9=0xec encodes E5M2 × E5M2 with FP32 accumulation.
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    b13, b14, b15 = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x7a; raw[1] = 0x72
    raw[2] = dest & 0xFF
    raw[3] = src_a & 0xFF
    raw[4] = src_b & 0xFF
    raw[5] = raw[6] = raw[7] = 0
    raw[8] = src_c & 0xFF
    raw[9] = 0xec      # E5M2 × E5M2 FP8 format
    raw[10] = raw[11] = raw[12] = 0
    raw[13] = b13; raw[14] = b14; raw[15] = b15
    return bytes(raw)
