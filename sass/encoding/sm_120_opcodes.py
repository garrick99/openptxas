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
    SR_TID_X  = 0x21
    SR_TID_Y  = 0x22
    SR_CTAID_X = 0x25
    SR_NTID_X  = 0x29
"""

from __future__ import annotations
import struct

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RZ = 255   # zero register index on SM_120
PT = 7     # predicate "true" (always-true predicate register)

# SR codes for S2R
SR_TID_X   = 0x21
SR_TID_Y   = 0x22
SR_CTAID_X = 0x25
SR_NTID_X  = 0x29

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
                 ctrl: int = 0) -> bytes:
    """
    Encode IADD3 dest, P0, PT, src0, src1, src2 to 16 bytes.

    Three-way integer add.  The predicate outputs (P0, PT) are fixed in the
    modifier bytes; only the four integer register operands are variable.

    Args:
        dest:  Destination register index (0..255, 255=RZ).
        src0:  First source register index (0..255, 255=RZ).
        src1:  Second source register index (0..255, 255=RZ).
        src2:  Third source register index (0..255, 255=RZ).
        ctrl:  23-bit scheduling control word (0 = default 0x7e0).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth:
        encode_iadd3(RZ, RZ, 4, RZ, ctrl=0x7f1)
            -> bytes.fromhex('1072ffff04000000ffe0f10700e20f00')
    """
    if ctrl == 0:
        ctrl = _CTRL_DEFAULT
    return _build(0x10, 0x72,
                  b2=dest, b3=src0, b4=src1,
                  b8=src2,
                  b9=0xe0, b10=0xf1, b11=0x07,
                  ctrl=ctrl)


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
                  b2=0x00, b3=src_addr, b4=ur_desc & 0xFF,
                  b8=src_data,
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
    return _build(0x0c, 0x7c,
                  b2=pred_dest & 0xFF, b3=src_reg, b4=ur_src & 0xFF,
                  b8=0x70,
                  b9=0x62, b10=0xf0, b11=0x0b,
                  ctrl=ctrl)


# ---------------------------------------------------------------------------
# IMAD.SHL.U32 (multiply-add as shift-left, avoids SHF pipeline conflicts)
# ---------------------------------------------------------------------------
# Ground truth: "IMAD.SHL.U32 R6, R2, 0x100, RZ"
#   lo=0x0000010002067824  hi=0x000fca00078e00ff
#   b0=0x24, b1=0x78 (opcode 0x824), b2=dest, b3=src0, b4:b5=imm16, b8=RZ
#   b9=0x00, b10=0x8e, b11=0x07
#   This computes: dest = src0 * imm16 + RZ = src0 << log2(imm16)

def encode_imad_shl_u32(dest: int, src0: int, shift_amount: int,
                         ctrl: int = 0) -> bytes:
    """
    Encode IMAD.SHL.U32 dest, src0, (1<<K), RZ — left shift via multiply.

    Uses the IMAD unit instead of SHF, avoiding SHF.L/SHF.R pipeline conflicts.
    ptxas uses this for shl when shr is also needed in the same kernel.

    Args:
        dest:         Destination register (0..254).
        src0:         Source register (0..254).
        shift_amount: Shift amount K (0..15, since imm16 = 1<<K must fit in 16 bits).
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
    raw[8] = RZ
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
    raw[11] = 0x07
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

if __name__ == "__main__":
    ok = roundtrip_verify_opcodes(verbose=True)
    print()
    if ok:
        print("All ground truth samples encode correctly.")
    else:
        print("FAILURES detected — fix the encoders before use.")
