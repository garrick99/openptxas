"""
SM_120 SHF instruction encoder — template-based, confirmed against ground truth.

Field positions (128-bit instruction = lo[63:0] | hi[63:0], both little-endian):

  lo word:
    bits [11:0]   = opcode family (SHF family = 0x819)
    bits [15:12]  = reserved / fixed (part of opcode extended bits)
    bits [23:16]  = destination register index
    bits [31:24]  = src0 register index
    bits [39:32]  = shift amount K (8-bit immediate)
    bits [63:40]  = reserved / zero

  hi word:
    bits [71:64]  = src1 register index  (= byte 8 of 16-byte word = hi bits [7:0])
    bits [79:72]  = modifier byte 1 (variant-specific, confirmed below)
    bits [87:80]  = modifier byte 2 (variant-specific, confirmed below)
    bits [95:88]  = 0x00 (always)
    bits [103:96] = 0x00 (always)
    bits [112:104]= low byte of control scheduling word (bits [7:0] of ctrl)
    bits [120:113]= high byte of control scheduling word (bits [14:8] of ctrl)
    bits [127:121]= stall count portion of control (bits [22:16] of ctrl, 7 bits)

  Control word (23 bits) = bytes [13:15] >> 1  (i.e., stored at bits [113:104])
  Equivalently: ctrl is packed as (bytes[15] << 16 | bytes[14] << 8 | bytes[13]) >> 1

Modifier bytes [79:72] and [87:80] per variant (100% confirmed from ground truth):
  SHF.L.W.U32.HI:  byte9=0x0e, byte10=0x01  (modifier word = 0x010e at bits [87:72])
  SHF.L.U32:       byte9=0x06, byte10=0x00  (modifier word = 0x0006 at bits [87:72])
  SHF.L.U64.HI:    byte9=0x02, byte10=0x01  (modifier word = 0x0102 at bits [87:72])

Bit interpretation of bits [87:72]:
  bit[73] = 1 always (shared base bit)
  bit[74] = 1 -> U32 width  (set for SHF.L.W.U32.HI and SHF.L.U32; 0 for U64.HI)
  bit[75] = 1 -> .W (wrap)  (set only for SHF.L.W.U32.HI)
  bit[80] = 1 -> .HI output (set for SHF.L.W.U32.HI and SHF.L.U64.HI; 0 for SHF.L.U32)

Control word encoding:
  Bits [22:17] of ctrl = stall count (6 bits, 0..63 cycles)
  Bit  [16]    of ctrl = yield hint
  Bit  [15]    of ctrl = write-after-read dependency barrier
  Bits [14:10] of ctrl = read-after-write dependency barrier
  Bits [9:4]   of ctrl = write dependency slot
  Bits [3:0]   of ctrl = ???

  Safe default ctrl (stall=15, no yield, no barriers): 0x007e0
    -> This equals control = 0x007e0 which places 15-cycle stall.
  Observed ctrl values from ptxas: 0x47f2 and 0x7e5 (scheduling decisions)

  For encoding purposes ctrl=0 uses a max-stall safe default.

Ground truth samples (from sm_120_encoding_tables.json + fresh ptxas output):
  SHF.L.W.U32.HI R5, R3, 0x1f, R2: lo=0x0000001f03057819 hi=0x008fe40000010e02
  SHF.L.W.U32.HI R4, R2, 0x1f, R3: lo=0x0000001f02047819 hi=0x000fca0000010e03
  SHF.L.W.U32.HI R5, R3, 0x8,  R2: lo=0x0000000803057819 hi=0x008fe40000010e02
  SHF.L.W.U32.HI R4, R2, 0x8,  R3: lo=0x0000000802047819 hi=0x000fca0000010e03
  SHF.L.U32      R9, R2, 0x1f, RZ: lo=0x0000001f02097819 hi=0x000fc800000006ff
  SHF.L.U32      R9, R2, 0x8,  RZ: lo=0x0000000802097819 hi=0x000fc800000006ff
  SHF.L.U32      R8, R2, 0x1f, RZ: lo=0x0000001f02087819 hi=0x000fe200000006ff

  Fresh ptxas K=1..31 confirm encoding: K goes linearly in bits[39:32].
  K=32 not encodable as SHF (ptxas emits LOP3).
"""

from __future__ import annotations
import struct

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPCODE_SHF = 0x819            # bits [11:0] for all SHF variants

# Modifier bytes 9 and 10 (bits [79:72] and [87:80]) per variant
# Confirmed: constant across ALL instances in the encoding table
_MODIFIER_SHF_L_W_U32_HI = (0x0e, 0x01)   # byte9=0x0e, byte10=0x01
_MODIFIER_SHF_L_U32       = (0x06, 0x00)   # byte9=0x06, byte10=0x00
_MODIFIER_SHF_L_U64_HI    = (0x02, 0x01)   # byte9=0x02, byte10=0x01

# Fixed bytes 11 and 12 (bits [95:88] and [103:96]) — always 0x00 for all SHF
_FIXED_BYTES_11_12 = (0x00, 0x00)

# byte15 (bits [127:120]):
#   SHF.L.W.U32.HI -> 0x00
#   SHF.L.U32      -> 0x00
#   SHF.L.U64.HI   -> 0x04  (reuse flag in control context)
_BYTE15_SHF_L_W_U32_HI = 0x00
_BYTE15_SHF_L_U32       = 0x00
_BYTE15_SHF_L_U64_HI    = 0x04

# RZ (zero register) index on SM_120
RZ = 255

# Safe default control word: stall=15 cycles, no yield, no barriers
# Encodes to bytes 13-15 as: raw24 = ctrl << 1
# ctrl = 0x007e0 -> stall field = bits[22:17] = 0x3f = 63 -> max stall
# Actually from ptxas: 0x7e4 is commonly used for single-issue stall-15
# Using 0x007e0 as "max stall, safe": raw24 = 0x007e0 << 1 = 0x00fc0
#   byte13 = 0xc0, byte14 = 0x0f, byte15 = 0x00
_CTRL_MAX_STALL = 0x007e0      # 23-bit control word: ~15 cycle stall
_CTRL_RAW24_MAX_STALL = _CTRL_MAX_STALL << 1   # = 0x00fc0


def _ctrl_to_bytes(ctrl: int) -> tuple[int, int, int]:
    """
    Convert a 23-bit control word to bytes 13, 14, 15.
    The control word is stored left-shifted by 1 in bytes [13:15].
    """
    raw24 = (ctrl & 0x7FFFFF) << 1
    b13 = raw24 & 0xFF
    b14 = (raw24 >> 8) & 0xFF
    b15 = (raw24 >> 16) & 0xFF
    return b13, b14, b15


def _build_shf(dest: int, src0: int, k: int, src1: int,
               mod_byte9: int, mod_byte10: int, byte15_fixed: int,
               ctrl: int) -> bytes:
    """
    Low-level 16-byte SHF instruction builder.

    Layout (bytes 0..15, little-endian):
      byte 0:  opcode bits [7:0]   = 0x19  (low 8 bits of 0x819)
      byte 1:  opcode bits [11:8] + upper 4 bits = 0x78 (from ground truth: 0x...7819)
      byte 2:  reserved            = 0x00
      byte 3:  dest reg            = dest & 0xFF
      byte 4:  src0 reg            = src0 & 0xFF  (wait — see below)
      byte 5:  reserved            = 0x00
      byte 6:  reserved            = 0x00
      byte 7:  reserved            = 0x00
      byte 8:  src1 reg            = src1 & 0xFF
      byte 9:  modifier byte 1     = mod_byte9
      byte10:  modifier byte 2     = mod_byte10
      byte11:  0x00 (fixed)
      byte12:  0x00 (fixed)
      byte13:  ctrl low            = from ctrl
      byte14:  ctrl high           = from ctrl
      byte15:  variant fixed byte  = byte15_fixed (or ctrl top bits)

    Wait — looking at actual ground truth hex more carefully:
      R5, R3, 0x1f, R2: hex = 197805031f000000 020e010000e48f00
        bytes 0..7: 19 78 05 03 1f 00 00 00
        bytes 8..15: 02 0e 01 00 00 e4 8f 00

      So:
        byte0 = 0x19, byte1 = 0x78  => lo bits[15:0] = 0x7819 = opcode+extra
        byte2 = 0x00  (bits [23:16] — but wait, dest=5 is at byte2 in 0-indexed)

    Re-examining: hex = "197805031f000000020e010000e48f00"
    Reading as bytes in order: 19 78 05 03 1f 00 00 00 | 02 0e 01 00 00 e4 8f 00
    As LE uint64:
      lo = 0x0000001f03057819
      hi = 0x008fe40000010e02

    So byte[0]=0x19, byte[1]=0x78, byte[2]=0x05 (dest=5), byte[3]=0x03 (src0=3),
       byte[4]=0x1f (K=31), byte[5]=0x00, byte[6]=0x00, byte[7]=0x00
       byte[8]=0x02 (src1=2), byte[9]=0x0e, byte[10]=0x01, byte[11]=0x00,
       byte[12]=0x00, byte[13]=0xe4 (ctrl low), byte[14]=0x8f (ctrl high), byte[15]=0x00

    Field positions in byte array (0-indexed):
      byte[0:2]  = opcode word (fixed per variant): 0x7819
      byte[2]    = dest register
      byte[3]    = src0 register
      byte[4]    = K (shift amount)
      byte[5:8]  = 0x000000 (fixed)
      byte[8]    = src1 register
      byte[9]    = modifier byte 1
      byte[10]   = modifier byte 2
      byte[11]   = 0x00 (fixed)
      byte[12]   = 0x00 (fixed)
      byte[13]   = ctrl bits [7:0] (stored as ctrl<<1 low byte)
      byte[14]   = ctrl bits [15:8] (stored as ctrl<<1 mid byte)
      byte[15]   = ctrl bits [22:16] | byte15_fixed

    The byte15 situation:
      For SHF.L.W.U32.HI and SHF.L.U32: byte15 is purely ctrl top bits.
      For SHF.L.U64.HI: byte15 = 0x04 (reuse/ctrl bit from ptxas).
      When ctrl=0 (safe default with no stall override): byte15 = byte15_fixed.
    """
    assert 0 <= dest <= 255, f"dest={dest} out of range"
    assert 0 <= src0 <= 255, f"src0={src0} out of range"
    assert 0 <= k <= 63,     f"k={k} out of range (SHF shift amount)"
    assert 0 <= src1 <= 255, f"src1={src1} out of range (255=RZ)"

    b13, b14, b15_ctrl = _ctrl_to_bytes(ctrl)
    # byte15 may have both a fixed variant bit and ctrl bits ORed in
    b15 = b15_ctrl | byte15_fixed

    raw = bytearray(16)
    # Opcode bytes (fixed)
    raw[0] = 0x19
    raw[1] = 0x78
    # Variable fields
    raw[2] = dest & 0xFF
    raw[3] = src0 & 0xFF
    raw[4] = k    & 0xFF
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8]  = src1 & 0xFF
    raw[9]  = mod_byte9
    raw[10] = mod_byte10
    raw[11] = 0x00
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15

    return bytes(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_shf_l_w_u32_hi(dest: int, src0: int, k: int, src1: int,
                           ctrl: int = 0) -> bytes:
    """
    Encode SHF.L.W.U32.HI dest, src0, k, src1 to 16 bytes.

    SHF.L.W.U32.HI performs a funnel-shift left with wrap on 32-bit inputs,
    producing the high 32-bit result. Used to implement 64-bit rotates:
        rotate64(x, k) = SHF.L.W.U32.HI(x.hi, x.lo, k) | SHF.L.W.U32.HI(x.lo, x.hi, 64-k)

    Args:
        dest:  Destination register index (0..254)
        src0:  First source register index (0..254)
        k:     Shift amount (0..63)
        src1:  Second source register index (0..254, 255=RZ)
        ctrl:  23-bit scheduling control word (0 = max-stall safe default).
               Use 0 when generating standalone instructions without pipeline context.
               Bits [22:17] = stall count (0..63 cycles)
               Bit  [16]    = yield hint
               etc.

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth validation:
        encode_shf_l_w_u32_hi(5, 3, 0x1f, 2, ctrl=0x47f2)
          -> bytes.fromhex('197805031f000000020e010000e48f00')
        encode_shf_l_w_u32_hi(4, 2, 0x1f, 3, ctrl=0x07e5)
          -> bytes.fromhex('197804021f000000030e010000ca0f00')
    """
    if ctrl == 0:
        ctrl = _CTRL_MAX_STALL
    return _build_shf(dest, src0, k, src1,
                      _MODIFIER_SHF_L_W_U32_HI[0],
                      _MODIFIER_SHF_L_W_U32_HI[1],
                      _BYTE15_SHF_L_W_U32_HI,
                      ctrl)


def encode_shf_l_u32(dest: int, src0: int, k: int, ctrl: int = 0) -> bytes:
    """
    Encode SHF.L.U32 dest, src0, k, RZ to 16 bytes.

    SHF.L.U32 (no .HI, no .W) shifts src0 left by k bits, zeroing the low bits.
    src1 is always RZ (255) for this variant.

    Args:
        dest:  Destination register index (0..254)
        src0:  Source register index (0..254)
        k:     Shift amount (0..31 for U32 — upper half is implementation-defined)
        ctrl:  23-bit scheduling control word (0 = max-stall safe default)

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth validation:
        encode_shf_l_u32(9, 2, 0x1f, ctrl=0x07e4)
          -> bytes.fromhex('197809021f000000ff06000000c80f00')
        encode_shf_l_u32(8, 2, 0x1f, ctrl=0x07f1)
          -> bytes.fromhex('197808021f000000ff06000000e20f00')
    """
    if ctrl == 0:
        ctrl = _CTRL_MAX_STALL
    return _build_shf(dest, src0, k, RZ,
                      _MODIFIER_SHF_L_U32[0],
                      _MODIFIER_SHF_L_U32[1],
                      _BYTE15_SHF_L_U32,
                      ctrl)


def encode_shf_l_u64_hi(dest: int, src0: int, k: int, src1: int,
                         ctrl: int = 0) -> bytes:
    """
    Encode SHF.L.U64.HI dest, src0, k, src1 to 16 bytes.

    SHF.L.U64.HI performs a 64-bit funnel-shift left, returning the high 32 bits.
    Used for 64-bit shift operations where both halves of a 64-bit register pair
    are involved.

    Args:
        dest:  Destination register index (0..254)
        src0:  First source register index (0..254)
        k:     Shift amount (0..63)
        src1:  Second source register index (0..254)
        ctrl:  23-bit scheduling control word (0 = max-stall safe default)
               Note: ptxas uses byte15=0x04 for this variant (reuse marker).

    Returns:
        16-byte little-endian instruction encoding.

    Ground truth validation:
        encode_shf_l_u64_hi(9, 2, 0x1f, 3, ctrl=0x207f2)
          -> bytes.fromhex('197809021f0000000302010000e40f04')
    """
    if ctrl == 0:
        ctrl = _CTRL_MAX_STALL
    return _build_shf(dest, src0, k, src1,
                      _MODIFIER_SHF_L_U64_HI[0],
                      _MODIFIER_SHF_L_U64_HI[1],
                      _BYTE15_SHF_L_U64_HI,
                      ctrl)


# ---------------------------------------------------------------------------
# SHF.R (right-shift) variants — confirmed 2026-03-17 from ptxas 13.0
# ---------------------------------------------------------------------------

# Modifier bytes for right-shift variants:
#   SHF.R.U64:     byte9=0x12, byte10=0x00   (SHF.L.U64.HI byte9 | 0x10)
#   SHF.R.U32.HI:  byte9=0x16, byte10=0x01   (SHF.L.U32 byte9 | 0x10)
#   SHF.R.S32.HI:  byte9=0x14, byte10=0x01   (arithmetic right shift; S32 type)
_MODIFIER_SHF_R_U64      = (0x12, 0x00)
_MODIFIER_SHF_R_U32_HI   = (0x16, 0x01)
_MODIFIER_SHF_R_S32_HI   = (0x14, 0x01)


def encode_shf_r_u32(dest: int, src0: int, k: int, src1: int,
                      ctrl: int = 0) -> bytes:
    """
    Encode SHF.R.U64 dest, src0, k, src1 to 16 bytes.

    Right-shift funnel: used for the low word of a 64-bit logical right shift.
    dest = (src1:src0 >> k)[31:0]  (funnel the pair and extract low 32 bits)

    Ground truth (ptxas 13.0, sm_120):
        SHF.R.U64 R6, R2, 0x8, R3:
          lo=0x0000000802067819  hi=0x004fc40000001203
          ctrl=0x27e2

    Note: despite the name SHF.R.U64, this is how ptxas encodes the low-word
    result of shr.u64.  The .U64 suffix indicates the 64-bit funnel mode.
    """
    if ctrl == 0:
        ctrl = _CTRL_MAX_STALL
    return _build_shf(dest, src0, k, src1,
                      _MODIFIER_SHF_R_U64[0],
                      _MODIFIER_SHF_R_U64[1],
                      0x00,
                      ctrl)


def encode_shf_r_u32_hi(dest: int, src0: int, k: int,
                          ctrl: int = 0) -> bytes:
    """
    Encode SHF.R.U32.HI dest, RZ, k, src0 to 16 bytes.

    Right-shift with zero-fill for the high word of a 64-bit logical right shift.
    dest = src0 >> k (with zeros shifted in from the left via RZ as src0).

    Actually: ptxas emits SHF.R.U32.HI dest, RZ, k, src1 where src1 has the
    high word.  The RZ in src0 position means the low word is zero.

    Ground truth (ptxas 13.0, sm_120):
        SHF.R.U32.HI R7, RZ, 0x8, R3:
          lo=0x00000008ff077819  hi=0x000fca0000011603
          ctrl=0x7e5
    """
    if ctrl == 0:
        ctrl = _CTRL_MAX_STALL
    return _build_shf(dest, RZ, k, src0,
                      _MODIFIER_SHF_R_U32_HI[0],
                      _MODIFIER_SHF_R_U32_HI[1],
                      0x00,
                      ctrl)


# ---------------------------------------------------------------------------
# Variable-shift SHF variants (shift amount from a register, not immediate)
# ---------------------------------------------------------------------------
#
# Ground truth (ptxas 13.0, sm_120), from var_shl probe:
#   SHF.L.U32 R6, R4, R5, RZ  (shl.b32 %r2, %r0, %r1 where %r1 is runtime):
#     bytes: 9972060405000000ff06000800e21f00
#   SHF.R.U32.HI R4, RZ, R5, R4  (shr.u32 %r3, %r0, %r1 where %r1 is runtime):
#     bytes: 997204ff050000000416010800ca0f00
#
# Key differences from constant-shift SHF (opcode 0x7819):
#   - opcode bytes: 0x99, 0x72  (vs 0x19, 0x78 for constant)
#   - byte[4] = k_reg (register index) instead of immediate K
#   - byte[12] = 0x08 (vs 0x00 for constant)
#   - modifier bytes (b9, b10) are IDENTICAL to the constant variants
#   - ctrl word (bytes 13-15) works the same way


def _build_shf_var(dest: int, src0: int, k_reg: int, src1: int,
                   mod_byte9: int, mod_byte10: int,
                   ctrl: int) -> bytes:
    """Build a variable-shift SHF instruction (shift amount in a register)."""
    if ctrl == 0:
        ctrl = _CTRL_MAX_STALL
    b13, b14, b15_ctrl = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x99   # opcode low byte (variable-shift variant)
    raw[1] = 0x72   # opcode high byte (variable-shift variant)
    raw[2] = dest  & 0xFF
    raw[3] = src0  & 0xFF
    raw[4] = k_reg & 0xFF   # shift-amount register (was immediate K for constant)
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8]  = src1 & 0xFF
    raw[9]  = mod_byte9
    raw[10] = mod_byte10
    raw[11] = 0x08   # register-source flag (vs 0x00 for constant-shift SHF)
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15_ctrl
    return bytes(raw)


def encode_shf_l_u32_var(dest: int, src0: int, k_reg: int,
                          ctrl: int = 0) -> bytes:
    """
    Encode SHF.L.U32 dest, src0, k_reg, RZ (variable left shift) to 16 bytes.

    k_reg is a GPR holding the shift amount (0-based register index).
    Used for shl.b32 with a runtime shift amount.

    Ground truth (ptxas 13.0, sm_120, shl.b32 reg-shift):
        SHF.L.U32 R5, R0, R5, RZ → 1972050005000000ff06000000ca0f00

    NOTE 2026-04-29: byte 0 must be 0x19 (NOT 0x99 as _build_shf_var
    produces) and byte 11 must be 0x00 (NOT 0x08).  Same hardware-
    decode quirk as encode_shf_r_u32_hi_var.  Fixed via inline build
    instead of _build_shf_var.
    """
    if ctrl == 0:
        ctrl = _CTRL_MAX_STALL
    b13, b14, b15_ctrl = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x19
    raw[1] = 0x72
    raw[2] = dest   & 0xFF
    raw[3] = src0   & 0xFF   # data at b3 (different from R-form which uses b8)
    raw[4] = k_reg  & 0xFF
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8] = RZ     & 0xFF
    raw[9]  = _MODIFIER_SHF_L_U32[0]   # 0x06
    raw[10] = _MODIFIER_SHF_L_U32[1]   # 0x00
    raw[11] = 0x00
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15_ctrl
    return bytes(raw)


def encode_shf_r_u32_hi_var(dest: int, src_hi: int, k_reg: int,
                              ctrl: int = 0) -> bytes:
    """
    Encode SHF.R.U32.HI dest, RZ, k_reg, src_hi (variable right shift) to 16 bytes.

    src_hi holds the data to shift; k_reg holds the shift amount.
    Used for shr.u32/b32 with a runtime shift amount.

    Ground truth (ptxas 13.0, sm_120, shr.b32 reg-shift):
        SHF.R.U32.HI R5, RZ, R5, R4 → 197205ff050000000016010000ca0f00

    NOTE 2026-04-29: an earlier encoder produced byte[0]=0x99 byte[11]=0x08
    via _build_shf_var (matching SHF.L.U32 var).  Hardware decodes that as
    a different SHF variant — output was tid-dependent garbage.  ptxas
    actually uses byte[0]=0x19 byte[11]=0x00 (same as SHF.R.S32.HI var,
    just with the U32 modifier byte 9 = 0x16 instead of 0x14).  Surfaced
    via probe mower's bfe.u32 reg-pos and shr.b32 reg-shift.
    """
    if ctrl == 0:
        ctrl = _CTRL_MAX_STALL
    b13, b14, b15_ctrl = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x19   # opcode low byte (matches ptxas; was 0x99)
    raw[1] = 0x72   # opcode high byte
    raw[2] = dest   & 0xFF
    raw[3] = RZ     & 0xFF
    raw[4] = k_reg  & 0xFF
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8] = src_hi & 0xFF
    raw[9]  = _MODIFIER_SHF_R_U32_HI[0]   # 0x16
    raw[10] = _MODIFIER_SHF_R_U32_HI[1]   # 0x01
    raw[11] = 0x00   # NOT 0x08 — matches s32 var encoder, mirrors ptxas
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15_ctrl
    return bytes(raw)


def encode_shf_r_s32_hi(dest: int, src0: int, k: int,
                         ctrl: int = 0) -> bytes:
    """
    Encode SHF.R.S32.HI dest, RZ, k, src0 (arithmetic right shift, constant) to 16 bytes.

    Like SHF.R.U32.HI but sign-extends from the MSB (S32 type).
    Used for shr.s32 with a constant shift amount.

    Ground truth (ptxas 13.0, sm_120):
        SHF.R.S32.HI R5, RZ, 4, R5 → 197805ff040000000514010000ca1f00
    """
    if ctrl == 0:
        ctrl = _CTRL_MAX_STALL
    return _build_shf(dest, RZ, k, src0,
                      _MODIFIER_SHF_R_S32_HI[0],
                      _MODIFIER_SHF_R_S32_HI[1],
                      0x00,
                      ctrl)


def encode_shf_r_s32_hi_var(dest: int, src_hi: int, k_reg: int,
                              ctrl: int = 0) -> bytes:
    """
    Encode SHF.R.S32.HI dest, RZ, k_reg, src_hi (arithmetic right shift, variable) to 16 bytes.

    NOTE: SHF.R.S32.HI var uses byte[0]=0x19 (not 0x99) and byte[11]=0x00 (not 0x08).
    This differs from SHF.L.U32 var and SHF.R.U32.HI var.

    Ground truth (ptxas 13.0, sm_120):
        SHF.R.S32.HI R5, RZ, R5, R4 → 197205ff050000000414010000ca1f00
    """
    if ctrl == 0:
        ctrl = _CTRL_MAX_STALL
    b13, b14, b15_ctrl = _ctrl_to_bytes(ctrl)
    raw = bytearray(16)
    raw[0] = 0x19   # opcode low byte (same as constant-shift SHF)
    raw[1] = 0x72   # opcode high byte (variable-shift variant, not 0x78)
    raw[2] = dest   & 0xFF
    raw[3] = RZ     & 0xFF   # src0 = RZ (lo word is zero for arithmetic shift)
    raw[4] = k_reg  & 0xFF   # shift-amount register
    raw[5] = 0x00
    raw[6] = 0x00
    raw[7] = 0x00
    raw[8]  = src_hi & 0xFF  # the value being shifted
    raw[9]  = _MODIFIER_SHF_R_S32_HI[0]   # 0x14
    raw[10] = _MODIFIER_SHF_R_S32_HI[1]   # 0x01
    raw[11] = 0x00   # no register-source flag (differs from U32 var which uses 0x08)
    raw[12] = 0x00
    raw[13] = b13
    raw[14] = b14
    raw[15] = b15_ctrl
    return bytes(raw)


# ---------------------------------------------------------------------------
# Round-trip verification against ground truth
# ---------------------------------------------------------------------------

# Ground truth table: (variant, dest, src0, k, src1, ctrl, expected_hex)
# ctrl values are taken directly from ptxas output (JSON control field).
# For round-trip: we encode WITH the exact ctrl value and compare full 16 bytes.
_GROUND_TRUTH: list[tuple[str, int, int, int, int, int, str]] = [
    # SHF.L.W.U32.HI samples from sm_120_encoding_tables.json
    ("SHF.L.W.U32.HI", 5, 3, 0x1f, 2, 0x047f2, "197805031f000000020e010000e48f00"),
    ("SHF.L.W.U32.HI", 4, 2, 0x1f, 3, 0x007e5, "197804021f000000030e010000ca0f00"),
    ("SHF.L.W.U32.HI", 5, 3, 0x08, 2, 0x047f2, "1978050308000000020e010000e48f00"),
    ("SHF.L.W.U32.HI", 4, 2, 0x08, 3, 0x007e5, "1978040208000000030e010000ca0f00"),
    ("SHF.L.W.U32.HI", 4, 3, 0x1f, 2, 0x047f2, "197804031f000000020e010000e48f00"),
    ("SHF.L.W.U32.HI", 5, 2, 0x1f, 3, 0x007e5, "197805021f000000030e010000ca0f00"),
    ("SHF.L.W.U32.HI", 4, 3, 0x01, 2, 0x047f2, "1978040301000000020e010000e48f00"),
    ("SHF.L.W.U32.HI", 5, 2, 0x01, 3, 0x007e5, "1978050201000000030e010000ca0f00"),
    # SHF.L.U32 samples
    ("SHF.L.U32",      9, 2, 0x1f, RZ, 0x007e4, "197809021f000000ff06000000c80f00"),
    ("SHF.L.U32",      9, 2, 0x08, RZ, 0x007e4, "1978090208000000ff06000000c80f00"),
    ("SHF.L.U32",      8, 2, 0x1f, RZ, 0x007f1, "197808021f000000ff06000000e20f00"),
    ("SHF.L.U32",      8, 2, 0x01, RZ, 0x007f1, "1978080201000000ff06000000e20f00"),
    # SHF.L.U64.HI samples (ctrl is 0x207f2 — note byte15=0x04 from reuse/ctrl top)
    ("SHF.L.U64.HI",  9, 2, 0x1f, 3, 0x207f2, "197809021f0000000302010000e40f04"),
    ("SHF.L.U64.HI",  9, 2, 0x08, 3, 0x207f2, "19780902080000000302010000e40f04"),
    ("SHF.L.U64.HI",  9, 2, 0x01, 3, 0x207f2, "19780902010000000302010000e40f04"),
    # SHF.R.U64 (right-shift funnel, 64-bit mode) — confirmed ptxas 13.0, sm_120
    ("SHF.R.U64",     6, 2, 0x08, 3, 0x027e2, "19780602080000000312000000c44f00"),
    # SHF.R.U32.HI (right-shift with zero-fill) — confirmed ptxas 13.0, sm_120
    ("SHF.R.U32.HI",  7, RZ, 0x08, 3, 0x007e5, "197807ff080000000316010000ca0f00"),
]


def roundtrip_verify(verbose: bool = True) -> bool:
    """
    Verify encode() produces bytes matching ground truth samples.

    Returns True if all samples pass.
    """
    encoders = {
        "SHF.L.W.U32.HI": lambda d, s0, k, s1, c: encode_shf_l_w_u32_hi(d, s0, k, s1, ctrl=c),
        "SHF.L.U32":       lambda d, s0, k, s1, c: encode_shf_l_u32(d, s0, k, ctrl=c),
        "SHF.L.U64.HI":    lambda d, s0, k, s1, c: encode_shf_l_u64_hi(d, s0, k, s1, ctrl=c),
        "SHF.R.U64":       lambda d, s0, k, s1, c: encode_shf_r_u32(d, s0, k, s1, ctrl=c),
        # Ground truth: src0=RZ, src1=R3 → encoder takes src_hi=s1 (the hi word reg)
        "SHF.R.U32.HI":    lambda d, s0, k, s1, c: encode_shf_r_u32_hi(d, s1, k, ctrl=c),
    }

    all_pass = True
    results = []

    for variant, dest, src0, k, src1, ctrl, expected_hex in _GROUND_TRUTH:
        expected = bytes.fromhex(expected_hex)
        encode_fn = encoders[variant]
        got = encode_fn(dest, src0, k, src1, ctrl)

        if got == expected:
            status = "PASS"
        else:
            status = "FAIL"
            all_pass = False

        results.append((variant, dest, src0, k, src1, ctrl, status, got, expected))

    if verbose:
        _print_results(results)

    return all_pass


def _print_results(results):
    print("=" * 90)
    print("SM_120 SHF Encode Round-Trip Verification")
    print("=" * 90)
    print(f"{'Variant':<22} {'Operands':<22} {'ctrl':>8} | {'Status':<6} | Notes")
    print("-" * 90)

    pass_count = 0
    fail_count = 0

    for variant, dest, src0, k, src1, ctrl, status, got, expected in results:
        if src1 == RZ:
            op_str = f"R{dest}, R{src0}, 0x{k:x}, RZ"
        else:
            op_str = f"R{dest}, R{src0}, 0x{k:x}, R{src1}"

        mark = "OK" if status == "PASS" else "!!"
        print(f"{variant:<22} {op_str:<22} {ctrl:#010x} | {status:<6} | {mark}")

        if status == "FAIL":
            fail_count += 1
            print(f"  expected: {expected.hex()}")
            print(f"  got:      {got.hex()}")
            # Show byte differences
            diffs = [i for i in range(16) if got[i] != expected[i]]
            for i in diffs:
                print(f"  byte[{i:2d}]: got=0x{got[i]:02x} expected=0x{expected[i]:02x}  "
                      f"bits[{i*8+7}:{i*8}]")
        else:
            pass_count += 1

    print("-" * 90)
    print(f"Results: {pass_count} PASS, {fail_count} FAIL out of {len(results)} total")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Disassembly helper (decode bytes -> field dict, for debugging)
# ---------------------------------------------------------------------------

def decode_shf_bytes(raw: bytes) -> dict:
    """
    Decode a 16-byte SHF instruction back into field values.
    Useful for debugging and cross-checking.
    """
    assert len(raw) == 16, f"Expected 16 bytes, got {len(raw)}"
    lo = struct.unpack_from('<Q', raw, 0)[0]
    hi = struct.unpack_from('<Q', raw, 8)[0]

    opcode = lo & 0xFFF
    dest   = (lo >> 16) & 0xFF
    src0   = (lo >> 24) & 0xFF
    k      = (lo >> 32) & 0xFF
    src1   = hi & 0xFF
    mod9   = (hi >> 8)  & 0xFF
    mod10  = (hi >> 16) & 0xFF
    # Control: bytes[13:15] >> 1
    b13, b14, b15 = raw[13], raw[14], raw[15]
    ctrl_raw24 = (b15 << 16) | (b14 << 8) | b13
    ctrl = ctrl_raw24 >> 1

    # Identify variant from modifier bytes
    mod_key = (mod9, mod10)
    variant_map = {
        (0x0e, 0x01): "SHF.L.W.U32.HI",
        (0x06, 0x00): "SHF.L.U32",
        (0x02, 0x01): "SHF.L.U64.HI",
    }
    variant = variant_map.get(mod_key, f"SHF.?(mod=0x{mod10:02x}{mod9:02x})")

    src1_str = "RZ" if src1 == RZ else f"R{src1}"
    return {
        "variant":  variant,
        "opcode":   hex(opcode),
        "dest":     f"R{dest}",
        "src0":     f"R{src0}",
        "k":        k,
        "k_hex":    hex(k),
        "src1":     src1_str,
        "mod9":     hex(mod9),
        "mod10":    hex(mod10),
        "ctrl":     hex(ctrl),
        "ctrl_raw": ctrl,
        "stall":    (ctrl >> 17) & 0x3F,
    }


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ok = roundtrip_verify(verbose=True)
    print()
    if ok:
        print("All ground truth samples encode correctly.")
    else:
        print("FAILURES detected — encoding has bugs.")

    print()
    print("Example decode:")
    ex = encode_shf_l_w_u32_hi(5, 3, 0x1f, 2, ctrl=0x047f2)
    print(f"  encode_shf_l_w_u32_hi(5, 3, 0x1f, 2, ctrl=0x47f2) = {ex.hex()}")
    print(f"  decode: {decode_shf_bytes(ex)}")
