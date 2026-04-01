"""
capmerc_gen.py — Generate correct .nv.capmerc.text.{kernel} records for SM_120.

Reverse-engineered from ptxas reference cubins. The capmerc section encodes
Mercury compiler metadata: register allocation, instruction class descriptors,
barrier regions, and scheduling hints.

Format overview (all little-endian):

  Header (16 bytes):
    [0:8]   Magic: 0c 00 00 00 01 00 00 c0
    [8]     Register count (num_gprs allocated)
    [9:12]  Reserved (zeros)
    [12:16] Capability bitmask (encodes instruction classes used)

  Body (variable): sequence of records:
    Type 01 (16B) — Instruction class descriptors
    Type 02 sub=0x22 (32B) — Barrier region descriptors
    Type 02 sub=0x38 (32B) — Terminal record (always last body record)
    Filler blocks (4B each): 41 0c XX 04 — inserted before terminal

  Trailer (2 bytes):
    [0]  0xd0 (standard) or 0x50 (high-register alt)
    [1]  Sequence/complexity marker
"""

import struct
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAPMERC_MAGIC = bytes.fromhex('0c000000010000c0')

# Universal prologue — always the first type-01 record
PROLOGUE_RECORD = bytes.fromhex('010b040af80004000000410000040000')

# STG instruction class descriptor
STG_DESCRIPTOR = bytes.fromhex('010b0e0afa0005000000030139040000')

# Standard filler block patterns
FILLER_WITH_BRANCH = bytes.fromhex('410c5404')   # kernels with branch/conditional exit
FILLER_NO_BRANCH   = bytes.fromhex('410c5004')   # straight-line kernels


# ---------------------------------------------------------------------------
# Capability bitmask computation
# ---------------------------------------------------------------------------

# Observed capability mask bits from ptxas cubins:
#   Bit 3  (0x08)  — always set (base ALU)
#   Bit 6  (0x40)  — STG present
#   Bit 7  (0x80)  — conditional branch / predicated exit
#   Bit 8  (0x100) — SHF / shift operations
#   Bit 9  (0x200) — extended ALU (IADD3.X, etc.)
#   Bit 10 (0x400) — ISETP / predicate operations
#   Bit 11 (0x800) — IMAD.WIDE / wide multiply
#   Bit 12 (0x1000)— LDG / global loads
#   Bit 13 (0x2000)— register pressure > 14
#   Bit 16+ — scale with instruction count / scheduling complexity

def compute_capability_mask(
    has_stg: bool,
    has_ldg: bool,
    has_branch: bool,
    has_shift: bool,
    has_ur_ops: bool,
    has_imad: bool,
    has_isetp: bool,
    has_fadd: bool,
    num_gprs: int,
    num_barrier_regions: int,
    text_size: int,
) -> int:
    """Compute the capability bitmask for the capmerc header.

    This is an approximation — the exact bit semantics are not fully decoded,
    but matches ptxas output for the kernel classes we support.
    """
    mask = 0x08  # base ALU always set

    if has_stg:
        mask |= 0x40
    if has_ldg:
        mask |= 0x1000
    if has_branch:
        mask |= 0x80 | 0x100 | 0x400
    if has_shift:
        mask |= 0x100
    if has_ur_ops:
        mask |= 0x200
    if has_imad:
        mask |= 0x800
    if has_isetp:
        mask |= 0x400
    if has_fadd:
        mask |= 0x20

    if num_gprs > 14:
        mask |= 0x2000

    # Higher bits scale with code size / complexity
    # text_size_pages = number of 256-byte pages
    text_pages = max(text_size // 256, 1)
    if text_pages > 1:
        mask |= (text_pages << 16)

    # Barrier region count contributes to upper bits
    if num_barrier_regions > 2:
        mask |= (num_barrier_regions << 8)

    return mask


# ---------------------------------------------------------------------------
# Type-01 record builders
# ---------------------------------------------------------------------------

def _build_type01_prologue() -> bytes:
    """Universal prologue record — always first."""
    return PROLOGUE_RECORD


def _build_type01_alu_basic(region_idx: int = 0) -> bytes:
    """ALU-only instruction class descriptor (byte[2]=0x04).

    region_idx: 0-based index of this record's scheduling region.
    byte[10] encodes a register liveness hint, byte[11] the region index,
    byte[12:13] encode instruction class sub-info.
    """
    rec = bytearray(PROLOGUE_RECORD)  # start from prologue template
    # Modify for specific region
    if region_idx == 0:
        # Second ALU record: byte[10] encodes register liveness.
        # 0x81 restricts to low register range; 0x01 allows full range.
        # ptxas uses 0x01 for all register counts (verified across 8-24 GPR probes).
        rec[10] = 0x01
        rec[11] = 0x00
        rec[12] = 0x01
        rec[13] = 0x02
    elif region_idx == 1:
        # Third ALU record: byte[10]=0x01 for full register range (was 0xc1)
        rec[10] = 0x01
        rec[11] = 0x00
        rec[12] = 0x01
        rec[13] = 0x04
    else:
        rec[10] = 0x41
        rec[11] = region_idx
        rec[12] = 0x01
        rec[13] = 0x02 * (region_idx + 1)
    return bytes(rec)


def _build_type01_ur_alu(region_idx: int = 0) -> bytes:
    """UR-using ALU instruction class descriptor (byte[2]=0x06, mask=0xfa).

    Used when the kernel references uniform registers (UR ops like ULDC.64,
    S2UR, etc).
    """
    rec = bytearray(16)
    rec[0] = 0x01
    rec[1] = 0x0b
    rec[2] = 0x06       # UR-ALU class
    rec[3] = 0x0a
    rec[4] = 0xfa       # mask includes UR bit
    rec[5] = 0x00
    rec[6] = 0x04
    rec[7] = 0x00
    rec[8] = 0x00
    rec[9] = 0x00
    rec[10] = 0x01
    rec[11] = 0x01
    rec[12] = 0x04
    rec[13] = 0x02
    rec[14] = 0x00
    rec[15] = 0x00
    return bytes(rec)


def _build_type01_stg(region_idx: int = 1) -> bytes:
    """STG (store global) instruction class descriptor."""
    rec = bytearray(STG_DESCRIPTOR)
    rec[11] = region_idx  # region counter
    return bytes(rec)


# ---------------------------------------------------------------------------
# Type-02 record builders
# ---------------------------------------------------------------------------

def _build_type02_barrier_first(has_branch: bool, has_ur_ops: bool,
                                 barrier_idx: int = 0,
                                 text_size_pages: int = 1) -> bytes:
    """First barrier region record (type 02 sub=0x22).

    The first barrier record always has a non-zero second-half encoding
    the initial scheduling window size.
    """
    rec = bytearray(32)
    rec[0] = 0x02
    rec[1] = 0x22
    # byte[2]: 0x08 if branch/UR, 0x0e if no branch
    rec[2] = 0x08 if (has_branch or has_ur_ops) else 0x0e
    rec[3] = 0x06
    rec[4] = 0xfa if (has_branch or has_ur_ops) else 0xf8
    rec[5] = 0x00
    # byte[6] mode: 0x42=first-with-branch, 0x52=first-no-branch, 0x62=complex
    if has_branch:
        rec[6] = 0x42
    else:
        rec[6] = 0x52
    rec[7] = 0x00
    # bytes 8-9: zeros
    # byte[10:11]: register liveness hint
    if has_branch:
        rec[10] = 0x01 if barrier_idx == 0 else 0x41
        rec[11] = 0x01
    else:
        rec[10] = 0x83 if barrier_idx == 0 else 0x03
        rec[11] = 0x00 if barrier_idx == 0 else 0x01
    rec[12] = 0x40
    rec[13] = 0x00
    rec[14] = 0x02
    rec[15] = 0x00
    # Second half: scheduling window
    # byte[28] (offset 12 in second half) = window size hint
    # 0x08 for 1-page, 0x10 for 2-page, 0x18 for 3-page
    rec[28] = text_size_pages * 0x08
    return bytes(rec)


def _build_type02_barrier_mid(barrier_idx: int, has_branch: bool,
                               has_ur_ops: bool,
                               text_size_pages: int = 1) -> bytes:
    """Middle/continuation barrier region record (type 02 sub=0x22).

    Subsequent barrier records after the first. These may or may not have
    a non-zero second half depending on scheduling complexity.
    """
    rec = bytearray(32)
    rec[0] = 0x02
    rec[1] = 0x22
    rec[2] = 0x08 if (has_branch or has_ur_ops) else 0x0e
    rec[3] = 0x06
    rec[4] = 0xfa if (has_branch or has_ur_ops) else 0xf8
    rec[5] = 0x00
    # mode byte: 0x52 for standard continuation, 0x62 for complex
    rec[6] = 0x52
    rec[7] = 0x00
    rec[10] = 0x83
    rec[11] = 0x01
    rec[12] = 0x40
    rec[13] = 0x00
    rec[14] = 0x02
    rec[15] = 0x00
    # Second half: may have scheduling hint for multi-page
    if text_size_pages > 1:
        rec[28] = 0x10
    return bytes(rec)


def _build_type02_barrier_last(barrier_idx: int, has_branch: bool,
                                has_ur_ops: bool) -> bytes:
    """Last barrier region record before the terminal (type 02 sub=0x22).

    Always has a zeroed second half.
    """
    rec = bytearray(32)
    rec[0] = 0x02
    rec[1] = 0x22
    rec[2] = 0x08 if (has_branch or has_ur_ops) else 0x0e
    rec[3] = 0x06
    rec[4] = 0xfa if (has_branch or has_ur_ops) else 0xf8
    rec[5] = 0x00
    rec[6] = 0x52
    rec[7] = 0x00
    rec[10] = 0x03
    rec[11] = barrier_idx
    rec[12] = 0x40
    rec[13] = 0x00
    rec[14] = 0x02
    rec[15] = 0x00
    return bytes(rec)


def _build_terminal(text_size: int, has_branch: bool,
                     num_gprs: int, num_barrier_regions: int) -> bytes:
    """Terminal record (type 02 sub=0x38) — always the last body record.

    Second-half byte[4] = text_size_pages - 1 (for pages > 1) or indicator.
    """
    rec = bytearray(32)
    rec[0] = 0x02
    rec[1] = 0x38
    rec[2] = 0x0e
    rec[3] = 0x32
    rec[4] = 0xf8
    rec[5] = 0x00
    # byte[6]: 0x40 if branch, 0x50 if no branch
    rec[6] = 0x40 if has_branch else 0x50
    rec[7] = 0x11
    # bytes 8-9: zeros
    # byte 10-11: zeros (always)
    # byte[12]: 0x82 if reg>8, 0x02 if reg<=8
    rec[12] = 0x82 if num_gprs > 8 else 0x02
    # byte[13]: barrier region count indicator
    rec[13] = max(num_barrier_regions - 1, 0) if not has_branch else 0x00
    rec[14] = 0x0a
    rec[15] = 0x00

    # Second half
    rec[16] = 0x00
    rec[17] = 0x02
    # byte[18]: number of scheduling regions (1-based)
    rec[18] = max(num_barrier_regions, 1)
    # byte[19]: mode/summary
    if has_branch:
        rec[19] = 0x40
    elif num_gprs <= 8:
        rec[19] = 0x82
    else:
        rec[19] = 0x02

    # byte[20]: (text_size - 256) / 128, i.e. (text_size >> 7) - 2
    # 256B -> 0, 384B -> 1, 512B -> 2, 768B -> 4
    rec[20] = max((text_size >> 7) - 2, 0)

    return bytes(rec)


def _build_trailer(num_gprs: int, has_branch: bool,
                   num_barrier_regions: int) -> bytes:
    """2-byte trailer.

    byte[0]: 0xd0 for standard, 0x50 for high-register (>14 GPRs)
    byte[1]: complexity marker — observed values correlate with record count.
    """
    # Trailer byte 0
    if num_gprs > 14:
        b0 = 0x50
    else:
        b0 = 0xd0

    # Trailer byte 1: appears to be a hash/counter of the body
    # Observed patterns:
    #   reg=8, no branch: 0x04
    #   reg=10, no branch: 0x07
    #   reg=10-12, with branch: 0x06-0x07
    #   reg=15, no branch: 0x05
    #   reg=19+, with branch: 0x05-0x07
    # Approximation: number of type-01 records + base
    b1 = 0x04 + num_barrier_regions
    if has_branch:
        b1 += 1
    if b1 > 0x0f:
        b1 = 0x0f

    return bytes([b0, b1])


# ---------------------------------------------------------------------------
# Filler computation
# ---------------------------------------------------------------------------

def compute_filler_count(num_gprs: int, num_type01: int,
                          num_type02_22: int) -> int:
    """Compute the number of 4-byte filler blocks to insert before terminal.

    Fillers pad the capmerc body for register pressure. Analysis shows:
    - reg <= 14 with sufficient records: 0 fillers
    - reg = 15 with few records: up to 5 fillers
    - reg >= 16: typically 3 fillers for complex kernels

    The pattern appears to be: fillers ensure minimum body size thresholds.
    For reg > 14, there's a minimum body size of ~130 bytes before terminal.
    """
    if num_gprs <= 14:
        # Low register pressure: no fillers needed unless body is very small
        return 0

    # Records contribute: type01 = 16B, type02_22 = 32B each
    body_recs_size = num_type01 * 16 + num_type02_22 * 32

    # Target minimum body before terminal (excluding terminal itself)
    # Observed: force_highreg has 2*16 + 2*32 = 96B records + 5*4=20B filler = 116B
    # decomposed has 2*16 + 3*32 = 128B records + 1*4=4B filler = 132B
    # vecadd_noif has 5*16 + 2*32 = 144B records + 3*4=12B filler = 156B
    # Minimum body before terminal appears to be ~128-132 bytes
    MIN_BODY_BEFORE_TERM = 128

    if body_recs_size >= MIN_BODY_BEFORE_TERM:
        # Already have enough records, but complex kernels still add 3
        if num_gprs >= 16:
            return 3
        else:
            return 1
    else:
        # Pad up to minimum
        deficit = MIN_BODY_BEFORE_TERM - body_recs_size
        return max((deficit + 3) // 4, 1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_capmerc(
    num_gprs: int,
    text_size: int,
    has_stg: bool = True,
    has_ldg: bool = False,
    has_branch: bool = False,
    has_ur_ops: bool = True,
    has_shift: bool = False,
    has_imad: bool = False,
    has_isetp: bool = False,
    has_fadd: bool = False,
    num_barrier_regions: int = 2,
) -> bytes:
    """Build a complete .nv.capmerc.text.{kernel} section.

    Args:
        num_gprs: Number of GPRs allocated (will be clamped to min 8).
        text_size: Size of .text section in bytes.
        has_stg: Kernel uses STG (store global).
        has_ldg: Kernel uses LDG (load global).
        has_branch: Kernel has conditional branches or predicated EXIT.
        has_ur_ops: Kernel uses uniform register operations.
        has_shift: Kernel uses SHF (shift) instructions.
        has_imad: Kernel uses IMAD instructions.
        has_isetp: Kernel uses ISETP (set predicate) instructions.
        has_fadd: Kernel uses FADD/FFMA (floating point) instructions.
        num_barrier_regions: Number of scoreboard barrier regions in SASS.

    Returns:
        Complete capmerc section bytes.
    """
    num_gprs = max(num_gprs, 8)
    text_size_pages = max(text_size // 256, 1)

    # --- Build body records ---
    body_records = []

    # 1. Universal prologue (always first)
    body_records.append(_build_type01_prologue())

    # 2. Additional type-01 records based on instruction classes
    if has_branch:
        # Branch-containing kernels get an extra ALU record for the branch path
        body_records.append(_build_type01_alu_basic(region_idx=0))

    if has_ur_ops and (has_branch or has_ldg or has_fadd):
        # UR-ALU descriptor for kernels using uniform regs with complexity
        body_records.append(_build_type01_ur_alu())

    # 3. First barrier region (type 02 sub=0x22)
    body_records.append(
        _build_type02_barrier_first(
            has_branch=has_branch,
            has_ur_ops=has_ur_ops,
            barrier_idx=0,
            text_size_pages=text_size_pages,
        )
    )

    # 4. Additional type-01 records for complex kernels
    if num_gprs > 14 or text_size_pages > 1:
        body_records.append(_build_type01_alu_basic(region_idx=1))

    # 5. Middle barrier regions (for multi-region kernels)
    if num_barrier_regions > 2:
        for i in range(1, num_barrier_regions - 1):
            body_records.append(
                _build_type02_barrier_mid(
                    barrier_idx=i,
                    has_branch=has_branch,
                    has_ur_ops=has_ur_ops,
                    text_size_pages=text_size_pages,
                )
            )

    # 6. Last barrier region
    body_records.append(
        _build_type02_barrier_last(
            barrier_idx=max(num_barrier_regions - 1, 1),
            has_branch=has_branch,
            has_ur_ops=has_ur_ops,
        )
    )

    # 7. STG descriptor (if kernel uses stores)
    if has_stg:
        stg_region = max(num_barrier_regions - 1, 1)
        body_records.append(_build_type01_stg(region_idx=stg_region))

    # Count record types for filler computation
    num_type01 = sum(1 for r in body_records if r[0] == 0x01)
    num_type02_22 = sum(1 for r in body_records if r[0] == 0x02 and r[1] == 0x22)

    # 8. Filler blocks
    filler_count = compute_filler_count(num_gprs, num_type01, num_type02_22)
    filler_block = FILLER_WITH_BRANCH if has_branch else FILLER_NO_BRANCH

    # 9. Terminal record (always last)
    terminal = _build_terminal(text_size, has_branch, num_gprs,
                                num_barrier_regions)

    # --- Assemble header ---
    cap_mask = compute_capability_mask(
        has_stg=has_stg,
        has_ldg=has_ldg,
        has_branch=has_branch,
        has_shift=has_shift,
        has_ur_ops=has_ur_ops,
        has_imad=has_imad,
        has_isetp=has_isetp,
        has_fadd=has_fadd,
        num_gprs=num_gprs,
        num_barrier_regions=num_barrier_regions,
        text_size=text_size,
    )

    header = bytearray(16)
    header[0:8] = CAPMERC_MAGIC
    header[8] = num_gprs
    struct.pack_into('<I', header, 12, cap_mask)

    # --- Assemble trailer ---
    trailer = _build_trailer(num_gprs, has_branch, num_barrier_regions)

    # --- Concatenate everything ---
    buf = bytearray(header)
    for rec in body_records:
        buf.extend(rec)
    for _ in range(filler_count):
        buf.extend(filler_block)
    buf.extend(terminal)
    buf.extend(trailer)

    return bytes(buf)


# ---------------------------------------------------------------------------
# Convenience: analyze SASS instructions to determine flags
# ---------------------------------------------------------------------------

# SM_120 opcode categories
_MEM_STORE_OPCODES = {0x385, 0x38d, 0x986}  # STG.E, STS, STG.E.64
_MEM_LOAD_OPCODES = {0x981, 0x389, 0x30c}   # LDG.E (0x981, was 0x381), LDS, LDSM
_BRANCH_OPCODES = {0x947, 0x94d}             # BRA, EXIT (predicated)
_SHIFT_OPCODES = {0x819}                     # SHF
_IMAD_OPCODES = {0x824, 0x825, 0xc24, 0xc11, 0x224, 0x225}  # IMAD variants (incl R-UR 0xc24)
_ISETP_OPCODES = {0x86c, 0xc0c, 0x20c}      # ISETP variants (R-R 0x20c, R-UR 0xc0c)
_UR_OPCODES = {0x9c3, 0x7ac, 0xb82}         # S2UR (0x9c3), ULDC.64, LDCU
_FADD_OPCODES = {0x221, 0x220, 0x223}       # FADD, FMUL, FFMA
_CONST_LOAD_OPCODES = {0x182, 0x189, 0x431, 0x7ac, 0xb82}  # LDCU variants


def analyze_sass_for_capmerc(sass_bytes: bytes) -> dict:
    """Analyze a SASS instruction stream and return capmerc parameters.

    Args:
        sass_bytes: Raw SASS .text section bytes.

    Returns:
        Dict with keys matching build_capmerc() parameters.
    """
    has_stg = False
    has_ldg = False
    has_branch = False
    has_shift = False
    has_ur_ops = False
    has_imad = False
    has_isetp = False
    has_fadd = False
    max_reg = 0
    barrier_count = 0
    has_predicated_exit = False

    for i in range(0, len(sass_bytes), 16):
        if i + 16 > len(sass_bytes):
            break

        lo = struct.unpack_from('<Q', sass_bytes, i)[0]
        hi = struct.unpack_from('<Q', sass_bytes, i + 8)[0]

        opcode = lo & 0xFFF
        pred = (lo >> 12) & 0xF
        dest = (lo >> 16) & 0xFF
        src0 = (lo >> 24) & 0xFF
        src1 = (lo >> 32) & 0xFF
        src2 = (hi >> 0) & 0xFF

        # Track max register
        for r in [dest, src0, src1, src2]:
            if r != 255 and r < 128 and r > max_reg:
                max_reg = r

        # Classify opcode
        if opcode in _MEM_STORE_OPCODES:
            has_stg = True
        if opcode in _MEM_LOAD_OPCODES:
            has_ldg = True
        if opcode in _SHIFT_OPCODES:
            has_shift = True
        if opcode in _IMAD_OPCODES:
            has_imad = True
        if opcode in _ISETP_OPCODES:
            has_isetp = True
        if opcode in _UR_OPCODES:
            has_ur_ops = True
        if opcode in _CONST_LOAD_OPCODES:
            has_ur_ops = True  # const loads use uniform path
        if opcode in _FADD_OPCODES:
            has_fadd = True

        # Detect predicated EXIT (conditional branch)
        if opcode == 0x94d and pred != 7:  # EXIT with predicate != PT
            has_predicated_exit = True
        if opcode == 0x947:  # BRA
            # Check if this is a self-loop (padding) vs real branch
            # Self-loops go to offset-16 (the previous instruction)
            pass

        # Count barrier write operations from control words
        raw24 = sass_bytes[i + 13] | (sass_bytes[i + 14] << 8) | (sass_bytes[i + 15] << 16)
        ctrl = raw24 >> 1
        wbar = (ctrl >> 15) & 1
        if wbar:
            barrier_count += 1

    if has_predicated_exit:
        has_branch = True

    # Number of barrier regions = barrier writes + 1 (initial region)
    # Minimum 2 (prologue region + main body)
    num_barrier_regions = max(barrier_count + 1, 2)

    # GPR count: round up to nearest even, minimum 8
    num_gprs = max(max_reg + 1, 8)
    if num_gprs % 2 == 1:
        num_gprs += 1

    return {
        'num_gprs': num_gprs,
        'text_size': len(sass_bytes),
        'has_stg': has_stg,
        'has_ldg': has_ldg,
        'has_branch': has_branch,
        'has_ur_ops': has_ur_ops,
        'has_shift': has_shift,
        'has_imad': has_imad,
        'has_isetp': has_isetp,
        'has_fadd': has_fadd,
        'num_barrier_regions': num_barrier_regions,
    }


def build_capmerc_from_sass(sass_bytes: bytes, num_gprs: Optional[int] = None) -> bytes:
    """Convenience: analyze SASS and generate capmerc in one call.

    Args:
        sass_bytes: Raw SASS .text section bytes.
        num_gprs: Override for GPR count (if None, auto-detected from SASS).

    Returns:
        Complete capmerc section bytes.
    """
    params = analyze_sass_for_capmerc(sass_bytes)
    if num_gprs is not None:
        params['num_gprs'] = max(num_gprs, 8)
    return build_capmerc(**params)
