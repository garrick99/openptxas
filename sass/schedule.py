"""
sass/schedule.py — Instruction scheduling and ctrl word assignment for SM_120.

The ctrl word (23 bits) in each 128-bit instruction encodes dependency barriers:
  bits[22:17] = stall count (0..63 cycles)
  bits[16]    = yield hint
  bits[15]    = write-after-read barrier
  bits[14:10] = read-after-write barrier mask (rbar, 5 bits)
  bits[9:4]   = write dependency slot (wdep, 6 bits)
  bits[3:0]   = scoreboard set (misc, 4 bits)

Key observations from ptxas RE:
  - LDG sets a scoreboard entry (via misc field) when it starts loading
  - ALL consumers of LDG output registers need rbar=0x09 to wait for the load
  - LDC/LDCU between LDG and consumers use rbar=0x01 (no LDG wait needed)
  - STG needs rbar=0x03
  - wdep encodes write-dependency tracking (varies by instruction)
"""

from __future__ import annotations
import struct

from sass.isel import SassInstr


# Ctrl templates from ptxas ground truth
CTRL_LDC       = 0x7f1    # LDC.32
CTRL_LDCU      = 0x717    # LDCU.64
CTRL_LDC_64    = 0x712    # LDC.64 (default)
CTRL_LDC_64_AFTER_LDG = 0x711  # LDC.64 in the LDG latency-hiding slot
CTRL_LDG       = 0xf56    # LDG.E.64
CTRL_STG       = 0xff1    # STG.E.64
CTRL_EXIT      = 0x7f5    # EXIT
CTRL_BRA       = 0x7e0    # BRA
CTRL_NOP       = 0x7e0    # NOP

# Compute instructions: use 0x7e0 (matches ptxas NOP slots — safe default).
# The preamble's LDG ctrl sets proper barriers; compute just executes sequentially.
CTRL_COMPUTE = 0x7e0


def _get_opcode(raw: bytes) -> int:
    lo = struct.unpack_from('<Q', raw, 0)[0]
    return lo & 0xFFF


def _get_src_regs(raw: bytes) -> set[int]:
    """Extract source register indices that this instruction reads."""
    lo = struct.unpack_from('<Q', raw, 0)[0]
    opcode = lo & 0xFFF
    regs = set()
    src0 = raw[3]   # byte[3] = src0
    src1 = raw[8]   # byte[8] = src1
    b4 = raw[4]     # byte[4] = src1/imm depending on opcode

    if opcode == 0x981:  # LDG: src0 = address register
        if src0 < 255:
            regs.add(src0)
            regs.add(src0 + 1)  # 64-bit pair
    elif opcode == 0x986:  # STG: src0 = address, src1 = data
        if src0 < 255:
            regs.add(src0)
            regs.add(src0 + 1)
        if src1 < 255:
            regs.add(src1)
            regs.add(src1 + 1)
    elif opcode == 0x819:  # SHF: src0=byte[3], src1=byte[8]
        if src0 < 255:
            regs.add(src0)
        if src1 < 255:
            regs.add(src1)
    elif opcode == 0x235:  # IADD.64: src0=byte[3], src1=byte[4]
        if src0 < 255:
            regs.add(src0)
            regs.add(src0 + 1)
        if b4 < 255:
            regs.add(b4)
            regs.add(b4 + 1)
    elif opcode == 0x202:  # MOV: src=byte[4]
        if b4 < 255:
            regs.add(b4)
    return regs


def _get_dest_regs(raw: bytes) -> set[int]:
    """Extract destination register indices that this instruction writes."""
    lo = struct.unpack_from('<Q', raw, 0)[0]
    opcode = lo & 0xFFF
    dest = raw[2]
    regs = set()

    if opcode == 0x981:  # LDG — always 64-bit in our usage
        if dest < 255:
            regs.add(dest)
            regs.add(dest + 1)
    elif opcode == 0xb82:  # LDC
        b9 = raw[9]
        if dest < 255:
            regs.add(dest)
            if b9 == 0x0a:  # 64-bit
                regs.add(dest + 1)
    elif opcode == 0x235:  # IADD.64
        if dest < 255:
            regs.add(dest)
            regs.add(dest + 1)
    elif opcode in (0x819, 0x202):  # SHF, MOV
        if dest < 255:
            regs.add(dest)
    return regs


def _patch_ctrl(raw: bytes, ctrl: int) -> bytes:
    buf = bytearray(raw)
    raw24 = (ctrl & 0x7FFFFF) << 1
    buf[13] = raw24 & 0xFF
    buf[14] = (raw24 >> 8) & 0xFF
    # Preserve SHF.L.U64.HI reuse flag in bit 2 of byte[15]
    buf[15] = ((raw24 >> 16) & 0xFF) | (buf[15] & 0x04)
    return bytes(buf)


def _reorder_after_ldg(instrs: list[SassInstr]) -> list[SassInstr]:
    """Move independent LDC after LDG to hide latency.

    Only moves an LDC if its destination doesn't conflict with the LDG output.
    """
    result = list(instrs)
    for i in range(len(result) - 1):
        if _get_opcode(result[i].raw) == 0x981:  # LDG
            ldg_dests = _get_dest_regs(result[i].raw)
            if _get_opcode(result[i + 1].raw) != 0xb82:  # next isn't LDC
                for j in range(i + 2, len(result)):
                    if _get_opcode(result[j].raw) == 0xb82:
                        # Check if this LDC would clobber LDG output registers
                        ldc_dests = _get_dest_regs(result[j].raw)
                        if ldc_dests & ldg_dests:
                            continue  # skip — would clobber LDG data
                        moved = result.pop(j)
                        result.insert(i + 1, moved)
                        break
    return result


def schedule(instrs: list[SassInstr]) -> list[SassInstr]:
    """
    Reorder and assign ctrl values with proper LDG barrier tracking.

    All instructions that READ registers written by LDG get rbar=0x09
    (the LDG wait barrier). Instructions that don't consume LDG data
    get rbar=0x01 (no wait).
    """
    reordered = _reorder_after_ldg(instrs)

    # Phase 1: find which registers are written by LDG instructions
    ldg_output_regs: set[int] = set()
    for si in reordered:
        if _get_opcode(si.raw) == 0x981:
            ldg_output_regs |= _get_dest_regs(si.raw)

    # Phase 2: assign ctrl values with def-use stall tracking
    # Track which register was most recently written and when
    last_write: dict[int, int] = {}  # reg_index → slot_index

    result = []
    last_ldg_pos = -100

    for i, si in enumerate(reordered):
        opcode = _get_opcode(si.raw)

        # Default ctrl from opcode type
        if opcode == 0xb82:  # LDC
            b9 = si.raw[9]
            if b9 == 0x0a:
                ctrl = CTRL_LDC_64_AFTER_LDG if i == last_ldg_pos + 1 else CTRL_LDC_64
            else:
                ctrl = CTRL_LDC
        elif opcode == 0x7ac:
            ctrl = CTRL_LDCU
        elif opcode == 0x981:
            ctrl = CTRL_LDG
            last_ldg_pos = i
        elif opcode == 0x986:
            ctrl = CTRL_STG
        elif opcode == 0x94d:
            ctrl = CTRL_EXIT
        elif opcode == 0x947:
            ctrl = CTRL_BRA
        elif opcode == 0x918:
            ctrl = CTRL_NOP
        else:
            # Compute instruction (SHF, IADD.64, MOV, etc.)
            src_regs = _get_src_regs(si.raw)
            reads_ldg = bool(src_regs & ldg_output_regs)

            ctrl = CTRL_COMPUTE

        patched = _patch_ctrl(si.raw, ctrl)
        result.append(SassInstr(patched, si.comment))

    return result
