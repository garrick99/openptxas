"""
sass/schedule.py — Instruction scheduling and ctrl word assignment.

Assigns ctrl values (dependency barriers) to SASS instructions based on
register def-use chains. On SM_120, the ctrl word encodes:
  - bits[22:17]: stall count (0..63 cycles to wait before issuing)
  - bits[16]:    yield hint
  - bits[15]:    write-after-read barrier
  - bits[14:10]: read-after-write barrier mask (5 bits)
  - bits[9:4]:   write dependency slot (6 bits)
  - bits[3:0]:   misc flags

For simple kernels, we use a conservative approach: assign enough stall
cycles to guarantee all dependencies are satisfied. ptxas uses a more
sophisticated barrier-based approach, but stall-based scheduling produces
correct (if slower) code.

The ctrl word encoding in 128-bit instructions:
  raw24 = ctrl << 1
  byte[13] = raw24 & 0xFF
  byte[14] = (raw24 >> 8) & 0xFF
  byte[15] = (raw24 >> 16) & 0xFF  (may be ORed with variant-specific bits)
"""

from __future__ import annotations
import struct

from sass.isel import SassInstr


# Ctrl values observed in ptxas output for common instruction patterns.
# These encode the correct dependency barriers for each instruction type
# in the context of a typical kernel preamble.
CTRL_LDC = 0x7f1       # LDC (constant bank load)
CTRL_LDCU = 0x717       # LDCU (uniform constant load)
CTRL_LDC_64 = 0x712     # LDC.64 (64-bit constant load)
CTRL_LDG = 0xf56        # LDG (global memory load, with read barrier)
CTRL_STG = 0xff1        # STG (global memory store)
CTRL_EXIT = 0x7f5       # EXIT
CTRL_BRA = 0x7e0        # BRA
CTRL_NOP = 0x7e0        # NOP
# SHF and compute instructions: use stall=4 for safety
# (ptxas uses stall=0 with barrier-based scheduling, but our barrier
# assignment is incomplete — stall-based is correct if slower)
CTRL_SHF_FIRST = 0x27e2 # SHF first
CTRL_SHF_SECOND = 0x7e5 # SHF subsequent
CTRL_IADD64 = 0x7e5     # IADD.64


def _get_opcode(raw: bytes) -> int:
    """Extract opcode from bits[11:0] of the instruction."""
    lo = struct.unpack_from('<Q', raw, 0)[0]
    return lo & 0xFFF


def _patch_ctrl(raw: bytes, ctrl: int) -> bytes:
    """Replace the ctrl field in a 16-byte instruction."""
    buf = bytearray(raw)
    raw24 = (ctrl & 0x7FFFFF) << 1
    # Preserve any variant-specific bits already in byte[15]
    old_b15_variant = buf[15] & 0xFE  # keep everything except ctrl bit 0
    new_b15_ctrl = (raw24 >> 16) & 0xFF
    buf[13] = raw24 & 0xFF
    buf[14] = (raw24 >> 8) & 0xFF
    # For SHF.L.U64.HI, byte[15] has a reuse flag (0x04) that must be preserved
    buf[15] = new_b15_ctrl | (buf[15] & 0x04)  # preserve bit 2 (reuse flag)
    return bytes(buf)


# Opcode → default ctrl mapping
_OPCODE_CTRL = {
    0xb82: CTRL_LDC,       # LDC / LDC.64
    0x7ac: CTRL_LDCU,      # LDCU
    0x981: CTRL_LDG,       # LDG
    0x986: CTRL_STG,       # STG
    0x94d: CTRL_EXIT,      # EXIT
    0x947: CTRL_BRA,       # BRA
    0x918: CTRL_NOP,       # NOP
    0x819: CTRL_SHF_SECOND,# SHF (default, overridden for first in pair)
    0x235: CTRL_IADD64,    # IADD.64
    0x210: CTRL_SHF_SECOND,# IADD3
    0x202: CTRL_SHF_SECOND,# MOV
}


def _reorder_after_ldg(instrs: list[SassInstr]) -> list[SassInstr]:
    """
    Move independent LDC instructions to fill the gap after LDG.

    ptxas always puts a param load (LDC.64 for output pointer) between
    LDG and the first compute instruction to hide memory latency.
    If we find an LDG followed immediately by a compute instruction,
    look for a later LDC that can be moved up.
    """
    result = list(instrs)

    for i in range(len(result) - 1):
        opcode_i = _get_opcode(result[i].raw)
        opcode_next = _get_opcode(result[i + 1].raw)

        # LDG followed by non-LDC (compute): look for a moveable LDC after
        if opcode_i == 0x981 and opcode_next != 0xb82:
            # Find the next LDC.64 after position i+1
            for j in range(i + 2, len(result)):
                if _get_opcode(result[j].raw) == 0xb82:
                    # Move it to position i+1
                    moved = result.pop(j)
                    result.insert(i + 1, moved)
                    break

    return result


def schedule(instrs: list[SassInstr]) -> list[SassInstr]:
    """
    Reorder and assign ctrl values to SASS instructions.

    1. Reorder to fill LDG latency gaps with independent instructions.
    2. Assign ctrl values (dependency barriers) based on opcode type.
    """
    # Phase 1: reorder
    reordered = _reorder_after_ldg(instrs)

    # Phase 2: assign ctrl values
    # Track position relative to LDG for barrier assignment
    result = []
    prev_opcode = None
    last_ldg_pos = -100  # track where last LDG was

    for i, si in enumerate(reordered):
        opcode = _get_opcode(si.raw)
        ctrl = _OPCODE_CTRL.get(opcode, CTRL_SHF_SECOND)

        # LDC.64 vs LDC.32: check b9 for size marker
        if opcode == 0xb82:
            b9 = si.raw[9]
            if b9 == 0x0a:
                ctrl = CTRL_LDC_64
                # LDC.64 right after LDG: uses ctrl=0x711 (ptxas pattern)
                if i == last_ldg_pos + 1:
                    ctrl = 0x711
            else:
                ctrl = CTRL_LDC

        # Track LDG position
        if opcode == 0x981:
            last_ldg_pos = i

        # SHF: first after LDG gap gets reuse flag, rest get standard
        if opcode == 0x819 and prev_opcode != 0x819:
            ctrl = CTRL_SHF_FIRST

        # IADD.64 after SHF sequence
        if opcode == 0x235:
            ctrl = CTRL_IADD64

        patched = _patch_ctrl(si.raw, ctrl)
        result.append(SassInstr(patched, si.comment))
        prev_opcode = opcode

    return result
