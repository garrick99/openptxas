"""
sass/scoreboard.py — SM_120 scoreboard emulator for ctrl word generation.

Generates correct dependency barrier (ctrl) values for any instruction stream.
Replaces the manual ptxas-matching approach with automated tracking.

SM_120 ctrl word (23 bits):
  [22:17] stall  — cycle count to wait (0 for barrier-based scheduling)
  [16]    yield  — yield hint
  [15]    wbar   — write-after-read barrier flag
  [14:10] rbar   — read barrier mask (which scoreboard slots to wait for)
  [9:4]   wdep   — write dependency slot (scoreboard slot for this result)
  [3:0]   misc   — instruction sequence counter (wraps mod 8)

Scoreboard slots (from ptxas RE):
  0x31 = LDC/LDCU result slot
  0x33 = LDS result slot
  0x35 = LDG result slot (first)
  0x37 = LDG result slot (second)
  0x3e = ALU result slot (SHF, IADD, FADD, etc.)
  0x3f = no write tracking (EXIT, BRA, STG, BAR)

Read barrier encoding:
  0x01 = no wait (default)
  0x03 = wait for LDG/STG slot
  0x05 = wait for 2nd LDG slot
  0x09 = wait for LDG data (first consumer after LDG)
"""

from __future__ import annotations
import struct
from sass.isel import SassInstr


# Opcode classification
_OPCODES_LDG = {0x981}
_OPCODES_LDC = {0xb82, 0x7ac, 0x919, 0x9c3}  # LDC, LDCU, S2R, S2UR
_OPCODES_LDS = {0x984}
_OPCODES_STG = {0x986}
_OPCODES_STS = {0x988}
_OPCODES_BAR = {0xb1d}
_OPCODES_CTRL = {0x94d, 0x947, 0x918}  # EXIT, BRA, NOP
_OPCODES_ALU = {0x819, 0x235, 0x210, 0x212, 0x221, 0x223, 0x202,
                0x824, 0x825, 0x224, 0x23c, 0x237, 0x431}
# Note: IADD.64-UR (0xc35) uses wdep=0x3f (no tracking) + stall=1.
# The 1-cycle stall ensures the result is ready for the subsequent LDG/STG.
_OPCODES_IADD64_UR = {0xc35}
_OPCODES_SMEM_SETUP = {0x9c3, 0x882, 0x291}  # S2UR, UMOV, ULEA


def _get_opcode(raw: bytes) -> int:
    return struct.unpack_from('<Q', raw, 0)[0] & 0xFFF


def _get_dest_reg(raw: bytes) -> int:
    """Get the destination register index, or -1 if none."""
    opcode = _get_opcode(raw)
    if opcode in _OPCODES_CTRL | _OPCODES_STG | _OPCODES_STS | _OPCODES_BAR:
        return -1  # no GPR dest
    return raw[2]


def _get_src_regs(raw: bytes) -> set[int]:
    """Get source register indices this instruction reads from GPRs."""
    opcode = _get_opcode(raw)
    regs = set()

    if opcode in _OPCODES_LDG:
        # LDG: src_addr at b3
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
    elif opcode in _OPCODES_STG:
        # STG: addr at b3, data at b4 (NOT b8 — b8 is UR descriptor)
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
        if raw[4] < 255: regs.add(raw[4])
    elif opcode in _OPCODES_STS:
        # STS: data at b4
        if raw[4] < 255: regs.add(raw[4])
    elif opcode in _OPCODES_ALU:
        # ALU: src0 at b3, src1 at b4, src2 at b8 (varies by opcode)
        if raw[3] < 255: regs.add(raw[3])
        if opcode == 0x210:  # IADD3: src0=b3, src1=b4, src2=b8
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255: regs.add(raw[8])
        elif opcode == 0x235:  # IADD.64: src0=b3 pair, src1=b4 pair
            if raw[3] < 255: regs.add(raw[3]+1)
            if raw[4] < 255: regs |= {raw[4], raw[4]+1}
        elif opcode in (0x819,):  # SHF: src0=b3, src1=b8
            if raw[8] < 255: regs.add(raw[8])
        elif opcode in (0x23c, 0x237):  # HMMA/IMMA: a=b3(4 regs), b=b4(2), c=b8(4)
            for r in range(4): regs.add(raw[3]+r)
            for r in range(2): regs.add(raw[4]+r)
            for r in range(4): regs.add(raw[8]+r)
        elif opcode in (0x221, 0x223):  # FADD/FFMA: src0=b3, src1=b4, src2=b8
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255 and opcode == 0x223: regs.add(raw[8])
        elif opcode in (0x824, 0x825, 0x224):  # IMAD variants: src0=b3, src1=b4, src2=b8
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255: regs.add(raw[8])
    return regs


def _get_dest_regs(raw: bytes) -> set[int]:
    """Get destination register indices this instruction writes."""
    opcode = _get_opcode(raw)
    dest = raw[2]
    regs = set()

    if opcode in _OPCODES_LDG:
        if dest < 255:
            regs.add(dest)
            # Check if 64-bit or 128-bit load
            b9 = raw[9]
            if b9 in (0x1b, 0x9b):  # LDG.E.64
                regs.add(dest+1)
            elif b9 in (0x1d, 0x9d):  # LDG.E.128
                regs |= {dest+1, dest+2, dest+3}
            else:
                regs.add(dest+1)  # default to 64-bit
    elif opcode in _OPCODES_LDC:
        if dest < 255:
            regs.add(dest)
            if raw[9] == 0x0a:  # 64-bit
                regs.add(dest+1)
    elif opcode in _OPCODES_LDS:
        if dest < 255: regs.add(dest)
    elif opcode == 0x235:  # IADD.64
        if dest < 255: regs |= {dest, dest+1}
    elif opcode in (0x23c, 0x237):  # HMMA/IMMA: writes 4 regs
        if dest < 255: regs |= {dest, dest+1, dest+2, dest+3}
    elif opcode in _OPCODES_ALU:
        if dest < 255: regs.add(dest)
    return regs


_ldcu_slot_counter = [0]  # mutable counter for rotating LDCU wdep slots
_ldc_slot_counter = [0]   # mutable counter for rotating LDC wdep slots

def _wdep_for_opcode(opcode: int) -> int:
    """Assign the scoreboard write-dependency slot for an opcode."""
    if opcode == 0x7ac:  # LDCU: rotating slots 0x31, 0x33, 0x35, 0x37
        slots = [0x31, 0x33, 0x35, 0x37]
        slot = slots[_ldcu_slot_counter[0] % len(slots)]
        _ldcu_slot_counter[0] += 1
        return slot
    if opcode in _OPCODES_LDC:
        return 0x31
    if opcode in _OPCODES_LDS:
        return 0x33
    if opcode in _OPCODES_LDG:
        return 0x35
    if opcode in _OPCODES_IADD64_UR:
        return 0x3f  # No write tracking — stall=1 handles the dependency
    if opcode in _OPCODES_ALU | _OPCODES_SMEM_SETUP:
        return 0x3e
    # No write tracking for control flow, stores, barriers
    return 0x3f


def _patch_ctrl(raw: bytes, ctrl: int) -> bytes:
    buf = bytearray(raw)
    raw24 = (ctrl & 0x7FFFFF) << 1
    buf[13] = raw24 & 0xFF
    buf[14] = (raw24 >> 8) & 0xFF
    buf[15] = ((raw24 >> 16) & 0xFF) | (buf[15] & 0x04)  # preserve SHF reuse flag
    return bytes(buf)


def assign_ctrl(instrs: list[SassInstr]) -> list[SassInstr]:
    """
    Assign ctrl values to an instruction stream using scoreboard emulation.

    Tracks which registers are written by long-latency ops (LDG, LDC, LDS)
    and sets rbar on consumer instructions to wait for the result.
    """
    # Track which registers have pending long-latency writes
    # Maps reg_index → (slot_index, wdep_slot) for pending writes
    pending_writes: dict[int, tuple[int, int]] = {}
    # Track UR writes separately (LDCU destinations)
    pending_ur_writes: dict[int, tuple[int, int]] = {}  # ur_index → (slot_index, wdep)

    # rbar encoding: maps wdep_slot → rbar bit pattern
    _WDEP_TO_RBAR = {
        0x31: 0x03,   # Slot 0 → rbar=0x03 (bit 1+0)
        0x33: 0x05,   # Slot 1 → rbar=0x05 (bit 2+0)
        0x35: 0x09,   # Slot 2 → rbar=0x09 (bit 3+0)
        0x37: 0x11,   # Slot 3 → rbar=0x11 (bit 4+0) - used for UR4 descriptor
        0x3e: 0x03,   # ALU → rbar=0x03
    }

    misc_counter = 0
    ldg_count = 0
    _ldcu_slot_counter[0] = 0  # reset per kernel
    _ldc_slot_counter[0] = 0   # reset per kernel
    result = []

    for i, si in enumerate(instrs):
        opcode = _get_opcode(si.raw)

        # Determine wdep for this instruction
        wdep = _wdep_for_opcode(opcode)

        # For second LDG, use a different slot
        if opcode in _OPCODES_LDG:
            ldg_count += 1
            if ldg_count > 1:
                wdep = 0x37

        # Determine rbar: check if any source register has a pending long-latency write
        rbar = 0x01  # default: no wait
        src_regs = _get_src_regs(si.raw)
        for r in src_regs:
            if r in pending_writes:
                _, pending_wdep = pending_writes[r]
                candidate_rbar = _WDEP_TO_RBAR.get(pending_wdep, 0x01)
                # For LDG consumers: first consumer gets 0x09, subsequent get 0x03
                if pending_wdep == 0x35:
                    candidate_rbar = 0x09
                rbar = max(rbar, candidate_rbar)

        # STS needs rbar=0x09 if writing data that came from LDG
        if opcode in _OPCODES_STS:
            for r in src_regs:
                if r in pending_writes:
                    rbar = 0x09

        # STG needs rbar=0x03
        if opcode in _OPCODES_STG:
            rbar = max(rbar, 0x03)

        # LDCU consumers: any instruction using UR operands needs rbar for LDCU
        # Check byte 4 for UR source in R-UR instructions
        if opcode in (0xc35, 0xc0c, 0xc24):  # IADD.64-UR, ISETP R-UR, IMAD R-UR
            ur_src = si.raw[4]
            if ur_src in pending_ur_writes:
                _, pw = pending_ur_writes[ur_src]
                if pw in _WDEP_TO_RBAR:
                    rbar = max(rbar, _WDEP_TO_RBAR[pw])
        # LDG/STG use descriptor from UR (LDG: b4=UR, STG: b8=UR)
        # NOTE: UR4 descriptor is loaded via LDCU in the preamble with
        # hardcoded ctrl. Body LDCUs (for pointer params) also track in
        # pending_ur_writes. Only apply rbar if UR4 is in pending_ur_writes
        # (i.e., LDCU UR4 went through the scoreboard in the body).
        if opcode in _OPCODES_LDG:
            ur_desc = si.raw[4]
            if ur_desc in pending_ur_writes:
                _, pw = pending_ur_writes[ur_desc]
                if pw in _WDEP_TO_RBAR:
                    rbar = max(rbar, _WDEP_TO_RBAR[pw])
        if opcode in _OPCODES_STG:
            ur_desc = si.raw[8]
            if ur_desc in pending_ur_writes:
                _, pw = pending_ur_writes[ur_desc]
                if pw in _WDEP_TO_RBAR:
                    rbar = max(rbar, _WDEP_TO_RBAR[pw])

        # BAR.SYNC and EXIT get special ctrl
        if opcode in _OPCODES_BAR:
            wdep = 0x3f
            rbar = 0x01
        if opcode == 0x94d:  # EXIT
            rbar = 0x01
            wdep = 0x3f

        # Build ctrl — mostly stall=0, barrier-based scheduling.
        # Exception: first IADD.64-UR after LDCU gets stall=1 (ptxas pattern).
        stall = 0
        if opcode == 0xc35:  # IADD.64-UR: add stall=1 for LDCU dependency
            stall = 1
        if opcode == 0x224:  # IMAD R-R: needs stall=1 (multi-cycle, wide write)
            stall = 1
        if opcode == 0xb82:  # LDC/S2R: add stall=4 for WAW hazard
            # SM_120 LDC.64 has delayed writes. When reusing the same
            # register pair for multiple LDC.64s, the previous LDC's
            # delayed write can overwrite the new value. Stall ensures
            # the previous LDC completes before the new one starts.
            src_regs_set = _get_src_regs(si.raw)
            dest_r = si.raw[2]
            if dest_r in pending_writes:
                stall = max(stall, 4)
        misc = misc_counter & 0xF
        ctrl = (stall << 17) | (rbar << 10) | (wdep << 4) | misc
        misc_counter += 1

        # Track this instruction's writes for future consumers.
        dest_regs = _get_dest_regs(si.raw)
        if wdep != 0x3f:
            for r in dest_regs:
                pending_writes[r] = (i, wdep)
        # Track UR writes: LDCU (0x7ac) and S2UR (0x9c3)
        if opcode == 0x7ac:
            ur_dest = si.raw[2]  # UR destination index
            pending_ur_writes[ur_dest] = (i, wdep)
            # Also track UR+1 for 64-bit pairs
            pending_ur_writes[ur_dest + 1] = (i, wdep)
        elif opcode == 0x9c3:  # S2UR: writes single UR (dest at byte 2)
            ur_dest = si.raw[2]
            pending_ur_writes[ur_dest] = (i, wdep)
        elif wdep == 0x3e:
            # ALU write: clear pending long-latency tracking for written regs
            # (the register now has a new value from a fast ALU op)
            for r in dest_regs:
                pending_writes.pop(r, None)
        # Note: we do NOT clear pending_writes for consumed SOURCE registers.
        # Multiple instructions may read from the same LDG output, and each
        # needs the rbar wait independently.

        patched = _patch_ctrl(si.raw, ctrl)
        result.append(SassInstr(patched, si.comment))

    return result
