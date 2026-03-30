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
from typing import NamedTuple
from sass.isel import SassInstr


class _OpMeta(NamedTuple):
    name: str
    min_gpr_gap: int  # minimum instruction gap between write and immediate GPR reader (0 = no constraint)
    wdep: int         # scoreboard write-dep slot (0x3e=ALU, 0x3f=no-track, etc.)
    misc: int         # ctrl misc nibble value (hardware-verified per opcode)


# SM_120 ALU instructions that require ≥1 instruction gap before a GPR consumer.
# The stall field is ignored by hardware; rbar alone does not gate adjacent ALU reads.
_OPCODE_META: dict[int, _OpMeta] = {
    0x210: _OpMeta('IADD3',      1, 0x3e, 1),
    0x212: _OpMeta('IADD3X',     1, 0x3e, 1),
    0x224: _OpMeta('IMAD.32',    1, 0x3e, 1),
    0x2a4: _OpMeta('IMAD.RR',   1, 0x3e, 1),   # R-R-R multiply (opcode 0x2a4, SM_120 validated)
    0x824: _OpMeta('IMAD',       1, 0x3e, 1),
    0x825: _OpMeta('IMAD.WIDE',  1, 0x3e, 1),   # IMAD.WIDE R-imm (64-bit result)
    0x225: _OpMeta('IMAD.WIDE.RR', 1, 0x3e, 1),  # IMAD.WIDE R-R
    0x819: _OpMeta('SHF',        1, 0x3e, 1),
    0x299: _OpMeta('SHF.VAR',   1, 0x3e, 1),   # variable-shift SHF (opcode 0x7299)
    0x219: _OpMeta('SHF.R.S32.HI.VAR', 1, 0x3e, 1),  # SHF.R.S32.HI variable-shift (shr.s32)
    0x221: _OpMeta('FADD',       1, 0x3e, 1),
    0x223: _OpMeta('FFMA',       1, 0x3e, 1),
    0x235: _OpMeta('IADD.64',    1, 0x3e, 1),
    0xc35: _OpMeta('IADD.64-UR', 1, 0x3e, 5),  # misc=5 per hardware bisect 2026-03-25
    0x202: _OpMeta('MOV',        0, 0x3e, 1),
    0x20c: _OpMeta('ISETP.RR',   0, 0x3e, 0),  # ISETP R-R: misc=0 (SM_120 predicate)
    0xc0c: _OpMeta('ISETP.RU',   0, 0x3e, 0),  # ISETP R-UR: misc=0 on SM_120
    0x431: _OpMeta('HFMA2',      1, 0x3e, 1),  # HFMA2 (half-precision FMA2, used as zero-init in div.u32)
    0x810: _OpMeta('IADD3.IMM',  1, 0x3e, 1),  # IADD3 with 32-bit immediate operand
    0x306: _OpMeta('I2F.U32.RP', 1, 0x3e, 1),  # I2F unsigned int to float, round toward +inf
    0x305: _OpMeta('F2I.FTZ.U32',1, 0x3e, 1),  # F2I float to unsigned int, truncate
    0x310: _OpMeta('F2F',        1, 0x3e, 1),  # F2F float-to-float precision conversion (F32↔F64)
    0x311: _OpMeta('F2I.F64',   1, 0x3e, 1),  # F2I.F64 float64-to-int32 conversion
    0x312: _OpMeta('I2F.F64',   1, 0x3e, 1),  # I2F.F64 int32-to-float64 conversion (writes pair)
    0x81a: _OpMeta('BFE_SEXT',  1, 0x3e, 1),  # BFE sign-extension step (bfe.s32 lowering)
}


# Opcode classification
_OPCODES_LDG = {0x981}
_OPCODES_ATOMG = {0x3a9}  # ATOMG.CAS (and future ATOMG variants with opcode 0x3a9)
_OPCODES_LDC = {0xb82, 0x7ac, 0x919, 0x9c3}  # LDC, LDCU, S2R, S2UR
_OPCODES_LDS = {0x984}
_OPCODES_STG = {0x986}
_OPCODES_STS = {0x988}
_OPCODES_BAR = {0xb1d}
_OPCODES_DFPU = {0xe29, 0xc28, 0xc2b}  # DADD, DMUL, DFMA (double-precision, wdep=0x33)
_OPCODES_CTRL = {0x94d, 0x947, 0x918}  # EXIT, BRA, NOP
_OPCODES_ALU = {
    # Integer arithmetic
    0x210,        # IADD3
    0x235,        # IADD.64
    0x202,        # IADD3.X (with carry)
    0x224, 0x2a4, 0xc24,  # IMAD R-R (old), IMAD R-R (validated), IMAD R-UR
    0x824, 0x825, 0x225, # IMAD.SHL.U32, IMAD.WIDE (imm), IMAD.WIDE (R-R)
    0x227,        # IMAD.HI.U32
    0x213,        # IABS
    0x248, 0x848, # VIMNMX R-R, R-imm (integer min/max)
    0x309, 0x301, # POPC, BREV
    0x300,        # FLO
    # Float arithmetic
    0x221,        # FADD
    0x223,        # FMUL / FFMA
    0x209,        # FMNMX (float min/max)
    0x308,        # MUFU (RCP, SQRT, SIN, COS, EX2, LG2)
    # Type conversion
    0x245,        # I2FP.F32.U32
    0x305,        # F2I.U32
    # Logic
    0x212,        # LOP3.LUT
    0x819,        # SHF (all variants: L/R, U32/U64/S32, HI/LO/W, constant shift)
    0x299,        # SHF.VAR (variable-shift SHF, shift amount in register)
    0x219,        # SHF.R.S32.HI.VAR (arithmetic right shift, variable amount)
    # Select / predicate
    0x207,        # SEL
    0x208,        # FSEL
    0x20b,        # FSETP
    0x20c,        # ISETP R-R
    0xc0c,        # ISETP R-UR
    # Permute / misc
    0x416,        # PRMT (immediate selector, opc=0x416)
    0x216,        # PRMT.REG (register selector, opc=0x216)
    0x589, 0xf89, 0x989,  # SHFL (reg-reg, reg-imm, imm-imm)
    0x806,        # VOTE.ANY
    # Matrix multiply (HMMA, IMMA)
    0x23c, 0x237,
    # Miscellaneous / div.u32 helpers
    0x431,        # HFMA2 (zero-init trick)
    0x810,        # IADD3 immediate form
    0x306,        # I2F.U32.RP
    0x305,        # F2I.FTZ.U32.TRUNC
    # Float precision conversion / integer↔float F64
    0x310,        # F2F (F32↔F64)
    0x311,        # F2I.F64 (F64→int32)
    0x312,        # I2F.F64 (int32→F64)
    # BFE helpers
    0x81a,        # BFE_SEXT (bfe.s32 sign-extend step)
}
# Note: IADD.64-UR (0xc35) uses wdep=0x3f (no tracking) + stall=1.
# The 1-cycle stall ensures the result is ready for the subsequent LDG/STG.
_OPCODES_IADD64_UR = {0xc35}
_OPCODES_SMEM_SETUP = {0x9c3, 0x882, 0x291}  # S2UR, UMOV, ULEA


def _get_opcode(raw: bytes) -> int:
    return struct.unpack_from('<Q', raw, 0)[0] & 0xFFF


def _get_dest_reg(raw: bytes) -> int:
    """Get the destination register index, or -1 if none."""
    opcode = _get_opcode(raw)
    # LDCU/S2UR write UR registers, not GPR
    if opcode in (_OPCODES_CTRL | _OPCODES_STG | _OPCODES_STS | _OPCODES_BAR | {0x7ac, 0x9c3}):
        return -1  # no GPR dest
    return raw[2]


def _get_src_regs(raw: bytes) -> set[int]:
    """Get source register indices this instruction reads from GPRs."""
    opcode = _get_opcode(raw)
    regs = set()

    if opcode in _OPCODES_LDG:
        # LDG: src_addr at b3
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
    elif opcode in _OPCODES_ATOMG:
        # ATOMG.CAS: addr(b3, 64-bit pair), compare(b4), new_val(b8)
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
        if raw[4] < 255: regs.add(raw[4])
        if raw[8] < 255: regs.add(raw[8])
    elif opcode in _OPCODES_DFPU:
        # DADD/DMUL: src0(b3, pair), src1(b4, pair); DFMA also src2(b8, pair)
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
        if raw[4] < 255: regs |= {raw[4], raw[4]+1}
        if opcode == 0xc2b and raw[8] < 255: regs |= {raw[8], raw[8]+1}  # DFMA src2
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
        elif opcode in (0x819,):  # SHF (const): src0=b3, K=b4(imm), src1=b8
            if raw[8] < 255: regs.add(raw[8])
        elif opcode in (0x299,):  # SHF.VAR: src0=b3, k_reg=b4(reg), src1=b8
            if raw[4] < 255: regs.add(raw[4])   # shift-amount register
            if raw[8] < 255: regs.add(raw[8])
        elif opcode in (0x23c, 0x237):  # HMMA/IMMA: a=b3(4 regs), b=b4(2), c=b8(4)
            for r in range(4): regs.add(raw[3]+r)
            for r in range(2): regs.add(raw[4]+r)
            for r in range(4): regs.add(raw[8]+r)
        elif opcode in (0x221, 0x223):  # FADD/FFMA: src0=b3, src1=b4, src2=b8
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255 and opcode == 0x223: regs.add(raw[8])
        elif opcode in (0x825, 0x225):  # IMAD.WIDE (R-imm 0x825, R-R 0x225): src2 is 64-bit pair
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255: regs |= {raw[8], raw[8]+1}
        elif opcode in (0x824, 0x224, 0x2a4):  # IMAD non-wide variants: src0=b3, src1=b4, src2=b8
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255: regs.add(raw[8])
        elif opcode == 0xc24:  # IMAD R-UR: src0=b3 (GPR), src1=b4 (UR, not GPR), src2=b8 (GPR)
            if raw[8] < 255: regs.add(raw[8])
        elif opcode == 0x20c:  # ISETP R-R: src0=b3, src1=b4
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
    return regs


def _get_dest_regs(raw: bytes) -> set[int]:
    """Get destination register indices this instruction writes."""
    opcode = _get_opcode(raw)
    if opcode in (0x7ac, 0x9c3):  # LDCU, S2UR: write UR bank, not GPR
        return set()
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
    elif opcode in _OPCODES_ATOMG:
        # ATOMG.CAS: writes single dest (b2) — the old value read from memory
        if dest < 255: regs.add(dest)
    elif opcode in _OPCODES_DFPU:
        # DADD/DMUL/DFMA: writes 64-bit dest pair (b2, b2+1)
        if dest < 255: regs |= {dest, dest+1}
    elif opcode in _OPCODES_LDC:
        if dest < 255:
            regs.add(dest)
            if raw[9] == 0x0a:  # 64-bit
                regs.add(dest+1)
    elif opcode in _OPCODES_LDS:
        if dest < 255: regs.add(dest)
    elif opcode in (0x235, 0xc35):  # IADD.64 / IADD.64-UR: writes GPR pair
        if dest < 255: regs |= {dest, dest+1}
    elif opcode in (0x23c, 0x237):  # HMMA/IMMA: writes 4 regs
        if dest < 255: regs |= {dest, dest+1, dest+2, dest+3}
    elif opcode in (0x825, 0x225):  # IMAD.WIDE: writes dest pair
        if dest < 255: regs |= {dest, dest+1}
    elif opcode == 0x310:  # F2F: F2F.F64.F32 (b9=0x18) writes pair; F2F.F32.F64 writes single
        if dest < 255:
            regs.add(dest)
            if raw[9] == 0x18:  # F2F.F64.F32: dest is f64 pair
                regs.add(dest + 1)
    elif opcode == 0x312:  # I2F.F64: always writes dest pair
        if dest < 255: regs |= {dest, dest + 1}
    elif opcode in _OPCODES_ALU:
        if dest < 255: regs.add(dest)
    return regs


_ldcu_slot_counter = [0]  # mutable counter for rotating LDCU wdep slots
_ldc_slot_counter = [0]   # mutable counter for rotating LDC wdep slots

def _wdep_for_opcode(opcode: int, raw: bytes = None) -> int:
    """Assign the scoreboard write-dependency slot for an opcode."""
    if opcode == 0x7ac:  # LDCU: use 0x35 for .64 (descriptor), rotate 0x31/0x33 for .32
        if raw is not None and raw[9] == 0x0a:  # LDCU.64 (descriptor load)
            return 0x35  # LDG slot — so consumer LDG gets rbar=0x09
        slots = [0x31, 0x33]
        slot = slots[_ldcu_slot_counter[0] % len(slots)]
        _ldcu_slot_counter[0] += 1
        return slot
    if opcode == 0x918:  # NOP: even wdep (misc=0 paired with 0x3e is safe)
        return 0x3e
    if opcode in _OPCODES_LDC:
        return 0x31
    if opcode in _OPCODES_LDS | _OPCODES_DFPU:
        return 0x33
    if opcode in _OPCODES_LDG | _OPCODES_ATOMG:
        return 0x35
    if opcode in _OPCODES_IADD64_UR:
        return 0x3e  # ALU slot — consumer LDG/STG gets rbar via pending_writes
    if opcode in _OPCODES_ALU | _OPCODES_SMEM_SETUP:
        return 0x3e
    # No write tracking for control flow (EXIT/BRA), stores, barriers
    return 0x3f


# Opcode-specific misc nibble (hardware-verified on RTX 5090, 2026-03-25).
# misc is NOT a counter — each opcode has a fixed value required by hardware.
_OPCODE_MISC: dict[int, int] = {
    0x918: 0,   # NOP: misc=0
    0x947: 0,   # BRA: misc=0
    0x94d: 5,   # EXIT: misc=5
    0x981: 6,   # LDG.E: misc=6
    0x3a9: 4,   # ATOMG.CAS: misc=4 (from RTX 5090 probe 2026-03-27)
    0xe29: 2,   # DADD: misc=2 (from RTX 5090 probe 2026-03-27)
    0xc28: 2,   # DMUL: misc=2
    0xc2b: 2,   # DFMA: misc=2
    0xc35: 5,   # IADD.64-UR: misc=5 (wide ALU result)
    0xc0c: 0,   # ISETP R-UR: misc=0 (SM_120: misc 1-12 → wrong predicate)
    0x20c: 0,   # ISETP R-R: misc=0 (same SM_120 predicate correctness requirement)
    0x986: 1,   # STG.E: misc=1 (from ptxas ground truth)
    0x988: 4,   # STS.E: misc=4
}

# All opcodes recognised by assign_ctrl.  Unknown opcodes raise ValueError.
_ALL_KNOWN_OPCODES: frozenset = frozenset(
    _OPCODES_LDG | _OPCODES_LDC | _OPCODES_LDS |
    _OPCODES_STG | _OPCODES_STS | _OPCODES_BAR |
    _OPCODES_CTRL | _OPCODES_ALU | _OPCODES_IADD64_UR |
    _OPCODES_SMEM_SETUP | _OPCODES_ATOMG | _OPCODES_DFPU
)


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
    # Track predicate writes from ISETP/SETP instructions
    pending_pred_writes: dict[int, tuple[int, int]] = {}  # pred_reg → (slot_index, wdep)

    # rbar encoding: maps wdep_slot → rbar bit pattern
    _WDEP_TO_RBAR = {
        0x31: 0x03,   # LDC/LDCU slot → rbar=0x03
        0x33: 0x05,   # LDS/LDCU.32 slot → rbar=0x05
        0x35: 0x09,   # LDG first slot → rbar=0x09
        0x37: 0x09,   # LDG second slot → rbar=0x09 (same as first — ptxas verified)
        0x3e: 0x03,   # ALU → rbar=0x03
    }

    misc_counter = 0
    ldg_count = 0
    _ldcu_slot_counter[0] = 0  # reset per kernel
    _ldc_slot_counter[0] = 0   # reset per kernel
    result = []

    for i, si in enumerate(instrs):
        opcode = _get_opcode(si.raw)
        if opcode not in _ALL_KNOWN_OPCODES:
            raise ValueError(f"assign_ctrl: unrecognized opcode 0x{opcode:03x} at instruction {i}")

        # Determine wdep for this instruction
        wdep = _wdep_for_opcode(opcode, si.raw)

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

        # STG: ptxas uses rbar=1 for STG that stores ALU results. Override
        # the general rbar — STG doesn't need to wait for ALU address registers
        # (they're available within 1 cycle). Only wait if data came from LDG.
        if opcode in _OPCODES_STG:
            data_reg = si.raw[4]  # b4 = data register for STG
            print(f"[STG DEBUG] data_reg=R{data_reg} pending={pending_writes.get(data_reg)} current_rbar={rbar}")
            if data_reg in pending_writes:
                _, pw = pending_writes[data_reg]
                if pw in (0x35, 0x37):  # LDG result slots
                    rbar = 0x09  # wait for LDG data
                else:
                    rbar = 0x01
            else:
                rbar = 0x01

        # ATOMG needs rbar=0x03 (memory ordering, same as STG)
        if opcode in _OPCODES_ATOMG:
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
        # STG UR descriptor: ptxas uses rbar=1 for STG, relying on instruction
        # scheduling to ensure the descriptor is available. Don't override rbar
        # for the UR descriptor — it's guaranteed ready by the time STG executes.
        # (LDG DOES need rbar for the descriptor — see above.)

        # Check predicate-register hazards: any instruction guarded by @Px must
        # wait for the instruction that wrote Px to complete.
        guard = (si.raw[1] >> 4) & 0xF
        if guard != 0x7:  # 0x7 = PT (unconditional)
            if guard in pending_pred_writes:
                _, pw = pending_pred_writes[guard]
                candidate = _WDEP_TO_RBAR.get(pw, 0x01)
                rbar = max(rbar, candidate)

        # BAR.SYNC and EXIT get special ctrl
        if opcode in _OPCODES_BAR:
            wdep = 0x3f
            rbar = 0x01
        if opcode == 0x94d:  # EXIT — SM_120 auto-tracks predicate hazards
            # Always use rbar=0x01 for EXIT (both conditional and unconditional).
            # SM_120 hardware enforces predicate read-after-write automatically;
            # ptxas uses rbar=1 for @Px EXIT. Using rbar=3 waits for the wrong
            # barrier slot and may execute @P0 EXIT before P0 is ready.
            rbar = 0x01
            wdep = 0x3f

        # Build ctrl — bits[22:17] = OPEX (instruction extension / hardware modifier).
        # These are NOT stall counters on SM_120. Each opcode has a fixed OPEX value
        # determined by reverse-engineering ptxas output. Wrong OPEX → hardware
        # misinterprets the instruction (e.g., LDG without OPEX=15 crashes).
        # SM_120: OPEX bits (ctrl[22:17]) are ALWAYS 0 for all opcodes.
        # Verified by extracting ctrl from multiple ptxas SM_120 cubins.
        # Non-zero OPEX corrupts the instruction encoding.
        stall = 0
        if opcode == 0x94d:  # EXIT: if predicated (@Px EXIT)
            guard = (si.raw[1] >> 4) & 0xF
            if guard != 0x7:  # 0x7 = PT (unconditional); any other guard = @Px
                # Predicated EXIT: reset LDCU slot counter so post-branch LDCU
                # instructions start from slot 0 again, matching ptxas behavior.
                _ldcu_slot_counter[0] = 0
        # BRA (opcode 0x947) with a non-PT guard also resets the LDCU counter.
        if opcode == 0x947:
            guard = (si.raw[1] >> 4) & 0xF
            if guard != 0x7:
                _ldcu_slot_counter[0] = 0
        # Misc nibble: opcode-specific where hardware requires it, counter elsewhere.
        if opcode == 0x7ac and si.raw[9] == 0x0a:
            misc = 7   # LDCU.64: misc=7 (CRITICAL: misc=1 → ILLEGAL_ADDRESS)
        elif opcode in _OPCODE_MISC:
            misc = _OPCODE_MISC[opcode]
        else:
            misc = misc_counter & 0xF
        # Hardware rule: odd wdep requires misc != 0 (misc=0 → ILLEGAL_INSTRUCTION)
        if (wdep & 1) and misc == 0:
            misc = 1
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
        # ALU writes (wdep=0x3e) DO need to be tracked for GPR-gap enforcement.
        # The min_gpr_gap ensures ≥1 instruction between ALU write and consumer.
        # Do NOT clear pending_writes here — consumers need the dependency info.
        # Track predicate writes from ISETP: pred dest at raw[2], wdep tells consumers when ready
        if opcode in (0xc0c, 0x20c):  # ISETP R-UR, ISETP R-R
            pred_dest = si.raw[2]  # destination predicate index (0..6)
            pending_pred_writes[pred_dest] = (i, wdep)
        # Note: we do NOT clear pending_writes for consumed SOURCE registers.
        # Multiple instructions may read from the same LDG output, and each
        # needs the rbar wait independently.

        patched = _patch_ctrl(si.raw, ctrl)
        result.append(SassInstr(patched, si.comment))

    return result
