"""
tests/test_scoreboard_regression.py — Regression tests for SM_120 scoreboard rules.

Hardware-verified constraints (GPU bisect on RTX 5090, 2026-03-21 / 2026-03-25):

  Rule 1: LDCU always uses wdep=0x31 (never rotate into LDG slots 0x35/0x37)
  Rule 2: LDCU/S2UR write UR registers — must NOT appear in GPR pending_writes
  Rule 3: misc field is opcode-specific (see _OPCODE_META), NOT a flat counter:
            LDCU.64 → misc=7   (CRITICAL: misc=1 + IADD64-UR misc=5 → ILLEGAL_ADDRESS)
            LDG.E.64 → misc=6
            IADD.64-UR → misc=5
            NOP / BRA → misc=0  (wdep=0x3e)
            EXIT → misc=5
            General ALU → misc=1
  Rule 4: IADD.64-UR uses wdep=0x3e with 64-bit dest pair {R, R+1} tracked

Run: python -m pytest tests/test_scoreboard_regression.py -v
"""
import struct
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass.encoding.sm_120_opcodes import (
    encode_nop, encode_exit, encode_s2r, encode_ldc, encode_ldcu_64,
    encode_ldcu_32, encode_iadd64_ur, encode_ldg_e_64, encode_stg_e_64,
    encode_iadd3, encode_iadd64,
    SR_TID_X, SR_CTAID_X, RZ,
)
from sass.isel import SassInstr
from sass.scoreboard import assign_ctrl, _get_dest_regs, _get_dest_reg, _OPCODE_META


def _wdep_for_opcode(opcode: int) -> int:
    """Look up wdep from the declarative opcode table."""
    return _OPCODE_META[opcode].wdep


def decode_ctrl(raw: bytes) -> dict:
    raw24 = (raw[15] << 16) | (raw[14] << 8) | raw[13]
    ctrl = raw24 >> 1
    return {
        'stall': (ctrl >> 17) & 0x3f,
        'rbar': (ctrl >> 10) & 0x1f,
        'wdep': (ctrl >> 4) & 0x3f,
        'misc': ctrl & 0xf,
    }


# -------------------------------------------------------------------------
# Rule 1: LDCU always wdep=0x31
# -------------------------------------------------------------------------

class TestLdcuSlotAssignment:
    def test_ldcu_never_uses_ldg_slot(self):
        """Multiple LDCUs must never get wdep=0x35 or 0x37 (those are LDG-only slots)."""
        body = [
            SassInstr(encode_s2r(0, SR_TID_X), 'S2R'),
            SassInstr(encode_ldcu_64(4, 0, 0x358), 'LDCU UR4'),
            SassInstr(encode_ldcu_64(6, 0, 0x380), 'LDCU UR6'),
            SassInstr(encode_ldcu_64(8, 0, 0x388), 'LDCU UR8'),
            SassInstr(encode_ldcu_64(10, 0, 0x390), 'LDCU UR10'),
            SassInstr(encode_exit(), 'EXIT'),
        ]
        result = assign_ctrl(body)
        for si in result:
            opcode = struct.unpack_from('<Q', si.raw, 0)[0] & 0xFFF
            if opcode == 0x7ac:  # LDCU
                c = decode_ctrl(si.raw)
                assert c['wdep'] in (0x31, 0x33), \
                    f"LDCU {si.comment} got wdep=0x{c['wdep']:02x}, must be 0x31/0x33 (not LDG slots)"

    def test_ldcu_does_not_alias_ldg_slot(self):
        """After an LDG (wdep=0x35), subsequent LDCU must NOT get 0x35."""
        body = [
            SassInstr(encode_s2r(0, SR_TID_X), 'S2R'),
            SassInstr(encode_ldcu_64(4, 0, 0x358), 'LDCU UR4'),
            SassInstr(encode_iadd64_ur(2, RZ, 4), 'IADD64-UR'),
            SassInstr(encode_ldg_e_64(4, 4, 2), 'LDG.E.64'),
            SassInstr(encode_nop(), 'NOP'),
            SassInstr(encode_ldcu_64(6, 0, 0x380), 'LDCU UR6 (after LDG)'),
            SassInstr(encode_exit(), 'EXIT'),
        ]
        result = assign_ctrl(body)
        for si in result:
            if 'after LDG' in si.comment:
                c = decode_ctrl(si.raw)
                assert c['wdep'] not in (0x35, 0x37), \
                    f"LDCU after LDG got wdep=0x{c['wdep']:02x} (LDG slot collision!)"
                assert c['wdep'] in (0x31, 0x33), \
                    f"LDCU got wdep=0x{c['wdep']:02x}, expected 0x31 or 0x33"


# -------------------------------------------------------------------------
# Rule 2: LDCU/S2UR do not enter GPR pending_writes
# -------------------------------------------------------------------------

class TestLdcuGprTracking:
    def test_ldcu_dest_reg_returns_minus1(self):
        """_get_dest_reg for LDCU must return -1 (not a GPR)."""
        raw = encode_ldcu_64(4, 0, 0x358)  # UR4
        assert _get_dest_reg(raw) == -1, \
            f"LDCU _get_dest_reg returned {_get_dest_reg(raw)}, expected -1"

    def test_ldcu_dest_regs_returns_empty(self):
        """_get_dest_regs for LDCU must return empty set (byte 2 is UR, not GPR)."""
        raw = encode_ldcu_64(4, 0, 0x358)
        regs = _get_dest_regs(raw)
        assert regs == set(), f"LDCU _get_dest_regs returned {regs}, expected empty"

    def test_ldcu_32_dest_regs_returns_empty(self):
        """_get_dest_regs for LDCU.32 must return empty set."""
        raw = encode_ldcu_32(7, 0, 0x398)
        regs = _get_dest_regs(raw)
        assert regs == set(), f"LDCU.32 _get_dest_regs returned {regs}, expected empty"

    def test_ldcu_ur4_does_not_poison_gpr4(self):
        """LDCU UR4 must NOT cause GPR R4 to appear as pending in the scoreboard."""
        body = [
            SassInstr(encode_ldcu_64(4, 0, 0x358), 'LDCU UR4'),
            # IADD3 reads R4 — should NOT get rbar from LDCU UR4
            SassInstr(encode_iadd3(5, 4, RZ, RZ), 'IADD3 R5, R4, RZ, RZ'),
            SassInstr(encode_exit(), 'EXIT'),
        ]
        result = assign_ctrl(body)
        iadd_ctrl = decode_ctrl(result[1].raw)
        assert iadd_ctrl['rbar'] == 0x01, \
            f"IADD3 got rbar=0x{iadd_ctrl['rbar']:02x} — LDCU UR4 poisoned GPR R4 tracking"


# -------------------------------------------------------------------------
# Rule 3: misc field per opcode (hardware-verified 2026-03-25)
# -------------------------------------------------------------------------

class TestMiscField:
    def test_nop_misc_zero(self):
        """NOP must have misc=0, wdep=0x3e (ptxas-observed)."""
        result = assign_ctrl([SassInstr(encode_nop(), 'NOP'), SassInstr(encode_exit(), 'EXIT')])
        c = decode_ctrl(result[0].raw)
        assert c['wdep'] == 0x3e, f"NOP wdep=0x{c['wdep']:02x}, expected 0x3e"
        assert c['misc'] == 0, f"NOP misc={c['misc']}, expected 0"

    def test_exit_misc_is_5(self):
        """EXIT must have misc=5, wdep=0x3f (ptxas-observed)."""
        result = assign_ctrl([SassInstr(encode_exit(), 'EXIT')])
        c = decode_ctrl(result[0].raw)
        assert c['misc'] == 5, f"EXIT misc={c['misc']}, expected 5"
        assert c['wdep'] == 0x3f, f"EXIT wdep=0x{c['wdep']:02x}, expected 0x3f"

    def test_alu_even_wdep_misc_nonzero(self):
        """ALU instructions (IADD3, etc.) with wdep=0x3e must have misc != 0
        (misc=0 + even wdep → ILLEGAL_INSTRUCTION on hardware)."""
        body = [
            SassInstr(encode_s2r(0, SR_TID_X), 'S2R'),
            SassInstr(encode_iadd3(2, 0, RZ, RZ), 'IADD3'),
            SassInstr(encode_exit(), 'EXIT'),
        ]
        result = assign_ctrl(body)
        c = decode_ctrl(result[1].raw)
        assert c['wdep'] == 0x3e, f"IADD3 wdep=0x{c['wdep']:02x}, expected 0x3e"
        assert c['misc'] != 0, \
            f"IADD3 misc=0 with even wdep → ILLEGAL_INSTRUCTION on hardware"

    def test_iadd64_ur_misc_is_5(self):
        """IADD.64-UR must have misc=5 (ptxas-observed: wide ALU result)."""
        body = [
            SassInstr(encode_s2r(0, SR_TID_X), 'S2R'),
            SassInstr(encode_ldcu_64(4, 0, 0x358), 'LDCU UR4'),
            SassInstr(encode_iadd64_ur(2, RZ, 4), 'IADD64-UR'),
            SassInstr(encode_exit(), 'EXIT'),
        ]
        result = assign_ctrl(body)
        c = decode_ctrl(result[2].raw)
        assert c['misc'] == 5, f"IADD64-UR misc={c['misc']}, expected 5"

    def test_ldcu_misc_is_7(self):
        """LDCU.64 must have misc=7 (ptxas-observed for uniform register loads).
        CRITICAL: misc=1 + subsequent IADD64-UR misc=5 → CUDA_ERROR_ILLEGAL_ADDRESS.
        Hardware bisect confirmed on RTX 5090 (2026-03-25)."""
        body = [
            SassInstr(encode_s2r(0, SR_TID_X), 'S2R'),
            SassInstr(encode_ldcu_64(4, 0, 0x358), 'LDCU UR4'),
            SassInstr(encode_exit(), 'EXIT'),
        ]
        result = assign_ctrl(body)
        c = decode_ctrl(result[1].raw)
        assert c['wdep'] in (0x31, 0x33), f"LDCU wdep=0x{c['wdep']:02x}"
        assert c['misc'] == 7, f"LDCU misc={c['misc']}, expected 7"

    def test_ldg_misc_is_6(self):
        """LDG.E.64 must have misc=6 (ptxas-observed for global load results)."""
        body = [
            SassInstr(encode_s2r(0, SR_TID_X), 'S2R'),
            SassInstr(encode_ldcu_64(4, 0, 0x358), 'LDCU UR4'),
            SassInstr(encode_ldcu_64(6, 0, 0x388), 'LDCU UR6'),
            SassInstr(encode_iadd64_ur(2, RZ, 6), 'IADD64-UR'),
            SassInstr(encode_ldg_e_64(0, 4, 2), 'LDG.E.64'),
            SassInstr(encode_exit(), 'EXIT'),
        ]
        result = assign_ctrl(body)
        c = decode_ctrl(result[4].raw)
        assert c['wdep'] == 0x35, f"LDG wdep=0x{c['wdep']:02x}"
        assert c['misc'] == 6, f"LDG misc={c['misc']}, expected 6"


# -------------------------------------------------------------------------
# Rule 4: IADD.64-UR uses wdep=0x3e with 64-bit dest pair tracking
# -------------------------------------------------------------------------

class TestIadd64UrTracking:
    def test_iadd64_ur_wdep_is_3e(self):
        """IADD.64-UR must use wdep=0x3e (ALU tracking, not 0x3f)."""
        assert _wdep_for_opcode(0xc35) == 0x3e

    def test_iadd64_ur_dest_regs_are_pair(self):
        """IADD.64-UR R2 must report dest regs {2, 3} (64-bit pair)."""
        raw = encode_iadd64_ur(2, RZ, 6)
        regs = _get_dest_regs(raw)
        assert regs == {2, 3}, f"IADD.64-UR R2 dest_regs={regs}, expected {{2, 3}}"

    def test_iadd64_ur_no_stall(self):
        """IADD.64-UR must not have a static stall — rbar handles the dependency."""
        body = [
            SassInstr(encode_s2r(0, SR_TID_X), 'S2R'),
            SassInstr(encode_ldcu_64(6, 0, 0x388), 'LDCU UR6'),
            SassInstr(encode_iadd64_ur(2, RZ, 6), 'IADD64-UR'),
            SassInstr(encode_exit(), 'EXIT'),
        ]
        result = assign_ctrl(body)
        c = decode_ctrl(result[2].raw)
        assert c['stall'] == 0, f"IADD.64-UR has stall={c['stall']}, should be 0"

    def test_iadd64_ur_consumer_gets_rbar(self):
        """LDG reading IADD.64-UR's output must have rbar > 0x01 (waits for ALU slot)."""
        body = [
            SassInstr(encode_s2r(0, SR_TID_X), 'S2R'),
            SassInstr(encode_ldcu_64(4, 0, 0x358), 'LDCU UR4'),
            SassInstr(encode_ldcu_64(6, 0, 0x388), 'LDCU UR6'),
            SassInstr(encode_iadd64_ur(2, RZ, 6), 'IADD64-UR'),
            SassInstr(encode_ldg_e_64(4, 4, 2), 'LDG.E.64'),
            SassInstr(encode_exit(), 'EXIT'),
        ]
        result = assign_ctrl(body)
        ldg_ctrl = decode_ctrl(result[4].raw)
        assert ldg_ctrl['rbar'] > 0x01, \
            f"LDG has rbar=0x{ldg_ctrl['rbar']:02x}, should wait for IADD64-UR result"


# -------------------------------------------------------------------------
# Integration: full LDG64 copy kernel ctrl correctness
# -------------------------------------------------------------------------

class TestLdg64CopyKernel:
    def test_ldg64_copy_ctrl_values(self):
        """Full 64-bit LDG copy kernel must produce valid ctrl for all instructions."""
        body = [
            SassInstr(encode_s2r(0, SR_TID_X), 'S2R R0'),
            SassInstr(encode_ldcu_64(4, 0, 0x358), 'LDCU.64 UR4'),
            SassInstr(encode_ldcu_64(6, 0, 0x388), 'LDCU.64 UR6 (in_ptr)'),
            SassInstr(encode_iadd64_ur(2, RZ, 6), 'IADD.64-UR R2 (in)'),
            SassInstr(encode_ldg_e_64(4, 4, 2), 'LDG.E.64 R4'),
            SassInstr(encode_nop(), 'NOP'),
            SassInstr(encode_ldcu_64(8, 0, 0x380), 'LDCU.64 UR8 (out_ptr)'),
            SassInstr(encode_iadd64_ur(6, RZ, 8), 'IADD.64-UR R6 (out)'),
            SassInstr(encode_stg_e_64(4, 6, 4), 'STG.E.64'),
            SassInstr(encode_exit(), 'EXIT'),
        ]
        result = assign_ctrl(body)

        # Rule: odd wdep → misc must be non-zero (even misc=0 → ILLEGAL_INSTRUCTION)
        for si in result:
            c = decode_ctrl(si.raw)
            if c['wdep'] & 1:
                assert c['misc'] != 0, \
                    f"{si.comment} (wdep=0x{c['wdep']:02x}, odd): misc=0 → ILLEGAL_INSTRUCTION"

        # LDG: correct slot and barrier
        ldg = decode_ctrl(result[4].raw)
        assert ldg['wdep'] == 0x35, "LDG wdep must be 0x35 (first LDG slot)"
        assert ldg['rbar'] > 0x01, "LDG must have barrier wait for address register"
        assert ldg['misc'] == 6, "LDG misc must be 6"

        # STG: must wait for both IADD result (address) and LDG result (data)
        stg = decode_ctrl(result[8].raw)
        assert stg['rbar'] >= 0x09, "STG must wait for LDG data (rbar >= 0x09)"

        # All LDCUs must use wdep in {0x31, 0x33} (never LDG slots 0x35/0x37)
        for i in [1, 2, 6]:
            c = decode_ctrl(result[i].raw)
            assert c['wdep'] in (0x31, 0x33), \
                f"LDCU at idx {i} ({result[i].comment}): wdep=0x{c['wdep']:02x}, not in LDCU slots"
            assert c['misc'] == 7, \
                f"LDCU at idx {i}: misc={c['misc']}, expected 7"

        # IADD64-UR: misc=5 (wide result)
        for i in [3, 7]:
            c = decode_ctrl(result[i].raw)
            assert c['misc'] == 5, \
                f"IADD64-UR at idx {i}: misc={c['misc']}, expected 5"

        # NOP: misc=0
        nop = decode_ctrl(result[5].raw)
        assert nop['misc'] == 0, f"NOP misc={nop['misc']}, expected 0"


# -------------------------------------------------------------------------
# Invariant: unknown opcode raises immediately
# -------------------------------------------------------------------------

class TestUnknownOpcodeAssertion:
    def test_unknown_opcode_raises(self):
        """assign_ctrl must raise ValueError for unrecognised opcodes, not silently emit 0x3f."""
        import struct
        # Construct a fake instruction with opcode 0xABC (not in table)
        raw = bytearray(16)
        raw[0] = 0xBC
        raw[1] = 0x0A  # little-endian: bytes[0..1] bits [11:0] = 0xABC
        with pytest.raises(ValueError, match="unrecognized opcode"):
            assign_ctrl([SassInstr(bytes(raw), 'fake')])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
