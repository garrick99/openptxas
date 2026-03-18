"""
test_opcodes.py — Round-trip encoding tests for all SM_120 non-SHF opcodes.

Covers: NOP, EXIT, MOV, LDC, LDC.64, S2R, IADD3, IMAD.WIDE,
        LDG.E.64, STG.E.64, ISETP.GE.AND, BRA.

Ground truth: bytes extracted from sm_120_encoding_tables.json (ptxas output
on RTX 5090 / SM_120).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sass.encoding.sm_120_opcodes import (
    encode_nop, encode_exit, encode_mov,
    encode_ldc, encode_ldc_64,
    encode_s2r,
    encode_iadd3, encode_iadd3x,
    encode_iadd64,
    encode_imad_wide,
    encode_ldg_e_64, encode_stg_e_64,
    encode_isetp_ge_and,
    encode_bra,
    roundtrip_verify_opcodes,
    RZ, PT, SR_TID_X, SR_CTAID_X,
)


def _chk(label, got, expected_hex):
    expected = bytes.fromhex(expected_hex)
    assert got == expected, (
        f"\n  {label}\n"
        f"  expected: {expected.hex()}\n"
        f"  got:      {got.hex()}\n"
        f"  diff at:  {[i for i in range(16) if got[i] != expected[i]]}"
    )


def test_nop():
    _chk("NOP ctrl=0x7e0",
         encode_nop(ctrl=0x7e0),
         "18790000000000000000000000c00f00")


def test_exit():
    _chk("EXIT ctrl=0x7f5",
         encode_exit(ctrl=0x7f5),
         "4d790000000000000000800300ea0f00")


def test_mov():
    _chk("MOV R6,R5", encode_mov(6, 5, ctrl=0x7e5),
         "0272060005000000000f000000ca0f00")
    _chk("MOV R3,R2", encode_mov(3, 2, ctrl=0x7e3),
         "0272030002000000000f000000c60f00")


def test_ldc_32():
    _chk("LDC R1,c[0][0x37c]", encode_ldc(1, 0, 0x37c, ctrl=0x7f1),
         "827b01ff00df00000008000000e20f00")
    _chk("LDC R13,c[0][0x360]", encode_ldc(13, 0, 0x360, ctrl=0x712),
         "827b0dff00d800000008000000240e00")
    _chk("LDC R9,c[0][0x360]",  encode_ldc(9,  0, 0x360, ctrl=0x712),
         "827b09ff00d800000008000000240e00")


def test_ldc_64():
    _chk("LDC.64 R2,c[0][0x388]",  encode_ldc_64(2,  0, 0x388, ctrl=0x711),
         "827b02ff00e20000000a000000220e00")
    _chk("LDC.64 R10,c[0][0x380]", encode_ldc_64(10, 0, 0x380, ctrl=0x751),
         "827b0aff00e00000000a000000a20e00")
    _chk("LDC.64 R6,c[0][0x380]",  encode_ldc_64(6,  0, 0x380, ctrl=0x751),
         "827b06ff00e00000000a000000a20e00")


def test_s2r():
    _chk("S2R R0,SR_TID.X", encode_s2r(0, SR_TID_X, ctrl=0x717),
         "197900000000000000210000002e0e00")


def test_iadd3():
    _chk("IADD3 RZ,RZ,R4,RZ", encode_iadd3(RZ, RZ, 4, RZ, ctrl=0x7f1),
         "1072ffff04000000ffe0f10700e20f00")


def test_iadd3x():
    _chk("IADD3.X R7,RZ,RZ,RZ", encode_iadd3x(7, RZ, RZ, RZ, ctrl=0x7f2),
         "107207ffff000000ffe47f0000e40f00")


def test_imad_wide():
    _chk("IMAD.WIDE R2,R13,0x8,R2",   encode_imad_wide(2, 13, 8, 2,  ctrl=0x0fe6),
         "2578020d0800000002028e0700cc1f00")
    _chk("IMAD.WIDE R6,R13,0x8,R10",  encode_imad_wide(6, 13, 8, 10, ctrl=0x27f1),
         "2578060d080000000a028e0700e24f00")
    _chk("IMAD.WIDE R2,R9,0x8,R2",    encode_imad_wide(2, 9,  8, 2,  ctrl=0x0fe6),
         "257802090800000002028e0700cc1f00")
    _chk("IMAD.WIDE R6,R9,0x8,R6",    encode_imad_wide(6, 9,  8, 6,  ctrl=0x27f1),
         "257806090800000006028e0700e24f00")


def test_ldg_e_64():
    _chk("LDG.E.64 R2,desc[UR4][R2]", encode_ldg_e_64(2, 4, 2, ctrl=0x1771),
         "8179020204000000001b1e0c00e22e00")


def test_stg_e_64():
    _chk("STG.E.64 desc[UR4][R6],R4", encode_stg_e_64(4, 6, 4, ctrl=0x7f1),
         "8679000604000000041b100c00e20f00")


def test_isetp_ge_and():
    _chk("ISETP.GE.AND P0,R13,UR5", encode_isetp_ge_and(0, 13, 5, ctrl=0x17ed),
         "0c7c000d050000007062f00b00da2f00")
    _chk("ISETP.GE.AND P0,R9,UR5",  encode_isetp_ge_and(0, 9,  5, ctrl=0x17ed),
         "0c7c0009050000007062f00b00da2f00")


def test_iadd64_sub():
    """IADD.64 with negation = 64-bit subtract. Ground truth from ptxas 13.0."""
    _chk("IADD.64 R6,R2,-R4", encode_iadd64(6, 2, 4, negate_src1=True, ctrl=0x27e6),
         "357206020400008000028e0700cc4f00")


def test_iadd64_add():
    """IADD.64 without negation = 64-bit add."""
    got = encode_iadd64(6, 2, 4, negate_src1=False, ctrl=0x27e6)
    # Same as sub but byte7=0x00 instead of 0x80
    assert got[7] == 0x00
    assert got[:7] == bytes.fromhex('35720602040000')


def test_bra_self_loop():
    """BRA with -16 byte offset = jump to self = the ground truth pattern."""
    _chk("BRA offset=-16", encode_bra(-16, ctrl=0x7e0),
         "4779fc00fcffffffffff830300c00f00")


def test_roundtrip_all():
    """Bulk verify: all 21 ground truth samples via roundtrip_verify_opcodes()."""
    assert roundtrip_verify_opcodes(verbose=False), \
        "roundtrip_verify_opcodes() reported failures"
