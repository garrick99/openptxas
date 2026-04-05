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


# ---------------------------------------------------------------------------
# New encoder tests (2026-04-04 batch: rare opcodes)
# ---------------------------------------------------------------------------

from sass.encoding.sm_120_opcodes import (
    encode_errbar, encode_cgaerrbar, encode_membar_sys,
    encode_pmtrig, encode_call_rel, encode_ret_rel,
    encode_bra_u, encode_umov_imm, encode_uiadd3_imm,
    encode_uisetp, encode_uisetp_imm, encode_usel_imm,
    encode_b2r_result, encode_bar_red_or, encode_sel_imm,
    encode_ufsetp_imm, encode_ufmul_imm,
    encode_redux_min_u32, encode_redux_max_u32,
    encode_idp4a_ur,
    encode_ucgabar_arv, encode_ucgabar_wait,
    encode_ulea, encode_membar_all_gpu, encode_umov_rr,
    encode_match_any, encode_nanosleep,
    encode_vote_all, encode_vote_any, encode_flo_sh,
    UISETP_NE,
)


def test_errbar():
    """ERRBAR ground truth from ptxas membar kernel."""
    _chk("ERRBAR", encode_errbar(ctrl=0x7f6),
         "ab790000000000000000000000ec0f00")


def test_cgaerrbar():
    """CGAERRBAR ground truth from ptxas membar kernel."""
    _chk("CGAERRBAR", encode_cgaerrbar(ctrl=0x7f6),
         "ab750000000000000000000000ec0f00")


def test_membar_sys():
    """MEMBAR.SC.SYS ground truth from ptxas membar kernel."""
    _chk("MEMBAR.SC.SYS", encode_membar_sys(ctrl=0x7f6),
         "92790000000000000040000000ec0f00")


def test_pmtrig():
    """PMTRIG 0x1 ground truth from ptxas pmevent kernel."""
    _chk("PMTRIG 0x1", encode_pmtrig(event=1, ctrl=0x7f1),
         "01780000010000000000000000e20f00")


def test_call_rel():
    """CALL.REL.NOINC ground truth from ptxas call kernel."""
    _chk("CALL.REL 0x30", encode_call_rel(pc_offset_bytes=0x30, ctrl=0xff5),
         "44790c00000000000000c00300ea1f00")


def test_ret_rel():
    """RET.REL.NODEC ground truth from ptxas call kernel."""
    _chk("RET.REL R2", encode_ret_rel(ret_addr_reg=2, ctrl=0x7f6),
         "5079d002fcffffffffffc30300ec0f00")


def test_umov_imm():
    """UMOV UR6, 0x10002 ground truth from ptxas packed kernel."""
    _chk("UMOV UR6,0x10002", encode_umov_imm(dest_ur=6, imm32=0x10002, ctrl=0x7f1),
         "8278060002000100000f000000e20f00")


def test_uiadd3_imm():
    """UIADD3 UR4, UPT, UPT, UR4, 0x2a, URZ ground truth."""
    _chk("UIADD3 UR4,UR4,0x2a,URZ",
         encode_uiadd3_imm(dest_ur=4, src0_ur=4, imm32=0x2a, src2_ur=0xFF, ctrl=0xfe6),
         "907804042a000000ffe0ff0f00cc1f00")


def test_uisetp_ne_rr():
    """UISETP.NE.U32.AND UP0, UPT, UR6, URZ, UPT ground truth."""
    _chk("UISETP.NE RR", encode_uisetp(0, 6, 0xFF, UISETP_NE, ctrl=0x711),
         "8c720006ff0000007050f00300220e00")


def test_uisetp_ne_imm():
    """UISETP.NE.U32.AND UP0, UPT, UR6, 0x1, UPT ground truth."""
    _chk("UISETP.NE imm", encode_uisetp_imm(0, 6, 1, UISETP_NE, ctrl=0x711),
         "8c780006010000007050f00300220e00")


def test_usel_imm():
    """USEL UR6, URZ, 0x190, UP0 ground truth."""
    _chk("USEL UR6,URZ,0x190,UP0",
         encode_usel_imm(dest_ur=6, src0_ur=0xFF, imm32=0x190, upred=0, ctrl=0x7e4),
         "877806ff900100000000000000c80f00")


def test_b2r_result():
    """B2R.RESULT RZ, P0 ground truth."""
    _chk("B2R.RESULT RZ,P0", encode_b2r_result(pred_dest=0, ctrl=0x712),
         "1c73ff00000000000040000000240e00")


def test_bar_red_or():
    """BAR.RED.OR.DEFER_BLOCKING 0x0, !P0 ground truth."""
    _chk("BAR.RED.OR 0,!P0",
         encode_bar_red_or(0, pred=0, pred_neg=True, ctrl=0x7f6),
         "1d7b0000000000000048010400ec0f00")


def test_sel_imm():
    """SEL R5, RZ, 0x1, !P0 ground truth."""
    _chk("SEL R5,RZ,0x1,!P0",
         encode_sel_imm(dest=5, src0=0xFF, imm32=1, pred=0, pred_neg=True, ctrl=0xfe5),
         "077805ff010000000000000400ca1f00")


# ---------------------------------------------------------------------------
# Cluster operation tests (2026-04-04)
# ---------------------------------------------------------------------------

def test_ucgabar_arv():
    """UCGABAR_ARV ground truth from ptxas cluster kernel."""
    _chk("UCGABAR_ARV", encode_ucgabar_arv(ctrl=0x7f1),
         "c7790000000000000000000800e20f00")

def test_ucgabar_wait():
    """UCGABAR_WAIT ground truth from ptxas cluster kernel."""
    _chk("UCGABAR_WAIT", encode_ucgabar_wait(ctrl=0x7f1),
         "c77d0000000000000000000800e20f00")

def test_membar_all_gpu():
    """MEMBAR.ALL.GPU ground truth from ptxas cluster kernel."""
    _chk("MEMBAR.ALL.GPU", encode_membar_all_gpu(ctrl=0x7f6),
         "927900000000000000a0000000ec0f00")

def test_umov_rr():
    """UMOV UR5, URZ (register-to-register) ground truth."""
    _chk("UMOV UR5,URZ", encode_umov_rr(dest_ur=5, src_ur=0xFF, ctrl=0x7f1),
         "827c0500ff0000000000000800e20f00")

def test_ulea():
    """ULEA UR4, UR5, UR4, 0x18 ground truth."""
    _chk("ULEA UR4,UR5,UR4,0x18",
         encode_ulea(dest_ur=4, base_ur=5, index_ur=4, scale=0x18, acc_ur=0xFF, ctrl=0xff1),
         "9172040504000000ffc08e0f00e21f00")


def test_match_any():
    """MATCH.ANY R0, R4 ground truth."""
    got = encode_match_any(dest=0, src=4, ctrl=0xf18)
    lo = int.from_bytes(got[0:8], 'little')
    hi = int.from_bytes(got[8:16], 'little')
    assert lo == 0x00000000040073a1, f"lo={lo:#018x}"
    assert hi == 0x001e3000000e8000, f"hi={hi:#018x}"


def test_nanosleep():
    """NANOSLEEP 0x64 ground truth."""
    got = encode_nanosleep(duration_ns=100, ctrl=0x7f5)
    lo = int.from_bytes(got[0:8], 'little')
    hi = int.from_bytes(got[8:16], 'little')
    assert lo == 0x000000640000795d, f"lo={lo:#018x}"
    assert hi == 0x000fea0003800000, f"hi={hi:#018x}"


def test_flo_sh():
    """FLO.U32.SH R2, R6 ground truth."""
    got = encode_flo_sh(dest=2, src=6, ctrl=0x712)
    lo = int.from_bytes(got[0:8], 'little')
    hi = int.from_bytes(got[8:16], 'little')
    assert lo == 0x0000000600027300, f"lo={lo:#018x}"
    assert hi == 0x000e2400000e0400, f"hi={hi:#018x}"
