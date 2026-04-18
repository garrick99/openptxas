"""PTXAS-R39 regression — extend R38 post-EXIT S2R gap rule to UIADD (0x835)
and LOP3.LUT (0x812) consumer classes.

R39 probe proof (with R38-only fix active):

    mov.u32 %r2, %ctaid.x;
    add.u32 %r3, %r2, 42;       → SASS: S2R Rx; UIADD Ry, Rx, 42
                                  (opc 0x835 reads S2R dest at b3)
                                  FAILS without gap — stored value is
                                  garbage + 42 instead of ctaid + 42.

    mov.u32 %r2, %ctaid.x;
    or.b32  %r3, %r2, 99;       → SASS: S2R Rx; LOP3.LUT Ry, Rx, 99, RZ
                                  (opc 0x812 reads S2R dest at b3)
                                  FAILS without gap — same garbage-read
                                  pattern.

Both share the encoding "b3 = src0 GPR reading the S2R dest
immediately after a predicated EXIT", identical to the R38-proven
IMAD.SHL.U32 (0x824) hazard.  R39 extends the R38 rule to cover all
three opcodes {0x824, 0x835, 0x812}.

Non-SHL multiply (mul.lo.u32) lowers via S2UR → IMAD R-UR and does NOT
hit the hazard (UR dest, not GPR), so R39 does not need to cover it.
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.parser import parse
from sass.pipeline import compile_function


_OPC_S2R       = 0x919
_OPC_NOP       = 0x918
_OPC_EXIT      = 0x94d
_OPC_IMAD_SHL  = 0x824
_OPC_UIADD     = 0x835
_OPC_LOP3_LUT  = 0x812
_SR_CTAID_X    = 0x25


_PTX_UIADD = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k_uiadd(.param .u64 in, .param .u64 out) {
    .reg .b32 %r<5>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;
    ld.param.u64 %rd0, [in];
    ld.param.u64 %rd1, [out];
    ld.global.u32 %r0, [%rd0];
    mov.u32 %r1, %tid.x;
    setp.eq.u32 %p0, %r1, 0;
    @!%p0 ret;
    mov.u32 %r2, %ctaid.x;
    add.u32 %r3, %r2, 42;
    add.u32 %r4, %r0, %r3;
    st.global.u32 [%rd1], %r4;
    ret;
}
"""


_PTX_LOP3 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k_lop3(.param .u64 in, .param .u64 out) {
    .reg .b32 %r<5>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;
    ld.param.u64 %rd0, [in];
    ld.param.u64 %rd1, [out];
    ld.global.u32 %r0, [%rd0];
    mov.u32 %r1, %tid.x;
    setp.eq.u32 %p0, %r1, 0;
    @!%p0 ret;
    mov.u32 %r2, %ctaid.x;
    or.b32 %r3, %r2, 99;
    add.u32 %r4, %r0, %r3;
    st.global.u32 [%rd1], %r4;
    ret;
}
"""


def _text(cubin: bytes, kernel: str) -> bytes:
    e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]

    def sh(i):
        return struct.unpack_from('<IIQQQQIIQQ', cubin, e_shoff + i * 64)

    _, _, _, _, so, ss, *_ = sh(e_shstrndx)
    shs = cubin[so:so + ss]
    target = f'.text.{kernel}'.encode()
    for i in range(e_shnum):
        nm, ty, _, _, off, sz, *_ = sh(i)
        end = shs.index(b'\x00', nm)
        if shs[nm:end] == target and ty == 1:
            return cubin[off:off + sz]
    raise AssertionError(f'no .text.{kernel}')


def _find_post_exit_s2r_ctaid(text: bytes) -> int | None:
    seen_pexit = False
    for a in range(0, len(text), 16):
        raw = text[a:a + 16]
        opc = (raw[0] | (raw[1] << 8)) & 0xFFF
        guard = (raw[1] >> 4) & 0xF
        if opc == _OPC_EXIT and guard != 0x7:
            seen_pexit = True
            continue
        if seen_pexit and opc == _OPC_S2R and raw[9] == _SR_CTAID_X:
            return a
    return None


# ---------------------------------------------------------------------------
# Test 1 — R39 fires for UIADD consumer: post-EXIT S2R CTAID is NOT
#          immediately followed by UIADD (0x835) reading the S2R dest.
# ---------------------------------------------------------------------------

def test_r39_gap_after_s2r_ctaid_uiadd():
    cubin = compile_function(parse(_PTX_UIADD).functions[0],
                             verbose=False, sm_version=120)
    text = _text(cubin, 'k_uiadd')
    s2r_pos = _find_post_exit_s2r_ctaid(text)
    assert s2r_pos is not None, 'k_uiadd must contain a post-EXIT S2R CTAID_X'
    s2r_dest = text[s2r_pos + 2]
    nxt = text[s2r_pos + 16:s2r_pos + 32]
    nxt_opc = (nxt[0] | (nxt[1] << 8)) & 0xFFF

    assert not (nxt_opc == _OPC_UIADD and nxt[3] == s2r_dest), (
        f"R39 violation: UIADD (0x835) immediately follows post-EXIT "
        f"S2R CTAID reading its dest at b3={nxt[3]}, s2r_dest={s2r_dest}. "
        f"R39 must insert a NOP between them — HW does not honor the "
        f"scoreboard slot-0x31 dependency for this post-EXIT shape and "
        f"UIADD reads stale data.")


# ---------------------------------------------------------------------------
# Test 2 — R39 fires for LOP3.LUT consumer: post-EXIT S2R CTAID is NOT
#          immediately followed by LOP3.LUT (0x812) reading the S2R dest.
# ---------------------------------------------------------------------------

def test_r39_gap_after_s2r_ctaid_lop3():
    cubin = compile_function(parse(_PTX_LOP3).functions[0],
                             verbose=False, sm_version=120)
    text = _text(cubin, 'k_lop3')
    s2r_pos = _find_post_exit_s2r_ctaid(text)
    assert s2r_pos is not None, 'k_lop3 must contain a post-EXIT S2R CTAID_X'
    s2r_dest = text[s2r_pos + 2]
    nxt = text[s2r_pos + 16:s2r_pos + 32]
    nxt_opc = (nxt[0] | (nxt[1] << 8)) & 0xFFF

    assert not (nxt_opc == _OPC_LOP3_LUT and nxt[3] == s2r_dest), (
        f"R39 violation: LOP3.LUT (0x812) immediately follows post-EXIT "
        f"S2R CTAID reading its dest at b3={nxt[3]}, s2r_dest={s2r_dest}. "
        f"R39 must insert a NOP between them — HW does not honor the "
        f"scoreboard slot-0x31 dependency for this post-EXIT shape and "
        f"LOP3.LUT reads stale data.")


# ---------------------------------------------------------------------------
# Test 3 — R39 narrow-scope: only fires for opcodes {0x824, 0x835, 0x812}.
#          The inserted gap instruction is a NOP (not a BSYNC / MEMBAR /
#          SYNC primitive), because probe proved a 1-cycle delay is
#          sufficient — a full sync primitive would be over-engineered.
# ---------------------------------------------------------------------------

def test_r39_inserted_gap_is_a_plain_nop():
    cubin = compile_function(parse(_PTX_UIADD).functions[0],
                             verbose=False, sm_version=120)
    text = _text(cubin, 'k_uiadd')
    s2r_pos = _find_post_exit_s2r_ctaid(text)
    assert s2r_pos is not None
    nxt = text[s2r_pos + 16:s2r_pos + 32]
    nxt_opc = (nxt[0] | (nxt[1] << 8)) & 0xFFF
    assert nxt_opc == _OPC_NOP, (
        f"R39 inserted instruction must be a plain NOP (opc 0x918), "
        f"not a sync primitive.  Got opc=0x{nxt_opc:03x}.  Probe proved "
        f"BSYNC / MEMBAR / WARPSYNC / NOP all fix the hazard at this "
        f"position, and NOP is the cheapest option (zero warp-sync cost).")
