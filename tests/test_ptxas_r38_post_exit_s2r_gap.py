"""PTXAS-R38 regression — post-EXIT S2R -> immediate IMAD.SHL.U32 gap.

R38 probe proof: the SASS pattern

    @!P0 EXIT
    S2R Rx, SR_CTAID_X
    IMAD.SHL.U32 Ry, Rx, imm, RZ   ; reads Rx at b3 (src0)

produces CUDA_ERROR_ILLEGAL_ADDRESS on SM_120 despite scoreboard rbar
nominally covering the S2R's slot-0x31 producer.  Kernel `s2_fail` is
the canonical repro.  R38 proves the hazard with cubin-level probes:

    * +0..+4 NOPs inserted before `IADD.64 R-UR` consumer: FAIL
    * NOP inserted BEFORE S2R CTAID: FAIL
    * NOP inserted AFTER IMAD.SHL (before MOV R2, R3): FAIL
    * NOP inserted AFTER MOV R2,R3: FAIL
    * NOP inserted AFTER MOV R3,RZ: FAIL
    * NOP inserted BETWEEN S2R and IMAD.SHL.U32: **PASS** (sync=0, out=777)
    * BSYNC/MEMBAR.CTA/WARPSYNC at the same position: also PASS

Passing kernels like G4 already have a natural gap (the scheduler
places a descriptor `LDCU.64` between `S2R CTAID` and `IMAD.SHL` in G4).
The fix only fires when no natural gap exists.
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.parser import parse
from sass.pipeline import compile_function


_OPC_S2R        = 0x919
_OPC_IMAD_SHL   = 0x824
_OPC_NOP        = 0x918
_OPC_EXIT       = 0x94d
_SR_CTAID_X     = 0x25


_PTX_S2_FAIL = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry s2_fail(.param .u64 in, .param .u64 out) {
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;
    ld.param.u64 %rd0, [in];
    ld.param.u64 %rd1, [out];
    ld.global.u32 %r0, [%rd0];
    mov.u32 %r1, %tid.x;
    setp.eq.u32 %p0, %r1, 0;
    @!%p0 ret;
    mov.u32 %r2, %ctaid.x;
    shl.b32 %r3, %r2, 2;
    cvt.u64.u32 %rd2, %r3;
    add.u64 %rd1, %rd1, %rd2;
    st.global.u32 [%rd1], %r0;
    ret;
}
"""


_PTX_G4_NATURAL_GAP = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry g4(.param .u64 out) {
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;
    ld.param.u64 %rd0, [out];
    mov.u32 %r0, %tid.x;
    setp.eq.u32 %p0, %r0, 0;
    @!%p0 ret;
    mov.u32 %r1, %ctaid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd2, %r2;
    add.u64 %rd1, %rd0, %rd2;
    st.global.u32 [%rd1], 2864434397;
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
# Test 1 — the R38 gap fires: post-EXIT S2R CTAID is NOT immediately
#          followed by IMAD.SHL.U32 reading the S2R's dest at b3.
# ---------------------------------------------------------------------------

def test_r38_gap_after_post_exit_s2r_ctaid():
    cubin = compile_function(parse(_PTX_S2_FAIL).functions[0],
                             verbose=False, sm_version=120)
    text = _text(cubin, 's2_fail')
    s2r_pos = _find_post_exit_s2r_ctaid(text)
    assert s2r_pos is not None, (
        's2_fail fixture: kernel must contain a post-EXIT S2R CTAID_X')
    s2r_dest = text[s2r_pos + 2]
    nxt = text[s2r_pos + 16:s2r_pos + 32]
    nxt_opc = (nxt[0] | (nxt[1] << 8)) & 0xFFF

    assert not (nxt_opc == _OPC_IMAD_SHL and nxt[3] == s2r_dest), (
        f'R38 violation: post-EXIT S2R CTAID at byte {s2r_pos} is '
        f'immediately followed by IMAD.SHL.U32 reading its dest '
        f'(b3={nxt[3]} == s2r_dest={s2r_dest}).  The R38 rule must '
        f'insert a NOP between the two instructions; otherwise HW '
        f'scoreboard does not honor the slot-0x31 dependency and the '
        f'downstream pair-build / IADD.64 R-UR / STG chain crashes '
        f'with CUDA_ERROR_ILLEGAL_ADDRESS.')


# ---------------------------------------------------------------------------
# Test 2 — the R38 gap is narrow: kernels that already have a natural
#          gap (G4's descriptor LDCU between S2R CTAID and IMAD.SHL)
#          are unchanged.  We do NOT add a second NOP on top.  This
#          verifies the rule only fires when the next instruction
#          is IMAD.SHL.U32 reading the S2R dest; G4 has an LDCU.64
#          between them instead.
# ---------------------------------------------------------------------------

def test_r38_narrow_scope_g4_unchanged():
    cubin = compile_function(parse(_PTX_G4_NATURAL_GAP).functions[0],
                             verbose=False, sm_version=120)
    text = _text(cubin, 'g4')
    s2r_pos = _find_post_exit_s2r_ctaid(text)
    assert s2r_pos is not None, 'g4 fixture: kernel must contain post-EXIT S2R CTAID_X'
    s2r_dest = text[s2r_pos + 2]
    nxt = text[s2r_pos + 16:s2r_pos + 32]
    nxt_opc = (nxt[0] | (nxt[1] << 8)) & 0xFFF

    # G4 must NOT have a NOP directly after S2R CTAID — the natural
    # LDCU gap already satisfies the hazard, and R38 should not stack
    # an extra NOP.
    assert nxt_opc != _OPC_NOP, (
        f'R38 scope violation: g4 already has a natural gap '
        f'(opc 0x{nxt_opc:03x} after S2R CTAID), but R38 inserted an '
        f'additional NOP.  The rule must only fire when the immediate '
        f'next instruction is `IMAD.SHL.U32 Ry, Rx, imm` reading the '
        f'S2R dest at b3.  Stacking is harmless but wasteful and '
        f'breaks the narrow-scope invariant.')

    # Sanity: the immediate next instruction is NOT IMAD.SHL.U32 in G4
    # (it's an LDCU or other non-conflicting op), so the rule does not
    # need to fire.
    assert not (nxt_opc == _OPC_IMAD_SHL and nxt[3] == s2r_dest), (
        f'g4 fixture: expected no S2R->IMAD.SHL direct consumer; got '
        f'nxt_opc=0x{nxt_opc:03x} b3={nxt[3]} dest={s2r_dest}.  If the '
        f'scheduler regressed G4 into the unsafe shape, R38 would fire '
        f'and the test would need updating.')
