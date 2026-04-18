"""PTXAS-R32' regression — WB-8 LDCU.128 pack guard for post-EXIT
HIGH-half consumers.

R30 / R32 proof: when WB-8 packs two LDCU.64 loads into one LDCU.128,
only the primary UR (``d``) reliably arms the scoreboard slot.  The
packed load's HIGH half (``d+2``, ``d+3``) does NOT reliably stall a
subsequent consumer that sits AFTER a predicated ``@!P0 EXIT`` — the
pack crosses the control-flow boundary without a reliable wait, and
the IADD.64 R-UR consumer reads a stale/garbage value for the HIGH
half.  Downstream STG writes to ``out_ptr + garbage`` and produces
CUDA_ERROR_ILLEGAL_ADDRESS (``k_2p_offset`` repro).

R32' adds a narrow guard to ``_wb8_pack_ldcu_128``: if the HIGH-half
UR (``d+2``) has an ``IADD.64 R-UR`` consumer (opc 0xc35) AFTER the
first predicated EXIT, do NOT pack — leave both LDCU.64s intact so
each arms its own scoreboard slot.  All other WB-8 packings (the
common all-pre-EXIT case, and HIGH-half consumers that stay
pre-EXIT) are unaffected.
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.parser import parse
from sass.pipeline import compile_function


_OPC_LDCU = 0x7ac


# The R30 / R32 proven unsafe class: two u64 params (0x380 / 0x388),
# both redefined in-place across a predicated EXIT (R31 routes them
# through UR via preamble LDCU.64), and a post-EXIT `IADD.64 R-UR`
# consuming the HIGH-half UR of the would-be pack.  Without R32',
# WB-8 packs LDCU.64 UR8 @ 0x380 + LDCU.64 UR10 @ 0x388 into
# LDCU.128 UR8 and crashes at runtime (CUDA_ERROR_ILLEGAL_ADDRESS).
_PTX_GUARDED = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k_r32_guard(.param .u64 in, .param .u64 out) {
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;
    ld.param.u64 %rd0, [in];
    ld.param.u64 %rd1, [out];
    mov.u32 %r0, 777;
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


# ---------------------------------------------------------------------------
# Test 1 — the R32' guard FIRES on the proven unsafe class: the
#          emitted text contains TWO separate LDCU.64 param loads
#          (b9=0x0a) at offsets 0x380 and 0x388, and NO LDCU.128
#          (b9=0x0c) param load.  Each LDCU.64 arms its own slot-0x31
#          entry so the post-EXIT `IADD.64 R-UR` wait resolves.
# ---------------------------------------------------------------------------

def test_r32_guard_prevents_pack_on_post_exit_consumer():
    """The R32' guard's effect is that the HIGH-half partner stays a real
    LDCU.64 (not NOP-absorbed by WB-8).  Rule #29 may still upcast the
    PRIMARY LDCU.64 to LDCU.128 for post-EXIT alignment reasons (that is
    a separate, proven-safe rewrite at raw[5] even + >= 0x70), but the
    partner must remain a real LDCU.64 at the sibling offset so the
    HIGH-half UR is re-armed on its own scoreboard slot.  WB-8 packing
    would have turned the partner into a NOP."""
    cubin = compile_function(parse(_PTX_GUARDED).functions[0],
                             verbose=False, sm_version=120)
    text = _text(cubin, 'k_r32_guard')
    ldcu_param_entries: list[tuple[int, int]] = []  # (offset_bytes, b9)
    for a in range(0, len(text), 16):
        raw = text[a:a + 16]
        opc = (raw[0] | (raw[1] << 8)) & 0xFFF
        if opc != _OPC_LDCU:
            continue
        if raw[5] < 0x70:
            continue  # skip the descriptor LDCU at c[0][0x358]
        ldcu_param_entries.append((raw[5] * 8, raw[9]))

    offsets = sorted({off for off, _ in ldcu_param_entries})
    assert 0x380 in offsets and 0x388 in offsets, (
        f"R32' expected LDCU loads for both params at 0x380 and 0x388; "
        f"got offsets {[hex(o) for o in offsets]}.  Packing would have "
        f"replaced the 0x388 entry with a NOP, breaking this invariant.")
    # Specifically, there must be a REAL LDCU.64 at 0x388 (b9=0x0a).  WB-8
    # packing would have NOP-absorbed it; R32' blocks that.
    assert any(off == 0x388 and b9 == 0x0a for off, b9 in ldcu_param_entries), (
        f"R32' violated: no LDCU.64 (b9=0x0a) at 0x388 in the emitted text; "
        f"entries={[(hex(o), hex(b)) for o, b in ldcu_param_entries]}. "
        f"WB-8 must not pack the 0x380+0x388 pair when the HIGH-half UR "
        f"has a post-EXIT `IADD.64 R-UR` consumer.")


# ---------------------------------------------------------------------------
# Test 2 — end-to-end: the post-EXIT `IADD.64 R-UR` exists (opc 0xc35)
#          and its UR-src (b4) is the HIGH-half of the NOT-packed pair,
#          confirming the guard ran for exactly the right reason.
# ---------------------------------------------------------------------------

def test_r32_guard_high_half_is_post_exit_iadd64_rur_src():
    cubin = compile_function(parse(_PTX_GUARDED).functions[0],
                             verbose=False, sm_version=120)
    text = _text(cubin, 'k_r32_guard')
    seen_exit = False
    post_exit_iadd64_rur_urs: list[int] = []
    for a in range(0, len(text), 16):
        raw = text[a:a + 16]
        opc = (raw[0] | (raw[1] << 8)) & 0xFFF
        guard = (raw[1] >> 4) & 0xF
        if opc == 0x94d and guard != 0x7:
            seen_exit = True
            continue
        if seen_exit and opc == 0xc35:
            post_exit_iadd64_rur_urs.append(raw[4])

    assert seen_exit, "R32' test fixture: kernel must contain a predicated EXIT"
    assert post_exit_iadd64_rur_urs, (
        "R32' test fixture: post-EXIT body must contain at least one "
        "`IADD.64 R-UR` (opc 0xc35) — this is the consumer that motivates "
        "the guard.  If missing, either the pipeline lowered the add.u64 "
        "into a different form, or R31/R32' is not routing the param to UR.")
    # The UR index read by the post-EXIT IADD.64 R-UR must be the HIGH half
    # of the would-be pack (UR10 since `out` param at offset 0x388 lands at
    # UR10:UR11, while `in` at 0x380 goes to UR8:UR9).
    assert any(ur in (8, 10) for ur in post_exit_iadd64_rur_urs), (
        f"R32' expected at least one post-EXIT IADD.64 R-UR reading UR8 or "
        f"UR10 (param UR slots); got b4 values {post_exit_iadd64_rur_urs}.")
