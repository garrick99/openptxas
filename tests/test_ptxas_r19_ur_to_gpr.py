"""PTXAS-R19 regression — UR→GPR routing for SR-derived values.

Proven defect (FB-1 Phase A): OpenPTXas's single-CTAID fast path routed
`%ctaid.x` into a UR via `S2UR`, assuming all consumers could read the
UR directly (IMAD R-UR, ISETP R-UR).  When a later block contained a
non-fusable 32-bit consumer — e.g. `shl.b32 %addr_off, %ctaid, 2` for a
store address — the allocator's pre-assigned GPR slot for `%ctaid` was
never written, and the consumer read garbage (CUDA_ERROR_ILLEGAL_ADDRESS
on STG).

These tests prove the semantic class, not the exact pilot text:
  1. `%ctaid.x` with a mixed UR-friendly + GPR-only consumer set must be
     routed through S2R (not S2UR) so the GPR slot is valid for the
     late consumer.
  2. A cross-basic-block GPR consumer of a SR-derived value must still
     force S2R routing.
  3. A kernel where `%ctaid.x` is only consumed by IMAD R-UR (pure
     UR-friendly) must retain the S2UR fast path.
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.parser import parse
from sass.pipeline import compile_function


# Opcodes used for direct assertion
_OPC_S2UR  = 0x9c3
_OPC_S2R   = 0x919
_OPC_STG_E = 0x986
_OPC_IMAD_UR_RR = 0xc24   # IMAD R-UR variant


def _text(cubin: bytes, kernel: str) -> bytes:
    e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]

    def sh(i):
        base = e_shoff + i * 64
        return struct.unpack_from('<IIQQQQIIQQ', cubin, base)

    _, _, _, _, shstr_off, shstr_size, *_ = sh(e_shstrndx)
    shstr = cubin[shstr_off:shstr_off + shstr_size]
    target = f'.text.{kernel}'.encode()
    for i in range(e_shnum):
        nm, ty, _, _, off, sz, *_ = sh(i)
        end = shstr.index(b'\x00', nm)
        if shstr[nm:end] == target and ty == 1:
            return cubin[off:off + sz]
    raise AssertionError(f'no .text.{kernel} in cubin')


def _has_opcode(text: bytes, opc: int) -> bool:
    for addr in range(0, len(text), 16):
        if ((text[addr] | (text[addr + 1] << 8)) & 0xFFF) == opc:
            return True
    return False


def _compile(ptx_src: str) -> bytes:
    mod = parse(ptx_src)
    return compile_function(mod.functions[0], verbose=False, sm_version=120)


# ---------------------------------------------------------------------------
# Test 1 — mixed UR-friendly + GPR-only consumer forces S2R
# ---------------------------------------------------------------------------

_PTX_MIXED = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_mixed_ctaid_use(
    .param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd0, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r3, %r1, %r2;     // UR-friendly: IMAD R-UR candidate
    add.u32 %r4, %r3, %r0;
    shl.b32 %r5, %r1, 2;           // GPR-required: shl of ctaid
    cvt.u64.u32 %rd1, %r5;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""


def test_mixed_ctaid_consumers_route_through_s2r():
    cubin = _compile(_PTX_MIXED)
    text = _text(cubin, 'k_mixed_ctaid_use')

    # The defining instruction for %ctaid.x must be S2R (GPR-writing).
    # S2UR (UR-writing) would leave the GPR slot undefined and a later
    # `shl.b32` on %r1 would read garbage.
    assert _has_opcode(text, _OPC_S2R), (
        'expected S2R for %ctaid.x when a GPR-only consumer exists '
        '(shl.b32). PTXAS-R19 regressed.')

    # The kernel still needs a GPR-writing S2R for the ctaid slot.
    s2r_count = sum(
        1 for addr in range(0, len(text), 16)
        if ((text[addr] | (text[addr + 1] << 8)) & 0xFFF) == _OPC_S2R
    )
    assert s2r_count >= 2, (
        f'expected ≥2 S2R (tid + ctaid when ctaid has GPR consumer); got {s2r_count}')


# ---------------------------------------------------------------------------
# Test 2 — cross-block GPR consumer forces S2R routing
# ---------------------------------------------------------------------------

_PTX_CROSS_BLOCK = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_cross_block_gpr(
    .param .u64 out, .param .u32 n)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [out];
    ld.param.u32 %r0, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.lo.u32 %r4, %r2, %r3;       // UR-friendly in entry block
    add.u32 %r5, %r4, %r1;
    setp.lt.u32 %p0, %r5, %r0;
    @%p0 bra store_path;
    ret;
store_path:
    shl.b32 %r6, %r2, 2;             // %ctaid consumer in LATER block
    cvt.u64.u32 %rd1, %r6;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r5;
    ret;
}
"""


def test_cross_block_gpr_consumer_forces_s2r():
    cubin = _compile(_PTX_CROSS_BLOCK)
    text = _text(cubin, 'k_cross_block_gpr')

    # The cross-block `shl.b32 %r6, %r2, 2` sits in `store_path` and
    # consumes `%ctaid.x` (aliased as `%r2`). The isel's PTXAS-R19 scan
    # MUST treat blocks other than the definition block as in-scope for
    # "GPR-required consumer" detection.
    assert _has_opcode(text, _OPC_S2R), (
        'cross-block GPR consumer of %ctaid.x did not trigger S2R '
        'routing — PTXAS-R19 cross-block detection regressed.')


# ---------------------------------------------------------------------------
# Test 3 — pure UR-friendly consumer set retains the S2UR fast path
# ---------------------------------------------------------------------------

_PTX_PURE_UR = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_pure_ur_ctaid(
    .param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd0, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r3, %r1, %r2;     // IMAD R-UR candidate (R1=mul src0, R2=ntid but is UR-eligible)
    mad.lo.u32 %r4, %r1, %r2, %r0;  // second UR-friendly consumer
    cvt.u64.u32 %rd1, %r4;
    shl.b64 %rd2, %rd1, 2;
    add.u64 %rd3, %rd0, %rd2;
    st.global.u32 [%rd3], %r4;
    ret;
}
"""


def test_pure_ur_consumers_keep_s2ur_fast_path():
    cubin = _compile(_PTX_PURE_UR)
    text = _text(cubin, 'k_pure_ur_ctaid')

    # Every %ctaid.x consumer here is a `mul.lo` / `mad.lo` which can
    # fuse the UR directly via IMAD R-UR (opcode 0xc24).  PTXAS-R19 must
    # leave this kernel on the S2UR fast path — emitting S2R here would
    # regress the uniform-register optimization the original code was
    # deliberately built for.
    assert _has_opcode(text, _OPC_S2UR), (
        'pure UR-friendly %ctaid consumers should retain the S2UR fast '
        'path (no GPR-only consumer exists, so no fallback needed).')
    assert _has_opcode(text, _OPC_IMAD_UR_RR), (
        'expected IMAD R-UR (0xc24) to fuse the mul.lo/mad.lo on ctaid*ntid')
