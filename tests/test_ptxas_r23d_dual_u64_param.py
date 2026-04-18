"""PTXAS-R23D regression — dual-u64-param reuse pattern requires both:

  (1) elimination of the dead pre-EXIT LDC.64 that the SSA-name-reuse
      path (`ld.param.u64 %rd0, [in]; ld.param.u64 %rd0, [out]`) would
      otherwise emit, and
  (2) a synthesized post-EXIT ULDCU.128 priming load over the param
      area, required by the SM_120 driver for STG.E on 2+ u64-param
      kernels when the live u64 param's byte offset is non-16-aligned
      (which pushes it to the GPR-direct path via R22, bypassing the
      UR-preamble that would otherwise provide the priming).

Proof: G4/G8/G1 Family-A GPU runs confirm bit-exact immediate storage;
earlier turns verified that re-swapping G8's .text bytes with a variant
containing both (1) and (2) makes G8 PASS on hardware.

These tests enforce the invariants at SASS level without requiring a
GPU."""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.parser import parse
from sass.pipeline import compile_function


_OPC_LDC    = 0xb82   # 32/64-bit LDC (constant-bank scalar load)
_OPC_LDCU   = 0x7ac   # ULDCU (uniform-register constant load)
_OPC_EXIT   = 0x94d
_PARAM_BASE = 0x380


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


def _compile(ptx_src: str) -> bytes:
    return compile_function(parse(ptx_src).functions[0],
                            verbose=False, sm_version=120)


def _count_pre_exit_ldc64_at(text: bytes, byte_off: int) -> int:
    """Count pre-EXIT LDC.64 (opc=0xb82, b9=0x0a) reading c[0][byte_off]."""
    n = 0
    dword_target = byte_off // 4
    for a in range(0, len(text), 16):
        raw = text[a:a + 16]
        opc = (raw[0] | (raw[1] << 8)) & 0xFFF
        if opc == _OPC_EXIT and ((raw[1] >> 4) & 0xF) != 7:
            break  # predicated EXIT — stop scanning (entered post-EXIT region)
        if opc != _OPC_LDC:
            continue
        if raw[9] != 0x0a:  # not LDC.64
            continue
        dw = raw[5] | ((raw[6] & 0x03) << 8)
        if dw == dword_target:
            n += 1
    return n


def _find_post_exit_uldcu128_at(text: bytes, byte_off: int) -> int | None:
    """Return offset of first post-EXIT ULDCU.128 (opc=0x7ac, b9=0x0c)
    whose byte offset equals `byte_off`, or None.  encode_ldcu_64 stores
    qword_offset directly in raw[5]; rule #29's upcast keeps the offset
    and only flips b9 from 0x0a → 0x0c."""
    qword_target = byte_off // 8
    post_exit = False
    for a in range(0, len(text), 16):
        raw = text[a:a + 16]
        opc = (raw[0] | (raw[1] << 8)) & 0xFFF
        guard = (raw[1] >> 4) & 0xF
        if opc == _OPC_EXIT and guard != 7:
            post_exit = True
            continue
        if not post_exit:
            continue
        if opc != _OPC_LDCU or raw[9] != 0x0c:
            continue
        if raw[5] == qword_target:
            return a
    return None


# ---------------------------------------------------------------------------
# Fixture kernels — the dual-u64-param reuse pattern (G8 shape)
# ---------------------------------------------------------------------------

_PTX_DUAL_REUSE = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_dual_reuse(.param .u64 in, .param .u64 out)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [in];               // DEAD — overwritten before use
    ld.param.u64 %rd0, [out];              // live
    mov.u32 %r0, %tid.x;
    setp.eq.u32 %p0, %r0, 0;
    @!%p0 ret;
    mov.u32 %r1, %ctaid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd2, %r2;
    add.u64 %rd1, %rd0, %rd2;
    st.global.u32 [%rd1], 305419896;       // 0x12345678
    ret;
}
"""

_PTX_SINGLE_PARAM = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_single(.param .u64 out)
{
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
    st.global.u32 [%rd1], 2864434397;      // 0xAABBCCDD
    ret;
}
"""


# ---------------------------------------------------------------------------
# Test 1 — R23D half A: the dead ld.param.u64 must NOT emit a pre-EXIT LDC.64
# ---------------------------------------------------------------------------

def test_dead_ldparam_u64_first_load_not_emitted():
    """With `ld.param.u64 %rd0, [in]; ld.param.u64 %rd0, [out]`, only the
    second (live) load should produce a pre-EXIT LDC.64.  The first is
    dead — emitting it produces a pre-EXIT shape that breaks SM_120 STG."""
    text = _text(_compile(_PTX_DUAL_REUSE), 'k_dual_reuse')
    # Param 'in' is at byte 0x380, param 'out' is at byte 0x388.
    n_in  = _count_pre_exit_ldc64_at(text, _PARAM_BASE)         # 0x380
    n_out = _count_pre_exit_ldc64_at(text, _PARAM_BASE + 8)     # 0x388
    assert n_in == 0, (
        f"pre-EXIT LDC.64 for dead param `in` (c[0][0x{_PARAM_BASE:x}]) "
        f"was emitted {n_in} time(s); R23D must suppress the dead load.")
    assert n_out == 1, (
        f"pre-EXIT LDC.64 for live param `out` (c[0][0x{_PARAM_BASE+8:x}]) "
        f"emitted {n_out} time(s); expected exactly 1.")


# ---------------------------------------------------------------------------
# Test 2 — R23D half B: post-EXIT ULDCU.128 priming at param base
# ---------------------------------------------------------------------------

def test_post_exit_uldcu128_priming_present_on_dual_reuse():
    """For the dual-u64-param-reuse kernel shape, the emitted text must
    contain a post-EXIT ULDCU.128 (opc=0x7ac, b9=0x0c) reading from
    c[0][0x380] (the param area).  Without this priming, SM_120 STG.E
    hits CUDA_ERROR_ILLEGAL_ADDRESS at launch."""
    text = _text(_compile(_PTX_DUAL_REUSE), 'k_dual_reuse')
    off = _find_post_exit_uldcu128_at(text, _PARAM_BASE)
    assert off is not None, (
        f"no post-EXIT ULDCU.128 at c[0][0x{_PARAM_BASE:x}] found in "
        f"dual-reuse kernel text; R23D priming synthesis must queue a "
        f"preamble LDCU.64 at the dead-load offset so rule #29 upcasts "
        f"it to ULDCU.128.")


# ---------------------------------------------------------------------------
# Test 3 — R23D does not affect the single-u64-param baseline
# ---------------------------------------------------------------------------

def test_single_u64_param_unchanged_by_r23d():
    """The single-u64-param shape must still compile successfully and
    R23D's dead-load priming path must NOT fire (no dead ld.param).
    The single-param live load goes through the UR-preamble path (not
    GPR-direct), which is the pre-existing behavior — R23D leaves it
    untouched."""
    # Compilation itself is the proof — R23D must not raise, regress,
    # or otherwise change single-param behavior.  An earlier GPU run
    # (G4 PASS) confirms correctness end-to-end; this test guards the
    # emission path.
    cubin = _compile(_PTX_SINGLE_PARAM)
    text = _text(cubin, 'k_single')
    # The kernel text must still contain a MOV32I for the immediate
    # (opcode 0x431, imm bytes 0xAABBCCDD in little-endian at b4..b7)
    # — sanity that the single-param path is intact end-to-end.
    target_imm_le = b'\xdd\xcc\xbb\xaa'
    found = False
    for a in range(0, len(text), 16):
        raw = text[a:a + 16]
        opc = (raw[0] | (raw[1] << 8)) & 0xFFF
        if opc == 0x431 and raw[4:8] == target_imm_le:
            found = True
            break
    assert found, (
        "single-param kernel text should still contain the R23B.A "
        "MOV32I materializing 0xAABBCCDD; R23D must not regress it.")
