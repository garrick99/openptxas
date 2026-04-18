"""PTXAS-R23A.1 regression — LDC opex (ctrl-byte) must stay in the
decoder-valid range.

Proven defect (R23A residual): the SM_120 instruction decoder reads a
5-bit `opex` field from each instruction's control word, where for LDC
the opex is formed as `(wdep[0] << 4) | misc`.  LDC's wdep is `0x31`
(bit 0 set), so ctrl-bit-4 = 1 and `opex = 0x10 | misc`.  Empirically,
`nvdisasm` and the hardware decoder reject opex == `0x1d`/`0x1e`/`0x1f`
(i.e. misc 13/14/15) for the LDC opclass with
`Opclass 'ldc__RaRZ', undefined value 0x1d for table TABLES_opex_0`.
The scoreboard's `assign_ctrl` previously used `misc = misc_counter &
0xF` unconditionally — fine for most ops but unsafe when an LDC landed
at a counter position ≥ 13.  R23A.1 narrows misc to `& 0x7` for LDC
when the unconstrained counter would produce misc ≥ 0xd.

These tests prove the invariant at SASS level:
  1. Every LDC emitted by OpenPTXas must carry a ctrl word whose
     opex (bits 0-4) stays out of the rejected set {0x1d, 0x1e, 0x1f}.
  2. Non-LDC opcodes are unaffected — the clamp fires only for opcode
     0xb82.
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.parser import parse
from sass.pipeline import compile_function


_OPC_LDC = 0xb82
_INVALID_LDC_OPEX = {0x1d, 0x1e, 0x1f}


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


def _compile(ptx_src: str) -> bytes:
    return compile_function(parse(ptx_src).functions[0],
                            verbose=False, sm_version=120)


def _ldc_opex(raw16: bytes) -> int:
    """Extract the 5-bit opex field (bits 0-4 of the ctrl word) from a
    16-byte SASS instruction.  `_patch_ctrl` stores `(ctrl & 0x7FFFFF) <<
    1` into bytes [13..15].  We reverse that shift to recover ctrl,
    then pick bits 0-4."""
    raw24 = raw16[13] | (raw16[14] << 8) | ((raw16[15] & 0xFB) << 16)
    ctrl = raw24 >> 1
    return ctrl & 0x1F


# ---------------------------------------------------------------------------
# Test 1 — G8-shape kernel: LDC ctrl must decode as valid opex
# ---------------------------------------------------------------------------

_PTX_G8_SHAPE = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_g8_shape(.param .u64 in, .param .u64 out)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [in];
    ld.param.u64 %rd0, [out];
    mov.u32 %r0, %tid.x;
    setp.eq.u32 %p0, %r0, 0;
    @!%p0 ret;
    mov.u32 %r1, %ctaid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd2, %r2;
    add.u64 %rd1, %rd0, %rd2;
    st.global.u32 [%rd1], 305419896;
    ret;
}
"""


def test_g8_shape_ldc_opex_is_decoder_valid():
    """Every LDC in the G8-shape cubin must have an opex value the
    SM_120 decoder accepts.  The pre-R23A.1 failure emitted opex == 0x1d
    at the literal-materialization LDC."""
    cubin = _compile(_PTX_G8_SHAPE)
    text = _text(cubin, 'k_g8_shape')

    seen_ldc_count = 0
    for addr in range(0, len(text), 16):
        opc = (text[addr] | (text[addr + 1] << 8)) & 0xFFF
        if opc != _OPC_LDC:
            continue
        seen_ldc_count += 1
        opex = _ldc_opex(text[addr:addr + 16])
        assert opex not in _INVALID_LDC_OPEX, (
            f'LDC at 0x{addr:x} has opex=0x{opex:02x} — rejected by the '
            f'SM_120 `ldc__RaRZ` decoder table.  PTXAS-R23A.1 must clamp '
            f'misc to avoid opex in {{0x1d, 0x1e, 0x1f}}.')
    assert seen_ldc_count > 0, (
        'expected at least one LDC in this kernel (frame-ptr, params, '
        'and the literal-materialization load)')


# ---------------------------------------------------------------------------
# Test 2 — every LDC in a complex kernel keeps valid opex
# ---------------------------------------------------------------------------
#
# Generate a kernel long enough that misc_counter climbs past 12 before
# hitting a body LDC.  Without the R23A.1 clamp, one of those LDCs would
# land at counter ∈ {13, 14, 15} and produce an invalid opex.

_PTX_COUNTER_STRESS = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_counter_stress(.param .u64 out)
{
    .reg .b32 %r<20>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r3, %r1, %r2;
    add.u32 %r4, %r3, %r0;
    add.u32 %r5, %r4, 1;
    add.u32 %r6, %r5, 2;
    add.u32 %r7, %r6, 3;
    add.u32 %r8, %r7, 4;
    add.u32 %r9, %r8, 5;
    add.u32 %r10, %r9, 6;
    add.u32 %r11, %r10, 7;
    setp.eq.u32 %p0, %r0, 0;
    @!%p0 ret;
    st.global.u32 [%rd0], %r11;
    ret;
}
"""


def test_counter_stress_all_ldc_opex_valid():
    """Runs a long-ish kernel whose `misc_counter` crosses 12 before the
    post-EXIT / late body region.  Every LDC in the output must have
    opex outside the rejected set."""
    cubin = _compile(_PTX_COUNTER_STRESS)
    text = _text(cubin, 'k_counter_stress')

    for addr in range(0, len(text), 16):
        opc = (text[addr] | (text[addr + 1] << 8)) & 0xFFF
        if opc != _OPC_LDC:
            continue
        opex = _ldc_opex(text[addr:addr + 16])
        assert opex not in _INVALID_LDC_OPEX, (
            f'counter-stress kernel: LDC at 0x{addr:x} has opex=0x{opex:02x}, '
            f'rejected by SM_120 decoder. R23A.1 clamp did not fire.')
