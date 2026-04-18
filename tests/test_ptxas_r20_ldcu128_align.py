"""PTXAS-R20 regression — guard post-EXIT LDCU.64→LDCU.128 upcast on alignment.

Proven defect (FB-1 Phase A reduction): the post-EXIT "Rule #29" rewrite
unconditionally flipped byte-9 of the first post-EXIT LDCU.64 from 0x0a
(64-bit) to 0x0c (128-bit).  LDCU.128 requires a 16-byte-aligned byte
offset — `encode_ldcu_128()` itself asserts this with a ValueError.  The
post-EXIT rewrite bypassed that guard, so any kernel whose first post-
EXIT deferred param lives at an 8-byte-aligned-but-not-16-byte-aligned
offset (the canonical case: a 2nd u64 param at c[0][0x388]) ended up
with a malformed LDCU.128.  The UR pair's high half held garbage, a
downstream IADD.64 produced an invalid 64-bit store address, and STG
raised CUDA_ERROR_ILLEGAL_ADDRESS on the active lane.

These tests prove the semantic class:
  1. A kernel with exactly one u64 param at c[0][0x380] retains the
     valid LDCU.64→LDCU.128 upcast (offset 0x380 is 16-byte aligned).
  2. A kernel with a second u64 param at c[0][0x388] does NOT get the
     upcast — the first post-EXIT LDCU.64 keeps byte-9 = 0x0a.
  3. Every LDCU with byte-9 = 0x0c in a compiled cubin has a 16-byte-
     aligned byte offset (invariant: no illegal upcast anywhere).
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.parser import parse
from sass.pipeline import compile_function


_OPC_LDCU = 0x7ac


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


def _ldcu_instrs(text: bytes):
    """Yield (addr, raw[5], raw[9]) for each LDCU in the text section."""
    for addr in range(0, len(text), 16):
        opc = (text[addr] | (text[addr + 1] << 8)) & 0xFFF
        if opc == _OPC_LDCU:
            yield (addr, text[addr + 5], text[addr + 9])


# ---------------------------------------------------------------------------
# Shared-shape PTX: one u32 param + predicated ret after a LDG + st.global.u32
# with the specified output-ptr layout.
# ---------------------------------------------------------------------------

_PTX_ONE_U64_OUT = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_one_u64_out(.param .u64 out)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [out];
    mov.u32 %r0, %tid.x;
    setp.eq.u32 %p0, %r0, 0;
    @!%p0 ret;
    st.global.u32 [%rd0], %r0;
    ret;
}
"""

_PTX_TWO_U64_PARAMS = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_two_u64_params(.param .u64 a, .param .u64 out)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<4>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [a];
    ld.param.u64 %rd1, [out];
    mov.u32 %r0, %tid.x;
    setp.eq.u32 %p0, %r0, 0;
    @!%p0 ret;
    st.global.u32 [%rd1], %r0;
    ret;
}
"""


def test_aligned_post_exit_ldcu_retains_128_upcast():
    """The `out` param sits at c[0][0x380] (16-byte aligned) — the
    post-EXIT LDCU.64→LDCU.128 upcast is LEGAL here and must still fire.
    """
    cubin = _compile(_PTX_ONE_U64_OUT)
    text = _text(cubin, 'k_one_u64_out')

    saw_128_at_aligned = False
    for addr, qw_idx, b9 in _ldcu_instrs(text):
        byte_off = qw_idx * 8
        if b9 == 0x0c:
            # Any 128-bit LDCU must be 16-byte aligned.
            assert byte_off % 16 == 0, (
                f'LDCU.128 at kernel-addr 0x{addr:x} reads c[0][0x{byte_off:x}] '
                f'which is NOT 16-byte aligned — illegal upcast regressed.')
            if byte_off == 0x380:
                saw_128_at_aligned = True
    assert saw_128_at_aligned, (
        'expected at least one LDCU.128 at c[0][0x380] for this kernel '
        '(aligned post-EXIT upcast); R20 should not have removed the '
        'valid upcast path.')


def test_misaligned_post_exit_ldcu_stays_64():
    """The 2nd u64 param `out` sits at c[0][0x388] (8-byte but NOT 16-byte
    aligned) — the post-EXIT upcast must NOT fire here.  All LDCUs that
    read from 0x388 must keep byte-9 = 0x0a (LDCU.64).
    """
    cubin = _compile(_PTX_TWO_U64_PARAMS)
    text = _text(cubin, 'k_two_u64_params')

    for addr, qw_idx, b9 in _ldcu_instrs(text):
        byte_off = qw_idx * 8
        if byte_off == 0x388:
            assert b9 == 0x0a, (
                f'LDCU at kernel-addr 0x{addr:x} reads c[0][0x388] with '
                f'byte-9 = 0x{b9:02x} — the Rule #29 upcast fired on a '
                f'NON-16-byte-aligned offset, producing a malformed '
                f'LDCU.128. Byte-9 must stay 0x0a for this offset.')


def test_no_ldcu128_on_unaligned_offset_globally():
    """Invariant guard over both kernels: NO LDCU.128 (byte-9 = 0x0c) may
    point at a non-16-byte-aligned byte offset anywhere in either cubin.
    This is the exact constraint `encode_ldcu_128` asserts at encoder
    time; the PTXAS-R20 guard brings the post-emission rewrite path into
    alignment with that contract.
    """
    for src, kernel in (
        (_PTX_ONE_U64_OUT, 'k_one_u64_out'),
        (_PTX_TWO_U64_PARAMS, 'k_two_u64_params'),
    ):
        cubin = _compile(src)
        text = _text(cubin, kernel)
        for addr, qw_idx, b9 in _ldcu_instrs(text):
            byte_off = qw_idx * 8
            if b9 == 0x0c:
                assert byte_off % 16 == 0, (
                    f'kernel {kernel}: LDCU.128 at 0x{addr:x} points at '
                    f'c[0][0x{byte_off:x}] — byte offset not 16-byte '
                    f'aligned. Illegal upcast.')
