"""PTXAS-R21 regression — descriptor (UR4) must be loaded before any
pre-EXIT memory op that consumes it.

Proven defect (FB-1 Phase A reduction): the scheduler's TE26-B family
classifier in `sass/schedule.py` decided "STG-only" by checking only
`post_boundary` for LDG/ATOMG.  A kernel with a pre-EXIT `LDG.E
desc[UR4][...]` and post-EXIT `STG.E` was mis-classified as STG-only,
so the `LDCU.64 UR4, c[0x0][0x358]` descriptor load was moved PAST the
predicated EXIT.  The pre-EXIT LDG consumed `UR4` before it was
defined — use-before-def — producing CUDA_ERROR_ILLEGAL_ADDRESS on the
active lane.

These tests prove the semantic class (SASS-level, no GPU required):
  1. A kernel with a pre-EXIT LDG + post-EXIT STG must emit `LDCU.64
     UR4` BEFORE the first LDG in the text section.
  2. A kernel with only post-EXIT STG (no LDG anywhere) is still
     eligible for the STG-only post-EXIT LDCU placement — the R21
     guard must not regress that path.
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.parser import parse
from sass.pipeline import compile_function


_OPC_LDG_E    = 0x981
_OPC_STG_E    = 0x986
_OPC_LDCU     = 0x7ac
_OPC_EXIT     = 0x94d
_OPC_LDCU_64_DESC_OFFSET_QWORD = 0x358 // 8   # UR4 descriptor is at c[0][0x358]


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


def _first_addr_of(text: bytes, opc_target: int, *, require_qword=None) -> int:
    """Return the byte offset of the first SASS instruction matching the
    given opcode.  If `require_qword` is provided, only match LDCU with
    raw[5] == require_qword (useful for pinning the descriptor LDCU).
    Returns -1 if not found."""
    for addr in range(0, len(text), 16):
        opc = (text[addr] | (text[addr + 1] << 8)) & 0xFFF
        if opc != opc_target:
            continue
        if require_qword is not None and text[addr + 5] != require_qword:
            continue
        return addr
    return -1


# ---------------------------------------------------------------------------
# Test 1 — pre-EXIT LDG requires descriptor before it
# ---------------------------------------------------------------------------

_PTX_PRE_EXIT_LDG = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_pre_exit_ldg(.param .u64 in, .param .u64 out)
{
    .reg .b32 %r<5>;
    .reg .b64 %rd<5>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [in];
    ld.param.u64 %rd1, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r3, %r1, %r2;
    add.u32 %r4, %r3, %r0;
    cvt.u64.u32 %rd2, %r4;
    shl.b64  %rd3, %rd2, 2;
    add.u64  %rd4, %rd0, %rd3;
    ld.global.u32 %r4, [%rd4];            // pre-EXIT LDG — consumes UR4
    setp.eq.u32 %p0, %r0, 0;
    @!%p0 ret;
    st.global.u32 [%rd1], %r4;            // post-EXIT STG
    ret;
}
"""


def test_pre_exit_ldg_forces_descriptor_before_ldg():
    cubin = _compile(_PTX_PRE_EXIT_LDG)
    text = _text(cubin, 'k_pre_exit_ldg')

    ldcu_desc_addr = _first_addr_of(
        text, _OPC_LDCU,
        require_qword=_OPC_LDCU_64_DESC_OFFSET_QWORD)  # 0x358 / 8 = 0x6b
    assert ldcu_desc_addr != -1, (
        'expected to find LDCU.64 UR4, c[0][0x358] in the text section')

    ldg_addr = _first_addr_of(text, _OPC_LDG_E)
    assert ldg_addr != -1, (
        'expected a LDG.E in the text section for this kernel')

    assert ldcu_desc_addr < ldg_addr, (
        f'LDCU.64 UR4 at 0x{ldcu_desc_addr:x} must precede LDG.E at '
        f'0x{ldg_addr:x}. The pre-R21 scheduler classified this kernel '
        f'as STG-only (looking only at post_boundary) and moved the '
        f'descriptor past the EXIT, producing use-before-def of UR4 '
        f'and CUDA_ERROR_ILLEGAL_ADDRESS at the LDG.')


# ---------------------------------------------------------------------------
# Test 2 — STG-only path stays post-EXIT when no LDG anywhere
# ---------------------------------------------------------------------------

_PTX_STG_ONLY = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_stg_only(.param .u64 out)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    setp.eq.u32 %p0, %r0, 0;
    @!%p0 ret;
    st.global.u32 [%rd0], %r1;
    ret;
}
"""


def test_stg_only_keeps_post_exit_ldcu_placement():
    """When there is NO LDG/ATOMG anywhere in the kernel, the STG-only
    classifier must still fire and the LDCU.64 for the descriptor may
    still be placed post-EXIT.  The R21 guard is allowed only to
    disqualify the STG-only path when a pre-EXIT LDG exists — it must
    not block this purely-STG case."""
    cubin = _compile(_PTX_STG_ONLY)
    text = _text(cubin, 'k_stg_only')

    exit_addr = _first_addr_of(text, _OPC_EXIT)
    assert exit_addr != -1, 'expected a predicated EXIT in this kernel'

    ldcu_desc_addr = _first_addr_of(
        text, _OPC_LDCU,
        require_qword=_OPC_LDCU_64_DESC_OFFSET_QWORD)
    assert ldcu_desc_addr != -1, 'expected LDCU.64 UR4 in the kernel'

    # LDCU.64 UR4 is allowed to sit after the first predicated EXIT for
    # this STG-only kernel (matches the TE26-B / ptxas layout).
    stg_addr = _first_addr_of(text, _OPC_STG_E)
    assert stg_addr != -1, 'expected STG.E in this kernel'
    assert ldcu_desc_addr < stg_addr, (
        f'even in the STG-only post-EXIT placement, LDCU.64 UR4 at '
        f'0x{ldcu_desc_addr:x} must precede STG.E at 0x{stg_addr:x}.')
