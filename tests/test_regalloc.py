"""Tests for the register allocator."""

import pytest
from ptx.parser import parse
from sass.regalloc import allocate, PARAM_BASE_SM120


PROBE_PTX = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry probe_k1(
    .param .u64 out_ptr,
    .param .u64 in_ptr)
{
    .reg .b64   %rd<8>;
    ld.param.u64    %rd0, [in_ptr];
    ld.global.u64   %rd1, [%rd0];
    shl.b64         %rd2, %rd1, 1;
    shr.u64         %rd3, %rd1, 63;
    add.s64         %rd4, %rd2, %rd3;
    ld.param.u64    %rd5, [out_ptr];
    st.global.u64   [%rd5], %rd4;
    ret;
}
"""

MIXED_PTX = """\
.version 8.0
.target sm_120
.address_size 64

.visible .entry mixed_regs(
    .param .u64 out_ptr,
    .param .u64 in_ptr,
    .param .s32 n)
{
    .reg .pred  %p<2>;
    .reg .s32   %r<4>;
    .reg .u64   %rd<6>;
    ret;
}
"""


def test_alloc_basic():
    """64-bit regs get pairs starting at R2, no coalescing."""
    mod = parse(PROBE_PTX)
    fn = mod.functions[0]
    result = allocate(fn)

    # %rd0 → R2 (lo), R3 (hi)
    assert result.ra.lo("%rd0") == 2
    assert result.ra.hi("%rd0") == 3

    # %rd1 gets next available pair (no LDG coalescing)
    assert result.ra.lo("%rd1") == 4
    assert result.ra.hi("%rd1") == 5


def test_alloc_param_offsets():
    """Params get sequential offsets starting at PARAM_BASE."""
    mod = parse(PROBE_PTX)
    fn = mod.functions[0]
    result = allocate(fn)

    # Two u64 params: out_ptr at 0x380, in_ptr at 0x388
    assert result.param_offsets["out_ptr"] == PARAM_BASE_SM120
    assert result.param_offsets["in_ptr"] == PARAM_BASE_SM120 + 8


def test_alloc_mixed_types():
    """Mixed kernel: unused regs are skipped."""
    mod = parse(MIXED_PTX)
    fn = mod.functions[0]
    result = allocate(fn)

    # MIXED_PTX has only 'ret;' — no regs are actually used
    # Unused regs are skipped by the allocator
    assert result.num_gprs == 2  # only R0-R1 reserved


def test_alloc_three_params():
    """Three params with mixed sizes get correct offsets."""
    mod = parse(MIXED_PTX)
    fn = mod.functions[0]
    result = allocate(fn)

    # out_ptr (u64) at 0x380, in_ptr (u64) at 0x388, n (s32) at 0x390
    assert result.param_offsets["out_ptr"] == PARAM_BASE_SM120
    assert result.param_offsets["in_ptr"] == PARAM_BASE_SM120 + 8
    assert result.param_offsets["n"] == PARAM_BASE_SM120 + 16


def test_num_gprs():
    """GPR count is correct for cubin metadata."""
    mod = parse(PROBE_PTX)
    fn = mod.functions[0]
    result = allocate(fn)

    # Linear scan with reuse: rd0→R2, rd1→R4, rd2→R2(reuse), rd3→R6, rd4→R4(reuse), rd5→R2(reuse)
    # Highest reg used = R7, so num_gprs = 8
    assert result.num_gprs == 8


def test_64bit_alignment():
    """64-bit regs are always aligned to even register indices."""
    mod = parse(PROBE_PTX)
    fn = mod.functions[0]
    result = allocate(fn)

    for i in range(6):  # only %rd0-%rd5 are used in probe_k1
        name = f"%rd{i}"
        if name in result.ra.int_regs:
            lo = result.ra.lo(name)
            assert lo % 2 == 0, f"{name} lo={lo} not even-aligned"
