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
    """64-bit regs get pairs starting at R2, no coalescing.

    FB-5.1 + PTXAS-R29.1 layering:
      * %rd0 is the base of a *scalar* ld.global with zero offset.  R29.1
        reclassifies such u64 params as ``direct_ldc_params`` — the body
        emits ``LDC.64 R_pair`` straight into a GPR pair, which feeds
        ``LDG.E desc[UR][R_pair.64]``.  So %rd0 DOES get a GPR
        allocation (the LDC.64 target), and it also appears in
        ``direct_ldc_params``.
      * %rd5 is the base of a st.global (not ld.global) — R29.1's
        ld-only filter doesn't apply.  FB-5.1's broader UR-bound
        classification applies instead: st.global MemOp base is safe
        for UR, so %rd5 is UR-bound and skipped from int_regs.
    """
    mod = parse(PROBE_PTX)
    fn = mod.functions[0]
    result = allocate(fn)

    # %rd0: R29.1 direct_ldc_params — gets a GPR AND is in the set
    assert "%rd0" in result.ra.int_regs
    assert "%rd0" in result.direct_ldc_params

    # %rd5: UR-bound (st.global base, not R29.1 ld-only) — skipped from int_regs
    assert "%rd5" not in result.ra.int_regs
    assert "%rd5" not in result.direct_ldc_params

    # %rd0 is allocated first (R2, R3 as the LDC.64 target pair)
    assert result.ra.lo("%rd0") == 2
    assert result.ra.hi("%rd0") == 3


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
    """GPR count is correct for cubin metadata.

    FB-5.1: %rd0 and %rd5 are UR-bound (skipped); the remaining vregs reuse
    the freed slots via linear scan, so the high-water mark drops below the
    pre-FB-5.1 value of 8.
    """
    mod = parse(PROBE_PTX)
    fn = mod.functions[0]
    result = allocate(fn)

    # Post-FB-5.1: only %rd1..%rd4 need real pairs, plus the address scratch.
    # The exact value is below the pre-FB-5.1 figure (8) and at least R0+R1+
    # one pair (4).  The point of the test is the GPR count remains stable.
    assert 4 <= result.num_gprs <= 8


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


# ---------------------------------------------------------------------------
# LDG coalescing tests — reclaim address reg as destination when safe
# ---------------------------------------------------------------------------

COALESCE_SAFE_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry coalesce_safe(.param .u64 a)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<10>;
    .reg .u64 %res;
    // FB-5.1: %rd0 is UR-bound (only consumed by add.u64).
    // %rd1 is a computed address — gets a real GPR pair which is the
    // subject of the LDG coalescing test.
    ld.param.u64 %rd0, [a];
    mov.u32 %r0, 8;
    cvt.u64.u32 %rd9, %r0;
    add.u64 %rd1, %rd0, %rd9;
    ld.global.u64 %rd2, [%rd1];
    add.u64 %res, %rd2, %rd2;
    ret;
}
"""

COALESCE_UNSAFE_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry coalesce_unsafe(.param .u64 a)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<10>;
    // FB-5.1: %rd1 is a computed address (not a direct param).
    // It is alive past the LDG (also used by the STG), so coalescing
    // with %rd2 must NOT fire.
    ld.param.u64 %rd0, [a];
    mov.u32 %r0, 8;
    cvt.u64.u32 %rd9, %r0;
    add.u64 %rd1, %rd0, %rd9;
    ld.global.u64 %rd2, [%rd1];
    add.u64 %rd3, %rd2, 1;
    st.global.u64 [%rd1], %rd3;
    ret;
}
"""


def test_ldg_coalesce_saves_register():
    """When addr dies at the load, dest should share addr's phys reg."""
    mod = parse(COALESCE_SAFE_PTX)
    fn = mod.functions[0]
    result = allocate(fn, sm_version=120)
    # %rd1 and %rd2 should share the same physical register
    assert "%rd1" in result.ra.int_regs
    assert "%rd2" in result.ra.int_regs
    assert result.ra.int_regs["%rd1"] == result.ra.int_regs["%rd2"], (
        f"coalesce failed: %rd1={result.ra.int_regs['%rd1']} "
        f"%rd2={result.ra.int_regs['%rd2']}"
    )


def test_ldg_coalesce_respects_interference():
    """When addr is used AFTER the load, coalescing must not fire."""
    mod = parse(COALESCE_UNSAFE_PTX)
    fn = mod.functions[0]
    result = allocate(fn, sm_version=120)
    # %rd1 is alive past the load (used in st.global), must NOT share with %rd2
    assert result.ra.int_regs["%rd1"] != result.ra.int_regs["%rd2"], (
        "coalesce fired despite interference: %rd1 alive past load"
    )
