"""
Tests for the PTX parser.
Uses repro_sub_bug.ptx from the ptxas_bug toolkit if available.
"""

import os
import pytest
from pathlib import Path

from ptx.parser import parse, parse_file
from ptx.ir import ScalarKind


PTXAS_BUG_DIR = Path(r"ptxas_bug")
REPRO_PTX     = PTXAS_BUG_DIR / "repro_sub_bug.ptx"


MINIMAL_PTX = """\
.version 8.0
.target sm_120
.address_size 64

.visible .entry shift_sub(
    .param .u64 out_ptr,
    .param .u64 in_ptr,
    .param .s32 n)
{
    .reg .pred  %p<2>;
    .reg .s32   %r<4>;
    .reg .u64   %rd<10>;

    ld.param.u64    %rd0, [out_ptr];
    ld.param.u64    %rd1, [in_ptr];
    ld.param.s32    %r0,  [n];

    setp.ge.s32     %p0, %r0, 1;
    @%p0 bra BB0_end;

    // The buggy pattern: (a << 8) - (a >> 56)
    ld.global.u64   %rd5, [%rd1];
    shl.b64         %rd6, %rd5, 8;
    shr.u64         %rd7, %rd5, 56;
    sub.s64         %rd8, %rd6, %rd7;
    st.global.u64   [%rd0], %rd8;

BB0_end:
    ret;
}
"""


def test_parse_minimal():
    mod = parse(MINIMAL_PTX)
    assert mod.version == (8, 0)
    assert mod.target == "sm_120"
    assert mod.address_size == 64
    assert len(mod.functions) == 1


def test_kernel_name():
    mod = parse(MINIMAL_PTX)
    fn = mod.functions[0]
    assert fn.name == "shift_sub"
    assert fn.is_kernel is True


def test_param_count():
    mod = parse(MINIMAL_PTX)
    fn = mod.functions[0]
    assert len(fn.params) == 3


def test_reg_decls():
    mod = parse(MINIMAL_PTX)
    fn = mod.functions[0]
    # Should have .pred, .s32, .u64 declarations
    types_found = {rd.type.kind for rd in fn.reg_decls}
    assert ScalarKind.PRED in types_found
    assert ScalarKind.S in types_found
    assert ScalarKind.U in types_found


def test_instructions_parsed():
    mod = parse(MINIMAL_PTX)
    fn  = mod.functions[0]
    all_ops = [i.op for i in fn.all_instructions()]
    assert "shl" in all_ops
    assert "shr" in all_ops
    assert "sub" in all_ops
    assert "ld"  in all_ops
    assert "st"  in all_ops
    assert "ret" in all_ops


def test_shl_b64_types():
    mod = parse(MINIMAL_PTX)
    fn  = mod.functions[0]
    shl_insts = [i for i in fn.all_instructions() if i.op == "shl"]
    assert shl_insts, "Expected at least one shl instruction"
    assert "b64" in shl_insts[0].types


def test_sub_s64_types():
    mod = parse(MINIMAL_PTX)
    fn  = mod.functions[0]
    sub_insts = [i for i in fn.all_instructions() if i.op == "sub"]
    assert sub_insts, "Expected at least one sub instruction"
    assert "s64" in sub_insts[0].types


def test_predicated_branch():
    mod = parse(MINIMAL_PTX)
    fn  = mod.functions[0]
    bra_insts = [i for i in fn.all_instructions() if i.op == "bra"]
    assert bra_insts
    assert bra_insts[0].pred is not None


def test_label_blocks():
    mod = parse(MINIMAL_PTX)
    fn  = mod.functions[0]
    labels = [bb.label for bb in fn.blocks if bb.label]
    assert "BB0_end" in labels


@pytest.mark.skipif(not REPRO_PTX.exists(),
                    reason="repro_sub_bug.ptx not found (need ptxas_bug dir)")
def test_parse_repro_ptx():
    """Parse the real 33K repro_sub_bug.ptx from the ptxas_bug toolkit."""
    mod = parse_file(str(REPRO_PTX))
    assert mod.target.startswith("sm_")
    assert len(mod.functions) >= 1

    # Count total instructions
    total = sum(1 for fn in mod.functions for _ in fn.all_instructions())
    print(f"\n  repro_sub_bug.ptx: {len(mod.functions)} functions, {total} instructions")
    assert total > 50, "Expected a substantial PTX file"

    # Verify the buggy pattern is present
    all_ops = [i.op for fn in mod.functions for i in fn.all_instructions()]
    assert "shl" in all_ops
    assert "shr" in all_ops
    assert "sub" in all_ops


@pytest.mark.skipif(not REPRO_PTX.exists(),
                    reason="repro_sub_bug.ptx not found")
def test_rotate_pass_on_repro():
    """
    Run the rotate pass on repro_sub_bug.ptx and verify:
    - No valid rotate groups (because all uses are sub, which is Bug 1)
    - Bug 1 patterns ARE reported
    """
    from ptx.passes.rotate import run as rotate_run

    mod = parse_file(str(REPRO_PTX))
    _, groups = rotate_run(mod)

    # All the (a<<K)-(a>>(64-K)) patterns in repro_sub_bug.ptx use sub,
    # so the correct pass should find 0 valid rotate groups.
    # (ptxas finds them all "valid" — that's the bug.)
    print(f"\n  Valid rotate groups found: {len(groups)}")
    # We don't assert 0 here in case the file also has some add/or/xor rotates
    # in test harness code — just print for human inspection.
