"""Tests for PTX dead code elimination pass."""

from ptx.parser import parse
from ptx.passes.dce import run_function, is_side_effecting


DEAD_ADD_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<10>;
    .reg .u64 %res;
    ld.param.u64 %rd1, [a];
    ld.global.u64 %rd2, [%rd1];
    add.u64 %rd3, %rd2, 1;       // dead: never read
    add.u64 %rd4, %rd2, 2;       // dead: never read
    st.global.u64 [%rd1], %rd2;
    ret;
}
"""

CHAIN_DEAD_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<10>;
    ld.param.u64 %rd1, [a];
    ld.global.u64 %rd2, [%rd1];
    add.u64 %rd3, %rd2, 1;       // dead transitively
    add.u64 %rd4, %rd3, 2;       // dead
    shl.b64 %rd5, %rd4, 1;       // dead, uses %rd4
    st.global.u64 [%rd1], %rd2;
    ret;
}
"""

PRESERVE_STORE_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<10>;
    ld.param.u64 %rd1, [a];
    add.u64 %rd2, %rd1, 8;       // live: used by store
    st.global.u64 [%rd2], %rd1;  // side-effecting: keep
    ret;
}
"""


def test_dce_removes_simple_dead_adds():
    mod = parse(DEAD_ADD_PTX)
    fn = mod.functions[0]
    before = sum(len(bb.instructions) for bb in fn.blocks)
    removed = run_function(fn)
    after = sum(len(bb.instructions) for bb in fn.blocks)
    assert removed == 2, f"expected 2 dead inst removed, got {removed}"
    assert after == before - 2


def test_dce_removes_chained_dead_code():
    mod = parse(CHAIN_DEAD_PTX)
    fn = mod.functions[0]
    removed = run_function(fn)
    # 3 dead instructions: %rd3, %rd4, %rd5 chain
    assert removed == 3, f"expected 3 dead inst removed, got {removed}"


def test_dce_preserves_stores():
    mod = parse(PRESERVE_STORE_PTX)
    fn = mod.functions[0]
    removed = run_function(fn)
    # Nothing to remove: %rd2 feeds the store, st.global is side-effecting
    assert removed == 0
    ops = [i.op for bb in fn.blocks for i in bb.instructions]
    assert "st" in ops
    assert "add" in ops


def test_is_side_effecting_flags():
    # Side-effecting ops
    mod = parse(PRESERVE_STORE_PTX)
    fn = mod.functions[0]
    st_inst = next(i for bb in fn.blocks for i in bb.instructions if i.op == "st")
    assert is_side_effecting(st_inst)
    add_inst = next(i for bb in fn.blocks for i in bb.instructions if i.op == "add")
    assert not is_side_effecting(add_inst)
