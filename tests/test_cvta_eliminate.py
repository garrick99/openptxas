"""Tests for the cvta-eliminate PTX pass (Phase 14)."""

from ptx.parser import parse
from ptx.passes.cvta_eliminate import run_function


CVTA_TO_GLOBAL_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<10>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd0, [a];
    cvta.to.global.u64 %rd1, %rd0;
    st.global.u32 [%rd1], %r0;
    ret;
}
"""


CVTA_GLOBAL_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<10>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd0, [a];
    cvta.global.u64 %rd1, %rd0;
    st.global.u32 [%rd1], %r0;
    ret;
}
"""


MULTI_CVTA_SAME_SRC_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<10>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd0, [a];
    cvta.to.global.u64 %rd1, %rd0;
    cvta.to.global.u64 %rd2, %rd0;
    cvta.to.global.u64 %rd3, %rd0;
    st.global.u32 [%rd1], %r0;
    st.global.u32 [%rd2], %r1;
    st.global.u32 [%rd3], %r2;
    ret;
}
"""


PREDICATED_CVTA_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<10>;
    .reg .u32 %r<4>;
    .reg .pred %p0;
    ld.param.u64 %rd0, [a];
    setp.ne.u64 %p0, %rd0, 0;
    @%p0 cvta.to.global.u64 %rd1, %rd0;
    @%p0 st.global.u32 [%rd1], %r0;
    ret;
}
"""


CVTA_TRANSITIVE_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<10>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd0, [a];
    cvta.to.global.u64 %rd1, %rd0;
    cvta.to.global.u64 %rd2, %rd1;
    st.global.u32 [%rd2], %r0;
    ret;
}
"""


def _ops(fn):
    return [i.op for bb in fn.blocks for i in bb.instructions]


def _instructions(fn):
    return [i for bb in fn.blocks for i in bb.instructions]


def test_cvta_to_global_dropped_and_use_rewritten():
    mod = parse(CVTA_TO_GLOBAL_PTX)
    fn = mod.functions[0]
    dropped = run_function(fn)
    assert dropped == 1
    ops = _ops(fn)
    assert "cvta" not in ops
    # The store's MemOp base must now be %rd0 (the original source).
    st = next(i for i in _instructions(fn) if i.op == "st")
    assert st.srcs[0].base == "%rd0"


def test_cvta_global_form_dropped():
    mod = parse(CVTA_GLOBAL_PTX)
    fn = mod.functions[0]
    dropped = run_function(fn)
    assert dropped == 1
    assert "cvta" not in _ops(fn)
    st = next(i for i in _instructions(fn) if i.op == "st")
    assert st.srcs[0].base == "%rd0"


def test_multi_cvta_same_src_all_aliased():
    mod = parse(MULTI_CVTA_SAME_SRC_PTX)
    fn = mod.functions[0]
    dropped = run_function(fn)
    assert dropped == 3
    assert "cvta" not in _ops(fn)
    bases = [i.srcs[0].base for i in _instructions(fn) if i.op == "st"]
    assert bases == ["%rd0", "%rd0", "%rd0"]


def test_predicated_cvta_skipped():
    mod = parse(PREDICATED_CVTA_PTX)
    fn = mod.functions[0]
    dropped = run_function(fn)
    assert dropped == 0
    # cvta is still there
    assert "cvta" in _ops(fn)
    # store's base is still %rd1 (untouched because cvta wasn't dropped)
    st = next(i for i in _instructions(fn) if i.op == "st")
    assert st.srcs[0].base == "%rd1"


def test_transitive_cvta_chain_collapses_to_root():
    mod = parse(CVTA_TRANSITIVE_PTX)
    fn = mod.functions[0]
    dropped = run_function(fn)
    assert dropped == 2
    assert "cvta" not in _ops(fn)
    st = next(i for i in _instructions(fn) if i.op == "st")
    # Transitive chain %rd2 -> %rd1 -> %rd0 should resolve to %rd0.
    assert st.srcs[0].base == "%rd0"
