"""Tests for the PTX-IR copy_prop pass (Phase 17)."""

from ptx.parser import parse
from ptx.ir import ImmOp, RegOp
from ptx.passes.copy_prop import run_function


def _all_instrs(fn):
    for bb in fn.blocks:
        for inst in bb.instructions:
            yield inst


def _find(fn, op, types_first=None):
    for inst in _all_instrs(fn):
        if inst.op != op:
            continue
        if types_first is not None and (not inst.types or inst.types[0] != types_first):
            continue
        yield inst


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------

SIMPLE_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<10>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, %r1;
    add.u32 %r3, %r2, %r1;
    st.global.u32 [%rd1], %r3;
    ret;
}
"""


def test_simple_reg_reg_mov_substitutes_and_drops_mov():
    """`mov %r2, %r1; add %r3, %r2, %r1` -> `add %r3, %r1, %r1` and the
    mov is removed."""
    mod = parse(SIMPLE_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 1, f"expected 1 substitution, got {n}"
    # Mov should be dead.
    assert not list(_find(fn, "mov", "u32")), \
        "mov %r2, %r1 should have been DCE'd"
    adds = list(_find(fn, "add", "u32"))
    assert len(adds) == 1
    add = adds[0]
    assert isinstance(add.srcs[0], RegOp) and add.srcs[0].name == "%r1"
    assert isinstance(add.srcs[1], RegOp) and add.srcs[1].name == "%r1"


MULTI_USE_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<10>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    ld.global.u32 %r5, [%rd1+4];
    mov.u32 %r2, %r1;
    add.u32 %r3, %r2, %r5;
    xor.b32 %r4, %r2, %r5;
    st.global.u32 [%rd1], %r3;
    st.global.u32 [%rd1+8], %r4;
    ret;
}
"""


def test_multi_use_propagates_into_all_consumers_and_drops_mov():
    """`mov %r2, %r1; add ...; xor ...` — both uses substituted and
    mov dropped."""
    mod = parse(MULTI_USE_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 2, f"expected 2 substitutions, got {n}"
    assert not list(_find(fn, "mov", "u32")), \
        "mov should be DCE'd after both uses substituted"
    adds = list(_find(fn, "add", "u32"))
    xors = list(_find(fn, "xor", "b32"))
    assert len(adds) == 1 and len(xors) == 1
    assert isinstance(adds[0].srcs[0], RegOp) and adds[0].srcs[0].name == "%r1"
    assert isinstance(xors[0].srcs[0], RegOp) and xors[0].srcs[0].name == "%r1"


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------

REDEFINED_SRC_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<10>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, %r1;
    ld.global.u32 %r1, [%rd1+4];
    add.u32 %r3, %r2, %r1;
    st.global.u32 [%rd1], %r3;
    ret;
}
"""


def test_redefined_source_aborts_fold():
    """%r1 is redefined between `mov %r2, %r1` and the use of %r2 —
    propagation must skip this use; mov stays live."""
    mod = parse(REDEFINED_SRC_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0, f"expected 0 substitutions when source is redefined, got {n}"
    movs = list(_find(fn, "mov", "u32"))
    assert len(movs) == 1, "mov must survive (source was clobbered)"
    add = next(_find(fn, "add", "u32"))
    assert isinstance(add.srcs[0], RegOp) and add.srcs[0].name == "%r2"


STORE_ADDR_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<5>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    mov.b64 %rd3, %rd1;
    st.global.u32 [%rd3], %r1;
    ret;
}
"""


def test_store_address_position_is_skipped():
    """st.global.* is a memory op — copy-prop must skip its operand
    list entirely (its address operand has alignment / pair-encoding
    semantics that copy-prop can't validate)."""
    mod = parse(STORE_ADDR_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    # The mov is consumed only by st.global, which we skip.  No subs,
    # mov stays live.
    assert n == 0, f"store-address operand should not be substituted, got {n}"
    movs = list(_find(fn, "mov", "b64"))
    assert len(movs) == 1


PREDICATED_MOV_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<10>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    setp.eq.u32 %p1, %r1, 0;
    @%p1 mov.u32 %r2, %r1;
    add.u32 %r3, %r2, %r1;
    st.global.u32 [%rd1], %r3;
    ret;
}
"""


def test_predicated_mov_is_not_propagated():
    """A predicated mov may not execute on every path — propagation
    would change observable behavior."""
    mod = parse(PREDICATED_MOV_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0, "predicated mov must not be propagated"
    movs = list(_find(fn, "mov", "u32"))
    assert len(movs) == 1


MULTI_DEF_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<10>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, %r1;
    add.u32 %r2, %r1, %r1;
    add.u32 %r3, %r2, %r1;
    st.global.u32 [%rd1], %r3;
    ret;
}
"""


def test_multi_def_dest_is_skipped():
    """%r2 has two definitions (mov and add) — copy_prop must skip it."""
    mod = parse(MULTI_DEF_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0, "multi-def %d must not enter copy_defs"
    movs = list(_find(fn, "mov", "u32"))
    assert len(movs) == 1


WIDTH_MISMATCH_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<5>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, %r1;
    cvt.u64.u32 %rd3, %r2;
    st.global.u64 [%rd1], %rd3;
    ret;
}
"""


def test_width_mismatch_is_skipped():
    """A 32-bit mov source folded into a 64-bit consumer is unsafe —
    skip when widths don't match."""
    mod = parse(WIDTH_MISMATCH_PTX)
    fn = mod.functions[0]
    # The cvt's first type is the destination type (u64) which is the
    # consumer width.  mov is u32.  Widths differ -> skip.  But cvt's
    # second type is u32 — since we're checking types[0] only, this
    # also tests that we conservatively gate on types[0].
    n = run_function(fn)
    # cvt.u64.u32: types[0] is "u64", mov_width is 32 -> mismatch -> skip.
    assert n == 0, "width-mismatched fold must be skipped"
    movs = list(_find(fn, "mov", "u32"))
    assert len(movs) == 1


SPECIAL_REG_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<10>;
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %r1;
    add.u32 %r3, %r2, %r2;
    ld.param.u64 %rd1, [a];
    st.global.u32 [%rd1], %r3;
    ret;
}
"""


def test_special_reg_source_not_propagated_directly():
    """When the mov's source is %tid.x (a special PTX state reg),
    copy_prop must NOT substitute it into consumers — those need the
    S2R lowering with the original name.  The chained `mov %r2, %r1`
    where %r1 is a normal reg IS allowed."""
    mod = parse(SPECIAL_REG_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    # %r1 = mov %tid.x (skipped — special-reg source)
    # %r2 = mov %r1 — eligible.  add %r3, %r2, %r2 -> add %r3, %r1, %r1.
    # Both srcs of add substitute -> 2 substitutions, mov %r2 dropped.
    assert n == 2, f"expected 2 substitutions, got {n}"
    # mov %r1, %tid.x should still be present (special-reg, not a
    # candidate for the rewrite).
    movs = list(_find(fn, "mov", "u32"))
    assert len(movs) == 1
    assert isinstance(movs[0].srcs[0], RegOp) and movs[0].srcs[0].name == "%tid.x"
