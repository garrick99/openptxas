"""Tests for the PTX-IR imm_propagate pass."""

from ptx.parser import parse
from ptx.ir import ImmOp, RegOp
from ptx.passes.imm_propagate import run_function


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

SHL_HOT_PATH_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a, .param .u64 b)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<5>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, 32;
    shl.b32 %r3, %r1, %r2;
    ld.param.u64 %rd2, [b];
    st.global.u32 [%rd2], %r3;
    ret;
}
"""


def test_shl_hot_path_substitutes_and_drops_mov():
    """`mov %r, 32; shl.b32 %d, %s, %r` → `shl.b32 %d, %s, 32` and the
    mov is removed."""
    mod = parse(SHL_HOT_PATH_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 1, f"expected 1 substitution, got {n}"

    # Mov is dead and should be gone.
    assert not list(_find(fn, "mov", "u32")), \
        "mov %r2, 32 should have been removed"

    # The shl now has an ImmOp at position 1.
    shls = list(_find(fn, "shl", "b32"))
    assert len(shls) == 1
    assert isinstance(shls[0].srcs[1], ImmOp)
    assert shls[0].srcs[1].value == 32


SUB_NOT_FOLDED_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<5>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, 16;
    sub.u32 %r3, %r1, %r2;
    st.global.u32 [%rd1], %r3;
    ret;
}
"""


def test_sub_is_not_in_whitelist():
    """sub is excluded — folding causes scheduler-induced NOP bloat."""
    mod = parse(SUB_NOT_FOLDED_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0
    assert list(_find(fn, "mov", "u32")), "mov must survive"


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
    mov.u32 %r2, 8;
    shr.u32 %r3, %r1, %r2;
    shl.b32 %r4, %r3, %r2;
    st.global.u32 [%rd1], %r4;
    ret;
}
"""


def test_multi_use_propagates_into_all_consumers_and_drops_mov():
    """`mov %r, 8; shr ...; shl ...` — both shift counts substituted,
    mov dropped."""
    mod = parse(MULTI_USE_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 2, f"expected 2 substitutions, got {n}"
    assert not list(_find(fn, "mov", "u32")), \
        "mov should be dead after both uses substituted"
    shrs = list(_find(fn, "shr", "u32"))
    shls = list(_find(fn, "shl", "b32"))
    assert len(shrs) == 1 and len(shls) == 1
    assert isinstance(shrs[0].srcs[1], ImmOp) and shrs[0].srcs[1].value == 8
    assert isinstance(shls[0].srcs[1], ImmOp) and shls[0].srcs[1].value == 8


XOR_NOT_FOLDED_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<5>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, 1779033703;
    xor.b32 %r3, %r1, %r2;
    st.global.u32 [%rd1], %r3;
    ret;
}
"""


def test_xor_is_not_in_whitelist():
    """and/or/xor are intentionally excluded from the whitelist
    (LOP3.IMM scheduler-ctrl issue).  The mov must survive."""
    mod = parse(XOR_NOT_FOLDED_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0
    assert list(_find(fn, "mov", "u32")), "mov must survive"
    xors = list(_find(fn, "xor", "b32"))
    assert isinstance(xors[0].srcs[1], RegOp)


U64_ADD_NOT_FOLDED_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<10>;
    .reg .u32 %r<5>;
    ld.param.u64 %rd1, [a];
    ld.global.u64 %rd2, [%rd1];
    mov.u64 %rd3, 4;
    add.u64 %rd4, %rd2, %rd3;
    st.global.u64 [%rd1], %rd4;
    ret;
}
"""


def test_add_is_not_in_whitelist():
    """add is excluded — narrow whitelist limits to shl/shr only."""
    mod = parse(U64_ADD_NOT_FOLDED_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0
    assert list(_find(fn, "mov", "u64")), "mov must survive"


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------

STORE_VALUE_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<5>;
    ld.param.u64 %rd1, [a];
    mov.u32 %r2, 42;
    st.global.u32 [%rd1], %r2;
    ret;
}
"""


def test_store_value_position_is_not_folded():
    """Stores aren't in the whitelist; mov must survive."""
    mod = parse(STORE_VALUE_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0, f"expected 0 substitutions, got {n}"
    movs = list(_find(fn, "mov", "u32"))
    assert len(movs) == 1, "mov should survive (its only use is a store)"
    assert isinstance(movs[0].srcs[0], ImmOp)


REDEFINED_REG_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<5>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, 5;
    add.u32 %r2, %r2, %r1;
    shl.b32 %r3, %r1, %r2;
    st.global.u32 [%rd1], %r3;
    ret;
}
"""


def test_redefined_reg_aborts_fold():
    """%r2 is written twice (mov then add) — propagation must skip it.
    The pass should leave both writes intact and not substitute."""
    mod = parse(REDEFINED_REG_PTX)
    fn = mod.functions[0]
    # Capture the shl's pre-pass second source.
    shl = next(_find(fn, "shl", "b32"))
    assert isinstance(shl.srcs[1], RegOp) and shl.srcs[1].name == "%r2"

    n = run_function(fn)
    assert n == 0, f"expected 0 substitutions for redefined reg, got {n}"

    # mov must survive (we didn't propagate so it's still live) and
    # the shl's second source must still be the register, not an imm.
    movs = list(_find(fn, "mov", "u32"))
    assert len(movs) == 1
    shl = next(_find(fn, "shl", "b32"))
    assert isinstance(shl.srcs[1], RegOp) and shl.srcs[1].name == "%r2"


PREDICATED_MOV_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<5>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    setp.eq.u32 %p1, %r1, 0;
    @%p1 mov.u32 %r2, 32;
    shl.b32 %r3, %r1, %r2;
    st.global.u32 [%rd1], %r3;
    ret;
}
"""


def test_predicated_mov_is_not_propagated():
    mod = parse(PREDICATED_MOV_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0, "predicated mov must not be propagated"
    # mov stays.
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
    ld.global.u64 %rd2, [%rd1];
    mov.u64 %rd3, 4;
    shr.u64 %rd4, %rd2, %rd3;
    st.global.u64 [%rd1], %rd4;
    ret;
}
"""


def test_width_mismatch_aborts_fold():
    """`mov.u64 %r, 4` consumed by `shr.u64` shift count.  PTX shift
    count is u32 (32-bit), but mov is u64 (64-bit) — widths do not
    match, no fold."""
    mod = parse(WIDTH_MISMATCH_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0, "u64 mov must not fold into shift-count position"
    assert list(_find(fn, "mov", "u64")), "mov must survive"
