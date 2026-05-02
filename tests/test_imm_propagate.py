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


SUB_FOLDED_PTX = """\
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


def test_sub_at_pos1_is_folded():
    """Phase 8: sub at position 1 (subtrahend) is folded.  IADD3.IMM
    consumer NOPs are absorbed by the _SCHED_FORWARDING_SAFE
    (0x810, ...) promotions in sass/schedule.py."""
    mod = parse(SUB_FOLDED_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 1
    assert not list(_find(fn, "mov", "u32")), "mov should be DCE'd"
    subs = list(_find(fn, "sub", "u32"))
    assert len(subs) == 1
    assert isinstance(subs[0].srcs[1], ImmOp) and subs[0].srcs[1].value == 16


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


XOR_FOLDED_PTX = """\
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


def test_xor_at_pos1_is_folded():
    """Phase 8: xor at position 1 is folded into LOP3.IMM (opcode
    0x812).  The opex_4 collision was closed by remapping invalid
    misc values for 0x812 in sass/scoreboard.py."""
    mod = parse(XOR_FOLDED_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 1
    assert not list(_find(fn, "mov", "u32")), "mov should be DCE'd"
    xors = list(_find(fn, "xor", "b32"))
    assert isinstance(xors[0].srcs[1], ImmOp)
    assert xors[0].srcs[1].value == 1779033703


U32_ADD_FOLDED_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<5>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, 100;
    add.u32 %r3, %r1, %r2;
    st.global.u32 [%rd1], %r3;
    ret;
}
"""


def test_add_at_pos1_is_folded():
    """Phase 8: add at position 1 (second source) is folded for u32.
    The IADD3.IMM consumer NOP gap was closed by the _SCHED_FORWARDING_SAFE
    promotions in sass/schedule.py."""
    mod = parse(U32_ADD_FOLDED_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 1
    assert not list(_find(fn, "mov", "u32")), "mov should be DCE'd"
    adds = list(_find(fn, "add", "u32"))
    assert len(adds) == 1
    assert isinstance(adds[0].srcs[1], ImmOp) and adds[0].srcs[1].value == 100


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


# ---------------------------------------------------------------------------
# Phase 27: pos-0 fold for and/or/xor/add + binary-constant evaluation
# ---------------------------------------------------------------------------

XOR_BOTH_CONST_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<8>;
    ld.param.u64 %rd1, [a];
    mov.u32 %r2, 5;
    mov.u32 %r3, 7;
    xor.b32 %r4, %r2, %r3;
    st.global.u32 [%rd1], %r4;
    ret;
}
"""


def test_xor_both_constants_collapses_to_mov():
    """Phase 27 hero case: `mov %a, IV_A; mov %b, IV_B; xor %d, %a, %b`
    folds both srcs to IMM, then const-eval rewrites to a single
    `mov %d, IV_A^IV_B`.  Mirrors merkle's Blake2 IV-XOR pattern."""
    mod = parse(XOR_BOTH_CONST_PTX)
    fn = mod.functions[0]
    run_function(fn)
    # The xor should be gone (rewritten to mov by const-eval).
    assert not list(_find(fn, "xor", "b32")), "xor should be const-eval'd away"
    # Two old movs (5 and 7) DCE'd; one new mov with 5^7=2.
    movs = list(_find(fn, "mov", "b32"))
    assert len(movs) == 1, f"expected 1 mov (5^7=2), got {len(movs)}"
    assert isinstance(movs[0].srcs[0], ImmOp) and movs[0].srcs[0].value == 2


AND_HIGHLOW_BYTES_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<5>;
    ld.param.u64 %rd1, [a];
    mov.u32 %r2, 65280;
    and.b32 %r3, %r2, 255;
    st.global.u32 [%rd1], %r3;
    ret;
}
"""


def test_and_pos0_fold_then_const_eval():
    """`mov %a, 0xff00; and.b32 %d, %a, 0xff` — pos-0 fold makes both
    srcs IMM, const-eval reduces to `mov %d, 0`."""
    mod = parse(AND_HIGHLOW_BYTES_PTX)
    fn = mod.functions[0]
    run_function(fn)
    assert not list(_find(fn, "and", "b32")), "and should be const-eval'd"
    movs = list(_find(fn, "mov", "b32"))
    assert len(movs) == 1
    assert isinstance(movs[0].srcs[0], ImmOp) and movs[0].srcs[0].value == 0


OR_POS0_REG_POS1_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a, .param .u64 b)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<5>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, 4096;
    or.b32 %r3, %r2, %r1;
    ld.param.u64 %rd2, [b];
    st.global.u32 [%rd2], %r3;
    ret;
}
"""


def test_or_pos0_fold_only_when_pos1_is_runtime_reg():
    """Asymmetric case: pos 0 IMM, pos 1 is a runtime register.  Phase
    27 still folds pos 0 (no const-eval since pos 1 isn't IMM); the
    isel handler at sass/isel.py:3809 calls _materialize_imm on
    srcs[0], so encoding stays valid."""
    mod = parse(OR_POS0_REG_POS1_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 1, f"expected 1 substitution (pos-0 fold), got {n}"
    assert not list(_find(fn, "mov", "u32")), "mov should be DCE'd"
    ors = list(_find(fn, "or", "b32"))
    assert len(ors) == 1
    # Pos 0 is now IMM, pos 1 stays as register.
    assert isinstance(ors[0].srcs[0], ImmOp) and ors[0].srcs[0].value == 4096
    assert isinstance(ors[0].srcs[1], RegOp)


ADD_BOTH_CONST_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<8>;
    ld.param.u64 %rd1, [a];
    mov.u32 %r2, 100;
    mov.u32 %r3, 250;
    add.u32 %r4, %r2, %r3;
    st.global.u32 [%rd1], %r4;
    ret;
}
"""


def test_add_both_constants_collapses_to_mov():
    """Both operands of add are foldable movs → const-eval to single
    `mov %d, 350`.  Phase 27 added pos-0 fold for add at width <= 32."""
    mod = parse(ADD_BOTH_CONST_PTX)
    fn = mod.functions[0]
    run_function(fn)
    assert not list(_find(fn, "add", "u32")), "add should be const-eval'd"
    movs = list(_find(fn, "mov", "u32"))
    assert len(movs) == 1
    assert isinstance(movs[0].srcs[0], ImmOp) and movs[0].srcs[0].value == 350


XOR_REVERSE_ORDER_PTX = """\
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
    xor.b32 %r3, %r2, %r1;
    st.global.u32 [%rd1], %r3;
    ret;
}
"""


def test_xor_pos0_fold_with_runtime_pos1():
    """Merkle's exact pattern: `mov %a, IV; xor %d, %a, %loaded`.
    Pos-0 fold collapses the explicit MOV.IMM, leaving the IV inline
    at position 0 of the xor; isel materializes it transparently."""
    mod = parse(XOR_REVERSE_ORDER_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 1, f"expected 1 substitution (pos-0 fold), got {n}"
    assert not list(_find(fn, "mov", "u32")), "mov should be DCE'd"
    xors = list(_find(fn, "xor", "b32"))
    assert len(xors) == 1
    assert isinstance(xors[0].srcs[0], ImmOp)
    assert xors[0].srcs[0].value == 1779033703
    assert isinstance(xors[0].srcs[1], RegOp)


CHAINED_XOR_CONST_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<10>;
    ld.param.u64 %rd1, [a];
    mov.u32 %r2, 5;
    mov.u32 %r3, 7;
    mov.u32 %r4, 11;
    xor.b32 %r5, %r2, %r3;
    xor.b32 %r6, %r5, %r4;
    st.global.u32 [%rd1], %r6;
    ret;
}
"""


def test_chained_xor_const_via_fixpoint():
    """Three-deep const xor: requires the fixpoint loop because the
    second xor's pos-0 (%r5) only becomes IMM after the first xor
    is const-eval'd into a mov."""
    mod = parse(CHAINED_XOR_CONST_PTX)
    fn = mod.functions[0]
    run_function(fn)
    assert not list(_find(fn, "xor", "b32")), "all xors should be const-eval'd"
    movs = list(_find(fn, "mov", "b32"))
    assert len(movs) == 1
    expected = (5 ^ 7) ^ 11
    assert movs[0].srcs[0].value == expected


U64_AND_NOT_FOLDED_AT_POS0_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [a];
    ld.global.u64 %rd2, [%rd1];
    mov.b64 %rd3, 0xff00ff00ff00ff00;
    and.b64 %rd4, %rd3, %rd2;
    st.global.u64 [%rd1], %rd4;
    ret;
}
"""


def test_u64_and_pos0_not_folded():
    """64-bit and/or/xor pos-0 fold is deliberately disallowed because
    the u64 isel handler reads `srcs[0].name` without an ImmOp branch.
    The mov must survive."""
    mod = parse(U64_AND_NOT_FOLDED_AT_POS0_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0, "u64 and pos-0 must not fold"
    movs = list(_find(fn, "mov", "b64"))
    assert len(movs) == 1, "mov.b64 must survive (pos-0 fold blocked at >32)"


U64_ADD_NOT_FOLDED_AT_POS0_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [a];
    ld.global.u64 %rd2, [%rd1];
    mov.u64 %rd3, 16;
    add.u64 %rd4, %rd3, %rd2;
    st.global.u64 [%rd1], %rd4;
    ret;
}
"""


def test_u64_add_pos0_not_folded():
    """64-bit add pos-0 fold is disallowed: _select_add_u64 raises
    ISelError when srcs[0]=ImmOp.  The mov must survive."""
    mod = parse(U64_ADD_NOT_FOLDED_AT_POS0_PTX)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0, "u64 add pos-0 must not fold"
    movs = list(_find(fn, "mov", "u64"))
    assert len(movs) == 1, "mov.u64 must survive (pos-0 fold blocked at >32)"


B32_BITS_TYPE_FOLDS_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .b32 %r<5>;
    ld.param.u64 %rd1, [a];
    mov.b32 %r1, 0xff00;
    mov.b32 %r2, 0x00ff;
    or.b32 %r3, %r1, %r2;
    st.global.b32 [%rd1], %r3;
    ret;
}
"""


def test_b32_bits_type_or_const_eval():
    """b32 (bits-typed) or with both-const inputs: width matches,
    fold + const-eval applies the same as u32."""
    mod = parse(B32_BITS_TYPE_FOLDS_PTX)
    fn = mod.functions[0]
    run_function(fn)
    assert not list(_find(fn, "or", "b32")), "or should be const-eval'd"
    movs = list(_find(fn, "mov", "b32"))
    assert len(movs) == 1
    assert movs[0].srcs[0].value == 0xffff


ADD_OVERFLOW_WRAP_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{
    .reg .b64 %rd<5>;
    .reg .u32 %r<5>;
    ld.param.u64 %rd1, [a];
    mov.u32 %r2, 4294967295;
    mov.u32 %r3, 5;
    add.u32 %r4, %r2, %r3;
    st.global.u32 [%rd1], %r4;
    ret;
}
"""


def test_add_const_eval_wraps_at_width():
    """Add of two constants that overflow u32: result must wrap mod
    2^32.  0xFFFFFFFF + 5 = 4 (mod 2^32)."""
    mod = parse(ADD_OVERFLOW_WRAP_PTX)
    fn = mod.functions[0]
    run_function(fn)
    assert not list(_find(fn, "add", "u32"))
    movs = list(_find(fn, "mov", "u32"))
    assert len(movs) == 1
    assert movs[0].srcs[0].value == 4


PREDICATED_XOR_PTX = """\
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
    mov.u32 %r2, 5;
    mov.u32 %r3, 7;
    @%p1 xor.b32 %r4, %r2, %r3;
    st.global.u32 [%rd1], %r4;
    ret;
}
"""


def test_predicated_consumer_const_eval_skipped():
    """Const-eval skips predicated instructions (the rewrite-to-mov
    would change the side-effect shape)."""
    mod = parse(PREDICATED_XOR_PTX)
    fn = mod.functions[0]
    run_function(fn)
    # Pos-0 and pos-1 fold still happen (they're safe under predicates).
    xors = list(_find(fn, "xor", "b32"))
    assert len(xors) == 1, "predicated xor must remain (const-eval skipped)"
    # Both srcs may be ImmOp after fold, but op stays as xor (not mov).
    assert xors[0].pred is not None
