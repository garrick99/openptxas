"""
Tests for the rotate-left pattern recognizer.

Validates the three bugs ptxas has — our pass must:
  - NOT emit a rotate for sub (Bug 1)
  - NOT emit a rotate for shr.s64 (Bug 2)
  - Handle operand order correctly (Bug 3)
  - DO emit a rotate for add/or/xor + shr.u64 (correct case)
"""

from ptx.ir import (
    Instruction, RegOp, ImmOp, BasicBlock, Function, Module,
)
from ptx.passes.rotate import match_rotate, find_rotate_groups, run


def _make_shl(dest: str, src: str, k: int) -> Instruction:
    return Instruction(op="shl", types=["b64"],
                       dest=RegOp(dest), srcs=[RegOp(src), ImmOp(k)])

def _make_shr_u(dest: str, src: str, k: int) -> Instruction:
    return Instruction(op="shr", types=["u64"],
                       dest=RegOp(dest), srcs=[RegOp(src), ImmOp(k)])

def _make_shr_s(dest: str, src: str, k: int) -> Instruction:
    return Instruction(op="shr", types=["s64"],
                       dest=RegOp(dest), srcs=[RegOp(src), ImmOp(k)])

def _make_combine(op: str, dest: str, a: str, b: str) -> Instruction:
    return Instruction(op=op, types=["s64"],
                       dest=RegOp(dest), srcs=[RegOp(a), RegOp(b)])

def _make_fn(insts: list[Instruction]) -> Function:
    bb = BasicBlock(label=None, instructions=insts)
    return Function(name="test", is_kernel=True, blocks=[bb])


# ---------------------------------------------------------------------------
# Correct rotate cases (should MATCH)
# ---------------------------------------------------------------------------

def test_add_u64_matches():
    """add.s64 + shr.u64 = valid rotate-left."""
    shl = _make_shl("%lo", "%a", 8)
    shr = _make_shr_u("%hi", "%a", 56)
    add = _make_combine("add", "%res", "%lo", "%hi")
    grp = match_rotate(shl, shr, add)
    assert grp is not None
    assert grp.k == 8
    assert grp.src == "%a"

def test_or_u64_matches():
    shl = _make_shl("%lo", "%a", 32)
    shr = _make_shr_u("%hi", "%a", 32)
    combine = _make_combine("or", "%res", "%lo", "%hi")
    assert match_rotate(shl, shr, combine) is not None

def test_xor_u64_matches():
    shl = _make_shl("%lo", "%a", 1)
    shr = _make_shr_u("%hi", "%a", 63)
    combine = _make_combine("xor", "%res", "%lo", "%hi")
    assert match_rotate(shl, shr, combine) is not None

def test_all_k_values_match_for_add():
    for k in range(1, 64):
        shl = _make_shl("%lo", "%a", k)
        shr = _make_shr_u("%hi", "%a", 64 - k)
        add = _make_combine("add", "%res", "%lo", "%hi")
        grp = match_rotate(shl, shr, add)
        assert grp is not None, f"k={k} should match"
        assert grp.k == k


# ---------------------------------------------------------------------------
# Bug 1: sub.s64 must NOT match
# ---------------------------------------------------------------------------

def test_bug1_sub_does_not_match():
    """
    ptxas miscompiles (a<<K) - (a>>(64-K)) as a rotate.
    Our pass must reject it.
    """
    shl = _make_shl("%lo", "%a", 8)
    shr = _make_shr_u("%hi", "%a", 56)
    sub = _make_combine("sub", "%res", "%lo", "%hi")
    grp = match_rotate(shl, shr, sub)
    assert grp is None, "sub.s64 must NOT be recognized as rotate (Bug 1)"

def test_bug1_all_k_values_sub_rejected():
    for k in range(1, 64):
        shl = _make_shl("%lo", "%a", k)
        shr = _make_shr_u("%hi", "%a", 64 - k)
        sub = _make_combine("sub", "%res", "%lo", "%hi")
        assert match_rotate(shl, shr, sub) is None, f"k={k} sub should be rejected"


# ---------------------------------------------------------------------------
# Bug 2: shr.s64 (arithmetic) must NOT match
# ---------------------------------------------------------------------------

def test_bug2_signed_shr_does_not_match():
    """
    ptxas ignores shift signedness — applies rotate even for shr.s64.
    Our pass must reject it.
    """
    shl = _make_shl("%lo", "%a", 8)
    shr = _make_shr_s("%hi", "%a", 56)   # SIGNED shift
    add = _make_combine("add", "%res", "%lo", "%hi")
    grp = match_rotate(shl, shr, add)
    assert grp is None, "shr.s64 must NOT be recognized as rotate (Bug 2)"

def test_bug2_signed_shr_with_sub_also_rejected():
    shl = _make_shl("%lo", "%a", 8)
    shr = _make_shr_s("%hi", "%a", 56)
    sub = _make_combine("sub", "%res", "%lo", "%hi")
    assert match_rotate(shl, shr, sub) is None


# ---------------------------------------------------------------------------
# Bug 3: operand order / non-commutative ops
# ---------------------------------------------------------------------------

def test_bug3_reversed_sub_operands_rejected():
    """hi - lo is also miscompiled identically by ptxas."""
    shl = _make_shl("%lo", "%a", 8)
    shr = _make_shr_u("%hi", "%a", 56)
    sub = _make_combine("sub", "%res", "%hi", "%lo")  # reversed
    assert match_rotate(shl, shr, sub) is None


# ---------------------------------------------------------------------------
# Boundary / non-pattern cases (must NOT match)
# ---------------------------------------------------------------------------

def test_different_sources_no_match():
    """(a << K) OP (b >> (64-K)) — different source vars, not a rotate."""
    shl = _make_shl("%lo", "%a", 8)
    shr = _make_shr_u("%hi", "%b", 56)  # different source
    add = _make_combine("add", "%res", "%lo", "%hi")
    assert match_rotate(shl, shr, add) is None

def test_non_complementary_shifts_no_match():
    """(a << 3) + (a >> 5) — 3+5=8 ≠ 64, not a rotate."""
    shl = _make_shl("%lo", "%a", 3)
    shr = _make_shr_u("%hi", "%a", 5)
    add = _make_combine("add", "%res", "%lo", "%hi")
    assert match_rotate(shl, shr, add) is None

def test_runtime_k_no_match():
    """Runtime shift amount — can't be a peephole rotate."""
    shl = Instruction(op="shl", types=["b64"],
                      dest=RegOp("%lo"), srcs=[RegOp("%a"), RegOp("%k")])  # reg, not imm
    shr = _make_shr_u("%hi", "%a", 56)
    add = _make_combine("add", "%res", "%lo", "%hi")
    assert match_rotate(shl, shr, add) is None


# ---------------------------------------------------------------------------
# Pass-level test
# ---------------------------------------------------------------------------

def test_pass_finds_valid_and_rejects_buggy():
    """
    The find_rotate_groups pass should:
    - Find one valid rotate group (add + shr.u64)
    - Reject buggy patterns (sub + shr.u64, add + shr.s64)
    - Not include buggy patterns in returned groups
    """
    insts = [
        # Valid rotate
        _make_shl("%lo1", "%a", 8),
        _make_shr_u("%hi1", "%a", 56),
        _make_combine("add", "%res1", "%lo1", "%hi1"),

        # Bug 1: sub.s64 (ptxas would miscompile)
        _make_shl("%lo2", "%b", 8),
        _make_shr_u("%hi2", "%b", 56),
        _make_combine("sub", "%res2", "%lo2", "%hi2"),

        # Bug 2: shr.s64 (ptxas would miscompile)
        _make_shl("%lo3", "%c", 8),
        _make_shr_s("%hi3", "%c", 56),
        _make_combine("add", "%res3", "%lo3", "%hi3"),
    ]
    fn = _make_fn(insts)
    groups = find_rotate_groups(fn)

    assert len(groups) == 1, f"Expected 1 valid group, got {len(groups)}"
    assert groups[0].k == 8
    assert groups[0].src == "%a"
