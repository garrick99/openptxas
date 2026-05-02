"""
Tests for the 32-bit rotate-fusion pass (Phase 11).

Mirrors tests/test_rotate_pass.py for the 32-bit case.  The pass
recognizes (shr.u32 + shl.b32 + or/add/xor.b32) triples that constitute
a 32-bit rotate emulation and rewrites them to a single synthetic
`rot.b32 %dst, %src, ImmOp(K_left)` instruction.

The same three semantic invariants of the 64-bit pass apply:
  - OP ∈ {add, or, xor}    (not sub — Bug 1 invariant)
  - shr is LOGICAL (shr.u32) — Bug 2 invariant
  - same source register for both shifts
"""

from ptx.ir import (
    Instruction, RegOp, ImmOp, BasicBlock, Function,
)
from ptx.passes.rotate32 import run_function as rotate32_run


def _make_shl_b32(dest: str, src: str, k: int) -> Instruction:
    return Instruction(op="shl", types=["b32"],
                       dest=RegOp(dest), srcs=[RegOp(src), ImmOp(k)])

def _make_shr_u32(dest: str, src: str, k: int) -> Instruction:
    return Instruction(op="shr", types=["u32"],
                       dest=RegOp(dest), srcs=[RegOp(src), ImmOp(k)])

def _make_shr_s32(dest: str, src: str, k: int) -> Instruction:
    return Instruction(op="shr", types=["s32"],
                       dest=RegOp(dest), srcs=[RegOp(src), ImmOp(k)])

def _make_combine_b32(op: str, dest: str, a: str, b: str) -> Instruction:
    return Instruction(op=op, types=["b32"],
                       dest=RegOp(dest), srcs=[RegOp(a), RegOp(b)])

def _make_fn(insts: list[Instruction]) -> Function:
    bb = BasicBlock(label=None, instructions=insts)
    return Function(name="test", is_kernel=True, blocks=[bb])


# ---------------------------------------------------------------------------
# Positive cases — pattern matched, fused to single `rot.b32`
# ---------------------------------------------------------------------------

def test_or_b32_matches_and_fuses():
    """shr.u32 + shl.b32 + or.b32 with K1+K2=32 → rot.b32."""
    insts = [
        _make_shr_u32("%a", "%x", 16),
        _make_shl_b32("%b", "%x", 16),
        _make_combine_b32("or", "%c", "%a", "%b"),
    ]
    fn = _make_fn(insts)
    n = rotate32_run(fn)
    assert n == 1
    bb = fn.blocks[0]
    # Triple collapses to a single instruction (shr/shl deleted, or rewritten).
    assert len(bb.instructions) == 1
    rot = bb.instructions[0]
    assert rot.op == "rot"
    assert rot.types == ["b32"]
    assert isinstance(rot.dest, RegOp) and rot.dest.name == "%c"
    assert isinstance(rot.srcs[0], RegOp) and rot.srcs[0].name == "%x"
    assert isinstance(rot.srcs[1], ImmOp) and rot.srcs[1].value == 16  # K_left = shl K


def test_xor_b32_matches():
    insts = [
        _make_shr_u32("%a", "%x", 8),
        _make_shl_b32("%b", "%x", 24),
        _make_combine_b32("xor", "%c", "%a", "%b"),
    ]
    fn = _make_fn(insts)
    assert rotate32_run(fn) == 1
    rot = fn.blocks[0].instructions[0]
    assert rot.op == "rot"
    assert rot.srcs[1].value == 24  # shl K


def test_add_b32_matches():
    insts = [
        _make_shr_u32("%a", "%x", 1),
        _make_shl_b32("%b", "%x", 31),
        _make_combine_b32("add", "%c", "%a", "%b"),
    ]
    fn = _make_fn(insts)
    assert rotate32_run(fn) == 1
    assert fn.blocks[0].instructions[0].op == "rot"


def test_all_k_values_match():
    """Every K in 1..31 should fuse with K2 = 32 - K."""
    for k in range(1, 32):
        insts = [
            _make_shr_u32("%a", "%x", k),
            _make_shl_b32("%b", "%x", 32 - k),
            _make_combine_b32("or", "%c", "%a", "%b"),
        ]
        fn = _make_fn(insts)
        assert rotate32_run(fn) == 1, f"k={k} should fuse"
        rot = fn.blocks[0].instructions[0]
        assert rot.srcs[1].value == 32 - k


def test_combine_operand_order_swapped():
    """combine(%shl_dest, %shr_dest) — operand order should not matter for or/add/xor."""
    insts = [
        _make_shr_u32("%a", "%x", 16),
        _make_shl_b32("%b", "%x", 16),
        _make_combine_b32("or", "%c", "%b", "%a"),  # swapped order
    ]
    fn = _make_fn(insts)
    assert rotate32_run(fn) == 1


def test_u32_shl_type_also_matches():
    """shl.u32 (instead of shl.b32) is just as valid — both are 32-bit shifts."""
    insts = [
        _make_shr_u32("%a", "%x", 16),
        Instruction(op="shl", types=["u32"],
                    dest=RegOp("%b"), srcs=[RegOp("%x"), ImmOp(16)]),
        _make_combine_b32("or", "%c", "%a", "%b"),
    ]
    fn = _make_fn(insts)
    assert rotate32_run(fn) == 1


# ---------------------------------------------------------------------------
# Negative cases — must NOT match
# ---------------------------------------------------------------------------

def test_shr_s32_does_not_match():
    """Bug 2 invariant: arithmetic right shift is NOT a rotate primitive."""
    insts = [
        _make_shr_s32("%a", "%x", 16),    # SIGNED right shift
        _make_shl_b32("%b", "%x", 16),
        _make_combine_b32("or", "%c", "%a", "%b"),
    ]
    fn = _make_fn(insts)
    assert rotate32_run(fn) == 0
    # Triple is preserved.
    assert len(fn.blocks[0].instructions) == 3


def test_sub_does_not_match():
    """Bug 1 invariant: sub is NOT rotation-equivalent."""
    insts = [
        _make_shr_u32("%a", "%x", 16),
        _make_shl_b32("%b", "%x", 16),
        Instruction(op="sub", types=["s32"],
                    dest=RegOp("%c"), srcs=[RegOp("%a"), RegOp("%b")]),
    ]
    fn = _make_fn(insts)
    assert rotate32_run(fn) == 0


def test_different_sources_no_match():
    """(a >> K) | (b << (32-K)) — different sources, not a rotate of any one register."""
    insts = [
        _make_shr_u32("%a", "%x", 16),
        _make_shl_b32("%b", "%y", 16),    # different src
        _make_combine_b32("or", "%c", "%a", "%b"),
    ]
    fn = _make_fn(insts)
    assert rotate32_run(fn) == 0


def test_non_complementary_shifts_no_match():
    """K1 + K2 != 32 — algebraically not a rotate."""
    insts = [
        _make_shr_u32("%a", "%x", 8),
        _make_shl_b32("%b", "%x", 16),    # 8 + 16 = 24, not 32
        _make_combine_b32("or", "%c", "%a", "%b"),
    ]
    fn = _make_fn(insts)
    assert rotate32_run(fn) == 0


def test_runtime_shift_amount_no_match():
    """Variable (register-typed) shift amount cannot fuse."""
    insts = [
        Instruction(op="shr", types=["u32"],
                    dest=RegOp("%a"), srcs=[RegOp("%x"), RegOp("%k1")]),
        _make_shl_b32("%b", "%x", 16),
        _make_combine_b32("or", "%c", "%a", "%b"),
    ]
    fn = _make_fn(insts)
    assert rotate32_run(fn) == 0


def test_zero_shift_amount_no_match():
    """K=0 is degenerate (shr by 0 → identity, shl by 32 → zero in PTX)."""
    insts = [
        _make_shr_u32("%a", "%x", 0),
        _make_shl_b32("%b", "%x", 32),
        _make_combine_b32("or", "%c", "%a", "%b"),
    ]
    fn = _make_fn(insts)
    assert rotate32_run(fn) == 0


# ---------------------------------------------------------------------------
# Pass-level scenarios
# ---------------------------------------------------------------------------

def test_pass_finds_valid_rejects_buggy_in_one_block():
    """In a single block: 1 valid + 1 sub-buggy + 1 shr.s32-buggy → 1 fuse."""
    insts = [
        # Valid rotate (should fuse)
        _make_shr_u32("%a1", "%x", 16),
        _make_shl_b32("%b1", "%x", 16),
        _make_combine_b32("or", "%c1", "%a1", "%b1"),

        # Bug 1: sub (must NOT fuse)
        _make_shr_u32("%a2", "%y", 8),
        _make_shl_b32("%b2", "%y", 24),
        Instruction(op="sub", types=["s32"],
                    dest=RegOp("%c2"), srcs=[RegOp("%a2"), RegOp("%b2")]),

        # Bug 2: shr.s32 (must NOT fuse)
        _make_shr_s32("%a3", "%z", 4),
        _make_shl_b32("%b3", "%z", 28),
        _make_combine_b32("or", "%c3", "%a3", "%b3"),
    ]
    fn = _make_fn(insts)
    n = rotate32_run(fn)
    assert n == 1
    # Find the rewritten rot instruction.
    rot_insts = [i for i in fn.blocks[0].instructions if i.op == "rot"]
    assert len(rot_insts) == 1
    assert rot_insts[0].dest.name == "%c1"


def test_dead_shr_shl_removed():
    """After fuse, shl/shr instructions whose dest had no other readers are removed."""
    insts = [
        _make_shr_u32("%a", "%x", 16),
        _make_shl_b32("%b", "%x", 16),
        _make_combine_b32("or", "%c", "%a", "%b"),
    ]
    fn = _make_fn(insts)
    rotate32_run(fn)
    ops = [i.op for i in fn.blocks[0].instructions]
    assert "shr" not in ops
    assert "shl" not in ops
    assert ops == ["rot"]


def test_shr_dest_used_elsewhere_kept():
    """If the shr's dest has another reader beyond the combine, it must NOT be deleted."""
    insts = [
        _make_shr_u32("%a", "%x", 16),
        _make_shl_b32("%b", "%x", 16),
        _make_combine_b32("or", "%c", "%a", "%b"),
        # %a is also used here — must survive.
        Instruction(op="add", types=["u32"],
                    dest=RegOp("%d"), srcs=[RegOp("%a"), ImmOp(1)]),
    ]
    fn = _make_fn(insts)
    rotate32_run(fn)
    ops = [i.op for i in fn.blocks[0].instructions]
    # Combine still fuses (dest of fused rot is %c, %a flow is independent).
    assert "rot" in ops
    assert "shr" in ops  # %a producer is kept (still used by the trailing add)
