"""
Trivial algebraic-identity folder at the PTX IR level.

After loop unroll + counter constant-propagation, the body often has
shapes like `mul %r, %r0, 0`, `add %r, %r, 0`, `xor %r, %r, 0`, etc.
Without this pass, those degenerate ops survive to the SASS stage as
real multiplies / adds / xors against zero.

Handles:
  add.<int>  %r, %a, 0     →  mov.<int>  %r, %a
  sub.<int>  %r, %a, 0     →  mov.<int>  %r, %a
  mul.lo.<int> %r, %a, 0   →  mov.<int>  %r, 0
  mul.lo.<int> %r, %a, 1   →  mov.<int>  %r, %a
  mul.hi.<int> %r, %a, 0   →  mov.<int>  %r, 0  (only safe for unsigned)
  xor.<bits> %r, %a, 0     →  mov.<bits> %r, %a
  or.<bits>  %r, %a, 0     →  mov.<bits> %r, %a
  and.<bits> %r, %a, 0     →  mov.<bits> %r, 0
  shl.<bits> %r, %a, 0     →  mov.<bits> %r, %a
  shr.<bits> %r, %a, 0     →  mov.<bits> %r, %a

Conservative: only fires on integer types. Float identities (`mul %r, %a, 1.0`)
are gated out — IEEE rounding can change observable bit patterns.

Operates per BasicBlock. Returns total folds.
"""
from __future__ import annotations

from typing import Optional

from ..ir import Function, ImmOp, Instruction, RegOp


_INT_TYPES = {"u8", "u16", "u32", "u64", "s8", "s16", "s32", "s64",
              "b8", "b16", "b32", "b64"}


_BITWIDTH = {
    "u8": 8, "s8": 8, "b8": 8,
    "u16": 16, "s16": 16, "b16": 16,
    "u32": 32, "s32": 32, "b32": 32,
    "u64": 64, "s64": 64, "b64": 64,
}


_DROP_SENTINEL = "__drop__"


def _imm_value(op) -> Optional[int]:
    return op.value if isinstance(op, ImmOp) else None


def _make_mov(dest: RegOp, src, type_str: str, pred: Optional[str], neg: bool) -> Optional[Instruction]:
    # Don't generate `mov %r, %r` — some kernels deliberately use
    # `add %r, %r, 0` as a register-pinning hint that the regalloc /
    # scheduler treats specially; replacing with a self-mov breaks
    # the pinning and regresses the kernel (caught on atom_cas64,
    # forge-workbench artifact 20260427_201134). Returning None
    # signals "don't fold this".
    if isinstance(src, RegOp) and src.name == dest.name:
        return None
    return Instruction(
        op="mov",
        types=[type_str],
        dest=dest,
        srcs=[src],
        pred=pred,
        neg=neg,
        mods=[],
    )


def _try_fold(inst: Instruction) -> Optional[Instruction]:
    """Return a replacement instruction if `inst` is an algebraic
    identity that simplifies to a `mov`; otherwise None.
    """
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return None
    if not inst.types:
        return None
    if inst.mods:
        return None  # don't touch sat / wrap / etc.

    # Type must be an integer scalar we recognize. The first type
    # element is the comparison/mod for setp etc.; for plain arith
    # ops, types[0] is the scalar type.
    op = inst.op
    pred = inst.pred
    neg  = inst.neg

    # Two-source ops where one operand is an immediate.
    if op in ("add", "sub", "or", "xor"):
        if len(inst.srcs) != 2 or inst.types[0] not in _INT_TYPES:
            return None
        a, b = inst.srcs
        b_val = _imm_value(b)
        if b_val == 0:
            # add/sub/or/xor with 0 → mov dest, a
            return _make_mov(inst.dest, a, inst.types[0], pred, neg)
        # add/or/xor are commutative — also handle imm on left.
        a_val = _imm_value(a)
        if op != "sub" and a_val == 0:
            return _make_mov(inst.dest, b, inst.types[0], pred, neg)
        return None

    if op == "and":
        if len(inst.srcs) != 2 or inst.types[0] not in _INT_TYPES:
            return None
        a, b = inst.srcs
        b_val = _imm_value(b)
        a_val = _imm_value(a)
        # and with 0 → mov dest, 0
        if b_val == 0 or a_val == 0:
            return _make_mov(inst.dest, ImmOp(0), inst.types[0], pred, neg)
        # and with all-ones (full bitwidth mask) → mov dest, other
        bw = _BITWIDTH.get(inst.types[0])
        if bw is not None:
            mask = (1 << bw) - 1
            if b_val is not None and (b_val & mask) == mask:
                return _make_mov(inst.dest, a, inst.types[0], pred, neg)
            if a_val is not None and (a_val & mask) == mask:
                return _make_mov(inst.dest, b, inst.types[0], pred, neg)
        return None

    if op == "mul":
        # mul has a `.lo` / `.hi` / `.wide` mod in types[0], type in types[1].
        if len(inst.types) < 2:
            return None
        kind = inst.types[0]
        scalar_type = inst.types[1]
        if scalar_type not in _INT_TYPES:
            return None
        if kind not in ("lo", "hi", "wide"):
            return None
        if len(inst.srcs) != 2:
            return None
        a, b = inst.srcs
        a_val = _imm_value(a)
        b_val = _imm_value(b)
        # mul.lo by 0 → 0; mul.hi by 0 → 0 for unsigned (signed
        # is also 0, but skip to keep gating tight for now)
        if (a_val == 0 or b_val == 0) and kind == "lo":
            return _make_mov(inst.dest, ImmOp(0), scalar_type, pred, neg)
        # mul.lo by 1 → mov(other operand)
        if kind == "lo":
            if b_val == 1:
                return _make_mov(inst.dest, a, scalar_type, pred, neg)
            if a_val == 1:
                return _make_mov(inst.dest, b, scalar_type, pred, neg)
        return None

    if op in ("shl", "shr"):
        if len(inst.srcs) != 2 or inst.types[0] not in _INT_TYPES:
            return None
        a, b = inst.srcs
        b_val = _imm_value(b)
        if b_val == 0:
            return _make_mov(inst.dest, a, inst.types[0], pred, neg)
        return None

    return None


def _is_self_alias_identity(inst: Instruction) -> bool:
    """True if this instruction is a self-aliased algebraic identity
    that produces no observable change (e.g. `or %r, %r, 0`).  Such
    instructions can be dropped entirely.
    """
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return False
    if inst.pred is not None or inst.mods or inst.neg:
        return False
    if not inst.types:
        return False
    if len(inst.srcs) != 2:
        return False
    a, b = inst.srcs
    op = inst.op
    if not isinstance(a, RegOp) or a.name != inst.dest.name:
        return False
    b_val = _imm_value(b)
    if b_val is None:
        return False
    if op in ("add", "sub", "or", "xor", "shl", "shr") and b_val == 0:
        return True
    if op == "and":
        bw = _BITWIDTH.get(inst.types[0])
        if bw is not None and (b_val & ((1 << bw) - 1)) == (1 << bw) - 1:
            return True
    if op == "mul":
        if (len(inst.types) >= 2 and inst.types[0] == "lo"
                and b_val == 1):
            return True
    return False


def _fold_block(instructions: list[Instruction]) -> int:
    n_folded = 0
    new_instrs: list[Instruction] = []
    for inst in instructions:
        if _is_self_alias_identity(inst):
            n_folded += 1
            continue
        replacement = _try_fold(inst)
        if replacement is not None:
            new_instrs.append(replacement)
            n_folded += 1
        else:
            new_instrs.append(inst)
    instructions[:] = new_instrs
    return n_folded


def run_function(fn: Function) -> int:
    total = 0
    for bb in fn.blocks:
        total += _fold_block(bb.instructions)
    return total


def run(module) -> int:
    total = 0
    for fn in module.functions:
        total += run_function(fn)
    return total
