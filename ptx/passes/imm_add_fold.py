"""
Imm-add chain fold at the PTX IR level.

Collapses runs of `add.<t> %r, %r, IMM_k` (same dest, same self-source,
type-compatible) into a single `add.<t> %r, %r, sum(IMM_k)`. Designed
to run AFTER loop unrolling, where the unroller leaves N copies of each
counter increment that ptxas would have const-folded.

Pattern (3+ instructions to be a clear win, 2 is also handled):

    add.u32 %r3, %r3, 1
    add.u32 %r3, %r3, 1
    add.u32 %r3, %r3, 1
    add.u32 %r3, %r3, 1

becomes:

    add.u32 %r3, %r3, 4

Conservative gating:
  - Same op (`add` only, not `sub`/`madd`/`mul`/etc.)
  - Same dest register and same self-source register (`add %r,%r,IMM`)
  - Same type list
  - Same predicate (predicate name + neg flag)
  - No mods (sat / wrap / etc.)
  - No instructions between the adds may read OR write `%r`
    (writes are obvious; reads would observe an intermediate value
    that no longer exists after folding)

Operates in-place on each BasicBlock; doesn't cross block boundaries.
"""
from __future__ import annotations

from typing import Optional

from ..ir import Function, ImmOp, Instruction, RegOp


def _is_eligible_add(inst: Instruction) -> Optional[tuple[str, str, tuple[str, ...]]]:
    """Return (dest_name, types_key, mods_key) if `inst` is an
    `add.t %r, %r, IMM` we can fold across. Otherwise None.

    The returned key is what must match across instructions in the
    chain; predicate matches are checked separately by the caller.
    """
    if inst.op != "add":
        return None
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return None
    if len(inst.srcs) != 2:
        return None
    src0, src1 = inst.srcs
    if not isinstance(src0, RegOp) or src0.name != inst.dest.name:
        return None
    if not isinstance(src1, ImmOp):
        return None
    if inst.mods:
        return None
    return (inst.dest.name, ".".join(inst.types), tuple(inst.mods))


def _reads_or_writes(inst: Instruction, reg_name: str) -> bool:
    if inst.dest is not None and isinstance(inst.dest, RegOp) and inst.dest.name == reg_name:
        return True
    for src in inst.srcs:
        if isinstance(src, RegOp) and src.name == reg_name:
            return True
    return False


def _fold_block(instructions: list[Instruction]) -> int:
    """Fold imm-add chains within a single block. Returns number of
    folded chains. Mutates `instructions` in place.
    """
    n_folded = 0
    i = 0
    while i < len(instructions):
        head = instructions[i]
        key = _is_eligible_add(head)
        if key is None:
            i += 1
            continue

        head_pred = head.pred
        head_neg  = head.neg
        reg_name, types_key, _ = key
        chain_indices = [i]
        chain_sum = head.srcs[1].value

        # Walk forward gathering candidate adds. Allow other instructions
        # between as long as they don't read or write `reg_name`.
        j = i + 1
        while j < len(instructions):
            cand = instructions[j]
            cand_key = _is_eligible_add(cand)
            if (cand_key is not None
                    and cand_key[0] == reg_name
                    and cand_key[1] == types_key
                    and cand.pred == head_pred
                    and cand.neg  == head_neg):
                # Foldable.
                chain_indices.append(j)
                chain_sum += cand.srcs[1].value
                j += 1
                continue
            # Not foldable. If it touches `reg_name` we have to stop.
            if _reads_or_writes(cand, reg_name):
                break
            # Independent instruction — skip past it and keep looking.
            j += 1

        if len(chain_indices) >= 2:
            # Mutate the head to carry the summed immediate, then drop
            # the rest. Mask to the operand width so the immediate stays
            # legal (e.g. u32 add wraps at 2^32).
            width_bits: Optional[int] = None
            if head.types:
                t = head.types[0]
                if t.startswith(("u", "s", "b")):
                    try:
                        width_bits = int(t[1:])
                    except ValueError:
                        width_bits = None
            folded_imm = chain_sum
            if width_bits is not None:
                folded_imm = folded_imm & ((1 << width_bits) - 1)
            head.srcs[1] = ImmOp(folded_imm)
            # Remove the folded instructions in reverse order so indices
            # stay valid.
            for idx in reversed(chain_indices[1:]):
                del instructions[idx]
            n_folded += 1
            # Don't advance `i` — the next iteration starts at the new
            # instruction at position `i+1`, which may itself be a
            # candidate chain head.
            i += 1
        else:
            i += 1
    return n_folded


def run_function(fn: Function) -> int:
    """Fold imm-add chains in every block of `fn`. Returns total chains folded."""
    total = 0
    for bb in fn.blocks:
        total += _fold_block(bb.instructions)
    return total


def run(module) -> int:
    total = 0
    for fn in module.functions:
        total += run_function(fn)
    return total
