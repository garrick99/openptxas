"""
Strength reduction for repeated reg-reg adds: collapse
`add.<t> %r1, %r1, %r2` × N (same %r1, same %r2) into a single
`mad.lo.<t> %r1, %r2, N, %r1`.

Designed to run AFTER loop unrolling and AFTER imm_add_fold, where
the unroller has produced N copies of an accumulator update that
const-fold can't reach (because the addend is a register, not an
immediate).

Pattern (4 adds → 1 mad):

    add.u32 %r2, %r2, %r0
    add.u32 %r2, %r2, %r0
    add.u32 %r2, %r2, %r0
    add.u32 %r2, %r2, %r0

becomes:

    mad.lo.u32 %r2, %r0, 4, %r2

Conservative gating:
  - Same op (`add` only)
  - Same dest = same self-source (accumulator pattern)
  - Same other-source register name across the chain
  - Same scalar type (u32/s32/u64/s64 only — float rounding is too
    different to safely strength-reduce here)
  - Same predicate (name + neg flag)
  - No mods
  - The other-source register `%r2` must NOT be written between the
    adds (would change the multiplicand mid-chain)
  - The accumulator register `%r1` must NOT be read or written by
    other instructions between the adds (would observe an
    intermediate sum)

Operates in-place per BasicBlock.
"""
from __future__ import annotations

from typing import Optional

from ..ir import Function, ImmOp, Instruction, RegOp


# Integer types where `mad.lo.<t>` has the same semantics as a chain
# of `add.<t>` (modular arithmetic). Float types are excluded — the
# corresponding fma instruction has different rounding from a sequence
# of adds, so this transform would change observable results.
_MAD_TYPES = {"u32", "s32", "u64", "s64"}


def _is_eligible_add(inst: Instruction) -> Optional[tuple[str, str, str, tuple[str, ...]]]:
    """Return (acc_reg, addend_reg, type_key, mods_key) if `inst` is
    `add.t %acc, %acc, %addend` with type in _MAD_TYPES; else None.
    """
    if inst.op != "add":
        return None
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return None
    if not inst.types or inst.types[0] not in _MAD_TYPES:
        return None
    if len(inst.srcs) != 2:
        return None
    src0, src1 = inst.srcs
    if not isinstance(src0, RegOp) or src0.name != inst.dest.name:
        return None
    if not isinstance(src1, RegOp):
        return None
    if src1.name == inst.dest.name:
        return None  # add %r,%r,%r — degenerate; skip
    if inst.mods:
        return None
    return (inst.dest.name, src1.name, inst.types[0], tuple(inst.mods))


def _reads_or_writes(inst: Instruction, reg_name: str) -> bool:
    if inst.dest is not None and isinstance(inst.dest, RegOp) and inst.dest.name == reg_name:
        return True
    for src in inst.srcs:
        if isinstance(src, RegOp) and src.name == reg_name:
            return True
    return False


def _writes(inst: Instruction, reg_name: str) -> bool:
    return (inst.dest is not None
            and isinstance(inst.dest, RegOp)
            and inst.dest.name == reg_name)


def _reduce_block(instructions: list[Instruction]) -> int:
    """Collapse repeated reg-reg-add chains in a single block.
    Returns the number of chains converted to `mad.lo`.
    """
    n_reduced = 0
    i = 0
    while i < len(instructions):
        head = instructions[i]
        key = _is_eligible_add(head)
        if key is None:
            i += 1
            continue
        acc_reg, addend_reg, type_key, _ = key
        head_pred = head.pred
        head_neg = head.neg
        chain_indices = [i]

        j = i + 1
        while j < len(instructions):
            cand = instructions[j]
            cand_key = _is_eligible_add(cand)
            if (cand_key is not None
                    and cand_key[0] == acc_reg
                    and cand_key[1] == addend_reg
                    and cand_key[2] == type_key
                    and cand.pred == head_pred
                    and cand.neg  == head_neg):
                chain_indices.append(j)
                j += 1
                continue
            # Bail-out: any write to acc_reg or addend_reg, or any read
            # of acc_reg, breaks the chain. Reads of addend_reg are
            # fine (mad reads it once anyway).
            if _writes(cand, acc_reg) or _writes(cand, addend_reg):
                break
            # acc_reg used in a non-add (read) — the intermediate sum
            # would be observable, so stop the chain.
            for src in cand.srcs:
                if isinstance(src, RegOp) and src.name == acc_reg:
                    break
            else:
                # Independent — skip past.
                j += 1
                continue
            break

        if len(chain_indices) >= 2:
            # Replace head with a mad, drop the rest.
            n = len(chain_indices)
            head.op = "mad"
            head.types = ["lo", type_key]
            head.srcs = [
                RegOp(addend_reg),  # multiplicand a
                ImmOp(n),           # multiplicand b (the chain length)
                RegOp(acc_reg),     # addend c
            ]
            # head.dest stays as RegOp(acc_reg), pred/neg unchanged.
            for idx in reversed(chain_indices[1:]):
                del instructions[idx]
            n_reduced += 1
            i += 1
        else:
            i += 1
    return n_reduced


def run_function(fn: Function) -> int:
    total = 0
    for bb in fn.blocks:
        total += _reduce_block(bb.instructions)
    return total


def run(module) -> int:
    total = 0
    for fn in module.functions:
        total += run_function(fn)
    return total
