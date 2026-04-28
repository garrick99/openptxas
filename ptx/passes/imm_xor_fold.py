"""
Imm-xor chain fold at the PTX IR level.

Collapses runs of `xor.<bits> %r, %r, K_i` (same dest, same self-source,
type-compatible) into a single `xor.<bits> %r, %r, XOR(K_0..K_{N-1})`.
If the XOR-sum is zero, drops the instruction entirely (the chain is
algebraically a no-op).

Designed to run AFTER loop unroll + per-iteration counter constant-
propagation, where the unroller produces N copies of `xor %r, %r, K_i`
with the counter values substituted in. Without this fold, those N
xors survive to SASS as N separate xor instructions.

For example, w1_loop_xor's body is `xor %r2, %r2, %r3` with counter
%r3 going 0..7. After unroll+const-prop: 8 `xor %r2, %r2, K_i` with
K_i = 0..7. XOR-sum = 0 (since 0^1^2^...^7 = 0), so this pass drops
the entire chain — the loop is algebraically a no-op for that
accumulator.

Conservative gating:
  - Same op (`xor` only)
  - Same dest = same self-source
  - Type must be `b32` / `b64` (xor is bitwise, signed/unsigned distinction
    doesn't matter at PTX level but type must match across chain)
  - Same predicate
  - No mods
  - Between members: any instruction is allowed AS LONG AS it doesn't
    read or write %r
"""
from __future__ import annotations

from typing import Optional

from ..ir import Function, ImmOp, Instruction, RegOp


_BIT_TYPES = {"b32", "b64", "u32", "s32", "u64", "s64"}


def _is_eligible_xor(inst: Instruction) -> Optional[tuple[str, str]]:
    """Return (dest_name, type_str) if `inst` is `xor.<t> %r, %r, IMM`;
    else None.
    """
    if inst.op != "xor":
        return None
    if inst.pred is not None or inst.mods:
        return None
    if not inst.types or inst.types[0] not in _BIT_TYPES:
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
    return (inst.dest.name, inst.types[0])


def _reads_or_writes(inst: Instruction, reg_name: str) -> bool:
    if inst.dest is not None and isinstance(inst.dest, RegOp) and inst.dest.name == reg_name:
        return True
    for src in inst.srcs:
        if isinstance(src, RegOp) and src.name == reg_name:
            return True
    return False


def _bitmask(t: str) -> int:
    if t in ("b32", "u32", "s32"):
        return 0xFFFFFFFF
    if t in ("b64", "u64", "s64"):
        return 0xFFFFFFFF_FFFFFFFF
    return 0xFFFFFFFF


def _fold_block(instructions: list[Instruction]) -> int:
    n_changed = 0
    skip = set()  # indices to drop entirely from the final list
    i = 0
    while i < len(instructions):
        if i in skip:
            i += 1
            continue
        head = instructions[i]
        key = _is_eligible_xor(head)
        if key is None:
            i += 1
            continue
        reg_name, type_str = key
        head_pred = head.pred
        head_neg = head.neg
        chain_indices = [i]
        chain_xor = head.srcs[1].value

        j = i + 1
        while j < len(instructions):
            cand = instructions[j]
            cand_key = _is_eligible_xor(cand)
            if (cand_key is not None
                    and cand_key[0] == reg_name
                    and cand_key[1] == type_str
                    and cand.pred == head_pred
                    and cand.neg  == head_neg):
                chain_indices.append(j)
                chain_xor ^= cand.srcs[1].value
                j += 1
                continue
            if _reads_or_writes(cand, reg_name):
                break
            j += 1

        if len(chain_indices) >= 2:
            mask = _bitmask(type_str)
            xor_val = chain_xor & mask
            if xor_val == 0:
                # Whole chain is a no-op. Drop the head AND the rest.
                skip.update(chain_indices)
            else:
                # Mutate head's immediate; drop the rest.
                head.srcs[1] = ImmOp(xor_val)
                skip.update(chain_indices[1:])
            n_changed += 1
        i += 1

    if skip:
        instructions[:] = [inst for idx, inst in enumerate(instructions)
                           if idx not in skip]
    return n_changed


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
