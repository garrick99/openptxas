"""
Forward-substitution of linear `add.<intT> %dst, %src, K_i` chains.

Recognizes:

    add.u32 %d_0, %s_0, K_0
    add.u32 %d_1, %d_0, K_1
    add.u32 %d_2, %d_1, K_2
    ...
    add.u32 %d_{N-1}, %d_{N-2}, K_{N-1}

(consecutive in a single basic block) and folds to:

    add.u32 %d_{N-1}, %s_0, (K_0 + K_1 + ... + K_{N-1}) mod 2^width

Safety:
  - All adds same integer type, sum taken mod 2^width.
  - Original src %s_0 must not be written by any chain instruction
    (else the consolidated add would read a stale base value).
  - For each intermediate dest %d_i (i < N-1), it is "safe to drop"
    iff EITHER (a) %d_i is overwritten later in the chain itself,
    OR (b) %d_i is not read or written anywhere outside the chain
    in the whole function.  Case (a) covers the common idiom where
    the chain ends back at the same register it started in (e.g.
    `add %r2, %r0, 1; add %r3, %r2, 1; ...; add %r2, %r9, 1`).

Surfaces in `k300_nasty_deep_dep` (9-deep serial add chain) where
ptxas folds 9 instructions to 1.
"""
from __future__ import annotations

from typing import Optional

from ..ir import Function, ImmOp, Instruction, RegOp


_INT_TYPES = {"u32", "s32", "u64", "s64"}


def _bitwidth(t: str) -> int:
    if t in ("u32", "s32"):
        return 32
    if t in ("u64", "s64"):
        return 64
    return 32


def _match_add_imm(inst: Instruction) -> Optional[tuple[str, str, str, int]]:
    """Recognize `add.<intT> %dst, %src, K`. Returns (dst, src, type, K)."""
    if inst.pred is not None or inst.mods:
        return None
    if inst.op != "add":
        return None
    if not inst.types or inst.types[0] not in _INT_TYPES:
        return None
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return None
    if len(inst.srcs) != 2:
        return None
    a, b = inst.srcs
    if not isinstance(a, RegOp) or not isinstance(b, ImmOp):
        return None
    return (inst.dest.name, a.name, inst.types[0], b.value)


def _is_referenced(fn: Function, reg_name: str,
                   skip: set[tuple[int, int]]) -> bool:
    """True if reg_name is read OR written anywhere outside skip."""
    for bi, bb in enumerate(fn.blocks):
        for ii, inst in enumerate(bb.instructions):
            if (bi, ii) in skip:
                continue
            for src in inst.srcs:
                if isinstance(src, RegOp) and src.name == reg_name:
                    return True
            if (inst.dest is not None
                    and isinstance(inst.dest, RegOp)
                    and inst.dest.name == reg_name):
                return True
    return False


def _writes_in(fn: Function, reg_name: str,
               skip: set[tuple[int, int]]) -> bool:
    """True if reg_name is written outside skip."""
    for bi, bb in enumerate(fn.blocks):
        for ii, inst in enumerate(bb.instructions):
            if (bi, ii) in skip:
                continue
            if (inst.dest is not None
                    and isinstance(inst.dest, RegOp)
                    and inst.dest.name == reg_name):
                return True
    return False


def _reduce_block(fn: Function, bi: int) -> int:
    bb = fn.blocks[bi]
    instructions = bb.instructions
    n = len(instructions)
    n_chains = 0
    new_instrs: list[Instruction] = []
    i = 0
    while i < n:
        first = instructions[i]
        first_match = _match_add_imm(first)
        if first_match is None:
            new_instrs.append(first)
            i += 1
            continue
        dst0, src0, type0, k0 = first_match

        chain_ends = i
        K_total = k0
        last_dst = dst0
        chain_indices = [i]
        j = i + 1
        while j < n:
            cand = instructions[j]
            cand_match = _match_add_imm(cand)
            if cand_match is None:
                break
            dst_j, src_j, type_j, k_j = cand_match
            if type_j != type0:
                break
            if src_j != last_dst:
                break
            chain_indices.append(j)
            K_total += k_j
            last_dst = dst_j
            chain_ends = j
            j += 1

        if len(chain_indices) < 2:
            new_instrs.append(first)
            i += 1
            continue

        skip = {(bi, idx) for idx in chain_indices}

        chain_dsts: list[str] = []
        for idx in chain_indices:
            chain_dsts.append(_match_add_imm(instructions[idx])[0])

        if src0 in chain_dsts:
            new_instrs.append(first)
            i += 1
            continue

        intermediates_ok = True
        for pos, idx in enumerate(chain_indices[:-1]):
            inter_dst = chain_dsts[pos]
            overwritten_later_in_chain = False
            for later_dst in chain_dsts[pos + 1:]:
                if later_dst == inter_dst:
                    overwritten_later_in_chain = True
                    break
            if overwritten_later_in_chain:
                continue
            if _is_referenced(fn, inter_dst, skip):
                intermediates_ok = False
                break
        if not intermediates_ok:
            new_instrs.append(first)
            i += 1
            continue

        mask = (1 << _bitwidth(type0)) - 1
        sum_k = K_total & mask

        new_instrs.append(Instruction(
            op="add",
            types=[type0],
            dest=RegOp(last_dst),
            srcs=[RegOp(src0), ImmOp(sum_k)],
            pred=None, neg=False, mods=[],
        ))
        n_chains += 1
        i = chain_ends + 1

    instructions[:] = new_instrs
    return n_chains


def run_function(fn: Function) -> int:
    total = 0
    for bi in range(len(fn.blocks)):
        total += _reduce_block(fn, bi)
    return total


def run(module) -> int:
    total = 0
    for fn in module.functions:
        total += run_function(fn)
    return total
