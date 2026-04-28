"""
Forward-substitution of `mul.lo.<intT>` immediate chains.

Recognizes:

    mul.lo.u32 %d_0, %s_0, K_0
    mul.lo.u32 %d_1, %d_0, K_1
    ...
    mul.lo.u32 %d_{N-1}, %d_{N-2}, K_{N-1}

(consecutive in a single basic block) and folds to:

    mul.lo.u32 %d_{N-1}, %s_0, (K_0 * K_1 * ... * K_{N-1}) mod 2^width

Multiplication is associative and commutative with constants, so
the chain reduces to a single mul with the wrapped product.

Surfaces in `k300_long_mul_chain` and similar.

Conservative gating mirrors add_forward_chain:
  - All same `mul.lo.<intT>`
  - Original src not later overwritten by chain
  - Each intermediate dest must EITHER be overwritten later in
    the chain itself OR not be referenced anywhere outside.
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


def _match_mul_imm(inst: Instruction) -> Optional[tuple[str, str, str, int]]:
    """Recognize `mul.lo.<intT> %dst, %src, K`. Returns (dst, src, type, K)."""
    if inst.pred is not None or inst.mods:
        return None
    if inst.op != "mul":
        return None
    if not inst.types or len(inst.types) < 2:
        return None
    if inst.types[0] != "lo":
        return None
    if inst.types[1] not in _INT_TYPES:
        return None
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return None
    if len(inst.srcs) != 2:
        return None
    a, b = inst.srcs
    if not isinstance(a, RegOp) or not isinstance(b, ImmOp):
        return None
    return (inst.dest.name, a.name, inst.types[1], b.value)


def _is_referenced(fn: Function, reg_name: str,
                   skip: set[tuple[int, int]]) -> bool:
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


def _reduce_block(fn: Function, bi: int) -> int:
    bb = fn.blocks[bi]
    instructions = bb.instructions
    n = len(instructions)
    n_chains = 0
    new_instrs: list[Instruction] = []
    i = 0
    while i < n:
        first = instructions[i]
        first_match = _match_mul_imm(first)
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
            cand_match = _match_mul_imm(cand)
            if cand_match is None:
                break
            dst_j, src_j, type_j, k_j = cand_match
            if type_j != type0:
                break
            if src_j != last_dst:
                break
            chain_indices.append(j)
            K_total *= k_j
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
            chain_dsts.append(_match_mul_imm(instructions[idx])[0])

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
        prod_k = K_total & mask

        new_instrs.append(Instruction(
            op="mul",
            types=["lo", type0],
            dest=RegOp(last_dst),
            srcs=[RegOp(src0), ImmOp(prod_k)],
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
