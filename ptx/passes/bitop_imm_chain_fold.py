"""
Forward-substitution of linear `xor.<bT> %dst, %src, K_i` and
`or.<bT> %dst, %src, K_i` chains.

Same shape as ptx/passes/add_forward_chain.py but for the
bit-operation family (xor, or):

    xor.b32 %d_0, %s_0, K_0
    xor.b32 %d_1, %d_0, K_1
    ...
    xor.b32 %d_{N-1}, %d_{N-2}, K_{N-1}

folds to:

    xor.b32 %d_{N-1}, %s_0, K_0 ^ K_1 ^ ... ^ K_{N-1}

Likewise `or` chains fold via bitwise OR.  Both ops are associative
and commutative with constant operands, identity element 0, so
constant chains fold trivially.

Surfaces in:
  - k200_xor_reduce, k300_triple_xor (XOR chains)
  - k300_or_chain (OR chain)
  - and any other bit-mask accumulator that ptxas's optimizer
    constant-folds.

Conservative gating mirrors add_forward_chain:
  - Contiguous chain in a single block, all same type
  - Original src not later overwritten by chain instructions
  - Each intermediate dest must EITHER be overwritten later in the
    chain itself OR not be referenced anywhere outside the chain.
"""
from __future__ import annotations

from typing import Optional

from ..ir import Function, ImmOp, Instruction, RegOp


_BIT_TYPES = {"b32", "b64"}
_BIT_OPS = {"xor", "or"}


def _bitwidth(t: str) -> int:
    if t == "b32":
        return 32
    if t == "b64":
        return 64
    return 32


def _combine(op: str, a: int, b: int) -> int:
    if op == "xor":
        return a ^ b
    if op == "or":
        return a | b
    raise ValueError(f"unsupported op {op}")


def _identity(op: str) -> int:
    return 0


def _match_bitop_imm(inst: Instruction) -> Optional[tuple[str, str, str, str, int]]:
    """Recognize `xor.<bT> %dst, %src, K` or `or.<bT> %dst, %src, K`.

    Returns (op, dst, src, type, K).
    """
    if inst.pred is not None or inst.mods:
        return None
    if inst.op not in _BIT_OPS:
        return None
    if not inst.types or inst.types[0] not in _BIT_TYPES:
        return None
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return None
    if len(inst.srcs) != 2:
        return None
    a, b = inst.srcs
    if not isinstance(a, RegOp) or not isinstance(b, ImmOp):
        return None
    return (inst.op, inst.dest.name, a.name, inst.types[0], b.value)


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
        first_match = _match_bitop_imm(first)
        if first_match is None:
            new_instrs.append(first)
            i += 1
            continue
        op0, dst0, src0, type0, k0 = first_match

        chain_ends = i
        K_total = k0
        last_dst = dst0
        chain_indices = [i]
        j = i + 1
        while j < n:
            cand = instructions[j]
            cand_match = _match_bitop_imm(cand)
            if cand_match is None:
                break
            op_j, dst_j, src_j, type_j, k_j = cand_match
            if op_j != op0 or type_j != type0:
                break
            if src_j != last_dst:
                break
            chain_indices.append(j)
            K_total = _combine(op0, K_total, k_j)
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
            chain_dsts.append(_match_bitop_imm(instructions[idx])[1])

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
        folded_k = K_total & mask

        if folded_k == _identity(op0):
            new_instrs.append(Instruction(
                op="mov",
                types=[type0],
                dest=RegOp(last_dst),
                srcs=[RegOp(src0)],
                pred=None, neg=False, mods=[],
            ))
        else:
            new_instrs.append(Instruction(
                op=op0,
                types=[type0],
                dest=RegOp(last_dst),
                srcs=[RegOp(src0), ImmOp(folded_k)],
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
