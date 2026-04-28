"""
Common-multiplier sum-tree fold.

Recognizes:

    mul.lo.<intT> %r_0, %a, K_0
    mul.lo.<intT> %r_1, %a, K_1
    ...
    mul.lo.<intT> %r_{N-1}, %a, K_{N-1}
    [add-tree that combines all %r_i into a single %dst]

and folds the whole thing to:

    mul.lo.<intT> %dst, %a, (K_0 + K_1 + ... + K_{N-1}) mod 2^width

The add-tree is verified by simulation: a virtual `weights` table
tracks which mul-output combinations are live in each register.
Each `add.<intT> %d, %s1, %s2` is consumed if BOTH %s1 and %s2 are
in the table — %d then carries the sum of their weights, and %s1/%s2
are marked consumed.  At the end exactly ONE register must remain
live with weight = ΣK_i.

Surfaces in:
  - k200_triple_acc / k200_quad_acc (linear sum chains)
  - k300_nasty_accum5 / k300_nasty_long_live (balanced tree merges)

Conservative gating:
  - Muls are CONTIGUOUS, all integer type, all sharing %a as src0
  - K_i are integer immediates
  - Each %r_i appears exactly once as a destination (no aliasing)
  - %a is not written by any mul in the sequence
  - Add-tree starts immediately after the last mul; once tree
    starts, only `add.<intT>` instructions consuming the live
    table are allowed (any non-add instruction that reads a live
    reg aborts; non-add that doesn't read the table is allowed
    past, but I keep this conservative for the first version).
  - Tree must terminate with exactly one live register holding
    the total weight; that becomes the final %dst.
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
    """Recognize `mul.lo.<intT> %dst, %a, K`. Returns (dst, a, type, K)."""
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
    if a.name == inst.dest.name:
        return None
    return (inst.dest.name, a.name, inst.types[1], b.value)


def _match_add_reg_reg(inst: Instruction, type_key: str) -> Optional[tuple[str, str, str]]:
    """Recognize `add.<type_key> %dst, %s1, %s2`. Returns (dst, s1, s2)."""
    if inst.pred is not None or inst.mods:
        return None
    if inst.op != "add":
        return None
    if not inst.types or inst.types[0] != type_key:
        return None
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return None
    if len(inst.srcs) != 2:
        return None
    s1, s2 = inst.srcs
    if not isinstance(s1, RegOp) or not isinstance(s2, RegOp):
        return None
    return (inst.dest.name, s1.name, s2.name)


def _writes(inst: Instruction, name: str) -> bool:
    return (inst.dest is not None
            and isinstance(inst.dest, RegOp)
            and inst.dest.name == name)


def _reads(inst: Instruction, name: str) -> bool:
    for src in inst.srcs:
        if isinstance(src, RegOp) and src.name == name:
            return True
    return False


def _reduce_block(fn: Function, bi: int) -> int:
    bb = fn.blocks[bi]
    instructions = bb.instructions
    n_total = len(instructions)
    n_chains = 0
    new_instrs: list[Instruction] = []
    i = 0
    while i < n_total:
        first = instructions[i]
        first_match = _match_mul_imm(first)
        if first_match is None:
            new_instrs.append(first)
            i += 1
            continue
        first_dst, a_name, type_key, k0 = first_match

        mul_indices = [i]
        weights: dict[str, int] = {first_dst: k0}
        j = i + 1
        while j < n_total:
            cand_match = _match_mul_imm(instructions[j])
            if cand_match is None:
                break
            d, a, t, k = cand_match
            if a != a_name or t != type_key:
                break
            if d in weights or d == a_name:
                break
            mul_indices.append(j)
            weights[d] = k
            j += 1

        if len(mul_indices) < 2:
            new_instrs.append(first)
            i += 1
            continue

        tree_indices: list[int] = []
        live = dict(weights)
        scan = j
        ok = True
        while scan < n_total and len(live) > 1:
            inst = instructions[scan]
            am = _match_add_reg_reg(inst, type_key)
            if am is None:
                ok = False
                break
            d, s1, s2 = am
            if s1 not in live or s2 not in live:
                ok = False
                break
            if s1 == s2:
                ok = False
                break
            if d in live and d != s1 and d != s2:
                ok = False
                break
            new_w = live[s1] + live[s2]
            del live[s1]
            del live[s2]
            live[d] = new_w
            tree_indices.append(scan)
            scan += 1
        if not ok or len(live) != 1:
            new_instrs.append(first)
            i += 1
            continue

        final_dst, final_w = next(iter(live.items()))
        consumed = set(mul_indices) | set(tree_indices)

        for orig_dst in weights.keys():
            if orig_dst == final_dst:
                continue
            future_use = False
            for k_idx in range(scan, n_total):
                if k_idx in consumed:
                    continue
                if _reads(instructions[k_idx], orig_dst):
                    future_use = True
                    break
                if _writes(instructions[k_idx], orig_dst):
                    break
            if future_use:
                ok = False
                break
        if not ok:
            new_instrs.append(first)
            i += 1
            continue

        mask = (1 << _bitwidth(type_key)) - 1
        sum_k = final_w & mask

        new_instrs.append(Instruction(
            op="mul",
            types=["lo", type_key],
            dest=RegOp(final_dst),
            srcs=[RegOp(a_name), ImmOp(sum_k)],
            pred=None, neg=False, mods=[],
        ))
        n_chains += 1
        i = scan

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
