"""
3-operand add chain reduction at the PTX IR level.

Recognizes the pattern produced by unroll + counter constant-prop on
loops whose body computes `tmp = base + counter` (or a similar
shape that becomes `tmp = base + K_i` after substitution) and
accumulates `acc += tmp`:

    add.<t>  %tmp, %a, K_0
    add.<t>  %acc, %acc, %tmp
    add.<t>  %tmp, %a, K_1
    add.<t>  %acc, %acc, %tmp
    ...
    add.<t>  %tmp, %a, K_{N-1}
    add.<t>  %acc, %acc, %tmp

becomes:

    mad.lo.<t> %acc, %a, N, %acc
    add.<t>    %acc, %acc, ΣK_i      (omitted if Σ == 0)

For N=4 the chain shrinks from 8 to 1 or 2 instructions. Designed to
run AFTER unroll + load_cse and BEFORE imm_add_fold, so the chain
is intact when this pass walks it.

The first instruction of each pair can also be `mov.<t> %tmp, %a`
(treated as `add %tmp, %a, 0`) — that handles the case where
trivial_fold has already simplified the K_0 = 0 form to a mov.

Conservative gating:
  - Same `%a`, `%acc`, scalar type across all pairs in a chain
  - %tmp can be the same register name across pairs (it gets
    rewritten each iteration; the value is consumed only by the
    immediately-following second-add)
  - K values are integer immediates
  - No `pred`, no `mods`
  - Between pairs: any instruction is allowed AS LONG AS it doesn't
    write %a, write %acc, or read %tmp (the latter would observe a
    stale tmp value after fold)
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
    return 32  # default


def _is_first(inst: Instruction) -> Optional[tuple[str, str, str, int]]:
    """Recognize `add.<t> %tmp, %a, K` or `mov.<t> %tmp, %a` (= add 0).
    Returns (tmp_name, a_name, type, K_value) or None.
    """
    if inst.pred is not None or inst.mods:
        return None
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return None
    if inst.op == "add":
        if not inst.types or inst.types[0] not in _INT_TYPES:
            return None
        if len(inst.srcs) != 2:
            return None
        a, b = inst.srcs
        if not isinstance(a, RegOp) or not isinstance(b, ImmOp):
            return None
        if a.name == inst.dest.name:
            return None  # not a fresh tmp; could be an accumulator chain handled elsewhere
        return (inst.dest.name, a.name, inst.types[0], b.value)
    if inst.op == "mov":
        if not inst.types or inst.types[0] not in _INT_TYPES:
            return None
        if len(inst.srcs) != 1 or not isinstance(inst.srcs[0], RegOp):
            return None
        if inst.srcs[0].name == inst.dest.name:
            return None
        return (inst.dest.name, inst.srcs[0].name, inst.types[0], 0)
    return None


def _is_second(inst: Instruction, tmp_name: str, type_key: str) -> Optional[str]:
    """Recognize `add.<t> %acc, %acc, %tmp`. Returns acc_name or None."""
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
    a, b = inst.srcs
    if not isinstance(a, RegOp) or not isinstance(b, RegOp):
        return None
    if a.name != inst.dest.name:
        return None
    if b.name != tmp_name:
        return None
    return inst.dest.name


def _writes(inst: Instruction, reg_name: str) -> bool:
    return (inst.dest is not None
            and isinstance(inst.dest, RegOp)
            and inst.dest.name == reg_name)


def _reads(inst: Instruction, reg_name: str) -> bool:
    for src in inst.srcs:
        if isinstance(src, RegOp) and src.name == reg_name:
            return True
    return False


def _reduce_block(instructions: list[Instruction]) -> int:
    """Recognize and collapse 3-op add chains in a single block.
    Returns count of chains collapsed.
    """
    n_chains = 0
    new_instrs: list[Instruction] = []
    i = 0
    while i < len(instructions):
        first = instructions[i]
        first_key = _is_first(first)
        if first_key is None:
            new_instrs.append(first)
            i += 1
            continue
        tmp_name, a_name, type_key, k0 = first_key
        if i + 1 >= len(instructions):
            new_instrs.append(first)
            i += 1
            continue
        second = instructions[i + 1]
        acc_name = _is_second(second, tmp_name, type_key)
        if acc_name is None or acc_name == a_name or acc_name == tmp_name:
            new_instrs.append(first)
            i += 1
            continue

        # Walk forward gathering more pairs. Between pairs, allow
        # any instruction that doesn't WRITE %a, WRITE %acc, or READ
        # %tmp. (Writes to %tmp itself are fine — they're the next
        # pair's first.)
        chain_indices = [(i, i + 1)]
        K_values = [k0]
        j = i + 2
        while j < len(instructions):
            # Skip "neutral" instructions until we find another candidate first.
            scan = j
            while scan < len(instructions):
                cand = instructions[scan]
                if _writes(cand, a_name) or _writes(cand, acc_name):
                    scan = -1
                    break
                if _reads(cand, tmp_name) and not (
                        _is_second(cand, tmp_name, type_key) == acc_name):
                    # Reading tmp for non-second-add purpose: bail.
                    scan = -1
                    break
                # Try to interpret as a new "first".
                fk = _is_first(cand)
                if (fk is not None and fk[0] == tmp_name
                        and fk[1] == a_name and fk[2] == type_key):
                    break
                # Otherwise it's a neutral instruction; skip past.
                scan += 1
            if scan < 0 or scan + 1 >= len(instructions):
                break
            cand_first = instructions[scan]
            cand_second = instructions[scan + 1]
            cand_first_key = _is_first(cand_first)
            if cand_first_key is None:
                break
            cand_second_acc = _is_second(cand_second, tmp_name, type_key)
            if cand_second_acc != acc_name:
                break
            # Verify any instructions BETWEEN the previous pair and this
            # one are still safe (rescan the gap above did it).
            chain_indices.append((scan, scan + 1))
            K_values.append(cand_first_key[3])
            j = scan + 2

        if len(chain_indices) < 2:
            new_instrs.append(first)
            i += 1
            continue

        # Collapse the chain. Emit:
        #   mad.lo.<type_key> %acc, %a, N, %acc
        #   add.<type_key>    %acc, %acc, ΣK_i   (only if ΣK != 0)
        n = len(chain_indices)
        mask = (1 << _bitwidth(type_key)) - 1
        sum_k = sum(K_values) & mask
        new_instrs.append(Instruction(
            op="mad",
            types=["lo", type_key],
            dest=RegOp(acc_name),
            srcs=[RegOp(a_name), ImmOp(n), RegOp(acc_name)],
            pred=None, neg=False, mods=[],
        ))
        if sum_k != 0:
            new_instrs.append(Instruction(
                op="add",
                types=[type_key],
                dest=RegOp(acc_name),
                srcs=[RegOp(acc_name), ImmOp(sum_k)],
                pred=None, neg=False, mods=[],
            ))

        # Append all "neutral" instructions that fell BETWEEN chain
        # pairs in original order. They were verified safe above.
        chain_starts = [s for s, _ in chain_indices]
        chain_seconds = {sec for _, sec in chain_indices}
        next_pair_idx = 1
        for k in range(chain_indices[0][1] + 1, chain_indices[-1][1] + 1):
            if k in chain_seconds:
                continue
            if k == chain_starts[next_pair_idx] if next_pair_idx < len(chain_starts) else False:
                next_pair_idx += 1
                continue
            # It's a neutral inter-pair instruction — keep it.
            if k not in chain_starts:
                new_instrs.append(instructions[k])

        n_chains += 1
        i = chain_indices[-1][1] + 1

    instructions[:] = new_instrs
    return n_chains


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
