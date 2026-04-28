"""
Mul-then-add chain reduction at the PTX IR level.

STATUS (2026-04-27): ACTIVE — re-wired after the underlying SASS
issue was diagnosed.  The fused 3-operand form `IMAD R, A, K, R`
(dest aliasing src2) emits wrong GPU output for non-pow-2 K
immediates.  sass/isel.py now decomposes that case into a separate
IMAD (with RZ accumulator) followed by IADD3, sidestepping the
fused-form bug.  The pass IR transform was always correct — only
the downstream lowering needed fixing.

Recognizes the pattern produced by unroll + counter constant-prop on
loops whose body computes `tmp = a * counter` and accumulates
`acc += tmp`:

    mul.lo.<t> %tmp, %a, K_0
    add.<t>    %acc, %acc, %tmp
    mul.lo.<t> %tmp, %a, K_1
    add.<t>    %acc, %acc, %tmp
    ...
    mul.lo.<t> %tmp, %a, K_{N-1}
    add.<t>    %acc, %acc, %tmp

becomes:

    mad.lo.<t> %acc, %a, ΣK_i, %acc

For N=4 with K = 0,1,2,3: ΣK = 6, so `add %r2, %r2, %r0*6` =
`mad.lo %r2, %r0, 6, %r2` — a single instruction replacing 8.
If ΣK = 0 the chain becomes a no-op and is dropped entirely.

Designed to run AFTER unroll + load_cse + add3_chain_reduce, BEFORE
trivial_fold (which would otherwise convert `mul.lo %tmp, %a, 0` to
`mov %tmp, 0` and break the chain detection).

Conservative gating:
  - First insn must be `mul.lo.<t> %tmp, %a, K_i` with int_type
  - Second must be `add.<t> %acc, %acc, %tmp` with same type
  - Same %a, %acc, type across all pairs
  - %tmp may share register names across pairs (it gets rewritten
    each iter)
  - K_i are integer immediates
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
    return 32


def _is_mul_first(inst: Instruction) -> Optional[tuple[str, str, str, int]]:
    """Recognize `mul.lo.<t> %tmp, %a, K`. Returns (tmp, a, type, K)."""
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
        return None  # `mul %r, %r, K` — degenerate
    return (inst.dest.name, a.name, inst.types[1], b.value)


def _is_add_second(inst: Instruction, tmp_name: str, type_key: str) -> Optional[str]:
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
    n_chains = 0
    new_instrs: list[Instruction] = []
    i = 0
    while i < len(instructions):
        first = instructions[i]
        first_key = _is_mul_first(first)
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
        acc_name = _is_add_second(second, tmp_name, type_key)
        if acc_name is None or acc_name == a_name or acc_name == tmp_name:
            new_instrs.append(first)
            i += 1
            continue

        chain_indices = [(i, i + 1)]
        K_values = [k0]
        j = i + 2
        while j < len(instructions):
            scan = j
            while scan < len(instructions):
                cand = instructions[scan]
                if _writes(cand, a_name) or _writes(cand, acc_name):
                    scan = -1
                    break
                if _reads(cand, tmp_name) and not (
                        _is_add_second(cand, tmp_name, type_key) == acc_name):
                    scan = -1
                    break
                fk = _is_mul_first(cand)
                if (fk is not None and fk[0] == tmp_name
                        and fk[1] == a_name and fk[2] == type_key):
                    break
                scan += 1
            if scan < 0 or scan + 1 >= len(instructions):
                break
            cand_first = instructions[scan]
            cand_second = instructions[scan + 1]
            cand_first_key = _is_mul_first(cand_first)
            if cand_first_key is None:
                break
            cand_second_acc = _is_add_second(cand_second, tmp_name, type_key)
            if cand_second_acc != acc_name:
                break
            chain_indices.append((scan, scan + 1))
            K_values.append(cand_first_key[3])
            j = scan + 2

        if len(chain_indices) < 2:
            new_instrs.append(first)
            i += 1
            continue

        mask = (1 << _bitwidth(type_key)) - 1
        sum_k = sum(K_values) & mask

        if sum_k == 0:
            # The whole chain is a no-op (acc += a * 0). Drop everything.
            pass
        else:
            new_instrs.append(Instruction(
                op="mad",
                types=["lo", type_key],
                dest=RegOp(acc_name),
                srcs=[RegOp(a_name), ImmOp(sum_k), RegOp(acc_name)],
                pred=None, neg=False, mods=[],
            ))

        # Carry through any neutral inter-pair instructions in original
        # order. They were verified safe above.
        chain_starts = [s for s, _ in chain_indices]
        chain_seconds = {sec for _, sec in chain_indices}
        for k in range(chain_indices[0][1] + 1, chain_indices[-1][1] + 1):
            if k in chain_seconds or k in chain_starts:
                continue
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
