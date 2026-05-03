"""
Pair-fusion of two-input xors into a synthetic 3-input `xor3` op that
isel lowers to a single SASS LOP3.LUT (R-R-R 0x96) or LOP3.IMM
(R-IMM-R 0x96 — opcode 0x812).

Recognizes the pattern emitted by merkle/SHA-style hash-mixing rounds:

    xor.b32 %tmp, %a, %b
    xor.b32 %dst, %tmp, %c    (or %dst, %c, %tmp -- xor is commutative)

where each of %a, %b, %c is either a register or an immediate, with at
most ONE immediate across the three operands, both xors share the same
bit-equivalent type, and neither is predicated.  Rewrites to:

    xor3.<t> %dst, %a, %b, %c

The synthetic `xor3` op is consumed by isel's xor path:
  - all-reg case   -> LOP3.LUT R{d}, R{a}, R{b}, R{c}, 0x96  (Phase 42)
  - one-imm case   -> LOP3.LUT R{d}, R{a}, IMM, R{b}, 0x96    (Phase 43)

Phase 41's imm_propagate-via-commute lands constants in the first xor
of merkle's SHA-mixing rounds at srcs[1], producing chains like
`xor %tmp, %r17, 0x6b08e647 ; xor %r22, %tmp, %r22` that Phase 42's
all-register matcher missed.  Phase 43 catches them.

When BOTH xors have IMM operands (= two IMMs total), const-eval should
collapse the chain to a single mov %d, IMM_xor_IMM_xor_a; we skip those
to avoid stepping on _try_const_eval_to_mov.

Conservative gating:
  - Both xors: same type in {b32, u32, s32} (all bit-equivalent for XOR)
  - No predicate, no mods on either
  - Each xor's two srcs are RegOp or ImmOp (no MemOp etc.)
  - At most ONE ImmOp across {%a, %b, %c}.
  - %tmp must NOT alias any RegOp in {%a, %b, %c} (avoid self-cycle)
  - %dst must NOT alias %a or %b (write doesn't shadow a still-needed src)
  - %tmp must be read EXACTLY ONCE in the entire function (= the
    second xor).  Cheapest sufficient liveness check.
"""
from __future__ import annotations

from collections import Counter

from ..ir import Function, ImmOp, Instruction, RegOp


_XOR_TYPES = {"b32", "u32", "s32"}


def _is_simple_xor(inst: Instruction) -> bool:
    """xor of the right type with two RegOp/ImmOp srcs and a RegOp dest."""
    if inst.op != "xor" or inst.pred is not None or inst.mods:
        return False
    if not inst.types or inst.types[0] not in _XOR_TYPES:
        return False
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return False
    if len(inst.srcs) != 2:
        return False
    return all(isinstance(s, (RegOp, ImmOp)) for s in inst.srcs)


def _count_reg_reads(fn: Function) -> Counter:
    """Count how many times each register name appears as a source operand
    anywhere in the function (memory base names included via MemOp.base)."""
    from ..ir import MemOp
    counts: Counter = Counter()
    for bb in fn.blocks:
        for inst in bb.instructions:
            for src in inst.srcs:
                if isinstance(src, RegOp):
                    counts[src.name] += 1
                elif isinstance(src, MemOp):
                    base = src.base
                    if base.startswith("%"):
                        counts[base] += 1
                    else:
                        counts["%" + base] += 1
    return counts


def run_function(fn: Function) -> int:
    read_counts = _count_reg_reads(fn)
    n_fused = 0
    for bb in fn.blocks:
        instrs = bb.instructions
        new_instrs: list[Instruction] = []
        i = 0
        while i < len(instrs):
            first = instrs[i]
            if i + 1 >= len(instrs) or not _is_simple_xor(first):
                new_instrs.append(first)
                i += 1
                continue
            second = instrs[i + 1]
            if not _is_simple_xor(second):
                new_instrs.append(first)
                i += 1
                continue
            if first.types[0] != second.types[0]:
                new_instrs.append(first)
                i += 1
                continue

            tmp = first.dest.name
            a_op = first.srcs[0]
            b_op = first.srcs[1]

            # Locate %tmp in the second xor.  Exactly one src must be
            # %tmp; the other (RegOp or ImmOp) becomes c_op.
            sec0, sec1 = second.srcs[0], second.srcs[1]
            sec0_is_tmp = isinstance(sec0, RegOp) and sec0.name == tmp
            sec1_is_tmp = isinstance(sec1, RegOp) and sec1.name == tmp
            if sec0_is_tmp and not sec1_is_tmp:
                c_op = sec1
            elif sec1_is_tmp and not sec0_is_tmp:
                c_op = sec0
            else:
                new_instrs.append(first)
                i += 1
                continue

            # At most one ImmOp across the three operands.  Two IMMs is
            # a const-eval case (mov %d, IMM_xor_IMM_xor_a) and belongs
            # to _try_const_eval_to_mov, not here.
            imm_count = sum(1 for op in (a_op, b_op, c_op)
                            if isinstance(op, ImmOp))
            if imm_count >= 2:
                new_instrs.append(first)
                i += 1
                continue

            dst = second.dest.name

            # Self-cycle / shadowing checks apply only to RegOps.
            reg_names_in_srcs = {op.name for op in (a_op, b_op, c_op)
                                 if isinstance(op, RegOp)}
            if tmp in reg_names_in_srcs:
                new_instrs.append(first)
                i += 1
                continue
            ab_reg_names = {op.name for op in (a_op, b_op)
                            if isinstance(op, RegOp)}
            if dst in ab_reg_names:
                new_instrs.append(first)
                i += 1
                continue
            if read_counts.get(tmp, 0) != 1:
                new_instrs.append(first)
                i += 1
                continue

            new_instrs.append(Instruction(
                op="xor3",
                types=[first.types[0]],
                dest=RegOp(dst),
                srcs=[a_op, b_op, c_op],
                pred=None, neg=False, mods=[],
            ))
            n_fused += 1
            i += 2
        instrs[:] = new_instrs
    return n_fused


def run(module) -> int:
    total = 0
    for fn in module.functions:
        total += run_function(fn)
    return total
