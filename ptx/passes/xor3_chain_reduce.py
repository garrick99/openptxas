"""
Pair-fusion of register-only 2-input xors into a synthetic 3-input
`xor3` op that isel lowers to a single SASS LOP3.LUT 0x96.

Recognizes the pattern emitted by merkle/SHA-style hash-mixing rounds:

    xor.b32 %tmp, %a, %b
    xor.b32 %dst, %tmp, %c    (or %dst, %c, %tmp -- xor is commutative)

where every source is a register, both xors share the same bit-equivalent
type, neither is predicated, and `%tmp`'s only consumer in the function
is the second xor.  Rewrites to:

    xor3.<t> %dst, %a, %b, %c

The synthetic `xor3` op is consumed by isel's xor path and emitted as a
single `LOP3.LUT R{d}, R{a}, R{b}, R{c}, 0x96, !PT` (3-input XOR truth
table).

Mirrors the pattern of `iadd3_pair_reduce.py` (Phase 34) but for XOR.
XOR is associative and commutative, so the merge is provably safe at
the value level — but the LUT byte must be exactly correct (0x96 for
3-input XOR) to preserve correctness, hence the GPU-validation gate.

Conservative gating:
  - Both xors: same type in {b32, u32, s32} (all bit-equivalent for XOR)
  - No predicate, no mods on either
  - All three of %a, %b, %c are RegOps (no ImmOp -- the imm-form path
    lowers via LOP3.IMM and we don't want to interfere with that)
  - %tmp != %a, %b, %c  (avoid self-cycle confusion)
  - %dst != %a, %b      (write doesn't shadow a still-needed src)
  - %tmp must be read EXACTLY ONCE in the entire function (= the
    second xor).  Cheapest sufficient liveness check.
"""
from __future__ import annotations

from collections import Counter

from ..ir import Function, Instruction, RegOp


_XOR_TYPES = {"b32", "u32", "s32"}


def _is_simple_reg_xor(inst: Instruction) -> bool:
    if inst.op != "xor" or inst.pred is not None or inst.mods:
        return False
    if not inst.types or inst.types[0] not in _XOR_TYPES:
        return False
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return False
    if len(inst.srcs) != 2:
        return False
    return all(isinstance(s, RegOp) for s in inst.srcs)


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
            if i + 1 >= len(instrs) or not _is_simple_reg_xor(first):
                new_instrs.append(first)
                i += 1
                continue
            second = instrs[i + 1]
            if not _is_simple_reg_xor(second):
                new_instrs.append(first)
                i += 1
                continue
            if first.types[0] != second.types[0]:
                new_instrs.append(first)
                i += 1
                continue

            tmp = first.dest.name
            a = first.srcs[0].name
            b = first.srcs[1].name
            sec_src0 = second.srcs[0].name
            sec_src1 = second.srcs[1].name
            if sec_src0 == tmp and sec_src1 != tmp:
                c = sec_src1
            elif sec_src1 == tmp and sec_src0 != tmp:
                c = sec_src0
            else:
                new_instrs.append(first)
                i += 1
                continue

            dst = second.dest.name
            if tmp in (a, b, c):
                new_instrs.append(first)
                i += 1
                continue
            if dst in (a, b):
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
                srcs=[RegOp(a), RegOp(b), RegOp(c)],
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
