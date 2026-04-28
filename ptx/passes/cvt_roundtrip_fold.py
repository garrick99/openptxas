"""
cvt.u64.u32 → add.u64 with imm → cvt.u32.u64 roundtrip folding.

Recognizes the pattern:

    cvt.u64.u32 %rd_x, %r_a
    add.u64     %rd_y, %rd_x, K
    cvt.u32.u64 %r_b, %rd_y

and folds it to:

    add.u32 %r_b, %r_a, (K & 0xFFFFFFFF)

Algebraic justification: zero-extending a u32 to u64, adding any
constant K, then truncating back to u32 is equivalent to
u32-wrapping-add of the low 32 bits of K.  Specifically:

    cvt.u32.u64(cvt.u64.u32(x) + K)
      = ((x : u32 → u64) + K) mod 2^32
      = (x + (K mod 2^32)) mod 2^32
      = u32_add(x, K & 0xFFFFFFFF)

Surfaces in k200_alt_32_64 (alternating 32/64 chain with constant
adds) where ptxas folds 6 instructions to 2.

Conservative gating:
  - Three CONSECUTIVE instructions matching the shape
  - %rd_x and %rd_y are RegOps with `rd`-style 64-bit names
  - Middle add.u64 has an ImmOp K
  - Neither %rd_x nor %rd_y is read or written outside this triple
    in the same block
  - First cvt's src must be a RegOp; final cvt's dest a RegOp
"""
from __future__ import annotations

from ..ir import Function, ImmOp, Instruction, RegOp


def _is_cvt_u64_u32(inst: Instruction) -> bool:
    if inst.pred is not None or inst.mods:
        return False
    if inst.op != "cvt":
        return False
    if not inst.types or len(inst.types) < 2:
        return False
    if inst.types[0] != "u64" or inst.types[1] != "u32":
        return False
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return False
    if len(inst.srcs) != 1 or not isinstance(inst.srcs[0], RegOp):
        return False
    return True


def _is_add_u64_imm(inst: Instruction) -> bool:
    if inst.pred is not None or inst.mods:
        return False
    if inst.op != "add":
        return False
    if not inst.types or inst.types[0] != "u64":
        return False
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return False
    if len(inst.srcs) != 2:
        return False
    if not isinstance(inst.srcs[0], RegOp):
        return False
    if not isinstance(inst.srcs[1], ImmOp):
        return False
    return True


def _is_cvt_u32_u64(inst: Instruction) -> bool:
    if inst.pred is not None or inst.mods:
        return False
    if inst.op != "cvt":
        return False
    if not inst.types or len(inst.types) < 2:
        return False
    if inst.types[0] != "u32" or inst.types[1] != "u64":
        return False
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return False
    if len(inst.srcs) != 1 or not isinstance(inst.srcs[0], RegOp):
        return False
    return True


def _block_reads(insts: list[Instruction], skip: set[int], reg: str) -> bool:
    for idx, inst in enumerate(insts):
        if idx in skip:
            continue
        for src in inst.srcs:
            if isinstance(src, RegOp) and src.name == reg:
                return True
    return False


def _block_writes(insts: list[Instruction], skip: set[int], reg: str) -> bool:
    for idx, inst in enumerate(insts):
        if idx in skip:
            continue
        if (inst.dest is not None
                and isinstance(inst.dest, RegOp)
                and inst.dest.name == reg):
            return True
    return False


def _fold_block(instructions: list[Instruction]) -> int:
    n_folded = 0
    new_instrs: list[Instruction] = []
    i = 0
    n = len(instructions)
    while i < n:
        if i + 2 >= n:
            new_instrs.append(instructions[i])
            i += 1
            continue
        a, b, c = instructions[i], instructions[i + 1], instructions[i + 2]
        if not (_is_cvt_u64_u32(a) and _is_add_u64_imm(b) and _is_cvt_u32_u64(c)):
            new_instrs.append(instructions[i])
            i += 1
            continue
        rd_x = a.dest.name
        r_a = a.srcs[0].name
        if b.srcs[0].name != rd_x:
            new_instrs.append(instructions[i])
            i += 1
            continue
        rd_y = b.dest.name
        if c.srcs[0].name != rd_y:
            new_instrs.append(instructions[i])
            i += 1
            continue
        r_b = c.dest.name
        K = b.srcs[1].value & 0xFFFFFFFF
        triple = {i, i + 1, i + 2}
        if _block_reads(instructions, triple, rd_x):
            new_instrs.append(instructions[i])
            i += 1
            continue
        if _block_writes(instructions, triple, rd_x):
            new_instrs.append(instructions[i])
            i += 1
            continue
        if _block_reads(instructions, triple, rd_y):
            new_instrs.append(instructions[i])
            i += 1
            continue
        if _block_writes(instructions, triple, rd_y):
            new_instrs.append(instructions[i])
            i += 1
            continue

        new_instrs.append(Instruction(
            op="add",
            types=["u32"],
            dest=RegOp(r_b),
            srcs=[RegOp(r_a), ImmOp(K)],
            pred=None, neg=False, mods=[],
        ))
        n_folded += 1
        i += 3

    instructions[:] = new_instrs
    return n_folded


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
