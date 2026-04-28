"""
CSE for redundant `cvt.u64.u32` + `shl.b64` address-stride pairs.

Recognizes the pattern:

    cvt.u64.u32 %rd_x, %r_a
    shl.b64     %rd_x, %rd_x, K        (K is an immediate)

This is the canonical "u32 promoted to u64, then left-shifted by K"
shape used to produce element-stride byte offsets (K is typically 2
for 4-byte stride, 3 for 8-byte stride).  When the same pair recurs
with the same %r_a and same K, and neither %rd_x nor %r_a has been
written in between, the second pair is redundant — %rd_x already
holds the shifted value.  Drop the second pair entirely.

Surfaces in r1_dot4 (and similar kernels that recompute the
load/store byte offset twice).

Conservative gating:
  - Both cvt.u64.u32 instructions write to the same dest reg %rd_x
    AND read the same src reg %r_a.
  - The shl.b64 immediate K is the same.
  - Between the first pair and the second pair, no instruction
    writes %rd_x or %r_a.
  - The cvt and shl are immediately adjacent in both pairs (the
    only shape currently emitted by openptxas-style address calc).
"""
from __future__ import annotations

from typing import Optional

from ..ir import Function, ImmOp, Instruction, RegOp


def _match_cvt_u64_u32(inst: Instruction) -> Optional[tuple[str, str]]:
    if inst.pred is not None or inst.mods:
        return None
    if inst.op != "cvt":
        return None
    if not inst.types or len(inst.types) < 2:
        return None
    if inst.types[0] != "u64" or inst.types[1] != "u32":
        return None
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return None
    if len(inst.srcs) != 1 or not isinstance(inst.srcs[0], RegOp):
        return None
    return (inst.dest.name, inst.srcs[0].name)


def _match_shl_b64_imm(inst: Instruction) -> Optional[tuple[str, str, int]]:
    if inst.pred is not None or inst.mods:
        return None
    if inst.op != "shl":
        return None
    if not inst.types or inst.types[0] != "b64":
        return None
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return None
    if len(inst.srcs) != 2:
        return None
    s, k = inst.srcs
    if not isinstance(s, RegOp) or not isinstance(k, ImmOp):
        return None
    return (inst.dest.name, s.name, k.value)


def _writes(inst: Instruction, name: str) -> bool:
    return (inst.dest is not None
            and isinstance(inst.dest, RegOp)
            and inst.dest.name == name)


def _fold_block(instructions: list[Instruction]) -> int:
    n_dropped = 0
    new_instrs: list[Instruction] = []
    n = len(instructions)
    seen: list[tuple[str, str, int, int]] = []
    i = 0
    while i < n:
        if i + 1 >= n:
            new_instrs.append(instructions[i])
            i += 1
            continue
        cvt = _match_cvt_u64_u32(instructions[i])
        if cvt is None:
            inst = instructions[i]
            seen = [(rd, ra, k, p) for (rd, ra, k, p) in seen
                    if not _writes(inst, rd) and not _writes(inst, ra)]
            new_instrs.append(inst)
            i += 1
            continue
        rd_x, r_a = cvt
        shl = _match_shl_b64_imm(instructions[i + 1])
        if shl is None or shl[0] != rd_x or shl[1] != rd_x:
            inst = instructions[i]
            seen = [(rd, ra, k, p) for (rd, ra, k, p) in seen
                    if not _writes(inst, rd) and not _writes(inst, ra)]
            new_instrs.append(inst)
            i += 1
            continue
        K = shl[2]
        match_idx = None
        for s_idx, (rd_s, ra_s, k_s, _p) in enumerate(seen):
            if rd_s == rd_x and ra_s == r_a and k_s == K:
                match_idx = s_idx
                break
        if match_idx is not None:
            n_dropped += 1
            i += 2
            continue
        seen.append((rd_x, r_a, K, len(new_instrs)))
        new_instrs.append(instructions[i])
        new_instrs.append(instructions[i + 1])
        i += 2

    instructions[:] = new_instrs
    return n_dropped


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
