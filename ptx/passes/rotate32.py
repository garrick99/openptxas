"""
32-bit rotate-fusion pass — IR-level rewrite of (shr+shl+OR) → SHF.L.U32.HI.

Phase 11 of the merkle_hash_leaves bloat fix.  After Phase 10's
imm_propagate pos-0 fold + const-eval, merkle's PTX still contains
~640 instances of the 32-bit rotate-emulation triple:

    shr.u32 %a, %x, K1
    shl.b32 %b, %x, K2     ; K2 = 32 - K1, possibly via const-eval
    or.b32  %c, %a, %b     ; (or add.b32 / xor.b32 — all commutative)

ptxas emits each as a single SHF.L.U32.HI %c, %x, K2, %x.  This pass
recognizes the same pattern and rewrites the IR so isel emits the
single SHF.L.U32.HI directly.

Mirrors `ptx/passes/rotate.py` (the 64-bit pass) for invariants:
    1. OP ∈ {add, or, xor}              — commutative AND rotation-equivalent
    2. shr is LOGICAL (shr.u32, NOT shr.s32)
    3. Both shifts share the same source register
    4. K1 + K2 == 32 (compile-time constants)

Differences vs. the 64-bit pass:
    - Operates on 32-bit shifts (shl.b32 / shr.u32) and 32-bit combine.
    - REWRITES the IR (the 64-bit pass only annotates).  After this
      pass runs, the matched triple becomes a single synthetic
      `rot.b32 %c, %x, K2` instruction recognized by isel.
    - Runs INSIDE compile_function (after imm_propagate), not at the
      module-level pre-pass — needs constant shift amounts that
      imm_propagate reveals.
"""
from __future__ import annotations

from typing import Optional

from ..ir import Function, ImmOp, Instruction, RegOp, VectorRegOp


# Operators for which 32-bit rotate fusion is semantically valid.
_ROTATE_COMBINE_OPS = {"or", "add", "xor"}

# 32-bit data types accepted on the combine instruction.
_B32_TYPES = {"b32", "u32", "s32"}


def _reg_name(op) -> Optional[str]:
    if isinstance(op, RegOp) and not isinstance(op, VectorRegOp):
        return op.name
    return None


def _imm_val(op) -> Optional[int]:
    return op.value if isinstance(op, ImmOp) else None


def _is_shl_b32(inst: Instruction) -> bool:
    return (inst.op == "shl"
            and inst.types
            and inst.types[0] in _B32_TYPES)


def _is_shr_u32_logical(inst: Instruction) -> bool:
    """Logical 32-bit right shift only — shr.u32/b32, NEVER shr.s32 (Bug 2)."""
    return (inst.op == "shr"
            and inst.types
            and inst.types[0] in ("u32", "b32"))


def _is_simple_unpredicated_unmodded(inst: Instruction) -> bool:
    return inst.pred is None and not inst.mods


def _walk_def_counts(fn: Function) -> dict[str, int]:
    counts: dict[str, int] = {}
    for bb in fn.blocks:
        for inst in bb.instructions:
            d = inst.dest
            if d is None:
                continue
            if isinstance(d, VectorRegOp):
                for r in (d.regs or ()):
                    counts[r] = counts.get(r, 0) + 1
            elif isinstance(d, RegOp):
                counts[d.name] = counts.get(d.name, 0) + 1
    return counts


def _walk_use_counts(fn: Function) -> dict[str, int]:
    counts: dict[str, int] = {}
    for bb in fn.blocks:
        for inst in bb.instructions:
            for src in inst.srcs:
                if isinstance(src, RegOp) and not isinstance(src, VectorRegOp):
                    counts[src.name] = counts.get(src.name, 0) + 1
    return counts


def _try_match_rot32(combine: Instruction,
                     producers: dict[str, Instruction]) -> Optional[tuple[str, int, Instruction, Instruction]]:
    """
    Try to recognize `combine` as the OR step of a 32-bit rotate triple.

    Returns (src_reg_name, K_left, shl_inst, shr_inst) if matched, else None.
      K_left is the LEFT-rotate amount (== shl_K == 32 - shr_K).

    All three semantic invariants of the 64-bit rotate pass apply.
    """
    if combine.op not in _ROTATE_COMBINE_OPS:
        return None
    if not combine.types or combine.types[0] not in _B32_TYPES:
        return None
    if not _is_simple_unpredicated_unmodded(combine):
        return None
    if combine.dest is None or not isinstance(combine.dest, RegOp):
        return None
    if len(combine.srcs) != 2:
        return None

    a_name = _reg_name(combine.srcs[0])
    b_name = _reg_name(combine.srcs[1])
    if a_name is None or b_name is None:
        return None
    if a_name == b_name:
        # Pathological: combine reads same reg twice — not a rotate.
        return None

    prod_a = producers.get(a_name)
    prod_b = producers.get(b_name)
    if prod_a is None or prod_b is None:
        return None

    # Try both assignments: (shl, shr) and (shr, shl).
    for shl_cand, shr_cand in ((prod_a, prod_b), (prod_b, prod_a)):
        if not _is_shl_b32(shl_cand) or not _is_shr_u32_logical(shr_cand):
            continue
        if not _is_simple_unpredicated_unmodded(shl_cand):
            continue
        if not _is_simple_unpredicated_unmodded(shr_cand):
            continue
        if shl_cand.dest is None or shr_cand.dest is None:
            continue
        if len(shl_cand.srcs) < 2 or len(shr_cand.srcs) < 2:
            continue

        shl_src = _reg_name(shl_cand.srcs[0])
        shr_src = _reg_name(shr_cand.srcs[0])
        if shl_src is None or shr_src is None:
            continue
        if shl_src != shr_src:
            continue

        shl_k = _imm_val(shl_cand.srcs[1])
        shr_k = _imm_val(shr_cand.srcs[1])
        if shl_k is None or shr_k is None:
            continue
        # Restrict to non-degenerate rotates (1..31 each side).
        if shl_k <= 0 or shl_k >= 32 or shr_k <= 0 or shr_k >= 32:
            continue
        if shl_k + shr_k != 32:
            continue

        return (shl_src, shl_k, shl_cand, shr_cand)

    return None


def _build_producer_map(insts: list[Instruction]) -> dict[str, Instruction]:
    """Within a single basic block, last-def map for plain RegOp dests."""
    producers: dict[str, Instruction] = {}
    for inst in insts:
        d = inst.dest
        if isinstance(d, VectorRegOp):
            continue
        if isinstance(d, RegOp):
            producers[d.name] = inst
    return producers


def run_function(fn: Function) -> int:
    """
    Recognize 32-bit rotate triples and rewrite each as a single
    synthetic `rot.b32 %dst, %src, ImmOp(K_left)` instruction.

    Returns the number of triples fused.  shl/shr instructions whose
    dest has no remaining readers after the fuse are removed.
    """
    n_fused = 0

    # Pre-compute use counts across the whole function so we can decide
    # whether each shl/shr dest can be deleted (no other consumers).
    use_counts = _walk_use_counts(fn)
    def_counts = _walk_def_counts(fn)

    # Track which producer instructions (shl/shr) we've absorbed.
    absorbed_ids: set[int] = set()

    for bb in fn.blocks:
        producers = _build_producer_map(bb.instructions)

        for idx, inst in enumerate(bb.instructions):
            match = _try_match_rot32(inst, producers)
            if match is None:
                continue
            src_name, k_left, shl_inst, shr_inst = match

            # Each absorbed shl/shr must have exactly one definition (so
            # removing it is safe) and at most one use beyond the combine
            # we're rewriting.  When the use_count == 1, the only user
            # was the combine itself → safe to drop.
            shl_dest_name = _reg_name(shl_inst.dest)
            shr_dest_name = _reg_name(shr_inst.dest)
            if shl_dest_name is None or shr_dest_name is None:
                continue
            if def_counts.get(shl_dest_name, 0) != 1:
                continue
            if def_counts.get(shr_dest_name, 0) != 1:
                continue

            # Rewrite the combine into a synthetic `rot.b32 %dst, %src, K_left`.
            inst.op = "rot"
            inst.types = ["b32"]
            inst.srcs = [RegOp(src_name), ImmOp(k_left)]
            n_fused += 1

            # Mark the shl/shr for deletion only if they had a single user
            # (the combine we just rewrote).
            if use_counts.get(shl_dest_name, 0) == 1:
                absorbed_ids.add(id(shl_inst))
            if use_counts.get(shr_dest_name, 0) == 1:
                absorbed_ids.add(id(shr_inst))

        # Remove absorbed producers from this block.
        if absorbed_ids:
            bb.instructions = [i for i in bb.instructions
                               if id(i) not in absorbed_ids]

    return n_fused


def run(module) -> int:
    total = 0
    for fn in module.functions:
        total += run_function(fn)
    return total
