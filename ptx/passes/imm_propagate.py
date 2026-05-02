"""
Imm-propagate: fold `mov.<t> %r, IMM` into eligible consumer operands.

Phase 7 of the merkle_hash_leaves bloat fix.  Forge-emitted PTX contains
many `mov.u32 %r, K` instructions whose sole use is a single shift,
sub, xor, etc., that natively accepts an immediate at that position.
ptxas materializes those immediates inline; openptxas had been emitting
MOV.IMM + a register-form consumer.  This pass closes the gap.

Algorithm:
  1. Count def occurrences per RegOp destination across the whole
     function (any instruction with a RegOp dest contributes a def).
  2. Record `imm_defs[%r] = (value, width, mov_inst)` for every
     unpredicated, unmodded `mov.<t> %r, IMM` whose def_count[%r] == 1.
     PTX is generally already in single-assignment form for `%r`, so
     this is conservatively SSA-aware without dominance analysis.
  3. For each non-mov instruction, consult the per-op whitelist of
     source positions that accept IMM.  Replace each RegOp source whose
     name is in `imm_defs` AND whose mov-width matches the consumer's
     expected width at that position with the corresponding ImmOp.
  4. After substitution, drop the movs in `imm_defs` whose dest no
     longer has any readers (a localized DCE — the central DCE pass is
     gated to the factory/fuzzer path).

Width-compatibility (avoids zero/sign-extension ambiguity at lowering):
  - General consumer width = first-type bit-width (e.g. add.u32 → 32).
  - shl/shr at position 1 (shift count): always 32 by PTX spec.
  - Substitute only when widths match.

Whitelist scope (Phase 8 expansion):
  - shl/shr at position 1 — the shift count (Phase 7 baseline)
  - add/sub at position 1 — second source (Phase 8: scheduler-NOP gap
    closed by promoting (0x810, 0x824 / 0x812 / 0x984 / 0x20c) into
    `_SCHED_FORWARDING_SAFE` in sass/schedule.py).
  - and/or/xor at position 1 — second source (Phase 8: LOP3.IMM
    opex_4 / disassembler-validity collision closed by remapping
    invalid misc bits in sass/scoreboard.py's assign_ctrl — same
    hybrid model as the existing LDC clamp).

Other ops (mul, mad, selp, min, max, setp) remain off because:
  * mul/mad: lowering through IMAD often picks `acc==src` aliasing
    that is sensitive to the second-operand form;
  * selp/min/max/setp: predicate-bearing or comparison-shaped — the
    immediate slot at PTX position 1 doesn't match the consumer's
    SASS-side immediate slot in all cases.

Not touched in any case: mov, ld, st, atom, red, bar, membar, fence,
ret, bra, call, cvt (isel requires RegOp source), and predicate-related
ops.

Predicated movs are skipped (conservative: don't propagate a value
that may not be live at every consumer).

Pipeline-toggle: disable via OPENPTXAS_DISABLE_PASSES=imm_propagate.
"""
from __future__ import annotations

from typing import Optional

from ..ir import Function, ImmOp, Instruction, RegOp, VectorRegOp


def _bitwidth_of_type(t: str) -> Optional[int]:
    if not t:
        return None
    if t[0] in ("u", "s", "b", "f"):
        try:
            return int(t[1:])
        except ValueError:
            return None
    return None


def _consumer_width_at(inst: Instruction, pos: int) -> Optional[int]:
    """Bit-width the consumer expects at src position `pos`, or None."""
    op = inst.op
    if op in ("shl", "shr") and pos == 1:
        return 32  # PTX shift count is u32 regardless of data width.
    if not inst.types:
        return None
    return _bitwidth_of_type(inst.types[0])


def _is_int_type(t: str) -> bool:
    """True for integer / bit types (u/s/b<N>), False for floats / preds."""
    if not t:
        return False
    return t[0] in ("u", "s", "b")


def _allowed_positions(op: str, width: Optional[int]) -> tuple[int, ...]:
    # Phase 7: shl/shr shift count.
    if op == "shl":   return (1,)
    if op == "shr":   return (1,)
    # Phase 8: add/sub second source — IADD3.IMM lowering, with the
    # scheduler-side NOP gap closed by promoting IADD3.IMM-as-writer
    # pairs in sass/schedule.py's _SCHED_FORWARDING_SAFE.
    if op == "add":   return (1,)
    if op == "sub":   return (1,)
    # Phase 8: and/or/xor second source — LOP3.IMM lowering, with the
    # ctrl-byte / opex_4 collision closed by remapping invalid misc
    # values for opcode 0x812 in sass/scoreboard.py.
    if op in ("and", "or", "xor"):
        return (1,)
    return ()


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


def _is_simple_mov_imm(inst: Instruction) -> Optional[tuple[str, int, int]]:
    """If `inst` is an unpredicated, unmodded `mov.<t> %r, IMM` with a
    plain (non-vector) RegOp dest, return (dest_name, value, bit_width)."""
    if inst.op != "mov":
        return None
    if inst.pred is not None or inst.mods:
        return None
    if inst.dest is None:
        return None
    if not isinstance(inst.dest, RegOp) or isinstance(inst.dest, VectorRegOp):
        return None
    if len(inst.srcs) != 1:
        return None
    src = inst.srcs[0]
    if not isinstance(src, ImmOp):
        return None
    if not inst.types:
        return None
    width = _bitwidth_of_type(inst.types[0])
    if width is None:
        return None
    return (inst.dest.name, src.value, width)


def run_function(fn: Function) -> int:
    """Run imm_propagate on a single function.  Returns the number of
    operand substitutions performed."""
    def_counts = _walk_def_counts(fn)
    imm_defs: dict[str, tuple[int, int, Instruction]] = {}

    for bb in fn.blocks:
        for inst in bb.instructions:
            sm = _is_simple_mov_imm(inst)
            if sm is None:
                continue
            name, value, width = sm
            if def_counts.get(name, 0) != 1:
                continue
            imm_defs[name] = (value, width, inst)

    if not imm_defs:
        return 0

    n_subs = 0
    for bb in fn.blocks:
        for inst in bb.instructions:
            if inst.op == "mov":
                continue
            # Restrict to integer/bit-typed consumers.  Float ops (f32/f64)
            # and predicate ops have lowering paths that don't uniformly
            # accept ImmOp; folding into them risks isel crashes.
            if not inst.types or not _is_int_type(inst.types[0]):
                continue
            consumer_width = _bitwidth_of_type(inst.types[0])
            allowed = _allowed_positions(inst.op, consumer_width)
            if not allowed:
                continue
            for i, src in enumerate(inst.srcs):
                if i not in allowed:
                    continue
                if not isinstance(src, RegOp) or isinstance(src, VectorRegOp):
                    continue
                rec = imm_defs.get(src.name)
                if rec is None:
                    continue
                value, mov_width, _ = rec
                cw = _consumer_width_at(inst, i)
                if cw is None or cw != mov_width:
                    continue
                inst.srcs[i] = ImmOp(value)
                n_subs += 1

    if n_subs == 0:
        return 0

    used: set[str] = set()
    for bb in fn.blocks:
        for inst in bb.instructions:
            for src in inst.srcs:
                if isinstance(src, RegOp):
                    used.add(src.name)

    dead_ids: set[int] = set()
    for name, (_, _, mov_inst) in imm_defs.items():
        if name not in used:
            dead_ids.add(id(mov_inst))

    if dead_ids:
        for bb in fn.blocks:
            bb.instructions = [i for i in bb.instructions
                               if id(i) not in dead_ids]

    return n_subs


def run(module) -> int:
    total = 0
    for fn in module.functions:
        total += run_function(fn)
    return total
