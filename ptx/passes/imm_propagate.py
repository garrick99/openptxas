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
    # Phase 10: sub also accepts IMM at position 0 — the rotate-emulation
    # pattern `sub %d, 32, %n` (Blake2s 32-bit right-rotate emulation).
    # The isel `sub.<int_t>` handler emits `IADD R, -R, IMM` (IADD-IMM
    # with negate_src0=True; b9 bit 0 set) for this shape — same 1-instr
    # form ptxas natural compile uses, no MOV.IMM required.
    if op == "sub":   return (0, 1)
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


def _try_const_eval_to_mov(inst: Instruction) -> bool:
    """Phase 10: if `inst` has reduced to a fully-constant binary op
    (both srcs ImmOp), evaluate at compile time and rewrite in-place
    as `mov.<t> dest, IMM_result`.  Returns True if rewritten.

    Why this matters: imm_propagate's pos-0 fold for sub combined with
    pos-1 fold (Phase 9) collapses Blake2s's rotate emulation
    (`mov %r, 32; sub %d, %r, %n`) to `sub %d, 32, IMM_n` when %n is
    also a foldable mov-imm (e.g. `mov %n, 16`).  Without this eval,
    isel materializes 32 via IADD3.IMM (0x810) then emits IADD-IMM
    (0x835) for the difference — a pair that requires a scheduler NOP.
    With this eval, sub becomes `mov %d, 16`, the new mov gets
    re-folded into the downstream `shl %r, %x, %d` shift count, and
    the rotate step collapses from 5 SASS instructions to 3.

    Conservative: only sub.<int> for now (the Phase 10 target).  The
    other allowed ops (add/and/or/xor/shl/shr) could be extended the
    same way, but each invites a separate validation against the
    forge corpus.
    """
    if inst.op != "sub":
        return False
    if not inst.types or not _is_int_type(inst.types[0]):
        return False
    if len(inst.srcs) != 2:
        return False
    if not (isinstance(inst.srcs[0], ImmOp) and isinstance(inst.srcs[1], ImmOp)):
        return False
    if inst.pred is not None or inst.mods:
        return False
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return False
    width = _bitwidth_of_type(inst.types[0])
    if width is None:
        return False
    mask = (1 << width) - 1
    diff = (inst.srcs[0].value - inst.srcs[1].value) & mask
    inst.op = "mov"
    inst.srcs = [ImmOp(diff)]
    return True


def run_function(fn: Function) -> int:
    """Run imm_propagate on a single function.  Returns the total number
    of operand substitutions performed across all fixpoint iterations."""
    total_subs = 0
    # Fixpoint loop: a sub-pos-0 + pos-1 fold can produce `sub %d, IMM,
    # IMM`, which _try_const_eval_to_mov rewrites as `mov %d, IMM_diff`.
    # That new mov may then be foldable into a downstream consumer
    # (e.g. shl-pos-1 shift count), so we re-iterate until no further
    # folds occur.  Bounded by 4 iterations to cap pathological inputs.
    for _ in range(4):
        n_subs = _propagate_once(fn)
        if n_subs == 0:
            break
        total_subs += n_subs
    return total_subs


def _propagate_once(fn: Function) -> int:
    """One iteration of imm_propagate.  Returns substitution count."""
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
            # Phase 10: if both srcs of a sub became ImmOp this round,
            # evaluate at compile time and rewrite as mov.  See
            # _try_const_eval_to_mov for the rationale (avoids the
            # IADD3.IMM→IADD-IMM scheduler-NOP penalty on Blake2s
            # rotate emulation).
            _try_const_eval_to_mov(inst)

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
