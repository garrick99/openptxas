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

Whitelist scope:
  - shl/shr at position 1 — the shift count (Phase 7 baseline).
  - add at positions 0, 1 — second source (Phase 8 added pos 1; Phase
    27 added pos 0 since add is commutative). Both-IMM result is
    collapsed by `_try_const_eval_to_mov`.
  - sub at positions 0, 1 — Phase 8 (pos 1) + Phase 10 (pos 0 for
    rotate emulation `sub %d, 32, %n`).
  - and/or/xor at positions 0, 1 — second source (Phase 8 added pos 1
    after closing the LOP3.IMM opex_4 / disassembler-validity collision
    by remapping invalid misc bits in sass/scoreboard.py's assign_ctrl;
    Phase 27 added pos 0 to absorb merkle's Blake2/SHA-256 IV-XOR
    pattern).  Both-IMM operand pairs are collapsed by
    `_try_const_eval_to_mov` so the SASS encoder never sees a two-IMM
    LOP3.

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
    # Phase 8: add second source — IADD3.IMM lowering, with the
    # scheduler-side NOP gap closed by promoting IADD3.IMM-as-writer
    # pairs in sass/schedule.py's _SCHED_FORWARDING_SAFE.
    # Phase 27: also pos 0 at width <=32 (add is commutative; the u32
    # isel handler at isel.py:3711 swaps operands so the IMM lands in
    # the IADD-IMM slot).  64-bit add deliberately omitted: the u64
    # isel path (_select_add_u64) doesn't handle srcs[0]=ImmOp and
    # would raise ISelError.  Both-IMM result is collapsed by
    # _try_const_eval_to_mov so the SASS encoder never sees two IMMs.
    if op == "add":
        return (0, 1) if (width is not None and width <= 32) else (1,)
    # Phase 10: sub also accepts IMM at position 0 — the rotate-emulation
    # pattern `sub %d, 32, %n` (Blake2s 32-bit right-rotate emulation).
    # The isel `sub.<int_t>` handler emits `IADD R, -R, IMM` (IADD-IMM
    # with negate_src0=True; b9 bit 0 set) for this shape — same 1-instr
    # form ptxas natural compile uses, no MOV.IMM required.
    if op == "sub":   return (0, 1)
    # Phase 8: and/or/xor second source — LOP3.IMM lowering, with the
    # ctrl-byte / opex_4 collision closed by remapping invalid misc
    # values for opcode 0x812 in sass/scoreboard.py.
    # Phase 27: also pos 0 at width <=32 (these ops are commutative;
    # the 32/16-bit isel handler at isel.py:3809 calls _materialize_imm
    # on srcs[0] which transparently handles ImmOp).  64-bit deliberately
    # omitted: the u64 LOP3 lowering at isel.py:3840 reads `srcs[0].name`
    # without an ImmOp branch — would AttributeError.  Merkle's SHA-256
    # / Blake2 IV-XOR pattern is u32/b32, so the 32-bit fold absorbs it;
    # both-IMM pairs are collapsed by _try_const_eval_to_mov.
    if op in ("and", "or", "xor"):
        return (0, 1) if (width is not None and width <= 32) else (1,)
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


_CONST_EVAL_OPS = ("add", "sub", "and", "or", "xor")


def _try_const_eval_to_mov(inst: Instruction) -> bool:
    """If `inst` has reduced to a fully-constant binary op (both srcs
    ImmOp), evaluate at compile time and rewrite in-place as
    `mov.<t> dest, IMM_result`.  Returns True if rewritten.

    Phase 10 added the sub-only case.  Phase 27 extends to add/and/or/
    xor: with pos-0 IMM-fold now allowed for these ops, pairs like
    `mov %a, IV_A; mov %b, IV_B; xor %d, %a, %b` (Blake2 IV constant
    XORs in merkle) collapse to `xor %d, IMM_A, IMM_B`, which we then
    evaluate at compile time to a single `mov %d, IMM_A^IMM_B`.  The
    new mov participates in the next fixpoint iteration and is
    typically folded into a downstream IADD/LOP3 immediate slot.

    Width handling: integer widths mask to (1<<width)-1.  Bit ops on
    b<N> types use the same mask.  Add/sub use unsigned wrap (two's
    complement is bit-equivalent at this level).
    """
    if inst.op not in _CONST_EVAL_OPS:
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
    a = inst.srcs[0].value & mask
    b = inst.srcs[1].value & mask
    op = inst.op
    if op == "add":
        result = (a + b) & mask
    elif op == "sub":
        result = (a - b) & mask
    elif op == "and":
        result = a & b
    elif op == "or":
        result = a | b
    elif op == "xor":
        result = a ^ b
    else:
        return False
    inst.op = "mov"
    inst.srcs = [ImmOp(result)]
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
            # If both srcs became ImmOp this round (or already were),
            # evaluate at compile time and rewrite as mov.  See
            # _try_const_eval_to_mov for the per-op rationale.
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
