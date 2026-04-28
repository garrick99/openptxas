"""
Selective loop unrolling at the PTX IR level.

STATUS (2026-04-27): IMPLEMENTED + CORRECT, but NOT wired into the
pipeline. Enabling unrolling alone REGRESSES `w1_loop_two_acc` from
+7 to +14 non-NOPs vs ptxas. Reason: ptxas's loop wins come from
THREE chained optimizations, and selective unroll is only one:

  1. Loop unroll (this pass)
  2. Constant-folding of consecutive imm-add chains:
     `add %r,%r,1; add %r,%r,1; ...` → `add %r,%r,N`
  3. Strength reduction for reg-reg-add chains:
     `add %r2,%r2,%r0; ...` × N → `IMAD %r2, %r0, N, %r2`

After unrolling, the body has N copies of every counter-style
increment. Without (2) collapsing the imm-add chains and (3)
collapsing the reg-reg-add chains, the unrolled output is strictly
larger than the rolled form. Once (2) and (3) land in this
directory, wire this pass back in via:

  from ptx.passes.unroll import run_function as _unroll_run_function
  _unroll_run_function(fn)

placed between `_if_convert(fn)` and `_sink_param_loads(fn)` in
sass/pipeline.py. The follow-up passes should run AFTER this one.

Converts small constant-bounded single-block loops into straight-line
code so downstream isel + scoreboard can schedule across the body
without back-edge constraints. Preserves correctness by:

  - Cloning the entire body (including the induction increment) per
    iteration, so register state evolves identically to the original.
  - Only firing when iteration count is statically derivable from
    `mov %ctr, IMM_init` (preceding) + `add %ctr, %ctr, IMM_step` +
    `setp.<cmp> %p, %ctr, IMM_bound` + `@%p bra LOOP`.
  - Bailing out when the loop's predicate or any of its registers are
    referenced after the loop, so removing the `setp` is safe.
  - Capping at MAX_ITER iterations and MAX_UNROLLED_BODY total
    instructions — beyond that, the rolled form is preferable.

The pass closes the bulk of the GAP bucket in the suite_all benchmark
where ptxas's aggressive unroller emits 5-10 fewer non-NOP
instructions than OpenPTXas's pre-unroll output. After this pass runs,
constant folding inside isel + the existing scheduler can typically
collapse repeated `add %r2, %r2, K` chains to a single op, matching
ptxas density.

Conservative on purpose: nested loops are handled via fixed-point
iteration (inner loops unroll first, then the outer becomes a single-
block candidate). Loops whose body contains nested labels, multiple
back-edges, or non-trivial control flow are skipped — the cost of a
wrong unroll is silent miscompile.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

from ..ir import Function, Instruction, ImmOp, LabelOp, RegOp


# Tuning: don't unroll loops with trip count above this, even if static.
MAX_UNROLL_ITER = 8
# Total non-pred non-NOP instructions after unroll. Beyond this, the
# branch overhead becomes cheaper than the code-size blowup.
MAX_UNROLLED_BODY = 64


def _is_imm(op) -> bool:
    return isinstance(op, ImmOp)


def _reg_name(op) -> Optional[str]:
    return op.name if isinstance(op, RegOp) else None


def _evaluate_setp(cmp_op: str, lhs: int, rhs: int, neg: bool) -> bool:
    """Return True if the back-edge `@(!?)pred bra LOOP` would be taken.

    The setp emits `pred = (lhs cmp rhs)`. The branch is taken iff
    `pred XOR neg` is True (i.e. `pred` when not negated, `not pred`
    when @!pred).
    """
    if cmp_op == "lt":   r = lhs < rhs
    elif cmp_op == "le": r = lhs <= rhs
    elif cmp_op == "gt": r = lhs > rhs
    elif cmp_op == "ge": r = lhs >= rhs
    elif cmp_op == "eq": r = lhs == rhs
    elif cmp_op == "ne": r = lhs != rhs
    else:
        # Unknown comparison — conservative: pretend loop never re-enters.
        return False
    return (not r) if neg else r


def _find_init_value(fn: Function, ctr_name: str, loop_idx: int) -> Optional[int]:
    """Search predecessor blocks for `mov.<t> %ctr, IMM`. Last write wins.

    Conservative: only matches a literal `mov` from immediate. Does
    not chase through other definitions (cvt, computed values, etc.).
    """
    init: Optional[int] = None
    for bb in fn.blocks[:loop_idx]:
        for inst in bb.instructions:
            if (inst.op == "mov"
                    and inst.dest is not None
                    and _reg_name(inst.dest) == ctr_name
                    and len(inst.srcs) == 1
                    and _is_imm(inst.srcs[0])
                    and inst.pred is None):
                init = inst.srcs[0].value
    return init


def _pred_used_after_loop(fn: Function, pred_name: str, loop_idx: int) -> bool:
    """Return True if `pred_name` is read by any instruction in a block
    that follows the loop block (defensive: assume linear fall-through
    means everything after `loop_idx` is reachable).
    """
    for bb in fn.blocks[loop_idx + 1:]:
        for inst in bb.instructions:
            if inst.pred == pred_name:
                return True
            for src in inst.srcs:
                if isinstance(src, RegOp) and src.name == pred_name:
                    return True
    return False


def _try_unroll(fn: Function, idx: int) -> bool:
    bb = fn.blocks[idx]
    if not bb.label or len(bb.instructions) < 3:
        return False

    # The PTX parser keeps fall-through code in the same block (it splits
    # only on labels). So the back-edge `@%p bra THIS_LABEL` may be in
    # the middle of the block; everything BEFORE it is the loop body
    # (run on every iteration including the last entry); everything
    # AFTER it is post-loop fall-through (run once when the predicate
    # is false). Find the back-edge by scanning.
    bra_idx = None
    for j, inst in enumerate(bb.instructions):
        if (inst.op == "bra"
                and inst.pred is not None
                and inst.srcs
                and isinstance(inst.srcs[0], LabelOp)
                and inst.srcs[0].name == bb.label):
            bra_idx = j
            break
    if bra_idx is None or bra_idx < 2:
        return False
    bra = bb.instructions[bra_idx]
    pred_name = bra.pred
    pred_neg = bra.neg

    # Setp must define that predicate, immediately preceding the bra.
    setp = bb.instructions[bra_idx - 1]
    if setp.op != "setp":
        return False
    if setp.dest is None or _reg_name(setp.dest) != pred_name:
        return False
    if setp.pred is not None:
        return False  # guarded setp — out of scope
    # types[0] is the comparison; types[1:] is the operand type.
    if not setp.types:
        return False
    cmp_op = setp.types[0]
    if cmp_op not in ("lt", "le", "gt", "ge", "eq", "ne"):
        return False
    if len(setp.srcs) < 2:
        return False
    ctr_op, bound_op = setp.srcs[0], setp.srcs[1]
    if not isinstance(ctr_op, RegOp) or not _is_imm(bound_op):
        return False
    ctr_name = ctr_op.name
    bound_val = bound_op.value

    # Counter increment must immediately precede the setp.
    incr = bb.instructions[bra_idx - 2]
    if incr.op != "add":
        return False
    if incr.dest is None or _reg_name(incr.dest) != ctr_name:
        return False
    if incr.pred is not None:
        return False  # predicated increment — out of scope
    if len(incr.srcs) < 2:
        return False
    if _reg_name(incr.srcs[0]) != ctr_name or not _is_imm(incr.srcs[1]):
        return False
    step_val = incr.srcs[1].value
    if step_val == 0:
        return False

    # Locate counter initialization in a predecessor block.
    init_val = _find_init_value(fn, ctr_name, idx)
    if init_val is None:
        return False

    # Simulate the loop: each iteration runs the body (which ends with
    # the increment), then the setp+bra checks the post-increment value.
    iterations = 0
    ctr_val = init_val
    while iterations <= MAX_UNROLL_ITER:
        iterations += 1
        ctr_val += step_val
        if not _evaluate_setp(cmp_op, ctr_val, bound_val, pred_neg):
            break
    else:
        return False  # exceeded MAX_UNROLL_ITER

    # Bail out on infinite loops or loops that exit without entering.
    if iterations < 1:
        return False

    # Body is everything before the setp (the increment is the last
    # body instruction). Tail is everything after the bra (post-loop
    # fall-through).
    body = bb.instructions[:bra_idx - 1]
    tail = bb.instructions[bra_idx + 1:]
    if len(body) * iterations > MAX_UNROLLED_BODY:
        return False

    # Safety: the predicate `pred_name` must be dead after the loop.
    # Check both the in-block tail and any subsequent blocks.
    for inst in tail:
        if inst.pred == pred_name:
            return False
        for src in inst.srcs:
            if isinstance(src, RegOp) and src.name == pred_name:
                return False
    if _pred_used_after_loop(fn, pred_name, idx):
        return False

    # The induction counter must not be WRITTEN by any body
    # instruction other than its own increment (which is body[-1]).
    # READS are now allowed: per-iteration constant-propagation
    # (the clone loop below substitutes %ctr → ImmOp(init+i*step)
    # in body[:-1]) feeds the resulting `add %tmp, %a, K_i` pairs
    # into add3_chain_reduce, which collapses N pairs into a single
    # mad.lo + add(ΣK_i). Surfaced as a win for w1_loop_countdown
    # once that reducer landed.
    body_excl_incr = body[:-1]
    for inst in body_excl_incr:
        if inst.dest is not None and isinstance(inst.dest, RegOp) and inst.dest.name == ctr_name:
            return False

    # Cheap-body gating: each non-increment body instruction must be
    # one of:
    #   - `add.<int_type>` reg+reg or reg+imm
    #   - `xor.<bits>` reg+reg or reg+imm (foldable post-unroll +
    #     const-prop via imm_xor_fold)
    #   - `ld.<space>.<type>` from a register-base MemOp
    # AND: no register may be the destination of more than one op
    # KIND (e.g. both `add` and `xor` writing the same %acc). Mixed-
    # op writers are non-commutative — fold passes can't combine the
    # cloned bodies into a single chain, and unroll just bloats the
    # output. Surfaced as a regression on w1_loop_shift (mixed add+
    # xor on %r2, MIXED → GAP, artifact 20260427_203853).
    _CHEAP_INT_TYPES = {"u32", "s32", "u64", "s64", "b32", "b64"}
    _ALLOWED_LD_SPACES = {"global", "shared", "const", "param"}
    dest_op_kinds: dict[str, set[str]] = {}
    for inst in body_excl_incr:
        if inst.dest is not None and isinstance(inst.dest, RegOp):
            dest_op_kinds.setdefault(inst.dest.name, set()).add(inst.op)
    for reg, kinds in dest_op_kinds.items():
        if len(kinds) > 1:
            return False

    for inst in body_excl_incr:
        if inst.op == "add":
            if not inst.types or inst.types[0] not in _CHEAP_INT_TYPES:
                return False
            if inst.pred is not None or inst.mods:
                return False
            if (inst.dest is None or not isinstance(inst.dest, RegOp)
                    or len(inst.srcs) != 2
                    or not isinstance(inst.srcs[0], RegOp)):
                return False
            if not isinstance(inst.srcs[1], (RegOp, ImmOp)):
                return False
            continue
        if inst.op == "xor":
            if not inst.types or inst.types[0] not in _CHEAP_INT_TYPES:
                return False
            if inst.pred is not None or inst.mods:
                return False
            if (inst.dest is None or not isinstance(inst.dest, RegOp)
                    or len(inst.srcs) != 2
                    or not isinstance(inst.srcs[0], RegOp)):
                return False
            if not isinstance(inst.srcs[1], (RegOp, ImmOp)):
                return False
            continue
        # NB: `mul.lo.<int>` body shape was previously allowed here to
        # feed ptx/passes/mul3_chain_reduce.py post-unroll. That pass
        # is currently DORMANT (see its header) due to a SASS-level
        # correctness failure on w1_loop_mul_acc. Without the reducer
        # downstream, allowing mul-body unroll alone makes things
        # worse (+8 SASS vs +4 unrolled-and-reduced). Re-add the gate
        # here when mul3_chain_reduce is re-wired.
        if inst.op == "ld":
            if inst.pred is not None or inst.mods:
                return False
            if not inst.types or len(inst.types) < 2:
                return False
            if inst.types[0] not in _ALLOWED_LD_SPACES:
                return False
            if (inst.dest is None or not isinstance(inst.dest, RegOp)
                    or len(inst.srcs) != 1):
                return False
            from ..ir import MemOp as _MemOp
            if not isinstance(inst.srcs[0], _MemOp):
                return False
            continue
        return False

    # Disallow loops whose body contains:
    #   - mid-iteration control flow (would break linear cloning)
    #   - side-effecting ops where unrolling N times produces N copies
    #     that the downstream pipeline (regalloc / scoreboard) doesn't
    #     yet handle correctly (atomics in particular: a suite_all run
    #     showed correctness regression on `w2_loop_atom_add` when the
    #     loop body's `atom.add` got cloned 3× and the resulting
    #     multiple-write-to-%r3 pattern wasn't WAW-renamed).
    _UNSAFE_OPS = {
        "bra", "call", "ret", "exit", "trap",          # control flow
        "atom", "red",                                  # atomic RMW / reduction
        "bar", "barrier", "membar", "fence",           # synchronization
        "cp",                                           # cp.async, cp.async.bulk
        "mbarrier",
    }
    for inst in body:
        if inst.op in _UNSAFE_OPS:
            return False

    # Transform: replace the block's instructions with N cloned copies
    # of `body`, then the post-loop tail. We drop the setp + bra. Each
    # iteration's body[:-1] (everything before the counter increment)
    # gets %ctr substituted with the iteration-specific constant value
    # so downstream constant-folding can fire on `mul %r,%r0,%ctr`,
    # `xor %r,%r,%ctr`, etc. The increment itself stays as-is so the
    # counter ends up at init + N*step at the post-loop point.
    new_instrs: list[Instruction] = []
    iter_ctr_val = init_val
    for _ in range(iterations):
        for inst in body_excl_incr:
            cloned = deepcopy(inst)
            new_srcs = []
            for src in cloned.srcs:
                if isinstance(src, RegOp) and src.name == ctr_name:
                    new_srcs.append(ImmOp(iter_ctr_val))
                else:
                    new_srcs.append(src)
            cloned.srcs = new_srcs
            new_instrs.append(cloned)
        # Counter increment: unchanged (it writes to %ctr and reads %ctr —
        # we don't substitute the increment because it IS the source of
        # the next iteration's counter value when not unrolled, and the
        # cumulative effect of N un-substituted increments is the
        # post-loop value we want).
        new_instrs.append(deepcopy(body[-1]))
        iter_ctr_val += step_val
    new_instrs.extend(tail)
    bb.instructions = new_instrs
    return True


def run_function(fn: Function) -> int:
    """Unroll all eligible loops in `fn`. Returns number of loops unrolled.

    Iterates to a fixed point so nested loops eventually unroll: an
    inner loop's unroll converts an outer loop into a single-block
    candidate that becomes eligible next iteration.
    """
    n_unrolled = 0
    progress = True
    safety_iter = 0
    while progress and safety_iter < 32:
        progress = False
        safety_iter += 1
        for i in range(len(fn.blocks)):
            if _try_unroll(fn, i):
                n_unrolled += 1
                progress = True
                break  # block list shape preserved, but restart for cascade
    return n_unrolled


def run(module) -> int:
    total = 0
    for fn in module.functions:
        total += run_function(fn)
    return total
