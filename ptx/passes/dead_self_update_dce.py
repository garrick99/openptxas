"""
Narrow DCE for dead self-updating ALU accumulators.

STATUS (2026-04-27): DORMANT — the IR-level transform is correct
but DROPPING leftover counter increments exposes a latent SASS
scheduling hazard.  The spacer slot the dead instruction occupied
between `IADD3 R, R, R` and the subsequent `IADD3.UR R2, R3, UR6`
(ALLOC addr-lo) is required for the scoreboard to insert a wait;
removing it makes IADD3.UR fire too early on w1_loop_sum and
w1_loop_mul_acc (CUDA error 700 / wrong output).

Re-wire in sass/pipeline.py only after auditing the
IADD3 -> IADD3.UR scheduling rule (sass/scoreboard.py /
sass/schedule.py) so the wait is inserted regardless of upstream
spacer presence.


Several chain-fold passes consolidate counter / running-sum
increments into a single instruction at the end of the chain:

    add.u32 %r3, %r3, 4     (after imm_add_fold collapses 4× +1)
    mad.lo.u32 %r2, %r0, 4, %r2  (after repeated_add_reduce)

When the destination of such a self-update is never read after the
fold, the entire instruction is dead — but the existing DCE pass
(ptx/passes/dce.py) is gated to the factory/fuzzer path because it
prunes patterns that some baseline tests assert on.

This pass is narrow: it removes ONLY pure-ALU instructions whose
destination ALSO appears as a source of the same instruction (i.e.
self-update form `op %r, %r, ...`) AND whose destination is never
read or written elsewhere in the function.  The self-update gate
prevents pruning instructions that produce a fresh result some
other code path might depend on.

Surfaces leftover counters in unrolled-and-folded loops, e.g.
`w1_loop_load_acc` (the `%r3 += 4` synthesized from a 4-iteration
counter is dead after the loop body folds away).

Conservative: function-wide live-out is approximated by "dest
appears anywhere else as src OR dest" — any later read or write
keeps the instruction.
"""
from __future__ import annotations

from ..ir import Function, Instruction, MemOp, RegOp


_PURE_SELF_UPDATE_OPS = {
    "add", "sub", "mul", "mad",
    "and", "or", "xor",
    "shl", "shr",
}


def _is_pure_self_update(inst: Instruction) -> bool:
    if inst.op not in _PURE_SELF_UPDATE_OPS:
        return False
    if inst.pred is not None or inst.mods or inst.neg:
        return False
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return False
    if not inst.srcs:
        return False
    dst_name = inst.dest.name
    for src in inst.srcs:
        if isinstance(src, RegOp) and src.name == dst_name:
            return True
    return False


def _read_elsewhere(fn: Function, name: str,
                    skip: tuple[int, int]) -> bool:
    """True if `name` is read by any instruction other than skip.

    A self-update `op %r, %r, K` only matters to consumers that read
    %r — earlier writes are irrelevant (they get overwritten anyway).
    Conservative: the check is function-wide, not CFG-aware, so a
    pre-op read also counts as 'elsewhere' even though dropping would
    not change its observed value.  This errs on the safe side.

    Critical: MemOp base registers count as reads (e.g. `[%rd5]` in a
    load/store reads %rd5 even though it's not a RegOp source).  And
    a MemOp dest (the address operand of a store) likewise counts.
    """
    target = name if name.startswith('%') else f'%{name}'
    bare = target.lstrip('%')
    def mem_reads(op):
        if not isinstance(op, MemOp):
            return False
        if not op.base:
            return False
        b = op.base if op.base.startswith('%') else f'%{op.base}'
        return b == target or op.base == bare
    for bi, bb in enumerate(fn.blocks):
        for ii, inst in enumerate(bb.instructions):
            if (bi, ii) == skip:
                continue
            for src in inst.srcs:
                if isinstance(src, RegOp) and src.name == name:
                    return True
                if mem_reads(src):
                    return True
            if mem_reads(inst.dest):
                return True
            if inst.pred:
                pn = inst.pred if inst.pred.startswith('%') else f'%{inst.pred}'
                if pn == target:
                    return True
    return False


def run_function(fn: Function) -> int:
    total = 0
    while True:
        n_this_pass = 0
        for bi in range(len(fn.blocks)):
            bb = fn.blocks[bi]
            keep = [True] * len(bb.instructions)
            for ii, inst in enumerate(bb.instructions):
                if not _is_pure_self_update(inst):
                    continue
                dst = inst.dest.name
                if _read_elsewhere(fn, dst, (bi, ii)):
                    continue
                keep[ii] = False
                n_this_pass += 1
            if not all(keep):
                bb.instructions[:] = [inst for inst, k in zip(bb.instructions, keep) if k]
        if n_this_pass == 0:
            break
        total += n_this_pass
    return total


def run(module) -> int:
    total = 0
    for fn in module.functions:
        total += run_function(fn)
    return total
