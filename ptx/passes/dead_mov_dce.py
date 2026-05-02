"""
Narrow DCE for dead `mov` instructions (any width).

A `mov %r, IMM` or `mov %r, %src` whose destination is never read by
anything else in the function is purely dead — `mov` has no side
effects, so removing it cannot change observable behaviour.  The
broader ptx/passes/dce.py is gated to the factory/fuzzer path because
some baseline tests assert on emit patterns DCE would prune; this
narrow variant restricts itself to `mov` opcodes only, which the
forge emitter writes in unit-typed assigns at the tail of every
diamond merge (`mov.u32 %r34, 0; // unit`) that ptxas DCEs.

Iterates to fixed point — removing `mov %r34, 0` makes
`mov %r33, %r34` unread, then that becomes dead too.
"""
from __future__ import annotations

from ..ir import Function, Instruction, MemOp, Module, RegOp, VectorRegOp


def _uses_of(inst: Instruction) -> set[str]:
    used: set[str] = set()
    for s in inst.srcs:
        if isinstance(s, VectorRegOp):
            for r in s.regs:
                used.add(r)
        elif isinstance(s, RegOp):
            used.add(s.name)
        elif isinstance(s, MemOp) and s.base:
            bn = s.base if s.base.startswith('%') else f'%{s.base}'
            used.add(bn)
    if inst.dest is not None and isinstance(inst.dest, MemOp) and inst.dest.base:
        bn = inst.dest.base if inst.dest.base.startswith('%') else f'%{inst.dest.base}'
        used.add(bn)
    if inst.pred:
        pn = inst.pred if inst.pred.startswith('%') else f'%{inst.pred}'
        used.add(pn)
    return used


def run_function(fn: Function) -> int:
    """Remove dead `mov` instructions until fixed point. Returns # removed.

    Skipped when the function contains an `mma` (or `wgmma`) instruction:
    MMA-test kernels (`tests/test_gpu_correctness.py::TestQmma`) emit
    "padding" `mov.b32 %rN, 0` writes whose only purpose is to reserve
    GPR slots so the regalloc lands the D/A/C operand quad on a 4-aligned
    boundary, even though no instruction reads %rN.  DCE-ing them
    breaks the layout the hardware expects.
    """
    has_mma = any(inst.op in ('mma', 'wgmma')
                  for bb in fn.blocks for inst in bb.instructions)
    if has_mma:
        return 0

    total = 0
    while True:
        readers: set[str] = set()
        for bb in fn.blocks:
            for inst in bb.instructions:
                readers |= _uses_of(inst)
        n_this_pass = 0
        for bb in fn.blocks:
            kept = []
            for inst in bb.instructions:
                if (inst.op == 'mov'
                        and inst.dest is not None
                        and isinstance(inst.dest, RegOp)
                        and inst.dest.name not in readers):
                    n_this_pass += 1
                    continue
                kept.append(inst)
            bb.instructions = kept
        if n_this_pass == 0:
            break
        total += n_this_pass
    return total


def run(mod: Module) -> int:
    total = 0
    for fn in mod.functions:
        total += run_function(fn)
    return total
