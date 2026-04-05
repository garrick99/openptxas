"""
Dead code elimination (DCE) at the PTX IR level.

Removes instructions whose results are never consumed, while preserving
side-effecting instructions: memory stores, atomics, barriers, control
flow, and kernel EXIT/RET.

Applied once per function as a fixed-point pass — removing one dead
instruction may make its sources dead too.

Conservative: any instruction with no dest, or whose op is in the
`_SIDE_EFFECTING` set, is always preserved. Anything else is removed
when its dest register has no readers in the rest of the function.
"""

from __future__ import annotations

from ..ir import Function, Module, RegOp, MemOp, Instruction


# Operations that have externally-visible side effects. These are never
# removed by DCE even if their destination register is unused.
_SIDE_EFFECTING = {
    "st",       # any store (.global, .shared, .local, .param)
    "atom",     # atomic RMW
    "red",      # atomic reduction
    "bar",      # barrier sync
    "membar",   # memory barrier
    "fence",    # memory fence
    "bra",      # branch
    "call",     # function call
    "ret",      # return
    "exit",     # kernel exit
    "trap",     # trap
    "brkpt",    # breakpoint
    "setp",     # writes predicate — flag-like, keep conservative
    "cp",       # cp.async etc.
    "mbarrier",
    "griddepcontrol",
    "wgmma",
    "mma",      # MMA side effect (writes many regs; keep)
    "tex",      # texture fetch
    "suld",     # surface load
    "sust",     # surface store
    "sured",    # surface reduction
    "vote",     # warp vote — side-effecting via shfl sync
    "shfl",     # warp shuffle — synchronizing
}


def _uses_of(inst: Instruction) -> set[str]:
    """Return the set of register names this instruction reads."""
    used: set[str] = set()
    for s in inst.srcs:
        if isinstance(s, RegOp):
            used.add(s.name)
        elif isinstance(s, MemOp) and s.base:
            bname = s.base if s.base.startswith('%') else f'%{s.base}'
            used.add(bname)
    if inst.pred:
        pn = inst.pred if inst.pred.startswith('%') else f'%{inst.pred}'
        used.add(pn)
    return used


def _def_of(inst: Instruction) -> str | None:
    """Return the name of the single register defined by this instruction, or None."""
    if inst.dest is None:
        return None
    if isinstance(inst.dest, RegOp):
        return inst.dest.name
    return None


def is_side_effecting(inst: Instruction) -> bool:
    """Return True if this instruction may have externally-visible effects."""
    if inst.op in _SIDE_EFFECTING:
        return True
    # Predicated stores/exits still count; their op tag already matches above.
    # ld.volatile / ld.acquire are memory-ordered and should be kept even if
    # the result is unused.
    if inst.op == "ld" and any(t in ("volatile", "acquire", "relaxed") for t in inst.types):
        return True
    return False


def run_function(fn: Function) -> int:
    """Run DCE on a single function until fixed point. Returns # removed."""
    removed_total = 0
    while True:
        # Collect all register reads across the function.
        readers: set[str] = set()
        for bb in fn.blocks:
            for inst in bb.instructions:
                readers |= _uses_of(inst)
        # Also treat function return values as reads (conservative: none
        # modeled explicitly in IR, so only reg-regs are tracked).

        removed_this_pass = 0
        for bb in fn.blocks:
            kept = []
            for inst in bb.instructions:
                if is_side_effecting(inst):
                    kept.append(inst)
                    continue
                d = _def_of(inst)
                if d is None:
                    kept.append(inst)
                    continue
                if d in readers:
                    kept.append(inst)
                    continue
                # Dead: skip
                removed_this_pass += 1
            bb.instructions = kept
        if removed_this_pass == 0:
            break
        removed_total += removed_this_pass
    return removed_total


def run(mod: Module) -> int:
    """Run DCE on every function in the module. Returns total # removed."""
    total = 0
    for fn in mod.functions:
        total += run_function(fn)
    return total
