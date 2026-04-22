"""
Rename multi-written virtual registers to eliminate WAW hazards at the
SASS level.

If a PTX virtual register `%rN` is written by more than one instruction
in the function, split each later write into a fresh name (`%rN_v2`,
`%rN_v3`, ...) and rewrite all subsequent reads of `%rN` to use the
new name, up until the next write of `%rN` (or end of function).

Why this exists: OpenPTXas's regalloc tends to reuse a single
physical GPR across all writes of a given `%rN`.  When two consecutive
SASS ALU instructions both write the same physical register with no
intervening read of the first write's value, SM_120 exhibits a
write-after-write hazard that silently zeroes the live write's result
(observed as the SHIFT_BOUNDARY / OTHER-cluster bug families).

By giving each write its own virtual name, regalloc is free to
allocate distinct physical registers, so the hazard cannot occur.  It
also sharpens downstream DCE: if a later write is actually dead, its
dedicated vreg is unused and DCE can remove the whole chain leading
to it.

Conservative: multi-write vregs that are only ever read by the NEXT
instruction are kept as-is (renaming there doesn't help: the reader
still reads the first write's value before the second write happens,
and regalloc's natural reuse is already correct).  The pass splits
only when there is a gap of at least one intervening instruction
between the two writes.
"""
from __future__ import annotations

from ..ir import Function, Module, RegOp, MemOp


def _uses_of(inst) -> set[str]:
    """Registers this instruction reads."""
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


def _def_of(inst) -> str | None:
    if inst.dest is None:
        return None
    if isinstance(inst.dest, RegOp):
        return inst.dest.name
    return None


def _substitute_reads(inst, old: str, new: str):
    """Replace reads of `old` with `new` in this instruction's srcs/pred."""
    for i, s in enumerate(inst.srcs):
        if isinstance(s, RegOp) and s.name == old:
            inst.srcs[i] = RegOp(name=new)
        elif isinstance(s, MemOp):
            bname = s.base if s.base and s.base.startswith('%') else (
                f'%{s.base}' if s.base else None)
            if bname == old:
                # Rewrite MemOp base (preserve offset/other fields)
                new_base = new.lstrip('%') if not s.base.startswith('%') else new
                # Replace via dataclass-style copy; assume MemOp has a 'base' field.
                inst.srcs[i] = MemOp(
                    base=new_base,
                    offset=getattr(s, 'offset', 0),
                    space=getattr(s, 'space', None))
    if inst.pred == old:
        inst.pred = new


def run_function(fn: Function) -> int:
    """Split multi-write vregs.  Returns number of renames applied."""
    # Collect all instructions in order, noting which basic block.
    all_insts = []
    for bb in fn.blocks:
        for inst in bb.instructions:
            all_insts.append(inst)

    # Per-vreg: list of (index, instruction) where it's written
    writes: dict[str, list[int]] = {}
    for idx, inst in enumerate(all_insts):
        d = _def_of(inst)
        if d is not None:
            writes.setdefault(d, []).append(idx)

    renames = 0
    # For each vreg with >=2 writes, split at every write past the first
    for name, write_indices in writes.items():
        if len(write_indices) < 2:
            continue
        # Use a counter so names collide-free across multiple renames of same reg
        counter = 2
        for k in range(1, len(write_indices)):
            write_idx = write_indices[k]
            # Don't bother renaming if the immediately-preceding instruction
            # is the previous write and no reader is between them (regalloc
            # already handles that case correctly, and the rename would only
            # add noise).
            prev_write_idx = write_indices[k-1]
            if write_idx == prev_write_idx + 1:
                continue

            new_name = f'{name}_v{counter}'
            counter += 1
            renames += 1

            # Rename this write's dest
            target = all_insts[write_idx]
            target.dest = RegOp(name=new_name)

            # Rewrite all reads of `name` from this point forward, up until
            # the next write of `name` (exclusive).  Find next write index.
            next_write_idx = (write_indices[k+1]
                              if k+1 < len(write_indices)
                              else len(all_insts))
            for j in range(write_idx + 1, next_write_idx):
                _substitute_reads(all_insts[j], name, new_name)

    return renames


def run(mod: Module) -> int:
    total = 0
    for fn in mod.functions:
        total += run_function(fn)
    return total
