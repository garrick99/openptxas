"""
Single-block load CSE: when the same `[base + offset]` is loaded more
than once in a basic block with no intervening write to the same
memory and no barrier/atomic, drop the redundant loads.

Designed to run AFTER loop unrolling, where the unroller leaves N
copies of `ld.global.u32 %r4, [%rd3]` from the same address. ptxas
hoists the load out of the loop entirely; we approximate that by CSE-
ing the unrolled body.

Conservative gating:
  - Only `ld.global.<t>` (and `ld.shared.<t>` after a corresponding
    store-clear) are CSE candidates.
  - Any `st`, `atom`, `red`, `bar`, `membar`, `fence`, `cp` between
    two loads invalidates ALL cached entries (we don't try to track
    address aliasing — the cost of being wrong is silent miscompile).
  - Any write to a register that was the BASE of a cached load
    invalidates that one entry (the address now points elsewhere).
  - Any write to the destination register of a cached load
    invalidates that one entry (we'd be reading a stale value).

The transform replaces the redundant load with a `mov.<t> %dest,
%cached_dest`. Downstream copy propagation / DCE can collapse the
mov, but the load itself is gone and the memory traffic is saved.
"""
from __future__ import annotations

from copy import copy
from typing import Optional

from ..ir import Function, Instruction, MemOp, RegOp


# Operations that invalidate all cached loads (memory side effects).
_INVALIDATE_ALL = {
    "st", "atom", "red", "bar", "barrier", "membar", "fence",
    "cp", "mbarrier",
}


def _ld_address_key(inst: Instruction) -> Optional[tuple[str, str, str, int]]:
    """Return (space, type, base_reg, offset) if `inst` is a CSE-able
    `ld.<space>.<type>` from a register-base MemOp; else None.
    """
    if inst.op != "ld":
        return None
    if not inst.types or len(inst.types) < 2:
        return None
    space = inst.types[0]
    if space not in ("global", "shared", "const", "param"):
        return None
    if inst.pred is not None:
        return None  # predicated ld — out of scope
    if inst.mods:
        return None
    if len(inst.srcs) != 1 or not isinstance(inst.srcs[0], MemOp):
        return None
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return None
    mem = inst.srcs[0]
    return (space, inst.types[1], mem.base, mem.offset)


def _cse_block(instructions: list[Instruction]) -> int:
    """CSE redundant loads in a single block. Returns count of
    redundant loads converted to movs."""
    cache: dict[tuple[str, str, str, int], str] = {}  # key → dest_name
    n_cse = 0
    new_instrs: list[Instruction] = []
    for inst in instructions:
        # Side-effecting / barrier ops invalidate the whole cache.
        if inst.op in _INVALIDATE_ALL:
            cache.clear()
            new_instrs.append(inst)
            continue

        # Try to CSE this load.
        key = _ld_address_key(inst)
        if key is not None and key in cache:
            cached_reg = cache[key]
            if cached_reg == inst.dest.name:
                # Redundant load to the same dest register — the
                # value is already there from the cached load, just
                # drop this instruction entirely.
                n_cse += 1
                continue
            # Different dest — replace with mov from cached register.
            mov = Instruction(
                op="mov",
                types=[inst.types[1]],
                dest=inst.dest,
                srcs=[RegOp(cached_reg)],
                pred=None,
                neg=False,
                mods=[],
            )
            new_instrs.append(mov)
            # Both `cached_reg` and `inst.dest` now hold the same
            # value; cache stays the same (we keep the original
            # `cached_reg` as the canonical name).
            n_cse += 1
            continue

        # If this is a CSE-candidate load that's the first time we've
        # seen this address, FIRST run the invalidation step (so any
        # prior cache entry whose dest reg gets clobbered by this
        # load's write disappears), THEN record the new entry. Order
        # matters — the invalidation must not delete the entry we're
        # about to add.
        if inst.dest is not None and isinstance(inst.dest, RegOp):
            written = inst.dest.name
            for k in list(cache.keys()):
                _, _, base, _ = k
                if base == written or cache[k] == written:
                    del cache[k]

        if key is not None:
            cache[key] = inst.dest.name

        new_instrs.append(inst)

    if n_cse > 0:
        instructions[:] = new_instrs
    return n_cse


def run_function(fn: Function) -> int:
    total = 0
    for bb in fn.blocks:
        total += _cse_block(bb.instructions)
    return total


def run(module) -> int:
    total = 0
    for fn in module.functions:
        total += run_function(fn)
    return total
