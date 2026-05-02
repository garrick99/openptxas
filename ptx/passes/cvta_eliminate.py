"""
cvta-eliminate: drop redundant cvta address-space conversions.

Phase 14 of the merkle_hash_leaves bloat fix.

On SM_120 (and SM_70+), pointers are unified — `cvta.to.<space>.<t>`
and `cvta.<space>.<t>` between generic-space and a specific space are
no-ops at the hardware level when the source is already a register.
ptxas emits 0 instructions for them; openptxas's isel cvta handler
lowers each to a `MOV R{d_lo}, R{s_lo}` (and a second MOV for u64),
inflating SASS line count for kernels with many call sites against a
common base pointer.

Algorithm:
  1. Walk each function once.  Collect every unpredicated `cvta.<...>`
     whose dest is a plain RegOp and source is a plain RegOp, where
     the type list mentions an eligible address space (global / shared
     / local / param / const).
  2. Build an alias map `dest -> source`, resolving transitively when
     a cvta's source itself was a previous cvta dest.
  3. Rewrite every subsequent register/MemOp-base use of the aliased
     dest to the resolved source.
  4. Drop the eligible cvta instructions.

Skipped (conservative):
  - Predicated cvta — may not always execute.
  - cvta whose source is a LabelOp (e.g. `cvta.shared.u64 %rd, label`)
    — that materializes a smem offset, not an identity.
  - cvta whose dest is a vector reg or whose source is anything other
    than a plain register.

Pipeline-toggle: disable via OPENPTXAS_DISABLE_PASSES=cvta_eliminate.
"""
from __future__ import annotations

from ..ir import Function, Instruction, MemOp, Module, RegOp, VectorRegOp


# Address spaces eligible for elimination on SM_120's unified address
# space.  cvta.<space>.<type> with a register source is identity.
_ELIGIBLE_SPACES = frozenset({"global", "shared", "local", "param", "const"})


def _is_simple_reg(op) -> bool:
    return isinstance(op, RegOp) and not isinstance(op, VectorRegOp)


def _eligible_cvta(inst: Instruction) -> bool:
    """True if `inst` is an eligible identity cvta we can drop."""
    if inst.op != "cvta":
        return False
    if inst.pred is not None:
        return False
    if not any(t in _ELIGIBLE_SPACES for t in inst.types):
        return False
    if not _is_simple_reg(inst.dest):
        return False
    if len(inst.srcs) != 1 or not _is_simple_reg(inst.srcs[0]):
        return False
    return True


def _rewrite_uses(fn: Function, alias_map: dict[str, str]) -> int:
    """Rewrite RegOp / MemOp-base reads of any alias-mapped dest to the
    canonical source.  Skips the cvta instructions themselves (their
    own srcs are the canonical sources already)."""
    if not alias_map:
        return 0
    n = 0
    for bb in fn.blocks:
        for inst in bb.instructions:
            if (inst.op == "cvta"
                    and _is_simple_reg(inst.dest)
                    and inst.dest.name in alias_map):
                continue
            new_srcs = []
            for src in inst.srcs:
                if isinstance(src, RegOp) and not isinstance(src, VectorRegOp):
                    new_name = alias_map.get(src.name)
                    if new_name is not None and new_name != src.name:
                        new_srcs.append(RegOp(new_name))
                        n += 1
                        continue
                if isinstance(src, MemOp) and src.base.startswith('%'):
                    new_base = alias_map.get(src.base)
                    if new_base is not None and new_base != src.base:
                        new_srcs.append(MemOp(new_base, src.offset))
                        n += 1
                        continue
                new_srcs.append(src)
            inst.srcs = new_srcs
    return n


def run_function(fn: Function) -> int:
    """Run cvta-eliminate on a single function.  Returns # cvtas dropped."""
    alias_map: dict[str, str] = {}
    for bb in fn.blocks:
        for inst in bb.instructions:
            if not _eligible_cvta(inst):
                continue
            d = inst.dest.name
            s = inst.srcs[0].name
            # Resolve transitively in case the source was itself the dest
            # of a prior cvta we already aliased.
            seen: set[str] = set()
            while s in alias_map and s not in seen:
                seen.add(s)
                s = alias_map[s]
            alias_map[d] = s

    if not alias_map:
        return 0

    _rewrite_uses(fn, alias_map)

    dropped = 0
    drop_dests = set(alias_map.keys())
    for bb in fn.blocks:
        kept = []
        for inst in bb.instructions:
            if (inst.op == "cvta"
                    and _is_simple_reg(inst.dest)
                    and inst.dest.name in drop_dests
                    and _eligible_cvta(inst)):
                dropped += 1
                continue
            kept.append(inst)
        bb.instructions = kept
    return dropped


def run(mod: Module) -> int:
    total = 0
    for fn in mod.functions:
        total += run_function(fn)
    return total
