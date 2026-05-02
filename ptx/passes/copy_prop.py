"""
Copy-propagate: substitute reg-reg copies of the form `mov.<t> %d, %s`
into eligible consumer source operands, then DCE the now-dead movs.

Phase 17 of the merkle_hash_leaves bloat fix.  Phase 8's `imm_propagate`
folds 0-immediates into add/sub/or/xor consumers, after which Phase 9's
`trivial_fold` rewrites `add.<t> %d, %a, 0` -> `mov.<t> %d, %a`.  These
freshly-created reg-reg movs accumulate as working-slot copies in the
post-unroll body of merkle_hash_leaves.  This pass finishes the cleanup
ptxas's interference-graph coalescer would do — substituting %a for %d
at every use and dropping the mov.

Algorithm:
  1. Walk each function once and assign a global linear position to
     every instruction.  Build def_positions[%r] = list of positions
     where %r is written.
  2. Build copy_defs[%d] = (mov_pos, %s, mov_inst) for every
     unpredicated, unmodded `mov.<t> %d, %s` whose dest %d has
     exactly one def in the function and whose source %s is a plain
     virtual register (not a special PTX state register, not a
     vector reg).
  3. For every non-mov, non-memory-op instruction at linear position
     p, walk its source operands.  For each RegOp source whose name
     %d is in copy_defs and whose mov was at position mov_pos:
       - skip if any def of %s sits at position k with mov_pos < k < p
         (i.e. %s was rewritten between the mov and this use).
       - skip if the consumer instruction's first-type bit-width
         doesn't match the mov's bit-width (no cross-width fold).
       - otherwise: substitute %s for %d at this src position.
  4. After all substitutions, drop every mov in copy_defs whose dest
     no longer has any readers.

Conservative gating:
  - Single definition of %d (the mov is the only writer).
  - Source %s is not a special PTX state register (%tid, %ctaid, ...);
    those need to flow through the S2R lowering path with their
    original name.
  - Skip predicated movs (the value may not be live at every consumer).
  - Skip movs with modifiers.
  - Skip same-name movs (`mov %r, %r` — degenerate, leave to DCE).
  - Skip memory-address operand positions: don't propagate into ld/
    st/ldg/stg/atom/red sources at all (their address operands have
    register-pair / bank-encoding requirements that copy-prop can't
    safely reason about).
  - Same-type-width only.

Pipeline-toggle: disable via OPENPTXAS_DISABLE_PASSES=copy_prop.
"""
from __future__ import annotations

from typing import Optional

from ..ir import Function, ImmOp, Instruction, RegOp, VectorRegOp


_SPECIAL_REG_PREFIXES = (
    "%tid", "%ntid", "%ctaid", "%nctaid",
    "%warpid", "%nwarpid", "%laneid", "%lanemask",
    "%clock", "%clock64", "%smid", "%nsmid",
    "%gridid", "%pm", "%envreg",
)


# ld/st-family ops are skipped entirely from substitution — their
# source positions encode address registers (lo+hi pair semantics for
# u64) or have specific isel constraints we don't want copy-prop to
# perturb.
_MEMORY_OPS = frozenset({
    "ld", "st", "ldu", "ldg", "ldc", "stg",
    "atom", "red",
    "tex", "sust", "suld", "suq", "txq",
    "cp",  # cp.async.*
    "wmma", "mma",
    "bar", "membar", "fence",
    "prefetch", "prefetchu",
})


def _bitwidth_of_type(t: str) -> Optional[int]:
    if not t:
        return None
    if t[0] in ("u", "s", "b", "f"):
        try:
            return int(t[1:])
        except ValueError:
            return None
    return None


def _is_special_reg(name: str) -> bool:
    return any(name.startswith(p) for p in _SPECIAL_REG_PREFIXES)


def _is_simple_reg_reg_mov(inst: Instruction) -> Optional[tuple[str, str, int]]:
    """If `inst` is an unpredicated, unmodded `mov.<t> %d, %s` reg-reg
    copy with both operands plain (non-vector) RegOps, return
    (dest_name, src_name, bit_width).  Returns None otherwise."""
    if inst.op != "mov":
        return None
    if inst.pred is not None or inst.mods or inst.neg:
        return None
    if inst.dest is None or len(inst.srcs) != 1:
        return None
    if not isinstance(inst.dest, RegOp) or isinstance(inst.dest, VectorRegOp):
        return None
    src = inst.srcs[0]
    if not isinstance(src, RegOp) or isinstance(src, VectorRegOp):
        return None
    if not inst.types:
        return None
    width = _bitwidth_of_type(inst.types[0])
    if width is None:
        return None
    if inst.dest.name == src.name:
        return None  # degenerate self-mov, leave for DCE
    if _is_special_reg(src.name):
        return None  # skip %tid/%ctaid/... — they need S2R lowering
    return (inst.dest.name, src.name, width)


def _walk_def_positions(linear: list[tuple[int, Instruction]]) -> dict[str, list[int]]:
    """Return {reg_name: [position, ...]} for every register that is
    written somewhere in the linear instruction stream."""
    defs: dict[str, list[int]] = {}
    for pos, inst in linear:
        d = inst.dest
        if d is None:
            continue
        if isinstance(d, VectorRegOp):
            for r in (d.regs or ()):
                defs.setdefault(r, []).append(pos)
        elif isinstance(d, RegOp):
            defs.setdefault(d.name, []).append(pos)
    return defs


def _linearize(fn: Function) -> list[tuple[int, Instruction]]:
    out: list[tuple[int, Instruction]] = []
    p = 0
    for bb in fn.blocks:
        for inst in bb.instructions:
            out.append((p, inst))
            p += 1
    return out


def _has_def_in_open_range(positions: list[int], lo: int, hi: int) -> bool:
    """True if any value in `positions` lies in (lo, hi) (exclusive)."""
    for k in positions:
        if lo < k < hi:
            return True
    return False


def run_function(fn: Function) -> int:
    """Run copy_prop on a single function.  Returns total number of
    src-operand substitutions performed."""
    linear = _linearize(fn)
    if not linear:
        return 0

    def_positions = _walk_def_positions(linear)

    # Build copy_defs only for movs whose dest has exactly one def
    # (the mov itself) — guarantees no other writer of %d exists.
    copy_defs: dict[str, tuple[int, str, int, Instruction]] = {}
    for pos, inst in linear:
        sm = _is_simple_reg_reg_mov(inst)
        if sm is None:
            continue
        dest_name, src_name, width = sm
        if len(def_positions.get(dest_name, ())) != 1:
            continue
        copy_defs[dest_name] = (pos, src_name, width, inst)

    if not copy_defs:
        return 0

    n_subs = 0
    for pos, inst in linear:
        if inst.op == "mov":
            continue
        if inst.op in _MEMORY_OPS:
            continue
        consumer_width: Optional[int] = None
        if inst.types:
            consumer_width = _bitwidth_of_type(inst.types[0])

        for i, src in enumerate(inst.srcs):
            if not isinstance(src, RegOp) or isinstance(src, VectorRegOp):
                continue
            rec = copy_defs.get(src.name)
            if rec is None:
                continue
            mov_pos, src_name, mov_width, _mov_inst = rec
            # Width-match guard: don't fold a 32-bit mov source into a
            # 64-bit consumer (or vice versa).  Skip when consumer
            # width is unknown (defensive — better to leave in place).
            if consumer_width is not None and consumer_width != mov_width:
                continue
            # Liveness check: ensure %s wasn't redefined between the
            # mov and this use.
            src_defs = def_positions.get(src_name, ())
            if _has_def_in_open_range(src_defs, mov_pos, pos):
                continue
            # Substitute %s for %d at this src position.
            inst.srcs[i] = RegOp(src_name)
            n_subs += 1

    if n_subs == 0:
        return 0

    # DCE: drop movs in copy_defs whose dest is no longer read.
    used: set[str] = set()
    for _, inst in linear:
        for src in inst.srcs:
            if isinstance(src, VectorRegOp):
                for r in (src.regs or ()):
                    used.add(r)
            elif isinstance(src, RegOp):
                used.add(src.name)
        # Predicate registers are also "uses".
        if inst.pred is not None:
            used.add(inst.pred)

    dead_ids: set[int] = set()
    for dest_name, (_, _, _, mov_inst) in copy_defs.items():
        if dest_name not in used:
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
