"""
m31_mod_fast_path — Mersenne reduction for `rem.u64 %d, %x, 2^31 - 1`.

Recognizes the M31 modular-reduction shape that appears in FORGE's NTT
kernels (m31_scale et al.) and rewrites it to a constant-time
shift/and/add/conditional-subtract sequence.

Without this pass, isel falls through to the generic bit-serial divide
loop (`_select_div_rem_u64`), which emits ~460 SASS instructions per
`rem.u64`.  ptxas emits ~10 — Mersenne reduction collapses to:

    hi    = x >> 31
    lo    = x & M31
    sum   = hi + lo                (sum ≤ 2^33)
    hi2   = sum >> 31
    lo2   = sum & M31
    sum2  = hi2 + lo2              (sum2 ≤ 2 * M31)
    if sum2 >= M31: sum2 -= M31
    d = sum2

Bound proof: with x ≤ 2^64 - 1, hi ≤ 2^33 - 1 and lo ≤ M31, so
sum ≤ 2^33 + 2^31 - 2 ≈ 2^33.4.  The second fold yields sum2 with
hi2 ≤ 4 and lo2 ≤ M31, so sum2 ≤ M31 + 4 < 2 * M31 — a single
conditional subtract finalizes.

Recognized forms:
    Direct:    rem.u64 %d, %x, 2147483647
    Indirect:  mov.u64 %const, 2147483647    ; %const has 1 def, 1 use
               rem.u64 %d, %x, %const

The indirect form is necessary because the parser leaves the divisor
as a RegOp when it appeared in the source as a `mov` of M31 — and
imm_propagate's whitelist does not cover `rem` (folding into rem
would mix concerns).  The pass walks the per-block last-def map to
recognize the mov-then-rem shape and rewrite in one step.

Conservative gating:
    - Divisor must be exactly 2147483647 (0x7FFFFFFF, M31).  Other
      Mersenne primes (M61, M89, ...) are out of scope — M31 is unique
      and algorithmically dominant in FORGE's small-field NTT.
    - rem.u64 must be unpredicated.  Predicated rems are skipped; the
      bit-serial divide loop remains valid, just slow.  Composing the
      outer predicate with the inner setp.ge guard requires `and.pred`
      with careful initialization, and the m31_scale shape — where
      the rem appears in straight-line code, before _if_convert
      predicates anything — does not need it.
    - Indirect form: the mov producing the divisor must be unpredicated
      and its dest must have exactly one def + one use (so removing
      the mov is safe).

Pipeline ordering: this pass runs BEFORE _if_convert so that NTT-style
rems in straight-line code are recognized while still unpredicated.
After _if_convert any rem.u64 inside an if-diamond becomes predicated,
which the conservative gate rejects.

Pipeline-toggle: `OPENPTXAS_DISABLE_PASSES=m31_mod_fast_path`.
"""
from __future__ import annotations

from typing import Optional

from ..ir import Function, ImmOp, Instruction, RegOp, VectorRegOp, RegDecl, U64, PRED


M31 = 0x7FFF_FFFF  # 2^31 - 1


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


def _walk_use_counts(fn: Function) -> dict[str, int]:
    counts: dict[str, int] = {}
    for bb in fn.blocks:
        for inst in bb.instructions:
            for src in inst.srcs:
                if isinstance(src, RegOp) and not isinstance(src, VectorRegOp):
                    counts[src.name] = counts.get(src.name, 0) + 1
    return counts


def _is_unpred_unmod(inst: Instruction) -> bool:
    return inst.pred is None and not inst.mods


def _alloc_vreg(fn: Function, base: str, type_=U64) -> str:
    """Allocate a fresh register name with `base` prefix, register the
    decl on `fn`. Counter is per-function to avoid collisions across
    multiple rewrites in the same function."""
    if not hasattr(fn, "_m31_mod_next_id"):
        fn._m31_mod_next_id = 0
    while True:
        n = fn._m31_mod_next_id
        fn._m31_mod_next_id += 1
        candidate = f"%{base}_{n}"
        if not any(rd.names and candidate in rd.names for rd in fn.reg_decls):
            fn.reg_decls.append(
                RegDecl(type=type_, name=candidate.lstrip('%'), count=1))
            return candidate


def _is_m31_mov(inst: Instruction) -> bool:
    """True iff `inst` is an unpredicated `mov.{u64,b64,s64} %r, M31`."""
    if inst.op != "mov" or not _is_unpred_unmod(inst):
        return False
    if not inst.types or inst.types[0] not in ("u64", "b64", "s64"):
        return False
    if not isinstance(inst.dest, RegOp) or isinstance(inst.dest, VectorRegOp):
        return False
    if len(inst.srcs) != 1 or not isinstance(inst.srcs[0], ImmOp):
        return False
    return (inst.srcs[0].value & 0xFFFFFFFFFFFFFFFF) == M31


def _is_m31_rem(inst: Instruction,
                last_def: dict[str, Instruction],
                def_count: dict[str, int],
                use_count: dict[str, int]) -> Optional[Instruction]:
    """If `inst` is `rem.u64 %d, %x, M31` (direct or via single-use mov),
    return the mov instruction to delete (None for direct form), or
    the sentinel `inst` itself when no mov is consumed.

    Returns:
        - inst if direct form (no mov to delete)
        - mov_inst if indirect form (mov should be deleted)
        - None if not a match
    """
    if inst.op != "rem":
        return None
    if not _is_unpred_unmod(inst):
        return None
    if not inst.types or inst.types[0] != "u64":
        return None
    if inst.dest is None or not isinstance(inst.dest, RegOp):
        return None
    if len(inst.srcs) != 2:
        return None
    if not isinstance(inst.srcs[0], RegOp):
        return None

    divisor = inst.srcs[1]
    if isinstance(divisor, ImmOp):
        if (divisor.value & 0xFFFFFFFFFFFFFFFF) == M31:
            return inst   # direct form, sentinel = self
        return None

    if isinstance(divisor, RegOp) and not isinstance(divisor, VectorRegOp):
        # Trace divisor back to a mov in the same block.
        mov = last_def.get(divisor.name)
        if mov is None or not _is_m31_mov(mov):
            return None
        # Conservative: divisor reg must have exactly one def and one
        # use (the rem itself).
        if def_count.get(divisor.name, 0) != 1:
            return None
        if use_count.get(divisor.name, 0) != 1:
            return None
        return mov   # indirect form

    return None


def _build_m31_sequence(fn: Function, rem: Instruction) -> list[Instruction]:
    """Construct the rewrite sequence for `rem.u64 %d, %x, M31`.

    Allocates fresh u64 regs (hi, lo, sum, hi2, lo2, sum2, tmp) and one
    fresh predicate (p_ge).  The conditional final-subtract is expressed
    as an unconditional sub into a scratch + a predicated mov over %d
    so the IADD3/IADD3.X carry chain is never itself predicated — the
    scoreboard does not track the IADD3.X P1 carry-in dependency, so
    keeping the sub unpredicated avoids stale-carry hazards.
    """
    x_reg: RegOp = rem.srcs[0]
    d_reg: RegOp = rem.dest

    hi   = RegOp(_alloc_vreg(fn, "m31_hi"))
    lo   = RegOp(_alloc_vreg(fn, "m31_lo"))
    s1   = RegOp(_alloc_vreg(fn, "m31_sum"))
    hi2  = RegOp(_alloc_vreg(fn, "m31_hi2"))
    lo2  = RegOp(_alloc_vreg(fn, "m31_lo2"))
    s2   = RegOp(_alloc_vreg(fn, "m31_sum2"))
    tmp  = RegOp(_alloc_vreg(fn, "m31_tmp"))
    p_ge = _alloc_vreg(fn, "m31_p", type_=PRED)

    seq = [
        # First fold: x -> hi + lo
        Instruction(op="shr", types=["u64"], dest=hi,  srcs=[x_reg, ImmOp(31)]),
        Instruction(op="and", types=["b64"], dest=lo,  srcs=[x_reg, ImmOp(M31)]),
        Instruction(op="add", types=["u64"], dest=s1,  srcs=[hi, lo]),
        # Second fold: sum -> hi2 + lo2
        Instruction(op="shr", types=["u64"], dest=hi2, srcs=[s1, ImmOp(31)]),
        Instruction(op="and", types=["b64"], dest=lo2, srcs=[s1, ImmOp(M31)]),
        Instruction(op="add", types=["u64"], dest=s2,  srcs=[hi2, lo2]),
        # Unconditional sub — produces tmp = sum2 - M31 regardless.
        # Bound math guarantees sum2 ≤ 2*M31 - 2, so tmp is the correct
        # reduced value when sum2 >= M31; otherwise tmp underflows to a
        # large value but is discarded by the predicated mov below.
        Instruction(op="sub",  types=["u64"], dest=tmp, srcs=[s2, ImmOp(M31)]),
        # Default %d = sum2 (the unreduced value).
        Instruction(op="mov",  types=["u64"], dest=d_reg, srcs=[s2]),
        # Override %d <- tmp when sum2 >= M31.
        Instruction(op="setp", types=["ge", "u64"],
                    dest=RegOp(p_ge), srcs=[s2, ImmOp(M31)]),
        Instruction(op="mov",  types=["u64"], dest=d_reg, srcs=[tmp],
                    pred=p_ge, neg=False),
    ]
    return seq


def run_function(fn: Function) -> int:
    """Recognize M31-rem patterns and rewrite. Returns the number of
    rem.u64 instructions converted."""
    def_count = _walk_def_counts(fn)
    use_count = _walk_use_counts(fn)

    n_rewrites = 0

    for bb in fn.blocks:
        # Per-block last-def map (PTX is largely SSA but we still scope
        # to the block to avoid cross-block reasoning about liveness).
        last_def: dict[str, Instruction] = {}
        new_instrs: list[Instruction] = []
        drop_ids: set[int] = set()

        for inst in bb.instructions:
            mov_to_drop = _is_m31_rem(inst, last_def, def_count, use_count)
            if mov_to_drop is None:
                # Update last_def AFTER the check (so a rem can read its
                # divisor's prior def, not its own dest).
                d = inst.dest
                if isinstance(d, RegOp) and not isinstance(d, VectorRegOp):
                    last_def[d.name] = inst
                new_instrs.append(inst)
                continue

            # Match. Build replacement.
            if mov_to_drop is not inst:
                # Indirect form: drop the mov from the new list. Since
                # we're building new_instrs in order, the mov is already
                # appended; mark its id for filtering below. Cleaner:
                # remove from new_instrs directly.
                drop_ids.add(id(mov_to_drop))

            new_instrs.extend(_build_m31_sequence(fn, inst))
            n_rewrites += 1

            # Update last_def from the rewrite's writes so subsequent
            # instructions see them.
            for replaced in new_instrs[-10:]:
                rd = replaced.dest
                if isinstance(rd, RegOp) and not isinstance(rd, VectorRegOp):
                    last_def[rd.name] = replaced

        if drop_ids:
            new_instrs = [i for i in new_instrs if id(i) not in drop_ids]
        bb.instructions = new_instrs

    return n_rewrites


def run(module) -> int:
    total = 0
    for fn in module.functions:
        total += run_function(fn)
    return total
