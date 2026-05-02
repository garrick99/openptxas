"""
Phase 22 + Phase 24: strength reduction + slot-0 hoist for chained
multiply-add address arithmetic.

Recognizes the FORGE-emitted pattern (per-slot store address compute):

    mul.lo.u64  %M1, %X, K1_imm        (M1 may be multi-used; preserved)
    add.u64     %A,  %M1, Y_imm        (A single-use; consumed by next mul)
    mul.lo.u64  %M2, %A,  K2_imm       (M2 single-use; consumed by addr add)
    add.u64     %F,  %B,  %M2          (F single-use as MemOp base)
    st.global   [%F], data             (single store)

and rewrites the SECOND mul + the consumer chain to:

    mul.lo.u64  %M_new, %X, (K1*K2)_imm   (NEW combined mul, fresh vreg)
    add.u64     %F,  %B, %M_new           (consumer; %M2 -> %M_new)
    add.u64     %F_off, %F, (Y*K2)        (NEW offset add)
    st.global   [%F_off], data            (consumer base bumped)

Mathematically equivalent: (X*K1 + Y) * K2 == X*(K1*K2) + Y*K2.

Downstream the chain collapses to 2 SASS instructions:

  IMAD.WIDE.U32 R_F, R_X, K1*K2, R_B    (Phase 19v2 fuses mul + add)
  STG.E [R_F + Y*K2], R_data             (analyze_addr_offset_fold folds
                                          F_off += imm into MemOp offset)

vs ~5+ SASS instructions before strength reduction.

The original `mul.lo.u64 %M1, %X, K1_imm` is preserved if it has any
remaining uses (typical for FORGE: M1 = idx*8 is reused across N
unrolled per-slot store chains).  The transform never deletes M1 — it
only DELETES the leading `add %A` and rewrites the second `mul %M2`.

# Conservative gating (correctness)

- %A is single-defined AND single-used (only by `mul %M2`).
- %M2 is single-defined AND single-used (only by `add %F`).
- %F is single-defined AND single-used (only as MemOp.base in store).
- K1, K2, Y must be 64-bit immediates or `mov.u64 %r, IMM` mov-imms
  whose only use is the chain instruction.
- K1 * K2 must not overflow u64 (computed % 2^64; rejected on overflow).
- |Y * K2| must fit in 24-bit signed (the LDG.E/STG.E offset range
  enforced by analyze_addr_offset_fold).
- All four chain instructions are in the same BasicBlock and
  immediately consecutive (no intervening writes to %M1, %X, or %B).
- The store and its address-add are in the same BB as the chain.

# Pipeline placement

After: unroll, cvta_eliminate, imm_propagate (so K1/K2/Y mov-imms are
already folded into ImmOps where the propagator allowed it).

Before: add3_chain_reduce, load_cse, mul_imm_chain_fold, common_mul_sum
(so they see the rewritten chain and don't double-fold).  Also before
the IMAD.WIDE-fuse analysis (which runs in pipeline.compile_function
AFTER all PTX-IR passes finish).

Toggle: OPENPTXAS_DISABLE_PASSES=mul_distribute.
"""
from __future__ import annotations

from typing import Optional

from ..ir import Function, ImmOp, Instruction, MemOp, RegOp, VectorRegOp


_INT64_TYPES = ("u64", "s64", "b64")


def _is_int64_typed(inst: Instruction) -> bool:
    """True iff `inst` has any 64-bit int type qualifier.

    PTX dotted forms vary by op:
      - `add.u64`     -> types = ['u64']
      - `mul.lo.u64`  -> types = ['lo', 'u64']    (the 'lo'/'hi' is in types[0])
      - `mov.u64`     -> types = ['u64']
    Check the whole list to handle all forms uniformly.
    """
    return bool(inst.types) and any(t in _INT64_TYPES for t in inst.types)


def _has_lo_modifier(inst: Instruction) -> bool:
    """For `mul.lo.<t>` PTX, the parser places 'lo' in types (not mods).
    Returns True if the instruction has the .lo qualifier."""
    return ("lo" in (inst.types or [])) or ("lo" in (inst.mods or []))


def _walk_def_use_counts(fn: Function) -> tuple[dict[str, int], dict[str, int],
                                                  dict[str, Instruction]]:
    """Return (def_count, use_count, def_instr) for every vreg in `fn`.

    A vreg's "use" includes RegOp source operands and MemOp.base
    references.  VectorRegOp registers count toward def_count for each
    component but are not unpacked into use_count by this pass (the
    address-arithmetic patterns we target only touch scalar u64s).
    """
    def_count: dict[str, int] = {}
    use_count: dict[str, int] = {}
    def_instr: dict[str, Instruction] = {}
    for bb in fn.blocks:
        for inst in bb.instructions:
            d = inst.dest
            if isinstance(d, VectorRegOp):
                for r in (d.regs or ()):
                    def_count[r] = def_count.get(r, 0) + 1
                    def_instr[r] = inst
            elif isinstance(d, RegOp):
                def_count[d.name] = def_count.get(d.name, 0) + 1
                def_instr[d.name] = inst
            for src in (inst.srcs or []):
                if isinstance(src, RegOp) and not isinstance(src, VectorRegOp):
                    use_count[src.name] = use_count.get(src.name, 0) + 1
                elif isinstance(src, MemOp):
                    base = src.base
                    if isinstance(base, str) and base.startswith('%'):
                        use_count[base] = use_count.get(base, 0) + 1
    return def_count, use_count, def_instr


def _resolve_u64_imm(op, def_instr: dict[str, Instruction],
                     def_count: dict[str, int],
                     use_count: dict[str, int],
                     allow_pred: Optional[tuple[Optional[str], bool]] = None
                     ) -> Optional[tuple[int, Optional[Instruction]]]:
    """If `op` is a u64-resolvable constant, return (value, dead_mov_or_None).

    - ImmOp -> direct value, no dead instruction.
    - RegOp whose single def is `mov.u64 %r, IMM` AND whose only use is
      the consumer-chain instruction we are folding -> (value, mov_instr).

    `allow_pred` (optional): when the mov is predicated, accept it only
    if its predicate matches the given (pred_name, neg) tuple.  Used to
    fold movs that share the chain's predicate after if-conversion.
    """
    if isinstance(op, ImmOp):
        return (op.value & 0xFFFFFFFFFFFFFFFF, None)
    if isinstance(op, RegOp) and not isinstance(op, VectorRegOp):
        df = def_instr.get(op.name)
        if df is None:
            return None
        if def_count.get(op.name, 0) != 1:
            return None
        if use_count.get(op.name, 0) != 1:
            return None
        if df.op != "mov":
            return None
        if not df.types or df.types[0] not in _INT64_TYPES:
            return None
        if df.mods:
            return None
        if df.pred is not None:
            if allow_pred is None:
                return None
            if (df.pred, df.neg) != allow_pred:
                return None
        if not df.srcs or not isinstance(df.srcs[0], ImmOp):
            return None
        return (df.srcs[0].value & 0xFFFFFFFFFFFFFFFF, df)
    return None


# 24-bit signed range for STG.E / LDG.E byte offset fold (matches
# analyze_addr_offset_fold's bound).
_OFF_MAX = (1 << 23) - 1
_OFF_MIN = -(1 << 23)


def _is_first_mul(inst: Instruction) -> Optional[tuple[str, RegOp]]:
    """Match `mul.lo.u64 %M1, %X, K1` and return (M1_name, X_reg).
    K1 is resolved at the caller (RegOp or ImmOp); types must be int64.
    The first mul is allowed to be unpredicated (it typically lives
    above the chain's if-converted region, computing %M1 unconditionally).
    """
    if inst.op != "mul" or not _has_lo_modifier(inst):
        return None
    if not _is_int64_typed(inst):
        return None
    if not isinstance(inst.dest, RegOp) or isinstance(inst.dest, VectorRegOp):
        return None
    if len(inst.srcs or []) != 2:
        return None
    x_op = inst.srcs[0]
    if not isinstance(x_op, RegOp) or isinstance(x_op, VectorRegOp):
        return None
    return (inst.dest.name, x_op)


def _is_add_match(inst: Instruction, m1_name: str) -> Optional[str]:
    """Match `[@pred] add.u64 %A, %M1, Y` (M1 in either source position).
    Returns A_name or None.  Predicate is permitted (the if-converted
    chain shares one); Y resolution is done by the caller.
    """
    if inst.op != "add" or not _is_int64_typed(inst):
        return None
    if inst.mods:
        return None
    if not isinstance(inst.dest, RegOp) or isinstance(inst.dest, VectorRegOp):
        return None
    if len(inst.srcs or []) != 2:
        return None
    a, b = inst.srcs[0], inst.srcs[1]
    if isinstance(a, RegOp) and a.name == m1_name:
        return inst.dest.name
    if isinstance(b, RegOp) and b.name == m1_name:
        return inst.dest.name
    return None


def _y_operand(add_inst: Instruction, m1_name: str):
    """Return the non-M1 source of the add (the Y operand)."""
    a, b = add_inst.srcs[0], add_inst.srcs[1]
    if isinstance(a, RegOp) and a.name == m1_name:
        return b
    return a


def _is_second_mul(inst: Instruction, a_name: str) -> Optional[tuple[str, object]]:
    """Match `[@pred] mul.lo.u64 %M2, %A, K2` and return (M2_name, K2_op)."""
    if inst.op != "mul" or not _has_lo_modifier(inst):
        return None
    if not _is_int64_typed(inst):
        return None
    if not isinstance(inst.dest, RegOp) or isinstance(inst.dest, VectorRegOp):
        return None
    if len(inst.srcs or []) != 2:
        return None
    a_op = inst.srcs[0]
    if not isinstance(a_op, RegOp) or a_op.name != a_name:
        return None
    return (inst.dest.name, inst.srcs[1])


def _is_consumer_add(inst: Instruction, m2_name: str) -> Optional[tuple[str, str]]:
    """Match `[@pred] add.u64 %F, %B, %M2`.  Returns (F_name, B_name) or None.
    %B must be a plain RegOp (the base pointer).  Predicate permitted.
    """
    if inst.op != "add" or not _is_int64_typed(inst):
        return None
    if inst.mods:
        return None
    if not isinstance(inst.dest, RegOp) or isinstance(inst.dest, VectorRegOp):
        return None
    if len(inst.srcs or []) != 2:
        return None
    a, b = inst.srcs[0], inst.srcs[1]
    if isinstance(a, RegOp) and a.name == m2_name and isinstance(b, RegOp):
        return (inst.dest.name, b.name)
    if isinstance(b, RegOp) and b.name == m2_name and isinstance(a, RegOp):
        return (inst.dest.name, a.name)
    return None


def _is_global_store_of(inst: Instruction, base_name: str) -> bool:
    """Match `[@pred] st.global.<t> [%F], data` whose memop base is %F."""
    if inst.op != "st":
        return False
    if "global" not in (inst.types or []):
        return False
    if not inst.srcs or not isinstance(inst.srcs[0], MemOp):
        return False
    base = inst.srcs[0].base
    if not isinstance(base, str):
        return False
    n = base if base.startswith('%') else f'%{base}'
    return n == base_name


def _pred_match(a: Instruction, b: Instruction) -> bool:
    """True iff two instructions share the same predicate (incl. None)."""
    return a.pred == b.pred and a.neg == b.neg


def _alloc_vreg(fn: Function, base: str) -> str:
    """Allocate a fresh u64 vreg name with the given base prefix.
    Maintains a per-function counter on `fn` to avoid collisions.
    """
    if not hasattr(fn, "_mul_distribute_next_id"):
        fn._mul_distribute_next_id = 0
    while True:
        n = fn._mul_distribute_next_id
        fn._mul_distribute_next_id += 1
        candidate = f"%{base}_md{n}"
        if not any(rd.names and candidate in rd.names for rd in fn.reg_decls):
            # Add a .reg .u64 declaration for the new vreg so the
            # downstream regalloc / parser do not balk.
            from ..ir import RegDecl, U64
            fn.reg_decls.append(RegDecl(type=U64, name=candidate.lstrip('%'), count=1))
            return candidate


def _writes(inst: Instruction, name: str) -> bool:
    d = inst.dest
    if isinstance(d, VectorRegOp):
        return name in (d.regs or ())
    return isinstance(d, RegOp) and d.name == name


def _try_rewrite_block_with_cse(fn: Function, bb,
                                  cse: dict[tuple, tuple[str, str]],
                                  def_count: dict[str, int],
                                  use_count: dict[str, int],
                                  def_instr: dict[str, Instruction]) -> int:
    """Walk a single block, find chain leading-add patterns, and rewrite.

    For each `add.u64 %A, %M1, Y` in the block where:
      * %M1 is defined ANYWHERE by `mul.lo.u64 %M1, %X, K1_imm`
      * %A and %M1's downstream chain meet the gating constraints
    rewrite the chain.

    The leading mul %M1 is preserved (we don't delete it — it likely has
    other uses beyond this chain in the FORGE per-slot pattern).
    The `cse` dict is shared across calls so subsequent passes (e.g.
    slot-0 sweep) see the same shared-mul/base-add allocations.
    Returns the number of chains rewritten.

    Intra-block CSE: when multiple chain rewrites within the same block
    share (X, K_combined, B, predicate), they reuse a single combined
    `mul %M_shared, %X, K1*K2 + add %F_shared, %B, %M_shared` and only
    emit a per-slot offset add.  This collapses the 7-fold duplication
    in the FORGE merkle per-slot store pattern to a single shared
    IMAD.WIDE.U32 plus N STG.Es with constant offset.
    """
    n_rewrites = 0
    i = 0
    while i < len(bb.instructions):
        instrs = bb.instructions
        cand = instrs[i]

        # Pattern starts at the LEADING ADD: `[@pred] add.u64 %A, %M1, Y_imm`.
        # The chain may be predicated (post if-conversion); ALL chain
        # instructions must share the SAME predicate / negation, and
        # the new instructions inherit that predicate.
        if cand.op != "add" or not _is_int64_typed(cand):
            i += 1
            continue
        if cand.mods:
            i += 1
            continue
        if not isinstance(cand.dest, RegOp) or isinstance(cand.dest, VectorRegOp):
            i += 1
            continue
        chain_pred = cand.pred
        chain_neg = cand.neg
        chain_pred_tuple = (chain_pred, chain_neg)
        if len(cand.srcs or []) != 2:
            i += 1
            continue
        a_name = cand.dest.name
        if def_count.get(a_name, 0) != 1 or use_count.get(a_name, 0) != 1:
            i += 1
            continue

        # Identify which source is %M1 (a u64 mul result) and which is Y.
        m1_op = None
        y_op = None
        for src in cand.srcs:
            if isinstance(src, RegOp) and not isinstance(src, VectorRegOp):
                df = def_instr.get(src.name)
                if df is None:
                    continue
                if df.op == "mul" and _has_lo_modifier(df) and _is_int64_typed(df):
                    m1_op = src
            else:
                pass
        if m1_op is None:
            # Maybe BOTH srcs are RegOps where one happens to also resolve;
            # do a second pass to assign Y.
            i += 1
            continue
        # Y is the OTHER source.
        y_op = cand.srcs[0] if cand.srcs[1] is m1_op or (
                isinstance(cand.srcs[1], RegOp) and cand.srcs[1].name == m1_op.name) \
              else cand.srcs[1]
        if y_op is m1_op or (isinstance(y_op, RegOp) and y_op.name == m1_op.name):
            i += 1
            continue

        m1_def = def_instr[m1_op.name]
        # m1_def: `mul.lo.u64 %M1, %X, K1`.  m1_def must NOT be predicated
        # (we read %X unconditionally in the new mul; if m1_def were
        # guarded, %X*K1*K2 might read undefined inputs in code paths
        # where the original guard didn't fire).  In FORGE PTX, the outer
        # mul %M1 = %X*K1 is computed unconditionally above the
        # if-converted region, so this is the typical case anyway.
        if m1_def.pred is not None:
            i += 1
            continue
        if (len(m1_def.srcs or []) != 2
                or not isinstance(m1_def.srcs[0], RegOp)
                or isinstance(m1_def.srcs[0], VectorRegOp)):
            i += 1
            continue
        x_reg = m1_def.srcs[0]
        k1 = _resolve_u64_imm(m1_def.srcs[1], def_instr, def_count, use_count,
                               allow_pred=None)
        if k1 is None:
            i += 1
            continue
        K1, k1_dead = k1
        # K1's mov is shared by m1_def's mul; if dead, removing it would
        # break m1_def itself (which we KEEP).  So always keep the K1 mov.
        # We don't add it to dead_inst_ids.
        if K1 == 0:
            i += 1
            continue

        # Resolve Y.  If Y is a mov-imm, accept either an unpredicated
        # mov OR one that shares the chain's predicate.
        y = _resolve_u64_imm(y_op, def_instr, def_count, use_count,
                              allow_pred=chain_pred_tuple)
        if y is None:
            i += 1
            continue
        Y, y_dead = y

        # The next instruction (or one mov ahead) must be the second mul.
        # The second mul must share the chain predicate.
        if i + 1 >= len(instrs):
            i += 1
            continue
        mul2_inst = instrs[i + 1]
        second_match = _is_second_mul(mul2_inst, a_name)
        if second_match is None:
            # Allow a single intervening mov-imm (the K2 mov, if not yet
            # propagated).  The mov's predicate must match the chain.
            if (mul2_inst.op == "mov"
                    and _is_int64_typed(mul2_inst)
                    and isinstance(mul2_inst.dest, RegOp)
                    and not isinstance(mul2_inst.dest, VectorRegOp)
                    and not mul2_inst.mods
                    and use_count.get(mul2_inst.dest.name, 0) == 1
                    and (mul2_inst.pred is None
                         or (mul2_inst.pred, mul2_inst.neg) == chain_pred_tuple)
                    and i + 2 < len(instrs)):
                second_match = _is_second_mul(instrs[i + 2], a_name)
                if second_match is not None:
                    mul2_inst = instrs[i + 2]
            if second_match is None:
                i += 1
                continue
        if not _pred_match(mul2_inst, cand):
            i += 1
            continue
        m2_name, k2_op = second_match
        k2 = _resolve_u64_imm(k2_op, def_instr, def_count, use_count,
                               allow_pred=chain_pred_tuple)
        if k2 is None:
            i += 1
            continue
        K2, k2_dead = k2
        if K2 == 0:
            i += 1
            continue
        if def_count.get(m2_name, 0) != 1 or use_count.get(m2_name, 0) != 1:
            i += 1
            continue

        # Overflow check on K1 * K2.
        full_product = K1 * K2
        if full_product != (full_product & 0xFFFFFFFFFFFFFFFF):
            i += 1
            continue
        K_combined = full_product & 0xFFFFFFFFFFFFFFFF
        if K_combined == 0 or (K_combined >> 32) != 0:
            i += 1
            continue
        Y_offset = (Y * K2) & 0xFFFFFFFFFFFFFFFF
        if Y_offset >= (1 << 63):
            signed_off = Y_offset - (1 << 64)
        else:
            signed_off = Y_offset
        if not (_OFF_MIN <= signed_off <= _OFF_MAX):
            i += 1
            continue

        # Find the consumer add (`add.u64 %F, %B, %M2`) within this BB.
        # Locate mul2_inst's index (already known: it's instrs[i+1] or [i+2]).
        mul2_idx = i + 1 if instrs[i + 1] is mul2_inst else i + 2
        consumer_add = None
        consumer_add_idx = -1
        for k in range(mul2_idx + 1, len(instrs)):
            later = instrs[k]
            if _writes(later, m2_name) and later is not mul2_inst:
                break
            if _writes(later, x_reg.name):
                break
            cm = _is_consumer_add(later, m2_name)
            if cm is not None:
                consumer_add = later
                consumer_add_idx = k
                break
            if any(isinstance(s, RegOp) and s.name == m2_name
                   for s in (later.srcs or [])):
                break
        if consumer_add is None:
            i += 1
            continue
        if not _pred_match(consumer_add, cand):
            i += 1
            continue
        f_name, b_name = _is_consumer_add(consumer_add, m2_name)
        if def_count.get(f_name, 0) != 1 or use_count.get(f_name, 0) != 1:
            i += 1
            continue

        # Find the store, must be in same BB after consumer_add.
        store_inst = None
        for k in range(consumer_add_idx + 1, len(instrs)):
            later = instrs[k]
            if _writes(later, f_name) and later is not consumer_add:
                break
            if _is_global_store_of(later, f_name):
                store_inst = later
                break
            # %F escaping into anything else (read or as MemOp.base of a
            # non-target instruction): bail.
            if any(isinstance(s, RegOp) and s.name == f_name
                   for s in (later.srcs or [])):
                break
            if any(isinstance(s, MemOp)
                   and isinstance(s.base, str)
                   and (s.base if s.base.startswith('%') else f'%{s.base}') == f_name
                   for s in (later.srcs or [])):
                break
        if store_inst is None:
            i += 1
            continue
        if not _pred_match(store_inst, cand):
            i += 1
            continue
        store_memop = store_inst.srcs[0]
        if not isinstance(store_memop, MemOp) or store_memop.offset != 0:
            i += 1
            continue

        # ---- All gates passed.  Perform the rewrite. ----

        cse_key = (x_reg.name, K_combined, b_name, chain_pred, chain_neg)
        cached = cse.get(cse_key)

        f_off_name = _alloc_vreg(fn, "rd_md")
        # Track which existing instructions become dead this iteration.
        # K1's mov is kept (m1_def still uses it).
        dead_inst_ids = {id(cand)}
        for dm in (k2_dead, y_dead):
            if dm is not None and dm is not cand and dm is not m1_def:
                dead_inst_ids.add(id(dm))

        if cached is None:
            # First chain in this CSE bucket: emit the shared mul + base add.
            m_new_name = _alloc_vreg(fn, "rd_md")
            new_mul = Instruction(
                op="mul",
                types=list(mul2_inst.types),
                dest=RegOp(m_new_name),
                srcs=[RegOp(x_reg.name), ImmOp(K_combined)],
                pred=chain_pred,
                neg=chain_neg,
                mods=list(mul2_inst.mods),
            )
            # Modify consumer_add to read %M_new, keeping its dest %F.
            consumer_add.srcs = [
                RegOp(m_new_name) if (isinstance(s, RegOp) and s.name == m2_name) else s
                for s in consumer_add.srcs
            ]
            # Replace mul2_inst with new_mul (in-place at mul2_idx).
            instrs[mul2_idx] = new_mul

            f_shared_name = f_name
            cse[cse_key] = (m_new_name, f_shared_name)
            base_for_off_add = RegOp(f_shared_name)
        else:
            # CSE hit: reuse the shared %M_new and %F_shared.  Drop both
            # the second mul and the consumer add (the slot's own
            # `add %F, %B, %M2` is no longer needed — the offset add
            # references %F_shared instead).  Mul2 and consumer_add are
            # marked dead.
            m_new_name, f_shared_name = cached
            dead_inst_ids.add(id(mul2_inst))
            dead_inst_ids.add(id(consumer_add))
            base_for_off_add = RegOp(f_shared_name)

        new_off_add = Instruction(
            op="add",
            types=list(consumer_add.types),
            dest=RegOp(f_off_name),
            srcs=[base_for_off_add, ImmOp(Y_offset)],
            pred=chain_pred,
            neg=chain_neg,
            mods=[],
        )

        # Re-base store on %F_off.
        old_memop = store_inst.srcs[0]
        store_inst.srcs[0] = MemOp(base=f_off_name, offset=old_memop.offset)

        # Splice instructions: drop dead, insert new_off_add immediately
        # after consumer_add (which itself may be dead — in that case we
        # insert at the position consumer_add USED to occupy).
        new_instrs: list[Instruction] = []
        inserted_off = False
        for inst in instrs:
            if id(inst) in dead_inst_ids:
                # If the dropped instruction is the consumer add (the
                # CSE-hit case), insert the offset add at this spot.
                if (cached is not None) and inst is consumer_add and not inserted_off:
                    new_instrs.append(new_off_add)
                    inserted_off = True
                continue
            new_instrs.append(inst)
            if (cached is None) and inst is consumer_add and not inserted_off:
                new_instrs.append(new_off_add)
                inserted_off = True
        bb.instructions = new_instrs

        # Refresh use/def maps for further matches.
        def_count, use_count, def_instr = _walk_def_use_counts(fn)
        n_rewrites += 1
        # Don't increment i; restart from current position to catch any
        # adjacent chains the rewrite may have exposed.

    return n_rewrites


def _match_slot0_chain(instrs, idx, x_name, b_name, K_combined,
                        chain_pred_tuple, def_count, use_count, def_instr):
    """Try to match a slot-0-style chain at instrs[idx..idx+2]:

        mul.lo.u64 %m2, %m1, K2     (m1 = X * K1; K1 * K2 == K_combined)
        add.u64    %f,  %B,  %m2    (or with %B, %m2 swapped)
        st.global  [%f], data       (offset 0)

    All three instructions must share `chain_pred_tuple`.  %m2 and %f
    must be single-def, single-use.  The leading mul `m1_def` must be
    unpredicated and have %X as its first source.

    Returns (idx, mul_inst, cons_inst, store_inst, k2_dead_mov) or None.
    """
    if idx + 2 >= len(instrs):
        return None

    mul_cand = instrs[idx]
    if (mul_cand.op != "mul" or not _has_lo_modifier(mul_cand)
            or not _is_int64_typed(mul_cand) or mul_cand.mods
            or len(mul_cand.srcs or []) != 2):
        return None
    if (not isinstance(mul_cand.dest, RegOp)
            or isinstance(mul_cand.dest, VectorRegOp)):
        return None
    if (mul_cand.pred, mul_cand.neg) != chain_pred_tuple:
        return None

    m1_op = mul_cand.srcs[0]
    k2_op = mul_cand.srcs[1]
    if not isinstance(m1_op, RegOp) or isinstance(m1_op, VectorRegOp):
        return None

    m1_def = def_instr.get(m1_op.name)
    if (m1_def is None or m1_def.op != "mul"
            or not _has_lo_modifier(m1_def) or not _is_int64_typed(m1_def)
            or m1_def.pred is not None or len(m1_def.srcs or []) != 2):
        return None
    x_op = m1_def.srcs[0]
    if (not isinstance(x_op, RegOp) or x_op.name != x_name
            or isinstance(x_op, VectorRegOp)):
        return None

    k1 = _resolve_u64_imm(m1_def.srcs[1], def_instr, def_count, use_count)
    if k1 is None:
        return None
    K1, _ = k1
    if K1 == 0:
        return None
    k2 = _resolve_u64_imm(k2_op, def_instr, def_count, use_count,
                           allow_pred=chain_pred_tuple)
    if k2 is None:
        return None
    K2, k2_dead = k2
    if K2 == 0:
        return None
    full_product = K1 * K2
    if full_product != (full_product & 0xFFFFFFFFFFFFFFFF):
        return None
    if (full_product & 0xFFFFFFFFFFFFFFFF) != K_combined:
        return None

    m2_name = mul_cand.dest.name
    if (def_count.get(m2_name, 0) != 1
            or use_count.get(m2_name, 0) != 1):
        return None

    cons_cand = instrs[idx + 1]
    cm = _is_consumer_add(cons_cand, m2_name)
    if cm is None or (cons_cand.pred, cons_cand.neg) != chain_pred_tuple:
        return None
    f_name, b_local = cm
    if b_local != b_name:
        return None
    if (def_count.get(f_name, 0) != 1
            or use_count.get(f_name, 0) != 1):
        return None

    store_cand = instrs[idx + 2]
    if (not _is_global_store_of(store_cand, f_name)
            or (store_cand.pred, store_cand.neg) != chain_pred_tuple):
        return None
    store_memop = store_cand.srcs[0]
    if not isinstance(store_memop, MemOp) or store_memop.offset != 0:
        return None

    return (idx, mul_cand, cons_cand, store_cand, k2_dead)


def _hoist_rebase_one_cse_entry(fn: Function, bb, cse_key, cse_value,
                                  def_count: dict[str, int],
                                  use_count: dict[str, int],
                                  def_instr: dict[str, Instruction]) -> int:
    """Process one CSE entry: rebase any slot-0 chains in `bb` onto its
    `(M_new, F_shared)` pair, hoisting the shared definitions if a
    chain physically precedes them in program order.
    """
    m_new_name, f_shared_name = cse_value
    x_name, K_combined, b_name, chain_pred, chain_neg = cse_key
    chain_pred_tuple = (chain_pred, chain_neg)

    instrs = bb.instructions
    m_new_idx = -1
    f_shared_idx = -1
    for i, inst in enumerate(instrs):
        d = inst.dest
        if isinstance(d, RegOp) and not isinstance(d, VectorRegOp):
            if d.name == m_new_name:
                m_new_idx = i
            elif d.name == f_shared_name:
                f_shared_idx = i
    if m_new_idx < 0 or f_shared_idx < 0:
        return 0
    if m_new_idx >= f_shared_idx:
        # Pass A always emits the shared mul before the consumer add.
        # If this isn't true, something has shuffled them — bail.
        return 0

    candidates = []
    i = 0
    while i < len(instrs):
        cand = _match_slot0_chain(instrs, i, x_name, b_name, K_combined,
                                    chain_pred_tuple, def_count, use_count,
                                    def_instr)
        if cand is None:
            i += 1
            continue
        slot_idx = cand[0]
        # Don't match the M_new + consumer_add + (whatever) — the new_mul
        # itself looks like a slot-0 mul, but the consumer_add is followed
        # by the offset_add, not a store, so it falls out at the store
        # check.  Defensive belt-and-suspenders here in case of overlap.
        if slot_idx in (m_new_idx, f_shared_idx):
            i += 1
            continue
        candidates.append(cand)
        i += 3

    if not candidates:
        return 0

    before = [c for c in candidates if c[0] < m_new_idx]
    after = [c for c in candidates if c[0] > f_shared_idx]
    n_done = 0

    # Apply "after" rebases first — no hoist required.
    if after:
        dead_inst_ids = set()
        for _, m, c, st, k2_dead in after:
            dead_inst_ids.add(id(m))
            dead_inst_ids.add(id(c))
            if k2_dead is not None:
                dead_inst_ids.add(id(k2_dead))
            old_memop = st.srcs[0]
            st.srcs[0] = MemOp(base=f_shared_name, offset=old_memop.offset)
        bb.instructions = [inst for inst in instrs
                            if id(inst) not in dead_inst_ids]
        instrs = bb.instructions
        n_done += len(after)
        def_count, use_count, def_instr = _walk_def_use_counts(fn)
        m_new_idx = -1
        f_shared_idx = -1
        for j, inst in enumerate(instrs):
            d = inst.dest
            if isinstance(d, RegOp) and not isinstance(d, VectorRegOp):
                if d.name == m_new_name:
                    m_new_idx = j
                elif d.name == f_shared_name:
                    f_shared_idx = j
        if m_new_idx < 0 or f_shared_idx < 0:
            return n_done

    if not before:
        return n_done

    m_new_inst = instrs[m_new_idx]
    f_sh_inst = instrs[f_shared_idx]
    earliest_idx = min(c[0] for c in before)

    # Hoist safety: between earliest_idx and f_shared_idx, nothing may
    # write %X, %B, or the chain predicate.  We skip the M_new and
    # F_shared positions themselves since those are the instructions we
    # are moving.
    for j in range(earliest_idx, f_shared_idx + 1):
        if j == m_new_idx or j == f_shared_idx:
            continue
        inst = instrs[j]
        if _writes(inst, x_name) or _writes(inst, b_name):
            return n_done
        if chain_pred is not None and _writes(inst, chain_pred):
            return n_done

    # The chain predicate (if any) must be defined BEFORE earliest_idx.
    # If it's defined in another block, treat as dominator-defined.
    if chain_pred is not None:
        pred_def = def_instr.get(chain_pred)
        if pred_def is not None:
            try:
                pred_idx = instrs.index(pred_def)
                if pred_idx >= earliest_idx:
                    return n_done
            except ValueError:
                pass

    # %X and %B (the new mul/consumer-add operands) must each be defined
    # before earliest_idx in this block, or in a dominating block.
    for needed in (x_name, b_name):
        d_inst = def_instr.get(needed)
        if d_inst is None:
            continue
        try:
            d_idx = instrs.index(d_inst)
        except ValueError:
            continue  # external — dominator-defined.
        if d_idx >= earliest_idx:
            return n_done

    dead_inst_ids = {id(m_new_inst), id(f_sh_inst)}
    for _, m, c, st, k2_dead in before:
        dead_inst_ids.add(id(m))
        dead_inst_ids.add(id(c))
        if k2_dead is not None:
            dead_inst_ids.add(id(k2_dead))
        old_memop = st.srcs[0]
        st.srcs[0] = MemOp(base=f_shared_name, offset=old_memop.offset)

    new_instrs = []
    inserted_hoist = False
    for j, inst in enumerate(instrs):
        if j == earliest_idx and not inserted_hoist:
            new_instrs.append(m_new_inst)
            new_instrs.append(f_sh_inst)
            inserted_hoist = True
        if id(inst) in dead_inst_ids:
            continue
        new_instrs.append(inst)
    if not inserted_hoist:
        return n_done
    bb.instructions = new_instrs
    return n_done + len(before)


def _try_slot0_hoist_rebase_block(fn: Function, bb,
                                    cse: dict[tuple, tuple[str, str]],
                                    def_count: dict[str, int],
                                    use_count: dict[str, int],
                                    def_instr: dict[str, Instruction]) -> int:
    """Phase 24: slot-0 chain hoist + rebase.

    Pass A emits a shared `mul %M_new, %X, K_combined` plus
    `add %F_shared, %B, %M_new` at the position of the FIRST leading-add
    chain in a block (typically slot 1 of a per-slot store group).
    Slots 1..N (with leading adds) reuse those via CSE and only need a
    constant-offset add per slot.

    Slot 0 lacks the leading add — its PTX form is

        mul.lo.u64 %m2, %m1, K2
        add.u64    %f,  %B, %m2
        st.global  [%f], data

    Phase 22's reverted attempt rebased slot-0 stores onto %F_shared
    blindly, producing a use-before-def hazard whenever the slot-0 chain
    physically preceded the Pass-A-inserted shared definitions.

    Phase 24 detects that case and HOISTS the (M_new + F_shared) pair to
    immediately before the earliest such "before" slot-0 chain, after
    verifying that:

      - %X and %B are not redefined between the hoist target index and
        F_shared's original index;
      - The chain predicate (if any) is defined before the hoist target;
      - %X and %B are defined before the hoist target (or in a
        dominating block);

    then drops the slot-0's mul + consumer_add (and any dead K2 mov)
    and repoints the slot-0 store to %F_shared.  Slot-0 chains that
    occur AFTER F_shared's definition do not need the hoist and are
    rebased directly.

    Returns the number of slot-0 chains rewritten.
    """
    if not cse:
        return 0
    n = 0
    for cse_key in list(cse.keys()):
        added = _hoist_rebase_one_cse_entry(fn, bb, cse_key, cse[cse_key],
                                              def_count, use_count, def_instr)
        if added:
            n += added
            def_count, use_count, def_instr = _walk_def_use_counts(fn)
    return n


def run_function(fn: Function) -> int:
    """Run mul_distribute on a single function.  Returns # chains rewritten."""
    def_count, use_count, def_instr = _walk_def_use_counts(fn)
    total = 0
    for bb in fn.blocks:
        # Pass A: leading-add chains; populates CSE within `bb` scope.
        # Pass A's shared mul/base-add are inserted at the position of
        # the FIRST chain that triggers them.  Slot-0 chains that
        # physically precede that position cannot be rebased without
        # hoisting; Phase 24's pass below handles the hoist + rebase.
        bb_cse: dict[tuple, tuple[str, str]] = {}
        n = _try_rewrite_block_with_cse(fn, bb, bb_cse,
                                        def_count, use_count, def_instr)
        if n:
            total += n
            def_count, use_count, def_instr = _walk_def_use_counts(fn)
        # Phase 24: hoist + rebase slot-0 chains using the CSE entries
        # populated by Pass A above.
        n2 = _try_slot0_hoist_rebase_block(fn, bb, bb_cse,
                                            def_count, use_count, def_instr)
        if n2:
            total += n2
            def_count, use_count, def_instr = _walk_def_use_counts(fn)
    return total


def run(module) -> int:
    total = 0
    for fn in module.functions:
        total += run_function(fn)
    return total
