"""
Rotate-left pattern recognizer — the CORRECT implementation.

This is the optimization that ptxas gets wrong.  We implement it right.

The pattern:
    shl.b64  %dst_lo, %src, K
    shr.u64  %dst_hi, %src, (64-K)
    OP       %result, %dst_lo, %dst_hi

is a rotate-left of %src by K bits ONLY when:
    1. OP is in {add, or, xor}     — commutative AND rotation-equivalent
    2. The right shift is LOGICAL  — shr.u64, NOT shr.s64 (arithmetic)
    3. Both shifts are of the SAME source register
    4. K and (64-K) are compile-time constants that sum to 64

ptxas applies this optimization even when OP=sub (Bug 1), when the
right shift is signed/arithmetic (Bug 2), and ignores operand order (Bug 3).
We check all three conditions explicitly.

The pass marks matched groups with a RotateGroup annotation; downstream
codegen can emit a single SHF.L.W instruction for each group.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from ..ir import (
    Module, Function, BasicBlock, Instruction,
    RegOp, ImmOp, Operand,
)


# ---------------------------------------------------------------------------
# Rotate group annotation
# ---------------------------------------------------------------------------

@dataclass
class RotateGroup:
    """
    Three instructions that form a valid rotate-left pattern.

    shl_inst:   the shl.b64 instruction
    shr_inst:   the shr.u64 instruction
    combine_inst: the add/or/xor instruction
    src:        the source register name (shared by both shifts)
    k:          the left-shift amount (rotate distance)
    width:      bit width (currently always 64)
    result_reg: the destination register of combine_inst
    """
    shl_inst:     Instruction
    shr_inst:     Instruction
    combine_inst: Instruction
    src:          str      # register name e.g. "%rd6"
    k:            int
    width:        int = 64
    result_reg:   Optional[str] = None


# ---------------------------------------------------------------------------
# Pattern matcher
# ---------------------------------------------------------------------------

# Operators for which rotate-left is semantically correct
_ROTATE_OPS = {"add", "or", "xor"}

# Operators that ptxas INCORRECTLY accepts (the bug)
_BUGGY_OPS   = {"sub"}


def _reg_name(op: Operand) -> Optional[str]:
    return op.name if isinstance(op, RegOp) else None


def _imm_val(op: Operand) -> Optional[int]:
    return op.value if isinstance(op, ImmOp) else None


def _is_shl_b64(inst: Instruction) -> bool:
    return inst.op == "shl" and "b64" in inst.types


def _is_shr_u64(inst: Instruction) -> bool:
    """Logical (unsigned) 64-bit right shift — valid for rotate."""
    return inst.op == "shr" and "u64" in inst.types


def _is_shr_s64(inst: Instruction) -> bool:
    """Arithmetic (signed) 64-bit right shift — INVALID for rotate (Bug 2)."""
    return inst.op == "shr" and "s64" in inst.types


def match_rotate(
    shl: Instruction,
    shr: Instruction,
    combine: Instruction,
    width: int = 64,
) -> Optional[RotateGroup]:
    """
    Try to match three instructions as a valid rotate-left pattern.
    Returns a RotateGroup if all conditions are satisfied, None otherwise.

    Raises ValueError with a diagnostic if the pattern structurally matches
    but would be miscompiled by ptxas (i.e. is one of the three known bugs).
    """

    # -----------------------------------------------------------------------
    # Structural checks: does this look like a rotate candidate at all?
    # -----------------------------------------------------------------------
    if not _is_shl_b64(shl):
        return None
    if not (_is_shr_u64(shr) or _is_shr_s64(shr)):
        return None

    # shl must have a dest and two srcs (dest, src, shift_amount)
    if shl.dest is None or len(shl.srcs) < 2:
        return None
    if shr.dest is None or len(shr.srcs) < 2:
        return None
    if combine.dest is None or len(combine.srcs) < 2:
        return None

    shl_src   = _reg_name(shl.srcs[0])
    shl_k     = _imm_val(shl.srcs[1])
    shr_src   = _reg_name(shr.srcs[0])
    shr_k     = _imm_val(shr.srcs[1])

    # Both shifts must have compile-time constant amounts
    if shl_k is None or shr_k is None:
        return None

    # Both shifts must be of the SAME source register
    if shl_src is None or shr_src is None:
        return None
    if shl_src != shr_src:
        return None

    # Shift amounts must be complementary (sum to width)
    if shl_k + shr_k != width:
        return None

    # combine must use both shift results as operands
    shl_dest = _reg_name(shl.dest)
    shr_dest = _reg_name(shr.dest)
    combine_srcs = [_reg_name(s) for s in combine.srcs]
    if shl_dest not in combine_srcs or shr_dest not in combine_srcs:
        return None

    # -----------------------------------------------------------------------
    # Semantic validity checks — THIS IS WHERE ptxas FAILS
    # -----------------------------------------------------------------------

    # Bug 2 check: right shift must be LOGICAL (shr.u64), not arithmetic
    if _is_shr_s64(shr):
        # Pattern structurally matches but is semantically wrong for rotate.
        # ptxas would miscompile this (Bug 2).  We do NOT emit a rotate.
        return None

    # Bug 1 check: combining operator must be in {add, or, xor}
    if combine.op not in _ROTATE_OPS:
        # ptxas would miscompile sub/and/etc. as a rotate (Bug 1).
        # We do NOT emit a rotate.
        return None

    # Bug 3 is subsumed: if we only match add/or/xor, operand order doesn't
    # matter for correctness (all three are commutative).

    # -----------------------------------------------------------------------
    # All checks passed — this is a valid rotate-left
    # -----------------------------------------------------------------------
    return RotateGroup(
        shl_inst=shl,
        shr_inst=shr,
        combine_inst=combine,
        src=shl_src,
        k=shl_k,
        width=width,
        result_reg=_reg_name(combine.dest),
    )


# ---------------------------------------------------------------------------
# Pass: scan a function for rotate groups
# ---------------------------------------------------------------------------

def find_rotate_groups(fn: Function) -> list[RotateGroup]:
    """
    Walk every basic block and return all valid rotate-left groups.
    Also logs any patterns that would be miscompiled by ptxas.
    """
    groups: list[RotateGroup] = []
    buggy:  list[str]         = []

    for bb in fn.blocks:
        insts = bb.instructions
        n     = len(insts)

        # Build a register → definition map for the block
        reg_def: dict[str, Instruction] = {}
        for inst in insts:
            if inst.dest:
                name = _reg_name(inst.dest)
                if name:
                    reg_def[name] = inst

        # For each combine instruction, look up its two shl/shr producers
        for inst in insts:
            if inst.op not in (_ROTATE_OPS | _BUGGY_OPS):
                continue
            if inst.dest is None or len(inst.srcs) < 2:
                continue

            src0_name = _reg_name(inst.srcs[0])
            src1_name = _reg_name(inst.srcs[1])
            if src0_name is None or src1_name is None:
                continue

            prod0 = reg_def.get(src0_name)
            prod1 = reg_def.get(src1_name)
            if prod0 is None or prod1 is None:
                continue

            # Try (shl=prod0, shr=prod1) and (shl=prod1, shr=prod0)
            for shl_cand, shr_cand in [(prod0, prod1), (prod1, prod0)]:
                # First check: would ptxas miscompile this? (Bug 1 or Bug 2)
                if _is_shl_b64(shl_cand) and (_is_shr_u64(shr_cand) or _is_shr_s64(shr_cand)):
                    shl_src  = _reg_name(shl_cand.srcs[0]) if shl_cand.srcs else None
                    shr_src  = _reg_name(shr_cand.srcs[0]) if shr_cand.srcs else None
                    shl_k    = _imm_val(shl_cand.srcs[1]) if len(shl_cand.srcs) > 1 else None
                    shr_k    = _imm_val(shr_cand.srcs[1]) if len(shr_cand.srcs) > 1 else None

                    if (shl_src and shr_src and shl_src == shr_src
                            and shl_k is not None and shr_k is not None
                            and shl_k + shr_k == 64):

                        if inst.op in _BUGGY_OPS:
                            buggy.append(
                                f"  PTXAS BUG (Bug 1): {inst} — "
                                f"sub.s64 would be miscompiled as rotate by ptxas"
                            )
                        if _is_shr_s64(shr_cand):
                            buggy.append(
                                f"  PTXAS BUG (Bug 2): {inst} — "
                                f"shr.s64 (arithmetic) would be miscompiled as rotate by ptxas"
                            )

                # Now check if it's a VALID rotate
                grp = match_rotate(shl_cand, shr_cand, inst)
                if grp:
                    groups.append(grp)
                    break  # don't double-count

    if buggy:
        print(f"[rotate pass] Found {len(buggy)} pattern(s) that ptxas would miscompile:")
        for b in buggy:
            print(b)

    return groups


# ---------------------------------------------------------------------------
# Pass entry point
# ---------------------------------------------------------------------------

def run(module: Module) -> tuple[Module, list[RotateGroup]]:
    """
    Run the rotate recognition pass over the entire module.
    Returns the (unchanged) module and all found RotateGroups.
    The module is not modified — codegen uses the groups to emit SHF.L.W.
    """
    all_groups: list[RotateGroup] = []
    for fn in module.functions:
        groups = find_rotate_groups(fn)
        all_groups.extend(groups)
        if groups:
            print(f"[rotate pass] {fn.name}: {len(groups)} valid rotate-left group(s)")
    return module, all_groups
