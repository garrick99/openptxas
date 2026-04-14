"""
sass/isel.py — PTX-to-SASS instruction selector for SM_120 (and SM_89).

Maps PTX IR instructions to sequences of 16-byte SASS instructions.
Handles 60+ instruction encodings verified byte-for-byte against ptxas 13.0.

Architecture:
  - Input: ptx.ir.Function with allocated physical registers (from regalloc.py)
  - Output: list of SassInstr (16-byte bytes + comment string)

Register mapping convention (set by regalloc, read here):
  PTX %rd0..%rdN → SASS R0..R(N*2+1)   (64-bit pairs: lo=even, hi=odd)
  PTX %r0..%rN   → SASS R(BASE+N)      (32-bit singles)
  PTX %f0..%fN   → SASS R(BASE+N)      (float, same bank as int32)
  PTX %fd0..%fdN → SASS R(BASE+N*2)    (f64, 64-bit pairs like rd)
  PTX %p0..%pN   → SASS P0..P5         (predicates)
  PTX %ur0..%urN → SASS UR0..UR63      (uniform registers, LDCU/S2UR targets)

Key SM_120 encoding constraints (see ARCHITECTURE.md for full details):
  - IMAD R-R (0x2a4) is BROKEN; use IMAD R-UR (0xc24) for all 32-bit mul
  - ISETP (0x20c/0xc0c) corrupts FSETP; use FSEL.step (0x80a) for int+float pred
  - rbar is a bitmask (OR-combine): bit1=LDC, bit2=LDS, bit3=LDG
  - S2R / S2UR are asynchronous (wdep=0x31 required)
  - SM_120 uses predicated execution for warp divergence (no intra-kernel BRA)
  - DSETP ordered comparison codes silently give P=false; use unordered (GEU etc.)
  - QMMA requires dest==src_a in encoding (in-place accumulate)
  - IMMA B register base must be < 8
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from ptx.ir import Instruction, Function, Operand, RegOp, ImmOp, LabelOp

from sass.encoding.sm_120_opcodes import (
    encode_nop, encode_exit, encode_mov,
    encode_ldc, encode_ldc_64,
    encode_s2r,
    encode_iadd3, encode_iadd3x,
    encode_iadd64,
    encode_imad_wide, encode_imad_wide_rr, encode_imad_wide_u32, encode_imad_wide_u32_carry, encode_imad_wide_u32x,
    encode_imad, encode_imad_rr, encode_imad_ur, encode_imad_hi, encode_imad_shl_u32,
    encode_s2ur,
    encode_ldg_e, encode_ldg_e_64,
    encode_stg_e, encode_stg_e_64,
    encode_lds, encode_sts, encode_lds_r, encode_sts_r,
    encode_ldcu_64, encode_ldcu_32,
    encode_iadd64_ur,
    encode_bar_sync,
    encode_isetp_ge_and, encode_isetp_ur,
    encode_isetp, ISETP_LT, ISETP_EQ, ISETP_LE, ISETP_GT, ISETP_NE, ISETP_GE,
    encode_fsetp, FSETP_LT, FSETP_EQ, FSETP_LE, FSETP_GT, FSETP_NE, FSETP_GE,
    encode_bra, patch_pred,
    encode_fadd, encode_fmul, encode_fmul_imm, encode_ffma, encode_ffma_imm,
    encode_mufu, MUFU_RCP, MUFU_SQRT, MUFU_SIN, MUFU_COS, MUFU_EX2, MUFU_LG2,
    encode_sel, encode_sel_imm, encode_fsel,
    encode_vimnmx_s32, encode_vimnmx_u32,
    encode_fmnmx,
    encode_prmt, encode_prmt_reg,
    encode_popc, encode_brev, encode_flo, encode_iabs, encode_bfe_sext,
    encode_shfl, SHFL_IDX, SHFL_UP, SHFL_DOWN, SHFL_BFLY,
    encode_vote_ballot,
    encode_atomg_cas_b32, encode_atomg_cas_b64, encode_atomg_u32, encode_atomg_add_f32,
    ATOMG_ADD, ATOMG_MIN, ATOMG_MAX, ATOMG_EXCH, ATOMG_OR, ATOMG_AND, ATOMG_XOR,
    encode_membar, MEMBAR_GPU, MEMBAR_CTA,
    encode_idp4a,
    encode_dadd, encode_dmul, encode_dfma, encode_dfma_ur_ur,
    encode_dsetp, DSETP_LT, DSETP_EQ, DSETP_LE, DSETP_GT, DSETP_NE, DSETP_GE,
    DSETP_LTU, DSETP_EQU, DSETP_LEU, DSETP_GTU, DSETP_NEU, DSETP_GEU,
    encode_i2fp_u32, encode_f2i_u32, encode_i2f_f32_s32, encode_f2i_s32_f32,
    encode_f2fp_f16_f32, encode_cvt_f16_f32,
    encode_f2f_f32_f64, encode_f2f_f64_f32,
    encode_f2i_s32_f64, encode_f2i_u32_f64, encode_i2f_f64_s32, encode_i2f_f64_u32,
    encode_i2f_u32_rp, encode_i2f_s32_rp, encode_f2i_ftz_u32_trunc, encode_hfma2_zero,
    encode_hmma_f16_f32, encode_hmma_f16_f32_k8, encode_hmma_bf16_f32, encode_hmma_tf32_f32,
    encode_imma_s8_s32, encode_dmma_8x8x4,
    encode_qmma_e4m3_f32, encode_qmma_e5m2_f32,
    encode_ldsm_x4, encode_ldsm_x2, encode_ldsm_x1,
    encode_redux_sum, encode_redux_sum_s32, encode_redux_min_s32, encode_redux_max_s32,
    encode_redux_and_b32, encode_redux_or_b32, encode_redux_xor_b32,
    encode_ldgsts_e, encode_ldgdepbar, encode_depbar_le,
    encode_syncs_exch_64, encode_syncs_arrive, encode_syncs_trywait,
    encode_ublkcp_s_g, encode_ublkcp_g_s,
    encode_utmaldg_1d, encode_utmaldg_2d, encode_utmastg_1d,
    encode_utmacmdflush, encode_elect, encode_cctl_ivall,
    encode_mov_gpr_from_ur,
    encode_iadd3_imm32, encode_iadd3_imm32_neg_src0,
    encode_iadd3_neg_b4, encode_iadd3_neg_b3,
    encode_iadd3_pred_neg_b4, encode_iadd3_pred_small_imm,
    encode_iadd3_pred_neg_b3, encode_lop3_pred,
    encode_lop3, LOP3_AND, LOP3_OR, LOP3_XOR,
    encode_lop3_imm32, LOP3_IMM_XOR, LOP3_IMM_AND, LOP3_IMM_OR,
    RZ, PT, SR_TID_X, SR_TID_Y, SR_TID_Z,
    SR_CTAID_X, SR_CTAID_Y, SR_CTAID_Z,
    SR_NTID_X, SR_NTID_Y, SR_NTID_Z,
    SR_NCTAID_X, SR_NCTAID_Y, SR_NCTAID_Z,
    SR_LANEID,
    encode_tex, encode_tld_lz, encode_tld4, encode_txq,
    encode_suld, encode_sust,
    TEX_DIM_1D, TEX_DIM_2D, TEX_DIM_3D,
    SURF_DIM_1D, SURF_DIM_2D,
    SURF_MODE_B32, SURF_MODE_B64,
    TXQ_WIDTH, TXQ_HEIGHT, TXQ_DEPTH,
)
from sass.encoding.sm_120_encode import (
    encode_shf_l_w_u32_hi,
    encode_shf_l_u32,
    encode_shf_l_u64_hi,
    encode_shf_r_u32, encode_shf_r_u32_hi,
    encode_shf_l_u32_var, encode_shf_r_u32_hi_var,
    encode_shf_r_s32_hi, encode_shf_r_s32_hi_var,
)
from sass.regalloc import RegAlloc


# ---------------------------------------------------------------------------
# Output: sequence of encoded SASS instructions
# ---------------------------------------------------------------------------

@dataclass
class SassInstr:
    """One encoded 16-byte SM_120 instruction with metadata."""
    raw:     bytes          # 16 bytes, little-endian
    comment: str = ''       # human-readable annotation

    def hex(self) -> str:
        return self.raw.hex()


# ---------------------------------------------------------------------------
# Instruction selector
# ---------------------------------------------------------------------------

class ISelError(Exception):
    pass


def _get_reg(op: Operand, ra: RegAlloc, bits: int = 32) -> int:
    """Extract physical register index from an operand."""
    if isinstance(op, RegOp):
        name = op.name
        if name == 'RZ' or name == '%rz':
            return RZ
        if bits == 64:
            return ra.lo(name)
        return ra.r32(name)
    raise ISelError(f"Expected register operand, got {op!r}")


def _get_imm(op: Operand) -> int:
    if isinstance(op, ImmOp):
        return op.value
    raise ISelError(f"Expected immediate operand, got {op!r}")


def _nop(comment: str = '') -> SassInstr:
    return SassInstr(encode_nop(), comment or 'NOP')


def _emit_ur_to_gpr(dest: int, ur_idx: int, comment: str = '',
                    ctx: 'ISelContext' = None) -> list[SassInstr]:
    """Materialize a UR pair into a GPR pair via IADD.64 R, RZ, UR.

    SM_120 rule #27 (OBSOLETE): IADD.64 R-UR with RZ as src_r was
    reported broken, but hardware testing (v1.8, driver 595.79) confirms
    IADD.64 R, RZ, UR works correctly. ptxas uses this form.
    """
    return [
        SassInstr(encode_iadd64_ur(dest, RZ, ur_idx),
                  f'IADD.64 R{dest}, RZ, UR{ur_idx}  // {comment or "UR->GPR"}'),
    ]


def _f64_to_gpr(name: str, ctx, output: list) -> int:
    """Return the lo GPR index for an f64 register.
    If the register is GPR-backed, return it directly.
    If it is UR-backed (loaded via LDCU.64), emit IADD.64 RZ,UR→tmp and cache + return tmp.
    Caching ensures each f64 param is only materialized once, reducing NOP insertions."""
    if name in ctx.ra.int_regs:
        return ctx.ra.int_regs[name]
    # Check materialization cache (avoids re-emitting IADD.64-UR for same param)
    if not hasattr(ctx, '_f64_gpr_cache'):
        ctx._f64_gpr_cache = {}
    if name in ctx._f64_gpr_cache:
        return ctx._f64_gpr_cache[name]
    ur = ctx._ur_params.get(name)
    if ur is None:
        raise ISelError(f'f64 register {name!r} not in GPR or UR')
    t = _alloc_gpr(ctx)
    if t % 2 != 0:
        t = _alloc_gpr(ctx)
    _alloc_gpr(ctx)  # reserve t+1
    output.extend(_emit_ur_to_gpr(t, ur, "materialize f64 {name}"))
    ctx._f64_gpr_cache[name] = t
    return t


def _alloc_scratch(ctx: 'ISelContext', count: int = 1) -> list[int]:
    """Allocate scratch GPRs from the pool. Returns list of register indices."""
    regs = []
    for _ in range(count):
        if ctx._scratch_pool:
            r = ctx._scratch_pool.pop()
        else:
            r = ctx._next_gpr
            ctx._next_gpr += 1
            ctx._scratch_highwater = max(ctx._scratch_highwater, ctx._next_gpr)
        regs.append(r)
    return regs


def _free_scratch(ctx: 'ISelContext', regs: list[int]):
    """Return scratch GPRs to the pool for reuse."""
    ctx._scratch_pool.extend(regs)


_GPR_HARD_LIMIT_DEFAULT = 14  # Without capmerc, R14+ triggers ERR715
_GPR_HARD_LIMIT_CAPMERC = 255  # With ptxas capmerc, full range available

def _alloc_gpr(ctx: 'ISelContext') -> int:
    """Allocate a single GPR, preferring the scratch pool."""
    limit = getattr(ctx, '_gpr_limit', _GPR_HARD_LIMIT_DEFAULT)
    while ctx._scratch_pool:
        r = ctx._scratch_pool.pop()
        if r < limit:
            return r
    if ctx._next_gpr < limit:
        r = ctx._next_gpr
        ctx._next_gpr += 1
        ctx._scratch_highwater = max(ctx._scratch_highwater, ctx._next_gpr)
        return r
    return 0


def analyze_mma_zero_subst(fn, enable: set[str] | None = None) -> tuple[dict, set]:
    """WB-2/WB-4: mma "all-zero inputs" optimization.

    Detect mma.sync instructions whose source operand vregs are provably
    zero (defined exactly once by `mov X, 0` and used only by this mma's
    input slots — possibly multiple slots of the same mma).

    For any mma shape, the math is D = A*B + C, so any of A/B/C being
    all zero forces D = 0 (mod NaN/inf).  ptxas exploits this by
    substituting RZ for B and C in the encoding and aliasing src_a with
    the dst quad, eliding all input-init MOVs entirely.

    Parameters
    ----------
    enable : optional set of {'hmma', 'imma', 'dmma', 'qmma'}.  Only mma
        instructions of an enabled class will be considered for
        substitution.  Defaults to {'hmma', 'imma', 'qmma'} (WB-4.1).
        DMMA is added separately in WB-4.2.

    Returns
    -------
    rz_subst : dict[id(mma_instr) -> set[str]]
        Subset of {'a','b','c'} marking which encoding slots should
        use RZ.
    dead_movs : set[id(mov_instr)]
        Init MOVs that can be skipped because their target vreg is no
        longer read by any non-substituted slot.
    """
    if enable is None:
        enable = {'hmma', 'imma', 'qmma', 'dmma'}  # WB-4.2: full coverage
    from ptx.ir import RegOp, ImmOp, VectorRegOp, MemOp

    all_instrs = []
    for bb in fn.blocks:
        all_instrs.extend(bb.instructions)

    def _dest_vregs(dst):
        if isinstance(dst, VectorRegOp):
            return tuple(dst.regs)
        if isinstance(dst, RegOp):
            return (dst.name,)
        return ()

    # 1. Find single-def zero-init vregs (mov X, 0  where X has no other def).
    def_count: dict[str, int] = {}
    zero_def_inst: dict[str, object] = {}
    for inst in all_instrs:
        for nm in _dest_vregs(inst.dest):
            def_count[nm] = def_count.get(nm, 0) + 1
        if (inst.op == 'mov' and inst.srcs
                and isinstance(inst.srcs[0], ImmOp)
                and inst.srcs[0].value == 0
                and isinstance(inst.dest, RegOp)
                and not isinstance(inst.dest, VectorRegOp)):
            zero_def_inst[inst.dest.name] = inst

    # Drop vregs with too many defs.  We allow up to 2 defs: the zero mov
    # + at most one mma redefinition (the mma's dst write, which doesn't
    # flow values BACK into the mma's input read).
    for nm in list(zero_def_inst):
        if def_count.get(nm, 0) > 2:
            del zero_def_inst[nm]
            continue
        if def_count.get(nm, 0) == 2:
            ok = False
            for inst in all_instrs:
                if (inst.op == 'mma' and 'sync' in inst.types
                        and nm in _dest_vregs(inst.dest)):
                    ok = True
                    break
            if not ok:
                del zero_def_inst[nm]

    # 2. For each mma.sync, determine the substitutable slots.
    rz_subst: dict[int, set[str]] = {}
    dead_movs: set[int] = set()

    def _slot_vregs(src):
        if isinstance(src, VectorRegOp):
            return tuple(src.regs)
        if isinstance(src, RegOp):
            return (src.name,)
        return ()

    def _mma_class(inst):
        """Classify an mma instruction by its operand types.
        Returns one of {'hmma','imma','dmma','qmma'} or None."""
        if inst.op != 'mma' or 'sync' not in inst.types:
            return None
        ts = set(inst.types)
        if 'f64' in ts:
            return 'dmma'
        if 'e4m3' in ts or 'e5m2' in ts:
            return 'qmma'
        if 's8' in ts or 'u8' in ts:
            return 'imma'
        return 'hmma'  # f16/bf16/tf32 + f32 accumulator

    # Per-class encoding metadata.
    #
    # ENCODED: slots that have a real GPR field in the SASS encoding.
    #     For QMMA the encoder forces src_a := dst (hardware constraint
    #     "dest == src_a"), so the PTX-level A vregs are never actually
    #     encoded.  src_a is implicitly "substituted" — it consumes no
    #     register from the analysis's perspective.
    #
    # RZ_SUBST: subset of ENCODED that we are allowed to set to RZ.
    #     Plus by convention, slot 'a' is rewired to `dst` (alias).
    #     QMMA explicitly excludes 'b' from RZ_SUBST: SM_120 QMMA
    #     hardware requires the B register base to be < 8 and rejects
    #     RZ (R255).  ptxas works around this with CS2R-zeroed R2.
    ENCODED_SLOTS = {
        'hmma': {'a', 'b', 'c'},
        'imma': {'a', 'b', 'c'},
        'qmma': {'b', 'c'},          # a is auto-aliased to dst
        'dmma': {'a', 'b', 'c'},     # WB-4.2 — not yet enabled
    }
    RZ_SUBST_SLOTS = {
        'hmma': {'a', 'b', 'c'},
        'imma': {'a', 'b', 'c'},
        'qmma': {'a', 'c'},          # 'a' marker for dead-mov bookkeeping
        'dmma': {'a', 'b', 'c'},     # WB-4.2
    }

    # Pre-compute per-vreg def positions (instruction indices) so we can
    # bound use-counting to the live range of a particular def.
    vreg_defs: dict[str, list[int]] = {}
    for idx, inst in enumerate(all_instrs):
        for nm in _dest_vregs(inst.dest):
            vreg_defs.setdefault(nm, []).append(idx)

    def _uses_in_range(vreg: str, lo_excl: int, hi_incl: int,
                       skip_inst: object) -> int:
        """Count source uses of `vreg` in instructions (lo_excl, hi_incl],
        excluding `skip_inst`."""
        n = 0
        for j in range(lo_excl + 1, hi_incl + 1):
            o = all_instrs[j]
            if o is skip_inst:
                continue
            for s in (o.srcs or []):
                if vreg in _slot_vregs(s):
                    n += 1
                    break
        return n

    for mma_idx, inst in enumerate(all_instrs):
        cls = _mma_class(inst)
        if cls is None or cls not in enable:
            continue
        srcs = inst.srcs or []
        if len(srcs) < 3:
            continue
        slots = ('a', 'b', 'c')
        slot_vregs: dict[str, tuple] = {
            slot: _slot_vregs(srcs[i]) for i, slot in enumerate(slots)
        }
        encoded = ENCODED_SLOTS[cls]
        rz_sub_class = RZ_SUBST_SLOTS[cls]

        # vreg → set of slots it appears in within this mma.
        vreg_slots: dict[str, set[str]] = {}
        for slot, vs in slot_vregs.items():
            for v in vs:
                vreg_slots.setdefault(v, set()).add(slot)

        # Iterative fixpoint: a slot is RZ-substituted iff it's in
        # rz_sub_class AND all its vregs become "eliminable" — that is,
        # they have no remaining encoded use anywhere in this mma after
        # substitution.  A vreg in slot 'a' for QMMA has 0 encoded uses
        # from 'a' regardless (encoder forces src_a := dst), so it's
        # eliminable from the outset for that slot.
        sub_slots = set(rz_sub_class)
        eliminable: set[str] = set()
        while True:
            new_eliminable: set[str] = set()
            for v, vslots in vreg_slots.items():
                remaining = [s for s in vslots
                             if s in encoded and s not in sub_slots]
                if not remaining:
                    new_eliminable.add(v)
            new_sub: set[str] = set()
            for s in sub_slots:
                if all(v in new_eliminable for v in slot_vregs.get(s, ())):
                    new_sub.add(s)
            if new_sub == sub_slots and new_eliminable == eliminable:
                eliminable = new_eliminable
                break
            sub_slots = new_sub
            eliminable = new_eliminable

        if not sub_slots:
            continue

        # Per-vreg liveness: only mark a mov dead if its vreg is
        # eliminable AND has no live def/use between the mov and this
        # mma.  A vreg in `eliminable` whose mov fails this check stays
        # alive — we keep the slot substituted (the encoded bytes don't
        # depend on the mov), the init mov just isn't elided.
        candidate_dead: set[str] = set()
        for v in eliminable:
            if v not in zero_def_inst:
                continue
            mov_inst = zero_def_inst[v]
            try:
                mov_idx = next(i for i, x in enumerate(all_instrs) if x is mov_inst)
            except StopIteration:
                continue
            if mov_idx >= mma_idx:
                continue
            # No intervening def between mov and mma.
            bad = False
            for d_idx in vreg_defs.get(v, []):
                if mov_idx < d_idx < mma_idx:
                    bad = True
                    break
            if bad:
                continue
            # No intervening use (mma itself excluded).
            if _uses_in_range(v, mov_idx, mma_idx - 1, inst) > 0:
                continue
            candidate_dead.add(v)

        rz_subst[id(inst)] = sub_slots
        for v in candidate_dead:
            dead_movs.add(id(zero_def_inst[v]))

    # WB-4.1: when at least one mma was substituted, also elide any
    # zero-init mov whose dest vreg is never read anywhere in the
    # function ("dead-on-def" padding inits).  This catches the
    # `mov.b32 %r6, 0; mov.b32 %r7, 0` pattern in the QMMA canary
    # where the test author wrote 2 padding inits to push the D/A
    # quad to the next 4-aligned boundary — those vregs are never
    # consumed but the regalloc + isel still emit IADD3 R, RZ, 0, RZ
    # for them.  Restricting to "kernel has a substituted mma" keeps
    # this gated to the WB-4 optimization scope; we don't run a
    # general dead-store-elimination pass.
    if rz_subst:
        all_used: set[str] = set()
        for inst in all_instrs:
            for src in (inst.srcs or []):
                if isinstance(src, VectorRegOp):
                    all_used.update(src.regs)
                elif isinstance(src, RegOp):
                    all_used.add(src.name)
                elif isinstance(src, MemOp) and isinstance(src.base, str):
                    bn = src.base if src.base.startswith('%') else f'%{src.base}'
                    all_used.add(bn)
        for inst in all_instrs:
            if (inst.op == 'mov' and inst.srcs
                    and isinstance(inst.srcs[0], ImmOp)
                    and inst.srcs[0].value == 0
                    and isinstance(inst.dest, RegOp)
                    and not isinstance(inst.dest, VectorRegOp)
                    and inst.dest.name not in all_used):
                dead_movs.add(id(inst))

    return rz_subst, dead_movs


def analyze_addr_offset_fold(fn) -> tuple[dict, set]:
    """WB-7: aliased-base address chain folding.

    Detect `add.u64 %A, %B, IMM` patterns where:
      - IMM is a small constant (fits in LDG.E's 24-bit signed offset)
      - %A's only use is as a single MemOp.base in a global ld/st/atom
      - %A has a single def (the add)

    For each match, record `%A → (%B, IMM)` so the consuming load/store
    isel emits LDG.E [%B + IMM] instead of materializing a new address
    pair.  The add.u64 is marked dead and skipped at emission time.

    Returns
    -------
    fold_map : dict[str, tuple[str, int]]
        vreg name → (base vreg, immediate offset)
    dead_adds : set[int]
        id() of add.u64 instructions to skip
    """
    from ptx.ir import RegOp, ImmOp, MemOp, VectorRegOp

    all_instrs = []
    for bb in fn.blocks:
        all_instrs.extend(bb.instructions)

    def _dest_vregs(dst):
        if isinstance(dst, VectorRegOp):
            return tuple(dst.regs)
        if isinstance(dst, RegOp):
            return (dst.name,)
        return ()

    # Count defs per vreg.
    def_count: dict[str, int] = {}
    for inst in all_instrs:
        for nm in _dest_vregs(inst.dest):
            def_count[nm] = def_count.get(nm, 0) + 1

    # Count uses per vreg (across the whole function), by category.
    base_uses: dict[str, int] = {}    # uses as MemOp.base in ld/st/atom global
    other_uses: dict[str, int] = {}   # any other source use
    for inst in all_instrs:
        for src in (inst.srcs or []):
            if isinstance(src, MemOp) and isinstance(src.base, str):
                bn = src.base if src.base.startswith('%') else f'%{src.base}'
                if (inst.op in ('ld', 'st', 'atom')
                        and 'global' in inst.types):
                    base_uses[bn] = base_uses.get(bn, 0) + 1
                else:
                    other_uses[bn] = other_uses.get(bn, 0) + 1
            elif isinstance(src, VectorRegOp):
                for v in src.regs:
                    other_uses[v] = other_uses.get(v, 0) + 1
            elif isinstance(src, RegOp):
                other_uses[src.name] = other_uses.get(src.name, 0) + 1

    fold_map: dict[str, tuple[str, int]] = {}
    dead_adds: set[int] = set()

    # 24-bit signed range for the LDG.E offset field.
    OFF_MAX = (1 << 23) - 1
    OFF_MIN = -(1 << 23)

    for inst in all_instrs:
        if (inst.op != 'add'
                or not any(t in ('u64', 's64', 'b64') for t in inst.types)):
            continue
        if not isinstance(inst.dest, RegOp) or len(inst.srcs or []) < 2:
            continue
        a, b = inst.srcs[0], inst.srcs[1]
        # Only fold the form (reg, imm).  (imm, reg) is rare in PTX.
        if not isinstance(a, RegOp) or not isinstance(b, ImmOp):
            continue
        imm = b.value
        if not (OFF_MIN <= imm <= OFF_MAX):
            continue
        dest_name = inst.dest.name
        # %A must have exactly one def (this add) and exactly one use,
        # and that use must be a MemOp.base in a global ld/st/atom.
        if def_count.get(dest_name, 0) != 1:
            continue
        if base_uses.get(dest_name, 0) != 1:
            continue
        if other_uses.get(dest_name, 0) != 0:
            continue
        # Base vreg must still be alive at the consumer.  We don't model
        # liveness here precisely; we trust that PTX vregs aren't
        # redefined in the typical aliased-base pattern.  If %B has more
        # than one def, skip — its value at the load may differ from
        # what the add saw.
        if def_count.get(a.name, 0) != 1:
            continue
        fold_map[dest_name] = (a.name, imm)
        dead_adds.add(id(inst))

    return fold_map, dead_adds


def _alloc_gpr_pair(ctx: 'ISelContext') -> int:
    """Allocate an even-aligned GPR pair (lo, lo+1) for 64-bit scratch use.

    Unlike calling _alloc_gpr twice and retrying on odd results, this properly
    returns odd-indexed registers to the pool instead of discarding them.
    Returns the lo (even) register index; the hi is lo+1.
    """
    limit = getattr(ctx, '_gpr_limit', _GPR_HARD_LIMIT_DEFAULT)
    # First pass: look for an even register already in the pool.
    odd_rejects: list[int] = []
    while ctx._scratch_pool:
        r = ctx._scratch_pool.pop()
        if r >= limit:
            continue  # discard out-of-range
        if r % 2 == 0:
            # Found an even base — put the rejects back and return it.
            ctx._scratch_pool.extend(odd_rejects)
            return r
        odd_rejects.append(r)
    # No even register in pool — put rejects back and allocate fresh.
    ctx._scratch_pool.extend(odd_rejects)
    if ctx._next_gpr % 2 != 0:
        # Advance to next even boundary; give the skipped odd reg to the pool.
        ctx._scratch_pool.append(ctx._next_gpr)
        ctx._next_gpr += 1
    if ctx._next_gpr + 1 < limit:
        r = ctx._next_gpr
        ctx._next_gpr += 2  # consume both lo and hi
        ctx._scratch_highwater = max(ctx._scratch_highwater, ctx._next_gpr)
        return r
    return 0


def _mark_scratch(ctx: 'ISelContext'):
    """Save current GPR watermark. Call before a multi-instruction sequence."""
    ctx._scratch_mark = ctx._next_gpr


def _release_scratch(ctx: 'ISelContext'):
    """Release all GPRs allocated since the last _mark_scratch call."""
    if ctx._scratch_mark >= 0:
        for r in range(ctx._scratch_mark, ctx._next_gpr):
            if r not in ctx._scratch_pool:
                ctx._scratch_pool.append(r)
        ctx._scratch_mark = -1


def _emit_lop3(output: list, ctx: 'ISelContext', dest: int, src0: int,
               src1: int, src2: int, lut: int, comment: str = ''):
    """Emit LOP3.LUT with register safety. On SM_120, LOP3 dest must be < R14
    (hardware limitation of the logic execution unit). If dest >= 14, use a
    scratch register and MOV the result."""
    if dest < 14:
        output.append(SassInstr(encode_lop3(dest, src0, src1, src2, lut), comment))
    else:
        # LOP3 to a low scratch, then MOV to actual dest.
        used = {src0, src1, src2, dest}
        scratch = next((r for r in range(14) if r not in used), 0)
        output.append(SassInstr(encode_lop3(scratch, src0, src1, src2, lut),
                                f'{comment} (via R{scratch})'))
        output.append(SassInstr(encode_iadd3(dest, scratch, RZ, RZ),
                                f'MOV R{dest}, R{scratch}  // lop3 fixup'))


def _alloc_scratch_pred(ctx: 'ISelContext', count: int = 1) -> list[int]:
    """Allocate scratch predicate registers."""
    regs = []
    for _ in range(count):
        r = ctx._next_pred
        ctx._next_pred += 1
    regs.append(r)
    return regs


def _materialize_imm(op: Operand, ctx: 'ISelContext', ra: RegAlloc,
                     output: list, bits: int = 32) -> int:
    """If op is an ImmOp, materialize it into a scratch GPR and return the index.
    If op is a RegOp, just return the register index. Handles 32-bit values."""
    if isinstance(op, RegOp):
        return ra.r32(op.name) if bits == 32 else ra.lo(op.name)
    if isinstance(op, ImmOp):
        val = op.value & 0xFFFFFFFF
        scratch = _alloc_gpr(ctx)
        # Use IADD3 Rd, RZ, imm, RZ to load a 32-bit immediate
        output.append(SassInstr(encode_iadd3_imm32(scratch, RZ, val, RZ),
                                f'MOV R{scratch}, 0x{val:x}  // materialize imm'))
        return scratch
    raise ISelError(f"Expected register or immediate operand, got {op!r}")


# ---------------------------------------------------------------------------
# PTX → SASS per-instruction mappers
# ---------------------------------------------------------------------------

_SPECIAL_REGS = {
    '%tid.x': SR_TID_X, '%tid.y': SR_TID_Y, '%tid.z': SR_TID_Z,
    '%ctaid.x': SR_CTAID_X, '%ctaid.y': SR_CTAID_Y, '%ctaid.z': SR_CTAID_Z,
    '%ntid.x': SR_NTID_X, '%ntid.y': SR_NTID_Y, '%ntid.z': SR_NTID_Z,
    '%nctaid.x': SR_NCTAID_X, '%nctaid.y': SR_NCTAID_Y, '%nctaid.z': SR_NCTAID_Z,
    # FG-2.6: %laneid maps to SR_LANEID (0x00) — the warp lane index.
    # PTXAS ground truth: `mov.u32 %r, %laneid;` lowers to
    # `S2R R<n>, SR_LANEID` (bytes 19 79 00 00 00 00 00 00 ...).
    '%laneid': SR_LANEID,
}

# Constant bank offsets for system values (driver-populated)
_CBANK_NTID_X = 0x360      # SM_120: blockDim.x at c[0][0x360]
_CBANK_NTID_X_SM89 = 0x0   # SM_89:  blockDim.x at c[0][0x0]


def _select_mov(instr: Instruction, ra: RegAlloc,
                ctx: 'ISelContext' = None) -> list[SassInstr]:
    """mov.u32 or mov.u64 (register-register or special register read)."""
    typ = instr.types[0] if instr.types else 'u32'
    dest = instr.dest
    src = instr.srcs[0]
    sm_ver = ctx.sm_version if ctx else 120

    if not isinstance(dest, RegOp):
        raise ISelError(f"MOV dest must be register: {dest!r}")

    # Check for special register source (threadIdx.x, blockIdx.x, etc.)
    if isinstance(src, RegOp) and src.name in _SPECIAL_REGS:
        d = ra.r32(dest.name)
        sr = _SPECIAL_REGS[src.name]

        # ntid.x: load from constant bank instead of S2R.
        # The driver populates the cbank offset with blockDim.x.
        if src.name == '%ntid.x':
            if sm_ver == 89:
                # SM_89: ntid.x at c[0][0x0], load via IMAD.MOV.U32
                from sass.encoding.sm_89_opcodes import encode_imad_mov_u32_cbuf
                ntid_off = _CBANK_NTID_X_SM89
                if ctx:
                    ctx._reg_param_off[dest.name] = ntid_off
                return [SassInstr(encode_imad_mov_u32_cbuf(d, 0, ntid_off),
                                  f'IMAD.MOV.U32 R{d}, RZ, RZ, c[0][0x{ntid_off:x}]  // ntid.x')]
            else:
                return [SassInstr(encode_ldc(d, 0, _CBANK_NTID_X),
                                  f'LDC R{d}, c[0][0x{_CBANK_NTID_X:x}]  // ntid.x')]

        return [SassInstr(encode_s2r(d, sr),
                          f'S2R R{d}, SR_{src.name}  // {dest.name} = {src.name}')]

    if isinstance(src, ImmOp):
        if typ in ('u64', 's64', 'b64', 'f64'):
            # 64-bit immediate: split into lo/hi 32-bit halves
            bits = src.value & 0xFFFFFFFFFFFFFFFF
            lo = bits & 0xFFFFFFFF
            hi = (bits >> 32) & 0xFFFFFFFF
            d_lo = ra.lo(dest.name)
            d_hi = d_lo + 1
            if ctx and hi == 0:
                if not hasattr(ctx, '_zero_regs'):
                    ctx._zero_regs = set()
                ctx._zero_regs.add(d_hi)
            return [
                SassInstr(encode_iadd3_imm32(d_lo, RZ, lo, RZ),
                          f'IADD3 R{d_lo}, RZ, 0x{lo:x}, RZ  // {dest.name}.lo = imm'),
                SassInstr(encode_iadd3_imm32(d_hi, RZ, hi, RZ),
                          f'IADD3 R{d_hi}, RZ, 0x{hi:x}, RZ  // {dest.name}.hi = imm'),
            ]
        raise ISelError("MOV from immediate not yet supported in isel (use LDC for params)")

    if not isinstance(src, RegOp):
        # Handle mov.u64 %rd, smem_name — shared memory base address
        from ptx.ir import LabelOp
        if isinstance(src, LabelOp) and ctx and hasattr(ctx, '_smem_offsets'):
            smem_off = ctx._smem_offsets.get(src.name, None)
            if smem_off is not None:
                # Shared memory base = offset within shared space (always 0 for first decl)
                d_lo = ra.lo(dest.name)
                d_hi = d_lo + 1
                return [
                    SassInstr(encode_iadd3_imm32(d_lo, RZ, smem_off, RZ),
                              f'IADD3 R{d_lo}, RZ, 0x{smem_off:x}, RZ  // smem base lo'),
                    SassInstr(encode_iadd3_imm32(d_hi, RZ, 0, RZ),
                              f'IADD3 R{d_hi}, RZ, 0, RZ  // smem base hi'),
                ]
        raise ISelError(f"MOV src must be register: {src!r}")

    if typ in ('u64', 's64', 'b64', 'f64'):
        # 64-bit: two 32-bit MOVs
        d_lo = ra.lo(dest.name)
        d_hi = d_lo + 1
        s_lo = ra.lo(src.name)
        s_hi = s_lo + 1
        instrs = []
        if d_lo != s_lo:
            instrs.append(SassInstr(encode_iadd3(d_lo, s_lo, RZ, RZ),
                                    f'MOV R{d_lo}, R{s_lo}  // {dest.name}.lo = {src.name}.lo'))
        if d_hi != s_hi:
            instrs.append(SassInstr(encode_iadd3(d_hi, s_hi, RZ, RZ),
                                    f'MOV R{d_hi}, R{s_hi}  // {dest.name}.hi = {src.name}.hi'))
        return instrs or [_nop(f'MOV {dest.name} = {src.name} (same reg, elided)')]
    else:
        # 32-bit
        d = ra.r32(dest.name)
        s = ra.r32(src.name)
        return [SassInstr(encode_iadd3(d, s, RZ, RZ), f'MOV R{d}, R{s}  // {dest.name} = {src.name}')]


def _select_shl_b64(instr: Instruction, ra: RegAlloc,
                    ctx: 'ISelContext' = None,
                    output: list = None) -> list[SassInstr]:
    """
    shl.b64 dest, src, K → SHF.L.U32 (lo) + SHF.L.U64.HI (hi).

    64-bit left shift by constant K:
      dest.lo = src.lo << K               (via SHF.L.U32, src1=RZ for low bits)
      dest.hi = funnel_shift(src.hi, src.lo, K)  (via SHF.L.U64.HI)
    """
    dest = instr.dest
    src  = instr.srcs[0]
    k_op = instr.srcs[1]
    if not isinstance(dest, RegOp) or not isinstance(src, RegOp):
        raise ISelError(f"shl.b64: dest/src must be registers")
    k = _get_imm(k_op)
    d_lo = ra.lo(dest.name); d_hi = d_lo + 1
    s_lo = ra.lo(src.name);  s_hi = s_lo + 1

    # IMAD.WIDE fusion: if source was cvt.u64.u32, the hi word is zero.
    # Use IMAD.WIDE to compute both lo and hi in one instruction:
    #   (dest, dest+1) = src32 * (1<<K) + RZ
    # This replaces MOV+MOV+IMAD.SHL+SHF with a single wide multiply.
    if (ctx and k < 32 and k <= 7
            and hasattr(ctx, '_cvt_src_map') and src.name in ctx._cvt_src_map):
        src32 = ctx._cvt_src_map[src.name]
        from sass.encoding.sm_120_opcodes import encode_imad_wide
        # Remove the dead MOV+MOV from cvt.u64.u32 (they're the last 1-2 instrs)
        if output is not None:
            # Remove trailing MOV hi=0
            if output and 'cvt.64.32 hi=0' in output[-1].comment:
                output.pop()
            # Remove trailing MOV lo copy (if present)
            if output and 'cvt.64.32 lo' in output[-1].comment:
                output.pop()
        # TE11: mark dest as widened from 32 bits for IADD3.R-UR substitution
        if ctx:
            if not hasattr(ctx, '_widened_from_32'):
                ctx._widened_from_32 = set()
            ctx._widened_from_32.add(dest.name)
        return [
            SassInstr(encode_imad_wide(d_lo, src32, 1 << k, RZ),
                      f'IMAD.WIDE R{d_lo}, R{src32}, {1<<k:#x}, RZ  // LEA: ({dest.name}.lo,hi) = {src.name} * {1<<k}'),
        ]

    if k < 32 and k <= 15:
        # Use IMAD.SHL for lo (avoids SHF.L/SHF.R pipeline conflicts)
        return [
            SassInstr(encode_imad_shl_u32(d_lo, s_lo, k),
                      f'IMAD.SHL.U32 R{d_lo}, R{s_lo}, {1<<k:#x}, RZ  // {dest.name}.lo = {src.name}.lo << {k}'),
            SassInstr(encode_shf_l_u64_hi(d_hi, s_lo, k, s_hi),
                      f'SHF.L.U64.HI R{d_hi}, R{s_lo}, 0x{k:x}, R{s_hi}  // {dest.name}.hi'),
        ]
    elif k < 32:
        return [
            SassInstr(encode_shf_l_u32(d_lo, s_lo, k),
                      f'SHF.L.U32 R{d_lo}, R{s_lo}, 0x{k:x}, RZ  // {dest.name}.lo = {src.name}.lo << {k}'),
            SassInstr(encode_shf_l_u64_hi(d_hi, s_lo, k, s_hi),
                      f'SHF.L.U64.HI R{d_hi}, R{s_lo}, 0x{k:x}, R{s_hi}  // {dest.name}.hi'),
        ]
    else:
        # K >= 32: result.hi = src.lo << (K-32), result.lo = 0
        k32 = k - 32
        return [
            SassInstr(encode_iadd3(d_lo, RZ, RZ, RZ),
                      f'MOV R{d_lo}, RZ  // shl.b64 lo = 0 (K>={k})'),
            SassInstr(encode_shf_l_u32(d_hi, s_lo, k32),
                      f'SHF.L.U32 R{d_hi}, R{s_lo}, 0x{k32:x}, RZ  // shl.b64 hi'),
        ]


def _select_rotl64(instr: Instruction, ra: RegAlloc) -> list[SassInstr]:
    """
    Correct 64-bit rotate-left: produces two SHF.L.W.U32.HI instructions.
    The source PTX pattern: add(shl(a,K), shr(a, 64-K)).

    This is the CORRECT transformation that ptxas gets wrong when it sees
    sub(shl(a,K), shr(a, 64-K)) — our rotate.py pass detects this.
    """
    dest = instr.dest
    src  = instr.srcs[0]
    k_op = instr.srcs[1]
    if not isinstance(dest, RegOp) or not isinstance(src, RegOp):
        raise ISelError(f"rotl64: dest/src must be registers")
    k = _get_imm(k_op)
    d_lo = ra.lo(dest.name); d_hi = d_lo + 1
    s_lo = ra.lo(src.name);  s_hi = s_lo + 1
    return [
        SassInstr(encode_shf_l_w_u32_hi(d_lo, s_lo, k, s_hi),
                  f'SHF.L.W.U32.HI R{d_lo}, R{s_lo}, 0x{k:x}, R{s_hi}  // rotl64 lo'),
        SassInstr(encode_shf_l_w_u32_hi(d_hi, s_hi, k, s_lo),
                  f'SHF.L.W.U32.HI R{d_hi}, R{s_hi}, 0x{k:x}, R{s_lo}  // rotl64 hi'),
    ]


def _select_shr_u64(instr: Instruction, ra: RegAlloc) -> list[SassInstr]:
    """
    shr.u64 dest, src, K → right shift by constant K.

    For K < 32: standard SHF.R.U64 + SHF.R.U32.HI pair.
    For K >= 32: optimized — shift the HIGH word right by (K-32), dest.hi = 0.
    """
    dest = instr.dest
    src  = instr.srcs[0]
    k_op = instr.srcs[1]
    if not isinstance(dest, RegOp) or not isinstance(src, RegOp):
        raise ISelError(f"shr.u64: dest/src must be registers")
    k = _get_imm(k_op)
    d_lo = ra.lo(dest.name); d_hi = d_lo + 1
    s_lo = ra.lo(src.name);  s_hi = s_lo + 1


    if k < 32:
        return [
            SassInstr(encode_shf_r_u32(d_lo, s_lo, k, s_hi),
                      f'SHF.R.U64 R{d_lo}, R{s_lo}, 0x{k:x}, R{s_hi}  // shr.u64 lo'),
            SassInstr(encode_shf_r_u32_hi(d_hi, s_hi, k),
                      f'SHF.R.U32.HI R{d_hi}, RZ, 0x{k:x}, R{s_hi}  // shr.u64 hi'),
        ]
    else:
        # K >= 32: result.lo = src.hi >> (K-32), result.hi = 0
        k32 = k - 32
        return [
            SassInstr(encode_shf_r_u32_hi(d_lo, s_hi, k32),
                      f'SHF.R.U32.HI R{d_lo}, RZ, 0x{k32:x}, R{s_hi}  // shr.u64 lo (K>={k})'),
            SassInstr(encode_iadd3(d_hi, RZ, RZ, RZ),
                      f'MOV R{d_hi}, RZ  // shr.u64 hi = 0'),
        ]


def _select_sub_u64(instr: Instruction, ra: RegAlloc) -> list[SassInstr]:
    """
    sub.u64/s64 dest, a, b → IADD.64 with negation on b.

    Uses dest=a_lo (in-place) to keep registers within R0-R7 range.
    IADD.64 reads both sources before writing, so dest=src0 is safe.
    """
    dest = instr.dest
    a    = instr.srcs[0]
    b    = instr.srcs[1]
    if not isinstance(dest, RegOp) or not isinstance(a, RegOp) or not isinstance(b, RegOp):
        raise ISelError(f"sub.u64: all operands must be registers")

    a_lo = ra.lo(a.name); a_hi = a_lo + 1
    b_lo = ra.lo(b.name); b_hi = b_lo + 1
    d_lo = ra.lo(dest.name); d_hi = d_lo + 1

    # SM_120 rule: IADD.64 R-R (0x235) is broken. Use IADD3+IADD3.X.
    # sub.u64: d = a + (-b). IADD3 with negate on src1.
    return [
        SassInstr(encode_iadd3(d_lo, a_lo, b_lo, RZ, negate_src1=True),
                  f'IADD3 R{d_lo}, R{a_lo}, -R{b_lo}, RZ  // sub.u64 lo'),
        SassInstr(encode_iadd3x(d_hi, a_hi, b_hi, RZ, negate_src1=True),
                  f'IADD3.X R{d_hi}, R{a_hi}, -R{b_hi}, RZ  // sub.u64 hi'),
    ]


def _select_add_u64(instr: Instruction, ra: RegAlloc,
                    ctx: 'ISelContext' = None) -> list[SassInstr]:
    """add.u64 dest, a, b → IADD.64 (SM_120) or IADD3+IADD3.X (SM_89)."""
    from sass.encoding.sm_120_opcodes import encode_iadd64_ur
    # WB-7: skip add.u64 marked dead by analyze_addr_offset_fold —
    # the immediate offset has been folded into the consuming load/store
    # and this add.u64's destination is no longer needed.
    if id(instr) in getattr(ctx, '_addr_fold_dead_adds', set()):
        return []
    dest = instr.dest
    a    = instr.srcs[0]
    b    = instr.srcs[1]
    sm_ver = ctx.sm_version if ctx else 120
    # WB-10: skip self-add-zero (`add.u64 %X, %X, 0`) — used in PTX
    # as a "materialize to GPR" hint that's redundant when the param
    # is already loaded directly via LDC.64 (direct_ldc_params path).
    # Only fires when the param vreg is in direct_ldc_params, so
    # legitimate `add.u64 X, X, 0` outside that context still emits
    # (preserves whatever side-effect the original code expected).
    if (isinstance(a, RegOp) and isinstance(dest, RegOp)
            and a.name == dest.name
            and isinstance(b, ImmOp) and b.value == 0
            and dest.name in getattr(ctx, '_direct_ldc_params', set())):
        return []

    # Handle immediate operand: add.u64 dest, a, imm  (e.g. loop counter increment by 1)
    if isinstance(b, ImmOp) and isinstance(dest, RegOp) and isinstance(a, RegOp):
        # If 'a' is a deferred param, emit inline LDCU.64 UR6 + materialize
        a_deferred_imm = (ctx and a.name in ctx._deferred_ur_params
                          and a.name not in getattr(ctx, '_gpr_written', set()))
        if a_deferred_imm:
            param_off = ctx._deferred_ur_params.pop(a.name)
            ur_tmp = 6  # always reuse UR6
            d_lo = ra.lo(dest.name)
            limit = getattr(ctx, '_gpr_limit', _GPR_HARD_LIMIT_DEFAULT)
            if d_lo >= limit:
                if not hasattr(ctx, '_addr_scratch'):
                    ctx._addr_scratch = 10
                d_lo = ctx._addr_scratch
                ra.int_regs[dest.name] = d_lo
            ctx._gpr_written.add(dest.name)
            prefix = [SassInstr(encode_ldcu_64(ur_tmp, 0, param_off),
                                f'LDCU.64 UR{ur_tmp}, c[0][0x{param_off:x}]  // deferred param')]
            if b.value == 0:
                return prefix + _emit_ur_to_gpr(d_lo, ur_tmp, 'add.u64 imm0 (deferred UR->GPR)')
            else:
                imm_lo = b.value & 0xFFFFFFFF
                return prefix + _emit_ur_to_gpr(d_lo, ur_tmp, 'deferred UR->GPR') + [
                    SassInstr(encode_iadd3_imm32(d_lo, d_lo, imm_lo, RZ),
                              f'IADD3.IMM R{d_lo}, R{d_lo}, {imm_lo:#x}, RZ  // add.u64 lo imm'),
                    SassInstr(encode_iadd3x(d_lo + 1, d_lo + 1, RZ, RZ),
                              f'IADD3.X R{d_lo+1}, R{d_lo+1}, RZ, RZ  // add.u64 hi carry'),
                ]
        # If 'a' is in UR space (loaded via LDCU.64), must materialize first.
        # Skip if 'a' was already written to GPR (e.g., by a previous add.u64).
        a_in_ur = (ctx and a.name in ctx._ur_params
                   and a.name not in getattr(ctx, '_gpr_written', set()))
        if a_in_ur:
            ur_idx = ctx._ur_params[a.name]
            d_lo = ra.lo(dest.name)
            limit = getattr(ctx, '_gpr_limit', _GPR_HARD_LIMIT_DEFAULT)
            if d_lo >= limit:
                if not hasattr(ctx, '_addr_scratch'):
                    ctx._addr_scratch = 10
                d_lo = ctx._addr_scratch
                ra.int_regs[dest.name] = d_lo
            if ctx:
                ctx._gpr_written.add(dest.name)
            if b.value == 0:
                # add.u64 dest, ur_param, 0 → just materialize UR to GPR
                return _emit_ur_to_gpr(d_lo, ur_idx, 'add.u64 imm0 (UR->GPR)')
            else:
                # Materialize UR→GPR, then add immediate
                imm_lo = b.value & 0xFFFFFFFF
                return _emit_ur_to_gpr(d_lo, ur_idx, 'materialize UR->GPR') + [
                    SassInstr(encode_iadd3_imm32(d_lo, d_lo, imm_lo, RZ),
                              f'IADD3.IMM R{d_lo}, R{d_lo}, {imm_lo:#x}, RZ  // add.u64 lo imm'),
                    SassInstr(encode_iadd3x(d_lo + 1, d_lo + 1, RZ, RZ),
                              f'IADD3.X R{d_lo+1}, R{d_lo+1}, RZ, RZ  // add.u64 hi carry'),
                ]
        d_lo = ra.lo(dest.name)
        a_lo = ra.lo(a.name)
        imm_lo = b.value & 0xFFFFFFFF
        if ctx:
            ctx._gpr_written.add(dest.name)
        # IADD3.IMM lo + IADD3.X hi (carry propagates via hardcoded predicate bits)
        return [
            SassInstr(encode_iadd3_imm32(d_lo, a_lo, imm_lo, RZ),
                      f'IADD3.IMM R{d_lo}, R{a_lo}, {imm_lo:#x}, RZ  // add.u64 lo imm'),
            SassInstr(encode_iadd3x(d_lo + 1, a_lo + 1, RZ, RZ),
                      f'IADD3.X R{d_lo+1}, R{a_lo+1}, RZ, RZ  // add.u64 hi carry'),
        ]

    if not isinstance(dest, RegOp) or not isinstance(a, RegOp) or not isinstance(b, RegOp):
        raise ISelError(f"add.u64: all operands must be registers")

    if sm_ver == 89:
        # SM_89: no IADD.64 instruction. Use IADD3.cb + IADD3.X.cb when one
        # operand is a 64-bit param (read directly from constant bank), or
        # IADD3 + IADD3.X R-R for two GPR operands.
        from sass.encoding.sm_89_opcodes import encode_iadd3_cbuf, encode_iadd3x_cbuf

        a_cbuf = ctx._reg_param_off.get(a.name) if ctx else None
        b_cbuf = ctx._reg_param_off.get(b.name) if ctx else None
        a_in_gpr = ctx and a.name in ctx._gpr_written
        b_in_gpr = ctx and b.name in ctx._gpr_written

        if a_cbuf is not None and not a_in_gpr:
            # a is in cbuf, b is in GPR → IADD3.cb dest, b, c[0][a_off], RZ
            # Use P1 for carry to avoid clobbering the execution predicate (P0).
            r_lo = ra.lo(b.name)
            d_lo = ra.lo(dest.name); d_hi = d_lo + 1
            if ctx:
                ctx._gpr_written.add(dest.name)
            return [
                SassInstr(encode_iadd3_cbuf(d_lo, r_lo, 0, a_cbuf, RZ, pred_out=1),
                          f'IADD3 R{d_lo}, P1, R{r_lo}, c[0][0x{a_cbuf:x}], RZ  // add.u64 lo cbuf'),
                SassInstr(encode_iadd3x_cbuf(d_hi, r_lo + 1, 0, a_cbuf + 4, RZ, pred_in=1),
                          f'IADD3.X R{d_hi}, R{r_lo+1}, c[0][0x{a_cbuf+4:x}], RZ, P1  // add.u64 hi cbuf'),
            ]
        elif b_cbuf is not None and not b_in_gpr:
            # b is in cbuf, a is in GPR → IADD3.cb dest, a, c[0][b_off], RZ
            r_lo = ra.lo(a.name)
            d_lo = ra.lo(dest.name); d_hi = d_lo + 1
            if ctx:
                ctx._gpr_written.add(dest.name)
            return [
                SassInstr(encode_iadd3_cbuf(d_lo, r_lo, 0, b_cbuf, RZ, pred_out=1),
                          f'IADD3 R{d_lo}, P1, R{r_lo}, c[0][0x{b_cbuf:x}], RZ  // add.u64 lo cbuf'),
                SassInstr(encode_iadd3x_cbuf(d_hi, r_lo + 1, 0, b_cbuf + 4, RZ, pred_in=1),
                          f'IADD3.X R{d_hi}, R{r_lo+1}, c[0][0x{b_cbuf+4:x}], RZ, P1  // add.u64 hi cbuf'),
            ]
        else:
            # Both in GPR → IADD3 + IADD3.X R-R
            a_lo = ra.lo(a.name)
            a_hi = (ctx._pair_hi_override.get(a.name, a_lo + 1)
                    if ctx and hasattr(ctx, '_pair_hi_override') else a_lo + 1)
            b_lo = ra.lo(b.name)
            b_hi = (ctx._pair_hi_override.get(b.name, b_lo + 1)
                    if ctx and hasattr(ctx, '_pair_hi_override') else b_lo + 1)
            d_lo = ra.lo(dest.name); d_hi = d_lo + 1
            if ctx:
                ctx._gpr_written.add(dest.name)
            return [
                SassInstr(encode_iadd3(d_lo, a_lo, b_lo, RZ),
                          f'IADD3 R{d_lo}, R{a_lo}, R{b_lo}, RZ  // add.u64 lo (R-R safe)'),
                SassInstr(encode_iadd3x(d_hi, a_hi, b_hi, RZ),
                          f'IADD3.X R{d_hi}, R{a_hi}, R{b_hi}, RZ  // add.u64 hi (R-R safe)'),
            ]

    # SM_120 path
    # Check for deferred params (4th+ pointer param, not yet loaded)
    deferred = getattr(ctx, '_deferred_ur_params', {}) if ctx else {}
    a_deferred = a.name in deferred
    b_deferred = b.name in deferred
    if a_deferred or b_deferred:
        # Inline LDCU.64 UR6 → IADD.64 R-UR for deferred param
        if a_deferred:
            param_off = deferred.get(a.name)
            r_lo = ra.lo(b.name)
        else:
            param_off = deferred.get(b.name)
            r_lo = ra.lo(a.name)
        d_lo = ra.lo(dest.name) if dest.name in ra.int_regs else r_lo
        limit = getattr(ctx, '_gpr_limit', _GPR_HARD_LIMIT_DEFAULT)
        if d_lo >= limit:
            if not hasattr(ctx, '_addr_scratch'):
                ctx._addr_scratch = 10
            d_lo = ctx._addr_scratch
            ra.int_regs[dest.name] = d_lo
        ur_tmp = 6  # always reuse UR6 for deferred loads
        if ctx:
            ctx._gpr_written.add(dest.name)
        return [
            SassInstr(encode_ldcu_64(ur_tmp, 0, param_off),
                      f'LDCU.64 UR{ur_tmp}, c[0][0x{param_off:x}]  // deferred param'),
            SassInstr(encode_iadd64_ur(d_lo, r_lo, ur_tmp),
                      f'IADD.64 R{d_lo}, R{r_lo}, UR{ur_tmp}  // add.u64 (deferred UR)'),
        ]

    # Check if either operand is a UR param (loaded via LDCU).
    # Skip if the register was already written to GPR (e.g., by a previous
    # add.u64 increment) — _gpr_written tracks modified registers.
    gpr_written = getattr(ctx, '_gpr_written', set()) if ctx else set()
    a_in_ur = ctx and a.name in ctx._ur_params and a.name not in gpr_written
    b_in_ur = ctx and b.name in ctx._ur_params and b.name not in gpr_written

    if a_in_ur or b_in_ur:
        if a_in_ur:
            ur_idx = ctx._ur_params[a.name]
            r_lo = ra.lo(b.name)
        else:
            ur_idx = ctx._ur_params[b.name]
            r_lo = ra.lo(a.name)
        d_lo = ra.lo(dest.name) if dest.name in ra.int_regs else r_lo
        limit = getattr(ctx, '_gpr_limit', _GPR_HARD_LIMIT_DEFAULT)
        if d_lo >= limit:
            if not hasattr(ctx, '_addr_scratch'):
                ctx._addr_scratch = 10
            d_lo = ctx._addr_scratch
            ra.int_regs[dest.name] = d_lo
        if ctx:
            ctx._gpr_written.add(dest.name)

        # TE11-B note: IADD3.R-UR carry chain (0xc11) was attempted as a
        # replacement for IADD.64-UR.  61 regressions — the encoding needs
        # GPU verification and the carry propagation between lo/hi halves
        # requires additional scoreboard support.  Kept as future work.

        # IADD.64-UR (original, working path)
        return [
            SassInstr(encode_iadd64_ur(d_lo, r_lo, ur_idx),
                      f'IADD.64 R{d_lo}, R{r_lo}, UR{ur_idx}  // add.u64 (UR base)'),
        ]
    else:
        # Both operands in R bank.
        # SM_120 rule: IADD.64 R-R (0x235) is broken (causes 715).
        # Use IADD3 + IADD3.X pair instead (same as SM_89 path).
        _hi_map = getattr(ctx, '_pair_hi_override', {}) if ctx else {}
        a_lo = ra.lo(a.name)
        a_hi = _hi_map.get(a.name, a_lo + 1)
        b_lo = ra.lo(b.name)
        b_hi = _hi_map.get(b.name, b_lo + 1)
        d_lo = ra.lo(dest.name); d_hi = d_lo + 1
        if ctx:
            ctx._gpr_written.add(dest.name)
        return [
            SassInstr(encode_iadd3(d_lo, a_lo, b_lo, RZ),
                      f'IADD3 R{d_lo}, R{a_lo}, R{b_lo}, RZ  // add.u64 lo (R-R safe)'),
            SassInstr(encode_iadd3x(d_hi, a_hi, b_hi, RZ),
                      f'IADD3.X R{d_hi}, R{a_hi}, R{b_hi}, RZ  // add.u64 hi (R-R safe)'),
        ]


def _select_ld_param(instr: Instruction, ra: RegAlloc,
                     param_offsets: dict[str, int],
                     ctx: 'ISelContext' = None) -> list[SassInstr]:
    """
    ld.param.u64 → LDCU.64 (SM_120) or 2x IMAD.MOV.U32 (SM_89).
    ld.param.u32 → LDC (SM_120) or IMAD.MOV.U32 (SM_89).

    SM_120 descriptor-based memory model requires pointer params in UR bank.
    SM_89 loads params directly into GPR (no LDCU/LDC instructions).
    """
    dest = instr.dest
    src  = instr.srcs[0]
    if not isinstance(dest, RegOp):
        raise ISelError(f"ld.param dest must be register")

    from ptx.ir import MemOp
    if not isinstance(src, MemOp):
        raise ISelError(f"ld.param src must be MemOp, got {src!r}")

    param_name = src.base
    if isinstance(src.offset, int):
        byte_off = param_offsets.get(param_name, 0) + src.offset
    else:
        byte_off = param_offsets.get(param_name, 0)

    sm_ver = ctx.sm_version if ctx else 120
    typ = instr.types[-1] if instr.types else 'u32'

    if sm_ver == 89:
        # SM_89: use inline cbuf operands (IADD3.cb) for 64-bit params.
        # Don't load into GPR — just record the cbuf offset. The add.u64
        # handler will emit IADD3.cb + IADD3.X.cb to add register + cbuf directly.
        from sass.encoding.sm_89_opcodes import encode_imad_mov_u32_cbuf
        if typ in ('u64', 's64', 'b64'):
            # Record cbuf offset for inline use by add.u64
            if ctx:
                ctx._reg_param_off[dest.name] = byte_off
                # Mark as "cbuf only" — NOT in GPR, NOT in UR
            return []  # No GPR load — read inline from cbuf
        else:
            # u32: single IMAD.MOV.U32 into GPR
            if dest.name not in ra.int_regs:
                return []  # dead parameter
            d = ra.r32(dest.name)
            if ctx:
                ctx._reg_param_off[dest.name] = byte_off
            return [SassInstr(encode_imad_mov_u32_cbuf(d, 0, byte_off),
                              f'IMAD.MOV.U32 R{d}, RZ, RZ, c[0][0x{byte_off:x}]  // {param_name}')]

    # SM_120 path (original)
    if typ == 'f64':
        # SM_120: Load f64 param into a UR pair via LDCU.64.
        # DFMA R-R-UR-UR uses the UR operands directly, keeping all regular
        # GPRs within R0-R13 and avoiding the R14+ ILLEGAL_INSTRUCTION restriction.
        ur_idx = ctx._next_ur if ctx else 6
        if ctx:
            if ur_idx % 2 != 0:
                ur_idx += 1
                ctx._next_ur = ur_idx
            ctx._next_ur += 2
            ctx._ur_params[dest.name] = ur_idx
        return [
            SassInstr(encode_ldcu_64(ur_idx, 0, byte_off),
                      f'LDCU.64 UR{ur_idx}, c[0][0x{byte_off:x}]  // {param_name} (f64)'),
        ]

    if typ in ('u64', 's64', 'b64'):
        # WB-5.0: tiny-kernel direct LDC.64 path.  Set by the allocator
        # when this is the only u64 param AND its only use is a single
        # MemOp.base in a global memory op.  Loading via LDC.64 avoids
        # the LDCU.64 + IADD.64 R-UR materialization (saves 2 SASS
        # instructions) AND lets the kernel skip the unconditional S2R
        # because no LDCU param load remains in the body.  Matches
        # ptxas's tight tiny-kernel pattern.
        if (ctx and ctx.sm_version == 120
                and dest.name in getattr(ctx, '_direct_ldc_params', set())
                and dest.name in ra.int_regs):
            dr = ctx.ra.lo(dest.name)
            ctx._gpr_written.add(dest.name)
            return [
                SassInstr(encode_ldc_64(dr, 0, byte_off),
                          f'LDC.64 R{dr}, c[0][0x{byte_off:x}]  // {param_name} (tiny direct)'),
            ]
        # SM_120 preamble interleaving: ALL pointer params are deferred.
        # Each add.u64 / ld.global / atom emits inline LDCU.64 UR6 + IADD.64,
        # reusing UR6 each time. No UR pressure, no clobber hazards.
        # For post-EXIT params (after bounds check), materialize immediately
        # via LDCU.64 UR6 + _emit_ur_to_gpr so LDCU stays in post-EXIT region.
        if ctx and ctx.sm_version == 120:
            # UR4+5 for descriptor, UR6+7 reserved. Params start at UR8.
            if ctx._next_ur < 8:
                ctx._next_ur = 8
            # Safety: predicated ld.param.u64 is in a divergent path (if-converted).
            # The deferred LDCU.64 would execute unconditionally (warp-wide) but
            # the consumer is predicated — UR/GPR conflicts across paths.
            # Fall back to LDC pair for predicated params.
            #
            # FG-4.4 Bug 1 fix: if the allocator has already given this
            # param register a GPR (dest.name in ra.int_regs), the
            # register is being redefined later (u64_def_count > 1 path
            # in regalloc.py) and cannot participate in the UR-bound
            # code path — later consumers via _select_add_u64 would
            # see a stale ur_params entry AND a ra.lo() lookup would
            # KeyError.  Route directly to the GPR via LDC.64.
            if instr.pred or dest.name in ra.int_regs:
                dr = ctx.ra.lo(dest.name)
                ctx._gpr_written.add(dest.name)
                return [
                    SassInstr(encode_ldc_64(dr, 0, byte_off),
                              f'LDC.64 R{dr}, c[0][0x{byte_off:x}]  // {param_name} (GPR direct)'),
                ]
            # Preamble preload: assign UR pair, record for preamble emission.
            # The LDCU is emitted by compile_function in the preamble window.
            # Body code uses _ur_params to find the UR index.
            ur_idx = ctx._next_ur
            if ur_idx % 2 != 0:
                ur_idx += 1          # LDCU.64 requires even-aligned UR
            ctx._next_ur = ur_idx + 2
            if not hasattr(ctx, '_preamble_ldcus'):
                ctx._preamble_ldcus = []
            ctx._preamble_ldcus.append(
                SassInstr(encode_ldcu_64(ur_idx, 0, byte_off),
                          f'LDCU.64 UR{ur_idx}, c[0][0x{byte_off:x}]  // preamble param {param_name}'))
            ctx._ur_params[dest.name] = ur_idx
            ctx._has_inline_deferred = True
            return []
        # Non-SM_120 fallback: load into UR pair
        ur_idx = ctx._next_ur if ctx else 6
        if ctx:
            if ur_idx % 2 != 0:
                ur_idx += 1
                ctx._next_ur = ur_idx
            ctx._next_ur = ur_idx + 2
            ctx._ur_params[dest.name] = ur_idx
        return [
            SassInstr(encode_ldcu_64(ur_idx, 0, byte_off),
                      f'LDCU.64 UR{ur_idx}, c[0][0x{byte_off:x}]  // {param_name}'),
        ]
    else:
        if ctx:
            ctx._reg_param_off[dest.name] = byte_off
        if dest.name not in ra.int_regs:
            return []
        d = ra.r32(dest.name)

        # u32 params consumed only by setp:
        # TE9-B: UR-native path — load to UR via LDCU.32 (not GPR via LDC)
        # to enable ISETP.R-UR in the setp handler.  This is a 1:1
        # instruction swap (LDCU.32 replaces LDC), no extra instructions.
        # Trigger: SM_120 + no VOTE + no BAR + no atom.xor template.
        if ctx and dest.name in getattr(ctx, '_setp_only_params', set()):
            ctx._reg_param_off[dest.name] = byte_off
            d = ra.r32(dest.name)
            has_bar = getattr(ctx, '_has_bar_sync', False)
            has_vote = getattr(ctx, '_has_vote', False)
            has_ur_act = getattr(ctx, '_ur_activation_sr', None) is not None
            ur_native_ok = (ctx.sm_version == 120
                            and not has_bar and not has_vote and not has_ur_act)
            if ur_native_ok:
                # TE10-B: UR-native path.  LDCU.32 goes in the BODY so
                # the scoreboard sets correct rbar on the ISETP.R-UR.
                # GUARD: only for kernels with exactly ONE setp-only u32
                # param.  Multiple LDCU.32→ISETP.R-UR pairs in the same
                # kernel produce wrong comparison results (TE10 finding:
                # second ISETP.R-UR fails for 3-param kernels).
                # Guard: single setp-only u32 param AND param is used by at
                # most one setp instruction.  Multiple setp→EXIT sequences
                # reusing the same UR cause the second ISETP.R-UR to fail
                # (TE10 finding: @Px EXIT disrupts subsequent UR reads).
                _n_setp_only_u32 = sum(1 for p in getattr(ctx, '_setp_only_params', set())
                                        if p in ctx._reg_param_off)
                _n_setp_uses = getattr(ctx, '_setp_use_count', {}).get(dest.name, 1)
                if _n_setp_only_u32 <= 1 and _n_setp_uses <= 1:
                    ur_idx = ctx._next_ur
                    ctx._next_ur = ur_idx + 1
                    ctx._ur_params[dest.name] = ur_idx
                    return [SassInstr(encode_ldcu_32(ur_idx, 0, byte_off),
                                      f'LDCU.32 UR{ur_idx}, c[0][0x{byte_off:x}]  // TE10: setp UR-native')]
            # Fallback: GPR LDC path
            if ctx.sm_version == 120 and has_bar:
                if not hasattr(ctx, '_preamble_ldcus'):
                    ctx._preamble_ldcus = []
                ctx._preamble_ldcus.append(
                    SassInstr(encode_ldc(d, 0, byte_off),
                              f'LDC R{d}, c[0][0x{byte_off:x}]  // preamble setp param'))
                return []
            return [SassInstr(encode_ldc(d, 0, byte_off),
                              f'LDC R{d}, c[0][0x{byte_off:x}]  // setp param')]

        # SM_120 rule #25: body LDC (0xb82) causes ERR715 in kernels
        # with VOTE+LDG. ptxas loads scalar params via LDCU.64 instead.
        _has_vote_fn = getattr(ctx, '_has_vote', False) if ctx else False

        if _has_vote_fn:
            # VOTE present: load param in preamble (body LDC causes ERR715).
            if ctx:
                if not hasattr(ctx, '_preamble_ldcus'):
                    ctx._preamble_ldcus = []
                ctx._preamble_ldcus.append(
                    SassInstr(encode_ldc(d, 0, byte_off),
                              f'LDC R{d}, c[0][0x{byte_off:x}]  // preamble setp param'))
            return []
        else:
            return [SassInstr(encode_ldc(d, 0, byte_off, ctrl=0x7f1),
                              f'LDC R{d}, c[0][0x{byte_off:x}]  // {param_name}')]


def _select_ld_global(instr: Instruction, ra: RegAlloc,
                      ur_desc: int, ctx: 'ISelContext' = None) -> list[SassInstr]:
    """ld.global → LDG.E with appropriate width."""
    dest = instr.dest
    src  = instr.srcs[0]
    if not isinstance(dest, RegOp):
        raise ISelError(f"ld.global dest must be register")
    from ptx.ir import MemOp
    if not isinstance(src, MemOp):
        raise ISelError(f"ld.global src must be MemOp")

    typ = instr.types[-1] if instr.types else 'u32'
    is_64 = typ in ('u64', 's64', 'b64', 'f64')

    # WB-7: address-chain fold.  If the MemOp.base is a vreg defined
    # ONLY by `add.u64 %X, %B, IMM`, redirect the base to %B and add
    # IMM to the immediate offset.  The add.u64 is marked dead and
    # will be skipped by _select_add_u64.
    fold_map = getattr(ctx, '_addr_fold_map', {}) if ctx else {}
    extra_offset = 0
    base_name_raw = src.base if src.base.startswith('%') else f'%{src.base}'
    if base_name_raw in fold_map:
        new_base, extra_offset = fold_map[base_name_raw]
        # Rebuild src with new base; keep src.offset (may be 0).
        src = MemOp(base=new_base, offset=src.offset)

    base_name = src.base if src.base.startswith('%') else f'%{src.base}'

    # Resolve address register: if the register was written to GPR (by add.u64 etc.),
    # use the GPR value. Otherwise, if it's only in a UR (raw pointer from ld.param.u64),
    # materialize via IADD.64-UR. For deferred params (4th+ pointer), emit inline LDCU.64.
    result = []
    ur_params = getattr(ctx, '_ur_params', {}) if ctx else {}
    deferred = getattr(ctx, '_deferred_ur_params', {}) if ctx else {}
    gpr_written = getattr(ctx, '_gpr_written', set()) if ctx else set()
    if base_name in gpr_written and src.base in ra.int_regs:
        addr = ra.lo(src.base)
    elif base_name in deferred:
        # Deferred param: emit inline LDCU.64 UR6 → materialize to GPR.
        # Materialize into the ALLOCATED register for the address variable
        # (not a scratch pair), so subsequent add.u64 %rd, %rd, K reads
        # the correct GPR and can increment in-place.
        param_off = deferred.get(base_name)
        ur_tmp = 6
        if src.base in ra.int_regs:
            addr = ra.lo(src.base)
        else:
            addr = getattr(ctx, '_addr_scratch_lo', None)
            if addr is None:
                addr = _alloc_gpr_pair(ctx)
        result.append(SassInstr(encode_ldcu_64(ur_tmp, 0, param_off),
                                f'LDCU.64 UR{ur_tmp}, c[0][0x{param_off:x}]  // deferred param'))
        result.extend(_emit_ur_to_gpr(addr, ur_tmp, "deferred UR->GPR"))
        # Mark address register as written so subsequent add.u64 %rd, %rd, K
        # increments the GPR value instead of re-materializing from the param.
        if ctx:
            ctx._gpr_written.add(base_name)
    elif base_name in ur_params:
        # Register only exists as a UR (raw pointer from ld.param.u64, no add.u64).
        # Use the dedicated addr-scratch pair from context when available.
        # This pair is reserved above the static allocation and reused across
        # all address materializations, preventing register pressure growth.
        ur_idx = ur_params[base_name]
        addr = getattr(ctx, '_addr_scratch_lo', None)
        if addr is None:
            addr = _alloc_gpr_pair(ctx)
        result.extend(_emit_ur_to_gpr(addr, ur_idx, "UR->GPR addr"))
    else:
        addr = RZ

    # BREAK-1B: combine WB-7 fold offset with MemOp inline offset [base+N]
    mem_offset = src.offset if isinstance(src.offset, int) else 0
    total_offset = extra_offset + mem_offset
    off_str = f' + 0x{total_offset:x}' if total_offset else ''
    if is_64:
        d = ra.lo(dest.name)
        result.append(SassInstr(encode_ldg_e_64(d, ur_desc, addr, imm_offset=total_offset),
                          f'LDG.E.64 R{d}, desc[UR{ur_desc}][R{addr}.64{off_str}]'))
    else:
        d = ra.r32(dest.name)
        result.append(SassInstr(encode_ldg_e(d, ur_desc, addr, width=32, imm_offset=total_offset),
                          f'LDG.E R{d}, desc[UR{ur_desc}][R{addr}.64{off_str}]'))
    return result


def _select_atom_cas(instr: Instruction, ra: RegAlloc,
                     ctx: 'ISelContext' = None) -> list[SassInstr]:
    """atom.cas.b32 → ATOMG.E.CAS.b32."""
    from ptx.ir import MemOp
    dest_op = instr.dest
    addr_op = instr.srcs[0]
    cmp_op  = instr.srcs[1]
    new_op  = instr.srcs[2]
    if not isinstance(addr_op, MemOp):
        raise ISelError("atom.cas addr must be MemOp")
    d   = ra.r32(dest_op.name)
    cmp = ra.r32(cmp_op.name)
    nv  = ra.r32(new_op.name)

    # Resolve address: prefer GPR (if written by add.u64) over stale UR entry
    prefix = []
    base_name = addr_op.base if addr_op.base.startswith('%') else f'%{addr_op.base}'
    ur_params = getattr(ctx, '_ur_params', {}) if ctx else {}
    deferred = getattr(ctx, '_deferred_ur_params', {}) if ctx else {}
    gpr_written = getattr(ctx, '_gpr_written', set()) if ctx else set()
    if base_name in gpr_written and addr_op.base in ra.int_regs:
        addr = ra.lo(addr_op.base)
    elif base_name in deferred:
        param_off = deferred.get(base_name)
        ur_tmp = 6
        addr = getattr(ctx, '_addr_scratch_lo', None)
        if addr is None:
            addr = _alloc_gpr_pair(ctx)
        prefix.append(SassInstr(encode_ldcu_64(ur_tmp, 0, param_off),
                                f'LDCU.64 UR{ur_tmp}, c[0][0x{param_off:x}]  // deferred param'))
        prefix.extend(_emit_ur_to_gpr(addr, ur_tmp, "deferred UR->GPR addr"))
    elif base_name in ur_params:
        ur_idx = ur_params[base_name]
        addr = getattr(ctx, '_addr_scratch_lo', None)
        if addr is None:
            addr = _alloc_gpr_pair(ctx)
        prefix.extend(_emit_ur_to_gpr(addr, ur_idx, "UR->GPR addr"))
    else:
        addr = RZ

    return prefix + [SassInstr(encode_atomg_cas_b32(d, addr, cmp, nv),
                      f'ATOMG.E.CAS.b32 R{d}, [R{addr}], R{cmp}, R{nv}')]


def _select_atom_add_u32(instr: Instruction, ra: RegAlloc,
                         ctx: 'ISelContext' = None) -> list[SassInstr]:
    """atom.global.add.u32 / atom.add.u32 → ATOMG.E.ADD.u32.

    Emits ATOMG.E.ADD with PT guard (b1=0x79). Uses descriptor-based
    addressing via UR descriptor. Address resolution mirrors _select_atom_cas:
    UR-only pointers are materialized to GPR via IADD.64 first.
    """
    from ptx.ir import MemOp
    dest_op = instr.dest
    addr_op = instr.srcs[0]
    data_op = instr.srcs[1]
    if not isinstance(addr_op, MemOp):
        raise ISelError("atom.add addr must be MemOp")
    d    = ra.r32(dest_op.name)

    # OC-4 fix: materialize the ADDRESS first, then the DATA.
    # Previously data was allocated a scratch GPR via _alloc_gpr()
    # before the address materialization, but _emit_ur_to_gpr could
    # allocate an overlapping pair (e.g., data=R6 then addr=R6:R7),
    # clobbering the data before the ATOMG reads it.
    prefix = []
    base_name = addr_op.base if addr_op.base.startswith('%') else f'%{addr_op.base}'
    ur_params = getattr(ctx, '_ur_params', {}) if ctx else {}
    deferred = getattr(ctx, '_deferred_ur_params', {}) if ctx else {}
    gpr_written = getattr(ctx, '_gpr_written', set()) if ctx else set()
    if base_name in gpr_written and addr_op.base in ra.int_regs:
        addr = ra.lo(addr_op.base)
    elif base_name in deferred:
        param_off = deferred.get(base_name)
        ur_tmp = 6
        addr = getattr(ctx, '_addr_scratch_lo', None)
        if addr is None:
            addr = _alloc_gpr_pair(ctx)
        prefix.append(SassInstr(encode_ldcu_64(ur_tmp, 0, param_off),
                                f'LDCU.64 UR{ur_tmp}, c[0][0x{param_off:x}]  // deferred param'))
        prefix.extend(_emit_ur_to_gpr(addr, ur_tmp, "deferred UR->GPR addr"))
    elif base_name in ur_params:
        ur_idx = ur_params[base_name]
        addr = getattr(ctx, '_addr_scratch_lo', None)
        if addr is None:
            addr = _alloc_gpr_pair(ctx)
        prefix.extend(_emit_ur_to_gpr(addr, ur_idx, "UR->GPR addr"))
    else:
        addr = RZ

    # Materialize data AFTER address so _alloc_gpr returns a register
    # that doesn't overlap with the address pair.  Explicitly bump
    # _next_gpr past the address pair (which is allocated via the
    # fixed _addr_scratch_lo, outside _alloc_gpr's knowledge).
    if isinstance(data_op, ImmOp):
        if ctx and addr != RZ and hasattr(ctx, '_next_gpr'):
            if ctx._next_gpr <= addr + 1:
                ctx._next_gpr = addr + 2
        data = _alloc_gpr(ctx)
        prefix.append(SassInstr(encode_iadd3_imm32(data, RZ, data_op.value & 0xFFFFFFFF, RZ),
                                f'MOV R{data}, {data_op.value:#x}  // atom data imm'))
    else:
        data = ra.r32(data_op.name)

    ur_d = ctx.ur_desc if ctx else 4
    return prefix + [SassInstr(encode_atomg_u32(d, addr, 0, data, ATOMG_ADD, ur_desc=ur_d),
                     f'ATOMG.E.ADD.u32 R{d}, desc[UR{ur_d}][R{addr}.64], R{data}')]


def _select_atom_generic_u32(instr: Instruction, ra: RegAlloc,
                              ctx: 'ISelContext', atom_op: int,
                              op_name: str) -> list[SassInstr]:
    """atom.global.{exch|min|max|and|or}.{b32|s32|u32} → ATOMG.E.{op}."""
    from ptx.ir import MemOp
    dest_op = instr.dest
    addr_op = instr.srcs[0]
    data_op = instr.srcs[1]
    if not isinstance(addr_op, MemOp):
        raise ISelError(f"atom.{op_name} addr must be MemOp")
    d    = ra.r32(dest_op.name)
    data = ra.r32(data_op.name)

    prefix = []
    base_name = addr_op.base if addr_op.base.startswith('%') else f'%{addr_op.base}'
    ur_params = getattr(ctx, '_ur_params', {}) if ctx else {}
    deferred = getattr(ctx, '_deferred_ur_params', {}) if ctx else {}
    gpr_written = getattr(ctx, '_gpr_written', set()) if ctx else set()
    if base_name in gpr_written and addr_op.base in ra.int_regs:
        addr = ra.lo(addr_op.base)
    elif base_name in deferred:
        param_off = deferred.get(base_name)
        ur_tmp = 6
        addr = getattr(ctx, '_addr_scratch_lo', None)
        if addr is None:
            addr = _alloc_gpr_pair(ctx)
        prefix.append(SassInstr(encode_ldcu_64(ur_tmp, 0, param_off),
                                f'LDCU.64 UR{ur_tmp}, c[0][0x{param_off:x}]  // deferred param'))
        prefix.extend(_emit_ur_to_gpr(addr, ur_tmp, "deferred UR->GPR addr"))
    elif base_name in ur_params:
        ur_idx = ur_params[base_name]
        addr = getattr(ctx, '_addr_scratch_lo', None)
        if addr is None:
            addr = _alloc_gpr_pair(ctx)
        prefix.extend(_emit_ur_to_gpr(addr, ur_idx, "UR->GPR addr"))
    else:
        addr = RZ

    ur_d = ctx.ur_desc if ctx else 4
    return prefix + [SassInstr(encode_atomg_u32(d, addr, 0, data, atom_op, ur_desc=ur_d),
                     f'ATOMG.E.{op_name} R{d}, desc[UR{ur_d}][R{addr}.64], R{data}')]


def _select_atom_add_f32(instr: Instruction, ra: RegAlloc,
                          ctx: 'ISelContext' = None) -> list[SassInstr]:
    """atom.global.add.f32 → ATOMG.E.ADD.F32."""
    from ptx.ir import MemOp
    dest_op = instr.dest
    addr_op = instr.srcs[0]
    data_op = instr.srcs[1]
    if not isinstance(addr_op, MemOp):
        raise ISelError("atom.add.f32 addr must be MemOp")
    d    = ra.r32(dest_op.name)
    data = ra.r32(data_op.name)

    prefix = []
    base_name = addr_op.base if addr_op.base.startswith('%') else f'%{addr_op.base}'
    ur_params = getattr(ctx, '_ur_params', {}) if ctx else {}
    deferred = getattr(ctx, '_deferred_ur_params', {}) if ctx else {}
    gpr_written = getattr(ctx, '_gpr_written', set()) if ctx else set()
    if base_name in gpr_written and addr_op.base in ra.int_regs:
        addr = ra.lo(addr_op.base)
    elif base_name in deferred:
        param_off = deferred.get(base_name)
        ur_tmp = 6
        addr = getattr(ctx, '_addr_scratch_lo', None)
        if addr is None:
            addr = _alloc_gpr_pair(ctx)
        prefix.append(SassInstr(encode_ldcu_64(ur_tmp, 0, param_off),
                                f'LDCU.64 UR{ur_tmp}, c[0][0x{param_off:x}]  // deferred param'))
        prefix.extend(_emit_ur_to_gpr(addr, ur_tmp, "deferred UR->GPR addr"))
    elif base_name in ur_params:
        ur_idx = ur_params[base_name]
        addr = getattr(ctx, '_addr_scratch_lo', None)
        if addr is None:
            addr = _alloc_gpr_pair(ctx)
        prefix.extend(_emit_ur_to_gpr(addr, ur_idx, "UR->GPR addr"))
    else:
        addr = RZ

    ur_d = ctx.ur_desc if ctx else 4
    return prefix + [SassInstr(encode_atomg_add_f32(d, addr, 0, data, ur_desc=ur_d),
                     f'ATOMG.E.ADD.F32 R{d}, desc[UR{ur_d}][R{addr}.64], R{data}')]


def _select_atom_cas_b64(instr: Instruction, ra: RegAlloc,
                          ctx: 'ISelContext' = None) -> list[SassInstr]:
    """atom.cas.b64 → ATOMG.E.CAS.64.

    All three operands (addr, compare, new_val) may be in UR space (loaded via
    LDCU.64 for kernel parameters). We need to materialize each into GPR pairs
    via IADD.64 if they're still in UR space.
    """
    from ptx.ir import MemOp
    dest_op = instr.dest
    addr_op = instr.srcs[0]
    cmp_op  = instr.srcs[1]
    new_op  = instr.srcs[2]
    if not isinstance(addr_op, MemOp):
        raise ISelError("atom.cas.b64 addr must be MemOp")

    ur_params = getattr(ctx, '_ur_params', {}) if ctx else {}
    deferred = getattr(ctx, '_deferred_ur_params', {}) if ctx else {}
    gpr_written = getattr(ctx, '_gpr_written', set()) if ctx else set()
    prefix = []

    def _materialize_u64(op, label):
        """Ensure a u64 operand is in GPR pair, materializing from UR if needed."""
        name = op.name
        base_name = name if name.startswith('%') else f'%{name}'
        if base_name in gpr_written and name in ra.int_regs:
            return ra.lo(name)
        elif base_name in deferred:
            param_off = deferred.get(base_name)
            ur_tmp = 6
            gpr = _alloc_gpr_pair(ctx)
            prefix.append(SassInstr(encode_ldcu_64(ur_tmp, 0, param_off),
                                    f'LDCU.64 UR{ur_tmp}, c[0][0x{param_off:x}]  // deferred param'))
            prefix.extend(_emit_ur_to_gpr(gpr, ur_tmp, f"deferred UR->GPR {label}"))
            return gpr
        elif base_name in ur_params:
            ur_idx = ur_params[base_name]
            gpr = _alloc_gpr_pair(ctx)
            prefix.extend(_emit_ur_to_gpr(gpr, ur_idx, f"UR->GPR {label}"))
            return gpr
        else:
            return ra.lo(name)

    # Materialize addr from MemOp
    addr_base = addr_op.base if addr_op.base.startswith('%') else f'%{addr_op.base}'
    if addr_base in gpr_written and addr_op.base in ra.int_regs:
        addr = ra.lo(addr_op.base)
    elif addr_base in deferred:
        param_off = deferred.get(addr_base)
        ur_tmp = 6
        addr = _alloc_gpr_pair(ctx)
        prefix.append(SassInstr(encode_ldcu_64(ur_tmp, 0, param_off),
                                f'LDCU.64 UR{ur_tmp}, c[0][0x{param_off:x}]  // deferred param'))
        prefix.extend(_emit_ur_to_gpr(addr, ur_tmp, "deferred UR->GPR addr"))
    elif addr_base in ur_params:
        ur_idx = ur_params[addr_base]
        addr = _alloc_gpr_pair(ctx)
        prefix.extend(_emit_ur_to_gpr(addr, ur_idx, "UR->GPR addr"))
    else:
        addr = RZ

    cmp = _materialize_u64(cmp_op, 'cmp')
    nv  = _materialize_u64(new_op, 'new')
    d   = ra.lo(dest_op.name)

    return prefix + [SassInstr(encode_atomg_cas_b64(d, addr, cmp, nv),
                      f'ATOMG.E.CAS.64 R{d}, [R{addr}], R{cmp}, R{nv}')]


def _select_dp4a(instr: Instruction, ra: RegAlloc,
                  ctx: 'ISelContext' = None) -> list[SassInstr]:
    """dp4a.u32.u32 → IDP.4A.U8.U8."""
    d = ra.r32(instr.dest.name)
    a = ra.r32(instr.srcs[0].name)
    b = ra.r32(instr.srcs[1].name)
    c = ra.r32(instr.srcs[2].name)
    return [SassInstr(encode_idp4a(d, a, b, c),
                      f'IDP.4A.U8.U8 R{d}, R{a}, R{b}, R{c}')]


def _select_st_global(instr: Instruction, ra: RegAlloc,
                      ur_desc: int, ctx: 'ISelContext' = None) -> list[SassInstr]:
    """st.global → STG.E with appropriate width."""
    dest_op = instr.srcs[0]  # address
    src_op  = instr.srcs[1]  # data
    from ptx.ir import MemOp
    if not isinstance(dest_op, MemOp):
        raise ISelError(f"st.global addr must be MemOp")
    if not isinstance(src_op, RegOp):
        # Immediate data: materialize into a temporary register first.
        if isinstance(src_op, ImmOp):
            t = _alloc_gpr(ctx)
            lit_off = ctx._alloc_literal(src_op.value & 0xFFFFFFFF)
            from sass.encoding.sm_120_opcodes import encode_ldc
            preamble = [SassInstr(encode_ldc(t, 4, lit_off),
                                  f'LDC R{t}, c[0x4][{lit_off:#x}]  // materialize imm for st')]
        else:
            raise ISelError(f"st.global data must be register or immediate")

    typ = instr.types[-1] if instr.types else 'u32'
    is_64 = typ in ('u64', 's64', 'b64', 'f64')

    # WB-7: address-chain fold (same logic as ld.global).
    fold_map = getattr(ctx, '_addr_fold_map', {}) if ctx else {}
    extra_offset = 0
    base_name_raw = dest_op.base if dest_op.base.startswith('%') else f'%{dest_op.base}'
    if base_name_raw in fold_map:
        new_base, extra_offset = fold_map[base_name_raw]
        dest_op = MemOp(base=new_base, offset=dest_op.offset)

    base_name = dest_op.base if dest_op.base.startswith('%') else f'%{dest_op.base}'

    # Resolve address: prefer GPR (if written by add.u64) over stale UR entry
    prefix = []
    ur_params = getattr(ctx, '_ur_params', {}) if ctx else {}
    deferred = getattr(ctx, '_deferred_ur_params', {}) if ctx else {}
    gpr_written = getattr(ctx, '_gpr_written', set()) if ctx else set()
    if base_name in gpr_written and dest_op.base in ra.int_regs:
        addr = ra.lo(dest_op.base)
    elif base_name in deferred:
        param_off = deferred.get(base_name)
        ur_tmp = 6
        addr = getattr(ctx, '_addr_scratch_lo', None)
        if addr is None:
            addr = _alloc_gpr_pair(ctx)
        prefix.append(SassInstr(encode_ldcu_64(ur_tmp, 0, param_off),
                                f'LDCU.64 UR{ur_tmp}, c[0][0x{param_off:x}]  // deferred param'))
        prefix.extend(_emit_ur_to_gpr(addr, ur_tmp, "deferred UR->GPR"))
    elif base_name in ur_params:
        ur_idx = ur_params[base_name]
        addr = getattr(ctx, '_addr_scratch_lo', None)
        if addr is None:
            addr = _alloc_gpr_pair(ctx)
        prefix.extend(_emit_ur_to_gpr(addr, ur_idx, "UR->GPR addr"))
    else:
        addr = RZ

    off_str = f' + 0x{extra_offset:x}' if extra_offset else ''

    # Handle materialized immediate
    if not isinstance(src_op, RegOp):
        data = t  # from materialized temp above
        result = prefix + preamble + [SassInstr(encode_stg_e(ur_desc, addr, data, width=32, imm_offset=extra_offset, ctrl=0xff1),
                                       f'STG.E desc[UR{ur_desc}][R{addr}.64{off_str}], R{data}')]
        return result

    if is_64:
        data = ra.lo(src_op.name)
        return prefix + [SassInstr(encode_stg_e_64(ur_desc, addr, data, imm_offset=extra_offset, ctrl=0xff1),
                          f'STG.E.64 desc[UR{ur_desc}][R{addr}.64{off_str}], R{data}')]
    else:
        data = ra.r32(src_op.name)
        return prefix + [SassInstr(encode_stg_e(ur_desc, addr, data, width=32, imm_offset=extra_offset, ctrl=0xff1),
                          f'STG.E desc[UR{ur_desc}][R{addr}.64{off_str}], R{data}')]


# ---------------------------------------------------------------------------
# Main instruction selector entry point
# ---------------------------------------------------------------------------

@dataclass
class ISelContext:
    """Context passed through the instruction selector."""
    ra:            RegAlloc
    # Byte offset of each kernel parameter in c[0][...] (ABI layout)
    param_offsets: dict[str, int] = field(default_factory=dict)
    # Uniform register to use for global memory descriptor
    ur_desc:       int = 4  # UR4 by default (matches ptxas convention)
    # Label → instruction index within output for branch fixup
    label_map:     dict[str, int] = field(default_factory=dict)
    # Next available uniform register for LDCU param loading (UR6+)
    _next_ur:      int = 6  # UR4 = mem desc, UR6+ for params
    _ur_free:      list = field(default_factory=list)  # freed UR pairs for reuse
    # Deferred u64 params: name → cbuf byte offset. Loaded inline via LDCU.64 UR6
    # at point of use (add.u64 / ld.global / atom), keeping max live URs to UR6-UR7.
    _deferred_ur_params: dict[str, int] = field(default_factory=dict)
    # Map PTX register name → UR index (for params loaded via LDCU)
    _ur_params:    dict[str, int] = field(default_factory=dict)
    # Map PTX register name → param byte offset (for setp LDCU fallback)
    _reg_param_off: dict[str, int] = field(default_factory=dict)
    # Map PTX register name → SR code (for S2UR in mad.lo)
    _reg_sr_source: dict[str, int] = field(default_factory=dict)
    # Map PTX register name → UR index (u32 params loaded via LDCU.32 for ISETP R-UR)
    _ur_for_param:  dict[str, int] = field(default_factory=dict)
    # Set of PTX register names that have been written to GPR (overriding any UR value)
    _gpr_written:   set = field(default_factory=set)
    # Literal constant pool: value → c[0] byte offset (baked into .nv.constant0)
    # Base offset is set by the pipeline after regalloc (after the param area ends).
    _const_pool_base: int = 0
    _const_pool:      dict[int, int] = field(default_factory=dict)
    # Next available scratch GPR (for isel-internal temporaries, e.g. bfe mask)
    # Initialized from alloc.num_gprs by the pipeline; may grow during isel.
    # SM_120 HARDWARE LIMIT: Without proper merc 0x5a metadata, the GPU only
    # allows access to R0..R(capmerc_byte8 - 1). Default capmerc allocates
    # based on num_gprs. To avoid ERR715, we cap scratch allocation and
    # reuse temporaries via a free-list.
    _next_gpr: int = 0
    _scratch_pool: list = field(default_factory=list)  # free scratch GPRs
    _scratch_highwater: int = 0  # max _next_gpr reached (for capmerc)
    _scratch_mark: int = -1  # saved _next_gpr for batch free
    # Next available scratch predicate register (for isel-internal use, e.g. div.u32)
    # Initialized from alloc.num_pred by the pipeline; may grow during isel.
    _next_pred: int = 0
    # Target SM version (89 = Ada Lovelace / RTX 4090, 120 = Blackwell / RTX 5090)
    sm_version: int = 120
    # Whether the kernel contains VOTE instructions (for SM_120 rule #25)
    _has_vote: bool = False

    def _alloc_literal(self, value: int) -> int:
        """Return the c[0] byte offset for a 32-bit literal constant.

        Allocates a new slot in the literal pool if the value has not been
        seen before.  Slots are 4 bytes each.
        """
        value = value & 0xFFFFFFFF  # normalise to u32 bit pattern
        if value not in self._const_pool:
            offset = self._const_pool_base + len(self._const_pool) * 4
            self._const_pool[value] = offset
        return self._const_pool[value]


# ---------------------------------------------------------------------------
# Texture/surface instruction selectors
# ---------------------------------------------------------------------------

def _select_tex(instr: 'Instruction', ctx: 'ISelContext') -> list[SassInstr]:
    """Select TEX or TLD.LZ for PTX tex.* instructions.

    PTX syntax: tex.{1d|2d|3d}.v4.{f32|u32|s32}.{s32|f32} {d0,d1,d2,d3}, [tex_desc, {coords}]
    For 1D integer coords → TLD.LZ (level-zero fetch)
    For 2D/3D float coords → TEX
    """
    from ptx.ir import MemOp
    result = []

    # Determine dimension from types
    dim_str = '1d'
    for t in instr.types:
        if t in ('1d', '2d', '3d'):
            dim_str = t
            break

    # Dest is the first register in the vector (parser extracts first from {})
    d = _get_reg(instr.dest, ctx.ra) if instr.dest else _alloc_gpr(ctx)

    # Source: texture descriptor (UR) and coordinate register
    # The parser gives us srcs[0] as MemOp (the [tex_desc, {coord}] part)
    # Since the parser consumed the coordinate vector inside the brackets,
    # the MemOp base is the texture descriptor register.
    # For bindless textures, the descriptor is a u64 in a UR pair.
    # We need the UR register allocated for the texture param.

    # Get the texture descriptor UR from the source memory operand
    coord = d  # Default: coord collocated with dest (ptxas pattern)
    ur_desc = 4  # Default UR4 (standard texture descriptor slot)

    if instr.srcs:
        src = instr.srcs[0]
        if isinstance(src, MemOp):
            # base is the texture descriptor register name
            name = src.base
            if name in ctx._ur_params:
                ur_desc = ctx._ur_params[name]
            elif name in ctx.ra.int_regs:
                # Texture descriptor loaded into a GPR — need to copy to UR
                ur_desc = 4  # Use UR4 as default slot
        elif isinstance(src, RegOp):
            name = src.name
            if name in ctx._ur_params:
                ur_desc = ctx._ur_params[name]

    # For 1D with integer coords → TLD.LZ
    if dim_str == '1d':
        mask = 0x0f  # Default RGBA
        # Check if we only use 1 component (optimization)
        result.append(SassInstr(
            encode_tld_lz(d, d, ur_desc, mask=mask),
            f'TLD.LZ R{d}, R{d}, UR{ur_desc}, 1D  // tex.1d'))
    elif dim_str == '2d':
        mask = 0x0f
        result.append(SassInstr(
            encode_tex(d, d, ur_desc, TEX_DIM_2D, mask=mask),
            f'TEX R{d}, R{d}, UR{ur_desc}, 2D  // tex.2d'))
    elif dim_str == '3d':
        mask = 0x0f
        result.append(SassInstr(
            encode_tex(d, d, ur_desc, TEX_DIM_3D, mask=mask),
            f'TEX R{d}, R{d}, UR{ur_desc}, 3D  // tex.3d'))

    return result


def _select_tld4(instr: 'Instruction', ctx: 'ISelContext') -> list[SassInstr]:
    """Select TLD4.R for PTX tld4.* instructions.

    PTX syntax: tld4.{r|g|b|a}.2d.v4.f32.f32 {d0,d1,d2,d3}, [tex_desc, {cx, cy}]
    """
    from ptx.ir import MemOp
    result = []

    d = _get_reg(instr.dest, ctx.ra) if instr.dest else _alloc_gpr(ctx)
    ur_desc = 4

    if instr.srcs:
        src = instr.srcs[0]
        if isinstance(src, MemOp):
            name = src.base
            if name in ctx._ur_params:
                ur_desc = ctx._ur_params[name]

    # TLD4 always returns 4 values; dest_hi = dest+2
    dest_hi = (d + 2) & 0xFF
    result.append(SassInstr(
        encode_tld4(d, d, ur_desc, dest_hi=dest_hi),
        f'TLD4.R R{d}, R{d}, UR{ur_desc}, 2D  // tld4'))

    return result


def _select_txq(instr: 'Instruction', ctx: 'ISelContext') -> list[SassInstr]:
    """Select TXQ for PTX txq.* instructions.

    PTX syntax: txq.{width|height|depth}.b32 %r, [tex_desc]
    """
    from ptx.ir import MemOp
    result = []

    d = _get_reg(instr.dest, ctx.ra) if instr.dest else _alloc_gpr(ctx)
    ur_desc = 4

    # Determine query type from modifiers
    query = TXQ_WIDTH  # default
    for t in instr.types:
        if t == 'width':
            query = TXQ_WIDTH
        elif t == 'height':
            query = TXQ_HEIGHT
        elif t == 'depth':
            query = TXQ_DEPTH

    if instr.srcs:
        src = instr.srcs[0]
        if isinstance(src, MemOp):
            name = src.base
            if name in ctx._ur_params:
                ur_desc = ctx._ur_params[name]

    query_name = {TXQ_WIDTH: 'width', TXQ_HEIGHT: 'height', TXQ_DEPTH: 'depth'}[query]
    result.append(SassInstr(
        encode_txq(d, ur_desc, query),
        f'TXQ R{d}, UR{ur_desc}, {query_name}  // txq'))

    return result


def _select_suld(instr: 'Instruction', ctx: 'ISelContext') -> list[SassInstr]:
    """Select SULD for PTX suld.* instructions.

    PTX syntax: suld.b.{1d|2d}.{b32|v2.b32}.trap {d}, [surf_desc, {coord}]
    """
    from ptx.ir import MemOp
    result = []

    d = _get_reg(instr.dest, ctx.ra) if instr.dest else _alloc_gpr(ctx)
    ur_desc = 4

    # Dimension
    dim = SURF_DIM_1D
    for t in instr.types:
        if t == '2d':
            dim = SURF_DIM_2D

    # Data width
    mode = SURF_MODE_B32
    if 'v2' in instr.types:
        mode = SURF_MODE_B64

    if instr.srcs:
        src = instr.srcs[0]
        if isinstance(src, MemOp):
            name = src.base
            if name in ctx._ur_params:
                ur_desc = ctx._ur_params[name]

    dim_name = '1D' if dim == SURF_DIM_1D else '2D'
    mode_name = 'b32' if mode == SURF_MODE_B32 else 'b64'
    result.append(SassInstr(
        encode_suld(d, d, ur_desc, dim, mode),
        f'SULD R{d}, [R{d}], UR{ur_desc}, {dim_name}, {mode_name}  // suld'))

    return result


def _select_sust(instr: 'Instruction', ctx: 'ISelContext') -> list[SassInstr]:
    """Select SUST for PTX sust.* instructions.

    PTX syntax: sust.b.{1d|2d}.{b32|v2.b32}.trap [surf_desc, {coord}], {data}
    """
    from ptx.ir import MemOp
    result = []

    ur_desc = 4

    # Dimension
    dim = SURF_DIM_1D
    for t in instr.types:
        if t == '2d':
            dim = SURF_DIM_2D

    # Data width
    mode = SURF_MODE_B32
    if 'v2' in instr.types:
        mode = SURF_MODE_B64

    # srcs[0] = MemOp (surface descriptor + coord)
    # srcs[1] = RegOp (data register)
    coord = 0
    data = 0

    if len(instr.srcs) >= 1:
        src = instr.srcs[0]
        if isinstance(src, MemOp):
            name = src.base
            if name in ctx._ur_params:
                ur_desc = ctx._ur_params[name]
            elif name in ctx.ra.int_regs:
                coord = ctx.ra.int_regs[name]

    if len(instr.srcs) >= 2:
        data = _get_reg(instr.srcs[1], ctx.ra)

    dim_name = '1D' if dim == SURF_DIM_1D else '2D'
    mode_name = 'b32' if mode == SURF_MODE_B32 else 'b64'
    result.append(SassInstr(
        encode_sust(data, coord, ur_desc, dim, mode),
        f'SUST [R{coord}], R{data}, UR{ur_desc}, {dim_name}, {mode_name}  // sust'))

    return result


def select_function(fn: Function, ctx: ISelContext) -> list[SassInstr]:
    """
    Select SASS instructions for every PTX instruction in a function.

    Returns a flat list of SassInstr.  Branch targets are not yet resolved
    (encode_bra is called with offset=0 as a placeholder); a second pass
    over the output would patch BRA offsets using label_map.
    """
    output: list[SassInstr] = []

    # Reorder blocks: move ret-only blocks to the end so they don't disrupt
    # BRA target offsets between jump sites and their targets.
    _ret_only = set()
    for bb in fn.blocks:
        if (bb.label and len(bb.instructions) == 1
                and bb.instructions[0].op == 'ret'):
            _ret_only.add(bb.label)
    ordered_blocks = [bb for bb in fn.blocks if bb.label not in _ret_only]
    ordered_blocks += [bb for bb in fn.blocks if bb.label in _ret_only]

    for bb in ordered_blocks:
        # Record label position and mark the first instruction with label tag
        block_start_idx = len(output)
        _label_tag = None
        if bb.label:
            ctx.label_map[bb.label] = len(output) * 16
            _label_tag = bb.label  # tag first emitted instruction for BRA fixup

        for _instr_idx, instr in enumerate(bb.instructions):
            if hasattr(ctx, '_skip_instrs') and id(instr) in ctx._skip_instrs:
                continue
            # Mark scratch watermark before each instruction so temporaries
            # (div/rem/mul.hi scratch regs) are reclaimed after emission.
            _mark_scratch(ctx)
            op = instr.op.lower()
            # typ = last type qualifier (the data type). Earlier elements are modifiers (lo, hi, approx, etc.)
            typ = instr.types[-1].lower() if instr.types else ''

            # Track output length before this instruction so we can apply
            # predicates to all newly-generated SASS after the handler runs.
            _pre_len = len(output)
            # Snapshot _negated_preds BEFORE processing: a predicated setp that
            # writes to the same predicate as its guard must use the OUTER guard
            # sense (from before the setp), not the NEW sense (after inversion).
            _neg_preds_snapshot = set(ctx._negated_preds) if hasattr(ctx, '_negated_preds') else set()

            try:
                # Vector unpack: mov.b64 {%rLo, %rHi}, %rdSrc
                # Alias the 32-bit destinations to the 64-bit source pair.
                # This avoids materializing MOVs — subsequent consumers of
                # %rLo/%rHi will directly use the accumulator registers.
                from ptx.ir import VectorRegOp
                if (op == 'mov' and typ in ('b64', 'u64', 's64')
                        and isinstance(instr.dest, VectorRegOp)
                        and len(instr.dest.regs) == 2
                        and isinstance(instr.srcs[0], RegOp)):
                    src_name = instr.srcs[0].name
                    s_lo = ctx.ra.lo(src_name)
                    s_hi = s_lo + 1
                    # Alias: make %rLo and %rHi point to same physical regs
                    lo_name = instr.dest.regs[0]
                    hi_name = instr.dest.regs[1]
                    ctx.ra.int_regs[lo_name] = s_lo
                    ctx.ra.int_regs[hi_name] = s_hi
                    # No instructions needed — pure alias
                    continue

                # Vector pack: mov.b64 %rdDest, {%rLo, %rHi}
                # Always alias — track non-consecutive hi separately.
                if (op == 'mov' and typ in ('b64', 'u64', 's64')
                        and isinstance(instr.srcs[0], VectorRegOp)
                        and len(instr.srcs[0].regs) == 2
                        and isinstance(instr.dest, RegOp)):
                    dest_name = instr.dest.name
                    s_lo = ctx.ra.r32(instr.srcs[0].regs[0])
                    s_hi = ctx.ra.r32(instr.srcs[0].regs[1])
                    ctx.ra.int_regs[dest_name] = s_lo
                    if s_hi != s_lo + 1:
                        # Non-consecutive: record actual hi register
                        if not hasattr(ctx, '_pair_hi_override'):
                            ctx._pair_hi_override = {}
                        ctx._pair_hi_override[dest_name] = s_hi
                    elif hasattr(ctx, '_pair_hi_override') and dest_name in ctx._pair_hi_override:
                        del ctx._pair_hi_override[dest_name]
                    continue

                if op == 'mov' and typ in ('u32', 's32', 'b32', 'f32', 'u64', 's64', 'b64', 'f64'):
                    # WB-2: skip mov if it's a zero-init that an HMMA RZ
                    # substitution made dead.  See analyze_mma_zero_subst.
                    if id(instr) in getattr(ctx, '_hmma_dead_movs', set()):
                        continue
                    # Immediate source: load via IADD3_IMM32 (integer) or FMUL_IMM (float)
                    if isinstance(instr.srcs[0], ImmOp) and typ in ('u32', 's32', 'b32', 'f32'):
                        d = ctx.ra.r32(instr.dest.name)
                        imm = instr.srcs[0].value & 0xFFFFFFFF
                        if imm == 0:
                            if not hasattr(ctx, '_zero_regs'):
                                ctx._zero_regs = set()
                            ctx._zero_regs.add(d)
                        # Track known-immediate registers for FFMA.IMM fusion
                        if not hasattr(ctx, '_imm_regs'):
                            ctx._imm_regs = {}
                        ctx._imm_regs[d] = imm
                        # Use IADD3_IMM32 to load immediate directly (works for any 32-bit pattern)
                        output.append(SassInstr(encode_iadd3_imm32(d, RZ, imm, RZ),
                                                f'IADD3 R{d}, RZ, 0x{imm:x}, RZ  // mov.{typ} imm'))
                        continue
                    # Track special register sources
                    if (isinstance(instr.srcs[0], RegOp) and
                        instr.srcs[0].name in _SPECIAL_REGS and
                        isinstance(instr.dest, RegOp)):
                        ctx._reg_sr_source[instr.dest.name] = _SPECIAL_REGS[instr.srcs[0].name]
                        # ntid.x loaded from constant bank — track as param-like source
                        if instr.srcs[0].name == '%ntid.x':
                            ctx._reg_param_off[instr.dest.name] = (
                                _CBANK_NTID_X_SM89 if ctx.sm_version == 89 else _CBANK_NTID_X)
                        elif instr.srcs[0].name in ('%ctaid.x', '%ctaid.y', '%ctaid.z'):
                            if ctx.sm_version == 89:
                                # SM_89: use S2R directly into GPR (IMAD.WIDE R-R handles mul)
                                pass  # fall through to _select_mov → S2R
                            else:
                                # SM_120: Put ctaid into a fresh UR via S2UR so mad.lo can use IMAD R-UR.
                                # IMAD R-R (0x224) is not validated on SM_120; IMAD R-UR (0xc24)
                                # is confirmed by ptxas. Must NOT use UR4 (reserved for mem
                                # descriptor by pipeline.py) to avoid a WAR hazard where the
                                # descriptor LDCU overwrites UR4 before IMAD finishes reading it.
                                sr_code = _SPECIAL_REGS[instr.srcs[0].name]
                                sr_label = instr.srcs[0].name.lstrip('%').replace('.', '_').upper()
                                # SM_120 rule: S2UR for ctaid uses UR6 when UR
                                # space is exhausted. UR4 holds the mem descriptor
                                # and must not be clobbered. UR6 is safe: it's
                                # consumed by IMAD immediately and not used again.
                                _has_deferred = getattr(ctx, '_had_deferred_params', False)
                                if ctx._next_ur >= 14 or _has_deferred:
                                    ur_ctaid = 6  # UR6 is safe (UR4 = mem desc)
                                else:
                                    ur_ctaid = ctx._next_ur; ctx._next_ur += 1
                                ctx._ur_for_param[instr.dest.name] = ur_ctaid
                                output.append(SassInstr(encode_s2ur(ur_ctaid, sr_code),
                                                        f'S2UR UR{ur_ctaid}, SR_{sr_label}  // {instr.dest.name} = {instr.srcs[0].name.lstrip("%")}'))
                                # S2UR latency: insert NOPs to ensure UR is ready
                                # before IMAD reads it. Needed when UR4 is reused
                                # (fewer pipeline slots than normal allocation).
                                pass  # no S2UR NOPs
                                continue
                    # HARD-FINISH-1: mov+@pred add → @pred IMAD coalescing.
                    # Pattern:
                    #   mul.lo.u32 %tmp, %a, K        (already emitted)
                    #   mov.u32 %dest, %val            (this instruction)
                    #   @%pred add.u32 %dest, %val, %tmp  (next instruction)
                    # Fused into: @pred IMAD R_dest, R_a, K, R_dest
                    # where R_dest = R_val (coalesced).  Saves MOV + separate mul.
                    _did_pred_fuse = False
                    if (typ in ('u32', 's32') and isinstance(instr.dest, RegOp)
                            and isinstance(instr.srcs[0], RegOp)
                            and not instr.pred
                            and _instr_idx + 1 < len(bb.instructions)):
                        _nxt = bb.instructions[_instr_idx + 1]
                        if (_nxt.op == 'add' and _nxt.pred
                                and _nxt.types and _nxt.types[-1] in ('u32', 's32')
                                and isinstance(_nxt.dest, RegOp)
                                and _nxt.dest.name == instr.dest.name
                                and isinstance(_nxt.srcs[0], RegOp)
                                and isinstance(_nxt.srcs[1], RegOp)):
                            val_name = instr.srcs[0].name
                            dest_name = instr.dest.name
                            # Identify which add source is val and which is tmp
                            if _nxt.srcs[0].name == val_name:
                                tmp_name = _nxt.srcs[1].name
                            elif _nxt.srcs[1].name == val_name:
                                tmp_name = _nxt.srcs[0].name
                            else:
                                tmp_name = None
                            if tmp_name is not None:
                                # Look backward for mul.lo.u32 producing tmp with imm K
                                _mul_i = None
                                for _la in range(max(0, _instr_idx - 4), _instr_idx):
                                    _c = bb.instructions[_la]
                                    if (_c.op == 'mul' and 'lo' in _c.types
                                            and _c.types[-1] in ('u32', 's32')
                                            and isinstance(_c.dest, RegOp)
                                            and _c.dest.name == tmp_name
                                            and isinstance(_c.srcs[1], ImmOp)
                                            and isinstance(_c.srcs[0], RegOp)):
                                        _mul_i = _c
                                        break
                                if (_mul_i is not None
                                        and (_mul_i.srcs[1].value & 0xFFFFFFFF) <= 0xFFFF):
                                    from sass.encoding.sm_120_opcodes import encode_imad_r_imm
                                    mul_k = _mul_i.srcs[1].value & 0xFFFFFFFF
                                    mul_a = ctx.ra.r32(_mul_i.srcs[0].name)
                                    val_r = ctx.ra.r32(val_name)
                                    # Coalesce dest with val
                                    ctx.ra.int_regs[dest_name] = val_r
                                    # Remove the mul's SASS from output (last IMAD for this tmp)
                                    tmp_r = ctx.ra.r32(tmp_name)
                                    for _ri in range(len(output) - 1, max(len(output) - 6, -1), -1):
                                        if _ri >= 0 and f'R{tmp_r},' in output[_ri].comment and 'mul.lo' in output[_ri].comment:
                                            output.pop(_ri)
                                            break
                                    # Emit @pred IMAD (predication applied manually since
                                    # the fusion is emitted at the mov's slot, not the add's)
                                    pd = ctx.ra.pred(_nxt.pred) if _nxt.pred in ctx.ra.pred_regs else 0
                                    neg = _nxt.neg
                                    if hasattr(ctx, '_negated_preds') and pd in ctx._negated_preds:
                                        neg = not neg
                                    imad_raw = patch_pred(
                                        encode_imad_r_imm(val_r, mul_a, mul_k, val_r),
                                        pred=pd, neg=neg)
                                    pred_str = f'@{"!" if neg else ""}P{pd} '
                                    output.append(SassInstr(imad_raw,
                                        f'{pred_str}IMAD R{val_r}, R{mul_a}, 0x{mul_k:x}, R{val_r}  // fused mov+@pred mul+add'))
                                    # Skip the @pred add
                                    if not hasattr(ctx, '_skip_instrs'):
                                        ctx._skip_instrs = set()
                                    ctx._skip_instrs.add(id(_nxt))
                                    _did_pred_fuse = True
                    if not _did_pred_fuse:
                        output.extend(_select_mov(instr, ctx.ra, ctx))

                elif op == 'shl' and typ in ('b64', 'u64', 's64'):
                    output.extend(_select_shl_b64(instr, ctx.ra, ctx, output))

                elif op == 'shl' and typ in ('b32', 'u32', 's32'):
                    # 32-bit shift left: IMAD.SHL or SHF.L.U32 for constants,
                    # SHF.L.U32.VAR (opcode 0x7299) for runtime register shifts.
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    if isinstance(instr.srcs[1], ImmOp):
                        k = instr.srcs[1].value
                        if k <= 15:
                            output.append(SassInstr(encode_imad_shl_u32(d, a, k),
                                                    f'IMAD.SHL.U32 R{d}, R{a}, {1<<k:#x}, RZ  // shl.{typ} {k}'))
                        else:
                            output.append(SassInstr(encode_shf_l_u32(d, a, k, RZ),
                                                    f'SHF.L.U32 R{d}, R{a}, 0x{k:x}, RZ  // shl.{typ} {k}'))
                    else:
                        k_reg = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                        output.append(SassInstr(encode_shf_l_u32_var(d, a, k_reg),
                                                f'SHF.L.U32 R{d}, R{a}, R{k_reg}, RZ  // shl.{typ} (var)'))

                elif op == 'shr' and typ in ('b32', 'u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    is_signed = (typ == 's32')
                    if isinstance(instr.srcs[1], ImmOp):
                        k = instr.srcs[1].value
                        if is_signed:
                            output.append(SassInstr(encode_shf_r_s32_hi(d, a, k),
                                                    f'SHF.R.S32.HI R{d}, RZ, 0x{k:x}, R{a}  // shr.s32 {k}'))
                        else:
                            output.append(SassInstr(encode_shf_r_u32_hi(d, a, k),
                                                    f'SHF.R.U32.HI R{d}, RZ, 0x{k:x}, R{a}  // shr.{typ} {k}'))
                    else:
                        k_reg = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                        if is_signed:
                            output.append(SassInstr(encode_shf_r_s32_hi_var(d, a, k_reg),
                                                    f'SHF.R.S32.HI R{d}, RZ, R{k_reg}, R{a}  // shr.s32 (var)'))
                        else:
                            output.append(SassInstr(encode_shf_r_u32_hi_var(d, a, k_reg),
                                                    f'SHF.R.U32.HI R{d}, RZ, R{k_reg}, R{a}  // shr.{typ} (var)'))

                elif op == 'shr' and typ in ('u64', 'b64'):
                    output.extend(_select_shr_u64(instr, ctx.ra))

                elif op == 'shr' and typ in ('s64',):
                    # shr.s64: arithmetic 64-bit right shift (sign-extends).
                    # K < 32: lo = SHF.R.U64(s_lo, k, s_hi) [pull in hi bits]
                    #          hi = SHF.R.S32.HI(s_hi, k)   [arithmetic shift of hi]
                    # K >= 32: lo = SHF.R.S32.HI(s_hi, k-32) [lo gets shifted hi]
                    #           hi = SHF.R.S32.HI(s_hi, 31)  [hi = all sign bits]
                    d_lo = ctx.ra.lo(instr.dest.name); d_hi = d_lo + 1
                    s_lo = ctx.ra.lo(instr.srcs[0].name); s_hi = s_lo + 1
                    k = _get_imm(instr.srcs[1])
                    if k < 32:
                        output.append(SassInstr(encode_shf_r_u32(d_lo, s_lo, k, s_hi),
                            f'SHF.R.U64 R{d_lo}, R{s_lo}, 0x{k:x}, R{s_hi}  // shr.s64 lo'))
                        output.append(SassInstr(encode_shf_r_s32_hi(d_hi, s_hi, k),
                            f'SHF.R.S32.HI R{d_hi}, RZ, 0x{k:x}, R{s_hi}  // shr.s64 hi'))
                    else:
                        k32 = k - 32
                        output.append(SassInstr(encode_shf_r_s32_hi(d_lo, s_hi, k32),
                            f'SHF.R.S32.HI R{d_lo}, RZ, 0x{k32:x}, R{s_hi}  // shr.s64 lo (K>={k})'))
                        output.append(SassInstr(encode_shf_r_s32_hi(d_hi, s_hi, 31),
                            f'SHF.R.S32.HI R{d_hi}, RZ, 0x1f, R{s_hi}  // shr.s64 hi=sign'))

                elif op == 'sub' and typ in ('u64', 's64'):
                    output.extend(_select_sub_u64(instr, ctx.ra))

                elif op == 'add' and typ in ('u64', 's64'):
                    output.extend(_select_add_u64(instr, ctx.ra, ctx))

                elif op == 'add' and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    if isinstance(instr.srcs[1], ImmOp):
                        # IADD3 with 32-bit immediate: dest = src0 + imm
                        a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        output.append(SassInstr(encode_iadd3_imm32(d, a, imm, RZ),
                                                f'IADD3 R{d}, R{a}, 0x{imm:x}, RZ  // add.{typ} imm'))
                    elif isinstance(instr.srcs[0], ImmOp):
                        b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                        imm = instr.srcs[0].value & 0xFFFFFFFF
                        output.append(SassInstr(encode_iadd3_imm32(d, b, imm, RZ),
                                                f'IADD3 R{d}, R{b}, 0x{imm:x}, RZ  // add.{typ} imm'))
                    else:
                        a = ctx.ra.r32(instr.srcs[0].name)
                        b = ctx.ra.r32(instr.srcs[1].name)
                        output.append(SassInstr(encode_iadd3(d, a, b, RZ),
                                                f'IADD3 R{d}, R{a}, R{b}, RZ  // add.{typ}'))

                elif op == 'sub' and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    if isinstance(instr.srcs[1], ImmOp):
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        lit_off = ctx._alloc_literal(imm)
                        # KERNEL-100.1 bugfix: when d==a, loading the literal
                        # into d clobbers the source before IADD3 reads it.
                        # Use a scratch register for the literal in that case.
                        lit_reg = d if d != a else _alloc_gpr(ctx)
                        output.append(SassInstr(encode_ldc(lit_reg, 0, lit_off),
                                                f'LDC R{lit_reg}, c[0][0x{lit_off:x}]  // sub imm={imm:#x}'))
                        output.append(SassInstr(encode_iadd3(d, a, lit_reg, RZ, negate_src1=True),
                                                f'IADD3 R{d}, R{a}, -R{lit_reg}, RZ  // sub.{typ} imm'))
                    else:
                        b = ctx.ra.r32(instr.srcs[1].name)
                        output.append(SassInstr(encode_iadd3(d, a, b, RZ, negate_src1=True),
                                                f'IADD3 R{d}, R{a}, -R{b}, RZ  // sub.{typ}'))

                elif op in ('and', 'or', 'xor') and typ in ('b32', 'u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    lut = {'and': LOP3_AND, 'or': LOP3_OR, 'xor': LOP3_XOR}[op]
                    if isinstance(instr.srcs[1], ImmOp):
                        # IMAD-FUSE-1: Use LOP3.IMM (opcode 0x812) to encode the
                        # immediate inline, avoiding the separate materialize step.
                        # Ground truth: ptxas uses LOP3.IMM for xor/and/or with
                        # 32-bit immediate on SM_120 (LUT values differ from R-R form).
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        lut_imm = {'and': LOP3_IMM_AND, 'or': LOP3_IMM_OR, 'xor': LOP3_IMM_XOR}[op]
                        if d < 14:
                            output.append(SassInstr(
                                encode_lop3_imm32(d, a, imm, RZ, lut_imm),
                                f'LOP3.LUT R{d}, R{a}, 0x{imm:x}, RZ, 0x{lut_imm:02x}  // {op}.{typ} imm'))
                        else:
                            # LOP3 dest must be < R14 on SM_120 — use scratch + MOV
                            used = {a, d}
                            scratch = next((r for r in range(14) if r not in used), 0)
                            output.append(SassInstr(
                                encode_lop3_imm32(scratch, a, imm, RZ, lut_imm),
                                f'LOP3.LUT R{scratch}, R{a}, 0x{imm:x}, RZ, 0x{lut_imm:02x}  // {op}.{typ} imm (via R{scratch})'))
                            output.append(SassInstr(encode_iadd3(d, scratch, RZ, RZ),
                                f'MOV R{d}, R{scratch}  // lop3 fixup'))
                    else:
                        b = ctx.ra.r32(instr.srcs[1].name)
                        _emit_lop3(output, ctx, d, a, b, RZ, lut, f'LOP3.LUT R{d}, R{a}, R{b}, RZ, 0x{lut:02x}  // {op}.{typ}')

                elif op in ('and', 'or', 'xor') and typ in ('b64', 'u64', 's64'):
                    # 64-bit logic: apply LOP3 to lo and hi words separately.
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = ctx.ra.lo(instr.srcs[0].name)
                    lut = {'and': LOP3_AND, 'or': LOP3_OR, 'xor': LOP3_XOR}[op]
                    if isinstance(instr.srcs[1], ImmOp):
                        imm = instr.srcs[1].value & 0xFFFF_FFFF_FFFF_FFFF
                        imm_lo = imm & 0xFFFFFFFF
                        imm_hi = (imm >> 32) & 0xFFFFFFFF
                        t = _alloc_gpr(ctx)
                        lit_lo = ctx._alloc_literal(imm_lo)
                        output.append(SassInstr(encode_ldc(t, 0, lit_lo),
                                                f'LDC R{t}, c[0][0x{lit_lo:x}]  // {op}.b64 imm_lo'))
                        _emit_lop3(output, ctx, d_lo, a_lo, t, RZ, lut, f'LOP3.LUT R{d_lo}, R{a_lo}, R{t}, RZ, 0x{lut:02x}  // {op}.b64 lo')
                        lit_hi = ctx._alloc_literal(imm_hi)
                        output.append(SassInstr(encode_ldc(t, 0, lit_hi),
                                                f'LDC R{t}, c[0][0x{lit_hi:x}]  // {op}.b64 imm_hi'))
                        _emit_lop3(output, ctx, d_lo+1, a_lo+1, t, RZ, lut, f'LOP3.LUT R{d_lo+1}, R{a_lo+1}, R{t}, RZ, 0x{lut:02x}  // {op}.b64 hi')
                    else:
                        b_lo = ctx.ra.lo(instr.srcs[1].name)
                        _emit_lop3(output, ctx, d_lo, a_lo, b_lo, RZ, lut, f'LOP3.LUT R{d_lo}, R{a_lo}, R{b_lo}, RZ, 0x{lut:02x}  // {op}.b64 lo')
                        _emit_lop3(output, ctx, d_lo+1, a_lo+1, b_lo+1, RZ, lut, f'LOP3.LUT R{d_lo+1}, R{a_lo+1}, R{b_lo+1}, RZ, 0x{lut:02x}  // {op}.b64 hi')

                elif op == 'not' and typ in ('b32', 'u32', 's32'):
                    # not.b32 d, a  →  LOP3.LUT d, a, RZ, RZ, 0x0F  (~a)
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    _emit_lop3(output, ctx, d, a, RZ, RZ, 0x0F, f'LOP3.LUT R{d}, R{a}, RZ, RZ, 0x0f  // not.{typ}')

                elif op == 'not' and typ in ('b64', 'u64', 's64'):
                    # not.b64 d, a  →  two LOP3.LUT on lo and hi words
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = ctx.ra.lo(instr.srcs[0].name)
                    _emit_lop3(output, ctx, d_lo, a_lo, RZ, RZ, 0x0F, f'LOP3.LUT R{d_lo}, R{a_lo}, RZ, RZ, 0x0f  // not.{typ} lo')
                    _emit_lop3(output, ctx, d_lo+1, a_lo+1, RZ, RZ, 0x0F, f'LOP3.LUT R{d_lo+1}, R{a_lo+1}, RZ, RZ, 0x0f  // not.{typ} hi')

                elif op == 'mul' and 'lo' in instr.types and typ in ('u32', 's32'):
                    # PEEPHOLE: mul+add fusion → IMAD with third operand (DISABLED FOR TESTING)
                    if False:
                        pass
                    # Look ahead: find add.u32 within next 3 instructions that uses our result
                    _next = None
                    _next_offset = 0
                    # Skip peephole if mul srcs aren't both RegOp (e.g., immediate multiplier)
                    if isinstance(instr.srcs[0], RegOp) and isinstance(instr.srcs[1], RegOp):
                     for _la in range(1, min(4, len(bb.instructions) - _instr_idx)):
                        _cand = bb.instructions[_instr_idx + _la]
                        if (_cand.op == 'add' and _cand.types and _cand.types[-1] in ('u32', 's32')
                                and isinstance(_cand.srcs[0], RegOp) and isinstance(_cand.srcs[1], RegOp)):
                            _next = _cand
                            _next_offset = _la
                            break
                    if _next:
                        # Check if one source of the add is the mul's dest
                        mul_dest_name = instr.dest.name
                        add_src0, add_src1 = _next.srcs[0].name, _next.srcs[1].name
                        add_other = None
                        if add_src0 == mul_dest_name:
                            add_other = add_src1
                        elif add_src1 == mul_dest_name:
                            add_other = add_src0
                        if add_other is not None:
                            # FUSION: mul a*b + c → IMAD dest, a, b_ur, c
                            fused_dest = ctx.ra.r32(_next.dest.name)
                            mul_a = instr.srcs[0].name
                            mul_b = instr.srcs[1].name
                            c_reg = ctx.ra.r32(add_other)
                            # Check if mul_a or mul_b is in UR (ctaid.x)
                            a_ur = ctx._ur_for_param.get(mul_a)
                            b_ur = ctx._ur_for_param.get(mul_b)
                            # Determine the GPR source (the mul operand that is NOT
                            # in a uniform register).
                            a_gpr = None
                            if a_ur is not None:
                                a_gpr = ctx.ra.r32(mul_b)
                            elif b_ur is not None:
                                a_gpr = ctx.ra.r32(mul_a)
                            # Both in GPR → can't fuse with R-UR IMAD, fall through
                            if a_gpr is not None:
                                # FG-1.14C / FG-2.2: NARROW defensive check at
                                # this specific fusion site, not a global rule.
                                #
                                # FG-2.2 verified across the 21-kernel suite +
                                # FG-1 reproducers + FG-2.1 repros that PTXAS
                                # itself emits aliased IMADs (dest ∈ {srcs})
                                # aggressively — 67 of its 106 IMADs alias a
                                # source, and it matches OURS byte-for-byte on
                                # 11 PARITY kernels that contain 65 such
                                # aliased IMADs.  The STRICT invariant dest ∉
                                # {src0,src1,src2} is therefore NOT a
                                # correctness requirement; it is empirically
                                # violated by the reference compiler.
                                #
                                # The REAL hardware requirement — enforced by
                                # sass.schedule._enforce_gpr_latency reading
                                # sass.scoreboard._OPCODE_META — is that the
                                # dest must not be read with 0 instructions of
                                # gap by a non-latency-aware consumer.  Since
                                # FG-1 closeout registered IMAD.R-UR (0xc24)
                                # in _OPCODE_META with min_gpr_gap=1, the
                                # scheduler now inserts the required NOP and
                                # the raw aliasing at this very site is safe
                                # even when dest == src0.
                                #
                                # We keep the fresh-GPR rebind here anyway
                                # because this particular fusion path creates
                                # a quirky ownership pattern: the mul-operand
                                # vreg and the add-dest vreg collapse into the
                                # same physical register AFTER the linear-scan
                                # allocator already made its decisions.  Having
                                # the isel pick a scratch register keeps that
                                # collapse local to the fusion site and avoids
                                # surprising downstream consumers that were
                                # expecting separate registers.  This is
                                # defense-in-depth, not structural correctness.
                                if fused_dest == a_gpr or fused_dest == c_reg:
                                    fresh = _alloc_gpr(ctx)
                                    ctx.ra.int_regs[_next.dest.name] = fresh
                                    ctx.ra.int_regs[mul_dest_name] = fresh
                                    fused_dest = fresh
                                else:
                                    # Still alias mul's dest to the add's dest,
                                    # as before.
                                    ctx.ra.int_regs[mul_dest_name] = fused_dest
                                    ctx.ra.int_regs[_next.dest.name] = fused_dest
                                # Emit the IMAD (source ordering preserved from
                                # the original branches).  The ALU-latency NOP
                                # that used to follow this instruction (FG-1.14C)
                                # was removed in the FG-1 closeout after opcode
                                # 0xc24 was registered in
                                # sass.scoreboard._OPCODE_META with
                                # min_gpr_gap=1, so the global scheduler now
                                # enforces the 1-instruction reader gap from
                                # the model rather than via a local hack.
                                if a_ur is not None:
                                    output.append(SassInstr(
                                        encode_imad_ur(fused_dest, a_gpr, a_ur, c_reg),
                                        f'IMAD R{fused_dest}, R{a_gpr}, UR{a_ur}, R{c_reg}  // fused mul+add'))
                                else:
                                    output.append(SassInstr(
                                        encode_imad_ur(fused_dest, a_gpr, b_ur, c_reg),
                                        f'IMAD R{fused_dest}, R{a_gpr}, UR{b_ur}, R{c_reg}  // fused mul+add'))
                                # Mark the add instruction to skip
                                if not hasattr(ctx, '_skip_instrs'):
                                    ctx._skip_instrs = set()
                                ctx._skip_instrs.add(id(_next))
                                continue

                    # mul.lo.s32 → IMAD R-UR or IMAD.WIDE with immediate
                    # NOTE: IMAD R-R (0x224) is NOT valid on SM_120!
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    if isinstance(instr.srcs[1], ImmOp):
                        # PEEPHOLE: mul.lo.s32 + cvt.u64.u32 → IMAD.WIDE
                        # If the next instruction is cvt to 64-bit using our result,
                        # emit IMAD.WIDE directly (1 instruction instead of 3).
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        _next_cvt = None
                        if _instr_idx + 1 < len(bb.instructions):
                            _ni = bb.instructions[_instr_idx + 1]
                            if (_ni.op == 'cvt'
                                    and any(t in ('u64', 's64') for t in _ni.types)
                                    and isinstance(_ni.srcs[0], RegOp)
                                    and _ni.srcs[0].name == instr.dest.name):
                                _next_cvt = _ni
                        if _next_cvt is not None and imm > 0 and imm <= 0xFFFF:
                            # Fuse: emit IMAD.WIDE Rd_lo, src, imm, RZ
                            d_lo = ctx.ra.lo(_next_cvt.dest.name)
                            output.append(SassInstr(
                                encode_imad_wide(d_lo, a, imm, RZ),
                                f'IMAD.WIDE R{d_lo}, R{a}, 0x{imm:x}, RZ  // fused mul+cvt64'))
                            if not hasattr(ctx, '_skip_instrs'):
                                ctx._skip_instrs = set()
                            ctx._skip_instrs.add(id(_next_cvt))
                            continue

                        # Immediate multiplier: use IMAD.SHL.U32 if power-of-2 and ≤15,
                        # else try IMAD R-imm (16-bit immediate, opcode 0x824),
                        # else load into UR via literal pool and use IMAD R-UR.
                        if imm > 0 and (imm & (imm - 1)) == 0:
                            # Power of two: IMAD.SHL.U32 dest, src, imm, RZ
                            shift = imm.bit_length() - 1
                            if shift <= 15:
                                output.append(SassInstr(encode_imad_shl_u32(d, a, shift),
                                    f'IMAD.SHL.U32 R{d}, R{a}, 0x{imm:x}, RZ  // mul.lo imm={imm}'))
                            elif imm <= 0xFFFF:
                                from sass.encoding.sm_120_opcodes import encode_imad_r_imm
                                output.append(SassInstr(encode_imad_r_imm(d, a, imm, RZ),
                                    f'IMAD R{d}, R{a}, 0x{imm:x}, RZ  // mul.lo imm'))
                            else:
                                lit_off = ctx._alloc_literal(imm)
                                ur_tmp = ctx._next_ur; ctx._next_ur += 1
                                output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, lit_off),
                                    f'LDCU.32 UR{ur_tmp}, c[0][0x{lit_off:x}]  // mul.lo imm'))
                                output.append(SassInstr(encode_imad_ur(d, a, ur_tmp, RZ),
                                    f'IMAD R{d}, R{a}, UR{ur_tmp}, RZ  // mul.lo imm'))
                        elif imm <= 0xFFFF:
                            # 16-bit immediate: use IMAD R-imm directly (ptxas pattern).
                            from sass.encoding.sm_120_opcodes import encode_imad_r_imm
                            output.append(SassInstr(encode_imad_r_imm(d, a, imm, RZ),
                                f'IMAD R{d}, R{a}, 0x{imm:x}, RZ  // mul.lo imm'))
                        else:
                            lit_off = ctx._alloc_literal(imm)
                            ur_tmp = ctx._next_ur; ctx._next_ur += 1
                            output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, lit_off),
                                f'LDCU.32 UR{ur_tmp}, c[0][0x{lit_off:x}]  // mul.lo imm'))
                            output.append(SassInstr(encode_imad_ur(d, a, ur_tmp, RZ),
                                f'IMAD R{d}, R{a}, UR{ur_tmp}, RZ  // mul.lo imm'))
                        continue
                    b = ctx.ra.r32(instr.srcs[1].name)
                    # Check if either source lives in a UR (ctaid.x via S2UR)
                    a_ur = ctx._ur_for_param.get(
                        instr.srcs[0].name if isinstance(instr.srcs[0], RegOp) else None)
                    b_ur = ctx._ur_for_param.get(
                        instr.srcs[1].name if isinstance(instr.srcs[1], RegOp) else None)
                    if a_ur is not None:
                        # src0 is in UR (e.g., ctaid.x) — use IMAD R{b}, UR{a_ur}, RZ
                        output.append(SassInstr(encode_imad_ur(d, b, a_ur, RZ),
                            f'IMAD R{d}, R{b}, UR{a_ur}, RZ  // mul.lo.{typ} (src0 in UR)'))
                        continue
                    if b_ur is not None:
                        # src1 is in UR — use IMAD R{a}, UR{b_ur}, RZ
                        output.append(SassInstr(encode_imad_ur(d, a, b_ur, RZ),
                            f'IMAD R{d}, R{a}, UR{b_ur}, RZ  // mul.lo.{typ} (src1 in UR)'))
                        continue
                    # Check if either source is a param → use IMAD R-UR
                    b_param = ctx._reg_param_off.get(
                        instr.srcs[1].name if isinstance(instr.srcs[1], RegOp) else None)
                    a_param = ctx._reg_param_off.get(
                        instr.srcs[0].name if isinstance(instr.srcs[0], RegOp) else None)
                    if ctx.sm_version == 89:
                        # SM_89: use IMAD.cb (constant bank multiply) instead of LDCU.32+R-UR.
                        # If a source is tracked in _reg_param_off, emit IMAD.cb directly.
                        if b_param is not None:
                            from sass.encoding.sm_89_opcodes import encode_imad_cbuf
                            output.append(SassInstr(encode_imad_cbuf(d, a, 0, b_param, RZ),
                                f'IMAD R{d}, R{a}, c[0][0x{b_param:x}], RZ  // mul.lo.{typ} cbuf'))
                            continue
                        elif a_param is not None:
                            from sass.encoding.sm_89_opcodes import encode_imad_cbuf
                            output.append(SassInstr(encode_imad_cbuf(d, b, 0, a_param, RZ),
                                f'IMAD R{d}, R{b}, c[0][0x{a_param:x}], RZ  // mul.lo.{typ} cbuf'))
                            continue
                        # else: neither in cbuf, fall through to IMAD.WIDE R-R
                        b_param = None; a_param = None
                    if b_param is not None:
                        ur_tmp = ctx._next_ur; ctx._next_ur += 1
                        output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, b_param),
                            f'LDCU.32 UR{ur_tmp}, c[0][0x{b_param:x}]'))
                        output.append(SassInstr(encode_imad_ur(d, a, ur_tmp, RZ),
                            f'IMAD R{d}, R{a}, UR{ur_tmp}, RZ  // mul.lo.{typ}'))
                    elif a_param is not None:
                        ur_tmp = ctx._next_ur; ctx._next_ur += 1
                        output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, a_param),
                            f'LDCU.32 UR{ur_tmp}, c[0][0x{a_param:x}]'))
                        output.append(SassInstr(encode_imad_ur(d, b, ur_tmp, RZ),
                            f'IMAD R{d}, R{b}, UR{ur_tmp}, RZ  // mul.lo.{typ}'))
                    else:
                        # IMAD R-R (0x2a4) is BROKEN on SM_120 but IMAD.WIDE R-R
                        # (0x225) works. Use WIDE to get the full 64-bit product,
                        # then take only the low 32 bits (dest register).
                        # IMAD.WIDE writes dest AND dest+1, so allocate a scratch
                        # for the high word to avoid clobbering live registers.
                        t = _alloc_gpr(ctx)
                        if t % 2 != 0:
                            t = _alloc_gpr(ctx)  # even-align for pair
                        _alloc_gpr(ctx)  # reserve t+1
                        output.append(SassInstr(encode_imad_wide_rr(t, a, b, RZ),
                            f'IMAD.WIDE R{t}, R{a}, R{b}, RZ  // mul.lo.{typ} R-R via WIDE'))
                        if t != d:
                            output.append(SassInstr(encode_mov(d, t),
                                f'MOV R{d}, R{t}  // mul.lo result'))

                elif op == 'mul' and 'lo' in instr.types and typ in ('u64', 's64', 'b64'):
                    # mul.lo.u64 d, a, b = lower 64 bits of a * b
                    # Decomposed into three IMAD operations:
                    #   IMAD.WIDE d_lo, a_lo, b_lo, RZ  → d_lo:d_hi = a_lo × b_lo
                    #   IMAD.RR   d_hi, a_lo, b_hi, d_hi → d_hi += a_lo × b_hi (lo bits only)
                    #   IMAD.RR   d_hi, a_hi, b_lo, d_hi → d_hi += a_hi × b_lo
                    #
                    # FG-1.12 — UR→GPR residency enforcement.
                    #
                    # IMAD.WIDE R-R (opcode 0x225) reads its source operands as
                    # GPR pairs.  But a u64 source value may currently live
                    # only in UR space — e.g., a u64 parameter loaded via
                    # LDCU into UR8..UR9 in the preamble and never copied to
                    # a GPR pair.  `ctx.ra.lo(name)` returns a GPR index for
                    # the vreg, but that GPR may have never been written if
                    # the value was only UR-resident.
                    #
                    # Root cause of FG-1-D (diagnosed in FG-1.11): Forge
                    # reduce_step loaded `stride` into UR8..UR9, then the
                    # mul.lo.u64 lowering emitted `IMAD.WIDE R12, R2, R10`
                    # reading R2 — which was never written.  Loop counter
                    # update `i += stride*2` produced garbage, loop exited
                    # after 1 iteration.
                    #
                    # Fix: enforce the residency invariant locally.  If a
                    # source vreg is UR-resident and has not been copied
                    # to a GPR yet (`not in _gpr_written`), emit two MOV
                    # R,UR instructions (one per half) to materialize the
                    # u64 value into a fresh GPR pair, update the
                    # allocator's mapping, and mark the vreg as GPR-written.
                    # Then proceed with the standard IMAD.WIDE R-R lowering.
                    #
                    # Narrowness: this block only fires when at least one
                    # source is in `_ur_params` and not already in
                    # `_gpr_written`.  Kernels whose u64 operands are
                    # already GPR-resident (the existing 21 hand-crafted
                    # kernels) never hit this path and are unaffected.
                    _gpr_written = getattr(ctx, '_gpr_written', set())
                    _ur_params   = getattr(ctx, '_ur_params', {})

                    def _ensure_u64_gpr(src):
                        """Return the GPR lo index for a u64 source, materializing
                        from UR via MOV R,UR if the vreg is UR-resident and has
                        not yet been copied to a GPR pair.
                        """
                        if not isinstance(src, RegOp):
                            return ctx.ra.lo(src.name)
                        if (src.name in _ur_params
                                and src.name not in _gpr_written):
                            ur_base = _ur_params[src.name]
                            tmp_lo = _alloc_gpr_pair(ctx)
                            output.append(SassInstr(
                                encode_mov_gpr_from_ur(tmp_lo, ur_base),
                                f'MOV R{tmp_lo}, UR{ur_base}  '
                                f'// FG-1.12: materialize {src.name}.lo for mul.lo.{typ}'))
                            output.append(SassInstr(
                                encode_mov_gpr_from_ur(tmp_lo + 1, ur_base + 1),
                                f'MOV R{tmp_lo+1}, UR{ur_base+1}  '
                                f'// FG-1.12: materialize {src.name}.hi for mul.lo.{typ}'))
                            # Rebind allocator so subsequent consumers see the
                            # GPR pair, and flag as GPR-written so UR-path
                            # checks (e.g. add.u64 R-UR) don't fire again.
                            ctx.ra.int_regs[src.name] = tmp_lo
                            if hasattr(ctx, '_gpr_written'):
                                ctx._gpr_written.add(src.name)
                            return tmp_lo
                        return ctx.ra.lo(src.name)

                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = _ensure_u64_gpr(instr.srcs[0])
                    b_lo = _ensure_u64_gpr(instr.srcs[1])
                    output.append(SassInstr(encode_imad_wide_rr(d_lo, a_lo, b_lo, RZ),
                        f'IMAD.WIDE R{d_lo}, R{a_lo}, R{b_lo}, RZ  // mul.lo.{typ} wide'))
                    # IMAD R-R (0x2a4) is broken on SM_120. Use IMAD.WIDE for cross terms:
                    # cross1 = a_lo * b_hi; cross2 = a_hi * b_lo; d_hi += cross1 + cross2
                    # Skip any cross term whose multiplier register is known to be zero.
                    _zero_regs = getattr(ctx, '_zero_regs', set())
                    b_hi = b_lo + 1
                    a_hi = a_lo + 1
                    need_cross1 = b_hi not in _zero_regs
                    need_cross2 = a_hi not in _zero_regs
                    if need_cross1 or need_cross2:
                        t = _alloc_gpr_pair(ctx)
                    if need_cross1:
                        # cross1: t = a_lo * b_hi (low 32 of wide product)
                        output.append(SassInstr(encode_imad_wide_rr(t, a_lo, b_hi, RZ),
                            f'IMAD.WIDE R{t}, R{a_lo}, R{b_hi}, RZ  // cross a_lo*b_hi'))
                        output.append(SassInstr(encode_iadd3(d_lo+1, d_lo+1, t, RZ),
                            f'IADD3 R{d_lo+1}, R{d_lo+1}, R{t}, RZ  // d_hi += cross1'))
                    if need_cross2:
                        # cross2: t = a_hi * b_lo (low 32 of wide product)
                        output.append(SassInstr(encode_imad_wide_rr(t, a_hi, b_lo, RZ),
                            f'IMAD.WIDE R{t}, R{a_hi}, R{b_lo}, RZ  // cross a_hi*b_lo'))
                        output.append(SassInstr(encode_iadd3(d_lo+1, d_lo+1, t, RZ),
                            f'IADD3 R{d_lo+1}, R{d_lo+1}, R{t}, RZ  // d_hi += cross2'))
                    if need_cross1 or need_cross2:
                        # Free cross-product scratch for reuse
                        _free_scratch(ctx, [t, t + 1])

                elif op == 'st' and 'shared' in instr.types:
                    from ptx.ir import MemOp
                    addr_op = instr.srcs[0]
                    data_op = instr.srcs[1]
                    data_r = ctx.ra.r32(data_op.name) if isinstance(data_op, RegOp) else RZ
                    if isinstance(addr_op, MemOp):
                        offset = addr_op.offset
                        base = addr_op.base
                        # Check if base is a register (starts with %)
                        if base.startswith('%') and base in ctx.ra.int_regs:
                            # 32-bit register → use directly
                            addr_r = ctx.ra.r32(base)
                            output.append(SassInstr(encode_sts_r(4, addr_r, data_r, offset),
                                f'STS [UR4+R{addr_r}+{offset:#x}], R{data_r}  // st.shared'))
                        elif base.startswith('%') and hasattr(ctx.ra, 'lo') and base in getattr(ctx.ra, 'int64_regs', {}):
                            # 64-bit register → use low 32 bits for smem addressing
                            addr_r = ctx.ra.lo(base)
                            output.append(SassInstr(encode_sts_r(4, addr_r, data_r, offset),
                                f'STS [UR4+R{addr_r}+{offset:#x}], R{data_r}  // st.shared (64->32)'))
                        else:
                            # Shared variable name or fixed offset → immediate-only
                            smem_off = ctx._smem_offsets.get(base, 0) + offset if hasattr(ctx, '_smem_offsets') else offset
                            output.append(SassInstr(encode_sts(4, smem_off, data_r),
                                f'STS [UR4+{smem_off:#x}], R{data_r}  // st.shared'))
                    else:
                        output.append(SassInstr(encode_sts(4, 0, data_r),
                                                f'STS [UR4+0x0], R{data_r}  // st.shared'))

                elif op == 'ld' and 'shared' in instr.types:
                    from ptx.ir import MemOp
                    dest_r = ctx.ra.r32(instr.dest.name)
                    addr_op = instr.srcs[0]
                    if isinstance(addr_op, MemOp):
                        offset = addr_op.offset
                        base = addr_op.base
                        if base.startswith('%') and base in ctx.ra.int_regs:
                            addr_r = ctx.ra.r32(base)
                            output.append(SassInstr(encode_lds_r(dest_r, 4, addr_r, offset),
                                f'LDS R{dest_r}, [UR4+R{addr_r}+{offset:#x}]  // ld.shared'))
                        elif base.startswith('%') and hasattr(ctx.ra, 'lo') and base in getattr(ctx.ra, 'int64_regs', {}):
                            addr_r = ctx.ra.lo(base)
                            output.append(SassInstr(encode_lds_r(dest_r, 4, addr_r, offset),
                                f'LDS R{dest_r}, [UR4+R{addr_r}+{offset:#x}]  // ld.shared (64->32)'))
                        else:
                            smem_off = ctx._smem_offsets.get(base, 0) + offset if hasattr(ctx, '_smem_offsets') else offset
                            output.append(SassInstr(encode_lds(dest_r, 4, smem_off),
                                f'LDS R{dest_r}, [UR4+{smem_off:#x}]  // ld.shared'))
                    else:
                        output.append(SassInstr(encode_lds(dest_r, 4, 0),
                                                f'LDS R{dest_r}, [UR4+0x0]  // ld.shared'))

                elif op == 'bar':
                    # SM_120: BSYNC before BAR.SYNC for shared memory visibility
                    _has_sts_in_kernel = any(
                        inst2.op == 'st' and 'shared' in inst2.types
                        for bb2 in fn.blocks for inst2 in bb2.instructions)
                    if _has_sts_in_kernel:
                        BSYNC_RAW = bytes.fromhex('41790000000000000002800300ea1f00')
                        output.append(SassInstr(BSYNC_RAW, 'BSYNC  // pre-barrier sync'))
                    output.append(SassInstr(encode_bar_sync(0),
                                            f'BAR.SYNC 0'))

                elif op == 'add' and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_fadd(d, a, b),
                                            f'FADD R{d}, R{a}, R{b}  // add.f32'))

                elif op == 'sub' and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_fadd(d, b, a, negate_src0=True),
                                            f'FADD R{d}, -R{b}, R{a}  // sub.f32'))

                elif op == 'mul' and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    # Use FMUL with inline immediate (0x820) when one operand is constant
                    if isinstance(instr.srcs[1], ImmOp):
                        a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        output.append(SassInstr(encode_fmul_imm(d, a, imm),
                                                f'FMUL R{d}, R{a}, 0x{imm:08x}  // mul.f32 imm'))
                    elif isinstance(instr.srcs[0], ImmOp):
                        b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                        imm = instr.srcs[0].value & 0xFFFFFFFF
                        output.append(SassInstr(encode_fmul_imm(d, b, imm),
                                                f'FMUL R{d}, R{b}, 0x{imm:08x}  // mul.f32 imm'))
                    else:
                        a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                        b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                        output.append(SassInstr(encode_fmul(d, a, b),
                                                f'FMUL R{d}, R{a}, R{b}  // mul.f32'))

                elif op == 'fma' and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    c = _materialize_imm(instr.srcs[2], ctx, ctx.ra, output)
                    # Use RZ for known-zero addend and remove the dead zeroing instruction
                    if hasattr(ctx, '_zero_regs') and c in ctx._zero_regs:
                        if output and f'R{c}, RZ, 0x0, RZ' in output[-1].comment:
                            output.pop()
                        c = RZ
                    # FFMA.IMM: if the multiplier is a known immediate, bake it in
                    _imm_regs = getattr(ctx, '_imm_regs', {})
                    if b in _imm_regs:
                        imm_val = _imm_regs[b]
                        # Remove the IADD3 that loaded this immediate
                        for j in range(len(output) - 1, max(len(output) - 5, -1), -1):
                            if j >= 0 and f'R{b}, RZ, 0x{imm_val:x}, RZ' in output[j].comment:
                                output.pop(j)
                                break
                        output.append(SassInstr(encode_ffma_imm(d, a, imm_val, c),
                                                f'FFMA.IMM R{d}, R{a}, 0x{imm_val:x}, R{c}  // fma.f32 (imm)'))
                    else:
                        output.append(SassInstr(encode_ffma(d, a, b, c),
                                                f'FFMA R{d}, R{a}, R{b}, R{c}  // fma.f32'))
                    # Clear zero-reg status: dest is no longer zero after FMA
                    if hasattr(ctx, '_zero_regs'):
                        ctx._zero_regs.discard(d)

                elif op == 'add' and typ == 'f64':
                    d = ctx.ra.lo(instr.dest.name)
                    a = _f64_to_gpr(instr.srcs[0].name, ctx, output)
                    b = _f64_to_gpr(instr.srcs[1].name, ctx, output)
                    output.append(SassInstr(encode_dadd(d, a, b),
                                            f'DADD R{d}, R{a}, R{b}  // add.f64'))

                elif op == 'sub' and typ == 'f64':
                    # sub.f64 d, a, b → d = a - b = -b + a → DADD d, -b, a
                    # Mirrors sub.f32 which uses FADD(d, b, a, negate_src0=True).
                    d = ctx.ra.lo(instr.dest.name)
                    a = _f64_to_gpr(instr.srcs[0].name, ctx, output)
                    b = _f64_to_gpr(instr.srcs[1].name, ctx, output)
                    output.append(SassInstr(encode_dadd(d, b, a, negate_src0=True),
                                            f'DADD R{d}, -R{b}, R{a}  // sub.f64'))

                elif op == 'mul' and typ == 'f64':
                    d = ctx.ra.lo(instr.dest.name)
                    a = _f64_to_gpr(instr.srcs[0].name, ctx, output)
                    b = _f64_to_gpr(instr.srcs[1].name, ctx, output)
                    output.append(SassInstr(encode_dmul(d, a, b),
                                            f'DMUL R{d}, R{a}, R{b}  // mul.f64'))

                elif op == 'fma' and typ == 'f64':
                    d = ctx.ra.lo(instr.dest.name)
                    a = _f64_to_gpr(instr.srcs[0].name, ctx, output)
                    b_ur = ctx._ur_params.get(instr.srcs[1].name) if ctx else None
                    c_ur = ctx._ur_params.get(instr.srcs[2].name) if ctx else None
                    if b_ur is not None and c_ur is not None:
                        # Both multiplier and addend are in UR — DFMA R-R-UR-UR
                        output.append(SassInstr(encode_dfma_ur_ur(d, a, b_ur, c_ur),
                                                f'DFMA R{d}, R{a}, UR{b_ur}, UR{c_ur}  // fma.f64 (UR×UR)'))
                    else:
                        b = _f64_to_gpr(instr.srcs[1].name, ctx, output)
                        c = _f64_to_gpr(instr.srcs[2].name, ctx, output)
                        output.append(SassInstr(encode_dfma(d, a, b, c),
                                                f'DFMA R{d}, R{a}, R{b}, R{c}  // fma.f64'))

                elif op == 'mma' and 'sync' in instr.types and 'aligned' in instr.types:
                    _types_set = set(instr.types)
                    shape = next((t for t in instr.types if t.startswith('m')), None)
                    # PTX tuple operands: extract base register (first element)
                    def _tuple_base(op_node):
                        nm = op_node.name if hasattr(op_node, 'name') else str(op_node)
                        # strip leading '{' and trailing '}' if present
                        nm = nm.lstrip('{').split(',')[0].rstrip('}').strip()
                        return nm
                    d_nm = _tuple_base(instr.dest) if instr.dest else None
                    srcs = instr.srcs or []
                    a_nm = _tuple_base(srcs[0]) if len(srcs) > 0 else None
                    b_nm = _tuple_base(srcs[1]) if len(srcs) > 1 else None
                    c_nm = _tuple_base(srcs[2]) if len(srcs) > 2 else None
                    def _r(nm): return ctx.ra.r32(nm) if nm else RZ
                    d = _r(d_nm); a = _r(a_nm); b = _r(b_nm); c = _r(c_nm)
                    # WB-2: HMMA RZ-substitution.  When all input slots are
                    # provably-zero (analyze_mma_zero_subst), B and C are
                    # encoded as RZ and src_a is aliased to dst (matches
                    # ptxas's hmma_zero pattern, lets the input init MOVs
                    # be elided).  HMMA-only — IMMA/DMMA/QMMA branches
                    # below are unaffected.
                    _rz_slots = getattr(ctx, '_hmma_rz_subst', {}).get(id(instr), set())
                    if 'a' in _rz_slots:
                        a = d
                    if 'b' in _rz_slots:
                        b = RZ
                    if 'c' in _rz_slots:
                        c = RZ
                    if shape == 'm8n8k4' and 'f64' in _types_set:
                        output.append(SassInstr(encode_dmma_8x8x4(d, a, b, c),
                                                f'DMMA.8x8x4 R{d}, R{a}, R{b}, R{c}'))
                    elif 'e4m3' in _types_set:
                        # SM_120 QMMA hardware constraint: dest register == src_a register
                        # (the A matrix values must be pre-loaded into the D register positions).
                        # PTX must use the same virtual regs for D and A operands.
                        output.append(SassInstr(encode_qmma_e4m3_f32(d, d, b, c),
                                                f'QMMA.16832.F32.E4M3.E4M3 R{d}, R{d}, R{b}, R{c}'))
                    elif 'e5m2' in _types_set:
                        # SM_120 QMMA hardware constraint: dest register == src_a register.
                        output.append(SassInstr(encode_qmma_e5m2_f32(d, d, b, c),
                                                f'QMMA.16832.F32.E5M2.E5M2 R{d}, R{d}, R{b}, R{c}'))
                    elif 's8' in _types_set or 'u8' in _types_set:
                        output.append(SassInstr(encode_imma_s8_s32(d, a, b, c),
                                                f'IMMA.16832.S8 R{d}, R{a}, R{b}, R{c}'))
                    elif 'tf32' in _types_set:
                        output.append(SassInstr(encode_hmma_tf32_f32(d, a, b, c),
                                                f'HMMA.TF32 R{d}, R{a}, R{b}, R{c}'))
                    elif 'bf16' in _types_set:
                        output.append(SassInstr(encode_hmma_bf16_f32(d, a, b, c),
                                                f'HMMA.BF16 R{d}, R{a}, R{b}, R{c}'))
                    elif shape == 'm16n8k8':
                        output.append(SassInstr(encode_hmma_f16_f32_k8(d, a, b, c),
                                                f'HMMA.1688.F32 R{d}, R{a}, R{b}, R{c}'))
                    else:  # m16n8k16 and other shapes
                        output.append(SassInstr(encode_hmma_f16_f32(d, a, b, c),
                                                f'HMMA.16816.F32 R{d}, R{a}, R{b}, R{c}'))

                elif op == 'ldmatrix' and 'sync' in instr.types and 'aligned' in instr.types:
                    _types_set = set(instr.types)
                    # ldmatrix.sync.aligned.x4.m8n8.shared.b16 {d0,d1,d2,d3}, [addr]
                    # dest is a tuple of 1/2/4 registers; addr is srcs[0]
                    def _tuple_base(op_node):
                        nm = op_node.name if hasattr(op_node, 'name') else str(op_node)
                        nm = nm.lstrip('{').split(',')[0].rstrip('}').strip()
                        return nm
                    d_nm = _tuple_base(instr.dest) if instr.dest else None
                    addr_nm = (instr.srcs[0].name if instr.srcs and hasattr(instr.srcs[0], 'name')
                               else None)
                    d = ctx.ra.r32(d_nm) if d_nm else RZ
                    a = ctx.ra.r32(addr_nm) if addr_nm else RZ
                    if 'x1' in _types_set:
                        output.append(SassInstr(encode_ldsm_x1(d, a),
                                                f'LDSM.x1 R{d}, [R{a}]'))
                    elif 'x2' in _types_set:
                        output.append(SassInstr(encode_ldsm_x2(d, a),
                                                f'LDSM.x2 R{d}, [R{a}]'))
                    else:  # x4 default
                        output.append(SassInstr(encode_ldsm_x4(d, a),
                                                f'LDSM.x4 R{d}, [R{a}]'))

                elif op == 'redux' and 'sync' in instr.types:
                    _types_set = set(instr.types)
                    # redux.sync.add.s32 dest, src, mask
                    # REDUX writes to a UR; MOV R, UR copies result to GPR.
                    d_nm = instr.dest.name if instr.dest and hasattr(instr.dest, 'name') else None
                    s_nm = (instr.srcs[0].name if instr.srcs and hasattr(instr.srcs[0], 'name')
                            else None)
                    d = ctx.ra.r32(d_nm) if d_nm else RZ
                    a = ctx.ra.r32(s_nm) if s_nm else RZ
                    # Allocate a UR temp for the REDUX result
                    ur_tmp = ctx._next_ur if ctx else 6
                    if ctx:
                        ctx._next_ur += 1
                    if 'min' in _types_set and 's32' in _types_set:
                        output.append(SassInstr(encode_redux_min_s32(ur_tmp, a),
                                                f'REDUX.MIN.S32 UR{ur_tmp}, R{a}'))
                    elif 'max' in _types_set and 's32' in _types_set:
                        output.append(SassInstr(encode_redux_max_s32(ur_tmp, a),
                                                f'REDUX.MAX.S32 UR{ur_tmp}, R{a}'))
                    elif 'and' in _types_set:
                        output.append(SassInstr(encode_redux_and_b32(ur_tmp, a),
                                                f'REDUX.AND.B32 UR{ur_tmp}, R{a}'))
                    elif 'or' in _types_set:
                        output.append(SassInstr(encode_redux_or_b32(ur_tmp, a),
                                                f'REDUX.OR.B32 UR{ur_tmp}, R{a}'))
                    elif 'xor' in _types_set:
                        output.append(SassInstr(encode_redux_xor_b32(ur_tmp, a),
                                                f'REDUX.XOR.B32 UR{ur_tmp}, R{a}'))
                    elif 'add' in _types_set and 'u32' in _types_set:
                        output.append(SassInstr(encode_redux_sum(ur_tmp, a),
                                                f'REDUX.SUM UR{ur_tmp}, R{a}'))
                    else:
                        # Default: signed sum (redux.sync.add.s32 or untyped)
                        output.append(SassInstr(encode_redux_sum_s32(ur_tmp, a),
                                                f'REDUX.SUM.S32 UR{ur_tmp}, R{a}'))
                    # Copy UR result to GPR dest (matches ptxas MOV R, UR pattern)
                    if d_nm:
                        output.append(SassInstr(encode_mov_gpr_from_ur(d, ur_tmp),
                                                f'MOV R{d}, UR{ur_tmp}  // redux result'))

                elif op == 'ld' and 'param' in instr.types:
                    output.extend(_select_ld_param(instr, ctx.ra, ctx.param_offsets, ctx))

                elif op == 'ld' and 'global' in instr.types:
                    output.extend(_select_ld_global(instr, ctx.ra, ctx.ur_desc, ctx))

                elif op == 'st' and 'global' in instr.types:
                    output.extend(_select_st_global(instr, ctx.ra, ctx.ur_desc, ctx))

                elif op == 'atom' and 'cas' in instr.types and 'b32' in instr.types:
                    output.extend(_select_atom_cas(instr, ctx.ra, ctx))

                elif op == 'atom' and 'add' in instr.types and 'u32' in instr.types:
                    output.extend(_select_atom_add_u32(instr, ctx.ra, ctx))

                elif op == 'atom' and 'add' in instr.types and 's32' in instr.types:
                    # s32 add is bitwise-identical to u32 add — same ATOMG encoding
                    output.extend(_select_atom_add_u32(instr, ctx.ra, ctx))

                elif op == 'atom' and 'exch' in instr.types and 'b32' in instr.types:
                    output.extend(_select_atom_generic_u32(instr, ctx.ra, ctx, ATOMG_EXCH, 'EXCH'))

                elif op == 'atom' and 'min' in instr.types and 's32' in instr.types:
                    output.extend(_select_atom_generic_u32(instr, ctx.ra, ctx, ATOMG_MIN, 'MIN.S32'))

                elif op == 'atom' and 'max' in instr.types and 's32' in instr.types:
                    output.extend(_select_atom_generic_u32(instr, ctx.ra, ctx, ATOMG_MAX, 'MAX.S32'))

                elif op == 'atom' and 'add' in instr.types and 'f32' in instr.types:
                    output.extend(_select_atom_add_f32(instr, ctx.ra, ctx))

                elif op == 'atom' and 'or' in instr.types and 'b32' in instr.types:
                    output.extend(_select_atom_generic_u32(instr, ctx.ra, ctx, ATOMG_OR, 'OR.b32'))

                elif op == 'atom' and 'and' in instr.types and 'b32' in instr.types:
                    output.extend(_select_atom_generic_u32(instr, ctx.ra, ctx, ATOMG_AND, 'AND.b32'))

                elif op == 'atom' and 'xor' in instr.types and 'b32' in instr.types:
                    # P3-3: atom.xor via 0x98e with UR-indexed data pipeline.
                    # Full pipeline: S2UR→UIADD(opt)→UMOV→sync→ATOMG_XOR
                    from sass.encoding.sm_120_opcodes import (
                        encode_atomg_xor_u32, encode_uiadd_imm,
                        encode_umov_gpr_to_ur)
                    from ptx.ir import MemOp as _MemOp
                    _xd = ctx.ra.r32(instr.dest.name) if isinstance(instr.dest, RegOp) else RZ
                    _xa = instr.srcs[0]
                    _xv = instr.srcs[1]

                    if isinstance(_xa, _MemOp):
                        # Resolve address (same as other atomics)
                        _xbn = _xa.base if _xa.base.startswith('%') else f'%{_xa.base}'
                        _xgw = getattr(ctx, '_gpr_written', set())
                        _xpre = []
                        if _xbn in _xgw and _xa.base in ctx.ra.int_regs:
                            _xaddr = ctx.ra.lo(_xa.base)
                        elif _xbn in getattr(ctx, '_ur_params', {}):
                            _xur = ctx._ur_params[_xbn]
                            _xaddr = getattr(ctx, '_addr_scratch_lo', None) or _alloc_gpr_pair(ctx)
                            _xpre = list(_emit_ur_to_gpr(_xaddr, _xur, 'atom.xor addr'))
                        else:
                            _xaddr = ctx.ra.lo(_xa.base) if _xa.base in ctx.ra.int_regs else RZ
                        output.extend(_xpre)

                        # Route data to UR via S2UR + UMOV pipeline.
                        # Check if data is tid.x (directly from S2R).
                        _data_ur = 5  # target UR for ATOMG data
                        # P3-7: ALL S2UR + activation goes in preamble (pipeline.py).
                        # Isel ONLY emits UMOV + sync + ATOMG.
                        _xdata_name = _xv.name if isinstance(_xv, RegOp) else None
                        _xsr = ctx._reg_sr_source.get(_xdata_name) if _xdata_name else None
                        _sr_for_preamble = _xsr
                        _add_for_preamble = 0
                        if _sr_for_preamble is None and _xdata_name:
                            for _bi in range(_instr_idx - 1, max(_instr_idx - 5, -1), -1):
                                _bdef = bb.instructions[_bi]
                                if (isinstance(_bdef.dest, RegOp) and _bdef.dest.name == _xdata_name
                                        and _bdef.op == 'add' and len(_bdef.srcs) >= 2):
                                    _s0, _s1 = _bdef.srcs[0], _bdef.srcs[1]
                                    if isinstance(_s0, RegOp) and isinstance(_s1, ImmOp):
                                        _sr_for_preamble = ctx._reg_sr_source.get(_s0.name)
                                        _add_for_preamble = _s1.value & 0xFFFFFFFF
                                    elif isinstance(_s1, RegOp) and isinstance(_s0, ImmOp):
                                        _sr_for_preamble = ctx._reg_sr_source.get(_s1.name)
                                        _add_for_preamble = _s0.value & 0xFFFFFFFF
                                    break
                        if _sr_for_preamble is None:
                            _sr_for_preamble = SR_TID_X
                        if not hasattr(ctx, '_ur_activation_sr'):
                            ctx._ur_activation_sr = _sr_for_preamble
                            ctx._ur_activation_add = _add_for_preamble
                        # UMOV is in preamble (P3-7). Only emit sync + ATOMG here.
                        # Sync/commit (0xc02 MOV R5, UR5)
                        output.append(SassInstr(encode_mov_gpr_from_ur(5, _data_ur),
                            f'MOV R5, UR{_data_ur}  // atom.xor: sync'))

                        # ATOMG_XOR with UR data
                        output.append(SassInstr(
                            encode_atomg_xor_u32(_xd, _xaddr, 0, _data_ur, ur_desc=ctx.ur_desc),
                            f'ATOMG.E.XOR R{_xd}, desc[UR{ctx.ur_desc}][R{_xaddr}.64], UR{_data_ur}'))

                elif op == 'atom' and 'min' in instr.types and 'u32' in instr.types:
                    output.extend(_select_atom_generic_u32(instr, ctx.ra, ctx, ATOMG_MIN, 'MIN.u32'))

                elif op == 'atom' and 'max' in instr.types and 'u32' in instr.types:
                    output.extend(_select_atom_generic_u32(instr, ctx.ra, ctx, ATOMG_MAX, 'MAX.u32'))

                elif op == 'atom' and 'cas' in instr.types and 'b64' in instr.types:
                    output.extend(_select_atom_cas_b64(instr, ctx.ra, ctx))

                elif op == 'membar':
                    if 'gl' in instr.types:
                        output.append(SassInstr(encode_membar(MEMBAR_GPU),
                                                'MEMBAR.SC.GPU  // membar.gl'))
                    elif 'cta' in instr.types:
                        output.append(SassInstr(encode_membar(MEMBAR_CTA),
                                                'MEMBAR.SC.CTA  // membar.cta'))
                    else:
                        # Default to GPU scope
                        output.append(SassInstr(encode_membar(MEMBAR_GPU),
                                                'MEMBAR.SC.GPU  // membar (default)'))

                elif op == 'cp' and 'async' in instr.types:
                    from ptx.ir import MemOp
                    if 'commit_group' in instr.types:
                        # cp.async.commit_group → LDGDEPBAR
                        output.append(SassInstr(encode_ldgdepbar(),
                                                'LDGDEPBAR  // cp.async.commit_group'))
                    elif 'wait_group' in instr.types:
                        # cp.async.wait_group N → DEPBAR.LE SB0, N
                        count = 0
                        if instr.srcs and isinstance(instr.srcs[0], ImmOp):
                            count = instr.srcs[0].value
                        output.append(SassInstr(encode_depbar_le(sb=0, count=count),
                                                f'DEPBAR.LE SB0, {count}  // cp.async.wait_group {count}'))
                    elif 'ca' in instr.types and 'shared' in instr.types and 'global' in instr.types:
                        # cp.async.ca.shared.global [smem], [gmem], size
                        # srcs[0] = MemOp (shared dest), srcs[1] = MemOp (global src), srcs[2] = ImmOp (size)
                        smem_op = instr.srcs[0]
                        gmem_op = instr.srcs[1]
                        # Get shared memory address register
                        if isinstance(smem_op, MemOp):
                            base = smem_op.base
                            if base.startswith('%') and base in ctx.ra.int_regs:
                                smem_r = ctx.ra.r32(base)
                            else:
                                smem_r = 0
                        else:
                            smem_r = 0
                        # Resolve global address: same logic as _select_ld_global
                        glob_r = RZ
                        if isinstance(gmem_op, MemOp):
                            gbase = gmem_op.base
                            gbase_n = gbase if gbase.startswith('%') else f'%{gbase}'
                            ur_params = getattr(ctx, '_ur_params', {})
                            deferred = getattr(ctx, '_deferred_ur_params', {})
                            gpr_written = getattr(ctx, '_gpr_written', set())
                            if gbase_n in gpr_written and gbase in ctx.ra.int_regs:
                                glob_r = ctx.ra.lo(gbase)
                            elif gbase_n in deferred:
                                param_off = deferred.get(gbase_n)
                                ur_tmp = 6
                                addr = _alloc_gpr_pair(ctx)
                                output.append(SassInstr(encode_ldcu_64(ur_tmp, 0, param_off),
                                    f'LDCU.64 UR{ur_tmp}, c[0][0x{param_off:x}]  // deferred cp.async addr'))
                                output.extend(_emit_ur_to_gpr(addr, ur_tmp, "deferred UR->GPR cp.async"))
                                glob_r = addr
                            elif gbase_n in ur_params:
                                ur_idx = ur_params[gbase_n]
                                addr = getattr(ctx, '_addr_scratch_lo', None)
                                if addr is None:
                                    addr = _alloc_gpr_pair(ctx)
                                output.extend(_emit_ur_to_gpr(addr, ur_idx, "cp.async UR->GPR addr"))
                                glob_r = addr
                            elif gbase in ctx.ra.int_regs:
                                glob_r = ctx.ra.lo(gbase)
                        ldgsts_raw = encode_ldgsts_e(smem_r, glob_r, ctx.ur_desc)
                        # cp.async may be in a conditional fall-through (after @%p bra).
                        # The parser doesn't always if-convert this. Scan backward in the
                        # current block for a conditional BRA and inherit its negated guard.
                        if not instr.pred:
                            for prev_inst in reversed(bb.instructions[:_instr_idx]):
                                if prev_inst.op == 'bra' and prev_inst.pred:
                                    pred_name = prev_inst.pred
                                    pd = ctx.ra.pred(pred_name) if pred_name in ctx.ra.pred_regs else 0
                                    neg = not prev_inst.neg  # fall-through = negated
                                    if hasattr(ctx, '_negated_preds') and pd in ctx._negated_preds:
                                        neg = not neg
                                    ldgsts_raw = patch_pred(ldgsts_raw, pred=pd, neg=neg)
                                    break
                        output.append(SassInstr(ldgsts_raw,
                            f'LDGSTS.E [R{smem_r}], desc[UR{ctx.ur_desc}][R{glob_r}.64]  // cp.async.ca.shared.global'))
                    elif 'bulk' in instr.types:
                        # cp.async.bulk.* — TMA instructions
                        from ptx.ir import MemOp
                        types_set = set(instr.types)
                        if 'commit_group' in types_set:
                            # cp.async.bulk.commit_group → UTMACMDFLUSH
                            output.append(SassInstr(encode_utmacmdflush(),
                                                    'UTMACMDFLUSH  // cp.async.bulk.commit_group'))
                        elif 'wait_group' in types_set:
                            # cp.async.bulk.wait_group N → DEPBAR.LE SB0, N
                            count = 0
                            if instr.srcs and isinstance(instr.srcs[0], ImmOp):
                                count = instr.srcs[0].value
                            output.append(SassInstr(encode_depbar_le(sb=0, count=count),
                                                    f'DEPBAR.LE SB0, {count}  // cp.async.bulk.wait_group {count}'))
                        elif 'tensor' in types_set:
                            # cp.async.bulk.tensor.Nd.shared::cluster.global.tile...
                            # Determine dimension from types
                            dim = 1
                            if '2d' in types_set:
                                dim = 2
                            elif '3d' in types_set:
                                dim = 3
                            # Check direction
                            is_store = False
                            for t in instr.types:
                                # "global" before "shared" = store direction
                                if 'global' in t and 'shared' not in t:
                                    # Check ordering: global.shared::cta = store
                                    idx_g = None
                                    idx_s = None
                                    for i, q in enumerate(instr.types):
                                        if 'global' in q and idx_g is None:
                                            idx_g = i
                                        if 'shared' in q and idx_s is None:
                                            idx_s = i
                                    if idx_g is not None and idx_s is not None and idx_g < idx_s:
                                        is_store = True
                                    break
                            if is_store:
                                # TMA tensor store: uses UTMASTG
                                # Allocate UR pairs for smem addr and descriptor
                                ur_smem = ctx._next_ur; ctx._next_ur += 1
                                ur_desc = ctx._next_ur; ctx._next_ur += 1
                                output.append(SassInstr(encode_utmastg_1d(ur_smem, ur_desc),
                                    f'UTMASTG.{dim}D [UR{ur_smem}], [UR{ur_desc}]  // cp.async.bulk.tensor.{dim}d store'))
                                output.append(SassInstr(encode_utmacmdflush(),
                                    'UTMACMDFLUSH  // TMA store flush'))
                            else:
                                # TMA tensor load: uses UTMALDG
                                ur_smem = ctx._next_ur; ctx._next_ur += 1
                                ur_desc = ctx._next_ur; ctx._next_ur += 1
                                if dim == 1:
                                    output.append(SassInstr(encode_utmaldg_1d(ur_smem, ur_desc),
                                        f'UTMALDG.1D [UR{ur_smem}], [UR{ur_desc}]  // cp.async.bulk.tensor.1d load'))
                                elif dim == 2:
                                    output.append(SassInstr(encode_utmaldg_2d(ur_smem, ur_desc),
                                        f'UTMALDG.2D [UR{ur_smem}], [UR{ur_desc}]  // cp.async.bulk.tensor.2d load'))
                                else:
                                    # 3D+ not yet supported; emit 1D as fallback
                                    output.append(SassInstr(encode_utmaldg_1d(ur_smem, ur_desc),
                                        f'UTMALDG.1D [UR{ur_smem}], [UR{ur_desc}]  // cp.async.bulk.tensor.{dim}d (fallback 1D)'))
                        elif any('shared' in t for t in instr.types) and any('global' in t for t in instr.types):
                            # cp.async.bulk.shared::cluster.global — non-tensor bulk copy
                            # or cp.async.bulk.global.shared::cta — reverse direction
                            is_store = False
                            for i, t in enumerate(instr.types):
                                if 'global' in t:
                                    # If global appears before shared in type list, it's a store
                                    for j, t2 in enumerate(instr.types):
                                        if 'shared' in t2 and j > i:
                                            is_store = True
                                    break
                            ur_dst  = ctx._next_ur; ctx._next_ur += 1
                            ur_src  = ctx._next_ur; ctx._next_ur += 1
                            ur_size = ctx._next_ur; ctx._next_ur += 1
                            if is_store:
                                output.append(SassInstr(encode_ublkcp_g_s(ur_dst, ur_src, ur_size),
                                    f'UBLKCP.G.S [UR{ur_dst}], [UR{ur_src}], UR{ur_size}  // cp.async.bulk global<-shared'))
                                output.append(SassInstr(encode_utmacmdflush(),
                                    'UTMACMDFLUSH  // bulk store flush'))
                            else:
                                output.append(SassInstr(encode_ublkcp_s_g(ur_dst, ur_src, ur_size),
                                    f'UBLKCP.S.G [UR{ur_dst}], [UR{ur_src}], UR{ur_size}  // cp.async.bulk shared<-global'))

                elif op == 'mbarrier':
                    # mbarrier.init / mbarrier.arrive / mbarrier.try_wait
                    types_set = set(instr.types)
                    if 'init' in types_set:
                        # mbarrier.init.shared::cta.b64 [mbar], count
                        ur_mbar  = ctx._next_ur; ctx._next_ur += 1
                        ur_count = ctx._next_ur; ctx._next_ur += 1
                        output.append(SassInstr(encode_syncs_exch_64(ur_mbar, ur_count),
                            f'SYNCS.EXCH.64 URZ, [UR{ur_mbar}], UR{ur_count}  // mbarrier.init'))
                    elif 'arrive' in types_set:
                        # mbarrier.arrive.shared::cta.b64 %rd, [mbar]
                        ur_mbar = ctx._next_ur; ctx._next_ur += 1
                        output.append(SassInstr(encode_syncs_arrive(ur_mbar),
                            f'SYNCS.ARRIVE [UR{ur_mbar}]  // mbarrier.arrive'))
                    elif 'try_wait' in types_set:
                        # mbarrier.try_wait.parity.shared::cta.b64 %p, [mbar], phase
                        ur_mbar = ctx._next_ur; ctx._next_ur += 1
                        # Phase register (R0 typically holds SHF.L.U32 RZ, 0x1f, RZ)
                        r_phase = 0  # default R0
                        output.append(SassInstr(encode_syncs_trywait(ur_mbar, r_phase),
                            f'SYNCS.TRYWAIT PT, [UR{ur_mbar}], R{r_phase}  // mbarrier.try_wait'))

                elif op == 'dp4a':
                    output.extend(_select_dp4a(instr, ctx.ra, ctx))

                elif op == 'bfind' and typ in ('u32',):
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_flo(d, a),
                                            f'FLO.U32 R{d}, R{a}  // bfind.u32'))

                elif op == 'ret':
                    output.append(SassInstr(encode_exit(ctrl=0x7f5), 'EXIT'))
                    if instr.pred:
                        ctx._past_predicated_exit = True

                elif op == 'bra':
                    from ptx.ir import LabelOp
                    target = None
                    if instr.srcs:
                        if isinstance(instr.srcs[0], LabelOp):
                            target = instr.srcs[0].name

                    # Optimization: if @Px bra TARGET and TARGET is a ret-only block,
                    # emit @Px EXIT instead of @Px BRA TARGET. On SM_120 (Blackwell),
                    # this is what ptxas does — predicated EXIT correctly exits
                    # idle threads without requiring reconvergence management.
                    if instr.pred and target:
                        target_is_ret = False
                        for tbb in fn.blocks:
                            if tbb.label == target:
                                if (len(tbb.instructions) == 1
                                        and tbb.instructions[0].op == 'ret'):
                                    target_is_ret = True
                                break
                        if target_is_ret:
                            pd = ctx.ra.pred(instr.pred) if instr.pred in ctx.ra.pred_regs else 0
                            neg = instr.neg
                            if hasattr(ctx, '_negated_preds') and pd in ctx._negated_preds:
                                neg = not neg
                            exit_raw = patch_pred(encode_exit(), pred=pd, neg=neg)
                            pred_str = f'@{"!" if neg else ""}P{pd} '
                            output.append(SassInstr(exit_raw,
                                                    f'{pred_str}EXIT  // early exit (idle threads)'))
                            # Flush deferred params: emit LDCU.64 right after EXIT.
                            # Use UR indices starting at ctx._next_ur (post-EXIT,
                            # UR pressure is lower because exited threads freed slots).
                            deferred = getattr(ctx, '_deferred_ur_params', {})
                            if deferred:
                                first = True
                                for pname, poff in list(deferred.items()):
                                    ur_tmp = ctx._next_ur
                                    if ur_tmp % 2 != 0:
                                        ur_tmp += 1
                                    ctx._next_ur = ur_tmp + 2
                                    raw = bytearray(encode_ldcu_64(ur_tmp, 0, poff))
                                    if first:
                                        raw[9] = 0x0c  # Rule #29: first post-EXIT
                                        first = False
                                    output.append(SassInstr(bytes(raw),
                                        f'LDCU.64 UR{ur_tmp}, c[0][0x{poff:x}]  // post-EXIT deferred'))
                                    ctx._ur_params[pname] = ur_tmp
                                deferred.clear()
                            continue

                    # Unconditional BRA to ret-only block → EXIT
                    # Peephole: if previous was @!Px BRA body, merge into @Px EXIT
                    if not instr.pred and target:
                        _tgt_is_ret = False
                        for tbb in fn.blocks:
                            if tbb.label == target:
                                if (len(tbb.instructions) == 1
                                        and tbb.instructions[0].op == 'ret'):
                                    _tgt_is_ret = True
                                break
                        if _tgt_is_ret:
                            # Check if previous instruction was @!Px BRA (negated predicated BRA)
                            if output and output[-1].raw[0] == 0x47:  # BRA opcode low byte
                                prev_guard = (output[-1].raw[1] >> 4) & 0xF
                                prev_neg = (output[-1].raw[1] >> 3) & 1
                                if prev_guard != 0x7 and prev_neg:
                                    # @!Px BRA body; bra exit → @Px EXIT
                                    # Remove the @!Px BRA
                                    prev_pred = prev_guard & 0x7
                                    output.pop()
                                    # Remove BRA fixup for the removed instruction
                                    if hasattr(ctx, '_bra_fixups') and ctx._bra_fixups:
                                        ctx._bra_fixups = ctx._bra_fixups[:-1]
                                    # Emit @Px EXIT (non-negated predicate)
                                    exit_raw = patch_pred(encode_exit(), pred=prev_pred, neg=False)
                                    output.append(SassInstr(exit_raw,
                                                            f'@P{prev_pred} EXIT  // bounds check'))
                                    # Flush deferred params right after EXIT
                                    deferred = getattr(ctx, '_deferred_ur_params', {})
                                    if deferred:
                                        first_d = True
                                        for dpname, dpoff in list(deferred.items()):
                                            dur = ctx._next_ur
                                            if dur % 2 != 0: dur += 1
                                            ctx._next_ur = dur + 2
                                            draw = bytearray(encode_ldcu_64(dur, 0, dpoff))
                                            if first_d:
                                                draw[9] = 0x0c  # Rule #29
                                                first_d = False
                                            output.append(SassInstr(bytes(draw),
                                                f'LDCU.64 UR{dur}, c[0][0x{dpoff:x}]  // post-EXIT'))
                                            ctx._ur_params[dpname] = dur
                                        deferred.clear()
                                    continue
                            # No peephole match — emit plain EXIT
                            output.append(SassInstr(encode_exit(),
                                                    f'EXIT  // unconditional return'))
                            continue

                    # General BRA with offset fixup
                    bra_idx = len(output)
                    if ctx.sm_version == 89:
                        from sass.encoding.sm_89_opcodes import encode_bra as _sm89_bra
                        bra_raw = _sm89_bra(0)
                    else:
                        bra_raw = encode_bra(0)
                    if instr.pred:
                        pd = ctx.ra.pred(instr.pred) if instr.pred in ctx.ra.pred_regs else 0
                        # Check if the predicate was negated by the setp handler
                        # (e.g., setp.lt emits GE + negate)
                        neg = instr.neg
                        if hasattr(ctx, '_negated_preds') and pd in ctx._negated_preds:
                            neg = not neg  # flip the negation
                        bra_raw = patch_pred(bra_raw, pred=pd, neg=neg)
                        pred_str = f'@{"!" if neg else ""}P{pd} '
                    else:
                        pred_str = ''
                    output.append(SassInstr(bra_raw,
                                            f'{pred_str}BRA {target or "?"}'))
                    if target:
                        if not hasattr(ctx, '_bra_fixups'):
                            ctx._bra_fixups = []
                        ctx._bra_fixups.append((bra_idx, target))

                elif op == 'nop':
                    output.append(_nop())

                elif op == 'cvt':
                    # Type conversion — handle widening to 64-bit
                    # CSE: if same source was already converted, reuse the result
                    d = instr.dest
                    s = instr.srcs[0]
                    if isinstance(d, RegOp) and isinstance(s, RegOp):
                        _types_set = set(instr.types)
                        _is_64_dst = any(t in ('u64','s64','b64') for t in instr.types[:1])
                        # Only zero-extend from unsigned 32-bit; signed widening needs
                        # SHF.R.S32.HI (no encoder yet) for correct sign extension.
                        _is_32_src = any(t in ('u32','b32') for t in instr.types[1:])
                        if _is_64_dst and _is_32_src:
                            s_r = ctx.ra.r32(s.name)
                            # CSE: check if we already converted this source register
                            if not hasattr(ctx, '_cvt_cache'):
                                ctx._cvt_cache = {}
                            if s.name in ctx._cvt_cache:
                                # Source was already widened once. The cached physical
                                # destination may have been overwritten by now (e.g. by
                                # a DADD that reused the same register slot). Re-emit
                                # the widening into d's own allocated physical register
                                # using the original 32-bit source (s_r still holds r9).
                                d_lo = ctx.ra.lo(d.name)
                                if d_lo != s_r:
                                    output.append(SassInstr(
                                        encode_iadd3(d_lo, s_r, RZ, RZ),
                                        f'MOV R{d_lo}, R{s_r}  // cvt.64.32 lo (CSE src)'))
                                if not hasattr(ctx, '_zero_regs'):
                                    ctx._zero_regs = set()
                                ctx._zero_regs.add(d_lo+1)
                                output.append(SassInstr(
                                    encode_iadd3(d_lo+1, RZ, RZ, RZ),
                                    f'MOV R{d_lo+1}, RZ  // cvt.64.32 hi=0 (CSE)'))
                                # Record for IMAD.WIDE fusion (CSE path)
                                if not hasattr(ctx, '_cvt_src_map'):
                                    ctx._cvt_src_map = {}
                                ctx._cvt_src_map[d.name] = s_r
                                continue
                            d_lo = ctx.ra.lo(d.name)
                            ctx._cvt_cache[s.name] = d_lo
                            if not hasattr(ctx, '_zero_regs'):
                                ctx._zero_regs = set()
                            ctx._zero_regs.add(d_lo+1)
                            # Record 32-bit source for IMAD.WIDE fusion
                            if not hasattr(ctx, '_cvt_src_map'):
                                ctx._cvt_src_map = {}
                            ctx._cvt_src_map[d.name] = s_r
                            if d_lo != s_r:
                                output.append(SassInstr(encode_iadd3(d_lo, s_r, RZ, RZ),
                                                        f'MOV R{d_lo}, R{s_r}  // cvt.64.32 lo'))
                            output.append(SassInstr(encode_iadd3(d_lo+1, RZ, RZ, RZ),
                                                    f'MOV R{d_lo+1}, RZ  // cvt.64.32 hi=0'))
                        elif _is_64_dst and any(t == 's32' for t in instr.types[1:]):
                            # Sign-extend s32 → s64/u64/b64
                            # SHF.R.U32.HI d_hi, RZ, 31, s_r → d_hi = 0 or 1
                            # INEG d_hi, d_hi               → d_hi = 0 or 0xFFFFFFFF
                            # MOV  d_lo, s_r                → lo word
                            s_r = ctx.ra.r32(s.name)
                            d_lo = ctx.ra.lo(d.name)
                            # Always use the regalloc's assignment for d_lo.
                            # Previous code aliased d_lo=s_r when s_r was even,
                            # but this mutated int_regs after allocation, causing
                            # later register conflicts (e.g., %f regs overlapping
                            # the aliased %rd pair). Emit a MOV when needed.
                            if d_lo != s_r:
                                output.append(SassInstr(encode_iadd3(d_lo, s_r, RZ, RZ),
                                                        f'MOV R{d_lo}, R{s_r}  // cvt.s64.s32 lo'))
                                s_r = d_lo  # sign-extend from the copy
                            d_hi = d_lo + 1
                            output.append(SassInstr(
                                encode_shf_r_s32_hi(d_hi, s_r, 31),
                                f'SHF.R.S32.HI R{d_hi}, RZ, 0x1f, R{s_r}  // cvt.s64.s32 sign'))
                        else:
                            # General 32-bit and float conversions
                            _ROUNDING = {'rn','rz','rm','rp','rni','rzi','rmi','rpi'}
                            _core = [t for t in instr.types if t not in _ROUNDING]
                            _dst_t = _core[0] if _core else 'u32'
                            _src_t = _core[1] if len(_core) > 1 else 'u32'
                            _32B = {'u32', 's32', 'b32', 'f32'}
                            _64B = {'u64', 's64', 'b64', 'f64'}
                            if _dst_t == 'f32' and _src_t == 'f64':
                                # cvt.rn.f32.f64: double-precision → single-precision
                                d_r  = ctx.ra.r32(d.name)
                                a_lo = ctx.ra.lo(s.name)
                                output.append(SassInstr(encode_f2f_f32_f64(d_r, a_lo),
                                                        f'F2F.F32.F64 R{d_r}, R{a_lo}'))
                            elif _dst_t == 'f64' and _src_t == 'f32':
                                # cvt.f64.f32: single-precision → double-precision
                                d_lo = ctx.ra.lo(d.name)
                                a_r  = ctx.ra.r32(s.name)
                                output.append(SassInstr(encode_f2f_f64_f32(d_lo, a_r),
                                                        f'F2F.F64.F32 R{d_lo}, R{a_r}'))
                            elif _dst_t == 's32' and _src_t == 'f64':
                                # cvt.rzi.s32.f64: double → signed int32
                                d_r  = ctx.ra.r32(d.name)
                                a_lo = ctx.ra.lo(s.name)
                                output.append(SassInstr(encode_f2i_s32_f64(d_r, a_lo),
                                                        f'F2I.S32.F64 R{d_r}, R{a_lo}'))
                            elif _dst_t == 'u32' and _src_t == 'f64':
                                # cvt.rzi.u32.f64: double → unsigned int32
                                d_r  = ctx.ra.r32(d.name)
                                a_lo = ctx.ra.lo(s.name)
                                output.append(SassInstr(encode_f2i_u32_f64(d_r, a_lo),
                                                        f'F2I.U32.F64 R{d_r}, R{a_lo}'))
                            elif _dst_t == 'f64' and _src_t == 's32':
                                # cvt.rn.f64.s32: signed int32 → double
                                d_lo = ctx.ra.lo(d.name)
                                a_r  = ctx.ra.r32(s.name)
                                output.append(SassInstr(encode_i2f_f64_s32(d_lo, a_r),
                                                        f'I2F.F64.S32 R{d_lo}, R{a_r}'))
                            elif _dst_t == 'f64' and _src_t == 'u32':
                                # cvt.rn.f64.u32: unsigned int32 → double
                                d_lo = ctx.ra.lo(d.name)
                                a_r  = ctx.ra.r32(s.name)
                                output.append(SassInstr(encode_i2f_f64_u32(d_lo, a_r),
                                                        f'I2F.F64.U32 R{d_lo}, R{a_r}'))
                            elif _dst_t == 'f16' and _src_t == 'f32':
                                # cvt.rn.f16.f32: FP32 → FP16 (packed into low 16 bits)
                                d_r = ctx.ra.r32(d.name)
                                a_r = ctx.ra.r32(s.name)
                                output.append(SassInstr(encode_cvt_f16_f32(d_r, a_r),
                                                        f'CVT.F16.F32 R{d_r}, R{a_r}  // cvt.f16.f32'))
                            elif 'f32' in _types_set and ('u32' in _types_set or 's32' in _types_set):
                                d_r = ctx.ra.r32(d.name)
                                a_r = ctx.ra.r32(s.name)
                                _fi = instr.types.index('f32')
                                _ii = (instr.types.index('u32') if 'u32' in instr.types
                                       else instr.types.index('s32'))
                                _is_signed = 's32' in _types_set
                                if _fi < _ii:
                                    # int → float
                                    if _is_signed:
                                        output.append(SassInstr(encode_i2f_f32_s32(d_r, a_r),
                                                                f'I2FP.F32.S32 R{d_r}, R{a_r}  // cvt.f32.s32'))
                                    else:
                                        output.append(SassInstr(encode_i2fp_u32(d_r, a_r),
                                                                f'I2FP.F32.U32 R{d_r}, R{a_r}  // cvt.f32.u32'))
                                else:
                                    # float → int
                                    if _is_signed:
                                        output.append(SassInstr(encode_f2i_s32_f32(d_r, a_r),
                                                                f'F2I.S32 R{d_r}, R{a_r}  // cvt.s32.f32'))
                                    else:
                                        output.append(SassInstr(encode_f2i_u32(d_r, a_r),
                                                                f'F2I.U32 R{d_r}, R{a_r}  // cvt.u32.f32'))
                            elif _dst_t in ('u8', 's8', 'b8') and _src_t in _32B:
                                # Truncate to 8 bits: AND with 0xFF
                                d_r = ctx.ra.r32(d.name)
                                a_r = ctx.ra.r32(s.name)
                                lit_off = ctx._alloc_literal(0xFF)
                                t = _alloc_gpr(ctx)
                                output.append(SassInstr(encode_ldc(t, 0, lit_off),
                                                        f'LDC R{t}, c[0][0x{lit_off:x}]  // 0xFF mask'))
                                _emit_lop3(output, ctx, d_r, a_r, t, RZ, LOP3_AND, f'LOP3.AND R{d_r}, R{a_r}, R{t}, RZ  // cvt.{_dst_t}.{_src_t}')
                            elif _dst_t in ('u16', 's16', 'b16') and _src_t in _32B:
                                # Truncate to 16 bits: AND with 0xFFFF
                                d_r = ctx.ra.r32(d.name)
                                a_r = ctx.ra.r32(s.name)
                                lit_off = ctx._alloc_literal(0xFFFF)
                                t = _alloc_gpr(ctx)
                                output.append(SassInstr(encode_ldc(t, 0, lit_off),
                                                        f'LDC R{t}, c[0][0x{lit_off:x}]  // 0xFFFF mask'))
                                _emit_lop3(output, ctx, d_r, a_r, t, RZ, LOP3_AND, f'LOP3.AND R{d_r}, R{a_r}, R{t}, RZ  // cvt.{_dst_t}.{_src_t}')
                            elif _dst_t in _32B and _src_t in ('u8', 's8', 'b8', 'u16', 's16', 'b16'):
                                # Widening from narrow: just copy (narrow stored as u32, already zero-extended)
                                d_r = ctx.ra.r32(d.name)
                                a_r = ctx.ra.r32(s.name)
                                if d_r != a_r:
                                    output.append(SassInstr(encode_iadd3(d_r, a_r, RZ, RZ),
                                                            f'MOV R{d_r}, R{a_r}  // cvt.{_dst_t}.{_src_t}'))
                            elif _dst_t in _32B and _src_t in _32B:
                                # Same-width int conversion (s32↔u32, etc.) — alias to same register
                                a_r = ctx.ra.r32(s.name)
                                ctx.ra.int_regs[d.name] = a_r  # alias output to input
                                d_r = a_r
                                if d_r != a_r:  # always false now, but keep for safety
                                    output.append(SassInstr(encode_iadd3(d_r, a_r, RZ, RZ),
                                                            f'MOV R{d_r}, R{a_r}  // cvt.{_dst_t}.{_src_t}'))
                                else:
                                    output.append(_nop(f'cvt.{_dst_t}.{_src_t} nop (d==a)'))
                            elif _dst_t in _32B and _src_t in _64B:
                                d_r = ctx.ra.r32(d.name)
                                a_lo = ctx.ra.lo(s.name)
                                if d_r != a_lo:
                                    output.append(SassInstr(encode_iadd3(d_r, a_lo, RZ, RZ),
                                                            f'MOV R{d_r}, R{a_lo}  // cvt.{_dst_t}.{_src_t} trunc'))
                                else:
                                    output.append(_nop(f'cvt.{_dst_t}.{_src_t} nop (d==a_lo)'))
                            elif _dst_t in _64B and _src_t in _64B:
                                # 64-bit reinterpret (u64↔s64, b64↔u64, etc.) — identity copy
                                d_lo = ctx.ra.lo(d.name)
                                a_lo = ctx.ra.lo(s.name)
                                if d_lo != a_lo:
                                    output.append(SassInstr(encode_iadd3(d_lo, a_lo, RZ, RZ),
                                                            f'MOV R{d_lo}, R{a_lo}  // cvt.{_dst_t}.{_src_t} lo'))
                                    output.append(SassInstr(encode_iadd3(d_lo+1, a_lo+1, RZ, RZ),
                                                            f'MOV R{d_lo+1}, R{a_lo+1}  // cvt.{_dst_t}.{_src_t} hi'))
                                # else: same register, nothing to do (NOP omitted)
                            elif _dst_t in _64B and _src_t in _32B:
                                # 32→64 widening: zero-extend (u64.u32/b64.b32) or sign-extend (s64.s32)
                                d_lo = ctx.ra.lo(d.name)
                                a_r  = ctx.ra.r32(s.name)
                                # lo = src
                                if d_lo != a_r:
                                    output.append(SassInstr(encode_iadd3(d_lo, a_r, RZ, RZ),
                                                            f'MOV R{d_lo}, R{a_r}  // cvt.{_dst_t}.{_src_t} lo'))
                                # hi = sign extension (s32) or 0 (u32/b32)
                                if _src_t == 's32' and _dst_t == 's64':
                                    # encode_shf_r_s32_hi already imported at module level
                                    output.append(SassInstr(
                                        encode_shf_r_s32_hi(d_lo+1, a_r, 31),
                                        f'SHF.R.S32.HI R{d_lo+1}, RZ, 31, R{a_r}  // cvt.s64.s32 sign'))
                                else:
                                    output.append(SassInstr(encode_iadd3(d_lo+1, RZ, RZ, RZ),
                                                            f'MOV R{d_lo+1}, RZ  // cvt.{_dst_t}.{_src_t} zero-ext'))
                            else:
                                # Unsupported cvt type combo — no encoder available.
                                # Known gaps: f64↔s64, f64↔u64 (no SASS encoder),
                                # f16↔f32 (use F2FP path), narrow↔narrow (unusual).
                                import sys as _sys
                                print(f'WARNING: unimplemented cvt type combination: cvt.{".".join(instr.types)}',
                                      file=_sys.stderr)
                                output.append(_nop(f'WARNING: unimplemented cvt {".".join(instr.types)}'))

                elif op == 'setp':
                    pred = instr.dest
                    a    = instr.srcs[0]
                    b    = instr.srcs[1]
                    if isinstance(pred, RegOp) and isinstance(a, RegOp):
                        pd = ctx.ra.pred(pred.name) if pred.name in ctx.ra.pred_regs else 0
                        ar = ctx.ra.r32(a.name)
                        is_f64  = 'f64' in instr.types
                        is_float = is_f64 or 'f32' in instr.types
                        cmp_name = next((t for t in instr.types if t in ('lt','le','gt','ge','eq','ne')), 'ge')
                        if is_f64:
                            # FP64 comparison: emit DSETP using register pairs.
                            # SM_120 DSETP only reliably supports unordered comparison
                            # codes; ordered codes (LT=1..GE=6) give wrong results.
                            # ptxas ground truth: setp.lt.f64 → DSETP.GEU (unordered
                            # complement) + predicate marked as negated so @P → @!P.
                            # We use unordered complements for all ordered comparisons:
                            #   NOT(ordered LT) = unordered GEU, etc.
                            ar64 = ctx.ra.lo(a.name)
                            cmp_map64 = {
                                'lt': DSETP_GEU, 'le': DSETP_GTU,
                                'gt': DSETP_LEU, 'ge': DSETP_LTU,
                                'eq': DSETP_NEU, 'ne': DSETP_EQU,
                            }
                            if isinstance(b, ImmOp):
                                # Materialize FP64 immediate as a register pair
                                imm_bits = b.value & 0xFFFFFFFF
                                br_lo = _alloc_gpr(ctx)
                                br_hi = _alloc_gpr(ctx)
                                output.append(SassInstr(encode_iadd3_imm32(br_lo, RZ, 0, RZ),
                                    f'IADD3 R{br_lo}, RZ, 0, RZ  // dsetp imm lo'))
                                output.append(SassInstr(encode_iadd3_imm32(br_hi, RZ, imm_bits, RZ),
                                    f'IADD3 R{br_hi}, RZ, 0x{imm_bits:x}, RZ  // dsetp imm hi'))
                                br_lo64 = br_lo
                            elif isinstance(b, RegOp):
                                br_lo64 = ctx.ra.lo(b.name)
                            else:
                                br_lo64 = RZ
                            dsetp_cmp = cmp_map64.get(cmp_name, DSETP_GEU)
                            # Emit the complemented comparison; mark pred as negated
                            # so @P guards become @!P (matching ptxas semantics).
                            cmp_label = {DSETP_GEU:'GEU', DSETP_GTU:'GTU',
                                         DSETP_LEU:'LEU', DSETP_LTU:'LTU',
                                         DSETP_NEU:'NEU', DSETP_EQU:'EQU'}.get(dsetp_cmp, 'GEU')
                            output.append(SassInstr(
                                encode_dsetp(pd, ar64, br_lo64, dsetp_cmp),
                                f'DSETP.{cmp_label} P{pd}, R{ar64}, R{br_lo64}  // setp.{cmp_name}.f64'))
                            if not hasattr(ctx, '_negated_preds'):
                                ctx._negated_preds = set()
                            ctx._negated_preds.add(pd)
                        elif is_float:
                            # PEEPHOLE: check if next 2 instructions are @p mov.f32 imm + @!p mov.f32 imm
                            # with values 1.0 and 0.0 (step function). If so, fuse into FSEL.step.
                            # This avoids the SM_120 bug where ISETP corrupts FSETP state.
                            from sass.encoding.sm_120_opcodes import encode_fsel_step, FSEL_GT, FSEL_LT, FSEL_GE, FSEL_LE, FSEL_EQ, FSEL_NE
                            _fsel_cmp = {'lt': FSEL_LT, 'le': FSEL_LE, 'gt': FSEL_GT,
                                         'ge': FSEL_GE, 'eq': FSEL_EQ, 'ne': FSEL_NE}
                            remaining = bb.instructions[_instr_idx+1:]
                            can_fsel = False
                            if (len(remaining) >= 2
                                and remaining[0].op == 'mov' and remaining[0].pred == pred.name
                                and remaining[1].op == 'mov' and remaining[1].pred == pred.name
                                and isinstance(remaining[0].srcs[0], ImmOp)
                                and isinstance(remaining[1].srcs[0], ImmOp)):
                                v_true = remaining[0].srcs[0].value & 0xFFFFFFFF
                                v_false = remaining[1].srcs[0].value & 0xFFFFFFFF
                                neg0 = remaining[0].neg
                                neg1 = remaining[1].neg
                                # @p mov true_val + @!p mov false_val (or reversed negation)
                                if (not neg0 and neg1 and v_true == 0x3F800000 and v_false == 0):
                                    can_fsel = True
                                elif (neg0 and not neg1 and v_true == 0 and v_false == 0x3F800000):
                                    can_fsel = True
                            if can_fsel and isinstance(b, ImmOp):
                                # FSEL.step: dest = (src cmp threshold) ? 1.0 : 0.0
                                threshold = b.value & 0xFFFFFFFF
                                dest_name = remaining[0].dest.name
                                d = ctx.ra.r32(dest_name)
                                output.append(SassInstr(
                                    encode_fsel_step(d, ar, threshold, _fsel_cmp.get(cmp_name, FSEL_GT)),
                                    f'FSEL.step R{d}, R{ar}, 0x{threshold:08x}, {cmp_name.upper()}'))
                                # Skip the next 2 instructions (predicated movs)
                                if not hasattr(ctx, '_skip_instrs'):
                                    ctx._skip_instrs = set()
                                ctx._skip_instrs.add(id(remaining[0]))
                                ctx._skip_instrs.add(id(remaining[1]))
                            # PEEPHOLE 2: setp.gt.f32 + selp.f32 → FSETP + FSEL.imm
                            # When the ONLY consumer of the predicate is selp.f32
                            # (no branch), we can use FSETP directly because FSETP
                            # predicates work for data-path consumers (SEL/FSEL).
                            # This matches ptxas's pattern: FSETP + FSEL.imm = 2 instrs
                            # vs FSEL.step + ISETP.NE + MOV + MOV + SEL = 5 instrs.
                            elif (not can_fsel and len(remaining) >= 1
                                  and remaining[0].op == 'selp'
                                  and remaining[0].srcs[2].name == pred.name
                                  and isinstance(remaining[0].srcs[0], ImmOp)
                                  and isinstance(remaining[0].srcs[1], ImmOp)):
                                from sass.encoding.sm_120_opcodes import encode_fsel_imm
                                _fsetp_cmp = {'lt': FSETP_LT, 'le': FSETP_LE,
                                              'gt': FSETP_GT, 'ge': FSETP_GE,
                                              'eq': FSETP_EQ, 'ne': FSETP_NE}
                                fsetp_c = _fsetp_cmp.get(cmp_name, FSETP_GT)

                                # Materialize threshold (if immediate)
                                if isinstance(b, ImmOp):
                                    br = _alloc_gpr(ctx)
                                    imm_val = b.value & 0xFFFFFFFF
                                    output.append(SassInstr(
                                        encode_iadd3_imm32(br, RZ, imm_val, RZ),
                                        f'IADD3 R{br}, RZ, 0x{imm_val:08x}, RZ  // fsetp threshold'))
                                elif isinstance(b, RegOp):
                                    br = ctx.ra.r32(b.name)
                                else:
                                    br = RZ

                                # FSETP: write predicate (data-path only, safe for FSEL)
                                output.append(SassInstr(
                                    encode_fsetp(pd, ar, br, cmp=fsetp_c),
                                    f'FSETP.{cmp_name.upper()} P{pd}, R{ar}, R{br}'))

                                # Fuse selp.f32 into IADD3(true_val) + FSEL.imm(false_val)
                                selp_instr = remaining[0]
                                true_val = selp_instr.srcs[0].value & 0xFFFFFFFF
                                false_val = selp_instr.srcs[1].value & 0xFFFFFFFF
                                d = ctx.ra.r32(selp_instr.dest.name)

                                # Load true_val into dest, then FSEL.imm selects
                                # between dest (when pred TRUE) and false_val (when FALSE)
                                output.append(SassInstr(
                                    encode_iadd3_imm32(d, RZ, true_val, RZ),
                                    f'IADD3 R{d}, RZ, 0x{true_val:08x}, RZ  // selp true'))
                                output.append(SassInstr(
                                    encode_fsel_imm(d, d, false_val, pred=pd),
                                    f'FSEL.imm R{d}, R{d}, 0x{false_val:08x}, P{pd}'))

                                # Skip the selp instruction (already fused)
                                if not hasattr(ctx, '_skip_instrs'):
                                    ctx._skip_instrs = set()
                                ctx._skip_instrs.add(id(selp_instr))
                                # Clear negated_preds (FSETP uses natural sense)
                                if hasattr(ctx, '_negated_preds'):
                                    ctx._negated_preds.discard(pd)
                            else:
                                # SM_120 FSETP GUARD PREDICATE LIMITATION:
                                # FSETP writes predicates that work for SEL/FSEL
                                # (data-path predicate reads) but NOT for BRA/EXIT
                                # guards (control-flow predicate reads). ptxas knows
                                # this and never uses FSETP predicates as branch guards.
                                #
                                # Workaround: FSEL.step (compare+select → 1.0/0.0)
                                # then ISETP.NE to convert to a branch-compatible pred.
                                if isinstance(b, ImmOp):
                                    threshold = b.value & 0xFFFFFFFF
                                elif isinstance(b, RegOp):
                                    br = ctx.ra.r32(b.name)
                                    threshold = None  # register form
                                else:
                                    threshold = 0

                                _fsel_cmp2 = {'lt': FSEL_LT, 'le': FSEL_LE, 'gt': FSEL_GT,
                                              'ge': FSEL_GE, 'eq': FSEL_EQ, 'ne': FSEL_NE}
                                fsel_c = _fsel_cmp2.get(cmp_name, FSEL_GT)

                                tmp_r = _alloc_gpr(ctx)
                                if isinstance(b, RegOp) and threshold is None:
                                    # Reg-reg: FSUB + FSEL.step + ISETP.NE
                                    diff_r = _alloc_gpr(ctx)
                                    output.append(SassInstr(
                                        encode_fadd(diff_r, br, ar, negate_src0=True),
                                        f'FADD R{diff_r}, -R{br}, R{ar}  // fsub for cmp'))
                                    output.append(SassInstr(
                                        encode_fsel_step(tmp_r, diff_r, 0, fsel_c),
                                        f'FSEL.step R{tmp_r}, R{diff_r}, 0x0, {cmp_name.upper()}'))
                                    output.append(SassInstr(
                                        encode_isetp(pd, tmp_r, RZ, ISETP_NE),
                                        f'ISETP.NE P{pd}, R{tmp_r}, RZ  // float reg cmp -> pred'))
                                    if hasattr(ctx, '_negated_preds'):
                                        ctx._negated_preds.discard(pd)
                                    ctx._scratch_mark = ctx._next_gpr
                                else:
                                    # Reg-imm: FSEL.step + ISETP.NE
                                    output.append(SassInstr(
                                        encode_fsel_step(tmp_r, ar, threshold, fsel_c),
                                        f'FSEL.step R{tmp_r}, R{ar}, 0x{threshold:08x}, {cmp_name.upper()}'))
                                    output.append(SassInstr(
                                        encode_isetp(pd, tmp_r, RZ, ISETP_NE),
                                        f'ISETP.NE P{pd}, R{tmp_r}, RZ  // float cmp -> pred'))
                                    if hasattr(ctx, '_negated_preds'):
                                        ctx._negated_preds.discard(pd)
                        else:
                            # Integer comparison
                            # SM_120 R-UR ISETP only emits cmp=GE (6) or cmp=GT (4)
                            # in ptxas ground truth.  Direct LT/LE encodings are
                            # never used by ptxas — both produce wrong results on
                            # hardware (FG-2.1: setp.le emitted as ISETP.LE direct
                            # silently computes GE; setp.gt -> ISETP.LE+neg likewise
                            # silently computes LT).  We therefore canonicalize
                            # away from LT/LE the same way ptxas does:
                            #   setp.lt → ISETP.GE + negate predicate
                            #   setp.le → ISETP.GT + negate predicate
                            #   setp.eq → ISETP.NE + negate predicate  (FG-2.3)
                            # GE, GT, and NE are emitted directly.  IMM path
                            # supports the full comparison set so it's exempt.
                            #
                            # FG-2.3 addition: EQ falls through to NE + neg so
                            # the R-UR path can be used for equality compares
                            # against UR-bound params.  Without this, EQ/NE
                            # would fall to the ISETP R-R path, which cannot
                            # read a UR-bound operand — the isel ended up
                            # reading an uninitialised GPR slot and silently
                            # produced a constant predicate.
                            _INVERT = {'lt': 'ge', 'le': 'gt', 'eq': 'ne'}
                            _can_use_ur_direct = False
                            _can_use_imm_direct = False
                            # ISETP.IMM (0x80c) supports all comparison codes directly.
                            # No inversion needed for immediate operands on SM_120.
                            if (cmp_name in _INVERT and ctx.sm_version != 89
                                    and isinstance(b, ImmOp)):
                                _can_use_imm_direct = True
                            if cmp_name in _INVERT and ctx.sm_version != 89 and not _can_use_ur_direct and not _can_use_imm_direct:
                                cmp_name = _INVERT[cmp_name]
                                if not hasattr(ctx, '_negated_preds'):
                                    ctx._negated_preds = set()
                                ctx._negated_preds.add(pd)
                            else:
                                # Non-inverted comparison: clear any stale negation
                                # from a previous setp that wrote the same predicate.
                                if hasattr(ctx, '_negated_preds'):
                                    ctx._negated_preds.discard(pd)
                            cmp_map = {'lt': ISETP_LT, 'le': ISETP_LE, 'gt': ISETP_GT,
                                       'ge': ISETP_GE, 'eq': ISETP_EQ, 'ne': ISETP_NE}
                            isetp_cmp = cmp_map.get(cmp_name, ISETP_GE)
                            if ctx.sm_version == 89:
                                # SM_89: ISETP R-R (0x20c) works correctly.
                                br = ctx.ra.r32(b.name) if isinstance(b, RegOp) else RZ
                                if isinstance(b, ImmOp):
                                    imm_val = b.value & 0xFFFFFFFF
                                    br = _alloc_gpr(ctx)
                                    output.append(SassInstr(encode_iadd3_imm32(br, RZ, imm_val, RZ),
                                        f'IADD3 R{br}, RZ, 0x{imm_val:x}, RZ  // setp imm'))
                                output.append(SassInstr(
                                    encode_isetp(pd, ar, br, cmp=isetp_cmp),
                                    f'ISETP.{cmp_name.upper()}.U32.AND P{pd}, PT, R{ar}, R{br}, PT'))
                            elif isinstance(b, RegOp):
                                b_param_off = ctx._reg_param_off.get(b.name) if ctx else None
                                b_ur_idx = (ctx._ur_params.get(b.name) if ctx else None)
                                if b_ur_idx is not None and isetp_cmp in (ISETP_GE, ISETP_GT, ISETP_LE, ISETP_LT, ISETP_NE):
                                    # SM_120 rule #25: ISETP.UR + VOTE causes ERR715 when
                                    # LDG is present. Use GPR path for vote kernels.
                                    _vote_safe = getattr(ctx, '_has_vote', False)
                                    if _vote_safe:
                                        # Value already in GPR (from LDCU.64 + MOV).
                                        br = ctx.ra.r32(b.name)
                                        emit_pd = pd
                                        output.append(SassInstr(
                                            encode_isetp(emit_pd, ar, br, cmp=isetp_cmp),
                                            f'ISETP.{cmp_name.upper()}.U32.AND P{emit_pd}, PT, R{ar}, R{br}, PT  // vote-safe GPR'))
                                    else:
                                        emit_pd = pd
                                        output.append(SassInstr(
                                            encode_isetp_ur(emit_pd, ar, b_ur_idx, cmp=isetp_cmp),
                                            f'ISETP.{cmp_name.upper()}.U32.AND P{emit_pd}, PT, R{ar}, UR{b_ur_idx}, PT'))
                                elif b_param_off is not None and isetp_cmp in (ISETP_GE, ISETP_GT, ISETP_LE, ISETP_LT, ISETP_NE):
                                    # SM_120 rule: LDCU.32 in the body poisons IADD.64-UR.
                                    # Always use GPR R-R path. The param value was loaded
                                    # into a GPR by ld.param → LDC at the regular u32 path.
                                    br = ctx.ra.r32(b.name)
                                    emit_pd = pd
                                    output.append(SassInstr(
                                        encode_isetp(emit_pd, ar, br, cmp=isetp_cmp),
                                        f'ISETP.{cmp_name.upper()}.U32.AND P{emit_pd}, PT, R{ar}, R{br}, PT  // GPR path (no body LDCU.32)'))
                                else:
                                    # No UR/param available for ISETP.UR. Use ISETP R-R
                                    # as last resort. NOTE: ISETP R-R (0x20c) has toxic
                                    # interaction with VOTE on SM_120. For vote-feeding
                                    # compares, prefer ISETP.UR by materializing src1
                                    # via LDCU.32 from a scratch literal pool slot.
                                    br = ctx.ra.r32(b.name)
                                    emit_pd = pd
                                    output.append(SassInstr(
                                        encode_isetp(emit_pd, ar, br, cmp=isetp_cmp),
                                        f'ISETP.{cmp_name.upper()}.U32.AND P{emit_pd}, PT, R{ar}, R{br}, PT'))
                            elif isinstance(b, ImmOp):
                                # Immediate src1: ptxas uses ISETP R-R (0x20c) with RZ for imm=0,
                                # or materializes the constant in a GPR for non-zero immediates.
                                # The literal-pool path (LDCU.32 from c[0]) is unreliable because
                                # the driver only initializes the param area — bytes beyond the
                                # params are uninitialized garbage, so imm=0 from the literal pool
                                # at c[0][param_end] reads a nonzero value.
                                imm_val = b.value & 0xFFFFFFFF
                                emit_pd = pd
                                # P2-2: removed P0-forcing. SM_120 supports P0-P5.
                                # SM_120 rule: use ISETP.IMM (0x80c) for ALL immediate
                                # comparisons, including imm=0. ISETP R-R (0x20c) causes
                                # toxic interaction with VOTE on SM_120 (rule #23).
                                from sass.encoding.sm_120_opcodes import encode_isetp_imm
                                output.append(SassInstr(
                                    encode_isetp_imm(emit_pd, ar, imm_val, cmp=isetp_cmp),
                                    f'ISETP.{cmp_name.upper()}.IMM P{emit_pd}, R{ar}, {imm_val:#x}'))
                            else:
                                # Non-register src1 (e.g. memory operand) — materialize into GPR first
                                br = _materialize_imm(b, ctx, ctx.ra, output)
                                emit_pd = pd
                                output.append(SassInstr(
                                    encode_isetp(emit_pd, ar, br, cmp=isetp_cmp),
                                    f'ISETP.{cmp_name.upper()}.U32.AND P{emit_pd}, PT, R{ar}, R{br}, PT  // setp non-reg src1'))
                    else:
                        # Non-register pred dest or src0 — invalid PTX or unusual operand form
                        import sys as _sys
                        print(f'WARNING: setp with non-register pred/src0: {instr}', file=_sys.stderr)
                        output.append(_nop(f'WARNING: setp non-register pred/src0: {instr}'))

                elif op == 'testp' and 'finite' in instr.types and 'f32' in instr.types:
                    # testp.finite.f32 p, f:
                    #   p = isfinite(f) = (f_bits & 0x7F800000) < 0x7F800000
                    # Lowering (4 instructions):
                    #   R_mask = 0x7F800000         (IADD3_IMM)
                    #   R_abs  = f_bits & R_mask    (LOP3.AND)
                    #   UR_thr = 0x7F800000         (LDCU.32 from literal pool)
                    #   p      = (R_abs < UR_thr)   (ISETP.LT.U32.AND R-UR)
                    pred   = instr.dest
                    f_op   = instr.srcs[0]
                    pd = ctx.ra.pred(pred.name) if pred.name in ctx.ra.pred_regs else 0
                    emit_pd = pd
                    f_reg = ctx.ra.r32(f_op.name)
                    R_mask = _alloc_gpr(ctx)
                    R_abs  = _alloc_gpr(ctx)
                    FINITE_MASK = 0x7F800000
                    output.append(SassInstr(
                        encode_iadd3_imm32(R_mask, RZ, FINITE_MASK, RZ),
                        f'MOV R{R_mask}, 0x7f800000  // testp.finite mask'))
                    output.append(SassInstr(
                        encode_lop3(R_abs, f_reg, R_mask, RZ, LOP3_AND),
                        f'LOP3.AND R{R_abs}, R{f_reg}, R{R_mask}, RZ  // testp.finite & exp mask'))
                    lit_off = ctx._alloc_literal(FINITE_MASK)
                    ur_thr  = ctx._next_ur; ctx._next_ur += 1
                    output.append(SassInstr(
                        encode_ldcu_32(ur_thr, 0, lit_off),
                        f'LDCU.32 UR{ur_thr}, c[0][0x{lit_off:x}]  // testp.finite threshold'))
                    output.append(SassInstr(
                        encode_isetp_ur(emit_pd, R_abs, ur_thr, cmp=ISETP_LT),
                        f'ISETP.LT.U32.AND P{emit_pd}, PT, R{R_abs}, UR{ur_thr}, PT  // testp.finite'))

                elif op == 'neg' and typ in ('s32', 'u32'):
                    # neg: IADD3 with src0=RZ, src1=src, negate_src1
                    # dest = 0 - src
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_iadd3(d, RZ, a, RZ, negate_src1=True),
                                            f'IADD3 R{d}, RZ, -R{a}, RZ  // neg.{typ}'))

                elif op == 'neg' and typ in ('s64', 'u64', 'b64'):
                    # neg.s64: d = 0 - a (two's complement of 64-bit value)
                    # SM_120: IADD.64 R-R broken. Use IADD3+IADD3.X.
                    d_lo = ctx.ra.lo(instr.dest.name); d_hi = d_lo + 1
                    a_lo = ctx.ra.lo(instr.srcs[0].name); a_hi = a_lo + 1
                    output.append(SassInstr(encode_iadd3(d_lo, RZ, a_lo, RZ, negate_src1=True),
                                            f'IADD3 R{d_lo}, RZ, -R{a_lo}, RZ  // neg.{typ} lo'))
                    output.append(SassInstr(encode_iadd3x(d_hi, RZ, a_hi, RZ, negate_src1=True),
                                            f'IADD3.X R{d_hi}, RZ, -R{a_hi}, RZ  // neg.{typ} hi'))

                elif op == 'neg' and typ == 'f32':
                    # neg.f32: FADD with negated src and zero
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_fadd(d, RZ, a, negate_src0=True),
                                            f'FADD R{d}, -R{a}, RZ  // neg.f32'))

                elif op == 'abs' and typ == 'f32':
                    # abs.f32: FADD |src|, RZ (absolute value via abs modifier)
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_fadd(d, a, RZ, abs_src0=True),
                                            f'FADD R{d}, |R{a}|, RZ  // abs.f32'))

                elif op == 'neg' and typ == 'f64':
                    # neg.f64: flip sign bit (bit 31) of hi word via XOR 0x80000000.
                    # lo word is unchanged.
                    d = ctx.ra.lo(instr.dest.name)
                    a = ctx.ra.lo(instr.srcs[0].name)
                    tmp = _alloc_gpr(ctx)
                    output.append(SassInstr(encode_iadd3_imm32(tmp, RZ, 0x80000000, RZ),
                                            f'IADD3 R{tmp}, RZ, 0x80000000, RZ  // neg.f64 sign mask'))
                    output.append(SassInstr(encode_lop3(d+1, a+1, tmp, RZ, LOP3_XOR),
                                            f'LOP3 R{d+1}, R{a+1}, R{tmp}, RZ, XOR  // neg.f64 hi'))
                    if d != a:
                        output.append(SassInstr(encode_iadd3(d, a, RZ, RZ),
                                                f'IADD3 R{d}, R{a}, RZ, RZ  // neg.f64 lo'))

                elif op == 'abs' and typ == 'f64':
                    # abs.f64: clear sign bit (bit 31) of hi word via AND 0x7FFFFFFF.
                    # lo word is unchanged.
                    d = ctx.ra.lo(instr.dest.name)
                    a = ctx.ra.lo(instr.srcs[0].name)
                    tmp = _alloc_gpr(ctx)
                    output.append(SassInstr(encode_iadd3_imm32(tmp, RZ, 0x7FFFFFFF, RZ),
                                            f'IADD3 R{tmp}, RZ, 0x7FFFFFFF, RZ  // abs.f64 mask'))
                    output.append(SassInstr(encode_lop3(d+1, a+1, tmp, RZ, LOP3_AND),
                                            f'LOP3 R{d+1}, R{a+1}, R{tmp}, RZ, AND  // abs.f64 hi'))
                    if d != a:
                        output.append(SassInstr(encode_iadd3(d, a, RZ, RZ),
                                                f'IADD3 R{d}, R{a}, RZ, RZ  // abs.f64 lo'))

                elif op == 'selp' and typ == 'f64':
                    # selp.f64 dest, src0, src1, Pp  →  2×FSEL (lo then hi 32-bit word)
                    d = ctx.ra.lo(instr.dest.name)
                    a = ctx.ra.lo(instr.srcs[0].name)
                    b = ctx.ra.lo(instr.srcs[1].name)
                    pd = 0
                    neg = False
                    if len(instr.srcs) > 2 and isinstance(instr.srcs[2], RegOp):
                        pd = ctx.ra.pred(instr.srcs[2].name) if instr.srcs[2].name in ctx.ra.pred_regs else 0
                        neg = hasattr(ctx, '_negated_preds') and pd in ctx._negated_preds
                    output.append(SassInstr(encode_fsel(d,   a,   b,   pd, neg),
                                            f'FSEL R{d},   R{a},   R{b},   {"!" if neg else ""}P{pd}  // selp.f64 lo'))
                    output.append(SassInstr(encode_fsel(d+1, a+1, b+1, pd, neg),
                                            f'FSEL R{d+1}, R{a+1}, R{b+1}, {"!" if neg else ""}P{pd}  // selp.f64 hi'))

                elif op == 'selp':
                    d = ctx.ra.r32(instr.dest.name)
                    pd = 0
                    neg = False
                    if len(instr.srcs) > 2 and isinstance(instr.srcs[2], RegOp):
                        pd = ctx.ra.pred(instr.srcs[2].name) if instr.srcs[2].name in ctx.ra.pred_regs else 0
                        neg = hasattr(ctx, '_negated_preds') and pd in ctx._negated_preds
                    s0, s1 = instr.srcs[0], instr.srcs[1]
                    # Predicated-MOV path: avoids SEL predicate-read barrier races.
                    # SM_120 scoreboard handles instruction guards (@P) correctly,
                    # so we emit: MOV R, false_val; @P MOV R, true_val.
                    # For negated preds (setp inversion): @!P MOV R, true_val.
                    if isinstance(s0, ImmOp) and isinstance(s1, ImmOp):
                        true_val  = s0.value & 0xFFFFFFFF
                        false_val = s1.value & 0xFFFFFFFF
                        # Emit unconditional MOV for false value
                        output.append(SassInstr(
                            encode_iadd3_imm32(d, RZ, false_val, RZ),
                            f'IADD3 R{d}, RZ, {false_val:#x}, RZ  // selp false'))
                        # Emit predicated MOV for true value
                        pred_byte = (pd & 0x07)
                        if neg:
                            pred_byte |= 0x08  # negate
                        raw_pmov = bytearray(encode_iadd3_imm32(d, RZ, true_val, RZ))
                        raw_pmov[1] = (raw_pmov[1] & 0x0F) | (pred_byte << 4)
                        output.append(SassInstr(
                            bytes(raw_pmov),
                            f'@{"!" if neg else ""}P{pd} IADD3 R{d}, RZ, {true_val:#x}, RZ  // selp true'))
                    elif isinstance(s0, ImmOp) or isinstance(s1, ImmOp):
                        # One register, one immediate
                        if isinstance(s0, ImmOp):
                            true_val = s0.value & 0xFFFFFFFF
                            false_reg = ctx.ra.r32(s1.name) if isinstance(s1, RegOp) else RZ
                        else:
                            true_val = s1.value & 0xFFFFFFFF
                            false_reg = ctx.ra.r32(s0.name) if isinstance(s0, RegOp) else RZ
                            neg = not neg  # swap true/false semantics
                        # MOV R, false_reg; @P MOV R, true_val
                        output.append(SassInstr(
                            encode_iadd3(d, false_reg, RZ, RZ),
                            f'MOV R{d}, R{false_reg}  // selp false'))
                        pred_byte = (pd & 0x07)
                        if neg:
                            pred_byte |= 0x08
                        raw_pmov = bytearray(encode_iadd3_imm32(d, RZ, true_val, RZ))
                        raw_pmov[1] = (raw_pmov[1] & 0x0F) | (pred_byte << 4)
                        output.append(SassInstr(
                            bytes(raw_pmov),
                            f'@{"!" if neg else ""}P{pd} IADD3 R{d}, RZ, {true_val:#x}, RZ  // selp true'))
                    else:
                        # Both register sources: use predicated MOV (P2-2).
                        # SEL only supports P0 on SM_120, but predicated IADD3
                        # works with all physical predicates (P0-P5).
                        a = ctx.ra.r32(s0.name) if isinstance(s0, RegOp) else RZ
                        b = ctx.ra.r32(s1.name) if isinstance(s1, RegOp) else RZ
                        true_reg, false_reg = (a, b) if not neg else (b, a)
                        # MOV d = false_reg (unconditional)
                        output.append(SassInstr(encode_iadd3(d, false_reg, RZ, RZ),
                            f'MOV R{d}, R{false_reg}  // selp false'))
                        # @P MOV d = true_reg (predicated)
                        pred_byte = (pd & 0x07)
                        if neg:
                            pred_byte |= 0x08
                        raw_pmov = bytearray(encode_iadd3(d, true_reg, RZ, RZ))
                        raw_pmov[1] = (raw_pmov[1] & 0x0F) | (pred_byte << 4)
                        output.append(SassInstr(bytes(raw_pmov),
                            f'@{"!" if neg else ""}P{pd} MOV R{d}, R{true_reg}  // selp true'))

                elif op == 'min' and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    is_signed = typ == 's32'
                    enc = encode_vimnmx_s32 if is_signed else encode_vimnmx_u32
                    output.append(SassInstr(enc(d, a, b, is_max=False),
                        f'VIMNMX.{"S" if is_signed else "U"}32 R{d}, R{a}, R{b}, PT  // min.{typ}'))

                elif op == 'max' and typ in ('u32', 's32'):
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    is_signed = typ == 's32'
                    enc = encode_vimnmx_s32 if is_signed else encode_vimnmx_u32
                    output.append(SassInstr(enc(d, a, b, is_max=True),
                        f'VIMNMX.{"S" if is_signed else "U"}32 R{d}, R{a}, R{b}, !PT  // max.{typ}'))

                elif op == 'sad' and typ in ('u32', 's32'):
                    # sad.u32 d, a, b, c  →  d = |a - b| + c
                    # VIMNMX.MAX t0, a, b
                    # VIMNMX.MIN t1, a, b
                    # IADD3 d, t0, -t1, c  (d = max - min + c = |a-b| + c)
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    c = _materialize_imm(instr.srcs[2], ctx, ctx.ra, output) if len(instr.srcs) > 2 else RZ
                    t_max = _alloc_gpr(ctx)
                    t_min = _alloc_gpr(ctx)
                    is_signed = typ == 's32'
                    output.append(SassInstr(encode_vimnmx_s32(t_max, a, b, is_max=True) if is_signed else encode_vimnmx_u32(t_max, a, b, is_max=True),
                                            f'VIMNMX.{"S" if is_signed else "U"}32 R{t_max}, R{a}, R{b}  // sad max'))
                    output.append(SassInstr(encode_vimnmx_s32(t_min, a, b, is_max=False) if is_signed else encode_vimnmx_u32(t_min, a, b, is_max=False),
                                            f'VIMNMX.{"S" if is_signed else "U"}32 R{t_min}, R{a}, R{b}  // sad min'))
                    output.append(SassInstr(encode_iadd3_neg_b4(d, t_max, t_min, c),
                                            f'IADD3 R{d}, R{t_max}, -R{t_min}, R{c}  // sad |a-b|+c'))

                elif op == 'mad' and 'lo' in instr.types:
                    # mad.lo.s32 → dest = src0 * src1 + src2
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    c_op = instr.srcs[2] if len(instr.srcs) > 2 else None
                    # FG-4.4 Bug 2 fix: if src2 is an immediate, it was
                    # silently replaced with RZ here and the +src2
                    # addend was lost.  Materialize the immediate into
                    # a scratch GPR first and use that as the c operand.
                    if isinstance(c_op, RegOp):
                        c = ctx.ra.r32(c_op.name)
                    elif isinstance(c_op, ImmOp):
                        if (c_op.value & 0xFFFFFFFF) == 0:
                            c = RZ
                        else:
                            c = _materialize_imm(c_op, ctx, ctx.ra, output)
                    else:
                        c = RZ
                    if isinstance(instr.srcs[1], ImmOp):
                        # FG-4.4 Bug 2 fix: for 16-bit-fitting immediate
                        # multipliers, use encode_imad_r_imm (opcode
                        # 0x824, `IMAD dest, src0, imm16, src2`) which
                        # encodes the immediate inline in b4:b5.  This
                        # avoids the literal-pool + LDCU.32 path that
                        # was broken by the CUDA driver zeroing literals
                        # placed adjacent to the param area.
                        #
                        # Power-of-2 shortcut via IMAD.SHL.U32 is kept
                        # for its IADD3 fast path; the LDCU literal
                        # pool path is retired for immediates <= 0xffff
                        # and only used for imm > 0xffff.
                        imm = instr.srcs[1].value & 0xFFFFFFFF
                        from sass.encoding.sm_120_opcodes import encode_imad_r_imm as _encode_imad_r_imm
                        if imm > 0 and (imm & (imm - 1)) == 0:
                            shift = imm.bit_length() - 1
                            if shift <= 15:
                                t = _alloc_gpr(ctx)
                                output.append(SassInstr(encode_imad_shl_u32(t, a, shift),
                                    f'IMAD.SHL.U32 R{t}, R{a}, 0x{imm:x}, RZ  // mad.lo shift'))
                                output.append(SassInstr(encode_iadd3(d, t, c, RZ),
                                    f'IADD3 R{d}, R{t}, R{c}, RZ  // mad.lo add'))
                            else:
                                output.append(SassInstr(
                                    _encode_imad_r_imm(d, a, imm, c),
                                    f'IMAD R{d}, R{a}, 0x{imm:x}, R{c}  // mad.lo imm (inline)'))
                        elif imm <= 0xffff:
                            # Direct inline-immediate IMAD.
                            output.append(SassInstr(
                                _encode_imad_r_imm(d, a, imm, c),
                                f'IMAD R{d}, R{a}, 0x{imm:x}, R{c}  // mad.lo imm (inline)'))
                        else:
                            # imm > 0xffff: still need the literal-pool
                            # path.  The lit_pool_base alignment hack in
                            # pipeline.py ensures the pool is past the
                            # driver's cbuf-pad region, but for the
                            # common small-imm case we skip this entirely
                            # via the inline path above.
                            lit_off = ctx._alloc_literal(imm)
                            ur_tmp = ctx._next_ur; ctx._next_ur += 1
                            output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, lit_off),
                                f'LDCU.32 UR{ur_tmp}, c[0][0x{lit_off:x}]'))
                            output.append(_nop('ldcu32->imad gap'))
                            output.append(SassInstr(encode_imad_ur(d, a, ur_tmp, c),
                                f'IMAD R{d}, R{a}, UR{ur_tmp}, R{c}  // mad.lo imm'))
                        continue
                    b = ctx.ra.r32(instr.srcs[1].name)
                    src0_name = instr.srcs[0].name if instr.srcs else ''
                    src1_name = instr.srcs[1].name if len(instr.srcs) > 1 else ''
                    ur_map = getattr(ctx, '_ur_for_param', {})
                    if src0_name in ur_map:
                        # src0 is in a UR (e.g. ctaid via S2UR) — use IMAD R-UR.
                        # IMAD R-UR: dest = src0_gpr * ur + src2. Multiplication is
                        # commutative so we put the GPR operand (src1) in src0 position.
                        ur_src = ur_map[src0_name]
                        output.append(SassInstr(encode_imad_ur(d, b, ur_src, c),
                            f'IMAD R{d}, R{b}, UR{ur_src}, R{c}  // mad.lo.{typ}'))
                    elif src1_name in ur_map:
                        ur_src = ur_map[src1_name]
                        output.append(SassInstr(encode_imad_ur(d, a, ur_src, c),
                            f'IMAD R{d}, R{a}, UR{ur_src}, R{c}  // mad.lo.{typ}'))
                    else:
                        # IMAD R-R (0x2a4) is BROKEN on SM_120 — only IMAD R-UR (0xc24) works.
                        # SM_89: skip LDCU.32 path, go straight to IMAD.WIDE R-R fallback.
                        src1_param_off = ctx._reg_param_off.get(src1_name) if ctx else None
                        src0_param_off = ctx._reg_param_off.get(src0_name) if ctx else None
                        if ctx and ctx.sm_version == 89:
                            # Force R-R fallback — SM_89 has no LDCU.32/IMAD R-UR
                            src1_param_off = None
                            src0_param_off = None
                        if src1_param_off is not None:
                            ur_tmp = ctx._next_ur; ctx._next_ur += 1
                            output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, src1_param_off),
                                f'LDCU.32 UR{ur_tmp}, c[0][0x{src1_param_off:x}]  // mad src1->UR'))
                            output.append(_nop('ldcu32->imad gap 1'))
                            output.append(SassInstr(encode_imad_ur(d, a, ur_tmp, c),
                                f'IMAD R{d}, R{a}, UR{ur_tmp}, R{c}  // mad.lo.{typ} R-UR'))
                        elif src0_param_off is not None:
                            ur_tmp = ctx._next_ur; ctx._next_ur += 1
                            output.append(SassInstr(encode_ldcu_32(ur_tmp, 0, src0_param_off),
                                f'LDCU.32 UR{ur_tmp}, c[0][0x{src0_param_off:x}]  // mad src0->UR'))
                            output.append(_nop('ldcu32->imad gap 1'))
                            output.append(SassInstr(encode_imad_ur(d, b, ur_tmp, c),
                                f'IMAD R{d}, R{b}, UR{ur_tmp}, R{c}  // mad.lo.{typ} R-UR'))
                        else:
                            # IMAD R-R (0x2a4) is BROKEN on SM_120 but IMAD.WIDE R-R
                            # (0x225) works. Use WIDE for the multiply, then add the
                            # addend via IADD3.
                            t = _alloc_gpr(ctx)
                            if t % 2 != 0:
                                t = _alloc_gpr(ctx)
                            _alloc_gpr(ctx)  # reserve t+1
                            output.append(SassInstr(encode_imad_wide_rr(t, a, b, RZ),
                                f'IMAD.WIDE R{t}, R{a}, R{b}, RZ  // mad.lo.{typ} R-R via WIDE'))
                            if c != RZ:
                                output.append(SassInstr(encode_iadd3(d, t, c, RZ),
                                    f'IADD3 R{d}, R{t}, R{c}, RZ  // mad.lo add'))
                            elif t != d:
                                output.append(SassInstr(encode_mov(d, t),
                                    f'MOV R{d}, R{t}  // mad.lo result'))

                elif op == 'mad' and 'wide' in instr.types and typ in ('u32', 's32'):
                    # mad.wide.u32/s32 d64, a32, b32_or_imm, c64
                    # Result pair: (dest_lo, dest_hi) = a * b + c64
                    # IMAD.WIDE writes dest and dest+1 atomically.
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a    = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    c_lo = ctx.ra.lo(instr.srcs[2].name) if len(instr.srcs) > 2 else RZ
                    if isinstance(instr.srcs[1], ImmOp):
                        imm = instr.srcs[1].value & 0xFFFF_FFFF
                        if imm <= 0xFF:
                            output.append(SassInstr(
                                encode_imad_wide(d_lo, a, imm, c_lo),
                                f'IMAD.WIDE R{d_lo}, R{a}, 0x{imm:x}, R{c_lo}  // mad.wide.{typ}'))
                        else:
                            # Large immediate: load via literal pool into UR, then R-UR IMAD.WIDE
                            lit_off = ctx._alloc_literal(imm)
                            ur_tmp = ctx._next_ur; ctx._next_ur += 1
                            output.append(SassInstr(
                                encode_ldcu_32(ur_tmp, 0, lit_off),
                                f'LDCU.32 UR{ur_tmp}, c[0][0x{lit_off:x}]  // mad.wide imm={imm:#x}'))
                            # Use R-imm form with UR treated as immediate slot
                            output.append(SassInstr(
                                encode_imad_wide(d_lo, a, ur_tmp, c_lo),
                                f'IMAD.WIDE R{d_lo}, R{a}, UR{ur_tmp}, R{c_lo}  // mad.wide large imm'))
                    else:
                        b = ctx.ra.r32(instr.srcs[1].name)
                        output.append(SassInstr(
                            encode_imad_wide_rr(d_lo, a, b, c_lo),
                            f'IMAD.WIDE R{d_lo}, R{a}, R{b}, R{c_lo}  // mad.wide.{typ} R-R'))

                elif op == 'mul' and 'hi' in instr.types and typ in ('u32', 's32'):
                    # mul.hi.u32: upper 32 bits of 32×32 product.
                    # Use IMAD.WIDE to get full 64-bit result, then take upper word.
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    # Allocate even-aligned tmp pair for IMAD.WIDE dest
                    tmp_lo = _alloc_gpr(ctx)
                    if tmp_lo & 1:
                        tmp_lo = _alloc_gpr(ctx)
                    tmp_hi = tmp_lo + 1
                    ctx._next_gpr = max(ctx._next_gpr, tmp_hi + 1)
                    _wide_enc = encode_imad_wide_u32 if typ == 'u32' else encode_imad_wide_rr
                    output.append(SassInstr(
                        _wide_enc(tmp_lo, a, b, RZ),
                        f'IMAD.WIDE.{"U32" if typ=="u32" else "S32"} R{tmp_lo}, R{a}, R{b}, RZ  // mul.hi.{typ}'))
                    if d != tmp_hi:
                        output.append(SassInstr(encode_iadd3(d, tmp_hi, RZ, RZ),
                                                f'MOV R{d}, R{tmp_hi}  // mul.hi upper'))

                elif op == 'mul' and 'wide' in instr.types and typ in ('u32', 's32'):
                    # mul.wide.u32: 64-bit result = 32×32 product
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    _wide_enc = encode_imad_wide_u32 if typ == 'u32' else encode_imad_wide_rr
                    output.append(SassInstr(
                        _wide_enc(d_lo, a, b, RZ),
                        f'IMAD.WIDE.{"U32" if typ=="u32" else "S32"} R{d_lo}, R{a}, R{b}, RZ  // mul.wide.{typ}'))

                elif op == 'mul' and 'hi' in instr.types and typ in ('u64', 's64'):
                    # mul.hi.u64: upper 64 bits of 128-bit unsigned product.
                    # Algorithm (schoolbook using IMAD.WIDE.U32):
                    #   a_lo, a_hi = src0 pair; b_lo, b_hi = src1 pair
                    #   t0 = IMAD.WIDE.U32(a_hi, b_lo, 0)          → a_hi*b_lo [64-bit]
                    #   t1 = IMAD.WIDE.U32(a_lo, b_hi, t0, P0)     → a_lo*b_hi + t0 [64-bit, sets P0]
                    #   t2 = IMAD.WIDE.U32(a_lo, b_lo, 0)          → a_lo*b_lo [64-bit]
                    #   carry = 0 + P0 (IADD3.X)                   → capture carry from step 2
                    #   sum_hi = t1_hi (=R9 in ground truth)
                    #   IADD3 RZ, P0, t2_hi, t1_lo, RZ             → detect carry from bit-32 sum
                    #   d = IMAD.WIDE.U32.X(a_hi, b_hi, sum_hi, P0) → a_hi*b_hi + sum_hi + carry
                    # Ground truth verified from ptxas mul.hi.u64 on SM_120.
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = ctx.ra.lo(instr.srcs[0].name)
                    b_lo = ctx.ra.lo(instr.srcs[1].name)
                    a_hi = a_lo + 1;  b_hi = b_lo + 1
                    t0_lo = ctx._next_gpr; ctx._next_gpr += 2  # t0 pair
                    t1_lo = ctx._next_gpr; ctx._next_gpr += 2  # t1 pair
                    t2_lo = ctx._next_gpr; ctx._next_gpr += 2  # t2 pair
                    carry = _alloc_gpr(ctx)
                    # Step 1: t0 = a_hi * b_lo
                    output.append(SassInstr(encode_imad_wide_u32(t0_lo, a_hi, b_lo, RZ),
                        f'IMAD.WIDE.U32 R{t0_lo}, R{a_hi}, R{b_lo}, RZ  // mul.hi.u64 step1'))
                    # Step 2: t1 = a_lo * b_hi + t0 (sets P0 carry)
                    output.append(SassInstr(encode_imad_wide_u32_carry(t1_lo, a_lo, b_hi, t0_lo),
                        f'IMAD.WIDE.U32 R{t1_lo}, P0, R{a_lo}, R{b_hi}, R{t0_lo}  // mul.hi.u64 step2'))
                    # Step 3: t2 = a_lo * b_lo
                    output.append(SassInstr(encode_imad_wide_u32(t2_lo, a_lo, b_lo, RZ),
                        f'IMAD.WIDE.U32 R{t2_lo}, R{a_lo}, R{b_lo}, RZ  // mul.hi.u64 step3'))
                    # Step 4: carry = 0 + P0 carry from step 2
                    output.append(SassInstr(encode_iadd3x(carry, RZ, RZ, RZ),
                        f'IADD3.X R{carry}, PT, PT, RZ, RZ, RZ, P0, !PT  // mul.hi.u64 carry'))
                    # Step 5: save t1_hi (hi word of a_lo*b_hi + t0) for final product
                    sum_hi = _alloc_gpr(ctx)
                    output.append(SassInstr(encode_mov(sum_hi, t1_lo + 1),
                        f'MOV R{sum_hi}, R{t1_lo+1}  // mul.hi.u64 save t1_hi'))
                    # Step 6: IADD3 to detect carry from t2_hi + t1_lo into P0
                    output.append(SassInstr(encode_iadd3(RZ, t2_lo + 1, t1_lo, RZ),
                        f'IADD3 RZ, P0, PT, R{t2_lo+1}, R{t1_lo}, RZ  // mul.hi.u64 carry detect'))
                    # Step 7: d = a_hi * b_hi + sum_hi + P0 carry
                    output.append(SassInstr(encode_imad_wide_u32x(d_lo, a_hi, b_hi, sum_hi),
                        f'IMAD.WIDE.U32.X R{d_lo}, R{a_hi}, R{b_hi}, R{sum_hi}, P0  // mul.hi.u64 final'))

                elif op == 'popc' and typ in ('b32',):
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_popc(d, a),
                                            f'POPC R{d}, R{a}'))

                elif op == 'clz' and typ in ('b32',):
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    # CLZ = 31 - FLO(x).  FLO returns MSB position (0..31) or
                    # 0xFFFFFFFF for zero input.  31 - 0xFFFFFFFF = 32 (mod 2^32).
                    output.append(SassInstr(encode_flo(d, a),
                                            f'FLO.U32 R{d}, R{a}  // clz step 1'))
                    output.append(SassInstr(encode_iadd3_imm32_neg_src0(d, d, 31, RZ),
                                            f'IADD3 R{d}, -R{d}, 0x1f, RZ  // clz = 31 - FLO'))

                elif op == 'brev' and typ in ('b32',):
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_brev(d, a),
                                            f'BREV R{d}, R{a}'))

                elif op == 'abs' and typ in ('s32',):
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_iabs(d, a),
                                            f'IABS R{d}, R{a}'))

                elif op == 'abs' and typ in ('s64',):
                    # abs.s64 d, a  — branchless sign-bit trick:
                    #   sign = arithmetic-right-shift(a_hi, 31) = 0 or 0xFFFFFFFF
                    #   d    = (a XOR sign) + (-sign)   where -sign = 0 or 1
                    # This avoids predicated 64-bit instructions.
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = ctx.ra.lo(instr.srcs[0].name)
                    sign = _alloc_gpr(ctx)
                    t_lo = _alloc_gpr(ctx)  # addend lo (0 or 1)
                    t_hi = _alloc_gpr(ctx)  # addend hi (always 0)
                    output.append(SassInstr(encode_shf_r_s32_hi(sign, a_lo+1, 31),
                        f'SHF.R.S32.HI R{sign}, RZ, 0x1f, R{a_lo+1}  // abs.s64 sign'))
                    _emit_lop3(output, ctx, d_lo,   a_lo,   sign, RZ, LOP3_XOR, f'LOP3.XOR R{d_lo}, R{a_lo}, R{sign}, RZ  // abs.s64 lo XOR')
                    _emit_lop3(output, ctx, d_lo+1, a_lo+1, sign, RZ, LOP3_XOR, f'LOP3.XOR R{d_lo+1}, R{a_lo+1}, R{sign}, RZ  // abs.s64 hi XOR')
                    output.append(SassInstr(encode_iadd3(t_hi, RZ, RZ, RZ),
                        f'MOV R{t_hi}, RZ  // abs.s64 addend hi=0'))
                    output.append(SassInstr(encode_iadd3(t_lo, RZ, sign, RZ, negate_src1=True),
                        f'IADD3 R{t_lo}, RZ, -R{sign}, RZ  // abs.s64 addend=-sign'))
                    output.append(SassInstr(encode_iadd3(d_lo, d_lo, t_lo, RZ),
                        f'IADD3 R{d_lo}, R{d_lo}, R{t_lo}, RZ  // abs.s64 add lo'))
                    output.append(SassInstr(encode_iadd3x(d_lo+1, d_lo+1, t_hi, RZ),
                        f'IADD3.X R{d_lo+1}, R{d_lo+1}, R{t_hi}, RZ  // abs.s64 add hi'))

                elif op == 'min' and typ in ('u64', 's64'):
                    # min.u64 branchless: min(a,b) = b + ((a-b) & sign_mask(a-b))
                    #   diff = a - b; mask = sign_fill(diff_hi); d = b + (diff & mask)
                    # Works for unsigned because a < b → diff wraps to large value with sign=1.
                    # For signed min (s64), the same bit trick applies (signed subtraction).
                    d_lo  = ctx.ra.lo(instr.dest.name)
                    a_lo  = ctx.ra.lo(instr.srcs[0].name)
                    b_lo  = ctx.ra.lo(instr.srcs[1].name)
                    t_lo  = ctx._next_gpr; ctx._next_gpr += 2   # diff pair (t_lo, t_lo+1)
                    mask  = _alloc_gpr(ctx)
                    output.append(SassInstr(encode_iadd3(t_lo, a_lo, b_lo, RZ, negate_src1=True),
                        f'IADD3 R{t_lo}, R{a_lo}, -R{b_lo}, RZ  // min.{typ} diff lo'))
                    output.append(SassInstr(encode_iadd3x(t_lo+1, a_lo+1, b_lo+1, RZ, negate_src1=True),
                        f'IADD3.X R{t_lo+1}, R{a_lo+1}, -R{b_lo+1}, RZ  // min.{typ} diff hi'))
                    output.append(SassInstr(encode_shf_r_s32_hi(mask, t_lo+1, 31),
                        f'SHF.R.S32.HI R{mask}, RZ, 0x1f, R{t_lo+1}  // min.{typ} mask'))
                    _emit_lop3(output, ctx, t_lo,   t_lo,   mask, RZ, LOP3_AND, f'LOP3.AND R{t_lo}, R{t_lo}, R{mask}, RZ  // min.{typ} lo')
                    _emit_lop3(output, ctx, t_lo+1, t_lo+1, mask, RZ, LOP3_AND, f'LOP3.AND R{t_lo+1}, R{t_lo+1}, R{mask}, RZ  // min.{typ} hi')
                    output.append(SassInstr(encode_iadd3(d_lo, b_lo, t_lo, RZ),
                        f'IADD3 R{d_lo}, R{b_lo}, R{t_lo}, RZ  // min.{typ} result lo'))
                    output.append(SassInstr(encode_iadd3x(d_lo+1, b_lo+1, t_lo+1, RZ),
                        f'IADD3.X R{d_lo+1}, R{b_lo+1}, R{t_lo+1}, RZ  // min.{typ} result hi'))

                elif op == 'max' and typ in ('u64', 's64'):
                    # max.u64 branchless: max(a,b) = b + ((a-b) & ~sign_mask(a-b))
                    #   diff = a - b; mask = ~sign_fill(diff_hi); d = b + (diff & ~mask)
                    d_lo  = ctx.ra.lo(instr.dest.name)
                    a_lo  = ctx.ra.lo(instr.srcs[0].name)
                    b_lo  = ctx.ra.lo(instr.srcs[1].name)
                    t_lo  = ctx._next_gpr; ctx._next_gpr += 2   # diff pair
                    mask  = _alloc_gpr(ctx)   # inverted sign mask
                    output.append(SassInstr(encode_iadd3(t_lo, a_lo, b_lo, RZ, negate_src1=True),
                        f'IADD3 R{t_lo}, R{a_lo}, -R{b_lo}, RZ  // max.{typ} diff lo'))
                    output.append(SassInstr(encode_iadd3x(t_lo+1, a_lo+1, b_lo+1, RZ, negate_src1=True),
                        f'IADD3.X R{t_lo+1}, R{a_lo+1}, -R{b_lo+1}, RZ  // max.{typ} diff hi'))
                    output.append(SassInstr(encode_shf_r_s32_hi(mask, t_lo+1, 31),
                        f'SHF.R.S32.HI R{mask}, RZ, 0x1f, R{t_lo+1}  // max.{typ} sign'))
                    _emit_lop3(output, ctx, mask, mask, RZ, RZ, 0x0F, f'LOP3.NOT R{mask}, R{mask}, RZ, RZ  // max.{typ} ~sign')
                    _emit_lop3(output, ctx, t_lo,   t_lo,   mask, RZ, LOP3_AND, f'LOP3.AND R{t_lo}, R{t_lo}, R{mask}, RZ  // max.{typ} lo')
                    _emit_lop3(output, ctx, t_lo+1, t_lo+1, mask, RZ, LOP3_AND, f'LOP3.AND R{t_lo+1}, R{t_lo+1}, R{mask}, RZ  // max.{typ} hi')
                    output.append(SassInstr(encode_iadd3(d_lo, b_lo, t_lo, RZ),
                        f'IADD3 R{d_lo}, R{b_lo}, R{t_lo}, RZ  // max.{typ} result lo'))
                    output.append(SassInstr(encode_iadd3x(d_lo+1, b_lo+1, t_lo+1, RZ),
                        f'IADD3.X R{d_lo+1}, R{b_lo+1}, R{t_lo+1}, RZ  // max.{typ} result hi'))

                elif op == 'min' and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_fmnmx(d, a, b, is_max=False),
                                            f'FMNMX R{d}, R{a}, R{b}, PT  // min.f32'))

                elif op == 'max' and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    # Use RZ for known-zero operand (e.g., relu: max(x, 0))
                    if hasattr(ctx, '_zero_regs') and b in ctx._zero_regs:
                        if output and f'R{b}, RZ, 0x0, RZ' in output[-1].comment:
                            output.pop()
                        b = RZ
                    elif hasattr(ctx, '_zero_regs') and a in ctx._zero_regs:
                        if output and f'R{a}, RZ, 0x0, RZ' in output[-1].comment:
                            output.pop()
                        a = RZ
                    output.append(SassInstr(encode_fmnmx(d, a, b, is_max=True),
                                            f'FMNMX R{d}, R{a}, R{b}, !PT  // max.f32'))

                elif op == 'min' and typ == 'f64':
                    # min.f64 d, a, b → d = (a < b) ? a : b
                    # DSETP.GEU p, a, b → p_hw = (a >= b, unordered)
                    # !p_hw = (a < b, ordered when no NaN) → select a when !p_hw
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = _f64_to_gpr(instr.srcs[0].name, ctx, output)
                    b_lo = _f64_to_gpr(instr.srcs[1].name, ctx, output)
                    p_tmp = _alloc_scratch_pred(ctx)[0]
                    output.append(SassInstr(encode_dsetp(p_tmp, a_lo, b_lo, DSETP_GEU),
                                            f'DSETP.GEU P{p_tmp}, R{a_lo}, R{b_lo}  // min.f64 cmp'))
                    output.append(SassInstr(encode_fsel(d_lo,   a_lo,   b_lo,   p_tmp, negate_pred=True),
                                            f'FSEL R{d_lo},   R{a_lo},   R{b_lo},   !P{p_tmp}  // min.f64 lo'))
                    output.append(SassInstr(encode_fsel(d_lo+1, a_lo+1, b_lo+1, p_tmp, negate_pred=True),
                                            f'FSEL R{d_lo+1}, R{a_lo+1}, R{b_lo+1}, !P{p_tmp}  // min.f64 hi'))

                elif op == 'max' and typ == 'f64':
                    # max.f64 d, a, b → d = (a > b) ? a : b
                    # DSETP.LEU p, a, b → p_hw = (a <= b, unordered)
                    # !p_hw = (a > b, ordered when no NaN) → select a when !p_hw
                    d_lo = ctx.ra.lo(instr.dest.name)
                    a_lo = _f64_to_gpr(instr.srcs[0].name, ctx, output)
                    b_lo = _f64_to_gpr(instr.srcs[1].name, ctx, output)
                    p_tmp = _alloc_scratch_pred(ctx)[0]
                    output.append(SassInstr(encode_dsetp(p_tmp, a_lo, b_lo, DSETP_LEU),
                                            f'DSETP.LEU P{p_tmp}, R{a_lo}, R{b_lo}  // max.f64 cmp'))
                    output.append(SassInstr(encode_fsel(d_lo,   a_lo,   b_lo,   p_tmp, negate_pred=True),
                                            f'FSEL R{d_lo},   R{a_lo},   R{b_lo},   !P{p_tmp}  // max.f64 lo'))
                    output.append(SassInstr(encode_fsel(d_lo+1, a_lo+1, b_lo+1, p_tmp, negate_pred=True),
                                            f'FSEL R{d_lo+1}, R{a_lo+1}, R{b_lo+1}, !P{p_tmp}  // max.f64 hi'))

                elif op == 'shfl':
                    # PTX shfl.sync.<mode>.b32 dst[|p], src, lane, c, mask
                    # When the optional pred dest is present (dst|p), srcs[0] is the pred reg
                    # and srcs[1] is the source register. Without it, srcs[0] is the source.
                    # Subsequent srcs are lane/delta (Imm), clamp (Imm), membermask (Imm).
                    d = ctx.ra.r32(instr.dest.name)
                    # Detect presence of pred dest: first src is a RegOp starting with '%p'
                    if (len(instr.srcs) >= 2 and isinstance(instr.srcs[0], RegOp)
                            and instr.srcs[0].name.startswith('%p')):
                        src_idx = 1
                    else:
                        src_idx = 0
                    a = _materialize_imm(instr.srcs[src_idx], ctx, ctx.ra, output)
                    mode_map = {'idx': SHFL_IDX, 'up': SHFL_UP, 'down': SHFL_DOWN, 'bfly': SHFL_BFLY}
                    mode = SHFL_IDX
                    for t in instr.types:
                        if t in mode_map:
                            mode = mode_map[t]
                    lane = 0
                    clamp = 0x1f
                    # lane is src_idx+1, clamp is src_idx+2
                    if len(instr.srcs) > src_idx + 1 and isinstance(instr.srcs[src_idx + 1], ImmOp):
                        lane = instr.srcs[src_idx + 1].value
                    if len(instr.srcs) > src_idx + 2 and isinstance(instr.srcs[src_idx + 2], ImmOp):
                        clamp = instr.srcs[src_idx + 2].value
                    output.append(SassInstr(encode_shfl(d, a, lane, clamp, mode),
                                            f'SHFL R{d}, R{a}, 0x{lane:x}, 0x{clamp:x}  // shfl.sync'))

                elif op == 'vote':
                    # PTX: vote.sync.ballot.b32 %rD, <pred>, <mask>
                    # <pred> can be a predicate register (%p0..%p7) or immediate 0/1
                    d = ctx.ra.r32(instr.dest.name)
                    pred_num = 7   # default PT (always true)
                    pred_neg = False
                    if len(instr.srcs) >= 1:
                        s0 = instr.srcs[0]
                        if isinstance(s0, RegOp):
                            if s0.name in ctx.ra.pred_regs:
                                pred_num = ctx.ra.pred(s0.name) & 0x07
                                # If the producing setp was inverted (e.g.
                                # setp.lt → ISETP.GE with negated semantics),
                                # the predicate value is logically inverted.
                                if (hasattr(ctx, '_negated_preds')
                                        and pred_num in ctx._negated_preds):
                                    pred_neg = True
                            else:
                                pred_num = 7
                        elif isinstance(s0, ImmOp):
                            # imm 1 → PT (always true); imm 0 → !PT (always false)
                            if s0.value == 0:
                                pred_num = 7
                                pred_neg = True
                            else:
                                pred_num = 7
                                pred_neg = False
                    pred_label = 'PT' if pred_num == 7 else f'P{pred_num}'
                    if pred_neg:
                        pred_label = '!' + pred_label
                    output.append(SassInstr(
                        encode_vote_ballot(d, pred_src=pred_num, neg=pred_neg),
                        f'VOTE.ANY R{d}, PT, {pred_label}  // vote.sync.ballot'))

                elif op == 'div' and typ == 'u32':
                    # Unsigned 32-bit division via Newton-Raphson reciprocal.
                    # Uses only battle-tested encoders (no custom div-specific ones).
                    # Algorithm: rcp approx → scale → NR refine → quotient → correction.
                    d  = ctx.ra.r32(instr.dest.name)
                    a  = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b  = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    t0 = _alloc_gpr(ctx)  # float rcp
                    t1 = _alloc_gpr(ctx)  # NR error / scratch
                    t2 = _alloc_gpr(ctx)  # int approx
                    t3 = _alloc_gpr(ctx)  # remainder
                    # Even-aligned pair for IMAD.WIDE scratch
                    tw = _alloc_gpr(ctx)
                    if tw & 1: tw = _alloc_gpr(ctx)
                    ctx._next_gpr = max(ctx._next_gpr, tw + 2)
                    # Step 1: float reciprocal + scale to integer
                    output.append(SassInstr(encode_i2f_u32_rp(t0, b),
                        f'I2F.U32.RP R{t0}, R{b}  // div.u32: float(divisor)'))
                    output.append(SassInstr(encode_mufu(t0, t0, MUFU_RCP),
                        f'MUFU.RCP R{t0}, R{t0}  // rcp'))
                    output.append(SassInstr(encode_fmul_imm(t0, t0, 0x4F7FFFFE),
                        f'FMUL.IMM R{t0}, R{t0}, 0x4F7FFFFE  // scale rcp'))
                    output.append(SassInstr(encode_f2i_ftz_u32_trunc(t2, t0),
                        f'F2I.FTZ.U32.TRUNC R{t2}, R{t0}  // int approx'))
                    # Step 2: quotient = mulhi(approx, dividend)
                    output.append(SassInstr(
                        encode_imad_wide_u32(tw, t2, a, RZ),
                        f'IMAD.WIDE.U32 R{tw}, R{t2}, R{a}, RZ  // quotient'))
                    output.append(SassInstr(encode_iadd3(d, tw+1, RZ, RZ),
                        f'MOV R{d}, R{tw+1}  // quotient = hi word'))
                    # Step 3: remainder = dividend - quotient * divisor
                    output.append(SassInstr(encode_imad(t3, d, b, RZ),
                        f'IMAD R{t3}, R{d}, R{b}, RZ  // q*divisor'))
                    raw_sub = bytearray(encode_iadd3(t3, a, t3, RZ))
                    raw_sub[7] = 0x80  # negate src1 (b4)
                    output.append(SassInstr(bytes(raw_sub),
                        f'IADD3 R{t3}, R{a}, -R{t3}, RZ  // remainder = a - q*b'))
                    # Step 4: corrections (remainder >= divisor → quotient++)
                    output.append(SassInstr(encode_isetp(0, t3, b, ISETP_GE),
                        f'ISETP.GE.U32 P0, PT, R{t3}, R{b}, PT'))
                    raw_p = bytearray(encode_iadd3_imm32(d, d, 1, RZ))
                    raw_p[1] = (raw_p[1] & 0x0F) | (0x00 << 4)  # @P0
                    output.append(SassInstr(bytes(raw_p),
                        f'@P0 IADD3 R{d}, R{d}, 1, RZ  // correction 1'))
                    raw_s = bytearray(encode_iadd3(t3, t3, b, RZ))
                    raw_s[7] = 0x80  # negate src1
                    raw_s[1] = (raw_s[1] & 0x0F) | (0x00 << 4)  # @P0
                    output.append(SassInstr(bytes(raw_s),
                        f'@P0 IADD3 R{t3}, R{t3}, -R{b}, RZ  // sub divisor'))
                    output.append(SassInstr(encode_isetp(0, t3, b, ISETP_GE),
                        f'ISETP.GE.U32 P0, PT, R{t3}, R{b}, PT'))
                    raw_p2 = bytearray(encode_iadd3_imm32(d, d, 1, RZ))
                    raw_p2[1] = (raw_p2[1] & 0x0F) | (0x00 << 4)  # @P0
                    output.append(SassInstr(bytes(raw_p2),
                        f'@P0 IADD3 R{d}, R{d}, 1, RZ  // correction 2'))

                elif op == 'div' and typ == 's32':
                    # Signed 32-bit division via Newton-Raphson on absolute values.
                    # Matches ptxas sm_120 div.s32 sequence: IABS both operands,
                    # LOP3.XOR to capture sign, NR on |a|/|b|, then sign-correct.
                    # Ground truth: cuobjdump verified against ptxas 13.0 output.
                    d  = ctx.ra.r32(instr.dest.name)
                    a  = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b  = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    t0 = _alloc_gpr(ctx)
                    t1 = _alloc_gpr(ctx)
                    t2 = _alloc_gpr(ctx)
                    t3 = _alloc_gpr(ctx)
                    ab_s = _alloc_gpr(ctx)  # |a| temp / saved |a|
                    sign = _alloc_gpr(ctx)  # sign = a ^ b (bit 31)
                    ppos  = ctx._next_pred; ctx._next_pred += 1  # result is positive
                    pge1  = ctx._next_pred; ctx._next_pred += 1
                    pge2  = ctx._next_pred; ctx._next_pred += 1
                    pnz   = ctx._next_pred; ctx._next_pred += 1  # divisor != 0
                    # Compute |b| in t2 (reuse t2 for NR), |a| saved in ab_s
                    abs_b = _alloc_gpr(ctx)  # |b| for NR
                    output.append(SassInstr(encode_iabs(abs_b, b),
                        f'IABS R{abs_b}, R{b}  // div.s32: |b|'))
                    output.append(SassInstr(encode_iabs(ab_s, a),
                        f'IABS R{ab_s}, R{a}  // div.s32: |a|'))
                    output.append(SassInstr(encode_i2f_s32_rp(t0, abs_b),
                        f'I2F.S32.RP R{t0}, R{abs_b}  // float(|b|) round-up'))
                    _emit_lop3(output, ctx, sign, a, b, RZ, LOP3_XOR, f'LOP3.XOR R{sign}, R{a}, R{b}, RZ  // sign = a^b')
                    output.append(SassInstr(encode_mufu(t0, t0, MUFU_RCP),
                        f'MUFU.RCP R{t0}, R{t0}'))
                    output.append(SassInstr(encode_iadd3_imm32(t1, t0, 0x0ffffffe, RZ),
                        f'IADD3 R{t1}, R{t0}, 0xffffffe, RZ'))
                    output.append(SassInstr(encode_f2i_ftz_u32_trunc(t2, t1),
                        f'F2I.FTZ.U32.TRUNC R{t2}, R{t1}'))
                    output.append(SassInstr(encode_hfma2_zero(t1),
                        f'HFMA2 R{t1}, -RZ, RZ, 0, 0'))
                    output.append(SassInstr(encode_iadd3_neg_b4(t3, RZ, t2, RZ),
                        f'IADD3 R{t3}, RZ, -R{t2}, RZ'))
                    output.append(SassInstr(encode_imad(t3, t3, abs_b, RZ),
                        f'IMAD R{t3}, R{t3}, R{abs_b}, RZ'))
                    # ab_s = |a| (saved), use as dividend for NR
                    output.append(SassInstr(encode_imad_hi(t2, t2, t3, t1),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{t3}, R{t1}'))
                    output.append(SassInstr(encode_imad_hi(t2, t2, ab_s, RZ),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{ab_s}, RZ  // quotient approx'))
                    output.append(SassInstr(encode_iadd3_neg_b3(t3, t2, RZ, RZ),
                        f'IADD3 R{t3}, -R{t2}, RZ, RZ  // negate q'))
                    output.append(SassInstr(encode_imad(t3, abs_b, t3, ab_s),
                        f'IMAD R{t3}, R{abs_b}, R{t3}, R{ab_s}  // remainder'))
                    # Correction: if |b| > remainder, no correction needed
                    output.append(SassInstr(encode_isetp(pge1, abs_b, t3, ISETP_GT),
                        f'ISETP.GT.U32 P{pge1}, PT, R{abs_b}, R{t3}, PT'))
                    output.append(SassInstr(
                        encode_iadd3_pred_neg_b4(t3, t3, abs_b, RZ, pge1, inverted=True),
                        f'@!P{pge1} IADD3 R{t3}, R{t3}, -R{abs_b}, RZ'))
                    output.append(SassInstr(
                        encode_iadd3_pred_small_imm(t2, t2, 1, RZ, pge1, inverted=True),
                        f'@!P{pge1} IADD3 R{t2}, R{t2}, 0x1, RZ'))
                    # Sign check: if sign_bit >= 0 (positive), keep quotient as-is
                    output.append(SassInstr(encode_isetp(ppos, sign, RZ, ISETP_GE, signed=True),
                        f'ISETP.GE.S32 P{ppos}, PT, R{sign}, RZ, PT'))
                    output.append(SassInstr(encode_isetp(pge2, t3, abs_b, ISETP_GE),
                        f'ISETP.GE.U32 P{pge2}, PT, R{t3}, R{abs_b}, PT'))
                    output.append(SassInstr(
                        encode_iadd3_pred_small_imm(t2, t2, 1, RZ, pge2),
                        f'@P{pge2} IADD3 R{t2}, R{t2}, 0x1, RZ'))
                    # Check if divisor is zero
                    output.append(SassInstr(encode_isetp(pnz, b, RZ, ISETP_NE, signed=True),
                        f'ISETP.NE.S32 P{pnz}, PT, R{b}, RZ, PT'))
                    output.append(SassInstr(encode_mov(d, t2),
                        f'MOV R{d}, R{t2}'))
                    # Negate quotient if sign bit indicates negative result
                    output.append(SassInstr(
                        encode_iadd3_pred_neg_b3(d, d, RZ, RZ, ppos, inverted=True),
                        f'@!P{ppos} IADD3 R{d}, -R{d}, RZ, RZ'))
                    # Div-by-zero: result = 0xFFFFFFFF (CUDA signed div-by-zero behavior)
                    output.append(SassInstr(
                        encode_lop3_pred(d, RZ, b, RZ, 0x33, pnz, inverted=True),
                        f'@!P{pnz} LOP3.LUT R{d}, RZ, R{b}, RZ, 0x33  // div-by-zero'))

                elif op == 'rem' and typ == 'u32':
                    # rem.u32 d, a, b = a - (a/b)*b
                    # Uses same Newton-Raphson setup as div.u32 but outputs remainder.
                    # Ground truth: cuobjdump verified against ptxas 13.0 rem.u32 output.
                    d  = ctx.ra.r32(instr.dest.name)
                    a  = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b  = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    t0 = _alloc_gpr(ctx)
                    t1 = _alloc_gpr(ctx)
                    t2 = _alloc_gpr(ctx)
                    t3 = _alloc_gpr(ctx)
                    pnz  = ctx._next_pred; ctx._next_pred += 1
                    pge1 = ctx._next_pred; ctx._next_pred += 1
                    pge2 = ctx._next_pred; ctx._next_pred += 1
                    # NR setup (same as div.u32)
                    output.append(SassInstr(encode_i2f_u32_rp(t0, b),
                        f'I2F.U32.RP R{t0}, R{b}  // rem.u32: float(divisor)'))
                    output.append(SassInstr(encode_isetp(pnz, b, RZ, ISETP_NE),
                        f'ISETP.NE.U32 P{pnz}, PT, R{b}, RZ, PT'))
                    output.append(SassInstr(encode_mufu(t0, t0, MUFU_RCP),
                        f'MUFU.RCP R{t0}, R{t0}'))
                    output.append(SassInstr(encode_iadd3_imm32(t1, t0, 0x0ffffffe, RZ),
                        f'IADD3 R{t1}, R{t0}, 0xffffffe, RZ'))
                    output.append(SassInstr(encode_f2i_ftz_u32_trunc(t2, t1),
                        f'F2I.FTZ.U32.TRUNC R{t2}, R{t1}'))
                    output.append(SassInstr(encode_hfma2_zero(t1),
                        f'HFMA2 R{t1}, -RZ, RZ, 0, 0'))
                    output.append(SassInstr(encode_iadd3_neg_b4(t3, RZ, t2, RZ),
                        f'IADD3 R{t3}, RZ, -R{t2}, RZ'))
                    output.append(SassInstr(encode_imad(t3, t3, b, RZ),
                        f'IMAD R{t3}, R{t3}, R{b}, RZ'))
                    output.append(SassInstr(encode_imad_hi(t2, t2, t3, t1),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{t3}, R{t1}'))
                    output.append(SassInstr(encode_imad_hi(t2, t2, a, RZ),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{a}, RZ  // quotient approx in t2'))
                    # Compute remainder: negate quotient in-place, then IMAD
                    output.append(SassInstr(encode_iadd3_neg_b3(t2, t2, RZ, RZ),
                        f'IADD3 R{t2}, -R{t2}, RZ, RZ  // negate quotient'))
                    output.append(SassInstr(encode_imad(d, b, t2, a),
                        f'IMAD R{d}, R{b}, R{t2}, R{a}  // d = a - q*b = remainder'))
                    # Two correction loops (subtract divisor, not increment quotient)
                    output.append(SassInstr(encode_isetp(pge1, d, b, ISETP_GE),
                        f'ISETP.GE.U32 P{pge1}, PT, R{d}, R{b}, PT'))
                    output.append(SassInstr(encode_iadd3_pred_neg_b4(d, d, b, RZ, pge1),
                        f'@P{pge1} IADD3 R{d}, R{d}, -R{b}, RZ'))
                    output.append(SassInstr(encode_isetp(pge2, d, b, ISETP_GE),
                        f'ISETP.GE.U32 P{pge2}, PT, R{d}, R{b}, PT'))
                    output.append(SassInstr(encode_iadd3_pred_neg_b4(d, d, b, RZ, pge2),
                        f'@P{pge2} IADD3 R{d}, R{d}, -R{b}, RZ'))
                    output.append(SassInstr(encode_lop3_pred(d, RZ, b, RZ, 0x33, pnz, inverted=True),
                        f'@!P{pnz} LOP3.LUT R{d}, RZ, R{b}, RZ, 0x33  // rem of div-by-zero=0xFFFFFFFF'))

                elif op == 'rem' and typ == 's32':
                    # Signed 32-bit remainder via Newton-Raphson on absolute values.
                    # Sign of remainder = sign of dividend (C semantics: a = (a/b)*b + rem).
                    # Ground truth: cuobjdump verified against ptxas 13.0 rem.s32 output.
                    d     = ctx.ra.r32(instr.dest.name)
                    a     = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b     = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    abs_b = _alloc_gpr(ctx)
                    abs_a = _alloc_gpr(ctx)
                    t0    = _alloc_gpr(ctx)
                    t1    = _alloc_gpr(ctx)
                    t2    = _alloc_gpr(ctx)
                    t3    = _alloc_gpr(ctx)
                    pgt1  = ctx._next_pred; ctx._next_pred += 1  # |b| > rem (no correction)
                    psign = ctx._next_pred; ctx._next_pred += 1  # a >= 0
                    pgt2  = ctx._next_pred; ctx._next_pred += 1  # second correction check
                    pnz   = ctx._next_pred; ctx._next_pred += 1  # b != 0
                    output.append(SassInstr(encode_iabs(abs_b, b),
                        f'IABS R{abs_b}, R{b}  // rem.s32: |b|'))
                    output.append(SassInstr(encode_iabs(abs_a, a),
                        f'IABS R{abs_a}, R{a}  // rem.s32: |a|'))
                    output.append(SassInstr(encode_i2f_s32_rp(t0, abs_b),
                        f'I2F.S32.RP R{t0}, R{abs_b}  // float(|b|) round-up'))
                    output.append(SassInstr(encode_mufu(t0, t0, MUFU_RCP),
                        f'MUFU.RCP R{t0}, R{t0}'))
                    output.append(SassInstr(encode_iadd3_imm32(t1, t0, 0x0ffffffe, RZ),
                        f'IADD3 R{t1}, R{t0}, 0xffffffe, RZ'))
                    output.append(SassInstr(encode_f2i_ftz_u32_trunc(t2, t1),
                        f'F2I.FTZ.U32.TRUNC R{t2}, R{t1}'))
                    output.append(SassInstr(encode_hfma2_zero(t1),
                        f'HFMA2 R{t1}, -RZ, RZ, 0, 0'))
                    output.append(SassInstr(encode_iadd3_neg_b4(t3, RZ, t2, RZ),
                        f'IADD3 R{t3}, RZ, -R{t2}, RZ'))
                    output.append(SassInstr(encode_imad(t3, t3, abs_b, RZ),
                        f'IMAD R{t3}, R{t3}, R{abs_b}, RZ'))
                    output.append(SassInstr(encode_imad_hi(t2, t2, t3, t1),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{t3}, R{t1}'))
                    output.append(SassInstr(encode_imad_hi(t2, t2, abs_a, RZ),
                        f'IMAD.HI.U32 R{t2}, R{t2}, R{abs_a}, RZ  // quotient approx'))
                    output.append(SassInstr(encode_iadd3_neg_b3(t2, t2, RZ, RZ),
                        f'IADD3 R{t2}, -R{t2}, RZ, RZ  // negate quotient'))
                    output.append(SassInstr(encode_imad(d, abs_b, t2, abs_a),
                        f'IMAD R{d}, R{abs_b}, R{t2}, R{abs_a}  // rem = |a| + |b|*(-q)'))
                    # Correction 1: if |b| > rem, no subtract needed; else subtract |b|
                    output.append(SassInstr(encode_isetp(pgt1, abs_b, d, ISETP_GT),
                        f'ISETP.GT.U32 P{pgt1}, PT, R{abs_b}, R{d}, PT'))
                    output.append(SassInstr(
                        encode_iadd3_pred_neg_b4(d, d, abs_b, RZ, pgt1, inverted=True),
                        f'@!P{pgt1} IADD3 R{d}, R{d}, -R{abs_b}, RZ'))
                    # Sign check: P=1 if original dividend was non-negative
                    output.append(SassInstr(encode_isetp(psign, a, RZ, ISETP_GE, signed=True),
                        f'ISETP.GE.S32 P{psign}, PT, R{a}, RZ, PT'))
                    # Correction 2: second overshoot check
                    output.append(SassInstr(encode_isetp(pgt2, abs_b, d, ISETP_GT),
                        f'ISETP.GT.U32 P{pgt2}, PT, R{abs_b}, R{d}, PT'))
                    output.append(SassInstr(
                        encode_iadd3_pred_neg_b4(d, d, abs_b, RZ, pgt2, inverted=True),
                        f'@!P{pgt2} IADD3 R{d}, R{d}, -R{abs_b}, RZ'))
                    # Div-by-zero predicate
                    output.append(SassInstr(encode_isetp(pnz, b, RZ, ISETP_NE, signed=True),
                        f'ISETP.NE.S32 P{pnz}, PT, R{b}, RZ, PT'))
                    # Negate remainder if dividend was negative
                    output.append(SassInstr(
                        encode_iadd3_pred_neg_b3(d, d, RZ, RZ, psign, inverted=True),
                        f'@!P{psign} IADD3 R{d}, -R{d}, RZ, RZ'))
                    # Div-by-zero: result = 0xFFFFFFFF
                    output.append(SassInstr(
                        encode_lop3_pred(d, RZ, b, RZ, 0x33, pnz, inverted=True),
                        f'@!P{pnz} LOP3.LUT R{d}, RZ, R{b}, RZ, 0x33  // rem-by-zero'))

                elif op == 'rcp' and any(m in instr.types for m in ('approx','rn','rz','rm','rp')) and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_RCP),
                                            f'MUFU.RCP R{d}, R{a}'))

                elif op == 'sqrt' and any(m in instr.types for m in ('approx','rn','rz','rm','rp')) and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_SQRT),
                                            f'MUFU.SQRT R{d}, R{a}'))

                elif op == 'sin' and 'approx' in instr.types and typ == 'f32':
                    # MUFU.SIN expects input in revolutions (cycles), not radians.
                    # Scale: FMUL dst, src, 1/(2*pi) then MUFU.SIN dst, dst.
                    # 1/(2*pi) = 0x3e22f983 in IEEE754 float.
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_fmul_imm(d, a, 0x3e22f983),
                                            f'FMUL R{d}, R{a}, 0x3e22f983  // radians * 1/(2*pi)'))
                    output.append(SassInstr(encode_mufu(d, d, MUFU_SIN),
                                            f'MUFU.SIN R{d}, R{d}'))

                elif op == 'cos' and 'approx' in instr.types and typ == 'f32':
                    # MUFU.COS expects input in revolutions, not radians.
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_fmul_imm(d, a, 0x3e22f983),
                                            f'FMUL R{d}, R{a}, 0x3e22f983  // radians * 1/(2*pi)'))
                    output.append(SassInstr(encode_mufu(d, d, MUFU_COS),
                                            f'MUFU.COS R{d}, R{d}'))

                elif op == 'ex2' and 'approx' in instr.types and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_EX2),
                                            f'MUFU.EX2 R{d}, R{a}'))

                elif op == 'lg2' and 'approx' in instr.types and typ == 'f32':
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_LG2),
                                            f'MUFU.LG2 R{d}, R{a}'))

                elif op == 'rsqrt' and 'approx' in instr.types and typ == 'f32':
                    # rsqrt = rcp(sqrt(x)) but MUFU has dedicated RSQ function
                    MUFU_RSQ = 0x14  # ptxas SM_120 ground truth
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    output.append(SassInstr(encode_mufu(d, a, MUFU_RSQ),
                                            f'MUFU.RSQ R{d}, R{a}'))

                elif op == 'div' and typ == 'f32':
                    # Float division: MUFU.RCP + FMUL
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                    # temp = rcp(b), result = a * temp
                    output.append(SassInstr(encode_mufu(d, b, MUFU_RCP),
                                            f'MUFU.RCP R{d}, R{b}  // div.f32 step 1'))
                    output.append(SassInstr(encode_fmul(d, a, d),
                                            f'FMUL R{d}, R{a}, R{d}  // div.f32 step 2'))

                elif op == 'prmt':
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    if isinstance(instr.srcs[1], ImmOp):
                        # prmt d, a, sel_imm, c  (selector is 2nd arg, immediate)
                        sel = instr.srcs[1].value
                        c = _materialize_imm(instr.srcs[2], ctx, ctx.ra, output) if len(instr.srcs) > 2 else RZ
                        output.append(SassInstr(encode_prmt(d, a, sel, c),
                                                f'PRMT R{d}, R{a}, 0x{sel:04x}, R{c}'))
                    elif len(instr.srcs) >= 3 and isinstance(instr.srcs[2], ImmOp):
                        # prmt d, a, b, sel_imm  (selector is last arg, immediate)
                        b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                        sel = instr.srcs[2].value
                        output.append(SassInstr(encode_prmt(d, a, sel, b),
                                                f'PRMT R{d}, R{a}, 0x{sel:04x}, R{b}'))
                    elif len(instr.srcs) >= 3:
                        # prmt d, a, b, sel_reg  (all register operands)
                        b = _materialize_imm(instr.srcs[1], ctx, ctx.ra, output)
                        sel_r = _materialize_imm(instr.srcs[2], ctx, ctx.ra, output)
                        output.append(SassInstr(encode_prmt_reg(d, a, b, sel_r),
                                                f'PRMT.REG R{d}, R{a}, R{b}, R{sel_r}'))
                    else:
                        # prmt with < 2 source args — invalid PTX
                        import sys as _sys
                        print(f'WARNING: prmt requires at least 2 source operands, got {len(instr.srcs)}',
                              file=_sys.stderr)
                        output.append(_nop(f'WARNING: prmt invalid operand count: {len(instr.srcs)}'))

                elif op == 'bfe' and typ == 'u32':
                    # Bit field extract: dest = (src >> start) & ((1<<length)-1)
                    # Decomposed as: SHF.R.U32.HI + (optional LDC + LOP3 for masking)
                    d = ctx.ra.r32(instr.dest.name)
                    a = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    start  = instr.srcs[1].value if isinstance(instr.srcs[1], ImmOp) else 0
                    length = instr.srcs[2].value if (len(instr.srcs) > 2 and isinstance(instr.srcs[2], ImmOp)) else 32
                    mask = (1 << length) - 1 if length < 32 else 0xFFFFFFFF
                    if length >= 32:
                        # No masking needed — just shift
                        output.append(SassInstr(
                            encode_shf_r_u32_hi(d, a, start),
                            f'SHF.R.U32.HI R{d}, RZ, 0x{start:x}, R{a}  // bfe.u32 len={length}'))
                    elif start == 0:
                        # Shift is 0 — just mask (LDC d=mask, LOP3 d=src&d)
                        lit_off = ctx._alloc_literal(mask)
                        output.append(SassInstr(
                            encode_ldc(d, 0, lit_off),
                            f'LDC R{d}, c[0][0x{lit_off:x}]  // bfe.u32 mask=0x{mask:x}'))
                        output.append(SassInstr(
                            encode_lop3(d, a, d, RZ, LOP3_AND),
                            f'LOP3.LUT R{d}, R{a}, R{d}, RZ, 0xC0  // bfe.u32 &mask'))
                    else:
                        # General: SHF shift into d, load mask into temp, AND
                        output.append(SassInstr(
                            encode_shf_r_u32_hi(d, a, start),
                            f'SHF.R.U32.HI R{d}, RZ, 0x{start:x}, R{a}  // bfe.u32 >>start'))
                        lit_off = ctx._alloc_literal(mask)
                        t = _alloc_gpr(ctx)
                        output.append(SassInstr(
                            encode_ldc(t, 0, lit_off),
                            f'LDC R{t}, c[0][0x{lit_off:x}]  // bfe.u32 mask=0x{mask:x}'))
                        output.append(SassInstr(
                            encode_lop3(d, d, t, RZ, LOP3_AND),
                            f'LOP3.LUT R{d}, R{d}, R{t}, RZ, 0xC0  // bfe.u32 &mask'))

                elif op == 'bfe' and typ == 's32':
                    # bfe.s32 dest, src, pos, len: sign-extend bits [pos+len-1:pos]
                    # Two-instruction sequence (ptxas ground truth):
                    #   If pos > 0: SHF.R.S32.HI dest, RZ, pos, src
                    #   Then:       BFE_SEXT dest, src_or_dest, len
                    # encode_shf_r_s32_hi already imported at module level
                    d   = ctx.ra.r32(instr.dest.name)
                    a   = _materialize_imm(instr.srcs[0], ctx, ctx.ra, output)
                    pos = instr.srcs[1].value if isinstance(instr.srcs[1], ImmOp) else 0
                    length = instr.srcs[2].value if (len(instr.srcs) > 2 and isinstance(instr.srcs[2], ImmOp)) else 32
                    if pos > 0:
                        output.append(SassInstr(
                            encode_shf_r_s32_hi(d, a, pos),
                            f'SHF.R.S32.HI R{d}, RZ, {pos}, R{a}  // bfe.s32 pos={pos}'))
                        output.append(SassInstr(
                            encode_bfe_sext(d, d, length),
                            f'BFE_SEXT R{d}, R{d}, {length}  // bfe.s32 len={length}'))
                    else:
                        output.append(SassInstr(
                            encode_bfe_sext(d, a, length),
                            f'BFE_SEXT R{d}, R{a}, {length}  // bfe.s32 len={length}'))

                elif op == 'bfi' and typ in ('b32',):
                    # bfi.b32 d, a, b, start, len
                    #   d = (b & ~shifted_mask) | ((a << start) & shifted_mask)
                    # Requires constant start and len (most common case).
                    d  = ctx.ra.r32(instr.dest.name)
                    a  = ctx.ra.r32(instr.srcs[0].name) if isinstance(instr.srcs[0], RegOp) else RZ
                    b  = ctx.ra.r32(instr.srcs[1].name) if isinstance(instr.srcs[1], RegOp) else RZ
                    start = instr.srcs[2].value if len(instr.srcs) > 2 and isinstance(instr.srcs[2], ImmOp) else 0
                    count = instr.srcs[3].value if len(instr.srcs) > 3 and isinstance(instr.srcs[3], ImmOp) else 32
                    raw_mask  = (1 << count) - 1 if count < 32 else 0xFFFFFFFF
                    shifted_mask     = (raw_mask << start) & 0xFFFFFFFF
                    not_shifted_mask = (~shifted_mask) & 0xFFFFFFFF
                    t1 = _alloc_gpr(ctx)
                    t2 = _alloc_gpr(ctx)
                    # t1 = (a << start) & shifted_mask
                    output.append(SassInstr(
                        encode_shf_l_u32(t1, a, start),
                        f'SHF.L.U32 R{t1}, R{a}, 0x{start:x}, RZ  // bfi shift'))
                    lit_sm = ctx._alloc_literal(shifted_mask)
                    output.append(SassInstr(
                        encode_ldc(t2, 0, lit_sm),
                        f'LDC R{t2}, c[0][0x{lit_sm:x}]  // bfi shifted_mask'))
                    output.append(SassInstr(
                        encode_lop3(t1, t1, t2, RZ, LOP3_AND),
                        f'LOP3.LUT R{t1}, R{t1}, R{t2}, RZ, 0xC0  // bfi a&mask'))
                    # t2 = b & ~shifted_mask
                    lit_nsm = ctx._alloc_literal(not_shifted_mask)
                    output.append(SassInstr(
                        encode_ldc(t2, 0, lit_nsm),
                        f'LDC R{t2}, c[0][0x{lit_nsm:x}]  // bfi ~shifted_mask'))
                    output.append(SassInstr(
                        encode_lop3(t2, b, t2, RZ, LOP3_AND),
                        f'LOP3.LUT R{t2}, R{b}, R{t2}, RZ, 0xC0  // bfi b&~mask'))
                    # d = t1 | t2
                    output.append(SassInstr(
                        encode_lop3(d, t1, t2, RZ, LOP3_OR),
                        f'LOP3.LUT R{d}, R{t1}, R{t2}, RZ, 0xFC  // bfi insert'))

                # ---------------------------------------------------------------
                # Texture/surface instructions
                # ---------------------------------------------------------------
                elif op == 'tex':
                    output.extend(_select_tex(instr, ctx))

                elif op == 'tld4':
                    output.extend(_select_tld4(instr, ctx))

                elif op == 'txq':
                    output.extend(_select_txq(instr, ctx))

                elif op == 'suld':
                    output.extend(_select_suld(instr, ctx))

                elif op == 'sust':
                    output.extend(_select_sust(instr, ctx))

                else:
                    # Unrecognized PTX instruction — emit NOP placeholder
                    import sys as _sys
                    print(f'WARNING: unimplemented PTX instruction: {instr.op} '
                          f'{".".join(instr.types)} {instr.mods}', file=_sys.stderr)
                    output.append(_nop(f'WARNING: unimplemented PTX instruction: {instr.op} '
                                       f'{".".join(instr.types)} {instr.mods}'))

            except ISelError as e:
                # Emit NOP with error comment rather than crashing
                output.append(_nop(f'ISEL ERROR: {e}  [{instr.op}]'))

            finally:
                # Release scratch GPRs allocated during this instruction.
                # This reclaims temporaries used by div/rem/mul.hi sequences so
                # subsequent instructions can reuse the same physical registers.
                _release_scratch(ctx)

                # Apply predicate guard to all SASS instructions generated for
                # this PTX instruction (except bra/ret which handle it themselves).
                # LDCU (0x7ac) and S2UR (0x9c3) write to warp-uniform UR registers
                # and MUST NOT be predicated with divergent thread predicates —
                # the hardware ignores or mishandles divergent predicates on UR writes.
                # NOTE: This is in a finally block so that 'continue' statements
                # inside the try block cannot skip predicate application.
                _UR_WRITE_OPCODES = frozenset({0x7ac, 0x9c3})
                if instr.pred and op not in ('bra',):  # ret needs predication for early-exit pattern
                    pd = ctx.ra.pred(instr.pred) if instr.pred in ctx.ra.pred_regs else 0
                    neg = instr.neg
                    # Use the pre-instruction snapshot to determine guard sense.
                    # A predicated setp that writes to its own guard predicate
                    # must not flip the guard with its own inversion.
                    if pd in _neg_preds_snapshot:
                        neg = not neg
                    pred_str = f'@{"!" if neg else ""}P{pd} '
                    for si_idx in range(_pre_len, len(output)):
                        old = output[si_idx]
                        opcode = (old.raw[0] | (old.raw[1] << 8)) & 0xFFF
                        if opcode in _UR_WRITE_OPCODES:
                            continue  # UR-write instrs must be unconditional
                        new_raw = patch_pred(old.raw, pred=pd, neg=neg)
                        output[si_idx] = SassInstr(new_raw, pred_str + old.comment)

        # Tag the first instruction of this block with label marker for BRA fixup.
        # The scheduler may reorder instructions, so the pipeline needs to find
        # labels by scanning comments rather than using body-relative byte offsets.
        if bb.label and block_start_idx < len(output):
            si = output[block_start_idx]
            output[block_start_idx] = SassInstr(si.raw, f'// {bb.label}: {si.comment}')

    # BRA offset fixup: do NOT patch here — the pipeline handles final fixup
    # after preamble insertion and scheduling. We only do fall-through elimination
    # (body-relative) and mark entries as handled so the pipeline skips them.
    if hasattr(ctx, '_bra_fixups'):
        surviving = []
        for bra_idx, target_label in ctx._bra_fixups:
            if target_label in ctx.label_map:
                target_byte = ctx.label_map[target_label]
                bra_byte = (bra_idx + 1) * 16
                rel_offset = target_byte - bra_byte
                if rel_offset == 0:
                    # Fall-through BRA: replace with NOP, don't pass to pipeline
                    output[bra_idx] = SassInstr(encode_nop(),
                        f'NOP  // eliminated fall-through BRA {target_label}')
                    continue
            # Keep this fixup for the pipeline to handle
            surviving.append((bra_idx, target_label))
        ctx._bra_fixups = surviving

    return output
