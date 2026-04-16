# UI01 — UIADD / UR-family clustering and first-target selection

## 01.1 — Subcluster table

Of the 98 STRUCTURAL kernels, 32 have a missing UR-family opcode (0x835 /
0xc11 / 0xc0c / 0xc24 / 0xc35). They form 28 distinct opcode-diff
signatures — most subclusters are size 1–2.

| subcluster | count | delta | miss (UR family) | extra (R form) | representative kernels |
|---|---:|---:|---|---|---|
| **C01** | **2** | **+0** | **UIADD×1** | **IADD3.I×1** | k100_ldg_add_stg, w1_div_load_paths |
| C02 | 2 | +1 | UIADD×3 | IADD3.I×5 | k200_pred_chain, w1_div_multi_guard |
| C00 | 3 | −5 | ISETP.UR×1 | — | k100_atom_max, k100_atom_min, w2_atom_and_reduce |
| C05 | 1 | +0 | IADD3.UR×2 | IADD3, IMAD.I×2, IMAD.WIDE | ilp_dual_int32 |
| C09 | 1 | +0 | UIADD×2 | IADD3, IADD3.I×2 | k100_pred_arith (multi-pred) |
| C14 | 1 | +0 | UIADD×5 | IADD3, IADD3.I×5 | k300_nasty_multi_pred (multi-pred) |
| C16 | 1 | +0 | UIADD×3 | IADD3, IADD3.I×3 | k300_pred3 (multi-pred) |
| … (21 other size-1 subclusters) | | | | | |

Distinct UR families in the diff set:

| family opcode(s) | total kernels touching it |
|---|---:|
| UIADD (0x835) | 20 kernels, 70 missing instances |
| IADD3.UR (0xc11) | 6 kernels, 14 missing instances |
| ISETP.UR (0xc0c) | 9 kernels, 9 missing instances |
| IMAD.UR (0xc24) | 1 kernel |
| IADD.64-UR (0xc35) | 0 (OURS actually emits +16 extra) |

## 01.2 — Delta=0 prioritization

Delta=0 STRUCTURAL kernels that touch the UR family (24 total delta=0 in
corpus → 6 involve UR substitution only):

| subcluster | kernels | delta=0? | opcode-only? | recommended priority | reason |
|---|---|:-:|:-:|---|---|
| **C01** | 2 | ✓ | ✓ (1:1 IADD3.I→UIADD) | **HIGHEST** | narrowest possible 1:1 swap, no predicate, no loops |
| C09 | 1 | ✓ | ✓ (2:2 + IADD3 shift) | low | multi-predicate (MP02 kernel, risk of re-exposing) |
| C14 | 1 | ✓ | ✓ (5:5 + IADD3 shift) | low | multi-predicate (MP02 kernel) |
| C16 | 1 | ✓ | ✓ (3:3 + IADD3 shift) | low | multi-predicate (MP02 kernel) |
| C05 | 1 | ✓ | mixed (IADD3.UR + IMAD shuffle) | low | different UR opcode + register restructure |

Prohibitions followed: MP02 multi-pred kernels excluded from first target
per "NO predicate canonicalization beyond already-proven behavior" and
"NO touching existing BYTE_EXACT kernels unless exact-shape safety is
proven".

## 01.3 — Uniformity proof surface for C01

Evidence base established by the PTXAS-vs-OURS opcode ledger:

| pattern | required proof | already implemented? | missing piece |
|---|---|:-:|---|
| `add.u32 rd, rs, imm` where rs is SR-derived | tag-propagation through add/xor/and/or/shl/mul | ✓ (UNIF-1, isel.py line 2442-2453) | — |
| `add.u32 rd, rs, imm` where rs is LDG-derived from SR-derived pointer | propagate UR-eligibility through `ld.global.TYPE`  | ✗ | **new bounded rule**: LDG dest inherits UR-eligibility when address operand is UR-eligible |
| Predicated add `@Px add.u32 rd, rd, imm` (self-modify) | _pred_safe predicate already exists | ✓ | — |
| Predicated add `@Px add.u32 rd, rs, imm` (rd != rs) | would need liveness of UR[rd] | ✗ | out of scope for C01 |

**Evidence that the new rule is safe**:
* PTXAS emits UIADD (0x835) **128** times across the 144-kernel corpus.
* PTXAS emits IADD3.IMM (0x810) **5** times — all in loop kernels
  (`w1_loop_countdown`, `w1_loop_two_acc`, `k100_add_sub_chain`).
* The 25 BYTE_EXACT kernels already emitting UIADD show the encoder is
  proven-correct for dest=R0/R5/R7/R10 etc. across varied source registers
  (R0, R3, R4, R5, R6, R9) — no encoding surprises.

**Forbidden forms (UI02 safety proof obligations)**:
* Loop bodies (BRA back-edge detected) — must fall back to IADD3.IMM.
* VOTE-adjacent code (existing `_has_vote` guard).
* BAR.SYNC-adjacent code (existing `_has_bar_sync` guard).
* Multi-predicate bodies (existing MP02 FG33 skip gate).

## 01.4 — Exact first target

| chosen | count | why chosen | why others deferred |
|---|---:|---|---|
| **C01 (LDG→add.u32→STG, UR-eligible address)** | 2 | smallest size-2 cluster, single 1:1 IADD3.I→UIADD substitution, delta=+0, no predicate on add (k100_ldg_add_stg) or single-pred self-modify (w1_div_load_paths), no loops/VOTE/BAR.SYNC | multi-pred kernels (C09/C14/C16) carry MP02 interaction risk; ISETP.UR clusters (C00/C03/C04/C07/C17/C24/C27) require a different subsystem; larger clusters (C11/C06/C18) involve IMAD/IMAD.WIDE restructuring outside the IADD3.I→UIADD scope |

**Target kernels**:
1. `k100_ldg_add_stg` — `add.u32 %r3, %r2, 42` after `ld.global.u32 %r2, [%rd3]`. Not predicated.
2. `w1_div_load_paths` — `@%p1 add.u32 %r2, %r2, 100` after LDG. Predicated self-modify (dest == src) — already `_pred_safe` per existing UNIF-1 rule.

**One-line substitution rule (for UI02/UI03)**:

> When isel currently emits `encode_iadd3_imm32(d, src, imm, RZ)` for
> `add.u32 rd, rs, imm`, emit `encode_uiadd_imm(d, src, imm)` instead iff
> (a) ctx.sm_version == 120,
> (b) no VOTE, no BAR.SYNC, no BRA back-edge,
> (c) `_pred_safe` (pred is None or self-modify), and
> (d) the source operand is UR-eligible — either already SR-derived
>     (existing `_reg_sr_source`) **or** the single-use LDG result of a
>     load whose address operand chain is UR-eligible.

Only (d)'s second disjunct is new work. Everything else already exists.
