# UI04 — UIADD frontier recompute + continuation

Evidence for commit `fc6c617` (post-UI03).

## 04.1 — Frontier delta

Corpus: 144 kernels.

| metric                           | before UI01 | after UI04 |
|----------------------------------|------------:|-----------:|
| BYTE_EXACT                       | 46          | 46         |
| STRUCTURAL                       | 98          | 98         |
| MIXED                            | 0           | 0          |
| REG_AND_CTRL                     | 0           | 0          |
| errors                           | 0           | 0          |
| pytest                           | 865 / 865   | 865 / 865  |
| GPU PASS                         | 126         | 126        |
| GPU FAIL / RUN_EXC               | 11 / 7      | 11 / 7     |
| STRUCTURAL with UR-family gap    | 32          | 29         |
| missing UIADD (0x835)            | 70          | **67**     |
| missing IADD3.UR (0xc11)         | 14          | 14         |
| missing ISETP.UR (0xc0c)         | 9           | 9          |
| missing IMAD.UR (0xc24)          | 1           | 1          |
| **total missing UR-family**      | **94**      | **91**     |
| extra IADD3.IMM (0x810)          | 99          | **96**     |

**Net UI01–UI03 impact on bytes**: −3 missing UIADD, −3 extra IADD3.IMM.
Two kernels (k100_ldg_add_stg, w1_div_load_paths) now emit UIADD at the
correct position (byte-for-byte match with PTXAS on that instruction).
BYTE_EXACT count unchanged because these kernels have additional unrelated
STRUCTURAL differences (register numbering, LDCU ordering) outside the
UIADD scope.

No new REG_AND_CTRL. No new MIXED. No new errors.

## 04.2 — Continuation ranking

Top 12 remaining UR-family subclusters (size-ordered):

| rank | subcluster | kernels | size | delta | missing | extra | next blocker |
|-----:|------------|---------|-----:|------:|---------|-------|--------------|
| 1 | C00 | k100_atom_max, k100_atom_min, w2_atom_and_reduce | 3 | −5 | ISETP.UR | — | atom + reduce restructure; no opcode-substitution only |
| 2 | C01′ | k200_pred_chain, w1_div_multi_guard | 2 | +1 | UIADD×3 | IADD3.I×5 | multi-pred (MP02 guarded) |
| 3 | C02 | atom_or | 1 | −4 | ISETP.UR | IADD3.I | atom kernel |
| 4 | C03 | atomg_add | 1 | −4 | ISETP.UR, IMAD.UR | — | atom kernel |
| 5 | C04 | ilp_dual_int32 | 1 | +0 | IADD3.UR×2 | IADD3, IMAD.I×2, IMAD.WIDE | IMAD.WIDE restructure |
| 6 | C06 | k100_atom_add | 1 | −3 | ISETP.UR | IADD3.I | atom kernel |
| 7 | C08 | k100_pred_arith | 1 | +0 | UIADD×2 | IADD3, IADD3.I×2 | multi-pred (MP02) |
| 8 | C09 | k100_setp_combo | 1 | +1 | UIADD | IADD3.I×3 | multi-pred (MP02) |
| … | (18 size-1 subclusters) | | | | | | |

**Two readings of "remaining UIADD gap"**:

* Total `missing UIADD`: 67 instances across 20 kernels.
* **Kernels eligible for the same bounded scope as UI01–UI03**
  (non-multi-pred, no IMAD-WIDE coordination, no atom-reduction
  restructure, delta ≤ +1, no predicate hazards): **0**.

The easy UIADD wins have been extracted. Every remaining UIADD subcluster
touches at least one of:
1. MP02 multi-pred territory (prohibited by UI01 scope)
2. IMAD/IMAD.WIDE coordination (would need a separate isel subsystem)
3. atom-reduction restructure (different family)
4. ISETP.UR transitions (different opcode family)

## 04.3 — Roadmap answers

**1. Did the first UIADD isel-level slice land cleanly?**
Yes. 2/2 target kernels now emit UIADD at the expected byte position.
Zero pytest regressions, zero frontier-class changes, zero GPU harness
regressions, zero unrelated drift. The isel-level approach is proven.

**2. How many kernels improved?**
Two kernels materially (k100_ldg_add_stg, w1_div_load_paths) gained a
correct UIADD emission. Frontier bucket counts are unchanged because the
kernels remain STRUCTURAL from *other* unrelated diffs — those are
outside the UI01–UI03 bounded scope and require separate subsystems.

**3. Is UIADD still the highest-leverage next continuation?**
**Partially.** UIADD is still the single largest missing-UR-family
opcode by absolute count (67 instances). But within the "bounded, exact
shape, non-multi-pred, no-IMAD-coordination" slice the UI01 rules
established, **no new UIADD subcluster remains eligible**. Further
UIADD wins require explicitly expanding scope into one of:
- MP02 multi-pred territory (risk: re-exposing predicate hazards);
- IMAD/IMAD.WIDE coordinated substitution (new subsystem);
- atom-reduction family restructure (different subsystem).

## 04.4 — One next move

> **Next move: pause UIADD for the ISETP.UR subsystem precursor analysis.**
>
> **Because**:
> * UIADD's bounded-shape inventory is exhausted — every remaining
>   UIADD subcluster requires scope expansion explicitly out of bounds
>   for this run.
> * The SHF harvest alternative is prohibited by this run's rules.
> * ISETP.UR (0xc0c) is the **second-largest** UR-family gap (9 missing
>   instances across 5 subclusters — C00, C02, C03, C06, plus
>   multi_block_atomic, w2_loop_atom_add, w2_pred_load).
> * ISETP.UR substitution is **orthogonal** to the MP02 predicate-hazard
>   gate: ISETP.UR reads a uniform source at b4 (not an immediate at
>   b4-b7), so the WAW predicate-reuse pattern that caused MP02 does
>   not apply here.
> * The atom-adjacent cluster (C00+C02+C03+C06) is not in the MP02-
>   guarded multi-pred set — no cross-gate risk.
>
> **Scope for the next sprint (pattern-matching UI01–UI03 discipline)**:
> precursor cluster → bounded UR-eligibility analysis → isel-level
> ISETP.IMM→ISETP.UR substitution → frontier recompute.

## Commit table

| phase | commit | pushed |
|---|---|:-:|
| UI01 | 02c5f35 | ✓ |
| UI02 | e76aad3 | ✓ |
| UI03 | fc6c617 | ✓ |
| UI04 | (this doc) | pending commit |
