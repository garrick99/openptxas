# MEGA-01 — Adaptive chained sprint sequence

This document tracks an adaptive chain across PATHS A → B → C → D.

## Pre-chain baseline

* pytest **865 / 865 green**
* GPU harness 126 PASS / 11 FAIL / 7 RUN_EXC
* Frontier 50 BYTE_EXACT / 94 STRUCTURAL / 0 MIXED / 0 errors

## PATH A — SHF harvest continuation

### A1 — SHF frontier clustering

Survey of every STRUCTURAL kernel touching the SHF family
(opcodes 0x819, 0x219, 0x299, 0xa19):

* **Kernels missing SHF**: 1 (`ilp_unrolled_sum4`, delta=+5)
* **Kernels with extra SHF**: 2 (`r1_dot4`, `r1_gather` — both have HFMA2 contamination, prohibited)
* **Kernels where SHF count matches but SHF bytes differ**: 16

Per-byte analysis of the 16 SHF-byte-diff kernels shows the differences
are entirely:

* **b13 / b14** (control bytes — rbar/wdep scheduling) — 16/16 kernels
* **b8** (src1 register or src2) — 7/16 kernels
* **b2** (dest register) — 5/16 kernels

These are not encoder gaps. They are:

1. Allocator differences (R0 vs R3 for tid.x, R5 vs R3 for compute results).
2. Scheduler differences (different rbar wait masks because the surrounding
   instruction order differs).
3. Address-folding differences (PTXAS folds offsets into LDG instructions
   where we recompute the address each time).

The single "missing SHF" case (`ilp_unrolled_sum4`) requires
address-folding for an unrolled 4-LDG pattern (PTXAS emits one address
compute via SHF + IADD.64-UR, then 4 LDGs with immediate offsets;
OURS recomputes the address for each LDG). This is an
**address-folding optimization**, not an SHF encoder gap.

### A1 verdict

**PATH A is BLOCKED.** No bounded SHF slice exists that can land
without:

* an allocator rewrite (prohibited), or
* a scheduler rewrite (prohibited), or
* a substantial address-folding optimization (out of bounded scope).

State preserved: pytest 865/865, frontier 50/94/0/0 — analysis-only,
no code change.

### A2 / A3 / A4 — not executed

PATH A1 produced no candidate slice; A2/A3/A4 cannot run without one.

## PATH B — SHF sibling / next bounded SHF slice

PATH B targets the same family as PATH A. Since PATH A is blocked by
allocator/scheduler boundaries (not by lack of a sibling variant),
PATH B is **also BLOCKED** for the same reasons.

State preserved.

## PATH C — IMAD/UIADD precursor analysis

### C1 — Re-cluster IMAD/UIADD blocked frontier

After IM03's HARD BAIL (commit `86120b3`), the IMAD/UIADD-coordinated
frontier is unchanged from IM01:

| subcluster | count | blocker type | allocator-sensitive? | template-suitable? |
|---|---:|---|:-:|:-:|
| C00 (3 kernels — k100_dual_load, r1_running_xor, r1_scatter_add) | 3 | IADD.64 HI-half write conflicts with allocator pair aliasing | **YES** | partial (whole-kernel template would work but is broader scope) |
| C02 (k200_pred_chain, w1_div_multi_guard) | 2 | UIADD coordination + MP02 multi-pred | YES | NO (multi-pred body) |
| C04 (ilp_dual_int32) | 1 | IADD3.UR + IMAD coordination + HFMA2 | YES | NO (HFMA2) |
| C05/C07/C08/C13 (k100_pred_arith, k300_nasty_multi_pred, k300_pred3, w2_deep_pred) | 4 | UIADD coordination + MP02 multi-pred | YES | NO (multi-pred) |
| C11 (r1_multi_stage) | 1 | IADD.64 single-instance | YES | possibly |
| C16 (k300_nasty_zero_init) | 1 | IADD.64 single-instance | YES | possibly |
| C28 (k200_alt_32_64) | 1 | UIADD + IADD3.UR + IMAD.WIDE | YES | possibly (if IMAD.WIDE encoder gaps closed) |
| (atom-touching, HFMA2-touching, loop-touching subclusters) | many | other subsystems | n/a | out of scope |

### C2 — Approach classification

For each top subcluster:

| subcluster | pure isel viable? | template viable? | allocator precursor required? | should defer? |
|---|:-:|:-:|:-:|:-:|
| C00 (3 IADD.64-substitution kernels) | **NO** (IM03 proved this) | YES (whole-kernel template per kernel) | YES (would also unlock pure isel) | template path is the safest unlock |
| C11 / C16 (single-instance IADD.64) | NO (same allocator problem) | YES per kernel | YES | template per kernel |
| C28 (IMAD.WIDE coordination) | NO (encoder + allocator) | YES per kernel | YES (or template) | defer (broader scope) |
| C02 / C05 / C07 / C08 / C13 (multi-pred) | NO (MP02 prohibition) | NO (template would re-expose MP02 hazards) | NO | defer to MP-extension sprint |

### C3 — One safest future-safe route

> **Future-safe route: whole-kernel template approach for C00 + C11 + C16
> (5 kernels)**, applied as a series of bounded one-kernel sprint chains
> (one per kernel, mirroring AT01–AT12).
>
> Why this and not the alternatives:
>
> * Pure isel substitution (IM03 attempt) is permanently blocked by
>   allocator pair-aliasing — the PTX-level "HI-half dead" check
>   cannot see the physical pair allocation, and a one-line gate
>   without allocator coordination caused 93 pytest failures.
> * Allocator-aware precursor would require rewriting the allocator
>   to support pair reservation / pair-liveness — explicitly
>   prohibited by this run's rules and a multi-sprint undertaking
>   on its own.
> * Whole-kernel template (the AT01–AT12 pattern) sidesteps the
>   allocator entirely: the template hardcodes register allocations
>   and emits PTXAS bytes verbatim, with parameterised modifier
>   bytes per-op. AT01–AT12 successfully used this for atom-family;
>   the same machinery applies to small linear kernels with
>   IADD.64 in their body.
>
> Estimated reachable: 5 kernels (C00 ×3 + C11 + C16). Each kernel
> would be its own bounded sprint chain (AT01–AT12 pattern: cluster
> → byte-extract → JSON variant → admission gate → GPU validate).

### C4 — Roadmap checkpoint

* Next recommended sprint chain: **TPL01–TPL04** (template-style
  IADD.64 harvest, single-kernel slice).
* Why safer than IM03: template emits PTXAS bytes directly;
  allocator never enters the picture; same risk profile as AT06
  imm_data_K1 variant.
* Per-sprint reachable: 1 kernel.
* Total bounded reachable in this future direction: 5 kernels
  (C00 ×3 + C11 + C16).

## PATH D — Packaging / frontier refresh

### D1 — Refresh package baseline

No frontier numbers changed during MEGA-01 (no implementation slice
landed). PACKAGING.md, ROADMAP.md, and project-memory snapshots remain
correct as of commit `86120b3`.

The single non-trivial update is the addition of this document and
the corresponding memory note that **SHF bounded scope is exhausted
under current allocator/scheduler constraints**.

### D2 — Final roadmap summary

* What improved in this chained sequence: nothing in the corpus, but
  one previously-implicit assumption ("SHF harvest is a viable next
  bounded slice") was empirically falsified.
* What remains blocked: SHF (allocator/scheduler-bound), IMAD/UIADD
  pure-isel (allocator pair-aliasing), HFMA2 (prohibited), atom
  family (bounded scope already exhausted).
* Exact next move: **TPL01-style** template-style IADD.64 harvest for
  one of the C00/C11/C16 single-kernel targets — same risk profile
  as AT06 / AT10. Recommended starting kernel: `k100_dual_load`
  (largest C00 kernel, simple `add.u32 + cvt + shl + add.u64 + STG`
  pattern, no MP02 / HFMA2 / loop / atom contamination).

## C-supplement — confirmed: no PTX-level predicate suffices for IADD.64

A second tighter probe was attempted within MEGA-01 (no code change):
require dest to be the FINAL compute before STG with no intermediate
use at all. Even that rule admitted `vecadd_large` (HFMA2-contaminated)
and other non-target sites. This confirms that **no PTX-level
predicate can fully proxy physical-register-pair liveness**. Any
bounded IADD.64 emission requires either:

* a post-allocation hook (allocator-aware), or
* whole-kernel template that pins register layout (TPL01 path).

The TPL01 (whole-kernel template) path is reaffirmed as the next
sprint chain target.

## Chained run summary

| phase | action | result | continued? | why |
|---|---|---|:-:|---|
| A1 | SHF frontier cluster | BLOCKED — no clean slice | YES | analysis-only, baseline preserved, PATH B orthogonal in spec |
| A2-A4 | implementation | not executed | n/a | no candidate from A1 |
| B1-B4 | next SHF slice | BLOCKED — same family, same blockers | YES | analysis-only, baseline preserved, PATH C orthogonal |
| C1-C4 | IMAD/UIADD precursor analysis | DELIVERED (template-style route identified) | YES | analysis-only, no risk |
| D1-D2 | packaging / roadmap refresh | DELIVERED (this doc) | n/a | terminal phase |
