# TPL12 — Non-atom template frontier recompute and continuation

Evidence for commit `09e70bd` (TPL11), built on TPL10's `348848a`.

## 12.1 — Frontier delta

| metric                              | before TPL09 | after TPL12 |
|-------------------------------------|-------------:|------------:|
| Corpus BYTE_EXACT                   | 52           | **53**      |
| Corpus STRUCTURAL                   | 92           | **91**      |
| MIXED / REG_AND_CTRL / errors       | 0 / 0 / 0    | 0 / 0 / 0   |
| pytest                              | 865/865      | 865/865     |
| GPU PASS / FAIL / RUN_EXC           | 126 / 11 / 7 | 126 / 11 / 7|

| family | before | after |
|---|---:|---:|
| atom-template kernels BYTE_EXACT     | 6 | 6 (unchanged) |
| **non-atom-template kernels BYTE_EXACT** | 2 | **3** (+r1_scatter_add) |

Cumulative non-atom template harvest (TPL01 + TPL05 + TPL09):
**3 kernels**, **+3 BYTE_EXACT**, zero regressions across 12 sprint
phases (3 sprint chains × 4 phases each).

## 12.2 — Continuation ranking

2 remaining template-direction candidates (post-TPL09):

| kernel | exact-shape match to any prior template? | next blocker | recommendation |
|---|:-:|---|---|
| r1_running_xor | NO (1 param, no LDG, XOR + XOR + add + AND chain) | own per-kernel template (~14 PTX ops) | TPL13-style sprint chain — next |
| r1_multi_stage | NO (1 param, no LDG, mul + add + and + xor + add chain) | own per-kernel template (~15 PTX ops) | TPL17-style sprint chain |

The `_TPL_NON_ATOM_REGISTRY` is now stable infrastructure: TPL09
landed via a **single-line registry addition** plus its JSON file,
no other infrastructure changes needed. Reachable BYTE_EXACT gain in
this template direction: **+2 kernels** if both land cleanly.

## 12.3 — One next move

> **Next move: TPL13 — bounded whole-kernel template for `r1_running_xor`.**
>
> **Why this and not r1_multi_stage**:
>
> * Same risk profile as TPL01–TPL12 (one-kernel template, kernel-name
>   admission gate, GPU validation per AT07 lesson).
> * `r1_running_xor` body is ~14 PTX ops (XOR + XOR + add + AND chain)
>   vs `r1_multi_stage` ~15 PTX ops (mul + add + and + xor + add).
>   Both are reasonable, but r1_running_xor is shorter and the XOR ops
>   are well-trodden (already covered for atom kernels and other
>   template paths).
> * Workbench-driven byte extraction is a 1-line `compile_ptxas` call
>   away, exactly as it was for TPL01 / TPL05 / TPL09.
>
> **Why NOT pivot away from templates yet**:
>
> * 3 templates landed cleanly, zero regressions across 12 sprint
>   phases — the non-atom template machinery is now battle-proven.
> * 2 more bounded BYTE_EXACT kernels are still reachable in this exact
>   direction. Each costs ~3 commits (analysis doc + JSON + 1-line
>   registry entry, plus validation doc).
> * Allocator-aware precursor remains explicitly out of bounded scope.
> * SHF was conclusively shown by MEGA-01 to be allocator/scheduler
>   bound — no clean bounded slice today.

After TPL13–TPL16 (the next sprint chain), if r1_running_xor lands,
the corpus would be at **54 BYTE_EXACT / 90 STRUCTURAL**. One more
sprint chain after that (TPL17–TPL20 for r1_multi_stage) would close
out this direction at **55 BYTE_EXACT / 89 STRUCTURAL** — the
bounded ceiling for the template-direction candidates identified in
MEGA-01.

## Honest assessment

The `_TPL_NON_ATOM_REGISTRY` pattern has now been exercised three
times (TPL01 set up the infrastructure; TPL05 added the second entry;
TPL09 landed via a single-line registry addition with no other
infrastructure changes). It is **stable, well-understood, and
low-risk**.

Per-sprint leverage stays at **+1 BYTE_EXACT** (small but consistent),
and per-sprint risk stays at **AT06-class** (low). Cumulative gain
over 3 sprint chains: **+3 BYTE_EXACT** on top of MP03 baseline.

The wider 91-kernel STRUCTURAL frontier remains gated by subsystems
explicitly outside the template path: predicate-body work (MP02
expansion), HFMA2 subsystem, allocator-aware IADD.64 substitution,
loop / CAS / divergent-if subsystems. Templates harvest specific
exact-shape kernels one at a time — they do not unblock those wider
subsystems.

## Commit table for this sprint chain

| phase | commit | pushed |
|---|---|:-:|
| TPL09 | `a197f9b` | ✓ |
| TPL10 | `348848a` | ✓ |
| TPL11 | `09e70bd` | ✓ |
| TPL12 | (this doc) | pending commit |
