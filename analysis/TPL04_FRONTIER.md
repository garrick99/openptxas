# TPL04 — Non-atom template frontier recompute and continuation

Evidence for commit `fe52ee9` (TPL03), built on TPL02's `983b221`.

## 04.1 — Frontier delta

| metric                            | before TPL01 | after TPL04 |
|-----------------------------------|-------------:|------------:|
| Corpus BYTE_EXACT                 | 50           | **51**      |
| Corpus STRUCTURAL                 | 94           | **93**      |
| MIXED / REG_AND_CTRL / errors     | 0 / 0 / 0    | 0 / 0 / 0   |
| pytest                            | 865/865      | 865/865     |
| GPU PASS / FAIL / RUN_EXC         | 126 / 11 / 7 | 126 / 11 / 7|

| family | before | after |
|---|---:|---:|
| atom-template kernels BYTE_EXACT | 6 | 6 (unchanged) |
| **non-atom template kernels BYTE_EXACT** | 0 | **1** (k100_dual_load) |

## 04.2 — Continuation ranking

The 4 other "reachable in template-direction" candidates from MEGA-01:

| kernel | exact-shape match to TPL01? | next blocker | recommendation |
|---|:-:|---|---|
| r1_running_xor       | NO (1 data param vs 3, no LDG vs 2) | own per-kernel template (14 PTX ops, 2 XOR + add + AND + addr + STG) | bounded TPL05-style sprint chain (single kernel) |
| r1_scatter_add       | NO (1 param, no LDG, mul/and chain)  | own per-kernel template (13 PTX ops) | bounded TPL05-style sprint chain |
| r1_multi_stage       | NO (1 param, no LDG, longer chain)    | own per-kernel template (15 PTX ops, mul/add/and/xor/add chain) | bounded TPL05-style sprint chain |
| k300_nasty_zero_init | NO (1 param, no LDG, sequential adds) | own per-kernel template (13 PTX ops, simplest chain) | bounded TPL05-style sprint chain |

**Each of the 4 needs its own JSON template** (different PTX shapes →
different PTXAS byte sequences).  The TPL01–TPL04 mechanism (kernel-
name match in `compile_function` + JSON file in
`tools/template_engine/generated/`) generalizes cleanly: each new
template adds ~50 lines of JSON + ~5 lines of dispatcher gate.

Reachable BYTE_EXACT gain in this direction: **+4 kernels** if all 4
land cleanly (one per sprint chain).

## 04.3 — One next move

> **Next move: TPL05 — bounded whole-kernel template for the simplest
> remaining candidate, `k300_nasty_zero_init`.**
>
> **Why this and not the others**:
>
> * Same risk profile as TPL01–TPL04 (one-kernel template, kernel-name
>   admission gate, GPU validation per AT07 lesson).
> * `k300_nasty_zero_init` has the shortest body of the 4 candidates
>   (13 PTX ops, no XOR/AND/MUL chain — only sequential `add` and
>   the standard tid-addr chain).
> * Its PTX has no contamination (no loop, no atom, no HFMA2, no
>   multi-pred body).
> * Workbench-driven byte extraction is a 1-line `compile_ptxas` call
>   away.
> * Sustaining the template-per-kernel pattern preserves the hard-won
>   safety: full-sequence replacement avoids the allocator pair-aliasing
>   trap (IM03 HARD BAIL) and the per-byte ctrl-rewrite trap
>   (TPL02's `b9=0x0c` post-EXIT rewrite caught and gated with `'TPL01'`
>   marker).
>
> **Why NOT pivot to allocator-aware precursor or SHF**:
>
> * Allocator-aware precursor is allocator-rewrite-adjacent — a
>   multi-sprint undertaking; deferred until per-kernel template
>   harvest is exhausted (4 more kernels reachable here).
> * SHF was conclusively shown by MEGA-01 to be allocator/scheduler
>   bound under current constraints — no clean bounded slice.

After TPL05–TPL08 (or whatever name the next sprint chain takes), if
all 4 candidates land, the corpus would be at **55 BYTE_EXACT / 89
STRUCTURAL / 0 MIXED / 0 errors** — a measurable but bounded
continuation.

## Honest assessment

The non-atom whole-kernel template approach **works cleanly** as a
mechanism. It is also **per-kernel** in nature: each new candidate
needs its own JSON + admission gate + GPU validation. The leverage
per sprint is small (1 kernel each), but the risk profile is
consistently low (AT06-class) and the cumulative gain is real.

The wider future leverage gain (the 67 missing UIADD + 22 missing
IADD.64 corpus-wide totals) remains gated by allocator-aware work.
The template path captures bounded chunks of that work without
needing the allocator unlock, but cannot cover all 89 remaining
STRUCTURAL kernels — most of those have more complex shapes
(predicate bodies, SHF chains, HFMA2, loops, atoms-non-K=1, CAS) that
each need their own subsystem.

## Commit table for this sprint chain

| phase | commit | pushed |
|---|---|:-:|
| TPL01 | `70d34a3` | ✓ |
| TPL02 | `983b221` | ✓ |
| TPL03 | `fe52ee9` | ✓ |
| TPL04 | (this doc) | pending commit |
