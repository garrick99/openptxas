# MPT12 — Final predicate-template frontier and continuation

Evidence for commit `be93ecb` (MPT10+11). Both Slice A and Slice B
of the MPT05–MPT12 chained run landed cleanly.

## 12.1 — Frontier delta

| metric                              | before MPT05 | after MPT12 |
|-------------------------------------|-------------:|------------:|
| Corpus BYTE_EXACT                   | 56           | **58**      |
| Corpus STRUCTURAL                   | 88           | **86**      |
| MIXED / REG_AND_CTRL / errors       | 0 / 0 / 0    | 0 / 0 / 0   |
| pytest                              | 865/865      | 865/865     |
| GPU PASS / FAIL / RUN_EXC           | 127 / 10 / 7 | 127 / 10 / 7|
| Predicate-template kernels BYTE_EXACT | 1          | **3**       |
| Multi-pred predicate-body family    | 11 STRUCT/0 BE | 8 STRUCT / 3 BE |

Cumulative MPT01–MPT12 harvest: **3 kernels**
(k100_pred_arith + k200_double_guard + k300_pred3), zero regressions
across 12 sprint phases.

## 12.2 — Remaining predicate-template ranking

8 remaining MP02-protected multi-pred candidates:

| rank | kernel | shape vs prior MPT templates | next blocker | recommendation |
|---:|---|---|---|---|
| 1 | k100_setp_combo | 2-setp/2-@P (matches MPT01 shape count) but uses gt-16 + gt-8 with P2 reuse, delta=+1 | own JSON | **MPT13-style sprint chain** — closest shape to MPT01 |
| 2 | k200_nested_pred | 2-setp/3-@P (extra @P from merge), delta=−2 | own JSON; might need merge-template variant | medium |
| 3 | r1_minmax | mul + and + 2-setp/2-@P, delta=+4 | own JSON; longer body | medium |
| 4 | k200_pred_chain | 4-setp/4-@P all-on-P1-reuse, delta=+1 | own JSON | medium-low |
| 5 | w1_div_multi_guard | 4-setp/4-@P P1/P2 alternation, delta=+1 | own JSON; sibling of k200_pred_chain | low (sibling) |
| 6 | k300_nasty_pred_nest3 | 3-setp/5-@P nested, delta=−3 | own JSON; nested guards | low |
| 7 | k300_nasty_multi_pred | 5-setp/5-@P P1/P2 alternation, delta=0 | own JSON; longest body but delta=0 | low (longest) |
| 8 | w2_deep_pred | 5-setp/5-@P P1/P2 alternation, delta=+1 | own JSON; sibling of k300_nasty_multi_pred | low (sibling of #7) |

## 12.3 — One next move

> **Next move: MPT13 — bounded predicate-body template for `k100_setp_combo`.**
>
> **Why this and not the others**:
>
> * Same risk profile as MPT01–MPT12 (per-kernel JSON template +
>   single-line registry entry + GPU validation per AT07 lesson).
> * Shape complexity matches MPT01 (2-setp + 2-@P) — the closest
>   PTX-shape neighbour to the proven-clean MPT01 baseline.
> * delta=+1 means OURS only differs from PTXAS by 1 instruction;
>   the template will close that gap byte-exactly.
> * Workbench-driven byte extraction is one `compile_ptxas` call
>   away.
>
> **Why NOT pivot away from predicate templates yet**:
>
> * MPT01 + MPT05 + MPT09 all landed cleanly — 3 of 11 multi-pred
>   kernels harvested with zero regressions.
> * 8 more bounded BYTE_EXACT kernels are still reachable.
> * Allocator-aware precursor and HFMA2 work remain explicitly out
>   of bounded scope.
> * SHF was conclusively shown by MEGA-01 to be allocator/scheduler
>   bound.
>
> **Bounded ceiling for predicate-template direction**: **66 BYTE_EXACT
> / 78 STRUCTURAL** if all 8 remaining kernels land via per-kernel
> templates. Current pace: 1 kernel per ~3 commits.

## Cumulative project status

| metric | MP03 baseline | post-MPT12 |
|---|---:|---:|
| BYTE_EXACT | 46 | **58** (+12) |
| STRUCTURAL | 98 | **86** (−12) |
| GPU PASS | 126 | **127** (+1, from TPL13's r1_running_xor) |
| pytest | 865/865 | 865/865 (unchanged) |
| MIXED / errors | 0 / 0 | 0 / 0 |

12 BYTE_EXACT kernels harvested via the template path across 36
sprint phases (AT01–AT12 + TPL01–TPL20 + MPT01–MPT12), all with
**zero regressions**. The whole-kernel template approach now stands
as the project's most reliable correctness-preserving mechanism for
moving STRUCTURAL → BYTE_EXACT one kernel at a time.

## Commit table for this sprint chain

| phase | commit | pushed |
|---|---|:-:|
| MPT05 | `fad083f` | ✓ |
| MPT06+MPT07 (consolidated) | `245c3e4` | ✓ |
| MPT08 (gate) | `fe9f6fa` | ✓ |
| MPT09 | `811390f` | ✓ |
| MPT10+MPT11 (consolidated) | `be93ecb` | ✓ |
| MPT12 | (this doc) | pending commit |
