# MPT20 — Final predicate-template frontier and continuation (post-MPT17)

Evidence for commits `88d642f` (MPT13 Slice A) and the upcoming
MPT17 Slice B commit.  Both slices of the MPT13–MPT20 chained run
landed cleanly.

## 20.1 — Frontier delta

| metric                              | pre-MPT13   | post-MPT17  |
|-------------------------------------|------------:|------------:|
| Corpus BYTE_EXACT                   | 58          | **60**      |
| Corpus STRUCTURAL                   | 86          | **84**      |
| MIXED / REG_AND_CTRL / errors       | 0 / 0 / 0   | 0 / 0 / 0   |
| pytest                              | 865/865     | 865/865     |
| GPU PASS / FAIL / RUN_EXC           | 127 / 10 / 7| 127 / 10 / 7|
| Predicate-template kernels BYTE_EXACT | 3         | **5**       |
| Multi-pred predicate-body family    | 8 STRUCT / 3 BE | 6 STRUCT / 5 BE |

Cumulative MPT01–MPT17 harvest: **5 kernels**
(k100_pred_arith + k200_double_guard + k300_pred3 + k100_setp_combo
+ k300_nasty_multi_pred), zero regressions across 17 sprint phases.

## 20.2 — Remaining predicate-template ranking

6 remaining MP02-protected multi-pred candidates:

| rank | kernel | shape vs prior MPT templates | next blocker | recommendation |
|---:|---|---|---|---|
| 1 | k200_pred_chain | 4-setp/4-@P all-on-P1-reuse, delta=+1 | own JSON | **MPT21-style sprint** — small delta, simple reuse pattern |
| 2 | w1_div_multi_guard | 4-setp/4-@P P1/P2 alternation, delta=+1 | own JSON; sibling of k200_pred_chain | medium |
| 3 | w2_deep_pred | 5-setp/5-@P P1/P2 alternation, delta=+1 | own JSON; sibling of MPT17 (k300_nasty_multi_pred) | medium |
| 4 | k200_nested_pred | 2-setp/3-@P (extra @P from merge), delta=−2 | own JSON; merge-template variant | medium-low |
| 5 | k300_nasty_pred_nest3 | 3-setp/5-@P nested, delta=−3 | own JSON; nested guards | low |
| 6 | r1_minmax | mul + and + 2-setp/2-@P, delta=+7 | own JSON; longest body | low |

## 20.3 — One next move

> **Next move: MPT21 — bounded predicate-body template for
> `k200_pred_chain`.**
>
> **Why this and not the others**:
>
> * Same risk profile as MPT01–MPT17 (per-kernel JSON template +
>   single-line registry entry + GPU validation per AT07 lesson).
> * 4-setp/4-@P with all-on-P1-reuse — closest neighbour to
>   k200_double_guard (MPT05) which used 2-setp on P1/P2 alternating.
>   k200_pred_chain extends that pattern to 4 producers, all reusing
>   P1.
> * delta=+1 means OURS only differs from PTXAS by 1 instruction;
>   the template will close that gap byte-exactly.
> * Workbench-driven byte extraction is one `compile_ptxas` call away.
>
> **Why NOT pivot away from predicate templates yet**:
>
> * MPT01 + MPT05 + MPT09 + MPT13 + MPT17 all landed cleanly — 5 of
>   11 multi-pred kernels harvested with zero regressions.
> * 6 more bounded BYTE_EXACT kernels are still reachable.
> * Allocator-aware precursor and HFMA2 work remain explicitly out
>   of bounded scope.
> * SHF was conclusively shown by MEGA-01 to be allocator/scheduler
>   bound.
>
> **Bounded ceiling for predicate-template direction**: **66 BYTE_EXACT
> / 78 STRUCTURAL** if all 6 remaining kernels land via per-kernel
> templates.  Current pace: 1 kernel per ~2.5 commits (MPT13–MPT17
> chain harvested 2 in one chained run).

## 20.4 — Cumulative project status

| metric | MP03 baseline | post-MPT17 |
|---|---:|---:|
| BYTE_EXACT | 46 | **60** (+14) |
| STRUCTURAL | 98 | **84** (−14) |
| GPU PASS | 126 | **127** (+1, from TPL13's r1_running_xor) |
| pytest | 865/865 | 865/865 (unchanged) |
| MIXED / errors | 0 / 0 | 0 / 0 |

14 BYTE_EXACT kernels harvested via the template path across 41
sprint phases (AT01–AT12 + TPL01–TPL20 + MPT01–MPT17), all with
**zero regressions**.  The whole-kernel template approach now stands
as the project's most reliable correctness-preserving mechanism for
moving STRUCTURAL → BYTE_EXACT one kernel at a time.

## 20.5 — Commit table for this sprint chain

| phase | commit | pushed |
|---|---|:-:|
| MPT13 (Slice A — k100_setp_combo) | `88d642f` | ✓ |
| MPT17 (Slice B — k300_nasty_multi_pred) + MPT20 doc | (this commit) | pending |
