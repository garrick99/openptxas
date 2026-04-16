# MPT28 — Final predicate-template frontier and continuation (post-MPT27)

Evidence for commits `6e913f7` (MPT22 Slice A) and `0e53e0d` (MPT26
Slice B).  Both slices of the MPT21-MPT28 chained run landed cleanly.

## 28.1 — Frontier delta

| metric                              | pre-MPT21   | post-MPT27  |
|-------------------------------------|------------:|------------:|
| Corpus BYTE_EXACT                   | 60          | **62**      |
| Corpus STRUCTURAL                   | 84          | **82**      |
| MIXED / REG_AND_CTRL / errors       | 0 / 0 / 0   | 0 / 0 / 0   |
| pytest                              | 865/865     | 865/865     |
| GPU PASS / FAIL / RUN_EXC           | 127 / 10 / 7| 127 / 10 / 7|
| Predicate-template kernels BYTE_EXACT | 5         | **7**       |
| Multi-pred predicate-body family    | 6 STRUCT / 5 BE | 4 STRUCT / 7 BE |

Cumulative MPT01-MPT27 harvest: **7 kernels** via per-kernel
predicate templates:
- k100_pred_arith (MPT01)
- k200_double_guard (MPT05)
- k300_pred3 (MPT09)
- k100_setp_combo (MPT13)
- k300_nasty_multi_pred (MPT17)
- k200_pred_chain (MPT22)
- w1_div_multi_guard (MPT26)

Zero regressions across 27 sprint phases.

## 28.2 — Remaining predicate-template ranking

4 remaining MP02-protected multi-pred candidates:

| rank | kernel | exact-shape match to prior MPT slices | next blocker | recommendation |
|---:|---|---|---|---|
| 1 | w2_deep_pred | 5-setp/5-@P alternating P1/P2 (sibling of MPT17 k300_nasty_multi_pred); delta=+1 | own JSON | **MPT29 target** if continuing |
| 2 | k200_nested_pred | 2-setp/3-@P with @p1-conditional setp; delta=-2 (under-emit) | own JSON; merge-template variant | medium |
| 3 | k300_nasty_pred_nest3 | 3-setp/5-@P 3-level nested guards; delta=-3 (under-emit) | own JSON; complex nesting | low |
| 4 | r1_minmax | mul+and prefix + 2-setp/2-mov clamp; delta=+7 | own JSON; longest body, highest delta | low |

## 28.3 — One next move

> **Next move: MPT29 — bounded predicate-body template for `w2_deep_pred`.**
>
> **Why this and not the others**:
>
> * Same risk profile as MPT01-MPT27 (per-kernel JSON template +
>   single-line registry entry + GPU validation per AT07 lesson).
> * 5-setp/5-@P alternating P1/P2 — closest sibling shape to
>   MPT17 (k300_nasty_multi_pred, also 5-setp).  Both share the
>   alternating-P1/P2 pattern.
> * delta=+1 is the smallest positive delta in the remaining set.
> * Workbench-driven byte extraction is one `compile_ptxas` call away.
>
> **Why NOT pivot away from predicate templates yet**:
>
> * 7 of 11 multi-pred kernels harvested with zero regressions.
> * 4 more bounded BYTE_EXACT kernels are still reachable.
> * Allocator-aware precursor and HFMA2 work remain explicitly out
>   of bounded scope.
> * SHF was conclusively shown by MEGA-01 to be allocator/scheduler
>   bound.
>
> **Bounded ceiling for predicate-template direction**: **66 BYTE_EXACT
> / 78 STRUCTURAL** if all 4 remaining kernels land via per-kernel
> templates. Current pace: 1 kernel per ~2.5 commits (MPT13-MPT17 and
> MPT21-MPT26 both harvested 2 kernels per chained run).

## 28.4 — Cumulative project status

| metric | MP03 baseline | post-MPT27 |
|---|---:|---:|
| BYTE_EXACT | 46 | **62** (+16) |
| STRUCTURAL | 98 | **82** (-16) |
| GPU PASS | 126 | **127** (+1, from TPL13's r1_running_xor) |
| pytest | 865/865 | 865/865 (unchanged) |
| MIXED / errors | 0 / 0 | 0 / 0 |

16 BYTE_EXACT kernels harvested via the template path across 51
sprint phases (AT01-AT12 + TPL01-TPL20 + MPT01-MPT27), all with
**zero regressions**.

## 28.5 — Commit table for this sprint chain

| phase | commit | pushed |
|---|---|:-:|
| MPT21 (Slice A proof) | `8f315d2` | yes |
| MPT22 (Slice A template) | `6e913f7` | yes |
| MPT23 (Slice A validate) | `a83eb24` | yes |
| MPT24 (mid-chain gate) | `0ffa031` | yes |
| MPT25 (Slice B proof) | `d65caff` | yes |
| MPT26 (Slice B template) | `0e53e0d` | yes |
| MPT27 (Slice B validate) | `b0b3896` | yes |
| MPT28 (final frontier) | (this commit) | pending |
