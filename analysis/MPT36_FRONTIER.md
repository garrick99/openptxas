# MPT36 — Final predicate-template frontier and continuation (post-MPT35)

Evidence for commits `6659d34` (MPT30 Slice A) and `ec9c61c` (MPT34
Slice B). Both slices of the MPT29-MPT36 chained run landed cleanly.

## 36.1 — Frontier delta

| metric                              | pre-MPT29   | post-MPT35  |
|-------------------------------------|------------:|------------:|
| Corpus BYTE_EXACT                   | 62          | **64**      |
| Corpus STRUCTURAL                   | 82          | **80**      |
| MIXED / REG_AND_CTRL / errors       | 0 / 0 / 0   | 0 / 0 / 0   |
| pytest                              | 865/865     | 865/865     |
| GPU PASS / FAIL / RUN_EXC           | 127 / 10 / 7| 127 / 10 / 7|
| Predicate-template kernels BYTE_EXACT | 7         | **9**       |
| Multi-pred predicate-body family    | 4 STRUCT / 7 BE | 2 STRUCT / 9 BE |

Cumulative MPT01-MPT35 harvest: **9 kernels** via per-kernel
predicate templates:
- k100_pred_arith (MPT01)
- k200_double_guard (MPT05)
- k300_pred3 (MPT09)
- k100_setp_combo (MPT13)
- k300_nasty_multi_pred (MPT17)
- k200_pred_chain (MPT22)
- w1_div_multi_guard (MPT26)
- w2_deep_pred (MPT30)
- k200_nested_pred (MPT34)

Zero regressions across 35 sprint phases.

Side fix landed in MPT34: FG-2.3 INV B coverage allowlist updated for
0x81c (template-only opcode).

## 36.2 — Remaining predicate-template ranking

2 remaining MP02-protected multi-pred candidates:

| rank | kernel | exact-shape match to prior MPT slices | next blocker | recommendation |
|---:|---|---|---|---|
| 1 | k300_nasty_pred_nest3 | 3-setp/5-@P 3-level nesting; delta=-3 (under-emit) | own JSON; deeper conditional-setp chain than MPT34 | **MPT37 target** if continuing |
| 2 | r1_minmax | mul+and prefix + 2-mov clamp; delta=+7 | own JSON; entirely different family pattern | medium-low |

## 36.3 — One next move

> **Next move: MPT37 — bounded predicate-body template for
> `k300_nasty_pred_nest3`.**
>
> **Why this and not the others**:
>
> * Same risk profile as MPT01-MPT35 (per-kernel JSON template +
>   single-line registry entry + GPU validation per AT07 lesson).
> * Direct extension of MPT34's @P-conditional-setp pattern: this
>   kernel uses TWO conditional setps (@p1 setp p2, then @p2 setp p1)
>   plus three ordinary @P-add bodies, vs MPT34's single conditional
>   setp.
> * MPT34's ISETP-with-@P-guard byte encoding learning directly
>   applies; the 0x81c FG-2.3 allowlist also already in place.
> * delta=-3 (under-emit) — same direction as MPT34's delta=-2,
>   meaning OURS is missing the conditional-setp expansion that the
>   template will close byte-exactly.
>
> **Why NOT pivot away from predicate templates yet**:
>
> * 9 of 11 multi-pred kernels harvested with zero regressions.
> * 2 more bounded BYTE_EXACT kernels are still reachable (one a
>   direct extension of MPT34).
> * Allocator-aware precursor and HFMA2 work remain explicitly out
>   of bounded scope.
> * SHF was conclusively shown by MEGA-01 to be allocator/scheduler
>   bound.
>
> **Bounded ceiling for predicate-template direction**: **66 BYTE_EXACT
> / 78 STRUCTURAL** if both remaining kernels land via per-kernel
> templates. Current pace: 1 kernel per ~2.5 commits.

## 36.4 — Cumulative project status

| metric | MP03 baseline | post-MPT35 |
|---|---:|---:|
| BYTE_EXACT | 46 | **64** (+18) |
| STRUCTURAL | 98 | **80** (-18) |
| GPU PASS | 126 | **127** (+1, from TPL13's r1_running_xor) |
| pytest | 865/865 | 865/865 (unchanged) |
| MIXED / errors | 0 / 0 | 0 / 0 |

18 BYTE_EXACT kernels harvested via the template path across 59
sprint phases (AT01-AT12 + TPL01-TPL20 + MPT01-MPT35), all with
**zero regressions**.

## 36.5 — Commit table for this sprint chain

| phase | commit | pushed |
|---|---|:-:|
| MPT29 (Slice A proof) | `89cf25b` | yes |
| MPT30 (Slice A template) | `6659d34` | yes |
| MPT31 (Slice A validate) | `1ca5293` | yes |
| MPT32 (mid-chain gate) | `8286157` | yes |
| MPT33 (Slice B proof) | `8f35d27` | yes |
| MPT34 (Slice B template + 0x81c allowlist) | `ec9c61c` | yes |
| MPT35 (Slice B validate) | `c15d307` | yes |
| MPT36 (final frontier) | (this commit) | pending |
