# MPT04 — Predicate-template frontier and continuation

Evidence for commit `4f48d8b` (MPT02+MPT03 consolidated).

## 04.1 — Frontier delta

| metric                              | before MPT01 | after MPT04 |
|-------------------------------------|-------------:|------------:|
| Corpus BYTE_EXACT                   | 55           | **56**      |
| Corpus STRUCTURAL                   | 89           | **88**      |
| MIXED / REG_AND_CTRL / errors       | 0 / 0 / 0    | 0 / 0 / 0   |
| pytest                              | 865/865      | 865/865     |
| GPU PASS / FAIL / RUN_EXC           | 127 / 10 / 7 | 127 / 10 / 7|
| Atom-template kernels BYTE_EXACT    | 6            | 6 (unchanged) |
| Non-atom-template kernels BYTE_EXACT| 5            | **6** (+k100_pred_arith) |

Multi-pred predicate-body family: 11 STRUCTURAL → **10 STRUCTURAL** + 1 BYTE_EXACT.

## 04.2 — Continuation ranking

10 remaining MP02-protected multi-pred candidates:

| kernel                  | shape vs MPT01 | ours/ptxas | next blocker | recommendation |
|-------------------------|---------------|-----------:|--------------|----------------|
| k200_double_guard       | similar shape (2-setp/2-@P) but body has `mul.lo.u32` upfront | 17/17 | own JSON template (different body bytes) | **MPT05-style sprint chain** |
| k200_nested_pred        | 2-setp/3-@P, delta=−2 | 17/19 | own JSON | MPT05-adjacent |
| k100_setp_combo         | 2-setp/2-@P, delta=+1 | 17/16 | own JSON; non-zero delta indicates more diff | medium |
| r1_minmax               | 2-setp/2-@P, delta=+4 | 20/16 | own JSON; delta=+4 indicates 4 extra ops | low |
| k300_pred3              | 3-setp/3-@P, delta=+0 | 19/19 | own JSON | medium |
| k300_nasty_pred_nest3   | 3-setp/5-@P, delta=−3 | 19/22 | own JSON; nested guards | low |
| k200_pred_chain         | 4-setp/4-@P, delta=+1 | 21/20 | own JSON | low |
| w1_div_multi_guard      | 4-setp/4-@P, delta=+1 | 21/20 | own JSON; same shape as k200_pred_chain | low (sibling) |
| k300_nasty_multi_pred   | 5-setp/5-@P, delta=+0 | 23/23 | own JSON | low (largest) |
| w2_deep_pred            | 5-setp/5-@P, delta=+1 | 23/22 | own JSON; same shape as k300_nasty_multi_pred | low (sibling of k300_nasty) |

## 04.3 — One next move

> **Next move: MPT05 — bounded whole-kernel template for `k200_double_guard`.**
>
> **Why this and not the others**:
>
> * Same shape complexity as MPT01 (2-setp + 2-@P, delta=+0), only
>   differs by an upfront `mul.lo.u32 %r2, %r0, 3` before the
>   conditional adds.
> * Same risk profile as TPL01–TPL20 / MPT01–MPT04: per-kernel JSON
>   template + single-line registry entry + GPU validation per AT07
>   lesson. The MPT01 wiring already added all needed
>   pipeline-level skips (FG29-C / FG33 / post-EXIT b9), so MPT05
>   needs **only the JSON file + 1 registry entry**.
> * Delta=0 means the kernel is already as compact as PTXAS — the
>   template just supplies the exact byte sequence.
>
> **Why NOT pivot away from predicate templates yet**:
>
> * MPT01 landed cleanly with zero regressions. The infrastructure is
>   proven for MP02-protected kernels.
> * 10 more bounded BYTE_EXACT kernels are reachable in this exact
>   direction (each its own ~3-commit chain).
> * Allocator-aware precursor and HFMA2 work remain explicitly out
>   of bounded scope.
> * SHF was conclusively shown by MEGA-01 to be allocator/scheduler
>   bound.

After MPT05–MPT08, if k200_double_guard lands, corpus would be at
**57 BYTE_EXACT / 87 STRUCTURAL**. Continuing through all 10
remaining candidates (MPT09 / MPT13 / MPT17 / MPT21 / MPT25 / etc.)
would close out this direction at **66 BYTE_EXACT / 78 STRUCTURAL**.

## Honest assessment

The MP02-aware predicate-body template subsystem is now proven (MPT01
landed cleanly). The mechanism is identical to TPL01–TPL20: extract
PTXAS bytes, write a JSON file, add a registry entry. The MP02 fix
integrity is preserved because the template REPLACES the body
entirely — MP02's downstream FG33/FG56/FG60 byte-rewrites have
nothing to modify.

The single new pipeline.py change (the FG29-C `'MPT' in si.comment`
skip) is defensive: it ensures any future MPT-template's verified
PTXAS bytes are not silently mutated by FG29-C's body ALU register
normalization. Pattern matches the prior TPL14 FG33 skip and TPL02
post-EXIT b9 skip — all are "the template knows what it wants; do
not patch".

## Commit table for this sprint chain

| phase | commit | pushed |
|---|---|:-:|
| MPT01 | `9aaddeb` | ✓ |
| MPT02 + MPT03 (consolidated) | `4f48d8b` | ✓ |
| MPT04 | (this doc) | pending commit |
