# MPT24 — Mid-chain recompute and adaptive continuation gate

## 24.1 — Mid-chain recompute

| metric | before MPT21 | after MPT23 |
|---|---:|---:|
| Corpus BYTE_EXACT | 60 | **61** |
| Corpus STRUCTURAL | 84 | **83** |
| MIXED / REG_AND_CTRL / errors | 0 / 0 / 0 | 0 / 0 / 0 |
| pytest | 865/865 | 865/865 |
| GPU PASS / FAIL / RUN_EXC | 127 / 10 / 7 | 127 / 10 / 7 |
| Predicate-template kernels BYTE_EXACT | 5 | **6** |

Cumulative MPT01-MPT23 harvest: 6 kernels via per-kernel templates,
zero regressions across 23 sprint phases.

## 24.2 — Adaptive continuation ranking (5 remaining candidates)

| rank | kernel | exact-shape closeness to prior MPT slices | next blocker | recommendation |
|---:|---|---|---|---|
| **1** | **w1_div_multi_guard** | 4-setp/4-@P alternating P1/P2 (sibling of MPT22 k200_pred_chain in setp count, only PTX predicate-name pattern differs); delta=+1 | own JSON | **MPT25 target** |
| 2 | w2_deep_pred | 5-setp/5-@P alternating P1/P2 (sibling of MPT17 k300_nasty_multi_pred); delta=+1 | own JSON | medium |
| 3 | k200_nested_pred | 2-setp/3-@P with @p1-conditional setp; delta=-2 (under-emit) | own JSON; merge-template variant | medium-low |
| 4 | k300_nasty_pred_nest3 | 3-setp/5-@P 3-level nested guards; delta=-3 (under-emit) | own JSON; complex nesting | low |
| 5 | r1_minmax | mul+and prefix + 2-setp/2-mov clamp; delta=+7 | own JSON; longest body | low (highest delta) |

## 24.3 — Continuation gate

| question | answer |
|---|:-:|
| Did Slice A land cleanly? | **YES** |
| Is baseline preserved? | **YES** (pytest 865/865, GPU 127/10/7) |
| Is there exactly one clear next bounded candidate? | **YES** - w1_div_multi_guard (rank #1, delta=+1, sibling of MPT22) |
| Is it safe to continue immediately into Slice B? | **YES** |

**Decision: CONTINUE into Slice B with `w1_div_multi_guard`.**

Why w1_div_multi_guard over w2_deep_pred (both delta=+1):
- Same 4-setp count as just-landed MPT22 (k200_pred_chain), so PTXAS's
  emitted shape is closer to the same template family.
- w1_div_multi_guard uses 2 PTX predicate names (P1, P2 alternating);
  k200_pred_chain used 1 (P1 only). The PTXAS allocator-rewrite delta
  is therefore smaller for w1_div_multi_guard.
- w2_deep_pred is a 5-setp variant — sibling of MPT17, but defer until
  the 4-setp alternating-P1/P2 pattern is proven.
