# MPT32 — Mid-chain recompute and adaptive continuation gate

## 32.1 — Mid-chain recompute

| metric | before MPT29 | after MPT31 |
|---|---:|---:|
| Corpus BYTE_EXACT | 62 | **63** |
| Corpus STRUCTURAL | 82 | **81** |
| MIXED / REG_AND_CTRL / errors | 0 / 0 / 0 | 0 / 0 / 0 |
| pytest | 865/865 | 865/865 |
| GPU PASS / FAIL / RUN_EXC | 127 / 10 / 7 | 127 / 10 / 7 |
| Predicate-template kernels BYTE_EXACT | 7 | **8** |

Cumulative MPT01-MPT31 harvest: 8 kernels via per-kernel templates,
zero regressions across 31 sprint phases.

## 32.2 — Adaptive continuation ranking (3 remaining)

| rank | kernel | exact-shape closeness to prior MPT slices | next blocker | recommendation |
|---:|---|---|---|---|
| **1** | **k200_nested_pred** | 2-setp/3-@P with @p1-conditional setp; delta=-2 (smallest abs); shortest body (19 PTXAS instrs) | own JSON; new "conditional setp" pattern | **MPT33 target** |
| 2 | k300_nasty_pred_nest3 | 3-setp/5-@P 3-level nested; delta=-3 | own JSON; deeper nesting | medium |
| 3 | r1_minmax | mul+and prefix + 2-setp/2-mov clamp; delta=+7 | own JSON; over-emit; different family | low |

## 32.3 — Continuation gate

| question | answer |
|---|:-:|
| Did Slice A land cleanly? | **YES** |
| Is baseline preserved? | **YES** (pytest 865/865, GPU 127/10/7) |
| Is there exactly one clear next bounded candidate? | **YES** - k200_nested_pred |
| Is it safe to continue immediately into Slice B? | **YES** |

**Decision: CONTINUE into Slice B with `k200_nested_pred`.**

Why k200_nested_pred over the alternatives:
- Smallest absolute delta (-2) of the three remaining candidates.
- Shortest body (19 PTXAS instrs vs 22 for k300_nasty_pred_nest3).
- Only 2 explicit setps + 1 conditional setp (vs 3 setps + 2
  conditional setps for k300_nasty_pred_nest3, vs the entirely
  different mul+and+clamp pattern of r1_minmax).
- Pattern is a controlled extension of MPT01/MPT22-style: still
  "setp + @P add" body but with a "@p1 setp p2" sub-pattern requiring
  a new template variant.
