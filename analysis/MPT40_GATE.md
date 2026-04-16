# MPT40 — Mid-chain recompute and final adaptive continuation gate

## 40.1 — Mid-chain recompute

| metric | before MPT37 | after MPT39 |
|---|---:|---:|
| Corpus BYTE_EXACT | 64 | **65** |
| Corpus STRUCTURAL | 80 | **79** |
| MIXED / REG_AND_CTRL / errors | 0 / 0 / 0 | 0 / 0 / 0 |
| pytest | 865/865 | 865/865 |
| GPU PASS / FAIL / RUN_EXC | 127 / 10 / 7 | 127 / 10 / 7 |
| Predicate-template kernels BYTE_EXACT | 9 | **10** |

Cumulative MPT01-MPT39 harvest: 10 kernels via per-kernel templates,
zero regressions across 39 sprint phases.

## 40.2 — Final continuation gate

Only one candidate remains in the multi-pred predicate-body family:
**r1_minmax** (delta=+7, mul+and prefix + 2-mov clamp pattern).

| question | answer |
|---|:-:|
| Did Slice A land cleanly? | **YES** |
| Is baseline preserved? | **YES** (pytest 865/865, GPU 127/10/7) |
| Is there a remaining bounded candidate to attempt? | **r1_minmax (only one left)** |
| Is it safe to continue into Slice B? | **YES, ATTEMPT MPT41 PROOF** |

**Decision: CONTINUE into Slice B with r1_minmax proof.**  If MPT41
discovers the shape is not exact-bounded for whole-kernel template
treatment (e.g. the mul+and prefix interacts unsafely with predicate
allocation), STOP after MPT41 and preserve Slice A.
