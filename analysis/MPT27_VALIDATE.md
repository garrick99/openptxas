# MPT27 — w1_div_multi_guard template validation

## 27.1 — Validation matrix (target)

| kernel | class before | class after | BE before | BE after | residual delta | GPU | PASS/FAIL |
|---|---|---|---|---|---:|---|:-:|
| w1_div_multi_guard | STRUCTURAL | **BYTE_EXACT** | NO | YES | 0 | PASS | **PASS** |

Active-instruction byte-equality verified at all 20 indices (workbench
direct-compile probe + frontier classifier).

## 27.2 — Controls

| control kernel | family | class | unchanged? | PASS/FAIL |
|---|---|---|:-:|:-:|
| k200_pred_chain | predicate-template (MPT22, Slice A) | BYTE_EXACT | YES | PASS |
| k300_nasty_multi_pred | predicate-template (MPT17) | BYTE_EXACT | YES | PASS |
| k300_nasty_zero_init | non-atom template (TPL05) | BYTE_EXACT | YES | PASS |
| k100_atom_add | atom template | BYTE_EXACT | YES | PASS |
| k100_dual_load | non-atom template (TPL01, 4-param) | BYTE_EXACT | YES | PASS |
| k200_nested_pred | excluded sibling | STRUCTURAL | YES | PASS |
| w2_deep_pred | excluded sibling | STRUCTURAL | YES | PASS |
| SHF-family | not touched | unchanged | YES | PASS |
| HFMA2-family | not touched | unchanged | YES | PASS |

(Corpus-wide frontier confirms BYTE_EXACT 61 -> 62 (+1) with no other
deltas; STRUCTURAL 83 -> 82 (-1) accounts for w1_div_multi_guard
leaving STRUCTURAL.)

## 27.3 — Residual blocker check

**No residual blocker** — w1_div_multi_guard is now BYTE_EXACT.

## 27.4 — Validation summary

| harness | result |
|---|:-:|
| pytest | **865/865** |
| GPU harness | 127 PASS / 10 FAIL / 7 RUN_EXC (unchanged baseline) |
| Frontier (corpus) | BYTE_EXACT **62** / STRUCTURAL **82** |
| health.py --frontier-only | GREEN |

## 27.5 — Slice B clean — chain complete
