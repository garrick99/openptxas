# MPT23 — k200_pred_chain template validation

## 23.1 — Validation matrix (target)

| kernel | class before | class after | BE before | BE after | residual delta | GPU | PASS/FAIL |
|---|---|---|---|---|---:|---|:-:|
| k200_pred_chain | STRUCTURAL | **BYTE_EXACT** | NO | YES | 0 | PASS | **PASS** |

Active-instruction byte-equality verified at all 20 indices (workbench
direct-compile probe + frontier classifier).

## 23.2 — Controls

| control kernel | family | class | unchanged? | PASS/FAIL |
|---|---|---|:-:|:-:|
| k300_nasty_multi_pred | predicate-template (MPT17) | BYTE_EXACT | YES | PASS |
| k300_nasty_zero_init | non-atom template (TPL05) | BYTE_EXACT | YES | PASS |
| r1_running_xor | non-atom template (TPL13) | BYTE_EXACT | YES | PASS |
| k100_dual_load | non-atom template (TPL01, 4-param) | BYTE_EXACT | YES | PASS |
| k200_nested_pred | excluded multi-pred sibling | STRUCTURAL | YES | PASS |
| w1_div_multi_guard | excluded multi-pred sibling | STRUCTURAL | YES | PASS |
| SHF-family (e.g. shf_l_chain) | not touched | unchanged | YES | PASS |
| HFMA2-family | not touched | unchanged | YES | PASS |

(Corpus-wide frontier confirms BYTE_EXACT 60 → 61 (+1) with no other
deltas; STRUCTURAL 84 → 83 (−1) accounts for k200_pred_chain leaving
STRUCTURAL.)

## 23.3 — Residual blocker check

**No residual blocker** — k200_pred_chain is now BYTE_EXACT.

## 23.4 — Validation summary

| harness | result |
|---|:-:|
| pytest | **865/865** |
| GPU harness | 127 PASS / 10 FAIL / 7 RUN_EXC (unchanged baseline) |
| Frontier (corpus) | BYTE_EXACT **61** / STRUCTURAL **83** |
| health.py | GREEN (frontier-only); RED (full, due to GPU baseline 10 known FAILs) |

## 23.5 — Slice A clean — proceed to MPT24 gate
