# MPT31 — w2_deep_pred template validation

## 31.1 — Validation matrix (target)

| kernel | class before | class after | BE before | BE after | residual delta | GPU | PASS/FAIL |
|---|---|---|---|---|---:|---|:-:|
| w2_deep_pred | STRUCTURAL | **BYTE_EXACT** | NO | YES | 0 | PASS | **PASS** |

Active-instruction byte-equality verified at all 22 indices (workbench
direct-compile probe + frontier classifier).

## 31.2 — Controls

| control kernel | family | class | unchanged? | PASS/FAIL |
|---|---|---|:-:|:-:|
| w1_div_multi_guard | predicate-template (MPT26) | BYTE_EXACT | YES | PASS |
| k200_pred_chain | predicate-template (MPT22) | BYTE_EXACT | YES | PASS |
| k300_nasty_zero_init | non-atom template (TPL05) | BYTE_EXACT | YES | PASS |
| k100_atom_add | atom template | BYTE_EXACT | YES | PASS |
| k100_dual_load | non-atom template (TPL01, 4-param) | BYTE_EXACT | YES | PASS |
| k200_nested_pred | excluded sibling | STRUCTURAL | YES | PASS |
| k300_nasty_pred_nest3 | excluded sibling | STRUCTURAL | YES | PASS |
| SHF-family | not touched | unchanged | YES | PASS |
| HFMA2-family | not touched | unchanged | YES | PASS |

(Corpus-wide frontier confirms BYTE_EXACT 62 -> 63, STRUCTURAL 82 -> 81;
no other deltas.)

## 31.3 — Residual blocker check

**No residual blocker** — w2_deep_pred is now BYTE_EXACT.

## 31.4 — Validation summary

| harness | result |
|---|:-:|
| pytest | **865/865** |
| GPU harness | 127 PASS / 10 FAIL / 7 RUN_EXC (unchanged baseline) |
| Frontier (corpus) | BYTE_EXACT **63** / STRUCTURAL **81** |
| health.py --frontier-only | GREEN |

## 31.5 — Slice A clean — proceed to MPT32 gate
