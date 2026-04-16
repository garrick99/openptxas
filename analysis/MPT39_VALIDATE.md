# MPT39 — k300_nasty_pred_nest3 template validation

## 39.1 — Validation matrix (target)

| kernel | class before | class after | BE before | BE after | residual delta | GPU | PASS/FAIL |
|---|---|---|---|---|---:|---|:-:|
| k300_nasty_pred_nest3 | STRUCTURAL | **BYTE_EXACT** | NO | YES | 0 | PASS | **PASS** |

All 22 active instructions match PTXAS verbatim.

## 39.2 — Controls

| control kernel | family | class | unchanged? | PASS/FAIL |
|---|---|---|:-:|:-:|
| k200_nested_pred | predicate-template (MPT34, conditional-setp prior) | BYTE_EXACT | YES | PASS |
| w2_deep_pred | predicate-template (MPT30) | BYTE_EXACT | YES | PASS |
| k300_nasty_zero_init | non-atom template (TPL05) | BYTE_EXACT | YES | PASS |
| k100_atom_add | atom template | BYTE_EXACT | YES | PASS |
| k100_dual_load | non-atom template (TPL01, 4-param) | BYTE_EXACT | YES | PASS |
| r1_minmax | excluded sibling (different family) | STRUCTURAL | YES | PASS |
| SHF-family | not touched | unchanged | YES | PASS |
| HFMA2-family | not touched | unchanged | YES | PASS |

(Corpus-wide frontier: BYTE_EXACT 64 -> 65, STRUCTURAL 80 -> 79; no
other deltas.)

## 39.3 — Residual blocker check

**No residual blocker** — k300_nasty_pred_nest3 is now BYTE_EXACT.

## 39.4 — Validation summary

| harness | result |
|---|:-:|
| pytest | **865/865** |
| GPU harness | 127 PASS / 10 FAIL / 7 RUN_EXC (unchanged baseline) |
| Frontier (corpus) | BYTE_EXACT **65** / STRUCTURAL **79** |
| health.py --frontier-only | GREEN |

## 39.5 — Slice A clean — proceed to MPT40 gate
