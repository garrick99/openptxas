# MPT35 — k200_nested_pred template validation

## 35.1 — Validation matrix (target)

| kernel | class before | class after | BE before | BE after | residual delta | GPU | PASS/FAIL |
|---|---|---|---|---|---:|---|:-:|
| k200_nested_pred | STRUCTURAL | **BYTE_EXACT** | NO | YES | 0 | PASS | **PASS** |

Active-instruction byte-equality verified at all 19 indices.

## 35.2 — Controls

| control kernel | family | class | unchanged? | PASS/FAIL |
|---|---|---|:-:|:-:|
| w2_deep_pred | predicate-template (MPT30 Slice A) | BYTE_EXACT | YES | PASS |
| w1_div_multi_guard | predicate-template (MPT26) | BYTE_EXACT | YES | PASS |
| k300_nasty_zero_init | non-atom template (TPL05) | BYTE_EXACT | YES | PASS |
| k100_atom_add | atom template | BYTE_EXACT | YES | PASS |
| k100_dual_load | non-atom template (TPL01, 4-param) | BYTE_EXACT | YES | PASS |
| k300_nasty_pred_nest3 | excluded sibling | STRUCTURAL | YES | PASS |
| r1_minmax | excluded sibling | STRUCTURAL | YES | PASS |
| SHF-family | not touched | unchanged | YES | PASS |
| HFMA2-family | not touched | unchanged | YES | PASS |

(Corpus-wide frontier: BYTE_EXACT 63 -> 64, STRUCTURAL 81 -> 80; no
other deltas.)

## 35.3 — Residual blocker check

**No residual blocker** — k200_nested_pred is now BYTE_EXACT.

Side note: required updating FG-2.3 INV B coverage allowlist for the
0x81c uncommon ALU helper opcode (template-only, never via isel).

## 35.4 — Validation summary

| harness | result |
|---|:-:|
| pytest | **865/865** |
| GPU harness | 127 PASS / 10 FAIL / 7 RUN_EXC (unchanged baseline) |
| Frontier (corpus) | BYTE_EXACT **64** / STRUCTURAL **80** |
| health.py --frontier-only | GREEN |

## 35.5 — Slice B clean — chain complete
