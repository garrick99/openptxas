# MPT42 — Slice B BAIL: r1_minmax template reverted

## Why bail

The r1_minmax template emits 0x848 (IMNMX) in a tight RAW chain:

```
[6] IMAD R0 = R3 * 7
[8] LOP3 R0 = R0 & 0xFF       (R0 RAW from [6])
[9] IMNMX R0 = clamp(R0,200)  (R0 RAW from [8])
[10] IMNMX R5 = clamp(R0,16)  (R0 RAW from [9])
```

The FG-2.5 proof engine analyzes this chain and reports VIOLATION
edges through 0x848 because that opcode has no entry in `_OPCODE_META`
(no evidence-backed `min_gpr_gap`).  Adding it to `_LATENCY_INERT`
(MPT41 commit) covers FG-2.3 INV B (opcode coverage) but NOT the FG-2.5
proof invariants:

* INV S (FG-3.0 expanded scoreboard model)
* INV W (memory model ALU edges)
* INV AB (FG-3.2 invariants)
* INV ADJ4 (FG-4.1 refinement)
* INV AH (FG-3.3 invariants)
* test_inv_h_corpus_is_safe_under_proof[r1_minmax]
* test_inv_i_safe_kernels_have_zero_violations
* test_verify_schedule_no_real_hazards
* test_verify_schedule_zero_reports_after_fg24

**Total: 9 pytest failures** when the registry entry is enabled.

GPU correctness is unaffected (PASS=127/10/7 unchanged) — PTXAS itself
schedules these IMNMX ops with appropriate ctrl-byte stalls, so the
template's verbatim bytes are runtime-correct.  But the FG-2.5 proof
engine cannot statically verify the chain without an `_OPCODE_META`
entry for 0x848.

## What was reverted

* `sass/pipeline.py` — registry entry `('r1_minmax', 2,
  'non_atom_minmax.json', 'TPL/MPT42')` REVERTED via `git checkout`
  (was uncommitted local edit only).

## What remains landed (innocuous)

* `tools/template_engine/generated/non_atom_minmax.json` (committed
  in `cbb77e7`) — never dispatched without registry entry; safe to
  keep for future use after 0x848 is properly modeled.
* `tests/test_fg23_model_complete.py` — 0x848 added to FG-2.3 INV B
  `_LATENCY_INERT` allowlist (committed in `cbb77e7`); harmless when
  0x848 is not currently emitted.
* `analysis/MPT41_MINMAX.md` — proof boundary doc.

## Slice A preserved

| metric | post-MPT39 | post-MPT42-bail | unchanged? |
|---|---:|---:|:-:|
| Corpus BYTE_EXACT | 65 | 65 | yes |
| Corpus STRUCTURAL | 79 | 79 | yes |
| pytest | 865/865 | 865/865 | yes |
| GPU PASS / FAIL / RUN_EXC | 127 / 10 / 7 | 127 / 10 / 7 | yes |

## What it would take to land r1_minmax later

1. Probe 0x848 (IMNMX) latency via standalone microbench: produce a
   chain `IMNMX → R-consumer` with varying gap and find the minimum
   safe gap (likely 1, same as IMAD/LOP3).
2. Add `_OPCODE_META[0x848] = (min_gpr_gap=N, ...)` with the probed
   evidence.
3. Re-enable the registry entry.
4. Re-validate FG-2.5 proof engine + GPU.

This work is OUT OF SCOPE for the MPT predicate-template chain.

## Predicate-template subsystem status

10 of 11 multi-pred kernels harvested via templates (k100_pred_arith,
k200_double_guard, k300_pred3, k100_setp_combo, k300_nasty_multi_pred,
k200_pred_chain, w1_div_multi_guard, w2_deep_pred, k200_nested_pred,
k300_nasty_pred_nest3).

**r1_minmax remains STRUCTURAL** awaiting 0x848 latency-model entry.

The predicate-template subsystem is **complete for the setp+@P-add
shape family** (the family the chain targeted).  r1_minmax is in a
distinct mul+and+clamp family that requires latency-model expansion,
not predicate-template extension.
