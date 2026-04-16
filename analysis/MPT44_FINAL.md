# MPT44 — Final predicate-template subsystem frontier

Final state after the MPT37-MPT44 chain.  Slice A landed; Slice B
bailed cleanly with Slice A preserved.

## 44.1 — Frontier delta (full MPT01-MPT44)

| metric                              | MP03 baseline | post-MPT44 |
|-------------------------------------|--------------:|-----------:|
| Corpus BYTE_EXACT                   | 46            | **65**     |
| Corpus STRUCTURAL                   | 98            | **79**     |
| MIXED / REG_AND_CTRL / errors       | 0 / 0 / 0     | 0 / 0 / 0  |
| pytest                              | 865/865       | 865/865    |
| GPU PASS / FAIL / RUN_EXC           | 126 / 10 / 7  | 127 / 10 / 7 |
| Predicate-template kernels BYTE_EXACT | 0           | **10**     |
| Multi-pred predicate-body family    | 11 STRUCTURAL | 1 STRUCTURAL + 10 BE |

## 44.2 — Predicate-template subsystem complete

10 of 11 multi-pred predicate-body kernels harvested via per-kernel
templates:

| # | kernel | landed in | shape highlight |
|---|---|---|---|
| 1 | k100_pred_arith       | MPT01 | first SEL+@P-UIADD mux |
| 2 | k200_double_guard     | MPT05 | LT-as-NEG-GE with @P1+@!P0 |
| 3 | k300_pred3            | MPT09 | 3-distinct-predicate allocation |
| 4 | k100_setp_combo       | MPT13 | R0-direct TID + SEL.IMM |
| 5 | k300_nasty_multi_pred | MPT17 | 5-setp R0=R3+10 SEL fold |
| 6 | k200_pred_chain       | MPT22 | 4-setp all-on-P1 reuse |
| 7 | w1_div_multi_guard    | MPT26 | 4-setp alternating P1/P2 |
| 8 | w2_deep_pred          | MPT30 | 5-setp alternating P1/P2 |
| 9 | k200_nested_pred      | MPT34 | first @P-conditional setp + 0x81c |
| 10 | k300_nasty_pred_nest3 | MPT38 | 2 conditional setps + 2x 0x81c |

## 44.3 — r1_minmax: the final hold-out

| candidate | status | next blocker |
|---|---|---|
| r1_minmax | STRUCTURAL (delta=+7) | requires `_OPCODE_META[0x848]` entry from probed IMNMX latency before its template can land without breaking FG-2.5 proof invariants |

The r1_minmax template was BYTE_EXACT at all 16 instructions in
isolation, but enabling it triggered 9 FG-2.5 proof-engine VIOLATION
failures (INV S/W/AB/ADJ4/AH/H/I + FG-2.3 schedule verifications).
GPU correctness was unaffected; the proof engine is conservative on
unmodeled opcodes.  Reverted per operating rules.

## 44.4 — One next move

> **Next move: pivot OUT of predicate-template direction.**
>
> All 10 predicate-template kernels in the *setp + @P-add* shape
> family have been harvested.  The remaining `r1_minmax` belongs to
> a distinct *mul + and + clamp* family that requires latency-model
> expansion (probe + model 0x848 IMNMX), not predicate-template
> extension.
>
> **Recommended pivot options** (in priority order):
>
> 1. **0x848 IMNMX latency probe + `_OPCODE_META` entry** — small,
>    bounded work that re-enables the already-written r1_minmax
>    template.  Could land 1 more BYTE_EXACT kernel (66 total) and
>    establishes the pattern for future PTXAS-recognized-idiom
>    templates.
> 2. **Pivot to allocator-aware precursor analysis** — the bounded
>    ceiling for templates is exhausted; deeper STRUCTURAL → BYTE_EXACT
>    progress requires understanding why OURS' allocator differs from
>    PTXAS on the 79 remaining STRUCTURAL kernels.
> 3. **Forge focus** (per user's [feedback_forge_focus]) — pause
>    OpenPTXas direction and switch to Forge compiler work.

## 44.5 — Cumulative subsystem-completion record

| sprint family | sprints | BE harvested | regressions |
|---|---|---:|---:|
| AT01-AT12 (atom-family templates) | 12 | (varies) | 0 |
| TPL01-TPL20 (non-atom templates) | 20 | 5 | 0 |
| MPT01-MPT44 (predicate templates) | 44 | 10 | 0 |
| **TOTAL template-path harvest** | **76** | **18** | **0** |

19 BYTE_EXACT kernels (counting TPL13's r1_running_xor GPU-PASS bonus)
total via the template path with **zero regressions** across 76 sprint
phases.

## 44.6 — Commit table for MPT37-MPT44

| phase | commit | pushed |
|---|---|:-:|
| MPT37 (Slice A proof)            | `9171636` | yes |
| MPT38 (Slice A template)         | `525f36e` | yes |
| MPT39 (Slice A validate)         | `5302e31` | yes |
| MPT40 (mid-chain gate)           | `25d261c` | yes |
| MPT41 (Slice B proof + JSON + 0x848 allowlist) | `cbb77e7` | yes |
| MPT42 (Slice B BAIL: registry reverted) | `739096c` | yes |
| MPT44 (this final-frontier doc)  | (this commit) | pending |
