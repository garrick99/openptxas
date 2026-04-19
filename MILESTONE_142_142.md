# Milestone: 142/142 — practical workbench corpus green

**HEAD:** `23959b1`
**Date:** 2026-04-18
**Campaign:** R31 → R58

---

## Corpus status

Full `workbench.py` + `workbench_expanded.py` fixture set:

- **Total tested:** 142
- **Passing:** 142
- **Failing:** 0

Verified via `scripts/corpus_sweep.py` (fresh CUDA context per kernel, N=32, 1 block).

```
$ python scripts/corpus_sweep.py
[corpus_sweep] total=142 pass=142 fail=0
```

Exit status: `0` when all pass, `1` when any fail — suitable for CI gating.

---

## Closed clusters / surfaces

| Cluster | Size | Representative kernels | Closing commit |
|---|---|---|---|
| Family A (reduce_sum, conv2d, hmma) | 3 | reduce_sum, conv2d_looped, hmma_zero | pre-R48 |
| Family B ladder | many | k100/k200/k300 ALU, ILP, mixed-width | pre-R48 |
| Post-EXIT S2R→ALU hazard | class | s2_fail, R38/R39 probe set | 844282a, 16a0d06 |
| ISETP→@P adjacency (EXIT/ALU/BRA) | ≥ 20 | k100_early_exit, k300_nasty_pred_xor, all loops | 9f9bf02 |
| FG29-C b4-is-imm misrename | class | k200_xor_reduce | 68c3bb5 |
| FG29-C 64-bit pair hi-half tracking | class | k200_load_pred_store | 17a5951 |
| FG56b ISETP consumer rename | class | k100_early_exit | 63b174b |
| Multi-BB ld.param.u64 sink position | class | w1_div_if_else | dec0ab0 |
| FG26 UR4 reservation gated on TE10 | class | warp intrinsics (10), smem/bar (2) | fa602dd, d05045d |
| Preamble-hoisted LDC vs S2R GPR alias | class | bar_ldc_xor | c391209 |
| ATOMG_AND mode-byte encoding | 1 instr | w2_atom_and_reduce | 23959b1 |

---

## Known boundary

- **"Tested corpus green"** — the 142-fixture practical corpus (ALU chains, predicate, memory, loops, shuffle/vote/redux, shared-memory + bar.sync, atomics) now compiles and executes correctly on SM_120.
- This does **not** mean every conceivable PTX construct is supported — only that the fixtures in `workbench.py` + `workbench_expanded.py` pass.

---

## Deferred to next phase

- **Forge benchmark suite integration** — full Forge compiler output exercises the backend on higher-variance PTX; not yet swept
- **OpenCUDA ctest integration** — full OpenCUDA kernel corpus; regression gating on a larger surface
- **Performance / parity with ptxas** — current goal was correctness; perf vs ptxas on SAXPY is 1.48× better (previously measured) but broader perf work not revisited
- **Warp-reduce atomic optimization (REDUX→REDG)** — ptxas collapses per-lane atomics on a shared address into warp-level reductions; our backend still emits direct per-lane ATOMG. Correct but not optimal.
- **Feature families beyond the corpus** — cp.async, TMA async, cluster/CGA primitives, sm_120-specific tensor ops (beyond HMMA/DMMA/IMMA zero-accumulator), matrix multiply with real inputs.

---

## Campaign arc (condensed)

1. **R31 → R39** — fix hazards in UR routing, post-EXIT consumer gaps, pair safety, LDCU.128 pack bounds
2. **R48** — single root cause for predicated-EXIT / loop-back-edge / predicated-ALU failures: `ISETP → @P-matching-guard` adjacency hazard; one NOP-gap pass covered all three
3. **R49 → R51** — tighten rename passes (FG29-C b4 exclusions, FG29-C pair hi-half, FG56b ISETP rename)
4. **R52** — preserve ld.param.u64 PTX position in multi-BB labeled-consumer kernels
5. **R55 → R56** — gate FG26 UR4 admission on TE10-path availability (vote/shfl/redux AND bar.sync kernels)
6. **R57** — resolve preamble-hoist GPR alias between SR-derived vregs and non-setp-only LDC params
7. **R58** — correct `ATOMG_AND` encoding mode bytes to live ptxas ground truth; drove final failure to green

Each turn was proof-first, narrow, and gated — every commit lands a single-cause fix with a documented probe trace.

---

## Next best move

After freeze, prefer one of:

- **Coverage expansion**: sweep Forge's generated PTX corpus through the same sweep harness; classify any new failures
- **OpenCUDA regression gate**: hook `_r54/sweep2.py` into OpenCUDA's CI so regressions in the backend are caught early
- **Perf regression baseline**: record current SAXPY + conv2d timings vs ptxas as the new baseline before opening perf work
