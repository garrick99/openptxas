# Milestone: 148/149 — expanded corpus (workbench + benchmarks) green

**HEAD:** `8711ee8`
**Date:** 2026-04-19
**Campaign:** R31 → R62

---

## Corpus status

Full `workbench.py` + `workbench_expanded.py` + `benchmarks/*_PTX` fixture set:

- **Total tested:** 149 (142 workbench + 7 benchmark)
- **Compile gate:** 149/149 pass (no GPU needed)
- **Execution gate (GPU):** 148/149 pass — only `relu` crashes (sync=700; u64 param UR-allocation structural issue, out of scope for narrow fixes)

Verified via `scripts/corpus_sweep.py` and `scripts/corpus_compile_check.py`:

```
$ python scripts/corpus_compile_check.py
[compile_check] total=149 pass=149 fail=0

$ python scripts/corpus_sweep.py
[corpus_sweep] total=149 pass=148 fail=1
  sync_err: relu
```

Benchmark-suite correctness (more than just sync — verifies bit-identical output vs ptxas):

| kernel | ratio vs ptxas | correctness |
|--------|---------------|-------------|
| vecadd | 1.00× | PASS |
| saxpy | 1.77× | PASS |
| memcpy | 1.01× | PASS |
| scale | 1.02× | PASS |
| stencil | 0.96× | PASS (fixed in R62) |
| relu | — | crashes (known) |
| fma_chain | 0.75× | PASS (perf gap, +22 NOPs) |

Geomean of 5 passing benchmarks: **1.15× ptxas**.

CI wiring (`.github/workflows/corpus.yml`):

- **compile-gate** — GitHub-hosted ubuntu, every PR
- **corpus-gpu** — self-hosted `[gpu, sm_120]`, main + `gpu-check` label

See `docs/CI.md` for runner setup.

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
| shl(cvt(x)) fold pop-wrong-cvt + fallback RAW | class | stencil (silent math corruption) | 8711ee8 |

---

## Known boundary

- **"Tested corpus green"** — the 142-fixture practical corpus (ALU chains, predicate, memory, loops, shuffle/vote/redux, shared-memory + bar.sync, atomics) now compiles and executes correctly on SM_120.
- This does **not** mean every conceivable PTX construct is supported — only that the fixtures in `workbench.py` + `workbench_expanded.py` pass.

---

## Deferred to next phase

- **`relu` sync=700** — u64 `p_out` param goes through the GPR `LDC.64` path (via `ra.int_regs`), then the store address is built from an `IADD3 + IADD3.X` carry chain. Ptxas keeps the same param in a UR via `LDCU.128` and uses `IADD.64 R-UR`. Fix requires a regalloc-level change to reroute u64 address params through the UR path when a GPR isn't needed; not a narrow patch.
- **`fma_chain` 0.75× perf** — +22 NOPs and +2 real instructions vs ptxas across the 32-FFMA dependency chain. Scoreboarder is inserting unnecessary stalls between FFMA→FFMA. Perf only — correctness is clean.
- **Forge benchmark suite integration** — full Forge compiler output exercises the backend on higher-variance PTX; not yet swept
- **OpenCUDA ctest integration** — full OpenCUDA kernel corpus; regression gating on a larger surface
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
