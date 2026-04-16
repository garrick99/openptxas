# OpenPTXas — Packaging Baseline (MP03)

This document is the single source of truth for how to exercise, validate, and
understand the current state of the OpenPTXas backend. It supersedes ad-hoc
scripts and describes the **committed entrypoints**.

For architectural detail see `ARCHITECTURE.md`. For the user-facing pitch and
discoveries list see `README.md`.

---

## Entrypoints

### 1. Single kernel: compile + GPU run + workbench diff

```bash
python workbench.py run --kernel <NAME> --compare ptxas
python workbench.py kdiff --kernel <NAME>
```

* `workbench.py run` — full pipeline: compile through OpenPTXas, compile through
  PTXAS, load both cubins on GPU, run the correctness harness, print metrics
  delta + ours/ptxas timings.
* `workbench.py kdiff` — compile-only side-by-side SASS diff (no GPU run).

List the full kernel catalog:

```bash
python workbench.py list
```

### 2. Full validation: pytest + GPU harness + frontier health

```bash
python scripts/health.py
```

Runs three stages and prints a one-page summary:

1. **pytest** — `pytest tests/ -q` (non-GPU unit/integration tests).
2. **GPU harness** — compile every registered kernel, run its correctness
   harness on the GPU (subprocess-isolated so one faulty kernel cannot poison
   the driver context for the rest of the sweep). Reports PASS / FAIL /
   COMPILE_FAIL / RUN_EXC / NO_HARNESS counts.
3. **Frontier** — classify every kernel via `tools/template_engine/regdiff.py`
   (OURS vs PTXAS byte-level). Reports BYTE_EXACT / STRUCTURAL / MIXED counts.

Flags:
```bash
python scripts/health.py --quick           # pytest + frontier (skip GPU sweep)
python scripts/health.py --gpu-only
python scripts/health.py --frontier-only
```

Exit code 0 iff pytest green **and** GPU FAIL==0 **and** MIXED==0 **and** 0 errors.

### 3. Single-file PTX → cubin (no workbench)

```bash
python -m openptxas <file.ptx> --out <file.cubin>
python -m openptxas <file.ptx> --audit <file.cubin>   # bug/hazard scan
python demo.py                                         # vector_add smoke test
```

---

## Current Status (as of commit `a1a05ea`, MP02)

Frontier (144 kernels, `scripts/health.py --frontier-only`):

| Bucket        | Count |
|---------------|------:|
| BYTE_EXACT    | 46    |
| STRUCTURAL    | 98    |
| MIXED         | 0     |
| REG_AND_CTRL  | 0     |
| errors        | 0     |

pytest: **865 / 865 green**.

GPU harness (144 kernels, subprocess-isolated): 126 PASS, 11 FAIL, 7 RUN_EXC,
0 COMPILE_FAIL. The 18 failing kernels belong to the **known unsupported
families** below — none are regressions from MP02.

---

## Known Unsupported Families

Each entry is *evidence-driven* — either a GPU FAIL or a RUN_EXC from the most
recent `scripts/health.py --gpu-only` sweep.

| Family                       | Blocker                                             | Example kernels |
|------------------------------|-----------------------------------------------------|-----------------|
| BRA-based loops              | loop backedge + predicate interplay not yet modelled | `w1_loop_countdown`, `w1_loop_mul_acc`, `w1_loop_shift`, `w1_loop_sum`, `w1_loop_two_acc`, `w1_loop_load_acc`, `w1_loop_pred_acc`, `w2_div_loop`, `w2_nested_loop` |
| if/else divergence           | predicate-compose + merge pattern                   | `w1_div_if_else`, `k100_early_exit`, `k100_add64_chain`, `k300_nasty_pred_xor` |
| Running/accumulator reductions | multi-step accumulator + atomic hybrid             | `r1_accumulator`, `r1_running_xor` |
| Atom + reduce fusion         | `atom.and` combined with reduction                  | `w2_atom_and_reduce` |
| XOR-reduce / ILP-pred edge cases | single-kernel ad-hoc bugs                       | `k200_xor_reduce`, `ilp_pred_alu` |

None of these are in the multi-predicate guard family MP02 fixed.

---

## Intentionally Deferred

These are **not bugs**, they are scoped-out work items:

* **Forge / OpenCUDA wiring** — this repo only covers PTX → cubin. Cross-tool
  end-to-end (`.cu → .ptx → .cubin`) is provided by external tools (OpenCUDA)
  and is **not** part of the OpenPTXas packaging baseline.
* **HFMA2 / FMUL.I subsystem** — the FG66 HARD BAIL proved a post-scheduling
  HFMA2 rewrite corrupts the scoreboard. A proper isel-level integration is
  deferred; the 5 HFMA2 target kernels remain STRUCTURAL.
* **SHF harvest follow-up** — FG67-70 integrated isel-level SHF widening but
  full byte-exact convergence for the SHF family requires bottom-up encoder
  work outside the scope of MP02–MP04.
* **Capmerc/DRM authentication** — already auto-generated with the 0x5a
  universal signature; no further packaging work required.

---

## Validation Quickstart

Developer fresh-clone workflow:

```bash
git clone https://github.com/garrick99/openptxas
cd openptxas
python -m pytest tests/ -q              # non-GPU, ~75s, must be 865 passed
python scripts/health.py --frontier-only # ~6s, must show 0 MIXED, 0 errors
python scripts/health.py                 # full sweep (~1m20s on RTX 5090)
```

Requirements:
* Python 3.11+ (pure-Python backend, no pip dependencies)
* NVIDIA GPU + `nvcuda.dll` (or equivalent libcuda) for the GPU phases
* `ptxas` on PATH for comparison-based commands (`--compare ptxas`,
  `kdiff`, frontier classification)

### Interpreting `health.py` output

* `OVERALL: GREEN` — pytest green, 0 new GPU FAIL, 0 MIXED.
* `OVERALL: RED` — check whether the failing kernels intersect the "Known
  Unsupported Families" table. If yes, it is a known non-regression. If no,
  you have a regression to investigate.

---

## Artifact Layout

* `workbench.py` — suite driver, catalog, per-kernel measurement, leaderboard.
* `workbench_expanded.py` — registry of expansion kernels (k100_*, k200_*, etc).
* `sass/pipeline.py` — compile_ptx_source entrypoint; orchestrates parse →
  regalloc → isel → schedule → scoreboard → emit.
* `sass/scoreboard.py` — ctrl word generation (rbar/wdep/misc per-opcode).
* `sass/encoding/` — per-opcode encoders, all byte-verified against PTXAS.
* `tools/template_engine/regdiff.py` — byte-level OURS-vs-PTXAS classifier.
* `scripts/health.py` — the one-page health summary (this doc's entrypoint 2).
* `scripts/_health_gpu_worker.py` — subprocess worker used by health.py for
  per-kernel CUDA-context isolation.
