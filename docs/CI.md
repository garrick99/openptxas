# CI — OpenPTXas regression gating

Two-tier gate enforced by `.github/workflows/corpus.yml`:

| Tier | Job | Runner | Coverage | Required? |
|------|-----|--------|----------|-----------|
| 1 | `compile-gate` | GitHub-hosted Ubuntu | Compiles 142 fixtures; no execution | Yes (every PR) |
| 2 | `corpus-gpu` | Self-hosted SM_120 | Full `corpus_sweep.py` end-to-end | Main branch + `gpu-check` label |

## Tier 1 — compile gate

Runs on every PR using `ubuntu-latest`. Installs the package (`pip install -e .`) and runs `scripts/corpus_compile_check.py`. No CUDA driver needed.

Failure conditions:
- any fixture raises during `compile_function`
- any fixture emits an empty cubin

Expected output at `HEAD = 033f398`:
```
[compile_check] total=142 pass=142 fail=0
```

## Tier 2 — GPU corpus sweep

Runs on a self-hosted runner with the labels `[self-hosted, gpu, sm_120]`. Executes every fixture in a fresh CUDA context (one subprocess per kernel) via `scripts/corpus_sweep.py`.

Triggers:
- `push` to `main`
- `workflow_dispatch` (manual)
- PR with the `gpu-check` label (avoids queueing every PR on the GPU runner)

Expected output at `HEAD = 033f398`:
```
[corpus_sweep] total=142 pass=142 fail=0
```

## Registering a self-hosted runner

Any machine with an SM_120 GPU and the CUDA driver (>= 12.x) can serve as the Tier 2 runner.

1. From the repo: **Settings → Actions → Runners → New self-hosted runner**. Follow the OS-specific instructions.
2. When prompted for labels, add `gpu` and `sm_120` (alongside the default `self-hosted`, OS, arch labels).
3. Install dependencies the runner needs:
   ```bash
   python -m pip install --upgrade pip
   pip install -e /path/to/openptxas
   ```
4. Ensure `nvcuda` is loadable (on Windows: `nvcuda.dll` on the PATH; on Linux: `libcuda.so`).
5. Start the runner service. Confirm by triggering `workflow_dispatch` on the `corpus` workflow.

## Local equivalents

Both gates are runnable directly:

```bash
# Tier 1 — fast, no GPU, good for pre-commit
python scripts/corpus_compile_check.py

# Tier 2 — full execution sweep (needs SM_120 GPU + CUDA driver)
python scripts/corpus_sweep.py
```

Both exit `0` on green, `1` on any failure — suitable for Git hooks.

## Expected flow for a regression

1. Developer opens PR. Tier 1 runs immediately on the Ubuntu runner.
2. If Tier 1 fails: PR blocked until compile regression is fixed.
3. If the PR touches the backend (anything under `sass/`, `ptx/`, or `scripts/`), reviewer adds the `gpu-check` label. Tier 2 queues on the self-hosted runner.
4. If Tier 2 fails: PR blocked until the hardware regression is fixed.
5. On merge to `main`: Tier 2 runs unconditionally as the canonical gate.
