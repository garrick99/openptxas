"""MP03 backend health summary.

Single entrypoint that runs the full validation surface and emits a one-page
summary.  Intended to answer "is the backend currently green?" in one command.

Sections
--------
1. pytest        — non-GPU unit/integration tests (tests/)
2. GPU harness   — compiles every registered kernel, runs the correctness
                   harness on GPU for every kernel that registers one, and
                   reports PASS/FAIL per kernel.
3. Frontier      — classifies every kernel through regdiff (OURS vs PTXAS
                   byte-level classification: BYTE_EXACT / STRUCTURAL / ...).

Usage
-----
    python scripts/health.py              # full surface
    python scripts/health.py --quick      # pytest + frontier only (no GPU)
    python scripts/health.py --gpu-only   # GPU harness only
    python scripts/health.py --frontier-only

Exit code is 0 iff every section passed (pytest green, 0 GPU FAILs, 0 MIXED,
0 errors).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _hr(ch='='):
    print(ch * 72)


def run_pytest() -> tuple[bool, str]:
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-q', '--no-header'],
        cwd=str(_ROOT), capture_output=True, text=True,
    )
    dt = time.perf_counter() - t0
    last = r.stdout.splitlines()[-1] if r.stdout else r.stderr.splitlines()[-1]
    ok = r.returncode == 0
    return ok, f'{last}   [{dt:.1f}s]'


def run_gpu_harness() -> tuple[bool, dict]:
    """Run the GPU correctness harness for every registered kernel.

    Runs each kernel as an isolated subprocess.  Subprocess isolation is the
    only reliable way to continue after a kernel hits an illegal instruction
    or illegal address — the driver context is then permanently poisoned and
    even cuCtxDestroy+cuCtxCreate doesn't recover in a single process.
    """
    import workbench  # noqa: E402
    results = {'PASS': [], 'FAIL': [], 'NO_HARNESS': [], 'COMPILE_FAIL': [],
               'RUN_EXC': []}
    t0 = time.perf_counter()
    worker = _ROOT / 'scripts' / '_health_gpu_worker.py'
    for name in sorted(workbench.KERNELS):
        k = workbench.KERNELS[name]
        if k.get('harness') is None:
            results['NO_HARNESS'].append(name)
            continue
        r = subprocess.run(
            [sys.executable, str(worker), name],
            cwd=str(_ROOT), capture_output=True, text=True, timeout=60,
        )
        verdict = (r.stdout or '').strip().splitlines()
        verdict = verdict[-1] if verdict else ''
        if verdict == 'PASS':
            results['PASS'].append(name)
        elif verdict == 'FAIL':
            results['FAIL'].append(name)
        elif verdict.startswith('COMPILE_FAIL:'):
            results['COMPILE_FAIL'].append((name, verdict[len('COMPILE_FAIL:'):].strip()))
        elif verdict.startswith('NO_HARNESS'):
            results['NO_HARNESS'].append(name)
        else:
            stderr_tail = (r.stderr or '').splitlines()
            tail = stderr_tail[-1] if stderr_tail else f'rc={r.returncode}'
            results['RUN_EXC'].append((name, verdict or tail))
    dt = time.perf_counter() - t0
    results['_dt'] = dt
    ok = len(results['FAIL']) == 0 and len(results['RUN_EXC']) == 0
    return ok, results


def run_frontier() -> tuple[bool, dict]:
    """Classify every registered kernel via regdiff."""
    import workbench  # noqa: E402
    from benchmarks.bench_util import compile_openptxas, compile_ptxas
    from tools.template_engine.regdiff import diff_kernel, DiffClass
    buckets = {c.value: [] for c in DiffClass}
    errors = []
    t0 = time.perf_counter()
    for name in sorted(workbench.KERNELS):
        k = workbench.KERNELS[name]
        src = k.get('ptx_inline')
        if src is None:
            p = k.get('ptx_path')
            if p and Path(p).exists():
                src = Path(p).read_text()
        if src is None:
            errors.append((name, 'no src')); continue
        try:
            ours, _ = compile_openptxas(src)
        except Exception as e:
            errors.append((name, f'ours {type(e).__name__}')); continue
        try:
            ptxas_bin, _ = compile_ptxas(src)
        except Exception as e:
            errors.append((name, f'ptxas {type(e).__name__}')); continue
        try:
            r = diff_kernel(ours, ptxas_bin, name)
            buckets[r.classification.value].append(name)
        except Exception as e:
            errors.append((name, f'diff {type(e).__name__}'))
    dt = time.perf_counter() - t0
    result = {'buckets': buckets, 'errors': errors, '_dt': dt}
    ok = buckets['MIXED'] == [] and errors == []
    return ok, result


def main(argv=None):
    ap = argparse.ArgumentParser(prog='health.py',
        description='OpenPTXas backend health summary (MP03 entrypoint).')
    ap.add_argument('--quick', action='store_true',
                    help='skip GPU harness')
    ap.add_argument('--gpu-only', action='store_true',
                    help='only run the GPU harness')
    ap.add_argument('--frontier-only', action='store_true',
                    help='only run the frontier classification')
    args = ap.parse_args(argv)

    overall_ok = True

    # 1) pytest
    if not (args.gpu_only or args.frontier_only):
        _hr(); print('[1/3] pytest (tests/)'); _hr('-')
        ok, summary = run_pytest()
        print(summary)
        overall_ok = overall_ok and ok
        print()

    # 2) GPU harness
    if not args.quick and not args.frontier_only:
        _hr(); print('[2/3] GPU harness (compile + cuLaunch + verify per kernel)')
        _hr('-')
        ok, res = run_gpu_harness()
        print(f'  PASS={len(res["PASS"])}  FAIL={len(res["FAIL"])}  '
              f'COMPILE_FAIL={len(res["COMPILE_FAIL"])}  '
              f'RUN_EXC={len(res["RUN_EXC"])}  '
              f'NO_HARNESS={len(res["NO_HARNESS"])}   [{res["_dt"]:.1f}s]')
        for name in res['FAIL']:
            print(f'    FAIL: {name}')
        for name, e in res['COMPILE_FAIL'][:10]:
            print(f'    COMPILE_FAIL: {name}: {e}')
        for name, e in res['RUN_EXC'][:10]:
            print(f'    RUN_EXC: {name}: {e}')
        overall_ok = overall_ok and ok
        print()

    # 3) Frontier recompute
    if not args.gpu_only:
        _hr(); print('[3/3] Frontier (regdiff OURS vs PTXAS per kernel)')
        _hr('-')
        ok, res = run_frontier()
        total = sum(len(v) for v in res['buckets'].values())
        print(f'  total={total}   [{res["_dt"]:.1f}s]')
        for k, v in res['buckets'].items():
            if v:
                print(f'    {k}: {len(v)}')
        if res['errors']:
            print(f'    errors: {len(res["errors"])}')
            for n, e in res['errors'][:5]:
                print(f'      {n}: {e}')
        overall_ok = overall_ok and ok
        print()

    _hr()
    print('OVERALL:', 'GREEN' if overall_ok else 'RED')
    _hr()
    return 0 if overall_ok else 1


if __name__ == '__main__':
    sys.exit(main())
