"""Single-kernel GPU correctness worker for scripts/health.py.

Usage: python _health_gpu_worker.py <kernel_name>
Prints exactly one verdict line on stdout:
    PASS
    FAIL
    NO_HARNESS
    COMPILE_FAIL: <msg>
    RUN_EXC: <msg>
Exit code is 0 iff verdict is PASS.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def main():
    if len(sys.argv) != 2:
        print('RUN_EXC: bad args')
        return 2
    name = sys.argv[1]
    import workbench
    k = workbench.KERNELS.get(name)
    if k is None:
        print('RUN_EXC: unknown kernel')
        return 2
    if k.get('harness') is None:
        print('NO_HARNESS')
        return 0
    src = k.get('ptx_inline')
    if src is None:
        p = k.get('ptx_path')
        if p and Path(p).exists():
            src = Path(p).read_text()
    if src is None:
        print('NO_HARNESS')
        return 0
    from benchmarks.bench_util import compile_openptxas, CUDAContext
    try:
        cubin, _ = compile_openptxas(src)
    except Exception as e:
        msg = f'{type(e).__name__}: {e}'
        print(f'COMPILE_FAIL: {msg[:200]}')
        return 1
    ctx = CUDAContext()
    try:
        if not ctx.load(cubin):
            print('RUN_EXC: cuModuleLoadData failed')
            return 1
        try:
            func = ctx.get_func(k['kernel_name'])
        except Exception as e:
            print(f'RUN_EXC: get_func {type(e).__name__}')
            return 1
        try:
            r = k['harness'](ctx, func, 'correctness')
            print('PASS' if r.get('correct') else 'FAIL')
            return 0 if r.get('correct') else 1
        except Exception as e:
            msg = f'{type(e).__name__}: {e}'
            print(f'RUN_EXC: {msg[:200]}')
            return 1
    finally:
        try: ctx.close()
        except Exception: pass


if __name__ == '__main__':
    sys.exit(main())
