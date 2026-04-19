"""Compile-only corpus check — no GPU required.

Enumerates every .entry sm_120 fixture in workbench.py and
workbench_expanded.py and compiles each one through compile_function.
Does NOT load or execute the resulting cubin — safe to run on CPU-only
CI runners.

Exit status:
  0 — every fixture compiles without raising
  1 — one or more fixtures fail to compile

Usage:
  python scripts/corpus_compile_check.py

See scripts/corpus_sweep.py for the full GPU-execution gate.
"""
import re, sys, traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'benchmarks'))
import workbench_expanded as we
import workbench as wb
from ptx.parser import parse
from sass.pipeline import compile_function


def enumerate_fixtures():
    """Enumerate every .entry sm_120 PTX source:
       * workbench.py + workbench_expanded.py (inline fixtures)
       * benchmarks/*_vs_nvidia.py (`*_PTX` module constants)
    """
    out = []
    # inline fixtures
    for mod in (we, wb):
        for name in dir(mod):
            if not (name.startswith('_') and name.isupper()):
                continue
            val = getattr(mod, name)
            if not (isinstance(val, str) and '.entry' in val
                    and '.target sm_120' in val):
                continue
            m = re.search(r'\.entry\s+(\w+)', val)
            if m:
                out.append((m.group(1), val))
    # benchmark PTX
    bench_mods = [
        'saxpy_vs_nvidia', 'vecadd_vs_nvidia', 'memcpy_vs_nvidia',
        'scale_vs_nvidia', 'stencil_vs_nvidia', 'relu_vs_nvidia',
        'fmachain_vs_nvidia',
    ]
    for mname in bench_mods:
        try:
            mod = __import__(mname)
        except Exception:
            continue
        for attr in dir(mod):
            if not attr.endswith('_PTX'):
                continue
            val = getattr(mod, attr, None)
            if not (isinstance(val, str) and '.entry' in val
                    and '.target sm_120' in val):
                continue
            m = re.search(r'\.entry\s+(\w+)', val)
            if m:
                out.append((m.group(1), val))
    seen = set(); uniq = []
    for n, v in out:
        if n in seen:
            continue
        seen.add(n); uniq.append((n, v))
    return uniq


def main():
    fixtures = enumerate_fixtures()
    print(f'[compile_check] {len(fixtures)} fixtures')
    fails = []
    for name, ptx in fixtures:
        try:
            cubin = compile_function(parse(ptx).functions[0],
                                      verbose=False, sm_version=120)
            ok = isinstance(cubin, (bytes, bytearray)) and len(cubin) > 0
            if not ok:
                raise RuntimeError('empty cubin')
            print(f'  {name:40s} OK  ({len(cubin)} bytes)')
        except Exception as e:
            fails.append((name, e))
            print(f'  {name:40s} FAIL  {e!r}'[:140])
    total = len(fixtures)
    passes = total - len(fails)
    print(f'\n[compile_check] total={total} pass={passes} fail={len(fails)}')
    if fails:
        print('\nfailures:')
        for name, e in fails:
            print(f'  {name}: {e!r}'[:200])
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
