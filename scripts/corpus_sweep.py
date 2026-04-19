"""Green-corpus sweep for the OpenPTXas backend.

Enumerates every .entry sm_120 PTX fixture declared in workbench.py and
workbench_expanded.py, compiles each one through compile_function, and
runs it in a fresh CUDA context (one subprocess per kernel to avoid
context poisoning on illegal-address sync errors).

Exit status:
  0   — all fixtures pass sync=0
  1   — one or more fixtures fail

Usage:
  python scripts/corpus_sweep.py

Expected at HEAD 23959b1+: total=142 pass=142 fail=0
See MILESTONE_142_142.md for campaign context.
"""
import ctypes, struct, sys, re, subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
import workbench_expanded as we
import workbench as wb


def _ptx_fixture(name, ptx):
    params = re.findall(r'\.param\s+\.([us]\d+|f\d+|b\d+)', ptx)
    return {
        'name': name,
        'ptx': ptx,
        'params': params,
        'has_smem': '.shared' in ptx,
        'has_bar': 'bar.sync' in ptx,
    }


def _collect_mod(mod):
    out = []
    for name in dir(mod):
        if not (name.startswith('_') and name.isupper()):
            continue
        val = getattr(mod, name, None)
        if not (isinstance(val, str) and '.entry' in val and '.target sm_120' in val):
            continue
        m = re.search(r'\.entry\s+(\w+)', val)
        if m:
            out.append(_ptx_fixture(m.group(1), val))
    return out


def enumerate_fixtures():
    out = []
    out.extend(_collect_mod(we))
    out.extend(_collect_mod(wb))
    sys.path.insert(0, str(_REPO / 'benchmarks'))
    for mname in ('saxpy_vs_nvidia', 'vecadd_vs_nvidia', 'memcpy_vs_nvidia',
                  'scale_vs_nvidia', 'stencil_vs_nvidia', 'relu_vs_nvidia',
                  'fmachain_vs_nvidia'):
        try:
            mod = __import__(mname)
        except Exception:
            continue
        for attr in dir(mod):
            if not attr.endswith('_PTX'):
                continue
            val = getattr(mod, attr, None)
            if not (isinstance(val, str) and '.entry' in val and '.target sm_120' in val):
                continue
            m = re.search(r'\.entry\s+(\w+)', val)
            if m:
                out.append(_ptx_fixture(m.group(1), val))
    seen = set(); uniq = []
    for f in out:
        if f['name'] in seen:
            continue
        seen.add(f['name']); uniq.append(f)
    return uniq


def run_one(fix, repo_path):
    """Run one kernel in a fresh subprocess (own CUDA context)."""
    script = f"""
import ctypes, struct, sys
sys.path.insert(0, {str(repo_path)!r})
import workbench_expanded as we
import workbench as wb
from ptx.parser import parse
from sass.pipeline import compile_function

fix = {fix!r}
params = fix['params']
try:
    cubin = compile_function(parse(fix['ptx']).functions[0], verbose=False, sm_version=120)
except Exception as e:
    print(f'compile_err={{e!r}}'); sys.exit(0)

cuda = ctypes.WinDLL('nvcuda'); cuda.cuInit(0)
dev = ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p(); cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
try:
    mod = ctypes.c_void_p()
    err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
    if err:
        print(f'load_err={{err}}'); sys.exit(0)
    func = ctypes.c_void_p()
    err = cuda.cuModuleGetFunction(ctypes.byref(func), mod, fix['name'].encode())
    if err:
        print(f'getfunc_err={{err}}'); sys.exit(0)
    N = 32; sz = N * 4
    allocs = []; arg_containers = []
    try:
        for i, p in enumerate(params):
            if p == 'u64':
                d = ctypes.c_uint64()
                cuda.cuMemAlloc_v2(ctypes.byref(d), sz)
                cuda.cuMemsetD8_v2(d, 0, sz)
                allocs.append(d.value)
                if i == 1:
                    cuda.cuMemcpyHtoD_v2(d, struct.pack(f'<{{N}}I', *range(N)), sz)
                arg_containers.append(ctypes.c_uint64(d.value))
            elif p in ('u32', 's32'):
                arg_containers.append(ctypes.c_uint32(N))
            else:
                arg_containers.append(ctypes.c_uint32(0))
        argv = (ctypes.c_void_p * len(arg_containers))(*[
            ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in arg_containers])
        smem_bytes = 2048 if fix['has_smem'] else 0
        launch_err = cuda.cuLaunchKernel(func, 1,1,1, N,1,1, smem_bytes, None, argv, None)
        if launch_err:
            print(f'launch_err={{launch_err}}'); sys.exit(0)
        sync_err = cuda.cuCtxSynchronize()
        if sync_err:
            print(f'sync_err={{sync_err}}'); sys.exit(0)
        print('ok')
    finally:
        for a in allocs:
            cuda.cuMemFree_v2(ctypes.c_uint64(a))
finally:
    cuda.cuCtxDestroy_v2(ctx)
"""
    try:
        r = subprocess.run([sys.executable, '-c', script],
                            capture_output=True, text=True, timeout=15)
        out = r.stdout.strip().splitlines()
        if not out:
            return ('proc_err', (r.stderr or '')[:80])
        last = out[-1]
        if '=' in last:
            return (last.split('=')[0], last)
        return (last, last)
    except subprocess.TimeoutExpired:
        return ('timeout', 'timeout')
    except Exception as e:
        return ('proc_err', repr(e)[:80])


def _load_known_fails():
    """Read scripts/known_fail.txt.  One fixture name per line; `#` comments
    and blank lines ignored.  Returns a set of fixture names."""
    path = Path(__file__).resolve().parent / 'known_fail.txt'
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            out.add(line)
    return out


def main():
    fixtures = enumerate_fixtures()
    known_fails = _load_known_fails()
    print(f'[corpus_sweep] {len(fixtures)} fixtures  (known_fails allowlist: {len(known_fails)})')
    results = {}
    for fix in fixtures:
        status, detail = run_one(fix, _REPO)
        results[fix['name']] = (status, detail)
        marker = 'OK' if status == 'ok' else status.upper()
        print(f'  {fix["name"]:36s} {marker:15s} {detail[:60]}')
    total = len(results)
    passes = sum(1 for s, _ in results.values() if s == 'ok')
    fails = total - passes
    failing_set = {n for n, (s, _) in results.items() if s != 'ok'}
    unexpected_fails = failing_set - known_fails
    unexpected_passes = known_fails - failing_set  # fixture in allowlist that passed
    print(f'\n[corpus_sweep] total={total} pass={passes} fail={fails}')
    if fails:
        groups = {}
        for n, (s, d) in results.items():
            if s != 'ok':
                groups.setdefault(s, []).append(n)
        for k, v in sorted(groups.items()):
            print(f'\n  {k} ({len(v)}):')
            for n in v:
                flag = '' if n in known_fails else '  [UNEXPECTED]'
                print(f'    {n}: {results[n][1][:60]}{flag}')
    # Regression detection:
    #   * unexpected_fails: something NEW broke -> fail CI
    #   * unexpected_passes: something known-broken is now passing -> also fail
    #     CI so the allowlist gets reviewed (good news should still prompt a
    #     deliberate update of scripts/known_fail.txt).
    if unexpected_fails:
        print(f'\n[corpus_sweep] UNEXPECTED failures — NEW regressions not in allowlist:')
        for n in sorted(unexpected_fails):
            print(f'  {n}: {results[n][1][:60]}')
        return 1
    if unexpected_passes:
        print(f'\n[corpus_sweep] UNEXPECTED passes — known-fails that now work '
              f'(update scripts/known_fail.txt):')
        for n in sorted(unexpected_passes):
            print(f'  {n}')
        return 1
    print('\n[corpus_sweep] OK — failing set matches allowlist exactly.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
