"""R54: broad sweep of workbench_expanded fixtures.  Checks compile +
launch + sync status for every PTX fixture."""
import ctypes, struct, sys, re, traceback
sys.path.insert(0, 'C:/Users/kraken/openptxas')
import workbench_expanded as we
import workbench as wb
from ptx.parser import parse
from sass.pipeline import compile_function


def enumerate_fixtures():
    out = []
    for mod in (we, wb):
        for name in dir(mod):
            if name.startswith('_') and name.isupper():
                val = getattr(mod, name)
                if isinstance(val, str) and '.entry' in val and '.target sm_120' in val:
                    m = re.search(r'\.entry\s+(\w+)', val)
                    if m:
                        # count params
                        pm = re.findall(r'\.param\s+\.([us]\d+|f\d+|b\d+)', val)
                        has_smem = '.shared' in val
                        has_bar = 'bar.sync' in val
                        out.append({'name': m.group(1), 'ptx': val, 'params': pm,
                                    'has_smem': has_smem, 'has_bar': has_bar})
    # dedupe by name
    seen = set(); uniq = []
    for f in out:
        if f['name'] in seen: continue
        seen.add(f['name']); uniq.append(f)
    return uniq


def try_launch(fix, ctx, cuda):
    name = fix['name']
    params = fix['params']
    # Launch with N=32 threads, 1 block.  Allocate output + optional input + n param
    N = 32
    sz = N * 4
    allocs = []
    arg_containers = []
    # Build args based on params
    # Assume convention: first param is p_out (u64), optional p_in (u64), last may be n (u32)
    try:
        for i, p in enumerate(params):
            if p == 'u64':
                d = ctypes.c_uint64()
                cuda.cuMemAlloc_v2(ctypes.byref(d), sz)
                cuda.cuMemsetD8_v2(d, 0, sz)
                allocs.append(d.value)
                if i == 1:
                    # assume p_in, fill with tids
                    cuda.cuMemcpyHtoD_v2(d, struct.pack(f'<{N}I', *range(N)), sz)
                arg_containers.append(ctypes.c_uint64(d.value))
            elif p in ('u32', 's32'):
                # n param: thread count
                arg_containers.append(ctypes.c_uint32(N))
            else:
                arg_containers.append(ctypes.c_uint32(0))
        argv = (ctypes.c_void_p * len(arg_containers))(*[
            ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in arg_containers])

        # Compile
        try:
            cubin = compile_function(parse(fix['ptx']).functions[0], verbose=False, sm_version=120)
        except Exception as e:
            return ('compile_err', repr(e)[:80])
        mod = ctypes.c_void_p()
        err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
        if err:
            return ('load_err', f'cuModuleLoadData={err}')
        func = ctypes.c_void_p()
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, name.encode())
        # Launch
        smem_bytes = 2048 if fix['has_smem'] else 0
        launch_err = cuda.cuLaunchKernel(func, 1,1,1, N,1,1, smem_bytes, None, argv, None)
        sync_err = cuda.cuCtxSynchronize()
        if launch_err:
            return ('launch_err', f'launch={launch_err}')
        if sync_err:
            return ('sync_err', f'sync={sync_err}')
        return ('ok', '')
    finally:
        for a in allocs:
            cuda.cuMemFree_v2(ctypes.c_uint64(a))


def main():
    fixtures = enumerate_fixtures()
    print(f'[sweep] {len(fixtures)} fixtures')
    cuda = ctypes.WinDLL('nvcuda'); cuda.cuInit(0)
    dev = ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p(); cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    results = {}
    try:
        for fix in fixtures:
            try:
                status, detail = try_launch(fix, ctx, cuda)
            except Exception as e:
                status = 'exception'
                detail = repr(e)[:80]
            results[fix['name']] = (status, detail)
            marker = 'OK' if status == 'ok' else status.upper()
            print(f'  {fix["name"]:36s} {marker:12s} {detail}')
    finally:
        cuda.cuCtxDestroy_v2(ctx)
    # summary
    total = len(results)
    passes = sum(1 for s, _ in results.values() if s == 'ok')
    print(f'\n[sweep] total={total} pass={passes} fail={total-passes}')
    # fail groups
    groups = {}
    for n, (s, d) in results.items():
        if s != 'ok':
            groups.setdefault(s, []).append(n)
    for k, v in groups.items():
        print(f'  {k}: {len(v)} — {v[:10]}')


if __name__ == '__main__':
    main()
