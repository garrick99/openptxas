"""Differential oracle: compile one PTX with both backends, run both
cubins on the same inputs, diff outputs.  Records one of:

    'ok'                — outputs bit-identical
    'divergence'        — both ran, outputs differ
    'sync_err_ours'     — our cubin crashed (illegal addr, etc.)
    'sync_err_theirs'   — ptxas cubin crashed
    'compile_err_ours'  — our compiler refused this PTX
    'compile_err_theirs'— ptxas refused this PTX
    'load_err'          — driver rejected a cubin at load time

Every non-'ok' outcome generates an artifact: dir named by the PTX
normalized-form hash, containing input.ptx + both cubins + both
outputs + meta.json.
"""
import ctypes, hashlib, json, os, struct, subprocess, sys, tempfile, time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from ptx.parser import parse
from sass.pipeline import compile_function
from fuzzer.generator import generate, normalize

# Platform-aware ptxas path and libcuda loader.  Honor PTXAS_PATH env
# var if set; else pick a sensible default for the running OS.
_IS_WIN = sys.platform.startswith('win')
if _IS_WIN:
    _DEFAULT_PTXAS = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\ptxas.exe'
    _LIBCUDA_NAME = 'nvcuda'
else:
    _DEFAULT_PTXAS = '/usr/local/cuda-13.2/bin/ptxas'
    _LIBCUDA_NAME = 'libcuda.so.1'

PTXAS = os.environ.get('PTXAS_PATH', _DEFAULT_PTXAS)
ARTIFACT_DIR = _REPO / '_fuzz' / 'artifacts'
N_THREADS = 32


def _load_libcuda():
    if _IS_WIN:
        return ctypes.WinDLL(_LIBCUDA_NAME)
    return ctypes.CDLL(_LIBCUDA_NAME)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def compile_ours(ptx: str):
    try:
        # enable_dce: the fuzzer intentionally emits dead filler ops to
        # vary family signatures; without DCE, regalloc reuses a physical
        # reg across dead+live writes and the hardware zeroes the live
        # value (SHIFT_BOUNDARY / SIGN_FLIP_CHAIN bug classes).
        # error_on_unimplemented: fail-closed on unsupported PTX so the
        # differ classifies it as compile_err_ours instead of silently
        # NOPing the op and producing a 'theirs_correct' miscompile.
        cubin = compile_function(parse(ptx).functions[0],
                                  verbose=False, sm_version=120,
                                  enable_dce=True,
                                  error_on_unimplemented=True)
        return cubin, None
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'


def compile_theirs(ptx: str):
    with tempfile.NamedTemporaryFile(suffix='.ptx', delete=False, mode='w') as f:
        f.write(ptx); p = f.name
    cp = p.replace('.ptx', '.cubin')
    try:
        r = subprocess.run([PTXAS, '-arch=sm_120', p, '-o', cp],
                           capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None, r.stderr[:200]
        with open(cp, 'rb') as f:
            return f.read(), None
    except subprocess.TimeoutExpired:
        return None, 'ptxas timeout'
    finally:
        try: os.unlink(p)
        except: pass
        try: os.unlink(cp)
        except: pass


class CudaRunner:
    """Persistent CUDA context.  Load two modules per iteration,
    launch each, read outputs, unload."""
    def __init__(self):
        self.cuda = _load_libcuda()
        self.cuda.cuInit(0)
        dev = ctypes.c_int(); self.cuda.cuDeviceGet(ctypes.byref(dev), 0)
        self.dev = dev
        self.ctx = ctypes.c_void_p()
        err = self.cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev)
        if err:
            raise RuntimeError(f'cuCtxCreate failed: {err}')

    def reset(self):
        """Destroy + recreate ctx after a bad launch to clear driver state."""
        try: self.cuda.cuCtxDestroy_v2(self.ctx)
        except: pass
        self.ctx = ctypes.c_void_p()
        err = self.cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, self.dev)
        if err:
            raise RuntimeError(f'cuCtxCreate reset failed: {err}')

    def run_cubin(self, cubin: bytes, input_bytes: bytes, n: int):
        """Load cubin, launch, read output.  Returns (output_bytes, sync_err)
        or (None, sync_err_code) on failure.  Module is unloaded, all
        allocations freed."""
        cuda = self.cuda
        mod = ctypes.c_void_p()
        err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
        if err:
            return None, f'load_err={err}'
        func = ctypes.c_void_p()
        err = cuda.cuModuleGetFunction(ctypes.byref(func), mod, b'fuzz')
        if err:
            cuda.cuModuleUnload(mod)
            return None, f'getfunc_err={err}'
        sz = n * 4
        d_in = ctypes.c_uint64()
        d_out = ctypes.c_uint64()
        cuda.cuMemAlloc_v2(ctypes.byref(d_in), sz)
        cuda.cuMemAlloc_v2(ctypes.byref(d_out), sz)
        try:
            cuda.cuMemcpyHtoD_v2(d_in, input_bytes, sz)
            cuda.cuMemsetD8_v2(d_out, 0, sz)
            a_in = ctypes.c_uint64(d_in.value)
            a_out = ctypes.c_uint64(d_out.value)
            a_n = ctypes.c_uint32(n)
            argv = (ctypes.c_void_p * 3)(
                ctypes.cast(ctypes.byref(a_in), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(a_out), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(a_n), ctypes.c_void_p))
            err = cuda.cuLaunchKernel(func, 1,1,1, n,1,1, 0, None, argv, None)
            if err:
                return None, f'launch_err={err}'
            err = cuda.cuCtxSynchronize()
            if err:
                return None, f'sync_err={err}'
            buf = ctypes.create_string_buffer(sz)
            cuda.cuMemcpyDtoH_v2(buf, d_out, sz)
            return bytes(buf.raw), None
        finally:
            cuda.cuMemFree_v2(d_in)
            cuda.cuMemFree_v2(d_out)
            cuda.cuModuleUnload(mod)

    def close(self):
        self.cuda.cuCtxDestroy_v2(self.ctx)


def _save_artifact(tag, ptx, ours, theirs, input_bytes, out_ours, out_theirs,
                    meta):
    sha = _sha(normalize(ptx))
    d = ARTIFACT_DIR / f'{tag}_{sha}'
    if d.exists():
        # Dedup: bump counter
        mf = d / 'meta.json'
        m = json.loads(mf.read_text()) if mf.exists() else {}
        m['hits'] = m.get('hits', 1) + 1
        m['last_seen'] = int(time.time())
        mf.write_text(json.dumps(m, indent=2))
        return False  # not new
    d.mkdir(parents=True, exist_ok=True)
    (d / 'input.ptx').write_text(ptx)
    if ours: (d / 'ours.cubin').write_bytes(ours)
    if theirs: (d / 'theirs.cubin').write_bytes(theirs)
    if input_bytes: (d / 'input.bin').write_bytes(input_bytes)
    if out_ours is not None: (d / 'ours.out').write_bytes(out_ours)
    if out_theirs is not None: (d / 'theirs.out').write_bytes(out_theirs)
    meta['hits'] = 1
    meta['first_seen'] = int(time.time())
    meta['last_seen'] = int(time.time())
    (d / 'meta.json').write_text(json.dumps(meta, indent=2))
    return True  # new unique finding


def test_ptx(ptx: str, seed: int, runner: CudaRunner, input_bytes: bytes,
              family: str = 'alu_int'):
    """Compile+run+diff a PTX string.  Caller supplies the PTX so any
    family's generator can plug in."""
    ours, err_ours = compile_ours(ptx)
    theirs, err_theirs = compile_theirs(ptx)
    base_meta = {'seed': seed, 'family': family}

    if err_ours is not None and err_theirs is not None:
        return 'both_reject', False
    if err_ours is not None:
        new = _save_artifact('compile_err_ours', ptx, None, theirs, None,
                              None, None, {**base_meta, 'err': err_ours[:200]})
        return 'compile_err_ours', new
    if err_theirs is not None:
        new = _save_artifact('compile_err_theirs', ptx, ours, None, None,
                              None, None, {**base_meta, 'err': err_theirs[:200]})
        return 'compile_err_theirs', new

    out_ours, se_ours = runner.run_cubin(ours, input_bytes, N_THREADS)
    if se_ours is not None:
        new = _save_artifact('sync_err_ours', ptx, ours, theirs, input_bytes,
                              None, None, {**base_meta, 'err': se_ours})
        runner.reset()
        return 'sync_err_ours', new
    out_theirs, se_theirs = runner.run_cubin(theirs, input_bytes, N_THREADS)
    if se_theirs is not None:
        new = _save_artifact('sync_err_theirs', ptx, ours, theirs, input_bytes,
                              out_ours, None, {**base_meta, 'err': se_theirs})
        runner.reset()
        return 'sync_err_theirs', new

    if out_ours != out_theirs:
        # Filter spurious divergences where the kernel reads an undefined
        # register.  Both compilers allocate registers independently and
        # undef reads pick up whatever physical reg was last written —
        # OURS and ptxas can disagree on coloring, producing "different"
        # outputs that are both valid unspecified results.  These are
        # fuzzer noise, not real miscompiles.
        from fuzzer.minimize import _is_well_formed
        if not _is_well_formed(ptx):
            return 'spurious_divergence', False
        new = _save_artifact('divergence', ptx, ours, theirs, input_bytes,
                              out_ours, out_theirs, base_meta)
        return 'divergence', new
    return 'ok', False


def test_one(seed: int, runner: CudaRunner, input_bytes: bytes):
    """Backward-compat wrapper: generates alu_int PTX inline."""
    ptx, _ = generate(seed)
    return test_ptx(ptx, seed, runner, input_bytes, family='alu_int')


if __name__ == '__main__':
    # Smoke test: 10 iters
    import random
    runner = CudaRunner()
    try:
        for seed in range(10):
            inp = random.Random(seed + 10000).randbytes(N_THREADS * 4)
            status, new = test_one(seed, runner, inp)
            print(f'  seed={seed}: {status}{"  [NEW]" if new else ""}')
    finally:
        runner.close()
