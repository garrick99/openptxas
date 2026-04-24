"""Test the minimal.ptx multi-ret repro end-to-end on GPU.

Kernel:
    cond_test(out, a, threshold, n) — for each lane i<n, if a[i] > threshold:
        out[i] = a[i] * 2.0, else out[i] = a[i] * 0.5.
"""
import sys, pathlib, ctypes
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from sass.pipeline import compile_ptx_source

here = pathlib.Path(__file__).parent
ptx = (here / 'minimal.ptx').read_text()
cub = compile_ptx_source(ptx)['cond_test']
cubin_path = str(here / '_ours.cubin')
open(cubin_path, 'wb').write(cub)

cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')

def ck(e, tag=''):
    if e != 0:
        name = ctypes.c_char_p(); desc = ctypes.c_char_p()
        cuda.cuGetErrorName(e, ctypes.byref(name))
        cuda.cuGetErrorString(e, ctypes.byref(desc))
        raise RuntimeError(f'{tag}: err={e} {name.value.decode() if name.value else ""} {desc.value.decode() if desc.value else ""}')

ck(cuda.cuInit(0))
dev = ctypes.c_int()
ck(cuda.cuDeviceGet(ctypes.byref(dev), 0))
cctx = ctypes.c_void_p()
ck(cuda.cuCtxCreate_v2(ctypes.byref(cctx), 0, dev))

mod = ctypes.c_void_p()
ck(cuda.cuModuleLoad(ctypes.byref(mod), cubin_path.encode()), 'load')
fn = ctypes.c_void_p()
ck(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b'cond_test'), 'get_fn')

N = 8
import struct
h_in = [-32.0, -1.0, 0.5, 2.0, 7.5, -10.0, 0.0, 100.0]
d_in = ctypes.c_uint64()
d_out = ctypes.c_uint64()
ck(cuda.cuMemAlloc_v2(ctypes.byref(d_in), 4*N))
ck(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4*N))
buf = (ctypes.c_float * N)(*h_in)
ck(cuda.cuMemcpyHtoD_v2(d_in, buf, 4*N))
ck(cuda.cuMemsetD8_v2(d_out, 0, 4*N))

arg_out = ctypes.c_uint64(d_out.value)
arg_a = ctypes.c_uint64(d_in.value)
arg_t = ctypes.c_float(1.0)   # threshold
arg_n = ctypes.c_int32(N)
args = (ctypes.c_void_p * 4)(
    ctypes.cast(ctypes.pointer(arg_out), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(arg_a), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(arg_t), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(arg_n), ctypes.c_void_p),
)
extra = ctypes.c_void_p(0)
ck(cuda.cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, args, extra), 'launch')
ck(cuda.cuCtxSynchronize(), 'sync')

host = (ctypes.c_float * N)()
ck(cuda.cuMemcpyDtoH_v2(host, d_out, 4*N))

ok = True
print(f"threshold = {arg_t.value}")
print(f"{'idx':>3} {'in':>8} {'out':>10} {'expected':>10} {'ok':>3}")
for i, v in enumerate(h_in):
    got = host[i]
    want = v * 2.0 if v > arg_t.value else v * 0.5
    passed = abs(got - want) < 1e-6
    ok &= passed
    print(f"{i:>3} {v:>8.3f} {got:>10.3f} {want:>10.3f} {'OK' if passed else 'FAIL':>3}")
print("RESULT:", "ALL PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
