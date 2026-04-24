"""Reproduce the UR4-clobber bug: fuzz kernel with @P0 ret guard.

Expected per lane: ((input + 3) >> 2) + 256
If UR4 clobber happens, output buffer stays at sentinel 0xABCDEF01.
"""
import sys, pathlib, ctypes
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from sass.pipeline import compile_ptx_source

here = pathlib.Path(__file__).parent
ptx = (here / 'minimal.ptx').read_text()
cub = compile_ptx_source(ptx)['fuzz']
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
ck(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b'fuzz'), 'get_fn')

N = 32
d_in = ctypes.c_uint64(); d_out = ctypes.c_uint64()
ck(cuda.cuMemAlloc_v2(ctypes.byref(d_in), 4*N))
ck(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4*N))

buf_in = (ctypes.c_uint32 * N)(*[i * 4 + 5 for i in range(N)])  # inputs
ck(cuda.cuMemcpyHtoD_v2(d_in, buf_in, 4*N))
# Fill out buffer with sentinel
buf_sent = (ctypes.c_uint32 * N)(*[0xABCDEF01] * N)
ck(cuda.cuMemcpyHtoD_v2(d_out, buf_sent, 4*N))

arg_p_in = ctypes.c_uint64(d_in.value)
arg_p_out = ctypes.c_uint64(d_out.value)
arg_n = ctypes.c_uint32(N)
args = (ctypes.c_void_p * 3)(
    ctypes.cast(ctypes.pointer(arg_p_in), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(arg_p_out), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(arg_n), ctypes.c_void_p),
)
extra = ctypes.c_void_p(0)
ck(cuda.cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, args, extra), 'launch')
ck(cuda.cuCtxSynchronize(), 'sync')

host = (ctypes.c_uint32 * N)()
ck(cuda.cuMemcpyDtoH_v2(host, d_out, 4*N))

ok = True
unwritten = 0
print(f"{'idx':>3} {'in':>8} {'got':>10} {'want':>10} {'status':>8}")
for i in range(N):
    inv = buf_in[i]
    want = ((inv + 3) >> 2) + 256
    got = host[i]
    if got == 0xABCDEF01:
        unwritten += 1
        status = 'UNWRIT'
        ok = False
    elif got == want:
        status = 'OK'
    else:
        status = 'WRONG'
        ok = False
    print(f"{i:>3} {inv:>8} {got:>10} {want:>10} {status:>8}")
print(f"\nunwritten: {unwritten}/{N}")
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
