"""Diagnostic: check if kernel returns without crash. Also compare to ptxas."""
import sys, pathlib, ctypes, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from sass.pipeline import compile_ptx_source

here = pathlib.Path(__file__).parent
ptx = (here / 'minimal.ptx').read_text()
# Compile OURS
cub_ours = compile_ptx_source(ptx)['fuzz']
open(here / '_ours.cubin', 'wb').write(cub_ours)
# Compile ptxas THEIRS
subprocess.run(['ptxas', '-arch=sm_120', str(here/'minimal.ptx'), '-o', str(here/'_theirs.cubin')], check=True)

cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
def ck(e, tag=''):
    if e != 0:
        name = ctypes.c_char_p(); desc = ctypes.c_char_p()
        cuda.cuGetErrorName(e, ctypes.byref(name))
        cuda.cuGetErrorString(e, ctypes.byref(desc))
        raise RuntimeError(f'{tag}: err={e} {name.value.decode() if name.value else ""}')

ck(cuda.cuInit(0))
dev = ctypes.c_int()
ck(cuda.cuDeviceGet(ctypes.byref(dev), 0))
cctx = ctypes.c_void_p()
ck(cuda.cuCtxCreate_v2(ctypes.byref(cctx), 0, dev))

for label, cubin_path in [('OURS', str(here/'_ours.cubin')), ('THEIRS', str(here/'_theirs.cubin'))]:
    mod = ctypes.c_void_p()
    ck(cuda.cuModuleLoad(ctypes.byref(mod), cubin_path.encode()))
    fn = ctypes.c_void_p()
    ck(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b'fuzz'))

    N = 4
    d_in = ctypes.c_uint64(); d_out = ctypes.c_uint64()
    ck(cuda.cuMemAlloc_v2(ctypes.byref(d_in), 4*N))
    ck(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4*N))
    buf_in = (ctypes.c_uint32 * N)(*[i * 4 + 5 for i in range(N)])
    ck(cuda.cuMemcpyHtoD_v2(d_in, buf_in, 4*N))
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
    err = cuda.cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, args, extra)
    sync_err = cuda.cuCtxSynchronize()
    host = (ctypes.c_uint32 * N)()
    ck(cuda.cuMemcpyDtoH_v2(host, d_out, 4*N))
    print(f"{label}: launch={err} sync={sync_err} d_in={hex(d_in.value)} d_out={hex(d_out.value)}")
    for i in range(N):
        print(f"  [{i}] in={buf_in[i]} out=0x{host[i]:x}  (want {((buf_in[i]+3)>>2)+256})")

    ck(cuda.cuModuleUnload(mod))
    cuda.cuMemFree_v2(d_in); cuda.cuMemFree_v2(d_out)
