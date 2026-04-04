"""Standalone GPU test for OpenCUDA vector_add - can be run under compute-sanitizer."""
import ctypes, struct, sys, array
sys.path.insert(0, '.')
from pathlib import Path
from sass.pipeline import compile_ptx_source

ptx = Path('C:/Users/kraken/opencuda/tests/vector_add.ptx').read_text()
results = compile_ptx_source(ptx)
cubin = results['vector_add']
with open('probe_work/opencuda_vecadd_test.cubin', 'wb') as f:
    f.write(cubin)

cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
CUdevice = ctypes.c_int; CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p; CUfunction = ctypes.c_void_p; CUdeviceptr = ctypes.c_uint64
def check(err, msg=''):
    if err != 0:
        n = ctypes.c_char_p(); cuda.cuGetErrorName(err, ctypes.byref(n))
        print(f'CUDA {err}: {n.value.decode()} [{msg}]'); sys.exit(1)

cuda.cuInit(0); dev = CUdevice(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = CUcontext(); cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
mod = CUmodule(); check(cuda.cuModuleLoadData(ctypes.byref(mod), cubin), 'load')
func = CUfunction(); check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, b'vector_add'), 'func')

N = 1; nb = 4
d_out = CUdeviceptr(); d_a = CUdeviceptr(); d_b = CUdeviceptr()
check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), nb))
check(cuda.cuMemAlloc_v2(ctypes.byref(d_a), nb))
check(cuda.cuMemAlloc_v2(ctypes.byref(d_b), nb))
check(cuda.cuMemcpyHtoD_v2(d_a, struct.pack('<f', 1.0), 4))
check(cuda.cuMemcpyHtoD_v2(d_b, struct.pack('<f', 2.0), 4))
check(cuda.cuMemcpyHtoD_v2(d_out, struct.pack('<f', 0.0), 4))

arg_out = ctypes.c_uint64(d_out.value); arg_a = ctypes.c_uint64(d_a.value)
arg_b = ctypes.c_uint64(d_b.value); arg_n = ctypes.c_int32(N)
args = (ctypes.c_void_p * 4)(
    ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
    ctypes.cast(ctypes.byref(arg_a), ctypes.c_void_p),
    ctypes.cast(ctypes.byref(arg_b), ctypes.c_void_p),
    ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
)
check(cuda.cuLaunchKernel(func, 1,1,1, 1,1,1, 0, None, args, None), 'launch')
check(cuda.cuCtxSynchronize(), 'sync')

out = (ctypes.c_float * 1)()
check(cuda.cuMemcpyDtoH_v2(out, d_out, 4))
print(f'out[0] = {out[0]} (expected 3.0)')
cuda.cuMemFree_v2(d_out); cuda.cuMemFree_v2(d_a); cuda.cuMemFree_v2(d_b)
cuda.cuModuleUnload(mod); cuda.cuCtxDestroy_v2(ctx)
