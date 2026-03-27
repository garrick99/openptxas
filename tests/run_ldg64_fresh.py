"""Isolated test for ldg64_test_fresh.cubin — run in a clean process."""
import ctypes
import struct
import sys
import os

CUBIN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'probe_work', 'ldg64_test_fresh.cubin')
CUBIN = os.path.normpath(CUBIN)

cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64


def check(err, msg=""):
    if err != 0:
        name = ctypes.c_char_p()
        desc = ctypes.c_char_p()
        cuda.cuGetErrorName(err, ctypes.byref(name))
        cuda.cuGetErrorString(err, ctypes.byref(desc))
        n = name.value.decode() if name.value else "?"
        d = desc.value.decode() if desc.value else "?"
        raise RuntimeError(f"CUDA ERROR {err}: {n} - {d} [{msg}]")


with open(CUBIN, 'rb') as f:
    cubin_data = f.read()

print(f"Cubin: {len(cubin_data)} bytes")

check(cuda.cuInit(0), "cuInit")
dev = CUdevice()
check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ctx = CUcontext()
check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

mod = CUmodule()
err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin_data)
if err != 0:
    name = ctypes.c_char_p()
    cuda.cuGetErrorName(err, ctypes.byref(name))
    n = name.value.decode() if name.value else str(err)
    cuda.cuCtxDestroy_v2(ctx)
    print(f"LOAD_FAIL:{n}")
    sys.exit(1)

func = CUfunction()
check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, b'ldg64_min'), "GetFunction")

d_in = CUdeviceptr()
d_out = CUdeviceptr()
check(cuda.cuMemAlloc_v2(ctypes.byref(d_in), 8), "alloc d_in")
check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 8), "alloc d_out")

pattern = 0xDEADBEEFCAFEBABE
check(cuda.cuMemcpyHtoD_v2(d_in, struct.pack('<Q', pattern), 8), "H2D in")
check(cuda.cuMemcpyHtoD_v2(d_out, struct.pack('<Q', 0), 8), "H2D out")

arg_out = ctypes.c_uint64(d_out.value)
arg_in = ctypes.c_uint64(d_in.value)
args = (ctypes.c_void_p * 2)(
    ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
    ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
)

err = cuda.cuLaunchKernel(func, 1,1,1, 1,1,1, 0, None, args, None)
if err != 0:
    name = ctypes.c_char_p()
    cuda.cuGetErrorName(err, ctypes.byref(name))
    n = name.value.decode() if name.value else str(err)
    cuda.cuCtxDestroy_v2(ctx)
    print(f"LAUNCH_FAIL:{n}")
    sys.exit(1)

err = cuda.cuCtxSynchronize()
if err != 0:
    name = ctypes.c_char_p()
    cuda.cuGetErrorName(err, ctypes.byref(name))
    n = name.value.decode() if name.value else str(err)
    cuda.cuMemFree_v2(d_in)
    cuda.cuMemFree_v2(d_out)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy_v2(ctx)
    print(f"CRASH:{n}")
    sys.exit(1)

h_out = ctypes.create_string_buffer(8)
check(cuda.cuMemcpyDtoH_v2(h_out, d_out, 8), "D2H out")
val = struct.unpack('<Q', h_out.raw)[0]

cuda.cuMemFree_v2(d_in)
cuda.cuMemFree_v2(d_out)
cuda.cuModuleUnload(mod)
cuda.cuCtxDestroy_v2(ctx)

if val == pattern:
    print("PASS")
else:
    print(f"WRONG:0x{val:016x} (expected 0x{pattern:016x})")
    sys.exit(1)
