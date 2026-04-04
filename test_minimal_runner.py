"""Test if a minimal EXIT-only kernel crashes."""
import ctypes, sys

cubin_path = sys.argv[1]
cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
assert cuda.cuInit(0) == 0
dev = ctypes.c_int()
cuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
assert cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0

mod = ctypes.c_void_p()
with open(cubin_path, 'rb') as f:
    cubin = f.read()
err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
if err != 0:
    name = ctypes.c_char_p()
    cuda.cuGetErrorName(err, ctypes.byref(name))
    print(f"LOAD_FAIL:{name.value.decode()}({err})")
    sys.exit(1)

fn = ctypes.c_void_p()
err = cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b'test_minimal')
assert err == 0, f"GetFunction err={err}"

arr = (ctypes.c_void_p * 0)()
le = cuda.cuLaunchKernel(fn, 1,1,1, 1,1,1, 0, None, arr, None)
se = cuda.cuCtxSynchronize()

if le == 0 and se == 0:
    print("OK")
else:
    le_name = ctypes.c_char_p()
    cuda.cuGetErrorName(le, ctypes.byref(le_name))
    se_name = ctypes.c_char_p()
    cuda.cuGetErrorName(se, ctypes.byref(se_name))
    print(f"FAIL:le={le}({le_name.value.decode()}),se={se}({se_name.value.decode()})")
