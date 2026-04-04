"""Subprocess-based isolation test for fp64_bench crash."""
import subprocess, sys, os, tempfile

CUBIN_PATH = r'C:\Users\kraken\openptxas\fp64_bench.cubin'

RUNNER = r'''
import ctypes, sys, os

cubin_path = sys.argv[1]
out_len    = int(sys.argv[2])
n_iters    = int(sys.argv[3])

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
err = cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b'fp64_bench')
assert err == 0, f"GetFunction err={err}"

# alloc output buffer (at least 1 byte)
out_ptr = ctypes.c_uint64()
alloc_size = max(out_len * 8, 1)
err = cuda.cuMemAlloc_v2(ctypes.byref(out_ptr), alloc_size)
assert err == 0, f"alloc err={err}"

# build args: out_data, out_len, n_iters, b (float), c (float)
B = 1.0 + 1e-9
C = 1e-19
holders = [
    ctypes.c_uint64(out_ptr.value),
    ctypes.c_uint64(out_len),
    ctypes.c_uint64(n_iters),
    ctypes.c_double(B),
    ctypes.c_double(C),
]
ptrs = [(ctypes.cast(ctypes.byref(h), ctypes.c_void_p)) for h in holders]
arr  = (ctypes.c_void_p * len(ptrs))(*ptrs)

le = cuda.cuLaunchKernel(fn, 1,1,1, 1,1,1, 0, None, arr, None)
se = cuda.cuCtxSynchronize()

if le == 0 and se == 0:
    print(f"OK")
else:
    le_name = ctypes.c_char_p()
    cuda.cuGetErrorName(le, ctypes.byref(le_name))
    se_name = ctypes.c_char_p()
    cuda.cuGetErrorName(se, ctypes.byref(se_name))
    print(f"FAIL:le={le}({le_name.value.decode()}),se={se}({se_name.value.decode()})")
'''

# Write runner to temp file
with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
    f.write(RUNNER)
    runner_path = f.name

print(f"Runner: {runner_path}")
print()

cases = [
    (0, 0, "no-loop no-store"),
    (1, 0, "no-loop WITH-store"),
    (0, 1, "WITH-loop(1) no-store"),
    (1, 1, "WITH-loop(1) WITH-store"),
    (0, 4, "WITH-loop(4) no-store"),
]

for out_len, n_iters, label in cases:
    result = subprocess.run(
        [sys.executable, runner_path, CUBIN_PATH, str(out_len), str(n_iters)],
        capture_output=True, text=True, timeout=30
    )
    out = (result.stdout + result.stderr).strip()
    print(f"[out_len={out_len}, n_iters={n_iters}] {label:30s}  => {out}")

os.unlink(runner_path)
