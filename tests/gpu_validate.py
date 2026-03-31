#!/usr/bin/env python3
"""GPU validation runner for OpenPTXas cubins.

Uses CUDA Driver API via ctypes to load cubins and verify kernel correctness.
Supports multiple kernel signatures and verification functions.
"""

import ctypes, struct, sys, os, glob, math

# CUDA Driver API bindings
try:
    if sys.platform == 'win32':
        _cuda = ctypes.WinDLL('nvcuda')
    else:
        _cuda = ctypes.CDLL('libcuda.so')
except OSError:
    print("ERROR: CUDA driver not found"); sys.exit(1)

CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64

def _check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"CUDA error {err}: {msg}")

def cuda_init():
    _check(_cuda.cuInit(0), "cuInit")
    dev = CUdevice()
    _check(_cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = CUcontext()
    _check(_cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
    _check(_cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    return dev, ctx

def cuda_load_module(path):
    mod = CUmodule()
    err = _cuda.cuModuleLoad(ctypes.byref(mod), path.encode())
    if err != 0:
        raise RuntimeError(f"Failed to load cubin '{path}': error {err}")
    return mod

def cuda_get_function(mod, name):
    func = CUfunction()
    err = _cuda.cuModuleGetFunction(ctypes.byref(func), mod, name.encode())
    if err != 0:
        raise RuntimeError(f"Failed to find kernel '{name}': error {err}")
    return func

def cuda_alloc(nbytes):
    ptr = CUdeviceptr()
    _check(_cuda.cuMemAlloc_v2(ctypes.byref(ptr), nbytes), "cuMemAlloc")
    return ptr

def cuda_free(ptr):
    _cuda.cuMemFree_v2(ptr)

def cuda_htod(dst, src_buf, nbytes):
    _check(_cuda.cuMemcpyHtoD_v2(dst, src_buf, nbytes), "cuMemcpyHtoD")

def cuda_dtoh(dst_buf, src, nbytes):
    _check(_cuda.cuMemcpyDtoH_v2(dst_buf, src, nbytes), "cuMemcpyDtoH")

def cuda_launch(func, grid, block, args, smem=0):
    gx, gy, gz = grid
    bx, by, bz = block
    arg_ptrs = (ctypes.c_void_p * len(args))()
    arg_storage = []
    for i, a in enumerate(args):
        if isinstance(a, CUdeviceptr):
            storage = ctypes.c_uint64(a.value)
        elif isinstance(a, float):
            storage = ctypes.c_float(a)
        elif isinstance(a, int):
            storage = ctypes.c_int32(a)
        else:
            storage = a
        arg_storage.append(storage)
        arg_ptrs[i] = ctypes.cast(ctypes.pointer(storage), ctypes.c_void_p)

    err = _cuda.cuLaunchKernel(func, gx, gy, gz, bx, by, bz, smem, None, arg_ptrs, None)
    if err != 0:
        raise RuntimeError(f"cuLaunchKernel failed: error {err}")
    _check(_cuda.cuCtxSynchronize(), "cuCtxSynchronize")


# ---------- Test kernel definitions ----------

def _make_float_buf(n, init_fn):
    buf = (ctypes.c_float * n)()
    for i in range(n):
        buf[i] = init_fn(i)
    return buf

def _read_float_buf(ptr, n):
    buf = (ctypes.c_float * n)()
    cuda_dtoh(buf, ptr, n * 4)
    return list(buf)


def test_vector_add(cubin_path, kernel_name, n=1024):
    """(float *out, float *a, float *b, int n) → out[i] = a[i] + b[i]"""
    mod = cuda_load_module(cubin_path)
    func = cuda_get_function(mod, kernel_name)

    h_a = _make_float_buf(n, lambda i: float(i))
    h_b = _make_float_buf(n, lambda i: float(i * 2))

    d_out = cuda_alloc(n * 4); d_a = cuda_alloc(n * 4); d_b = cuda_alloc(n * 4)
    cuda_htod(d_a, h_a, n * 4); cuda_htod(d_b, h_b, n * 4)

    blocks = (n + 255) // 256
    cuda_launch(func, (blocks, 1, 1), (256, 1, 1), [d_out, d_a, d_b, n])

    out = _read_float_buf(d_out, n)
    errors = sum(1 for i in range(n) if abs(out[i] - (h_a[i] + h_b[i])) > 0.01)

    cuda_free(d_out); cuda_free(d_a); cuda_free(d_b)
    _cuda.cuModuleUnload(mod)
    return errors == 0, f"{errors}/{n} mismatches" if errors else "OK"


def test_generic_3ptr_int(cubin_path, kernel_name, verify_fn, n=1024,
                          init_a=None, init_b=None):
    """(float *out, float *a, float *b, int n) with custom verify."""
    mod = cuda_load_module(cubin_path)
    func = cuda_get_function(mod, kernel_name)

    if init_a is None: init_a = lambda i: float(i + 1)
    if init_b is None: init_b = lambda i: float(i * 2 + 1)

    h_a = _make_float_buf(n, init_a)
    h_b = _make_float_buf(n, init_b)

    d_out = cuda_alloc(n * 4); d_a = cuda_alloc(n * 4); d_b = cuda_alloc(n * 4)
    cuda_htod(d_a, h_a, n * 4); cuda_htod(d_b, h_b, n * 4)
    # Zero out output
    h_zero = _make_float_buf(n, lambda i: 0.0)
    cuda_htod(d_out, h_zero, n * 4)

    blocks = (n + 255) // 256
    cuda_launch(func, (blocks, 1, 1), (256, 1, 1), [d_out, d_a, d_b, n])

    out = _read_float_buf(d_out, n)
    errors = 0
    for i in range(n):
        expected = verify_fn(i, h_a[i], h_b[i])
        if expected is None:
            continue  # skip verification for this element
        if abs(out[i] - expected) > max(0.01, abs(expected) * 1e-5):
            errors += 1
            if errors <= 3:
                print(f"  [{i}] got={out[i]:.6f} expected={expected:.6f}")

    cuda_free(d_out); cuda_free(d_a); cuda_free(d_b)
    _cuda.cuModuleUnload(mod)
    return errors == 0, f"{errors}/{n} mismatches" if errors else "OK"


def test_launch_only(cubin_path, kernel_name, n=256):
    """Just launch the kernel and check for GPU errors (no verification)."""
    mod = cuda_load_module(cubin_path)
    func = cuda_get_function(mod, kernel_name)

    d_out = cuda_alloc(n * 4); d_a = cuda_alloc(n * 4); d_b = cuda_alloc(n * 4)

    # Initialize with non-zero data
    h_a = _make_float_buf(n, lambda i: float(i + 1))
    h_b = _make_float_buf(n, lambda i: float(i * 2 + 1))
    cuda_htod(d_a, h_a, n * 4); cuda_htod(d_b, h_b, n * 4)
    h_zero = _make_float_buf(n, lambda i: 0.0)
    cuda_htod(d_out, h_zero, n * 4)

    blocks = (n + 255) // 256
    try:
        cuda_launch(func, (blocks, 1, 1), (256, 1, 1), [d_out, d_a, d_b, n])
        cuda_free(d_out); cuda_free(d_a); cuda_free(d_b)
        _cuda.cuModuleUnload(mod)
        return True, "LAUNCH_OK"
    except RuntimeError as e:
        cuda_free(d_out); cuda_free(d_a); cuda_free(d_b)
        _cuda.cuModuleUnload(mod)
        return False, str(e)


# ---------- Main ----------

if __name__ == '__main__':
    dev, ctx = cuda_init()

    cubin_dir = os.path.dirname(os.path.abspath(__file__)) + '/..'
    cubins = sorted(glob.glob(os.path.join(cubin_dir, '_auto_*.cubin')))

    if not cubins:
        print("No _auto_*.cubin files found")
        sys.exit(1)

    # Known kernel verifications
    VERIFIED = {
        'vector_add': lambda i, a, b: a + b,
        'saxpy': None,  # different signature, skip
        'compound_ops': lambda i, a, b: a + b,  # approximate
    }

    launched = 0; passed = 0; failed = 0; errors = 0

    for cubin_path in cubins:
        kernel_name = os.path.basename(cubin_path).replace('_auto_', '').replace('.cubin', '')

        try:
            ok, msg = test_launch_only(cubin_path, kernel_name, n=256)
            launched += 1
            if ok:
                passed += 1
                status = "LAUNCH_OK"
            else:
                failed += 1
                status = f"LAUNCH_FAIL: {msg}"
        except Exception as e:
            errors += 1
            status = f"ERROR: {str(e)[:60]}"

        sym = "+" if "OK" in str(status) else "-"
        print(f"  {sym} {kernel_name:30s}: {status}")

    print(f"\n{launched} launched, {passed} passed, {failed} failed, {errors} errors")
