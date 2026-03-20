"""
GPU test for vector_add kernel: out[i] = a[i] + b[i]
Tests multi-thread execution with S2R, IMAD, branching, and 32-bit LDG/STG.
"""
import ctypes
import struct
import sys
from pathlib import Path

cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')

CUresult = ctypes.c_int
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
        print(f"CUDA ERROR {err}: {name.value.decode()} - {desc.value.decode()} [{msg}]")
        sys.exit(1)

def run_vecadd(cubin_path, kernel_name, N, block_size=32):
    check(cuda.cuInit(0), "cuInit")
    dev = CUdevice()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0))
    devname = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(devname, 256, dev)
    print(f"Device: {devname.value.decode()}")

    ctx = CUcontext()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))

    mod = CUmodule()
    err = cuda.cuModuleLoad(ctypes.byref(mod), cubin_path.encode())
    if err != 0:
        name = ctypes.c_char_p()
        cuda.cuGetErrorName(err, ctypes.byref(name))
        print(f"Failed to load cubin: {name.value.decode()}")
        return False
    print(f"Loaded: {cubin_path}")

    func = CUfunction()
    check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, kernel_name.encode()))
    print(f"Kernel: {kernel_name}")

    # Prepare data
    h_a = (ctypes.c_uint32 * N)(*[i + 1 for i in range(N)])
    h_b = (ctypes.c_uint32 * N)(*[i * 10 for i in range(N)])
    h_out = (ctypes.c_uint32 * N)(*[0] * N)
    byte_size = N * 4

    d_a = CUdeviceptr()
    d_b = CUdeviceptr()
    d_out = CUdeviceptr()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_a), byte_size))
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_b), byte_size))
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), byte_size))

    check(cuda.cuMemcpyHtoD_v2(d_a, h_a, byte_size))
    check(cuda.cuMemcpyHtoD_v2(d_b, h_b, byte_size))
    check(cuda.cuMemcpyHtoD_v2(d_out, h_out, byte_size))

    # Launch: vector_add(out, a, b, n)
    arg_out = ctypes.c_uint64(d_out.value)
    arg_a = ctypes.c_uint64(d_a.value)
    arg_b = ctypes.c_uint64(d_b.value)
    arg_n = ctypes.c_uint32(N)

    args = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(arg_a), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(arg_b), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
    )

    grid_x = (N + block_size - 1) // block_size
    print(f"Launch: grid={grid_x}, block={block_size}, N={N}")

    check(cuda.cuLaunchKernel(
        func,
        grid_x, 1, 1,
        block_size, 1, 1,
        0, None,
        args, None,
    ), "cuLaunchKernel")

    check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")

    # Read back
    check(cuda.cuMemcpyDtoH_v2(h_out, d_out, byte_size))

    # Verify
    errors = 0
    for i in range(N):
        expected = h_a[i] + h_b[i]
        if h_out[i] != expected:
            if errors < 10:
                print(f"  MISMATCH [{i}]: got {h_out[i]}, expected {expected} "
                      f"(a={h_a[i]}, b={h_b[i]})")
            errors += 1

    if errors == 0:
        print(f"\n*** PASS: {N} elements verified correct ***")
    else:
        print(f"\n*** FAIL: {errors}/{N} mismatches ***")

    cuda.cuMemFree_v2(d_a)
    cuda.cuMemFree_v2(d_b)
    cuda.cuMemFree_v2(d_out)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy_v2(ctx)
    return errors == 0


if __name__ == '__main__':
    cubin = sys.argv[1] if len(sys.argv) > 1 else 'probe_work/vector_add.cubin'
    kernel = sys.argv[2] if len(sys.argv) > 2 else 'vector_add'
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    ok = run_vecadd(cubin, kernel, N)
    sys.exit(0 if ok else 1)
