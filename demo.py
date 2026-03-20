#!/usr/bin/env python3
"""
OpenPTXas 30-second proof: compile PTX and run on RTX 5090.
No ptxas. No nvcc. Just OpenPTXas.

Usage:
    python demo.py                          # full demo
    python demo.py --compile-only           # just compile, don't run
    python demo.py examples/vector_add.ptx  # custom PTX file
"""
import sys
import ctypes
from pathlib import Path

from sass.pipeline import compile_ptx_source


def compile_kernel(ptx_path: str) -> tuple[str, bytes]:
    """Compile PTX to cubin. Returns (kernel_name, cubin_bytes)."""
    ptx_src = Path(ptx_path).read_text(encoding='utf-8')
    results = compile_ptx_source(ptx_src)
    name, cubin = next(iter(results.items()))
    return name, cubin


def run_vector_add(cubin_path: str, kernel_name: str, N: int = 1024):
    """Launch vector_add on GPU: out[i] = a[i] + b[i], verify all elements."""
    cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
    CUdeviceptr = ctypes.c_uint64

    def check(err, msg=""):
        if err != 0:
            name_buf = ctypes.c_char_p()
            cuda.cuGetErrorName(err, ctypes.byref(name_buf))
            print(f"CUDA ERROR: {name_buf.value.decode()} [{msg}]")
            sys.exit(1)

    check(cuda.cuInit(0))
    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0))
    devname = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(devname, 256, dev)
    print(f"Device: {devname.value.decode()}")

    ctx = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))
    mod = ctypes.c_void_p()
    check(cuda.cuModuleLoad(ctypes.byref(mod), cubin_path.encode()))
    func = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, kernel_name.encode()))

    # Prepare data: a[i] = i+1, b[i] = i*10
    block_size = 32
    h_a = (ctypes.c_uint32 * N)(*[i + 1 for i in range(N)])
    h_b = (ctypes.c_uint32 * N)(*[i * 10 for i in range(N)])
    h_out = (ctypes.c_uint32 * N)(*[0] * N)
    byte_size = N * 4

    d_a, d_b, d_out = CUdeviceptr(), CUdeviceptr(), CUdeviceptr()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_a), byte_size))
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_b), byte_size))
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), byte_size))
    check(cuda.cuMemcpyHtoD_v2(d_a, h_a, byte_size))
    check(cuda.cuMemcpyHtoD_v2(d_b, h_b, byte_size))

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
    print(f"Launch: {grid_x} blocks x {block_size} threads = {N} elements")

    check(cuda.cuLaunchKernel(func,
        grid_x, 1, 1, block_size, 1, 1, 0, None, args, None))
    check(cuda.cuCtxSynchronize())
    check(cuda.cuMemcpyDtoH_v2(h_out, d_out, byte_size))

    errors = sum(1 for i in range(N) if h_out[i] != h_a[i] + h_b[i])
    if errors == 0:
        print(f"[PASS] {N} elements verified correct")
    else:
        print(f"[FAIL] {errors}/{N} mismatches")

    cuda.cuMemFree_v2(d_a)
    cuda.cuMemFree_v2(d_b)
    cuda.cuMemFree_v2(d_out)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy_v2(ctx)
    return errors == 0


if __name__ == '__main__':
    ptx_file = 'examples/vector_add.ptx'
    compile_only = False

    for arg in sys.argv[1:]:
        if arg == '--compile-only':
            compile_only = True
        elif arg.endswith('.ptx'):
            ptx_file = arg

    print(f"Compiling: {ptx_file}")
    kernel_name, cubin = compile_kernel(ptx_file)
    cubin_path = ptx_file.replace('.ptx', '.cubin')
    Path(cubin_path).write_bytes(cubin)
    print(f"Output:    {cubin_path} ({len(cubin)} bytes)")
    print(f"Kernel:    {kernel_name}")
    print()

    if compile_only:
        print("Done (compile only).")
    else:
        print("--- Running on GPU (no NVIDIA compiler used) ---")
        ok = run_vector_add(cubin_path, kernel_name)
        print()
        if ok:
            print("Our code. Their GPU.")
        sys.exit(0 if ok else 1)
