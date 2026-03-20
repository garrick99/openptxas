"""
GPU test using CUDA driver API via ctypes.
Tests cubin loading and kernel execution on RTX 5090.
"""
import ctypes
import struct
import sys
from pathlib import Path

# CUDA driver API
cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')

# Type aliases
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

def run_copy_test(cubin_path, kernel_name):
    """Test LDG+STG: kernel copies uint64 from in_ptr to out_ptr."""
    check(cuda.cuInit(0), "cuInit")

    dev = CUdevice()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")

    devname = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(devname, 256, dev)
    print(f"Device: {devname.value.decode()}")

    ctx = CUcontext()
    # Use cuCtxCreate_v2 for compatibility
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

    mod = CUmodule()
    path_bytes = cubin_path.encode('utf-8')
    err = cuda.cuModuleLoad(ctypes.byref(mod), path_bytes)
    if err != 0:
        name = ctypes.c_char_p()
        desc = ctypes.c_char_p()
        cuda.cuGetErrorName(err, ctypes.byref(name))
        cuda.cuGetErrorString(err, ctypes.byref(desc))
        print(f"Failed to load cubin: {name.value.decode()} - {desc.value.decode()}")
        return False
    print(f"Loaded: {cubin_path}")

    func = CUfunction()
    check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, kernel_name.encode('utf-8')),
          "cuModuleGetFunction")
    print(f"Kernel: {kernel_name}")

    # Allocate device memory
    d_in = CUdeviceptr()
    d_out = CUdeviceptr()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_in), 8), "cuMemAlloc d_in")
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 8), "cuMemAlloc d_out")

    # Write test pattern to d_in
    h_in = struct.pack('<Q', 0xDEADBEEFCAFEBABE)
    check(cuda.cuMemcpyHtoD_v2(d_in, h_in, 8), "cuMemcpyHtoD d_in")

    # Zero d_out
    h_zero = struct.pack('<Q', 0)
    check(cuda.cuMemcpyHtoD_v2(d_out, h_zero, 8), "cuMemcpyHtoD d_out")

    print(f"d_out=0x{d_out.value:x}, d_in=0x{d_in.value:x}")

    # Launch kernel(out_ptr, in_ptr)
    # Args are pointers to the actual values (which are device pointers)
    arg_out = ctypes.c_uint64(d_out.value)
    arg_in = ctypes.c_uint64(d_in.value)
    args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
    )

    check(cuda.cuLaunchKernel(
        func,
        1, 1, 1,      # grid
        1, 1, 1,      # block
        0,             # shared mem
        None,          # stream
        args,          # kernel args
        None,          # extra
    ), "cuLaunchKernel")

    check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")

    # Read back
    h_out = ctypes.create_string_buffer(8)
    check(cuda.cuMemcpyDtoH_v2(h_out, d_out, 8), "cuMemcpyDtoH")

    out_val = struct.unpack('<Q', h_out.raw)[0]
    in_val = 0xDEADBEEFCAFEBABE

    print(f"Input:  0x{in_val:016x}")
    print(f"Output: 0x{out_val:016x}")

    if out_val == in_val:
        print("\n*** PASS — LDG+STG copy works! ***")
        ok = True
    else:
        print(f"\n*** FAIL — expected 0x{in_val:016x}, got 0x{out_val:016x} ***")
        ok = False

    cuda.cuMemFree_v2(d_in)
    cuda.cuMemFree_v2(d_out)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy_v2(ctx)
    return ok


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <cubin_path> <kernel_name>")
        sys.exit(1)
    ok = run_copy_test(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
