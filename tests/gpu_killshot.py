#!/usr/bin/env python3
"""
Kill shot: ptxas vs OpenPTXas on the rotate-sub bug.

Same kernel. Same GPU. Same input. Different output.
ptxas gets it wrong. OpenPTXas gets it right.

The kernel computes: (x << 8) - (x >> 56)
ptxas incorrectly compiles this as: rotate_left(x, 8)
"""
import ctypes
import struct
import sys

cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
CUdeviceptr = ctypes.c_uint64

def check(err, msg=""):
    if err != 0:
        name = ctypes.c_char_p()
        cuda.cuGetErrorName(err, ctypes.byref(name))
        print(f"CUDA ERROR: {name.value.decode()} [{msg}]")
        sys.exit(1)

def run_kernel(cubin_path, kernel_name, input_val):
    """Run sub_bug kernel with a single u64 input, return u64 output."""
    check(cuda.cuInit(0))
    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0))
    ctx = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))
    mod = ctypes.c_void_p()
    err = cuda.cuModuleLoad(ctypes.byref(mod), cubin_path.encode())
    if err != 0:
        cuda.cuCtxDestroy_v2(ctx)
        return None
    func = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func), mod, kernel_name.encode())

    d_in = CUdeviceptr()
    d_out = CUdeviceptr()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_in), 8))
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 8))
    check(cuda.cuMemcpyHtoD_v2(d_in, struct.pack('<Q', input_val), 8))
    check(cuda.cuMemcpyHtoD_v2(d_out, struct.pack('<Q', 0), 8))

    arg_out = ctypes.c_uint64(d_out.value)
    arg_in = ctypes.c_uint64(d_in.value)
    args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
    )
    check(cuda.cuLaunchKernel(func, 1,1,1, 1,1,1, 0, None, args, None))
    check(cuda.cuCtxSynchronize())

    buf = ctypes.create_string_buffer(8)
    check(cuda.cuMemcpyDtoH_v2(buf, d_out, 8))
    result = struct.unpack('<Q', buf.raw)[0]

    cuda.cuMemFree_v2(d_in)
    cuda.cuMemFree_v2(d_out)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy_v2(ctx)
    return result


def cpu_correct(x):
    """CPU reference: (x << 8) - (x >> 56)"""
    shl = (x << 8) & 0xFFFFFFFFFFFFFFFF
    shr = x >> 56
    return (shl - shr) & 0xFFFFFFFFFFFFFFFF


def cpu_rotate(x):
    """What ptxas produces: rotate_left(x, 8)"""
    return ((x << 8) | (x >> 56)) & 0xFFFFFFFFFFFFFFFF


if __name__ == '__main__':
    devname = ctypes.create_string_buffer(256)
    check(cuda.cuInit(0))
    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0))
    cuda.cuDeviceGetName(devname, 256, dev)

    print("=" * 60)
    print("  ptxas vs OpenPTXas — The Rotate-Sub Bug")
    print(f"  GPU: {devname.value.decode()}")
    print("=" * 60)
    print()
    print("Kernel: (x << 8) - (x >> 56)")
    print("ptxas miscompiles this as: rotate_left(x, 8)")
    print()

    test_val = 0x0123456789ABCDEF
    correct = cpu_correct(test_val)
    wrong_rotate = cpu_rotate(test_val)

    ptxas_result = run_kernel('probe_work/sub_bug_ptxas.cubin', 'sub_bug', test_val)
    openptxas_result = run_kernel('probe_work/sub_bug_openptxas.cubin', 'sub_bug', test_val)

    print(f"Input:         0x{test_val:016X}")
    print(f"Correct:       0x{correct:016X}  (subtract)")
    print(f"Wrong rotate:  0x{wrong_rotate:016X}  (rotate, NOT subtract)")
    print()
    print(f"ptxas output:      0x{ptxas_result:016X}  {'CORRECT' if ptxas_result == correct else 'WRONG'}")
    print(f"OpenPTXas output:  0x{openptxas_result:016X}  {'CORRECT' if openptxas_result == correct else 'WRONG'}")
    print()

    if ptxas_result != correct and openptxas_result == correct:
        print("ptxas produces WRONG answer.")
        print("OpenPTXas produces CORRECT answer.")
        print()
        print("Same kernel. Same GPU. Same input. Different output.")
    elif ptxas_result == correct and openptxas_result == correct:
        print("Both correct (ptxas may have been patched for this case).")
    else:
        print("Unexpected results.")

    print()
    print("=" * 60)
