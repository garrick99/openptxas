"""GPU test: verify ptxas-generated TMA reference cubins load and the
legacy cp.async path (LDGSTS/LDGDEPBAR/DEPBAR.LE) works correctly.

The TMA tensor path requires cuTensorMapEncode which is complex to set up
from ctypes alone, so we test the commit/wait kernel that uses legacy cp.async.
"""
import ctypes
import struct
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip if no GPU
try:
    if sys.platform == 'win32':
        _cuda = ctypes.WinDLL('nvcuda')
    else:
        _cuda = ctypes.CDLL('libcuda.so')
    CUresult = ctypes.c_int
    CUdevice = ctypes.c_int
    CUcontext = ctypes.c_void_p
    CUmodule = ctypes.c_void_p
    CUfunction = ctypes.c_void_p
    CUdeviceptr = ctypes.c_uint64
    HAS_GPU = True
except OSError:
    HAS_GPU = False

import pytest


def _check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"CUDA error {err}: {msg}")


@pytest.fixture(scope="module")
def tma_cuda_ctx():
    if not HAS_GPU:
        pytest.skip("No CUDA driver available")
    _check(_cuda.cuInit(0), "cuInit")
    dev = CUdevice()
    _check(_cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    # Check compute capability
    major = ctypes.c_int()
    minor = ctypes.c_int()
    _cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    _cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    if major.value * 10 + minor.value < 120:
        pytest.skip(f"Need SM_120+, got {major.value}.{minor.value}")
    ctx = CUcontext()
    _check(_cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
    _check(_cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    return ctx


def _load_cubin(path):
    mod = CUmodule()
    err = _cuda.cuModuleLoad(ctypes.byref(mod), path.encode())
    if err != 0:
        raise RuntimeError(f"Failed to load cubin '{path}': error {err}")
    return mod


def test_tma_commit_wait_loads(tma_cuda_ctx):
    """Verify the tma_commit_wait cubin loads on SM_120 GPU."""
    cubin = str(Path(__file__).parent.parent / "probe_work" / "tma_probes" / "tma_commit_wait.cubin")
    if not os.path.exists(cubin):
        pytest.skip("tma_commit_wait.cubin not found")
    mod = _load_cubin(cubin)
    func = CUfunction()
    err = _cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"tma_commit_wait")
    assert err == 0, f"Failed to find kernel: error {err}"


def test_tma_commit_wait_copies_data(tma_cuda_ctx):
    """Run tma_commit_wait kernel and verify data is copied correctly.

    This kernel uses legacy cp.async: LDGSTS.E.128 + LDGDEPBAR + DEPBAR.LE
    to copy 16 bytes from global to shared, then reads back via LDS.64 + STG.E.64.
    """
    cubin = str(Path(__file__).parent.parent / "probe_work" / "tma_probes" / "tma_commit_wait.cubin")
    if not os.path.exists(cubin):
        pytest.skip("tma_commit_wait.cubin not found")
    mod = _load_cubin(cubin)
    func = CUfunction()
    _check(_cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"tma_commit_wait"))

    # Allocate source and destination buffers
    nbytes = 128  # Kernel uses 128 bytes of shared memory
    d_src = CUdeviceptr()
    d_dst = CUdeviceptr()
    _check(_cuda.cuMemAlloc_v2(ctypes.byref(d_src), nbytes))
    _check(_cuda.cuMemAlloc_v2(ctypes.byref(d_dst), nbytes))

    # Fill source with known pattern
    h_src = (ctypes.c_uint8 * nbytes)()
    for i in range(nbytes):
        h_src[i] = (i * 7 + 13) & 0xFF
    _check(_cuda.cuMemcpyHtoD_v2(d_src, h_src, nbytes))

    # Zero destination
    h_zero = (ctypes.c_uint8 * nbytes)()
    _check(_cuda.cuMemcpyHtoD_v2(d_dst, h_zero, nbytes))

    # Launch: tma_commit_wait(u64 src_ptr, u64 dst_ptr)
    src_arg = ctypes.c_uint64(d_src.value)
    dst_arg = ctypes.c_uint64(d_dst.value)
    args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.pointer(src_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(dst_arg), ctypes.c_void_p),
    )
    err = _cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
    if err != 0:
        _cuda.cuMemFree_v2(d_src)
        _cuda.cuMemFree_v2(d_dst)
        pytest.fail(f"cuLaunchKernel failed: error {err}")
    _check(_cuda.cuCtxSynchronize())

    # Read back first 8 bytes (the kernel does: LDS.64 R4, [smem]; STG.64 [dst], R4)
    h_dst = (ctypes.c_uint8 * 8)()
    _check(_cuda.cuMemcpyDtoH_v2(h_dst, d_dst, 8))

    _cuda.cuMemFree_v2(d_src)
    _cuda.cuMemFree_v2(d_dst)

    # Verify first 8 bytes match source
    for i in range(8):
        assert h_dst[i] == h_src[i], f"byte[{i}]: got {h_dst[i]:#x}, expected {h_src[i]:#x}"


def test_tma_bulk_copy_loads(tma_cuda_ctx):
    """Verify the tma_bulk_copy cubin loads on SM_120 GPU."""
    cubin = str(Path(__file__).parent.parent / "probe_work" / "tma_probes" / "tma_bulk_copy.cubin")
    if not os.path.exists(cubin):
        pytest.skip("tma_bulk_copy.cubin not found")
    mod = _load_cubin(cubin)
    func = CUfunction()
    err = _cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"tma_bulk_copy")
    assert err == 0, f"Failed to find kernel: error {err}"


def test_tma_tensor_1d_loads(tma_cuda_ctx):
    """Verify the tma_tensor_1d cubin loads on SM_120 GPU."""
    cubin = str(Path(__file__).parent.parent / "probe_work" / "tma_probes" / "tma_tensor_1d.cubin")
    if not os.path.exists(cubin):
        pytest.skip("tma_tensor_1d.cubin not found")
    mod = _load_cubin(cubin)
    func = CUfunction()
    err = _cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"tma_tensor_1d")
    assert err == 0, f"Failed to find kernel: error {err}"
