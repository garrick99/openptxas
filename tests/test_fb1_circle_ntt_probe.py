"""FB-1 diagnostic probe: launch circle_ntt_layer through the open-toolchain
path and capture the exact failure mode.  Same shape as test_fb1_e2e.py,
different kernel."""
from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, r'C:\Users\kraken\openptxas')
sys.path.insert(0, r'C:\Users\kraken\opencuda')

from opencuda.tests.test_gpu_e2e import CUDAContext, _CUDA

gpu = pytest.mark.skipif(_CUDA is None, reason="No CUDA GPU available")

FORGE_WRAPPER = Path(r"C:\Users\kraken\VortexSTARK\cuda\forge\circle_ntt_layer_forge.cu")
CUDA_INCLUDE = Path(r"C:\Users\kraken\VortexSTARK\cuda\include")


def _build_cubins() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        ptx_path = Path(tmp) / "k.ptx"
        r = subprocess.run(
            [sys.executable, "-m", "opencuda", str(FORGE_WRAPPER),
             "--emit-ptx", "--out", str(ptx_path), "-I", str(CUDA_INCLUDE)],
            cwd=r"C:\Users\kraken\opencuda",
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, f"OpenCUDA failed: {r.stderr}"
        from sass.pipeline import compile_ptx_source
        return compile_ptx_source(ptx_path.read_text())


class _PrimaryCtxBag:
    """Mimics the Rust path: cuDevicePrimaryCtxRetain + cuCtxSetCurrent."""
    def __init__(self):
        self.cuda = _CUDA
        dev = ctypes.c_int()
        assert self.cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
        self._dev = dev
        ctx = ctypes.c_void_p()
        assert self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) == 0
        assert self.cuda.cuCtxSetCurrent(ctx) == 0
        self.ctx = ctx
        self.mod = ctypes.c_void_p()

    def load(self, cubin):
        err = self.cuda.cuModuleLoadData(ctypes.byref(self.mod), cubin)
        return err == 0

    def get_func(self, name):
        func = ctypes.c_void_p()
        err = self.cuda.cuModuleGetFunction(ctypes.byref(func), self.mod, name.encode())
        assert err == 0, f"cuModuleGetFunction({name}) failed: {err}"
        return func

    def alloc(self, nbytes):
        ptr = ctypes.c_uint64()
        assert self.cuda.cuMemAlloc_v2(ctypes.byref(ptr), nbytes) == 0
        return ptr.value

    def copy_to(self, dev_ptr, data):
        assert self.cuda.cuMemcpyHtoD_v2(ctypes.c_uint64(dev_ptr), data, len(data)) == 0

    def copy_from(self, dev_ptr, nbytes):
        buf = (ctypes.c_uint8 * nbytes)()
        assert self.cuda.cuMemcpyDtoH_v2(buf, ctypes.c_uint64(dev_ptr), nbytes) == 0
        return bytes(buf)

    def free(self, dev_ptr):
        self.cuda.cuMemFree_v2(ctypes.c_uint64(dev_ptr))

    def sync(self):
        return self.cuda.cuCtxSynchronize()


@gpu
def test_circle_ntt_layer_primary_ctx_probe():
    """Same as the Rust path: primary context + cuMemAlloc + cuLaunchKernel."""
    cuda_ctx = _PrimaryCtxBag()
    cubins = _build_cubins()
    print(f"\n[probe] cubin keys: {list(cubins.keys())}")
    cubin = cubins["circle_ntt_layer_forward"]
    print(f"[probe] forward cubin size: {len(cubin)}")

    assert cuda_ctx.load(cubin), "cuModuleLoadData rejected our cubin"
    func = cuda_ctx.get_func("circle_ntt_layer_forward")

    # Match the test in src/blake2s_m31.rs:
    #   log_n = 10, n = 1024, half_n = 512, layer_idx = 3
    n = 1024
    half_n = n // 2
    layer_idx = 3

    # Allocate buffers
    data = [(i * 0x9E3779B9) ^ 0x12345678 for i in range(n)]
    data = [v % 0x7FFFFFFF for v in data]
    twiddles = [((i * 0x6A09E667) ^ 0xCAFE) % 0x7FFFFFFF for i in range(half_n)]

    data_bytes = struct.pack(f'<{n}I', *data)
    tw_bytes = struct.pack(f'<{half_n}I', *twiddles)

    d_data = cuda_ctx.alloc(len(data_bytes))
    d_tw = cuda_ctx.alloc(len(tw_bytes))
    cuda_ctx.copy_to(d_data, data_bytes)
    cuda_ctx.copy_to(d_tw, tw_bytes)

    # Pack args matching OpenCUDA's flattened param signature:
    #   .param .u64 data_data
    #   .param .u64 data_len      (u32 element count)
    #   .param .u64 twiddles_data
    #   .param .u64 twiddles_len  (u32 element count)
    #   .param .u32 layer_idx
    #   .param .u64 half_n
    holders = [
        ctypes.c_uint64(d_data),
        ctypes.c_uint64(n),
        ctypes.c_uint64(d_tw),
        ctypes.c_uint64(half_n),
        ctypes.c_uint32(layer_idx),
        ctypes.c_uint64(half_n),
    ]
    ptrs = [ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders]
    args_arr = (ctypes.c_void_p * len(ptrs))(*ptrs)

    threads = 256
    blocks = (half_n + threads - 1) // threads

    rc = cuda_ctx.cuda.cuLaunchKernel(
        func, blocks, 1, 1, threads, 1, 1, 0, None, args_arr, None)
    print(f"[probe] cuLaunchKernel rc = {rc}")
    sync_rc = cuda_ctx.sync()
    print(f"[probe] cuCtxSynchronize rc = {sync_rc}")

    # Read back data and check it changed (sanity, not correctness)
    out_bytes = cuda_ctx.copy_from(d_data, len(data_bytes))
    out_data = list(struct.unpack(f'<{n}I', out_bytes))
    diffs = sum(1 for a, b in zip(data, out_data) if a != b)
    print(f"[probe] elements changed by kernel: {diffs} / {n}")

    cuda_ctx.free(d_data)
    cuda_ctx.free(d_tw)

    assert sync_rc == 0, (
        f"circle_ntt_layer_forward open-toolchain cubin failed at sync: "
        f"rc={sync_rc} (cudaErrorIllegalAddress=700 if value is 700)")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
