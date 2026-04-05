"""
tests/test_gpu_cvt_encoders.py -- GPU tests for newly wired cvt encoders.

Tests:
  1. cvt.rn.f32.s32 (signed int → float, uses I2FP.F32.S32)
  2. cvt.rzi.s32.f32 (float → signed int, uses F2I.S32)
  3. cvt.rn.f16.f32 (float → half, uses F2FP.F16.F32)

Run: python -m pytest tests/test_gpu_cvt_encoders.py -v -m gpu --tb=short
"""
import ctypes
import struct
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass.pipeline import compile_ptx_source

# ---------------------------------------------------------------------------
# GPU driver bootstrap
# ---------------------------------------------------------------------------

def _get_cuda():
    try:
        cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
        err = cuda.cuInit(0)
        if err != 0:
            return None
        return cuda
    except Exception:
        return None

_CUDA = _get_cuda()
gpu = pytest.mark.skipif(_CUDA is None, reason="No CUDA GPU available")


class CUDAContext:
    def __init__(self):
        self.cuda = _CUDA
        self.ctx = ctypes.c_void_p()
        self.mod = ctypes.c_void_p()
        dev = ctypes.c_int()
        err = self.cuda.cuDeviceGet(ctypes.byref(dev), 0)
        assert err == 0, f"cuDeviceGet failed: {err}"
        err = self.cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev)
        assert err == 0, f"cuCtxCreate_v2 failed: {err}"

    def load(self, cubin_bytes: bytes) -> bool:
        if self.mod and self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()
        err = self.cuda.cuModuleLoadData(ctypes.byref(self.mod), cubin_bytes)
        return err == 0

    def get_func(self, name: str):
        func = ctypes.c_void_p()
        err = self.cuda.cuModuleGetFunction(ctypes.byref(func), self.mod, name.encode())
        assert err == 0, f"cuModuleGetFunction({name}) failed: {err}"
        return func

    def alloc(self, nbytes: int) -> int:
        ptr = ctypes.c_uint64()
        err = self.cuda.cuMemAlloc_v2(ctypes.byref(ptr), nbytes)
        assert err == 0, f"cuMemAlloc_v2({nbytes}) failed: {err}"
        return ptr.value

    def copy_to(self, dev_ptr: int, host_data: bytes):
        err = self.cuda.cuMemcpyHtoD_v2(ctypes.c_uint64(dev_ptr), host_data, len(host_data))
        assert err == 0, f"cuMemcpyHtoD_v2 failed: {err}"

    def copy_from(self, dev_ptr: int, nbytes: int) -> bytes:
        buf = (ctypes.c_uint8 * nbytes)()
        err = self.cuda.cuMemcpyDtoH_v2(buf, ctypes.c_uint64(dev_ptr), nbytes)
        assert err == 0, f"cuMemcpyDtoH_v2 failed: {err}"
        return bytes(buf)

    def launch(self, func, grid, block, args_list, smem=0) -> int:
        arg_holders = []
        ptrs = []
        for a in args_list:
            if isinstance(a, float):
                holder = ctypes.c_float(a)
            elif isinstance(a, int) and a > 0xFFFFFFFF:
                holder = ctypes.c_uint64(a)
            else:
                holder = ctypes.c_int32(a)
            arg_holders.append(holder)
            ptrs.append(ctypes.cast(ctypes.byref(holder), ctypes.c_void_p))
        args_arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        gx, gy, gz = grid
        bx, by, bz = block
        return self.cuda.cuLaunchKernel(func, gx, gy, gz, bx, by, bz, smem, None, args_arr, None)

    def free(self, ptr: int):
        self.cuda.cuMemFree_v2(ctypes.c_uint64(ptr))

    def sync(self) -> int:
        return self.cuda.cuCtxSynchronize()

    def close(self):
        if self.mod and self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()
        if self.ctx and self.ctx.value:
            self.cuda.cuCtxSynchronize()
            self.cuda.cuCtxDestroy_v2(self.ctx)
            self.ctx = ctypes.c_void_p()


@pytest.fixture(scope="class")
def cuda_ctx():
    if _CUDA is None:
        pytest.skip("No CUDA GPU available")
    cctx = CUDAContext()
    yield cctx
    cctx.close()


def _compile(ptx_src: str) -> dict[str, bytes]:
    return compile_ptx_source(ptx_src)


# ===========================================================================
# Test 1: cvt.rn.f32.s32 — signed int to float (I2FP.F32.S32)
# ===========================================================================

_PTX_CVT_F32_S32 = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry cvt_f32_s32(
    .param .u64 p_out,
    .param .u32 p_val
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd0, [p_out];
    ld.param.u32 %r0, [p_val];

    // Convert signed int to float
    cvt.rn.f32.s32 %f0, %r0;

    // Store result (single thread, no offset needed)
    st.global.f32 [%rd0], %f0;
    ret;
}
"""

@gpu
class TestCvtF32S32:
    def test_positive(self, cuda_ctx):
        cubin = _compile(_PTX_CVT_F32_S32)
        assert cubin, "Compilation failed"
        name = [k for k in cubin if not k.startswith('_')][0]
        ok = cuda_ctx.load(cubin[name])
        assert ok, "Module load failed"
        func = cuda_ctx.get_func('cvt_f32_s32')
        d_out = cuda_ctx.alloc(4)
        err = cuda_ctx.launch(func, (1,1,1), (1,1,1), [d_out, 42])
        assert err == 0
        assert cuda_ctx.sync() == 0
        result = struct.unpack('f', cuda_ctx.copy_from(d_out, 4))[0]
        assert result == 42.0, f"Expected 42.0, got {result}"
        cuda_ctx.free(d_out)

    def test_large_positive(self, cuda_ctx):
        """cvt.rn.f32.s32 with large positive value: 1000000 should become 1000000.0"""
        cubin = _compile(_PTX_CVT_F32_S32)
        assert cubin
        name = [k for k in cubin if not k.startswith('_')][0]
        ok = cuda_ctx.load(cubin[name])
        assert ok
        func = cuda_ctx.get_func('cvt_f32_s32')
        d_out = cuda_ctx.alloc(4)
        err = cuda_ctx.launch(func, (1,1,1), (1,1,1), [d_out, 1000000])
        assert err == 0
        assert cuda_ctx.sync() == 0
        result = struct.unpack('f', cuda_ctx.copy_from(d_out, 4))[0]
        assert result == 1000000.0, f"Expected 1000000.0, got {result}"
        cuda_ctx.free(d_out)


# ===========================================================================
# Test 2: cvt.rzi.s32.f32 — float to signed int (F2I.S32)
# ===========================================================================

_PTX_CVT_S32_F32 = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry cvt_s32_f32(
    .param .u64 p_out,
    .param .u64 p_in
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_in];

    // Load float input (single thread)
    ld.global.f32 %f0, [%rd1];

    // Convert float to signed int (truncate)
    cvt.rzi.s32.f32 %r1, %f0;

    // Store result
    st.global.u32 [%rd0], %r1;
    ret;
}
"""

@gpu
class TestCvtS32F32:
    def test_positive(self, cuda_ctx):
        cubin = _compile(_PTX_CVT_S32_F32)
        assert cubin
        name = [k for k in cubin if not k.startswith('_')][0]
        ok = cuda_ctx.load(cubin[name])
        assert ok
        func = cuda_ctx.get_func('cvt_s32_f32')
        d_out = cuda_ctx.alloc(4)
        d_in = cuda_ctx.alloc(4)
        cuda_ctx.copy_to(d_in, struct.pack('f', 3.7))
        err = cuda_ctx.launch(func, (1,1,1), (1,1,1), [d_out, d_in])
        assert err == 0
        assert cuda_ctx.sync() == 0
        result = struct.unpack('i', cuda_ctx.copy_from(d_out, 4))[0]
        assert result == 3, f"Expected 3 (truncated from 3.7), got {result}"
        cuda_ctx.free(d_out)
        cuda_ctx.free(d_in)

    def test_negative(self, cuda_ctx):
        """cvt.rzi.s32.f32 with -5.9 should truncate to -5"""
        cubin = _compile(_PTX_CVT_S32_F32)
        assert cubin
        name = [k for k in cubin if not k.startswith('_')][0]
        ok = cuda_ctx.load(cubin[name])
        assert ok
        func = cuda_ctx.get_func('cvt_s32_f32')
        d_out = cuda_ctx.alloc(4)
        d_in = cuda_ctx.alloc(4)
        cuda_ctx.copy_to(d_in, struct.pack('f', -5.9))
        err = cuda_ctx.launch(func, (1,1,1), (1,1,1), [d_out, d_in])
        assert err == 0
        assert cuda_ctx.sync() == 0
        result = struct.unpack('i', cuda_ctx.copy_from(d_out, 4))[0]
        assert result == -5, f"Expected -5 (truncated from -5.9), got {result}"
        cuda_ctx.free(d_out)
        cuda_ctx.free(d_in)


# ===========================================================================
# Test 3: cvt.rn.f16.f32 — float to half (F2FP.F16.F32)
# ===========================================================================

_PTX_CVT_F16_F32 = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry cvt_f16_f32(
    .param .u64 p_out,
    .param .u64 p_in
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<4>;
    .reg .b16 %h<4>;

    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_in];

    // Load float input (single thread)
    ld.global.f32 %f0, [%rd1];

    // Convert float to half
    cvt.rn.f16.f32 %h0, %f0;

    // Store result as u32 (the packed f16 in low bits)
    mov.b32 %r1, %h0;
    st.global.u32 [%rd0], %r1;
    ret;
}
"""

@gpu
class TestCvtF16F32:
    def test_basic(self, cuda_ctx):
        """cvt.rn.f16.f32: 1.0f should become 0x3C00 in FP16"""
        cubin = _compile(_PTX_CVT_F16_F32)
        assert cubin
        name = [k for k in cubin if not k.startswith('_')][0]
        ok = cuda_ctx.load(cubin[name])
        assert ok
        func = cuda_ctx.get_func('cvt_f16_f32')
        d_out = cuda_ctx.alloc(4)
        d_in = cuda_ctx.alloc(4)
        cuda_ctx.copy_to(d_in, struct.pack('f', 1.0))
        err = cuda_ctx.launch(func, (1,1,1), (1,1,1), [d_out, d_in])
        assert err == 0
        assert cuda_ctx.sync() == 0
        result = struct.unpack('I', cuda_ctx.copy_from(d_out, 4))[0]
        # FP16 1.0 = 0x3C00; F2FP.PACK_AB puts it in low 16 bits
        f16_val = result & 0xFFFF
        assert f16_val == 0x3C00, f"Expected FP16 1.0 (0x3C00), got 0x{f16_val:04x}"
        cuda_ctx.free(d_out)
        cuda_ctx.free(d_in)
