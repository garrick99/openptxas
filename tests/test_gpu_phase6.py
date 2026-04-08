"""
tests/test_gpu_phase6.py -- Phase 6 GPU tests: scoreboard bug reproducers.

Bug 1: LDC from constant bank after BAR.SYNC — verify no stale data.
Bug 2: Dual LDG.E.64 feeding DADD — verify second load is not zero.

Run: python -m pytest tests/test_gpu_phase6.py -v -m gpu --tb=short
"""
import ctypes
import struct
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass.pipeline import compile_ptx_source

# ---------------------------------------------------------------------------
# GPU driver bootstrap (same pattern as phase1-5)
# ---------------------------------------------------------------------------


try:
    import ctypes; _c = ctypes.cdll.LoadLibrary("nvcuda.dll"); _CUDA = _c.cuInit(0) == 0
except Exception:
    _CUDA = False
gpu = pytest.mark.skipif(not _CUDA, reason="No CUDA GPU")


def _compile(ptx_src: str) -> dict[str, bytes]:
    return compile_ptx_source(ptx_src)


# ===========================================================================
# Bug 1: LDC after BAR.SYNC — XOR with param constant must be correct
# ===========================================================================

_PTX_BAR_LDC_XOR = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry bar_ldc_xor(
    .param .u64 p_out,
    .param .u32 p_n,
    .param .u32 p_mask
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<4>;
    .reg .pred %p0;
    .shared .align 4 .b32 smem[256];

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r5, [p_n];
    setp.ge.u32 %p0, %r0, %r5;
    @%p0 bra DONE;

    // Write tid+42 to shared memory
    shl.b32 %r1, %r0, 2;
    add.u32 %r2, %r0, 42;
    st.shared.b32 [%r1], %r2;
    bar.sync 0;

    // After barrier: load parameter constant and XOR
    ld.param.u32 %r3, [p_mask];
    xor.b32 %r4, %r2, %r3;

    // Store result
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
DONE:
    ret;
}
"""


class TestBarLdcXor:
    @gpu
    def test_bar_ldc_xor_correctness(self, cuda_ctx):
        """After BAR.SYNC, LDC-loaded param XOR must produce correct result."""
        cubins = _compile(_PTX_BAR_LDC_XOR)
        assert 'bar_ldc_xor' in cubins
        ok = cuda_ctx.load(cubins['bar_ldc_xor'])
        assert ok, "cuModuleLoadData failed for bar_ldc_xor"
        func = cuda_ctx.get_func('bar_ldc_xor')

        N = 32
        mask = 0x55  # small value to avoid ctypes sign issues
        d_out = cuda_ctx.alloc(N * 4)
        try:
            cuda_ctx.copy_to(d_out, b'\x00' * (N * 4))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1),
                                  [d_out, N, mask], smem=256 * 4)
            assert err == 0, f"launch failed: {err}"
            assert cuda_ctx.sync() == 0, "bar_ldc_xor kernel crashed"
            raw = cuda_ctx.copy_from(d_out, N * 4)
            results = struct.unpack(f'<{N}I', raw)
            for tid in range(N):
                val = tid + 42
                expected = (val ^ mask) & 0xFFFFFFFF
                assert results[tid] == expected, \
                    f"tid {tid}: got 0x{results[tid]:08x}, expected 0x{expected:08x}"
        finally:
            cuda_ctx.free(d_out)


# ===========================================================================
# Bug 2: Dual LDG.E.64 feeding DADD — second load must not be zero
# ===========================================================================

_PTX_DUAL_LDG64_DADD = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry dual_ldg64_dadd(
    .param .u64 p_out,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u32 p_n
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<16>;
    .reg .f64 %fd<4>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    // Compute byte offset: tid * 8 (f64 = 8 bytes)
    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 3;

    // Load a[tid]
    ld.param.u64 %rd1, [p_a];
    add.u64 %rd2, %rd1, %rd0;
    ld.global.f64 %fd0, [%rd2];

    // Load b[tid]
    ld.param.u64 %rd3, [p_b];
    add.u64 %rd4, %rd3, %rd0;
    ld.global.f64 %fd1, [%rd4];

    // Compute a[tid] + b[tid]
    add.f64 %fd2, %fd0, %fd1;

    // Store result
    ld.param.u64 %rd5, [p_out];
    add.u64 %rd6, %rd5, %rd0;
    st.global.f64 [%rd6], %fd2;
DONE:
    ret;
}
"""


class TestDualLdg64Dadd:
    @gpu
    def test_dual_ldg64_dadd(self, cuda_ctx):
        """Two LDG.E.64 loads feeding DADD: both values must be correct."""
        cubins = _compile(_PTX_DUAL_LDG64_DADD)
        assert 'dual_ldg64_dadd' in cubins
        ok = cuda_ctx.load(cubins['dual_ldg64_dadd'])
        assert ok, "cuModuleLoadData failed for dual_ldg64_dadd"
        func = cuda_ctx.get_func('dual_ldg64_dadd')

        N = 32
        # Use non-trivial f64 values so zero-from-second-load is detectable
        a_vals = [float(i) * 1.5 + 100.0 for i in range(N)]
        b_vals = [float(i) * 2.5 + 200.0 for i in range(N)]
        a_bytes = struct.pack(f'<{N}d', *a_vals)
        b_bytes = struct.pack(f'<{N}d', *b_vals)

        d_a = cuda_ctx.alloc(N * 8)
        d_b = cuda_ctx.alloc(N * 8)
        d_out = cuda_ctx.alloc(N * 8)
        try:
            cuda_ctx.copy_to(d_a, a_bytes)
            cuda_ctx.copy_to(d_b, b_bytes)
            cuda_ctx.copy_to(d_out, b'\x00' * (N * 8))

            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1),
                                  [d_out, d_a, d_b, N])
            assert err == 0, f"launch failed: {err}"
            assert cuda_ctx.sync() == 0, "dual_ldg64_dadd kernel crashed"

            raw = cuda_ctx.copy_from(d_out, N * 8)
            results = struct.unpack(f'<{N}d', raw)
            for i in range(N):
                expected = a_vals[i] + b_vals[i]
                assert abs(results[i] - expected) < 1e-10, \
                    f"idx {i}: got {results[i]}, expected {expected} " \
                    f"(a={a_vals[i]}, b={b_vals[i]})"
        finally:
            cuda_ctx.free(d_a)
            cuda_ctx.free(d_b)
            cuda_ctx.free(d_out)

    @gpu
    def test_dual_ldg64_dmul(self, cuda_ctx):
        """Two LDG.E.64 loads feeding DMUL: both values must be correct."""
        ptx = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry dual_ldg64_dmul(
    .param .u64 p_out,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u32 p_n
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<16>;
    .reg .f64 %fd<4>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 3;

    ld.param.u64 %rd1, [p_a];
    add.u64 %rd2, %rd1, %rd0;
    ld.global.f64 %fd0, [%rd2];

    ld.param.u64 %rd3, [p_b];
    add.u64 %rd4, %rd3, %rd0;
    ld.global.f64 %fd1, [%rd4];

    mul.f64 %fd2, %fd0, %fd1;

    ld.param.u64 %rd5, [p_out];
    add.u64 %rd6, %rd5, %rd0;
    st.global.f64 [%rd6], %fd2;
DONE:
    ret;
}
"""
        cubins = _compile(ptx)
        assert 'dual_ldg64_dmul' in cubins
        ok = cuda_ctx.load(cubins['dual_ldg64_dmul'])
        assert ok, "cuModuleLoadData failed for dual_ldg64_dmul"
        func = cuda_ctx.get_func('dual_ldg64_dmul')

        N = 32
        a_vals = [float(i) * 0.1 + 1.0 for i in range(N)]
        b_vals = [float(i) * 0.2 + 2.0 for i in range(N)]
        a_bytes = struct.pack(f'<{N}d', *a_vals)
        b_bytes = struct.pack(f'<{N}d', *b_vals)

        d_a = cuda_ctx.alloc(N * 8)
        d_b = cuda_ctx.alloc(N * 8)
        d_out = cuda_ctx.alloc(N * 8)
        try:
            cuda_ctx.copy_to(d_a, a_bytes)
            cuda_ctx.copy_to(d_b, b_bytes)
            cuda_ctx.copy_to(d_out, b'\x00' * (N * 8))

            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1),
                                  [d_out, d_a, d_b, N])
            assert err == 0, f"launch failed: {err}"
            assert cuda_ctx.sync() == 0, "dual_ldg64_dmul kernel crashed"

            raw = cuda_ctx.copy_from(d_out, N * 8)
            results = struct.unpack(f'<{N}d', raw)
            for i in range(N):
                expected = a_vals[i] * b_vals[i]
                assert abs(results[i] - expected) < 1e-10, \
                    f"idx {i}: got {results[i]}, expected {expected}"
        finally:
            cuda_ctx.free(d_a)
            cuda_ctx.free(d_b)
            cuda_ctx.free(d_out)
