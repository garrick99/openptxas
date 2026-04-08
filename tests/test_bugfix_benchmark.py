"""Regression tests for five bugs discovered during benchmark development (2026-04-04).

Each test exercises one bug:
  1. shfl.sync delta immediate encoded as 0
  2. ld.shared cross-address stale reads after bar.sync (LDC/LOP3 RAW hazard)
  3. atom.global.add.f32 aggregates (I2FP→ATOMG src tracking)
  4. multi-ld.global register aliasing (add.u64-imm _gpr_written tracking)
  5. mul.lo.s32 with immediate src1 isel crash

Requires RTX 5090 (SM_120). Marked with @pytest.mark.gpu.
"""
import struct
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass.pipeline import compile_ptx_source

# CUDA fixture provided by conftest.py (session-scoped primary context)
try:
    import ctypes as _ctypes; _c = _ctypes.cdll.LoadLibrary("nvcuda.dll"); _CUDA = _c.cuInit(0) == 0
except Exception:
    _CUDA = False
gpu = pytest.mark.skipif(not _CUDA, reason="No CUDA GPU")


def _compile(src):
    return compile_ptx_source(src)


# ---------------------------------------------------------------------------
# Bug 5: mul.lo.s32 with immediate 2nd operand
# ---------------------------------------------------------------------------

_PTX_MUL_IMM = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry mul_imm_test(.param .u64 p) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    mov.u32 %r0, %tid.x;
    mul.lo.s32 %r1, %r0, 4;
    mul.lo.s32 %r2, %r0, 7;
    add.u32 %r1, %r1, %r2;
    ld.param.u64 %rd0, [p];
    shl.b32 %r3, %r0, 2;
    cvt.u64.u32 %rd1, %r3;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r1;
    ret;
}
"""


def test_mul_lo_imm_compiles():
    """Bug 5: mul.lo.s32 with immediate src1 must not crash isel."""
    cubins = _compile(_PTX_MUL_IMM)
    assert 'mul_imm_test' in cubins
    assert len(cubins['mul_imm_test']) > 0


class TestMulImmGpu:
    @gpu
    def test_mul_imm_correct(self, cuda_ctx):
        cubins = _compile(_PTX_MUL_IMM)
        cuda_ctx.load(cubins['mul_imm_test'])
        func = cuda_ctx.get_func('mul_imm_test')
        N = 32
        d_out = cuda_ctx.alloc(N * 4)
        try:
            cuda_ctx.copy_to(d_out, b'\x00' * (N * 4))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1), [d_out])
            assert err == 0
            assert cuda_ctx.sync() == 0
            raw = cuda_ctx.copy_from(d_out, N * 4)
            vals = struct.unpack(f'<{N}I', raw)
            for i in range(N):
                assert vals[i] == i * 11, f"tid {i}: got {vals[i]}, expected {i*11}"
        finally:
            cuda_ctx.free(d_out)


# ---------------------------------------------------------------------------
# Bug 1: shfl.sync.down delta immediate
# ---------------------------------------------------------------------------

_PTX_SHFL_DOWN = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry shfl_down_test(.param .u64 p) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    .reg .pred %pp;
    mov.u32 %r0, %tid.x;
    add.u32 %r2, %r0, 100;
    shfl.sync.down.b32 %r1|%pp, %r2, 16, 0x1f, 0xffffffff;
    ld.param.u64 %rd0, [p];
    shl.b32 %r3, %r0, 2;
    cvt.u64.u32 %rd1, %r3;
    add.u64 %rd0, %rd0, %rd1;
    st.global.u32 [%rd0], %r1;
    ret;
}
"""


class TestShflDelta:
    @gpu
    def test_shfl_down_delta16(self, cuda_ctx):
        cubins = _compile(_PTX_SHFL_DOWN)
        cuda_ctx.load(cubins['shfl_down_test'])
        func = cuda_ctx.get_func('shfl_down_test')
        N = 32
        d_out = cuda_ctx.alloc(N * 4)
        try:
            cuda_ctx.copy_to(d_out, b'\x00' * (N * 4))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1), [d_out])
            assert err == 0
            assert cuda_ctx.sync() == 0
            raw = cuda_ctx.copy_from(d_out, N * 4)
            vals = struct.unpack(f'<{N}I', raw)
            for i in range(N):
                expected = (100 + i + 16) if i < 16 else (100 + i)
                assert vals[i] == expected, \
                    f"tid {i}: got {vals[i]}, expected {expected}"
        finally:
            cuda_ctx.free(d_out)


# ---------------------------------------------------------------------------
# Bug 2: ld.shared cross-thread via xor-derived address after bar.sync
# (root cause: LDC->LOP3 RAW hazard for xor imm)
# ---------------------------------------------------------------------------

_PTX_SMEM_CROSS = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry smem_cross_test(.param .u64 p) {
    .reg .u32 %r<8>;
    .reg .u64 %rd<4>;
    .shared .align 4 .b32 smem[32];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    add.u32 %r3, %r0, 1000;
    st.shared.u32 [%r1], %r3;
    bar.sync 0;
    xor.b32 %r4, %r0, 1;
    shl.b32 %r5, %r4, 2;
    ld.shared.u32 %r6, [%r5];
    ld.param.u64 %rd0, [p];
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r6;
    ret;
}
"""


class TestSmemCross:
    @gpu
    def test_smem_xor_cross(self, cuda_ctx):
        cubins = _compile(_PTX_SMEM_CROSS)
        cuda_ctx.load(cubins['smem_cross_test'])
        func = cuda_ctx.get_func('smem_cross_test')
        N = 32
        d_out = cuda_ctx.alloc(N * 4)
        try:
            cuda_ctx.copy_to(d_out, b'\x00' * (N * 4))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1),
                                  [d_out], smem=256)
            assert err == 0
            assert cuda_ctx.sync() == 0
            raw = cuda_ctx.copy_from(d_out, N * 4)
            vals = struct.unpack(f'<{N}I', raw)
            for i in range(N):
                expected = (i ^ 1) + 1000
                assert vals[i] == expected, \
                    f"tid {i}: got {vals[i]}, expected {expected}"
        finally:
            cuda_ctx.free(d_out)


# ---------------------------------------------------------------------------
# Bug 3: atom.global.add.f32 aggregates (I2FP source tracking)
# ---------------------------------------------------------------------------

_PTX_ATOM_F32 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry atom_f32_test(.param .u64 p) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<4>;
    mov.u32 %r0, %tid.x;
    cvt.rn.f32.u32 %f0, %r0;
    ld.param.u64 %rd0, [p];
    atom.global.add.f32 %f1, [%rd0], %f0;
    ret;
}
"""


class TestAtomF32:
    @gpu
    def test_atom_add_f32_aggregate(self, cuda_ctx):
        cubins = _compile(_PTX_ATOM_F32)
        cuda_ctx.load(cubins['atom_f32_test'])
        func = cuda_ctx.get_func('atom_f32_test')
        d_accum = cuda_ctx.alloc(4)
        try:
            cuda_ctx.copy_to(d_accum, struct.pack('<f', 0.0))
            err = cuda_ctx.launch(func, (1, 1, 1), (32, 1, 1), [d_accum])
            assert err == 0
            assert cuda_ctx.sync() == 0
            result = struct.unpack('<f', cuda_ctx.copy_from(d_accum, 4))[0]
            # Sum of 0..31 = 496
            assert result == 496.0, f"got {result}, expected 496.0"
        finally:
            cuda_ctx.free(d_accum)


# ---------------------------------------------------------------------------
# Bug 4: multi-ld.global with register aliasing
# ---------------------------------------------------------------------------

_PTX_MULTI_LDG = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry multi_ldg_test(.param .u64 pin, .param .u64 pout) {
    .reg .u32 %r<8>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd0, [pin];
    ld.param.u64 %rd1, [pout];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd2, %r1;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];
    add.u64 %rd4, %rd3, 4;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f0, %f1;
    add.u64 %rd5, %rd1, %rd2;
    st.global.f32 [%rd5], %f2;
    ret;
}
"""


class TestMultiLdg:
    @gpu
    def test_multi_ldg_aliased_base(self, cuda_ctx):
        cubins = _compile(_PTX_MULTI_LDG)
        cuda_ctx.load(cubins['multi_ldg_test'])
        func = cuda_ctx.get_func('multi_ldg_test')
        N = 4
        d_in = cuda_ctx.alloc((N + 1) * 4)
        d_out = cuda_ctx.alloc(N * 4)
        in_vals = [float(i + 1) for i in range(N + 1)]
        try:
            cuda_ctx.copy_to(d_in, struct.pack(f'<{N+1}f', *in_vals))
            cuda_ctx.copy_to(d_out, b'\x00' * (N * 4))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1), [d_in, d_out])
            assert err == 0
            assert cuda_ctx.sync() == 0
            raw = cuda_ctx.copy_from(d_out, N * 4)
            vals = struct.unpack(f'<{N}f', raw)
            for i in range(N):
                expected = in_vals[i] + in_vals[i + 1]
                assert vals[i] == expected, \
                    f"tid {i}: got {vals[i]}, expected {expected}"
        finally:
            cuda_ctx.free(d_in)
            cuda_ctx.free(d_out)
