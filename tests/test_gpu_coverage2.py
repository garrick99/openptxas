"""GPU coverage tests - part 2 (kernel families needing separate CUDA context).

Split from test_gpu_coverage.py to stay under the SM_120 UR cache
pollution threshold (~24 module loads per CUDA context).
"""
import struct
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sass.pipeline import compile_ptx_source

try:
    import ctypes; _c = ctypes.cdll.LoadLibrary("nvcuda.dll"); _CUDA = _c.cuInit(0) == 0
except Exception:
    _CUDA = False
gpu = pytest.mark.skipif(not _CUDA, reason="No CUDA GPU")

# ============================================================

_PTX_DOT = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry dot_product(.param .u64 p_out, .param .u64 p_a, .param .u64 p_b) {
    .reg .u64 %rd<8>; .reg .f32 %f<8>;
    ld.param.u64 %rd0, [p_a]; ld.param.u64 %rd1, [p_b];
    mov.f32 %f0, 0f00000000;
    ld.global.f32 %f1, [%rd0]; ld.global.f32 %f2, [%rd1];
    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.u64 %rd0, %rd0, 4; add.u64 %rd1, %rd1, 4;
    ld.global.f32 %f1, [%rd0]; ld.global.f32 %f2, [%rd1];
    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.u64 %rd0, %rd0, 4; add.u64 %rd1, %rd1, 4;
    ld.global.f32 %f1, [%rd0]; ld.global.f32 %f2, [%rd1];
    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.u64 %rd0, %rd0, 4; add.u64 %rd1, %rd1, 4;
    ld.global.f32 %f1, [%rd0]; ld.global.f32 %f2, [%rd1];
    fma.rn.f32 %f0, %f1, %f2, %f0;
    ld.param.u64 %rd2, [p_out];
    st.global.f32 [%rd2], %f0;
    ret;
}
"""


@gpu
class TestDotProduct:
    def test_dot4(self, cuda_ctx):
        """dot([1,2,3,4], [5,6,7,8]) = 5+12+21+32 = 70."""
        cubins = compile_ptx_source(_PTX_DOT)
        assert cuda_ctx.load(cubins['dot_product'])
        func = cuda_ctx.get_func('dot_product')
        a_data = struct.pack('<4f', 1, 2, 3, 4)
        b_data = struct.pack('<4f', 5, 6, 7, 8)
        d_a = cuda_ctx.alloc(16); cuda_ctx.copy_to(d_a, a_data)
        d_b = cuda_ctx.alloc(16); cuda_ctx.copy_to(d_b, b_data)
        d_out = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_out, b'\x00' * 4)
        cuda_ctx.launch(func, (1,1,1), (1,1,1), [d_out, d_a, d_b])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<f', cuda_ctx.copy_from(d_out, 4))[0]
        assert abs(val - 70.0) < 0.01, f"dot: {val}"
        cuda_ctx.free(d_a); cuda_ctx.free(d_b); cuda_ctx.free(d_out)


# ============================================================
# Integer min/max chain
# ============================================================

_PTX_IMIN = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry imin_test(.param .u64 p_out, .param .u64 p_a, .param .u64 p_b) {
    .reg .u32 %r<4>; .reg .u64 %rd<4>;
    ld.param.u64 %rd0, [p_a]; ld.global.u32 %r0, [%rd0];
    ld.param.u64 %rd1, [p_b]; ld.global.u32 %r1, [%rd1];
    min.u32 %r2, %r0, %r1;
    ld.param.u64 %rd2, [p_out];
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_PTX_IMAX = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry imax_test(.param .u64 p_out, .param .u64 p_a, .param .u64 p_b) {
    .reg .u32 %r<4>; .reg .u64 %rd<4>;
    ld.param.u64 %rd0, [p_a]; ld.global.u32 %r0, [%rd0];
    ld.param.u64 %rd1, [p_b]; ld.global.u32 %r1, [%rd1];
    max.u32 %r2, %r0, %r1;
    ld.param.u64 %rd2, [p_out];
    st.global.u32 [%rd2], %r2;
    ret;
}
"""


@gpu
class TestIntMinMax:
    def test_min_u32(self, cuda_ctx):
        cubins = compile_ptx_source(_PTX_IMIN)
        assert cuda_ctx.load(cubins['imin_test'])
        d_a = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_a, struct.pack('<I', 10))
        d_b = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_b, struct.pack('<I', 20))
        d = cuda_ctx.alloc(4); cuda_ctx.copy_to(d, b'\x00' * 4)
        cuda_ctx.launch(cuda_ctx.get_func('imin_test'), (1,1,1), (1,1,1), [d, d_a, d_b])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<I', cuda_ctx.copy_from(d, 4))[0]
        assert val == 10, f"min.u32(10,20): {val}"
        cuda_ctx.free(d_a); cuda_ctx.free(d_b); cuda_ctx.free(d)

    def test_max_u32(self, cuda_ctx):
        cubins = compile_ptx_source(_PTX_IMAX)
        assert cuda_ctx.load(cubins['imax_test'])
        d_a = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_a, struct.pack('<I', 10))
        d_b = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_b, struct.pack('<I', 5))
        d = cuda_ctx.alloc(4); cuda_ctx.copy_to(d, b'\x00' * 4)
        cuda_ctx.launch(cuda_ctx.get_func('imax_test'), (1,1,1), (1,1,1), [d, d_a, d_b])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<I', cuda_ctx.copy_from(d, 4))[0]
        assert val == 10, f"max.u32(10,5): {val}"
        cuda_ctx.free(d_a); cuda_ctx.free(d_b); cuda_ctx.free(d)


# ============================================================
# Conversion matrix (cvt variants)
# ============================================================

_PTX_CVT_MATRIX = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry cvt_matrix(.param .u64 p_out, .param .u64 p_in) {
    .reg .u32 %r<8>; .reg .u64 %rd<4>; .reg .f32 %f<4>;
    ld.param.u64 %rd0, [p_in]; ld.global.u32 %r0, [%rd0];
    cvt.rn.f32.u32 %f0, %r0;
    cvt.rzi.u32.f32 %r1, %f0;
    ld.param.u64 %rd1, [p_out];
    st.global.f32 [%rd1], %f0;
    add.u64 %rd2, %rd1, 4;
    st.global.u32 [%rd2], %r1;
    ret;
}
"""


@gpu
class TestCvtMatrix:
    
    def test_conversion_roundtrip(self, cuda_ctx):
        """u32->f32->u32 roundtrip."""
        cubins = compile_ptx_source(_PTX_CVT_MATRIX)
        assert cuda_ctx.load(cubins['cvt_matrix'])
        func = cuda_ctx.get_func('cvt_matrix')
        d_in = cuda_ctx.alloc(4)
        cuda_ctx.copy_to(d_in, struct.pack('<I', 42))
        d = cuda_ctx.alloc(8); cuda_ctx.copy_to(d, b'\x00' * 8)
        cuda_ctx.launch(func, (1,1,1), (1,1,1), [d, d_in])
        assert cuda_ctx.sync() == 0
        data = cuda_ctx.copy_from(d, 8)
        f_val = struct.unpack_from('<f', data, 0)[0]
        u_val = struct.unpack_from('<I', data, 4)[0]
        assert f_val == 42.0, f"cvt.f32.u32: {f_val}"
        assert u_val == 42, f"cvt.u32.f32: {u_val}"
        cuda_ctx.free(d_in); cuda_ctx.free(d)


# ============================================================
# 64-bit arithmetic chain
# ============================================================

_PTX_U64_ADD = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry u64_add(.param .u64 p_out, .param .u64 p_a, .param .u64 p_b) {
    .reg .u64 %rd<8>;
    ld.param.u64 %rd0, [p_a]; ld.global.u64 %rd1, [%rd0];
    ld.param.u64 %rd2, [p_b]; ld.global.u64 %rd3, [%rd2];
    add.u64 %rd4, %rd1, %rd3;
    ld.param.u64 %rd5, [p_out];
    st.global.u64 [%rd5], %rd4;
    ret;
}
"""


@gpu
class TestU64Arith:
    def test_64bit_add(self, cuda_ctx):
        """add.u64(0x1_0000_0000, 0x2_0000_0000)=0x3_0000_0000."""
        cubins = compile_ptx_source(_PTX_U64_ADD)
        assert cuda_ctx.load(cubins['u64_add'])
        d_a = cuda_ctx.alloc(8); cuda_ctx.copy_to(d_a, struct.pack('<Q', 0x100000000))
        d_b = cuda_ctx.alloc(8); cuda_ctx.copy_to(d_b, struct.pack('<Q', 0x200000000))
        d = cuda_ctx.alloc(8); cuda_ctx.copy_to(d, b'\x00' * 8)
        cuda_ctx.launch(cuda_ctx.get_func('u64_add'), (1,1,1), (1,1,1), [d, d_a, d_b])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<Q', cuda_ctx.copy_from(d, 8))[0]
        assert val == 0x300000000, f"add: {val:#x}"
        cuda_ctx.free(d_a); cuda_ctx.free(d_b); cuda_ctx.free(d)


# ============================================================
# Multi-pointer deferred params (5 u64 params)
# ============================================================

_PTX_MULTI_PARAM = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry multi_param(
    .param .u64 p0, .param .u64 p1, .param .u64 p2,
    .param .u64 p3, .param .u64 p4
) {
    .reg .u64 %rd<8>; .reg .f32 %f<8>;
    ld.param.u64 %rd0, [p0]; ld.global.f32 %f0, [%rd0];
    ld.param.u64 %rd1, [p1]; ld.global.f32 %f1, [%rd1];
    ld.param.u64 %rd2, [p2]; ld.global.f32 %f2, [%rd2];
    ld.param.u64 %rd3, [p3]; ld.global.f32 %f3, [%rd3];
    fma.rn.f32 %f0, %f0, %f1, %f0;
    fma.rn.f32 %f0, %f2, %f3, %f0;
    ld.param.u64 %rd4, [p4];
    st.global.f32 [%rd4], %f0;
    ret;
}
"""


@gpu
class TestMultiParam:
    
    def test_five_param_deferred(self, cuda_ctx):
        """5 u64 params, all deferred: fma(a,b,a) + fma(c,d,prev) = 28."""
        cubins = compile_ptx_source(_PTX_MULTI_PARAM)
        assert cuda_ctx.load(cubins['multi_param'])
        func = cuda_ctx.get_func('multi_param')
        vals = [2.0, 3.0, 4.0, 5.0]
        ptrs = []
        for v in vals:
            p = cuda_ctx.alloc(4)
            cuda_ctx.copy_to(p, struct.pack('<f', v))
            ptrs.append(p)
        d_out = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_out, b'\x00' * 4)
        cuda_ctx.launch(func, (1,1,1), (1,1,1), [*ptrs, d_out])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<f', cuda_ctx.copy_from(d_out, 4))[0]
        assert abs(val - 28.0) < 0.01, f"multi_param: {val}"
        for p in ptrs: cuda_ctx.free(p)
        cuda_ctx.free(d_out)
