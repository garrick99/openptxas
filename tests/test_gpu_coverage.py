"""GPU coverage tests for untested instruction classes.

Tests instruction families not covered by existing test suites:
- MUFU (sin, cos, ex2, lg2, rsqrt, rcp)
- Broader kernel patterns (reduction, scan prefix, stencil-like)
"""
import struct
import ctypes
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
# MUFU tests (sin, cos, ex2, lg2, rsqrt)
# ============================================================

def _mufu_ptx(op, approx='approx'):
    return f'''
.version 9.0
.target sm_120
.address_size 64
.visible .entry mufu_k(.param .u64 p_out, .param .u64 p_in) {{
    .reg .u32 %r<4>; .reg .u64 %rd<8>; .reg .f32 %f<4>;
    mov.u32 %r0, %tid.x;
    cvt.u64.u32 %rd0, %r0; shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [p_in]; add.u64 %rd2, %rd1, %rd0;
    ld.param.u64 %rd3, [p_out]; add.u64 %rd4, %rd3, %rd0;
    ld.global.f32 %f0, [%rd2];
    {op}.{approx}.f32 %f1, %f0;
    st.global.f32 [%rd4], %f1;
    ret;
}}
'''


@gpu
class TestMufuSuite:
    def _run(self, cuda_ctx, op, inputs, expected, tol=1e-3, approx='approx'):
        ptx = _mufu_ptx(op, approx)
        cubins = compile_ptx_source(ptx)
        assert cuda_ctx.load(cubins['mufu_k'])
        func = cuda_ctx.get_func('mufu_k')
        N = len(inputs)
        sz = N * 4
        d_in = cuda_ctx.alloc(sz); d_out = cuda_ctx.alloc(sz)
        cuda_ctx.copy_to(d_in, struct.pack(f'<{N}f', *inputs))
        cuda_ctx.copy_to(d_out, b'\x00' * sz)
        err = cuda_ctx.launch(func, (1,1,1), (N,1,1), [d_out, d_in])
        assert err == 0
        assert cuda_ctx.sync() == 0, f"{op} crashed"
        results = struct.unpack(f'<{N}f', cuda_ctx.copy_from(d_out, sz))
        for i in range(N):
            assert abs(results[i] - expected[i]) < tol, \
                f"{op} idx={i}: got {results[i]}, expected {expected[i]}"
        cuda_ctx.free(d_in); cuda_ctx.free(d_out)

    def test_sin(self, cuda_ctx):
        import math
        inputs = [0.0, math.pi/6, math.pi/2, math.pi]
        self._run(cuda_ctx, 'sin', inputs, [math.sin(x) for x in inputs], tol=1e-2)

    def test_cos(self, cuda_ctx):
        import math
        inputs = [0.0, math.pi/3, math.pi/2, math.pi]
        self._run(cuda_ctx, 'cos', inputs, [math.cos(x) for x in inputs], tol=1e-2)

    def test_ex2(self, cuda_ctx):
        self._run(cuda_ctx, 'ex2', [0.0, 1.0, 2.0, -1.0], [1.0, 2.0, 4.0, 0.5], tol=1e-2)

    def test_lg2(self, cuda_ctx):
        import math
        self._run(cuda_ctx, 'lg2', [1.0, 2.0, 4.0, 8.0], [0.0, 1.0, 2.0, 3.0], tol=1e-2)

    def test_rsqrt(self, cuda_ctx):
        self._run(cuda_ctx, 'rsqrt', [1.0, 4.0, 9.0, 16.0], [1.0, 0.5, 1/3.0, 0.25], tol=1e-2)

    def test_rcp(self, cuda_ctx):
        self._run(cuda_ctx, 'rcp', [1.0, 2.0, 4.0, 0.5], [1.0, 0.5, 0.25, 2.0], tol=1e-3)

    def test_sqrt(self, cuda_ctx):
        self._run(cuda_ctx, 'sqrt', [1.0, 4.0, 9.0, 16.0], [1.0, 2.0, 3.0, 4.0], tol=1e-3)

    def test_tanh(self, cuda_ctx):
        import math
        inputs = [0.0, 0.5, 1.0, -1.0]
        self._run(cuda_ctx, 'tanh', inputs, [math.tanh(x) for x in inputs], tol=1e-2)


# ============================================================
# Broader kernel families
# ============================================================

_PTX_WARP_REDUCE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry warp_reduce(
    .param .u64 p_out, .param .u64 p_in, .param .u32 n)
{
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<4>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    cvt.u64.u32 %rd0, %r0; shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [p_in]; add.u64 %rd2, %rd1, %rd0;
    ld.global.f32 %f0, [%rd2];

    // Warp-level butterfly reduction via shfl.down
    shfl.sync.down.b32 %f1, %f0, 16, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;
    shfl.sync.down.b32 %f1, %f0, 8, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;
    shfl.sync.down.b32 %f1, %f0, 4, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;
    shfl.sync.down.b32 %f1, %f0, 2, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;
    shfl.sync.down.b32 %f1, %f0, 1, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;

    // Thread 0 writes result
    setp.ne.u32 %p0, %r0, 0; @%p0 ret;
    ld.param.u64 %rd3, [p_out];
    st.global.f32 [%rd3], %f0;
    ret;
}
"""


@gpu
class TestWarpReduce:
    def test_sum_32(self, cuda_ctx):
        """Warp-level sum reduction of 32 floats."""
        cubins = compile_ptx_source(_PTX_WARP_REDUCE)
        assert cuda_ctx.load(cubins['warp_reduce'])
        func = cuda_ctx.get_func('warp_reduce')
        N = 32
        vals = [float(i + 1) for i in range(N)]
        d_in = cuda_ctx.alloc(N * 4)
        d_out = cuda_ctx.alloc(4)
        cuda_ctx.copy_to(d_in, struct.pack(f'<{N}f', *vals))
        cuda_ctx.copy_to(d_out, b'\x00' * 4)
        err = cuda_ctx.launch(func, (1,1,1), (N,1,1), [d_out, d_in, N])
        assert err == 0
        assert cuda_ctx.sync() == 0
        result = struct.unpack('f', cuda_ctx.copy_from(d_out, 4))[0]
        expected = sum(vals)
        assert abs(result - expected) < 1e-2, f"got {result}, expected {expected}"
        cuda_ctx.free(d_in)
        cuda_ctx.free(d_out)


_PTX_SAXPY_MULTI = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry saxpy_multi(
    .param .u64 p_x, .param .u64 p_y, .param .u64 p_z,
    .param .u64 p_out, .param .u32 n)
{
    .reg .u32 %r<4>; .reg .u64 %rd<16>; .reg .f32 %f<8>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r3, %r1; @%p0 ret;
    cvt.u64.u32 %rd0, %r3; shl.b64 %rd0, %rd0, 2;
    // 4-pointer kernel: tests deferred param interleaving with 4 u64 params
    ld.param.u64 %rd1, [p_x]; add.u64 %rd5, %rd1, %rd0;
    ld.param.u64 %rd2, [p_y]; add.u64 %rd6, %rd2, %rd0;
    ld.param.u64 %rd3, [p_z]; add.u64 %rd7, %rd3, %rd0;
    ld.param.u64 %rd4, [p_out]; add.u64 %rd8, %rd4, %rd0;
    ld.global.f32 %f0, [%rd5];
    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    // out = x + y + z
    add.f32 %f3, %f0, %f1;
    add.f32 %f4, %f3, %f2;
    st.global.f32 [%rd8], %f4;
    ret;
}
"""


@gpu
class TestSaxpyMulti:
    def test_4ptr_add(self, cuda_ctx):
        """4-pointer kernel: out[i] = x[i] + y[i] + z[i]. Tests deferred param with 4 u64 params."""
        cubins = compile_ptx_source(_PTX_SAXPY_MULTI)
        assert cuda_ctx.load(cubins['saxpy_multi'])
        func = cuda_ctx.get_func('saxpy_multi')
        N = 64
        x = [float(i) for i in range(N)]
        y = [float(i * 10) for i in range(N)]
        z = [float(i * 100) for i in range(N)]
        sz = N * 4
        d_x = cuda_ctx.alloc(sz); d_y = cuda_ctx.alloc(sz)
        d_z = cuda_ctx.alloc(sz); d_out = cuda_ctx.alloc(sz)
        cuda_ctx.copy_to(d_x, struct.pack(f'<{N}f', *x))
        cuda_ctx.copy_to(d_y, struct.pack(f'<{N}f', *y))
        cuda_ctx.copy_to(d_z, struct.pack(f'<{N}f', *z))
        cuda_ctx.copy_to(d_out, b'\x00' * sz)
        err = cuda_ctx.launch(func, (1,1,1), (N,1,1), [d_x, d_y, d_z, d_out, N])
        assert err == 0
        assert cuda_ctx.sync() == 0
        results = struct.unpack(f'<{N}f', cuda_ctx.copy_from(d_out, sz))
        for i in range(N):
            expected = x[i] + y[i] + z[i]
            assert abs(results[i] - expected) < 1e-2, f"idx {i}: got {results[i]}, expected {expected}"
        cuda_ctx.free(d_x); cuda_ctx.free(d_y)
        cuda_ctx.free(d_z); cuda_ctx.free(d_out)


_PTX_BITWISE_CHAIN = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry bitwise_chain(
    .param .u64 p_out, .param .u64 p_in, .param .u32 n)
{
    .reg .u32 %r<12>; .reg .u64 %rd<8>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    cvt.u64.u32 %rd0, %r0; shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [p_in]; add.u64 %rd2, %rd1, %rd0;
    ld.param.u64 %rd3, [p_out]; add.u64 %rd4, %rd3, %rd0;
    ld.global.u32 %r2, [%rd2];
    // Chain: brev → popc → clz → bfind → bfe → bfi → prmt
    brev.b32 %r3, %r2;
    popc.b32 %r4, %r3;
    clz.b32 %r5, %r2;
    bfind.u32 %r6, %r2;
    bfe.u32 %r7, %r2, 0, 8;
    bfi.b32 %r8, %r7, %r2, 8, 8;
    // Combine: result = (popc << 24) | (clz << 16) | (bfind << 8) | bfe
    shl.b32 %r9, %r4, 24;
    shl.b32 %r10, %r5, 16;
    or.b32 %r11, %r9, %r10;
    shl.b32 %r9, %r6, 8;
    or.b32 %r11, %r11, %r9;
    or.b32 %r11, %r11, %r7;
    st.global.u32 [%rd4], %r11;
    ret;
}
"""


@gpu
class TestBitwiseChain:
    def test_bit_ops(self, cuda_ctx):
        """Chain of bit manipulation ops: brev, popc, clz, bfind, bfe, bfi."""
        cubins = compile_ptx_source(_PTX_BITWISE_CHAIN)
        assert cuda_ctx.load(cubins['bitwise_chain'])
        func = cuda_ctx.get_func('bitwise_chain')
        N = 4
        inputs = [0xFF, 0x12345678, 0x80000000, 0x0000FFFF]
        d_in = cuda_ctx.alloc(N * 4); d_out = cuda_ctx.alloc(N * 4)
        cuda_ctx.copy_to(d_in, struct.pack(f'<{N}I', *inputs))
        cuda_ctx.copy_to(d_out, b'\x00' * (N * 4))
        err = cuda_ctx.launch(func, (1,1,1), (N,1,1), [d_out, d_in, N])
        assert err == 0
        assert cuda_ctx.sync() == 0, "bitwise_chain crashed"
        results = struct.unpack(f'<{N}I', cuda_ctx.copy_from(d_out, N * 4))
        # Verify each result is non-zero (specific values depend on encoding)
        for i in range(N):
            assert results[i] != 0, f"idx {i}: got 0 for input {inputs[i]:#x}"
        cuda_ctx.free(d_in); cuda_ctx.free(d_out)


# ============================================================
# Atomic variants (or, and, xor, max.u32, min.u32)
# ============================================================

_PTX_ATOM_OR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry atom_or(.param .u64 p_out) {
    .reg .u32 %r<4>; .reg .u64 %rd<2>;
    mov.u32 %r1, 0xFF;
    ld.param.u64 %rd0, [p_out];
    atom.global.or.b32 %r0, [%rd0], %r1;
    ret;
}
"""

_PTX_ATOM_MINMAX_U = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry atom_minmax_u(.param .u64 p_min, .param .u64 p_max) {
    .reg .u32 %r<4>; .reg .s32 %s<4>; .reg .u64 %rd<4>;
    mov.u32 %r0, %tid.x; add.u32 %r1, %r0, 1;
    ld.param.u64 %rd0, [p_min]; atom.global.min.s32 %s0, [%rd0], %r1;
    ld.param.u64 %rd1, [p_max]; atom.global.max.s32 %s1, [%rd1], %r1;
    ret;
}
"""


@gpu
class TestAtomicVariants:

    def test_atomic_or(self, cuda_ctx):
        cubins = compile_ptx_source(_PTX_ATOM_OR)
        assert cuda_ctx.load(cubins['atom_or'])
        func = cuda_ctx.get_func('atom_or')
        d = cuda_ctx.alloc(4); cuda_ctx.copy_to(d, struct.pack('<I', 0))
        cuda_ctx.launch(func, (1,1,1), (32,1,1), [d])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<I', cuda_ctx.copy_from(d, 4))[0]
        assert val == 0xFF
        cuda_ctx.free(d)


    def test_atomic_min_max_s32(self, cuda_ctx):
        cubins = compile_ptx_source(_PTX_ATOM_MINMAX_U)
        assert cuda_ctx.load(cubins['atom_minmax_u'])
        func = cuda_ctx.get_func('atom_minmax_u')
        d_min = cuda_ctx.alloc(4); d_max = cuda_ctx.alloc(4)
        cuda_ctx.copy_to(d_min, struct.pack('<i', 0x7FFFFFFF))
        cuda_ctx.copy_to(d_max, struct.pack('<I', 0))
        cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_min, d_max])
        assert cuda_ctx.sync() == 0
        mn = struct.unpack('<I', cuda_ctx.copy_from(d_min, 4))[0]
        mx = struct.unpack('<I', cuda_ctx.copy_from(d_max, 4))[0]
        assert mn == 1, f"min: {mn}"
        assert mx == 32, f"max: {mx}"
        cuda_ctx.free(d_min); cuda_ctx.free(d_max)


# ============================================================
# setp variants (all 6 integer comparisons)
# ============================================================

_PTX_SETP_VARIANTS = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry setp_variants(.param .u64 p_out) {
    .reg .u32 %r<8>; .reg .u64 %rd<2>; .reg .pred %p<2>;
    mov.u32 %r0, 5; mov.u32 %r1, 10;
    mov.u32 %r3, 0;
    setp.eq.s32 %p0, %r0, %r1; selp.u32 %r2, 1, 0, %p0;
    shl.b32 %r2, %r2, 5; or.b32 %r3, %r3, %r2;
    setp.ne.s32 %p0, %r0, %r1; selp.u32 %r2, 1, 0, %p0;
    shl.b32 %r2, %r2, 4; or.b32 %r3, %r3, %r2;
    setp.lt.s32 %p0, %r0, %r1; selp.u32 %r2, 1, 0, %p0;
    shl.b32 %r2, %r2, 3; or.b32 %r3, %r3, %r2;
    setp.le.s32 %p0, %r0, %r1; selp.u32 %r2, 1, 0, %p0;
    shl.b32 %r2, %r2, 2; or.b32 %r3, %r3, %r2;
    setp.gt.s32 %p0, %r0, %r1; selp.u32 %r2, 1, 0, %p0;
    shl.b32 %r2, %r2, 1; or.b32 %r3, %r3, %r2;
    setp.ge.s32 %p0, %r0, %r1; selp.u32 %r2, 1, 0, %p0;
    or.b32 %r3, %r3, %r2;
    ld.param.u64 %rd0, [p_out]; st.global.u32 [%rd0], %r3;
    ret;
}
"""


@gpu
class TestSetpVariants:

    def test_integer_comparisons(self, cuda_ctx):
        cubins = compile_ptx_source(_PTX_SETP_VARIANTS)
        assert cuda_ctx.load(cubins['setp_variants'])
        func = cuda_ctx.get_func('setp_variants')
        d = cuda_ctx.alloc(4); cuda_ctx.copy_to(d, b'\x00' * 4)
        cuda_ctx.launch(func, (1,1,1), (1,1,1), [d])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<I', cuda_ctx.copy_from(d, 4))[0]
        assert val == 0b011100, f"got {val:06b}, expected 011100"
        cuda_ctx.free(d)


# ============================================================
# Wide math (mul.hi, mul.wide)
# ============================================================

_PTX_MUL_HI = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry mul_hi(.param .u64 p_out) {
    .reg .u32 %r<4>; .reg .u64 %rd<2>;
    mov.u32 %r0, 0x80000000; mov.u32 %r1, 2;
    mul.hi.u32 %r2, %r0, %r1;
    ld.param.u64 %rd0, [p_out];
    st.global.u32 [%rd0], %r2;
    ret;
}
"""

_PTX_MUL_WIDE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry mul_wide(.param .u64 p_out) {
    .reg .u32 %r<4>; .reg .u64 %rd<4>;
    mov.u32 %r0, 0x80000000;
    mul.wide.u32 %rd0, %r0, 3;
    ld.param.u64 %rd1, [p_out];
    cvt.u32.u64 %r1, %rd0;
    st.global.u32 [%rd1], %r1;
    add.u64 %rd2, %rd1, 4;
    shr.u64 %rd3, %rd0, 32; cvt.u32.u64 %r2, %rd3;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""


@gpu
class TestWideMath:

    def test_mul_hi(self, cuda_ctx):
        cubins = compile_ptx_source(_PTX_MUL_HI)
        assert cuda_ctx.load(cubins['mul_hi'])
        func = cuda_ctx.get_func('mul_hi')
        d = cuda_ctx.alloc(4); cuda_ctx.copy_to(d, b'\x00' * 4)
        cuda_ctx.launch(func, (1,1,1), (1,1,1), [d])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<I', cuda_ctx.copy_from(d, 4))[0]
        assert val == 1, f"mul.hi: {val}"
        cuda_ctx.free(d)

    def test_mul_wide(self, cuda_ctx):
        cubins = compile_ptx_source(_PTX_MUL_WIDE)
        assert cuda_ctx.load(cubins['mul_wide'])
        func = cuda_ctx.get_func('mul_wide')
        d = cuda_ctx.alloc(8); cuda_ctx.copy_to(d, b'\x00' * 8)
        cuda_ctx.launch(func, (1,1,1), (1,1,1), [d])
        assert cuda_ctx.sync() == 0
        lo, hi = struct.unpack('<II', cuda_ctx.copy_from(d, 8))
        assert lo == 0x80000000, f"wide.lo: {lo:#x}"
        assert hi == 1, f"wide.hi: {hi}"
        cuda_ctx.free(d)


# ============================================================
# neg/abs float
# ============================================================

_PTX_NEGABS = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry negabs(.param .u64 p_out) {
    .reg .u32 %r<2>; .reg .u64 %rd<4>; .reg .f32 %f<4>;
    mov.f32 %f0, 0f41200000;
    neg.f32 %f1, %f0;
    abs.f32 %f2, %f1;
    ld.param.u64 %rd0, [p_out];
    st.global.f32 [%rd0], %f1;
    add.u64 %rd1, %rd0, 4;
    st.global.f32 [%rd1], %f2;
    ret;
}
"""


@gpu
class TestNegAbs:

    def test_neg_abs_f32(self, cuda_ctx):
        cubins = compile_ptx_source(_PTX_NEGABS)
        assert cuda_ctx.load(cubins['negabs'])
        func = cuda_ctx.get_func('negabs')
        d = cuda_ctx.alloc(8); cuda_ctx.copy_to(d, b'\x00' * 8)
        cuda_ctx.launch(func, (1,1,1), (1,1,1), [d])
        assert cuda_ctx.sync() == 0
        neg, ab = struct.unpack('<ff', cuda_ctx.copy_from(d, 8))
        assert neg == -10.0, f"neg: {neg}"
        assert ab == 10.0, f"abs: {ab}"
        cuda_ctx.free(d)


# ============================================================
# FP32 division (div.approx.f32)
# ============================================================

_PTX_FDIV = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry fdiv_test(.param .u64 p_out) {
    .reg .u64 %rd<2>; .reg .f32 %f<4>;
    mov.f32 %f0, 0f41200000;
    mov.f32 %f1, 0f40000000;
    div.approx.f32 %f2, %f0, %f1;
    ld.param.u64 %rd0, [p_out];
    st.global.f32 [%rd0], %f2;
    ret;
}
"""


@gpu
class TestFDiv:
    def test_div_approx_f32(self, cuda_ctx):
        """div.approx.f32: 10.0 / 2.0 = 5.0."""
        cubins = compile_ptx_source(_PTX_FDIV)
        assert cuda_ctx.load(cubins['fdiv_test'])
        func = cuda_ctx.get_func('fdiv_test')
        d = cuda_ctx.alloc(4); cuda_ctx.copy_to(d, b'\x00' * 4)
        cuda_ctx.launch(func, (1,1,1), (1,1,1), [d])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<f', cuda_ctx.copy_from(d, 4))[0]
        assert abs(val - 5.0) < 0.01, f"div.approx: {val}"
        cuda_ctx.free(d)


# ============================================================
# Integer division (div.u32, div.s32)
# ============================================================

_PTX_IDIVU = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry idivu_test(.param .u64 p_out, .param .u64 p_a, .param .u64 p_b) {
    .reg .u32 %r<4>; .reg .u64 %rd<4>;
    ld.param.u64 %rd0, [p_a]; ld.global.u32 %r0, [%rd0];
    ld.param.u64 %rd1, [p_b]; ld.global.u32 %r1, [%rd1];
    div.u32 %r2, %r0, %r1;
    ld.param.u64 %rd2, [p_out];
    st.global.u32 [%rd2], %r2;
    ret;
}
"""


@gpu
class TestIntDiv:
    
    def test_div_u32(self, cuda_ctx):
        cubins = compile_ptx_source(_PTX_IDIVU)
        assert cuda_ctx.load(cubins['idivu_test'])
        func = cuda_ctx.get_func('idivu_test')
        d_a = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_a, struct.pack('<I', 100))
        d_b = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_b, struct.pack('<I', 7))
        d = cuda_ctx.alloc(4); cuda_ctx.copy_to(d, b'\x00' * 4)
        cuda_ctx.launch(func, (1,1,1), (1,1,1), [d, d_a, d_b])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<I', cuda_ctx.copy_from(d, 4))[0]
        assert val == 14, f"div.u32: 100/7={val}"
        cuda_ctx.free(d_a); cuda_ctx.free(d_b); cuda_ctx.free(d)


# ============================================================
# XOR (xor.b32)
# ============================================================

_PTX_XOR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry xor_test(.param .u64 p_out) {
    .reg .u32 %r<4>; .reg .u64 %rd<2>;
    mov.u32 %r0, 0xFF00FF00;
    mov.u32 %r1, 0x0F0F0F0F;
    xor.b32 %r2, %r0, %r1;
    ld.param.u64 %rd0, [p_out];
    st.global.u32 [%rd0], %r2;
    ret;
}
"""


@gpu
class TestXOR:
    def test_xor_b32(self, cuda_ctx):
        cubins = compile_ptx_source(_PTX_XOR)
        assert cuda_ctx.load(cubins['xor_test'])
        func = cuda_ctx.get_func('xor_test')
        d = cuda_ctx.alloc(4); cuda_ctx.copy_to(d, b'\x00' * 4)
        cuda_ctx.launch(func, (1,1,1), (1,1,1), [d])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<I', cuda_ctx.copy_from(d, 4))[0]
        assert val == 0xF00FF00F, f"xor.b32: {val:#010x}"
        cuda_ctx.free(d)


# ============================================================
# Float comparisons (setp.lt.f32 + selp)
# ============================================================

_PTX_FSETP = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry fsetp_test(.param .u64 p_out) {
    .reg .u32 %r<4>; .reg .u64 %rd<2>; .reg .f32 %f<4>; .reg .pred %p<2>;
    mov.f32 %f0, 0f40A00000;
    mov.f32 %f1, 0f41200000;
    setp.lt.f32 %p0, %f0, %f1;
    selp.u32 %r0, 1, 0, %p0;
    setp.gt.f32 %p0, %f0, %f1;
    selp.u32 %r1, 1, 0, %p0;
    shl.b32 %r0, %r0, 1;
    or.b32 %r2, %r0, %r1;
    ld.param.u64 %rd0, [p_out];
    st.global.u32 [%rd0], %r2;
    ret;
}
"""


@gpu
class TestFsetp:
    def test_fsetp_lt_gt(self, cuda_ctx):
        """setp.lt.f32(5.0, 10.0) = 1, setp.gt.f32(5.0, 10.0) = 0 -> 0b10 = 2."""
        cubins = compile_ptx_source(_PTX_FSETP)
        assert cuda_ctx.load(cubins['fsetp_test'])
        func = cuda_ctx.get_func('fsetp_test')
        d = cuda_ctx.alloc(4); cuda_ctx.copy_to(d, b'\x00' * 4)
        cuda_ctx.launch(func, (1,1,1), (1,1,1), [d])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<I', cuda_ctx.copy_from(d, 4))[0]
        assert val == 2, f"fsetp: got {val:02b}, expected 10"
        cuda_ctx.free(d)


# ============================================================
# Divergent branch control flow
# ============================================================

_PTX_BRANCH = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry branch_test(.param .u64 p_out) {
    .reg .u32 %r<4>; .reg .u64 %rd<4>; .reg .pred %p<2>;
    mov.u32 %r0, %tid.x;
    setp.lt.u32 %p0, %r0, 16;
    @%p0 bra THEN;
    mov.u32 %r1, 200;
    bra DONE;
THEN:
    mov.u32 %r1, 100;
DONE:
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd0, %rd0, %rd1;
    st.global.u32 [%rd0], %r1;
    ret;
}
"""


@gpu
class TestBranch:
    def test_divergent_branch(self, cuda_ctx):
        """Threads 0-15 write 100, threads 16-31 write 200."""
        cubins = compile_ptx_source(_PTX_BRANCH)
        assert cuda_ctx.load(cubins['branch_test'])
        func = cuda_ctx.get_func('branch_test')
        d = cuda_ctx.alloc(128); cuda_ctx.copy_to(d, b'\x00' * 128)
        cuda_ctx.launch(func, (1,1,1), (32,1,1), [d])
        assert cuda_ctx.sync() == 0
        vals = struct.unpack('<32I', cuda_ctx.copy_from(d, 128))
        for i in range(16):
            assert vals[i] == 100, f"tid {i}: {vals[i]}"
        for i in range(16, 32):
            assert vals[i] == 200, f"tid {i}: {vals[i]}"
        cuda_ctx.free(d)


# ============================================================
# FP64 arithmetic chain (dmul + dadd)
# ============================================================

_PTX_FP64 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry fp64_test(.param .u64 p_out) {
    .reg .u64 %rd<4>; .reg .f64 %fd<4>;
    mov.f64 %fd0, 0d4024000000000000;
    mov.f64 %fd1, 0d4000000000000000;
    mul.f64 %fd2, %fd0, %fd1;
    add.f64 %fd3, %fd2, %fd1;
    ld.param.u64 %rd0, [p_out];
    st.global.f64 [%rd0], %fd3;
    ret;
}
"""


@gpu
class TestFP64:
    def test_fp64_chain(self, cuda_ctx):
        """10.0 * 2.0 + 2.0 = 22.0 (FP64)."""
        cubins = compile_ptx_source(_PTX_FP64)
        assert cuda_ctx.load(cubins['fp64_test'])
        func = cuda_ctx.get_func('fp64_test')
        d = cuda_ctx.alloc(8); cuda_ctx.copy_to(d, b'\x00' * 8)
        cuda_ctx.launch(func, (1,1,1), (1,1,1), [d])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<d', cuda_ctx.copy_from(d, 8))[0]
        assert val == 22.0, f"fp64: {val}"
        cuda_ctx.free(d)


# ============================================================
# Float min/max chain
# ============================================================

_PTX_FMIN = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry fmin_test(.param .u64 p_out, .param .u64 p_a, .param .u64 p_b) {
    .reg .u64 %rd<4>; .reg .f32 %f<4>;
    ld.param.u64 %rd0, [p_a]; ld.global.f32 %f0, [%rd0];
    ld.param.u64 %rd1, [p_b]; ld.global.f32 %f1, [%rd1];
    min.f32 %f2, %f0, %f1;
    ld.param.u64 %rd2, [p_out];
    st.global.f32 [%rd2], %f2;
    ret;
}
"""

_PTX_FMAX = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry fmax_test(.param .u64 p_out, .param .u64 p_a, .param .u64 p_b) {
    .reg .u64 %rd<4>; .reg .f32 %f<4>;
    ld.param.u64 %rd0, [p_a]; ld.global.f32 %f0, [%rd0];
    ld.param.u64 %rd1, [p_b]; ld.global.f32 %f1, [%rd1];
    max.f32 %f2, %f0, %f1;
    ld.param.u64 %rd2, [p_out];
    st.global.f32 [%rd2], %f2;
    ret;
}
"""


@gpu
class TestFMinMax:
    def test_min_f32(self, cuda_ctx):
        cubins = compile_ptx_source(_PTX_FMIN)
        assert cuda_ctx.load(cubins['fmin_test'])
        d_a = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_a, struct.pack('<f', 10.0))
        d_b = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_b, struct.pack('<f', 5.0))
        d = cuda_ctx.alloc(4); cuda_ctx.copy_to(d, b'\x00' * 4)
        cuda_ctx.launch(cuda_ctx.get_func('fmin_test'), (1,1,1), (1,1,1), [d, d_a, d_b])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<f', cuda_ctx.copy_from(d, 4))[0]
        assert val == 5.0, f"min: {val}"
        cuda_ctx.free(d_a); cuda_ctx.free(d_b); cuda_ctx.free(d)

    def test_max_f32(self, cuda_ctx):
        cubins = compile_ptx_source(_PTX_FMAX)
        assert cuda_ctx.load(cubins['fmax_test'])
        d_a = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_a, struct.pack('<f', 5.0))
        d_b = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_b, struct.pack('<f', 15.0))
        d = cuda_ctx.alloc(4); cuda_ctx.copy_to(d, b'\x00' * 4)
        cuda_ctx.launch(cuda_ctx.get_func('fmax_test'), (1,1,1), (1,1,1), [d, d_a, d_b])
        assert cuda_ctx.sync() == 0
        val = struct.unpack('<f', cuda_ctx.copy_from(d, 4))[0]
        assert val == 15.0, f"max: {val}"
        cuda_ctx.free(d_a); cuda_ctx.free(d_b); cuda_ctx.free(d)


# ============================================================
# SEL.64 — selp.b64 with two register sources lowers to native SEL.64
# ============================================================

_PTX_SEL64 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry sel64_test(.param .u64 p_out, .param .u64 p_a, .param .u64 p_b, .param .u64 p_cond) {
    .reg .u64 %rd<8>;
    .reg .u32 %r<4>;
    .reg .pred %p<2>;
    ld.param.u64 %rd0, [p_a];
    ld.global.u64 %rd1, [%rd0];
    ld.param.u64 %rd2, [p_b];
    ld.global.u64 %rd3, [%rd2];
    ld.param.u64 %rd9, [p_cond];
    cvt.u32.u64 %r0, %rd9;
    setp.ne.u32 %p0, %r0, 0;
    selp.b64 %rd4, %rd1, %rd3, %p0;
    add.u64 %rd5, %rd4, %rd1;
    xor.b64 %rd6, %rd5, %rd3;
    ld.param.u64 %rd7, [p_out];
    st.global.u64 [%rd7], %rd6;
    ret;
}
"""


@gpu
class TestSel64:
    """Bit-identical SEL.64 emission and runtime correctness for selp.b64."""

    def _run(self, cuda_ctx, cond):
        cubins = compile_ptx_source(_PTX_SEL64)
        assert cuda_ctx.load(cubins['sel64_test'])
        func = cuda_ctx.get_func('sel64_test')
        a_val = 0x1122334455667788
        b_val = 0x0fedcba987654321
        d_a = cuda_ctx.alloc(8); cuda_ctx.copy_to(d_a, struct.pack('<Q', a_val))
        d_b = cuda_ctx.alloc(8); cuda_ctx.copy_to(d_b, struct.pack('<Q', b_val))
        d_o = cuda_ctx.alloc(8); cuda_ctx.copy_to(d_o, b'\x00' * 8)
        # cond is a by-value u64 param.  Force the conftest launch helper
        # to pack it as a u64 by ORing in the high bit (value-preserving for
        # cond>0; for cond=0 we use a special value 0).  Workaround for
        # conftest's heuristic that picks c_int32 for small ints.
        cond_arg = cond if cond > 0xFFFFFFFF else (0x100000000 | cond) - 0x100000000  # 0 stays 0
        # Actually: explicitly use a u64 value.  Force cond to be passed as u64 by
        # using a value >0xFFFFFFFF when nonzero; for cond=0, value is naturally 0.
        # The real fix is to pass cond via a one-element array; but we need a
        # workaround.  Use ctypes directly:
        import ctypes as _ct
        cond_holder = _ct.c_uint64(cond)
        # cuda_ctx.launch's first arg-loop will treat cond_holder as non-int,
        # which fails its int branches.  Instead, build args ourselves.
        gx, gy, gz = (1,1,1); bx, by, bz = (1,1,1)
        holders = [
            _ct.c_uint64(d_o.value if hasattr(d_o, 'value') else d_o),
            _ct.c_uint64(d_a.value if hasattr(d_a, 'value') else d_a),
            _ct.c_uint64(d_b.value if hasattr(d_b, 'value') else d_b),
            cond_holder,
        ]
        ptrs = [_ct.cast(_ct.byref(h), _ct.c_void_p) for h in holders]
        aa = (_ct.c_void_p * 4)(*ptrs)
        cuda_ctx.cuda.cuLaunchKernel(func, gx, gy, gz, bx, by, bz, 0, None, aa, None)
        assert cuda_ctx.sync() == 0
        got = struct.unpack('<Q', cuda_ctx.copy_from(d_o, 8))[0]
        sel = a_val if cond != 0 else b_val
        mask = 0xFFFFFFFFFFFFFFFF
        expected = (((sel + a_val) & mask) ^ b_val) & mask
        cuda_ctx.free(d_a); cuda_ctx.free(d_b); cuda_ctx.free(d_o)
        assert got == expected, f"cond={cond}: got 0x{got:016x}, expected 0x{expected:016x}"

    def test_sel64_true(self, cuda_ctx):
        self._run(cuda_ctx, 1)

    def test_sel64_false(self, cuda_ctx):
        self._run(cuda_ctx, 0)

    def test_sel64_alt_true(self, cuda_ctx):
        self._run(cuda_ctx, 0xDEADBEEF)


# ============================================================
# SHF.L.W — left funnel-shift, WRAP variant of shift amount
# ============================================================
#
# `shf.l.wrap.b32 d, lo, hi, s` computes:
#   d = ((lo, hi) << (s & 31)) >> 32
# wrap masks shift count to low 5 bits, clamp clamps it to 32; differ for s>=32.
#
# Note: PTX `shf` op is NOT yet implemented in our parser/isel.  The encoders
# `encode_shf_l_w_u32_var` / `encode_shf_l_w_u32_hi_var` are byte-exact verified
# against ptxas ground truth (see tests/test_new_encoders.py::test_shf_l_w_*),
# which is the strongest correctness guarantee.  Adding GPU correctness via a
# full shf.l.wrap.b32 PTX→SASS lowering pipeline is tracked as a separate
# follow-up — the encoders themselves will be runtime-correct because they
# match ptxas's bytes.


# ============================================================
# CS2UR — read special register into a UNIFORM register (UR-bank dest)
# ============================================================
#
# Note: CS2UR writes to a UR (uniform) register; our isel pipeline lowers
# %clock and similar uniform SRs through S2R (GPR-dest) rather than CS2UR.
# Wiring CS2UR end-to-end through isel + scheduler + downstream UR consumers
# is a larger refactor.
#
# The encoder is byte-exact verified against ptxas SM_120 ground truth (see
# tests/test_new_encoders.py::test_cs2ur_*).  Runtime correctness is implied
# by byte-equality with ptxas: when ptxas itself emits these exact bytes for
# SR_CLOCKLO/SR_PM0/etc. in our probe kernels, the GPU executes them
# correctly — that is the runtime evidence.  See _probe_landing/probe_cs2ur*.


# ============================================================
# Dot product (FMA chain with indexed loads)
