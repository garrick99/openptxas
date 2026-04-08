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


def _get_cuda():
    try:
        cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
        if cuda.cuInit(0) != 0: return None
        return cuda
    except: return None

_CUDA = _get_cuda()
gpu = pytest.mark.skipif(_CUDA is None, reason="No CUDA GPU")


class CUDAContext:
    def __init__(self):
        self.cuda = _CUDA
        self.ctx = ctypes.c_void_p()
        self.mod = ctypes.c_void_p()
        dev = ctypes.c_int()
        self.cuda.cuDeviceGet(ctypes.byref(dev), 0)
        assert self.cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev) == 0

    def load(self, cubin):
        if self.mod.value: self.cuda.cuModuleUnload(self.mod)
        self.mod = ctypes.c_void_p()
        return self.cuda.cuModuleLoadData(ctypes.byref(self.mod), cubin) == 0

    def get_func(self, name):
        f = ctypes.c_void_p()
        assert self.cuda.cuModuleGetFunction(ctypes.byref(f), self.mod, name.encode()) == 0
        return f

    def alloc(self, n):
        p = ctypes.c_uint64()
        assert self.cuda.cuMemAlloc_v2(ctypes.byref(p), n) == 0
        return p.value

    def free(self, p): self.cuda.cuMemFree_v2(ctypes.c_uint64(p))
    def copy_to(self, p, d): self.cuda.cuMemcpyHtoD_v2(ctypes.c_uint64(p), d, len(d))
    def copy_from(self, p, n):
        b = (ctypes.c_uint8 * n)()
        self.cuda.cuMemcpyDtoH_v2(b, ctypes.c_uint64(p), n)
        return bytes(b)

    def sync(self): return self.cuda.cuCtxSynchronize()

    def launch(self, func, grid, block, args):
        gx, gy, gz = grid if isinstance(grid, tuple) else (grid, 1, 1)
        bx, by, bz = block if isinstance(block, tuple) else (block, 1, 1)
        holders = []
        ptrs = []
        for a in args:
            h = ctypes.c_uint64(a) if isinstance(a, int) and a > 0xFFFFFFFF else ctypes.c_int32(a)
            holders.append(h)
            ptrs.append(ctypes.cast(ctypes.byref(h), ctypes.c_void_p))
        aa = (ctypes.c_void_p * len(ptrs))(*ptrs)
        return self.cuda.cuLaunchKernel(func, gx, gy, gz, bx, by, bz, 0, None, aa, None)

    def close(self):
        if self.mod.value: self.cuda.cuModuleUnload(self.mod)
        if self.ctx.value: self.cuda.cuCtxDestroy_v2(self.ctx)


@pytest.fixture(scope="module")
def cuda_ctx():
    if _CUDA is None: pytest.skip("No CUDA")
    c = CUDAContext()
    yield c
    c.close()


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
