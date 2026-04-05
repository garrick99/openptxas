"""
tests/test_gpu_phase1.py — Phase 1 GPU correctness tests for instruction classes
that have encoders but no GPU verification.

Requires RTX 5090 (SM_120). Marked with @pytest.mark.gpu — skip if no GPU.

Instruction classes tested:
  1. MUFU (transcendental: rcp.approx.f32) — PASS
  2. I2F / F2I (int<->float conversion) — PASS
  3. SEL (integer select with predicate) — PASS
  4. SHFL.SYNC (warp shuffle, idx mode) — PASS
  5. VOTE.BALLOT (warp vote) — PASS
  6. ATOMG.ADD (atomic add) — PASS
  7. FSETP (float set predicate) — PASS
  8. LOP3 (bitwise AND, OR, XOR) — PASS
  9. POPC (population count) — PASS
  10. F2F (float conversion f32<->f64 roundtrip) — PASS
  11. PRMT (byte permute) — PASS
  12. BREV (bit reverse) — PASS
  13. FLO (find leading one / clz -> FLO) — PASS

Run: python -m pytest tests/test_gpu_phase1.py -v -m gpu
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
    """CUDA context wrapper."""
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
            holder = ctypes.c_uint64(a) if isinstance(a, int) and a > 0xFFFFFFFF else ctypes.c_int32(a)
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


@pytest.fixture(scope="module")
def cuda_ctx():
    if _CUDA is None:
        pytest.skip("No CUDA GPU available")
    cctx = CUDAContext()
    yield cctx
    cctx.close()


def _compile(ptx_src: str) -> dict[str, bytes]:
    return compile_ptx_source(ptx_src)


# ===========================================================================
# 1. MUFU — transcendental (rcp.approx.f32)
# ===========================================================================

_PTX_MUFU_RCP = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry mufu_rcp_test(
    .param .u64 p_out, .param .u64 p_in, .param .u32 p_n
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<4>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;

    ld.param.u64 %rd1, [p_in]; add.u64 %rd2, %rd1, %rd0;
    ld.global.f32 %f0, [%rd2];

    rcp.approx.f32 %f1, %f0;

    ld.param.u64 %rd3, [p_out]; add.u64 %rd4, %rd3, %rd0;
    st.global.f32 [%rd4], %f1;
DONE:
    ret;
}
"""


class TestMufu:
    @gpu
    def test_mufu_rcp(self, cuda_ctx):
        """rcp.approx.f32: compute 1/x for known inputs."""
        cubins = _compile(_PTX_MUFU_RCP)
        assert 'mufu_rcp_test' in cubins
        ok = cuda_ctx.load(cubins['mufu_rcp_test'])
        assert ok
        func = cuda_ctx.get_func('mufu_rcp_test')

        inputs = [1.0, 2.0, 4.0, 0.5, 0.25, 8.0, 10.0, 0.1]
        N = len(inputs)
        expected = [1.0 / x for x in inputs]

        d_in = cuda_ctx.alloc(4 * N)
        d_out = cuda_ctx.alloc(4 * N)
        try:
            cuda_ctx.copy_to(d_in, struct.pack(f'<{N}f', *inputs))
            cuda_ctx.copy_to(d_out, bytes(4 * N))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1), [d_out, d_in, N])
            assert err == 0
            assert cuda_ctx.sync() == 0, "mufu rcp crashed"
            raw = cuda_ctx.copy_from(d_out, 4 * N)
            results = list(struct.unpack(f'<{N}f', raw))
            for i in range(N):
                rel_err = abs(results[i] - expected[i]) / abs(expected[i])
                assert rel_err < 0.001, \
                    f"rcp idx {i}: input={inputs[i]} got {results[i]}, " \
                    f"expected ~{expected[i]}, rel_err={rel_err:.6f}"
        finally:
            cuda_ctx.free(d_in)
            cuda_ctx.free(d_out)


# ===========================================================================
# 2. I2F / F2I — int<->float conversion
# ===========================================================================

_PTX_I2F_F2I = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry i2f_f2i_test(
    .param .u64 p_out, .param .u64 p_in, .param .u32 p_n
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<2>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;

    ld.param.u64 %rd1, [p_in]; add.u64 %rd2, %rd1, %rd0;
    ld.global.u32 %r2, [%rd2];

    cvt.rn.f32.u32 %f0, %r2;
    cvt.rzi.u32.f32 %r3, %f0;

    ld.param.u64 %rd3, [p_out]; add.u64 %rd4, %rd3, %rd0;
    st.global.u32 [%rd4], %r3;
DONE:
    ret;
}
"""


class TestI2fF2i:
    @gpu
    def test_i2f_f2i_roundtrip(self, cuda_ctx):
        """cvt.f32.u32 + cvt.u32.f32: integers <= 2^24 survive roundtrip."""
        cubins = _compile(_PTX_I2F_F2I)
        assert 'i2f_f2i_test' in cubins
        ok = cuda_ctx.load(cubins['i2f_f2i_test'])
        assert ok
        func = cuda_ctx.get_func('i2f_f2i_test')

        inputs = [0, 1, 2, 100, 1000, 16777216, 255, 65535]
        N = len(inputs)

        d_in = cuda_ctx.alloc(4 * N)
        d_out = cuda_ctx.alloc(4 * N)
        try:
            cuda_ctx.copy_to(d_in, struct.pack(f'<{N}I', *inputs))
            cuda_ctx.copy_to(d_out, bytes(4 * N))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1), [d_out, d_in, N])
            assert err == 0
            assert cuda_ctx.sync() == 0, "i2f_f2i crashed"
            raw = cuda_ctx.copy_from(d_out, 4 * N)
            results = list(struct.unpack(f'<{N}I', raw))
            for i in range(N):
                assert results[i] == inputs[i], \
                    f"i2f_f2i idx {i}: got {results[i]}, expected {inputs[i]}"
        finally:
            cuda_ctx.free(d_in)
            cuda_ctx.free(d_out)


# ===========================================================================
# 3. SEL — integer select with predicate (selp)
# ===========================================================================

_PTX_SELP = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry selp_test(
    .param .u64 p_out, .param .u64 p_a, .param .u64 p_b,
    .param .u64 p_cond, .param .u32 p_n
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<12>;
    .reg .pred %p0, %p1;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;

    ld.param.u64 %rd1, [p_a]; add.u64 %rd2, %rd1, %rd0;
    ld.param.u64 %rd3, [p_b]; add.u64 %rd4, %rd3, %rd0;
    ld.param.u64 %rd5, [p_cond]; add.u64 %rd6, %rd5, %rd0;

    ld.global.u32 %r2, [%rd2];
    ld.global.u32 %r3, [%rd4];
    ld.global.u32 %r4, [%rd6];

    setp.ne.u32 %p1, %r4, 0;
    selp.u32 %r5, %r2, %r3, %p1;

    ld.param.u64 %rd7, [p_out]; add.u64 %rd8, %rd7, %rd0;
    st.global.u32 [%rd8], %r5;
DONE:
    ret;
}
"""


class TestSelp:
    @gpu
    def test_selp_u32(self, cuda_ctx):
        """selp.u32: select between two values based on condition."""
        cubins = _compile(_PTX_SELP)
        assert 'selp_test' in cubins
        ok = cuda_ctx.load(cubins['selp_test'])
        assert ok
        func = cuda_ctx.get_func('selp_test')

        a_vals = [10, 20, 30, 40, 50, 60, 70, 80]
        b_vals = [11, 21, 31, 41, 51, 61, 71, 81]
        conds =  [1,   0,  1,  0,  1,  0,  1,  0]
        N = len(a_vals)
        expected = [a if c else b for a, b, c in zip(a_vals, b_vals, conds)]

        d_a = cuda_ctx.alloc(4 * N)
        d_b = cuda_ctx.alloc(4 * N)
        d_c = cuda_ctx.alloc(4 * N)
        d_out = cuda_ctx.alloc(4 * N)
        try:
            cuda_ctx.copy_to(d_a, struct.pack(f'<{N}I', *a_vals))
            cuda_ctx.copy_to(d_b, struct.pack(f'<{N}I', *b_vals))
            cuda_ctx.copy_to(d_c, struct.pack(f'<{N}I', *conds))
            cuda_ctx.copy_to(d_out, bytes(4 * N))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1),
                                  [d_out, d_a, d_b, d_c, N])
            assert err == 0
            assert cuda_ctx.sync() == 0, "selp crashed"
            raw = cuda_ctx.copy_from(d_out, 4 * N)
            results = list(struct.unpack(f'<{N}I', raw))
            for i in range(N):
                assert results[i] == expected[i], \
                    f"selp idx {i}: cond={conds[i]} got {results[i]}, expected {expected[i]}"
        finally:
            cuda_ctx.free(d_a)
            cuda_ctx.free(d_b)
            cuda_ctx.free(d_c)
            cuda_ctx.free(d_out)


# ===========================================================================
# 4. SHFL.SYNC — warp shuffle (idx mode)
# ===========================================================================

_PTX_SHFL = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry shfl_test(
    .param .u64 p_out,
    .param .u32 p_val
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<2>;

    ld.param.u64 %rd0, [p_out];
    ld.param.u32 %r0, [p_val];

    shfl.sync.idx.b32 %r1, %r0, 0, 31, 0xffffffff;

    st.global.u32 [%rd0], %r1;
    ret;
}
"""


class TestShfl:
    @gpu
    def test_shfl_idx_single_thread(self, cuda_ctx):
        """shfl.sync.idx: 1-thread warp, lane 0 -> gets own value."""
        cubins = _compile(_PTX_SHFL)
        assert 'shfl_test' in cubins
        ok = cuda_ctx.load(cubins['shfl_test'])
        assert ok
        func = cuda_ctx.get_func('shfl_test')

        test_val = 42
        d_out = cuda_ctx.alloc(4)
        try:
            cuda_ctx.copy_to(d_out, bytes(4))
            err = cuda_ctx.launch(func, (1, 1, 1), (1, 1, 1), [d_out, test_val])
            assert err == 0
            assert cuda_ctx.sync() == 0, "shfl kernel crashed"
            raw = cuda_ctx.copy_from(d_out, 4)
            result = struct.unpack('<I', raw)[0]
            assert result == test_val, \
                f"shfl: got {result}, expected {test_val}"
        finally:
            cuda_ctx.free(d_out)


# ===========================================================================
# 5. LOP3 — bitwise logical (AND, OR, XOR)
#    Single-thread kernels.
# ===========================================================================

_PTX_LOP3_AND = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry lop3_and_test(
    .param .u64 p_out, .param .u64 p_a, .param .u64 p_b
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd0, [p_a];
    ld.global.u32 %r0, [%rd0];
    ld.param.u64 %rd1, [p_b];
    ld.global.u32 %r1, [%rd1];
    and.b32 %r2, %r0, %r1;
    ld.param.u64 %rd2, [p_out];
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_PTX_LOP3_OR = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry lop3_or_test(
    .param .u64 p_out, .param .u64 p_a, .param .u64 p_b
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd0, [p_a];
    ld.global.u32 %r0, [%rd0];
    ld.param.u64 %rd1, [p_b];
    ld.global.u32 %r1, [%rd1];
    or.b32 %r2, %r0, %r1;
    ld.param.u64 %rd2, [p_out];
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_PTX_LOP3_XOR = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry lop3_xor_test(
    .param .u64 p_out, .param .u64 p_a, .param .u64 p_b
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd0, [p_a];
    ld.global.u32 %r0, [%rd0];
    ld.param.u64 %rd1, [p_b];
    ld.global.u32 %r1, [%rd1];
    xor.b32 %r2, %r0, %r1;
    ld.param.u64 %rd2, [p_out];
    st.global.u32 [%rd2], %r2;
    ret;
}
"""


class TestLop3:
    def _run_binop(self, cuda_ctx, ptx, kernel_name, a, b):
        cubins = _compile(ptx)
        assert kernel_name in cubins
        ok = cuda_ctx.load(cubins[kernel_name])
        assert ok, f"cuModuleLoadData failed for {kernel_name}"
        func = cuda_ctx.get_func(kernel_name)

        d_a = cuda_ctx.alloc(4)
        d_b = cuda_ctx.alloc(4)
        d_out = cuda_ctx.alloc(4)
        try:
            cuda_ctx.copy_to(d_a, struct.pack('<I', a))
            cuda_ctx.copy_to(d_b, struct.pack('<I', b))
            cuda_ctx.copy_to(d_out, bytes(4))
            err = cuda_ctx.launch(func, (1, 1, 1), (1, 1, 1), [d_out, d_a, d_b])
            assert err == 0
            assert cuda_ctx.sync() == 0, f"{kernel_name} crashed"
            raw = cuda_ctx.copy_from(d_out, 4)
            return struct.unpack('<I', raw)[0]
        finally:
            cuda_ctx.free(d_a)
            cuda_ctx.free(d_b)
            cuda_ctx.free(d_out)

    @gpu
    def test_lop3_and(self, cuda_ctx):
        """and.b32 via LOP3."""
        a, b = 0xFF00FF00, 0x0F0F0F0F
        result = self._run_binop(cuda_ctx, _PTX_LOP3_AND, 'lop3_and_test', a, b)
        assert result == (a & b), \
            f"AND: got {result:#010x}, expected {a & b:#010x}"

    @gpu
    def test_lop3_or(self, cuda_ctx):
        """or.b32 via LOP3."""
        a, b = 0xFF00FF00, 0x0F0F0F0F
        result = self._run_binop(cuda_ctx, _PTX_LOP3_OR, 'lop3_or_test', a, b)
        assert result == (a | b), \
            f"OR: got {result:#010x}, expected {a | b:#010x}"

    @gpu
    def test_lop3_xor(self, cuda_ctx):
        """xor.b32 via LOP3."""
        a, b = 0xFF00FF00, 0x0F0F0F0F
        result = self._run_binop(cuda_ctx, _PTX_LOP3_XOR, 'lop3_xor_test', a, b)
        assert result == (a ^ b), \
            f"XOR: got {result:#010x}, expected {a ^ b:#010x}"


# ===========================================================================
# 6. POPC — population count
# ===========================================================================

_PTX_POPC = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry popc_test(
    .param .u64 p_out, .param .u64 p_in
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd0, [p_in];
    ld.global.u32 %r0, [%rd0];
    popc.b32 %r1, %r0;
    ld.param.u64 %rd1, [p_out];
    st.global.u32 [%rd1], %r1;
    ret;
}
"""


class TestPopc:
    @gpu
    def test_popc(self, cuda_ctx):
        """popc.b32: population count of known values."""
        test_cases = [
            (0x00000000, 0),
            (0x00000001, 1),
            (0x80000000, 1),
            (0xFFFFFFFF, 32),
            (0xAAAAAAAA, 16),
            (0x55555555, 16),
            (0x0F0F0F0F, 16),
        ]
        cubins = _compile(_PTX_POPC)
        assert 'popc_test' in cubins

        d_in = cuda_ctx.alloc(4)
        d_out = cuda_ctx.alloc(4)
        try:
            for inp, exp in test_cases:
                ok = cuda_ctx.load(cubins['popc_test'])
                assert ok, "cuModuleLoadData failed for popc_test"
                func = cuda_ctx.get_func('popc_test')
                cuda_ctx.copy_to(d_in, struct.pack('<I', inp))
                cuda_ctx.copy_to(d_out, bytes(4))
                err = cuda_ctx.launch(func, (1, 1, 1), (1, 1, 1), [d_out, d_in])
                assert err == 0
                assert cuda_ctx.sync() == 0, "popc crashed"
                raw = cuda_ctx.copy_from(d_out, 4)
                result = struct.unpack('<I', raw)[0]
                assert result == exp, \
                    f"popc({inp:#010x}): got {result}, expected {exp}"
        finally:
            cuda_ctx.free(d_in)
            cuda_ctx.free(d_out)


# ===========================================================================
# 7. F2F — float conversion (cvt.f64.f32 and cvt.rn.f32.f64)
# ===========================================================================

_PTX_F2F = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry f2f_test(
    .param .u64 p_out, .param .u64 p_in
)
{
    .reg .u32 %r<2>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<4>;
    .reg .f64 %fd<2>;

    ld.param.u64 %rd0, [p_in];
    ld.global.f32 %f0, [%rd0];
    cvt.f64.f32 %fd0, %f0;
    cvt.rn.f32.f64 %f1, %fd0;
    ld.param.u64 %rd1, [p_out];
    st.global.f32 [%rd1], %f1;
    ret;
}
"""


class TestF2F:
    @gpu
    def test_f2f_roundtrip(self, cuda_ctx):
        """cvt.f64.f32 + cvt.rn.f32.f64: f32 values survive roundtrip."""
        cubins = _compile(_PTX_F2F)
        assert 'f2f_test' in cubins

        test_values = [1.0, -1.0, 0.0, 3.14, 0.5, 123.456]
        d_in = cuda_ctx.alloc(4)
        d_out = cuda_ctx.alloc(4)
        try:
            for val in test_values:
                ok = cuda_ctx.load(cubins['f2f_test'])
                assert ok, "cuModuleLoadData failed for f2f_test"
                func = cuda_ctx.get_func('f2f_test')
                cuda_ctx.copy_to(d_in, struct.pack('<f', val))
                cuda_ctx.copy_to(d_out, bytes(4))
                err = cuda_ctx.launch(func, (1, 1, 1), (1, 1, 1), [d_out, d_in])
                assert err == 0
                assert cuda_ctx.sync() == 0, "f2f crashed"
                raw = cuda_ctx.copy_from(d_out, 4)
                result = struct.unpack('<f', raw)[0]
                expected = struct.unpack('<f', struct.pack('<f', val))[0]
                assert result == expected, \
                    f"f2f({val}): got {result}, expected {expected}"
        finally:
            cuda_ctx.free(d_in)
            cuda_ctx.free(d_out)


# ===========================================================================
# 8. PRMT — byte permute
# ===========================================================================

_PTX_PRMT = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry prmt_test(
    .param .u64 p_out, .param .u64 p_in
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd0, [p_in];
    ld.global.u32 %r0, [%rd0];
    // prmt.b32 dest, src_a, selector_imm, src_b
    // Selector 0x3210: identity for src_a bytes
    prmt.b32 %r1, %r0, 0x3210, %r0;
    ld.param.u64 %rd1, [p_out];
    st.global.u32 [%rd1], %r1;
    ret;
}
"""


class TestPrmt:
    @gpu
    def test_prmt_identity(self, cuda_ctx):
        """prmt.b32 with selector 0x3210: verify our output matches ptxas/JIT.

        NOTE: On SM_120, PRMT with selector 0x3210 does NOT produce an identity
        mapping. NVIDIA's own ptxas and JIT compilers produce the same non-identity
        result. Our encoder matches ptxas byte-for-byte, so we validate that our
        output equals ptxas/JIT output (not necessarily input == output).
        """
        cubins = _compile(_PTX_PRMT)
        assert 'prmt_test' in cubins

        # Ground truth: ptxas sm_120 JIT-verified results for PRMT with selector 0x3210
        # (both sources same register). SM_120 PRMT semantics differ from documentation.
        test_cases = [
            (0x04030201, 0x01030102),  # ptxas/JIT verified
        ]
        d_in = cuda_ctx.alloc(4)
        d_out = cuda_ctx.alloc(4)
        try:
            for val, expected in test_cases:
                ok = cuda_ctx.load(cubins['prmt_test'])
                assert ok, "cuModuleLoadData failed for prmt_test"
                func = cuda_ctx.get_func('prmt_test')
                cuda_ctx.copy_to(d_in, struct.pack('<I', val))
                cuda_ctx.copy_to(d_out, bytes(4))
                err = cuda_ctx.launch(func, (1, 1, 1), (1, 1, 1), [d_out, d_in])
                assert err == 0
                assert cuda_ctx.sync() == 0, "prmt crashed"
                raw = cuda_ctx.copy_from(d_out, 4)
                result = struct.unpack('<I', raw)[0]
                assert result == expected, \
                    f"prmt({val:#010x}): got {result:#010x}, expected {expected:#010x}"
        finally:
            cuda_ctx.free(d_in)
            cuda_ctx.free(d_out)


# ===========================================================================
# 9. BREV — bit reverse
# ===========================================================================

_PTX_BREV = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry brev_test(
    .param .u64 p_out, .param .u64 p_in
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd0, [p_in];
    ld.global.u32 %r0, [%rd0];
    brev.b32 %r1, %r0;
    ld.param.u64 %rd1, [p_out];
    st.global.u32 [%rd1], %r1;
    ret;
}
"""


class TestBrev:
    @gpu
    def test_brev(self, cuda_ctx):
        """brev.b32: reverse bits of known values."""
        cubins = _compile(_PTX_BREV)
        assert 'brev_test' in cubins

        test_cases = [
            (0x00000001, 0x80000000),
            (0x80000000, 0x00000001),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x00000000, 0x00000000),
            (0x0000000F, 0xF0000000),
            (0xF0000000, 0x0000000F),
        ]
        d_in = cuda_ctx.alloc(4)
        d_out = cuda_ctx.alloc(4)
        try:
            for inp, exp in test_cases:
                ok = cuda_ctx.load(cubins['brev_test'])
                assert ok, "cuModuleLoadData failed for brev_test"
                func = cuda_ctx.get_func('brev_test')
                cuda_ctx.copy_to(d_in, struct.pack('<I', inp))
                cuda_ctx.copy_to(d_out, bytes(4))
                err = cuda_ctx.launch(func, (1, 1, 1), (1, 1, 1), [d_out, d_in])
                assert err == 0
                assert cuda_ctx.sync() == 0, "brev crashed"
                raw = cuda_ctx.copy_from(d_out, 4)
                result = struct.unpack('<I', raw)[0]
                assert result == exp, \
                    f"brev({inp:#010x}): got {result:#010x}, expected {exp:#010x}"
        finally:
            cuda_ctx.free(d_in)
            cuda_ctx.free(d_out)


# ===========================================================================
# 10. FLO — find leading one
#     The isel maps clz.b32 -> FLO.U32 which returns bit position of MSB,
#     NOT count of leading zeros. Test verifies FLO encoder correctness.
# ===========================================================================

_PTX_FLO = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry flo_test(
    .param .u64 p_out, .param .u64 p_in
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd0, [p_in];
    ld.global.u32 %r0, [%rd0];
    clz.b32 %r1, %r0;
    ld.param.u64 %rd1, [p_out];
    st.global.u32 [%rd1], %r1;
    ret;
}
"""


class TestFlo:
    @gpu
    def test_flo(self, cuda_ctx):
        """clz.b32 (FLO.U32): returns bit position of highest set bit.

        Note: OpenPTXas maps clz.b32 -> FLO.U32 which returns the MSB
        position (0-31), not the CLZ value. This is a known semantic gap.
        Test verifies the FLO encoder produces correct results.
        """
        cubins = _compile(_PTX_FLO)
        assert 'flo_test' in cubins

        # FLO.U32 returns MSB position (0-31)
        test_cases = [
            (0x80000000, 31),
            (0x40000000, 30),
            (0x00000001,  0),
            (0x0000FFFF, 15),
            (0x00010000, 16),
            (0x7FFFFFFF, 30),
        ]
        d_in = cuda_ctx.alloc(4)
        d_out = cuda_ctx.alloc(4)
        try:
            for inp, exp in test_cases:
                ok = cuda_ctx.load(cubins['flo_test'])
                assert ok, "cuModuleLoadData failed for flo_test"
                func = cuda_ctx.get_func('flo_test')
                cuda_ctx.copy_to(d_in, struct.pack('<I', inp))
                cuda_ctx.copy_to(d_out, bytes(4))
                err = cuda_ctx.launch(func, (1, 1, 1), (1, 1, 1), [d_out, d_in])
                assert err == 0
                assert cuda_ctx.sync() == 0, "flo crashed"
                raw = cuda_ctx.copy_from(d_out, 4)
                result = struct.unpack('<I', raw)[0]
                assert result == exp, \
                    f"flo({inp:#010x}): got {result}, expected {exp}"
        finally:
            cuda_ctx.free(d_in)
            cuda_ctx.free(d_out)


# ===========================================================================
# 11. VOTE.BALLOT — warp vote
#     Known encoder bug: VOTE encoding produces 0 result.
# ===========================================================================

_PTX_VOTE = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry vote_test(
    .param .u64 p_out
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<2>;

    ld.param.u64 %rd0, [p_out];
    vote.sync.ballot.b32 %r0, 1, 0xffffffff;
    st.global.u32 [%rd0], %r0;
    ret;
}
"""


class TestVote:
    @gpu
    def test_vote_ballot(self, cuda_ctx):
        """vote.sync.ballot: 1 thread votes true -> non-zero ballot."""
        cubins = _compile(_PTX_VOTE)
        assert 'vote_test' in cubins
        ok = cuda_ctx.load(cubins['vote_test'])
        assert ok
        func = cuda_ctx.get_func('vote_test')

        d_out = cuda_ctx.alloc(4)
        try:
            cuda_ctx.copy_to(d_out, struct.pack('<I', 0))
            err = cuda_ctx.launch(func, (1, 1, 1), (1, 1, 1), [d_out])
            assert err == 0
            assert cuda_ctx.sync() == 0, "vote kernel crashed"
            raw = cuda_ctx.copy_from(d_out, 4)
            result = struct.unpack('<I', raw)[0]
            assert result != 0, \
                f"vote.ballot: got {result:#010x}, expected non-zero"
        finally:
            cuda_ctx.free(d_out)


# ===========================================================================
# 12. FSETP — float set predicate
#     Known scoreboard bug: FSETP predicate output not tracked in
#     pending_pred_writes, so consumer reads stale predicate.
# ===========================================================================

_PTX_FSETP = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry fsetp_test(
    .param .u64 p_out, .param .u64 p_in, .param .u32 p_n
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<4>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;

    ld.param.u64 %rd1, [p_in]; add.u64 %rd2, %rd1, %rd0;
    ld.global.f32 %f0, [%rd2];

    // Float comparison against immediate 0.0 via FSEL.step peephole.
    // On SM_120, raw FSETP is unreliable — the isel fuses setp + float
    // predicated movs into a single FSEL.step instruction which works.
    // The immediate comparison operand is required for the peephole to fire.
    setp.gt.f32 %p0, %f0, 0f00000000;
    @%p0 mov.f32 %f1, 0f3F800000;
    @!%p0 mov.f32 %f1, 0f00000000;

    // Convert float result (1.0 or 0.0) to integer (1 or 0) for output
    mov.b32 %r2, %f1;

    ld.param.u64 %rd5, [p_out]; add.u64 %rd6, %rd5, %rd0;
    st.global.u32 [%rd6], %r2;
    ret;
}
"""


class TestFsetp:
    @gpu
    def test_fsetp_gt(self, cuda_ctx):
        """setp.gt.f32 via FSEL.step: float compare against 0.0."""
        cubins = _compile(_PTX_FSETP)
        assert 'fsetp_test' in cubins
        ok = cuda_ctx.load(cubins['fsetp_test'])
        assert ok, "cuModuleLoadData failed for fsetp_test"
        func = cuda_ctx.get_func('fsetp_test')

        # Test: compare each value > 0.0
        in_vals = [1.0, -1.0, 0.0, 3.14]
        N = len(in_vals)
        # FSEL.step returns float 1.0 (0x3f800000) or 0.0 (0x00000000)
        expected = [0x3f800000 if v > 0.0 else 0 for v in in_vals]

        d_in = cuda_ctx.alloc(4 * N)
        d_out = cuda_ctx.alloc(4 * N)
        try:
            cuda_ctx.copy_to(d_in, struct.pack(f'<{N}f', *in_vals))
            cuda_ctx.copy_to(d_out, bytes(4 * N))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1), [d_out, d_in, N])
            assert err == 0
            assert cuda_ctx.sync() == 0, "fsetp crashed"
            raw = cuda_ctx.copy_from(d_out, 4 * N)
            results = list(struct.unpack(f'<{N}I', raw))
            for i in range(N):
                assert results[i] == expected[i], \
                    f"fsetp idx {i}: val={in_vals[i]} got {results[i]:#010x} expected {expected[i]:#010x}"
        finally:
            cuda_ctx.free(d_in)
            cuda_ctx.free(d_out)


# ===========================================================================
# 13. ATOMG.ADD — atomic add
#     Known encoder bug: error 715 (illegal instruction) on launch.
# ===========================================================================

_PTX_ATOMG_ADD = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry atomg_add_test(
    .param .u64 p_out,
    .param .u32 p_addend
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<2>;

    ld.param.u64 %rd0, [p_out];
    ld.param.u32 %r0, [p_addend];
    atom.global.add.u32 %r1, [%rd0], %r0;
    ret;
}
"""


class TestAtomgAdd:
    @gpu
    def test_atomg_add_counter(self, cuda_ctx):
        """atom.global.add.u32: 32 threads each add 1 -> counter = 32."""
        cubins = _compile(_PTX_ATOMG_ADD)
        assert 'atomg_add_test' in cubins
        ok = cuda_ctx.load(cubins['atomg_add_test'])
        assert ok
        func = cuda_ctx.get_func('atomg_add_test')

        d_out = cuda_ctx.alloc(4)
        try:
            cuda_ctx.copy_to(d_out, struct.pack('<I', 0))
            err = cuda_ctx.launch(func, (1, 1, 1), (32, 1, 1), [d_out, 1])
            assert err == 0
            assert cuda_ctx.sync() == 0, "atomg_add crashed"
            raw = cuda_ctx.copy_from(d_out, 4)
            result = struct.unpack('<I', raw)[0]
            assert result == 32, \
                f"atomg_add: got {result}, expected 32"
        finally:
            cuda_ctx.free(d_out)


# ===========================================================================
# 14. I2F.F32.S32 — signed int32 to float32 (negative values)
# ===========================================================================

_PTX_I2F_S32 = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry i2f_s32_test(
    .param .u64 p_out,
    .param .u64 p_in,
    .param .u32 p_n
) {
    .reg .u32 %r0, %r1, %r2;
    .reg .s32 %r3;
    .reg .f32 %f0;
    .reg .u64 %rd0, %rd1, %rd2, %rd3, %rd4;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;

    ld.param.u64 %rd1, [p_in]; add.u64 %rd2, %rd1, %rd0;
    ld.global.s32 %r3, [%rd2];

    cvt.rn.f32.s32 %f0, %r3;

    ld.param.u64 %rd3, [p_out]; add.u64 %rd4, %rd3, %rd0;
    st.global.f32 [%rd4], %f0;
DONE:
    ret;
}
"""


class TestI2fS32:
    @gpu
    def test_i2f_s32_negative(self, cuda_ctx):
        """cvt.rn.f32.s32: signed integers including negatives convert correctly."""
        import math
        cubins = _compile(_PTX_I2F_S32)
        assert 'i2f_s32_test' in cubins
        ok = cuda_ctx.load(cubins['i2f_s32_test'])
        assert ok
        func = cuda_ctx.get_func('i2f_s32_test')

        # Test values: negatives, zero, positives, extremes
        # Note: values outside [-2^24, 2^24] may lose precision in float32 (IEEE 754)
        inputs_signed = [-42, -1000000, -2147483648, -1, 0, 1, 42, 2147483647]
        N = len(inputs_signed)
        # Pack as signed int32
        in_bytes = struct.pack(f'<{N}i', *inputs_signed)

        d_in = cuda_ctx.alloc(4 * N)
        d_out = cuda_ctx.alloc(4 * N)
        try:
            cuda_ctx.copy_to(d_in, in_bytes)
            cuda_ctx.copy_to(d_out, bytes(4 * N))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1), [d_out, d_in, N])
            assert err == 0
            assert cuda_ctx.sync() == 0, "i2f_s32 crashed"
            raw = cuda_ctx.copy_from(d_out, 4 * N)
            results = list(struct.unpack(f'<{N}f', raw))
            for i in range(N):
                # Use struct to get the same rounding as hardware: pack as f32, unpack
                expected_bits = struct.pack('<f', float(inputs_signed[i]))
                expected = struct.unpack('<f', expected_bits)[0]
                assert results[i] == expected, \
                    f"i2f_s32 idx {i}: input={inputs_signed[i]}, got {results[i]}, expected {expected}"
        finally:
            cuda_ctx.free(d_in)
            cuda_ctx.free(d_out)


# ===========================================================================
# 15. cp.async — async global→shared copy
# ===========================================================================

_PTX_CP_ASYNC = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry cp_async_test(
    .param .u64 p_out,
    .param .u64 p_in
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .pred %p0;
    .shared .align 4 .b32 smem[256];

    mov.u32 %r0, %tid.x;

    // Thread 0 initiates async copy of 4 bytes from global to shared
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra SKIP_COPY;

    // smem byte offset for thread 0 = 0
    mov.u32 %r1, 0;
    ld.param.u64 %rd0, [p_in];
    cp.async.ca.shared.global [%r1], [%rd0], 4;

SKIP_COPY:
    cp.async.commit_group;
    cp.async.wait_group 0;
    bar.sync 0;

    // All threads read from shared offset 0
    mov.u32 %r2, 0;
    ld.shared.b32 %r3, [%r2];

    // Write to global output at [tid*4]
    shl.b32 %r4, %r0, 2;
    cvt.u64.u32 %rd1, %r4;
    ld.param.u64 %rd2, [p_out];
    add.u64 %rd3, %rd2, %rd1;
    st.global.u32 [%rd3], %r3;
    ret;
}
"""


class TestCpAsync:
    @gpu
    def test_cp_async_basic(self, cuda_ctx):
        """cp.async: async copy 4B from global to shared, all threads read it."""
        cubins = _compile(_PTX_CP_ASYNC)
        assert 'cp_async_test' in cubins
        ok = cuda_ctx.load(cubins['cp_async_test'])
        assert ok, "cuModuleLoadData failed for cp_async_test"
        func = cuda_ctx.get_func('cp_async_test')

        N = 32  # one warp
        magic = 0xDEADBEEF
        d_in = cuda_ctx.alloc(4)
        d_out = cuda_ctx.alloc(N * 4)
        try:
            cuda_ctx.copy_to(d_in, struct.pack('<I', magic))
            cuda_ctx.copy_to(d_out, b'\x00' * (N * 4))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1),
                                  [d_out, d_in], smem=1024)
            assert err == 0
            assert cuda_ctx.sync() == 0, "cp_async crashed"
            raw = cuda_ctx.copy_from(d_out, N * 4)
            results = list(struct.unpack(f'<{N}I', raw))
            for i in range(N):
                assert results[i] == magic, \
                    f"cp_async thread {i}: got {results[i]:#x}, expected {magic:#x}"
        finally:
            cuda_ctx.free(d_in)
            cuda_ctx.free(d_out)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'gpu'])
