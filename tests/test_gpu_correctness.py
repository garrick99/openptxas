"""
tests/test_gpu_correctness.py — Hardware execution correctness tests for SM_120.

Requires RTX 5090 (SM_120). Marked with @pytest.mark.gpu — skip if no GPU.

Tests:
  1. Multi-block launch — correct output across all thread blocks
  2. Predicated EXIT path — threads that hit early-exit produce no output (verified)
  3. Predicated arithmetic — conditional computation only for active threads
  4. Real ALU on even wdep — IMAD, IADD3, FADD all produce correct results
  5. Branchy kernel — divergent warp path (some threads branch, others don't)

Design note: a single module-scoped CUDA context is shared across all tests.
This avoids the "deferred GPU exception" problem where a GPU-side error 700
(CUDA_ERROR_ILLEGAL_ADDRESS) from one kernel run surfaces at the next
cuCtxCreate_v2 call, contaminating subsequent tests. Each test calls
cctx.load() which first unloads any existing module, then loads the new cubin.

Run: python -m pytest tests/test_gpu_correctness.py -v -m gpu
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
    """CUDA context wrapper. Supports repeated load() calls within the same context."""
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
        """Load a new cubin, unloading any previously loaded module first."""
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

    def launch(self, func, grid, block, args_list) -> int:
        arg_holders = []
        ptrs = []
        for a in args_list:
            holder = ctypes.c_uint64(a) if isinstance(a, int) and a > 0xFFFFFFFF else ctypes.c_int32(a)
            arg_holders.append(holder)
            ptrs.append(ctypes.cast(ctypes.byref(holder), ctypes.c_void_p))
        args_arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        gx, gy, gz = grid
        bx, by, bz = block
        return self.cuda.cuLaunchKernel(func, gx, gy, gz, bx, by, bz, 0, None, args_arr, None)

    def free(self, ptr: int):
        self.cuda.cuMemFree_v2(ctypes.c_uint64(ptr))

    def sync(self) -> int:
        return self.cuda.cuCtxSynchronize()

    def close(self):
        if self.mod and self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()
        if self.ctx and self.ctx.value:
            # cuCtxSynchronize consumes any pending deferred GPU error (e.g. error 715
            # from a kernel crash in an xfailed test).  Without this, the pending error
            # surfaces as error 700 on the next cuCtxCreate_v2 in a different test module.
            self.cuda.cuCtxSynchronize()
            self.cuda.cuCtxDestroy_v2(self.ctx)
            self.ctx = ctypes.c_void_p()


# ---------------------------------------------------------------------------
# Module-scoped CUDA context fixture
# One context per test module — avoids deferred error 700 on cuCtxCreate_v2.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cuda_ctx():
    """Shared CUDA context for all GPU tests in this module."""
    if _CUDA is None:
        pytest.skip("No CUDA GPU available")
    cctx = CUDAContext()
    yield cctx
    cctx.close()


# ---------------------------------------------------------------------------
# PTX kernels under test
# ---------------------------------------------------------------------------

_PTX_VECADD = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry vector_add(
    .param .u64 out, .param .u64 a, .param .u64 b, .param .u32 n)
{
    .reg .b32 %r<8>; .reg .b64 %rd<8>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;
    cvt.u64.u32 %rd0, %r3;
    shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [a]; add.u64 %rd2, %rd1, %rd0;
    ld.param.u64 %rd3, [b]; add.u64 %rd4, %rd3, %rd0;
    ld.global.u32 %r5, [%rd2];
    ld.global.u32 %r6, [%rd4];
    add.s32 %r7, %r5, %r6;
    ld.param.u64 %rd5, [out]; add.u64 %rd6, %rd5, %rd0;
    st.global.u32 [%rd6], %r7;
DONE:
    ret;
}
"""

# Kernel where output[i] = a[i] * mul + b[i]
# Tests IMAD (even wdep=0x3e) with register operands — mul is a u32 param, not immediate.
_PTX_ALU_CHAIN = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry alu_chain(
    .param .u64 out, .param .u64 a, .param .u64 b, .param .u32 mul, .param .u32 n)
{
    .reg .b32 %r<10>; .reg .b64 %rd<8>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;
    cvt.u64.u32 %rd0, %r3;
    shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [a]; add.u64 %rd2, %rd1, %rd0;
    ld.global.u32 %r5, [%rd2];
    ld.param.u64 %rd3, [b]; add.u64 %rd4, %rd3, %rd0;
    ld.global.u32 %r6, [%rd4];
    ld.param.u32 %r7, [mul];
    mad.lo.s32 %r8, %r5, %r7, %r6;
    ld.param.u64 %rd5, [out]; add.u64 %rd6, %rd5, %rd0;
    st.global.u32 [%rd6], %r8;
DONE:
    ret;
}
"""

# Kernel: output[i] = 0 if idx < half, else 1
# Tests intra-warp branch divergence: threads 0..half-1 go LEFT, half..N-1 go RIGHT.
_PTX_DIVERGENT = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry divergent(
    .param .u64 out, .param .u64 zero_val, .param .u64 one_val,
    .param .u32 half, .param .u32 n)
{
    .reg .b32 %r<8>; .reg .b64 %rd<8>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;
    ld.param.u32 %r5, [half];
    setp.ge.u32 %p1, %r3, %r5;
    @%p1 bra WRITE_ONE;
    ld.param.u64 %rd1, [zero_val]; ld.global.u32 %r6, [%rd1];
    bra STORE;
WRITE_ONE:
    ld.param.u64 %rd2, [one_val]; ld.global.u32 %r6, [%rd2];
STORE:
    cvt.u64.u32 %rd0, %r3;
    shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd3, [out]; add.u64 %rd4, %rd3, %rd0;
    st.global.u32 [%rd4], %r6;
DONE:
    ret;
}
"""


def _compile(ptx_src: str) -> dict[str, bytes]:
    return compile_ptx_source(ptx_src)


# ---------------------------------------------------------------------------
# Test: 1. Multi-block launch + predicated EXIT
# ---------------------------------------------------------------------------

class TestMultiBlock:
    @gpu
    def test_vector_add_single_block(self, cuda_ctx):
        """vector_add: 32 elements, 1 block."""
        cubins = _compile(_PTX_VECADD)
        assert 'vector_add' in cubins
        N = 32
        a = list(range(1, N + 1))
        b = list(range(100, 100 + N))
        self._run_and_verify(cuda_ctx, cubins['vector_add'], N, a, b,
                             grid=(1, 1, 1), block=(N, 1, 1))

    @gpu
    def test_vector_add_multi_block(self, cuda_ctx):
        """vector_add: 128 elements, 4 blocks of 32."""
        cubins = _compile(_PTX_VECADD)
        N = 128
        a = list(range(1, N + 1))
        b = [i * 10 for i in range(N)]
        self._run_and_verify(cuda_ctx, cubins['vector_add'], N, a, b,
                             grid=(4, 1, 1), block=(32, 1, 1))

    @gpu
    def test_vector_add_non_power_of_2(self, cuda_ctx):
        """vector_add: 100 elements, 4 blocks of 32 (last block has 4 active threads)."""
        cubins = _compile(_PTX_VECADD)
        N = 100
        a = list(range(N))
        b = [N - i for i in range(N)]
        self._run_and_verify(cuda_ctx, cubins['vector_add'], N, a, b,
                             grid=(4, 1, 1), block=(32, 1, 1))

    @gpu
    def test_predicated_exit_threads_write_nothing(self, cuda_ctx):
        """Threads with idx >= N must take the early EXIT and NOT write to out[idx].

        Verifies the @P0 EXIT path (predicated EXIT) actually fires for out-of-bounds
        threads. Output buffer is pre-filled with a sentinel value 0xDEADBEEF.
        Active threads (idx < N) write their result. Out-of-bounds threads must leave
        the sentinel untouched.
        """
        cubins = _compile(_PTX_VECADD)
        N = 17  # deliberately not power-of-2: 1 block of 32 has 15 idle threads
        a = list(range(N)) + [0] * (32 - N)
        b = list(range(100, 100 + N)) + [0] * (32 - N)
        total = 32
        sentinel = 0xDEADBEEF

        assert cuda_ctx.load(cubins['vector_add'])
        func = cuda_ctx.get_func('vector_add')

        byte_size = total * 4
        d_a = cuda_ctx.alloc(byte_size)
        d_b = cuda_ctx.alloc(byte_size)
        d_out = cuda_ctx.alloc(byte_size)
        try:
            cuda_ctx.copy_to(d_a, struct.pack(f'<{total}i', *a))
            cuda_ctx.copy_to(d_b, struct.pack(f'<{total}i', *b))
            cuda_ctx.copy_to(d_out, struct.pack(f'<{total}I', *([sentinel] * total)))

            err = cuda_ctx.launch(func, (1, 1, 1), (32, 1, 1),
                                  [d_out, d_a, d_b, N])
            assert err == 0, f"Launch failed: {err}"
            assert cuda_ctx.sync() == 0, "Kernel crashed"

            raw = cuda_ctx.copy_from(d_out, byte_size)
            results = list(struct.unpack(f'<{total}I', raw))

            for i in range(N):
                expected = a[i] + b[i]
                assert results[i] == expected, \
                    f"idx {i}: got {results[i]:#010x}, expected {expected}"

            for i in range(N, total):
                assert results[i] == sentinel, \
                    f"idx {i} (idle): got {results[i]:#010x}, " \
                    f"expected sentinel {sentinel:#010x} — predicated EXIT not taken!"
        finally:
            cuda_ctx.free(d_a)
            cuda_ctx.free(d_b)
            cuda_ctx.free(d_out)

    def _run_and_verify(self, cctx, cubin_bytes, N, a_vals, b_vals, grid, block):
        assert cctx.load(cubin_bytes), "cuModuleLoadData failed"
        func = cctx.get_func('vector_add')
        byte_size = N * 4
        d_a = cctx.alloc(byte_size)
        d_b = cctx.alloc(byte_size)
        d_out = cctx.alloc(byte_size)
        try:
            cctx.copy_to(d_a, struct.pack(f'<{N}i', *a_vals))
            cctx.copy_to(d_b, struct.pack(f'<{N}i', *b_vals))
            cctx.copy_to(d_out, struct.pack(f'<{N}i', *([0] * N)))
            err = cctx.launch(func, grid, block, [d_out, d_a, d_b, N])
            assert err == 0, f"Launch error {err}"
            assert cctx.sync() == 0, "Kernel crash on sync"
            raw = cctx.copy_from(d_out, byte_size)
            results = list(struct.unpack(f'<{N}i', raw))
            expected = [a_vals[i] + b_vals[i] for i in range(N)]
            mismatches = [i for i in range(N) if results[i] != expected[i]]
            assert not mismatches, \
                f"Mismatch at idx {mismatches[0]}: got {results[mismatches[0]]}, " \
                f"expected {expected[mismatches[0]]}"
        finally:
            cctx.free(d_a)
            cctx.free(d_b)
            cctx.free(d_out)


# ---------------------------------------------------------------------------
# Test: 2. Real ALU on even wdep=0x3e (IMAD chain)
# ---------------------------------------------------------------------------

class TestAluCorrectness:
    @gpu
    def test_imad_chain(self, cuda_ctx):
        """alu_chain kernel: out[i] = a[i] * mul + b[i] — tests IMAD R-UR with register operands."""
        cubins = _compile(_PTX_ALU_CHAIN)
        assert 'alu_chain' in cubins
        N = 64
        a_vals = list(range(N))
        b_vals = [i * 2 for i in range(N)]
        MUL = 3
        expected = [a_vals[i] * MUL + b_vals[i] for i in range(N)]

        assert cuda_ctx.load(cubins['alu_chain'])
        func = cuda_ctx.get_func('alu_chain')
        byte_size = N * 4
        d_a = cuda_ctx.alloc(byte_size)
        d_b = cuda_ctx.alloc(byte_size)
        d_out = cuda_ctx.alloc(byte_size)
        try:
            cuda_ctx.copy_to(d_a, struct.pack(f'<{N}i', *a_vals))
            cuda_ctx.copy_to(d_b, struct.pack(f'<{N}i', *b_vals))
            cuda_ctx.copy_to(d_out, struct.pack(f'<{N}i', *([0] * N)))
            # args: out, a, b, mul, n
            err = cuda_ctx.launch(func, (2, 1, 1), (32, 1, 1), [d_out, d_a, d_b, MUL, N])
            assert err == 0
            assert cuda_ctx.sync() == 0, "alu_chain crashed"
            raw = cuda_ctx.copy_from(d_out, byte_size)
            results = list(struct.unpack(f'<{N}i', raw))
            mismatches = [i for i in range(N) if results[i] != expected[i]]
            assert not mismatches, \
                f"ALU chain mismatch at idx {mismatches[0]}: " \
                f"got {results[mismatches[0]]}, expected {expected[mismatches[0]]}"
        finally:
            cuda_ctx.free(d_a)
            cuda_ctx.free(d_b)
            cuda_ctx.free(d_out)


# ---------------------------------------------------------------------------
# Test: 3. Divergent warp path
# ---------------------------------------------------------------------------

class TestDivergentWarp:
    def _run_divergent(self, cctx, cubin, N, half, total=None):
        """Launch divergent kernel. Returns (results, sentinel)."""
        if total is None:
            total = N
        assert cctx.load(cubin)
        func = cctx.get_func('divergent')
        byte_size = total * 4

        d_zero = cctx.alloc(4)
        d_one  = cctx.alloc(4)
        d_out  = cctx.alloc(byte_size)
        try:
            cctx.copy_to(d_zero, struct.pack('<I', 0))
            cctx.copy_to(d_one,  struct.pack('<I', 1))
            sentinel = 0xDEADBEEF
            cctx.copy_to(d_out, struct.pack(f'<{total}I', *([sentinel] * total)))

            # args: out, zero_val, one_val, half, n
            err = cctx.launch(func, (1, 1, 1), (total, 1, 1),
                              [d_out, d_zero, d_one, half, N])
            assert err == 0, f"launch error {err}"
            assert cctx.sync() == 0, "divergent kernel crashed"
            raw = cctx.copy_from(d_out, byte_size)
            return list(struct.unpack(f'<{total}I', raw)), sentinel
        finally:
            cctx.free(d_zero)
            cctx.free(d_one)
            cctx.free(d_out)

    @gpu
    def test_intra_warp_divergence(self, cuda_ctx):
        """Divergent kernel: threads 0..15 → 0, threads 16..31 → 1.

        All 32 threads in one block diverge at the half-split branch.
        Tests that both branch paths execute and reconverge.
        """
        cubins = _compile(_PTX_DIVERGENT)
        assert 'divergent' in cubins
        N = 32; half = 16
        results, _ = self._run_divergent(cuda_ctx, cubins['divergent'], N, half)
        for i in range(N):
            expected = 0 if i < half else 1
            assert results[i] == expected, \
                f"idx {i}: got {results[i]}, expected {expected}"

    @gpu
    def test_divergent_with_idle_threads(self, cuda_ctx):
        """Divergent kernel, N=20, half=10: threads [20,31] take the bounds-check EXIT."""
        cubins = _compile(_PTX_DIVERGENT)
        N = 20; half = 10; total = 32
        results, sentinel = self._run_divergent(cuda_ctx, cubins['divergent'], N, half, total)
        for i in range(N):
            expected = 0 if i < half else 1
            assert results[i] == expected, \
                f"active idx {i}: got {results[i]}, expected {expected}"
        for i in range(N, total):
            assert results[i] == sentinel, \
                f"idle idx {i}: got {results[i]:#010x}, expected sentinel {sentinel:#010x}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'gpu'])
