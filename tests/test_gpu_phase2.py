"""
tests/test_gpu_phase2.py — Phase 2 GPU correctness tests.

Instruction classes tested:
  1. MEMBAR.GL / MEMBAR.CTA — Memory fence
  2. ATOMG.EXCH.b32 — Atomic exchange
  3. ATOMG.MIN.S32 / ATOMG.MAX.S32 — Atomic signed min/max
  4. ATOMG.ADD.F32 — Atomic float add
  5. ATOMG.CAS.B64 — 64-bit compare-and-swap
  6. IDP.4A (dp4a.u32.u32) — Integer dot product
  7. FLO (bfind.u32) — Find highest set bit
  8. FLO (clz.b32) — Count leading zeros (via FLO)

Run: python -m pytest tests/test_gpu_phase2.py -v -m gpu
"""
import ctypes
import struct
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass.pipeline import compile_ptx_source

# ---------------------------------------------------------------------------
# GPU driver bootstrap (same as phase1)
# ---------------------------------------------------------------------------


try:
    import ctypes; _c = ctypes.cdll.LoadLibrary("nvcuda.dll"); _CUDA = _c.cuInit(0) == 0
except Exception:
    _CUDA = False
gpu = pytest.mark.skipif(not _CUDA, reason="No CUDA GPU")


def _compile(ptx_src: str) -> dict[str, bytes]:
    return compile_ptx_source(ptx_src)


# ===========================================================================
# 1. MEMBAR — Memory fence (membar.gl / membar.cta)
# ===========================================================================

_PTX_MEMBAR = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry membar_test(
    .param .u64 p_out,
    .param .u32 p_val
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd0, [p_out];
    ld.param.u32 %r0, [p_val];
    st.global.u32 [%rd0], %r0;
    membar.gl;
    membar.cta;
    ret;
}
"""


class TestMembar:
    @gpu
    def test_membar_basic(self, cuda_ctx):
        """membar.gl + membar.cta: kernel should not crash."""
        cubins = _compile(_PTX_MEMBAR)
        assert 'membar_test' in cubins
        ok = cuda_ctx.load(cubins['membar_test'])
        assert ok, "cuModuleLoadData failed for membar_test"
        func = cuda_ctx.get_func('membar_test')

        d_out = cuda_ctx.alloc(4)
        try:
            cuda_ctx.copy_to(d_out, struct.pack('<I', 0))
            err = cuda_ctx.launch(func, (1, 1, 1), (32, 1, 1), [d_out, 42])
            assert err == 0
            assert cuda_ctx.sync() == 0, "membar_test crashed"
            raw = cuda_ctx.copy_from(d_out, 4)
            result = struct.unpack('<I', raw)[0]
            # Last thread to write wins; all write 42
            assert result == 42, f"membar: got {result}, expected 42"
        finally:
            cuda_ctx.free(d_out)


# ===========================================================================
# 2. ATOMG.EXCH.b32 — Atomic exchange
# ===========================================================================

_PTX_ATOM_EXCH = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry atom_exch_test(
    .param .u64 p_addr,
    .param .u64 p_out,
    .param .u32 p_val
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<6>;

    mov.u32 %r0, %tid.x;
    ld.param.u64 %rd0, [p_addr];
    ld.param.u64 %rd1, [p_out];
    ld.param.u32 %r1, [p_val];

    atom.global.exch.b32 %r2, [%rd0], %r1;

    cvt.u64.u32 %rd2, %r0;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r2;
    ret;
}
"""


class TestAtomExch:
    @gpu
    def test_atom_exch(self, cuda_ctx):
        """atom.global.exch.b32: exchange returns old value."""
        cubins = _compile(_PTX_ATOM_EXCH)
        assert 'atom_exch_test' in cubins
        ok = cuda_ctx.load(cubins['atom_exch_test'])
        assert ok, "cuModuleLoadData failed for atom_exch_test"
        func = cuda_ctx.get_func('atom_exch_test')

        # Start with initial value 999, exchange with 42 from 1 thread
        d_addr = cuda_ctx.alloc(4)
        d_out = cuda_ctx.alloc(4)
        try:
            cuda_ctx.copy_to(d_addr, struct.pack('<I', 999))
            cuda_ctx.copy_to(d_out, struct.pack('<I', 0))
            err = cuda_ctx.launch(func, (1, 1, 1), (1, 1, 1), [d_addr, d_out, 42])
            assert err == 0
            assert cuda_ctx.sync() == 0, "atom_exch crashed"
            # The old value returned should be 999
            raw = cuda_ctx.copy_from(d_out, 4)
            old_val = struct.unpack('<I', raw)[0]
            assert old_val == 999, f"atom_exch: old value = {old_val}, expected 999"
            # The new value at addr should be 42
            raw = cuda_ctx.copy_from(d_addr, 4)
            new_val = struct.unpack('<I', raw)[0]
            assert new_val == 42, f"atom_exch: new value = {new_val}, expected 42"
        finally:
            cuda_ctx.free(d_addr)
            cuda_ctx.free(d_out)


# ===========================================================================
# 3. ATOMG.MIN.S32 / ATOMG.MAX.S32 — Atomic signed min/max
# ===========================================================================

_PTX_ATOM_MIN = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry atom_min_test(
    .param .u64 p_addr,
    .param .u64 p_vals,
    .param .u32 p_n
)
{
    .reg .u32 %r<4>;
    .reg .s32 %rs<4>;
    .reg .u64 %rd<6>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    ld.param.u64 %rd0, [p_addr];
    ld.param.u64 %rd1, [p_vals];

    cvt.u64.u32 %rd2, %r0;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.s32 %rs0, [%rd3];

    atom.global.min.s32 %rs1, [%rd0], %rs0;
DONE:
    ret;
}
"""

_PTX_ATOM_MAX = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry atom_max_test(
    .param .u64 p_addr,
    .param .u64 p_vals,
    .param .u32 p_n
)
{
    .reg .u32 %r<4>;
    .reg .s32 %rs<4>;
    .reg .u64 %rd<6>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    ld.param.u64 %rd0, [p_addr];
    ld.param.u64 %rd1, [p_vals];

    cvt.u64.u32 %rd2, %r0;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.s32 %rs0, [%rd3];

    atom.global.max.s32 %rs1, [%rd0], %rs0;
DONE:
    ret;
}
"""


class TestAtomMinMax:
    @gpu
    def test_atom_min_s32(self, cuda_ctx):
        """atom.global.min.s32: parallel min reduction with negative values."""
        cubins = _compile(_PTX_ATOM_MIN)
        assert 'atom_min_test' in cubins
        ok = cuda_ctx.load(cubins['atom_min_test'])
        assert ok, "cuModuleLoadData failed for atom_min_test"
        func = cuda_ctx.get_func('atom_min_test')

        vals = [100, -5, 42, -200, 7, 0, -1, 50]
        N = len(vals)
        d_addr = cuda_ctx.alloc(4)
        d_vals = cuda_ctx.alloc(4 * N)
        try:
            cuda_ctx.copy_to(d_addr, struct.pack('<i', 0x7FFFFFFF))  # INT_MAX
            cuda_ctx.copy_to(d_vals, struct.pack(f'<{N}i', *vals))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1), [d_addr, d_vals, N])
            assert err == 0
            assert cuda_ctx.sync() == 0, "atom_min crashed"
            raw = cuda_ctx.copy_from(d_addr, 4)
            result = struct.unpack('<i', raw)[0]
            assert result == min(vals), f"atom_min: got {result}, expected {min(vals)}"
        finally:
            cuda_ctx.free(d_addr)
            cuda_ctx.free(d_vals)

    @gpu
    def test_atom_max_s32(self, cuda_ctx):
        """atom.global.max.s32: parallel max reduction with negative values."""
        cubins = _compile(_PTX_ATOM_MAX)
        assert 'atom_max_test' in cubins
        ok = cuda_ctx.load(cubins['atom_max_test'])
        assert ok, "cuModuleLoadData failed for atom_max_test"
        func = cuda_ctx.get_func('atom_max_test')

        vals = [-100, -5, -42, -200, -7, -1, -50, -3]
        N = len(vals)
        d_addr = cuda_ctx.alloc(4)
        d_vals = cuda_ctx.alloc(4 * N)
        try:
            cuda_ctx.copy_to(d_addr, struct.pack('<i', -0x80000000))  # INT_MIN
            cuda_ctx.copy_to(d_vals, struct.pack(f'<{N}i', *vals))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1), [d_addr, d_vals, N])
            assert err == 0
            assert cuda_ctx.sync() == 0, "atom_max crashed"
            raw = cuda_ctx.copy_from(d_addr, 4)
            result = struct.unpack('<i', raw)[0]
            assert result == max(vals), f"atom_max: got {result}, expected {max(vals)}"
        finally:
            cuda_ctx.free(d_addr)
            cuda_ctx.free(d_vals)


# ===========================================================================
# 4. ATOMG.ADD.F32 — Atomic float add
# ===========================================================================

_PTX_ATOM_ADD_F32 = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry atom_add_f32_test(
    .param .u64 p_addr,
    .param .f32 p_val
)
{
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    .reg .u64 %rd<2>;

    ld.param.u64 %rd0, [p_addr];
    ld.param.f32 %f0, [p_val];
    atom.global.add.f32 %f1, [%rd0], %f0;
    ret;
}
"""


class TestAtomAddF32:
    @gpu
    def test_atom_add_f32(self, cuda_ctx):
        """atom.global.add.f32: 32 threads each add 1.0 -> result = 32.0."""
        cubins = _compile(_PTX_ATOM_ADD_F32)
        assert 'atom_add_f32_test' in cubins
        ok = cuda_ctx.load(cubins['atom_add_f32_test'])
        assert ok, "cuModuleLoadData failed for atom_add_f32_test"
        func = cuda_ctx.get_func('atom_add_f32_test')

        d_addr = cuda_ctx.alloc(4)
        try:
            cuda_ctx.copy_to(d_addr, struct.pack('<f', 0.0))
            # Pass 1.0 as float parameter (as u32 bits)
            one_f32 = struct.unpack('<I', struct.pack('<f', 1.0))[0]
            err = cuda_ctx.launch(func, (1, 1, 1), (32, 1, 1), [d_addr, one_f32])
            assert err == 0
            assert cuda_ctx.sync() == 0, "atom_add_f32 crashed"
            raw = cuda_ctx.copy_from(d_addr, 4)
            result = struct.unpack('<f', raw)[0]
            assert abs(result - 32.0) < 0.01, \
                f"atom_add_f32: got {result}, expected 32.0"
        finally:
            cuda_ctx.free(d_addr)


# ===========================================================================
# 5. IDP.4A (dp4a.u32.u32) — Integer dot product
# ===========================================================================

_PTX_DP4A = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry dp4a_test(
    .param .u64 p_out,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u32 p_n
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;

    ld.param.u64 %rd1, [p_a];
    add.u64 %rd2, %rd1, %rd0;
    ld.global.u32 %r2, [%rd2];

    ld.param.u64 %rd3, [p_b];
    add.u64 %rd4, %rd3, %rd0;
    ld.global.u32 %r3, [%rd4];

    mov.u32 %r4, 0;
    dp4a.u32.u32 %r5, %r2, %r3, %r4;

    ld.param.u64 %rd5, [p_out];
    add.u64 %rd6, %rd5, %rd0;
    st.global.u32 [%rd6], %r5;
DONE:
    ret;
}
"""


class TestDp4a:
    @gpu
    def test_dp4a_known_values(self, cuda_ctx):
        """dp4a.u32.u32: dot product of 4 packed bytes."""
        cubins = _compile(_PTX_DP4A)
        assert 'dp4a_test' in cubins
        ok = cuda_ctx.load(cubins['dp4a_test'])
        assert ok, "cuModuleLoadData failed for dp4a_test"
        func = cuda_ctx.get_func('dp4a_test')

        # Test cases: a and b as packed u8x4, expected = sum(a_i * b_i)
        # a = [1, 2, 3, 4] packed into u32 = 0x04030201
        # b = [5, 6, 7, 8] packed into u32 = 0x08070605
        # expected = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        test_a = [0x04030201, 0x01010101, 0x00000000, 0xFFFFFFFF]
        test_b = [0x08070605, 0x01010101, 0xFFFFFFFF, 0x01010101]
        expected = [
            1*5 + 2*6 + 3*7 + 4*8,    # 70
            1*1 + 1*1 + 1*1 + 1*1,    # 4
            0,                          # 0
            255*1 + 255*1 + 255*1 + 255*1,  # 1020
        ]
        N = len(test_a)

        d_out = cuda_ctx.alloc(4 * N)
        d_a = cuda_ctx.alloc(4 * N)
        d_b = cuda_ctx.alloc(4 * N)
        try:
            cuda_ctx.copy_to(d_a, struct.pack(f'<{N}I', *test_a))
            cuda_ctx.copy_to(d_b, struct.pack(f'<{N}I', *test_b))
            cuda_ctx.copy_to(d_out, bytes(4 * N))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1), [d_out, d_a, d_b, N])
            assert err == 0
            assert cuda_ctx.sync() == 0, "dp4a crashed"
            raw = cuda_ctx.copy_from(d_out, 4 * N)
            results = list(struct.unpack(f'<{N}I', raw))
            for i in range(N):
                assert results[i] == expected[i], \
                    f"dp4a idx {i}: got {results[i]}, expected {expected[i]}"
        finally:
            cuda_ctx.free(d_out)
            cuda_ctx.free(d_a)
            cuda_ctx.free(d_b)


# ===========================================================================
# 6. bfind.u32 — Find highest set bit (maps to FLO.U32)
# ===========================================================================

_PTX_BFIND = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry bfind_test(
    .param .u64 p_out,
    .param .u64 p_in,
    .param .u32 p_n
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;

    ld.param.u64 %rd1, [p_in];
    add.u64 %rd2, %rd1, %rd0;
    ld.global.u32 %r2, [%rd2];

    bfind.u32 %r3, %r2;

    ld.param.u64 %rd3, [p_out];
    add.u64 %rd4, %rd3, %rd0;
    st.global.u32 [%rd4], %r3;
DONE:
    ret;
}
"""


class TestBfind:
    @gpu
    def test_bfind_known_values(self, cuda_ctx):
        """bfind.u32: find highest set bit position."""
        cubins = _compile(_PTX_BFIND)
        assert 'bfind_test' in cubins
        ok = cuda_ctx.load(cubins['bfind_test'])
        assert ok, "cuModuleLoadData failed for bfind_test"
        func = cuda_ctx.get_func('bfind_test')

        # bfind.u32 returns the bit position of the highest set bit
        # For 0, ptx spec says 0xFFFFFFFF (-1)
        inputs = [0x80000000, 0x00000001, 0x00000100, 0x0000FFFF]
        # bfind = FLO = position of highest set bit (0-indexed from LSB)
        expected = [31, 0, 8, 15]
        N = len(inputs)

        d_out = cuda_ctx.alloc(4 * N)
        d_in = cuda_ctx.alloc(4 * N)
        try:
            cuda_ctx.copy_to(d_in, struct.pack(f'<{N}I', *inputs))
            cuda_ctx.copy_to(d_out, bytes(4 * N))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1), [d_out, d_in, N])
            assert err == 0
            assert cuda_ctx.sync() == 0, "bfind crashed"
            raw = cuda_ctx.copy_from(d_out, 4 * N)
            results = list(struct.unpack(f'<{N}I', raw))
            for i in range(N):
                assert results[i] == expected[i], \
                    f"bfind idx {i}: input={inputs[i]:#x} got {results[i]}, expected {expected[i]}"
        finally:
            cuda_ctx.free(d_out)
            cuda_ctx.free(d_in)


# ===========================================================================
# 7. CLZ — Count leading zeros (already tested in phase1, but now via
#    the clz PTX opcode which maps to FLO)
# ===========================================================================

_PTX_CLZ = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry clz_test(
    .param .u64 p_out,
    .param .u64 p_in,
    .param .u32 p_n
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;

    ld.param.u64 %rd1, [p_in];
    add.u64 %rd2, %rd1, %rd0;
    ld.global.u32 %r2, [%rd2];

    clz.b32 %r3, %r2;

    ld.param.u64 %rd3, [p_out];
    add.u64 %rd4, %rd3, %rd0;
    st.global.u32 [%rd4], %r3;
DONE:
    ret;
}
"""


class TestClz:
    @gpu
    def test_clz_known_values(self, cuda_ctx):
        """clz.b32: count leading zeros."""
        cubins = _compile(_PTX_CLZ)
        assert 'clz_test' in cubins
        ok = cuda_ctx.load(cubins['clz_test'])
        assert ok, "cuModuleLoadData failed for clz_test"
        func = cuda_ctx.get_func('clz_test')

        # clz.b32 = 31 - FLO(MSB_position).  FLO returns the bit index of
        # the highest set bit; the IADD3 step computes the complement.
        # Zero input: FLO returns 0xFFFFFFFF, 31 - 0xFFFFFFFF wraps to 32.
        inputs = [0x80000000, 0x00000001, 0x00000100, 0x0000FFFF]
        expected = [0, 31, 23, 16]
        N = len(inputs)

        d_out = cuda_ctx.alloc(4 * N)
        d_in = cuda_ctx.alloc(4 * N)
        try:
            cuda_ctx.copy_to(d_in, struct.pack(f'<{N}I', *inputs))
            cuda_ctx.copy_to(d_out, bytes(4 * N))
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1), [d_out, d_in, N])
            assert err == 0
            assert cuda_ctx.sync() == 0, "clz crashed"
            raw = cuda_ctx.copy_from(d_out, 4 * N)
            results = list(struct.unpack(f'<{N}I', raw))
            for i in range(N):
                assert results[i] == expected[i], \
                    f"clz idx {i}: input={inputs[i]:#x} got {results[i]}, expected {expected[i]}"
        finally:
            cuda_ctx.free(d_out)
            cuda_ctx.free(d_in)


# ===========================================================================
# 8. ATOMG.CAS.B64 — 64-bit compare-and-swap
# ===========================================================================

_PTX_ATOM_CAS64 = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry atom_cas64_test(
    .param .u64 p_addr,
    .param .u64 p_cmp,
    .param .u64 p_new,
    .param .u64 p_out
)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<4>;

    ld.param.u64 %rd0, [p_addr];
    ld.param.u64 %rd1, [p_cmp];
    ld.param.u64 %rd2, [p_new];

    // Materialize addr to GPR via add.u64 0
    add.u64 %rd0, %rd0, 0;
    // Materialize cmp and new via add.u64 0
    add.u64 %rd1, %rd1, 0;
    add.u64 %rd2, %rd2, 0;

    atom.global.cas.b64 %rd3, [%rd0], %rd1, %rd2;
    ld.param.u64 %rd4, [p_out];
    st.global.u64 [%rd4], %rd3;
    ret;
}
"""


class TestAtomCas64:
    @gpu
    def test_atom_cas64_success(self, cuda_ctx):
        """atom.global.cas.b64: successful CAS returns old value and updates."""
        cubins = _compile(_PTX_ATOM_CAS64)
        assert 'atom_cas64_test' in cubins
        ok = cuda_ctx.load(cubins['atom_cas64_test'])
        assert ok, "cuModuleLoadData failed for atom_cas64_test"
        func = cuda_ctx.get_func('atom_cas64_test')

        old_val = 0xDEADBEEFCAFEBABE
        cmp_val = 0xDEADBEEFCAFEBABE  # matches -> should swap
        new_val = 0x1234567890ABCDEF
        d_addr = cuda_ctx.alloc(8)
        d_out = cuda_ctx.alloc(8)
        try:
            cuda_ctx.copy_to(d_addr, struct.pack('<Q', old_val))
            cuda_ctx.copy_to(d_out, struct.pack('<Q', 0))
            err = cuda_ctx.launch(func, (1, 1, 1), (1, 1, 1),
                                  [d_addr, cmp_val, new_val, d_out])
            assert err == 0
            assert cuda_ctx.sync() == 0, "atom_cas64 crashed"
            # Old value should be returned
            raw = cuda_ctx.copy_from(d_out, 8)
            returned = struct.unpack('<Q', raw)[0]
            assert returned == old_val, \
                f"cas64: returned {returned:#018x}, expected {old_val:#018x}"
            # Memory should now contain new_val
            raw = cuda_ctx.copy_from(d_addr, 8)
            mem_val = struct.unpack('<Q', raw)[0]
            assert mem_val == new_val, \
                f"cas64: memory {mem_val:#018x}, expected {new_val:#018x}"
        finally:
            cuda_ctx.free(d_addr)
            cuda_ctx.free(d_out)

    @gpu
    def test_atom_cas64_fail(self, cuda_ctx):
        """atom.global.cas.b64: failed CAS (mismatch) leaves memory unchanged."""
        cubins = _compile(_PTX_ATOM_CAS64)
        assert 'atom_cas64_test' in cubins
        ok = cuda_ctx.load(cubins['atom_cas64_test'])
        assert ok
        func = cuda_ctx.get_func('atom_cas64_test')

        old_val = 0xAAAABBBBCCCCDDDD
        cmp_val = 0x1111111111111111  # does NOT match -> should NOT swap
        new_val = 0x2222222222222222
        d_addr = cuda_ctx.alloc(8)
        d_out = cuda_ctx.alloc(8)
        try:
            cuda_ctx.copy_to(d_addr, struct.pack('<Q', old_val))
            cuda_ctx.copy_to(d_out, struct.pack('<Q', 0))
            err = cuda_ctx.launch(func, (1, 1, 1), (1, 1, 1),
                                  [d_addr, cmp_val, new_val, d_out])
            assert err == 0
            assert cuda_ctx.sync() == 0, "atom_cas64_fail crashed"
            # Returned old value
            raw = cuda_ctx.copy_from(d_out, 8)
            returned = struct.unpack('<Q', raw)[0]
            assert returned == old_val, \
                f"cas64 fail: returned {returned:#018x}, expected {old_val:#018x}"
            # Memory should be UNCHANGED (CAS failed)
            raw = cuda_ctx.copy_from(d_addr, 8)
            mem_val = struct.unpack('<Q', raw)[0]
            assert mem_val == old_val, \
                f"cas64 fail: memory changed to {mem_val:#018x}, expected unchanged {old_val:#018x}"
        finally:
            cuda_ctx.free(d_addr)
            cuda_ctx.free(d_out)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'gpu'])
