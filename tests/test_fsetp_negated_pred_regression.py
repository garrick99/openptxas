"""
Regression test for FSETP stale _negated_preds bug.

Bug: When ISETP (setp.lt.s32) writes predicate P0 with inversion
(ISETP.GE + negation flag), and then FSETP (setp.gt.f32) writes
the SAME predicate P0 WITHOUT inversion, the stale negation flag
from the ISETP persists. Downstream consumers (@P0 bra) read the
predicate with flipped sense, taking the wrong branch.

This test captures the exact failure pattern from the OpenCUDA
cond_test kernel. It must pass before and after any fix to
_negated_preds handling.

Symptom: nested if/else with integer guard + float comparison
produces wrong branch selection, leading to CUDA error 700
(illegal address) or wrong output values.

Fix target: sass/isel.py — clear _negated_preds when FSETP
writes a predicate that was previously negated by ISETP.
"""
import struct
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sass.pipeline import compile_ptx_source

# Minimal kernel: ISETP sets P0 (negated), then FSETP reuses P0 (not negated).
# Thread 0 loads a[0]=15.0, compares > 10.0 → should take true branch → out = 30.0
# If pred state is stale (negated), it takes the false branch → out = 7.5
FSETP_PRED_REUSE_PTX = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry fsetp_pred_reuse(
    .param .u64 out, .param .u64 a, .param .s32 n)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<3>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [out];
    ld.param.u64 %rd1, [a];
    ld.param.s32 %r0, [n];
    mov.u32 %r1, %tid.x;

    // Step 1: ISETP with inversion (setp.lt → ISETP.GE + negate)
    // This sets P0 and marks it as negated in _negated_preds
    setp.lt.s32 %p0, %r1, %r0;
    @%p0 bra GUARDED;
    bra DONE;

GUARDED:
    // Inside guard: load a[tid]
    mul.lo.s32 %r2, %r1, 4;
    cvt.u64.u32 %rd2, %r2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.f32 %f0, [%rd3];

    // Step 2: FSETP reuses P0 (setp.gt.f32 → NOT inverted)
    // BUG: _negated_preds still has P0 from step 1
    setp.gt.f32 %p0, %f0, 0f41200000;

    // Step 3: Branch on P0 — if stale negation, takes wrong path
    @%p0 bra TRUE_PATH;
    // False path: multiply by 0.5
    mul.f32 %f1, %f0, 0f3F000000;
    add.u64 %rd4, %rd0, %rd2;
    st.global.f32 [%rd4], %f1;
    bra DONE;

TRUE_PATH:
    // True path: multiply by 2.0
    mul.f32 %f2, %f0, 0f40000000;
    add.u64 %rd5, %rd0, %rd2;
    st.global.f32 [%rd5], %f2;

DONE:
    ret;
}
"""


def test_fsetp_pred_reuse_compiles():
    """The kernel must compile without errors."""
    result = compile_ptx_source(FSETP_PRED_REUSE_PTX)
    assert 'fsetp_pred_reuse' in result
    cubin = result['fsetp_pred_reuse']
    assert len(cubin) > 100, f"cubin too small: {len(cubin)} bytes"


def _get_cuda():
    try:
        import ctypes
        cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
        if cuda.cuInit(0) != 0:
            return None
        return cuda
    except Exception:
        return None


@pytest.mark.skipif(_get_cuda() is None, reason="No CUDA GPU")
@pytest.mark.xfail(reason="FSETP stale _negated_preds: P0 negation from ISETP.GE persists through FSETP.GT rewrite — wrong branch taken")
def test_fsetp_pred_reuse_gpu_correctness():
    """
    Thread 0: a[0] = 15.0, threshold = 10.0
    15.0 > 10.0 is TRUE → should take TRUE_PATH → out = 15.0 * 2.0 = 30.0

    If _negated_preds is stale, the @P0 bra has flipped sense:
    TRUE becomes FALSE → out = 15.0 * 0.5 = 7.5 (WRONG)
    """
    import ctypes

    cuda = _get_cuda()
    ctx = ctypes.c_void_p()
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

    result = compile_ptx_source(FSETP_PRED_REUSE_PTX)
    cubin = result['fsetp_pred_reuse']

    mod = ctypes.c_void_p()
    err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
    assert err == 0, f"cuModuleLoadData failed: {err}"

    func = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func), mod, b'fsetp_pred_reuse')

    N = 1
    # a[0] = 15.0 (above threshold 10.0)
    a_data = struct.pack('f', 15.0)
    out_data = struct.pack('f', 0.0)

    d_out = ctypes.c_uint64()
    d_a = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4)
    cuda.cuMemAlloc_v2(ctypes.byref(d_a), 4)
    cuda.cuMemcpyHtoD_v2(d_a, a_data, 4)
    cuda.cuMemcpyHtoD_v2(d_out, out_data, 4)

    n_val = ctypes.c_int32(N)
    d_out_val = ctypes.c_uint64(d_out.value)
    d_a_val = ctypes.c_uint64(d_a.value)
    args = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.byref(d_out_val), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_a_val), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n_val), ctypes.c_void_p),
    )

    err = cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, None, args, None)
    assert err == 0, f"cuLaunchKernel failed: {err}"
    err = cuda.cuCtxSynchronize()
    assert err == 0, f"cuCtxSynchronize failed: {err}"

    buf = (ctypes.c_uint8 * 4)()
    cuda.cuMemcpyDtoH_v2(buf, d_out, 4)
    result_val = struct.unpack('f', bytes(buf))[0]

    cuda.cuMemFree_v2(d_out)
    cuda.cuMemFree_v2(d_a)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxSynchronize()
    cuda.cuCtxDestroy_v2(ctx)

    # 15.0 > 10.0 → TRUE path → 15.0 * 2.0 = 30.0
    # If stale negated pred: FALSE path → 15.0 * 0.5 = 7.5
    assert result_val == 30.0, (
        f"Expected 30.0 (TRUE path: 15.0 * 2.0), got {result_val}. "
        f"If 7.5: stale _negated_preds from ISETP flipped FSETP sense."
    )
