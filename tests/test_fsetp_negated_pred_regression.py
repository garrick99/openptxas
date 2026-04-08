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
@pytest.mark.gpu
def test_fsetp_pred_reuse_gpu_correctness(cuda_ctx):
    """
    Thread 0: a[0] = 15.0, threshold = 10.0
    15.0 > 10.0 is TRUE -> should take TRUE_PATH -> out = 15.0 * 2.0 = 30.0
    """
    result = compile_ptx_source(FSETP_PRED_REUSE_PTX)
    cubin = result['fsetp_pred_reuse']
    assert cuda_ctx.load(cubin), "cuModuleLoadData failed"
    func = cuda_ctx.get_func('fsetp_pred_reuse')

    d_out = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_out, struct.pack('f', 0.0))
    d_a = cuda_ctx.alloc(4); cuda_ctx.copy_to(d_a, struct.pack('f', 15.0))
    cuda_ctx.launch(func, (1,1,1), (1,1,1), [d_out, d_a, 1])
    assert cuda_ctx.sync() == 0
    result_val = struct.unpack('f', cuda_ctx.copy_from(d_out, 4))[0]
    cuda_ctx.free(d_out); cuda_ctx.free(d_a)

    assert result_val == 30.0, (
        f"Expected 30.0 (TRUE path: 15.0 * 2.0), got {result_val}. "
        f"If 7.5: stale _negated_preds from ISETP flipped FSETP sense."
    )
