"""FB-1 reduce_sum incremental probes."""
import ctypes
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sass.pipeline import compile_function, _extract_ptxas_metadata
from ptx.parser import parse

# Step 1: lane/wid guard added
STEP1 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry step1(
    .param .u64 data_ptr, .param .u64 output_ptr, .param .u64 n)
{
    .reg .b32 %r<20>; .reg .b64 %rd<8>; .reg .pred %p<2>;
    ld.param.u64 %rd0, [data_ptr];
    ld.param.u64 %rd1, [output_ptr];
    mov.u32 %r0, %tid.x;
    cvt.u64.u32 %rd2, %r0;
    shl.b64 %rd3, %rd2, 3;
    add.u64 %rd3, %rd0, %rd3;
    ld.global.u64 %rd4, [%rd3];

    mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r4, %r2, 16, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r5, %r3, 16, 31, 0xffffffff;
    mov.b64 %rd5, {%r4, %r5}; add.u64 %rd4, %rd4, %rd5; mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r6, %r2, 8, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r7, %r3, 8, 31, 0xffffffff;
    mov.b64 %rd5, {%r6, %r7}; add.u64 %rd4, %rd4, %rd5; mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r8, %r2, 4, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r9, %r3, 4, 31, 0xffffffff;
    mov.b64 %rd5, {%r8, %r9}; add.u64 %rd4, %rd4, %rd5; mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r10, %r2, 2, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r11, %r3, 2, 31, 0xffffffff;
    mov.b64 %rd5, {%r10, %r11}; add.u64 %rd4, %rd4, %rd5; mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r12, %r2, 1, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r13, %r3, 1, 31, 0xffffffff;
    mov.b64 %rd5, {%r12, %r13}; add.u64 %rd4, %rd4, %rd5;

    and.b32 %r16, %r0, 31; shr.u32 %r17, %r0, 5;
    setp.ne.u32 %p0, %r16, 0; @%p0 ret;
    setp.ne.u32 %p1, %r17, 0; @%p1 ret;
    st.global.u64 [%rd1], %rd4;
    ret;
}
"""

# Step 2: gid<n guard + lane/wid guard
STEP2 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry step2(
    .param .u64 data_ptr, .param .u64 output_ptr, .param .u64 n)
{
    .reg .b32 %r<20>; .reg .b64 %rd<8>; .reg .pred %p<2>;
    ld.param.u64 %rd0, [data_ptr];
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [n];
    mov.u32 %r0, %tid.x;
    cvt.u64.u32 %rd3, %r0;
    mov.u64 %rd4, 0;
    setp.lt.u64 %p0, %rd3, %rd2;
    @!%p0 bra REDUCE;
    shl.b64 %rd5, %rd3, 3;
    add.u64 %rd5, %rd0, %rd5;
    ld.global.u64 %rd4, [%rd5];
REDUCE:
    mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r4, %r2, 16, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r5, %r3, 16, 31, 0xffffffff;
    mov.b64 %rd5, {%r4, %r5}; add.u64 %rd4, %rd4, %rd5; mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r6, %r2, 8, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r7, %r3, 8, 31, 0xffffffff;
    mov.b64 %rd5, {%r6, %r7}; add.u64 %rd4, %rd4, %rd5; mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r8, %r2, 4, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r9, %r3, 4, 31, 0xffffffff;
    mov.b64 %rd5, {%r8, %r9}; add.u64 %rd4, %rd4, %rd5; mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r10, %r2, 2, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r11, %r3, 2, 31, 0xffffffff;
    mov.b64 %rd5, {%r10, %r11}; add.u64 %rd4, %rd4, %rd5; mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r12, %r2, 1, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r13, %r3, 1, 31, 0xffffffff;
    mov.b64 %rd5, {%r12, %r13}; add.u64 %rd4, %rd4, %rd5;

    and.b32 %r16, %r0, 31; shr.u32 %r17, %r0, 5;
    setp.ne.u32 %p0, %r16, 0; @%p0 ret;
    setp.ne.u32 %p1, %r17, 0; @%p1 ret;
    st.global.u64 [%rd1], %rd4;
    ret;
}
"""

# Step 3: 5 params
STEP3 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry step3(
    .param .u64 data_ptr, .param .u64 data_len,
    .param .u64 output_ptr, .param .u64 output_len, .param .u64 n)
{
    .reg .b32 %r<20>; .reg .b64 %rd<8>; .reg .pred %p<2>;
    ld.param.u64 %rd0, [data_ptr];
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [n];
    mov.u32 %r0, %tid.x;
    cvt.u64.u32 %rd3, %r0;
    mov.u64 %rd4, 0;
    setp.lt.u64 %p0, %rd3, %rd2;
    @!%p0 bra REDUCE;
    shl.b64 %rd5, %rd3, 3;
    add.u64 %rd5, %rd0, %rd5;
    ld.global.u64 %rd4, [%rd5];
REDUCE:
    mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r4, %r2, 16, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r5, %r3, 16, 31, 0xffffffff;
    mov.b64 %rd5, {%r4, %r5}; add.u64 %rd4, %rd4, %rd5; mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r6, %r2, 8, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r7, %r3, 8, 31, 0xffffffff;
    mov.b64 %rd5, {%r6, %r7}; add.u64 %rd4, %rd4, %rd5; mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r8, %r2, 4, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r9, %r3, 4, 31, 0xffffffff;
    mov.b64 %rd5, {%r8, %r9}; add.u64 %rd4, %rd4, %rd5; mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r10, %r2, 2, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r11, %r3, 2, 31, 0xffffffff;
    mov.b64 %rd5, {%r10, %r11}; add.u64 %rd4, %rd4, %rd5; mov.b64 {%r2, %r3}, %rd4;
    shfl.sync.bfly.b32 %r12, %r2, 1, 31, 0xffffffff;
    shfl.sync.bfly.b32 %r13, %r3, 1, 31, 0xffffffff;
    mov.b64 %rd5, {%r12, %r13}; add.u64 %rd4, %rd4, %rd5;

    and.b32 %r16, %r0, 31; shr.u32 %r17, %r0, 5;
    setp.ne.u32 %p0, %r16, 0; @%p0 ret;
    setp.ne.u32 %p1, %r17, 0; @%p1 ret;
    st.global.u64 [%rd1], %rd4;
    ret;
}
"""


def init_cuda():
    cuda = ctypes.CDLL('nvcuda.dll')
    for attr, args, ret in [
        ('cuInit', [ctypes.c_uint], ctypes.c_int),
        ('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int),
        ('cuCtxCreate_v2', [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int], ctypes.c_int),
        ('cuModuleLoadData', [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
        ('cuModuleGetFunction', [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
        ('cuMemAlloc_v2', [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t], ctypes.c_int),
        ('cuMemcpyHtoD_v2', [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
        ('cuMemcpyDtoH_v2', [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t], ctypes.c_int),
        ('cuLaunchKernel', [ctypes.c_void_p]+[ctypes.c_uint]*6+[ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p], ctypes.c_int),
        ('cuCtxSynchronize', [], ctypes.c_int),
        ('cuCtxDestroy_v2', [ctypes.c_void_p], ctypes.c_int),
    ]:
        f = getattr(cuda, attr); f.argtypes = args; f.restype = ret
    return cuda


def main():
    cuda = init_cuda()
    assert cuda.cuInit(0) == 0
    dev = ctypes.c_int(); assert cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
    ctx = ctypes.c_void_p(); assert cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value) == 0

    N = 32
    data = np.arange(1, N+1, dtype=np.uint64)
    expected = 528
    d_data = ctypes.c_uint64(); d_out = ctypes.c_uint64()
    assert cuda.cuMemAlloc_v2(ctypes.byref(d_data), N*8) == 0
    assert cuda.cuMemAlloc_v2(ctypes.byref(d_out), 8) == 0
    cuda.cuMemcpyHtoD_v2(d_data.value, data.ctypes.data, N*8)

    for label, ptx_src, kname, nparams in [
        ('Step1 (lane/wid guard)', STEP1, 'step1', 3),
        ('Step2 (+gid<n guard)', STEP2, 'step2', 3),
        ('Step3 (+5 params)', STEP3, 'step3', 5),
    ]:
        meta = _extract_ptxas_metadata(ptx_src)
        km = meta.get(kname, {})
        mod = parse(ptx_src)
        fn = mod.functions[0]
        cubin = compile_function(fn, verbose=False, ptxas_meta=km, sm_version=120)

        m = ctypes.c_void_p()
        r = cuda.cuModuleLoadData(ctypes.byref(m), cubin)
        if r != 0:
            print(f'{label}: LOAD FAILED ({r})')
            continue

        func = ctypes.c_void_p()
        assert cuda.cuModuleGetFunction(ctypes.byref(func), m, kname.encode()) == 0
        cuda.cuMemcpyHtoD_v2(d_out.value, np.zeros(1, dtype=np.uint64).ctypes.data, 8)

        if nparams == 3:
            p = [ctypes.c_uint64(d_data.value), ctypes.c_uint64(d_out.value), ctypes.c_uint64(N)]
            args = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in p])
        else:
            p = [ctypes.c_uint64(d_data.value), ctypes.c_uint64(N),
                 ctypes.c_uint64(d_out.value), ctypes.c_uint64(1), ctypes.c_uint64(N)]
            args = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in p])

        assert cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None) == 0
        r = cuda.cuCtxSynchronize()
        if r == 0:
            out = np.zeros(1, dtype=np.uint64)
            cuda.cuMemcpyDtoH_v2(out.ctypes.data, d_out.value, 8)
            status = 'PASS' if out[0] == expected else 'FAIL'
            print(f'{label}: got={out[0]} expected={expected} -> {status}')
        else:
            print(f'{label}: GPU ERROR {r}')

    cuda.cuCtxDestroy_v2(ctx)


if __name__ == '__main__':
    main()
