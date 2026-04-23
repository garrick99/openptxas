"""R34: instrument s2_fail to capture post-add %rd1 pair at STG time.

The key insight: in s2_fail, `%rd1` goes UR-routed via R31 (preamble
LDCU.64), and the post-add pair `%__r31_rd1_0` is what the final STG
would use as address.  To avoid disturbing that routing, we must NOT
use %rd1 as a non-add/non-base source.

Instead we split the post-add pair via `mov.b64 {%r_lo, %r_hi}, %rd1`
(which AFTER R31 rename reads %__r31_rd1_0, a fresh GPR pair).  That
preserves the exact s2_fail shape — %rd1 remains UR-routed, %rd2
stays as the offset, and %__r31_rd1_0 is the STG address candidate.

Debug layout (32 bytes):
  [0]  = post-add pair LO (= STG target lo in s2_fail) — should = d_out_lo
  [4]  = post-add pair HI (= STG target hi in s2_fail) — should = d_out_hi
  [8]  = LDG result (%r0) — should = 777
"""
from __future__ import annotations
import ctypes, struct, sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')
from ptx.parser import parse
from sass.pipeline import compile_function


_PTX_DEBUG = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry s2_debug(.param .u64 in, .param .u64 out, .param .u64 debug) {
    .reg .b32 %r<10>;
    .reg .b64 %rd<12>;
    .reg .pred %p<1>;
    ld.param.u64 %rd0, [in];
    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd3, [debug];
    ld.global.u32 %r0, [%rd0];
    mov.u32 %r1, %tid.x;
    setp.eq.u32 %p0, %r1, 0;
    @!%p0 ret;
    mov.u32 %r2, %ctaid.x;
    shl.b32 %r3, %r2, 2;
    cvt.u64.u32 %rd2, %r3;
    // split %rd2 (offset pair) for capture
    mov.b64 {%r6, %r7}, %rd2;
    add.u64 %rd1, %rd1, %rd2;
    mov.b64 {%r4, %r5}, %rd1;
    // explicit distinct addresses for captures
    add.u64 %rd4, %rd3, 0;
    add.u64 %rd5, %rd3, 4;
    add.u64 %rd6, %rd3, 8;
    add.u64 %rd7, %rd3, 12;
    add.u64 %rd8, %rd3, 16;
    add.u64 %rd9, %rd3, 20;
    st.global.u32 [%rd4], %r4;   // [0]  post-add lo
    st.global.u32 [%rd5], %r5;   // [4]  post-add hi
    st.global.u32 [%rd6], %r0;   // [8]  LDG result
    st.global.u32 [%rd7], %r6;   // [12] offset lo
    st.global.u32 [%rd8], %r7;   // [16] offset hi
    st.global.u32 [%rd9], %r2;   // [20] ctaid.x
    ret;
}
"""


def main():
    cuda = ctypes.WinDLL('nvcuda')
    cuda.cuInit(0)
    dev = ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p(); cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

    try:
        cubin = compile_function(parse(_PTX_DEBUG).functions[0],
                                 verbose=False, sm_version=120)
        mod = ctypes.c_void_p()
        err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
        if err != 0:
            print(f'LOAD ERROR: {err}')
            return
        func = ctypes.c_void_p()
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, b's2_debug')

        d_in = ctypes.c_uint64()
        d_out = ctypes.c_uint64()
        d_debug = ctypes.c_uint64()
        cuda.cuMemAlloc_v2(ctypes.byref(d_in), 4)
        cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4)
        cuda.cuMemAlloc_v2(ctypes.byref(d_debug), 32)
        cuda.cuMemcpyHtoD_v2(d_in, struct.pack('<I', 777), 4)
        cuda.cuMemcpyHtoD_v2(d_out, struct.pack('<I', 0), 4)
        cuda.cuMemcpyHtoD_v2(d_debug, b'\xaa' * 32, 32)

        a_in = ctypes.c_uint64(d_in.value)
        a_out = ctypes.c_uint64(d_out.value)
        a_debug = ctypes.c_uint64(d_debug.value)
        argv = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.byref(a_in), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(a_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(a_debug), ctypes.c_void_p),
        )
        cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, None, argv, None)
        err = cuda.cuCtxSynchronize()
        print(f'sync={err}')

        buf = ctypes.create_string_buffer(32)
        cuda.cuMemcpyDtoH_v2(buf, d_debug, 32)
        rd1_lo, rd1_hi, r0, off_lo, off_hi, ctaid = struct.unpack_from('<IIIIII', buf.raw, 0)

        d_out_lo = a_out.value & 0xFFFFFFFF
        d_out_hi = (a_out.value >> 32) & 0xFFFFFFFF
        print(f'd_in   =0x{a_in.value:016x}')
        print(f'd_out  =0x{a_out.value:016x}   lo=0x{d_out_lo:08x} hi=0x{d_out_hi:08x}')
        print(f'd_debug=0x{a_debug.value:016x}')
        print(f'---')
        def _m(v, e): return "MATCH" if v == e else "MISMATCH"
        print(f'ctaid.x                          : 0x{ctaid:08x}   expected=0x00000000   {_m(ctaid, 0)}')
        print(f'offset LO (= shl result)         : 0x{off_lo:08x}   expected=0x00000000   {_m(off_lo, 0)}')
        print(f'offset HI (= 0)                  : 0x{off_hi:08x}   expected=0x00000000   {_m(off_hi, 0)}')
        print(f'LDG result %r0                   : 0x{r0:08x}   expected=0x00000309   {_m(r0, 777)}')
        print(f'post-add pair LO (= STG addr lo) : 0x{rd1_lo:08x}   expected=0x{d_out_lo:08x}   {_m(rd1_lo, d_out_lo)}')
        print(f'post-add pair HI (= STG addr hi) : 0x{rd1_hi:08x}   expected=0x{d_out_hi:08x}   {_m(rd1_hi, d_out_hi)}')
    finally:
        cuda.cuCtxDestroy_v2(ctx)


if __name__ == '__main__':
    main()
