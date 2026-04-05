"""
Scale Benchmark: OpenPTXas vs NVIDIA ptxas.

out[i] = alpha * x[i] + beta    (1 FMA + 1 read + 1 write per element)
alpha, beta are passed via pointers.

Usage: python benchmarks/scale_vs_nvidia.py
"""
import ctypes
import struct
from statistics import median

from bench_util import (CUDAContext, compile_openptxas, compile_ptxas,
                        print_header, print_results)


SCALE_PTX = """.version 9.0
.target sm_120
.address_size 64

.visible .entry scale(
    .param .u64 p_ab,
    .param .u64 p_x,
    .param .u64 p_out,
    .param .u32 n
) {
    .reg .b32 %r<8>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<8>;
    .reg .pred %p<2>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    // Load alpha, beta from two separate pointers (p_ab[0]=alpha, p_ab[1]=beta)
    ld.param.u64 %rd0, [p_ab];
    add.s64 %rd7, %rd0, 4;
    ld.global.f32 %f0, [%rd0];
    ld.global.f32 %f1, [%rd7];

    cvt.u64.u32 %rd1, %r3;
    shl.b64 %rd2, %rd1, 2;
    ld.param.u64 %rd3, [p_x];
    add.s64 %rd4, %rd3, %rd2;
    ld.global.f32 %f2, [%rd4];

    fma.rn.f32 %f3, %f0, %f2, %f1;

    ld.param.u64 %rd5, [p_out];
    add.s64 %rd6, %rd5, %rd2;
    st.global.f32 [%rd6], %f3;

DONE:
    ret;
}
"""


def run_kernel(ctx, func, n, iters):
    nbytes = n * 4
    d_ab = ctx.alloc(8)
    d_x = ctx.alloc(nbytes)
    d_out = ctx.alloc(nbytes)

    alpha, beta = 3.0, 1.0
    ctx.copy_to(d_ab, struct.pack('ff', alpha, beta))

    h = bytearray(nbytes)
    for i in range(n):
        struct.pack_into('f', h, i * 4, (i % 1000) * 0.01)
    ctx.copy_to(d_x, bytes(h))

    block = 256
    grid = (n + block - 1) // block

    ab_v = ctypes.c_uint64(d_ab)
    x_v = ctypes.c_uint64(d_x)
    out_v = ctypes.c_uint64(d_out)
    n_v = ctypes.c_uint32(n)
    args = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.byref(ab_v), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(x_v), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(out_v), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n_v), ctypes.c_void_p),
    )

    s_evt = ctx.event_create()
    e_evt = ctx.event_create()

    for _ in range(5):
        ctx.launch(func, grid, block, args)
    ctx.sync()

    times = []
    for _ in range(iters):
        ctx.event_record(s_evt)
        ctx.launch(func, grid, block, args)
        ctx.event_record(e_evt)
        ctx.sync()
        times.append(ctx.event_elapsed_ms(s_evt, e_evt))

    result = ctx.copy_from(d_out, nbytes)
    ctx.free(d_ab); ctx.free(d_x); ctx.free(d_out)
    return times, result, alpha, beta


def main():
    print_header("Scale (out = a*x + b) Benchmark: OpenPTXas vs NVIDIA ptxas")

    n = 16 * 1024 * 1024
    iters = 100

    print(f"  Array size:    {n:,} floats ({n*4/1e6:.1f} MB)")
    print(f"  Iterations:    {iters}")
    print(f"  Memory/launch: 2 x {n*4/1e6:.0f} MB (1 read + 1 write)")
    print()

    compile_openptxas(SCALE_PTX)
    print("  [compile] OpenPTXas...")
    c_ours, t_ours = compile_openptxas(SCALE_PTX)
    print(f"            {t_ours*1000:.1f} ms, {len(c_ours)} bytes")
    print("  [compile] NVIDIA ptxas...")
    c_nvid, t_nvid = compile_ptxas(SCALE_PTX)
    print(f"            {t_nvid*1000:.1f} ms, {len(c_nvid)} bytes")
    print()

    ctx = CUDAContext()
    print("  [execute] OpenPTXas cubin...")
    assert ctx.load(c_ours)
    f = ctx.get_func("scale")
    t_ours_list, r_ours, alpha, beta = run_kernel(ctx, f, n, iters)
    med_ours = median(t_ours_list)
    bw_ours = (2 * n * 4) / (med_ours / 1000) / 1e9
    print(f"            median {med_ours*1000:.1f} us, {bw_ours:.1f} GB/s")

    print("  [execute] ptxas cubin...")
    assert ctx.load(c_nvid)
    f = ctx.get_func("scale")
    t_nvid_list, r_nvid, _, _ = run_kernel(ctx, f, n, iters)
    med_nvid = median(t_nvid_list)
    bw_nvid = (2 * n * 4) / (med_nvid / 1000) / 1e9
    print(f"            median {med_nvid*1000:.1f} us, {bw_nvid:.1f} GB/s")
    print()
    ctx.close()

    ok_math = True
    for i in range(0, min(n, 100000), 1371):
        got = struct.unpack_from('f', r_ours, i * 4)[0]
        x = (i % 1000) * 0.01
        exp = alpha * x + beta
        if abs(got - exp) > 1e-3:
            ok_math = False
            break
    correct = ok_math and (r_ours == r_nvid)
    print(f"  [verify]  Math: {'PASS' if ok_math else 'FAIL'}, "
          f"bit-identical: {'YES' if r_ours == r_nvid else 'NO'}")
    print()

    print_results(t_ours*1000, t_nvid*1000, c_ours, c_nvid,
                  med_ours*1000, med_nvid*1000,
                  "Mem bandwidth", bw_ours, bw_nvid, correct)
    return bw_ours, bw_nvid, correct


if __name__ == '__main__':
    main()
