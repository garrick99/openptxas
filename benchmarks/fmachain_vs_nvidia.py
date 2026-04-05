"""
FMA Chain Benchmark: OpenPTXas vs NVIDIA ptxas.

Each thread performs a long unrolled chain of FMA operations per element.
This stresses FP throughput rather than memory bandwidth.

out[i] = fma(x[i], k, fma(x[i], k, ... fma(x[i], k, 0)))  (N_FMA times)

Usage: python benchmarks/fmachain_vs_nvidia.py
"""
import ctypes
import struct
from statistics import median

from bench_util import (CUDAContext, compile_openptxas, compile_ptxas,
                        print_header, print_results)


N_FMA = 32  # FMAs per element


def gen_ptx(n_fma):
    # Each FMA: acc = x * k + acc
    header = """.version 9.0
.target sm_120
.address_size 64

.visible .entry fma_chain(
    .param .u64 p_x,
    .param .u64 p_out,
    .param .u32 n
) {
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .pred %p<2>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r3;
    shl.b64 %rd1, %rd0, 2;
    ld.param.u64 %rd2, [p_x];
    add.s64 %rd3, %rd2, %rd1;
    ld.global.f32 %f0, [%rd3];

    mov.f32 %f1, 0f3F800000;         // k = 1.0
    mov.f32 %f2, 0f00000000;         // acc = 0
"""
    body = ""
    for _ in range(n_fma):
        body += "    fma.rn.f32 %f2, %f0, %f1, %f2;\n"
    footer = """
    ld.param.u64 %rd4, [p_out];
    add.s64 %rd5, %rd4, %rd1;
    st.global.f32 [%rd5], %f2;

DONE:
    ret;
}
"""
    return header + body + footer


FMACHAIN_PTX = gen_ptx(N_FMA)


def run_kernel(ctx, func, n, iters):
    nbytes = n * 4
    d_x = ctx.alloc(nbytes)
    d_out = ctx.alloc(nbytes)

    h = bytearray(nbytes)
    for i in range(n):
        struct.pack_into('f', h, i * 4, 0.5)
    ctx.copy_to(d_x, bytes(h))

    block = 256
    grid = (n + block - 1) // block

    x_v = ctypes.c_uint64(d_x)
    out_v = ctypes.c_uint64(d_out)
    n_v = ctypes.c_uint32(n)
    args = (ctypes.c_void_p * 3)(
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
    ctx.free(d_x); ctx.free(d_out)
    return times, result


def main():
    print_header(f"FMA Chain ({N_FMA} FMAs/elt) Benchmark: OpenPTXas vs NVIDIA ptxas")

    n = 16 * 1024 * 1024
    iters = 100
    flops_per_launch = n * N_FMA * 2  # 2 flops per FMA

    print(f"  Array size:    {n:,} floats ({n*4/1e6:.1f} MB)")
    print(f"  FMAs/element:  {N_FMA}")
    print(f"  Iterations:    {iters}")
    print(f"  FLOPs/launch:  {flops_per_launch/1e9:.2f} G")
    print()

    compile_openptxas(FMACHAIN_PTX)
    print("  [compile] OpenPTXas...")
    c_ours, t_ours = compile_openptxas(FMACHAIN_PTX)
    print(f"            {t_ours*1000:.1f} ms, {len(c_ours)} bytes")
    print("  [compile] NVIDIA ptxas...")
    c_nvid, t_nvid = compile_ptxas(FMACHAIN_PTX)
    print(f"            {t_nvid*1000:.1f} ms, {len(c_nvid)} bytes")
    print()

    ctx = CUDAContext()
    print("  [execute] OpenPTXas cubin...")
    assert ctx.load(c_ours)
    f = ctx.get_func("fma_chain")
    t_ours_list, r_ours = run_kernel(ctx, f, n, iters)
    med_ours = median(t_ours_list)
    gf_ours = flops_per_launch / (med_ours / 1000) / 1e9
    print(f"            median {med_ours*1000:.1f} us, {gf_ours:.0f} GFLOPS")

    print("  [execute] ptxas cubin...")
    assert ctx.load(c_nvid)
    f = ctx.get_func("fma_chain")
    t_nvid_list, r_nvid = run_kernel(ctx, f, n, iters)
    med_nvid = median(t_nvid_list)
    gf_nvid = flops_per_launch / (med_nvid / 1000) / 1e9
    print(f"            median {med_nvid*1000:.1f} us, {gf_nvid:.0f} GFLOPS")
    print()
    ctx.close()

    expected = 0.5 * N_FMA  # x=0.5, k=1.0, acc starts at 0
    got = struct.unpack_from('f', r_ours, 0)[0]
    ok_math = abs(got - expected) < 1e-4
    correct = ok_math and (r_ours == r_nvid)
    print(f"  [verify]  Result={got}, expected={expected}: "
          f"{'PASS' if ok_math else 'FAIL'}, bit-identical: "
          f"{'YES' if r_ours == r_nvid else 'NO'}")
    print()

    print_results(t_ours*1000, t_nvid*1000, c_ours, c_nvid,
                  med_ours*1000, med_nvid*1000,
                  "GFLOPS", gf_ours, gf_nvid, correct, perf_fmt="{:.0f}")
    return gf_ours, gf_nvid, correct


if __name__ == '__main__':
    main()
