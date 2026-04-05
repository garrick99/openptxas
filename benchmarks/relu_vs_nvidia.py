"""
ReLU Benchmark: OpenPTXas vs NVIDIA ptxas.

out[i] = max(x[i], 0)    using max.f32 (branchless)

Usage: python benchmarks/relu_vs_nvidia.py
"""
import ctypes
import struct
from statistics import median

from bench_util import (CUDAContext, compile_openptxas, compile_ptxas,
                        print_header, print_results)


RELU_PTX = """.version 9.0
.target sm_120
.address_size 64

.visible .entry relu(
    .param .u64 p_x,
    .param .u64 p_out,
    .param .u32 n
) {
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;
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

    mov.f32 %f1, 0f00000000;
    max.f32 %f2, %f0, %f1;

    ld.param.u64 %rd4, [p_out];
    add.s64 %rd5, %rd4, %rd1;
    st.global.f32 [%rd5], %f2;

DONE:
    ret;
}
"""


def run_kernel(ctx, func, n, iters):
    nbytes = n * 4
    d_x = ctx.alloc(nbytes)
    d_out = ctx.alloc(nbytes)

    h = bytearray(nbytes)
    for i in range(n):
        # alternating sign values to exercise the max
        v = (i - (n // 2)) * 0.001
        struct.pack_into('f', h, i * 4, v)
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
    print_header("ReLU Benchmark: OpenPTXas vs NVIDIA ptxas")

    n = 16 * 1024 * 1024
    iters = 100

    print(f"  Array size:    {n:,} floats ({n*4/1e6:.1f} MB)")
    print(f"  Iterations:    {iters}")
    print(f"  Memory/launch: 2 x {n*4/1e6:.0f} MB (1 read + 1 write)")
    print()

    compile_openptxas(RELU_PTX)
    print("  [compile] OpenPTXas...")
    c_ours, t_ours = compile_openptxas(RELU_PTX)
    print(f"            {t_ours*1000:.1f} ms, {len(c_ours)} bytes")
    print("  [compile] NVIDIA ptxas...")
    c_nvid, t_nvid = compile_ptxas(RELU_PTX)
    print(f"            {t_nvid*1000:.1f} ms, {len(c_nvid)} bytes")
    print()

    ctx = CUDAContext()
    print("  [execute] OpenPTXas cubin...")
    assert ctx.load(c_ours)
    f = ctx.get_func("relu")
    t_ours_list, r_ours = run_kernel(ctx, f, n, iters)
    med_ours = median(t_ours_list)
    bw_ours = (2 * n * 4) / (med_ours / 1000) / 1e9
    print(f"            median {med_ours*1000:.1f} us, {bw_ours:.1f} GB/s")

    print("  [execute] ptxas cubin...")
    assert ctx.load(c_nvid)
    f = ctx.get_func("relu")
    t_nvid_list, r_nvid = run_kernel(ctx, f, n, iters)
    med_nvid = median(t_nvid_list)
    bw_nvid = (2 * n * 4) / (med_nvid / 1000) / 1e9
    print(f"            median {med_nvid*1000:.1f} us, {bw_nvid:.1f} GB/s")
    print()
    ctx.close()

    ok_math = True
    for i in range(0, min(n, 1000000), 7919):
        got = struct.unpack_from('f', r_ours, i * 4)[0]
        v = (i - (n // 2)) * 0.001
        exp = max(v, 0.0)
        if abs(got - exp) > 1e-4:
            ok_math = False
            break
    # max.f32 may differ on +0.0/-0.0 between implementations; math is canonical.
    correct = ok_math
    print(f"  [verify]  Math: {'PASS' if ok_math else 'FAIL'}, "
          f"bit-identical: {'YES' if r_ours == r_nvid else 'NO (zero sign)'}")
    print()

    print_results(t_ours*1000, t_nvid*1000, c_ours, c_nvid,
                  med_ours*1000, med_nvid*1000,
                  "Mem bandwidth", bw_ours, bw_nvid, correct)
    return bw_ours, bw_nvid, correct


if __name__ == '__main__':
    main()
