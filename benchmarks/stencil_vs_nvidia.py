"""
1D 5-point stencil benchmark:
  out[i] = (in[i-2] + in[i-1] + in[i] + in[i+1] + in[i+2]) * 0.2

Input is padded with 2 ghost cells on each side so every thread reads
uniformly, without branching for halos. The L1 texture/cache coalesces
the 5 reads per element into ~1 transaction per thread.

Usage: python benchmarks/stencil_vs_nvidia.py
"""
import ctypes
import struct
from statistics import median

from bench_util import (CUDAContext, compile_openptxas, compile_ptxas,
                        print_header, print_results)


# Block size 256. smem holds 256 + 4 = 260 floats.
STENCIL_PTX = """.version 9.0
.target sm_120
.address_size 64

// Input padded with 2 ghost cells each side. p_in points to start of ghosts.
// Thread gi writes out[gi] = sum(in_padded[gi .. gi+4]) * 0.2
.visible .entry stencil(
    .param .u64 p_in,
    .param .u64 p_out,
    .param .u32 n
) {
    .reg .b32 %r<16>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<16>;
    .reg .pred %p<2>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    // Fully-independent address computations (workaround for OpenPTXas
    // register-alloc issue with shared base regs across ld.global).
    add.s32 %r5, %r3, 1;
    add.s32 %r6, %r3, 2;
    add.s32 %r7, %r3, 3;
    add.s32 %r8, %r3, 4;
    cvt.u64.u32 %rd0, %r3;
    cvt.u64.u32 %rd1, %r5;
    cvt.u64.u32 %rd2, %r6;
    cvt.u64.u32 %rd3, %r7;
    cvt.u64.u32 %rd4, %r8;
    shl.b64 %rd5, %rd0, 2;
    shl.b64 %rd6, %rd1, 2;
    shl.b64 %rd7, %rd2, 2;
    shl.b64 %rd8, %rd3, 2;
    shl.b64 %rd9, %rd4, 2;
    ld.param.u64 %rd10, [p_in];
    add.s64 %rd11, %rd10, %rd5;
    add.s64 %rd12, %rd10, %rd6;
    add.s64 %rd13, %rd10, %rd7;
    add.s64 %rd14, %rd10, %rd8;
    add.s64 %rd15, %rd10, %rd9;
    ld.global.f32 %f0, [%rd11];
    ld.global.f32 %f1, [%rd12];
    ld.global.f32 %f2, [%rd13];
    ld.global.f32 %f3, [%rd14];
    ld.global.f32 %f4, [%rd15];

    add.f32 %f5, %f0, %f1;
    add.f32 %f6, %f5, %f2;
    add.f32 %f7, %f6, %f3;
    add.f32 %f8, %f7, %f4;
    mul.f32 %f9, %f8, 0f3E4CCCCD;

    ld.param.u64 %rd16, [p_out];
    add.s64 %rd17, %rd16, %rd5;
    st.global.f32 [%rd17], %f9;

DONE:
    ret;
}
"""


def run_kernel(ctx, func, n, iters):
    nbytes = n * 4
    # Input padded with 2 ghost cells on each side.
    in_elts = n + 4
    d_in = ctx.alloc(in_elts * 4)
    d_out = ctx.alloc(nbytes)

    h_in = bytearray(in_elts * 4)
    # Ghosts are zero, interior holds data.
    for i in range(n):
        struct.pack_into('f', h_in, (i + 2) * 4, (i % 1000) * 0.01)
    ctx.copy_to(d_in, bytes(h_in))
    ctx.memset_d8(d_out, 0, nbytes)

    block = 256
    grid = (n + block - 1) // block

    in_v = ctypes.c_uint64(d_in)
    out_v = ctypes.c_uint64(d_out)
    n_v = ctypes.c_uint32(n)
    args = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.byref(in_v), ctypes.c_void_p),
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
    ctx.free(d_in); ctx.free(d_out)
    return times, result


def verify(result_bytes, n):
    # Input is padded with 2 ghost zeros; out[i] uses in_padded[i..i+4] where in_padded[j+2]=in[j]
    def in_val(i):
        if i < 0 or i >= n:
            return 0.0
        return (i % 1000) * 0.01
    for i in range(100, min(n - 100, 100000), 313):
        got = struct.unpack_from('f', result_bytes, i * 4)[0]
        exp = (in_val(i-2) + in_val(i-1) + in_val(i) + in_val(i+1) + in_val(i+2)) * 0.2
        if abs(got - exp) > 1e-3:
            return False
    return True


def main():
    print_header("Stencil (1D, 5-point) Benchmark: OpenPTXas vs NVIDIA ptxas")

    n = 16 * 1024 * 1024
    iters = 100

    print(f"  Array size:    {n:,} floats ({n*4/1e6:.1f} MB each)")
    print(f"  Iterations:    {iters}")
    print(f"  Memory/launch: ~{2*n*4/1e6:.0f} MB effective (5-read + 1-write, L1 cached)")
    print(f"  Padding:       2 ghost cells per side")
    print()

    compile_openptxas(STENCIL_PTX)
    print("  [compile] OpenPTXas...")
    c_ours, t_ours = compile_openptxas(STENCIL_PTX)
    print(f"            {t_ours*1000:.1f} ms, {len(c_ours)} bytes")
    print("  [compile] NVIDIA ptxas...")
    c_nvid, t_nvid = compile_ptxas(STENCIL_PTX)
    print(f"            {t_nvid*1000:.1f} ms, {len(c_nvid)} bytes")
    print()

    ctx = CUDAContext()
    print("  [execute] OpenPTXas cubin...")
    assert ctx.load(c_ours)
    f = ctx.get_func("stencil")
    t_ours_list, r_ours = run_kernel(ctx, f, n, iters)
    med_ours = median(t_ours_list)
    bw_ours = (2 * n * 4) / (med_ours / 1000) / 1e9
    print(f"            median {med_ours*1000:.1f} us, {bw_ours:.1f} GB/s")

    print("  [execute] ptxas cubin...")
    assert ctx.load(c_nvid)
    f = ctx.get_func("stencil")
    t_nvid_list, r_nvid = run_kernel(ctx, f, n, iters)
    med_nvid = median(t_nvid_list)
    bw_nvid = (2 * n * 4) / (med_nvid / 1000) / 1e9
    print(f"            median {med_nvid*1000:.1f} us, {bw_nvid:.1f} GB/s")
    print()
    ctx.close()

    ok = verify(r_ours, n)
    correct = ok and (r_ours == r_nvid)
    print(f"  [verify]  Math: {'PASS' if ok else 'FAIL'}, bit-identical: {'YES' if r_ours == r_nvid else 'NO'}")
    print()

    print_results(t_ours*1000, t_nvid*1000, c_ours, c_nvid,
                  med_ours*1000, med_nvid*1000,
                  "Mem bandwidth", bw_ours, bw_nvid, correct)
    return bw_ours, bw_nvid, correct


if __name__ == '__main__':
    main()
