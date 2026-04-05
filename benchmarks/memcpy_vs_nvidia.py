"""
Memcpy Benchmark: device-to-device copy at peak bandwidth.

Each thread moves 4 bytes (u32) with coalesced warp accesses — the peak
bandwidth measurement for the toolchain.

Usage: python benchmarks/memcpy_vs_nvidia.py
"""
import ctypes
from statistics import median

from bench_util import (CUDAContext, compile_openptxas, compile_ptxas,
                        print_header, print_results)


MEMCPY_PTX = """.version 9.0
.target sm_120
.address_size 64

.visible .entry memcpy16(
    .param .u64 p_in,
    .param .u64 p_out,
    .param .u32 n_words
) {
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .reg .pred %p<2>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r4, [n_words];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r3;
    shl.b64 %rd1, %rd0, 2;
    ld.param.u64 %rd2, [p_in];
    ld.param.u64 %rd3, [p_out];
    add.s64 %rd4, %rd2, %rd1;
    add.s64 %rd5, %rd3, %rd1;

    ld.global.u32 %r10, [%rd4];
    st.global.u32 [%rd5], %r10;

DONE:
    ret;
}
"""


def run_kernel(ctx, func, n_vec, iters):
    nbytes = n_vec * 4
    d_in = ctx.alloc(nbytes)
    d_out = ctx.alloc(nbytes)
    ctx.memset_d8(d_in, 0xA5, nbytes)
    ctx.memset_d8(d_out, 0x00, nbytes)

    block = 256
    grid = (n_vec + block - 1) // block

    in_v = ctypes.c_uint64(d_in)
    out_v = ctypes.c_uint64(d_out)
    n_v = ctypes.c_uint32(n_vec)
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

    sample = ctx.copy_from(d_out, 256)
    ctx.free(d_in); ctx.free(d_out)
    return times, sample


def main():
    print_header("Memcpy Benchmark: OpenPTXas vs NVIDIA ptxas")

    # 256 MB
    total_bytes = 256 * 1024 * 1024
    n_vec = total_bytes // 4
    iters = 100

    print(f"  Buffer size:   {total_bytes/1e6:.0f} MB")
    print(f"  Word count:    {n_vec:,} x u32 (4 B each)")
    print(f"  Iterations:    {iters}")
    print(f"  Memory/launch: {2*total_bytes/1e6:.0f} MB (read + write)")
    print()

    compile_openptxas(MEMCPY_PTX)
    print("  [compile] OpenPTXas...")
    c_ours, t_ours = compile_openptxas(MEMCPY_PTX)
    print(f"            {t_ours*1000:.1f} ms, {len(c_ours)} bytes")
    print("  [compile] NVIDIA ptxas...")
    c_nvid, t_nvid = compile_ptxas(MEMCPY_PTX)
    print(f"            {t_nvid*1000:.1f} ms, {len(c_nvid)} bytes")
    print()

    ctx = CUDAContext()
    print("  [execute] OpenPTXas cubin...")
    assert ctx.load(c_ours)
    f = ctx.get_func("memcpy16")
    t_ours_list, r_ours = run_kernel(ctx, f, n_vec, iters)
    med_ours = median(t_ours_list)
    bw_ours = (2 * total_bytes) / (med_ours / 1000) / 1e9
    print(f"            median {med_ours*1000:.1f} us, {bw_ours:.1f} GB/s")

    print("  [execute] ptxas cubin...")
    assert ctx.load(c_nvid)
    f = ctx.get_func("memcpy16")
    t_nvid_list, r_nvid = run_kernel(ctx, f, n_vec, iters)
    med_nvid = median(t_nvid_list)
    bw_nvid = (2 * total_bytes) / (med_nvid / 1000) / 1e9
    print(f"            median {med_nvid*1000:.1f} us, {bw_nvid:.1f} GB/s")
    print()
    ctx.close()

    correct = all(b == 0xA5 for b in r_ours) and r_ours == r_nvid
    print(f"  [verify]  All 0xA5: {'YES' if all(b == 0xA5 for b in r_ours) else 'NO'}, "
          f"bit-identical: {'YES' if r_ours == r_nvid else 'NO'}")
    print()

    print_results(t_ours*1000, t_nvid*1000, c_ours, c_nvid,
                  med_ours*1000, med_nvid*1000,
                  "Mem bandwidth", bw_ours, bw_nvid, correct)
    return bw_ours, bw_nvid, correct


if __name__ == '__main__':
    main()
