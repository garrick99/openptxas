"""
SAXPY Benchmark: OpenPTXas vs NVIDIA ptxas.

y[i] = a * x[i] + y[i]  (FMA)
The scalar 'a' is passed via a pointer (scalar float params are unreliable).

Usage: python benchmarks/saxpy_vs_nvidia.py
"""
import ctypes
import struct
from statistics import median

from bench_util import (CUDAContext, compile_openptxas, compile_ptxas,
                        print_header, print_results)


SAXPY_PTX = """.version 9.0
.target sm_120
.address_size 64

.visible .entry saxpy(
    .param .u64 p_a,
    .param .u64 p_x,
    .param .u64 p_y,
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

    cvt.u64.u32 %rd0, %r3;
    shl.b64 %rd1, %rd0, 2;
    ld.param.u64 %rd2, [p_a];
    ld.param.u64 %rd3, [p_x];
    ld.param.u64 %rd4, [p_y];
    add.s64 %rd5, %rd3, %rd1;
    add.s64 %rd6, %rd4, %rd1;

    ld.global.f32 %f1, [%rd2];
    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f3, [%rd6];
    fma.rn.f32 %f4, %f1, %f2, %f3;
    st.global.f32 [%rd6], %f4;

DONE:
    ret;
}
"""


def run_kernel(ctx, func, n, iters):
    nbytes = n * 4
    d_a = ctx.alloc(4)
    d_x = ctx.alloc(nbytes)
    d_y = ctx.alloc(nbytes)

    a_val = 2.5
    ctx.copy_to(d_a, struct.pack('f', a_val))

    x_host = bytearray(nbytes)
    y_host = bytearray(nbytes)
    for i in range(n):
        struct.pack_into('f', x_host, i * 4, (i % 1024) * 0.001)
        struct.pack_into('f', y_host, i * 4, (i % 512) * 0.002)

    block = 256
    grid = (n + block - 1) // block

    da_v = ctypes.c_uint64(d_a)
    dx_v = ctypes.c_uint64(d_x)
    dy_v = ctypes.c_uint64(d_y)
    n_v = ctypes.c_uint32(n)
    args = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.byref(da_v), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(dx_v), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(dy_v), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n_v), ctypes.c_void_p),
    )

    s_evt = ctx.event_create()
    e_evt = ctx.event_create()

    # Warmup: reload fresh y each time to avoid accumulation
    ctx.copy_to(d_x, bytes(x_host))
    ctx.copy_to(d_y, bytes(y_host))
    for _ in range(5):
        ctx.launch(func, grid, block, args)
    ctx.sync()

    # Re-init and time
    ctx.copy_to(d_y, bytes(y_host))
    times = []
    for _ in range(iters):
        ctx.copy_to(d_y, bytes(y_host))
        ctx.event_record(s_evt)
        ctx.launch(func, grid, block, args)
        ctx.event_record(e_evt)
        ctx.sync()
        times.append(ctx.event_elapsed_ms(s_evt, e_evt))

    result = ctx.copy_from(d_y, nbytes)
    ctx.free(d_a); ctx.free(d_x); ctx.free(d_y)
    return times, result, a_val


def verify(result_bytes, n, a_val):
    for i in range(0, min(n, 4096), 37):
        got = struct.unpack_from('f', result_bytes, i * 4)[0]
        x = (i % 1024) * 0.001
        y = (i % 512) * 0.002
        exp = a_val * x + y
        if abs(got - exp) > 1e-4:
            return False, i, got, exp
    return True, 0, 0, 0


def main():
    print_header("SAXPY Benchmark: OpenPTXas vs NVIDIA ptxas")

    n = 16 * 1024 * 1024
    iters = 100
    print(f"  Array size:    {n:,} floats ({n*4/1e6:.1f} MB each)")
    print(f"  Iterations:    {iters}")
    print(f"  Memory/launch: 2 reads + 1 write = {3*n*4/1e6:.0f} MB (plus 4B scalar)")
    print(f"  FLOPs/launch:  {2*n/1e6:.0f} M (1 FMA per element = 2 flops)")
    print()

    compile_openptxas(SAXPY_PTX)  # warmup
    print("  [compile] OpenPTXas...")
    c_ours, t_ours = compile_openptxas(SAXPY_PTX)
    print(f"            {t_ours*1000:.1f} ms, {len(c_ours)} bytes")
    print("  [compile] NVIDIA ptxas...")
    c_nvid, t_nvid = compile_ptxas(SAXPY_PTX)
    print(f"            {t_nvid*1000:.1f} ms, {len(c_nvid)} bytes")
    print()

    ctx = CUDAContext()
    print("  [execute] OpenPTXas cubin...")
    assert ctx.load(c_ours)
    f = ctx.get_func("saxpy")
    t_ours_list, r_ours, a_val = run_kernel(ctx, f, n, iters)
    med_ours = median(t_ours_list)
    bw_ours = (3 * n * 4) / (med_ours / 1000) / 1e9
    print(f"            median {med_ours*1000:.1f} us, {bw_ours:.1f} GB/s")

    print("  [execute] ptxas cubin...")
    assert ctx.load(c_nvid)
    f = ctx.get_func("saxpy")
    t_nvid_list, r_nvid, _ = run_kernel(ctx, f, n, iters)
    med_nvid = median(t_nvid_list)
    bw_nvid = (3 * n * 4) / (med_nvid / 1000) / 1e9
    print(f"            median {med_nvid*1000:.1f} us, {bw_nvid:.1f} GB/s")
    print()
    ctx.close()

    ok_ours, _, _, _ = verify(r_ours, n, a_val)
    correct = r_ours == r_nvid and ok_ours
    print(f"  [verify]  Bit-identical: {'YES' if r_ours == r_nvid else 'NO'}, math: {'PASS' if ok_ours else 'FAIL'}")
    print()

    print_results(t_ours*1000, t_nvid*1000, c_ours, c_nvid,
                  med_ours*1000, med_nvid*1000,
                  "Mem bandwidth", bw_ours, bw_nvid, correct)
    return bw_ours, bw_nvid, correct


if __name__ == '__main__':
    main()
