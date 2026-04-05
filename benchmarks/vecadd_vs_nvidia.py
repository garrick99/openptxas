"""
Vector Add Benchmark: Our Toolchain vs NVIDIA

Compares OpenPTXas (PTX->cubin) vs ptxas (PTX->cubin) on SM_120.
Memory-bound kernel: 3 buffers (2 reads + 1 write) per element.

Usage: python benchmarks/vecadd_vs_nvidia.py
"""
import ctypes
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sass.pipeline import compile_ptx_source


SAXPY_PTX = """.version 9.0
.target sm_120
.address_size 64

.visible .entry vecadd(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
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
    ld.param.u64 %rd2, [a];
    ld.param.u64 %rd3, [b];
    ld.param.u64 %rd6, [out];
    add.s64 %rd4, %rd2, %rd1;
    add.s64 %rd5, %rd3, %rd1;
    add.s64 %rd7, %rd6, %rd1;

    ld.global.f32 %f1, [%rd4];
    ld.global.f32 %f2, [%rd5];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

DONE:
    ret;
}
"""


def _get_cuda():
    try:
        cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
        if cuda.cuInit(0) != 0:
            return None
        return cuda
    except Exception:
        return None


_CUDA = _get_cuda()
if _CUDA is None:
    print("ERROR: nvcuda.dll not found")
    sys.exit(1)


class CUDAContext:
    def __init__(self):
        self.cuda = _CUDA
        self.ctx = ctypes.c_void_p()
        self.mod = ctypes.c_void_p()
        dev = ctypes.c_int()
        assert self.cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
        assert self.cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev) == 0

    def load(self, cubin_bytes):
        if self.mod and self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()
        err = self.cuda.cuModuleLoadData(ctypes.byref(self.mod), cubin_bytes)
        return err == 0

    def get_func(self, name):
        func = ctypes.c_void_p()
        err = self.cuda.cuModuleGetFunction(ctypes.byref(func), self.mod, name.encode())
        assert err == 0, f"cuModuleGetFunction({name}) failed: {err}"
        return func

    def alloc(self, nbytes):
        ptr = ctypes.c_uint64()
        assert self.cuda.cuMemAlloc_v2(ctypes.byref(ptr), nbytes) == 0
        return ptr.value

    def copy_to(self, dev_ptr, host_data):
        assert self.cuda.cuMemcpyHtoD_v2(ctypes.c_uint64(dev_ptr), host_data, len(host_data)) == 0

    def copy_from(self, dev_ptr, nbytes):
        buf = (ctypes.c_uint8 * nbytes)()
        assert self.cuda.cuMemcpyDtoH_v2(buf, ctypes.c_uint64(dev_ptr), nbytes) == 0
        return bytes(buf)

    def free(self, ptr):
        self.cuda.cuMemFree_v2(ctypes.c_uint64(ptr))

    def sync(self):
        return self.cuda.cuCtxSynchronize()

    def close(self):
        if self.mod and self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
        if self.ctx and self.ctx.value:
            self.cuda.cuCtxSynchronize()
            self.cuda.cuCtxDestroy_v2(self.ctx)

    def event_create(self):
        evt = ctypes.c_void_p()
        assert self.cuda.cuEventCreate(ctypes.byref(evt), 0) == 0
        return evt

    def event_record(self, evt):
        self.cuda.cuEventRecord(evt, None)

    def event_elapsed_ms(self, start, stop):
        ms = ctypes.c_float()
        self.cuda.cuEventSynchronize(stop)
        self.cuda.cuEventElapsedTime(ctypes.byref(ms), start, stop)
        return ms.value


def compile_openptxas(ptx):
    t0 = time.perf_counter()
    result = compile_ptx_source(ptx)
    # Returns dict {kernel_name: cubin_bytes}
    cubin = next(iter(result.values())) if isinstance(result, dict) else result
    return cubin, time.perf_counter() - t0


def compile_ptxas(ptx):
    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "k.ptx"
        cf = Path(tmp) / "k.cubin"
        pf.write_text(ptx)
        t0 = time.perf_counter()
        r = subprocess.run(
            ["ptxas", "-arch=sm_120", "-o", str(cf), str(pf)],
            capture_output=True, text=True
        )
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            raise RuntimeError(f"ptxas: {r.stderr}")
        return cf.read_bytes(), dt


def run_saxpy(ctx, func, n, iters):
    import struct
    nbytes = n * 4
    d_a = ctx.alloc(nbytes)
    d_b = ctx.alloc(nbytes)
    d_out = ctx.alloc(nbytes)

    a_host = bytearray(nbytes)
    b_host = bytearray(nbytes)
    for i in range(n):
        struct.pack_into('f', a_host, i * 4, i * 0.001)
        struct.pack_into('f', b_host, i * 4, i * 0.002)

    block = 256
    grid = (n + block - 1) // block

    da_v = ctypes.c_uint64(d_a)
    db_v = ctypes.c_uint64(d_b)
    dout_v = ctypes.c_uint64(d_out)
    n_v = ctypes.c_uint32(n)
    args = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.byref(da_v), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(db_v), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(dout_v), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n_v), ctypes.c_void_p),
    )

    s_evt = ctx.event_create()
    e_evt = ctx.event_create()

    ctx.copy_to(d_a, bytes(a_host))
    ctx.copy_to(d_b, bytes(b_host))
    for _ in range(5):
        ctx.cuda.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, None, args, None)
    ctx.sync()

    times = []
    for _ in range(iters):
        ctx.event_record(s_evt)
        ctx.cuda.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, None, args, None)
        ctx.event_record(e_evt)
        ctx.sync()
        times.append(ctx.event_elapsed_ms(s_evt, e_evt))

    result = ctx.copy_from(d_out, nbytes)
    ctx.free(d_a)
    ctx.free(d_b)
    ctx.free(d_out)
    return times, result


def main():
    print("=" * 64)
    print("  Vector Add Benchmark: OpenPTXas vs NVIDIA ptxas")
    print("  GPU: RTX 5090 (SM_120)")
    print("=" * 64)
    print()

    n = 16 * 1024 * 1024
    iters = 100

    print(f"  Array size:    {n:,} floats ({n*4/1e6:.1f} MB each)")
    print(f"  Iterations:    {iters}")
    print(f"  Memory/launch: 3 x {n*4/1e6:.0f} MB = {3*n*4/1e6:.0f} MB read+write")
    print()

    # Warmup compile (excludes Python import overhead from measurement)
    _, _ = compile_openptxas(SAXPY_PTX)

    print("  [compile] OpenPTXas...")
    c_ours, t_ours = compile_openptxas(SAXPY_PTX)
    print(f"            {t_ours*1000:.1f} ms, {len(c_ours)} bytes")

    print("  [compile] NVIDIA ptxas...")
    c_nvid, t_nvid = compile_ptxas(SAXPY_PTX)
    print(f"            {t_nvid*1000:.1f} ms, {len(c_nvid)} bytes")
    print()

    ctx = CUDAContext()

    print("  [execute] OpenPTXas cubin...")
    assert ctx.load(c_ours), "Load failed"
    f = ctx.get_func("vecadd")
    t_list_ours, r_ours = run_saxpy(ctx, f, n, iters)
    med_ours = median(t_list_ours)
    bw_ours = (3 * n * 4) / (med_ours / 1000) / 1e9
    print(f"            median {med_ours*1000:.1f} us, {bw_ours:.1f} GB/s")

    print("  [execute] ptxas cubin...")
    assert ctx.load(c_nvid), "Load failed"
    f = ctx.get_func("vecadd")
    t_list_nvid, r_nvid = run_saxpy(ctx, f, n, iters)
    med_nvid = median(t_list_nvid)
    bw_nvid = (3 * n * 4) / (med_nvid / 1000) / 1e9
    print(f"            median {med_nvid*1000:.1f} us, {bw_nvid:.1f} GB/s")
    print()

    ctx.close()

    correct = r_ours == r_nvid
    print(f"  [verify]  Bit-identical: {'YES' if correct else 'NO'}")
    if not correct:
        import struct
        # Check first 5 and count exact matches
        print(f"  [debug]   First 5 values:")
        for i in range(5):
            v_o = struct.unpack_from('f', r_ours, i*4)[0]
            v_n = struct.unpack_from('f', r_nvid, i*4)[0]
            exp = (i * 0.001) + (i * 0.002)
            print(f"            i={i}: ours={v_o:.6f}, nvid={v_n:.6f}, expected={exp:.6f}")
        matches = sum(1 for i in range(0, len(r_ours), 4)
                      if r_ours[i:i+4] == r_nvid[i:i+4])
        print(f"  [debug]   Bit-matches: {matches}/{n}")
    print()

    print("=" * 64)
    print("  RESULTS")
    print("=" * 64)
    fmt = "  {:<22} {:>14}  {:>14}  {:>10}"
    print(fmt.format("Metric", "OpenPTXas", "NVIDIA ptxas", "Ratio"))
    print("-" * 64)
    print(fmt.format("Compile time",
                     f"{t_ours*1000:.1f} ms",
                     f"{t_nvid*1000:.1f} ms",
                     f"{t_nvid/t_ours:.1f}x"))
    print(fmt.format("Cubin size",
                     f"{len(c_ours)} B",
                     f"{len(c_nvid)} B",
                     f"{(len(c_nvid)-len(c_ours))/len(c_nvid)*100:+.0f}%"))
    print(fmt.format("Dependencies",
                     "0 binaries",
                     "ptxas binary",
                     ""))
    print(fmt.format("GPU exec (median)",
                     f"{med_ours*1000:.1f} us",
                     f"{med_nvid*1000:.1f} us",
                     f"{med_nvid/med_ours:.2f}x"))
    print(fmt.format("Mem bandwidth",
                     f"{bw_ours:.1f} GB/s",
                     f"{bw_nvid:.1f} GB/s",
                     f"{bw_ours/bw_nvid:.2f}x"))
    print(fmt.format("Correctness",
                     "identical" if correct else "DIFFERS",
                     "(baseline)",
                     ""))
    print("=" * 64)


if __name__ == '__main__':
    main()
