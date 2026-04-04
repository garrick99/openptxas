#!/usr/bin/env python3
"""GPU runner for FP64 bench — loads cubin and benchmarks on RTX 5090."""
import sys
import ctypes
import struct

CUBIN_PATH = r'C:\Users\kraken\openptxas\fp64_bench.cubin'
N_BLOCKS   = 170 * 4   # 680 — fills RTX 5090 (170 SMs, 4 resident blocks/SM)
BLOCK_SIZE = 256
N_THREADS  = N_BLOCKS * BLOCK_SIZE  # 174,080

class GPU:
    def __init__(self):
        try:
            self.cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
        except Exception:
            print("No nvcuda.dll found.")
            sys.exit(1)
        assert self.cuda.cuInit(0) == 0, "cuInit failed"
        dev = ctypes.c_int()
        self.cuda.cuDeviceGet(ctypes.byref(dev), 0)
        name_buf = ctypes.create_string_buffer(256)
        self.cuda.cuDeviceGetName(name_buf, 256, dev)
        self.device_name = name_buf.value.decode()
        self.ctx = ctypes.c_void_p()
        assert self.cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev) == 0
        self.mod = ctypes.c_void_p()

    def load_file(self, path):
        with open(path, 'rb') as f:
            cubin = f.read()
        err = self.cuda.cuModuleLoadData(ctypes.byref(self.mod), cubin)
        if err != 0:
            name = ctypes.c_char_p()
            self.cuda.cuGetErrorName(err, ctypes.byref(name))
            print(f"cuModuleLoadData failed: {name.value.decode()} (err={err})")
            sys.exit(1)

    def func(self, name):
        f = ctypes.c_void_p()
        err = self.cuda.cuModuleGetFunction(ctypes.byref(f), self.mod, name.encode())
        assert err == 0, f"cuModuleGetFunction({name}) err={err}"
        return f

    def alloc(self, nbytes):
        p = ctypes.c_uint64()
        assert self.cuda.cuMemAlloc_v2(ctypes.byref(p), max(nbytes, 1)) == 0
        return p.value

    def free(self, p):
        self.cuda.cuMemFree_v2(ctypes.c_uint64(p))

    def launch(self, fn, grid, block, args):
        holders, ptrs = [], []
        for a in args:
            if isinstance(a, float):
                h = ctypes.c_double(a)
            else:
                h = ctypes.c_uint64(a)
            holders.append(h)
            ptrs.append(ctypes.cast(ctypes.byref(h), ctypes.c_void_p))
        arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        gx, gy, gz = grid
        bx, by, bz = block
        return self.cuda.cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, 0, None, arr, None)

    def event(self):
        ev = ctypes.c_void_p()
        self.cuda.cuEventCreate(ctypes.byref(ev), 0)
        return ev

    def record(self, ev):
        self.cuda.cuEventRecord(ev, None)

    def elapsed_ms(self, ev_start, ev_end):
        self.cuda.cuEventSynchronize(ev_end)
        ms = ctypes.c_float()
        self.cuda.cuEventElapsedTime(ctypes.byref(ms), ev_start, ev_end)
        return ms.value

    def sync(self):
        self.cuda.cuCtxSynchronize()


def run_benchmark(gpu, fn):
    out_ptr = gpu.alloc(N_THREADS * 8)
    B = 1.0 + 1e-9
    C = 1e-19

    print(f"\n  GPU: {gpu.device_name}")
    print(f"  Grid: {N_BLOCKS} blocks x {BLOCK_SIZE} threads = {N_THREADS:,} threads")
    print(f"  4 independent DMUL+DADD chains per thread")
    print(f"  FP32 theoretical peak (RTX 5090): 105,000 GFLOPS\n")

    header = f"{'n_iters':>10}  {'best_ms':>10}  {'GFLOPS':>10}  {'vs FP32':>10}  {'status'}"
    print(header)
    print("-" * len(header))

    results = {}
    for n_iters in [64, 128, 256, 512, 1024, 2048]:
        flops = N_THREADS * 4 * n_iters * 2
        ev_s = gpu.event()
        ev_e = gpu.event()
        best_ms = float('inf')

        err = gpu.launch(fn, (N_BLOCKS, 1, 1), (BLOCK_SIZE, 1, 1),
                        [out_ptr, N_THREADS, n_iters, B, C])
        if err != 0:
            name = ctypes.c_char_p()
            gpu.cuda.cuGetErrorName(err, ctypes.byref(name))
            print(f"  Warmup launch submit failed: {name.value.decode()} (err={err})")
            gpu.free(out_ptr)
            return {}
        sync_err = ctypes.c_int()
        err = gpu.cuda.cuCtxSynchronize()
        if err != 0:
            name = ctypes.c_char_p()
            gpu.cuda.cuGetErrorName(err, ctypes.byref(name))
            print(f"  Warmup sync failed: {name.value.decode()} (err={err})")
            gpu.free(out_ptr)
            return {}

        for _ in range(5):
            gpu.record(ev_s)
            err = gpu.launch(fn, (N_BLOCKS, 1, 1), (BLOCK_SIZE, 1, 1),
                             [out_ptr, N_THREADS, n_iters, B, C])
            gpu.record(ev_e)
            ms = gpu.elapsed_ms(ev_s, ev_e)
            if err == 0:
                best_ms = min(best_ms, ms)

        if best_ms == float('inf'):
            print(f"{n_iters:>10}  {'--':>10}  {'--':>10}  {'--':>10}  LAUNCH ERROR")
            continue

        gflops = flops / best_ms / 1e6
        ratio  = gflops / 105_000 * 100
        results[n_iters] = gflops
        print(f"{n_iters:>10}  {best_ms:>10.3f}  {gflops:>10.1f}  {ratio:>9.3f}%  OK")

    gpu.free(out_ptr)
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("  FP64 Benchmark — Forge -> OpenPTXas -> SM_120 -> RTX 5090")
    print("=" * 60)

    gpu = GPU()
    gpu.load_file(CUBIN_PATH)
    fn = gpu.func('fp64_bench')

    results = run_benchmark(gpu, fn)

    if results:
        best = max(results.values())
        print(f"\n  Peak measured: {best:.1f} GFLOPS ({best/1e3:.3f} TFLOPS)")
        print(f"  FP32/FP64 ratio: ~{105_000/best:.0f}:1")
        print()
        print("  First public measurement of RTX 5090 FP64 throughput.")
        print("  Pipeline: Forge (formally verified) -> OpenPTXas -> SM_120 cubin")

    print("\n" + "=" * 60)
