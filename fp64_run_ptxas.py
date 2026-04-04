#!/usr/bin/env python3
"""GPU runner — uses ptxas-compiled cubin for comparison."""
import sys, ctypes

CUBIN_PATH = r'C:\Users\kraken\openptxas\fp64_bench_ptxas.cubin'
N_BLOCKS   = 170 * 4
BLOCK_SIZE = 256
N_THREADS  = N_BLOCKS * BLOCK_SIZE

class GPU:
    def __init__(self):
        self.cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
        assert self.cuda.cuInit(0) == 0
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
            h = ctypes.c_double(a) if isinstance(a, float) else ctypes.c_uint64(a)
            holders.append(h)
            ptrs.append(ctypes.cast(ctypes.byref(h), ctypes.c_void_p))
        arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        return self.cuda.cuLaunchKernel(fn, *grid, *block, 0, None, arr, None)

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
        return self.cuda.cuCtxSynchronize()


gpu = GPU()
print(f"GPU: {gpu.device_name}")
gpu.load_file(CUBIN_PATH)
fn = gpu.func('fp64_bench')

out_ptr = gpu.alloc(N_THREADS * 8)
B = 1.0 + 1e-9
C = 1e-19

for n_iters in [64, 128, 256]:
    flops = N_THREADS * 4 * n_iters * 2
    ev_s = gpu.event()
    ev_e = gpu.event()

    err = gpu.launch(fn, (N_BLOCKS, 1, 1), (BLOCK_SIZE, 1, 1),
                     [out_ptr, N_THREADS, n_iters, B, C])
    if err != 0:
        name = ctypes.c_char_p()
        gpu.cuda.cuGetErrorName(err, ctypes.byref(name))
        print(f"  n_iters={n_iters} LAUNCH FAILED: {name.value.decode()}")
        continue
    sync_err = gpu.sync()
    if sync_err != 0:
        name = ctypes.c_char_p()
        gpu.cuda.cuGetErrorName(sync_err, ctypes.byref(name))
        print(f"  n_iters={n_iters} SYNC FAILED: {name.value.decode()} (err={sync_err})")
        continue

    best_ms = float('inf')
    for _ in range(3):
        gpu.record(ev_s)
        err = gpu.launch(fn, (N_BLOCKS, 1, 1), (BLOCK_SIZE, 1, 1),
                         [out_ptr, N_THREADS, n_iters, B, C])
        gpu.record(ev_e)
        ms = gpu.elapsed_ms(ev_s, ev_e)
        if err == 0:
            best_ms = min(best_ms, ms)

    if best_ms != float('inf'):
        gflops = flops / best_ms / 1e6
        print(f"  n_iters={n_iters}: {best_ms:.3f}ms, {gflops:.1f} GFLOPS")

gpu.free(out_ptr)
