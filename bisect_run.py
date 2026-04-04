#!/usr/bin/env python3
"""Bisect ILLEGAL_INSTRUCTION by testing different n_iters values."""
import sys, ctypes

CUBIN_PATH = r'C:\Users\kraken\openptxas\fp64_bench.cubin'
N_BLOCKS   = 1   # minimal
BLOCK_SIZE = 32
N_THREADS  = N_BLOCKS * BLOCK_SIZE

class GPU:
    def __init__(self):
        self.cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
        assert self.cuda.cuInit(0) == 0
        dev = ctypes.c_int()
        self.cuda.cuDeviceGet(ctypes.byref(dev), 0)
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
        assert err == 0, f"{name} err={err}"
        return f

    def alloc(self, nbytes):
        p = ctypes.c_uint64()
        assert self.cuda.cuMemAlloc_v2(ctypes.byref(p), max(nbytes, 1)) == 0
        return p.value

    def free(self, p):
        self.cuda.cuMemFree_v2(ctypes.c_uint64(p))

    def run(self, fn, out_ptr, n_threads, n_iters, B, C):
        holders, ptrs = [], []
        for a in [out_ptr, n_threads, n_iters, B, C]:
            h = ctypes.c_double(a) if isinstance(a, float) else ctypes.c_uint64(a)
            holders.append(h)
            ptrs.append(ctypes.cast(ctypes.byref(h), ctypes.c_void_p))
        arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        err = self.cuda.cuLaunchKernel(fn, N_BLOCKS, 1, 1, BLOCK_SIZE, 1, 1, 0, None, arr, None)
        if err != 0:
            name = ctypes.c_char_p()
            self.cuda.cuGetErrorName(err, ctypes.byref(name))
            return f"LAUNCH_FAIL:{name.value.decode()}"
        sync_err = self.cuda.cuCtxSynchronize()
        if sync_err != 0:
            name = ctypes.c_char_p()
            self.cuda.cuGetErrorName(sync_err, ctypes.byref(name))
            return f"SYNC_FAIL:{name.value.decode()}(err={sync_err})"
        return "OK"


gpu = GPU()
gpu.load_file(CUBIN_PATH)
fn = gpu.func('fp64_bench')
out_ptr = gpu.alloc(N_THREADS * 8)
B = 1.0 + 1e-9
C = 1e-19

for n_iters in [0, 1, 2, 4, 8, 16, 32, 64]:
    result = gpu.run(fn, out_ptr, N_THREADS, n_iters, B, C)
    print(f"n_iters={n_iters:3d}: {result}")

gpu.free(out_ptr)
