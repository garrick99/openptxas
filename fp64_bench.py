#!/usr/bin/env python3
"""
FP64 Throughput Benchmark — RTX 5090 (SM_120 Blackwell)
Pipeline: Forge → OpenCUDA → OpenPTXas → GPU

Measures double-precision FLOPS via 4 independent DMUL+DADD chains.
Reports GFLOPS and ratio vs FP32 theoretical peak (105 TFLOPS).

Usage: python fp64_bench.py
"""
import sys
import os
import ctypes
import struct
import subprocess
import time

# Add openptxas and opencuda to path
BENCH_DIR  = os.path.dirname(os.path.abspath(__file__))
OPENCUDA   = os.path.join(BENCH_DIR, '..', 'opencuda')
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, OPENCUDA)

from sass.pipeline import compile_ptx_source

FORGE_BIN  = '/mnt/c/Users/kraken/forge/_build/default/bin/main.exe'
FORGE_SRC  = '/mnt/c/Users/kraken/forge/demos/1019_fp64_bench.fg'
FORGE_ROOT = '/mnt/c/Users/kraken/forge'

# ---------------------------------------------------------------------------
# Step 1: Forge → CUDA C
# ---------------------------------------------------------------------------
def forge_to_cuda_c():
    print("[1/3] Forge → CUDA C  (proving 4 obligations)...")
    result = subprocess.run(
        [FORGE_BIN, 'build', FORGE_SRC],
        capture_output=True, text=True, cwd=FORGE_ROOT
    )
    if result.returncode != 0:
        print("Forge error:\n", result.stderr)
        sys.exit(1)
    cu_path = FORGE_SRC.replace('.fg', '.cu')
    with open(cu_path) as f:
        src = f.read()
    proofs = result.stdout.count('✓')
    print(f"    {proofs} proofs discharged. Correct by construction.")
    return src

# ---------------------------------------------------------------------------
# Step 2: CUDA C → PTX (OpenCUDA)
# ---------------------------------------------------------------------------
def cuda_c_to_ptx(cu_src):
    print("[2/3] CUDA C → PTX  (OpenCUDA)...")
    sys.path.insert(0, os.path.join(BENCH_DIR, '..', 'opencuda'))
    from opencuda.frontend.preprocess import preprocess
    from opencuda.frontend.parser import parse
    from opencuda.ir.optimize import optimize
    from opencuda.codegen.emit import ir_to_ptx

    source = preprocess(cu_src)
    module = parse(source)
    module = optimize(module)
    ptx_map = ir_to_ptx(module)

    # Assemble full PTX module
    lines = ['.version 9.0', '.target sm_120', '.address_size 64', '']
    if '__preamble__' in ptx_map:
        lines.extend(ptx_map['__preamble__'].split('\n'))
        lines.append('')
    for name, text in ptx_map.items():
        if name.startswith('__'):
            continue
        # Skip header lines already added
        body = text.split('\n')
        start = next((i for i, l in enumerate(body)
                      if l.startswith('.visible') or l.startswith('{')), 0)
        lines.extend(body[start:])
        lines.append('')
    ptx = '\n'.join(lines)
    print(f"    PTX generated ({len(ptx)} chars, {sum(1 for k in ptx_map if not k.startswith('__'))} kernel(s)).")
    return ptx

# ---------------------------------------------------------------------------
# Step 3: PTX → cubin (OpenPTXas)
# ---------------------------------------------------------------------------
def ptx_to_cubin(ptx_src):
    print("[3/3] PTX → cubin  (OpenPTXas)...")
    results = compile_ptx_source(ptx_src)
    cubin = results['fp64_bench']
    print(f"    cubin: {len(cubin)} bytes.")
    return cubin

# ---------------------------------------------------------------------------
# CUDA driver wrapper
# ---------------------------------------------------------------------------
class GPU:
    def __init__(self):
        try:
            self.cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
        except Exception:
            print("No nvcuda.dll found — is this Windows with CUDA?")
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

    def load(self, cubin):
        if self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()
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
        err = self.cuda.cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, 0, None, arr, None)
        return err

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


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(gpu, fn):
    # RTX 5090: 170 SMs. 4 resident blocks/SM fills the machine.
    N_BLOCKS    = 170 * 4   # 680
    BLOCK_SIZE  = 256
    N_THREADS   = N_BLOCKS * BLOCK_SIZE  # 174,080

    # Allocate output buffer (one f64 per thread)
    out_ptr = gpu.alloc(N_THREADS * 8)
    out_len = N_THREADS

    # b ≈ 1.0+ε prevents overflow; c ≈ 0 adds tiny drift to prevent CSE
    B = 1.0 + 1e-9
    C = 1e-19

    print(f"\n  Grid: {N_BLOCKS} blocks × {BLOCK_SIZE} threads = {N_THREADS:,} threads")
    print(f"  4 independent DMUL+DADD chains per thread")
    print(f"  FP32 theoretical peak (RTX 5090): 105,000 GFLOPS\n")

    header = f"{'n_iters':>10}  {'best_ms':>10}  {'GFLOPS':>10}  {'vs FP32':>10}  {'status'}"
    print(header)
    print("-" * len(header))

    results = {}
    for n_iters in [64, 128, 256, 512, 1024, 2048]:
        # 4 chains × n_iters iters × 2 FLOPs (DMUL + DADD)
        flops = N_THREADS * 4 * n_iters * 2

        ev_s = gpu.event()
        ev_e = gpu.event()
        best_ms = float('inf')

        # Warmup
        gpu.launch(fn, (N_BLOCKS, 1, 1), (BLOCK_SIZE, 1, 1),
                   [out_ptr, out_len, n_iters, B, C])
        gpu.sync()

        # 5 timed runs
        for _ in range(5):
            gpu.record(ev_s)
            err = gpu.launch(fn, (N_BLOCKS, 1, 1), (BLOCK_SIZE, 1, 1),
                             [out_ptr, out_len, n_iters, B, C])
            gpu.record(ev_e)
            ms = gpu.elapsed_ms(ev_s, ev_e)
            if err == 0:
                best_ms = min(best_ms, ms)

        if best_ms == float('inf'):
            print(f"{n_iters:>10}  {'—':>10}  {'—':>10}  {'—':>10}  LAUNCH ERROR")
            continue

        gflops = flops / best_ms / 1e6
        ratio  = gflops / 105_000 * 100
        results[n_iters] = gflops
        status = "OK"
        print(f"{n_iters:>10}  {best_ms:>10.3f}  {gflops:>10.1f}  {ratio:>9.3f}%  {status}")

    gpu.free(out_ptr)
    return results


def main():
    print("=" * 60)
    print("  FP64 Throughput — Forge → OpenPTXas → SM_120")
    print("=" * 60)
    print()

    forge_to_cuda_c()

    # Use Forge-generated PTX directly (SWhile writeback fix applied)
    print("[2/3] Loading Forge PTX (SWhile writeback fix active)...")
    ptx_path = FORGE_SRC.replace('.fg', '.ptx')
    with open(ptx_path) as f:
        ptx_src = f.read()
    print(f"    PTX loaded ({len(ptx_src)} chars).")

    cubin = ptx_to_cubin(ptx_src)

    # Save cubin for Windows-side GPU execution
    cubin_path = os.path.join(BENCH_DIR, 'fp64_bench.cubin')
    with open(cubin_path, 'wb') as f:
        f.write(cubin)
    print(f"    cubin saved: {cubin_path}")

    # Launch GPU runner via Windows Python (nvcuda.dll requires Windows context)
    runner = os.path.join(BENCH_DIR, 'fp64_run.py')
    win_runner = runner.replace('/mnt/c/', 'C:\\').replace('/', '\\')
    print("\n  Handing off to Windows Python for GPU execution...")
    result = subprocess.run(
        ['cmd.exe', '/c', 'python.exe', win_runner],
        text=True
    )
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
