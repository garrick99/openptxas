# OpenPTXas Benchmark Suite

Compares OpenPTXas (pure-Python PTX assembler) against NVIDIA's `ptxas`
on the same PTX source code. Both are loaded via the CUDA driver API and
timed with CUDA events on an RTX 5090 (SM_120).

## Running

```
python benchmarks/run_all.py            # run all benchmarks + summary
python benchmarks/vecadd_vs_nvidia.py   # run a single benchmark
```

## Benchmarks

| Benchmark    | Kernel                              | Stresses                           |
|--------------|-------------------------------------|------------------------------------|
| `vecadd`     | `out = a + b`                       | Memory bandwidth, 3-buffer load/store |
| `saxpy`      | `y = a*x + y` (FMA, scalar via ptr) | FMA + 3-buffer bandwidth           |
| `memcpy`     | `out[i] = in[i]` (u32)              | Peak coalesced memory bandwidth    |
| `scale`      | `out = alpha*x + beta`              | 1 FMA + 2-buffer bandwidth         |
| `stencil`    | 1D 5-point, padded ghosts, no smem  | Cache-coalesced multi-load + FMA   |
| `relu`       | `out = max(x, 0)`                   | Compute + predicate, 2-buf bw      |
| `fma_chain`  | 32 FMAs per element                 | FP throughput (compute-bound)      |

## Example Results (RTX 5090, SM_120)

```
  Benchmark       OpenPTXas        NVIDIA     Ratio    Status
--------------------------------------------------------------------
  vecadd        1600.9 GB/s   1635.8 GB/s     0.98x      PASS
  saxpy         1422.3 GB/s   1004.6 GB/s     1.42x      PASS
  memcpy        1521.6 GB/s   1518.6 GB/s     1.00x      PASS
  scale         1586.6 GB/s   1747.6 GB/s     0.91x      PASS
  stencil       1421.1 GB/s   1664.4 GB/s     0.85x      PASS
  relu          1698.8 GB/s   1834.4 GB/s     0.93x      PASS
  fma_chain    9869.0 GFLOPS  14354.8 GFLOPS  0.69x      PASS
  Geomean perf ratio (passing): 0.947x of NVIDIA ptxas
```

Memory-bound kernels achieve **0.85x-1.42x** the bandwidth of cubins
compiled by NVIDIA ptxas.  Pure-compute (`fma_chain`) is **0.69x**, the
main gap area (scheduler/ILP).

## Methodology

- 5 warmup launches, 100 timed launches (median reported).
- CUDA events for timing (`cuEventRecord` + `cuEventElapsedTime`).
- Same PTX fed to both assemblers; cubins loaded via `cuModuleLoadData`.
- Correctness verified against Python reference (tolerance for float).

## Benchmark Workarounds

During development several OpenPTXas codegen limitations were found and
avoided in these kernels (see `stencil_vs_nvidia.py` for an example):

- Multiple `ld.global` instructions that share a base address register
  (e.g. loading `[%rd0]` and `[%rd0+4]` into different registers after
  computing `%rd0` from `%tid`) can produce an illegal instruction at
  runtime. Workaround: compute fully-independent address registers per
  load.
- `ld.shared` reading back from a different address than the matching
  `st.shared` can return stale/zero values. Workaround: avoid or use
  identical store/load addresses per thread.
- `shfl.sync.{down,bfly}.b32` with an immediate delta (e.g. `, 16, 31,`)
  emits delta=0 in SASS, breaking warp reductions.
- `atom.global.add.f32` returns wrong values (per-thread non-atomic
  behavior). `atom.global.add.u32` is correct.

These prevented benchmarking shared-memory tile transpose, warp-shuffle
reduction, and global-atomic dot-product.
