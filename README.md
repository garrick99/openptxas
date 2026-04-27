# OpenPTXas

**Open-source PTX assembler. Real cubins. Real GPU. GPU-verified.**

Compiles PTX into executable cubins for **SM_120 Blackwell** GPUs. Full pipeline: parse, register allocate, instruction select, schedule, scoreboard, ELF emit, GPU execute.

**No ptxas. No nvcc. Pure Python.**

## What is this?

A pure-Python implementation of NVIDIA's PTX assembler. Take any PTX source, get back an executable SM_120 cubin (ELF binary) that the CUDA driver loads via `cuModuleLoadData` and runs directly on a Blackwell GPU. Reverse-engineered from byte-exact ptxas 13.0 output across 108 unique SM_120 opcodes.

OpenPTXas is the back of a fully open-source GPU toolchain:

```
[Forge (.fg)]  ──►  [OpenCUDA]  ──►  PTX  ──►  OpenPTXas  ──►  cubin  ──►  GPU
                                                ↑ this repo    PTX → SASS → ELF
```

- **[Forge](https://github.com/garrick99/forge)** — formally-verified systems language (optional front-end)
- **[OpenCUDA](https://github.com/garrick99/opencuda)** — CUDA C → PTX, pure Python
- **OpenPTXas** (this repo) — PTX → SM_120 cubin, pure Python
- **[forge-workbench](https://github.com/garrick99/forge-workbench)** — cross-stack CLI cockpit (run / compare / benchmark / classify across all four projects above)
- **[VortexSTARK](https://github.com/garrick99/VortexSTARK)** — production user via the Forge front-end (9 forge-emitted kernels in the prover)

No NVIDIA compiler is invoked at any stage of the toolchain.

## What's Proven

GPU-verified on RTX 5090 (Blackwell SM_120), zero ptxas fallback anywhere in the pipeline:

| Signal | Number |
|--------|-------:|
| Pytest (parser, isel, scoreboard, encoders, codegen, regressions) | **904 / 904 pass** |
| 144-kernel frontier (byte-classified vs ptxas 13.0) | **63 BYTE_EXACT / 78 STRUCTURAL / 3 MIXED** |
| 7-kernel benchmark suite (all correctness-verified) | **geomean 1.06× vs ptxas**, SAXPY **1.72×** |
| Pair with [OpenCUDA](https://github.com/garrick99/opencuda) (CUDA C → PTX) GPU E2E | **88 / 88 pass** |
| SASS encoder coverage | **183 encoders / 108 unique SM_120 opcodes** |

The 904 pytest count breaks down as: 787 non-GPU (parser / regalloc / isel / encoder / scheduler / scoreboard regressions, run as one batch), 89 GPU (`test_gpu_*.py` files run one-at-a-time to avoid CUDA UR cache pollution), and 28 misc (`test_capmerc_gen.py`, `test_bugfix_benchmark.py`, `test_fg40_harness.py`).

"STRUCTURAL" means valid SASS that differs in instruction layout from ptxas (e.g. we pack LDCU.128 where ptxas uses two LDCU.64s). "MIXED" is where OURS and ptxas both produce correct SASS with a mix of register-only and control-byte-only byte diffs. Every kernel in all three buckets produces correct GPU output.

## Benchmarks (RTX 5090, SM_120)

| Kernel | OpenPTXas | NVIDIA ptxas | Ratio | Status |
|--------|----------:|-------------:|------:|--------|
| vecadd | 1626.3 GB/s | 1626.1 GB/s | 1.00× | PASS |
| **saxpy** | **1723.9 GB/s** | **1000.9 GB/s** | **1.72×** | PASS |
| memcpy | 1518.0 GB/s | 1516.9 GB/s | 1.00× | PASS |
| scale | 1746.9 GB/s | 1746.5 GB/s | 1.00× | PASS |
| stencil | 1599.0 GB/s | 1644.8 GB/s | 0.97× | PASS |
| relu | 1744.4 GB/s | 1803.6 GB/s | 0.97× | PASS |
| fma_chain | 13712 GFLOPS | 14352 GFLOPS | 0.96× | PASS |

Geomean: **1.06×** of ptxas across 7 kernels. All outputs byte-correct against ptxas reference.

## The Full Stack

```
Forge (.fg)                      ← formally-verified systems language
   │
   │  Z3 proof discharge + C99/CUDA C emission
   ▼
CUDA C (.cu)
   │
   │  OpenCUDA — CUDA C → PTX (Python compiler)
   ▼
PTX (.ptx)
   │
   │  OpenPTXas — PTX → SASS → cubin (Python assembler)
   ▼
SM_120 cubin (ELF binary)
   │
   │  cuModuleLoad + cuLaunchKernel
   ▼
RTX 5090 GPU — correct output, matching hand-written CUDA
```

No NVIDIA compiler is invoked at any stage. The Forge → OpenCUDA → OpenPTXas toolchain has landed tiled matmul, 2D stencil, tiled convolution, warp-shuffle reductions, and multi-block atomics — all with Z3-discharged proofs and GPU-verified correctness on RTX 5090.

## Differential Fuzzer

`openptxas/fuzzer/` is a grammar-based differential fuzzer. It emits random well-typed PTX, compiles through both OpenPTXas and ptxas, runs both on GPU, and flags divergences. Three families:

| Family | Focus | Divergence rate (fresh 1-min campaign) |
|--------|-------|---------------------------------------:|
| `alu_int` | integer ALU, carry chains, shifts, mul.wide | ~10% |
| `bitmanip` | bfe, bfi, popc, clz, brev, prmt | ~12% |
| `warp` | shfl, vote, ballot | ~18% |

The fuzzer is **self-filtering**: a strengthened well-formedness check in both the generator and oracle rejects PTX with undefined-register reads, so every flagged divergence is a real miscompile (OpenPTXas-side or ptxas-side), not noise. Closed miscompiles in recent campaigns:

- `vote.sync.{any,all}.pred` with predicate-destination (was `KeyError` in isel)
- `bfi.b32` with `c + len > 32` (spec-unspecified; now matches ptxas's pass-through)
- `shl.b64` with `K ≥ 64` after IR constant-folding of chained shifts
- `mul.lo R-R` via single IMAD (0x224) instead of IMAD.WIDE + MOV
- `mul.lo + cvt.u64` 8-bit immediate truncation in IMAD.WIDE peephole
- UIADD (0x835) narrowed to `_sr_source`-only (was admitting unsafe LDG-address-derived shapes)
- `sub.f32` / `setp.gt.f32` operand-order inversion (FADD negate-flag misnomer)
- Multi-ret basic block miscompile (`_sink_param_loads` placing `ld.param.u64` after trailing `bra`)
- UR4 clobber (FG26 admission + R22 WB-8 exemption chain, 4-attempt saga documented in `_fuzz_bugs/add_shr_add_with_tid_guard/REPRO.md`)
- Predicate allocation now uses linear-scan liveness: dead predicates post-`@P EXIT` are correctly reused, matching ptxas

Every fix is committed with a reproducer and GPU-verified before landing.

## ptxas Gets It Wrong

Differential testing found a miscompile in **NVIDIA's own ptxas 13.0** on RTX 5090:

```
Kernel:   (x << 8) - (x >> 56)
Input:    0x0123456789ABCDEF

ptxas 13.0    0x23456789ABCDEF01   WRONG
OpenPTXas     0x23456789ABCDEEFF   CORRECT
```

Verified over 500,000 iterations. Same kernel, same GPU, same input. Reproducer + NVIDIA dossier in `_nvidia_bugs/`.

Additional ptxas bugs surfaced by the stack (dossiered, pre-disclosure):
- `bfe.s32` out-of-range shift constant fold
- LOP3 + SHF commutator bug
- Hypervisor IOMMU BSOD chain (system-crash-class, not submitted)

## Quick Start

```bash
git clone https://github.com/garrick99/openptxas
cd openptxas
python demo.py                                   # compile + run vector_add on GPU
python benchmarks/run_all.py                     # benchmark vs ptxas
python scripts/health.py --frontier-only         # classify 144 kernels
python -m fuzzer.loop run --families alu_int --minutes 1  # differential fuzz

# CLI dashboard (run / compare / benchmark / classify) lives in forge-workbench:
pip install -e ../forge-workbench && workbench list
```

Pure Python 3.11+. No external dependencies beyond NVIDIA driver (for execution) and optional `ptxas` (for differential comparison).

## What's Inside

| Stage | File | Description |
|-------|------|-------------|
| Parser | `ptx/parser.py` | Recursive descent PTX → IR |
| Regalloc | `sass/regalloc.py` | Linear-scan liveness with safe eviction, predicate reuse |
| Isel | `sass/isel.py` | PTX → SASS selection, 183 encoders, 108 unique opcodes |
| Scheduler | `sass/schedule.py` | LDG latency hiding, LDCU.64 hoisting |
| Scoreboard | `sass/scoreboard.py` | Automated rbar/wdep/misc generation (bitmask-based) |
| Emitter | `sass/pipeline.py` | Full ELF cubin with `.nv.info`, `.nv.capmerc`, `.nv.merc` |
| Fuzzer | `fuzzer/` | Grammar-based differential fuzz with well-formedness filter |

The CLI dashboard (run / status / show / kdiff / explore / history / diff / stress) used to live here as `workbench.py`; it now ships as a stand-alone package — see [forge-workbench](https://github.com/garrick99/forge-workbench).

## SM_120 Blackwell Discoveries

Reverse-engineered during development. Not documented publicly elsewhere:

| Discovery | Detail |
|-----------|--------|
| **rbar is a bitmask** | OR-combine barrier waits: bit 1 = LDC, bit 2 = LDS, bit 3 = LDG |
| **IMAD R-R-R (0x224) works, (0x2a4) broken** | The `encode_imad_rr` (0x2a4) variant silently produces wrong results; (0x224) R-R-R is ptxas-verified |
| **ISETP corrupts FSETP state** | Both R-R and R-UR variants clobber subsequent FSETP output |
| **FSEL.step (0x80a)** | Combined float compare+select avoids ISETP/FSETP interaction |
| **S2R is asynchronous** | Requires `wdep=0x31` scoreboard tracking |
| **SM_120 uses predicated execution** | No BRA-based warp divergence; ptxas if-converts everything |
| **Capmerc DRM signature** | 0x5a universal ptxas signature authenticates register metadata |
| **Literal pool is broken** | Driver doesn't init `.nv.constant0` beyond params; all immediates must inline |
| **LDG shares one scoreboard slot** | All LDG instructions use `wdep=0x35`; slot 0x37 has no rbar bit |
| **BAR.SYNC resets scoreboard state** | Pending writes must be cleared after barrier; stale deps corrupt post-barrier LDC |
| **DADD/DMUL/DFMA use b1=0x72** | The b1=0x7e/0x7c forms (from decode_sass.py) silently produce wrong results |
| **LOP3 reads 3 source registers** | b3, b4, b8 all tracked for dependency; missing b4/b8 causes stale-data hazards |
| **DADD src1 at b8, not b4** | Unlike DMUL/DFMA, DADD places second operand at byte 8 |
| **UR cache leaks across launches** | Uniform register bank retains stale values across cuModuleLoad; ~20+ loads corrupt LDCU.64 |
| **SEL barrier race** | SEL predicate read shares ALU wdep slot; intermediate ALU can clear barrier before pred ready |
| **FMNMX pred encoding** | b10=0x80 (not 0xfe); b11=0x03 for min (PT), 0x07 for max (!PT) |
| **UIADD (0x835) dual-write** | Writes both R[dest] and UR[dest] simultaneously; only safe when source is ctaid/ntid SR-derived |
| **`bfi.b32` spec ambiguity** | For `c + len > 32`, ptxas passes `b` through unchanged (unspecified per spec) |
| **IMAD.WIDE imm is 8 bits** | `encode_imad_wide` b4 silently masks to 8 bits; larger multipliers must use another path |

## Instruction Coverage

108 unique SM_120 opcodes, 183 encoders. All byte-verified against ptxas 13.0; 99 GPU-verified on RTX 5090.

| Category | Instructions |
|----------|-------------|
| Integer | IADD3, IMAD, IMAD.WIDE, IMAD.SHL, IADD.64, IABS, LEA, IMNMX, IDP (dp4a) |
| Float | FADD, FMUL, FFMA, FSEL.step, FMNMX, FSWZADD, DADD, DMUL, DFMA, DSETP |
| Transcendentals | MUFU (RCP, SQRT, RSQ, SIN, COS, EX2, LG2) |
| Shifts / Bits | SHF (L/R, U32/U64/S32, HI/LO, const/var), LOP3, POPC, BREV, FLO, BMSK, SGXT, PRMT |
| Comparison | ISETP (6 modes), FSETP (8 modes), DSETP (unordered), VIMNMX |
| Memory | LDG / STG (32/64-bit), LDS / STS, LDC / LDCU (32/64/128), LDSM |
| Atomics | ATOMG (ADD, MIN, MAX, EXCH, CAS.32, CAS.64, ADD.F32) |
| Async copy | LDGSTS (cp.async), LDGDEPBAR, DEPBAR.LE |
| TMA | UBLKCP (bulk copy), UTMALDG (tensor 1D/2D), UTMASTG, UTMACMDFLUSH |
| Mbarrier | SYNCS.EXCH (init), SYNCS.ARRIVE, SYNCS.TRYWAIT |
| Warp | SHFL (4 modes), VOTE (BALLOT / ALL / ANY — with pred or GPR dest), REDUX (SUM / MIN / MAX), MATCH (ANY / ALL), NANOSLEEP |
| Texture | TEX, TLD.LZ, TLD4, TXQ, SULD, SUST |
| Type convert | I2F (u32/s32), F2I (u32/s32), F2F (f32 ↔ f64), F2FP (f16 ↔ f32), I2IP |
| Predicates | P2R, R2P, PLOP3 |
| Uniform | UMOV, UIADD (0x835), UIADD3, UISETP, USEL, UFSETP, UFMUL, ULEA |
| Cluster | UCGABAR (arrive / wait), MEMBAR.ALL.GPU |
| Control | MOV, NOP, EXIT, BRA, BRA.U, CALL.REL, RET.REL, S2R, S2UR, ELECT |
| Barriers | BAR.SYNC, BAR.RED.OR, ERRBAR, CGAERRBAR, B2R, CCTL |
| Tensor cores | HMMA (BF16 / TF32), IMMA (INT8), DMMA (FP64), QMMA (FP8 E4M3 / E5M2) |
| Capmerc / DRM | Fully automatic from SASS, 0x5a universal signature confirmed |

## Requirements

- Python 3.11+
- NVIDIA GPU + CUDA driver (for execution)
- NVIDIA ptxas (optional, for differential validation and fuzzing only)

## Design

No dependencies. No plugins. One Python package. Everything is readable source — parser, regalloc, isel, scheduler, scoreboard, encoders, emitter. No C extensions, no LLVM, no MLIR.

Every instruction encoder has a byte-exact ground-truth comment pinned from ptxas 13.0 output. Every scoreboard opcode has its misc/wdep/rbar documented with kernel-level provenance. Every architectural discovery has a reproducer.

## License

See LICENSE file.
