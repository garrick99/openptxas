# OpenPTXas

Open-source PTX-to-SASS assembler for NVIDIA GPUs. Compiles PTX assembly to executable cubin binaries — no NVIDIA ptxas required.

Targets SM_120 (Blackwell / RTX 5090) with 60+ ground-truth verified SASS instruction encoders, automated dependency barrier generation, and a fix for a critical ptxas miscompilation bug affecting every NVIDIA GPU since 2014.

## What You Can Do

```bash
# Compile PTX to cubin
python __main__.py kernel.ptx -v

# Scan an existing cubin for the ptxas rotate-sub bug
python __main__.py --scan suspicious.cubin

# Audit a cubin for scheduling hazards
python __main__.py --audit kernel.cubin

# Run tests (71 tests, all passing)
pytest tests/ -v
```

The generated cubins execute correctly on RTX 5090 hardware, verified against ptxas 13.0 output.

## Full Pipeline: CUDA C → PTX → cubin

With [OpenCUDA](https://github.com/garrick99/opencuda), you get a complete open-source GPU compilation pipeline:

```bash
# Step 1: CUDA C → PTX (OpenCUDA)
cd opencuda && python -m opencuda kernel.cu --emit-ptx

# Step 2: PTX → cubin (OpenPTXas)
cd openptxas && python __main__.py kernel.ptx -v
```

**33 CUDA kernels compile through this pipeline with zero errors** — from vector_add through tiled matrix multiply, shared memory reductions, atomic operations, warp shuffles, and control flow.

## Instruction Coverage (60+ SASS Encoders)

All encoders are byte-verified against ptxas 13.0 output on SM_120.

| Category | Instructions |
|---|---|
| **Integer arithmetic** | IADD3, IADD3.X, IADD.64, IMAD, IMAD.WIDE, IMAD.HI, IMAD.SHL |
| **Float arithmetic** | FADD, FMUL, FFMA |
| **Transcendentals** | MUFU.RCP, MUFU.SQRT, MUFU.RSQ, MUFU.SIN, MUFU.COS, MUFU.EX2, MUFU.LG2 |
| **Shifts** | SHF.L.W.U32.HI, SHF.L.U32, SHF.L.U64.HI, SHF.R.U64, SHF.R.U32.HI |
| **Bitwise** | LOP3.LUT (AND/OR/XOR via lookup table), POPC, BREV, FLO |
| **Comparison** | ISETP (6 modes, signed/unsigned), FSETP (8 modes) |
| **Min/Max** | VIMNMX.S32, VIMNMX.U32, FMNMX |
| **Conditional** | SEL, FSEL |
| **Memory** | LDG.E (u8/u16/u32/u64/u128), STG.E (u32/u64/u128), LDS, STS, LDC, LDC.64, LDCU.64 |
| **Atomics** | ATOMG.E (ADD, MIN, MAX, AND, OR, XOR, EXCH) — 7 operations |
| **Warp** | SHFL (IDX/UP/DOWN/BFLY), VOTE.BALLOT |
| **Type convert** | I2FP.F32 (S32/U32), F2I (S32/U32), CVT (u32↔u64) |
| **Byte manip** | PRMT (byte permute with immediate selector) |
| **Control** | MOV, NOP, EXIT, S2R, S2UR, BAR.SYNC, BRA (predicated), IABS |
| **Tensor** | HMMA.16816.F32, IMMA.16832.S8.S8, LDSM.16.M88.4 (encoders present, isel pending) |

## Key Features

### Automated Scoreboard Emulation
The SM_120 scoreboard protocol (rbar/wdep barriers) is generated automatically from def-use analysis — no manual control word matching required. Hardware-verified on RTX 5090.

### ptxas Miscompilation Bug Fix
Detects and correctly compiles a pattern that NVIDIA's ptxas has miscompiled since SM_50 (~2014):

```
shl.b64  %lo, %a, K
shr.u64  %hi, %a, (64-K)
sub.s64  %res, %lo, %hi    ← ptxas incorrectly emits ROTATE instead of SUBTRACT
```

OpenPTXas validates all three conditions (commutative op, unsigned shift, matching sources) before emitting a rotate instruction. ptxas skips these checks.

**Verified:** 500K iterations on RTX 5090 — OpenPTXas produces correct results, ptxas produces wrong answers.

### GPU Binary Auditor
6-check static analysis tool for existing cubins:
1. ptxas rotate-sub miscompilation detection
2. Scheduling hazard identification (missing rbar barriers)
3. Register pressure warnings
4. Memory access pattern analysis
5. Synchronization correctness
6. Instruction mix profiling

## Architecture

```
PTX source (.ptx)
    ↓
[Parser]       Hand-rolled recursive descent (full PTX grammar)
    ↓
[IR]           Module → Function → BasicBlock → Instruction
    ↓
[Passes]       Rotate-left bug detection and validation
    ↓
[RegAlloc]     Sequential GPR allocation, 64-bit pair alignment, LDG coalescing
    ↓
[ISel]         PTX → SASS instruction selection (60+ mappings)
    ↓
[Scheduler]    LDG latency hiding (moves LDC after LDG)
    ↓
[Scoreboard]   Automated rbar/wdep barrier generation
    ↓
[ELF Emitter]  Full cubin with .nv.info, .nv.capmerc, .note sections
    ↓
GPU execution  RTX 5090 verified ✓
```

Pure Python 3.11+. No dependencies beyond pytest for testing.

## Requirements

- Python 3.11+
- For GPU execution: NVIDIA CUDA toolkit + RTX 5090/4090
- For validation testing: NVIDIA ptxas (optional)

## Performance

```
OpenPTXas v0.1.0 vs NVIDIA ptxas 13.0 — RTX 5090 (SM_120)
500,000 iterations:

  Rotate-Left (both correct):
    ptxas 13.0:   5.446 µs/iter
    OpenPTXas:    5.250 µs/iter  — 1.04x faster

  Bug-Fix (ptxas WRONG, OpenPTXas CORRECT):
    ptxas 13.0:   5.632 µs/iter  — WRONG ANSWER
    OpenPTXas:    5.257 µs/iter  — correct
```

## Known Limitations

- No register spilling (fails if >255 GPRs needed)
- No liveness analysis (all registers treated as live)
- Integer div/rem emit placeholder (ptxas generates 20-instruction Newton-Raphson sequences)
- Scheduler only does one transformation (LDC after LDG)
- f64 SASS instructions not yet encoded
- SM_89 (Ada Lovelace) support is secondary — SM_120 is primary target

## License

See LICENSE file.
