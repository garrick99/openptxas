# OpenPTXas

**Open-source PTX assembler. Real cubins. Real GPU. Correct output.**

Compiles PTX into executable cubins for **SM_120 Blackwell** GPUs. Full pipeline: parse, register allocate, instruction select, schedule, scoreboard, ELF emit, GPU execute.

**No ptxas. No nvcc. Just Python.**

## The Proof

PTX compiled to working SM_120 cubins using only open-source Python tools.

**GPU-verified on RTX 5090:**

| Kernel | What it does | Status |
|--------|-------------|--------|
| `vector_add` | Float addition, multi-block indexing | **PASS** |
| `kernel_a` | Float multiply by constant (`in[i] * 2.0f`) | **PASS** |
| `increment` | Integer add (`in[i] + 1`) | **PASS** |
| `divergent_warp` | Predicated early exit, intra-warp divergence | **PASS** |
| `sel` | Float ternary with bounds check (`v > 0.5f ? 1.0f : 0.0f`) | **PASS** |
| `imad_chain` | Multi-block multiply-add chain | **PASS** |
| `predicated_exit` | Idle thread writes nothing | **PASS** |

### With [OpenCUDA](https://github.com/garrick99/opencuda): full CUDA C pipeline

```
CUDA C source (.cu)
    |  OpenCUDA (Python)
    v
PTX assembly
    |  OpenPTXas (Python)
    v
SM_120 cubin (ELF binary)
    |  cuModuleLoad + cuLaunchKernel
    v
RTX 5090 GPU --> correct results
```

No NVIDIA compiler involved at any stage.

## ptxas Gets It Wrong

On RTX 5090, NVIDIA's ptxas 13.0 miscompiles a PTX subtract pattern:

```
Kernel:   (x << 8) - (x >> 56)
Input:    0x0123456789ABCDEF

ptxas 13.0    0x23456789ABCDEF01  WRONG
OpenPTXas     0x23456789ABCDEEFF  CORRECT
```

Verified over 500,000 iterations. Same kernel, same GPU, same input.

## Quick Start

```bash
git clone https://github.com/garrick99/openptxas
cd openptxas
python demo.py                      # compile + run vector_add on GPU
pytest tests/ -x -q                 # 205 tests
```

## What's Inside

| Stage | Description |
|-------|-------------|
| **Parser** | Recursive descent PTX parser to IR |
| **RegAlloc** | Linear scan with liveness, safe eviction |
| **ISel** | PTX to SASS instruction selection (60+ encoders) |
| **Scheduler** | LDG latency hiding, LDCU.64 hoisting |
| **Scoreboard** | Automated rbar/wdep/misc generation (bitmask-based) |
| **Emitter** | Full ELF cubin with .nv.info, .nv.capmerc, .nv.merc |

Pure Python 3.11+. No dependencies.

## SM_120 Blackwell Discoveries

Reverse-engineered during development. Not documented publicly elsewhere:

| Discovery | Detail |
|-----------|--------|
| **rbar is a bitmask** | OR-combine barrier waits: bit1=LDC, bit2=LDS, bit3=LDG |
| **IMAD R-R (0x2a4) broken** | Produces garbage. Use IMAD.WIDE (0x225) or IMAD R-UR (0xc24) |
| **ISETP corrupts FSETP** | Both R-R and R-UR variants clobber subsequent FSETP output |
| **FSEL.step (0x80a)** | Combined float compare+select avoids ISETP/FSETP interaction |
| **S2R is asynchronous** | Requires wdep=0x31 scoreboard tracking |
| **SM_120 uses predicated execution** | No BRA-based warp divergence; ptxas if-converts everything |
| **Capmerc DRM system** | 0x5a universal ptxas signature authenticates register metadata |
| **Literal pool broken** | Driver doesn't init .nv.constant0 beyond params; all immediates inline |

## Instruction Coverage (60+)

All encoders byte-verified against ptxas 13.0 on SM_120.

| Category | Instructions |
|----------|-------------|
| Integer | IADD3, IMAD, IMAD.WIDE, IMAD.SHL, IADD.64, IABS |
| Float | FADD, FMUL, FFMA, FMUL.IMM, FFMA.IMM, FSEL.step |
| Transcendentals | MUFU (RCP, SQRT, RSQ, SIN, COS, EX2, LG2) |
| Shifts | SHF (L/R, U32/U64/S32, HI/LO, const/var) |
| Bitwise | LOP3.LUT (AND/OR/XOR/NOT), POPC, BREV, FLO |
| Comparison | ISETP (6 modes), FSETP (8 modes) |
| Memory | LDG/STG (u8-u128), LDS/STS, LDC/LDCU |
| Atomics | ATOMG.E (ADD, MIN, MAX, AND, OR, XOR, EXCH, CAS) |
| Warp | SHFL (IDX/UP/DOWN/BFLY), VOTE.BALLOT |
| Type convert | I2F, F2I, F2F, CVT (u32/u64/s32/s64) |
| Control | MOV, NOP, EXIT, BRA, S2R, S2UR, BAR.SYNC |
| Tensor | HMMA, IMMA, LDSM |

## Requirements

- Python 3.11+
- NVIDIA GPU + CUDA driver (for execution)
- NVIDIA ptxas (optional, for validation only)

## License

See LICENSE file.
