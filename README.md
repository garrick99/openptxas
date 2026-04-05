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
pytest tests/ -x -q                 # 356 tests
```

## What's Inside

| Stage | Description |
|-------|-------------|
| **Parser** | Recursive descent PTX parser to IR |
| **RegAlloc** | Linear scan with liveness, safe eviction |
| **ISel** | PTX to SASS instruction selection (183 encoders, 108 unique opcodes) |
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
| **LDG shares single scoreboard slot** | All LDG instructions use wdep=0x35; slot 0x37 has no rbar bit |
| **BAR.SYNC resets scoreboard state** | Pending writes must be cleared after barrier; stale deps corrupt post-barrier LDC |
| **DADD/DMUL/DFMA use b1=0x72** | The b1=0x7e/0x7c forms (from decode_sass.py) silently produce wrong results |
| **LOP3 reads 3 source registers** | b3, b4, b8 all tracked for dependency; missing b4/b8 causes stale-data hazards |
| **DADD src1 at b8, not b4** | Unlike DMUL/DFMA, DADD places second operand at byte 8 |

## Instruction Coverage (108 unique opcodes, 183 encoders)

All encoders byte-verified against ptxas 13.0 on SM_120. 70 GPU-verified on RTX 5090.

| Category | Instructions |
|----------|-------------|
| Integer | IADD3, IMAD, IMAD.WIDE, IMAD.SHL, IADD.64, IABS, LEA, IMNMX, IDP (dp4a) |
| Float | FADD, FMUL, FFMA, FSEL.step, FMNMX, FSWZADD, DADD, DMUL, DFMA, DSETP |
| Transcendentals | MUFU (RCP, SQRT, RSQ, SIN, COS, EX2, LG2) |
| Shifts/Bits | SHF (L/R, U32/U64/S32, HI/LO, const/var), LOP3, POPC, BREV, FLO, BMSK, SGXT, PRMT |
| Comparison | ISETP (6 modes), FSETP (8 modes), DSETP (unordered), VIMNMX |
| Memory | LDG/STG (32/64-bit), LDS/STS, LDC/LDCU, LDSM |
| Atomics | ATOMG (ADD, MIN, MAX, EXCH, CAS.32, CAS.64, ADD.F32) |
| Async copy | LDGSTS (cp.async), LDGDEPBAR, DEPBAR.LE |
| TMA | UBLKCP (bulk copy), UTMALDG (tensor 1D/2D), UTMASTG, UTMACMDFLUSH |
| Mbarrier | SYNCS.EXCH (init), SYNCS.ARRIVE, SYNCS.TRYWAIT |
| Warp | SHFL (4 modes), VOTE (BALLOT/ALL/ANY), REDUX (SUM/MIN/MAX), MATCH (ANY/ALL), NANOSLEEP |
| Texture | TEX, TLD.LZ, TLD4, TXQ, SULD, SUST |
| Type convert | I2F (u32/s32), F2I (u32/s32), F2F (f32↔f64), F2FP (f16↔f32), I2IP |
| Predicates | P2R, R2P, PLOP3 |
| Uniform | UMOV, UIADD3, UISETP, USEL, UFSETP, UFMUL, ULEA |
| Cluster | UCGABAR (arrive/wait), MEMBAR.ALL.GPU |
| Control | MOV, NOP, EXIT, BRA, BRA.U, CALL.REL, RET.REL, S2R, S2UR, ELECT |
| Barriers | BAR.SYNC, BAR.RED.OR, ERRBAR, CGAERRBAR, B2R, CCTL |
| Tensor cores | HMMA (BF16/TF32), IMMA (INT8), DMMA (FP64), QMMA (FP8 E4M3/E5M2) |
| Capmerc/DRM | Fully automatic from SASS, 0x5a universal signature confirmed |

## Requirements

- Python 3.11+
- NVIDIA GPU + CUDA driver (for execution)
- NVIDIA ptxas (optional, for validation only)

## License

See LICENSE file.
