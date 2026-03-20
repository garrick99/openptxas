# OpenPTXas

**PTX → OpenPTXas → cubin → RTX 5090 → correct output.**
**No ptxas. No nvcc.**

Compiles PTX directly into executable cubins for **RTX 5090 (SM_120 Blackwell)** — entirely outside NVIDIA's compiler toolchain.

## What this proves

- Generate valid cubins for Blackwell GPUs
- Execute multi-block kernels on real hardware
- Pass 71/71 unit tests
- Bit-exact correctness (rotate, memory, arithmetic)
- Full pipeline: PTX → parse → regalloc → isel → schedule → scoreboard → cubin → GPU

This is not a simulator. This is real machine code running on a real RTX 5090.

## 30-second proof

```bash
git clone https://github.com/garrick99/openptxas
cd openptxas
python demo.py
```

Output:
```
Compiling: examples/vector_add.ptx
Output:    examples/vector_add.cubin (4432 bytes)
Kernel:    vector_add

--- Running on GPU (no NVIDIA compiler used) ---
Device: NVIDIA GeForce RTX 5090
Launch: 32 blocks x 32 threads = 1024 elements
[PASS] 1024 elements verified correct

Our code. Their GPU.
```

No ptxas. No nvcc. Just OpenPTXas.

## What's inside

| Stage | Description |
|-------|-------------|
| **Parser** | Recursive descent PTX parser → IR |
| **RegAlloc** | Linear scan with liveness, 64-bit pair alignment |
| **ISel** | PTX → SASS instruction selection (60+ encoders) |
| **Scheduler** | LDG latency hiding |
| **Scoreboard** | Automated rbar/wdep barrier generation |
| **Emitter** | Full ELF cubin with .nv.info, .nv.capmerc |

Pure Python 3.11+. No dependencies beyond pytest.

## Instruction coverage (60+ SASS encoders)

All encoders are byte-verified against ptxas 13.0 output on SM_120.

| Category | Instructions |
|---|---|
| **Integer** | IADD3, IADD3.X, IADD.64, IMAD, IMAD.WIDE, IMAD.HI, IMAD.SHL |
| **Float** | FADD, FMUL, FFMA |
| **Transcendentals** | MUFU (RCP, SQRT, RSQ, SIN, COS, EX2, LG2) |
| **Shifts** | SHF.L.W.U32.HI, SHF.L.U32, SHF.L.U64.HI, SHF.R.U64, SHF.R.U32.HI |
| **Bitwise** | LOP3.LUT (AND/OR/XOR), POPC, BREV, FLO |
| **Comparison** | ISETP (6 modes), FSETP (8 modes) |
| **Min/Max** | VIMNMX.S32, VIMNMX.U32, FMNMX |
| **Conditional** | SEL, FSEL |
| **Memory** | LDG.E (u8-u128), STG.E (u32-u128), LDS, STS, LDC, LDC.64, LDCU.64, LDCU.32 |
| **Atomics** | ATOMG.E (ADD, MIN, MAX, AND, OR, XOR, EXCH) |
| **Warp** | SHFL (IDX/UP/DOWN/BFLY), VOTE.BALLOT |
| **Type convert** | I2FP.F32 (S32/U32), F2I (S32/U32), CVT (u32/u64) |
| **Control** | MOV, NOP, EXIT, S2R, S2UR, BAR.SYNC, BRA (predicated), IABS |
| **Tensor** | HMMA.16816.F32, IMMA.16832.S8, LDSM.16.M88.4 |

## SM_120 Blackwell discoveries

Reverse-engineered during development — not documented publicly elsewhere:

- **S2R is asynchronous**: requires `wdep=0x31` scoreboard tracking (not fire-and-forget)
- **Opcode 0x224 is NOT IMAD R-R**: kills non-lane-0 threads on Blackwell
- **c[0][0x360] = blockDim.x**: driver-populated constant bank offset
- **IMAD R-UR (0xc24)**: only non-wide multiply variant on SM_120
- **Capmerc byte[8]**: controls hardware GPR allocation per thread
- **LDC.64 single scoreboard slot**: multiple loads to same pair cause WAW hazards
- **Predicate encoding**: byte[1] bits 7:4 (0x7=PT, 0x0=P0, 0x8=!P0)
- **Pointers must use LDCU→UR path**: avoids GPR pressure and LDC.64 hazards

## ptxas bug detection

OpenPTXas detects and correctly compiles a pattern that NVIDIA's ptxas has miscompiled since SM_50 (~2014):

```
shl.b64  %lo, %a, K
shr.u64  %hi, %a, (64-K)
sub.s64  %res, %lo, %hi    ← ptxas incorrectly emits ROTATE instead of SUBTRACT
```

**Verified:** 500K iterations on RTX 5090 — OpenPTXas correct, ptxas wrong.

## Full pipeline: CUDA C → PTX → cubin

With [OpenCUDA](https://github.com/garrick99/opencuda):

```bash
# CUDA C → PTX
cd opencuda && python -m opencuda kernel.cu --emit-ptx

# PTX → cubin
cd openptxas && python demo.py kernel.ptx
```

33 CUDA kernels compile through this pipeline with zero errors.

## Status

| Feature | Status |
|---------|--------|
| Integer backend | GPU-verified |
| Multi-thread + multi-block | GPU-verified |
| 32-bit and 64-bit memory | GPU-verified |
| Predicated control flow | GPU-verified |
| Scoreboard generation | GPU-verified |
| Floating point | Implemented, validation in progress |
| Shared memory | Implemented, integration in progress |
| General loops/branches | In progress |

## Requirements

- Python 3.11+
- For GPU execution: NVIDIA GPU (RTX 5090/4090) + CUDA driver
- For validation: NVIDIA ptxas (optional)

## License

See LICENSE file.
