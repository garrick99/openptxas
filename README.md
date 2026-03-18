# OpenPTXas

Open-source PTX assembler for NVIDIA Blackwell (SM_120 / RTX 5090).

Generates executable cubins from PTX source, with correct results verified on hardware. Includes a scoreboard emulator that auto-generates dependency barriers, a cubin scanner that detects the ptxas rotate-miscompilation bug, and 34 ground-truth verified SASS instruction encoders.

## Status

**Working.** Three standalone cubins execute correctly on RTX 5090. Matches or beats ptxas 13.0 performance.

```
OpenPTXas v0.1.0 vs NVIDIA ptxas 13.0 — RTX 5090 (SM_120)
500,000 iterations:

  Test 1: Rotate-Left (both correct)
    ptxas 13.0:   5.446 us/iter
    OpenPTXas:    5.250 us/iter  — 1.04x faster

  Test 2: Bug-Fix (ptxas WRONG, OpenPTXas CORRECT)
    ptxas 13.0:   5.632 us/iter  — WRONG ANSWER
    OpenPTXas:    5.257 us/iter  — CORRECT
```

## The ptxas Bug

NVIDIA's `ptxas` has a miscompilation bug affecting SM_50 through SM_120 (every GPU since ~2014):

```
PTX:    (a << K) - (a >> (64-K))    // subtraction
ptxas:  SHF.L.W.U32.HI              // emits rotate — WRONG
```

The peephole optimizer pattern-matches `shl + shr` and converts to rotate without checking that the combining operation is subtraction (not add/or/xor). OpenPTXas correctly emits `IADD.64` with negate.

### Scan for the bug

```bash
python -m openptxas --scan suspicious.cubin
```

### Compile with the fix

```bash
python -m openptxas kernel.ptx --out kernel.cubin
```

## Usage

```bash
# Compile PTX to cubin
python -m openptxas kernel.ptx --out kernel.cubin

# Scan a cubin for rotate-miscompilation bugs
python -m openptxas --scan input.cubin

# Parse and dump IR
python -m openptxas kernel.ptx --dump-ir

# Verbose output (show SASS instructions)
python -m openptxas kernel.ptx -v
```

## Instruction Encoders (34)

All encoders produce byte-identical output to ptxas 13.0.

| Category | Instructions |
|----------|-------------|
| Integer | IADD3, IADD3.X, IADD.64, IMAD.WIDE, IMAD.SHL.U32, LOP3.LUT |
| FP32 | FADD, FMUL, FFMA |
| Shifts | SHF.L.W.U32.HI, SHF.L.U32, SHF.L.U64.HI, SHF.R.U64, SHF.R.U32.HI |
| Tensor | HMMA.16816.F32, IMMA.16832.S8.S8, LDSM.16.M88.4 |
| Memory | LDC, LDC.64, LDCU.64, LDG.E (32/64/128), STG.E (32/64/128), STS, LDS |
| CVT | I2FP.F32.S32, F2I.TRUNC.NTZ |
| Control | MOV, NOP, EXIT, S2R, S2UR, BAR.SYNC, BRA, ISETP.GE.AND |

## Architecture

```
  PTX source
      |
  [Parser]     — hand-rolled recursive descent
      |
  [IR]         — Module / Function / BasicBlock / Instruction
      |
  [Rotate Pass] — detects ptxas miscompilation patterns
      |
  [RegAlloc]   — sequential allocation with LDG coalescing
      |
  [ISel]       — PTX → SASS instruction selection
      |
  [Scheduler]  — LDG latency hiding reorder
      |
  [Scoreboard] — automated ctrl/depbar generation
      |
  [Emitter]    — ELF64 cubin with ELFOSABI_CUDA, .nv.info, .nv.capmerc
      |
  [RTX 5090]
```

## Key Technical Details

- **Scoreboard emulator**: auto-generates SM_120 dependency barriers (rbar/wdep) from register def-use chains. No manual ptxas-matching needed.
- **Capmerc section**: `.nv.capmerc.text.<kernel>` header byte[8] encodes physical register file allocation. Without it, R8+ registers are uninitialized.
- **ELFOSABI_CUDA**: `e_ident[7]=0x41, e_ident[8]=0x08` — required for CUDA driver to accept the binary.
- **CGA addressing**: shared memory on Blackwell uses `S2UR(CgaCtaId) + UMOV + ULEA` with a 24-bit shift for CTA-local base computation.

## Stack

| Layer | Project | Role |
|-------|---------|------|
| Frontend | [OpenCUDA](https://github.com/garrick99/opencuda) | C → PTX |
| Backend | **OpenPTXas** | PTX → cubin |
| Hardware | RTX 5090 | SM_120 / Blackwell |

## Tests

71 unit tests + 3 GPU integration tests (compute, bug-fix, shared memory).

```bash
python -m pytest tests/ -v
```

## Requirements

- Python 3.11+
- `lark` (optional, for grammar-based parsing)
- `pyelftools` (optional, for ELF inspection)
- CUDA toolkit (only for GPU tests)

## License

Private. All rights reserved.
