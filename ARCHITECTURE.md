# OpenPTXas — Architecture Guide

Internal reference for contributors. The README covers the user-facing story and the list of
SM_120 discoveries; this document covers what you need to know to safely modify the assembler.

---

## Pipeline Overview

```
PTX source text
  └─ parse()            ptx/parser.py       — recursive descent PTX → IR
      └─ regalloc()     sass/regalloc.py    — linear scan register allocation
          └─ isel()     sass/isel.py        — PTX IR → SASS instruction list
              └─ schedule()  sass/schedule.py  — reorder for latency hiding
                  └─ scoreboard()  sass/scoreboard.py  — ctrl word generation
                      └─ emit_cubin()  cubin/emitter.py  — ELF construction
```

All stages are in `sass/pipeline.py`, which orchestrates them and handles if-conversion.

---

## SM_120 Hardware — Critical Knowledge

**Read this before touching isel.py, scoreboard.py, or any encoding.**

These are reverse-engineered facts about SM_120 Blackwell that are not documented by NVIDIA.
Every one of them was discovered by running kernels on hardware and observing incorrect output
or crashes.

### Broken instructions

| Instruction | Opcode | Problem | Fix |
|-------------|--------|---------|-----|
| IMAD R-R | 0x2a4 | Produces garbage output | Use IMAD.WIDE (0x225) or IMAD R-UR (0xc24) |
| ISETP R-R | 0x20c | Corrupts subsequent FSETP output | Use FSEL.step (0x80a) instead |
| ISETP R-UR | 0xc0c | Same FSETP corruption | Same fix |

**The ISETP/FSETP interaction is the most dangerous footgun.** If you need a predicate from
an integer comparison followed by a float comparison, you cannot use ISETP for the integer
part. Use FSEL.step for the combined operation, or restructure to avoid adjacent ISETP+FSETP.

### rbar is a bitmask, not a maximum

The `rbar` field in the ctrl word is OR-combined across all instructions that need to be
complete before the current instruction reads their result. It is **not** a watermark.

```
bit 1 = LDC (constant memory load)
bit 2 = LDS (shared memory load)
bit 3 = LDG (global memory load)
```

If you have an LDG followed by an LDS followed by a consumer, the consumer's rbar is
`(1<<3) | (1<<2) = 0xc`, not `max(0x8, 0x4) = 0x8`.

### S2R is asynchronous

Reading a special register (e.g., `%tid.x`) via S2R is an asynchronous operation.
The result is not available until `wdep=0x31` is seen by the scoreboard. Any instruction
that consumes the S2R output must either have `wdep=0x31` or be separated from the S2R
by enough independent instructions.

### Predicated execution, not BRA

SM_120 does not use BRA for intra-kernel warp divergence. Instead, ptxas if-converts
everything: conditional branches become predicate-guarded (`@%p`) instruction sequences.
BRA is only used for backward jumps (loops) and device function calls.

The pipeline handles if-conversion in `sass/pipeline.py`. Do not emit BRA for if/else
constructs — the GPU will execute both sides, guarded by predicates.

### Literal pool broken

The `.nv.constant0` ELF section holds the kernel parameter area. The GPU driver initializes
it up to the end of the parameter declarations, but **no further**. If you try to store
floating-point immediates in a literal pool beyond the parameter area (as some assemblers do),
they will read as zero at runtime.

**All immediates must be encoded inline** in the instruction stream. There is no literal pool.

### EXIT ctrl word

EXIT always requires `wdep=0x3f` and `misc=5`, regardless of whether it is predicated or
unpredicated. This is not optional — kernels that exit with wrong ctrl words cause GPU hangs
or silent data corruption.

### FSEL.step (0x80a)

This is a combined float compare-and-select instruction: `FSEL.step dest, a, b, threshold`.
It is the correct way to implement float ternary on SM_120 without triggering the
ISETP/FSETP corruption bug. `misc=5` is required. The step suffix controls which comparison
mode is active.

---

## Capmerc System

Capmerc (`.nv.capmerc` ELF section) is a DRM-like metadata blob that the GPU driver
validates against the kernel's `.text` section at load time. If capmerc doesn't match the
instruction layout, `cuModuleLoad` fails silently (no error, kernel just doesn't run).

### What it contains

Capmerc encodes per-instruction-class metadata about the register usage pattern of the
kernel. The first 52 bytes are a universal ptxas 13.0 signature (`0x5a` magic, identical
across all cubins of the same ptxas version). After that, per-instruction metadata follows.

### capmerc_gen.py

`cubin/capmerc_gen.py` generates valid capmerc from a SASS instruction list by:
1. Scanning the instruction classes and their operand register ranges
2. Building the instruction-schedule-matching metadata required by the driver
3. Prepending the 0x5a universal signature

**GPR limit:** Without proper capmerc, reliably only 10 GPRs work. The capmerc system is
what enables >10 GPRs. If you modify instruction selection in a way that changes register
pressure, re-run the capmerc generator.

### The merc section

`.nv.merc` is a related section containing resource usage metadata (shared memory size,
register count, etc.). `cubin/emitter.py` generates this from the kernel's register
allocation results.

---

## Scoreboard (sass/scoreboard.py)

The scoreboard generates the 3-byte ctrl word prepended to each instruction group.

### Ctrl word fields

| Field | Bits | Meaning |
|-------|------|---------|
| `rbar` | [4:0] | Barrier wait bitmask (see above) |
| `wdep` | [12:5] | Write dependency tracking |
| `misc` | [15:13] | Instruction mode flags |
| `yield` | [16] | Yield hint for warp scheduler |

**rbar**: OR of barrier bits for all reads this instruction must wait for.
**wdep**: Tracks which write barriers the current instruction produces. The scoreboard
accumulates wdep values and emits them at instruction group boundaries.
**misc=5**: Required for FSEL.step and EXIT. Also required for certain S2R variants.

### Instruction groups

SM_120 processes instructions in groups (typically 4 per fetch). The ctrl word applies to
the group, not individual instructions. The scheduler in `sass/schedule.py` manages group
boundaries.

---

## Register Allocation (sass/regalloc.py)

Linear scan over a liveness interval computed from the instruction list. Key constraints:

- R0–R9: reliably usable without capmerc complications
- R10+: require instruction-schedule-matching capmerc
- Predicate registers (P0–P7): separate pool, do not alias with R registers
- UR registers (uniform registers): separate pool, used for IMAD R-UR patterns

If the allocator overflows 10 GPRs, `capmerc_gen.py` must be invoked to generate matching
capmerc. This is done automatically in `sass/pipeline.py`.

---

## Instruction Selection (sass/isel.py)

60+ encoders, one per SASS opcode class. Each encoder takes PTX-level operands and produces
a byte-encoded SASS instruction.

### Adding a new instruction

1. Add an encoder function in `sass/isel.py` following the naming convention `_emit_<opcode>`
2. Add the dispatch case to `isel()`
3. Add scoreboard handling in `sass/scoreboard.py` (which barrier bits does it need/produce?)
4. Write a PTX test file and add it to `tests/`
5. Verify the encoding byte-for-byte against ptxas 13.0 output on SM_120

**Byte verification is mandatory.** The SASS encoding is not documented. Every encoder was
derived by diffing ptxas output. Do not guess — verify.

### Encoding conventions

Instructions are 16 bytes (128 bits). The opcode occupies bits [11:0] of the first 8 bytes.
Operand fields are packed in a layout that varies by instruction class. See the existing
encoders for the bit layout patterns — they're the authoritative reference.

---

## ELF Structure (cubin/emitter.py)

A cubin is an ELF64 binary with NVIDIA-specific sections:

| Section | Purpose |
|---------|---------|
| `.text.kernel_name` | SASS instruction bytes |
| `.nv.info` | Kernel metadata (param sizes, etc.) |
| `.nv.constant0` | Parameter area (initialized by driver) |
| `.nv.capmerc` | Capmerc DRM blob |
| `.nv.merc` | Resource usage metadata |
| `.nv.shared` | Shared memory declarations |

The `emitter.py` constructs this ELF from the scheduler output and capmerc generator output.
`emitter_sm89.py` is the Ampere/Ada variant (SM89) retained for reference — SM_120 uses the
main `emitter.py`.

---

## Test Structure

```
tests/
  test_ptxas_validates/    — compile PTX, run on GPU, check output
  test_capmerc/            — capmerc generation correctness
  (205/207 passing)
```

Run tests: `pytest tests/ -x -q`

The 2 failing tests are pre-existing edge cases in IMAD.WIDE encoding for a specific
immediate value range. They do not affect the passing GPU-verified kernels.

---

## Common Pitfalls

**Never use IMAD R-R.** Even if you think the context is safe. Use IMAD.WIDE or IMAD R-UR.

**Never place ISETP before FSETP.** The ISETP/FSETP corruption affects any FSETP that
follows an ISETP in the same warp execution path, regardless of intervening instructions.
If you need both integer and float predicates, compute them in the opposite order
(FSETP first, ISETP second) or restructure to use FSEL.step.

**Always set misc=5 for EXIT.** A kernel that exits with the wrong ctrl word may appear to
succeed (no CUDA error) but produce garbage in subsequent kernel launches on the same stream.

**Do not try to store constants in .nv.constant0 beyond the parameter area.** The driver
will not initialize them. All float constants must be encoded as immediate fields in the
instruction, or loaded via LDC from a valid constant bank.

**capmerc must match the instruction layout.** If you change instruction selection, register
allocation, or instruction scheduling, regenerate capmerc. A stale capmerc causes silent
`cuModuleLoad` failure.
