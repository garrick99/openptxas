# OpenPTXas miscompile: scoreboard-propagation across @P0 EXIT

## Symptom

A minimal `load → add → shr → add → store` kernel with the standard
fuzz-kernel `@%p0 ret` tid-bounded guard writes **zero** to the output
buffer for every lane. Same kernel without the guard writes correct
values.

Surfaced by `factory/` differential fuzz 2026-04-22. `theirs_correct`
verdict on programs pid=68, pid=103 in the smoke DB (both have the
standard guard + short ALU chain + store).

## Minimal repro

See [minimal.ptx](minimal.ptx) (17 body instructions) and
[noguard.ptx](noguard.ptx) (same body, guard removed).

Expected output per lane: `((input + 3) >> 2) + 256`.

| | lane 0 | lane 1 | lane 2 |
|---|---|---|---|
| spec / ptxas | 0x100 | 0x100 | 0x2aaaabab |
| OpenPTXas (with guard) | 0 | 0 | 0 |
| OpenPTXas (no guard) | 0x100 | 0x100 | 0x2aaaabab |

## Root cause (narrowed)

SASS instructions are byte-identical between the broken and correct
versions for the output-address computation, **but the ctrl bytes
differ at the scoreboard-wait positions:**

| offset | instruction | broken ctrl | correct ctrl |
|---|---|---|---|
| LDC.64 R4, c[0x0][0x388] | p_out load | `0x000e22...` | `0x000e2a...` |
| IADD3 R2, P0, PT, R4, R2, RZ | lo32 add | `0x001fc4...` | `0x001fdc...` |
| IADD3.X R3, ..., P0, !PT | hi32 add-with-carry | `0x001fc6...` | `0x001fde...` |

In the broken version the scheduler drops wait bits needed to enforce
the `LDC.64 R4 → IADD3 R4` RAW dependency. The IADD3 reads stale R4
(still holding the earlier input-address IADD.64 result), computes a
bogus output address, and STG.E writes into nowhere useful. The output
buffer was `cuMemsetD8_v2` zero-filled before the launch, so the read
comes back as 0.

Without the `@P0 EXIT` sequence, the scheduler emits the correct wait
bits and the RAW dependency is honored.

## Suspected fix location

`sass/scoreboard.py`. The file already has special handling for
predicated EXIT/BRA that resets the LDCU slot counter (line 1532-1542)
— whatever state that reset touches appears to miss the GPR RAW
dependency tracker, so later instructions lose their wait bits.

## Why the 865-kernel corpus doesn't catch this

Every kernel in the corpus goes through a different pipeline path
(templates, precursor analysis, etc.). The fuzz-kernel-shape +
`compile_function(..., enable_dce=...)` direct path doesn't exercise
those paths; it's essentially a different compile mode that the
factory is the first to stress.

## Regression test

Once fixed: `python -c "import _fuzz_bugs.add_shr_add_with_tid_guard.repro"`
should compile + run both variants and assert both produce 0x100 for
lane 0.
