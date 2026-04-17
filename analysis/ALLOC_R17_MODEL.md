# ALLOC-R17 — Allocator correctness model

## Semantic value model

A **semantic value** is the result of a single PTX instruction's write
to a destination register.  PTX virtual register names (`%r2`, etc.)
are *labels* for storage locations; the same name can be re-bound
across the body of a function.  The allocator must reason about
*values* that flow between defs and uses, not about labels.

For each instruction `I` writing dest `%rX`:
* `I` defines a fresh semantic value `v(I)`.
* Any subsequent read of `%rX` before the next write to `%rX` binds
  to `v(I)`.
* The next write to `%rX` (call it `J`) defines a distinct semantic
  value `v(J)` — completely independent of `v(I)`.

## Lifetime model

Per semantic value `v`:
* **def(v)**:    instruction index where `v` is created.
* **last_use(v)**: latest instruction index that reads `v`.
* **kill(v)**:   the next write to the same vreg name (or end of
                 function); `v` is dead afterward.
* **interval(v)**: `[def(v), last_use(v)]`.

Two semantic values **interfere** iff their intervals overlap.

## Failure taxonomy from prior chains (evidence-backed)

| value_kind | example kernel | failure mode | required allocator behavior |
|---|---|---|---|
| `S2R(TID)` vs `LDC(blockDim)` | Forge tcopy | hoist + same-phys-reg overwrite | distinct phys for both |
| fusion `mul.dest` vs `add.dest` | Forge memory_slice (R09 repro) | fusion alias corrupted later use | distinct phys when later use exists |
| reused `%rN` for distinct LDG dests | Forge map_compose pre-OCUDA01 | scheduler reorder swapped values | distinct phys per LDG dest |
| loop-carried `%sum` | Forge loop_acc | loop back-edge extends live range | live across back-edge — distinct from later non-loop-carried writes |

## Exact invariant

> **No physical register assignment may cause two simultaneously-live
> semantic values to alias.**

Equivalently: if `interval(v1) ∩ interval(v2) ≠ ∅`, then `phys(v1) ≠ phys(v2)`.

## Why the prior fixes work

* **OCUDA01-08 (frontend SSA for b32)**: eliminates the source of
  multi-write vreg names at the OpenCUDA emit boundary. Each
  semantic value gets a unique `%rN` *before* the backend sees it,
  trivially satisfying the invariant.

* **ALLOC-R01-08 (S2R-hoist guard)**: prevents the scheduler from
  reordering a hoist past a same-phys-reg write, which would create
  a temporal interference invisible to the static interval check.

* **ALLOC-R09-16 (phys-reg-aware fusion guard)**: prevents the
  fusion peephole from creating an alias (mul.dest = add.dest)
  whose later use would be corrupted by an intervening write.

These are three orthogonal protections against the same root cause
manifest at different pipeline stages.  An in-backend SSA renaming
pass would unify them by enforcing the invariant directly at
allocation time, making the schedule/fusion guards defense-in-depth
rather than load-bearing.

## What "true value-lifetime splitting" would add

The remaining gap (only relevant for hand-written PTX corpus
kernels — Forge-emitted code is already SSA-clean from OCUDA):

* When a hand-written PTX kernel reuses `%rN` across distinct
  semantic values, an in-backend pass should split it into
  `%rN_v0`, `%rN_v1`, ... before regalloc, so the allocator
  naturally honors the invariant.

The implementation in R18-R20 is the bounded, evidence-driven version
of this — splitting only `r` (b32) prefix, only when multiple writes
exist, only within a single basic block (for now).
