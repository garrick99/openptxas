# ALLOC-R18-24 — In-backend value-lifetime split: BAIL

## What was attempted

`sass/regalloc.py::_split_b32_multi_writes` — a pre-allocation pass
that, for each basic block, identifies multi-written `%rN` (b32)
vregs and renames the second and subsequent writes to fresh
`%rN__vK` names with reads pointing at the latest dominating write.

The renaming itself worked correctly on the Forge tcopy minimal
repro: `mul.lo.u32 %r1, %r1, %r2` (after `mov %r1=ctaid; mov %r2=ntid`)
correctly read the original `%r1`/`%r2` and wrote to `%r1__v1`,
giving each semantic value a distinct allocator name.

## Why it bailed

After enabling the pass:

| metric | baseline | with split | delta |
|---|---:|---:|---:|
| pytest | 865/865 | 864/865 (1 flake) | -1 |
| GPU PASS | 127 | **125** | **-2** |
| GPU FAIL | 10 | **12** | **+2** |
| BYTE_EXACT | 66 | **65** | **-1** |
| STRUCTURAL | 78 | 78 | 0 |

Real GPU regressions in 2 corpus kernels + 1 BYTE_EXACT kernel
moving back to STRUCTURAL.  Per the operating rules ("HARD BAIL on
invariant explosion that cannot be localized" + "Zero regressions
mandatory"), the change was reverted.

## Root cause of the regression

Existing OURS passes downstream of allocation make assumptions that
are violated by the new vreg names:

1. **LDG dest coalescing** (`_find_ldg_coalesces`) keys on the
   original PTX vreg name; new `%rN__vK` names break the heuristic.
2. **WB-7 address-fold map** (`_addr_fold_map`) likewise keys on
   the original name.
3. **`ctx._reg_param_off` / `_ur_for_param` / `_reg_ur_safe_src`**
   tracking dictionaries set keys at PTX-load time using the
   original vreg name; later code paths key into them with the
   renamed name and miss.
4. **Control-flow guard checks** (e.g. the if-conversion's
   `_has_neg_sub` and `_overwrites_pred` guards) match by vreg name
   to detect specific patterns; renaming hides those patterns.

Each downstream pass that consumes the PTX vreg name would need
to be taught about the rename mapping.  This is exactly the
"cascading change" the operating rules forbid as a "broad rewrite".

## Why the existing composition already works

The combination of three orthogonal fixes shipped in earlier chains
already enforces the same invariant from R17 by other means:

* **OCUDA01-08** eliminates b32 vreg reuse at the OpenCUDA emit
  boundary — Forge-emitted PTX never has the multi-write pattern.
* **ALLOC-R01-08** prevents the scheduler from creating a
  same-phys-reg overwrite via S2R hoisting.
* **ALLOC-R09-16** prevents the IMAD fusion peephole from creating
  an alias that a later instruction would corrupt.

For Forge-emitted PTX, OCUDA SSA means there's nothing for an
in-backend split pass to *do* — every vreg already has exactly one
write.  For hand-written PTX in the workbench corpus, the schedule
+ fusion guards catch the failure modes that would surface from
multi-write reuse.

## What was preserved

* `sass/regalloc.py` reverted to baseline.
* pytest 865/865, GPU 127 PASS / 10 FAIL / 7 RUN_EXC, frontier
  BYTE_EXACT 66 / STRUCTURAL 78 — all unchanged.
* All 8 Forge slices (FORGE01, 05, 09, 13, 17, 21, 25, 29) still
  compile + load + GPU-PASS unchanged.

## Honest assessment

The semantic value-lifetime correctness invariant from
`ALLOC_R17_MODEL.md` IS the right model.  But:

* It is **already enforced** for Forge-emitted PTX via the
  frontend SSA fix (OCUDA01-08).
* It is **already enforced** for hand-written PTX via two narrow
  scheduler/isel guards (ALLOC-R01-08, ALLOC-R09-16).
* A genuine in-backend implementation of the invariant would
  require touching every downstream pass that keys on PTX vreg
  names — exactly the broad rewrite the operating rules forbid.

The current 3-fix composition is **load-bearing equivalent** to a
proper allocator rewrite for the entire current envelope (Forge
slices, workbench corpus, GPU baseline).  An allocator rewrite
would consolidate the protections but not unlock new capability.

## Classification

**ALLOC_REWRITE_BLOCKED** — bounded in-backend implementation
attempted, regressed 2 GPU PASS kernels + 1 BYTE_EXACT kernel due
to cascading dependencies in downstream passes.  Reverted.  The
invariant is enforced by the existing 3-fix composition shipped
in OCUDA01-08, ALLOC-R01-08, and ALLOC-R09-16.

## Next subsystem decision

Given the value-lifetime invariant is **already enforced** for the
current envelope, the right next move is **C: Return to Forge
expansion**.  Specifically: extend the Forge envelope into shapes
that the current backend can already handle but that no Forge slice
yet exercises (e.g., warp-level intrinsics, multi-block reductions,
floating-point arithmetic).  Each new Forge slice that lands without
backend changes is independent evidence that the value-lifetime
invariant holds.
