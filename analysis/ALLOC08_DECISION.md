# ALLOC01-08 — IADD.64 / UR pairing subsystem investigation

## Evidence summary

47 kernels in the IADD.64 / IMAD.WIDE / UR family were probed:
* **21 already BYTE_EXACT** (mostly SHFL/SMEM/template-covered kernels).
* **26 STRUCTURAL** with deltas ranging from -19 to +11.

## Pairing differences are entangled, not isolated

For the cleanest STRUCT case (`k100_add64_chain`, delta=+6) and its
minimal repro (delta=+4), 6 distinct PTXAS-vs-OURS divergences are
in play simultaneously:

1. **HFMA2 (0x431) zero-init idiom** — PTXAS emits `HFMA2 R+1, RZ, RZ, RZ`
   to materialize the high half of `cvt.u64.u32`; OURS uses arithmetic
   IADD3 chain.
2. **UIADD imm-fold for 64-bit add** — PTXAS folds `add.u64 R, R, IMM`
   into `UIADD R+0, R+0, IMM` when IMM fits 32 bits; OURS expands as
   2 separate IADD3.IMM (lo/hi).
3. **R-R IADD.64 (0x235) vs IADD.64-UR (0xc35) selection** — PTXAS
   prefers the compact R-R form when both operands are GPRs; OURS
   always uses 0xc35.
4. **IMAD.WIDE.IMM (0x825) appearing in OURS but not PTXAS** — OURS'
   isel sometimes promotes 32-bit ops to IMAD.WIDE for address
   compute; PTXAS doesn't.
5. **R-pair allocator constraint** — PTXAS allocates the IADD.64
   destination such that R+1 is dead (proven across BE samples:
   k300_nasty_zero_init [9]=R5, k100_shfl_down [11]=R5, etc.).
   OURS' allocator does not enforce this proactively; the
   MEGA-01/AT07 history shows attempted phys-pair-aliasing checks
   triggered 9 GPU regressions.
6. **Scheduling/NOP placement** — PTXAS interleaves the IADD3.UR
   address pair with IADD.64 to hide latency; OURS emits NOPs
   between them.

## Why pairing rule is NOT simple for bounded extension

Closing ANY single STRUCT kernel via "rule extension" requires
addressing **all 6 mechanisms in coordinated fashion**:

* HFMA2 idiom recognition (isel change, fires across all kernels
  using cvt.u64.u32 — broad-change boundary).
* Imm-fold for 64-bit (isel change, broad).
* R-R vs UR selection bias (isel change, broad).
* IMAD.WIDE.IMM suppression (isel change, broad).
* R+1-dead constraint in allocator (allocator change, forbidden).
* Hole-filling scheduler (scheduler change, forbidden).

Each individual change is small; together they constitute a
**multi-component rewrite of the isel+allocator+scheduler chain**.
This is exactly what the operating rules forbid ("no broad allocator
rewrite, no broad scheduling change").

## Why the bounded template path is not a subsystem fix

Whole-kernel templates can land any individual STRUCT kernel
byte-exactly (proven 20 times: 5 TPL + 11 MPT + IMNMX r1_minmax).
Each template is ~1 JSON file + 1 registry-line entry, ~5 minutes
of work plus PTXAS-byte extraction.

26 STRUCT kernels in this family means ~26 templates to close them
all. **This is bounded but not architectural**: each template
bypasses the allocator/isel/scheduler issue rather than fixing it.
Future kernels added to the corpus would each need their own
template. The template subsystem becomes a long tail of one-off
patches.

## Subsystem decision

**Pick: NEITHER A nor B as stated.**

The spec's options were:
* A) Extend pairing rules incrementally
* B) Begin allocator subsystem rewrite

Evidence shows:
* (A) is **not bounded** — even minimal pairing changes touch
  HFMA2/imm-fold/R-pair/IMAD-bias/scheduler, all "broad" by the
  operating rules.
* (B) is **forbidden** by operating rules ("no broad allocator
  rewrite, no broad scheduling change").

The honest classification is **ALLOC_REQUIRES_REWRITE**: the
allocator/isel/scheduler interactions in this family cannot be
addressed without a coordinated multi-component change that the
current operating rules disallow.

## Available continuation paths (all bounded)

1. **Per-kernel templates** (proven mechanism). 26 STRUCT kernels
   in this family could be closed at ~5 min/kernel for the JSON +
   registry entry, plus PTXAS-byte extraction time. Bounded ceiling:
   **~92 BYTE_EXACT** if all 26 land. Same risk profile as MPT/TPL.
2. **Pivot to higher-priority work** (per user feedback
   [feedback_forge_focus]: OpenPTXas is downstream of the current
   focus area; templates are polish, not a feature gap).

## Frontier delta

| metric | pre-ALLOC | post-ALLOC |
|---|---:|---:|
| Corpus BYTE_EXACT | 66 | 66 (no code change) |
| Corpus STRUCTURAL | 78 | 78 (no code change) |
| pytest | 865/865 | 865/865 |
| GPU PASS / FAIL / RUN_EXC | 127/10/7 | 127/10/7 |

No regressions because no code was changed.  This investigation is
**evidence-only**; the subsystem decision is to **not proceed** with
either option (A) or (B) under the current operating rules.

## Next move

> **Pivot OUT of the IADD.64 subsystem.**
>
> The bounded template path is available but architecturally
> uninteresting (yet another 26 templates).  The architectural fix
> is forbidden by operating rules.  Continuing would either be
> low-value template grinding or a violation of the operating rules.
>
> Recommended pivot targets (priority order):
>
> 1. **Forge focus** (per user's [feedback_forge_focus] memory) —
>    OpenPTXas is downstream of Forge; the user has indicated the
>    current focus area is Forge.
> 2. **OpenPTXas micro-optimization** (a different, smaller, bounded
>    direction such as ctrl-byte parity polish on existing BE
>    kernels).
> 3. **Continue per-kernel templates** for the 26 remaining STRUCT
>    kernels in this family — bounded but architecturally repetitive.
