# IM03 — HARD BAIL

## Status

* IM01 (clustering): clean ✓ commit `7a7a605`
* IM02 (analysis-only helper): clean ✓ commit `2ce83b8`
* **IM03 (emission wiring): HARD BAIL — reverted in this commit**

## What happened

When the IM02 predicate `_im_iadd64_admissible` was wired into the
`add.u32 reg-reg` isel branch as the gate for `IADD3 → IADD.64`
substitution, **pytest dropped from 865/865 green to 752 passed,
93 failed, 20 errors** — a catastrophic regression spanning many
test families (TMA, MUFU, atom, SHFL, warp reduce, predicate
compare, FSETP, etc.).

The wiring was reverted immediately; pytest is back to **865/865
green**.

## Exact blocker

The IM02 predicate's "HI-half is dead" check is performed at the
**PTX level**: it scans forward in the basic block to verify that no
subsequent instruction reads `%dest` as a 64-bit pair.

This check is **insufficient** because IADD.64 writes the **physical
register pair** `R+1` (where R is the allocated physical register for
`%dest`), and the corruption affects whatever the **register allocator
put at R+1** — which is *not* visible at the PTX level.

In other words:
* PTX-level: `%dest`'s HI half is unused → looks safe.
* Physical level: `R+1` may be allocated to a *different* live vreg
  whose value is read by a subsequent instruction. Writing garbage
  to `R+1` then silently corrupts that other value.

The eligibility-table probe in IM02 sampled 16 kernels and showed
zero hard mismatches against TARGET / forbidden controls. But the
broader corpus (tests/) contains hundreds of kernels whose register
allocations were not enumerated, and many of those have the
PTX-level pattern that admits the substitution while having an
allocated `R+1` carrying a live value.

## Why the eligibility-table probe missed this

The IM02 probe operated only on parsed PTX (no allocation, no isel
emission). It correctly verified the **PTX-level** gate logic. It
did *not* simulate the actual allocator's register-pair aliasing,
because that would have required driving the full pipeline per
kernel — exactly what `pytest tests/` does, and exactly where the
real failures emerged.

## What this means for the IMAD/UIADD subsystem

The bounded first slice as originally specified (single PTX-level
look-ahead deciding IADD3 → IADD.64) is **not byte-safe** under the
current allocator's pair-aliasing behavior. To safely admit IADD.64
emission we would need at least one of:

1. **Register-level liveness analysis at the IADD.64 candidate site**:
   verify `R+1` is not allocated to any live vreg's value at that
   physical-register snapshot. This requires either a post-allocation
   isel hook or a liveness-aware allocator API that does not currently
   exist in `sass/regalloc.py`.

2. **Allocator-level pair reservation**: if the allocator could
   guarantee that vregs participating in the `add.u32 reg-reg → STG`
   pattern are placed in physical pairs whose `R+1` is reserved as
   a scratch (free), the existing PTX-level helper would suffice.
   This is an allocator change.

3. **Whole-kernel template approach**: lift the entire kernel into a
   PTXAS-byte-template (analogous to AT01–AT12 atom-family work), so
   the whole register layout is fixed by the template and no post-hoc
   substitution is needed. This is the AT01–AT12 pattern, but applied
   to a much wider family — likely an oversized first slice.

None of these is a one-isel-helper-line fix; each is its own bounded
sub-subsystem.

## Per HARD BAIL rules

* Stop immediately ✓
* Revert unsafe changes ✓ (pytest 865/865 restored)
* Report exact blocker ✓ (this document)
* Do not continue (no IM04) — atom-family progress and all prior work
  intact.

## State preserved

* pytest **865 / 865 green** (verified post-revert)
* frontier **50 BYTE_EXACT / 94 STRUCTURAL / 0 MIXED / 0 errors**
* GPU harness baseline unchanged (would need re-run to confirm but
  pytest is the gate per AT07 lesson — the IM03 failure surfaced
  immediately in pytest)
* IM01 / IM02 commits intact (analysis docs + the inert
  `_im_iadd64_admissible` helper remain in the codebase as evidence
  for the next attempt)
* All prior subsystem work (MP02, UI03, AT01–AT12) intact

## Honest roadmap revision

The user's question "Is IMAD/UIADD still the highest-leverage next
continuation?" gets a more nuanced answer than UI04/AT12 implied:

* By raw missing-opcode count, yes (67 missing UIADD + 22 missing
  IADD.64 are still the largest single bucket).
* By **bounded-isel-substitution feasibility**, the IADD.64
  substitution is BLOCKED by allocator pair-aliasing. A successful
  attempt would require allocator-level coordination, not a pure
  isel-level slice.

The honest next attempts (each its own future sprint chain, none
runnable as a one-isel-line slice):

1. **Allocator pair-reservation precursor**: identify whether the
   regalloc has a clean hook to reserve `R+1` for chosen vregs.
   If yes, IADD.64 substitution can resume.
2. **SHF harvest** (the standing offer from MP03/UI04). Smaller
   per-sprint leverage but no allocator coordination required.
3. **Whole-kernel template approach** for specific delta=0 kernels,
   matching the AT01–AT12 atom-template pattern.

I am NOT recommending option 1 here — it would be an allocator-rewrite-
adjacent move that the run's prohibitions explicitly forbid. The
honest recommendation is **SHF harvest** for the next bounded sprint
chain.
