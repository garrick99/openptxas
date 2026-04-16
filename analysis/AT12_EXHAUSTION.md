# AT12 — Atom-frontier recompute and bounded-scope exhaustion

Evidence for commit `500bb6c` (AT11), built on AT10's `cd467bd`.

## 12.1 — Atom-frontier delta

| metric                            | before AT09 | after AT12 |
|-----------------------------------|------------:|-----------:|
| Atom kernels — BYTE_EXACT         | 5           | **6** (+multi_block_atomic) |
| Atom kernels — STRUCTURAL         | 8           | **7** |
| Corpus BYTE_EXACT                 | 49          | **50** |
| Corpus STRUCTURAL                 | 95          | **94** |
| MIXED / REG_AND_CTRL / errors     | 0 / 0 / 0   | 0 / 0 / 0  |
| pytest                            | 865/865     | 865/865    |
| GPU PASS / FAIL / RUN_EXC         | 126 / 11 / 7| 126 / 11 / 7 |

Per-kernel atom state after AT11 (7 STRUCTURAL remaining):

| kernel              | delta | hard blockers stacked on this kernel |
|---------------------|------:|---------------------------------------|
| atomg_add           | −4    | param-data routing (no tid guard, IMAD.UR needed) |
| r1_histogram8       | −1    | computed tid-derived address **AND** HFMA2 |
| w2_loop_atom_add    | −1    | loop body |
| w2_atom_and_reduce  | −5    | computed data (`or %tid, K`) **AND** tid prelude — needs new template |
| atom_or             | −4    | HFMA2 (PTXAS emits 0x431) **AND** data value ≠ 1 |
| k100_atom_cas32     | +1    | CAS shape (2 data operands) |
| atom_cas64          | −1    | CAS 64-bit (2 data operands, pair widths) |

## 12.2 — Bounded-scope exhaustion test

Each remaining STRUCTURAL atom kernel evaluated against the proven
atom-UR template machinery:

| kernel              | reachable by current machinery? | reason |
|---------------------|:-:|--------|
| atomg_add           | NO | param-data routing requires a new template variant for `atom.<op>.u32 [base], <ld.param-loaded reg>`. The data path differs (LDCU → MOV.UR → ATOMG with UR-data, where data UR is loaded from a param byte offset, not from a `mov %r, IMM`). Outside the K=1 immediate scope. |
| r1_histogram8       | NO | computed address `[%rd0 + (%tid & 7) * 4]` is not in `_ur_params` and `_gpr_written` is set via `add.u64`. The hardcoded ATOMG bytes assume the address is the param descriptor's UR4. AND PTXAS emits HFMA2 (0x431) elsewhere — HFMA2 prohibited this run. |
| w2_loop_atom_add    | NO | loop body. The 11- or 15-instruction template emits exactly one ATOMG; a per-iteration repeated atomic semantic cannot be expressed without a separate loop-aware template. Loop subsystem out of scope. |
| w2_atom_and_reduce  | NO | data is `or.b32 %r2, %tid, 0xFFFF0000` (computed). Has tid prelude + non-1 computed data + PTXAS emits HFMA2. Multiple blockers; needs a new template variant for "computed-data atom" plus HFMA2 work. |
| atom_or             | NO | data value 0xFF (≠ 1) means our K=1 imm-data variants all reject; PTXAS additionally emits HFMA2. Needs a new "K = arbitrary" template plus HFMA2 work. |
| k100_atom_cas32     | NO | atom.cas has TWO data operands (cmp + new). The 0x98e ATOMG opcode and template assume a single data UR; CAS uses a different ATOMG family (0x3a9). New CAS subsystem required. |
| atom_cas64          | NO | atom.cas.b64 — same as cas32 plus 64-bit register-pair widths. Different ATOMG.E.CAS encoding family. New subsystem. |

**Verdict: bounded atom-family scope is exhausted.** Every one of the 7
remaining STRUCTURAL atom kernels is blocked by an explicitly out-of-
scope item (HFMA2 work prohibited, looped/CAS/param-data each a new
subsystem). No further bounded sibling variant within the atom-UR
template machinery can reach any of them without crossing a subsystem
boundary.

## 12.3 — One next move

> **Next move: pivot to the IMAD/UIADD coordinated subsystem precursor
> analysis** (the next 4-phase sprint chain — name it however appropriate,
> e.g. IM01–IM04).
>
> **Why IMAD/UIADD over SHF**:
>
> * UI04 evidence: the IMAD/UIADD-coordinated bucket is the **largest
>   single remaining gap** in the corpus. After AT12 the corpus shows
>   50 BYTE_EXACT / 94 STRUCTURAL with the dominant remaining missing
>   opcode still being UIADD (67 instances) — most of those are
>   blocked behind IMAD coordination requirements that AT01–UI04
>   identified explicitly.
> * SHF harvest is a standing offer (deferred since MP03/UI04) but is
>   structurally a series of small bounded slices with limited
>   per-sprint leverage.
> * IMAD/UIADD has the same template-expansion + isel-eligibility
>   shape as the AT01–AT12 atom work, so the institutional pattern
>   is reusable: cluster → eligibility-tag → bounded substitution →
>   GPU validation.
> * Choosing IMAD/UIADD now also clears the path for a later HFMA2/
>   FMUL.I subsystem to be a clean self-contained sprint chain
>   (currently it is entangled with multiple atom-family kernels).
>
> **Concrete deliverables for the next sprint chain (IM01-style)**:
> 1. Cluster the 22 missing-IADD.64 + 67 missing-UIADD STRUCTURAL
>    kernels by exact opcode-diff signature against PTXAS.
> 2. Identify the bounded subcluster that is exact-shape safe under
>    isel-level (no post-scheduling rewrites — FG65/66 lesson).
> 3. Implement the eligibility tag + substitution.
> 4. Validate via pytest + GPU harness + frontier (AT07 lesson).

## Commit table for this sprint chain

| phase | commit | pushed |
|---|---|:-:|
| AT09 | `c1e67a6` | ✓ |
| AT10 | `cd467bd` | ✓ |
| AT11 | `500bb6c` | ✓ |
| AT12 | (this doc) | pending commit |
