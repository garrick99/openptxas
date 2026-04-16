# AT08 — Atom-frontier recompute and continuation

Evidence for commit `317109c` (AT07).

## 08.1 — Atom-frontier delta

| metric                            | before AT05 | after AT08 |
|-----------------------------------|------------:|-----------:|
| Atom kernels — BYTE_EXACT         | 4           | **5** (+k100_atom_add) |
| Atom kernels — STRUCTURAL         | 9           | **8** |
| Corpus BYTE_EXACT                 | 48          | **49** |
| Corpus STRUCTURAL                 | 96          | **95** |
| MIXED / REG_AND_CTRL / errors     | 0 / 0 / 0   | 0 / 0 / 0  |
| pytest                            | 865/865     | 865/865    |
| GPU PASS / FAIL / RUN_EXC         | 126 / 11 / 7| 126 / 11 / 7 (no change) |

Per-kernel atom state after AT07 (8 STRUCTURAL remaining):

| kernel              | delta | next blocker | reachable in bounded scope? |
|---------------------|------:|--------------|------:|
| **multi_block_atomic** | −1 | no-tid-guard prefix (4-instr difference vs imm_data_K1) | **YES — narrow variant** |
| atomg_add           | −4 | param data + IMAD.UR + no tid guard | new template machinery |
| r1_histogram8       | −1 | computed tid-derived address + HFMA2 | HFMA2 prohibited |
| w2_loop_atom_add    | −1 | loop body | loop subsystem |
| w2_atom_and_reduce  | −5 | computed data via or-with-const | new data path |
| atom_or             | −4 | HFMA2 + computed data | HFMA2 prohibited |
| k100_atom_cas32     | +1 | CAS shape (2 data ops) | CAS subsystem |
| atom_cas64          | −1 | 64-bit CAS shape | CAS subsystem |

PTXAS byte-evidence for multi_block_atomic vs k100_atom_add (11 vs 15
instructions): identical 11-instruction suffix; multi_block_atomic
simply omits the 4-instruction tid-guard prelude (positions [1–4] of
imm_data_K1: S2R-tid, LDCU(UR4), ISETP.UR, @P0 EXIT).

## 08.2 — Continuation ranking

| rank | next variant set | reachable | next blocker | recommendation |
|---:|------|------:|---|---|
| 1 | atom imm-data **no-tid-guard** prefix | 1 (multi_block_atomic) | new sibling JSON variant + admission-gate inversion (no tid prelude) | **bounded one-kernel slice** — same risk profile as AT06 |
| 2 | atom HFMA2-touching kernels | 3 | HFMA2 subsystem | prohibited this run |
| 3 | atom.cas subsystem | 2 | 2-data-operand ATOMG | new subsystem |
| 4 | atom param-data routing | 2 (atomg_add, w2_atom_and_reduce) | param→UR data routing + IMAD.UR | new subsystem |
| 5 | looped atom | 1 (w2_loop_atom_add) | loop body | loop subsystem |

Outside atom-family (corpus standing offers):

| competing subsystem | scope | leverage |
|---|---|---|
| SHF harvest (MP03 standing offer) | unknown precursor work needed | small bounded slices |
| IMAD/UIADD coordinated subsystem | new isel pass + IMAD.WIDE coordination | up to 22 kernels (largest single bucket) |
| HFMA2/FMUL.I subsystem | prohibited this run | 43 missing |

## 08.3 — One next move

> **Next move: continue atom-family expansion into the no-tid-guard
> sibling variant** (single-kernel target: `multi_block_atomic`).
>
> **Why this and not other options**:
>
> * Same template-expansion risk profile as AT06: one new JSON variant,
>   one new isel admission gate, per-op overrides already in place.
> * PTXAS byte-evidence already collected: multi_block_atomic's
>   11-instruction PTXAS output is byte-identical to the
>   imm_data_K1 11-instruction suffix; only the prelude differs.
> * Concrete +1 BYTE_EXACT, no exploration cost.
> * After this slice the atom-family bounded scope is genuinely
>   exhausted — remaining atom STRUCTURAL kernels need a separate
>   subsystem (HFMA2 / loops / CAS / param-routing) explicitly
>   prohibited or out of scope for the atom-template machinery.
> * Switching mid-stride to SHF or IMAD/UIADD would orphan the
>   atom-template momentum without any precursor analysis evidence.
>
> **Concrete deliverables for the next sprint chain (AT09-style)**:
> 1. New `imm_data_K1_no_tid_guard` variant in family_atom_ur.json
>    (11 instructions, identical suffix to imm_data_K1).
> 2. New isel admission shape: `atom.<op>.u32 <ld.param-base>, ImmOp(1)`
>    where the kernel does NOT have a tid-bounded guard prelude
>    (no SR_TID_X register in `_reg_sr_source`).
> 3. Pipeline dispatcher selector update.
> 4. Validation matrix mirroring AT07: target BYTE_EXACT, all current
>    BYTE_EXACT (5 atom + 5 unrelated controls) preserved, no GPU
>    regressions.
>
> After AT09–AT12 (or whatever the next 4-phase chain is named), the
> atom-family bounded scope will be exhausted and the next pivot
> should be SHF or IMAD/UIADD.

## Commit table for this sprint chain

| phase | commit | pushed |
|---|---|:-:|
| AT05 | `58b3e1b` | ✓ |
| AT06 | `02533d7` | ✓ |
| AT07 | `317109c` | ✓ (includes the gate-tightening fix) |
| AT08 | (this doc) | pending commit |
