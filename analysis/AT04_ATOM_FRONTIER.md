# AT04 — Atom-frontier recompute and continuation

Evidence for commit `8e910fd` (AT03), built on AT02's `ca86b4c`.

## 04.1 — Atom-frontier delta

| metric                            | before AT01 | after AT04 |
|-----------------------------------|------------:|-----------:|
| Atom kernels — BYTE_EXACT         | 2           | **4** (+k100_atom_max, +k100_atom_min) |
| Atom kernels — STRUCTURAL         | 11          | **9** |
| MIXED                             | 0           | 0          |
| REG_AND_CTRL                      | 0           | 0          |
| pytest                            | 865/865     | 865/865    |
| GPU PASS / FAIL / RUN_EXC         | 126 / 11 / 7| 126 / 11 / 7 (no change) |
| **Corpus BYTE_EXACT**             | 46          | **48 (+2)**|
| **Corpus STRUCTURAL**             | 98          | **96**     |
| Corpus missing UIADD (0x835)      | 67          | 67         |
| Corpus missing IADD3.UR (0xc11)   | 14          | 14         |
| Corpus missing ISETP.UR (0xc0c)   | 9           | **7 (−2)** |
| Corpus missing IMAD.UR (0xc24)    | 1           | 1          |

Per-kernel atom residuals after AT02:

| kernel              | class      | residual missing opcodes (atom-preamble bundle) |
|---------------------|------------|------------------------------------------------|
| k100_atom_xor       | BYTE_EXACT | (none) |
| w2_atom_xor_reduce  | BYTE_EXACT | (none) |
| **k100_atom_max**   | **BYTE_EXACT** | (none — closed) |
| **k100_atom_min**   | **BYTE_EXACT** | (none — closed) |
| k100_atom_add       | STRUCTURAL | LDC, S2R, ISETP.UR, 0x886, 0xd09, 0x2bd, 0x98e |
| atomg_add           | STRUCTURAL | LDC, 0x886, 0x2bd, 0x2bf, IMAD.UR, ISETP.UR, 0x98e |
| multi_block_atomic  | STRUCTURAL | 0x886, 0xd09, 0x2bd, ISETP.UR, 0x98e |
| r1_histogram8       | STRUCTURAL | IMAD.I, **HFMA2**, 0x802, 0x98e |
| w2_loop_atom_add    | STRUCTURAL | LDC, S2R, ISETP.UR, 0x886, 0xd09, 0x2bd, 0x98e×3 |
| w2_atom_and_reduce  | STRUCTURAL | LDC, S2R, ISETP.UR, 0x886, MOV.UR, 0x2bd, MOV.UR.alt, 0x98e |
| atom_or             | STRUCTURAL | LDC, **HFMA2**, 0x886, MOV.UR, 0x2bd, MOV.UR.alt, ISETP.UR, 0x98e |
| k100_atom_cas32     | STRUCTURAL | **HFMA2**, MOV.UR.alt |
| atom_cas64          | STRUCTURAL | LDCU, MOV.UR.alt |

## 04.2 — Continuation ranking

| rank | next variant set | atom kernels reachable | next blocker | recommendation |
|-----:|------------------|------------------------:|--------------|----------------|
| 1 | **atom.{add,or}.<op> with immediate data** | 2 (k100_atom_add, multi_block_atomic) | new template variant: data sourced from immediate-loaded UR (not S2UR-of-tid). Different UMOV preamble shape than AT02; family_atom_ur.json needs a third variant. | bounded but **new** template machinery, larger scope than AT02 |
| 2 | atom kernels needing HFMA2 | 3 (r1_histogram8, atom_or, k100_atom_cas32) | HFMA2 subsystem | **prohibited this run** |
| 3 | atom.cas (32 + 64) | 2 (k100_atom_cas32, atom_cas64) | 2-data-operand ATOMG shape; entirely different template | new subsystem |
| 4 | looped atom (w2_loop_atom_add) | 1 | loop body BRA back-edge | loop subsystem |
| 5 | param-data atom (atomg_add, w2_atom_and_reduce) | 2 | data is param-loaded or computed; needs param→UR data routing | new template variant + IMAD.UR support |

Outside the atom family (still the largest corpus gaps from MP04/UI04):

| competing subsystem | corpus gap | scope size |
|---|---|---|
| UIADD remaining (multi-pred / IMAD-coordinated) | 67 missing UIADD | large; requires MP02 expansion or IMAD subsystem |
| HFMA2/FMUL.I subsystem | 43 missing | medium; explicitly prohibited this run |
| IADD.64 (0x235) | 22 missing | medium; 64-bit address widening |
| **SHF harvest (standing offer from MP03/UI04)** | residual byte-encoder work | small bounded slices possible |

## 04.3 — One next move

> **Next move: continue atom-family expansion into the
> immediate-data variant** — add a third entry to `family_atom_ur.json`
> for `atom.<op>.u32 <addr>, <imm_data>` (no SR data, immediate
> materialized through a UMOV-of-immediate path). Targets:
> `k100_atom_add` (literal `1`) and, after constant folding through a
> `mov.u32 %r0, 1`, `multi_block_atomic`.
>
> **Why this and not the other options**:
>
> * The AT01–AT04 sprint chain proved the template-expansion model
>   works without regressions. Reusing the same shape (one new
>   variant, one new isel admission gate) keeps the next slice on the
>   *same risk profile* as AT02 — no new subsystem boundary.
> * SHF harvest is a valid alternative (offered by MP03/UI04 as a
>   standing opportunity) but is unrelated to this run's atom-template
>   subsystem; switching would orphan the atom progress mid-sprint.
> * UIADD remaining work and HFMA2 require crossing out-of-scope
>   gates from previous sprints (MP02 multi-pred / HFMA2 prohibition).
> * Atom.cas / looped / param-data atoms each need a different
>   subsystem than the atom-UR-template; they should each get their
>   own scoped sprint chain rather than being lumped under "atom".
>
> **Concrete deliverables for the next sprint chain (AT05-style)**:
> 1. PTXAS byte capture for `k100_atom_add` and a new IMM-data
>    `family_atom_ur.json` variant.
> 2. Isel admission gate: `atom.add.u32 <addr>, <ImmOp>` (and
>    optionally constant-folded reg-with-imm).
> 3. Validation matrix mirroring AT03: targets BYTE_EXACT, atom.xor
>    baseline + AT02 max/min controls unchanged.

## Commit table for this sprint chain

| phase | commit | pushed |
|---|---|:-:|
| AT01 | `4a9112a` | ✓ |
| AT02 | `ca86b4c` | ✓ |
| AT03 | `8e910fd` | ✓ |
| AT04 | (this doc) | pending commit |
