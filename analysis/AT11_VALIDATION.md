# AT11 — Validation matrix for the no-tid-guard atom sibling

Evidence for commit `cd467bd` (AT10).

## 11.1 — Target validation matrix

| target              | class before AT09 | class after AT11 | residual missing | residual extra | GPU |
|---------------------|-------------------|------------------|------------------|----------------|-----|
| multi_block_atomic  | STRUCTURAL        | **BYTE_EXACT**   | (none)           | (none)         | PASS |

Workbench `diff_kernel` confirms zero residual opcode delta. PTXAS and
OURS produce byte-identical 11-instruction sequences over `.text`.

## 11.2 — Controls

| control kernel       | expected   | actual     | unchanged? | GPU      | role |
|----------------------|------------|------------|:-:|----------|------|
| k100_atom_add        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | AT06 tid-guard K=1 (must remain on AT06 path) |
| k100_atom_xor        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | baseline atom.xor template |
| k100_atom_max        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | AT02 max baseline |
| k100_atom_min        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | AT02 min baseline |
| atomg_add            | STRUCTURAL | STRUCTURAL | ✓ | PASS     | param data — must NOT pick up template |
| r1_histogram8        | STRUCTURAL | STRUCTURAL | ✓ | PASS     | has tid prelude — must NOT pick up no-tid variant |
| w2_loop_atom_add     | STRUCTURAL | STRUCTURAL | ✓ | PASS     | looped — must NOT pick up template |
| atom_or              | STRUCTURAL | STRUCTURAL | ✓ | PASS     | data value 0xFF ≠ 1 — must NOT pick up template |
| k100_atom_cas32      | STRUCTURAL | STRUCTURAL | ✓ | PASS     | CAS shape |
| atom_cas64           | STRUCTURAL | STRUCTURAL | ✓ | PASS     | CAS 64-bit |
| w2_atom_and_reduce   | STRUCTURAL | STRUCTURAL | ✓ | RUN_EXC* | computed data + tid — pre-existing baseline failure (MP03) |
| k100_guarded_store   | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | unrelated single-pred |
| smem_exchange        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | unrelated UIADD |
| k100_pred_arith      | STRUCTURAL | STRUCTURAL | ✓ | PASS     | MP02 multi-pred control |
| ilp_alu_addr         | STRUCTURAL | STRUCTURAL | ✓ | PASS     | unrelated STRUCTURAL non-atom |

\* `w2_atom_and_reduce` RUN_EXC is the pre-existing baseline state (in
MP03's known-unsupported-families table); unchanged by AT10.

## 11.3 — Residual blocker check

multi_block_atomic reaches BYTE_EXACT with zero residual delta. No
remaining blockers for this kernel.

## Full validation summary

| stage | result |
|---|---|
| pytest | **865 / 865 green** |
| GPU harness | **126 PASS / 11 FAIL / 7 RUN_EXC / 0 COMPILE_FAIL** — identical to MP03 baseline |
| workbench frontier | **50 BYTE_EXACT / 94 STRUCTURAL / 0 MIXED / 0 errors** (+1 from AT08's 49/95/0) |
| targeted kdiff | multi_block_atomic byte-for-byte identical to PTXAS over `.text` |
| AT06 / AT02 / MP02 / UI03 | all unchanged |
| AT07 lesson | respected — GPU harness ran before AT10 declared clean |
