# AT03 — Validation matrix for the bounded atom-template expansion

Evidence for commit `ca86b4c` (AT02). All numbers from `scripts/health.py`.

## 03.1 — Target validation matrix

| kernel          | class before | class after | byte_match before | byte_match after | missing opcodes after | extra opcodes after | GPU |
|-----------------|--------------|-------------|------------------:|-----------------:|-----------------------|---------------------|-----|
| k100_atom_max   | STRUCTURAL   | **BYTE_EXACT** | 11/16          | **16/16**        | (none)                | (none)              | PASS |
| k100_atom_min   | STRUCTURAL   | **BYTE_EXACT** | 11/16          | **16/16**        | (none)                | (none)              | PASS |

Both targets transition to BYTE_EXACT with zero residual opcode delta.

## 03.2 — Controls

| control kernel       | expect      | actual      | unchanged? | GPU  | role |
|----------------------|-------------|-------------|:-:|------|------|
| k100_atom_xor        | BYTE_EXACT  | BYTE_EXACT  | ✓ | PASS | atom.xor template baseline |
| w2_atom_xor_reduce   | BYTE_EXACT  | BYTE_EXACT  | ✓ | PASS | atom.xor (tid+K) baseline |
| k100_atom_add        | STRUCTURAL  | STRUCTURAL  | ✓ | PASS | atom.add (imm data — must NOT pick up template) |
| atom_or              | STRUCTURAL  | STRUCTURAL  | ✓ | PASS | atom.or (imm-in-reg — must NOT pick up template) |
| w2_loop_atom_add     | STRUCTURAL  | STRUCTURAL  | ✓ | PASS | looped atom (must NOT pick up template) |
| k100_guarded_store   | BYTE_EXACT  | BYTE_EXACT  | ✓ | PASS | unrelated single-pred BYTE_EXACT |
| smem_exchange        | BYTE_EXACT  | BYTE_EXACT  | ✓ | PASS | unrelated UIADD BYTE_EXACT |
| k100_pred_arith      | STRUCTURAL  | STRUCTURAL  | ✓ | PASS | MP02 multi-pred control |
| ilp_alu_addr         | STRUCTURAL  | STRUCTURAL  | ✓ | PASS | unrelated STRUCTURAL non-atom |

## 03.3 — Residual blocker table

| target          | remaining blockers after AT02 | next blocker type |
|-----------------|-------------------------------|-------------------|
| k100_atom_max   | none — BYTE_EXACT achieved    | — |
| k100_atom_min   | none — BYTE_EXACT achieved    | — |

## Full validation summary

| stage | result |
|---|---|
| pytest | **865 / 865 green** (~83 s) |
| GPU harness | 126 PASS / 11 FAIL / 7 RUN_EXC / 0 COMPILE_FAIL — **identical to MP03 baseline** (no new GPU regressions; the 18 failing kernels are all in the documented known-unsupported families) |
| workbench frontier | **48 BYTE_EXACT / 96 STRUCTURAL / 0 MIXED / 0 errors** (+2 BYTE_EXACT vs AT01 baseline 46/98/0) |
| targeted kdiff (k100_atom_max, k100_atom_min) | byte-for-byte identical to PTXAS over the .text section (verified via `diff_kernel`) |
| MP02 multi-pred fixes | unchanged (k100_pred_arith STRUCTURAL+GPU PASS; existing FG33 / FG56 gates intact) |
| UI03 UIADD fixes | unchanged (k100_ldg_add_stg / w1_div_load_paths still emit UIADD at correct positions; smem_exchange BYTE_EXACT) |
