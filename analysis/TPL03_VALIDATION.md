# TPL03 — Validation matrix for k100_dual_load whole-kernel template

Evidence for commit `983b221` (TPL02).

## 03.1 — Target validation matrix

| target | class before TPL01 | class after TPL03 | residual delta | GPU |
|---|---|---|:-:|---|
| **k100_dual_load** | STRUCTURAL | **BYTE_EXACT** | zero | **PASS** |

Workbench `diff_kernel` confirms zero residual opcode delta. PTXAS
and OURS produce byte-identical 19-instruction sequences over `.text`
(plus identical NOP padding).

## 03.2 — Controls

| control kernel       | expected   | actual     | unchanged? | GPU      | role |
|----------------------|------------|------------|:-:|----------|------|
| r1_running_xor       | STRUCTURAL | STRUCTURAL | ✓ | FAIL\*   | C00 sibling — must NOT pick up template |
| r1_scatter_add       | STRUCTURAL | STRUCTURAL | ✓ | PASS     | C00 sibling — must NOT pick up template |
| r1_multi_stage       | STRUCTURAL | STRUCTURAL | ✓ | PASS     | C11 — must NOT pick up template |
| k300_nasty_zero_init | STRUCTURAL | STRUCTURAL | ✓ | PASS     | C16 — must NOT pick up template |
| k100_atom_xor        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | atom template baseline |
| k100_atom_max        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | AT02 |
| k100_atom_add        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | AT06 |
| multi_block_atomic   | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | AT10 |
| smem_exchange        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | unrelated UIADD BE |
| k100_guarded_store   | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | unrelated single-pred BE |
| k100_pred_arith      | STRUCTURAL | STRUCTURAL | ✓ | PASS     | MP02 multi-pred |
| w1_loop_countdown    | STRUCTURAL | STRUCTURAL | ✓ | FAIL\*   | loop kernel |
| vecadd_large         | STRUCTURAL | STRUCTURAL | ✓ | PASS     | HFMA2 kernel |
| ilp_unrolled_sum4    | STRUCTURAL | STRUCTURAL | ✓ | PASS     | SHF kernel |

\* `r1_running_xor` and `w1_loop_countdown` GPU FAIL is the
pre-existing baseline state (in MP03's known-unsupported-families
table) — unchanged by TPL02.

## 03.3 — Residual blocker check

k100_dual_load reaches BYTE_EXACT with zero residual delta. No
remaining blockers for this kernel.

## Full validation summary

| stage | result |
|---|---|
| pytest | **865 / 865 green** |
| GPU harness | **126 PASS / 11 FAIL / 7 RUN_EXC / 0 COMPILE_FAIL** — identical to MP03 baseline |
| workbench frontier | **51 BYTE_EXACT / 93 STRUCTURAL / 0 MIXED / 0 errors** (+1 from MEGA-01's 50/94/0) |
| targeted kdiff | k100_dual_load byte-for-byte identical to PTXAS over `.text` |
| TPL02 dispatcher trigger | confirmed via verbose: `[TPL01] whole-kernel template applied for k100_dual_load` |
| Atom-template baselines (AT01–AT12) | all unchanged |
| MP02 / UI03 | all unchanged |
| AT07 lesson | respected — GPU harness ran before TPL02 declared clean |
