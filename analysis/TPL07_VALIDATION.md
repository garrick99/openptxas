# TPL07 — Validation matrix for k300_nasty_zero_init whole-kernel template

Evidence for commit `432595c` (TPL06).

## 07.1 — Target validation matrix

| target | class before TPL05 | class after TPL07 | residual delta | GPU |
|---|---|---|:-:|---|
| **k300_nasty_zero_init** | STRUCTURAL | **BYTE_EXACT** | zero | **PASS** |

Workbench `diff_kernel` confirms zero residual opcode delta. PTXAS
and OURS produce byte-identical 13-instruction sequences over `.text`
(plus identical NOP padding).

## 07.2 — Controls

| control kernel       | expected   | actual     | unchanged? | GPU      | role |
|----------------------|------------|------------|:-:|----------|------|
| r1_running_xor       | STRUCTURAL | STRUCTURAL | ✓ | FAIL\*   | C00 sibling — must NOT pick up template |
| r1_scatter_add       | STRUCTURAL | STRUCTURAL | ✓ | PASS     | C00 sibling — must NOT pick up template |
| r1_multi_stage       | STRUCTURAL | STRUCTURAL | ✓ | PASS     | C11 — must NOT pick up template |
| **k100_dual_load**   | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | **TPL01 baseline (must remain BE)** |
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

\* Pre-existing baseline GPU FAIL state — unchanged by TPL06.

## 07.3 — Residual blocker check

k300_nasty_zero_init reaches BYTE_EXACT with zero residual delta.
No remaining blockers for this kernel.

## Full validation summary

| stage | result |
|---|---|
| pytest | **865 / 865 green** (~82 s) |
| GPU harness | **126 PASS / 11 FAIL / 7 RUN_EXC / 0 COMPILE_FAIL** — identical to MP03 baseline |
| workbench frontier | **52 BYTE_EXACT / 92 STRUCTURAL / 0 MIXED / 0 errors** (+1 from TPL04's 51/93/0) |
| targeted kdiff | k300_nasty_zero_init byte-for-byte identical to PTXAS over `.text` |
| TPL05 dispatcher trigger | confirmed via verbose: `[TPL05] whole-kernel template applied for k300_nasty_zero_init` |
| TPL01 baseline (k100_dual_load) | BYTE_EXACT preserved |
| Atom-template baselines (AT01–AT12) | all unchanged |
| MP02 / UI03 | all unchanged |
| AT07 lesson | respected — GPU harness ran before TPL06 declared clean |
