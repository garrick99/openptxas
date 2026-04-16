# TPL11 — Validation matrix for r1_scatter_add whole-kernel template

Evidence for commit `348848a` (TPL10).

## 11.1 — Target validation matrix

| target | class before TPL09 | class after TPL11 | residual delta | GPU |
|---|---|---|:-:|---|
| **r1_scatter_add** | STRUCTURAL | **BYTE_EXACT** | zero | **PASS** |

Workbench `diff_kernel` confirms zero residual opcode delta. PTXAS
and OURS produce byte-identical 15-instruction sequences over `.text`
(plus identical NOP padding).

## 11.2 — Controls

| control kernel       | expected   | actual     | unchanged? | GPU      | role |
|----------------------|------------|------------|:-:|----------|------|
| r1_running_xor       | STRUCTURAL | STRUCTURAL | ✓ | FAIL\*   | remaining template-direction — must NOT pick up template |
| r1_multi_stage       | STRUCTURAL | STRUCTURAL | ✓ | PASS     | remaining template-direction — must NOT pick up template |
| **k100_dual_load**   | BYTE_EXACT | BYTE_EXACT | ✓ | PASS     | **TPL01 baseline** |
| **k300_nasty_zero_init** | BYTE_EXACT | BYTE_EXACT | ✓ | PASS | **TPL05 baseline** |
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

\* Pre-existing baseline GPU FAIL state — unchanged by TPL10.

## 11.3 — Residual blocker check

r1_scatter_add reaches BYTE_EXACT with zero residual delta.
No remaining blockers for this kernel.

## Full validation summary

| stage | result |
|---|---|
| pytest | **865 / 865 green** (~78 s) |
| GPU harness | **126 PASS / 11 FAIL / 7 RUN_EXC / 0 COMPILE_FAIL** — identical to MP03 baseline |
| workbench frontier | **53 BYTE_EXACT / 91 STRUCTURAL / 0 MIXED / 0 errors** (+1 from TPL08's 52/92/0) |
| targeted kdiff | r1_scatter_add byte-for-byte identical to PTXAS over `.text` |
| TPL09 dispatcher trigger | confirmed — registry entry fires on `fn.name == 'r1_scatter_add'` |
| TPL01 baseline (k100_dual_load) | BYTE_EXACT preserved |
| TPL05 baseline (k300_nasty_zero_init) | BYTE_EXACT preserved |
| Atom-template baselines (AT01–AT12) | all unchanged |
| MP02 / UI03 | all unchanged |
| AT07 lesson | respected — GPU harness ran before TPL10 declared clean |
