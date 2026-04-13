# Full Corpus Byte-Exact Audit: OURS vs PTXAS

**Date:** 2026-04-13
**Corpus:** 144 kernels (workbench)
**Method:** Byte-level comparison of active (non-NOP) instructions

## Summary

| Category | Count | Percentage |
|---|---|---|
| Byte-exact | 29 | 20.1% |
| Structural | 114 | 79.2% |
| Mixed | 1 | 0.7% |

## Instruction Delta

| Metric | Value |
|---|---|
| Mean absolute delta | 1.9 instructions |
| Median absolute delta | 1 instruction |
| At parity (delta=0) | 69 kernels (47.9%) |
| OURS smaller | 25 kernels |
| OURS larger | 50 kernels |

## Byte-Exact Kernels (29)

Shared memory: smem_exchange, smem_cycle, w1_smem_copy, w1_smem_compute,
w1_smem_neighbor, w1_smem_xor_swap, w1_smem_guarded, w1_smem_reduce_pair,
w2_smem_loop

Warp operations: reduce_sum, redux_sum, warp_reduce, k100_shfl_down,
k100_shfl_up, k100_shfl_xor, k100_redux_and, k200_shfl_reduce2,
k300_shfl_idx, k300_nasty_shfl_chain, r1_scan_warp, r1_warp_sum,
r1_tile_compute

Special: conv2d_looped (306 instrs), conv2d_unrolled (281 instrs),
cp_async, fmax, bar_ldc_xor

Template-driven: k100_atom_xor, w2_atom_xor_reduce

## Top 5 Largest Gaps

| Kernel | Delta | Ours | PTXAS | Explanation |
|---|---|---|---|---|
| w1_loop_pred_acc | -57 | 19 | 76 | PTXAS unrolls loop, OURS doesn't |
| w2_div_loop | -20 | 20 | 40 | PTXAS unrolls loop |
| w2_nested_loop | +10 | 23 | 13 | OURS generates extra ALU for loop control |
| k300_nasty_long_live | +8 | 21 | 13 | OURS: more IMAD.WIDE for addr computation |
| k300_nasty_deep_dep | +8 | 21 | 13 | Same pattern as above |

## Root Cause of Structural Differences

The 79.2% structural difference is instruction selection, not register
allocation (TE7-A proved zero REG_ONLY differences exist).

PTXAS uses the UR file extensively for parameter handling:
- ISETP.R-UR (+114 instances over OURS)
- LDCU (+101), UIADD (+93)

OURS uses GPR equivalents:
- ISETP.R-R (+105 over PTXAS)
- IADD3.IMM (+154), IMAD.WIDE (+102)

Both produce correct code. The gap is in isel-level UR utilization.
