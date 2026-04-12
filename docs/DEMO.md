# OpenPTXas Demo

## What this is

Side-by-side comparison of NVIDIA's `ptxas` (reference compiler) vs OpenPTXas (OURS) on SM_120 (Blackwell / RTX 5090). Both compilers receive the same PTX input. Both cubins run on real GPU hardware. Output correctness is verified. Instruction counts, register usage, and NOP counts are compared. Every difference is explained.

## How to run

```
python demo/main.py --suite demo                    # 5-kernel showcase
python demo/main.py --suite ilp                     # 6 ILP kernels
python demo/main.py --suite full                    # all 27 kernels
python demo/main.py --kernel vecadd_large           # single kernel
python demo/main.py --kernel ilp_dual_int32 --diff  # transformation-grouped diff
python demo/main.py --suite demo --explain          # verbose explanations
python demo/main.py --proof                         # standalone proof check
```

## Output format

Suite summary columns:

| Column | Meaning |
|--------|---------|
| Correct | GPU correctness: PASS or FAIL |
| Instrs | Instruction delta (OURS - PTXAS). Negative = OURS wins. |
| Regs | Register delta (OURS - PTXAS). Negative = OURS wins. |
| Verdict | See below |

Verdict categories:

| Verdict | Meaning |
|---------|---------|
| PARITY | Same instruction count (registers may differ) |
| OURS WINS | Fewer instructions than PTXAS |
| +N gap (struct) | Bounded gap requiring allocator/architecture changes |
| +N gap (sched) | Scheduling or address generation style difference |
| +N gap (minor) | Diminishing returns, not worth pursuing |

## Example output

Full 27-kernel suite:

```
  Kernel                    Correct   Instrs     Regs Verdict
  ------------------------------------------------------------------
  reduce_sum                   PASS        0        0   PARITY
  conv2d_looped                PASS        0        0   PARITY
  conv2d_unrolled              PASS        0        0   PARITY
  hmma_zero                    PASS       -3        0   OURS WINS
  imma_zero                    PASS       -5        0   OURS WINS
  dmma_zero                    PASS        0        0   PARITY
  qmma_zero                    PASS       -1        0   OURS WINS
  cp_async                     PASS        0        0   PARITY
  warp_reduce                  PASS        0        0   PARITY
  atom_or                      PASS       -4        0   OURS WINS
  atomg_add                    PASS       -4        0   OURS WINS
  vecadd_large                 PASS       +1        0   +1 gap (sched)
  multi_ldg                    PASS        0        0   PARITY
  smem_exchange                PASS        0        0   PARITY
  fmax                         PASS        0        0   PARITY
  smem_cycle                   PASS        0        0   PARITY
  bar_ldc_xor                  PASS        0        0   PARITY
  dual_ldg64_dadd              PASS        0       +2   PARITY
  multi_block_atomic           PASS       -1       +3   OURS WINS
  atom_cas64                   PASS       -1       -2   OURS WINS
  redux_sum                    PASS        0        0   PARITY
  ilp_dual_int32               PASS        0        0   PARITY
  ilp_dual_int64               PASS        0       +2   PARITY
  ilp_alu_addr                 PASS        0       +2   PARITY
  ilp_unrolled_sum4            PASS       +3       +2   +3 gap (struct)
  ilp_pipeline_load            PASS       +1       +1   +1 gap (minor)
  ilp_pred_alu                 PASS        0       +2   PARITY
  ------------------------------------------------------------------
  TOTAL                                  -14            OURS 993 vs PTXAS 1007

  Wins: 7  |  Parity: 17  |  Bounded gaps: 3

  Proof: 51/51 adversarial CONFIRMED | 37/37 corpus SAFE
```

## Proof model

All scheduling decisions are validated by two independent proof systems:

- **Adversarial harness** (51 kernels): 7 families of adversarial kernels designed to expose false positives and false negatives in the dependency model. All 51 return MODEL_CONFIRMED.
- **Corpus proof** (37 kernels): Every workbench kernel, probe kernel, and predicate-test kernel is verified SAFE by the constructive proof engine (13 proof classes, evidence-backed forwarding pairs).

## Interpretation

- **Parity** is the expected outcome for most kernels. Both compilers target the same ISA and the instruction selection converges for standard patterns.
- **OURS WINS** shows where OpenPTXas optimizations (IMAD fusion, LOP3 immediate encoding, IMAD.WIDE address fusion) produce fewer instructions than PTXAS.
- **Bounded gaps** have known root causes (chained stride addresses, scheduling style) and are documented with category tags. No hidden correctness tradeoffs.
- **993 vs 1007 total instructions** across 27 kernels: OpenPTXas emits 14 fewer instructions than PTXAS (-1.4%).
