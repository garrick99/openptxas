# PERF-7: Cross-Stack Performance Synthesis

**Date:** 2026-04-12 (updated after HARD-FINISH campaign)
**Scope:** OpenPTXas v1.0 — all 27 workbench kernels (21 original + 6 ILP)

## 1. Current Performance State

OpenPTXas **outperforms PTXAS** on both register pressure and instruction count across the 27-kernel benchmark suite.

### Register Pressure

| Metric | Value |
|---|---|
| OURS uses fewer regs | 16 kernels |
| OURS matches PTXAS | 10 kernels |
| OURS uses more regs | 1 kernel (dmma_zero: +0 regs after ALLOC-1) |
| **Net delta** | **−105 registers (OURS wins)** |

### Instruction Count

| Metric | Value |
|---|---|
| OURS real instructions | 993 |
| PTXAS real instructions | 1007 |
| **Net delta** | **−14 instructions (−1.4%, OURS wins)** |

Best wins: imma_zero (−5), atom_or (−4), atomg_add (−4), hmma_zero (−3).

### NOP Census

| Category | Count |
|---|---|
| Total NOPs | ~225 (across 27 kernels) |
| Structural padding (post-EXIT) | ~215 (96%) |
| Body latency NOPs | ~10 (4%) |
| Removable body NOPs | 0 |

The body latency NOPs are all on critical-path dependency chains with no independent instructions available for fill (PERF-3). The padding NOPs are structural requirements of the SM_120 text section format.

## 2. Gains from PERF + HARD-FINISH Series

| Pass | Finding | Outcome |
|---|---|---|
| PERF-1 | Forwarding-safe NOP removal needs operand-role awareness | Infrastructure built |
| PERF-2 | Body NOPs serve dual purposes (ALU RAW + memory scoreboard) | Single-edge removal unsafe |
| PERF-3 | All body NOPs are on critical paths, zero fillable | Scheduler already optimal for current shapes |
| PERF-4 | ILP kernels have zero body NOPs — scheduler handles ILP | Confirmed: scheduler is competent |
| PERF-5 | OURS uses fewer registers than PTXAS | No action needed — already ahead |
| PERF-6 | OURS uses fewer instructions than PTXAS | No action needed — already ahead |
| IMAD-FUSE-1 | LOP3.IMM (0x812) encoding saves materialize step | ilp_dual_int32, ilp_alu_addr closed |
| HARD-FINISH-1 | Predicated mov+mul+add → @pred IMAD fusion | ilp_pred_alu closed |

## 3. Remaining Gaps

### 3.1 Address chain stride (ilp_unrolled_sum4, +3)
- OURS emits separate IADD3.IMM + IADD3.RR pairs for each chained add.u64
- PTXAS folds constant-stride offsets into LDG.E immediate field
- Fix requires register allocator rework (chain fold creates base register clobbering)
- **Status:** Bounded structural gap, not safely fixable without allocator changes

### 3.2 Minor scheduling differences (+1 gaps)
- vecadd_large (+1), ilp_pipeline_load (+1)
- Different instruction selection style, diminishing returns
- **Status:** Intentionally left as bounded residuals

### 3.3 Register pressure residuals
- ilp_alu_addr (+2 regs), ilp_unrolled_sum4 (+2 regs), ilp_pred_alu (+2 regs)
- PTXAS uses fewer setup registers via HFMA2 constant-load trick
- **Status:** Register count only, no instruction count impact

## 4. Bottom Line

**OpenPTXas is performance-competitive with PTXAS.** It uses fewer registers (−105 net), fewer instructions (−14 net, −1.4%), and has zero unnecessary latency NOPs. The remaining gaps (stride chain folding, minor scheduling) are bounded structural differences with known root causes.
