# PERF-7: Cross-Stack Performance Synthesis

**Date:** 2026-04-12  
**Scope:** OpenPTXas v1.0 — all 27 workbench kernels (21 original + 6 ILP)

## 1. Current Performance State

OpenPTXas **outperforms PTXAS** on both register pressure and instruction count across the 27-kernel benchmark suite.

### Register Pressure

| Metric | Value |
|---|---|
| OURS uses fewer regs | 16 kernels |
| OURS matches PTXAS | 10 kernels |
| OURS uses more regs | 1 kernel (dmma_zero: +3) |
| **Net delta** | **−101 registers (OURS wins)** |

Best wins: vecadd_large (−12), ilp_pipeline_load (−12), ilp_dual_int32 (−11).

### Instruction Count

| Metric | Value |
|---|---|
| OURS real instructions | 999 |
| PTXAS real instructions | 1007 |
| **Net delta** | **−8 instructions (−0.8%, OURS wins)** |

Best wins: imma_zero (−5), atom_or (−4), atomg_add (−4), hmma_zero (−3).

### NOP Census

| Category | Count |
|---|---|
| Total NOPs | 227 (across 27 kernels) |
| Structural padding (post-EXIT) | 217 (96%) |
| Body latency NOPs | 10 (4%) |
| Removable body NOPs | 0 |

The 10 body latency NOPs are all on critical-path dependency chains with no independent instructions available for fill (PERF-3). The 217 padding NOPs are structural requirements of the SM_120 text section format.

## 2. Gains from PERF Series

| Pass | Finding | Outcome |
|---|---|---|
| PERF-1 | Forwarding-safe NOP removal needs operand-role awareness | Infrastructure built |
| PERF-2 | Body NOPs serve dual purposes (ALU RAW + memory scoreboard) | Single-edge removal unsafe |
| PERF-3 | All 10 body NOPs are on critical paths, zero fillable | Scheduler already optimal for current shapes |
| PERF-4 | ILP kernels have zero body NOPs — scheduler handles ILP | Confirmed: scheduler is competent |
| PERF-5 | OURS uses 101 fewer registers than PTXAS | No action needed — already ahead |
| PERF-6 | OURS uses 8 fewer instructions than PTXAS | No action needed — already ahead |

## 3. Remaining Gaps

### 3.1 Register over-declaration (dmma_zero)
- OURS declares 10 GPRs vs PTXAS's 7
- Cause: allocator reserves scratch pairs in the high-water-mark
- Impact: reduced occupancy for DMMA-heavy kernels
- Fix: allocator architecture change (not surgical)

### 3.2 Address calculation pattern (ILP kernels)
- OURS emits IADD3+IADD3X (2 instrs) for 64-bit offset computation
- PTXAS uses LEA (1 fused shift+add instruction)
- Impact: +2 instructions per address calc in affected kernels
- Fix: LEA-fusion peephole in isel (medium complexity)

### 3.3 Body NOP dual-purpose limitation
- 10 body NOPs cannot be removed because they simultaneously guard ALU RAW edges AND memory-load scoreboard waits
- Fix: full multi-dependency rescheduler (high complexity)

## 4. Next Possible Directions

1. **LEA fusion peephole** — detect `cvt.u64 + shl.b64 + add.u64` and emit LEA+IADD3X. Would save ~2 instructions per affected kernel. Medium effort.

2. **Allocator high-water-mark tightening** — reduce the scratch-pair reservation for small kernels. Would fix the dmma_zero +3 regression. Medium effort.

3. **Multi-dependency NOP rescheduler** — full dependency analysis at each NOP position to safely remove dual-purpose NOPs. High effort, currently blocked by the 10-NOP critical-path finding.

4. **Loop-body scheduling** — for looped kernels (conv2d_looped, reduce_sum), software pipelining across loop iterations could hide memory latency. High effort.

## 5. Proof Model State

The FG-series proof engine (FG-2.4 → FG-4.8) is complete:
- 13 proof classes covering ALU, memory, UR, and forwarding edges
- 23 forwarding-safe pairs, all evidence-backed via GPU runtime
- 51/51 adversarial kernels confirmed (0 false positives, 0 false negatives)
- 54/54 corpus kernels SAFE, 554 proof edges, 0 violations

The proof model is no longer a performance bottleneck — it's a safety net that enables confident optimization.

## 6. Bottom Line

**OpenPTXas is performance-competitive with PTXAS.** It uses fewer registers (−101), fewer instructions (−8), and has zero unnecessary latency NOPs in the scheduling output. The remaining gaps (LEA fusion, allocator tightening) are incremental refinements, not systemic deficits.
