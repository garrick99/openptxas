# FREEZE-1: Verified and Performance-Tuned Stack Milestone

**Date:** 2026-04-12  
**Status:** FROZEN — all known defects resolved or bounded with evidence

---

## A. Stack Summary

### Forge Compiler
- **Commits:** 159
- **Language:** OCaml 5.0+ (Dune build, Z3 4.12+ runtime)
- **Demos:** 1,064 verified programs
- **Proofs:** 1,045+ obligations discharged (0 GCC failures on C99 output)
- **License:** Apache 2.0 + Commercial
- **HEAD:** `abf315d` (FG-2.6: close PTX backend integration bugs)

### OpenCUDA
- **Commits:** 565
- **Language:** Python 3.14
- **Tests:** 31,650 passed, 0 failed, 55 skipped
- **GPU E2E:** 30/30 on RTX 5090
- **HEAD:** `5a46ed8` (OC-6: update guarded-store test)

### OpenPTXas
- **Commits:** 315+
- **Language:** Python 3.14
- **Tests:** 585 passed, 0 failed
- **Proof corpus:** 54/54 SAFE, 554 edges, 0 violations
- **Adversarial harness:** 51/51 MODEL_CONFIRMED, 0 false positives, 0 false negatives
- **HEAD:** `03165ce` (PERF-6.1: LEA fusion investigation)

---

## B. Correctness Summary

| Suite | Total | Pass | Fail |
|---|---:|---:|---:|
| Forge demos | 1,064 | 1,064 | 0 |
| OpenCUDA pytest | 31,705 | 31,650 | 0 |
| OpenCUDA GPU E2E | 30 | 30 | 0 |
| OpenPTXas pytest | 585 | 585 | 0 |
| OpenPTXas proof corpus | 54 | 54 SAFE | 0 |
| OpenPTXas adversarial | 51 | 51 | 0 |
| **Cross-stack total** | **~33,489** | **~33,434** | **0** |

### Proof Model (FG-2.4 → FG-4.8)
- 13 proof classes (LATENCY_INERT through UR_MEMORY_VIOLATION)
- 23 forwarding-safe pairs (all evidence-backed via GPU runtime)
- 1 semantic fastpath (HFMA2 zero-init, ZERO_INIT_SAFE)
- CTRLWORD_SAFE and GAP_SAFE retired (R1 unified into R10)

### Bugs Fixed in This Campaign
- **OpenPTXas FG-4.4:** duplicate ld.param.u64 KeyError + mad.lo.u32 immediate miscompilation
- **OpenPTXas OC-4:** atomicAdd register clobber in atom isel
- **OpenCUDA OC-2:** branch target invalidation in false_is_ret peephole (30 tests)
- **OpenCUDA OC-3:** printf elimination in peephole (3 tests)
- **OpenCUDA OC-6:** guarded-store test expectation (1 test)

---

## C. Performance Summary

### Registers (27 kernels: OURS vs PTXAS)
| Metric | Value |
|---|---|
| OURS uses fewer regs | 16 kernels |
| OURS matches | 10 kernels |
| OURS uses more | 1 kernel (dmma_zero: +1 after PERF-5.1 fix) |
| **Net delta** | **−103 registers (OURS wins)** |

### Instructions (27 kernels: OURS vs PTXAS)
| Metric | Value |
|---|---|
| OURS total real | 993 |
| PTXAS total real | 1007 |
| **Net delta** | **−14 instructions (−1.4%, OURS wins)** |

*Updated after IMAD-FUSE-1 + HARD-FINISH-1 campaign (was −8 / −0.8% at freeze).*

### NOPs
- ~225 total: ~215 structural padding, ~10 body latency NOPs
- 0 body NOPs removable (all on critical dependency chains)
- ILP kernels: 0 body NOPs (scheduler handles ILP correctly)

### PERF + HARD-FINISH Series Findings
| Pass | Finding |
|---|---|
| PERF-1 | NOP removal needs operand-role awareness |
| PERF-2 | Body NOPs serve dual purposes (ALU + memory scoreboard) |
| PERF-3 | All body NOPs are critical-path, zero fillable |
| PERF-4 | ILP kernels have zero body NOPs — scheduler works |
| PERF-5 | OURS uses fewer registers than PTXAS |
| PERF-5.1 | Fixed dmma_zero over-declaration (10→8 regs) |
| PERF-6 | OURS uses fewer instructions than PTXAS |
| PERF-6.1 | LEA fusion investigated — instruction-count neutral (not needed) |
| PERF-7 | Cross-stack synthesis: OURS is performance-competitive |
| IMAD-FUSE-1 | LOP3.IMM encoding closes ilp_dual_int32 and ilp_alu_addr |
| HARD-FINISH-1 | @pred IMAD fusion closes ilp_pred_alu |

---

## D. Remaining Known Limits

1. **ilp_unrolled_sum4 +3 instructions** — chained stride addresses require register allocator rework to safely fold into base+offset LDG. Bounded structural gap.

2. **+1 instruction gaps** (vecadd_large, ilp_pipeline_load) — minor instruction selection / scheduling differences. Diminishing returns.

3. **~10 body latency NOPs** — serve dual purposes (ALU RAW + memory scoreboard). Require full multi-dependency rescheduler to safely eliminate.

None of these are correctness issues. All are bounded performance gaps with clear root causes and known fix paths.

---

## E. Milestone Designation

This commit marks the completion of:
- The FG-series proof engine (FG-2.4 → FG-4.8)
- The OC-series OpenCUDA cure pass (OC-1 → OC-6)
- The PERF-series performance campaign (PERF-1 → PERF-7, PERF-5.1, PERF-6.1)

The stack is ready for production use on SM_120 (Blackwell / RTX 5090).
