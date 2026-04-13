# FREEZE-3: Phase 3 Completion — Architecture Verified

**Date:** 2026-04-13
**Status:** STABLE / VERIFIED / ARCHITECTURALLY BOUNDED

---

## 1. System Summary

OpenPTXas is a PTX → SASS assembler for NVIDIA SM_120 (Blackwell / RTX 5090).
Part of the Forge → OpenCUDA → OpenPTXas compiler stack.
All validation runs on local RTX 5090 hardware.

---

## 2. Metrics

| Suite | Total | Pass | Fail |
|---|---|---|---|
| OpenPTXas pytest | 817 | 817 | 0 |
| Adversarial harness | 51 | 51 CONFIRMED | 0 |
| Workbench GPU correctness | 143 | 143 | 0 |
| Proof corpus | 37+ | all SAFE | 0 |

---

## 3. Capability Table

### Fully Supported
- Integer ALU: add, sub, mul, xor, and, or, shl, shr (32-bit and 64-bit)
- Memory: ld.global, st.global, ld.shared, st.shared with descriptor and offset addressing
- Atomics: add.u32, or.b32, and.b32, min, max, cas.b32, cas.b64
- Warp: shfl (down/up/bfly/idx), redux (add/and/or/xor), vote.ballot
- FP32: fadd, fmul, ffma (register-based)
- FP64: dadd, dmul, dfma
- Tensor: hmma, imma, dmma, qmma (zero-init patterns)
- Control: setp (all comparisons), selp (imm+imm, reg+reg), predicated ops, if/else, loops
- Proof: 13 proof classes, 36+ forwarding-safe pairs, gap-aware memory model

### Partially Supported
- atom.global.xor.b32: works for direct special-register data (tid.x, ctaid.x)
- FP32 immediates: register-based only (inline float literals not parsed)

### Unsupported (Bounded)
- atom.global.xor.b32 with (tid + constant): requires kernel-adaptive UR scheduling
- atom.global.xor.b32 with arbitrary divergent data: no GPR→UR path exists
- sub.u32 with immediate: literal-pool alias bug (workaround: scratch register)
- 4+ u64 parameter kernels: deferred LDCU corner cases

---

## 4. UR Architecture Findings

### Dual Register File
SM_120 has two independent register files:
- GPR (R0-R255): per-thread, divergent
- UR (UR0-UR63): per-warp, uniform

### UR Activation Sequence (for ATOMG 0x98e)
Required opcodes: S2UR(0x919), 0x886, 0x2bd, UIADD(0x835), UMOV(0x3c4)

### UR Pipeline Invariants
1. S2UR(0x919) must seed UR0 before UIADD
2. 0x886 (UR_PIPE_INIT) required
3. S2UR UR2 (UR_SEED) required
4. 0x2bd (UR_PIPE_FINAL) required
5. UIADD writes both GPR and UR simultaneously
6. UMOV must immediately follow UIADD
7. **ISETP.RUR (UR READ) required between UMOV and ATOMG** — triggers pipeline flush
8. **NO UR writes (S2UR, LDCU) between UMOV and ATOMG** — disrupts pipeline state

### Kernel-Adaptive Ordering
PTXAS generates **per-kernel** UR activation orderings. Different kernels have different sequences. A fixed template does not work across kernel variants.

---

## 5. Failure Boundary

**atom.global.xor.b32 with (tid + constant) requires kernel-adaptive UR sequence generation and cannot be expressed as a fixed template.**

The UR activation sequence must be interleaved with kernel-specific instructions (bounds checks, address generation, parameter loads) in an order that respects the UR pipeline invariants. PTXAS achieves this with an adaptive scheduling pass. OpenPTXas currently lacks this capability.

---

## 6. Evidence References

| Sprint | Finding |
|---|---|
| P3-1 | UR architecture mapping: dual register file, instruction classification |
| P3-2 | Gap-aware proof engine: MEMORY_GAP_SAFE for S2R |
| P3-3 | UIADD encoding grounded, dual-write confirmed |
| P3-4 | Execution dependency confirmed: 3 setup instructions required |
| P3-5 | Full RE: 0x886=UR_PIPE_INIT, 0x2bd=UR_PIPE_FINAL, S2UR UR2=UR_SEED |
| P3-7 | Preamble ordering fixed, S2UR 0x919 vs 0x9c3 identified |
| P3-8 | ISETP.RUR required for UR flush, body UR writes disruptive |
| P3-9 | Template approach fails: PTXAS uses kernel-adaptive ordering |

---

## 7. Performance (unchanged from FREEZE-1 update)

| Metric | Value |
|---|---|
| OURS total real instructions | 993 |
| PTXAS total real instructions | 1007 |
| Net delta | -14 (-1.4%, OURS wins) |
| Wins | 7 kernels |
| Parity | 17 kernels |
| Bounded gaps | 3 kernels |

---

## 8. Final Classification

**SYSTEM STATUS: STABLE / VERIFIED / ARCHITECTURALLY BOUNDED**

The compiler stack produces correct, GPU-verified SASS for 143 kernels across all tested instruction classes. The remaining unsupported path (atom.xor tid+constant) is precisely bounded with a clear engineering path forward (Phase 4: UR scheduler).
