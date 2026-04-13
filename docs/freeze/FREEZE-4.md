# FREEZE-4: Bounded SM_120 Stack with Uniform atom.xor Enabled

**Date:** 2026-04-13
**Status:** STABLE / VERIFIED / TEMPLATE_SUCCESS
**Commit:** 0fb1a4a

---

## 1. System Summary

OpenPTXas is a PTX to SASS assembler for NVIDIA SM_120 (Blackwell / RTX 5090).
Part of the Forge to OpenCUDA to OpenPTXas compiler stack.
All validation runs on local RTX 5090 hardware.

FREEZE-4 marks the completion of the uniform atom.xor feature path.  The
previously blocked `atom.global.xor.b32` with `tid + constant` data is now
enabled via PTXAS-faithful byte templating, bringing the total active GPU-
verified kernels from 143 (FREEZE-3, with atom.xor blocked) to 144 (143
passing + 1 pre-existing unrelated failure).

---

## 2. Metrics

| Suite | Total | Pass | Fail | Notes |
|---|---|---|---|---|
| OpenPTXas pytest | 817 | 817 | 0 | |
| Workbench GPU correctness | 144 | 143 | 1 | w2_atom_and_reduce: pre-existing error 715, unrelated to atom.xor |
| Hazard regression | 15 | 15 | 0 | |
| Proof corpus | 37+ | all SAFE | 0 | |

The single workbench failure (w2_atom_and_reduce) is a pre-existing
ILLEGAL_INSTRUCTION crash in the LOP3.IMM + ATOMG.AND path, present before
any Phase 4 work.  It is not an atom.xor regression.

---

## 3. Capability Table

### Fully Supported
- Integer ALU: add, sub, mul, xor, and, or, shl, shr (32-bit and 64-bit)
- Memory: ld.global, st.global, ld.shared, st.shared with descriptor and offset addressing
- Atomics: add.u32, or.b32, min, max, cas.b32, cas.b64
- **atom.global.xor.b32: direct SR data and SR + constant uniform data**
- Warp: shfl (down/up/bfly/idx), redux (add/and/or/xor), vote.ballot
- FP32: fadd, fmul, ffma (register-based)
- FP64: dadd, dmul, dfma
- Tensor: hmma, imma, dmma, qmma (zero-init patterns)
- Control: setp (all comparisons), selp (imm+imm, reg+reg), predicated ops, if/else, loops
- Proof: 13 proof classes, 36+ forwarding-safe pairs, gap-aware memory model

### Bounded / Unsupported
- atom.global.xor.b32 with arbitrary divergent data: no GPR-to-UR path exists
- atom.global.xor.b32 with >2 params or non-standard param layouts: template covers (.u64, .u32) signature only
- atom.global.and.b32 w2 variant: pre-existing LOP3.IMM encoding issue (error 715)
- sub.u32 with immediate: literal-pool alias bug (workaround: scratch register)
- 4+ u64 parameter kernels: deferred LDCU corner cases

---

## 4. UR Architecture Findings

### Dual Register File
SM_120 has two independent register files:
- GPR (R0-R255): per-thread, divergent
- UR (UR0-UR63): per-warp, uniform

### UR Activation Sequence (for ATOMG 0x98e)
Required opcodes: S2UR(0x919), 0x886(UR_PIPE_INIT), 0x2bd(UR_PIPE_FINAL),
UMOV(0x3c4), LDCU(0x7ac for descriptor), optional UIADD(0x835 for constant).

### Critical Findings from Phase 4
1. **ATOMG 0x98e reads data from GPR**, not UR directly.  Confirmed by NOPing
   MOV.UR in a PTXAS cubin (produces garbage, not tid.x).
2. **The UR pipeline activation state depends on exact surrounding instruction
   context**, not just the activation opcodes or their ordering.  Generic
   post-scheduling injection cannot reproduce the PTXAS context.
3. **Max stall on every instruction does not fix generic activation.**  The
   issue is functional, not timing.
4. **Any non-clobbered GPR works** for MOV.UR-to-ATOMG data delivery (R0, R1,
   R4-R7 all verified in PTXAS context).
5. **Descriptor UR register is flexible** (UR4 and UR6 both verified).
6. **PTXAS-faithful byte templating is the only proven path** for correct UR
   pipeline activation on SM_120.

### Template Parameterization Surface
The template for uniform atom.xor has **only 3 parameterized bytes**: the
UIADD immediate constant K (bytes 4-6 of the UIADD instruction in Variant B).
All other bytes are invariant PTXAS ground truth.

### Kernel-Adaptive Ordering
PTXAS generates per-kernel UR activation orderings.  Two grounded variants:

**Variant A (direct SR):**
S2UR UR0 -> LDCU(param) -> ISETP.RUR(bounds) -> EXIT -> S2UR UR2 ->
UMOV -> 0x886 -> LDCU(desc) -> 0x2bd -> MOV.UR -> ISETP.RUR(flush) ->
S2R(addr) -> ATOMG

**Variant B (tid+constant):**
S2UR UR0 -> LDCU(param) -> ISETP.RUR(bounds) -> EXIT -> S2UR UR2 ->
UIADD -> 0x886 -> LDCU(desc) -> UMOV -> 0x2bd -> MOV.UR ->
ISETP.RUR(flush) -> S2R(addr) -> ATOMG

---

## 5. Final Phase 4 Result

- Direct SR atom.xor path: **PASS** (k100_atom_xor, all N values correct)
- tid+constant atom.xor path: **PASS** (w2_atom_xor_reduce, 1-thread=1, 32-thread=32)
- The bounded uniform atom.xor feature is now **enabled**
- This was achieved by **PTXAS-faithful byte templating**, not general UR scheduling synthesis
- Classification: **TEMPLATE_SUCCESS**

---

## 6. Remaining Known Limits

1. atom.global.xor.b32 with arbitrary divergent GPR data (no path exists)
2. atom.global.and.b32 w2 variant (pre-existing LOP3 encoding issue)
3. sub.u32 with immediate (literal-pool alias, scratch-register workaround)
4. Inline float literals in FP32 operations (register-based only)

---

## 7. Why This Matters

The stack is now:
- **Stable**: 817 tests, 143 GPU-verified kernels, zero regressions across all phases
- **Explainable**: every UR pipeline finding is documented with hardware evidence
- **Strong enough to automate**: the PTXAS-faithful template approach is proven and bounded,
  making it a viable foundation for automated template discovery tooling
