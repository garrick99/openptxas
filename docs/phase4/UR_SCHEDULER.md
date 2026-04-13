# Phase 4: UR Scheduling Synthesis

**Status:** Planning (not yet implemented)

## 1. Problem Definition

Generate a correct UR activation + usage sequence for `atom.global.xor.b32` under arbitrary kernel shapes, where the data operand is a uniform computation (SR + constant).

The sequence must be interleaved with kernel-specific operations (bounds checks, address generation, parameter loads) while respecting all UR pipeline invariants.

## 2. Required Capabilities

The UR scheduler must:

1. **Detect** atom.global.xor with uniform data path
2. **Identify** the special register source and add constant
3. **Generate** the activation sequence:
   - S2UR(0x919) UR0 ← SR
   - 0x886 R4 (UR_PIPE_INIT)
   - S2UR(0x919) UR2 ← LANEID (UR_SEED)
   - 0x2bd R4, UR_desc (UR_PIPE_FINAL)
   - UIADD R0/UR0 += K (if constant needed)
   - UMOV UR5 ← UR0
4. **Place** an ISETP.RUR (UR READ) between UMOV and ATOMG
5. **Guarantee** no UR writes (S2UR, LDCU) between UMOV and ATOMG
6. **Interleave** with the kernel's bounds check, address computation, and parameter loads

## 3. Observed PTXAS Behavior

PTXAS generates per-kernel orderings. Two observed variants:

**Simple kernel (1 param, no bounds check):**
```
S2UR → 0x886 → LDCU → S2UR → 0x2bd → UIADD → UMOV
→ ISETP.RUR → S2R → MOV.UR → ATOMG
```

**Complex kernel (2 params, bounds check, add):**
```
S2UR → LDCU → ISETP.RUR(bounds) → EXIT
→ S2UR → UIADD → 0x886 → LDCU → UMOV → 0x2bd
→ MOV.UR → ISETP.RUR(flush) → S2R → ATOMG
```

Key observations:
- Activation order is NOT fixed
- 0x886 can come before OR after UIADD
- Bounds check can be interleaved
- Multiple ISETP.RUR can appear (bounds + flush)

## 4. Proposed Architecture

### Option A: Micro-scheduler for UR blocks
- Post-scheduling pass that reorders instructions within a marked UR-sensitive region
- Moves UR writes before the activation window
- Moves UR reads into the window
- Complexity: Medium. Risk: interaction with main scheduler.

### Option B: Pattern-matching templates with variants
- Define 3-5 template variants covering common kernel shapes
- Select template based on kernel structure analysis
- Complexity: Low. Risk: doesn't generalize to new shapes.

### Option C: Constraint-based scheduling
- Define UR constraints as hard scheduling rules
- Integrate into the main scheduler as constraints
- Complexity: High. Risk: performance regression on non-UR kernels.

**Recommended: Option A** — most practical balance of generality and complexity.

## 5. Phase 4.3 — PTXAS-Faithful Template (IMPLEMENTED)

### Problem
Phase 4.2 proved that generic post-scheduling UR activation injection cannot
produce the correct UR pipeline state.  The exact surrounding instruction
context matters — not just the activation opcodes or their ordering.

### Solution
Emit exact PTXAS instruction bytes for the atom.xor block, parameterized
only by the UIADD immediate constant K.  Two variants:

**Variant A (direct SR, no UIADD):**
```
S2UR UR0←TID → LDCU UR4(param n) → ISETP.RUR(bounds) → EXIT
→ S2UR UR2←LANEID → UMOV UR5←UR0 → 0x886 → LDCU UR6(desc)
→ 0x2bd → MOV.UR R5←UR5 → ISETP.RUR(flush) → S2R R2(addr) → ATOMG
```

**Variant B (tid+constant, has UIADD):**
```
S2UR UR0←TID → LDCU UR4(param n) → ISETP.RUR(bounds) → EXIT
→ S2UR UR2←LANEID → UIADD UR0+=K → 0x886 → LDCU UR6(desc)
→ UMOV UR5←UR0 → 0x2bd → MOV.UR R5←UR5 → ISETP.RUR(flush)
→ S2R R2(addr) → ATOMG
```

### Key findings (Phase 4.2 forensics)
1. ATOMG 0x98e reads data from **GPR** (not UR directly) — confirmed by
   NOPing MOV.UR in PTXAS cubin (produces garbage, not tid.x)
2. Any non-clobbered GPR works for MOV.UR→ATOMG data delivery
3. Descriptor UR register is flexible (UR4 and UR6 both work)
4. Max stall everywhere does NOT fix generic activation — issue is functional
5. The UR pipeline requires specific instruction context that generic
   scheduling cannot reproduce

### Parameterization surface
| Field | Location | Source | Why parameterized |
|---|---|---|---|
| UIADD immediate K | Variant B [6] b4-b6 | ctx._ur_activation_add | Per-kernel constant |

All other bytes are invariant PTXAS ground truth.

### Success criteria (MET)
- w2_atom_xor_reduce PASS (1-thread: 1, 32-thread: 32) ✓
- k100_atom_xor PASS (N=3,5,7,32,64 all correct) ✓
- All 143 existing kernels still PASS ✓
- Zero regressions ✓

## 6. Risks

| Risk | Mitigation |
|---|---|
| Main scheduler interaction | Post-scheduling pass only, no scheduler changes |
| Hidden UR hazards | Always validate on real GPU |
| Kernel-specific divergence | Start with known working PTXAS ordering, generalize later |

## 7. Success Criteria (Phase 4 overall)

- atom.global.xor.b32 with tid+constant: PASS
- No regressions on any existing path
- UR scheduling integrated cleanly (no global pipeline changes)
- Solution documented and bounded
