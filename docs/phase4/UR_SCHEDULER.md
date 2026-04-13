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

## 5. Phase 4.1 Prototype Plan

### Goal
Enable tid+constant atom.xor for the w2_atom_xor_reduce kernel.

### Steps
1. Analyze the w2 kernel's PTXAS instruction order
2. Build a specialized reordering pass for the atom.xor body
3. Move S2UR/LDCU before the activation window
4. Ensure ISETP.RUR + sync + ATOMG are contiguous after UMOV
5. Validate on GPU

### Success criteria
- w2_atom_xor_reduce PASS (1-thread: 1, 32-thread: 32)
- All 143 existing kernels still PASS
- No regressions

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
