# SM_120 Dual Register Architecture: GPR and UR

**Date:** 2026-04-12
**Status:** Evidence-backed model from P2-4 through P3-1

## 1. Overview

SM_120 (Blackwell) has two parallel register files:
- **GPR** (General Purpose Registers): R0-R255, per-thread, divergent
- **UR** (Uniform Registers): UR0-UR63, per-warp, warp-uniform

Most ALU instructions operate on GPR. A subset of instructions can read from or write to UR. Some instructions update both files simultaneously.

## 2. Instruction Classification

| Opcode | Name | GPR Write | UR Write | GPR Read | UR Read | Notes |
|--------|------|-----------|----------|----------|---------|-------|
| 0xb82 | S2R | yes | no | no | no | Special register → GPR |
| 0x919 | S2UR | no | yes | no | no | Special register → UR |
| 0x7ac | LDCU.64 | no | yes (pair) | no | no | Constant bank → UR pair |
| 0x3c4 | UMOV | no | yes | no | yes | UR → UR copy |
| 0x835 | UIADD | yes | yes | yes | yes | Uniform arithmetic, dual write |
| 0xc02 | MOV.UR | yes | sync? | no | yes | UR → GPR + sync/commit |
| 0x824 | IMAD.Ri | yes | no | yes | no | Standard ALU |
| 0x210 | IADD3 | yes | no | yes | no | Standard ALU |
| 0x810 | IADD3.IMM | yes | no | yes | no | Standard ALU |
| 0x812 | LOP3.IMM | yes | no | yes | no | Standard ALU |
| 0xc35 | IADD.64 R-UR | yes (pair) | no | yes | yes | 64-bit add with UR operand |
| 0x98e | ATOMG.XOR | yes | no | yes | yes | Atomic XOR, UR-indexed data |
| 0x9a8 | ATOMG.ADD/OR/AND | yes | no | yes | no | Atomic, GPR-indexed data |

## 3. Key Findings

### 3.1 UR is NOT a shadow of GPR
The UR and GPR files are independent. Writing R5 does NOT automatically update UR5. Writing UR5 does NOT automatically update R5. They are only synchronized by specific instructions.

### 3.2 Values reach UR via these paths only
1. **S2UR** (0x919): special register → UR (tid.x, ctaid.x, laneid, etc.)
2. **LDCU/LDCU.64** (0x7ac): constant bank → UR (parameters, descriptors)
3. **UIADD** (0x835): uniform arithmetic on existing UR values
4. **UMOV** (0x3c4): UR → UR copy

### 3.3 No known GPR → UR instruction
There is no observed instruction that moves an arbitrary GPR value into a UR register. This is a fundamental architectural boundary: per-thread divergent values (in GPR) cannot be promoted to warp-uniform (UR) space.

### 3.4 The 0xc02 role
The 0xc02 instruction appears to be `MOV Rdest, URsrc` (UR → GPR) but also serves as a synchronization/commit point. When NOPed, downstream UR consumers produce garbage. The exact mechanism is unclear — it may flush a UR write queue or commit a pending uniform operation.

### 3.5 Uniform arithmetic (0x835)
The UIADD instruction (0x835) updates BOTH the GPR and UR simultaneously. This is how computed UR values (like `tid.x + constant`) are created: S2UR puts the base into UR, then UIADD adds a constant to both R and UR.

## 4. ATOMG_XOR Feasibility

### 4.1 The 0x98e family
ATOMG_XOR uses opcode 0x98e with UR-indexed data at b4. The full data pipeline is:

```
S2UR UR0 ← tid.x         // load special register to UR
UIADD UR0 += K            // uniform add (updates both GPR and UR)
UMOV UR5 ← UR0            // copy to target UR
MOV.UR sync               // commit/synchronize
ATOMG.XOR [addr], UR5     // atomic XOR with UR data
```

### 4.2 Classification: IMPLEMENTABLE_WITH_CONSTRAINTS

ATOMG_XOR is possible when the data is:
- A special register (tid.x, ctaid.x, etc.)
- A constant from the constant bank
- A uniform computation on the above (via UIADD)

ATOMG_XOR is NOT possible when the data is:
- A per-thread divergent value loaded from global memory
- The result of a non-uniform computation (e.g., `a[tid] * 3`)

### 4.3 The 0x9a8 family
Our existing atom.add/or/and/min/max use the 0x9a8 family with GPR-indexed data. The 0x9a8 family does NOT support XOR (cubin load fails for all b11 candidates). XOR is exclusively in the 0x98e family.

## 5. Implications

1. **atom.global.xor support is limited to uniform data** — this is an SM_120 ISA constraint, not an OpenPTXas limitation
2. **Most practical atom.xor uses (e.g., reduce XOR of tid values) ARE uniform-data** — the constraint is rarely hit in practice
3. **Implementation requires**: S2UR + UIADD (0x835) encoder + UMOV (0x3c4) + sync (0xc02) + ATOMG (0x98e)
4. **Missing encoder**: UIADD (0x835) — ground truth available but encoder not yet implemented
