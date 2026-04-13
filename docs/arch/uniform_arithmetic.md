# SM_120 Uniform Arithmetic (UIADD / 0x835)

**Date:** 2026-04-13
**Status:** Grounded encoding, execution requires UR pipeline init

## 1. UIADD Encoding (fully grounded)

```
Opcode: 0x835 (b0=0x35, b1=0x78)
b2 = dest register index (R[b2] and UR[b2])
b3 = source register index (R[b3] and UR[b3])
b4-b7 = 32-bit immediate addend
b10=0x8e, b11=0x07
```

PTXAS only uses b2=0, b3=0 (R0/UR0). Other indices unverified.

## 2. Dual-Write Behavior (confirmed)

UIADD writes BOTH:
- GPR[dest] = GPR[src] + imm32
- UR[dest] = UR[src] + imm32

Evidence: PTXAS-compiled `add.u32 %r, %tid.x, 5` produces correct results
when consumed by both `st.global` (GPR path) and `atom.xor` (UR path).

## 3. Execution Dependency (P3-4 finding)

UIADD requires three preceding UR-pipeline initialization instructions:

| Instruction | Opcode | Role |
|-------------|--------|------|
| 0x886 | Unknown | UR pipeline setup (NOPing breaks output) |
| S2UR UR2 | 0x919 | Writes UR2 (side-effect: UR pipeline activation?) |
| 0x2bd | Unknown | UR4-related setup (NOPing breaks output) |

**Without these instructions, UIADD's UR write does not propagate.** The GPR
write may still work (untested), but the UR value remains 0.

Evidence: cubin-patch experiment on PTXAS working cubin. NOPing any of
[2] 0x886, [4] S2UR UR2, or [5] 0x2bd changes output from 1 (correct)
to 0 (UIADD result lost). Only [3] LDCU.64 can be NOPed safely.

## 4. Implications for OpenPTXas

### What works
- `atom.global.xor.b32 [ptr], %tid.x` — uses S2UR + UMOV (no UIADD)
- Direct special-register data reaches UR via S2UR + UMOV(0x3c4) + sync(0xc02)

### What doesn't work
- `atom.global.xor.b32 [ptr], (%tid.x + K)` — requires UIADD
- UIADD needs 0x886 and 0x2bd which are uncharacterized SM_120 opcodes

### What's needed to unblock
1. Reverse engineer 0x886 (UR pipeline initialization)
2. Reverse engineer 0x2bd (UR4-related setup)
3. Emit them in the correct sequence before UIADD

## 5. Opcodes to Investigate

| Opcode | Bytes | Role |
|--------|-------|------|
| 0x886 | `86 78 04 00 00 00 00 00 00 01 8e 03` | UR pipeline init, writes R4 |
| 0x2bd | `bd 72 04 00 04 00 00 00 00 00 0e 08` | UR4 setup, writes R4 from UR4? |
