# SM_120 UR Pipeline Activation Sequence

**Date:** 2026-04-13
**Status:** Fully decoded via cubin-patch forensics (P3-5)

## 1. Required Sequence

For UIADD (0x835) to propagate its UR write, the following instructions
must precede it in the EXACT order shown:

```
[0] S2R R1, SR_TID_X              // standard GPR tid load
[1] S2UR UR0, SR_TID_X            // MUST be first S2UR, targets UR0
[2] 0x886 R4                      // UR pipeline init (writes R4)
[3] LDCU.64 URdesc                // descriptor load (can be NOPed)
[4] S2UR UR2, SR_LANEID           // secondary UR seed
[5] 0x2bd R4, UR4                 // UR pipeline finalize
[6] UIADD R0/UR0, R0/UR0, imm    // uniform add (dual write)
[7] UMOV UR5, UR0                 // copy to target UR
[8] ... (bounds check, setup) ...
[N] MOV.UR R5, UR5                // sync/commit
[N+1] ATOMG.XOR via UR5           // consumes UR data
```

## 2. Per-Instruction Findings

### 0x886 (UR_PIPE_INIT)
- **Encoding:** `86 78 04 00 00 00 00 00 00 01 8e 03`
- **b2 (dest):** MUST be 0x04 (R4). Other values break output.
- **b3 (src):** Flexible (0x00, 0x02, 0x04 all work)
- **b9 (mode):** 0x00, 0x01, 0x02 work. 0x03 crashes.
- **Role:** UR_PIPELINE_PREP — initializes pipeline state. Writes R4 with internal state needed by 0x2bd.
- **Required:** YES (NOPing breaks UIADD's UR write)

### S2UR UR2 (UR_SEED)
- **Encoding:** Standard S2UR targeting UR2 with SR_LANEID (or similar)
- **Role:** UR_SEED — activates UR pipeline? Side-effect of writing a second UR register.
- **Required:** YES (NOPing breaks output)
- **UR index:** UR2 appears fixed; changing to other indices untested.

### 0x2bd (UR_PIPE_FINAL)
- **Encoding:** `bd 72 04 00 04 00 00 00 00 00 0e 08`
- **b2 (dest):** R4 (writes R4)
- **b4 (UR):** UR4 (reads descriptor UR). Value doesn't matter (all UR indices give same result).
- **Role:** UR_PIPELINE_COMMIT — finalizes the pipeline state.
- **Required:** YES (NOPing breaks output)

## 3. Order Dependence (P3-5 Critical Finding)

**THE ORDER IS CRITICAL.**

Slot-by-slot hybrid testing proved that replacing ANY slot with our
(differently-ordered) instruction breaks the output:

```
slot[1] S2UR UR0→UR2: BROKE
slot[2] 0x886→LDCU.64: BROKE
slot[4] S2UR UR2→S2UR UR0: BROKE
slot[5] 0x2bd→0x886: BROKE
slot[6] UIADD→S2UR: BROKE
```

The S2UR UR0 MUST be at slot 1 (before LDCU.64). Our pipeline puts
LDCU.64 first, which is the root cause of the failure.

## 4. What Works Without the Sequence

- `atom.global.xor.b32 [ptr], %tid.x` — uses S2UR + UMOV (no UIADD), works
- The UMOV (0x3c4) path doesn't need the activation sequence
- Only UIADD (dual GPR+UR write) requires the full activation

## 5. Implementation Path

To enable UIADD-based ATOMG_XOR:
1. Restructure pipeline preamble to emit S2UR UR0 BEFORE any LDCU.64
2. Insert 0x886, S2UR UR2, 0x2bd in the correct positions
3. Emit UIADD in the preamble (not inline)
4. The body emits UMOV + sync + ATOMG as before

This requires changing `pipeline.py` preamble construction order.
Current order: `[LDC R1] [LDCUs] [UR4 desc]`
Required order: `[LDC R1] [S2UR UR0] [0x886] [LDCUs] [S2UR UR2] [0x2bd] [UIADD] [UR4 desc]`

## 6. Classification Summary

| Instruction | Role | Required | Order-Sensitive |
|-------------|------|----------|-----------------|
| S2UR UR0 | Data seed | YES | MUST be slot 1 |
| 0x886 | UR_PIPELINE_PREP | YES | After S2UR UR0 |
| S2UR UR2 | UR_SEED | YES | After 0x886 |
| 0x2bd | UR_PIPELINE_COMMIT | YES | After S2UR UR2 |
| UIADD | Dual write | YES | After 0x2bd |
| UMOV | UR copy | YES | After UIADD |
| MOV.UR | Sync | YES | Before ATOMG |
