# Gap-Aware Proof Model (P3-2)

**Date:** 2026-04-12
**Status:** Implemented and validated

## Problem

The proof engine classifies S2R (0xb82) as `_OPCODES_LDC` (long-latency memory class) because the scoreboard requires it there for ctrl-word barrier generation. This causes false positives when S2R's output is consumed many instructions later — the proof engine flags missing rbar coverage even when the gap far exceeds S2R's actual latency.

## Solution

Added `MEMORY_GAP_SAFE` proof class: if a memory-class producer's effective latency is bounded AND the gap to the consumer exceeds a threshold, the edge is classified as safe without rbar evidence.

## Rule

```
IF:
  producer opcode ∈ {0xb82}  (S2R only)
  gap ≥ 4 instructions
THEN:
  classification = MEMORY_GAP_SAFE
```

## Rationale

S2R reads a warp special register (tid.x, ctaid.x, etc.). This is a register file read, not a memory operation. Effective latency is 1-2 cycles. A gap of 4 instructions (minimum ~4 cycles at 1 IPC) is more than sufficient.

## Scope

- **Applied to:** S2R (0xb82) only
- **NOT applied to:** LDCU (0x7ac), LDS (0x984), LDG (0x981), or any other memory producer
- **Threshold:** 4 instructions (conservative; actual latency is ~1-2)

## Impact

- Resolved false positive on w1_smem_reduce_pair (S2R[0]→IADD.64[8] at gap=7)
- Kernel re-enabled after being excluded since WEIRD-1
- All 142 kernels now active

## Latency Reference

| Producer | Effective Latency | Class |
|----------|-------------------|-------|
| S2R (0xb82) | ~1-2 cycles | Gap-safe at ≥4 |
| LDCU (0x7ac) | 4-7 cycles | Scoreboard (rbar) |
| LDS (0x984) | 2-4 cycles | Scoreboard (rbar) |
| LDG (0x981) | 200+ cycles | Scoreboard (rbar) |
| ALU (IADD3, etc.) | 1 cycle | Forwarding-safe |
