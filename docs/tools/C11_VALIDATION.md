# 0xc11 (IADD3.R-UR) Carry-Chain GPU Validation

**Date:** 2026-04-13
**Status:** GPU-VALIDATED

## Encoder Verification

Functional bytes (b0-b12) match PTXAS ground truth byte-for-byte:
- Low half: `117c020306000000ff10800f00` (R2 = R3 + UR6)
- High half: `117c030307000000ff140f0800` (R3 = R3 + UR7 + carry)

Source: 226 PTXAS instances across 144-kernel corpus.

## GPU Validation

Kernel: k100_mul_xor (PTXAS cubin containing 0xc11 at positions [10]-[11])
- Formula: out[tid] = ((tid * 17) ^ 0xDEAD) * 3
- 8 threads: ALL CORRECT
- Mutation test: changing UR source (b4) produces different results (field confirmed semantic)

## Carry Propagation

The low half generates carry, the high half consumes it:
- Low: R_addr_lo = R_byte_offset + UR_param_lo (carry flag set)
- High: R_addr_hi = R_or_RZ + UR_param_hi + carry
- PTXAS uses R3 (same as low source) for high src — the value is added to param_hi with carry

## Integration Status

Encoder: `sass/encoding/iadd3_ur.py` — ready
Scoreboard: not yet integrated (TE12-B)
Active codegen: not yet integrated (TE12-C)
