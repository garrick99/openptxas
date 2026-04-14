# TE13: Consumer-Aware 0xc11 Carry-Chain Emission

## Current Path (broken for convergence)

```
shl.b64 handler → IMAD.WIDE R_lo:R_hi = R_src * stride + RZ  (1 instr)
add.u64 handler → IADD.64-UR R_out = R_lo:R_hi + UR_param    (1 instr)
Total: 2 instructions (IMAD.WIDE + IADD.64-UR)
```

## PTXAS Path (target)

```
(shl.b64 produces 32-bit multiply only — no widening)
add site → 0xc11 low:  R_lo = R_offset + UR_param_lo  (carry out)
           0xc11 high: R_hi = R_offset + UR_param_hi  (carry in)
Total: 2 instructions (0xc11 + 0xc11)
```

Note: PTXAS's shl.b64 equivalent produces a 32-bit result (IMAD.SHL),
not a 64-bit widened pair.  The widening happens implicitly in the 0xc11
carry chain.

## Coordinated Emission Rule

### Trigger (ALL required)
1. SM_120
2. add.u64 has one UR-backed operand (u64 param loaded via LDCU.64)
3. Other operand was widened from 32 bits (tracked by `_widened_from_32`)
4. Not in atom.xor template path
5. Fallback available

### Suppression
When trigger fires at the add.u64 site:
- The IMAD.WIDE that was already emitted by shl.b64 must be REMOVED
- Mechanism: shl.b64 records itself in `_imad_wide_deferred` dict
  mapping dest_name → (SassInstr, src32_gpr, shift_amount)
- add.u64 checks `_imad_wide_deferred` for its source operand
- If found: POP the IMAD.WIDE from the output list, emit 32-bit
  multiply + 0xc11 pair instead

### Replacement Sequence
```
IMAD.SHL R_lo, R_src, stride      (32-bit multiply, replaces IMAD.WIDE)
0xc11 lo: R_out = R_lo + UR_lo    (carry out)
0xc11 hi: R_out+1 = R_lo + UR_hi  (carry in from lo)
```

Total: 3 instructions.  But IMAD.WIDE was 1 + IADD.64-UR was 1 = 2.
So this is +1 instruction.

### Problem: +1 instruction
The 0xc11 path produces 3 instructions vs the current 2.  But PTXAS
produces 2 (it doesn't emit a separate multiply — the multiply is fused
into the preceding ALU or eliminated).

### Resolution: skip the multiply when stride = element_size
For `out[tid] = f(tid)` kernels, the byte offset is `tid * 4`.
PTXAS computes this as part of the preceding ALU chain, not as a
separate IMAD.WIDE.  If the shl.b64 source is directly from S2UR
(tid.x) and the shift is by 2 (multiply by 4), we can fold the
multiply into the 0xc11 low half by adjusting the input.

Actually, looking at PTXAS more carefully: the byte offset is computed
by an earlier IMAD (body ALU) and the 0xc11 just adds it to the base.
So PTXAS also has a separate multiply — it just uses a different opcode.

The real answer: PTXAS uses 1 IMAD + 2 × 0xc11 = 3 instructions for the
same work as our 1 IMAD.WIDE + 1 IADD.64-UR = 2 instructions.  PTXAS has
MORE instructions for the address path, not fewer.  The instruction count
difference comes from elsewhere (ISETP encoding, param load ordering).
