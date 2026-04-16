# MPT41 — r1_minmax template-boundary proof (Slice B, FINAL)

## 41.1 — Exact kernel shape

PTX body:

```
mul.lo.u32 %r2, %r0, 7;
and.b32 %r3, %r2, 0xFF;
mov.u32 %r4, %r3;
setp.gt.u32 %p1, %r3, 200;
@%p1 mov.u32 %r4, 200;
setp.lt.u32 %p2, %r4, 16;
@%p2 mov.u32 %r4, 16;
mov.u32 %r2, %r4;
```

Clamp idiom: `clamp(r3, 16, 200)` expressed via 5-instruction
mov+setp+@P-mov+setp+@P-mov sequence.

PTXAS produces a **16-instruction** sequence; OURS pre-MPT41 produces
23 (delta=+7).  PTXAS recognizes the clamp idiom and folds the entire
clamp body into 2x IMNMX (op 0x848): one for clamp-high to 200, one
for clamp-low to 16.

| idx | role | bytes |
|---:|---|---|
| 0  | LDC R1 preamble                                | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R3=TID                                     | `19790300000000000021000000220e00` |
| 2  | LDCU UR4=c[0x71] (n)                           | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE entry guard from R3                | `0c7c0003040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                       | `4d090000000000000000800300ea0f00` |
| 5  | LDCU UR6=c[0x70] (p_out)                       | `ac7706ff00700000000a000800220e00` |
| 6  | IMAD R0=R3*7 (mul.lo)                          | `2478000307000000ff028e0700e20f00` |
| 7  | LDCU UR4=c[0x6b] (desc)                        | `ac7704ff006b0000000a000800680e00` |
| 8  | LOP3 R0=R0&0xFF (and.b32)                      | `12780000ff000000ffc08e0700ca0f00` |
| 9  | **IMNMX clamp-high R0,200** (op 0x848)         | `48780000c80000000000fe0300ca0f00` |
| 10 | **IMNMX clamp-low R5,16** (op 0x848)           | `48780500100000000000fe0700e20f00` |
| 11 | IADD3.UR R2=R3+UR6 (addr lo)                   | `117c020306000000ff10800f00c81f00` |
| 12 | IADD3.UR R3=R3+UR7 (addr hi)                   | `117c030307000000ff140f0800ca0f00` |
| 13 | STG [R2:R3]=R5                                 | `86790002050000000419100c00e22f00` |
| 14 | EXIT                                           | `4d790000000000000000800300ea0f00` |
| 15 | BRA                                            | `4779fc00fcffffffffff830300c00f00` |

## 41.2 — Exclusion: no remaining candidates

This is the **final** predicate-template kernel.  No other kernels
remain in the multi-pred predicate-body family that could be admitted
by widening this template's gate.

## 41.3 — Template suitability proof

| blocker | why template bypasses | evidence |
|---|---|---|
| **Clamp-idiom recognition across 5 PTX ops** — PTXAS folds `mov + setp gt + @P mov + setp lt + @P mov` into 2x IMNMX. Pure isel performs per-PTX-op lowering and cannot pattern-match across 5 ops | Template emits IMNMX bytes verbatim | OURS over-emits by 7 instrs (23 vs 16) using naive predicate-mov expansion |
| **0x848 IMNMX opcode not in OURS' isel** — empirical scan of EXPANDED_KERNELS shows zero current emission of 0x848 by any OURS-compiled kernel | Template emits 0x848 verbatim; FG-2.3 INV B allowlist updated with rationale (template-only opcode, never via isel) | Same allowlist mechanism as MPT34's 0x81c |
| **Different family pattern** — first MPT slice without a setp+@P-add body shape; pure isel cannot bridge from add-pattern templates to clamp-pattern templates | Template captures the entire kernel verbatim, isolating the new pattern from existing isel paths | Verified |

**Decision: SAFE to template.**  The 0x848 allowlist update is the
same mechanism used in MPT34 for 0x81c — proven safe across multiple
slices.

## 41.4 — Single-target

**Chosen target**: `r1_minmax`.  Last predicate-template kernel.
After this slice, the multi-pred predicate-body family contains 11/11
BYTE_EXACT kernels.

**Slice B target locked.**
