# MPT25 — w1_div_multi_guard template-boundary proof (Slice B)

## 25.1 — Exact kernel shape

PTX body:

```
mov.u32 %r2, 0;
setp.gt.u32 %p1, %r0, 8;   @%p1 add.u32 %r2, %r2, 1;
setp.gt.u32 %p2, %r0, 16;  @%p2 add.u32 %r2, %r2, 2;
setp.gt.u32 %p1, %r0, 32;  @%p1 add.u32 %r2, %r2, 4;
setp.gt.u32 %p2, %r0, 48;  @%p2 add.u32 %r2, %r2, 8;
```

4-setp / 4-@P alternating P1/P2.

PTXAS produces a **20-instruction** sequence; OURS pre-MPT25 produces 21
(delta=+1).

PTXAS structure is IDENTICAL to MPT22 (k200_pred_chain) at the opcode
level: same 20 opcodes in the same positions.  Only differences:

| field | MPT22 (k200_pred_chain) | MPT25 (w1_div_multi_guard) |
|---|---|---|
| ISETP imm @ [5] | gt-8  | gt-16 |
| ISETP imm @ [7] | gt-4  | gt-8  |
| ISETP imm @ [8] | gt-16 | gt-32 |
| ISETP imm @ [11] | gt-32 | gt-48 |
| SEL.IMM folds  | gt-4 → +1 | gt-8 → +1 |

Predicate-allocation map is identical: 4 PTX setps (named P1/P2
alternating) reallocated by PTXAS to {P0, P2, P1, P2} with P0 reused
across entry/gt-16, P2 reused across gt-8 (consumed by SEL.IMM)
and gt-48 (consumed by @P2 UIADD).

| idx | role | bytes |
|---:|---|---|
| 0  | LDC R1 preamble                                | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R0=TID                                     | `19790000000000000021000000220e00` |
| 2  | LDCU UR4=c[0x71] (n)                           | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE entry guard P0                     | `0c7c0000040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                       | `4d090000000000000000800300ea0f00` |
| 5  | ISETP.IMM.GT R0,16→P0 (reuse entry-P0 slot)    | `0c780000100000007040f00300e20f04` |
| 6  | LDCU UR6=c[0x70] (p_out)                       | `ac7706ff00700000000a000800220e00` |
| 7  | ISETP.IMM.GT R0,8→P2                           | `0c780000080000007040f40300e20f04` |
| 8  | ISETP.IMM.GT R0,32→P1                          | `0c780000200000007040f20300e20f04` |
| 9  | LDCU UR4=c[0x6b] (desc)                        | `ac7704ff006b0000000a000800660e00` |
| 10 | SEL.IMM R5,RZ,1,P2 (fold gt-8 +1)              | `077805ff010000000000000500e20f00` |
| 11 | ISETP.IMM.GT R0,48→P2 (P2 REUSE — kills gt-8)  | `0c780000300000007040f40300ca0f00` |
| 12 | @P0 UIADD R5+=2 (gt-16 contribution)           | `350805050200000000008e0700ca0f00` |
| 13 | @P1 UIADD R5+=4 (gt-32 contribution)           | `351805050400000000008e0700e20f00` |
| 14 | IADD3.UR R2=R0+UR6 (addr lo)                   | `117c020006000000ff10800f00c81f00` |
| 15 | IADD3.UR R3=R0+UR7 (addr hi)                   | `117c030007000000ff140f0800e20f00` |
| 16 | @P2 UIADD R5+=8 (gt-48, P2 reuse)              | `352805050800000000008e0700ca0f00` |
| 17 | STG [R2:R3]=R5                                 | `86790002050000000419100c00e22f00` |
| 18 | EXIT                                           | `4d790000000000000000800300ea0f00` |
| 19 | BRA                                            | `4779fc00fcffffffffff830300c00f00` |

## 25.2 — Exclusion proof

| kernel | differs where? | admitted in MPT25? |
|---|---|:-:|
| **w1_div_multi_guard** | — | YES |
| w2_deep_pred | 5-setp (not 4); extra setp at gt-2 | NO |
| k200_nested_pred | 2-setp + @p1-conditional setp; under-emit (-2) | NO |
| k300_nasty_pred_nest3 | 3-setp/5-@P 3-level nesting | NO |
| r1_minmax | mul+and prefix + 2-mov clamp pattern | NO |

## 25.3 — Template suitability proof

| blocker | why template bypasses | evidence |
|---|---|---|
| **MP02-protected predicate-name independence** — PTXAS allocates 4 PTX setps (named P1/P2 alternating) to {P0,P2,P1,P2} with P0/P2 slot reuse; pure isel cannot perform this rewrite | Template emits ISETP-to-{P0,P2,P1,P2} bytes verbatim | Same MP02 gate that protected MPT05/MPT09/MPT17/MPT22 |
| **SEL.IMM R5,RZ,1,P2 fold** — first +1 add folded into SEL operand instead of an @P UIADD; pure isel cannot rewrite mov+@P-add into 1-instr SEL fold | Template emits SEL.IMM bytes verbatim | Same trick as MPT13/MPT22 |
| **Add-imm reordering** — PTX add order is +1,+2,+4,+8 by threshold ascending; PTXAS reorders to fold gt-8 (smallest threshold of the +1 contribution) into SEL via P2; remaining +2,+4,+8 mapped to @P0,@P1,@P2 | Template emits PTXAS schedule verbatim | Verified |
