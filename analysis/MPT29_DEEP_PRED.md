# MPT29 — w2_deep_pred template-boundary proof (Slice A)

## 29.1 — Exact kernel shape

PTX body:

```
mov.u32 %r2, 0;
setp.gt.u32 %p1, %r0, 2;   @%p1 add.u32 %r2, %r2, 1;
setp.gt.u32 %p2, %r0, 6;   @%p2 add.u32 %r2, %r2, 2;
setp.gt.u32 %p1, %r0, 12;  @%p1 add.u32 %r2, %r2, 4;
setp.gt.u32 %p2, %r0, 24;  @%p2 add.u32 %r2, %r2, 8;
setp.gt.u32 %p1, %r0, 48;  @%p1 add.u32 %r2, %r2, 16;
```

5-setp / 5-@P alternating PTX P1/P2.

PTXAS produces a **22-instruction** sequence; OURS pre-MPT29 produces 23
(delta=+1).

PTXAS structure resembles MPT17 (k300_nasty_multi_pred, also 5-setp)
but uses SEL.IMM (R5,RZ,1) instead of MPT17's R0=R3+10 pre-compute +
SEL.REG variant.  Predicate-allocation map: 5 PTX setps -> {P2, P0,
P1, P0-reuse, P2-reuse} = 3 distinct slots, with P0 reused across
entry-guard EXIT, gt-2, and gt-24; P2 reused across gt-6 (consumed
by SEL.IMM) and gt-48.

| idx | role | bytes |
|---:|---|---|
| 0  | LDC R1 preamble                                     | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R0=TID                                          | `19790000000000000021000000220e00` |
| 2  | LDCU UR4=c[0x71] (n)                                | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE entry guard P0                          | `0c7c0000040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                            | `4d090000000000000000800300ea0f00` |
| 5  | ISETP.IMM.GT R0,6 -> P2                             | `0c780000060000007040f40300e20f04` |
| 6  | ISETP.IMM.GT R0,2 -> P0 (entry-P0 reuse)            | `0c780000020000007040f00300e20f04` |
| 7  | ISETP.IMM.GT R0,12 -> P1                            | `0c7800000c0000007040f20300e20f04` |
| 8  | LDCU UR6=c[0x70] (p_out)                            | `ac7706ff00700000000a000800260e00` |
| 9  | SEL.IMM R5,RZ,1,P0 (fold gt-2 +1)                   | `077805ff010000000000000400e20f00` |
| 10 | ISETP.IMM.GT R0,24 -> P0 (P0 REUSE - kills gt-2)    | `0c780000180000007040f00300e20f04` |
| 11 | LDCU UR4=c[0x6b] (desc)                             | `ac7704ff006b0000000a0008006a0e00` |
| 12 | @P2 UIADD R5+=2 (gt-6 contribution)                 | `352805050200000000008e0700e20f00` |
| 13 | ISETP.IMM.GT R0,48 -> P2 (P2 REUSE - kills gt-6)    | `0c780000300000007040f40300c80f00` |
| 14 | @P1 UIADD R5+=4 (gt-12 contribution)                | `351805050400000000008e0700e20f00` |
| 15 | IADD3.UR R2=R0+UR6 (addr lo)                        | `117c020006000000ff10820f00c81f00` |
| 16 | @P0 UIADD R5+=8 (gt-24, P0 reuse)                   | `350805050800000000008e0700e20f00` |
| 17 | IADD3.UR R3=R0+UR7 (addr hi)                        | `117c030007000000ff148f0800c80f00` |
| 18 | @P2 UIADD R5+=16 (gt-48, P2 reuse)                  | `352805051000000000008e0700ca0f00` |
| 19 | STG [R2:R3]=R5                                      | `86790002050000000419100c00e22f00` |
| 20 | EXIT                                                | `4d790000000000000000800300ea0f00` |
| 21 | BRA                                                 | `4779fc00fcffffffffff830300c00f00` |

OURS pre-MPT29 emits 23 instrs via naive 5x ISETP+LEA pattern (no SEL
fold, no @P-UIADD chain, no slot reuse).

## 29.2 — Candidate identity / exclusion table

| kernel | exact same shape as w2_deep_pred? | differs where? | admit in MPT29? | why |
|---|:-:|---|:-:|---|
| **w2_deep_pred** | yes | — | YES | target |
| k200_nested_pred | NO | 2-setp/3-@P with @p1-conditional setp; delta=-2 (under-emit) | NO | nested guards |
| k300_nasty_pred_nest3 | NO | 3-setp/5-@P 3-level nesting; delta=-3 | NO | 3-level nesting |
| r1_minmax | NO | mul+and prefix + 2-setp/2-mov clamp; delta=+7 | NO | extra ALU prefix |

## 29.3 — Template suitability proof

| blocker | why template bypasses | evidence |
|---|---|---|
| **MP02-protected predicate-slot reuse** — PTXAS reallocates 5 PTX setps (alternating P1/P2) to {P2,P0,P1,P0-reuse,P2-reuse}; pure isel cannot perform this without violating MP02 | Template emits the verbatim ISETP-to-{P2,P0,P1,P0,P2} bytes | MP02 fix's predicate-RAW gate forbids the unsafe pattern OURS would emit |
| **SEL.IMM R5,RZ,1,P0 fold of gt-2 +1** — first add folded into SEL operand instead of @P UIADD; pure isel cannot rewrite mov+@P-add into 1-instr SEL fold | Template emits SEL.IMM bytes verbatim | Same trick as MPT13/MPT22/MPT26 |
| **Schedule reorder + interleaving** — gt-6 emitted before gt-2; 4x@P-UIADD chain interleaved with IADD3.UR address pair | Template emits PTXAS schedule verbatim | Verified |

## 29.4 — Single-target confirmation

**Chosen target**: `w2_deep_pred` — 5-setp/5-@P alternating PTX P1/P2,
thresholds 2/6/12/24/48, add values 1/2/4/8/16.  Single-target only:
each sibling differs in setp count, nesting depth, or instruction
prefix.

**Slice A target locked: w2_deep_pred.**
