# MPT37 — k300_nasty_pred_nest3 template-boundary proof (Slice A)

## 37.1 — Exact kernel shape

PTX body:

```
mov.u32 %r2, 0;
setp.gt.u32 %p1, %r0, 5;
@%p1 add.u32 %r2, %r2, 1;
@%p1 setp.gt.u32 %p2, %r0, 15;   <- @P1-conditional setp #1
@%p2 add.u32 %r2, %r2, 2;
@%p2 setp.gt.u32 %p1, %r0, 30;   <- @P2-conditional setp #2 (P1 reuse!)
@%p1 add.u32 %r2, %r2, 4;
```

3 plain setp + 2 conditional setp + 5 @P consumers (some shared %p1).

PTXAS produces a **22-instruction** sequence; OURS pre-MPT37 produces 19
(delta=-3; UNDER-emit because OURS skips the chained @P-conditional-
setp expansion).

| idx | role | bytes |
|---:|---|---|
| 0  | LDC R1 preamble                                   | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R0=TID                                        | `19790000000000000021000000220e00` |
| 2  | LDCU UR4=c[0x71] (n)                              | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE entry guard from R0                   | `0c7c0000040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                          | `4d090000000000000000800300ea0f00` |
| 5  | ISETP.IMM.GT R0,5 -> P2 (PTX setp #1)             | `0c780000050000007040f40300e20f04` |
| 6  | ISETP.IMM @P-cond gt-5 (conditional-setp variant) | `0c780000050000007030720000e20f04` |
| 7  | LDCU UR6=c[0x70] (p_out)                          | `ac7706ff00700000000a000800260e00` |
| 8  | ISETP.IMM @P-cond gt-15 -> P0 (entry-P0 reuse)    | `0c7800000f0000007040700100e20f00` |
| 9  | LDCU UR4=c[0x6b] (desc)                           | `ac7704ff006b0000000a000800620e00` |
| 10 | SEL.IMM R5,RZ,1,P? (fold first +1 add)            | `077805ff010000000000000500c60f00` |
| 11 | 0x81c uncommon ALU helper #1                      | `1c788f0000000000703f700000ca0f00` |
| 12 | ISETP.IMM @P-cond gt-30 (conditional-setp #2)     | `0c7800001e0000007040720000e20f04` |
| 13 | ISETP.IMM gt-5 (re-emit, P1-reuse alignment)      | `0c780000050000007040760400e20f04` |
| 14 | IADD3.UR R2=R0+UR6 (addr lo)                      | `117c020006000000ff10840f00c81f00` |
| 15 | 0x81c uncommon ALU helper #2                      | `1c788f0000000000707ff20000e40f00` |
| 16 | @P0 UIADD R5+=2 (gt-15 contribution)              | `350805050200000000008e0700e20f00` |
| 17 | IADD3.UR R3=R0+UR7 (addr hi)                      | `117c030007000000ff140f0900d40f00` |
| 18 | @P1 UIADD R5+=4 (gt-30 contribution)              | `351805050400000000008e0700ca0f00` |
| 19 | STG [R2:R3]=R5                                    | `86790002050000000419100c00e22f00` |
| 20 | EXIT                                              | `4d790000000000000000800300ea0f00` |
| 21 | BRA                                               | `4779fc00fcffffffffff830300c00f00` |

**Key novel features (vs MPT34, the prior conditional-setp slice)**:
- Two chained @P-conditional setps at [6,8] and [12] — vs MPT34's
  single conditional setp.
- Two 0x81c uncommon-ALU helpers at [11] and [15] — vs MPT34's one.
- Re-emitted ISETP gt-5 at [13] for P1-slot alignment after the
  P1-reuse rewrite from `@p2 setp.gt %p1, r0, 30`.
- The 0x81c FG-2.3 allowlist established in MPT34 directly applies.

## 37.2 — Exclusion table

| kernel | differs where? | admit in MPT37? |
|---|---|:-:|
| **k300_nasty_pred_nest3** | — | YES |
| r1_minmax | mul+and prefix + 2-mov clamp; entirely different family pattern; delta=+7 | NO |

## 37.3 — Template suitability

| blocker | why template bypasses | evidence |
|---|---|---|
| **Two-level @P-conditional-setp chain** (`@p1 setp p2`, `@p2 setp p1` with P1 reuse) | Template emits the verbatim ISETP-with-@P-guard bytes for both conditional setps plus the P1-reuse re-emit | OURS under-emit by 3 instrs - proof isel skips the conditional-setp expansion entirely |
| **Two 0x81c uncommon ALU helpers** at [11] and [15] | Template emits 0x81c bytes verbatim; FG-2.3 INV B coverage allowlist already covers this opcode (MPT34 added it) | OURS lacks 0x81c entirely |
| **SEL.IMM R5,RZ,1,P? fold of first +1 add** | Template emits SEL.IMM verbatim | Same trick as MPT13/MPT22/MPT26/MPT30 |

## 37.4 — Single-target

**Chosen target**: `k300_nasty_pred_nest3`.  Only remaining
predicate-template kernel with chained @P-conditional setp pattern.
Single-target only: the only other remaining candidate (r1_minmax)
has a fundamentally different mul+and+clamp shape with no setp+@P-add
body.

**Slice A target locked.**
