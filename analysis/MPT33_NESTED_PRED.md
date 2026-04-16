# MPT33 — k200_nested_pred template-boundary proof (Slice B)

## 33.1 — Exact kernel shape

PTX body:

```
mov.u32 %r2, %r0;
setp.gt.u32 %p1, %r0, 8;
@%p1 add.u32 %r2, %r2, 10;
@%p1 setp.gt.u32 %p2, %r0, 24;   <- @P1-conditional setp
@%p2 add.u32 %r2, %r2, 20;
```

2-setp + 1-conditional-setp + 2-@P pattern, **distinct from all prior
MPT slices** (first occurrence of @P-conditional setp).

PTXAS produces a **19-instruction** sequence; OURS pre-MPT33 produces 17
(delta=-2; UNDER-emit because OURS skips the conditional-setp expansion).

| idx | role | bytes |
|---:|---|---|
| 0  | LDC R1 preamble                                           | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R3=TID                                                | `19790300000000000021000000220e00` |
| 2  | LDCU UR4=c[0x71] (n)                                      | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE entry guard from R3                           | `0c7c0003040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                                  | `4d090000000000000000800300ea0f00` |
| 5  | ISETP.IMM.GT R3,8 -> P1 (PTX setp #1)                     | `0c780003080000007040f20300e20f04` |
| 6  | LDCU UR6=c[0x70] (p_out)                                  | `ac7706ff00700000000a000800220e00` |
| 7  | ISETP.IMM @P1-conditional gt-8 -> P2 (conditional-setp!)  | `0c780003080000007030740000e20f04` |
| 8  | UIADD R0=R3+10 (pre-compute for SEL fold)                 | `357800030a00000000008e0700e40f04` |
| 9  | ISETP.IMM.GT R3,24 -> P0 (entry-P0 reuse)                 | `0c780003180000007040f00000e20f00` |
| 10 | LDCU UR4=c[0x6b] (desc)                                   | `ac7704ff006b0000000a000800640e00` |
| 11 | SEL.REG R5,R0,R3,P1 (fold @p1 add 10 via R0=R3+10)        | `07720500030000000000800000e40f00` |
| 12 | (uncommon 0x81c ALU helper)                               | `1c788f0000000000705f700000c40f00` |
| 13 | IADD3.UR R2=R3+UR6 (addr lo)                              | `117c020306000000ff10820f00c81f00` |
| 14 | IADD3.UR R3=R3+UR7 (addr hi)                              | `117c030307000000ff148f0800ce0f00` |
| 15 | @P0 UIADD R5+=20 (gt-24 contribution; P0 reuse)           | `350805051400000000008e0700ca0f00` |
| 16 | STG [R2:R3]=R5                                            | `86790002050000000419100c00e22f00` |
| 17 | EXIT                                                      | `4d790000000000000000800300ea0f00` |
| 18 | BRA                                                       | `4779fc00fcffffffffff830300c00f00` |

**Key novel features (vs all prior MPT slices)**:
- Index [7]: ISETP-with-@P1-guard — PTXAS encodes the conditional-
  setp pattern with a special byte layout (byte 9 = 0x30, byte 10 = 0x74)
  that combines the gt-8 comparison with the @P1 guard.
- Index [12]: uncommon 0x81c ALU helper — likely ISETP-derived MOV/SEL
  helper specific to this conditional-setp pattern.
- @p1-conditional setp is the first such pattern in the MPT
  predicate-template family.

## 33.2 — Exclusion proof

| kernel | differs where? | admitted in MPT33? | why |
|---|---|:-:|---|
| **k200_nested_pred** | — | YES | target |
| k300_nasty_pred_nest3 | 3-setp/5-@P 3-level nesting (vs 2/3); delta=-3 | NO | deeper nesting |
| r1_minmax | mul+and prefix + 2-mov clamp; delta=+7 | NO | different family |

## 33.3 — Template suitability proof

| blocker | why template bypasses | evidence |
|---|---|---|
| **@P1-conditional setp encoding** — PTXAS emits a special ISETP variant at [7] that encodes the @P1 guard alongside the gt-8 -> P2 producer; pure isel cannot construct this guarded-setp byte pattern | Template emits the verbatim conditional-setp bytes | OURS under-emit by 2 instrs (proof isel skips the conditional-setp expansion) |
| **R0=R3+10 pre-compute + SEL.REG fold** | Template emits UIADD + SEL.REG bytes verbatim | Same fold trick as MPT01/MPT09/MPT17 (SEL.REG variant) |
| **Uncommon 0x81c helper op** at [12] | Template emits 0x81c bytes verbatim | OURS lacks this opcode entirely |
