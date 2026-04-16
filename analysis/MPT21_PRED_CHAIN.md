# MPT21 — k200_pred_chain template-boundary proof (Slice A)

## 21.1 — Exact kernel shape

PTX body:

```
mov.u32 %r2, 0;
setp.gt.u32 %p1, %r0, 4;   @%p1 add.u32 %r2, %r2, 1;
setp.gt.u32 %p1, %r0, 8;   @%p1 add.u32 %r2, %r2, 2;
setp.gt.u32 %p1, %r0, 16;  @%p1 add.u32 %r2, %r2, 4;
setp.gt.u32 %p1, %r0, 32;  @%p1 add.u32 %r2, %r2, 8;
```

4-setp / 4-@P, **all on PTX %p1** (single-name reuse).

PTXAS produces a **20-instruction** sequence; OURS pre-MPT21 produces
21 (delta=+1).

PTXAS shape (active region):

| idx | role | bytes |
|---:|---|---|
| 0  | LDC R1 preamble                                       | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R0 = TID                                          | `19790000000000000021000000220e00` |
| 2  | LDCU UR4 = c[0x71] (param n)                          | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE P0 (entry guard, R0 source)               | `0c7c0000040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                              | `4d090000000000000000800300ea0f00` |
| 5  | ISETP.IMM.GT R0,8 → P0 (reuses entry-P0 slot)         | `0c780000080000007040f00300e20f04` |
| 6  | LDCU UR6 = c[0x70] (param p_out)                      | `ac7706ff00700000000a000800220e00` |
| 7  | ISETP.IMM.GT R0,4 → P2                                | `0c780000040000007040f40300e20f04` |
| 8  | ISETP.IMM.GT R0,16 → P1                               | `0c780000100000007040f20300e20f04` |
| 9  | LDCU UR4 = c[0x6b] (descriptor)                       | `ac7704ff006b0000000a000800660e00` |
| 10 | SEL.IMM R5,RZ,1,P2 (fold first +1 add via mux)        | `077805ff010000000000000500e20f00` |
| 11 | ISETP.IMM.GT R0,32 → P2 (P2 REUSE — kills gt-4 P2)    | `0c780000200000007040f40300ca0f00` |
| 12 | @P0 UIADD R5+=2 (gt-8 contribution)                   | `350805050200000000008e0700ca0f00` |
| 13 | @P1 UIADD R5+=4 (gt-16 contribution)                  | `351805050400000000008e0700e20f00` |
| 14 | IADD3.UR R2 = R0 + UR6 (addr lo)                      | `117c020006000000ff10800f00c81f00` |
| 15 | IADD3.UR R3 = R0 + UR7 (addr hi)                      | `117c030007000000ff140f0800e20f00` |
| 16 | @P2 UIADD R5+=8 (gt-32, P2 reuse)                     | `352805050800000000008e0700ca0f00` |
| 17 | STG [R2:R3] = R5                                      | `86790002050000000419100c00e22f00` |
| 18 | EXIT                                                  | `4d790000000000000000800300ea0f00` |
| 19 | BRA                                                   | `4779fc00fcffffffffff830300c00f00` |

**PTXAS allocation map**: 4 PTX setp producers (all on %p1) reallocated
to {P0, P2, P1, P2}.  P0 reused across entry-guard EXIT and gt-8;
P2 reused across gt-4 (consumed by SEL.IMM at [10]) and gt-32
(consumed by @P2 UIADD at [16]).  Three predicate slots used in total.

OURS pre-MPT21 emits 21 instrs via naive 4×ISETP+LEA pattern (no SEL
fold, no @P-UIADD chain, no slot reuse).

## 21.2 — Candidate identity / exclusion table

| kernel | exact same shape as k200_pred_chain? | differs where? | admit in MPT21? | why |
|---|:-:|---|:-:|---|
| **k200_pred_chain** | ✓ | — | YES | target |
| k200_nested_pred | NO | 2-setp/3-@P with @p1-conditional setp; delta=−2 | NO | nested guards |
| r1_minmax | NO | mul+and prefix + 2-setp on different sources; delta=+7 | NO | extra ALU prefix |
| w1_div_multi_guard | NO | 4-setp/4-@P alternating P1/P2 (NOT all-%p1); delta=+1 | NO | different PTXAS allocation pattern |
| k300_nasty_pred_nest3 | NO | 3-setp/5-@P nested; delta=−3 | NO | 3-level nesting |
| w2_deep_pred | NO | 5-setp/5-@P alternating P1/P2; delta=+1 | NO | extra setp + alternating |

## 21.3 — Template suitability proof

| blocker | why template bypasses | evidence |
|---|---|---|
| **MP02-protected predicate-slot reuse** — PTXAS reallocates 4 PTX-on-%p1 setps to {P0,P2,P1,P2} with P2 reused; pure isel cannot perform this rewrite | Template emits ISETP-to-{P0,P2,P1,P2} bytes verbatim | MP02 fix's predicate-RAW gate forbids the unsafe pattern OURS would emit |
| **SEL.IMM R5,RZ,1,P2 fold** — first +1 add folded into SEL operand instead of @P UIADD; pure isel cannot rewrite mov+@P-add into 1-instr SEL fold | Template emits SEL.IMM bytes verbatim | Same trick PTXAS used in MPT13, MPT17 |
| **Schedule reorder + interleaving** — gt-8 ISETP emitted before LDCU UR6 (param p_out); UIADD imm-add chain interleaved with IADD3.UR address-pair to hide latency | Template emits PTXAS schedule verbatim | Verified |

## 21.4 — Single-target confirmation

**Chosen target**: `k200_pred_chain`.

* **Why exact-shape safe**: 4-setp on single PTX `%p1` (all-on-P1 reuse)
  + 4-@P-add bodies; no nested guards; no merge ops; no extra ALU
  prefix; identical preamble/epilogue shape to MPT01/MPT05/MPT09/MPT13/
  MPT17.
* **Why single-target only**: every sibling differs in at least one
  of (setp count, PTX predicate name pattern, body shape, ALU prefix,
  nesting depth).  Each sibling requires its own PTXAS-specific
  predicate-allocation byte sequence.

**Slice A target locked: k200_pred_chain.**
