# MPT09 — k300_pred3 whole-kernel template proof (Slice B)

## 09.1 — Exact kernel shape

PTX body:

```
mov.u32 %r0, %tid.x;
ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
ld.param.u64 %rd0, [p_out];
mov.u32 %r2, %r0;
setp.gt.u32 %p1, %r0, 10;        ← P1 producer #1
@%p1 add.u32 %r2, %r2, 1;
setp.gt.u32 %p1, %r0, 20;        ← P1 reuse (#2)
@%p1 add.u32 %r2, %r2, 2;
setp.gt.u32 %p1, %r0, 40;        ← P1 reuse (#3)
@%p1 add.u32 %r2, %r2, 4;
cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
add.u64 %rd2, %rd0, %rd1;
st.global.u32 [%rd2], %r2;
ret;
```

PTXAS produces a **19-instruction** sequence; OURS produces 19 too
(delta=0).

| idx | role | bytes (hex) |
|----:|------|-------------|
| 0  | LDC R1 preamble                                       | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R3 = TID.X                                         | `19790300000000000021000000220e00` |
| 2  | LDCU UR4 = c[0x71] (param n)                           | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE P0, R3, UR4 (entry guard)                  | `0c7c0003040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                                | `4d090000000000000000800300ea0f00` |
| 5  | LDCU UR6 = c[0x70] (param p_out)                       | `ac7706ff00700000000a000800220e00` |
| 6  | ISETP.IMM.GT R3,20 → P1                               | `0c780003140000007040f20300e20f04` |
| 7  | ISETP.IMM.GT R3,10 → P2                               | `0c7800030a0000007040f40300e20f04` |
| 8  | UIADD R0=R3+1 (unconditional pre-compute)             | `357800030100000000008e0700e20f04` |
| 9  | LDCU UR4 = c[0x6b] (descriptor)                        | `ac7704ff006b0000000a000800620e00` |
| 10 | ISETP.IMM.GT R3,40 → P0 (reuse, kills entry P0)       | `0c780003280000007040f00300c60f00` |
| 11 | SEL R5, R0, R3, P2 (predicate-mux: pick R0 if P2)     | `07720500030000000000000100cc0f00` |
| 12 | @P1 UIADD R5 += 2                                      | `351805050200000000008e0700e20f00` |
| 13 | IADD3.UR R2 = R3 + UR6 (addr lo)                       | `117c020306000000ff10840f00c81f00` |
| 14 | @P0 UIADD R5 += 4                                      | `350805050400000000008e0700e20f00` |
| 15 | IADD3.UR R3 = R3 + UR7 (addr hi)                       | `117c030307000000ff140f0900ca0f00` |
| 16 | STG [R2:R3] = R5                                       | `86790002050000000419100c00e22f00` |
| 17 | EXIT                                                    | `4d790000000000000000800300ea0f00` |
| 18 | BRA                                                     | `4779fc00fcffffffffff830300c00f00` |

PTXAS uses a **3-predicate, SEL-based mux pattern** that uses P0/P1/P2
distinctly (no naive predicate reuse like the PTX). This avoids the
P1-reuse stale-value hazard MP02 uncovered.

## 09.2 — Exclusion proof

Remaining 8 multi-pred siblings:

| kernel | differs where? | admit in MPT09? |
|---|---|:-:|
| k100_setp_combo | 2-setp/2-@P, different imms (16/8), delta=+1 | NO |
| k200_nested_pred | 2-setp/3-@P, has merge-like @P/@!P pair, delta=−2 | NO |
| r1_minmax | mul + and + 2 setp, delta=+4 (4 extra ops) | NO |
| k300_nasty_pred_nest3 | 3-setp/5-@P nested, delta=−3 | NO |
| k200_pred_chain | 4-setp/4-@P all on P1 reuse, delta=+1 | NO |
| w1_div_multi_guard | 4-setp/4-@P alternating P1/P2, delta=+1 | NO |
| k300_nasty_multi_pred | 5-setp/5-@P alternating, delta=0 | NO |
| w2_deep_pred | 5-setp/5-@P alternating P1/P2, delta=+1 | NO |

Each sibling differs in setp count (3 vs 2/4/5) AND in body shape AND
in PTXAS predicate-allocation strategy. None match k300_pred3's exact
3-setp + 3-@P + delta=0 pattern.

## 09.3 — Template suitability proof

| blocker | why template bypasses | evidence |
|---|---|---|
| **PTXAS uses 3 distinct predicates (P0/P1/P2)** for what PTX expresses with P1 reuse — pure isel cannot perform this predicate-allocation rewrite without violating MP02 multi-pred safety | Template emits the verbatim ISETP-to-{P0,P1,P2} bytes | MP02 fix's predicate-RAW gate in scoreboard catches the unsafe pattern OURS emits when reusing P1 |
| **SEL R5, R0, R3, P2 mux** — pre-computes both branches of the first predicate as registers, then picks via SEL. Pure isel cannot reorder this unconditionally before P2 is computed | Template emits SEL bytes verbatim | Same trick PTXAS used in MPT01 |
| Multi-op sequencing (delta=0 means same count, but ctrl bytes differ) | Template emits PTXAS schedule verbatim | Verified |
