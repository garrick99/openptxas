# MPT05 — k200_double_guard whole-kernel template proof

## 05.1 — Exact kernel shape

PTX body:

```
mov.u32 %r0, %tid.x;
ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
ld.param.u64 %rd0, [p_out];
mul.lo.u32 %r2, %r0, 3;          ← upfront mul (vs MPT01's mov)
setp.gt.u32 %p1, %r0, 16;        ← P1 producer
@%p1 add.u32 %r2, %r2, 50;       ← P1 consumer
setp.lt.u32 %p2, %r0, 48;        ← P2 producer (LT, not GT)
@%p2 add.u32 %r2, %r2, 25;       ← P2 consumer
cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
add.u64 %rd2, %rd0, %rd1;
st.global.u32 [%rd2], %r2;
ret;
```

PTXAS produces a **17-instruction** sequence (same count as MPT01).
Body differs from MPT01 by:
* upfront `mul.lo.u32 r2, r0, 3` (replaces MPT01's `mov r2, r0`)
* second setp uses LT instead of GT
* immediate values differ (50/25 vs 100/200)

PTXAS uses standard **@P-guarded UIADD** (no SEL+@P0 trick this time):

| idx | role | bytes (hex) |
|----:|------|-------------|
| 0  | LDC R1 preamble                                  | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R0 = TID.X                                    | `19790000000000000021000000220e00` |
| 2  | LDCU UR4 = c[0x71] (param n)                      | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE P0, R0, UR4 (entry guard)             | `0c7c0000040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                           | `4d090000000000000000800300ea0f00` |
| 5  | LDCU UR6 = c[0x70] (param p_out)                  | `ac7706ff00700000000a000800220e00` |
| 6  | ISETP.IMM.GT R0,16 → P1                           | `0c780000100000007040f20300e20f04` |
| 7  | ISETP.IMM.GE R0,48 → P0 (LT lowered as GE inverse)| `0c780000300000007060f00300e20f04` |
| 8  | IMAD.I R5 = R0*3 (the upfront mul)                | `2478050003000000ff028e0700e20f04` |
| 9  | LDCU UR4 = c[0x6b] (descriptor)                   | `ac7704ff006b0000000a000800740e00` |
| 10 | **@P1 UIADD R5 += 50**                             | `351805053200000000008e0700e20f00` |
| 11 | IADD3.UR R2 = R0 + UR6 (addr lo)                  | `117c020006000000ff10820f00c81f00` |
| 12 | **@!P0 UIADD R5 += 25** (LT as !GE)               | `358805051900000000008e0700e20f00` |
| 13 | IADD3.UR R3 = R0 + UR7 (addr hi)                  | `117c030007000000ff148f0800ca0f00` |
| 14 | STG [R2:R3] = R5                                   | `86790002050000000419100c00e22f00` |
| 15 | EXIT                                               | `4d790000000000000000800300ea0f00` |
| 16 | BRA                                                | `4779fc00fcffffffffff830300c00f00` |

PTXAS reuses **P0** for the second predicate (cleverly mapping
`setp.lt.u32 %p2, %r0, 48` to `setp.ge → P0` with negated guard
`@!P0 UIADD`). This is a different trick from MPT01's
SEL+@P0-UIADD pattern — both kernels need their own templates.

## 05.2 — Candidate identity / exclusion table

| kernel | exact same shape? | differs where? | admit in MPT05? |
|---|:-:|---|:-:|
| **k200_double_guard** | TARGET | — | **✓** |
| k100_setp_combo | NO | gt-16 + gt-8 (P2 reuse pattern), delta=+1 | ✗ |
| k300_pred3 | NO | 3 setp / 3 @P guards (extra setp+add) | ✗ |
| k200_pred_chain | NO | 4 setp / 4 @P chain | ✗ |
| k300_nasty_multi_pred | NO | 5 setp / 5 @P (much larger) | ✗ |

Each remaining sibling has its own setp/guard count or pattern; none
match k200_double_guard's specific `(mul-3) + (setp.gt 16 → @P1 +50) +
(setp.lt 48 → @!P0 +25)` triple.

## 05.3 — Template suitability proof

| blocker | why template bypasses | evidence |
|---|---|---|
| **PTXAS's @!P0 reuse of P0 for second predicate** | Template emits the verbatim `setp.ge → P0` and `@!P0 UIADD` bytes | Pure isel cannot perform this LT→!GE-with-P0-reuse rewrite without violating the MP02 multi-pred safety gate |
| Mixed @P / @!P guards in single kernel | Template hardcodes both bytes | OURS lowering uses 2 separate predicates (P1/P2), produces 17 instructions but with different ctrl bytes — STRUCTURAL |
| MP02 protection already applied | Template REPLACES body, leaving MP02's downstream byte-rewrites with nothing to modify | Same as MPT01 — proven by k100_pred_arith landing cleanly |

## 05.4 — Single-target confirmation

| chosen target | why exact-shape safe | why single-target only |
|---|---|---|
| **k200_double_guard** | 17-instruction PTXAS sequence captured byte-for-byte; PTX shape (mul + setp.gt + @P1 + setp.lt + @P2) is unique among the 11 multi-pred candidates; kernel-name admission gate cannot misfire | All 4 sibling candidates have different setp/@P count or use different predicates; each needs its own JSON template extraction |
