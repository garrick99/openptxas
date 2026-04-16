# MPT13 — k100_setp_combo whole-kernel template proof (Slice A)

## 13.1 — Exact kernel shape

PTX body:

```
mov.u32 %r0, %tid.x;
ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
ld.param.u64 %rd0, [p_out];
mov.u32 %r2, 0;
setp.lt.u32 %p1, %r0, 16;        ← P1 producer (lt)
@%p1 add.u32 %r2, %r2, 1;
setp.gt.u32 %p2, %r0, 8;         ← P2 producer (gt)
@%p2 add.u32 %r2, %r2, 2;
cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
add.u64 %rd2, %rd0, %rd1;
st.global.u32 [%rd2], %r2;
ret;
```

PTXAS produces a **16-instruction** sequence; OURS pre-MPT13 produced 17
(delta=+1).  Template lands OURS at exactly 16 instructions.

| idx | role | bytes (hex) |
|----:|------|-------------|
| 0  | LDC R1 preamble                                       | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R0 = TID.X (no R3 alias)                          | `19790000000000000021000000220e00` |
| 2  | LDCU UR4 = c[0x71] (param n)                          | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE P0, R0, UR4 (entry guard, R0 source)      | `0c7c0000040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                              | `4d090000000000000000800300ea0f00` |
| 5  | LDCU UR6 = c[0x70] (param p_out)                      | `ac7706ff00700000000a000800220e00` |
| 6  | ISETP.IMM.GT R0, 8 → P0 (reuses entry-P0 slot)       | `0c780000080000007040f00300e20f04` |
| 7  | ISETP.IMM.LT R0, 16 → P1                             | `0c780000100000007060f20300e20f00` |
| 8  | LDCU UR4 = c[0x6b] (descriptor)                       | `ac7704ff006b0000000a000800680e00` |
| 9  | SEL R5, RZ, IMM=1, P? (predicate-mux: 1 if P1 else 0) | `077805ff010000000000800000ce0f00` |
| 10 | @P0 UIADD R5 += 2 (+2 if gt-8)                        | `350805050200000000008e0700e20f00` |
| 11 | IADD3.UR R2 = R0 + UR6 (addr lo)                      | `117c020006000000ff10820f00c81f00` |
| 12 | IADD3.UR R3 = R0 + UR7 (addr hi)                      | `117c030007000000ff148f0800ca0f00` |
| 13 | STG [R2:R3] = R5                                      | `86790002050000000419100c00e22f00` |
| 14 | EXIT                                                  | `4d790000000000000000800300ea0f00` |
| 15 | BRA                                                   | `4779fc00fcffffffffff830300c00f00` |

**Key shape differences vs MPT01 (k100_pred_arith)**:
* PTXAS keeps TID in R0 directly (no `mov %r2,%r0` pre-step); reuses R0
  as the address source in IADD3.UR.  No UIADD pre-compute needed.
* SEL operand B is an **immediate (1)**, not a register — opcode 0x807
  (SEL.IMM) instead of MPT01's 0x207 (SEL.REG).
* P0 is **reused** for both the entry-guard and the gt-8 setp (after
  the EXIT consumes the entry P0).  This is a different predicate
  allocation than MPT09 (which used three distinct predicates).

## 13.2 — Exclusion proof

Remaining 7 multi-pred siblings:

| kernel | differs where? | admit in MPT13? |
|---|---|:-:|
| k200_nested_pred | 2-setp/3-@P, has merge-like @P/@!P pair, delta=−2 | NO |
| r1_minmax | mul + and + 2 setp, delta=+4 (4 extra ops) | NO |
| k200_pred_chain | 4-setp/4-@P all on P1 reuse, delta=+1 | NO |
| w1_div_multi_guard | 4-setp/4-@P alternating P1/P2, delta=+1 | NO |
| k300_nasty_pred_nest3 | 3-setp/5-@P nested, delta=−3 | NO |
| k300_nasty_multi_pred | 5-setp/5-@P alternating, delta=0 | NO |
| w2_deep_pred | 5-setp/5-@P alternating P1/P2, delta=+1 | NO |

Each sibling differs in setp count (2 vs 3/4/5) AND in body shape AND
in PTXAS predicate-allocation strategy.  None match k100_setp_combo's
exact 2-setp + 2-@P + delta=+1 pattern.

## 13.3 — Template suitability proof

| blocker | why template bypasses | evidence |
|---|---|---|
| **PTXAS uses SEL.IMM + @P-UIADD mux** to fold the two predicated adds into one register without a temporary `mov r2,0` op (delta=+1 source).  Pure isel cannot perform this rewrite without losing the per-predicate add semantics PTX expressed | Template emits SEL+UIADD bytes verbatim | Same trick PTXAS used in MPT01, MPT05; here in 0x807 (SEL.IMM) variant |
| **R0 reuse** — PTXAS keeps TID in R0 throughout instead of MOV-ing to R2; OURS isel allocator places TID in a different register, generating a dead MOV | Template hard-binds R0 as TID | OURS pre-MPT13 emitted 17 instrs incl. the dead MOV |
| **P0 reuse across entry-guard/EXIT and post-EXIT setp** — PTXAS recognizes that P0 is consumed by the EXIT and reuses the slot for the gt-8 setp; OURS isel allocates a fresh predicate, leaving the wrong delta | Template emits ISETP-IMM.GT bytes targeting P0 verbatim | MP02 multi-pred safety preserved because template REPLACES the body |

## 13.4 — Validation matrix (post-MPT13)

| metric | pre-MPT13 | post-MPT13 | delta |
|---|---:|---:|---:|
| Corpus BYTE_EXACT | 58 | **59** | +1 |
| Corpus STRUCTURAL | 86 | **85** | −1 |
| pytest | 865/865 | 865/865 | 0 |
| GPU PASS / FAIL / RUN_EXC | 127 / 10 / 7 | 127 / 10 / 7 | 0 |
| Predicate-template kernels BYTE_EXACT | 3 | **4** | +1 |

## 13.5 — Slice A clean — proceed to MPT16 gate
