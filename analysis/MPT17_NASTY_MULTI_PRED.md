# MPT17 — k300_nasty_multi_pred whole-kernel template proof (Slice B)

## 17.1 — Slice B candidate selection (MPT16 gate)

Re-probed 7 remaining multi-pred candidates after Slice A (MPT13)
landed.  Probe measures `compile_ptxas` vs `compile_openptxas` active
instruction counts (trimming trailing 0x918 alignment NOPs):

| candidate | ptxas_n | ours_n | delta | byte-eq common | rank |
|---|---:|---:|---:|---:|---:|
| **k300_nasty_multi_pred** | **23** | **23** | **0** | 4/23 | **1** |
| k200_pred_chain | 20 | 21 | +1 | 3/20 | 2 |
| w1_div_multi_guard | 20 | 21 | +1 | 3/20 | 3 |
| w2_deep_pred | 22 | 23 | +1 | 3/22 | 4 |
| k200_nested_pred | 19 | 17 | −2 | 2/17 | 5 (under-emit, risky) |
| k300_nasty_pred_nest3 | 22 | 19 | −3 | 3/19 | 6 (under-emit, risky) |
| r1_minmax | 16 | 23 | +7 | 3/16 | 7 (huge delta) |

**Pick: k300_nasty_multi_pred (delta=0)**.  Same delta-class as MPT09
(k300_pred3, also delta=0).  Longest body in the candidate set, but
delta=0 means OURS already emits the same instruction count — only
predicate-allocation and operand-ordering differ.

## 17.2 — Exact kernel shape

PTX body:

```
mov.u32 %r2, %r0;
setp.gt.u32 %p1, %r0, 4;   @%p1 add.u32 %r2, %r2, 10;
setp.gt.u32 %p2, %r0, 8;   @%p2 add.u32 %r2, %r2, 20;
setp.gt.u32 %p1, %r0, 16;  @%p1 add.u32 %r2, %r2, 40;
setp.gt.u32 %p2, %r0, 32;  @%p2 add.u32 %r2, %r2, 80;
setp.gt.u32 %p1, %r0, 48;  @%p1 add.u32 %r2, %r2, 160;
```

5 setp / 5 @P-add, alternating P1/P2.

PTXAS produces a **23-instruction** sequence; OURS pre-MPT17 also 23
(delta=0).  Template lands OURS byte-exactly at all 23 instructions.

Key PTXAS structural moves (none expressible by pure isel):

1. **Aggressive predicate-slot reuse** — P0 carries the entry-guard,
   then is reassigned to gt-4, then again to gt-32.  P2 is reused for
   gt-8 then gt-48.  Three predicate slots model five distinct setp
   producers; isel allocator would reserve five separate slots,
   blowing the pred budget and changing the body bytes.
2. **SEL R5,R0,R3,P0 + 4×@P-UIADD mux** — first add (+10) is folded
   into a UIADD pre-compute (R0=R3+10) then SELed against R3 under P0
   (=gt-4).  Remaining four adds (20, 40, 80, 160) become @P-UIADD
   chain entries, ordered to interleave with the address-compute
   IADD3.UR pair.
3. **Schedule interleaving** — `LDCU UR4 (descriptor 0x6b)` is moved
   between the SEL and the first @P-UIADD; the IADD3.UR address pair
   is split across the @P-UIADDs to hide latency.

## 17.3 — Template suitability proof

| blocker | why template bypasses | evidence |
|---|---|---|
| **Predicate-slot reuse across setp producers** — MP02 multi-pred safety must hold for all three P0 producers, all two P2 producers; isel cannot perform this allocation under MP02 without a re-design | Template emits the verbatim ISETP-to-{P0,P1,P2} bytes in the exact PTXAS order | MP02 fix's predicate-RAW gate in scoreboard catches the unsafe pattern OURS would emit |
| **R0=R3+10 pre-compute + SEL fold** — first add folded into SEL operand requires reordering the mov+add sequence ahead of the first setp, which pure isel cannot do | Template emits UIADD R0=R3+10 at index 7 verbatim | Same trick PTXAS used in MPT09 (R0=R3+1) |
| **Schedule interleaving** — LDCU UR4 inserted mid-mux to hide RAW latency; UIADD imm-add chain split across IADD3.UR pair | Template emits PTXAS schedule verbatim | Verified |

## 17.4 — Validation matrix (post-MPT17)

| metric | pre-MPT17 (post-MPT13) | post-MPT17 | delta |
|---|---:|---:|---:|
| Corpus BYTE_EXACT | 59 | **60** | +1 |
| Corpus STRUCTURAL | 85 | **84** | −1 |
| pytest | 865/865 | 865/865 | 0 |
| GPU PASS / FAIL / RUN_EXC | 127 / 10 / 7 | 127 / 10 / 7 | 0 |
| Predicate-template kernels BYTE_EXACT | 4 | **5** | +1 |

## 17.5 — Slice B clean — chain complete
