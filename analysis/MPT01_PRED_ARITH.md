# MPT01 — Predicate-body frontier clustering and first MP02-aware target

## 01.1 — Family clustering

11 STRUCTURAL kernels with MP02-protected multi-predicate body shape
(post-EXIT body has ≥ 2 setp + ≥ 2 @P-guarded ops, no loops, no atoms):

| kernel                  | delta | ours/ptxas | #setp post-EXIT | #@P guards |
|-------------------------|------:|------------|----------------:|-----------:|
| **k100_pred_arith**     |    +0 | 17/17      | 2               | 2          |
| **k200_double_guard**   |    +0 | 17/17      | 2               | 2          |
| k200_nested_pred        |    −2 | 17/19      | 2               | 3          |
| k100_setp_combo         |    +1 | 17/16      | 2               | 2          |
| r1_minmax               |    +4 | 20/16      | 2               | 2          |
| k300_pred3              |    +0 | 19/19      | 3               | 3          |
| k300_nasty_pred_nest3   |    −3 | 19/22      | 3               | 5          |
| k200_pred_chain         |    +1 | 21/20      | 4               | 4          |
| w1_div_multi_guard      |    +1 | 21/20      | 4               | 4          |
| k300_nasty_multi_pred   |    +0 | 23/23      | 5               | 5          |
| w2_deep_pred            |    +1 | 23/22      | 5               | 5          |

## 01.2 — Clean-slice prioritization

| subcluster | count | clean? | priority | reason |
|---|---:|:-:|---|---|
| **delta=0 + 2-setp + 2-@P + simplest body** | 2 | ✓ | **HIGHEST** | k100_pred_arith and k200_double_guard tie on shape complexity |
| delta=0 + 3-setp + 3-@P (k300_pred3) | 1 | ✓ | medium | larger body, more PTXAS bytes to capture |
| delta=0 + 5-setp + 5-@P (k300_nasty_multi_pred) | 1 | ✓ | low | longest delta=0 body |
| delta ≠ 0 (8 others) | 8 | mixed | low | non-zero delta indicates additional structural difference beyond the predicate-body itself |

## 01.3 — Shape proof for k100_pred_arith (chosen)

PTX body:

```
mov.u32 %r0, %tid.x;
ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
ld.param.u64 %rd0, [p_out];
mov.u32 %r2, %r0;
setp.gt.u32 %p1, %r0, 16;        ← P1 producer
@%p1 add.u32 %r2, %r2, 100;      ← P1 consumer
setp.gt.u32 %p2, %r0, 48;        ← P2 producer
@%p2 add.u32 %r2, %r2, 200;      ← P2 consumer
cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
add.u64 %rd2, %rd0, %rd1;
st.global.u32 [%rd2], %r2;
ret;
```

* **# predicates**: 3 (entry P0 + body P1 + body P2)
* **# predicated body instructions**: 2 (`@%p1 add` + `@%p2 add`)
* **straight-line after entry guard**: yes (no loops, no merges)
* **MP02 guardrails**: already classify this kernel as multi-pred and
  apply the FG33 ctrl-byte skip + FG56 R3→R0 rename. Those keep it
  GPU-correct (PASS) but leave it STRUCTURAL because OURS does not
  produce the *exact* PTXAS schedule.
* **why partial isel substitution is wrong**: PTXAS hoists both
  ISETPs above both predicated UIADDs and uses an unconditional UIADD
  to pre-compute `r2 + 100`, then a `@P0 UIADD` to add 200, plus a
  `SEL` to pick a value. The reordering and the SEL trick cannot be
  expressed as a per-instruction byte rewrite without an allocator-
  aware pass.

PTXAS 17-instruction sequence:

| idx | bytes (hex) |
|----:|-------------|
| 0   | `827b01ff00df00000008000000e20f00`  LDC R1 preamble |
| 1   | `19790300000000000021000000220e00`  S2R R3 = TID.X |
| 2   | `ac7704ff007100000008000800240e00`  LDCU UR4 = c[0x71] (n) |
| 3   | `0c7c0003040000007060f00b00da1f00`  ISETP.UR.GE P0,R3,UR4 |
| 4   | `4d090000000000000000800300ea0f00`  @P0 EXIT |
| 5   | `ac7706ff00700000000a000800220e00`  LDCU UR6 = c[0x70] (p_out) |
| 6   | `0c780003300000007040f00300e20f04`  ISETP.IMM (cmp=GT, imm=48) |
| 7   | `0c780003100000007040f20300e20f04`  ISETP.IMM (cmp=GT, imm=16) |
| 8   | `357800036400000000008e0700e20f00`  UIADD R0=R3+100 (unconditional) |
| 9   | `ac7704ff006b0000000a000800680e00`  LDCU UR4 = c[0x6b] (descriptor) |
| 10  | `07720500030000000000800000cc0f00`  SEL R5,R0,R3 (predicate-mux) |
| 11  | `35080505c800000000008e0700e20f00`  @P0 UIADD R5 += 200 |
| 12  | `117c020306000000ff10820f00c81f00`  IADD3.UR R2 = R3 + UR6 (addr lo) |
| 13  | `117c030307000000ff148f0800ca0f00`  IADD3.UR R3 = R3 + UR7 (addr hi) |
| 14  | `86790002050000000419100c00e22f00`  STG [R2:R3] = R5 |
| 15  | `4d790000000000000000800300ea0f00`  EXIT |
| 16  | `4779fc00fcffffffffff830300c00f00`  BRA |

PTXAS uses **SEL + @P0-UIADD** instead of two separate `@P add`
patterns — exactly the kind of clever predicate-mux that pure isel
substitution cannot produce.

## 01.4 — Single-target confirmation

| chosen target | why exact-shape safe | why single-target only | why siblings deferred |
|---|---|---|---|
| **k100_pred_arith** | 17-instruction PTXAS sequence captured; PTX shape (3 params + 2 setp + 2 @P-add + addr + STG) is unique among the 11 multi-pred candidates by the exact body op + setp/guard pair counts; kernel-name admission gate is the strongest defense; MP02 guardrails already apply (no fight with existing predicate handling) | Of the 11 candidates, only k100_pred_arith and k200_double_guard match the 2-setp + 2-@P + delta=0 shape; k200_double_guard adds an upfront `mul.lo.u32` not present in k100_pred_arith — different PTX body | k200_double_guard is sibling-suitable for MPT05; the other 9 candidates have larger setp/guard counts (≥3) or non-zero delta indicating extra structural difference. Each needs its own template extraction |

## Proof obligations carried into MPT02

1. New JSON template `non_atom_pred_arith.json` (16 instructions =
   PTXAS [1..16]; preamble [0] LDC R1 from pipeline).
2. Add registry entry `('k100_pred_arith', 2, ..., 'MPT01')` to the
   `_TPL_NON_ATOM_REGISTRY` list. Defense-in-depth: the existing
   admission gate checks kernel-name + param count + no atom
   template; that fully suffices for this kernel.
3. The TPL06 generalised post-EXIT b9 skip (`'TPL'` marker) does NOT
   automatically cover MPT-tagged instructions. Either generalise
   further (use `'TPL' or 'MPT'` check, or just `'TPL' in comment or
   'MPT' in comment`) OR tag the new template with `'TPL'` prefix.
4. AT07 lesson: GPU harness mandatory before MPT03 declares clean.
5. MP02 fixes (FG33 / FG56 / FG60) must remain intact; the template
   path runs *before* those fixes in pipeline.py, so they never see
   the substituted bytes — but the proof needs verification.
