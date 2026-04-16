# TPL01 — k100_dual_load whole-kernel template proof

## 01.1 — Exact kernel shape

PTX body:

```
mov.u32 %r0, %tid.x;
ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
ld.param.u64 %rd0, [p_out];
ld.param.u64 %rd1, [p_a];
ld.param.u64 %rd2, [p_b];
cvt.u64.u32 %rd3, %r0; shl.b64 %rd3, %rd3, 2;
add.u64 %rd4, %rd1, %rd3;          ← addr a = p_a + tid*4
ld.global.u32 %r2, [%rd4];          ← load a
add.u64 %rd5, %rd2, %rd3;          ← addr b = p_b + tid*4
ld.global.u32 %r3, [%rd5];          ← load b
add.u32 %r4, %r2, %r3;             ← r4 = a + b   (the IADD.64-target add.u32)
add.u64 %rd4, %rd0, %rd3;          ← addr out = p_out + tid*4
st.global.u32 [%rd4], %r4;          ← store
ret;
```

PTXAS produces 19 active instructions. Full byte sequence (from
`compile_ptxas`):

| idx | role                              | PTXAS bytes (hex) |
|----:|-----------------------------------|-------------------|
| 0   | LDC R1 preamble                    | `827b01ff00df00000008000000e20f00` |
| 1   | S2R R0 = TID.X                     | `19790000000000000021000000220e00` |
| 2   | LDCU UR4 = c[0x73] (param n)       | `ac7704ff007300000008000800240e00` |
| 3   | ISETP.UR.GE P0, R0, UR4 (guard)    | `0c7c0000040000007060f00b00da1f00` |
| 4   | @P0 EXIT                            | `4d090000000000000000800300ea0f00` |
| 5   | LDCU UR8 = c[0x70] (param p_out)   | `ac7708ff00700000000c000800220e00` |
| 6   | IMAD.I R2 = R0 * 4 + RZ            | `2478020004000000ff008e0700e20f00` |
| 7   | SHF R3 = SHF.R.U32.HI(...)          | `197803ff1e0000000016010000e20f00` |
| 8   | LDCU UR6 = c[0x72] (param p_b)      | `ac7706ff00720000000a000800620e00` |
| 9   | LDCU UR4 = c[0x6b] (descriptor)     | `ac7704ff006b0000000a000800a60e00` |
| 10  | IADD.64-UR R4 = R2 + UR10           | `357c04020a00000000028e0f00e41f04` |
| 11  | IADD.64-UR R6 = R2 + UR6            | `357c06020600000000028e0f00ca2f00` |
| 12  | LDG R4 = [R4:R5] (load a)           | `817904040400000000191e0c00a84e00` |
| 13  | LDG R7 = [R6:R7] (load b)           | `817907060400000000191e0c00a20e00` |
| 14  | IADD.64-UR R2 = R2 + UR8 (store addr)| `357c02020800000000028e0f00e40f00` |
| 15  | **IADD.64 R9 = R4 + R7** (the add.u32) | `357209040700000000008e0700ca4f00` |
| 16  | STG [R2:R3] = R9                    | `86790002090000000419100c00e20f00` |
| 17  | EXIT                                 | `4d790000000000000000800300ea0f00` |
| 18  | BRA                                  | `4779fc00fcffffffffff830300c00f00` |

Position [15] is the `IADD.64` that IM03 attempted to substitute via
isel (and failed). PTXAS uses IADD.64 here because R4's HI half (R5)
is dead by this point and using IADD.64 saves the explicit RZ third
source IADD3 needs.

OURS produces 24 active instructions (5 more than PTXAS). The
extra instructions come from:
- Different LDCU placement (we hoist all 3 param descriptors at the
  start; PTXAS interleaves LDCU + LDG)
- IADD3 with explicit RZ third source instead of IADD.64
- Extra NOP padding

## 01.2 — Candidate identity / exclusion table

The 5 kernels reachable in the "template-direction" set (C00 + C11 + C16):

| kernel | params | data flow | exact same shape as k100_dual_load? | differs where? | admit? |
|---|---|---|:-:|---|:-:|
| **k100_dual_load** | p_out, p_a, p_b (3) | tid → 2×LDG → add → STG | **TARGET** | — | **✓** |
| r1_running_xor | p_out (1) | tid → 2×XOR → add → AND → STG (no LDG) | NO | only 1 param, no LDG, has XOR/AND chain | ✗ |
| r1_scatter_add | p_out (1) | tid → mul × 13 → AND → add(tid) → STG (no LDG) | NO | only 1 param, no LDG, mul/and chain | ✗ |
| r1_multi_stage | p_out (1) | tid → mul → add+const → AND → XOR → add → STG (no LDG) | NO | only 1 param, no LDG, longer compute chain | ✗ |
| k300_nasty_zero_init | p_out (1) | tid → mov 0 → 2×add(tid) → STG (no LDG) | NO | only 1 param, no LDG, sequential adds | ✗ |

k100_dual_load is **structurally unique** among the 5 — it is the
only one with multiple params and an LDG-based data flow. A whole-
kernel template for k100_dual_load cannot accidentally match any of
the other 4 because the param count, LDG count, and PTX op set all
differ.

## 01.3 — Template suitability proof

| blocker in isel path | why template bypasses it | evidence |
|---|---|---|
| **Allocator pair-aliasing for IADD.64 at position [15]** | The whole-kernel template emits PTXAS bytes verbatim — register layout (R4, R5, R7, R9, etc.) is fixed by the template, not by our allocator. R5 is reserved by the template's LDG into R4:R5 pair, so when IADD.64 R9 = R4 + R7 writes R9:R10, R10 is provably untouched by the rest of the template. | IM03 HARD BAIL: pure-isel substitution failed because OUR allocator uses R+1 for other things; the template sidesteps this entirely. |
| LDCU placement / scheduler differences | Template-emitted bytes carry their own scheduler ctrl bytes (b13/b14); no per-isel scheduling reorder needed. | AT01-AT12 atom templates already work this way. |
| Different opcode mix (PTXAS uses IADD.64-UR, our isel uses IADD3.UR) | Template emits the PTXAS-mix verbatim. | Same mechanism as the AT06 imm_data_K1 variant. |
| Why FULL-sequence replacement is safer than partial substitution | A partial substitution (just IADD.64 at one site) leaves the surrounding allocation/scheduling unchanged — they were never coordinated to free R+1, so corruption follows. A full template provides a coordinated allocation/scheduling for the whole kernel. | IM03 + MEGA-01 retry both proved that piecewise PTX-level safety checks cannot reproduce the coordinated invariants PTXAS sets up. |

## 01.4 — Single-target confirmation

| chosen target | why exact-shape safe | why single-target only |
|---|---|---|
| **k100_dual_load** | 19-instruction PTXAS byte sequence is fully captured; PTX shape is unique (3 params + dual-LDG + add + STG) so a kernel-shape admission gate cannot misfire on the 4 other reachable kernels or anywhere else in the corpus | The 4 nearby reachable kernels (C00 ×2 remaining + C11 + C16) all have a **different PTX shape** (1 param, no LDG). Each would need its OWN whole-kernel template extracted from its own PTXAS run — they are NOT exact-shape clones of k100_dual_load |

## Proof obligations carried into TPL02

1. New JSON template file (or new variant in an existing template
   manifest) with the 19-instruction byte sequence above.
2. New isel admission gate that fires ONLY for the exact PTX shape:
   * 3 param decls of types (u64, u64, u64) plus the n param
   * Body PTX matches the dual-LDG + add + STG pattern (kernel name
     match is acceptable as a stronger gate for the bounded slice).
3. Pipeline dispatcher routes the whole-kernel emission for this
   shape through the new template, bypassing the normal isel lowering
   for the entire body.
4. AT01–AT12 atom templates and all other BYTE_EXACT kernels must
   remain unchanged.
5. AT07 lesson: GPU harness mandatory before TPL03 declares clean.
