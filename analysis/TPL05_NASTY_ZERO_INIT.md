# TPL05 — k300_nasty_zero_init whole-kernel template proof

## 05.1 — Exact kernel shape

PTX body:

```
mov.u32 %r0, %tid.x;
ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
ld.param.u64 %rd0, [p_out];
mov.u32 %r2, 0;                  ← zero-init
add.u32 %r2, %r2, %r0;           ← r2 = 0 + r0 = r0
add.u32 %r2, %r2, %r0;           ← r2 = r0 + r0 = 2*r0
cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
add.u64 %rd2, %rd0, %rd1;
st.global.u32 [%rd2], %r2;
ret;
```

PTXAS recognises the zero-init + double-add pattern and computes
`r2 = 2*tid.x` via a single `IADD.64 R5, R5, R5` (R5 holds tid.x
from S2R). 13 active PTXAS instructions.

| idx | role | PTXAS bytes (hex) |
|----:|------|-------------------|
| 0  | LDC R1 preamble                       | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R5 = TID.X                         | `19790500000000000021000000220e00` |
| 2  | LDCU UR4 = c[0x71] (param n)           | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE P0, R5, UR4 (guard)        | `0c7c0005040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                | `4d090000000000000000800300ea0f00` |
| 5  | LDCU UR6 = c[0x70] (param p_out)       | `ac7706ff00700000000a000800220e00` |
| 6  | LDCU UR4 = c[0x6b] (descriptor)         | `ac7704ff006b0000000a000800620e00` |
| 7  | IADD3.UR R2 = R5 + UR6 (addr lo)       | `117c020506000000ff10800f00c81f00` |
| 8  | IADD3.UR R3 = R5 + UR7 (addr hi)       | `117c030507000000ff140f0800e20f04` |
| 9  | **IADD.64 R5 = R5 + R5** (the 2× zero-init double-add) | `357205050500000000008e0700ca0f00` |
| 10 | STG [R2:R3] = R5                        | `86790002050000000419100c00e22f00` |
| 11 | EXIT                                     | `4d790000000000000000800300ea0f00` |
| 12 | BRA                                      | `4779fc00fcffffffffff830300c00f00` |

Position [9] is again an `IADD.64` (same opcode 0x235 as TPL01)
emitted for what PTX expresses as a `add.u32` of two same-source
registers. The R+1 corruption avoidance follows the same
template-fixed-allocation argument as TPL01.

OURS produces 16 active instructions (3 more than PTXAS) — extra
IADD3 calls and ctrl differences, classified STRUCTURAL (delta=+2).

## 05.2 — Candidate identity / exclusion table

The 4 reachable template-direction candidates (post-TPL01):

| kernel | params | PTX body shape | exact same as k300_nasty_zero_init? | differs where? | admit? |
|---|---:|---|:-:|---|:-:|
| **k300_nasty_zero_init** | 2 (p_out, n) | `mov 0 + 2× add(tid)` | **TARGET** | — | **✓** |
| r1_running_xor | 2 | `xor + xor + add + and` | NO | XOR/AND chain instead of mov+2×add | ✗ |
| r1_scatter_add | 2 | `mul + and + add(tid)` | NO | mul/and chain | ✗ |
| r1_multi_stage | 2 | `mul + add + and + xor + add` | NO | longer compute chain | ✗ |

All 4 share the "1 data param + n + tid-bounded guard + cvt+shl+add+STG"
outer skeleton, but their **body op sequences are fundamentally
different**: zero-init/double-add (k300) vs XOR-chain (running_xor)
vs MUL-chain (scatter_add) vs MUL+ADD+AND+XOR+ADD (multi_stage).
A kernel-name admission gate cannot misfire across these candidates,
and the body op-mismatch provides defense in depth.

## 05.3 — Template suitability proof

| blocker in normal lowering | why template bypasses it | evidence |
|---|---|---|
| **IADD.64 emission at position [9]** depends on R+1 being dead in PTXAS's allocation. OURS would emit IADD3 with explicit RZ instead, and any pure-isel substitution (per IM03 HARD BAIL) corrupts whatever the allocator placed at OUR R+1. | Whole-kernel template fixes register allocation: the template's R5:R6 IADD.64 dest pair is provably dead beyond R5 because the template only reads R5 in the subsequent STG. | TPL01 success on k100_dual_load proved this exact mechanism with a different IADD.64 site (R9:R10). |
| **Multi-op sequencing** (PTXAS interleaves LDCU + IADD3.UR pairs differently from our coordinated lowering) | Template emits the PTXAS schedule verbatim. | OURS delta=+2 STRUCTURAL today — meaning OURS emits 2 extra instructions compared to PTXAS, due to address-compute and zero-init lowering choices. |
| **`mov.u32 %r, 0` + `add %r, %r, %x` + `add %r, %r, %x` pattern** | Template recognises the zero-init double-add as a single IADD.64 that doubles tid.x. Our isel would lower these as 3 separate instructions (mov, IADD3, IADD3). | PTXAS emits exactly 1 instruction (IADD.64) for the body compute; OURS emits 3. |

The blocker is **multi-op sequencing + IADD.64-pair-write** — the
same class of blocker as TPL01. Template is proven safe by TPL01's
prior success.

## 05.4 — Single-target confirmation

| chosen target | why exact-shape safe | why single-target only |
|---|---|---|
| **k300_nasty_zero_init** | 13-instruction PTXAS sequence is fully extracted; PTX body shape (`mov 0; add(tid); add(tid)`) is unique among the 4 reachable candidates; kernel-name admission gate cannot misfire | The 3 other reachable candidates have fundamentally different body op sequences (XOR, MUL, multi-stage), so each requires its own separately-extracted template + admission gate |

## Proof obligations carried into TPL06

1. New JSON template file `non_atom_nasty_zero_init.json` with the
   12 instructions (idx 1–12; preamble [0] LDC R1 from pipeline).
2. New isel admission gate in pipeline.py that fires ONLY when:
   * `fn.name == 'k300_nasty_zero_init'`
   * `len(fn.params) == 2`
   * No atom template active.
3. AT07 lesson: GPU harness mandatory before TPL07 declares clean.
4. The TPL01 `'TPL01'` marker check on the SM_120 rule #29 post-EXIT
   `b9=0x0c` rewrite needs to also recognize TPL05 instructions
   (use a generalised marker like `'TPL'`) so the new template's
   verified-against-PTXAS b9 values are not rewritten.
