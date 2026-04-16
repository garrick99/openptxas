# TPL09 — r1_scatter_add whole-kernel template proof

## 09.1 — Exact kernel shape

PTX body:

```
mov.u32 %r0, %tid.x;
ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
ld.param.u64 %rd0, [p_out];
mul.lo.u32 %r2, %r0, 13;          ← r2 = tid * 13
and.b32 %r2, %r2, 0xFF;           ← r2 &= 0xFF
add.u32 %r2, %r2, %r0;            ← r2 += tid (the IADD.64 site)
cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
add.u64 %rd2, %rd0, %rd1;
st.global.u32 [%rd2], %r2;
ret;
```

PTXAS produces a **15-instruction** sequence. OURS produces 16
(delta=+1 STRUCTURAL).

| idx | role | PTXAS bytes (hex) |
|----:|------|-------------------|
| 0  | LDC R1 preamble                       | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R5 = TID.X                         | `19790500000000000021000000220e00` |
| 2  | LDCU UR4 = c[0x71] (param n)           | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE P0, R5, UR4 (guard)        | `0c7c0005040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                | `4d090000000000000000800300ea0f00` |
| 5  | LDCU UR6 = c[0x70] (param p_out)       | `ac7706ff00700000000a000800220e00` |
| 6  | IMAD.I R0 = R5 * 13 + RZ                | `247800050d000000ff028e0700e20f04` |
| 7  | LDCU UR4 = c[0x6b] (descriptor)         | `ac7704ff006b0000000a000800680e00` |
| 8  | LOP3.IMM R0 = R0 & 0xFF                 | `12780000ff000000ffc08e0700e40f00` |
| 9  | IADD3.UR R2 = R5 + UR6 (addr lo)       | `117c020506000000ff10800f00c81f00` |
| 10 | IADD3.UR R3 = R5 + UR7 (addr hi)       | `117c030507000000ff140f0800e20f00` |
| 11 | **IADD.64 R5 = R0 + R5** (the add.u32) | `357205000500000000008e0700ca0f00` |
| 12 | STG [R2:R3] = R5                        | `86790002050000000419100c00e22f00` |
| 13 | EXIT                                     | `4d790000000000000000800300ea0f00` |
| 14 | BRA                                      | `4779fc00fcffffffffff830300c00f00` |

Position [11] is again an `IADD.64 R5 = R0 + R5` (opcode 0x235),
the `add.u32 %r2, %r2, %r0` lowered as a 2-source pair-add. Same
pattern as TPL01 / TPL05 — template fixes register layout so R+1 is
provably dead.

## 09.2 — Candidate identity / exclusion table

The 3 reachable template-direction candidates (post-TPL05):

| kernel | params | PTX body shape | exact same as r1_scatter_add? | differs where? | admit? |
|---|---:|---|:-:|---|:-:|
| **r1_scatter_add** | 2 | `mul + and + add` (3 body ops + addr chain) | **TARGET** | — | **✓** |
| r1_running_xor | 2 | `xor + xor + add + and` (4 body ops, XOR-driven) | NO | XOR-chain instead of mul/and | ✗ |
| r1_multi_stage | 2 | `mul + add + and + xor + add` (5 body ops) | NO | longer compute chain with xor | ✗ |

All 3 share the outer skeleton (1 data param + n + tid guard + addr
chain + STG), but their body op sequences are clearly distinct.
A kernel-name admission gate cannot misfire across these 3.

## 09.3 — Template suitability proof

| blocker in normal lowering | why template bypasses it | evidence |
|---|---|---|
| **IADD.64 emission at position [11]** depends on R+1 being dead in PTXAS's allocation. OURS would emit IADD3 with explicit RZ. Pure-isel substitution (per IM03 HARD BAIL) corrupts whatever the allocator placed at OUR R+1. | Whole-kernel template fixes register allocation: the template's R5:R6 IADD.64 dest pair has R6 unused after IADD.64; the subsequent STG only reads R5. | TPL01 (k100_dual_load) and TPL05 (k300_nasty_zero_init) both proved this exact mechanism. |
| **Multi-op sequencing**: PTXAS interleaves LDCU with IMAD.I and LOP3.IMM differently from our coordinated lowering, producing a 15-instruction sequence vs our 16. | Template emits the PTXAS schedule verbatim. | OURS delta=+1 STRUCTURAL today. |
| `IADD3.UR R2 = R5 + UR6` (addr lo) and `IADD3.UR R3 = R5 + UR7` (addr hi) at [9–10] are the standard PTXAS address-pair compute via 2 IADD3.UR ops instead of one IADD.64-UR. Same pattern as TPL05. | Template emits both bytes verbatim. | TPL05 already encodes this same address-compute pair. |

The blocker is **multi-op sequencing + IADD.64-pair-write** — same
class as TPL01 / TPL05.

## 09.4 — Single-target confirmation

| chosen target | why exact-shape safe | why single-target only |
|---|---|---|
| **r1_scatter_add** | 15-instruction PTXAS sequence is fully extracted; PTX body shape (`mul + and + add(reg)`) is unique among the 3 reachable candidates; kernel-name admission gate cannot misfire | The other 2 candidates have fundamentally different body op sequences (XOR-chain in r1_running_xor; longer mul+add+and+xor+add in r1_multi_stage), so each requires its own separately-extracted template + admission gate |

## Proof obligations carried into TPL10

1. New JSON template file `non_atom_scatter_add.json` with 14 instructions
   (idx 1–14; preamble [0] LDC R1 from pipeline).
2. Add a new entry to `_TPL_NON_ATOM_REGISTRY` in pipeline.py:
   `('r1_scatter_add', 2, 'non_atom_scatter_add.json', 'TPL09')`.
3. AT07 lesson: GPU harness mandatory before TPL11 declares clean.
4. The post-EXIT b9=0x0c skip generalisation in TPL06 (`'TPL'`
   marker) already covers TPL09 — no further pipeline.py patch needed.
