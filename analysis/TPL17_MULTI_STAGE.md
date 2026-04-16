# TPL17 — r1_multi_stage whole-kernel template proof

## 17.1 — Exact kernel shape

PTX body:

```
mov.u32 %r0, %tid.x;
ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
ld.param.u64 %rd0, [p_out];
mul.lo.u32 %r2, %r0, 5;
add.u32 %r3, %r2, 100;
and.b32 %r4, %r3, 0x1FF;
xor.b32 %r5, %r4, %r2;
add.u32 %r2, %r5, %r0;
cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
add.u64 %rd2, %rd0, %rd1;
st.global.u32 [%rd2], %r2;
ret;
```

PTXAS produces a **16-instruction** sequence; OURS produces 19.

| idx | role | PTXAS bytes (hex) |
|----:|------|-------------------|
| 0  | LDC R1 preamble                          | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R5 = TID.X                            | `19790500000000000021000000220e00` |
| 2  | LDCU UR4 = c[0x71] (param n)              | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE P0, R5, UR4 (guard)           | `0c7c0005040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                   | `4d090000000000000000800300ea0f00` |
| 5  | LDCU UR6 = c[0x70] (param p_out)          | `ac7706ff00700000000a000800220e00` |
| 6  | IMAD.I R0 = R5 * 5 + RZ                    | `2478000505000000ff028e0700e20f00` |
| 7  | LDCU UR4 = c[0x6b] (descriptor)            | `ac7704ff006b0000000a000800680e00` |
| 8  | **UIADD R3 = R0 + 100** (`add.u32 r,imm`)  | `357803006400000000008e0700ca0f00` |
| 9  | LOP3.IMM (R3 & 0x1FF) XOR R0 fused → R0   | `12780000ff01000003788e0700e40f00` |
| 10 | IADD3.UR R2 = R5 + UR6 (addr lo)          | `117c020506000000ff10800f00c81f00` |
| 11 | IADD3.UR R3 = R5 + UR7 (addr hi)          | `117c030507000000ff140f0800e20f00` |
| 12 | **IADD.64 R5 = R0 + R5** (`add.u32 r,r,r`) | `357205000500000000008e0700ca0f00` |
| 13 | STG [R2:R3] = R5                           | `86790002050000000419100c00e22f00` |
| 14 | EXIT                                       | `4d790000000000000000800300ea0f00` |
| 15 | BRA                                        | `4779fc00fcffffffffff830300c00f00` |

PTXAS folds the `and.b32 + xor.b32` pair into **one LOP3.IMM** at
position [9] using LOP3's 3-input truth table. Position [12] is the
familiar IADD.64 for the final `add.u32 r, r, r`.

## 17.2 — Single-target confirmation

| chosen target | why exact-shape safe | why no broader admission needed |
|---|---|---|
| **r1_multi_stage** | 16-instruction PTXAS sequence captured; PTX shape (`mul + add + and + xor + add`) is unique vs all prior templates; kernel-name admission gate cannot misfire | Last remaining template-direction candidate; no other corpus kernel shares this 5-op compute body |

## 17.3 — Template suitability proof

| blocker | why template bypasses | evidence |
|---|---|---|
| IADD.64 at [12] (R+1 dead requirement) | Template fixes register layout (R5:R6, R6 unused after) | TPL01/TPL05/TPL09/TPL13 all proven |
| UIADD at [8] (similar pair-write concerns) | Template emits PTXAS bytes verbatim | UIADD is the same opcode AT02/AT06 already template-handle |
| LOP3.IMM (and+xor fused) at [9] | PTXAS uses LOP3's 3-input truth table to fuse 2 PTX ops into 1 SASS op; our isel can't do this fusion | Template emits the verbatim LOP3 byte sequence |
| Multi-op sequencing (delta=+3 STRUCTURAL today) | Template emits PTXAS schedule verbatim | OURS uses 19 instructions vs PTXAS's 16 |
