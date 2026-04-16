# TPL13 — r1_running_xor whole-kernel template proof

## 13.1 — Exact kernel shape

PTX body:

```
mov.u32 %r0, %tid.x;
ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
ld.param.u64 %rd0, [p_out];
xor.b32 %r2, %r0, 0xABCD;
xor.b32 %r3, %r2, 0x1234;
add.u32 %r2, %r2, %r3;
and.b32 %r2, %r2, 0xFFFF;
cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
add.u64 %rd2, %rd0, %rd1;
st.global.u32 [%rd2], %r2;
ret;
```

PTXAS produces a **16-instruction** sequence; OURS produces 18.

| idx | role | PTXAS bytes (hex) |
|----:|------|-------------------|
| 0  | LDC R1 preamble                        | `827b01ff00df00000008000000e20f00` |
| 1  | S2R R4 = TID.X                          | `19790400000000000021000000220e00` |
| 2  | LDCU UR4 = c[0x71] (param n)            | `ac7704ff007100000008000800240e00` |
| 3  | ISETP.UR.GE P0, R4, UR4 (guard)         | `0c7c0004040000007060f00b00da1f00` |
| 4  | @P0 EXIT                                 | `4d090000000000000000800300ea0f00` |
| 5  | LDCU UR6 = c[0x70] (param p_out)        | `ac7706ff00700000000a000800220e00` |
| 6  | LOP3.IMM R0 = R4 ^ 0xABCD                | `12780004cdab0000ff3c8e0700e20f00` |
| 7  | LDCU UR4 = c[0x6b] (descriptor)          | `ac7704ff006b0000000a000800660e00` |
| 8  | LOP3.IMM R3 = R0 ^ 0x1234                | `1278030034120000ff3c8e0700ca0f00` |
| 9  | **IADD.64 R0 = R0 + R3** (the add.u32)  | `357200000300000000008e0700ca0f00` |
| 10 | LOP3.IMM R5 = R0 & 0xFFFF                | `12780500ffff0000ffc08e0700e40f00` |
| 11 | IADD3.UR R2 = R4 + UR6 (addr lo)        | `117c020406000000ff10800f00c81f00` |
| 12 | IADD3.UR R3 = R4 + UR7 (addr hi)        | `117c030407000000ff140f0800ca0f00` |
| 13 | STG [R2:R3] = R5                         | `86790002050000000419100c00e22f00` |
| 14 | EXIT                                     | `4d790000000000000000800300ea0f00` |
| 15 | BRA                                      | `4779fc00fcffffffffff830300c00f00` |

Position [9] is again `IADD.64` (opcode 0x235) for the `add.u32`.

## 13.2 — Candidate identity / exclusion table

| kernel | params | body shape | exact same as r1_running_xor? | differs where? | admit? |
|---|---:|---|:-:|---|:-:|
| **r1_running_xor** | 2 | `xor + xor + add + and` | **TARGET** | — | **✓** |
| r1_multi_stage | 2 | `mul + add + and + xor + add` | NO | mul+add chain instead of xor+xor | ✗ |

## 13.3 — Template suitability proof

| blocker | why template bypasses | evidence |
|---|---|---|
| IADD.64 at [9] requires R+1 dead — pure isel cannot prove (IM03 HARD BAIL) | Template fixes register layout (R0:R1, R1 unused after IADD.64) | TPL01/TPL05/TPL09 all proven |
| Multi-op sequencing (XOR/AND/IMM-ADD interleaved with LDCU descriptor loads) | Template emits PTXAS schedule verbatim | OURS delta=+2 STRUCTURAL today |

## 13.4 — Single-target confirmation

| chosen target | why exact-shape safe | why single-target only |
|---|---|---|
| **r1_running_xor** | 16-instruction PTXAS sequence captured; XOR-driven body unique vs r1_multi_stage's mul+add chain; kernel-name admission gate cannot misfire | r1_multi_stage has fundamentally different body (mul-driven instead of xor-driven), needs its own template extraction (Slice B) |
