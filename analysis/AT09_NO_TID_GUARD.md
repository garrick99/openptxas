# AT09 — No-tid-guard atom sibling proof

## 09.1 — Sibling byte comparison

`k100_atom_add` (15 active instructions) vs `multi_block_atomic`
(11 active instructions). Indices below are the active-instruction
indices (NOPs trimmed).

| ka_idx | mb_idx | role                       | ka bytes (last 16 hex)                | mb bytes                              | identical? |
|-------:|-------:|----------------------------|---------------------------------------|---------------------------------------|:-:|
| 0      | 0      | LDC R1 preamble            | `827b01ff00df00000008000000e20f00`    | `827b01ff00df00000008000000e20f00`    | ✓ |
| 1      | —      | S2R R0 = TID.X             | `19790000000000000021000000220e00`    | (absent)                              | n/a |
| 2      | —      | LDCU UR4 = c[0x71]         | `ac7704ff007100000008000800240e00`    | (absent)                              | n/a |
| 3      | —      | ISETP.UR.GE P0, R0, UR4    | `0c7c0000040000007060f00b00da1f00`    | (absent)                              | n/a |
| 4      | —      | @P0 EXIT                   | `4d090000000000000000800300ea0f00`    | (absent)                              | n/a |
| 5      | 1      | S2R R0 = LANEID            | `19790000000000000000000000220e00`    | `19790000000000000000000000220e00`    | ✓ |
| 6      | 2      | 0x886 R6 (UR pipe init)    | `867806000000000000018e0300e20f00`    | `867806000000000000018e0300e20f00`    | ✓ |
| 7      | 3      | LDCU UR4 = c[0x6b] (desc)  | `ac7704ff006b0000000a000800620e00`    | `ac7704ff006b0000000a000800620e00`    | ✓ |
| 8      | 4      | 0xd09 R5, R6               | `097d0500060000000000000800620e00`    | `097d0500060000000000000800680e00`    | **b13: 0x62 vs 0x68** |
| 9      | 5      | LDC R2 = c[0xe0]           | `827b02ff00e00000000a000000620e00`    | `827b02ff00e00000000a000000620e00`    | ✓ |
| 10     | 6      | 0x2bd R7, R6               | `bd7207000600000000000e0800cc0f00`    | `bd7207000600000000000e0800cc0f00`    | ✓ |
| 11     | 7      | ISETP.UR.GE P0, R0, UR7    | `0c7c0000070000007020f00b00da1f00`    | `0c7c0000070000007020f00b00da1f00`    | ✓ |
| 12     | 8      | ATOMG.ADD R0, R2, R5       | `8e0900020500000004e1120c00e22f00`    | `8e0900020500000004e1120c00e22f00`    | ✓ |
| 13     | 9      | EXIT                       | `4d790000000000000000800300ea0f00`    | `4d790000000000000000800300ea0f00`    | ✓ |
| 14     | 10     | BRA                        | `4779fc00fcffffffffff830300c00f00`    | `4779fc00fcffffffffff830300c00f00`    | ✓ |

Sibling shape proof:

* The 4-instruction tid-guard prelude (`S2R R0 = TID.X` + `LDCU UR4 =
  c[0x71]` + `ISETP.UR.GE P0, R0, UR4` + `@P0 EXIT`) is **dropped
  entirely** in the no-tid-guard variant.
* The 11-instruction shared suffix is **byte-identical except for one
  control-byte position**: at position [4] of the suffix (the 0xd09
  `UR_PIPE_FINAL` op), `b13` is `0x62` for k100_atom_add and `0x68`
  for multi_block_atomic. This is a scoreboard control-byte adjustment
  (rbar/wdep) reflecting the different scheduling context after the
  prelude is removed.

## 09.2 — Candidate exclusion table

PTX-level scan of every remaining STRUCTURAL atom kernel:

| kernel              | atom op    | %tid.x | bra | CAS | data form       | excluded by |
|---------------------|------------|:-:|:-:|:-:|----------------|--------------|
| **multi_block_atomic** | add.u32  | ✗ | ✗ | ✗ | reg=`mov.u32 1` | **TARGET (no exclusion)** |
| atomg_add           | add.u32    | ✗ | ✗ | ✗ | param-loaded reg | **param data** (excluded — different data routing) |
| r1_histogram8       | add.u32    | ✓ | ✗ | ✗ | literal `1`     | **has tid prelude** (already covered by AT06's tid path; that path was tightened in AT07 to exclude r1_histogram8 for its computed address) |
| w2_loop_atom_add    | add.u32    | ✓ | **✓** | ✗ | literal `1`     | **loop body** (no-tid gate excludes via tid-presence; loop gate excludes via bra) |
| w2_atom_and_reduce  | and.b32    | ✓ | ✗ | ✗ | computed `or %tid,K` | **has tid prelude AND computed data** |
| atom_or             | or.b32     | ✗ | ✗ | ✗ | reg=`mov.u32 0xFF` | **data value ≠ 1** (no-tid path admits only K=1) |
| k100_atom_cas32     | cas.b32    | ✓ | ✗ | **✓** | reg cmp + reg new | **CAS shape** (atom.cas has 2 data operands; opcode/template differs) |
| atom_cas64          | cas.b64    | ✗ | ✗ | **✓** | u64 cmp + u64 new | **CAS shape + 64-bit** |

Every kernel except multi_block_atomic is excluded by at least one
hard gate (tid-presence is the wrong direction for the no-tid path;
loop / CAS / non-1 data / param-data each block independently).

## 09.3 — Single-target confirmation

| chosen target       | why exact-shape safe | why single-target only |
|---------------------|----------------------|------------------------|
| **multi_block_atomic** | PTXAS byte-evidence: 11-instruction sequence byte-identical to AT06's imm_data_K1 suffix except for **one control byte** (b13=0x68 vs 0x62 at suffix position [4]). The data is `mov.u32 %r0, 1; atom.global.add.u32 %r, [%rd0], %r0` — effective K=1 immediate via constant-fold-aware lookback. | The 7 other STRUCTURAL atom kernels are blocked by independent gates: param-data (atomg_add), tid prelude (r1_histogram8, w2_loop_atom_add, w2_atom_and_reduce), looping (w2_loop_atom_add), non-1 data (atom_or), or CAS shape (k100_atom_cas32, atom_cas64). The no-tid-guard sibling cannot reach any of them without crossing into a separate subsystem. |

## Proof obligations carried into AT10

1. New `imm_data_K1_no_tid_guard` variant in `family_atom_ur.json`:
   11 instructions; byte-identical to imm_data_K1's suffix except
   position [4] b13 = 0x68 (one byte changed).
2. New isel admission shape: `atom.add.u32 <ld.param-base>, X` where:
   - `X` is `ImmOp(1)` OR a `RegOp` whose def (in the same bb) is a
     `mov.u32 ... ImmOp(1)`.
   - NO register in `ctx._reg_sr_source` is mapped to `SR_TID_X`
     (no tid prelude).
   - No basic block contains a `bra` instruction (no loops).
   - Atom op type matches the variant's ATOMG bytes (atom.add.u32 only
     for now; or/and/min/max with K=1 + no-tid would need their own
     overrides).
3. Pipeline dispatcher selects the new variant when
   `ctx._ur_activation_atom_no_tid_guard` is True.
4. AT06 imm_data_K1 path must remain intact; AT02 atom.xor/max/min
   baseline must remain intact.
5. AT07 lesson: GPU harness must run before AT10 declares clean —
   pytest+frontier alone cannot catch silent semantic regressions
   in template-replaced bodies.
