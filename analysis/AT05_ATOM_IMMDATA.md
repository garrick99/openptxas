# AT05 — Atom immediate-data variant clustering and target selection

## 05.1 — Immediate-data atom clustering

| kernel              | op       | data form        | tid guard? | loop? | HFMA2? | candidate? |
|---------------------|----------|------------------|:-:|:-:|:-:|:-:|
| k100_atom_add       | add.u32  | literal `1`      | ✓ | ✗ | ✗ | **TARGET** |
| multi_block_atomic  | add.u32  | reg=1 (folded)   | ✗ | ✗ | ✗ | deferred (no tid guard → different prefix) |
| r1_histogram8       | add.u32  | literal `1`      | ✓ | ✗ | **✓** | blocked (HFMA2 prohibited) |
| w2_loop_atom_add    | add.u32  | literal `1`      | ✓ | **✓** | ✗ | blocked (loop prohibited) |
| atom_or             | or.b32   | reg=0xFF (folded)| ✗ | ✗ | **✓** | blocked (HFMA2 + no tid guard) |

Only `k100_atom_add` matches the bounded shape: literal IMM data, tid-bounded
guard, no loop, no HFMA2.

## 05.2 — Critical PTXAS evidence: K=1 shortcut vs K≥2 materialization

Synthesized identical-PTX-shape kernels with varying immediate K and compared
PTXAS output:

| K (immediate) | PTXAS effective instructions | Distinguishing feature |
|---|---:|---|
| **1** | **15** | shortcut sequence — no IMM materialization opcode |
| 2     | 16 | IMM materialized via IMAD.I (0x824) at position [10] |
| 7     | 16 | same as K=2, IMM in IMAD.I bytes 4–7 |
| 0xff  | 16 | same |
| 0x12345678 | 16 | same — IMM written byte-for-byte at [10].b4–b7 |

**The K=1 shortcut and the K≥2 path are structurally different
templates.** PTXAS recognizes K=1 specially (likely because it can use
an "inc" semantic in the ATOMG operation field).

For corpus targets: every atom.add kernel in our 144-kernel corpus
uses **K=1**, so only the K=1 shortcut path is reachable for BYTE_EXACT
in this run. The K≥2 template is parameterizable but no current kernel
needs it.

## 05.3 — Baseline-vs-variant byte diff

Side-by-side: k100_atom_xor (BYTE_EXACT baseline, 16 instrs) vs k100_atom_add K=1 (target, 15 instrs).

| pos | atom.xor role | atom.add K=1 role | shape change |
|-----|--------------|-------------------|--------------|
| [0–4] | LDC, S2R(TID.X), LDCU(UR4), ISETP.UR, @P0 EXIT | identical | none |
| [5] | S2R **R2** = LANEID | S2R **R0** = LANEID | b2 register changes (R2→R0) |
| [6] | UMOV UR5=UR0 (data routing) | **0x886 R6** | template **drops** UMOV (no data routing for IMM) |
| [7] | 0x886 R4 | LDCU UR4=c[0x6b] (addr desc) | reordered |
| [8] | LDCU UR6=c[0x6b] | 0xd09 R5, R6 | reordered + new opcode 0xd09 |
| [9] | 0x2bd R4, R4 | LDC R2 = c[0xe0] | reordered |
| [10] | MOV.UR R5=UR5 | 0x2bd R7, R6 | template drops MOV.UR |
| [11] | ISETP.UR (flush) | ISETP.UR (flush) | identical role, different bytes |
| [12] | LDC R2 = c[0xe0] | **ATOMG.ADD** R0,R2,R5 | atom moved up; ATOMG modifier b9=0xe1 b10=0x12 b11=0x0c |
| [13] | ATOMG.XOR | EXIT | atom moved to [12] |
| [14] | EXIT | BRA | shifted |
| [15] | BRA | (NOP) | one fewer instruction |

This is **not** a per-op byte override on the same skeleton. It is a
**new template variant** with a different instruction count (15 vs 16),
different opcode sequence, and different reg/UR routing. It must be
implemented as a third entry in `family_atom_ur.json`, parallel to the
existing `direct_sr` and `tid_plus_constant` variants.

## 05.4 — Reuse compatibility table

| kernel              | reuse atom-UR template? | required edits | blocked? | why |
|---------------------|:-:|---|:-:|---|
| **k100_atom_add**   | ✓ | new JSON variant `imm_data_K1` (15 instrs); isel admission gate `atom.<add\|min\|max\|or\|and>.u32 <addr>, ImmOp(K=1)` with tid-guard shape; pipeline dispatcher selector | — | exact-shape match to PTXAS; only kernel meeting all gates |
| multi_block_atomic  | ✗ | no tid guard; PTXAS uses 11-instr prefix without S2R/ISETP.UR/EXIT | — | different prefix shape |
| r1_histogram8       | ✗ | PTXAS emits HFMA2 (0x431) | HFMA2 prohibited | out of scope |
| w2_loop_atom_add    | ✗ | loop body | loop prohibited | out of scope |
| atom_or             | ✗ | PTXAS emits HFMA2 + no tid guard | multiple blockers | out of scope |
| atom.add K≥2 hypothetical | (would need different variant) | new JSON variant `imm_data_general` (16 instrs) with IMM-materialization at [10] | — | no current corpus kernel uses K≥2 |

## 05.5 — Exact first target set

| chosen kernels | atom op | IMM | why chosen | why others deferred |
|---|---|---:|---|---|
| **k100_atom_add** | atom.global.add.u32 | 1 | exact-shape match to PTXAS K=1 shortcut sequence; tid-bounded guard; no loop; no HFMA2; 15-instr structure proven byte-for-byte against PTXAS at multiple synthesized K values | multi_block_atomic has different prefix (no tid guard); r1_histogram8/atom_or/k100_atom_cas* gated by HFMA2; w2_loop_atom_add gated by loop body; K≥2 path unused by current corpus |

## Proof obligations carried into AT06

1. New `imm_data_K1` variant entry in `family_atom_ur.json` with the
   15-instruction byte spec verified against PTXAS for k100_atom_add.
2. New isel admission shape: atom.add/min/max/or/and with ImmOp data
   AND tid-guard kernel pattern AND no loop. Default disposition is
   to fall through to the existing generic atom path.
3. Pipeline dispatcher reads a new ctx flag (e.g.
   `_ur_activation_atom_imm_K = 1`) to select the new variant.
4. AT02 atom.max/min and atom.xor baselines must remain BYTE_EXACT.
5. Excluded kernels must NOT silently pick up the new variant (verified
   by running them as controls in AT07).
