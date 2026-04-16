# IP01 — ISETP.UR clustering and HARD BAIL

Evidence-driven clustering of every STRUCTURAL kernel missing ISETP.UR
(0xc0c). Result: **no bounded isel-level substitution target exists.**
Per the IP01 sprint rules, declare HARD BAIL and stop before IP02.

## 01.1 — Family clustering

All 9 STRUCTURAL kernels missing at least one ISETP.UR, with full
opcode-diff signatures (`- means PTXAS emits, OURS does not`):

| kernel                 | ours | ptxas | delta | missing                                        | extra                        |
|------------------------|-----:|------:|------:|-----------------------------------------------|------------------------------|
| atom_or                | 9    | 13    | −4    | 0x2bd, 0x3c4, **HFMA2**, 0x886, ATOMG.XOR, LDC, MOV.UR, **ISETP.UR** | LDCU, IADD3.I, ATOMG, IADD.64-UR |
| atomg_add              | 9    | 13    | −4    | 0x2bd, 0x2bf, 0x886, ATOMG.XOR, LDC, **ISETP.UR**, IMAD.UR | LDCU, ATOMG, IADD.64-UR |
| k100_atom_add          | 12   | 15    | −3    | 0x2bd, 0x886, S2R, ATOMG.XOR, LDC, **ISETP.UR**, 0xd09 | LDCU, IADD3.I, ATOMG, IADD.64-UR |
| k100_atom_max          | 11   | 16    | −5    | 0x2bd, 0x3c4, 0x886, S2R, ATOMG.XOR, LDC, MOV.UR, **ISETP.UR** | LDCU, ATOMG, IADD.64-UR |
| k100_atom_min          | 11   | 16    | −5    | 0x2bd, 0x3c4, 0x886, S2R, ATOMG.XOR, LDC, MOV.UR, **ISETP.UR** | LDCU, ATOMG, IADD.64-UR |
| multi_block_atomic     | 10   | 11    | −1    | 0x2bd, 0x886, ATOMG.XOR, **ISETP.UR**, 0xd09 | IADD3, IADD3.I×2, ATOMG |
| w2_atom_and_reduce     | 12   | 17    | −5    | 0x2bd, 0x3c4, 0x886, S2R, ATOMG.XOR, LDC, MOV.UR, **ISETP.UR** | LDCU, ATOMG, IADD.64-UR |
| w2_loop_atom_add       | 16   | 17    | −1    | 0x2bd, 0x886, S2R, ATOMG.XOR×3, LDC, **ISETP.UR**, 0xd09 | LDCU, ISETP.I, IADD3.I×3, BRA, ATOMG, IADD.64-UR |
| w2_pred_load           | 18   | 22    | −4    | IADD.64, **HFMA2**, LDCU×2, BSYNC, 0x945, BRA, **ISETP.UR** | ISETP.R, IADD3, IADD3.I, LDC |

## 01.2 — Delta=0 prioritization

| subcluster | count | delta=0? | opcode-only? | priority | reason |
|---|---:|:-:|:-:|---|---|
| (no subcluster qualifies) | 0 | — | — | — | no ISETP.UR-missing kernel has delta=0; smallest is −1 (`multi_block_atomic`, `w2_loop_atom_add`) and both have ≥4 other missing opcodes |

## 01.3 — UR-legality proof surface

Side-by-side kdiff evidence against the **existing BYTE_EXACT reference
path** `k100_atom_xor` (which already emits two ISETP.UR instances
byte-for-byte matching PTXAS):

```
k100_atom_xor (BYTE_EXACT, 16 instrs):
  [0]  LDC         [1]  S2R         [2]  LDCU         [3]  ISETP.UR (guard tid vs n)
  [4]  @P0 EXIT    [5]  S2R         [6]  MOV.UR       [7]  0x886 (UCGABAR arrive)
  [8]  LDCU        [9]  0x2bd       [10] MOV.UR       [11] ISETP.UR (atom-predicate)
  [12] LDC         [13] @P0 ATOMG.XOR  [14] EXIT    [15] BRA

k100_atom_add (STRUCTURAL, 12 instrs):
  [0]  LDC         [1]  S2R         [2-4] LDCU×3      [5]  ISETP.UR (guard)
  [6]  @P0 EXIT    [7]  IADD.64-UR  [8]  IADD3.I      [9]  NOP
  [10] ATOMG (generic, not ATOMG.XOR)                  [11] EXIT  [12] BRA

Difference: PTXAS lowers atom.add via the SAME "atom.xor-style" preamble
(0x886 + LDCU + 0x2bd + MOV.UR + ISETP.UR + @P0 ATOMG.XOR), with the
ATOMG.XOR opcode carrying an atom-add variant. OURS lowers atom.add via
a simpler IADD3.I + ATOMG generic path that never reaches the second
ISETP.UR.
```

**Required proof per pattern (IP01 sprint-spec form)**:

| pattern | required proof | already implemented? | missing piece |
|---|---|:-:|---|
| `atom.add/max/min/and/or` → preamble + ISETP.UR | port the `_ur_activation_sr` / `family_atom_ur.json` template from `atom.xor` to other atom ops | partial (atom.xor only) | **new atom-family template variants + isel dispatch for add/max/min/and/or** |
| `setp.cmp R, UR` from a simple predicate compare | proven-correct ISETP.UR encoder exists and is emitted in 122 kernels corpus-wide | ✓ | — (existing path is correct when invoked) |
| compound atom pattern with HFMA2 consumer | new HFMA2 subsystem | ✗ | **prohibited this run** |

The ISETP.UR encoder and emission path are already proven across 122
kernels. **The gap is not the encoder — it is the atom-lowering
preamble.** The missing ISETP.UR instance is produced as a side-effect
of that preamble (alongside 0x886, 0x2bd, MOV.UR, LDCU rescheduling),
not by a direct ISETP emission.

## 01.4 — First target: none qualifying

| chosen | count | reason |
|---|---:|---|
| **NONE** | 0 | Every ISETP.UR-missing kernel requires either (a) atom-family template expansion (8 of 9) — a subsystem-level change emitting 4–8 new opcodes (0x886, 0x2bd, MOV.UR, new LDCU placement, ATOMG.XOR variant) in addition to ISETP.UR; or (b) HFMA2 subsystem (w2_pred_load) — prohibited this run. The "isel-level ISETP.UR substitution" described in the IP01 sprint spec does not apply: ISETP.UR is already emitted correctly by the existing isel path in 122 kernels. The 9 missing instances are produced by the atom-preamble template, not by isel's ISETP selection. |

## HARD BAIL declaration

Per the IP01 spec:

> If IP01 fails: HARD BAIL
> Hard bail means: stop immediately; revert unsafe changes if needed;
> report exact blocker; do not continue.

**Status**:
- Stop: IP02/IP03/IP04 not executed.
- Revert: not needed (IP01 is analysis-only; no code changes).
- Blocker: the ISETP.UR gap is a **disguised atom-family template gap**,
  not an isel-level opcode-substitution gap. Every bounded-scope rule
  from the IP01 sprint (no family drift, no subsystem expansion, no
  post-scheduling rewrite, exact-shape only) would be violated by any
  fix for the actual underlying pattern.

**Pre-existing state preserved**:
- pytest 865/865 green (verified post-IP01 analysis)
- frontier unchanged: BYTE_EXACT 46, STRUCTURAL 98, MIXED 0, errors 0
- UI03 UIADD fixes retained (commits a1a05ea, fc6c617 intact)
- MP02 multi-pred fixes retained

## Honest roadmap revision

The UI04 recommendation ("pause UIADD for ISETP.UR subsystem") was made
on the ranking that ISETP.UR had 9 missing instances across 5+
subclusters. The ranking was correct on the count, but wrong on the
**subsystem boundary** — ISETP.UR is not its own subsystem within the
OpenPTXas isel. It is a byproduct of the atom-family template, and
closing the gap requires **atom-family template expansion**, which is a
different (and larger) subsystem than IP01's stated scope.

**Correct next move** (to be picked in a future sprint, with explicit
scoping rules for atom-template expansion, not in this one):

* Either: scope an atom-template expansion sprint (add variants for
  atom.add / atom.max / atom.min / atom.and to `family_atom_ur.json`
  and wire the `_ur_activation_sr` trigger in the corresponding isel
  paths).
* Or: scope an IMAD/IMAD.WIDE coordinated subsystem that unlocks the
  UIADD gaps UI04 identified as blocked by IMAD coordination.
* Or: resume SHF harvest (was deferred in this run's prohibition list
  but is a standing orthogonal opportunity).

All three are larger than IP01's bounded-slice intent. A future sprint
chain should explicitly pick one with scoping rules appropriate to it.
