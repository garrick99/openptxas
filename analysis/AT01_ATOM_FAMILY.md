# AT01 — Atom-family UR template boundary

## 01.1 — Atom clustering (13 kernels)

| kernel                    | PTX atom op              | class      | delta | harness |
|---------------------------|--------------------------|------------|------:|:-:|
| **k100_atom_xor**         | atom.global.xor.b32 tid  | BYTE_EXACT |    0 | Y |
| **w2_atom_xor_reduce**    | atom.global.xor.b32 tid+K| BYTE_EXACT |    0 | Y |
| atomg_add                 | atom.global.add.u32 param| STRUCTURAL |   −4 | Y |
| k100_atom_add             | atom.global.add.u32 imm  | STRUCTURAL |   −3 | Y |
| multi_block_atomic        | atom.global.add.u32 reg  | STRUCTURAL |   −1 | Y |
| r1_histogram8             | atom.global.add.u32 idx  | STRUCTURAL |   −1 | Y |
| w2_loop_atom_add          | atom.global.add.u32 (loop)| STRUCTURAL|   −1 | Y |
| w2_atom_and_reduce        | atom.global.and.b32 computed| STRUCTURAL | −5 | Y |
| **k100_atom_max**         | atom.global.max.u32 tid  | STRUCTURAL |   −5 | Y |
| **k100_atom_min**         | atom.global.min.u32 tid  | STRUCTURAL |   −5 | Y |
| atom_or                   | atom.global.or.b32 imm   | STRUCTURAL |   −4 | Y |
| k100_atom_cas32           | atom.global.cas.b32      | STRUCTURAL |   +1 | Y |
| atom_cas64                | atom.global.cas.b64      | STRUCTURAL |   −1 | Y |

## 01.2 — Existing template machinery

| component | location | role | used by |
|---|---|---|---|
| `family_atom_ur.json` | `tools/template_engine/generated/` | 16/17-instruction byte specification with 2 variants (direct_sr, tid_plus_constant) | atom.xor kernels |
| `_ur_activation_sr` | ctx flag set in `sass/isel.py:3325` | triggers the template engine to replace body_scheduled | atom.xor kernels only |
| `_ur_activation_add` | ctx flag | selects variant (==0 ⇒ direct_sr, !=0 ⇒ tid_plus_constant) | atom.xor kernels only |
| template dispatcher | `sass/pipeline.py:1193-1268` | loads JSON, selects variant, applies `add_imm_K` parameter, replaces body | atom.xor kernels only |
| atom.xor isel branch | `sass/isel.py:3275-3336` | sets `_ur_activation_sr`, emits placeholder MOV+ATOMG.XOR (wiped by template) | atom.xor |
| `encode_atomg_xor_u32` | `sass/encoding/sm_120_opcodes.py` | 0x98e encoder with b10/b11 operation field | atom.xor |

The template already parameterizes the two positions that differ per-op:
* instruction [6] `UMOV_UR5_UR0`: param `mode_b9_10` at byte_offset=9, byte_length=2
* instruction [13] `ATOMG_XOR_…`: param `operand_b10_11` at byte_offset=10, byte_length=2

But the template dispatcher (`pipeline.py:1224-1227`) only applies the
`add_imm_K` parameter. The `mode_b9_10` and `operand_b10_11` parameters
are declared in the JSON but currently ignored — they fall through to
the hex-string default, which is the atom.xor bytes.

## 01.3 — Reuse compatibility table

Direct byte-diff evidence from `compile_ptxas` on identical-shape
kernels (only the PTX atom op varies):

| position | role | atom.xor (BE) | atom.max | atom.min |
|---|---|---|---|---|
| [6] | UMOV_UR5_UR0 b9/b10 | 0x80 0x00 | **0x40 0x01** | **0x00 0x01** |
| [13] | ATOMG op b10/b11   | 0x92 0x0f | **0x12 0x0d** | **0x92 0x0c** |

All other 14 of 16 instruction bytes are identical byte-for-byte. That
is what makes atom.max.u32 + atom.min.u32 the narrowest safe expansion.

| variant                     | reuse template? | required edits | blocked? | why |
|-----------------------------|:-:|---|:-:|---|
| atom.xor.b32 (tid)          | ✓ baseline | — | — | already BYTE_EXACT |
| atom.xor.b32 (tid+K)        | ✓ baseline | — | — | already BYTE_EXACT |
| **atom.max.u32 (tid)**      | ✓ | 2 byte positions parameterized per-op; isel dispatch + ctx atom-op tag | — | exact shape match to atom.xor (verified) |
| **atom.min.u32 (tid)**      | ✓ | same as max | — | exact shape match |
| atom.add.u32 (imm)          | ? | data is NOT %tid.x; template UMOV uses UR0=tid | BLOCKED-for-now | semantic data shape differs |
| atom.add.u32 (param)        | ? | data is param-loaded; entirely different data path | BLOCKED | different lowering |
| atom.or.b32 (imm)           | ? | data is immediate constant | BLOCKED | data shape differs |
| atom.and.b32 (computed)     | ✗ | data is OR(tid, 0xFFFF0000) — computed | BLOCKED | data computation pre-atom |
| w2_loop_atom_add            | ✗ | has loop body with back-edge | BLOCKED | template requires no loops |
| r1_histogram8               | ✗ | address-indexed: [%rd0 + (tid & 7) * 4] | BLOCKED | address arithmetic past template shape |
| multi_block_atomic          | ✗ | no %tid.x guard | BLOCKED | template requires tid-bounded guard |
| atomg_add                   | ✗ | no %tid.x guard | BLOCKED | template requires tid-bounded guard |
| k100_atom_cas32             | ✗ | 2 data operands | BLOCKED | different ATOMG shape |
| atom_cas64                  | ✗ | 64-bit, 2 data operands | BLOCKED | different ATOMG shape |

## 01.4 — Exact first target set

| chosen variants | kernels | why chosen | why others deferred |
|---|---|---|---|
| **atom.max.u32 (tid) + atom.min.u32 (tid)** | `k100_atom_max`, `k100_atom_min` | PTX shape byte-identical to `k100_atom_xor` BYTE_EXACT baseline except for the atom op; PTXAS output differs in exactly 2 byte positions (already declared as template parameters); both positions are deterministic per-op (verified against ptxas); no looping, no address math, no multi-data-operand atom | atom.add variants need a different data-routing path (imm / param / reg); atom.or/and need data-computation pre-atom; atom.cas has 2 data ops; looped and address-indexed atoms require additional machinery outside the atom.xor template |

## Proof obligations carried into AT02

1. `k100_atom_max.u32` and `k100_atom_min.u32` must be recognized by
   isel as eligible for the atom-UR template.
2. The atom-op identity must be routed to the template dispatcher so
   the UMOV[6] and ATOMG[13] bytes can be patched per-op.
3. The existing atom.xor BYTE_EXACT behavior must be preserved (same
   bytes for `k100_atom_xor`, `w2_atom_xor_reduce`).
4. No non-target atom kernel may silently pick up the template
   (atom.add / atom.or / atom.and / cas / looped / address-indexed).

## GPU harness coverage for the target set

Both `k100_atom_max` and `k100_atom_min` have existing workbench
harnesses (`_harness_atom_min`, `_harness_atom_max` in
`workbench_expanded.py`). Pre-AT02 GPU state: both currently PASS
under the non-template isel path. That means GPU correctness must
continue to hold under the template path too.
