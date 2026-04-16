# AT07 — Validation matrix for atom imm-data template (commit fixed in this phase)

## 07.0 — Adversarial finding during AT07 validation

The first AT06 admission gate was too loose. AT07's GPU harness sweep
exposed 2 regressions:
- `r1_histogram8`: address operand `[%rd2]` is computed from tid via
  `shl.b64` + `add.u64`. The hardcoded template assumes the address is
  a simple ld.param.u64 pointer; the per-thread offset was lost,
  producing wrong GPU output.
- `w2_loop_atom_add`: the atom call sits inside a loop body. The
  template emits exactly one ATOMG; the loop's repeated atomic
  semantics was silently collapsed.

Both kernels passed pytest and produced compilable cubins; only the GPU
correctness sweep caught them. This is the documented purpose of the
AT07 GPU validation step.

**Fix applied (within AT07)**: the helper `_try_atom_ur_imm_K1_template`
now also requires:
1. `_xa.base in ctx._ur_params AND not in ctx._gpr_written` — address
   must be a simple ld.param.u64 pointer with no GPR-modification
   chain.
2. The function (any basic block) contains no `bra` instruction —
   excludes loop bodies entirely.

After the fix, both regressions disappear and the only kernel admitted
is the original target `k100_atom_add`.

## 07.1 — Validation matrix (after fix)

| target          | class before AT05 | class after AT07 | residual missing | residual extra | GPU |
|-----------------|-------------------|------------------|------------------|----------------|-----|
| k100_atom_add   | STRUCTURAL        | **BYTE_EXACT**   | (none)           | (none)         | PASS |

## 07.2 — Controls

| control kernel       | expected   | actual     | unchanged? | GPU  | role |
|----------------------|------------|------------|:-:|------|------|
| k100_atom_xor        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS | atom.xor template baseline |
| k100_atom_max        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS | AT02 max baseline |
| k100_atom_min        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS | AT02 min baseline |
| multi_block_atomic   | STRUCTURAL | STRUCTURAL | ✓ | PASS | no-tid-guard (must NOT pick up template) |
| **r1_histogram8**    | STRUCTURAL | STRUCTURAL | ✓ | **PASS** | computed-address atom (must NOT pick up template) |
| **w2_loop_atom_add** | STRUCTURAL | STRUCTURAL | ✓ | **PASS** | looped atom (must NOT pick up template) |
| atom_or              | STRUCTURAL | STRUCTURAL | ✓ | PASS | HFMA2 atom (must NOT pick up template) |
| k100_atom_cas32      | STRUCTURAL | STRUCTURAL | ✓ | PASS | CAS shape |
| atomg_add            | STRUCTURAL | STRUCTURAL | ✓ | PASS | param-data |
| w2_atom_and_reduce   | STRUCTURAL | STRUCTURAL | ✓ | RUN_EXC (pre-existing baseline) | computed-data |
| k100_guarded_store   | BYTE_EXACT | BYTE_EXACT | ✓ | PASS | unrelated single-pred |
| smem_exchange        | BYTE_EXACT | BYTE_EXACT | ✓ | PASS | unrelated UIADD |
| k100_pred_arith      | STRUCTURAL | STRUCTURAL | ✓ | PASS | MP02 multi-pred control |
| ilp_alu_addr         | STRUCTURAL | STRUCTURAL | ✓ | PASS | unrelated STRUCTURAL |

`r1_histogram8` and `w2_loop_atom_add` were briefly broken under the
loose AT06 gate but are now back to baseline — verified PASS again
under the tightened AT07 gate.

## 07.3 — Residual blocker table

| target          | remaining blockers after AT07 | next blocker type |
|-----------------|-------------------------------|-------------------|
| k100_atom_add   | none — BYTE_EXACT achieved    | — |

## Full validation summary

| stage | result |
|---|---|
| pytest | **865 / 865 green** (~80 s) |
| GPU harness | **126 PASS / 11 FAIL / 7 RUN_EXC / 0 COMPILE_FAIL** — identical to MP03 baseline (no new regressions; r1_histogram8 and w2_loop_atom_add restored to PASS by the AT07 gate-tightening) |
| workbench frontier | **49 BYTE_EXACT / 95 STRUCTURAL / 0 MIXED / 0 errors** (+1 vs AT04 baseline 48/96/0) |
| targeted kdiff | k100_atom_add byte-for-byte identical to PTXAS over `.text` |
| MP02 / UI03 / AT02 fixes | all unchanged |
