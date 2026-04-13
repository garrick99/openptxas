# TEMPLATE-ENGINE-1: Automated PTXAS Template Discovery

**Status:** PROVEN (round-trip exact match for both variants)

## 1. Problem Statement

Phase 4 proved that correct atom.global.xor.b32 emission on SM_120 requires
exact PTXAS instruction byte sequences.  Generic scheduling cannot reproduce
the UR pipeline activation context.  Hand-copying bytes works but does not
scale.

The template engine automates extraction, parameterization, and rendering of
PTXAS-faithful instruction templates.

## 2. Bounded Scope

TEMPLATE-ENGINE-1 supports **only**:
- atom.global.xor.b32 with uniform SR-derived data
- Two variants: direct SR and tid + constant
- Kernel signature: (.u64 p_out, .u32 n)

This is not a general cubin lifter or disassembler.

## 3. Architecture

```
tools/template_engine/
  spec.py      — data structures: TemplateSpec, TemplateInstruction, ParamField
  extract.py   — extract a TemplateSpec from a PTXAS cubin
  render.py    — render a TemplateSpec back to instruction bytes
  cli.py       — command-line interface for extract / roundtrip
```

### Extract
Parses the .text section of a PTXAS cubin, classifies each instruction by
opcode and operand fields, assigns semantic roles, and marks parameterized
byte ranges (currently only the UIADD immediate K).

### Render
Takes a TemplateSpec and a parameter dict, patches the parameterized byte
ranges, and produces the exact instruction byte sequence.

### Round-trip
Extract -> Render -> Compare.  Proves the engine captures all information
needed to reproduce the original PTXAS output.

## 4. Supported Domain

| Variant | Condition | Parameterized | Param bytes |
|---|---|---|---|
| direct_sr | `ur_activation_add == 0` | None | 0 |
| tid_plus_constant | `ur_activation_add != 0` | UIADD K (b4-b6) | 3 |

## 5. Round-trip Proof

```
k100_atom_xor:      variant=direct_sr          instrs=16 param_bytes=0 -> PASS
w2_atom_xor_reduce: variant=tid_plus_constant  instrs=17 param_bytes=3 -> PASS
```

Both variants produce an **exact byte-for-byte match** between the original
PTXAS instruction bytes and the rendered output.

## 6. Usage

```bash
# Extract template spec (JSON)
python -m tools.template_engine.cli extract k100_atom_xor

# Round-trip validation
python -m tools.template_engine.cli roundtrip
python -m tools.template_engine.cli roundtrip --kernel k100_atom_xor w2_atom_xor_reduce
```

## 7. Next Expansion Directions

1. **Broader opcode coverage**: extract templates for other UR-dependent paths
   (atom.and, atom.or with UR data) if the same activation pattern applies.
2. **Parameter discovery**: automatically identify which byte ranges differ
   between multiple PTXAS outputs for the same opcode pattern with varying
   params/constants.
3. **Integration hook**: replace the hardcoded template bytes in
   `sass/pipeline.py` with spec-driven rendering from this engine.
4. **Cross-kernel generalization**: feed multiple PTXAS kernels with the same
   atom.xor pattern but different param layouts, and infer which bytes are
   layout-dependent.
