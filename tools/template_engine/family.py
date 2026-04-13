"""Unified family model for multi-variant template specs.

TEMPLATE-ENGINE-6A: group related template variants into one family with
shared metadata, a selector rule per variant, and per-variant instruction
sequences.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .spec import TemplateSpec, TemplateInstruction, ParamField


@dataclass
class FamilyVariant:
    """One variant within a family."""
    name: str
    selector: str           # e.g. "ur_activation_add == 0"
    instructions: list[TemplateInstruction]
    params: list[ParamField] = field(default_factory=list)

    def total_parameterized_bytes(self) -> int:
        return sum(p.byte_length for i in self.instructions for p in i.params)


@dataclass
class TemplateFamily:
    """A family of related template variants sharing a common activation pattern."""
    name: str
    description: str
    shared_prefix_count: int     # how many leading instructions are byte-identical
    variants: list[FamilyVariant] = field(default_factory=list)

    def select_variant(self, params: dict) -> Optional[FamilyVariant]:
        """Select the appropriate variant based on runtime parameters."""
        add_val = params.get('ur_activation_add', 0)
        for v in self.variants:
            if 'add != 0' in v.selector and add_val != 0:
                return v
            if 'add == 0' in v.selector and add_val == 0:
                return v
        return None

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'shared_prefix_count': self.shared_prefix_count,
            'variants': [
                {
                    'name': v.name,
                    'selector': v.selector,
                    'total_parameterized_bytes': v.total_parameterized_bytes(),
                    'instructions': [
                        {
                            'index': i.index,
                            'opcode': f'0x{i.opcode:03x}',
                            'role': i.role,
                            'bytes': i.raw_bytes.hex(),
                            'invariant': i.invariant,
                            'params': [
                                {'name': p.name, 'byte_offset': p.byte_offset,
                                 'byte_length': p.byte_length, 'description': p.description}
                                for p in i.params
                            ] if i.params else [],
                        }
                        for i in v.instructions
                    ],
                }
                for v in self.variants
            ],
        }


def build_atom_ur_family(spec_dir: Path) -> TemplateFamily:
    """Build the atom_ur family from the existing generated specs."""
    # Load the two variant specs
    va_path = spec_dir / 'atom_ur_generalized_xor_min_max.json'
    vb_path = spec_dir / 'atom_xor_uniform_tid_plus_constant.json'

    def _load_instrs(path: Path) -> list[TemplateInstruction]:
        data = json.loads(path.read_text(encoding='utf-8'))
        instrs = []
        for si in data['instructions']:
            params = [ParamField(p['name'], p['byte_offset'], p['byte_length'],
                                 p.get('description', ''))
                      for p in si.get('params', [])]
            instrs.append(TemplateInstruction(
                si['index'], int(si['opcode'], 16), si['role'],
                bytes.fromhex(si['bytes']), si['invariant'], params))
        return instrs

    ia = _load_instrs(va_path)
    ib = _load_instrs(vb_path)

    # Count shared prefix
    shared = 0
    for a, b in zip(ia, ib):
        if a.raw_bytes == b.raw_bytes:
            shared += 1
        else:
            break

    # Collect per-variant params
    va_params = [p for i in ia for p in i.params]
    vb_params = [p for i in ib for p in i.params]

    family = TemplateFamily(
        name='atom_ur',
        description=(
            'Unified family for atom.global.{xor,min,max}.b32 with UR activation. '
            'Variant A: direct SR data. Variant B: SR + constant (UIADD).'
        ),
        shared_prefix_count=shared,
        variants=[
            FamilyVariant(
                name='direct_sr',
                selector='ur_activation_add == 0',
                instructions=ia,
                params=va_params,
            ),
            FamilyVariant(
                name='tid_plus_constant',
                selector='ur_activation_add != 0',
                instructions=ib,
                params=vb_params,
            ),
        ],
    )
    return family


def render_family_variant(variant: FamilyVariant, params: dict) -> list[bytes]:
    """Render one variant's instructions with parameter patching."""
    result = []
    for instr in variant.instructions:
        raw = bytearray(instr.raw_bytes)
        for p in instr.params:
            if p.name in params:
                val = params[p.name]
                for i in range(p.byte_length):
                    raw[p.byte_offset + i] = (val >> (8 * i)) & 0xFF
        result.append(bytes(raw))
    return result
