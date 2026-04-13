"""Template spec data structures for PTXAS-faithful byte templates."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParamField:
    """A parameterized byte range within an instruction."""
    name: str
    byte_offset: int       # offset within the 16-byte instruction
    byte_length: int       # number of bytes (1-4)
    description: str = ""


@dataclass
class TemplateInstruction:
    """One instruction in a template."""
    index: int
    opcode: int
    role: str              # semantic label (e.g. "S2UR_UR0_TID")
    raw_bytes: bytes       # exact 16-byte instruction
    invariant: bool = True
    params: list[ParamField] = field(default_factory=list)


@dataclass
class TemplateSpec:
    """Complete template specification for an atom.xor variant."""
    name: str
    variant: str           # "direct_sr" or "tid_plus_constant"
    description: str
    instructions: list[TemplateInstruction] = field(default_factory=list)
    selector_condition: str = ""  # e.g. "ur_activation_add == 0"

    def total_parameterized_bytes(self) -> int:
        return sum(p.byte_length for instr in self.instructions for p in instr.params)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON/YAML output."""
        return {
            "name": self.name,
            "variant": self.variant,
            "description": self.description,
            "selector_condition": self.selector_condition,
            "total_parameterized_bytes": self.total_parameterized_bytes(),
            "instructions": [
                {
                    "index": i.index,
                    "opcode": f"0x{i.opcode:03x}",
                    "role": i.role,
                    "bytes": i.raw_bytes.hex(),
                    "invariant": i.invariant,
                    "params": [
                        {
                            "name": p.name,
                            "byte_offset": p.byte_offset,
                            "byte_length": p.byte_length,
                            "description": p.description,
                        }
                        for p in i.params
                    ] if i.params else [],
                }
                for i in self.instructions
            ],
        }
