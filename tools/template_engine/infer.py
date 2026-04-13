"""Automatic parameter inference for template clusters.

TEMPLATE-ENGINE-3: compare cluster members byte-by-byte, classify invariant
vs parameterized fields, generate template specs automatically.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .spec import TemplateSpec, TemplateInstruction, ParamField
from .detect import Cluster, Region


@dataclass
class ByteAnalysis:
    """Per-byte analysis across cluster members."""
    position: int          # byte offset within the instruction (0-15)
    instr_index: int       # instruction index within the region
    values: list[int]      # observed values across members
    is_invariant: bool     # True if all values identical
    inferred_type: str = ""  # "ctrl", "immediate", "register", "unknown"
    confidence: float = 1.0


def _classify_byte_position(instr_opc: int, byte_pos: int) -> str:
    """Classify what a byte position encodes based on opcode family."""
    if byte_pos >= 13:
        return "ctrl"  # bytes 13-15 are always control/scheduling
    if byte_pos <= 1:
        return "opcode"
    # Common register positions
    if byte_pos == 2:
        return "dest_reg"
    if byte_pos == 3:
        return "src_a"
    if byte_pos == 4:
        if instr_opc == 0x835:  # UIADD
            return "immediate"
        return "src_b_or_ur"
    if byte_pos in (5, 6):
        if instr_opc == 0x835:
            return "immediate"
        if instr_opc in (0x7ac, 0xb82):  # LDCU, S2R — constant buffer offset
            return "cbank_offset"
        return "operand"
    if byte_pos == 8:
        return "ur_desc_or_operand"
    if byte_pos == 9:
        if instr_opc == 0x919:  # S2UR — special register code
            return "sr_code"
        return "mode"
    return "operand"


def analyze_cluster(cluster: Cluster) -> list[list[ByteAnalysis]]:
    """Analyze byte-level variation across cluster members.

    Returns a list of instruction analyses, each containing 16 ByteAnalysis
    entries (one per byte).
    """
    if cluster.size < 2:
        # Single-member cluster: all bytes invariant by definition
        r = cluster.representative
        result = []
        for i, raw in enumerate(r.raw_bytes):
            opc = int.from_bytes(raw[:2], "little") & 0xFFF
            instr_analysis = []
            for b in range(16):
                ba = ByteAnalysis(
                    position=b,
                    instr_index=i,
                    values=[raw[b]],
                    is_invariant=True,
                    inferred_type=_classify_byte_position(opc, b),
                )
                instr_analysis.append(ba)
            result.append(instr_analysis)
        return result

    # Multi-member: compare byte by byte
    n_instrs = min(len(m.raw_bytes) for m in cluster.members)
    result = []
    for i in range(n_instrs):
        opc = int.from_bytes(cluster.representative.raw_bytes[i][:2], "little") & 0xFFF
        instr_analysis = []
        for b in range(16):
            values = [m.raw_bytes[i][b] for m in cluster.members]
            is_inv = len(set(values)) == 1
            ba = ByteAnalysis(
                position=b,
                instr_index=i,
                values=values,
                is_invariant=is_inv,
                inferred_type=_classify_byte_position(opc, b),
                confidence=1.0 if is_inv else 0.8,
            )
            instr_analysis.append(ba)
        result.append(instr_analysis)
    return result


def _group_varying_bytes(instr_analysis: list[ByteAnalysis]) -> list[ParamField]:
    """Group adjacent varying bytes into parameter fields."""
    params = []
    i = 0
    while i < len(instr_analysis):
        ba = instr_analysis[i]
        if ba.is_invariant or ba.inferred_type in ("ctrl", "opcode"):
            i += 1
            continue
        # Start a new param field
        start = i
        ptype = ba.inferred_type
        while (i < len(instr_analysis)
               and not instr_analysis[i].is_invariant
               and instr_analysis[i].inferred_type not in ("ctrl", "opcode")):
            i += 1
        length = i - start
        name = f"{ptype}_b{start}" if length == 1 else f"{ptype}_b{start}_{start+length-1}"
        params.append(ParamField(
            name=name,
            byte_offset=start,
            byte_length=length,
            description=f"Varying {ptype} field at bytes [{start}:{start+length})",
        ))
    return params


def generate_template_spec(cluster: Cluster, analysis: list[list[ByteAnalysis]],
                           name: Optional[str] = None) -> TemplateSpec:
    """Generate a TemplateSpec from cluster analysis."""
    rep = cluster.representative
    has_uiadd = 0x835 in rep.opcodes
    variant = "tid_plus_constant" if has_uiadd else "direct_sr"
    has_98e = 0x98e in rep.opcodes

    if name is None:
        if has_98e:
            name = f"atomg_cluster_{cluster.cluster_id}_{variant}"
        else:
            opcodes_short = "_".join(f"{o:03x}" for o in rep.opcodes[:4])
            name = f"cluster_{cluster.cluster_id}_{opcodes_short}"

    instrs = []
    for i, raw in enumerate(rep.raw_bytes):
        opc = int.from_bytes(raw[:2], "little") & 0xFFF
        # Derive role
        from .extract import _refine_role
        role = _refine_role(opc, raw, i, has_uiadd)

        # Get params from analysis
        params = _group_varying_bytes(analysis[i]) if i < len(analysis) else []
        is_inv = len(params) == 0

        instrs.append(TemplateInstruction(
            index=i,
            opcode=opc,
            role=role,
            raw_bytes=raw,
            invariant=is_inv,
            params=params,
        ))

    selector = ""
    if has_98e:
        selector = "ur_activation_add != 0" if has_uiadd else "ur_activation_add == 0"

    return TemplateSpec(
        name=name,
        variant=variant,
        description=f"Auto-generated from cluster C{cluster.cluster_id} ({cluster.size} members)",
        instructions=instrs,
        selector_condition=selector,
    )


def generate_and_save(cluster: Cluster, output_dir: str | Path,
                      name: Optional[str] = None) -> TemplateSpec:
    """Analyze cluster, generate spec, and save to JSON."""
    analysis = analyze_cluster(cluster)
    spec = generate_template_spec(cluster, analysis, name=name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{spec.name}.json"
    path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    return spec
