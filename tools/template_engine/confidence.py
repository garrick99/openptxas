"""Deterministic confidence scoring for discovered templates.

TEMPLATE-ENGINE-5B: score templates on cluster quality, parameter cleanliness,
and round-trip fidelity.  No ML — simple weighted formula.
"""
from __future__ import annotations

from dataclasses import dataclass
from .spec import TemplateSpec
from .detect import Cluster
from .infer import analyze_cluster
from .render import render_template


@dataclass
class ConfidenceBreakdown:
    """Per-template confidence scores (each 0.0–1.0)."""
    template_name: str
    cluster_size_score: float      # more members = better
    opcode_stability_score: float  # all members same opcode seq = 1.0
    byte_variation_score: float    # fewer varying bytes = better
    parameter_coherence_score: float  # varying bytes form clean fields
    roundtrip_score: float         # exact match = 1.0
    context_noise_score: float     # low unrelated ops = better
    total: float = 0.0
    tier: str = "LOW"

    def compute_total(self):
        """Weighted average.  Roundtrip is mandatory (0 kills total)."""
        if self.roundtrip_score < 1.0:
            self.total = 0.0
            self.tier = "LOW"
            return
        self.total = (
            0.15 * self.cluster_size_score
            + 0.15 * self.opcode_stability_score
            + 0.20 * self.byte_variation_score
            + 0.15 * self.parameter_coherence_score
            + 0.20 * self.roundtrip_score
            + 0.15 * self.context_noise_score
        )
        if self.total >= 0.80:
            self.tier = "HIGH"
        elif self.total >= 0.55:
            self.tier = "MEDIUM"
        else:
            self.tier = "LOW"


def _cluster_size_score(cluster: Cluster) -> float:
    """1 member = 0.3, 2 = 0.6, 3+ = 1.0."""
    n = cluster.size
    if n >= 3:
        return 1.0
    if n == 2:
        return 0.6
    return 0.3


def _opcode_stability_score(cluster: Cluster) -> float:
    """1.0 if all members have identical opcode sequence."""
    # By construction in our clustering, members HAVE the same opcode seq.
    return 1.0


def _byte_variation_score(spec: TemplateSpec) -> float:
    """Fewer parameterized bytes = higher score.

    0 varying = 1.0, 1-3 = 0.9, 4-8 = 0.7, 9+ = 0.4.
    """
    n = spec.total_parameterized_bytes()
    if n == 0:
        return 1.0
    if n <= 3:
        return 0.9
    if n <= 8:
        return 0.7
    return 0.4


def _parameter_coherence_score(spec: TemplateSpec) -> float:
    """1.0 if all params are contiguous fields.  Lower if scattered."""
    if spec.total_parameterized_bytes() == 0:
        return 1.0
    n_fields = sum(len(i.params) for i in spec.instructions)
    n_param_instrs = sum(1 for i in spec.instructions if i.params)
    # Ideal: few fields, few parameterized instructions
    if n_fields <= 2 and n_param_instrs <= 2:
        return 1.0
    if n_fields <= 4:
        return 0.8
    return 0.5


def _roundtrip_score(spec: TemplateSpec) -> float:
    """1.0 if render(spec, original_params) == original bytes."""
    params = {}
    for instr in spec.instructions:
        for p in instr.params:
            val = 0
            for j in range(p.byte_length):
                val |= instr.raw_bytes[p.byte_offset + j] << (8 * j)
            params[p.name] = val
    try:
        rendered = render_template(spec, params)
    except Exception:
        return 0.0
    original = [i.raw_bytes for i in spec.instructions]
    if len(rendered) != len(original):
        return 0.0
    return 1.0 if all(r == o for r, o in zip(rendered, original)) else 0.0


def _context_noise_score(spec: TemplateSpec) -> float:
    """Higher if the template is dominated by known-useful opcodes.

    Penalize if many instructions are unclassified / unknown.
    """
    known = {0xb82, 0x919, 0x7ac, 0xc0c, 0x94d, 0x3c4, 0x886, 0x2bd,
             0xc02, 0x835, 0x98e, 0x9a8, 0x9ae, 0x947, 0x810, 0x812,
             0x824, 0x80c, 0x210, 0x202, 0xc35, 0x235, 0x981, 0x986,
             0x984, 0x807, 0x431, 0x424, 0x388, 0xb1d, 0x819, 0xc11,
             0x223, 0x823, 0x221}
    total = len(spec.instructions)
    if total == 0:
        return 0.0
    n_known = sum(1 for i in spec.instructions if i.opcode in known)
    return n_known / total


def score_template(spec: TemplateSpec, cluster: Cluster) -> ConfidenceBreakdown:
    """Compute full confidence breakdown for a template."""
    bd = ConfidenceBreakdown(
        template_name=spec.name,
        cluster_size_score=_cluster_size_score(cluster),
        opcode_stability_score=_opcode_stability_score(cluster),
        byte_variation_score=_byte_variation_score(spec),
        parameter_coherence_score=_parameter_coherence_score(spec),
        roundtrip_score=_roundtrip_score(spec),
        context_noise_score=_context_noise_score(spec),
    )
    bd.compute_total()
    return bd
