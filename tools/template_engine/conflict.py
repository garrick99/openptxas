"""Deterministic template conflict detection and resolution.

TEMPLATE-ENGINE-6C: when multiple templates could match the same codegen
context, resolve deterministically to prevent bad emissions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .spec import TemplateSpec
from .family import TemplateFamily, FamilyVariant
from .confidence import ConfidenceBreakdown


# ── Conflict types ────────────────────────────────────────────────────────

@dataclass
class ConflictCandidate:
    """A template that could match the current codegen context."""
    source: str           # "family:atom_ur/direct_sr" or "spec:cluster_4_..."
    confidence: float
    tier: str             # HIGH / MEDIUM / LOW
    is_integrated: bool   # actively used in codegen?
    match_quality: str    # "exact_variant", "family_fallback", "cluster_match"


@dataclass
class ConflictResolution:
    """Result of conflict resolution."""
    candidates: list[ConflictCandidate]
    winner: Optional[ConflictCandidate]
    reason: str
    is_ambiguous: bool

    @property
    def safe_to_emit(self) -> bool:
        return self.winner is not None and not self.is_ambiguous


# ── Precedence rules ─────────────────────────────────────────────────────
#
# 1. Exact variant match beats family fallback
# 2. Integrated families beat exploratory templates
# 3. Higher confidence beats lower confidence
# 4. Narrower selector beats broader selector
# 5. Ambiguous match → fallback (no emit)
#
# These are applied in order. First rule that distinguishes → winner.


def _precedence_key(c: ConflictCandidate) -> tuple:
    """Sort key: lower = higher priority."""
    quality_rank = {'exact_variant': 0, 'family_fallback': 1, 'cluster_match': 2}
    integrated_rank = 0 if c.is_integrated else 1
    confidence_rank = -c.confidence  # higher confidence = lower rank
    return (
        quality_rank.get(c.match_quality, 9),
        integrated_rank,
        confidence_rank,
    )


def resolve_conflicts(candidates: list[ConflictCandidate]) -> ConflictResolution:
    """Deterministically resolve which template to use.

    Returns a ConflictResolution with the winner or ambiguous status.
    """
    if not candidates:
        return ConflictResolution(
            candidates=[], winner=None,
            reason="no candidates", is_ambiguous=False)

    if len(candidates) == 1:
        return ConflictResolution(
            candidates=candidates, winner=candidates[0],
            reason="single candidate", is_ambiguous=False)

    # Sort by precedence
    sorted_c = sorted(candidates, key=_precedence_key)

    # Check if top two are distinguishable
    top = sorted_c[0]
    second = sorted_c[1]
    top_key = _precedence_key(top)
    second_key = _precedence_key(second)

    if top_key == second_key:
        # Ambiguous — fallback
        return ConflictResolution(
            candidates=candidates, winner=None,
            reason=f"ambiguous: {top.source} vs {second.source} have equal precedence",
            is_ambiguous=True)

    return ConflictResolution(
        candidates=candidates, winner=top,
        reason=f"precedence: {top.source} beats {second.source} "
               f"(quality={top.match_quality}, integrated={top.is_integrated}, "
               f"confidence={top.confidence:.2f})",
        is_ambiguous=False)


# ── Match checking ────────────────────────────────────────────────────────

def find_matching_candidates(
    ur_activation_add: int,
    families: list[TemplateFamily],
    standalone_specs: list[tuple[TemplateSpec, ConfidenceBreakdown]],
) -> list[ConflictCandidate]:
    """Find all templates that could match the current atom.xor context."""
    candidates = []

    # Check family variants
    for fam in families:
        variant = fam.select_variant({'ur_activation_add': ur_activation_add})
        if variant is not None:
            candidates.append(ConflictCandidate(
                source=f"family:{fam.name}/{variant.name}",
                confidence=0.95,  # family models are high confidence
                tier="HIGH",
                is_integrated=True,
                match_quality="exact_variant",
            ))

    # Check standalone specs
    for spec, bd in standalone_specs:
        if 'add == 0' in spec.selector_condition and ur_activation_add == 0:
            candidates.append(ConflictCandidate(
                source=f"spec:{spec.name}",
                confidence=bd.total,
                tier=bd.tier,
                is_integrated=False,
                match_quality="cluster_match",
            ))
        elif 'add != 0' in spec.selector_condition and ur_activation_add != 0:
            candidates.append(ConflictCandidate(
                source=f"spec:{spec.name}",
                confidence=bd.total,
                tier=bd.tier,
                is_integrated=False,
                match_quality="cluster_match",
            ))

    return candidates
