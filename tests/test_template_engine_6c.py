"""Tests for TEMPLATE-ENGINE-6C: template conflict detection and resolution."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.template_engine.conflict import (
    ConflictCandidate, resolve_conflicts, find_matching_candidates,
)
from tools.template_engine.family import build_atom_ur_family
from tools.template_engine.confidence import ConfidenceBreakdown
from tools.template_engine.spec import TemplateSpec
from pathlib import Path


class TestResolution:
    def test_single_candidate_wins(self):
        c = ConflictCandidate("family:atom_ur/direct_sr", 0.95, "HIGH", True, "exact_variant")
        r = resolve_conflicts([c])
        assert r.winner == c
        assert not r.is_ambiguous
        assert r.safe_to_emit

    def test_no_candidates_safe(self):
        r = resolve_conflicts([])
        assert r.winner is None
        assert not r.is_ambiguous
        assert not r.safe_to_emit

    def test_exact_variant_beats_cluster_match(self):
        fam = ConflictCandidate("family:atom_ur/direct_sr", 0.95, "HIGH", True, "exact_variant")
        cluster = ConflictCandidate("spec:cluster_4", 0.94, "HIGH", False, "cluster_match")
        r = resolve_conflicts([cluster, fam])
        assert r.winner.source == "family:atom_ur/direct_sr"
        assert not r.is_ambiguous

    def test_integrated_beats_exploratory(self):
        integ = ConflictCandidate("family:atom_ur/direct_sr", 0.90, "HIGH", True, "family_fallback")
        explor = ConflictCandidate("spec:experimental", 0.90, "HIGH", False, "family_fallback")
        r = resolve_conflicts([explor, integ])
        assert r.winner.source == "family:atom_ur/direct_sr"

    def test_higher_confidence_breaks_tie(self):
        a = ConflictCandidate("spec:a", 0.95, "HIGH", False, "cluster_match")
        b = ConflictCandidate("spec:b", 0.90, "HIGH", False, "cluster_match")
        r = resolve_conflicts([b, a])
        assert r.winner.source == "spec:a"

    def test_ambiguous_equal_precedence(self):
        a = ConflictCandidate("spec:a", 0.95, "HIGH", False, "cluster_match")
        b = ConflictCandidate("spec:b", 0.95, "HIGH", False, "cluster_match")
        r = resolve_conflicts([a, b])
        assert r.is_ambiguous
        assert not r.safe_to_emit


class TestMatchFinding:
    def test_family_match_direct_sr(self):
        spec_dir = Path(__file__).resolve().parent.parent / 'tools' / 'template_engine' / 'generated'
        family = build_atom_ur_family(spec_dir)
        candidates = find_matching_candidates(0, [family], [])
        assert len(candidates) >= 1
        assert any(c.match_quality == "exact_variant" for c in candidates)

    def test_family_match_tid_plus_constant(self):
        spec_dir = Path(__file__).resolve().parent.parent / 'tools' / 'template_engine' / 'generated'
        family = build_atom_ur_family(spec_dir)
        candidates = find_matching_candidates(1, [family], [])
        assert len(candidates) >= 1

    def test_resolution_with_family_and_standalone(self):
        """Family variant should beat standalone spec."""
        spec_dir = Path(__file__).resolve().parent.parent / 'tools' / 'template_engine' / 'generated'
        family = build_atom_ur_family(spec_dir)

        # Create a fake standalone spec match
        fake_spec = TemplateSpec("fake", "direct_sr", "test",
                                 selector_condition="ur_activation_add == 0")
        fake_bd = ConfidenceBreakdown("fake", 0.9, 1.0, 0.9, 1.0, 1.0, 1.0)
        fake_bd.compute_total()

        candidates = find_matching_candidates(0, [family], [(fake_spec, fake_bd)])
        r = resolve_conflicts(candidates)
        assert r.safe_to_emit
        assert "family:" in r.winner.source
