"""Tests for TEMPLATE-ENGINE-5B: confidence scoring."""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench_util import compile_ptxas
from tools.template_engine.detect import scan_cubins, find_atom_xor_cluster
from tools.template_engine.infer import analyze_cluster, generate_template_spec
from tools.template_engine.confidence import score_template
import workbench

_CUBINS = None
_CLUSTERS = None

def _get():
    global _CUBINS, _CLUSTERS
    if _CUBINS is None:
        _CUBINS = {}
        for name, entry in workbench.KERNELS.items():
            src = entry.get("ptx_inline")
            if src is None:
                p = entry.get("ptx_path")
                if p and Path(p).exists(): src = Path(p).read_text()
            if src:
                try: _CUBINS[name], _ = compile_ptxas(src)
                except: pass
        _, _CLUSTERS = scan_cubins(_CUBINS)
    return _CUBINS, _CLUSTERS


class TestConfidence:
    def test_roundtrip_mandatory(self):
        """Templates with roundtrip=1.0 must not have total=0."""
        _, clusters = _get()
        xor_c = find_atom_xor_cluster(clusters)
        analysis = analyze_cluster(xor_c)
        spec = generate_template_spec(xor_c, analysis)
        bd = score_template(spec, xor_c)
        assert bd.roundtrip_score == 1.0
        assert bd.total > 0.0

    def test_all_selected_families_high(self):
        """All selected families should score HIGH or MEDIUM."""
        _, clusters = _get()
        for cid in [0, 1, 4, 6]:
            if cid < len(clusters):
                c = clusters[cid]
                analysis = analyze_cluster(c)
                spec = generate_template_spec(c, analysis)
                bd = score_template(spec, c)
                assert bd.tier in ("HIGH", "MEDIUM"), (
                    f"Cluster C{cid} scored {bd.tier} (total={bd.total:.2f})")

    def test_multi_member_scores_higher_cluster_size(self):
        """3+ member cluster should score higher than 1-member."""
        _, clusters = _get()
        multi = None
        single = None
        for c in clusters:
            if c.size >= 3 and multi is None:
                multi = c
            if c.size == 1 and single is None:
                single = c
            if multi and single:
                break
        if multi and single:
            a1 = analyze_cluster(multi)
            s1 = generate_template_spec(multi, a1)
            bd1 = score_template(s1, multi)
            a2 = analyze_cluster(single)
            s2 = generate_template_spec(single, a2)
            bd2 = score_template(s2, single)
            assert bd1.cluster_size_score > bd2.cluster_size_score

    def test_scores_deterministic(self):
        """Running twice produces identical scores."""
        _, clusters = _get()
        c = clusters[4]
        a = analyze_cluster(c)
        s = generate_template_spec(c, a)
        bd1 = score_template(s, c)
        bd2 = score_template(s, c)
        assert bd1.total == bd2.total
        assert bd1.tier == bd2.tier
