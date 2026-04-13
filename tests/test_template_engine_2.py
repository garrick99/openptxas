"""Tests for TEMPLATE-ENGINE-2: auto region detection and clustering."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench_util import compile_ptxas
from tools.template_engine.detect import (
    scan_cubins, find_atom_xor_cluster, detect_regions, _get_opcode, _NOP,
    _iter_text_sections,
)
import workbench


def _compile_all_ptxas():
    cubins = {}
    for name, entry in workbench.KERNELS.items():
        src = entry.get("ptx_inline")
        if src is None:
            from pathlib import Path
            p = entry.get("ptx_path")
            if p and Path(p).exists():
                src = Path(p).read_text()
        if src:
            try:
                cubins[name], _ = compile_ptxas(src)
            except Exception:
                pass
    return cubins


_CUBINS = None

def _get_cubins():
    global _CUBINS
    if _CUBINS is None:
        _CUBINS = _compile_all_ptxas()
    return _CUBINS


class TestRegionDetection:
    def test_detects_regions_across_corpus(self):
        cubins = _get_cubins()
        regions, clusters = scan_cubins(cubins)
        assert len(regions) >= 100, f"Expected 100+ regions, got {len(regions)}"

    def test_no_nops_in_regions(self):
        cubins = _get_cubins()
        regions, _ = scan_cubins(cubins)
        for r in regions:
            for raw in r.raw_bytes:
                assert _get_opcode(raw) != _NOP, f"NOP in region of {r.kernel}"


class TestClustering:
    def test_clusters_formed(self):
        cubins = _get_cubins()
        _, clusters = scan_cubins(cubins)
        assert len(clusters) >= 50, f"Expected 50+ clusters, got {len(clusters)}"

    def test_atom_xor_cluster_found(self):
        cubins = _get_cubins()
        _, clusters = scan_cubins(cubins)
        xor_c = find_atom_xor_cluster(clusters)
        assert xor_c is not None, "atom.xor cluster not found"
        assert xor_c.size >= 1

    def test_atom_xor_cluster_contains_known_kernel(self):
        cubins = _get_cubins()
        _, clusters = scan_cubins(cubins)
        xor_c = find_atom_xor_cluster(clusters)
        kernel_names = {m.kernel for m in xor_c.members}
        assert "k100_atom_xor" in kernel_names

    def test_w2_atom_xor_has_uiadd(self):
        """w2_atom_xor_reduce must be in a cluster containing UIADD (0x835)."""
        cubins = _get_cubins()
        _, clusters = scan_cubins(cubins)
        for c in clusters:
            for m in c.members:
                if m.kernel == "w2_atom_xor_reduce":
                    assert 0x835 in c.opcodes(), (
                        "w2_atom_xor_reduce cluster must contain UIADD"
                    )
                    return
        assert False, "w2_atom_xor_reduce not found in any cluster"
