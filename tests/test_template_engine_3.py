"""Tests for TEMPLATE-ENGINE-3: automatic parameter inference."""
from __future__ import annotations

import sys, os, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench_util import compile_ptxas
from tools.template_engine.detect import scan_cubins, find_atom_xor_cluster
from tools.template_engine.infer import analyze_cluster, generate_template_spec, generate_and_save
from tools.template_engine.render import render_template
import workbench


def _build_cubins():
    cubins = {}
    for name, entry in workbench.KERNELS.items():
        src = entry.get("ptx_inline")
        if src is None:
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
_CLUSTERS = None

def _get():
    global _CUBINS, _CLUSTERS
    if _CUBINS is None:
        _CUBINS = _build_cubins()
        _, _CLUSTERS = scan_cubins(_CUBINS)
    return _CUBINS, _CLUSTERS


class TestAnalysis:
    def test_multi_member_finds_variation(self):
        """Multi-member cluster should detect varying bytes."""
        _, clusters = _get()
        xor_c = find_atom_xor_cluster(clusters)
        assert xor_c.size >= 2, "Need multi-member cluster"
        analysis = analyze_cluster(xor_c)
        has_varying = False
        for instr_a in analysis:
            for ba in instr_a:
                if not ba.is_invariant:
                    has_varying = True
                    break
        assert has_varying, "Multi-member cluster should have varying bytes"

    def test_single_member_all_invariant(self):
        """Single-member cluster: all bytes invariant."""
        _, clusters = _get()
        for c in clusters:
            if c.size == 1:
                analysis = analyze_cluster(c)
                for instr_a in analysis:
                    for ba in instr_a:
                        assert ba.is_invariant, (
                            f"Single-member byte should be invariant: "
                            f"instr {ba.instr_index} byte {ba.position}"
                        )
                break  # test one is enough

    def test_ctrl_bytes_classified(self):
        """Bytes 13-15 should be classified as ctrl."""
        _, clusters = _get()
        xor_c = find_atom_xor_cluster(clusters)
        analysis = analyze_cluster(xor_c)
        for instr_a in analysis:
            for ba in instr_a:
                if ba.position >= 13:
                    assert ba.inferred_type == "ctrl"


class TestGeneration:
    def test_spec_generated_for_atom_xor(self):
        _, clusters = _get()
        xor_c = find_atom_xor_cluster(clusters)
        analysis = analyze_cluster(xor_c)
        spec = generate_template_spec(xor_c, analysis)
        assert spec.variant == "direct_sr"
        assert len(spec.instructions) > 0
        assert any(i.opcode == 0x98e for i in spec.instructions)

    def test_spec_json_serializable(self):
        _, clusters = _get()
        xor_c = find_atom_xor_cluster(clusters)
        analysis = analyze_cluster(xor_c)
        spec = generate_template_spec(xor_c, analysis)
        d = spec.to_dict()
        s = json.dumps(d)  # must not raise
        assert len(s) > 100


class TestRoundTrip:
    def test_variant_a_round_trip(self):
        _, clusters = _get()
        xor_c = find_atom_xor_cluster(clusters)
        analysis = analyze_cluster(xor_c)
        spec = generate_template_spec(xor_c, analysis)
        params = {}
        for instr in spec.instructions:
            for p in instr.params:
                val = 0
                for j in range(p.byte_length):
                    val |= instr.raw_bytes[p.byte_offset + j] << (8 * j)
                params[p.name] = val
        rendered = render_template(spec, params)
        original = [i.raw_bytes for i in spec.instructions]
        for i, (r, o) in enumerate(zip(rendered, original)):
            assert r == o, f"Mismatch at [{i}] {spec.instructions[i].role}"

    def test_variant_b_round_trip(self):
        _, clusters = _get()
        w2_c = None
        for c in clusters:
            for m in c.members:
                if m.kernel == "w2_atom_xor_reduce":
                    w2_c = c
                    break
            if w2_c:
                break
        assert w2_c is not None
        analysis = analyze_cluster(w2_c)
        spec = generate_template_spec(w2_c, analysis)
        rendered = render_template(spec, {})
        original = [i.raw_bytes for i in spec.instructions]
        for i, (r, o) in enumerate(zip(rendered, original)):
            assert r == o, f"Mismatch at [{i}] {spec.instructions[i].role}"
