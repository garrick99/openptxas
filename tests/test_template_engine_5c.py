"""Tests for TEMPLATE-ENGINE-5C: cross-kernel template generalization."""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench_util import compile_ptxas
from tools.template_engine.detect import scan_cubins, find_atom_xor_cluster
from tools.template_engine.infer import analyze_cluster, generate_template_spec
from tools.template_engine.render import render_template
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


class TestCrossKernelGeneralization:
    def test_all_members_exact_match(self):
        """One generalized spec must reconstruct every member exactly."""
        _, clusters = _get()
        xor_c = find_atom_xor_cluster(clusters)
        analysis = analyze_cluster(xor_c)
        spec = generate_template_spec(xor_c, analysis)

        for mi, member in enumerate(xor_c.members):
            params = {}
            for instr in spec.instructions:
                for p in instr.params:
                    val = 0
                    for j in range(p.byte_length):
                        val |= member.raw_bytes[instr.index][p.byte_offset + j] << (8 * j)
                    params[p.name] = val

            rendered = render_template(spec, params)
            for idx in range(len(rendered)):
                assert rendered[idx] == member.raw_bytes[idx], (
                    f"{member.kernel} instr[{idx}] mismatch: "
                    f"rendered={rendered[idx].hex()} "
                    f"original={member.raw_bytes[idx].hex()}"
                )

    def test_varying_fields_are_semantic(self):
        """The varying fields should be in UMOV and ATOMG (operation-dependent)."""
        _, clusters = _get()
        xor_c = find_atom_xor_cluster(clusters)
        analysis = analyze_cluster(xor_c)
        spec = generate_template_spec(xor_c, analysis)

        param_opcodes = set()
        for instr in spec.instructions:
            if instr.params:
                param_opcodes.add(instr.opcode)

        # UMOV (0x3c4) and ATOMG (0x98e) should be the parameterized ones
        assert 0x3c4 in param_opcodes, "UMOV should have params"
        assert 0x98e in param_opcodes, "ATOMG should have params"

    def test_covers_three_operations(self):
        """Cluster must contain xor, min, and max."""
        _, clusters = _get()
        xor_c = find_atom_xor_cluster(clusters)
        names = {m.kernel for m in xor_c.members}
        assert "k100_atom_xor" in names
        assert "k100_atom_min" in names
        assert "k100_atom_max" in names
