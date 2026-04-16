"""Tests for TEMPLATE-ENGINE-7A: register-aware diff classification."""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench_util import compile_ptxas, compile_openptxas
from tools.template_engine.regdiff import diff_kernel, DiffClass
import workbench


def _diff(name):
    k = workbench.KERNELS[name]
    src = k.get("ptx_inline")
    if src is None:
        p = k.get("ptx_path")
        if p and Path(p).exists(): src = Path(p).read_text()
    o, _ = compile_openptxas(src)
    p, _ = compile_ptxas(src)
    return diff_kernel(o, p, name)


class TestClassification:
    def test_byte_exact_detected(self):
        r = _diff("smem_exchange")
        assert r.classification == DiffClass.BYTE_EXACT

    def test_structural_detected(self):
        # Pick a kernel that is reliably STRUCTURAL across the current
        # backend.  k100_atom_add was STRUCTURAL pre-AT06 but is now
        # BYTE_EXACT via the imm_data_K1 atom-UR template.  ilp_alu_addr
        # remains a stable STRUCTURAL representative (IMAD->HFMA2 family
        # is still deferred).
        r = _diff("ilp_alu_addr")
        assert r.classification == DiffClass.STRUCTURAL

    def test_atom_xor_exact(self):
        """atom.xor is template-driven, should be byte-exact."""
        r = _diff("k100_atom_xor")
        assert r.classification == DiffClass.BYTE_EXACT

    def test_instr_delta_computed(self):
        r = _diff("smem_exchange")
        assert r.instr_delta == 0

    def test_normalized_match_for_exact(self):
        r = _diff("smem_exchange")
        assert r.normalized_match is True
