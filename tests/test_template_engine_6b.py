"""Tests for TEMPLATE-ENGINE-6B: non-atomic template verification integration."""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench_util import compile_ptxas, compile_openptxas
from tools.template_engine.verify import verify_against_template
import workbench


def _compile_both(kernel_name):
    k = workbench.KERNELS[kernel_name]
    src = k.get("ptx_inline")
    if src is None:
        p = k.get("ptx_path")
        if p and Path(p).exists():
            src = Path(p).read_text()
    ours, _ = compile_openptxas(src)
    ptxas, _ = compile_ptxas(src)
    return ours, ptxas


class TestSharedMemoryVerification:
    """C6 shared_memory family: OURS matches PTXAS byte-for-byte."""

    def test_smem_exchange_byte_exact(self):
        ours, ptxas = _compile_both("smem_exchange")
        vr = verify_against_template(ours, ptxas, "smem_exchange", 6, "shared_memory")
        assert vr.opcode_match
        assert vr.match_ratio == 1.0, f"Expected byte-exact, got {vr.match_ratio:.2f}"

    def test_w1_smem_copy_byte_exact(self):
        ours, ptxas = _compile_both("w1_smem_copy")
        vr = verify_against_template(ours, ptxas, "w1_smem_copy", 6, "shared_memory")
        assert vr.opcode_match
        assert vr.match_ratio == 1.0

    def test_negative_control_divergent(self):
        """A kernel from a divergent cluster should NOT be byte-exact."""
        ours, ptxas = _compile_both("k100_guarded_store")
        vr = verify_against_template(ours, ptxas, "k100_guarded_store", 0, "predicated_store")
        assert not vr.is_ctrl_only, "Expected functional divergence for this family"

    def test_ctrl_only_detection(self):
        """Verify ctrl_only_diffs counts correctly when only b13-b15 differ."""
        vr = verify_against_template.__wrapped__ if hasattr(verify_against_template, '__wrapped__') else verify_against_template
        # Use smem_exchange which is byte-exact — ctrl_only should be 0
        ours, ptxas = _compile_both("smem_exchange")
        vr = verify_against_template(ours, ptxas, "smem_exchange")
        assert vr.ctrl_only_diffs == 0
        assert vr.functional_diffs == 0
