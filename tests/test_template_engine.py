"""Tests for TEMPLATE-ENGINE-1: PTXAS template extraction and round-trip."""
from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench_util import compile_ptxas
from tools.template_engine.extract import extract_atom_xor_template
from tools.template_engine.render import render_template
import workbench


def _get_ptxas_cubin(kernel_name: str) -> bytes:
    k = workbench.KERNELS[kernel_name]
    src = k.get("ptx_inline")
    cubin, _ = compile_ptxas(src)
    return cubin


class TestExtract:
    def test_variant_a_detected(self):
        cubin = _get_ptxas_cubin("k100_atom_xor")
        spec = extract_atom_xor_template(cubin)
        assert spec.variant == "direct_sr"
        assert spec.total_parameterized_bytes() == 0

    def test_variant_b_detected(self):
        cubin = _get_ptxas_cubin("w2_atom_xor_reduce")
        spec = extract_atom_xor_template(cubin)
        assert spec.variant == "tid_plus_constant"
        assert spec.total_parameterized_bytes() == 3

    def test_variant_a_has_atomg(self):
        cubin = _get_ptxas_cubin("k100_atom_xor")
        spec = extract_atom_xor_template(cubin)
        opcodes = [i.opcode for i in spec.instructions]
        assert 0x98e in opcodes, "ATOMG_XOR must be in the template"

    def test_variant_b_has_uiadd(self):
        cubin = _get_ptxas_cubin("w2_atom_xor_reduce")
        spec = extract_atom_xor_template(cubin)
        opcodes = [i.opcode for i in spec.instructions]
        assert 0x835 in opcodes, "UIADD must be in the template"

    def test_no_nops_in_template(self):
        for name in ["k100_atom_xor", "w2_atom_xor_reduce"]:
            cubin = _get_ptxas_cubin(name)
            spec = extract_atom_xor_template(cubin)
            for instr in spec.instructions:
                assert instr.opcode != 0x918, f"NOP found in {name} template"


class TestRoundTrip:
    @pytest.mark.parametrize("kernel_name", ["k100_atom_xor", "w2_atom_xor_reduce"])
    def test_exact_round_trip(self, kernel_name):
        cubin = _get_ptxas_cubin(kernel_name)
        spec = extract_atom_xor_template(cubin)

        # Read original parameter values for rendering
        params = {}
        for instr in spec.instructions:
            for p in instr.params:
                val = 0
                for i in range(p.byte_length):
                    val |= instr.raw_bytes[p.byte_offset + i] << (8 * i)
                params[p.name] = val

        rendered = render_template(spec, params)
        original = [instr.raw_bytes for instr in spec.instructions]

        assert len(rendered) == len(original)
        for i, (r, o) in enumerate(zip(rendered, original)):
            assert r == o, (
                f"Mismatch at instruction [{i}] {spec.instructions[i].role}: "
                f"rendered={r.hex()} original={o.hex()}"
            )

    def test_variant_b_different_k(self):
        """Render Variant B with a different K value and verify only UIADD changes."""
        cubin = _get_ptxas_cubin("w2_atom_xor_reduce")
        spec = extract_atom_xor_template(cubin)
        original = [instr.raw_bytes for instr in spec.instructions]

        rendered_k42 = render_template(spec, {"add_imm_K": 42})

        changes = 0
        for i, (r, o) in enumerate(zip(rendered_k42, original)):
            if r != o:
                changes += 1
                assert spec.instructions[i].opcode == 0x835, (
                    f"Non-UIADD instruction [{i}] {spec.instructions[i].role} changed"
                )
        assert changes == 1, f"Expected exactly 1 changed instruction, got {changes}"
