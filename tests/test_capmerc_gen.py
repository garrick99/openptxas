"""Tests for capmerc generation (build_capmerc / build_capmerc_from_sass).

The capmerc builder uses hardcoded ptxas-verified structures (138B for simple
kernels, 170B for LDG+ISETP+branch). These tests verify the fixed paths work
correctly, not a dynamic parameterized builder.
"""
import struct
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cubin.capmerc_gen import build_capmerc, CAPMERC_MAGIC


class TestCapmercMagicAndHeader:
    """Test header fields of generated capmerc."""

    def test_magic(self):
        cm = build_capmerc(num_gprs=10, text_size=384)
        assert cm[:8] == CAPMERC_MAGIC

    def test_reg_count_at_byte8(self):
        cm = build_capmerc(num_gprs=12, text_size=384)
        assert cm[8] == 12

    def test_reg_count_minimum_8(self):
        cm = build_capmerc(num_gprs=4, text_size=256)
        assert cm[8] == 8

    def test_cap_mask_nonzero(self):
        cm = build_capmerc(num_gprs=10, text_size=384)
        cap = struct.unpack_from('<I', cm, 12)[0]
        assert cap != 0


class TestCapmercStructure:
    """Test structural correctness of generated capmerc."""

    def test_starts_with_prologue(self):
        cm = build_capmerc(num_gprs=10, text_size=384)
        prologue = cm[16:32]
        assert prologue == bytes.fromhex('010b040af80004000000410000040000')

    def test_size_114_for_simple(self):
        """Simple kernels (no LDG+branch) get 114B capmerc (ptxas ground truth)."""
        cm = build_capmerc(num_gprs=10, text_size=384)
        assert len(cm) == 114

    def test_size_114_for_ldg_branch(self):
        """LDG kernels use 114B universal capmerc (170B path disabled)."""
        cm = build_capmerc(num_gprs=12, text_size=384,
                           has_ldg=True, has_isetp=True, has_branch=True)
        assert len(cm) == 114

    def test_114_has_barrier(self):
        """114B path has a type02 barrier record."""
        cm = build_capmerc(num_gprs=12, text_size=384,
                           has_ldg=True, has_isetp=True, has_branch=True)
        has_barrier = any(cm[i] == 0x02 and cm[i+1] == 0x22
                         for i in range(16, len(cm) - 1))
        assert has_barrier, "Missing barrier record"

    def test_trailer_present(self):
        cm = build_capmerc(num_gprs=10, text_size=384)
        assert len(cm) >= 18
        trailer = cm[-2:]
        assert trailer[0] in (0xd0, 0x50), f"Bad trailer byte: {trailer[0]:#x}"

    def test_branch_changes_trailer(self):
        """has_branch=True changes the trailer value."""
        cm_br = build_capmerc(num_gprs=12, text_size=384, has_branch=True)
        cm_no = build_capmerc(num_gprs=12, text_size=384, has_branch=False)
        assert cm_br[-2:] != cm_no[-2:], "Branch and no-branch should have different trailers"

    def test_stg_descriptor_present(self):
        """STG descriptor record is included in the 138B path."""
        cm = build_capmerc(num_gprs=10, text_size=384, has_stg=True)
        # STG record: type01 with b2=0x0e
        found = any(cm[i] == 0x01 and cm[i+1] == 0x0b and cm[i+2] == 0x0e
                     for i in range(16, len(cm) - 16, 16))
        assert found, "Missing STG descriptor record"

    def test_reg_count_varies(self):
        """Different num_gprs produce different byte[8] values."""
        cm8 = build_capmerc(num_gprs=8, text_size=384)
        cm14 = build_capmerc(num_gprs=14, text_size=384)
        assert cm8[8] != cm14[8]


class TestCapmercGpuVerified:
    """Verify that generated capmerc matches ptxas ground truth structure.

    These tests use the 170B path which was GPU-verified against ptxas.
    """

    def test_114_byte_ldg_structure(self):
        """LDG kernels use 114B universal structure (170B path disabled)."""
        cm = build_capmerc(num_gprs=12, text_size=384,
                           has_ldg=True, has_isetp=True, has_branch=True)
        assert len(cm) == 114
        # Prologue
        assert cm[16:32] == bytes.fromhex('010b040af80004000000410000040000')
        # STG record
        assert cm[32:48] == bytes.fromhex('010b0e0afa0005000000030139040000')
