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

    def test_size_138_for_simple(self):
        """Simple kernels (no LDG+branch) get 138B capmerc."""
        cm = build_capmerc(num_gprs=10, text_size=384)
        assert len(cm) == 138

    def test_size_170_for_ldg_branch(self):
        """LDG+ISETP+branch kernels get 170B capmerc."""
        cm = build_capmerc(num_gprs=12, text_size=384,
                           has_ldg=True, has_isetp=True, has_branch=True)
        assert len(cm) == 170

    def test_170_has_dual_barriers(self):
        """170B path has two type02 barrier records (0x42 pre-EXIT + 0x62 body)."""
        cm = build_capmerc(num_gprs=12, text_size=384,
                           has_ldg=True, has_isetp=True, has_branch=True)
        barriers = []
        for i in range(16, len(cm) - 1):
            if cm[i] == 0x02 and cm[i+1] == 0x22:
                barriers.append(cm[i+6] if i+6 < len(cm) else None)
        assert 0x42 in barriers, "Missing pre-EXIT barrier (0x42)"
        assert 0x62 in barriers, "Missing body barrier (0x62)"

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

    def test_170_byte_exact_structure(self):
        """170B path body matches ptxas ground truth (excluding header/ctrl)."""
        cm = build_capmerc(num_gprs=12, text_size=384,
                           has_ldg=True, has_isetp=True, has_branch=True)
        # Prologue (always identical)
        assert cm[16:32] == bytes.fromhex('010b040af80004000000410000040000')
        # ALU record
        assert cm[32:48] == bytes.fromhex('010b040af80004000000810001020000')
        # Pre-EXIT barrier (0x42)
        assert cm[48:54] == bytes.fromhex('02220806fa00')
        assert cm[54] == 0x42
        # EXIT boundary marker
        assert cm[76] == 0x10
        # Body barrier (0x62)
        assert cm[86] == 0x62
        # STG record
        assert cm[112:128] == bytes.fromhex('010b0e0afa0005000000030139040000')
        # Trailer
        assert cm[168:170] == b'\x50\x05'
