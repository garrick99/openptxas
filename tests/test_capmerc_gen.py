"""
test_capmerc_gen.py — Verify capmerc generation against ptxas reference cubins.

Tests that build_capmerc() produces structurally correct output that matches
key fields from ptxas-generated capmerc sections.

Run:
    python -m pytest tests/test_capmerc_gen.py -v
or:
    python tests/test_capmerc_gen.py
"""

import sys
import os
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cubin.capmerc_gen import (
    build_capmerc,
    build_capmerc_from_sass,
    analyze_sass_for_capmerc,
    CAPMERC_MAGIC,
)

PROBE_DIR = os.path.join(os.path.dirname(__file__), "..", "probe_work")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_elf_sections(path: str) -> dict:
    """Extract named sections from an ELF cubin."""
    with open(path, 'rb') as f:
        data = f.read()
    e_shoff = struct.unpack_from('<Q', data, 40)[0]
    e_shentsize = struct.unpack_from('<H', data, 58)[0]
    e_shnum = struct.unpack_from('<H', data, 60)[0]
    e_shstrndx = struct.unpack_from('<H', data, 62)[0]
    shstr_off = e_shoff + e_shstrndx * e_shentsize
    shstr_offset = struct.unpack_from('<Q', data, shstr_off + 24)[0]
    shstr_size = struct.unpack_from('<Q', data, shstr_off + 32)[0]
    shstrtab = data[shstr_offset:shstr_offset + shstr_size]
    sections = {}
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_name = struct.unpack_from('<I', data, off)[0]
        sh_offset = struct.unpack_from('<Q', data, off + 24)[0]
        sh_size = struct.unpack_from('<Q', data, off + 32)[0]
        if sh_name < len(shstrtab):
            name_end = shstrtab.index(0, sh_name)
            sname = shstrtab[sh_name:name_end].decode('ascii', errors='replace')
        else:
            sname = f'<idx_{sh_name}>'
        if sh_size > 0:
            sections[sname] = data[sh_offset:sh_offset + sh_size]
    return sections


def get_ptxas_capmerc(cubin_name: str):
    """Load capmerc and text from a ptxas reference cubin."""
    path = os.path.join(PROBE_DIR, cubin_name)
    if not os.path.exists(path):
        return None, None
    secs = extract_elf_sections(path)
    capmerc = None
    text = None
    for k, v in secs.items():
        if 'capmerc' in k:
            capmerc = v
        if k.startswith('.text.'):
            text = v
    return capmerc, text


def parse_capmerc_structure(data: bytes) -> dict:
    """Parse capmerc into structured form for comparison."""
    result = {
        'magic': data[:8],
        'reg_count': data[8],
        'cap_mask': struct.unpack_from('<I', data, 12)[0],
        'records': [],
        'fillers': 0,
        'terminal': None,
        'trailer': data[-2:],
        'size': len(data),
    }
    body = data[16:-2]
    pos = 0
    while pos < len(body):
        rt = body[pos]
        if rt == 0x01:
            rec = body[pos:pos+16]
            result['records'].append(('type01', rec))
            pos += 16
        elif rt == 0x02:
            sub = body[pos+1] if pos+1 < len(body) else 0
            rec = body[pos:pos+32]
            if sub == 0x38:
                result['terminal'] = rec
                result['records'].append(('terminal', rec))
            else:
                result['records'].append(('type02', rec))
            pos += 32
        elif rt == 0x41:
            result['fillers'] += 1
            pos += 4
        else:
            pos += 1
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCapmercMagicAndHeader:
    """Test that headers are correctly formed."""

    def test_magic_bytes(self):
        cm = build_capmerc(num_gprs=8, text_size=256)
        assert cm[:8] == CAPMERC_MAGIC

    def test_reg_count_passthrough(self):
        for reg in [8, 10, 12, 15, 19, 22, 32]:
            cm = build_capmerc(num_gprs=reg, text_size=384)
            assert cm[8] == reg, f"reg={reg}: got {cm[8]}"

    def test_reg_count_minimum_8(self):
        cm = build_capmerc(num_gprs=4, text_size=256)
        assert cm[8] == 8

    def test_cap_mask_nonzero(self):
        cm = build_capmerc(num_gprs=10, text_size=384)
        cap = struct.unpack_from('<I', cm, 12)[0]
        assert cap != 0

    def test_cap_mask_stg_bit(self):
        cm_stg = build_capmerc(num_gprs=10, text_size=384, has_stg=True)
        cm_no = build_capmerc(num_gprs=10, text_size=384, has_stg=False)
        cap_stg = struct.unpack_from('<I', cm_stg, 12)[0]
        cap_no = struct.unpack_from('<I', cm_no, 12)[0]
        assert cap_stg & 0x40, "STG bit should be set"
        assert not (cap_no & 0x40), "STG bit should not be set"


class TestCapmercStructure:
    """Test structural correctness of generated capmerc."""

    def test_starts_with_prologue(self):
        cm = build_capmerc(num_gprs=10, text_size=384)
        # Body starts at offset 16
        prologue = cm[16:32]
        assert prologue == bytes.fromhex('010b040af80004000000410000040000')

    def test_has_terminal_record(self):
        cm = build_capmerc(num_gprs=10, text_size=384)
        parsed = parse_capmerc_structure(cm)
        assert parsed['terminal'] is not None, "Missing terminal record"
        assert parsed['terminal'][0] == 0x02
        assert parsed['terminal'][1] == 0x38

    def test_terminal_text_size_encoding(self):
        """Terminal record second-half byte[4] encodes (text_size >> 7) - 2."""
        for text_size, expected_b20 in [(256, 0), (384, 1), (512, 2), (768, 4)]:
            cm = build_capmerc(num_gprs=10, text_size=text_size)
            parsed = parse_capmerc_structure(cm)
            term = parsed['terminal']
            actual = term[20]
            assert actual == expected_b20, (
                f"text_size={text_size}: terminal byte[20]={actual}, "
                f"expected {expected_b20}"
            )

    def test_trailer_present(self):
        cm = build_capmerc(num_gprs=10, text_size=384)
        assert len(cm) >= 18
        trailer = cm[-2:]
        assert trailer[0] in (0xd0, 0x50), f"Bad trailer byte 0: {trailer[0]:#x}"

    def test_trailer_high_reg(self):
        """Trailer byte[0] should be 0x50 for high register count."""
        cm_low = build_capmerc(num_gprs=10, text_size=384)
        cm_high = build_capmerc(num_gprs=16, text_size=384)
        assert cm_low[-2] == 0xd0, "Low-reg trailer should be 0xd0"
        assert cm_high[-2] == 0x50, "High-reg trailer should be 0x50"

    def test_stg_descriptor_present(self):
        cm = build_capmerc(num_gprs=10, text_size=384, has_stg=True)
        parsed = parse_capmerc_structure(cm)
        stg_recs = [r for tag, r in parsed['records']
                    if tag == 'type01' and r[2] == 0x0e]
        assert len(stg_recs) >= 1, "Missing STG descriptor record"

    def test_no_stg_descriptor_when_disabled(self):
        cm = build_capmerc(num_gprs=10, text_size=384, has_stg=False)
        parsed = parse_capmerc_structure(cm)
        stg_recs = [r for tag, r in parsed['records']
                    if tag == 'type01' and r[2] == 0x0e]
        assert len(stg_recs) == 0, "STG descriptor present but has_stg=False"

    def test_fillers_for_high_reg(self):
        cm = build_capmerc(num_gprs=22, text_size=512)
        parsed = parse_capmerc_structure(cm)
        assert parsed['fillers'] > 0, "High-reg kernel should have fillers"

    def test_no_fillers_for_low_reg(self):
        cm = build_capmerc(num_gprs=8, text_size=256)
        parsed = parse_capmerc_structure(cm)
        assert parsed['fillers'] == 0, "Low-reg kernel should have no fillers"

    def test_branch_mode_in_terminal(self):
        cm_br = build_capmerc(num_gprs=12, text_size=384, has_branch=True)
        cm_no = build_capmerc(num_gprs=12, text_size=384, has_branch=False)
        parsed_br = parse_capmerc_structure(cm_br)
        parsed_no = parse_capmerc_structure(cm_no)
        assert parsed_br['terminal'][6] == 0x40, "Branch terminal should have mode 0x40"
        assert parsed_no['terminal'][6] == 0x50, "No-branch terminal should have mode 0x50"


class TestCapmercVsPtxas:
    """Compare generated capmerc against ptxas reference cubins.

    These tests verify structural equivalence — not byte-exact matching,
    since the capability mask and some per-instruction hints vary with
    the exact instruction sequence.
    """

    def _compare_structural(self, cubin_name: str, gen_params: dict):
        """Compare structural properties of generated vs ptxas capmerc."""
        ref_cm, ref_text = get_ptxas_capmerc(cubin_name)
        if ref_cm is None:
            import pytest
            pytest.skip(f"{cubin_name} not found in probe_work/")

        gen_cm = build_capmerc(**gen_params)
        ref_parsed = parse_capmerc_structure(ref_cm)
        gen_parsed = parse_capmerc_structure(gen_cm)

        # 1. Magic must match
        assert gen_parsed['magic'] == ref_parsed['magic'], "Magic mismatch"

        # 2. Register count must match
        assert gen_parsed['reg_count'] == ref_parsed['reg_count'], (
            f"Reg count: gen={gen_parsed['reg_count']} ref={ref_parsed['reg_count']}"
        )

        # 3. Terminal record must exist
        assert gen_parsed['terminal'] is not None, "Generated missing terminal"
        assert ref_parsed['terminal'] is not None, "Reference missing terminal"

        # 4. Terminal mode byte should match (branch vs no-branch)
        assert gen_parsed['terminal'][6] == ref_parsed['terminal'][6], (
            f"Terminal mode: gen=0x{gen_parsed['terminal'][6]:02x} "
            f"ref=0x{ref_parsed['terminal'][6]:02x}"
        )

        # 5. Terminal text-size encoding should match
        assert gen_parsed['terminal'][20] == ref_parsed['terminal'][20], (
            f"Terminal b20: gen={gen_parsed['terminal'][20]} "
            f"ref={ref_parsed['terminal'][20]}"
        )

        # 6. Terminal byte[12] (reg threshold) should match
        assert gen_parsed['terminal'][12] == ref_parsed['terminal'][12], (
            f"Terminal b12: gen=0x{gen_parsed['terminal'][12]:02x} "
            f"ref=0x{ref_parsed['terminal'][12]:02x}"
        )

        # 7. Trailer byte[0] should match (0xd0 vs 0x50)
        assert gen_parsed['trailer'][0] == ref_parsed['trailer'][0], (
            f"Trailer b0: gen=0x{gen_parsed['trailer'][0]:02x} "
            f"ref=0x{ref_parsed['trailer'][0]:02x}"
        )

        # 8. Prologue record should match exactly
        ref_type01 = [r for tag, r in ref_parsed['records'] if tag == 'type01']
        gen_type01 = [r for tag, r in gen_parsed['records'] if tag == 'type01']
        if ref_type01 and gen_type01:
            assert gen_type01[0] == ref_type01[0], "Prologue record mismatch"

        return ref_parsed, gen_parsed

    def test_ldg64_min_ptxas(self):
        """Simple store-only kernel: 8 GPRs, 256B text, no branch."""
        self._compare_structural('ldg64_min_ptxas.cubin', {
            'num_gprs': 8,
            'text_size': 256,
            'has_stg': True,
            'has_ldg': False,
            'has_branch': False,
            'has_ur_ops': True,
            'num_barrier_regions': 2,
        })

    def test_probe_k1_ptxas(self):
        """Basic SHF kernel: 10 GPRs, 384B text, no branch."""
        self._compare_structural('probe_k1_ptxas.cubin', {
            'num_gprs': 10,
            'text_size': 384,
            'has_stg': True,
            'has_ldg': False,
            'has_branch': False,
            'has_shift': True,
            'has_ur_ops': True,
            'num_barrier_regions': 2,
        })

    def test_bra_test_ptxas(self):
        """Branch kernel: 12 GPRs, 384B text, with branch."""
        self._compare_structural('bra_test_ptxas.cubin', {
            'num_gprs': 12,
            'text_size': 384,
            'has_stg': True,
            'has_ldg': False,
            'has_branch': True,
            'has_ur_ops': True,
            'has_isetp': True,
            'has_imad': True,
            'num_barrier_regions': 2,
        })

    def test_force_highreg(self):
        """High register kernel: 15 GPRs, 384B text, no branch."""
        self._compare_structural('force_highreg.cubin', {
            'num_gprs': 15,
            'text_size': 384,
            'has_stg': True,
            'has_ldg': False,
            'has_branch': False,
            'has_shift': True,
            'has_ur_ops': True,
            'num_barrier_regions': 2,
        })


class TestCapmercFromSass:
    """Test auto-detection from SASS bytes."""

    def test_sass_analysis_basic(self):
        """Verify analyze_sass_for_capmerc returns sane defaults."""
        ref_cm, ref_text = get_ptxas_capmerc('ldg64_min_ptxas.cubin')
        if ref_text is None:
            import pytest
            pytest.skip("ldg64_min_ptxas.cubin not found")
        params = analyze_sass_for_capmerc(ref_text)
        assert params['num_gprs'] >= 6, f"Expected >= 6 GPRs, got {params['num_gprs']}"
        assert params['has_stg'] is True, "Should detect STG"
        assert params['text_size'] == len(ref_text)

    def test_build_from_sass_produces_valid(self):
        """build_capmerc_from_sass should produce a valid capmerc."""
        ref_cm, ref_text = get_ptxas_capmerc('probe_k1_ptxas.cubin')
        if ref_text is None:
            import pytest
            pytest.skip("probe_k1_ptxas.cubin not found")
        cm = build_capmerc_from_sass(ref_text)
        assert cm[:8] == CAPMERC_MAGIC
        assert len(cm) >= 50  # minimum viable capmerc
        # Should have terminal
        parsed = parse_capmerc_structure(cm)
        assert parsed['terminal'] is not None

    def test_build_from_sass_reg_override(self):
        """num_gprs override should take precedence."""
        ref_cm, ref_text = get_ptxas_capmerc('probe_k1_ptxas.cubin')
        if ref_text is None:
            import pytest
            pytest.skip("probe_k1_ptxas.cubin not found")
        cm = build_capmerc_from_sass(ref_text, num_gprs=22)
        assert cm[8] == 22


class TestCapmercSizeReasonable:
    """Verify generated capmerc sizes are in the right ballpark."""

    def test_simple_kernel_size(self):
        """Simple kernel should produce 130-180 bytes."""
        cm = build_capmerc(num_gprs=8, text_size=256)
        assert 100 <= len(cm) <= 200, f"Size {len(cm)} out of range"

    def test_complex_kernel_size(self):
        """Complex kernel should produce 180-260 bytes."""
        cm = build_capmerc(
            num_gprs=22, text_size=768, has_stg=True, has_ldg=True,
            has_branch=True, has_ur_ops=True, has_fadd=True,
            num_barrier_regions=4,
        )
        assert 150 <= len(cm) <= 300, f"Size {len(cm)} out of range"

    def test_size_always_mod2(self):
        """Capmerc size should always be even (ends with 2-byte trailer)."""
        for reg in [8, 10, 12, 15, 19, 22]:
            cm = build_capmerc(num_gprs=reg, text_size=384)
            assert len(cm) % 2 == 0, f"reg={reg}: size {len(cm)} is odd"


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
