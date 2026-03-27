"""
tests/test_exit_metadata.py — EIATTR_EXIT_INSTR_OFFSETS metadata integrity tests.

Verifies that the cubin emitter correctly finds and records all EXIT instruction
byte offsets in the .nv.info.kernel section (attribute 0x1c).

Bugs fixed 2026-03-21:
  - Old code used raw byte match [0x4d, 0x79] for EXIT, missing predicated exits
    (@P0 EXIT has byte[1]=0x09, not 0x79 — predicate bits change byte 1).
  - Old code used a single exit_offset int; driver needs ALL exits to avoid
    patching the wrong instruction and corrupting its encoding.

Run: python -m pytest tests/test_exit_metadata.py -v
"""
import struct
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass.encoding.sm_120_opcodes import (
    encode_nop, encode_exit, encode_s2r, encode_bra,
    encode_ldcu_32, encode_isetp_ge_and, patch_pred,
    SR_TID_X, RZ,
)
from sass.isel import SassInstr
from sass.scoreboard import assign_ctrl
from sass.pipeline import compile_ptx_source
from cubin.emitter import emit_cubin, KernelDesc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_exits_correct(sass_bytes: bytes) -> list[int]:
    """The correct EXIT scan: opcode mask catches predicated + unconditional."""
    return [i for i in range(0, len(sass_bytes), 16)
            if (struct.unpack_from('<Q', sass_bytes, i)[0] & 0xFFF) == 0x94d]


def _find_exits_naive(sass_bytes: bytes) -> list[int]:
    """The old broken EXIT scan: raw bytes only catches unconditional EXIT."""
    return [i for i in range(0, len(sass_bytes), 16)
            if sass_bytes[i:i+2] == bytes([0x4d, 0x79])]


def _parse_exit_offsets_from_nvinfo(cubin_bytes: bytes) -> list[int]:
    """Extract EIATTR_EXIT_INSTR_OFFSETS (0x1c) from the .nv.info.<kernel> section."""
    data = cubin_bytes
    e_shoff = struct.unpack_from('<Q', data, 40)[0]
    e_shnum = struct.unpack_from('<H', data, 60)[0]
    e_shstrndx = struct.unpack_from('<H', data, 62)[0]
    sh_base = e_shoff + e_shstrndx * 64
    shstrtab_off = struct.unpack_from('<Q', data, sh_base + 24)[0]
    shstrtab_size = struct.unpack_from('<Q', data, sh_base + 32)[0]
    shstrtab = data[shstrtab_off:shstrtab_off + shstrtab_size]

    for i in range(e_shnum):
        sh = e_shoff + i * 64
        name_off = struct.unpack_from('<I', data, sh)[0]
        sh_off = struct.unpack_from('<Q', data, sh + 24)[0]
        sh_size = struct.unpack_from('<Q', data, sh + 32)[0]
        end = shstrtab.find(b'\x00', name_off)
        name = shstrtab[name_off:end].decode('utf-8', errors='replace')
        if sh_size > 0 and name.startswith('.nv.info.') and not name.startswith('.nv.info.vector'):
            pass  # skip global
        if sh_size > 0 and name.startswith('.nv.info.') and '.' in name[9:]:
            sec_data = data[sh_off:sh_off + sh_size]
            offsets = _parse_attr_0x1c(sec_data)
            if offsets is not None:
                return offsets

    # Fallback: scan all nv.info sections
    for i in range(e_shnum):
        sh = e_shoff + i * 64
        name_off = struct.unpack_from('<I', data, sh)[0]
        sh_off = struct.unpack_from('<Q', data, sh + 24)[0]
        sh_size = struct.unpack_from('<Q', data, sh + 32)[0]
        end = shstrtab.find(b'\x00', name_off)
        name = shstrtab[name_off:end].decode('utf-8', errors='replace')
        if sh_size > 0 and 'nv.info.' in name and name != '.nv.info':
            sec_data = data[sh_off:sh_off + sh_size]
            offsets = _parse_attr_0x1c(sec_data)
            if offsets is not None:
                return offsets
    return []


def _parse_attr_0x1c(sec_data: bytes) -> list[int] | None:
    """Parse the 0x1c attribute from an .nv.info.kernel section."""
    pos = 0
    while pos + 2 <= len(sec_data):
        fmt = sec_data[pos]
        tag = sec_data[pos + 1]
        if fmt == 0x04 and tag == 0x1c:
            size = struct.unpack_from('<H', sec_data, pos + 2)[0]
            payload = sec_data[pos + 4:pos + 4 + size]
            offsets = []
            for j in range(0, len(payload), 4):
                offsets.append(struct.unpack_from('<I', payload, j)[0])
            return offsets
        if fmt == 0x04:
            size = struct.unpack_from('<H', sec_data, pos + 2)[0]
            pos += 4 + size
        elif fmt in (0x02, 0x03):
            pos += 4
        else:
            break
    return None


def _make_minimal_cubin(sass_bytes: bytes, name: str = 'test_k') -> bytes:
    # EXIT offsets are auto-scanned from sass_bytes inside emit_cubin.
    kd = KernelDesc(
        name=name,
        sass_bytes=sass_bytes,
        num_gprs=4,
        num_params=0,
        param_sizes=[],
        param_offsets={},
        const0_size=0x384,
        s2r_offset=0,
    )
    return emit_cubin(kd)


# ---------------------------------------------------------------------------
# Tests: EXIT opcode detection
# ---------------------------------------------------------------------------

class TestExitDetection:
    def test_unconditional_exit_detected(self):
        """Unconditional EXIT (byte[1]=0x79) must be found by both naive and correct scan."""
        sass = encode_s2r(0, SR_TID_X) + encode_exit()
        naive = _find_exits_naive(sass)
        correct = _find_exits_correct(sass)
        assert 16 in naive,   "Naive scan missed unconditional EXIT"
        assert 16 in correct, "Correct scan missed unconditional EXIT"

    def test_predicated_exit_missed_by_naive(self):
        """Predicated @P0 EXIT (byte[1]=0x09) must be MISSED by the naive scan."""
        # Build @P0 EXIT: patch predicate bits into the exit encoding
        exit_raw = bytearray(encode_exit())
        exit_raw[1] = (exit_raw[1] & 0x0F) | 0x00  # P0 = 0 in high nibble
        pred_exit = bytes(exit_raw)
        sass = encode_s2r(0, SR_TID_X) + pred_exit + encode_exit()

        naive = _find_exits_naive(sass)
        correct = _find_exits_correct(sass)

        # The predicated exit at offset 16 should be missed by naive, caught by correct
        assert 16 not in naive,   "Naive scan should NOT catch predicated @P0 EXIT"
        assert 16 in correct,     "Correct scan must catch predicated @P0 EXIT"
        # Both must find the unconditional exit at 32
        assert 32 in naive
        assert 32 in correct

    def test_predicated_exit_opcode_is_0x94d(self):
        """Predicated EXIT has same opcode bits as unconditional EXIT."""
        exit_raw = bytearray(encode_exit())
        exit_raw[1] = (exit_raw[1] & 0x0F) | 0x00  # P0
        pred_exit = bytes(exit_raw)
        lo = struct.unpack_from('<Q', pred_exit, 0)[0]
        assert (lo & 0xFFF) == 0x94d, \
            f"Predicated EXIT opcode bits = 0x{lo & 0xFFF:03x}, expected 0x94d"

    def test_two_exits_both_detected(self):
        """Kernel with two EXITs (unconditional) must report both offsets."""
        sass = encode_s2r(0, SR_TID_X) + encode_nop() + encode_exit() + encode_exit()
        correct = _find_exits_correct(sass)
        assert sorted(correct) == [32, 48], f"Expected exits at [32, 48], got {correct}"


# ---------------------------------------------------------------------------
# Tests: metadata emission
# ---------------------------------------------------------------------------

class TestExitOffsetMetadata:
    def test_single_exit_in_nvinfo(self):
        """Kernel with one EXIT must have that offset in .nv.info.kernel attr 0x1c."""
        sass = encode_s2r(0, SR_TID_X) + encode_nop() + encode_exit()
        cubin = _make_minimal_cubin(sass)
        offsets = _parse_exit_offsets_from_nvinfo(cubin)
        assert offsets == [32], f"Expected [32], got {offsets}"

    def test_predicated_and_final_exit_both_in_nvinfo(self):
        """Kernel with predicated exit + final exit must list both in attr 0x1c."""
        # Build: S2R + predicated-EXIT + NOP + EXIT
        exit_raw = bytearray(encode_exit())
        exit_raw[1] = (exit_raw[1] & 0x0F) | 0x00  # P0
        pred_exit = bytes(exit_raw)
        sass = encode_s2r(0, SR_TID_X) + pred_exit + encode_nop() + encode_exit()
        cubin = _make_minimal_cubin(sass)
        offsets = _parse_exit_offsets_from_nvinfo(cubin)
        assert sorted(offsets) == [16, 48], \
            f"Expected [16, 48], got {offsets}"

    def test_pipeline_exit_offsets_match_sass(self):
        """compile_ptx_source must produce cubin whose attr 0x1c matches actual EXIT positions."""
        ptx_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'probe_work', 'vector_add.ptx')
        ptx = open(ptx_path, encoding='utf-8').read()
        results = compile_ptx_source(ptx)
        assert len(results) == 1
        cubin = next(iter(results.values()))

        # Find actual EXIT positions in .text
        data = cubin
        e_shoff = struct.unpack_from('<Q', data, 40)[0]
        e_shnum = struct.unpack_from('<H', data, 60)[0]
        e_shstrndx = struct.unpack_from('<H', data, 62)[0]
        sh_base = e_shoff + e_shstrndx * 64
        shstrtab_off = struct.unpack_from('<Q', data, sh_base + 24)[0]
        shstrtab = data[shstrtab_off:]
        text_data = None
        for i in range(e_shnum):
            sh = e_shoff + i * 64
            name_off = struct.unpack_from('<I', data, sh)[0]
            end = shstrtab.find(b'\x00', name_off)
            name = shstrtab[name_off:end].decode('utf-8', errors='replace')
            if name.startswith('.text.'):
                sh_off = struct.unpack_from('<Q', data, sh + 24)[0]
                sh_size = struct.unpack_from('<Q', data, sh + 32)[0]
                text_data = data[sh_off:sh_off + sh_size]
                break

        assert text_data is not None, "No .text section found"
        actual_exits = _find_exits_correct(text_data)
        nvinfo_exits = _parse_exit_offsets_from_nvinfo(cubin)

        assert sorted(nvinfo_exits) == sorted(actual_exits), \
            f"NV info exits {nvinfo_exits} != actual exits in .text {actual_exits}"
        assert len(actual_exits) >= 2, \
            f"Expected at least 2 EXITs (predicated + final), got {actual_exits}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
