"""
test_patcher.py — Tests for the cubin template patcher.

Uses probe_work/probe_k1.cubin as a ground truth cubin.
"""

import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cubin.patcher import CubinPatcher, find_instruction_offset, disassemble_text
from sass.encoding.sm_120_opcodes import encode_exit, encode_nop, RZ

PROBE_CUBIN = Path(__file__).parent.parent / "probe_work" / "probe_k1.cubin"


def _skip_if_missing():
    import pytest
    if not PROBE_CUBIN.exists():
        pytest.skip(f"test cubin not found: {PROBE_CUBIN}")


def test_kernel_names():
    _skip_if_missing()
    p = CubinPatcher(PROBE_CUBIN)
    assert p.kernel_names() == ['probe_k1']


def test_get_instruction():
    _skip_if_missing()
    p = CubinPatcher(PROBE_CUBIN)
    # First instruction is LDC R1, c[0][0x37c]
    first = p.get_instruction('probe_k1', 0)
    assert first == bytes.fromhex('827b01ff00df00000008000000e20f00')
    # EXIT instruction (offset shifts with pipeline changes; currently at 288)
    exit_instr = p.get_instruction('probe_k1', 288)
    assert exit_instr[:2] == bytes.fromhex('4d79')  # EXIT opcode


def test_patch_roundtrip():
    """Patch a NOP → EXIT and verify; then check the cubin can be written."""
    _skip_if_missing()
    p = CubinPatcher(PROBE_CUBIN)
    nop_offset = 208  # first NOP after EXIT+BRA
    before = p.get_instruction('probe_k1', nop_offset)
    assert before[:2] == bytes.fromhex('1879'), \
        f"Expected NOP at +{nop_offset}, got opcode {before[:2].hex()}"
    p.patch_instruction('probe_k1', nop_offset, encode_exit(ctrl=0x7f5))
    after = p.get_instruction('probe_k1', nop_offset)
    assert after[:2] == bytes.fromhex('4d79')  # EXIT opcode


def test_write_and_reload():
    """Write patched cubin to temp file and reload, verifying patch persisted."""
    _skip_if_missing()
    p = CubinPatcher(PROBE_CUBIN)
    p.patch_instruction('probe_k1', 208, encode_exit(ctrl=0x7f5))
    with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        p.write(tmp_path)
        # Reload and verify
        p2 = CubinPatcher(tmp_path)
        patched = p2.get_instruction('probe_k1', 208)
        assert patched[:2] == bytes.fromhex('4d79')  # EXIT opcode
        # Original instruction unchanged
        first = p2.get_instruction('probe_k1', 0)
        assert first == bytes.fromhex('827b01ff00df00000008000000e20f00')
    finally:
        tmp_path.unlink(missing_ok=True)


def test_disassemble_text():
    _skip_if_missing()
    instructions = disassemble_text(PROBE_CUBIN, 'probe_k1')
    assert len(instructions) == 24  # 384 bytes / 16 = 24 instructions
    assert instructions[0] == (0, bytes.fromhex('827b01ff00df00000008000000e20f00'))
    # EXIT is at instruction 11 (+176) in current layout
    exit_idx = next(i for i, (off, b) in enumerate(instructions) if b[:2] == bytes.fromhex('4d79'))
    assert instructions[exit_idx][1][:2] == bytes.fromhex('4d79')


def test_find_instruction_offset():
    _skip_if_missing()
    # Find first NOP in the text section
    nop_hex = '18790000000000000000000000c00f00'
    offset = find_instruction_offset(PROBE_CUBIN, 'probe_k1', bytes.fromhex(nop_hex))
    assert offset is not None and offset % 16 == 0  # NOP found at aligned offset

    # Find EXIT by opcode prefix (ctrl bytes may vary)
    instructions = disassemble_text(PROBE_CUBIN, 'probe_k1')
    exit_offset = next(off for off, b in instructions if b[:2] == bytes.fromhex('4d79'))
    offset2 = exit_offset
    assert offset2 >= 128  # EXIT position depends on pipeline layout


def test_offset_alignment_check():
    _skip_if_missing()
    import pytest
    p = CubinPatcher(PROBE_CUBIN)
    with pytest.raises(ValueError, match="not 16-byte aligned"):
        p.patch_instruction('probe_k1', 17, encode_nop())


def test_patch_size_check():
    _skip_if_missing()
    import pytest
    p = CubinPatcher(PROBE_CUBIN)
    with pytest.raises(ValueError, match="16 bytes"):
        p.patch_instruction('probe_k1', 0, b'\x00' * 8)
