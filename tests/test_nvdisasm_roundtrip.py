"""
nvdisasm roundtrip validation for SM_120 encoders.

For each encoder, build a small instruction, dump it as a raw SM120 binary,
invoke NVIDIA's `nvdisasm --binary SM120`, and compare the resulting mnemonic
against the expected string. This catches encoding bugs that may pass internal
tests but not match real NVIDIA encoding.

The test is skipped if nvdisasm is not found on the system.

Inspired by NVK's nvdisasm_tests.rs (Mesa nouveau compiler).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import pytest

from sass.encoding.sm_120_opcodes import (
    encode_iadd3,
    encode_imad,
    encode_fadd,
    encode_fmul,
    encode_ffma,
    encode_dadd,
    encode_dmul,
    encode_dfma,
    encode_lds,
    encode_sts,
    encode_ldc,
    encode_bra,
    encode_exit,
    encode_mufu,
    encode_popc,
    encode_brev,
    encode_flo,
    encode_sel,
    encode_fsel,
    encode_isetp,
    encode_nop,
    encode_mov,
    encode_s2r,
    encode_iabs,
    encode_lop3,
    LOP3_AND,
)


def _find_nvdisasm() -> str | None:
    """Locate nvdisasm on this system."""
    # Try PATH first
    exe = shutil.which("nvdisasm")
    if exe:
        return exe
    # Common Windows install
    candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvdisasm.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvdisasm.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvdisasm.exe",
        "/usr/local/cuda/bin/nvdisasm",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


NVDISASM = _find_nvdisasm()
pytestmark = pytest.mark.skipif(
    NVDISASM is None, reason="nvdisasm not found (requires CUDA toolkit)"
)


def _disasm(raw: bytes) -> list[str]:
    """Disassemble a raw sm_120 binary and return the mnemonic lines."""
    with tempfile.NamedTemporaryFile(
        suffix=".bin", delete=False, dir=os.getcwd()
    ) as f:
        f.write(raw)
        path = f.name
    try:
        out = subprocess.run(
            [NVDISASM, "--binary", "SM120", path],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if out.returncode != 0:
        raise RuntimeError(
            f"nvdisasm failed (rc={out.returncode}):\n"
            f"stdout: {out.stdout}\nstderr: {out.stderr}"
        )
    lines = []
    for line in out.stdout.splitlines():
        # Lines look like: "        /*0000*/                   IADD3 R3, P0, PT, R1, R2, R0;"
        m = re.match(r"\s*/\*[0-9a-fA-F]+\*/\s*(.*);", line)
        if m:
            lines.append(m.group(1).strip())
    return lines


# Each entry: (encoder-output, expected-prefix-match-substring)
# Uses substring (not equality) because nvdisasm may print extra flags
# (e.g. predicates, scoreboard annotations) depending on ctrl bits.
ROUNDTRIP_CASES = [
    ("IADD3",    encode_iadd3(3, 1, 2, 0),        "IADD3 R3"),
    ("IMAD",     encode_imad(4, 1, 2, 3),         "IMAD R4, R1, R2, R3"),
    ("FADD",     encode_fadd(3, 1, 2),            "FADD R3, R1, R2"),
    # Note: FMUL is encoded as FFMA with RZ addend.
    ("FMUL",     encode_fmul(3, 1, 2),            "FFMA R3, R1, R2, RZ"),
    ("FFMA",     encode_ffma(4, 1, 2, 3),         "FFMA R4, R1, R2, R3"),
    ("DADD",     encode_dadd(4, 2, 0),            "DADD R4, R2, R0"),
    ("DMUL",     encode_dmul(4, 2, 0),            "DMUL R4, R2, R0"),
    ("DFMA",     encode_dfma(6, 2, 0, 4),         "DFMA R6, R2, R0, R4"),
    ("LDS",      encode_lds(3, 4, 0),             "LDS R3"),
    ("STS",      encode_sts(4, 0, 3),             "STS"),
    ("BRA",      encode_bra(0),                   "BRA"),
    ("EXIT",     encode_exit(),                   "EXIT"),
    ("MUFU",     encode_mufu(3, 2, 0),            "MUFU"),
    ("POPC",     encode_popc(3, 2),               "POPC R3, R2"),
    ("BREV",     encode_brev(3, 2),               "BREV R3, R2"),
    ("FLO",      encode_flo(3, 2),                "FLO"),
    ("SEL",      encode_sel(3, 1, 2),             "SEL R3, R1, R2"),
    ("FSEL",     encode_fsel(3, 1, 2),            "FSEL R3, R1, R2"),
    ("LDC",      encode_ldc(3, 0, 0x160),         "LDC R3, c[0x0][0x160]"),
    ("ISETP",    encode_isetp(0, 1, 2),           "ISETP.GE"),
    ("NOP",      encode_nop(),                    "NOP"),
    ("MOV",      encode_mov(3, 4),                "MOV R3, R4"),
    ("S2R",      encode_s2r(1, 0),                "S2R R1"),
    ("IABS",     encode_iabs(3, 2),               "IABS R3"),
    ("LOP3_AND", encode_lop3(3, 1, 2, 0xff, LOP3_AND, 0), "LOP3"),
]


def test_nvdisasm_roundtrip_batch():
    """Dump every encoder into a single raw binary and compare disassembly lines."""
    assert NVDISASM is not None
    data = b"".join(b for _, b, _ in ROUNDTRIP_CASES)
    lines = _disasm(data)
    assert len(lines) == len(ROUNDTRIP_CASES), (
        f"nvdisasm produced {len(lines)} lines, expected {len(ROUNDTRIP_CASES)}\n"
        f"Lines: {lines}"
    )
    mismatches = []
    for (name, _raw, expected), got in zip(ROUNDTRIP_CASES, lines):
        if expected not in got:
            mismatches.append(f"  {name}: expected substring '{expected}' in '{got}'")
    assert not mismatches, "nvdisasm roundtrip mismatches:\n" + "\n".join(mismatches)


@pytest.mark.parametrize(
    "name,raw,expected",
    [(n, r, e) for n, r, e in ROUNDTRIP_CASES],
    ids=[n for n, _, _ in ROUNDTRIP_CASES],
)
def test_nvdisasm_roundtrip_individual(name, raw, expected):
    """Per-encoder roundtrip so failures are pinpointed."""
    lines = _disasm(raw)
    assert len(lines) >= 1, f"no disassembly for {name}"
    assert expected in lines[0], (
        f"{name}: expected substring '{expected}' in disassembly '{lines[0]}'\n"
        f"raw bytes: {raw.hex()}"
    )
