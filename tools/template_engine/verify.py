"""Template verification: compare compiler output against discovered PTXAS templates.

TEMPLATE-ENGINE-6B: post-scheduling verification that our output matches
known PTXAS patterns.  Not a replacer — a confidence signal.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VerifyResult:
    """Result of comparing compiler output to a template."""
    kernel_name: str
    cluster_id: int
    family_name: str
    opcode_match: bool         # same opcode sequence
    byte_match_count: int      # instructions with exact byte match
    total_instructions: int
    ctrl_only_diffs: int       # instructions differing only in ctrl bytes (b13-b15)
    functional_diffs: int      # instructions differing in functional bytes (b0-b12)

    @property
    def match_ratio(self) -> float:
        return self.byte_match_count / self.total_instructions if self.total_instructions else 0.0

    @property
    def is_opcode_equivalent(self) -> bool:
        return self.opcode_match

    @property
    def is_ctrl_only(self) -> bool:
        """True if all diffs are in ctrl bytes only (functionally equivalent)."""
        return self.opcode_match and self.functional_diffs == 0


def _get_opcode(raw: bytes) -> int:
    return int.from_bytes(raw[:2], 'little') & 0xFFF


def _get_active_instrs(cubin: bytes) -> list[bytes]:
    """Extract active (non-NOP) instructions from first .text section."""
    e_shoff = struct.unpack_from('<Q', cubin, 40)[0]
    e_shnum = struct.unpack_from('<H', cubin, 60)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 62)[0]
    stoff = struct.unpack_from('<Q', cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    for i in range(e_shnum):
        base = e_shoff + i * 64
        nm = struct.unpack_from('<I', cubin, base)[0]
        ne = cubin.index(0, stoff + nm)
        name = cubin[stoff + nm:ne]
        if name.startswith(b'.text.'):
            off = struct.unpack_from('<Q', cubin, base + 24)[0]
            sz = struct.unpack_from('<Q', cubin, base + 32)[0]
            active = []
            for j in range(0, sz, 16):
                raw = cubin[off + j:off + j + 16]
                if len(raw) == 16 and _get_opcode(raw) != 0x918:
                    active.append(bytes(raw))
            return active
    return []


def verify_against_template(ours_cubin: bytes, ptxas_cubin: bytes,
                            kernel_name: str, cluster_id: int = -1,
                            family_name: str = '') -> VerifyResult:
    """Compare our compiled output against a PTXAS reference."""
    ours = _get_active_instrs(ours_cubin)
    ptxas = _get_active_instrs(ptxas_cubin)

    # Opcode sequence match
    ours_opc = [_get_opcode(r) for r in ours]
    ptxas_opc = [_get_opcode(r) for r in ptxas]
    opc_match = ours_opc == ptxas_opc

    # Byte-level comparison
    n = min(len(ours), len(ptxas))
    exact = 0
    ctrl_only = 0
    functional = 0
    for i in range(n):
        if ours[i] == ptxas[i]:
            exact += 1
        elif ours[i][:13] == ptxas[i][:13]:
            ctrl_only += 1  # only b13-b15 differ
        else:
            functional += 1

    return VerifyResult(
        kernel_name=kernel_name,
        cluster_id=cluster_id,
        family_name=family_name,
        opcode_match=opc_match,
        byte_match_count=exact,
        total_instructions=max(len(ours), len(ptxas)),
        ctrl_only_diffs=ctrl_only,
        functional_diffs=functional,
    )
