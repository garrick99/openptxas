"""Register-aware diff classification for OURS vs PTXAS instruction streams.

TEMPLATE-ENGINE-7A: classify differences as REG_ONLY, STRUCTURAL, or MIXED,
and provide operand-normalized comparison.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DiffClass(Enum):
    BYTE_EXACT = "BYTE_EXACT"
    REG_ONLY = "REG_ONLY"
    CTRL_ONLY = "CTRL_ONLY"
    REG_AND_CTRL = "REG_AND_CTRL"
    STRUCTURAL = "STRUCTURAL"
    MIXED = "MIXED"


@dataclass
class InstrDiff:
    """Per-instruction diff detail."""
    index: int
    opcode_ours: int
    opcode_ptxas: int
    byte_exact: bool
    reg_bytes_differ: list[int]    # byte positions 2-12 that differ
    ctrl_bytes_differ: list[int]   # byte positions 13-15 that differ


@dataclass
class KernelDiffResult:
    """Full diff result for one kernel."""
    kernel: str
    classification: DiffClass
    ours_count: int
    ptxas_count: int
    instr_delta: int           # ours_count - ptxas_count
    opcodes_match: bool
    byte_exact_instrs: int
    reg_only_instrs: int
    ctrl_only_instrs: int
    mixed_instrs: int
    normalized_match: bool     # after register renaming
    details: list[InstrDiff] = field(default_factory=list)


def _get_opcode(raw: bytes) -> int:
    return int.from_bytes(raw[:2], "little") & 0xFFF


def _get_active(cubin: bytes) -> list[bytes]:
    """Extract active (non-NOP) instructions from first .text section."""
    e_shoff = struct.unpack_from("<Q", cubin, 40)[0]
    e_shnum = struct.unpack_from("<H", cubin, 60)[0]
    e_shstrndx = struct.unpack_from("<H", cubin, 62)[0]
    stoff = struct.unpack_from("<Q", cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    for i in range(e_shnum):
        base = e_shoff + i * 64
        nm = struct.unpack_from("<I", cubin, base)[0]
        ne = cubin.index(0, stoff + nm)
        name = cubin[stoff + nm:ne]
        if name.startswith(b".text."):
            off = struct.unpack_from("<Q", cubin, base + 24)[0]
            sz = struct.unpack_from("<Q", cubin, base + 32)[0]
            active = []
            for j in range(0, sz, 16):
                raw = cubin[off + j:off + j + 16]
                if len(raw) == 16 and _get_opcode(raw) != 0x918:
                    active.append(bytes(raw))
            return active
    return []


# Byte positions that encode register indices (operand bytes).
# Positions 0-1 = opcode, 2-12 = operands/mode, 13-15 = ctrl.
_REG_BYTE_POSITIONS = {2, 3, 4, 8}  # common register field positions
_CTRL_BYTE_POSITIONS = {13, 14, 15}


def _classify_instr_diff(ours: bytes, ptxas: bytes, idx: int) -> InstrDiff:
    """Classify a single instruction difference."""
    reg_diffs = []
    ctrl_diffs = []
    for b in range(16):
        if ours[b] != ptxas[b]:
            if b in _CTRL_BYTE_POSITIONS:
                ctrl_diffs.append(b)
            elif b >= 2 and b <= 12:
                reg_diffs.append(b)
    return InstrDiff(
        index=idx,
        opcode_ours=_get_opcode(ours),
        opcode_ptxas=_get_opcode(ptxas),
        byte_exact=(ours == ptxas),
        reg_bytes_differ=reg_diffs,
        ctrl_bytes_differ=ctrl_diffs,
    )


def _normalize_registers(instrs: list[bytes]) -> list[bytes]:
    """Remap register indices to canonical form for structural comparison.

    For each instruction, remap the register fields (b2, b3, b4, b8) using
    a first-seen mapping.  This strips register allocation differences while
    preserving structural patterns.
    """
    reg_map: dict[int, int] = {}
    next_id = 0
    result = []
    for raw in instrs:
        norm = bytearray(raw)
        for pos in sorted(_REG_BYTE_POSITIONS):
            val = raw[pos]
            if val == 0xFF:  # RZ / special
                continue
            if val not in reg_map:
                reg_map[val] = next_id
                next_id += 1
            norm[pos] = reg_map[val]
        # Also zero out ctrl bytes for pure structural comparison
        norm[13] = 0
        norm[14] = 0
        norm[15] = 0
        result.append(bytes(norm))
    return result


def diff_kernel(ours_cubin: bytes, ptxas_cubin: bytes, kernel_name: str) -> KernelDiffResult:
    """Full register-aware diff for one kernel."""
    ours = _get_active(ours_cubin)
    ptxas = _get_active(ptxas_cubin)

    ours_opc = [_get_opcode(r) for r in ours]
    ptxas_opc = [_get_opcode(r) for r in ptxas]
    opcodes_match = ours_opc == ptxas_opc

    # Per-instruction classification (only if same count)
    details = []
    byte_exact = 0
    reg_only = 0
    ctrl_only = 0
    mixed = 0

    if len(ours) == len(ptxas):
        for i in range(len(ours)):
            d = _classify_instr_diff(ours[i], ptxas[i], i)
            details.append(d)
            if d.byte_exact:
                byte_exact += 1
            elif not d.reg_bytes_differ and d.ctrl_bytes_differ:
                ctrl_only += 1
            elif d.reg_bytes_differ and not d.ctrl_bytes_differ:
                reg_only += 1
            else:
                mixed += 1

    # Normalized comparison
    if opcodes_match and len(ours) == len(ptxas):
        norm_ours = _normalize_registers(ours)
        norm_ptxas = _normalize_registers(ptxas)
        normalized_match = norm_ours == norm_ptxas
    else:
        normalized_match = False

    # Overall classification
    if all(o == p for o, p in zip(ours, ptxas)) and len(ours) == len(ptxas):
        classification = DiffClass.BYTE_EXACT
    elif not opcodes_match or len(ours) != len(ptxas):
        classification = DiffClass.STRUCTURAL
    elif mixed > 0:
        classification = DiffClass.MIXED
    elif reg_only > 0 and ctrl_only > 0:
        classification = DiffClass.REG_AND_CTRL
    elif reg_only > 0:
        classification = DiffClass.REG_ONLY
    elif ctrl_only > 0:
        classification = DiffClass.CTRL_ONLY
    else:
        classification = DiffClass.BYTE_EXACT

    return KernelDiffResult(
        kernel=kernel_name,
        classification=classification,
        ours_count=len(ours),
        ptxas_count=len(ptxas),
        instr_delta=len(ours) - len(ptxas),
        opcodes_match=opcodes_match,
        byte_exact_instrs=byte_exact,
        reg_only_instrs=reg_only,
        ctrl_only_instrs=ctrl_only,
        mixed_instrs=mixed,
        normalized_match=normalized_match,
        details=details,
    )
