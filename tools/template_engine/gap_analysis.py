"""Gap analysis: quantify and explain OURS vs PTXAS structural differences.

TEMPLATE-ENGINE-7B: since TE7-A proved zero REG_ONLY differences exist,
this module explains WHY the differences are structural and where the
isel-level gap lies.
"""
from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GapSummary:
    """Summary of opcode-level gap between OURS and PTXAS."""
    ours_extra: Counter        # opcodes OURS uses more
    ptxas_extra: Counter       # opcodes PTXAS uses more
    structural_kernels: int
    same_count_structural: int # structural but same instruction count
    byte_exact_count: int
    total_kernels: int

    @property
    def structural_pct(self) -> float:
        return 100 * self.structural_kernels / self.total_kernels if self.total_kernels else 0

    @property
    def exact_pct(self) -> float:
        return 100 * self.byte_exact_count / self.total_kernels if self.total_kernels else 0


# Opcode names for readable output
_OPC_NAMES = {
    0x810: "IADD3.IMM", 0x210: "IADD3", 0x20c: "ISETP.R-R",
    0xb82: "S2R/LDC", 0x825: "IMAD.WIDE", 0xc35: "IADD.64-UR",
    0x824: "IMAD", 0x80c: "FSETP", 0x812: "LOP3.IMM",
    0x947: "BRA", 0xc11: "ISETP.UR-UR", 0xc0c: "ISETP.R-UR",
    0x7ac: "LDCU", 0x835: "UIADD", 0x431: "HFMA2",
    0x235: "IADD.64", 0x424: "FMUL.IMM", 0x819: "SHF",
    0x202: "MOV",
}

# The structural gap explanation
STRUCTURAL_GAP_EXPLANATION = """
## Structural Gap: OURS vs PTXAS

The 79% structural difference is NOT register allocation. It is instruction
selection — PTXAS and OpenPTXas use different opcodes for the same operations:

### Parameter handling
- PTXAS: ld.param → LDCU to UR → use UR in comparisons (ISETP.R-UR)
- OURS: ld.param → LDC/LDCU → move to GPR → compare GPR-GPR (ISETP.R-R)

### Address computation
- PTXAS: param addr via S2R → offset → UIADD for uniform adds
- OURS: param addr via LDCU + IADD.64-UR → IMAD.WIDE for wide adds

### Predicate comparisons
- PTXAS: ISETP.R-UR (0xc0c) +164 over OURS
- OURS: ISETP.R-R (0x20c) +105 over PTXAS

### Key insight
Both produce CORRECT code. The difference is in the degree of UR file
utilization. PTXAS uses the UR file for everything parameter-related,
while OpenPTXas routes most values through GPR.

Closing this gap requires isel-level changes to emit UR-centric instruction
sequences, not register allocation changes.
"""


def format_gap_table(summary: GapSummary) -> str:
    """Format a readable gap analysis."""
    lines = []
    lines.append(f"Total kernels: {summary.total_kernels}")
    lines.append(f"Byte-exact:    {summary.byte_exact_count} ({summary.exact_pct:.1f}%)")
    lines.append(f"Structural:    {summary.structural_kernels} ({summary.structural_pct:.1f}%)")
    lines.append(f"  same count:  {summary.same_count_structural}")
    lines.append("")
    lines.append("Opcodes OURS uses more (top 5):")
    for opc, cnt in summary.ours_extra.most_common(5):
        name = _OPC_NAMES.get(opc, f"0x{opc:03x}")
        lines.append(f"  {name:<15s} (0x{opc:03x}): +{cnt}")
    lines.append("")
    lines.append("Opcodes PTXAS uses more (top 5):")
    for opc, cnt in summary.ptxas_extra.most_common(5):
        name = _OPC_NAMES.get(opc, f"0x{opc:03x}")
        lines.append(f"  {name:<15s} (0x{opc:03x}): +{cnt}")
    return "\n".join(lines)
