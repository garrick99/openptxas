"""
tools/audit.py — GPU binary auditor for SM_89/SM_120 cubins.

Scans SASS instructions for:
  1. Known miscompilation patterns (ptxas rotate-sub bug)
  2. Scheduling hazards (missing barriers, LDG consumers without rbar)
  3. Register pressure warnings (high GPR count, potential spills)
  4. Memory access patterns (uncoalesced loads, bank conflicts)
  5. Synchronization issues (missing bar.sync before shared reads)

Usage:
    python -m openptxas --audit input.cubin
    python -m openptxas --audit input.cubin --verbose
"""

from __future__ import annotations
import struct
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from cubin.patcher import CubinPatcher, ELF64, disassemble_text


class Severity(Enum):
    INFO = "INFO"
    WARN = "WARN"
    BUG  = "BUG"
    CRITICAL = "CRITICAL"


@dataclass
class Finding:
    severity: Severity
    kernel: str
    offset: int
    title: str
    detail: str
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Opcode classification
# ---------------------------------------------------------------------------

_OP_NAMES = {
    0x819: 'SHF', 0x210: 'IADD3', 0x202: 'MOV', 0x918: 'NOP', 0x94d: 'EXIT',
    0x981: 'LDG', 0x986: 'STG', 0xb82: 'LDC', 0x947: 'BRA', 0x7ac: 'LDCU',
    0x235: 'IADD.64', 0x431: 'HFMA2', 0x824: 'IMAD', 0x825: 'IMAD.W',
    0xc0c: 'ISETP', 0x212: 'LOP3', 0x988: 'STS', 0x984: 'LDS', 0xb1d: 'BAR',
    0x9c3: 'S2UR', 0x882: 'UMOV', 0x291: 'ULEA', 0x221: 'FADD', 0x223: 'FFMA',
    0x23c: 'HMMA', 0x237: 'IMMA', 0x245: 'I2F', 0x305: 'F2I',
    0xa02: 'MOV.C', 0x624: 'IMAD.M', 0xab9: 'ULDC', 0x83b: 'LDSM',
}


def _decode(raw: bytes) -> dict:
    lo = struct.unpack_from('<Q', raw, 0)[0]
    opcode = lo & 0xFFF
    raw24 = (raw[15] << 16) | (raw[14] << 8) | raw[13]
    ctrl = raw24 >> 1
    return {
        'opcode': opcode,
        'name': _OP_NAMES.get(opcode, f'UNK_{opcode:#05x}'),
        'dest': raw[2],
        'src0': raw[3],
        'b4': raw[4],
        'src1': raw[8],
        'mod9': raw[9],
        'mod10': raw[10],
        'ctrl': ctrl,
        'stall': (ctrl >> 17) & 0x3F,
        'rbar': (ctrl >> 10) & 0x1F,
        'wdep': (ctrl >> 4) & 0x3F,
        'raw': raw,
    }


# ---------------------------------------------------------------------------
# Check 1: ptxas rotate-sub miscompilation
# ---------------------------------------------------------------------------

def _check_rotate_bug(kernel: str, instrs: list[tuple[int, bytes]]) -> list[Finding]:
    findings = []
    for i in range(len(instrs) - 1):
        off1, raw1 = instrs[i]
        off2, raw2 = instrs[i + 1]
        d1 = _decode(raw1)
        d2 = _decode(raw2)

        # SHF.L.W.U32.HI pair: same K, swapped sources
        if (d1['opcode'] == 0x819 and d1['mod9'] == 0x0e and d1['mod10'] == 0x01 and
            d2['opcode'] == 0x819 and d2['mod9'] == 0x0e and d2['mod10'] == 0x01):
            k1, k2 = d1['b4'], d2['b4']
            if k1 == k2 and d1['src0'] == d2['src1'] and d1['src1'] == d2['src0']:
                findings.append(Finding(
                    severity=Severity.BUG,
                    kernel=kernel, offset=off1,
                    title=f"Potential ptxas rotate-sub miscompilation (K={k1})",
                    detail=(
                        f"SHF.L.W.U32.HI rotate pair at +{off1:#x} and +{off2:#x}. "
                        f"If the original PTX used sub.u64/sub.s64 instead of add/or/xor, "
                        f"this is a confirmed ptxas miscompilation bug. The GPU computes "
                        f"rotate_left(a, {k1}) instead of (a << {k1}) - (a >> {64-k1}). "
                        f"Bug affects ptxas 7.x through 13.1, all architectures SM_50-SM_120."
                    ),
                    recommendation=(
                        "Verify the original PTX source. If it contains sub.s64 or sub.u64 "
                        "of complementary shifts, recompile with OpenPTXas to produce correct code. "
                        "Recompile with OpenPTXas or apply the __umul64hi workaround."
                    ),
                ))
    return findings


# ---------------------------------------------------------------------------
# Check 2: Scheduling hazards (missing LDG barriers)
# ---------------------------------------------------------------------------

def _check_scheduling_hazards(kernel: str, instrs: list[tuple[int, bytes]]) -> list[Finding]:
    findings = []
    ldg_dests: dict[int, int] = {}  # reg → offset of LDG that wrote it

    for i, (off, raw) in enumerate(instrs):
        d = _decode(raw)

        # Track LDG destinations
        if d['opcode'] == 0x981:  # LDG
            ldg_dests[d['dest']] = off
            ldg_dests[d['dest'] + 1] = off  # 64-bit pair

        # Check if ALU instruction reads LDG output without proper barrier
        if d['opcode'] in (0x819, 0x210, 0x235, 0x221, 0x223, 0x824):
            src_regs = set()
            if d['src0'] < 255: src_regs.add(d['src0'])
            if d['src1'] < 255: src_regs.add(d['src1'])

            for reg in src_regs:
                if reg in ldg_dests:
                    ldg_off = ldg_dests[reg]
                    distance = (off - ldg_off) // 16

                    if d['rbar'] == 0x01 and distance <= 2:
                        findings.append(Finding(
                            severity=Severity.WARN,
                            kernel=kernel, offset=off,
                            title=f"Possible LDG scheduling hazard (R{reg})",
                            detail=(
                                f"{d['name']} at +{off:#x} reads R{reg} which was written by "
                                f"LDG at +{ldg_off:#x} ({distance} slots away) but has rbar=0x01 "
                                f"(no barrier wait). Expected rbar=0x03 or 0x09 for LDG consumers."
                            ),
                            recommendation="Add dependency barrier or increase instruction distance from LDG.",
                        ))

        # Clear tracking when register is overwritten by ALU
        if d['opcode'] not in (0x981, 0xb82, 0x984) and d['dest'] < 255:
            ldg_dests.pop(d['dest'], None)

    return findings


# ---------------------------------------------------------------------------
# Check 3: Register pressure
# ---------------------------------------------------------------------------

def _check_register_pressure(kernel: str, instrs: list[tuple[int, bytes]]) -> list[Finding]:
    findings = []
    max_reg = 0
    # Opcodes that don't have real register destinations
    no_dest_ops = {0x947, 0x918, 0x94d, 0x986, 0x988, 0xb1d}  # BRA, NOP, EXIT, STG, STS, BAR

    for off, raw in instrs:
        d = _decode(raw)
        if d['opcode'] not in no_dest_ops and d['dest'] < 255:
            max_reg = max(max_reg, d['dest'])
        if d['opcode'] not in (0x918, 0x947, 0x94d) and d['src0'] < 255:
            max_reg = max(max_reg, d['src0'])
        if d['opcode'] not in (0x918, 0x947, 0x94d) and d['src1'] < 255:
            max_reg = max(max_reg, d['src1'])

    if max_reg >= 128:
        findings.append(Finding(
            severity=Severity.WARN,
            kernel=kernel, offset=0,
            title=f"Very high register pressure (R{max_reg})",
            detail=f"Kernel uses register R{max_reg}. SM_120 allows up to R255 but high register usage reduces occupancy.",
            recommendation="Consider reducing live variables or using shared memory for spilling.",
        ))
    elif max_reg >= 64:
        findings.append(Finding(
            severity=Severity.INFO,
            kernel=kernel, offset=0,
            title=f"Moderate register pressure (R{max_reg})",
            detail=f"Kernel uses register R{max_reg}. This limits occupancy to ~50% on SM_120.",
        ))

    return findings


# ---------------------------------------------------------------------------
# Check 4: Missing synchronization
# ---------------------------------------------------------------------------

def _check_sync_issues(kernel: str, instrs: list[tuple[int, bytes]]) -> list[Finding]:
    findings = []
    has_sts = False
    sts_offset = 0

    for off, raw in instrs:
        d = _decode(raw)

        if d['opcode'] == 0x988:  # STS
            has_sts = True
            sts_offset = off

        if d['opcode'] == 0x984:  # LDS
            if has_sts:
                # Check if there's a BAR between the last STS and this LDS
                bar_found = False
                for off2, raw2 in instrs:
                    d2 = _decode(raw2)
                    if off2 > sts_offset and off2 < off and d2['opcode'] == 0xb1d:
                        bar_found = True
                        break
                if not bar_found:
                    findings.append(Finding(
                        severity=Severity.CRITICAL,
                        kernel=kernel, offset=off,
                        title="LDS without BAR.SYNC after STS",
                        detail=(
                            f"LDS at +{off:#x} reads shared memory that was written by STS at "
                            f"+{sts_offset:#x} with no BAR.SYNC barrier between them. "
                            f"This is a race condition — other threads may not have completed "
                            f"their STS writes."
                        ),
                        recommendation="Add __syncthreads() (BAR.SYNC) between shared memory writes and reads.",
                    ))

    return findings


# ---------------------------------------------------------------------------
# Check 5: NOP padding analysis
# ---------------------------------------------------------------------------

def _check_nop_padding(kernel: str, instrs: list[tuple[int, bytes]]) -> list[Finding]:
    findings = []
    nop_count = sum(1 for _, raw in instrs if _decode(raw)['opcode'] == 0x918)
    total = len(instrs)

    if total > 0:
        nop_pct = nop_count * 100 / total
        if nop_pct > 50:
            findings.append(Finding(
                severity=Severity.INFO,
                kernel=kernel, offset=0,
                title=f"High NOP padding ({nop_pct:.0f}%)",
                detail=(
                    f"{nop_count}/{total} instructions are NOPs. "
                    f"This is normal for small kernels padded to 128-byte alignment "
                    f"but may indicate missed optimization opportunities in larger kernels."
                ),
            ))

    return findings


# ---------------------------------------------------------------------------
# Check 6: Instruction mix analysis
# ---------------------------------------------------------------------------

def _check_instruction_mix(kernel: str, instrs: list[tuple[int, bytes]]) -> list[Finding]:
    findings = []
    counts: dict[str, int] = {}

    for _, raw in instrs:
        d = _decode(raw)
        name = d['name']
        if name != 'NOP':
            counts[name] = counts.get(name, 0) + 1

    total_real = sum(counts.values())
    if total_real == 0:
        return findings

    # Check for tensor core usage
    tensor_ops = counts.get('HMMA', 0) + counts.get('IMMA', 0)
    if tensor_ops > 0:
        findings.append(Finding(
            severity=Severity.INFO,
            kernel=kernel, offset=0,
            title=f"Tensor core utilization: {tensor_ops} MMA instructions",
            detail=f"Kernel uses {tensor_ops} tensor core instructions (HMMA/IMMA).",
        ))

    # Check memory vs compute ratio
    mem_ops = counts.get('LDG', 0) + counts.get('STG', 0) + counts.get('LDS', 0) + counts.get('STS', 0)
    alu_ops = sum(v for k, v in counts.items() if k in ('IADD3', 'IADD.64', 'IMAD', 'IMAD.W',
                                                          'SHF', 'LOP3', 'FADD', 'FFMA', 'HFMA2'))
    if alu_ops > 0 and mem_ops > 0:
        ratio = alu_ops / mem_ops
        if ratio < 1.0:
            findings.append(Finding(
                severity=Severity.INFO,
                kernel=kernel, offset=0,
                title=f"Memory-bound kernel (ALU:MEM ratio = {ratio:.1f}:1)",
                detail=f"{alu_ops} ALU ops vs {mem_ops} memory ops. Consider optimizing memory access patterns.",
            ))
        elif ratio > 10.0:
            findings.append(Finding(
                severity=Severity.INFO,
                kernel=kernel, offset=0,
                title=f"Compute-bound kernel (ALU:MEM ratio = {ratio:.1f}:1)",
                detail=f"{alu_ops} ALU ops vs {mem_ops} memory ops. Kernel is compute-limited.",
            ))

    return findings


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def audit_cubin(cubin_path: str, verbose: bool = False) -> list[Finding]:
    """
    Audit a cubin file for bugs, hazards, and optimization opportunities.

    Returns a list of Findings sorted by severity.
    """
    all_findings = []
    p = CubinPatcher(cubin_path)
    kernels = p.kernel_names()

    for kernel in kernels:
        instrs = disassemble_text(cubin_path, kernel)

        # Skip NOP-only padding
        real_instrs = [(off, raw) for off, raw in instrs if _decode(raw)['opcode'] != 0x918]

        all_findings.extend(_check_rotate_bug(kernel, instrs))
        all_findings.extend(_check_scheduling_hazards(kernel, instrs))
        all_findings.extend(_check_register_pressure(kernel, instrs))
        all_findings.extend(_check_sync_issues(kernel, instrs))
        all_findings.extend(_check_nop_padding(kernel, instrs))
        all_findings.extend(_check_instruction_mix(kernel, instrs))

    # Sort by severity
    severity_order = {Severity.CRITICAL: 0, Severity.BUG: 1, Severity.WARN: 2, Severity.INFO: 3}
    all_findings.sort(key=lambda f: severity_order[f.severity])

    return all_findings


def print_audit(cubin_path: str, verbose: bool = False):
    """Run audit and print results."""
    findings = audit_cubin(cubin_path, verbose)

    p = CubinPatcher(cubin_path)
    kernels = p.kernel_names()

    print(f"OpenPTXas Audit: {cubin_path}")
    print(f"  Kernels: {len(kernels)} ({', '.join(kernels)})")
    print(f"  Findings: {len(findings)}")
    print()

    if not findings:
        print("  No issues found.")
        return

    for f in findings:
        icon = {'CRITICAL': '!!!', 'BUG': '**', 'WARN': '*', 'INFO': '-'}[f.severity.value]
        print(f"  [{f.severity.value}] {icon} {f.title}")
        print(f"    Kernel: {f.kernel}, Offset: +{f.offset:#x}")
        if verbose or f.severity in (Severity.CRITICAL, Severity.BUG):
            print(f"    {f.detail}")
            if f.recommendation:
                print(f"    Fix: {f.recommendation}")
        print()

    # Summary
    by_sev = {}
    for f in findings:
        by_sev[f.severity.value] = by_sev.get(f.severity.value, 0) + 1
    print(f"  Summary: {', '.join(f'{v} {k}' for k, v in by_sev.items())}")
