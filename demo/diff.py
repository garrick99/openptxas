"""Instruction-level diff between OURS and PTXAS SASS."""
from __future__ import annotations

from demo.compare import opcode_name


# Known transformation explanations keyed by (ours_opcode, ptxas_opcode) patterns
_TRANSFORM_EXPLAIN = {
    # LOP3.IMM replaces IADD3.IMM + LOP3 (materialize + bitwise op)
    'LOP3.IMM': 'LOP3 inline immediate: eliminated separate constant materialization (IMAD-FUSE-1)',
    # Predicated IMAD replaces MOV + @pred IADD3
    '@pred IMAD': 'predicated IMAD fusion: mov + @pred add + mul absorbed into one instruction (HARD-FINISH-1)',
    # IMAD.WIDE replaces separate cvt + shl
    'IMAD.WIDE': 'IMAD.WIDE fusion: cvt.u64.u32 + shl.b64 absorbed into single wide multiply',
    # IADD.64 R-UR replaces IADD3 + IADD3.X
    'IADD64.RUR': 'IADD.64 R-UR: 64-bit add with uniform register in one instruction',
}


def diff_streams(ours_ops: list[dict], ptxas_ops: list[dict]) -> list[dict]:
    """Compare two non-NOP instruction streams.

    Returns a list of diff records:
      {'type': 'match'|'ours_only'|'ptxas_only'|'differ',
       'ours': {...}|None, 'ptxas': {...}|None,
       'explanation': str|None}
    """
    # Build opcode sequences
    ours_seq = [(opcode_name(o['opcode']), o) for o in ours_ops]
    ptxas_seq = [(opcode_name(o['opcode']), o) for o in ptxas_ops]

    # Simple LCS-based diff
    m, n = len(ours_seq), len(ptxas_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if ours_seq[i][0] == ptxas_seq[j][0]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    result = []
    i, j = 0, 0
    while i < m or j < n:
        if i < m and j < n and ours_seq[i][0] == ptxas_seq[j][0]:
            result.append({
                'type': 'match',
                'ours': ours_seq[i][1],
                'ptxas': ptxas_seq[j][1],
                'explanation': None,
            })
            i += 1; j += 1
        elif j >= n or (i < m and dp[i + 1][j] >= dp[i][j + 1]):
            oname = ours_seq[i][0]
            expl = None
            for key, val in _TRANSFORM_EXPLAIN.items():
                if key in oname:
                    expl = val
                    break
            result.append({
                'type': 'ours_only',
                'ours': ours_seq[i][1],
                'ptxas': None,
                'explanation': expl,
            })
            i += 1
        else:
            result.append({
                'type': 'ptxas_only',
                'ours': None,
                'ptxas': ptxas_seq[j][1],
                'explanation': None,
            })
            j += 1

    return result


def summarize_transforms(diff_records: list[dict]) -> list[str]:
    """Extract human-readable transformation notes from a diff."""
    notes = []
    ours_only_ops = [opcode_name(r['ours']['opcode'])
                     for r in diff_records if r['type'] == 'ours_only']
    ptxas_only_ops = [opcode_name(r['ptxas']['opcode'])
                      for r in diff_records if r['type'] == 'ptxas_only']

    # Detect specific patterns
    if 'LOP3.IMM' in ours_only_ops:
        notes.append('OURS uses LOP3.IMM (inline immediate) -- saves 1 instruction per bitwise-with-constant')

    if any('IMAD' in o for o in ours_only_ops):
        for r in diff_records:
            if r['type'] == 'ours_only' and r['explanation']:
                if r['explanation'] not in notes:
                    notes.append(r['explanation'])

    if 'LEA.RUR' in ptxas_only_ops:
        notes.append('PTXAS uses LEA R-UR for address calc (instruction-count neutral vs IMAD.WIDE + IADD.64)')

    if 'HFMA2' in ptxas_only_ops:
        notes.append('PTXAS uses HFMA2 constant-load trick (extra setup instruction)')

    if 'SHF' in ptxas_only_ops:
        notes.append('PTXAS uses SHF for stride address calc (structural difference)')

    if 'IMAD.424' in ptxas_only_ops:
        notes.append('PTXAS uses IMAD.424 (32-bit addend form) for fused mul+add')

    if not notes:
        if len(ours_only_ops) == 0 and len(ptxas_only_ops) == 0:
            notes.append('Instruction streams are identical')
        else:
            notes.append(f'OURS unique: {", ".join(ours_only_ops) or "(none)"}')
            notes.append(f'PTXAS unique: {", ".join(ptxas_only_ops) or "(none)"}')

    return notes
