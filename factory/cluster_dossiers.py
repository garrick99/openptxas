"""Cluster OpenPTXas bug dossiers by SASS diff fingerprint.

For each dossier in _openptxas_bugs_auto/, extract the canonical PTX
from REPORT.md, recompile ours + theirs, byte-diff the .text.fuzz
sections, and bucket by the first-differing-instruction fingerprint.

Output: "N dossiers collapse to K unique bug fingerprints."

Usage:
    python -m factory.cluster_dossiers
"""
from __future__ import annotations
import os, re, sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from factory import db
from factory.sass_diff import first_diff, pretty_op, InstrDiff
from fuzzer.oracle import compile_ours, compile_theirs


_PTX_BLOCK = re.compile(r'```ptx\s*\n(.*?)```', re.DOTALL)


def extract_ptx(report_md: str) -> str:
    m = _PTX_BLOCK.search(report_md)
    if not m:
        raise ValueError('no ```ptx``` block found in REPORT.md')
    return m.group(1).strip()


def cluster_dossier_dirs(dirs: list[Path]) -> dict[str, list[tuple[Path, InstrDiff]]]:
    buckets: dict[str, list[tuple[Path, InstrDiff]]] = defaultdict(list)
    errors = 0
    for d in dirs:
        rpath = d / 'REPORT.md'
        if not rpath.exists():
            continue
        try:
            ptx = extract_ptx(rpath.read_text(encoding='utf-8'))
        except Exception:
            errors += 1
            continue
        cubin_ours, err_o = compile_ours(ptx)
        if err_o or not cubin_ours:
            buckets[f'UNCOMPILABLE_OURS:{(err_o or "").strip()[:30]}'].append((d, None))
            continue
        cubin_theirs, err_t = compile_theirs(ptx)
        if err_t or not cubin_theirs:
            buckets[f'UNCOMPILABLE_THEIRS:{(err_t or "").strip()[:30]}'].append((d, None))
            continue
        diff = first_diff(cubin_ours, cubin_theirs)
        if diff is None:
            buckets['IDENTICAL'].append((d, None))
            continue
        buckets[diff.fingerprint()].append((d, diff))
    if errors:
        print(f'[cluster] {errors} dossiers with missing/unparseable REPORT.md', file=sys.stderr)
    return buckets


def main():
    root = db.OPENPTXAS_REPORT_DIR
    if not root.exists():
        print(f'no dossier dir at {root}'); return
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    print(f'[cluster] scanning {len(dirs)} dossiers...\n')
    buckets = cluster_dossier_dirs(dirs)

    # Sort clusters by size descending
    ordered = sorted(buckets.items(), key=lambda kv: -len(kv[1]))

    print(f'{"count":>6} {"fingerprint":<70} {"canonical"}')
    print(f'{"-"*6:>6} {"-"*70:<70} {"-"*40}')
    for fp, items in ordered:
        canonical = items[0][0].name
        diff = items[0][1]
        if diff:
            op = pretty_op(diff.ours_opcode)
            if diff.theirs_opcode != diff.ours_opcode:
                op = f'{pretty_op(diff.ours_opcode)}=>{pretty_op(diff.theirs_opcode)}'
        print(f'{len(items):>6} {fp[:70]:<70} {canonical}')

    print(f'\n[cluster] {len(dirs)} dossiers collapse to {len(buckets)} unique fingerprints.')


if __name__ == '__main__':
    main()
