"""Cluster dossiers by normalized PTX body shape.

Extract the PTX body (lines between `ld.global.u32 %r3` and
`ld.param.u64 %rd3`), rename registers and immediates into canonical
slots (%r3 stays as the input, every other reg becomes %ta, %tb, ...
and immediates become :IMM:), then hash.  Two dossiers with the same
shape hash are semantic siblings — they exercise the same bug family.
"""
from __future__ import annotations
import hashlib, re, sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from factory import db

_PTX_BLOCK = re.compile(r'```ptx\s*\n(.*?)```', re.DOTALL)


def extract_body(report_md: str) -> list[str]:
    m = _PTX_BLOCK.search(report_md)
    if not m:
        return []
    lines = m.group(1).strip().splitlines()
    body = []
    in_body = False
    for ln in lines:
        s = ln.strip()
        if 'ld.global.u32 %r3' in s:
            in_body = True
            continue
        if 'ld.param.u64 %rd3' in s:
            break
        if in_body and s and not s.startswith('//'):
            body.append(s)
    return body


_REG_RE = re.compile(r'%(r|rd|p)(\d+)')
_IMM_RE = re.compile(r'\b(-?\d{4,}|0x[0-9a-fA-F]+)\b')


def normalize(body: list[str]) -> str:
    """Map %rN→%tN (sequential, stable), %rdN→%TN, %pN→%PN.  Keep %r3.
    Replace multi-digit literals with :IMM:."""
    reg_map: dict[str, str] = {'%r3': '%r3'}
    rd_map: dict[str, str] = {}
    p_map: dict[str, str] = {}
    out_lines = []
    rc = rd_c = p_c = 0
    for line in body:
        # Strip trailing semicolons and whitespace
        line = line.rstrip(';').strip()
        def sub_reg(m):
            nonlocal rc, rd_c, p_c
            kind, num = m.group(1), m.group(2)
            key = f'%{kind}{num}'
            if kind == 'r':
                if key == '%r3': return '%r3'
                if key not in reg_map:
                    reg_map[key] = f'%t{rc}'; rc += 1
                return reg_map[key]
            if kind == 'rd':
                if key not in rd_map:
                    rd_map[key] = f'%T{rd_c}'; rd_c += 1
                return rd_map[key]
            if kind == 'p':
                if key not in p_map:
                    p_map[key] = f'%P{p_c}'; p_c += 1
                return p_map[key]
            return m.group(0)
        line = _REG_RE.sub(sub_reg, line)
        line = _IMM_RE.sub(':IMM:', line)
        out_lines.append(line)
    return '\n'.join(out_lines)


def main():
    root = db.OPENPTXAS_REPORT_DIR
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    buckets: dict[str, list[Path]] = defaultdict(list)
    for d in dirs:
        rp = d / 'REPORT.md'
        if not rp.exists(): continue
        body = extract_body(rp.read_text(encoding='utf-8'))
        if not body:
            buckets['NO_BODY'].append(d); continue
        norm = normalize(body)
        h = hashlib.sha256(norm.encode()).hexdigest()[:10]
        buckets[(h, norm)].append(d)

    ordered = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    print(f'[body-cluster] scanning {len(dirs)} dossiers, {len(buckets)} unique body shapes\n')
    for i, (key, items) in enumerate(ordered):
        if key == 'NO_BODY':
            print(f'[UNPARSEABLE] {len(items)} dossiers')
            continue
        h, norm = key
        print(f'=== cluster {i+1}: n={len(items)}  canonical={items[0].name} ===')
        for line in norm.splitlines():
            print(f'    {line}')
        print()
        if i >= 14:  # top 15 only
            remainder = sum(len(v) for _, v in ordered[i+1:])
            print(f'... +{len(ordered)-i-1} more clusters covering {remainder} dossiers')
            break


if __name__ == '__main__':
    main()
