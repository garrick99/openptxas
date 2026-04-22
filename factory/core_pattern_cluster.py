"""Cluster OpenPTXas dossiers by the core 'dangerous operation' in the body.

Most dossiers have unique SASS and unique PTX body shapes because the
danger generator intentionally wraps each core pattern in random filler
ops.  To see the TRUE number of underlying bugs, we look past the
filler and identify which of a small set of known-danger PTX patterns
is present in the body:

  POOL_ZERO_MUL    — mul.lo.{s32,u32} _, %r3, <imm > 0x7FFF>
                      (literal-pool-zero class, already characterized)
  BUG2_SHAPE       — or.b32 _, _, <msb-set>; shr.s32; shr.u32
                      (known ptxas constant-fold; ours typically matches
                       spec, but some compositions surface ours-wrong)
  SHIFT_BOUNDARY   — shl.b32 then shr.{s32,u32} with amounts near 0/31
  SIGN_FLIP_CHAIN  — xor/or/and with 0x80000000 then signed ALU
  BFI_UB           — bfi.b32 with pos+len > 32 (PTX UB)
  BFE_S32_OOR      — bfe.s32 with start >= 32 (Bug 1 shape, already known)

Each dossier is assigned to the FIRST matching pattern.  If none match,
'OTHER' captures novel shapes — those are the interesting-investigate
candidates.
"""
from __future__ import annotations
import re, sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from factory import db

_PTX_BLOCK = re.compile(r'```ptx\s*\n(.*?)```', re.DOTALL)

_RE_MUL_LARGE = re.compile(
    r'mul\.lo\.[us]32\s+%r\d+\s*,\s*%r\d+\s*,\s*(\d+|0x[0-9a-fA-F]+|-\d+)')
_RE_OR_MSB = re.compile(
    r'or\.b32\s+%r\d+\s*,\s*%r\d+\s*,\s*(2147483648|-2147483648|0x80000000|0xFFFFFFFF|4294967295|2147483647|0x7FFFFFFF)')
_RE_SHR_S32 = re.compile(r'shr\.s32\s+%r\d+\s*,\s*%r\d+\s*,\s*(\d+)')
_RE_SHR_U32 = re.compile(r'shr\.u32\s+%r\d+\s*,\s*%r\d+\s*,\s*(\d+)')
_RE_SHL = re.compile(r'shl\.b32\s+%r\d+\s*,\s*%r\d+\s*,\s*(\d+)')
_RE_BFI = re.compile(
    r'bfi\.b32\s+%r\d+\s*,\s*%r\d+\s*,\s*%r\d+\s*,\s*(\d+)\s*,\s*(\d+)')
_RE_BFE_S32 = re.compile(
    r'bfe\.s32\s+%r\d+\s*,\s*%r\d+\s*,\s*(\d+)\s*,\s*(\d+)')
_RE_XOR_MSB = re.compile(
    r'xor\.b32\s+%r\d+\s*,\s*%r\d+\s*,\s*(2147483648|-2147483648|0x80000000)')


def _body(ptx: str) -> list[str]:
    lines = ptx.splitlines()
    body = []
    in_body = False
    for ln in lines:
        s = ln.strip()
        if 'ld.global.u32 %r3' in s:
            in_body = True
            continue
        if 'ld.param.u64 %rd3' in s:
            break
        if in_body and s:
            body.append(s)
    return body


def classify(body: list[str]) -> str:
    text = '\n'.join(body)
    # BFE.S32 OOR = Bug 1 shape
    for m in _RE_BFE_S32.finditer(text):
        if int(m.group(1)) >= 32:
            return 'BFE_S32_OOR (Bug1)'
    # BFI UB
    for m in _RE_BFI.finditer(text):
        if int(m.group(1)) + int(m.group(2)) > 32:
            return 'BFI_UB'
    # Bug 2 shape: or-MSB then shr.s32 then shr.u32
    if _RE_OR_MSB.search(text) and _RE_SHR_S32.search(text) and _RE_SHR_U32.search(text):
        return 'BUG2_SHAPE'
    # Pool-zero mul: mul.lo.{s,u}32 with immediate > 0x7FFF
    for m in _RE_MUL_LARGE.finditer(text):
        imm_str = m.group(1)
        try:
            imm = int(imm_str, 16) if imm_str.startswith(('0x', '-0x')) else int(imm_str)
        except ValueError:
            continue
        # Consider power-of-2 shifts > 15 as pool-path, plus anything above 0x7FFF
        if abs(imm) > 0x7FFF:
            return 'POOL_ZERO_MUL'
    # Shift boundary: shl+shr with amounts at/near 0 or 31
    if _RE_SHL.search(text) and (_RE_SHR_S32.search(text) or _RE_SHR_U32.search(text)):
        return 'SHIFT_BOUNDARY'
    # Sign-flip chain
    if _RE_XOR_MSB.search(text):
        return 'SIGN_FLIP_CHAIN'
    return 'OTHER'


def main():
    root = db.OPENPTXAS_REPORT_DIR
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    buckets: dict[str, list[Path]] = defaultdict(list)
    for d in dirs:
        rp = d / 'REPORT.md'
        if not rp.exists(): continue
        m = _PTX_BLOCK.search(rp.read_text(encoding='utf-8'))
        if not m: continue
        body = _body(m.group(1))
        if not body: continue
        k = classify(body)
        buckets[k].append(d)

    ordered = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    print(f'{"count":>6}  {"pattern":<28} {"canonical":<40}  {"status"}')
    print(f'{"-"*6:>6}  {"-"*28:<28} {"-"*40:<40}  {"-"*25}')
    status_map = {
        'POOL_ZERO_MUL':    'parked in known_residuals.md',
        'BUG2_SHAPE':       'ptxas Bug 2 variant, already submitted',
        'BFE_S32_OOR (Bug1)': 'ptxas Bug 1 variant, already submitted',
        'BFI_UB':           'PTX undefined behavior, not fixable',
        'SHIFT_BOUNDARY':   '*** possibly NEW OpenPTXas bug class',
        'SIGN_FLIP_CHAIN':  '*** possibly NEW OpenPTXas bug class',
        'OTHER':            '*** needs hand review',
    }
    for k, items in ordered:
        print(f'{len(items):>6}  {k:<28} {items[0].name:<40}  {status_map.get(k, "")}')


if __name__ == '__main__':
    main()
