"""Bug classification: extract a canonical op-family signature from PTX.

Two PTXs with the same multiset of integer ops (ignoring operand register
numbers and immediate values) get the same signature.  This lets us
cluster raw fuzzer artifacts and ask questions like:
  - which op combinations produce divergences?
  - which minimal-PTX signatures are most common (= most frequent bug)?
"""
import re
from collections import Counter
from pathlib import Path

_OP_RE = re.compile(r'^\s*(?:@!?%p\d+\s+)?([a-z][a-z0-9.]*)(?:\s|;|$)')

# Ops that are fuzz-kernel plumbing, not payload.
_PLUMBING = frozenset([
    'ret', 'bra', 'bar.sync',
    'mov.u32', 'ld.param.u32', 'ld.param.u64', 'ld.global.u32',
    'st.global.u32', 'cvt.u64.u32', 'shl.b64', 'add.u64', 'setp.ge.u32',
])


def body_ops(ptx: str) -> list[str]:
    """Return the sequence of op-family tokens appearing between the
    input-load and the output-store (the fuzz-body proper)."""
    lines = ptx.splitlines()
    bstart = bend = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if bstart is None and s.startswith('ld.global.u32 %r3'):
            bstart = i + 1
        elif bstart is not None and s.startswith('ld.param.u64 %rd3'):
            bend = i
            break
    if bstart is None or bend is None:
        return []
    ops = []
    for ln in lines[bstart:bend]:
        m = _OP_RE.match(ln)
        if m:
            op = m.group(1)
            if op not in _PLUMBING:
                ops.append(op)
    return ops


def signature(ptx: str) -> str:
    """Canonical multiset signature: 'op*count,op*count,...' sorted."""
    c = Counter(body_ops(ptx))
    return ','.join(f'{op}*{n}' for op, n in sorted(c.items()))


def family(op: str) -> str:
    """Collapse 'add.u32' / 'add.s32' / 'add.cc.u32' -> 'add'."""
    return op.split('.', 1)[0]


def family_signature(ptx: str) -> str:
    """Coarser signature: one token per op family (add, mul, cvt, ...)."""
    c = Counter(family(op) for op in body_ops(ptx))
    return ','.join(f'{f}*{n}' for f, n in sorted(c.items()))


if __name__ == '__main__':
    import sys
    for path in sys.argv[1:]:
        ptx = Path(path).read_text()
        print(f'{path}:')
        print(f'  sig:    {signature(ptx)}')
        print(f'  family: {family_signature(ptx)}')
