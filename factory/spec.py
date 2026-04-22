"""PTX spec simulator — pure-Python reference.

Evaluates a fuzz-kernel body lane-by-lane for a given input buffer.
Covers the integer-ALU subset our generators emit, plus bfe / bfi /
shifts / cvt / selp / setp.  Raises `Unsupported` if any body
instruction falls outside the subset — the oracle daemon catches this
and marks the program as 'unsupported'.

Kernel shape (from fuzzer.generator):

    ...prologue...
    ld.global.u32 %r3, [%rd2];    <-- input for this lane
    <body instructions>
    ld.param.u64 %rd3, [p_out];
    add.u64 %rd4, %rd3, %rd1;
    st.global.u32 [%rd4], %rN;    <-- the "output" register

We identify the store's value register, simulate the body for each
input word, and return 32 u32 expected outputs.
"""
from __future__ import annotations
import re, struct
from typing import Optional


class Unsupported(Exception):
    """Body contains an instruction the simulator can't evaluate."""


_BODY_START_RE = re.compile(r'^\s*ld\.global\.u32\s+%r3\s*,')
_BODY_END_RE   = re.compile(r'^\s*ld\.param\.u64\s+%rd3\s*,')
_STORE_RE      = re.compile(r'^\s*st\.global\.u32\s+\[\s*%rd4\s*\]\s*,\s*(%r\d+)')
_INSTR_RE      = re.compile(
    r'^\s*(?:@(!?)(%p\d+)\s+)?([a-z][a-z0-9.]*)\s+(.*?)\s*;?\s*$')


def _u32(x: int) -> int: return x & 0xFFFFFFFF
def _s32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x
def _u64(x: int) -> int: return x & 0xFFFFFFFFFFFFFFFF


def _parse_reg(tok: str):
    """Return (kind, name) where kind ∈ {'r','rd','p'} or ('imm', value)."""
    tok = tok.strip().rstrip(',')
    if tok.startswith('%r') and not tok.startswith('%rd'):
        return ('r', tok)
    if tok.startswith('%rd'):
        return ('rd', tok)
    if tok.startswith('%p'):
        return ('p', tok)
    # Immediate — Python int literal with optional 0x
    try:
        if tok.startswith('0x') or tok.startswith('-0x'):
            return ('imm', int(tok, 16))
        return ('imm', int(tok, 0))
    except ValueError:
        raise Unsupported(f'cannot parse operand: {tok!r}')


def _split_body(ptx: str):
    """Return (body_lines, out_reg_name) or raise Unsupported."""
    lines = ptx.splitlines()
    bstart = bend = None
    for i, ln in enumerate(lines):
        if bstart is None and _BODY_START_RE.match(ln):
            bstart = i
        if bstart is not None and bend is None and _BODY_END_RE.match(ln):
            bend = i
    if bstart is None or bend is None:
        raise Unsupported('cannot locate kernel body')
    body = lines[bstart+1:bend]
    # Find the store
    out_reg = None
    for ln in lines[bend:]:
        m = _STORE_RE.match(ln)
        if m:
            out_reg = m.group(1); break
    if out_reg is None:
        raise Unsupported('cannot find st.global.u32')
    return body, out_reg


def _binop(op: str, a: int, b: int) -> int:
    """Evaluate a binary opcode on two u32 values."""
    if op == 'add.u32' or op == 'add.s32':
        return _u32(a + b)
    if op == 'sub.u32' or op == 'sub.s32':
        return _u32(a - b)
    if op == 'mul.lo.u32' or op == 'mul.lo.s32':
        return _u32(a * b)
    if op == 'mul.hi.u32':
        return _u32((a * b) >> 32)
    if op == 'mul.hi.s32':
        return _u32((_s32(a) * _s32(b)) >> 32) & 0xFFFFFFFF
    if op == 'and.b32':
        return _u32(a & b)
    if op == 'or.b32':
        return _u32(a | b)
    if op == 'xor.b32':
        return _u32(a ^ b)
    if op == 'min.u32':
        return min(_u32(a), _u32(b))
    if op == 'max.u32':
        return max(_u32(a), _u32(b))
    if op == 'min.s32':
        return _u32(min(_s32(a), _s32(b)))
    if op == 'max.s32':
        return _u32(max(_s32(a), _s32(b)))
    if op == 'shl.b32':
        n = _u32(b) & 0x1F
        return _u32(a << n)
    if op == 'shr.u32':
        n = _u32(b) & 0x1F
        return _u32(_u32(a) >> n)
    if op == 'shr.s32':
        n = _u32(b) & 0x1F
        # arithmetic right shift (sign-preserving)
        sv = _s32(a) >> n
        return _u32(sv)
    if op == 'add.sat.s32':
        return _sat_s32(_s32(a) + _s32(b))
    if op == 'sub.sat.s32':
        return _sat_s32(_s32(a) - _s32(b))
    raise Unsupported(op)


def _sat_s32(x: int) -> int:
    """Saturate a Python int to signed 32-bit range, return as u32."""
    if x > 0x7FFFFFFF: return 0x7FFFFFFF
    if x < -0x80000000: return _u32(-0x80000000)
    return _u32(x)


def _shf_l(a: int, b: int, c: int, clamp: bool) -> int:
    """shf.l: funnel-shift left of concat(b:a) by c; return upper 32 bits.
    wrap: n = c & 0x1f.  clamp: n = min(c, 32)."""
    c = _u32(c)
    n = min(c, 32) if clamp else (c & 0x1F)
    a = _u32(a); b = _u32(b)
    if n == 0: return b
    if n == 32: return a
    return _u32((b << n) | (a >> (32 - n)))


def _shf_r(a: int, b: int, c: int, clamp: bool) -> int:
    """shf.r: funnel-shift right of concat(b:a) by c; return lower 32 bits."""
    c = _u32(c)
    n = min(c, 32) if clamp else (c & 0x1F)
    a = _u32(a); b = _u32(b)
    if n == 0: return a
    if n == 32: return b
    return _u32((a >> n) | (b << (32 - n)))


def _setp_cmp(cmp_: str, a: int, b: int) -> bool:
    if cmp_ == 'eq': return _u32(a) == _u32(b)
    if cmp_ == 'ne': return _u32(a) != _u32(b)
    if cmp_ == 'lt': return _s32(a) <  _s32(b)
    if cmp_ == 'le': return _s32(a) <= _s32(b)
    if cmp_ == 'gt': return _s32(a) >  _s32(b)
    if cmp_ == 'ge': return _s32(a) >= _s32(b)
    raise Unsupported('cmp ' + cmp_)


def _setp_u_cmp(cmp_: str, a: int, b: int) -> bool:
    a = _u32(a); b = _u32(b)
    return {'eq': a == b, 'ne': a != b,
            'lt': a < b,  'le': a <= b,
            'gt': a > b,  'ge': a >= b}.get(cmp_, None) \
        if cmp_ in ('eq','ne','lt','le','gt','ge') else None


def _bfe_u32(a: int, pos: int, length: int) -> int:
    if length == 0: return 0
    pos &= 0xFF
    length &= 0xFF
    if pos >= 32:
        return 0
    a = _u32(a)
    length = min(length, 32 - pos)
    mask = (1 << length) - 1
    return (a >> pos) & mask


def _bfe_s32(a: int, pos: int, length: int) -> int:
    # PTX spec: result is sign-extended.  When pos >= 32, all bits are
    # sign-bit replications of a[31].
    if length == 0: return 0
    pos &= 0xFF
    length &= 0xFF
    sign = (a >> 31) & 1
    if pos >= 32:
        return 0xFFFFFFFF if sign else 0
    # Extract field, then sign-extend from top bit of the extracted field.
    effective_len = min(length, 32 - pos)
    mask = (1 << effective_len) - 1
    val = (_u32(a) >> pos) & mask
    # Sign-extend: top bit of the extracted field becomes the sign
    # (but if field goes beyond bit 31, the top bits are sign-bit replications of a[31]).
    if effective_len < length:
        # Part of the field is beyond bit 31 -> sign-bit fills
        full_mask = (1 << length) - 1
        if sign:
            val |= (full_mask << effective_len) & full_mask
        top_bit = sign
    else:
        top_bit = (val >> (effective_len - 1)) & 1
    if top_bit:
        # Sign-extend from bit `length`
        val |= 0xFFFFFFFF << length
    return _u32(val)


def _bfi_b32(a: int, b: int, pos: int, length: int) -> int:
    """Insert bits [0..length-1] of a into b starting at pos."""
    pos &= 0xFF
    length &= 0xFF
    if length == 0 or pos >= 32:
        return _u32(b)
    length = min(length, 32 - pos)
    mask = ((1 << length) - 1) << pos
    return _u32((_u32(b) & ~mask) | ((_u32(a) << pos) & mask))


def simulate(ptx: str, inputs: bytes) -> bytes:
    """Run the body on each of the 32 input u32 words; return 128 output bytes.

    Raises Unsupported if any body instruction is outside the handled subset.
    """
    body, out_reg = _split_body(ptx)
    n = len(inputs) // 4
    out = bytearray(n * 4)

    for lane in range(n):
        r3 = struct.unpack_from('<I', inputs, lane*4)[0]
        state = {'%r3': r3, '%rz': 0, '%rZ': 0}
        preds = {}
        for line in body:
            s = line.strip()
            if not s or s.startswith('//'):
                continue
            m = _INSTR_RE.match(s)
            if not m:
                continue
            neg_pred, pred_reg, op, args = m.group(1), m.group(2), m.group(3), m.group(4)
            # Predication: skip if predicate not satisfied
            if pred_reg is not None:
                pv = preds.get(pred_reg, False)
                if neg_pred == '!': pv = not pv
                if not pv: continue
            # Split args by commas
            operands = [x.strip() for x in args.split(',')]

            if op in ('mov.u32', 'mov.b32'):
                d = operands[0]
                kind, val = _parse_reg(operands[1])
                state[d] = state.get(operands[1], 0) if kind == 'r' else val
                continue

            if op == 'selp.b32':
                # selp.b32 d, a, b, p   =>  d = p ? a : b
                d = operands[0]
                kind_a, va = _parse_reg(operands[1])
                kind_b, vb = _parse_reg(operands[2])
                pv = preds.get(operands[3], False)
                a = state.get(operands[1], 0) if kind_a == 'r' else va
                b = state.get(operands[2], 0) if kind_b == 'r' else vb
                state[d] = _u32(a) if pv else _u32(b)
                continue

            if op.startswith('setp.'):
                # setp.<cmp>.<type>  pd, a, b                           (3-arg)
                # setp.<cmp>.<bool>.<type>  pd, a, b, {!}pred_in         (4-arg)
                # bool ∈ {and, or, xor}; fuse cmp result with pred_in.
                parts = op.split('.')
                if len(parts) < 3:
                    raise Unsupported(op)
                cmp_ = parts[1]
                bool_op = None
                ptype = parts[2]
                if len(parts) == 4 and parts[2] in ('and', 'or', 'xor'):
                    bool_op = parts[2]
                    ptype = parts[3]
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                kb, vb = _parse_reg(operands[2])
                a = state.get(operands[1], 0) if ka == 'r' else va
                b = state.get(operands[2], 0) if kb == 'r' else vb
                if ptype in ('u32', 'b32'):
                    base = {'eq': _u32(a)==_u32(b), 'ne': _u32(a)!=_u32(b),
                             'lt': _u32(a)<_u32(b), 'le': _u32(a)<=_u32(b),
                             'gt': _u32(a)>_u32(b), 'ge': _u32(a)>=_u32(b),
                             }.get(cmp_, None)
                elif ptype == 's32':
                    base = _setp_cmp(cmp_, a, b)
                else:
                    raise Unsupported(op)
                if base is None: raise Unsupported(op)
                if bool_op is not None:
                    # 4th operand is the predicate input, possibly negated
                    ptok = operands[3].strip()
                    neg = False
                    if ptok.startswith('!'):
                        neg = True; ptok = ptok[1:].strip()
                    pin = preds.get(ptok, False)
                    if neg: pin = not pin
                    if bool_op == 'and': base = base and pin
                    elif bool_op == 'or': base = base or pin
                    else: base = bool(base) ^ bool(pin)
                preds[d] = base
                continue

            if op.startswith('shf.'):
                # shf.{l,r}.{wrap,clamp}.b32 d, a, b, c
                parts = op.split('.')
                if len(parts) != 4 or parts[1] not in ('l', 'r') \
                        or parts[2] not in ('wrap', 'clamp') or parts[3] != 'b32':
                    raise Unsupported(op)
                if len(operands) != 4: raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                kb, vb = _parse_reg(operands[2])
                kc, vc = _parse_reg(operands[3])
                a = state.get(operands[1], 0) if ka == 'r' else va
                b = state.get(operands[2], 0) if kb == 'r' else vb
                c = state.get(operands[3], 0) if kc == 'r' else vc
                clamp = (parts[2] == 'clamp')
                state[d] = _shf_l(a, b, c, clamp) if parts[1] == 'l' \
                            else _shf_r(a, b, c, clamp)
                continue

            if op in ('mad.lo.u32', 'mad.lo.s32'):
                # mad.lo d, a, b, c  =>  (a*b) + c, low 32 bits
                if len(operands) != 4: raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                kb, vb = _parse_reg(operands[2])
                kc, vc = _parse_reg(operands[3])
                a = state.get(operands[1], 0) if ka == 'r' else va
                b = state.get(operands[2], 0) if kb == 'r' else vb
                c = state.get(operands[3], 0) if kc == 'r' else vc
                if op.endswith('s32'):
                    state[d] = _u32(_s32(a) * _s32(b) + _s32(c))
                else:
                    state[d] = _u32(_u32(a) * _u32(b) + _u32(c))
                continue

            if op in ('mul.wide.u32', 'mul.wide.s32'):
                # mul.wide d, a, b  where d is %rd<n> (64-bit).  Store the
                # full u64 in a side dict; if the store reads %rd we can
                # still report correctly.
                if len(operands) != 3: raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                kb, vb = _parse_reg(operands[2])
                a = state.get(operands[1], 0) if ka == 'r' else va
                b = state.get(operands[2], 0) if kb == 'r' else vb
                if op.endswith('s32'):
                    prod = _u64(_s32(a) * _s32(b))
                else:
                    prod = _u64(_u32(a) * _u32(b))
                state[d] = prod
                continue

            if op.startswith('bfe.'):
                t = op.split('.')[1]
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                kp, vp = _parse_reg(operands[2])
                kl, vl = _parse_reg(operands[3])
                a = state.get(operands[1], 0) if ka == 'r' else va
                p = state.get(operands[2], 0) if kp == 'r' else vp
                l = state.get(operands[3], 0) if kl == 'r' else vl
                if t == 's32':
                    state[d] = _bfe_s32(a, p, l)
                elif t == 'u32':
                    state[d] = _bfe_u32(a, p, l)
                else:
                    raise Unsupported(op)
                continue

            if op == 'bfi.b32':
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                kb, vb = _parse_reg(operands[2])
                kp, vp = _parse_reg(operands[3])
                kl, vl = _parse_reg(operands[4])
                a = state.get(operands[1], 0) if ka == 'r' else va
                b = state.get(operands[2], 0) if kb == 'r' else vb
                p = state.get(operands[3], 0) if kp == 'r' else vp
                l = state.get(operands[4], 0) if kl == 'r' else vl
                state[d] = _bfi_b32(a, b, p, l)
                continue

            # Binary ALU pattern: <op> d, a, b
            if op in ('add.u32','add.s32','sub.u32','sub.s32',
                       'mul.lo.u32','mul.lo.s32','mul.hi.u32','mul.hi.s32',
                       'and.b32','or.b32','xor.b32',
                       'min.u32','max.u32','min.s32','max.s32',
                       'shl.b32','shr.u32','shr.s32',
                       'add.sat.s32','sub.sat.s32'):
                if len(operands) != 3:
                    raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                kb, vb = _parse_reg(operands[2])
                a = state.get(operands[1], 0) if ka == 'r' else va
                b = state.get(operands[2], 0) if kb == 'r' else vb
                state[d] = _binop(op, a, b)
                continue

            if op.startswith('cvt.'):
                # cvt.<dst>.<src> d, a   — only handle identity-width and truncation
                parts = op.split('.')
                if len(parts) != 3:
                    raise Unsupported(op)
                dst_t, src_t = parts[1], parts[2]
                if dst_t not in ('u32','s32','u16','s16','u8','s8','b32') or \
                   src_t not in ('u32','s32','u16','s16','u8','s8','b32'):
                    raise Unsupported(op)
                if len(operands) != 2: raise Unsupported(op)
                ka, va = _parse_reg(operands[1])
                a = state.get(operands[1], 0) if ka == 'r' else va
                width = {'u8':8,'s8':8,'u16':16,'s16':16,
                          'u32':32,'s32':32,'b32':32}
                sw = width[src_t]; dw = width[dst_t]
                if sw < 32:
                    mask = (1 << sw) - 1
                    a &= mask
                    if src_t.startswith('s') and a & (1 << (sw-1)):
                        a -= (1 << sw)
                if dw < 32:
                    mask = (1 << dw) - 1
                    v = a & mask
                    if dst_t.startswith('s') and v & (1 << (dw-1)):
                        v -= (1 << dw)
                    state[operands[0]] = _u32(v)
                else:
                    state[operands[0]] = _u32(a)
                continue

            # Anything else: bail
            raise Unsupported(op)

        struct.pack_into('<I', out, lane*4, _u32(state.get(out_reg, 0)))
    return bytes(out)
