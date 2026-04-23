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
def _s64(x: int) -> int:
    x &= 0xFFFFFFFFFFFFFFFF
    return x - 0x10000000000000000 if x & 0x8000000000000000 else x


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


def _binop_u64(op: str, a: int, b: int) -> int:
    """Evaluate a binary opcode on two u64 values."""
    if op == 'add.u64' or op == 'add.s64':
        return _u64(a + b)
    if op == 'sub.u64' or op == 'sub.s64':
        return _u64(a - b)
    if op == 'mul.lo.u64' or op == 'mul.lo.s64':
        return _u64(a * b)
    if op == 'and.b64':
        return _u64(a & b)
    if op == 'or.b64':
        return _u64(a | b)
    if op == 'xor.b64':
        return _u64(a ^ b)
    if op == 'min.u64':
        return min(_u64(a), _u64(b))
    if op == 'max.u64':
        return max(_u64(a), _u64(b))
    if op == 'min.s64':
        return _u64(min(_s64(a), _s64(b)))
    if op == 'max.s64':
        return _u64(max(_s64(a), _s64(b)))
    if op == 'shl.b64':
        n = _u32(b) & 0x3F
        return _u64(a << n)
    if op == 'shr.u64':
        n = _u32(b) & 0x3F
        return _u64(_u64(a) >> n)
    if op == 'shr.s64':
        n = _u32(b) & 0x3F
        return _u64(_s64(a) >> n)
    raise Unsupported(op)


_U64_BINOPS = frozenset((
    'add.u64','add.s64','sub.u64','sub.s64',
    'mul.lo.u64','mul.lo.s64',
    'and.b64','or.b64','xor.b64',
    'min.u64','max.u64','min.s64','max.s64',
    'shl.b64','shr.u64','shr.s64',
))


def _clz_b32(a: int) -> int:
    """Count leading zeros of a 32-bit value.  clz(0) == 32."""
    a = _u32(a)
    if a == 0: return 32
    n = 0
    for i in range(31, -1, -1):
        if (a >> i) & 1: break
        n += 1
    return n


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


def _exec_shfl(op: str, operands: list, states: list,
                preds_per_lane: list, pred_reg, neg_pred, n: int) -> None:
    """Handle shfl.sync.{bfly,up,down,idx}.b32 across the warp.

    PTX: ``shfl.sync.<mode>.b32 d, a, b, c, membermask`` —

    - ``d``, ``a``: 32-bit registers (dest / source-value)
    - ``b``: shift/mask/index operand (register or imm)
    - ``c``: clamp + segmentation.  Low 5 bits = clamp; we treat higher
      bits as "full warp" for now (our fuzz kernels pass 0x1f).
    - ``membermask``: active lane bitmask; ignored here (assume all active).

    ``d[L] := a[src_lane]`` when the source lane is valid for the mode,
    else ``d[L] := a[L]`` (no exchange).  The predicate-output variant
    ``d|%p`` raises Unsupported; we don't emit kernels that need it yet.
    """
    if len(operands) != 5:
        raise Unsupported(op + '  (expected d, a, b, c, mask)')
    parts = op.split('.')
    if len(parts) != 4 or parts[1] != 'sync' or parts[3] != 'b32':
        raise Unsupported(op)
    mode = parts[2]
    if mode not in ('bfly', 'up', 'down', 'idx'):
        raise Unsupported(op)

    d_reg = operands[0]
    if '|' in d_reg:
        raise Unsupported(op + '  (predicate-output form)')
    a_tok, b_tok, c_tok = operands[1], operands[2], operands[3]

    def _eval_in(tok: str, lane: int) -> int:
        k, v = _parse_reg(tok)
        if k in ('r', 'rd'): return _u32(states[lane].get(tok, 0))
        return _u32(v)

    # Snapshot source values BEFORE any writes — a warp-wide shfl reads
    # every lane's pre-shuffle ``a`` simultaneously.
    a_vals = [_eval_in(a_tok, L) for L in range(n)]

    for L in range(n):
        # Honor per-lane predication (same semantics as other ops).
        if pred_reg is not None:
            pv = preds_per_lane[L].get(pred_reg, False)
            if neg_pred == '!': pv = not pv
            if not pv: continue
        b = _eval_in(b_tok, L) & 0x1F
        c = _eval_in(c_tok, L)
        clamp = c & 0x1F
        if mode == 'bfly':
            src = L ^ b
            valid = src < n
        elif mode == 'up':
            src = L - b
            # Lane L reads from L-b only if (L-b) >= clamp_lo derived from c;
            # for canonical c=0x1f, clamp_lo = 0 → src >= 0 is the test.
            valid = src >= 0
        elif mode == 'down':
            src = L + b
            # For canonical c=0x1f, clamp_hi = 31 → src <= 31 is the test.
            valid = src < n and src <= clamp
        else:  # idx
            src = b
            valid = src < n
        states[L][d_reg] = a_vals[src] if valid else a_vals[L]


def simulate(ptx: str, inputs: bytes) -> bytes:
    """Run the body on each of the 32 input u32 words; return 128 output bytes.

    Line-outer loop so warp-level ops (shfl.sync) can read every lane's
    register snapshot at the instant they execute.  Per-lane ops are
    dispatched in the inner loop and mutate their lane's state dict.

    Raises Unsupported if any body instruction is outside the handled subset.
    """
    body, out_reg = _split_body(ptx)
    n = len(inputs) // 4
    out = bytearray(n * 4)
    states = [{'%r3': struct.unpack_from('<I', inputs, lane*4)[0],
                '%rz': 0, '%rZ': 0} for lane in range(n)]
    preds_per_lane = [{} for _ in range(n)]

    for line in body:
        s = line.strip()
        if not s or s.startswith('//'):
            continue
        m = _INSTR_RE.match(s)
        if not m:
            continue
        neg_pred, pred_reg, op, args = m.group(1), m.group(2), m.group(3), m.group(4)
        operands = [x.strip() for x in args.split(',')]

        # Warp-level op — reads pre-shuffle state across all lanes.
        if op.startswith('shfl.sync.'):
            _exec_shfl(op, operands, states, preds_per_lane, pred_reg, neg_pred, n)
            continue

        # vote.sync.ballot.b32 d, pred_src, mask —
        # Each lane's d gets the same 32-bit mask where bit L = pred[L].
        # Inactive (membermask) lanes and predicated-off lanes contribute 0.
        if op == 'vote.sync.ballot.b32':
            if len(operands) != 3: raise Unsupported(op)
            d_reg = operands[0]
            pred_src = operands[1].lstrip('!')
            pred_neg = operands[1].startswith('!')
            # Build the mask by reading each lane's predicate value.
            mask = 0
            for L in range(n):
                pv = preds_per_lane[L].get(pred_src, False)
                if pred_neg: pv = not pv
                # Honor outer predication (@%pX vote.sync...): only lanes
                # where the guard is TRUE contribute.
                if pred_reg is not None:
                    g = preds_per_lane[L].get(pred_reg, False)
                    if neg_pred == '!': g = not g
                    if not g: continue
                if pv: mask |= (1 << L)
            # Write to all active lanes; skip predicated-off lanes' dest.
            for L in range(n):
                if pred_reg is not None:
                    g = preds_per_lane[L].get(pred_reg, False)
                    if neg_pred == '!': g = not g
                    if not g: continue
                states[L][d_reg] = _u32(mask)
            continue

        for lane in range(n):
            state = states[lane]
            preds = preds_per_lane[lane]
            # Predication: skip if predicate not satisfied for this lane.
            if pred_reg is not None:
                pv = preds.get(pred_reg, False)
                if neg_pred == '!': pv = not pv
                if not pv: continue

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
                    ua, ub = _u32(a), _u32(b)
                    # hi/lo/hs/ls are the unsigned carry-based names
                    # (equivalent to gt/lt/ge/le on unsigned compare).
                    base = {'eq': ua==ub, 'ne': ua!=ub,
                             'lt': ua<ub, 'le': ua<=ub,
                             'gt': ua>ub, 'ge': ua>=ub,
                             'lo': ua<ub, 'ls': ua<=ub,
                             'hi': ua>ub, 'hs': ua>=ub,
                             }.get(cmp_, None)
                elif ptype == 's32':
                    base = _setp_cmp(cmp_, a, b)
                elif ptype in ('u64', 'b64'):
                    ua, ub = _u64(a), _u64(b)
                    base = {'eq': ua==ub, 'ne': ua!=ub,
                             'lt': ua<ub, 'le': ua<=ub,
                             'gt': ua>ub, 'ge': ua>=ub,
                             'lo': ua<ub, 'ls': ua<=ub,
                             'hi': ua>ub, 'hs': ua>=ub,
                             }.get(cmp_, None)
                elif ptype == 's64':
                    sa, sb = _s64(a), _s64(b)
                    base = {'eq': sa==sb, 'ne': sa!=sb,
                             'lt': sa<sb, 'le': sa<=sb,
                             'gt': sa>sb, 'ge': sa>=sb}.get(cmp_, None)
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
                a = state.get(operands[1], 0) if ka in ('r','rd') else va
                b = state.get(operands[2], 0) if kb in ('r','rd') else vb
                state[d] = _binop(op, a, b)
                continue

            # 64-bit binary ALU — operands live in %rd registers.  State dict
            # treats %r as u32 and %rd as u64; no separate tables needed.
            if op in _U64_BINOPS:
                if len(operands) != 3:
                    raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                kb, vb = _parse_reg(operands[2])
                a = state.get(operands[1], 0) if ka in ('r','rd') else va
                b = state.get(operands[2], 0) if kb in ('r','rd') else vb
                state[d] = _binop_u64(op, a, b)
                continue

            # add.cc / addc — 32-bit extended-precision adds.  We model the
            # carry flag in state['__CC__']; the fuzz kernels chain these
            # only within a single thread so the flag is per-lane.
            if op in ('add.cc.u32', 'add.cc.s32'):
                if len(operands) != 3: raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                kb, vb = _parse_reg(operands[2])
                a = _u32(state.get(operands[1], 0) if ka == 'r' else va)
                b = _u32(state.get(operands[2], 0) if kb == 'r' else vb)
                s = a + b
                state[d] = _u32(s)
                state['__CC__'] = 1 if s > 0xFFFFFFFF else 0
                continue
            if op in ('addc.u32', 'addc.cc.u32', 'addc.s32', 'addc.cc.s32'):
                if len(operands) != 3: raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                kb, vb = _parse_reg(operands[2])
                a = _u32(state.get(operands[1], 0) if ka == 'r' else va)
                b = _u32(state.get(operands[2], 0) if kb == 'r' else vb)
                c = state.get('__CC__', 0)
                s = a + b + c
                state[d] = _u32(s)
                if op.startswith('addc.cc.'):
                    state['__CC__'] = 1 if s > 0xFFFFFFFF else 0
                continue

            # Unary: clz.b32
            if op == 'clz.b32':
                if len(operands) != 2: raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                a = state.get(operands[1], 0) if ka == 'r' else va
                state[d] = _clz_b32(a)
                continue

            # popc.b32 — population count
            if op == 'popc.b32':
                if len(operands) != 2: raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                a = _u32(state.get(operands[1], 0) if ka == 'r' else va)
                state[d] = bin(a).count('1')
                continue

            # prmt.b32 d, a, b, c  (default "sel" mode) — permute bytes.
            # Low 16 bits of c form 4x 4-bit selectors; each picks one of 8
            # bytes from concat(b:a) = bytes 0..3 of a then 0..3 of b.
            # High bit of each selector forces sign-replication of the
            # picked byte (0x00 or 0xFF).  .f4e/.b4e/.rc8/.ecl/.ecr/.rc16
            # variants are not emitted by our generators; leave unsupported.
            if op == 'prmt.b32':
                if len(operands) != 4: raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                kb, vb = _parse_reg(operands[2])
                kc, vc = _parse_reg(operands[3])
                a = _u32(state.get(operands[1], 0) if ka == 'r' else va)
                b = _u32(state.get(operands[2], 0) if kb == 'r' else vb)
                c = _u32(state.get(operands[3], 0) if kc == 'r' else vc)
                src = [a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, (a >> 24) & 0xFF,
                       b & 0xFF, (b >> 8) & 0xFF, (b >> 16) & 0xFF, (b >> 24) & 0xFF]
                r = 0
                for i in range(4):
                    sel = (c >> (i * 4)) & 0xF
                    byt = src[sel & 7]
                    if sel & 8:
                        byt = 0xFF if (byt & 0x80) else 0x00
                    r |= byt << (i * 8)
                state[d] = _u32(r)
                continue

            # brev.b32 — bit reverse
            if op == 'brev.b32':
                if len(operands) != 2: raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                a = _u32(state.get(operands[1], 0) if ka == 'r' else va)
                r = 0
                for i in range(32):
                    r = (r << 1) | ((a >> i) & 1)
                state[d] = r
                continue

            # neg.{s32,s64} d, a  => d = -a (two's complement)
            if op in ('neg.s32', 'neg.s64'):
                if len(operands) != 2: raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                a = state.get(operands[1], 0) if ka in ('r','rd') else va
                if op == 'neg.s32':
                    state[d] = _u32(-_s32(a))
                else:
                    state[d] = _u64(-_s64(a))
                continue

            # not.{b32,b64} d, a
            if op in ('not.b32', 'not.b64'):
                if len(operands) != 2: raise Unsupported(op)
                d = operands[0]
                ka, va = _parse_reg(operands[1])
                a = state.get(operands[1], 0) if ka in ('r','rd') else va
                if op == 'not.b32':
                    state[d] = _u32(~a)
                else:
                    state[d] = _u64(~a)
                continue

            if op.startswith('cvt.'):
                # cvt.<dst>.<src> d, a  — integer zext / sext / truncate.
                # Skip rounding modes (cvt.rn.f32.u32 etc.) — not in the
                # integer generators.
                parts = op.split('.')
                if len(parts) != 3:
                    raise Unsupported(op)
                dst_t, src_t = parts[1], parts[2]
                widths = {'u8':8,'s8':8,'u16':16,'s16':16,
                           'u32':32,'s32':32,'b32':32,
                           'u64':64,'s64':64,'b64':64}
                if dst_t not in widths or src_t not in widths:
                    raise Unsupported(op)
                if len(operands) != 2: raise Unsupported(op)
                ka, va = _parse_reg(operands[1])
                a = state.get(operands[1], 0) if ka in ('r','rd') else va
                sw = widths[src_t]; dw = widths[dst_t]
                # Interpret source according to its signedness.
                if sw < 64:
                    mask = (1 << sw) - 1
                    a &= mask
                    if src_t.startswith('s') and a & (1 << (sw-1)):
                        a -= (1 << sw)
                else:
                    if src_t.startswith('s') and a & (1 << 63):
                        a -= (1 << 64)
                # Write the dest with width-appropriate mask.
                if dw <= 32:
                    mask = (1 << dw) - 1
                    v = a & mask
                    if dst_t.startswith('s') and dw < 32 and v & (1 << (dw-1)):
                        v -= (1 << dw)
                    state[operands[0]] = _u32(v)
                else:
                    # 64-bit dest: write into %rd
                    state[operands[0]] = _u64(a & 0xFFFFFFFFFFFFFFFF)
                continue

            # Anything else: bail
            raise Unsupported(op)

    # All lines evaluated — emit each lane's output register.
    for lane in range(n):
        struct.pack_into('<I', out, lane*4, _u32(states[lane].get(out_reg, 0)))
    return bytes(out)
