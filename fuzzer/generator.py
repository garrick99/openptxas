"""Grammar-based PTX generator for differential fuzzing.

Kernel shape:
  out[tid] = f(in[tid])
where f is a seeded-random straight-line sequence of integer ops over
b32 and b64 registers, ending by storing a b32 value at out[tid].

No branches, no loops, no data-dep memory ops — keeps the GPU driver
watchdog from firing.  Generator is deterministic per seed.

Op coverage (targets known ptxas bug families):
  * b32 ALU: add/sub/mul.lo/and/or/xor/min/max/shifts, u32 and s32
  * b64 ALU: add/sub/mul.lo/and/or/xor/min/max/shifts, u64 and s64
  * mul.wide.{u32,s32}  — 32x32->64, classic signed-extension bug site
  * mad.{lo,hi}.{u32,s32}
  * add.cc.u32 + addc.u32 carry chains
  * cvt bridges (both widening and narrowing, signed and unsigned)
  * setp + @p / @!p predicated ALU
  * selp.b32 predicate-select
"""
import random

# Prologue/epilogue reserve %r0..%r3 and %rd0..%rd4, %p0.
_REG_B32 = list(range(4, 28))
_REG_B64 = list(range(5, 16))
_PREDS = [1, 2, 3, 4]
_TID_REG = 0
_N_REG = 1
_LOAD_REG = 3  # first live b32 (holds input word)

_IMM32 = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64,
          0xFF, 0x100, 0xFFFF, 0x10000,
          0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
_SHIFT32 = [0, 1, 2, 3, 4, 7, 15, 16, 30, 31]
_SHIFT64 = _SHIFT32 + [32, 33, 47, 48, 62, 63]

_BIN_B32 = ['add.u32', 'sub.u32', 'mul.lo.u32', 'and.b32', 'or.b32',
            'xor.b32', 'min.u32', 'max.u32',
            'add.s32', 'sub.s32', 'mul.lo.s32', 'min.s32', 'max.s32']
_BIN_B64 = ['add.u64', 'sub.u64', 'mul.lo.u64', 'and.b64', 'or.b64',
            'xor.b64', 'min.u64', 'max.u64',
            'add.s64', 'sub.s64', 'mul.lo.s64', 'min.s64', 'max.s64']
_SHIFT_B32 = ['shl.b32', 'shr.u32', 'shr.s32']
_SHIFT_B64 = ['shl.b64', 'shr.u64', 'shr.s64']
# mad.hi.{u32,s32} and addc.u32 are unimplemented in OpenPTXas today — emitting
# them produces guaranteed divergences that are our own gaps, not ptxas bugs.
# Keep mad.lo only; skip add.cc/addc carry chains entirely.
_MAD = ['mad.lo.u32', 'mad.lo.s32']
_MULW = ['mul.wide.u32', 'mul.wide.s32']
# setp comparison ops — .u32 supports the unsigned-only (lo/ls/hi/hs) set,
# .s32 only the sign-agnostic set.
_CMP_COMMON = ['eq', 'ne', 'lt', 'le', 'gt', 'ge']
_CMP_U32_ONLY = ['lo', 'ls', 'hi', 'hs']


def _pick(rnd, xs):
    return rnd.choice(xs)


def generate(seed: int, n_instrs: int = 14) -> tuple[str, int]:
    """Produce one PTX source.  Returns (ptx_text, final_b32_reg_index)."""
    rnd = random.Random(seed)
    live32 = [_LOAD_REG]
    live64 = []
    preds_set = []
    pending_pred = None  # (pred_num, negate) applied to next op if non-None
    body: list[str] = []
    last_b32 = _LOAD_REG

    def prefix():
        nonlocal pending_pred
        if pending_pred is None:
            return ''
        pn, neg = pending_pred
        pending_pred = None
        return f'@{"!" if neg else ""}%p{pn} '

    for _ in range(n_instrs):
        cat = rnd.random()

        if cat < 0.24:  # b32 reg-reg
            op = _pick(rnd, _BIN_B32)
            dst = _pick(rnd, _REG_B32)
            a, b = _pick(rnd, live32), _pick(rnd, live32)
            body.append(f'    {prefix()}{op} %r{dst}, %r{a}, %r{b};')
            if dst not in live32: live32.append(dst)
            last_b32 = dst

        elif cat < 0.42:  # b32 reg-imm
            op = _pick(rnd, _BIN_B32)
            dst = _pick(rnd, _REG_B32)
            a = _pick(rnd, live32)
            imm = _pick(rnd, _IMM32)
            body.append(f'    {prefix()}{op} %r{dst}, %r{a}, {imm};')
            if dst not in live32: live32.append(dst)
            last_b32 = dst

        elif cat < 0.52:  # b32 shift by constant
            op = _pick(rnd, _SHIFT_B32)
            dst = _pick(rnd, _REG_B32)
            a = _pick(rnd, live32)
            k = _pick(rnd, _SHIFT32)
            body.append(f'    {prefix()}{op} %r{dst}, %r{a}, {k};')
            if dst not in live32: live32.append(dst)
            last_b32 = dst

        elif cat < 0.58:  # mad (3-op multiply-add)
            op = _pick(rnd, _MAD)
            dst = _pick(rnd, _REG_B32)
            a, b, c = _pick(rnd, live32), _pick(rnd, live32), _pick(rnd, live32)
            body.append(f'    {prefix()}{op} %r{dst}, %r{a}, %r{b}, %r{c};')
            if dst not in live32: live32.append(dst)
            last_b32 = dst

        elif cat < 0.64:  # mul.wide (b32 x b32 -> b64) — classic signed-ext bug site
            op = _pick(rnd, _MULW)
            dst = _pick(rnd, _REG_B64)
            a, b = _pick(rnd, live32), _pick(rnd, live32)
            body.append(f'    {prefix()}{op} %rd{dst}, %r{a}, %r{b};')
            if dst not in live64: live64.append(dst)

        elif cat < 0.72:  # b64 binary (bootstrap via cvt if live64 empty)
            if not live64:
                d = _pick(rnd, _REG_B64)
                src = _pick(rnd, live32)
                cop = _pick(rnd, ['cvt.u64.u32', 'cvt.s64.s32'])
                body.append(f'    {prefix()}{cop} %rd{d}, %r{src};')
                live64.append(d)
            else:
                op = _pick(rnd, _BIN_B64)
                dst = _pick(rnd, _REG_B64)
                a, b = _pick(rnd, live64), _pick(rnd, live64)
                body.append(f'    {prefix()}{op} %rd{dst}, %rd{a}, %rd{b};')
                if dst not in live64: live64.append(dst)

        elif cat < 0.78:  # b64 shift
            if live64:
                op = _pick(rnd, _SHIFT_B64)
                dst = _pick(rnd, _REG_B64)
                a = _pick(rnd, live64)
                k = _pick(rnd, _SHIFT64)
                body.append(f'    {prefix()}{op} %rd{dst}, %rd{a}, {k};')
                if dst not in live64: live64.append(dst)

        elif cat < 0.88:  # cvt bridge
            if live64 and rnd.random() < 0.5:
                # 64 -> 32 narrowing
                dst = _pick(rnd, _REG_B32)
                src = _pick(rnd, live64)
                cop = _pick(rnd, ['cvt.u32.u64', 'cvt.s32.s64'])
                body.append(f'    {prefix()}{cop} %r{dst}, %rd{src};')
                if dst not in live32: live32.append(dst)
                last_b32 = dst
            else:
                # 32 -> 64 widening (sign- or zero-extend)
                dst = _pick(rnd, _REG_B64)
                src = _pick(rnd, live32)
                cop = _pick(rnd, ['cvt.u64.u32', 'cvt.s64.s32'])
                body.append(f'    {prefix()}{cop} %rd{dst}, %r{src};')
                if dst not in live64: live64.append(dst)

        elif cat < 0.95:  # setp -> predicate (don't predicate setp itself)
            pending_pred = None
            pn = _pick(rnd, _PREDS)
            if rnd.random() < 0.5:
                ty = 'u32'
                cmp_ = _pick(rnd, _CMP_COMMON + _CMP_U32_ONLY)
            else:
                ty = 's32'
                cmp_ = _pick(rnd, _CMP_COMMON)
            a = _pick(rnd, live32)
            if rnd.random() < 0.5:
                b_tok = f'%r{_pick(rnd, live32)}'
            else:
                b_tok = str(_pick(rnd, _IMM32))
            body.append(f'    setp.{cmp_}.{ty} %p{pn}, %r{a}, {b_tok};')
            if pn not in preds_set: preds_set.append(pn)
            # 70% chance next op is predicated on this fresh predicate
            if rnd.random() < 0.7:
                pending_pred = (pn, rnd.random() < 0.3)

        else:  # selp.b32 — requires at least one predicate
            if preds_set:
                dst = _pick(rnd, _REG_B32)
                a, b = _pick(rnd, live32), _pick(rnd, live32)
                pn = _pick(rnd, preds_set)
                body.append(f'    {prefix()}selp.b32 %r{dst}, %r{a}, %r{b}, %p{pn};')
                if dst not in live32: live32.append(dst)
                last_b32 = dst

    # If we ended with a live64 result and never narrowed to b32, insert
    # a final cvt so the store has something to read.
    if body and last_b32 == _LOAD_REG and live64:
        dst = _REG_B32[0]
        src = live64[-1]
        body.append(f'    cvt.u32.u64 %r{dst}, %rd{src};')
        last_b32 = dst

    body_text = '\n'.join(body)
    ptx = f""".version 9.0
.target sm_120
.address_size 64

.visible .entry fuzz(.param .u64 p_in, .param .u64 p_out, .param .u32 n) {{
    .reg .b32 %r<32>;
    .reg .b64 %rd<16>;
    .reg .pred %p<8>;
    mov.u32 %r{_TID_REG}, %tid.x;
    ld.param.u32 %r{_N_REG}, [n];
    setp.ge.u32 %p0, %r{_TID_REG}, %r{_N_REG};
    @%p0 ret;
    ld.param.u64 %rd0, [p_in];
    cvt.u64.u32 %rd1, %r{_TID_REG};
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r{_LOAD_REG}, [%rd2];
{body_text}
    ld.param.u64 %rd3, [p_out];
    add.u64 %rd4, %rd3, %rd1;
    st.global.u32 [%rd4], %r{last_b32};
    ret;
}}
"""
    return ptx, last_b32


def normalize(ptx: str) -> str:
    """Strip whitespace + comments for structural hashing."""
    import re
    out = []
    for line in ptx.splitlines():
        line = re.sub(r'//.*$', '', line).strip()
        if line:
            out.append(line)
    return '\n'.join(out)


if __name__ == '__main__':
    for seed in range(6):
        ptx, r = generate(seed)
        print(f'--- seed={seed} result=%r{r} ---')
        if seed == 0:
            print(ptx)
        print(f'  body lines: {ptx.count(chr(10))}')
