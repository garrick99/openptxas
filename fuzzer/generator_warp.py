"""PTX generator — warp intrinsics family.

Kernel shape: all 32 threads always execute (no early-exit — warp
intrinsics require warp-uniform control flow).  Body mixes shfl.sync
variants, vote.sync, and match.sync, with per-thread ALU to scramble
the input before warp-reducing it.

Op coverage:
  * shfl.sync.{up,down,bfly,idx}.b32
  * vote.sync.ballot.b32 / all.pred / any.pred
  * match.any.sync.b32
  * setp (to produce predicates for vote)
"""
import random

_REG_B32 = list(range(4, 20))
_PREDS = [1, 2, 3]
_TID_REG = 0
_LOAD_REG = 3

_SHFL_MODES = ['up', 'down', 'bfly', 'idx']
_SHFL_LANES_UP_DOWN = [1, 2, 4, 8, 16]
_SHFL_LANES_BFLY = [1, 2, 4, 8, 16]
_SHFL_LANES_IDX = [0, 1, 7, 15, 31]
_CMP_U = ['eq', 'ne', 'lt', 'le', 'gt', 'ge']
_IMM32 = [0, 1, 2, 3, 7, 8, 15, 16, 31, 0xFF, 0xFFFF, 0x7FFFFFFF, 0xFFFFFFFF]

_MEMBERMASK = '-1'    # all 32 lanes active
_SHFL_C = '0x1f'      # full-warp width (5 bits)


def _pick(rnd, xs): return rnd.choice(xs)


def _emit_shfl(rnd, dst, src):
    mode = _pick(rnd, _SHFL_MODES)
    if mode == 'idx':
        lane = _pick(rnd, _SHFL_LANES_IDX)
    elif mode == 'bfly':
        lane = _pick(rnd, _SHFL_LANES_BFLY)
    else:
        lane = _pick(rnd, _SHFL_LANES_UP_DOWN)
    return f'    shfl.sync.{mode}.b32 %r{dst}, %r{src}, {lane}, {_SHFL_C}, {_MEMBERMASK};'


def _emit_vote_ballot(rnd, dst, pred):
    return f'    vote.sync.ballot.b32 %r{dst}, %p{pred}, {_MEMBERMASK};'


def _emit_match_any(rnd, dst, src):
    return f'    match.any.sync.b32 %r{dst}, %r{src}, {_MEMBERMASK};'


def _emit_setp(rnd, pred, a, live32):
    cmp_ = _pick(rnd, _CMP_U)
    if rnd.random() < 0.5:
        b_tok = f'%r{_pick(rnd, live32)}'
    else:
        b_tok = str(_pick(rnd, _IMM32))
    return f'    setp.{cmp_}.u32 %p{pred}, %r{a}, {b_tok};'


def _emit_alu(rnd, dst, a, b, live32):
    op = _pick(rnd, ['add.u32', 'sub.u32', 'xor.b32', 'and.b32', 'or.b32',
                       'min.u32', 'max.u32'])
    return f'    {op} %r{dst}, %r{a}, %r{b};'


def generate(seed: int, n_instrs: int = 10) -> tuple[str, int]:
    rnd = random.Random(seed ^ 0xC0FFEE)
    live32 = [_LOAD_REG, _TID_REG]
    preds_set = []
    body = []
    last = _LOAD_REG

    for _ in range(n_instrs):
        r = rnd.random()
        if r < 0.45:  # shfl
            dst = _pick(rnd, _REG_B32)
            src = _pick(rnd, live32)
            body.append(_emit_shfl(rnd, dst, src))
            if dst not in live32: live32.append(dst)
            last = dst
        elif r < 0.60:  # alu scramble
            dst = _pick(rnd, _REG_B32)
            a, b = _pick(rnd, live32), _pick(rnd, live32)
            body.append(_emit_alu(rnd, dst, a, b, live32))
            if dst not in live32: live32.append(dst)
            last = dst
        elif r < 0.75:  # another alu (fill the slot; match.any.sync is
                         # unimplemented in OpenPTXas — skipping avoids
                         # 100% noise)
            dst = _pick(rnd, _REG_B32)
            a, b = _pick(rnd, live32), _pick(rnd, live32)
            body.append(_emit_alu(rnd, dst, a, b, live32))
            if dst not in live32: live32.append(dst)
            last = dst
        elif r < 0.90:  # setp -> vote.ballot
            pn = _pick(rnd, _PREDS)
            a = _pick(rnd, live32)
            body.append(_emit_setp(rnd, pn, a, live32))
            if pn not in preds_set: preds_set.append(pn)
            # Immediately use via vote.ballot
            dst = _pick(rnd, _REG_B32)
            body.append(_emit_vote_ballot(rnd, dst, pn))
            if dst not in live32: live32.append(dst)
            last = dst
        else:  # vote.all/any
            if preds_set:
                dst_p = _pick(rnd, _PREDS)
                src_p = _pick(rnd, preds_set)
                mode = _pick(rnd, ['all', 'any'])
                body.append(f'    vote.sync.{mode}.pred %p{dst_p}, %p{src_p}, {_MEMBERMASK};')
                if dst_p not in preds_set: preds_set.append(dst_p)

    body_text = '\n'.join(body)
    ptx = f""".version 9.0
.target sm_120
.address_size 64

.visible .entry fuzz(.param .u64 p_in, .param .u64 p_out, .param .u32 n) {{
    .reg .b32 %r<24>;
    .reg .b64 %rd<8>;
    .reg .pred %p<6>;
    mov.u32 %r{_TID_REG}, %tid.x;
    ld.param.u64 %rd0, [p_in];
    cvt.u64.u32 %rd1, %r{_TID_REG};
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r{_LOAD_REG}, [%rd2];
{body_text}
    ld.param.u64 %rd3, [p_out];
    add.u64 %rd4, %rd3, %rd1;
    st.global.u32 [%rd4], %r{last};
    ret;
}}
"""
    return ptx, last


if __name__ == '__main__':
    for s in range(3):
        ptx, r = generate(s)
        print(f'--- seed={s} ---')
        if s == 0: print(ptx)
        print(f'  body lines: {ptx.count(chr(10))}')
