"""PTX generator — bit-manipulation family.

Kernel shape: same as alu_int (early-exit guard, per-thread load/store).
Body emits PTX bit-extraction, bit-insertion, population-count, count-
leading-zeros, bit-reversal, byte-permute.

Op coverage:
  * bfe.{u32,s32}
  * bfi.b32
  * popc.b32
  * clz.b32
  * brev.b32
  * prmt.b32 (with a range of selector values)
  * plus a small ALU spice (and/or/xor/add) to feed operands
"""
import random

_REG_B32 = list(range(4, 24))
_TID_REG = 0
_N_REG = 1
_LOAD_REG = 3

# bfe / bfi: start positions + lengths worth trying
_BF_START = [0, 1, 4, 7, 8, 15, 16, 23, 24, 31]
_BF_LEN   = [1, 2, 4, 7, 8, 16, 24, 31, 32]

# prmt.b32 selector nibbles: each nibble selects one byte from source
_PRMT_C = [0x0000, 0x3210, 0x0123, 0x5410, 0x4444, 0xFEDC, 0x7320,
           0xBBBB, 0x1032, 0x4455]

_IMM32 = [0, 1, 2, 3, 7, 8, 15, 16, 31, 32,
          0xFF, 0x100, 0xFFFF, 0x10000,
          0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]

_ALU = ['add.u32', 'sub.u32', 'and.b32', 'or.b32', 'xor.b32',
         'min.u32', 'max.u32']


def _pick(rnd, xs): return rnd.choice(xs)


def generate(seed: int, n_instrs: int = 12) -> tuple[str, int]:
    rnd = random.Random(seed ^ 0xB17BA71)
    live = [_LOAD_REG]
    body = []
    last = _LOAD_REG

    for _ in range(n_instrs):
        r = rnd.random()
        if r < 0.25:  # bfe
            sgn = _pick(rnd, ['u32', 's32'])
            dst = _pick(rnd, _REG_B32)
            src = _pick(rnd, live)
            start = _pick(rnd, _BF_START)
            length = _pick(rnd, _BF_LEN)
            body.append(f'    bfe.{sgn} %r{dst}, %r{src}, {start}, {length};')
            if dst not in live: live.append(dst)
            last = dst
        elif r < 0.40:  # bfi
            dst = _pick(rnd, _REG_B32)
            srcbits = _pick(rnd, live)
            srcbase = _pick(rnd, live)
            start = _pick(rnd, _BF_START)
            length = _pick(rnd, _BF_LEN)
            body.append(f'    bfi.b32 %r{dst}, %r{srcbits}, %r{srcbase}, {start}, {length};')
            if dst not in live: live.append(dst)
            last = dst
        elif r < 0.55:  # popc
            dst = _pick(rnd, _REG_B32)
            src = _pick(rnd, live)
            body.append(f'    popc.b32 %r{dst}, %r{src};')
            if dst not in live: live.append(dst)
            last = dst
        elif r < 0.70:  # clz
            dst = _pick(rnd, _REG_B32)
            src = _pick(rnd, live)
            body.append(f'    clz.b32 %r{dst}, %r{src};')
            if dst not in live: live.append(dst)
            last = dst
        elif r < 0.80:  # brev
            dst = _pick(rnd, _REG_B32)
            src = _pick(rnd, live)
            body.append(f'    brev.b32 %r{dst}, %r{src};')
            if dst not in live: live.append(dst)
            last = dst
        elif r < 0.88:  # prmt
            dst = _pick(rnd, _REG_B32)
            a = _pick(rnd, live)
            b = _pick(rnd, live)
            c = _pick(rnd, _PRMT_C)
            body.append(f'    prmt.b32 %r{dst}, %r{a}, %r{b}, {c};')
            if dst not in live: live.append(dst)
            last = dst
        else:  # ALU spice (feeds fresh operands)
            op = _pick(rnd, _ALU)
            dst = _pick(rnd, _REG_B32)
            a = _pick(rnd, live)
            if rnd.random() < 0.5:
                b_tok = f'%r{_pick(rnd, live)}'
            else:
                b_tok = str(_pick(rnd, _IMM32))
            body.append(f'    {op} %r{dst}, %r{a}, {b_tok};')
            if dst not in live: live.append(dst)
            last = dst

    body_text = '\n'.join(body)
    ptx = f""".version 9.0
.target sm_120
.address_size 64

.visible .entry fuzz(.param .u64 p_in, .param .u64 p_out, .param .u32 n) {{
    .reg .b32 %r<28>;
    .reg .b64 %rd<8>;
    .reg .pred %p<4>;
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
