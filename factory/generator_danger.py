"""Bug-adjacent PTX generator.

Emits kernel bodies built around PTX shapes that are empirically close
to known ptxas and OpenPTXas bugs.  Each invocation picks one danger
template, fills it with parameters drawn from the high-bug region,
then wraps it in 0-4 lines of random filler so the family_signature
varies enough that two hits of the same bug land in the same class
(3-specimen escalation threshold) without every single kernel
collapsing to an identical sha.

Target hit rate: ~1/50 (vs ~1/500 for uniform random alu/bitmanip).

Danger patterns (sourced from bugs we've already confirmed):
  * or_msb_shr_fold   — Bug 2 shape: (x | 0x80000000) >>s N >>u (32-N)
  * bfe_s32_oor       — Bug 1 shape: bfe.s32 with start >= 32
  * mul_lo_largeimm   — literal-pool-zero class: mul.lo with imm >= 0x7FFFFFFF
  * shift_boundary    — shl.b32 then shr.{s,u}32 with amounts near 0/31
  * sign_flip_chain   — xor/or with 0x80000000 followed by signed ALU
  * bfi_oor           — bfi.b32 with pos+len > 32 (characterized UB region)
"""
import random

_TID_REG  = 0
_N_REG    = 1
_LOAD_REG = 3

# Signed-right-shift amounts that trigger the ptxas or+shr fold bug
_FOLD_N = [1, 2, 3, 4, 5]          # N in {1,2,3} trigger; 4,5 adjacent
# bfe start positions straddling the 32-bit boundary
_BFE_START_OOR = [32, 33, 40, 48, 56, 63]
_BFE_LEN = [1, 2, 4, 7, 8, 15, 16, 23, 24, 31, 32]
# mul.lo immediates that hit the literal-pool-zero class
_MUL_LARGE_IMM = [0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF,
                  0x7FFF0000, 0x80000001, 0xCAFEBABE, 0xDEADBEEF,
                  0x00010000, 0x00100000, 0x40000000]
# Shift boundaries
_SHIFT_BOUNDARY = [0, 1, 2, 15, 16, 17, 30, 31]
# Sign-bit forcing constants
_SIGN_CONST = [0x80000000, 0x80000001, 0xFFFFFFFF, 0x7FFFFFFF,
                0x80000002, 0xC0000000]

# Spice operations for random filler
_ALU_FILLER = ['add.u32', 'sub.u32', 'and.b32', 'or.b32', 'xor.b32',
                'min.u32', 'max.u32', 'add.s32', 'sub.s32']
_IMM_FILLER = [0, 1, 2, 3, 7, 15, 16, 31, 32,
                0xFF, 0x100, 0xFFFF, 0x10000]


def _pick(rnd, xs):
    return rnd.choice(xs)


# Each template returns (list[str] body_lines, int last_reg).
# `last_reg` is the final register the store should read.

def _tmpl_or_msb_shr_fold(rnd, fillers):
    """Bug 2 shape: force MSB, arithmetic shift, logical extract."""
    a = _pick(rnd, [4, 5, 6, 7])
    b = a + 1
    c = b + 1
    const = _pick(rnd, _SIGN_CONST)
    N = _pick(rnd, _FOLD_N)
    U = 32 - N
    lines = [
        f'    or.b32 %r{a}, %r{_LOAD_REG}, {const};',
        f'    shr.s32 %r{b}, %r{a}, {N};',
        f'    shr.u32 %r{c}, %r{b}, {U};',
    ]
    return lines, c


def _tmpl_bfe_s32_oor(rnd, fillers):
    """Bug 1 shape: bfe.s32 with start >= 32."""
    a = _pick(rnd, [4, 5, 6, 7, 8])
    start = _pick(rnd, _BFE_START_OOR)
    length = _pick(rnd, _BFE_LEN)
    lines = [f'    bfe.s32 %r{a}, %r{_LOAD_REG}, {start}, {length};']
    return lines, a


def _tmpl_mul_lo_largeimm(rnd, fillers):
    """Pool-zero class: mul.lo.u32 with large immediate."""
    a = _pick(rnd, [4, 5, 6, 7])
    b = a + 1
    imm = _pick(rnd, _MUL_LARGE_IMM)
    typ = _pick(rnd, ['u32', 's32'])
    lines = [f'    mul.lo.{typ} %r{a}, %r{_LOAD_REG}, {imm};']
    # Sometimes feed into a subsequent ALU op — pool-zero bugs often
    # surface only after a consumer reads the (wrongly-loaded) value.
    if rnd.random() < 0.7:
        op = _pick(rnd, ['add.s32', 'sub.s32', 'xor.b32', 'or.b32'])
        lines.append(f'    {op} %r{b}, %r{_LOAD_REG}, %r{a};')
        return lines, b
    return lines, a


def _tmpl_shift_boundary(rnd, fillers):
    """shl then shr near 0/31 boundaries."""
    a = _pick(rnd, [4, 5, 6, 7])
    b = a + 1
    L = _pick(rnd, _SHIFT_BOUNDARY)
    R = _pick(rnd, _SHIFT_BOUNDARY)
    shr_kind = _pick(rnd, ['shr.s32', 'shr.u32'])
    lines = [
        f'    shl.b32 %r{a}, %r{_LOAD_REG}, {L};',
        f'    {shr_kind} %r{b}, %r{a}, {R};',
    ]
    return lines, b


def _tmpl_sign_flip_chain(rnd, fillers):
    """XOR/OR with 0x80000000 then signed ALU — sign-bit-sensitive
    codegen tends to diverge in this region."""
    a = _pick(rnd, [4, 5, 6, 7])
    b = a + 1
    flip = _pick(rnd, ['xor.b32', 'or.b32', 'and.b32'])
    const = _pick(rnd, _SIGN_CONST)
    op = _pick(rnd, ['add.s32', 'sub.s32', 'max.s32', 'min.s32',
                      'shr.s32', 'shl.b32'])
    if op.startswith(('shr', 'shl')):
        rhs = _pick(rnd, _SHIFT_BOUNDARY)
    else:
        rhs = _pick(rnd, _IMM_FILLER + [0x80000000, 0x7FFFFFFF])
    lines = [
        f'    {flip} %r{a}, %r{_LOAD_REG}, {const};',
        f'    {op} %r{b}, %r{a}, {rhs};',
    ]
    return lines, b


def _tmpl_bfi_oor(rnd, fillers):
    """bfi.b32 with pos+len > 32 — NVIDIA-acknowledged UB but OpenPTXas
    and ptxas sometimes disagree here."""
    a = _pick(rnd, [4, 5, 6, 7])
    pos = _pick(rnd, [0, 1, 8, 16, 24, 31])
    length = _pick(rnd, [16, 24, 31, 32])
    lines = [f'    bfi.b32 %r{a}, %r{_LOAD_REG}, %r{_LOAD_REG}, {pos}, {length};']
    return lines, a


def _tmpl_mad_wide(rnd, fillers):
    """mul.wide.{u32,s32} — 32x32->64 multiply.  Historically the home
    of the sub.s64 rotate miscompile family; still a rich area.
    Uses %rd6 (declared by the kernel template, not used by pro/epilogue)."""
    a = _pick(rnd, [4, 5, 6, 7])
    imm = _pick(rnd, _MUL_LARGE_IMM + [1, 2, 3, 7, 16, 256])
    kind = _pick(rnd, ['mul.wide.u32', 'mul.wide.s32'])
    lines = [
        f'    {kind} %rd6, %r{_LOAD_REG}, {imm};',
        f'    cvt.u32.u64 %r{a}, %rd6;',
    ]
    return lines, a


def _tmpl_cvt_chain(rnd, fillers):
    """cvt narrowing + widening — rounding mode edge cases."""
    # cvt.u16.u32 truncates, cvt.u32.u16 zero-extends; saturation modes
    # on float conversions are a separate nightmare, but we stick to int here.
    a = _pick(rnd, [4, 5, 6, 7])
    b = a + 1
    c = b + 1
    lines = [
        f'    cvt.u16.u32 %r{a}, %r{_LOAD_REG};',   # truncate to 16 bits
        f'    cvt.s32.s16 %r{b}, %r{a};',           # sign-extend back to 32
        # Arithmetic on the sign-extended value
        f'    shr.s32 %r{c}, %r{b}, {_pick(rnd, [1, 15, 16, 31])};',
    ]
    return lines, c


def _tmpl_shf_funnel(rnd, fillers):
    """shf.l/shf.r funnel shift — 64-bit pair shifted to produce 32-bit."""
    a = _pick(rnd, [4, 5, 6, 7])
    b = a + 1
    shift_amt = _pick(rnd, _SHIFT_BOUNDARY)
    direction = _pick(rnd, ['shf.l.wrap.b32', 'shf.r.wrap.b32'])
    # shf.l.wrap %rdst, %rlo, %rhi, %shift  — lo=%r3, hi=%r3 (self for simplicity)
    lines = [
        f'    {direction} %r{a}, %r{_LOAD_REG}, %r{_LOAD_REG}, {shift_amt};',
        f'    add.u32 %r{b}, %r{a}, {_pick(rnd, _IMM_FILLER)};',
    ]
    return lines, b


def _tmpl_addcc_chain(rnd, fillers):
    """add.cc / addc carry chain — the sub.s64 rotate-miscompile family."""
    a = _pick(rnd, [4, 5, 6, 7])
    b = a + 1
    c = b + 1
    shift = _pick(rnd, [1, 2, 7, 15, 16, 31])
    # Classic pattern: shift then add-with-carry.  This is the shape
    # that triggered the original NVIDIA sub.s64 miscompile family.
    lines = [
        f'    add.cc.u32 %r{a}, %r{_LOAD_REG}, {_pick(rnd, _IMM_FILLER)};',
        f'    addc.u32 %r{b}, %r{_LOAD_REG}, {_pick(rnd, _IMM_FILLER)};',
        f'    shr.u32 %r{c}, %r{a}, {shift};',
    ]
    return lines, c


_TEMPLATES = [
    _tmpl_or_msb_shr_fold,
    # _tmpl_bfe_s32_oor,   # DISABLED: start>=32 triggers hardware trap on
                            # SM_120, poisons context irrecoverably.
                            # Bug 1 is already characterized offline.
    _tmpl_mul_lo_largeimm,
    _tmpl_shift_boundary,
    _tmpl_sign_flip_chain,
    _tmpl_mad_wide,          # NEW: 32x32->64 multiply surface
    _tmpl_cvt_chain,         # NEW: narrowing+widening cvt
    # _tmpl_shf_funnel,      # DISABLED: OpenPTXas doesn't implement shf.l/shf.r
    _tmpl_addcc_chain,       # NEW: carry-chain (sub.s64 family)
    # _tmpl_bfi_oor,       # DISABLED: same reason as bfe_oor.
]


def _filler_line(rnd, avail_regs):
    """Emit one random ALU op using available registers as sources and
    writing to an unused destination.  Returns (line, new_reg)."""
    op = _pick(rnd, _ALU_FILLER)
    a = _pick(rnd, avail_regs)
    use_imm = rnd.random() < 0.5
    if use_imm:
        b = str(_pick(rnd, _IMM_FILLER))
    else:
        b = f'%r{_pick(rnd, avail_regs)}'
    # Pick a destination register that's not currently special
    dst = _pick(rnd, [20, 21, 22, 23, 24, 25])
    return f'    {op} %r{dst}, %r{a}, {b};', dst


def generate(seed: int, n_instrs: int = 0) -> tuple[str, int]:
    """Emit one bug-adjacent kernel.  n_instrs is ignored (kept for
    interface parity with the other generators)."""
    rnd = random.Random(seed ^ 0xDA17E6)
    tmpl = _pick(rnd, _TEMPLATES)

    # How many pre-filler and post-filler ALU lines to spice with.
    # Small counts so the family_signature stays concise enough for
    # collisions with 3-specimen escalation.
    n_pre  = rnd.randint(0, 2)
    n_post = rnd.randint(0, 2)

    body = []
    avail = [_LOAD_REG]
    for _ in range(n_pre):
        line, new_reg = _filler_line(rnd, avail)
        body.append(line)
        if new_reg not in avail:
            avail.append(new_reg)

    core_lines, last = tmpl(rnd, avail)
    body.extend(core_lines)
    avail.append(last)

    for _ in range(n_post):
        line, new_reg = _filler_line(rnd, avail)
        body.append(line)
        if new_reg not in avail:
            avail.append(new_reg)
        last = new_reg

    body_text = '\n'.join(body)
    ptx = f""".version 9.0
.target sm_120
.address_size 64

.visible .entry fuzz(.param .u64 p_in, .param .u64 p_out, .param .u32 n) {{
    .reg .b32 %r<32>;
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
    # Quick visual check
    for s in range(6):
        ptx, r = generate(s)
        print(f'=== seed={s}, output=%r{r} ===')
        # Extract body
        lines = ptx.splitlines()
        bstart = next(i for i, ln in enumerate(lines) if 'ld.global.u32' in ln)
        bend = next(i for i, ln in enumerate(lines) if 'ld.param.u64 %rd3' in ln)
        for ln in lines[bstart+1:bend]:
            print(f'  {ln.strip()}')
        print()
