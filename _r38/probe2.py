"""R38 follow-up: pinpoint which exact gap fixes s2_fail."""
from __future__ import annotations
import ctypes, struct, subprocess, sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')
sys.path.insert(0, '_r38')
from probe import build_variant as _build_variant_base, run, NOP

from sass.pipeline import compile_function
from ptx.parser import parse


_PTX = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry s2_fail(.param .u64 in, .param .u64 out) {
    .reg .b32 %r<4>; .reg .b64 %rd<3>; .reg .pred %p<1>;
    ld.param.u64 %rd0, [in]; ld.param.u64 %rd1, [out];
    ld.global.u32 %r0, [%rd0]; mov.u32 %r1, %tid.x; setp.eq.u32 %p0, %r1, 0; @!%p0 ret;
    mov.u32 %r2, %ctaid.x; shl.b32 %r3, %r2, 2; cvt.u64.u32 %rd2, %r3;
    add.u64 %rd1, %rd1, %rd2; st.global.u32 [%rd1], %r0; ret;
}
"""


def _find_text(cubin):
    e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]
    def sh(i): return struct.unpack_from('<IIQQQQIIQQ', cubin, e_shoff + i * 64)
    _, _, _, _, so, ss, *_ = sh(e_shstrndx)
    shs = cubin[so:so + ss]
    for i in range(e_shnum):
        nm, ty, _, _, off, sz, *_ = sh(i)
        end = shs.index(b'\x00', nm)
        if shs[nm:end] == b'.text.s2_fail' and ty == 1:
            return off, sz
    raise AssertionError('no .text.s2_fail')


def build_insert_at(insert_idx: int) -> bytes:
    """Insert NOP at the given post-compact instruction index via in-place
    shift (eats one trailing padding NOP)."""
    cubin = bytearray(compile_function(parse(_PTX).functions[0],
                                        verbose=False, sm_version=120))
    sec_off, sec_sz = _find_text(cubin)
    text = bytearray(cubin[sec_off:sec_off + sec_sz])
    n = len(text) // 16
    instrs = [bytes(text[i * 16:(i + 1) * 16]) for i in range(n)]

    trailing = 0
    for i in range(n - 1, -1, -1):
        if instrs[i] == NOP:
            trailing += 1
        else:
            break

    # Shift [insert_idx, n-trailing) by 1, NOP at insert_idx.
    body = instrs[insert_idx : n - trailing]
    new_instrs = (
        instrs[:insert_idx]
        + [NOP]
        + body
        + instrs[n - trailing + 1:]
    )
    assert len(new_instrs) == n
    new_text = b''.join(new_instrs)
    new_cubin = bytearray(cubin)
    new_cubin[sec_off:sec_off + sec_sz] = new_text
    return bytes(new_cubin)


# Map labels to insertion indices based on s2_fail baseline SASS.
# Baseline post-compact layout:
#   [0] LDC frame    [5] LDG.E        [10] NOP         [15] NOP
#   [1] S2R tid      [6] ISETP        [11] IADD3 R2    [16] STG.E
#   [2] LDCU UR4     [7] EXIT         [12] IADD3 R3    [17] EXIT
#   [3] LDCU UR8     [8] S2R CTAID    [13] NOP         [18] BRA
#   [4] LDC in_ptr   [9] IMAD.SHL     [14] IADD.64
VARIANTS = [
    ('baseline (none)',                 -1),
    ('NOP before S2R-CTAID  (@8)',       8),
    ('NOP after S2R-CTAID   (@9)',       9),
    ('NOP after IMAD.SHL    (@10)',     10),
    ('NOP after MOV R2,R3   (@11)',     11),
    ('NOP after MOV R3,RZ   (@12)',     12),
]


if __name__ == '__main__':
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        label, insert_idx = VARIANTS[idx]
        if insert_idx == -1:
            cubin = _build_variant_base(None, 'none')
        else:
            cubin = build_insert_at(insert_idx)
        run(cubin, label)
    else:
        for i in range(len(VARIANTS)):
            subprocess.run([sys.executable, __file__, str(i)])
