"""R42 audit: dump label_abs_byte, BRA fixup decisions, and final-stream
label positions for w1_loop_sum.  Compare against decoded BRA target.
"""
from __future__ import annotations
import struct, sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')

import workbench_expanded as we
import sass.pipeline as P
from ptx.parser import parse


# Instrument the BRA fixup by wrapping/monkey-patching.  We wrap the
# SassInstr constructor inside the fixup block by hooking the fixup's
# `label_abs_byte` dict and BRA iteration.  Simpler: just re-run the
# fixup logic manually using the internal state.
#
# Since the fixup uses ctx._bra_fixups and ctx.label_map which are set
# during compile_function, we recompile and capture those values.

_orig_compile = P.compile_function


def compile_and_dump(fn):
    # Patch: replace the fixup block's logging by monkey-patching.
    # Approach: inject a print just before the fixup mutates the BRA.
    # We do this by re-reading ctx from a wrapped compile_function.
    import types

    # Inject a hook that prints label_abs_byte and BRA fixup decisions.
    # The fixup runs inline in compile_function, so we can't easily hook
    # mid-flight without modifying pipeline.py.  Instead, we'll:
    # 1. Compile normally -> get final cubin
    # 2. Re-compute label_abs_byte-like mapping by scanning comments
    # 3. Decode BRA targets from the emitted cubin
    cubin = _orig_compile(fn, verbose=False, sm_version=120)

    # Find .text section.
    e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]
    def sh(i): return struct.unpack_from('<IIQQQQIIQQ', cubin, e_shoff + i * 64)
    _, _, _, _, so, ss, *_ = sh(e_shstrndx)
    shs = cubin[so:so + ss]
    for i in range(e_shnum):
        nm, ty, _, _, off, sz, *_ = sh(i)
        end = shs.index(b'\x00', nm)
        if shs[nm:end].startswith(b'.text.') and ty == 1:
            text = cubin[off:off + sz]
            kname = shs[nm:end].decode()
            break

    return cubin, text, kname


def audit(ptx):
    # Hook the BRA fixup to print intermediate state.  We monkey-patch
    # SassInstr's constructor to log when a BRA is rewritten, but easier:
    # just reimplement the label_abs_byte computation via ctx snapshot.
    import sass.pipeline as P

    # Wrap compile_function to capture internal state.
    orig = P.compile_function
    captured = {}

    def wrap(fn, verbose=False, sm_version=120, **kw):
        # Instrument by replacing SassInstr with a recording class in the
        # fixup range is too invasive.  Instead: run with verbose=True and
        # parse stdout for the "BRA ... (offset=...)" comments.
        return orig(fn, verbose=True, sm_version=sm_version, **kw)

    P.compile_function = wrap
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cubin = P.compile_function(parse(ptx).functions[0], verbose=True, sm_version=120)
        verbose_out = buf.getvalue()
    finally:
        P.compile_function = orig

    # Find .text.
    e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]
    def sh(i): return struct.unpack_from('<IIQQQQIIQQ', cubin, e_shoff + i * 64)
    _, _, _, _, so, ss, *_ = sh(e_shstrndx)
    shs = cubin[so:so + ss]
    text = b''
    for i in range(e_shnum):
        nm, ty, _, _, off, sz, *_ = sh(i)
        end = shs.index(b'\x00', nm)
        if shs[nm:end].startswith(b'.text.') and ty == 1:
            text = cubin[off:off + sz]
            break

    # Parse verbose output for BRA fixup decisions and label tags.
    print('=== Verbose BRA-related lines ===')
    for line in verbose_out.splitlines():
        if 'BRA' in line or 'LOOP' in line or 'loop' in line or 'offset=' in line:
            print('  ', line)
    print()

    # Decode every BRA in the final cubin.  For a predicated BRA, decode
    # the 18-bit signed offset per pipeline.py:2059-2077 encoding:
    #   total = offset_instrs * 4 (signed 18-bit)
    #   b2 = total & 0xFF
    #   b4 = ((total >> 8) << 2) & 0xFF  (6 bits, shifted up by 2)
    #   b10 = 0x80 | ((total >> 16) & 0x03)
    print('=== Final-stream SASS (opc + targets) ===')
    loop_label_pos = None
    for a in range(0, len(text), 16):
        r = text[a:a + 16]
        opc = (r[0] | (r[1] << 8)) & 0xFFF
        if opc == 0x947:
            guard = (r[1] >> 4) & 0xF
            # Reconstruct total (signed): b2 is [7:0], b4[7:2] is [13:8],
            # b10[1:0] is [17:16].  Bits [15:14] are sign-extension from
            # bit 17 (they are not encoded).  Reconstruct by taking the
            # 14-bit encoded value and sign-extending from bit 17.
            total = r[2] | (((r[4] >> 2) & 0x3F) << 8) | ((r[10] & 0x03) << 16)
            # Sign-extend from bit 17.
            if (r[10] & 0x02):  # bit 17 set
                total |= 0xC000  # fill bits 15:14
                total -= (1 << 18)
            offset_instrs = total // 4
            rel_offset_bytes = offset_instrs * 16
            # Target = next_pc + rel_offset = (pos+1)*16 + rel_offset
            next_pc = (a // 16 + 1) * 16
            target_byte = next_pc + rel_offset_bytes
            target_pos = target_byte // 16
            ctrl24 = (r[15] << 16) | (r[14] << 8) | r[13]
            ctrl = ctrl24 >> 1
            wdep = (ctrl >> 4) & 0x3f
            rbar = (ctrl >> 10) & 0x1f
            print(f'  pos {a//16:2d} BRA guard={guard} total={total} '
                  f'off_instrs={offset_instrs} target_byte=0x{target_byte:03x} '
                  f'(pos {target_pos})  wdep=0x{wdep:02x} rbar=0x{rbar:02x}')
        elif opc in (0x94d,):
            guard = (r[1] >> 4) & 0xF
            print(f'  pos {a//16:2d} EXIT guard={guard}')
        # Flag likely LOOP-body start — first IADD3 with R4 dest reading R4 (loop-carried acc)
        if opc == 0x210 and r[2] == 0x04 and r[3] == 0x04:
            if loop_label_pos is None:
                loop_label_pos = a // 16
                print(f'  pos {a//16:2d} IADD3 R4,R4,...  (candidate LOOP label)')
    print()

    print(f'=== Candidate LOOP label position in final stream: pos {loop_label_pos} ===')


if __name__ == '__main__':
    print('\n### w1_loop_sum ###\n')
    audit(we._W1_LOOP_SUM)
