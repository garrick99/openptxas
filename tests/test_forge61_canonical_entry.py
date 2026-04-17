"""FORGE61-64 regression: canonical single-entry invariant for PTX merge blocks.

Proven defect (2026-04-17): OpenPTXas emitted BRA.U !UP0 with an unconditional
`+1` offset bump, designed to skip past a BSYNC.RECONVERGENT preamble that
isel.py emits for blocks whose first PTX instruction is `bar.sync`.  For PTX
merge blocks that begin with a *different* instruction (e.g. `setp`, which
becomes ISETP on SM_120), there is no BSYNC preamble, so the `+1` skipped the
canonical entry instruction, and incoming fall-through edges consumed a stale
predicate left by an earlier unrelated ISETP.

These tests construct small synthetic PTX fixtures that exercise the two
shape classes from the tiled-matmul slice and assert, at cubin level, that
every unconditional forward BRA lands exactly on the canonical entry of its
target PTX block.

Shape classes covered:
  1. `test_merge_block_inner_bra_and_outer_fallthrough` — setp-first merge
     with an inner true-branch edge AND an outer entry-fallthrough edge
     (models `if_merge_4`).
  2. `test_loop_cond_backedge_and_entry_edge` — setp-first while-condition
     block with a back-edge from the body AND an entry edge from the
     pre-loop initializer (models `while_cond_20`).
  3. `test_bar_merge_block_still_skips_bsync_preamble` — merge block that
     begins with `bar.sync`; BSYNC preamble is present, so BRA.U must
     still skip past it to the real BAR.SYNC (regression guard for the
     transpose slice, FORGE45-48).
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.parser import parse
from sass.pipeline import compile_function


def _text_section(cubin: bytes, kernel_name: str) -> bytes:
    e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]

    def sh(i):
        base = e_shoff + i * 64
        return struct.unpack_from('<IIQQQQIIQQ', cubin, base)

    _, _, _, _, shstr_off, shstr_size, *_ = sh(e_shstrndx)
    shstr = cubin[shstr_off:shstr_off + shstr_size]

    target = f'.text.{kernel_name}'.encode()
    for i in range(e_shnum):
        nm, ty, _, _, off, sz, *_ = sh(i)
        end = shstr.index(b'\x00', nm)
        if shstr[nm:end] == target and ty == 1:  # PROGBITS
            return cubin[off:off + sz]
    raise AssertionError(f'no .text.{kernel_name} section in cubin')


def _decode_bra_u_up0_target(text: bytes, addr: int) -> int:
    """Return the hardware-resolved target byte for a BRA.U !UP0 at addr.

    Hardware semantics on SM_120: target = next_pc + offset_instrs * 16,
    where offset_instrs = b2 / 4 (with high bits contributed by b4).
    """
    raw = text[addr:addr + 16]
    opc = (raw[0] | (raw[1] << 8)) & 0xFFF
    assert opc == 0x547, (
        f'expected BRA.U !UP0 at 0x{addr:x}, got opcode 0x{opc:03x}')
    b2 = raw[2]
    b4 = raw[4]
    # Low 6 bits of offset_instrs are in b2 >> 2 (since b2 = offset * 4).
    # High bits are encoded via (b4 - 1) // 4 shifted up by 6.
    offset_instrs = b2 // 4 + ((b4 - 1) // 4) * 64
    next_pc = addr + 16
    return next_pc + offset_instrs * 16


def _find_bra_u_targets(text: bytes) -> list[tuple[int, int]]:
    results = []
    for addr in range(0, len(text), 16):
        opc = (text[addr] | (text[addr + 1] << 8)) & 0xFFF
        if opc == 0x547:  # BRA.U !UP0
            results.append((addr, _decode_bra_u_up0_target(text, addr)))
    return results


def _compile(ptx_src: str) -> bytes:
    mod = parse(ptx_src)
    fn = mod.functions[0]
    return compile_function(fn, verbose=False, sm_version=120)


# ---------------------------------------------------------------------------
# Test 1 — setp-first merge block (FORGE61-64 if_merge_4 shape)
# ---------------------------------------------------------------------------

_PTX_INNER_BRA_AND_OUTER_FALLTHROUGH = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_merge_setp_first(
    .param .u64 out, .param .u32 n)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [out];
    ld.param.u32 %r0, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;

    setp.lt.u32 %p0, %r2, %r0;
    @%p0 bra if_true;
    bra merge;
if_true:
    add.u32 %r3, %r1, 1;
    bra merge;
merge:
    setp.lt.u32 %p0, %r1, %r0;
    @%p0 bra store_block;
    bra tail;
store_block:
    cvt.u64.u32 %rd1, %r1;
    shl.b64 %rd2, %rd1, 2;
    add.u64 %rd3, %rd0, %rd2;
    st.global.u32 [%rd3], %r1;
tail:
    ret;
}
"""


def test_merge_block_inner_bra_and_outer_fallthrough():
    cubin = _compile(_PTX_INNER_BRA_AND_OUTER_FALLTHROUGH)
    text = _text_section(cubin, 'k_merge_setp_first')

    # Every BRA.U !UP0 must land on the *canonical entry* of its target —
    # an ISETP for this kernel shape (the merge block recomputes its guard).
    # Landing one instruction past the ISETP would mean the fix regressed.
    # ISETP has two SM_120 opcode variants: 0xc0c (R-UR) and 0x20c (R-R).
    _ISETP_OPCODES = {0xc0c, 0x20c}
    bras = _find_bra_u_targets(text)
    assert bras, 'expected at least one BRA.U !UP0 in kernel'
    for addr, tgt in bras:
        opc = (text[tgt] | (text[tgt + 1] << 8)) & 0xFFF
        # Canonical entry of a setp-first merge block is an ISETP. BSYNC
        # (0x941) is allowed only as an intentional preamble skip target of
        # another BRA.U (covered by test 3) — it must NOT appear here. A
        # BRA-after-ISETP landing (opc=0x947) would indicate the pre-fix
        # behavior where `+1` skipped the guard-compute.
        assert opc != 0x941, (
            f'BRA.U at 0x{addr:x} landed on BSYNC at 0x{tgt:x} — expected a '
            f'real canonical entry instruction; regressed to pre-fix behavior')
        assert opc != 0x947, (
            f'BRA.U at 0x{addr:x} → 0x{tgt:x} landed on a BRA (0x947); the '
            f'`+1` offset bump was applied to a non-BSYNC target and skipped '
            f'the setp/ISETP at the canonical entry — FORGE61-64 regression.')
        assert opc in _ISETP_OPCODES, (
            f'BRA.U at 0x{addr:x} → 0x{tgt:x} opc=0x{opc:03x}; '
            f'expected ISETP (0xc0c or 0x20c) for setp-first merge block. '
            f'Landing on a non-canonical interior instruction would skip '
            f'the merge block\'s guard-compute.')


# ---------------------------------------------------------------------------
# Test 2 — setp-first while-condition with back-edge + entry edge
# ---------------------------------------------------------------------------

_PTX_LOOP_COND_BACKEDGE_AND_ENTRY = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_while_setp_first(
    .param .u64 out, .param .u32 n)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [out];
    ld.param.u32 %r0, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
    bra loop_cond;
loop_body:
    add.u32 %r2, %r2, %r1;
    add.u32 %r3, %r3, 1;
    bra loop_cond;
loop_cond:
    setp.lt.u32 %p0, %r3, %r0;
    @%p0 bra loop_body;
    bra loop_exit;
loop_exit:
    cvt.u64.u32 %rd1, %r1;
    shl.b64 %rd2, %rd1, 2;
    add.u64 %rd3, %rd0, %rd2;
    st.global.u32 [%rd3], %r2;
    ret;
}
"""


def test_loop_cond_backedge_and_entry_edge():
    cubin = _compile(_PTX_LOOP_COND_BACKEDGE_AND_ENTRY)
    text = _text_section(cubin, 'k_while_setp_first')

    bras = _find_bra_u_targets(text)
    assert bras, 'expected at least one BRA.U !UP0 in kernel'

    # The entry edge `bra loop_cond` is emitted as a forward BRA.U !UP0.
    # It must land on the ISETP at the head of `loop_cond` so the guard is
    # recomputed for iteration 0.  Landing past the ISETP would re-use a
    # stale predicate and cause the dot-product-style failure class.
    _ISETP_OPCODES = {0xc0c, 0x20c}
    found_forward_to_isetp = False
    for addr, tgt in bras:
        opc = (text[tgt] | (text[tgt + 1] << 8)) & 0xFFF
        assert opc != 0x941, (
            f'BRA.U at 0x{addr:x} landed on BSYNC at 0x{tgt:x}')
        assert opc != 0x947, (
            f'BRA.U at 0x{addr:x} → 0x{tgt:x} landed on a BRA (0x947); the '
            f'`+1` bump was applied to a non-BSYNC target and skipped the '
            f'loop-condition guard — FORGE61-64 regression class.')
        if opc in _ISETP_OPCODES:
            found_forward_to_isetp = True
    assert found_forward_to_isetp, (
        'expected at least one forward BRA.U !UP0 landing on an ISETP '
        '(the while-condition guard-compute)')


# ---------------------------------------------------------------------------
# Test 3 — bar-first merge block must still skip the BSYNC preamble
# ---------------------------------------------------------------------------

_PTX_BAR_FIRST_MERGE = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_merge_bar_first(
    .param .u64 out, .param .u32 n)
{
    .shared .u32 smem[64];
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [out];
    ld.param.u32 %r0, [n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u64 %rd1, smem;

    setp.lt.u32 %p0, %r2, %r0;
    @%p0 bra if_true;
    bra merge;
if_true:
    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd1, %rd1, %rd3;
    st.shared.u32 [%rd1], %r1;
    bra merge;
merge:
    bar.sync 0;
    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd1, %rd0, %rd3;
    st.global.u32 [%rd1], %r1;
    ret;
}
"""


def test_bar_merge_block_still_skips_bsync_preamble():
    cubin = _compile(_PTX_BAR_FIRST_MERGE)
    text = _text_section(cubin, 'k_merge_bar_first')

    bras = _find_bra_u_targets(text)
    assert bras, 'expected at least one BRA.U !UP0 in kernel'

    # For bar-first merge blocks, isel emits BSYNC.RECONVERGENT (opcode
    # 0x941) immediately before BAR.SYNC (opcode 0xb1d).  The canonical
    # entry is the BSYNC (which carries the label tag); BRA.U must
    # offset one instruction past it so the jump lands on the real
    # BAR.SYNC and does not execute the reconvergence marker
    # out-of-context.  Regression guard for FORGE45-48 transpose slice.
    landed_on_bar_sync = False
    for addr, tgt in bras:
        opc = (text[tgt] | (text[tgt + 1] << 8)) & 0xFFF
        if opc == 0xb1d:
            # Previous instruction should be the BSYNC preamble.
            prev_opc = (text[tgt - 16] | (text[tgt - 15] << 8)) & 0xFFF
            assert prev_opc == 0x941, (
                f'BRA.U at 0x{addr:x} → BAR.SYNC at 0x{tgt:x} but the '
                f'preceding instruction is 0x{prev_opc:03x}, not BSYNC. '
                f'BSYNC preamble is required for correct reconvergence.')
            landed_on_bar_sync = True
    assert landed_on_bar_sync, (
        'expected at least one BRA.U !UP0 to land on BAR.SYNC past a '
        'BSYNC preamble (the bar-first merge entry path)')
