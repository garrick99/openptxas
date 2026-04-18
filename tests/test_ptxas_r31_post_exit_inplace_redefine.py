"""PTXAS-R31 regression — UR-route u64 params redefined in-place across
a predicated EXIT.

R30 proof: the PTX pattern

    ld.param.u64 %rdN, [p];
    ... @!%pX ret; ...
    add.u64 %rdN, %rdN, X;
    st.global.u32 [%rdN], ...;

forces regalloc to see ``%rdN`` as GPR-resident (u64 has >1 def) which
routes isel's ``_select_ld_param`` into the "GPR direct" body ``LDC.64
R_pair, c[0][param_off]`` branch.  That body LDC.64 pre-EXIT +
in-place ``IADD3`` post-EXIT + STG combination produces
CUDA_ERROR_ILLEGAL_ADDRESS on SM_120.  The safe path — proven by the
``offset_distinct_dest`` R30 repro — routes ``%rdN`` through UR
(preamble ``ULDCU.64`` + body ``IADD.64 R-UR`` into a fresh pair +
STG via fresh pair).

R31's transform renames the redefine's dest to a fresh vreg
(``%__r31_<orig>_<idx>``) and rewrites every subsequent use of
``%rdN`` to the fresh vreg.  After the rename, ``%rdN`` has exactly
one def (the ``ld.param.u64``) and exactly one use (the new add's
src), so regalloc leaves it on the UR path and isel emits
preamble ``ULDCU.64``.  ``regalloc`` also consults
``fn._r31_force_ur_params`` to override the R22 misaligned-addr-arith
exclusion for these specific params.
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.ir import RegOp
from ptx.parser import parse
from sass.pipeline import (
    _if_convert,
    _sink_param_loads,
    _r31_rename_inplace_u64_redefine_across_exit,
    compile_function,
)


_PTX_UNSAFE_INPLACE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k_r31_unsafe(.param .u64 out) {
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    .reg .pred %p<1>;
    ld.param.u64 %rd1, [out];
    mov.u32 %r0, 777;
    mov.u32 %r1, %tid.x;
    setp.eq.u32 %p0, %r1, 0;
    @!%p0 ret;
    add.u64 %rd1, %rd1, 0;
    st.global.u32 [%rd1], %r0;
    ret;
}
"""


_PTX_SAFE_PRE_EXIT = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k_r31_safe_pre(.param .u64 out) {
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    .reg .pred %p<1>;
    ld.param.u64 %rd1, [out];
    mov.u32 %r0, 777;
    add.u64 %rd1, %rd1, 0;
    mov.u32 %r1, %tid.x;
    setp.eq.u32 %p0, %r1, 0;
    @!%p0 ret;
    st.global.u32 [%rd1], %r0;
    ret;
}
"""


# ---------------------------------------------------------------------------
# Test 1 — the PTX-IR transform triggers on the unsafe pattern: the
#          redefine instruction's dest is renamed to a fresh `%__r31_*`
#          vreg, and subsequent uses are rewritten to the fresh vreg.
# ---------------------------------------------------------------------------

def test_r31_rename_triggers_on_unsafe_pattern():
    fn = parse(_PTX_UNSAFE_INPLACE).functions[0]
    _if_convert(fn)
    _sink_param_loads(fn)
    _r31_rename_inplace_u64_redefine_across_exit(fn)

    # The `add.u64` that used to redefine %rd1 must now write to a
    # fresh `%__r31_rd1_*` vreg.
    add_insts = [
        inst
        for bb in fn.blocks for inst in bb.instructions
        if inst.op == 'add' and 'u64' in inst.types
    ]
    assert len(add_insts) == 1, (
        f"expected exactly one add.u64; got {len(add_insts)}")
    add_dest = add_insts[0].dest
    assert isinstance(add_dest, RegOp), f"add.u64 dest not RegOp: {add_dest}"
    assert add_dest.name.startswith('%__r31_rd1_'), (
        f"R31 rename did not fire: add.u64 dest still {add_dest.name!r}, "
        f"expected a fresh %__r31_rd1_* vreg")

    # The store's MemOp base must reference the renamed vreg, not %rd1.
    st_insts = [
        inst
        for bb in fn.blocks for inst in bb.instructions
        if inst.op == 'st' and 'global' in inst.types
    ]
    assert len(st_insts) == 1
    memop = st_insts[0].srcs[0]
    assert memop.base == add_dest.name, (
        f"st.global base {memop.base!r} does not match renamed add dest "
        f"{add_dest.name!r}; post-rename uses must be rewritten")

    # fn._r31_force_ur_params must include %rd1 so regalloc forces UR.
    assert '%rd1' in getattr(fn, '_r31_force_ur_params', set()), (
        "R31 must mark %rd1 in fn._r31_force_ur_params so regalloc "
        "overrides the R22 misaligned-addr-arith exclusion and routes "
        "the param via preamble ULDCU.64")


# ---------------------------------------------------------------------------
# Test 2 — the safe pattern (redefine BEFORE the predicated EXIT) is
#          left unchanged.  R31 must not fire on pre-EXIT redefines.
# ---------------------------------------------------------------------------

def test_r31_does_not_fire_on_pre_exit_redefine():
    fn = parse(_PTX_SAFE_PRE_EXIT).functions[0]
    _if_convert(fn)
    _sink_param_loads(fn)
    _r31_rename_inplace_u64_redefine_across_exit(fn)

    # No fresh %__r31_* vreg should be introduced.
    for bb in fn.blocks:
        for inst in bb.instructions:
            for op in [inst.dest] + list(inst.srcs):
                if isinstance(op, RegOp) and op.name.startswith('%__r31_'):
                    raise AssertionError(
                        f"R31 wrongly fired on safe pre-EXIT redefine: "
                        f"introduced {op.name!r} in {inst}")

    # fn._r31_force_ur_params must be empty / unset.
    assert not getattr(fn, '_r31_force_ur_params', set()), (
        "R31 must not mark any param in _r31_force_ur_params when the "
        "redefine is before the predicated EXIT")


# ---------------------------------------------------------------------------
# Test 3 — end-to-end: the emitted SASS for the unsafe pattern no
#          longer contains a body ``LDC.64`` (GPR-direct param load)
#          for the `out` param — it goes through preamble ``LDCU.64``
#          and body ``IADD.64 R-UR`` into a fresh pair instead.
# ---------------------------------------------------------------------------

_OPC_LDC = 0xb82
_OPC_LDCU = 0x7ac
_OPC_IADD64_RUR = 0xc35


def _text(cubin: bytes, kernel: str) -> bytes:
    e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]

    def sh(i):
        return struct.unpack_from('<IIQQQQIIQQ', cubin, e_shoff + i * 64)

    _, _, _, _, so, ss, *_ = sh(e_shstrndx)
    shs = cubin[so:so + ss]
    target = f'.text.{kernel}'.encode()
    for i in range(e_shnum):
        nm, ty, _, _, off, sz, *_ = sh(i)
        end = shs.index(b'\x00', nm)
        if shs[nm:end] == target and ty == 1:
            return cubin[off:off + sz]
    raise AssertionError(f'no .text.{kernel}')


def test_r31_no_body_ldc_param_for_unsafe_pattern():
    cubin = compile_function(parse(_PTX_UNSAFE_INPLACE).functions[0],
                             verbose=False, sm_version=120)
    text = _text(cubin, 'k_r31_unsafe')

    saw_param_ldc = False
    saw_param_ldcu = False
    saw_iadd64_rur = False
    for a in range(0, len(text), 16):
        raw = text[a:a + 16]
        opc = (raw[0] | (raw[1] << 8)) & 0xFFF
        # LDC / LDCU byte-offset: raw[5] is dword_offset; * 4 == bytes.
        # Param area starts at c[0][0x380] → raw[5] >= 0x70 (0x380/8=0x70).
        if opc == _OPC_LDC and raw[9] == 0x0a and raw[5] >= 0x70:
            saw_param_ldc = True
        if opc == _OPC_LDCU and raw[9] in (0x0a, 0x0c) and raw[5] >= 0x70:
            saw_param_ldcu = True
        if opc == _OPC_IADD64_RUR:
            saw_iadd64_rur = True

    assert not saw_param_ldc, (
        "R31 regression: the unsafe pattern still emits a body LDC.64 "
        "(opc 0xb82, b9=0x0a, raw[5]>=0x70) for a u64 param — this is "
        "the GPR-direct path R30 proved crashes when combined with an "
        "in-place post-EXIT redefine.  The param must route through "
        "preamble LDCU.64 instead.")
    assert saw_param_ldcu, (
        "R31 expected a preamble LDCU.64 (opc 0x7ac, b9=0x0a or 0x0c, "
        "raw[5]>=0x70) for the UR-routed param, but none found.  "
        "The UR path is the proven-safe lowering.")
    assert saw_iadd64_rur, (
        "R31 expected a body IADD.64 R-UR (opc 0xc35) that consumes the "
        "UR-routed param and writes a fresh GPR pair for the store, "
        "but none found.")
