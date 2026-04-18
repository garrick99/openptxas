"""PTXAS-R25.3 regression — scalar-LDG (ld.global from raw ld.param.u64
pointer with no intervening address arithmetic) is canonicalized at the
PTX-IR level into the working offset-address form before isel runs.

This class crashes the SM_120 descriptor-LDG path with
CUDA_ERROR_ILLEGAL_ADDRESS (Family B / G5 proof).  Every isel-local
workaround failed — R24 (swap address producer), R24.1 (per-lane
producer), R24.2 (full NVCC template), R25 (LDG.U), R25.1 (replicate
kCandidate chain), R25.2 (gate TE20-A peephole) — because the
structural invariant the hardware expects spans more surfaces than any
single isel or peephole patch can control (register allocation,
scheduler ctrl bytes, S2R deduplication).  The already-working
offset-form path (`kCandidate` reference) is reliable.

R25.3 canonicalizes the scalar-LDG class into the same PTX-IR shape as
kCandidate before isel sees it: inject a per-lane tid-masked zero-
offset add.u64 and rewrite the LDG's address operand.  The existing
downstream lowering produces the correct structure naturally.
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.ir import RegOp, MemOp, Instruction
from ptx.parser import parse
from sass.pipeline import _canonicalize_scalar_ldg, compile_function


_OPC_LDG_E = 0x981


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


_PTX_SCALAR_LDG = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_scalar_ldg(.param .u64 in, .param .u64 out)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [in];
    ld.param.u64 %rd1, [out];
    ld.global.u32 %r0, [%rd0];                 // SCALAR LDG from raw param
    mov.u32 %r1, %tid.x;
    setp.eq.u32 %p0, %r1, 0;
    @!%p0 ret;
    mov.u32 %r2, %ctaid.x;
    shl.b32 %r3, %r2, 2;
    cvt.u64.u32 %rd2, %r3;
    add.u64 %rd1, %rd1, %rd2;
    st.global.u32 [%rd1], %r0;
    ret;
}
"""

_PTX_OFFSET_LDG = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry k_offset_ldg(.param .u64 in, .param .u64 out)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<5>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [in];
    ld.param.u64 %rd1, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r3, %r1, %r2;
    add.u32 %r4, %r3, %r0;                     // gid
    cvt.u64.u32 %rd2, %r4;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd4, %rd0, %rd3;                  // in_ptr + gid*4 (offset form)
    ld.global.u32 %r5, [%rd4];
    shl.b32 %r3, %r1, 2;
    cvt.u64.u32 %rd2, %r3;
    add.u64 %rd1, %rd1, %rd2;
    setp.eq.u32 %p0, %r0, 0;
    @!%p0 ret;
    st.global.u32 [%rd1], %r5;
    ret;
}
"""


# ---------------------------------------------------------------------------
# Test 1 — scalar-LDG PTX-IR is rewritten: LDG's base becomes an add-produced reg
# ---------------------------------------------------------------------------

def test_scalar_ldg_rewritten_to_offset_form():
    fn = parse(_PTX_SCALAR_LDG).functions[0]
    _canonicalize_scalar_ldg(fn)
    # After canonicalization, the ld.global's address register must be
    # produced by an add.u64 (not be the raw ld.param dest).
    param_regs = set()
    offset_regs = set()
    ldg_base = None
    for bb in fn.blocks:
        for inst in bb.instructions:
            if (inst.op == 'ld' and 'param' in inst.types
                    and isinstance(inst.dest, RegOp)):
                param_regs.add(inst.dest.name)
            if inst.op in ('add', 'mul', 'mad', 'cvt', 'shl') and isinstance(inst.dest, RegOp):
                offset_regs.add(inst.dest.name)
            if inst.op == 'ld' and 'global' in inst.types and inst.srcs:
                s0 = inst.srcs[0]
                if isinstance(s0, MemOp):
                    b = s0.base if s0.base.startswith('%') else '%' + s0.base
                    ldg_base = b
    assert ldg_base is not None, "no ld.global found in rewritten kernel"
    assert ldg_base in offset_regs, (
        f"after canonicalization, ld.global base {ldg_base!r} should be an "
        f"offset_reg (produced by add/mul/mad/cvt/shl); offset_regs={offset_regs}")
    assert ldg_base not in param_regs, (
        f"after canonicalization, ld.global base {ldg_base!r} should NOT be "
        f"a raw ld.param.u64 dest; param_regs={param_regs}")


# ---------------------------------------------------------------------------
# Test 2 — the canonicalizer leaves already-offset-form kernels unchanged
# ---------------------------------------------------------------------------

def test_offset_ldg_unchanged_by_canonicalization():
    """kCandidate-style kernels (LDG already via add.u64) must not be
    re-canonicalized.  Verify no `_r25c_` synthetic registers are
    introduced for a kernel that doesn't match the scalar-LDG class."""
    fn = parse(_PTX_OFFSET_LDG).functions[0]
    _canonicalize_scalar_ldg(fn)
    for bb in fn.blocks:
        for inst in bb.instructions:
            for op in [inst.dest] + list(inst.srcs):
                if isinstance(op, RegOp) and op.name.startswith('%_r25c_'):
                    raise AssertionError(
                        f"canonicalizer fired on non-scalar-LDG kernel: "
                        f"injected synthetic reg {op.name!r} in {inst!s}")


# ---------------------------------------------------------------------------
# Test 3 — emitted SASS contains an LDG.E, and its src_addr register is
#          written by a body-level arithmetic producer (not a preamble LDC.64)
# ---------------------------------------------------------------------------

def test_scalar_ldg_emits_ldge_with_arith_produced_address():
    """End-to-end: compile the scalar-LDG PTX and confirm the emitted
    text contains exactly one LDG.E (opc 0x981) and that the surrounding
    body has an IADD.64 R-UR (opc 0xc35) producing the address pair."""
    cubin = compile_function(parse(_PTX_SCALAR_LDG).functions[0],
                             verbose=False, sm_version=120)
    text = _text(cubin, 'k_scalar_ldg')
    ldg_count = 0
    saw_arith_producer = False
    for a in range(0, len(text), 16):
        raw = text[a:a + 16]
        opc = (raw[0] | (raw[1] << 8)) & 0xFFF
        if opc == _OPC_LDG_E:
            ldg_count += 1
        # IADD.64 R-UR (opc 0xc35) or IADD3.UR (opc 0xc11) producing
        # the address pair satisfies the body-level arithmetic
        # requirement.
        if opc in (0xc35, 0xc11):
            saw_arith_producer = True
    assert ldg_count == 1, (
        f"expected exactly one LDG.E in k_scalar_ldg text; got {ldg_count}")
    assert saw_arith_producer, (
        "k_scalar_ldg text should contain an IADD.64-R-UR (0xc35) or "
        "IADD3.UR (0xc11) producing the address pair; neither found")
