"""Tests for the M31 Mersenne-prime modular-reduction PTX-IR pass.

Phase 40: covers `rem.u64 %d, %x, 2147483647` (2^31 - 1) recognition
and rewrite to a 9-op shift/and/add/setp/sub chain.
"""
from __future__ import annotations

import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from ptx.parser import parse
from ptx.ir import ImmOp, RegOp
from ptx.passes.m31_mod_fast_path import run_function


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_instrs(fn):
    for bb in fn.blocks:
        yield from bb.instructions


def _ops(fn):
    return [inst.op for inst in _all_instrs(fn)]


def _wrap(body: str) -> str:
    """Wrap a PTX body in a minimal kernel envelope."""
    return f"""\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a, .param .u64 b)
{{
    .reg .b64 %rd<32>;
    .reg .pred %p<4>;
    ld.param.u64 %rd1, [a];
    ld.global.u64 %rd2, [%rd1];
{body}
    ld.param.u64 %rd4, [b];
    st.global.u64 [%rd4], %rd3;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Part C: positive recognition
# ---------------------------------------------------------------------------

def test_recognizes_m31_mod_direct_form():
    """`rem.u64 %d, %x, 2147483647` is rewritten."""
    mod = parse(_wrap("    rem.u64 %rd3, %rd2, 2147483647;"))
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 1
    ops = _ops(fn)
    assert "rem" not in ops
    assert "shr" in ops
    assert "and" in ops
    assert "add" in ops
    assert "setp" in ops
    assert "sub" in ops


def test_recognizes_m31_mod_indirect_form():
    """`mov %const, M31; rem %d, %x, %const` (single-use mov) is rewritten,
    and the mov is removed."""
    body = (
        "    mov.u64 %rd5, 2147483647;\n"
        "    rem.u64 %rd3, %rd2, %rd5;"
    )
    mod = parse(_wrap(body))
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 1
    ops = _ops(fn)
    assert "rem" not in ops
    # The original mov.u64 %rd5, 2147483647 must be gone.
    movs = [i for i in _all_instrs(fn)
            if i.op == "mov"
            and i.types and i.types[0] in ("u64", "b64", "s64")
            and i.srcs and isinstance(i.srcs[0], ImmOp)
            and i.srcs[0].value == 0x7FFFFFFF]
    assert not movs, "the M31 const-mov should be DCE'd by the pass"


def test_skips_other_divisors():
    """Other divisors are not rewritten."""
    mod = parse(_wrap("    rem.u64 %rd3, %rd2, 12345;"))
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0
    assert "rem" in _ops(fn)


def test_skips_other_mersenne_primes():
    """Conservative: only M31 (2^31 - 1) — not M61 or other Mersennes."""
    mod = parse(_wrap("    rem.u64 %rd3, %rd2, 2305843009213693951;"))  # M61
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0
    assert "rem" in _ops(fn)


def test_skips_predicated_rem():
    """Predicated rem.u64 is conservatively skipped."""
    body = (
        "    setp.ne.u64 %p1, %rd2, 0;\n"
        "    @%p1 rem.u64 %rd3, %rd2, 2147483647;"
    )
    mod = parse(_wrap(body))
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0
    rems = [i for i in _all_instrs(fn) if i.op == "rem"]
    assert len(rems) == 1
    assert rems[0].pred == "%p1"


def test_skips_rem_u32():
    """Only u64 — rem.u32 with M31 is left alone."""
    body = "    rem.u32 %r1, %r0, 2147483647;"
    ptx = f"""\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{{
    .reg .b64 %rd<5>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd1, [a];
    ld.global.u32 %r0, [%rd1];
{body}
    st.global.u32 [%rd1], %r1;
    ret;
}}
"""
    mod = parse(ptx)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0


def test_skips_indirect_when_mov_redefined():
    """If the mov-defined reg has another writer (def_count > 1), skip."""
    body = (
        "    mov.u64 %rd5, 2147483647;\n"
        "    mov.u64 %rd5, 99;\n"          # second def of %rd5
        "    rem.u64 %rd3, %rd2, %rd5;"
    )
    mod = parse(_wrap(body))
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0
    assert "rem" in _ops(fn)


def test_skips_indirect_when_const_used_elsewhere():
    """If the M31 constant has another reader, skip (don't DCE the mov)."""
    body = (
        "    mov.u64 %rd5, 2147483647;\n"
        "    rem.u64 %rd3, %rd2, %rd5;\n"
        "    add.u64 %rd6, %rd5, %rd2;\n"   # second use of %rd5
        "    st.global.u64 [%rd1], %rd6;"
    )
    ptx = f"""\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 a)
{{
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [a];
    ld.global.u64 %rd2, [%rd1];
{body}
    ret;
}}
"""
    mod = parse(ptx)
    fn = mod.functions[0]
    n = run_function(fn)
    assert n == 0


# ---------------------------------------------------------------------------
# Part C: structural shape of the rewrite
# ---------------------------------------------------------------------------

def test_rewrite_emits_expected_op_sequence():
    """Verify the rewrite shape: shr, and, add, shr, and, add, sub
    (unconditional), mov (default), setp, mov (predicated override).

    The sub is unconditional (not predicated) so the IADD3/IADD3.X
    carry chain at SASS level isn't itself predicated — the scoreboard
    does not track the IADD3.X P1 carry-in dependency; predicated
    carry-chain subs see stale-carry hazards on SM_120.
    """
    mod = parse(_wrap("    rem.u64 %rd3, %rd2, 2147483647;"))
    fn = mod.functions[0]
    run_function(fn)

    body_ops = []
    for inst in _all_instrs(fn):
        if inst.op in ("ld", "st", "ret"):
            continue
        body_ops.append(inst)

    expected_ops = ["shr", "and", "add", "shr", "and", "add",
                    "sub", "mov", "setp", "mov"]
    assert [b.op for b in body_ops] == expected_ops

    # The two shr instructions shift by 31.
    shrs = [b for b in body_ops if b.op == "shr"]
    for shr in shrs:
        assert isinstance(shr.srcs[1], ImmOp) and shr.srcs[1].value == 31

    # The two and instructions mask with 0x7FFFFFFF.
    ands = [b for b in body_ops if b.op == "and"]
    for a in ands:
        assert isinstance(a.srcs[1], ImmOp) and a.srcs[1].value == 0x7FFFFFFF

    # The sub is unconditional and subtracts M31.
    sub = [b for b in body_ops if b.op == "sub"][0]
    assert sub.pred is None, "sub must be unpredicated"
    assert isinstance(sub.srcs[1], ImmOp) and sub.srcs[1].value == 0x7FFFFFFF

    # The setp is .ge.u64 against M31.
    setp = [b for b in body_ops if b.op == "setp"][0]
    assert "ge" in setp.types and "u64" in setp.types
    assert isinstance(setp.srcs[1], ImmOp) and setp.srcs[1].value == 0x7FFFFFFF

    # Two movs: the first is the default (unpredicated) write of %d
    # = sum2; the second is the predicated override %d = tmp.
    movs = [b for b in body_ops if b.op == "mov"]
    assert len(movs) == 2
    assert movs[0].pred is None, "default mov must be unpredicated"
    assert movs[1].pred == setp.dest.name, \
        "override mov must be guarded by the setp predicate"


# ---------------------------------------------------------------------------
# End-to-end pipeline integration
# ---------------------------------------------------------------------------

def test_pipeline_compile_drops_rem_loop():
    """Compiling a kernel through the full pipeline with the pass enabled
    must NOT emit the bit-serial divide loop's signature (192 SHF.L.U64.HI
    instructions per rem.u64).  A simple proxy: total emitted SASS
    instructions stay well below the un-rewritten baseline."""
    from sass.pipeline import compile_ptx_source

    ptx = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 inp, .param .u64 outp)
{
    .reg .b64 %rd<8>;
    .reg .u32 %r<4>;
    mov.u32 %r0, %tid.x;
    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 3;
    ld.param.u64 %rd1, [inp];  add.u64 %rd2, %rd1, %rd0;
    ld.global.u64 %rd3, [%rd2];
    rem.u64 %rd4, %rd3, 2147483647;
    ld.param.u64 %rd5, [outp]; add.u64 %rd6, %rd5, %rd0;
    st.global.u64 [%rd6], %rd4;
    ret;
}
"""
    cubins = compile_ptx_source(ptx)
    assert "k" in cubins
    cubin = cubins["k"]
    # The bit-serial divide loop emits 64 iterations × ~7 SASS = ~448
    # body instrs in addition to the kernel scaffolding (~30 instrs).
    # The fast-path drops body to ~10 SASS, total ~50 with scaffolding.
    # Use a generous threshold (200) — anything close to 552 indicates
    # the pass didn't fire.  Cubins are ELF; their text section grows
    # roughly linearly with instruction count.
    assert len(cubin) < 8000, (
        f"cubin size {len(cubin)} suggests the rem.u64 bit-serial loop "
        f"survived — fast-path pass may have failed to fire")


def test_pipeline_disable_via_env(monkeypatch):
    """OPENPTXAS_DISABLE_PASSES=m31_mod_fast_path leaves rem.u64 in place."""
    from sass.pipeline import compile_ptx_source

    ptx = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry k(.param .u64 inp, .param .u64 outp)
{
    .reg .b64 %rd<8>;
    .reg .u32 %r<4>;
    mov.u32 %r0, %tid.x;
    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 3;
    ld.param.u64 %rd1, [inp];  add.u64 %rd2, %rd1, %rd0;
    ld.global.u64 %rd3, [%rd2];
    rem.u64 %rd4, %rd3, 2147483647;
    ld.param.u64 %rd5, [outp]; add.u64 %rd6, %rd5, %rd0;
    st.global.u64 [%rd6], %rd4;
    ret;
}
"""
    monkeypatch.setenv("OPENPTXAS_DISABLE_PASSES", "m31_mod_fast_path")
    cubin_disabled = compile_ptx_source(ptx)["k"]
    monkeypatch.delenv("OPENPTXAS_DISABLE_PASSES", raising=False)
    cubin_enabled = compile_ptx_source(ptx)["k"]
    # With the bit-serial loop in play, the disabled cubin must be
    # noticeably larger than the enabled one.
    assert len(cubin_disabled) > len(cubin_enabled), (
        f"disabled cubin ({len(cubin_disabled)}) should be larger than "
        f"enabled cubin ({len(cubin_enabled)}) — toggle has no effect")


# ---------------------------------------------------------------------------
# Part D: GPU correctness — bit-identical with Python oracle
# ---------------------------------------------------------------------------

try:
    import ctypes as _ctypes
    _c = _ctypes.cdll.LoadLibrary("nvcuda.dll")
    _CUDA_AVAILABLE = _c.cuInit(0) == 0
except Exception:
    _CUDA_AVAILABLE = False

gpu = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="No CUDA GPU")


_M31 = (1 << 31) - 1


_PTX_M31_MOD = """\
.version 9.0
.target sm_120
.address_size 64
.visible .entry m31_mod_kernel(
    .param .u64 inp, .param .u64 outp, .param .u32 n)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<16>;
    .reg .pred %p<2>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r3;
    shl.b64    %rd0, %rd0, 3;
    ld.param.u64 %rd1, [inp];  add.u64 %rd2, %rd1, %rd0;
    ld.global.u64 %rd3, [%rd2];

    rem.u64 %rd4, %rd3, 2147483647;

    ld.param.u64 %rd5, [outp]; add.u64 %rd6, %rd5, %rd0;
    st.global.u64 [%rd6], %rd4;
DONE:
    ret;
}
"""


@gpu
class TestM31ModGpu:
    def test_m31_mod_matches_python_oracle(self, cuda_ctx):
        """End-to-end GPU run: out[i] = in[i] % M31, bit-identical with
        Python oracle for a panel of inputs that exercises the bound
        math (zero, M31 itself, M31+1, 2*M31, large 64-bit values)."""
        from sass.pipeline import compile_ptx_source

        cubins = compile_ptx_source(_PTX_M31_MOD)
        assert "m31_mod_kernel" in cubins
        assert cuda_ctx.load(cubins["m31_mod_kernel"])
        func = cuda_ctx.get_func("m31_mod_kernel")

        inputs = [
            0,
            1,
            _M31 - 1,
            _M31,
            _M31 + 1,
            2 * _M31,
            2 * _M31 + 1,
            (1 << 31),
            (1 << 32),
            (1 << 33) - 1,
            (1 << 33),
            (1 << 33) + _M31,
            (1 << 60),
            (1 << 62),
            (1 << 63),
            (1 << 64) - 1,
            12345678901234567,
            0xDEADBEEF_CAFEBABE,
            0xFEEDFACE_DEADBEEF,
            0x8000_0000_0000_0001,
        ]
        # Pad up to a round multiple for clean grid sizing.
        while len(inputs) < 32:
            inputs.append(((len(inputs) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF))
        N = len(inputs)
        expected = [x % _M31 for x in inputs]

        byte_size = N * 8
        d_in  = cuda_ctx.alloc(byte_size)
        d_out = cuda_ctx.alloc(byte_size)
        try:
            cuda_ctx.copy_to(d_in,  struct.pack(f"<{N}Q", *inputs))
            cuda_ctx.copy_to(d_out, b"\x00" * byte_size)
            err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1),
                                  [d_in, d_out, N])
            assert err == 0, f"cuLaunchKernel failed: {err}"
            assert cuda_ctx.sync() == 0, "kernel crashed at cuCtxSynchronize"

            raw = cuda_ctx.copy_from(d_out, byte_size)
            results = list(struct.unpack(f"<{N}Q", raw))
            for i in range(N):
                assert results[i] == expected[i], (
                    f"x={inputs[i]:#018x}: ours={results[i]}, "
                    f"oracle={expected[i]}")
        finally:
            cuda_ctx.free(d_in)
            cuda_ctx.free(d_out)

    def test_m31_mod_disabled_vs_enabled_match(self, cuda_ctx):
        """Cross-check: with the fast-path enabled vs disabled, both
        should produce identical bit-for-bit results across the full
        input panel.  This is the strongest correctness guarantee — it
        catches any divergence between the rewrite and the bit-serial
        divide reference."""
        from sass.pipeline import compile_ptx_source
        import os

        inputs = [
            0, 1, _M31 - 1, _M31, _M31 + 1, 2 * _M31,
            (1 << 31), (1 << 32), (1 << 33), (1 << 40), (1 << 60),
            (1 << 63), (1 << 64) - 1,
            0xDEADBEEF_CAFEBABE, 0xFEEDFACE_DEADBEEF,
            12345678901234567,
        ]
        while len(inputs) < 32:
            inputs.append((len(inputs) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
        N = len(inputs)

        prev = os.environ.get("OPENPTXAS_DISABLE_PASSES")
        try:
            os.environ["OPENPTXAS_DISABLE_PASSES"] = "m31_mod_fast_path"
            cubins_off = compile_ptx_source(_PTX_M31_MOD)
            os.environ.pop("OPENPTXAS_DISABLE_PASSES", None)
            cubins_on = compile_ptx_source(_PTX_M31_MOD)
        finally:
            if prev is not None:
                os.environ["OPENPTXAS_DISABLE_PASSES"] = prev
            else:
                os.environ.pop("OPENPTXAS_DISABLE_PASSES", None)

        def run(cubin):
            assert cuda_ctx.load(cubin)
            func = cuda_ctx.get_func("m31_mod_kernel")
            byte_size = N * 8
            d_in  = cuda_ctx.alloc(byte_size)
            d_out = cuda_ctx.alloc(byte_size)
            try:
                cuda_ctx.copy_to(d_in,  struct.pack(f"<{N}Q", *inputs))
                cuda_ctx.copy_to(d_out, b"\x00" * byte_size)
                err = cuda_ctx.launch(func, (1, 1, 1), (N, 1, 1),
                                      [d_in, d_out, N])
                assert err == 0
                assert cuda_ctx.sync() == 0
                return list(struct.unpack(f"<{N}Q",
                                          cuda_ctx.copy_from(d_out, byte_size)))
            finally:
                cuda_ctx.free(d_in)
                cuda_ctx.free(d_out)

        results_off = run(cubins_off["m31_mod_kernel"])
        results_on  = run(cubins_on["m31_mod_kernel"])
        assert results_off == results_on, (
            f"fast-path divergence at indices: "
            f"{[i for i in range(N) if results_off[i] != results_on[i]]}")
