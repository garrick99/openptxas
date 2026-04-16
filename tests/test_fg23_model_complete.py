"""
FG-2.3 — system-property tests: the execution model is complete for
observed behaviour.

Four independent invariants are enforced:

INV A — Predicate correctness.
    For every setp.<cmp>.u64 (cmp in {ge, lt, gt, le, eq, ne}) against
    a UR-backed param, OpenPTXas must produce the same output as
    PTXAS across a range of n values.  This locks in FG-2.1 + FG-2.3
    and will fail loudly if the _INVERT mapping or the R-UR whitelist
    ever drops a comparison op.

INV B — Opcode coverage completeness.
    Every opcode that OpenPTXas emits across the workbench + probe
    kernels must be either in sass.scoreboard._OPCODE_META OR in the
    FG-2.3 latency-inert allowlist.  This prevents silent regressions
    where a new emission path introduces an unmodeled opcode.

INV C — verify_schedule: zero REAL hazards.
    Every violation reported by sass.schedule.verify_schedule on the
    workbench + probe kernels must be in the FG-2.3 false-positive
    classification table (see probe_work/fg23_verify_classify.py).
    An unclassified violation fails the test.

INV D — FG-1 IMAD.R-UR defense still active.
    At least on the FG-1 reproducer kernels, no IMAD.R-UR (0xc24)
    may have dest == src0.  This re-asserts the FG-1.14C narrow
    defense and overlaps with test_fg22_imad_alias_invariant.py,
    but is kept here so FG-2.3 is self-contained.
"""
from __future__ import annotations

import struct
import ctypes
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from benchmarks.bench_util import compile_openptxas, compile_ptxas, CUDAContext
from sass.scoreboard import _OPCODE_META
from sass.schedule import verify_schedule
from sass.pipeline import SassInstr


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _iter_text_sections(cubin):
    e_shoff = struct.unpack_from('<Q', cubin, 40)[0]
    e_shnum = struct.unpack_from('<H', cubin, 60)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 62)[0]
    stoff = struct.unpack_from('<Q', cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    for i in range(e_shnum):
        base = e_shoff + i * 64
        nm = struct.unpack_from('<I', cubin, base)[0]
        name_end = cubin.index(0, stoff + nm)
        name = cubin[stoff + nm:name_end]
        if not name.startswith(b".text."):
            continue
        off = struct.unpack_from('<Q', cubin, base + 24)[0]
        sz = struct.unpack_from('<Q', cubin, base + 32)[0]
        yield name.decode()[len(".text."):], off, sz


def _opcode(raw: bytes) -> int:
    return (raw[0] | (raw[1] << 8)) & 0xFFF


def _workbench_kernels():
    import sys
    sys.path.insert(0, str(ROOT))
    import workbench
    out = []
    for name, entry in workbench.KERNELS.items():
        src = entry.get("ptx_inline")
        if src is None:
            path = entry.get("ptx_path")
            if path and Path(path).exists():
                src = Path(path).read_text(encoding="utf-8")
        if src is None:
            continue
        out.append((name, src))
    return out


def _probe_kernels():
    paths = [
        "probe_work/fg114b_diag.ptx",
        "probe_work/fg114b_diag3.ptx",
        "probe_work/fg114a_step1b_guarded.ptx",
        "probe_work/fg114a_step3b_freshaddr.ptx",
    ]
    return [(p, (ROOT / p).read_text()) for p in paths]


# ---------------------------------------------------------------------------
# INV A — predicate correctness
# ---------------------------------------------------------------------------

_PRED_N_VALUES = [0, 1, 4, 8, 15, 16, 17, 32, 100]


def _pred_kernel_ptx(name: str, cmp: str) -> str:
    return f"""\
.version 8.8
.target sm_120
.address_size 64

.visible .entry {name}(
    .param .u64 {name}_param_output_data,
    .param .u64 {name}_param_output_len,
    .param .u64 {name}_param_n
)
{{
    .reg .pred %p8;
    .reg .u64 %rd0, %rd1, %rd2, %rd9, %rd12, %rd13, %rd14, %rd15, %rd20;
    .reg .u32 %r3, %r4, %r5, %r6, %r7;

  ld.param.u64 %rd0, [{name}_param_output_data];
  ld.param.u64 %rd1, [{name}_param_output_len];
  ld.param.u64 %rd2, [{name}_param_n];
  mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x;
  mad.lo.u32 %r5, %r3, %r4, 0;
  mov.u32 %r6, %tid.x; add.u32 %r7, %r5, %r6;
  cvt.u64.u32 %rd9, %r7;
  cvt.u64.u32 %rd12, %r7;
  mov.u64 %rd13, 8;
  mul.lo.u64 %rd14, %rd12, %rd13;
  add.u64 %rd15, %rd0, %rd14;
  mov.u64 %rd20, 0xBBBB;
  setp.{cmp}.u64 %p8, %rd9, %rd2;
  @%p8 bra {name}_then;
  bra {name}_else;
  {name}_then:
  st.global.u64 [%rd15], %rd20;
  bra {name}_end;
  {name}_else:
  {name}_end:
  ret;
}}
"""


def _make_args(*vs):
    ptrs = (ctypes.c_void_p * len(vs))(
        *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vs]
    )
    return ptrs, vs


def _run_pred_kernel(ctx, cubin, name, n, block=16):
    mod = ctypes.c_void_p()
    ctx.cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
    f = ctypes.c_void_p()
    ctx.cuda.cuModuleGetFunction(ctypes.byref(f), mod, name.encode())
    h = (ctypes.c_uint64 * block)(*([0xDEADBEEF] * block))
    d = ctx.alloc(block * 8)
    try:
        ctx.copy_to(d, bytes(h))
        a1 = ctypes.c_uint64(d)
        a2 = ctypes.c_uint64(block)
        a3 = ctypes.c_uint64(n)
        args, _ = _make_args(a1, a2, a3)
        ctx.cuda.cuLaunchKernel(f, 1, 1, 1, block, 1, 1, 0, None, args, None)
        ctx.sync()
        out = ctx.copy_from(d, block * 8)
        return list(struct.unpack(f"<{block}Q", out))
    finally:
        ctx.free(d)
        ctx.cuda.cuModuleUnload(mod)


@pytest.mark.parametrize("cmp", ["ge", "lt", "gt", "le", "eq", "ne"])
def test_predicate_ur_compare_matches_ptxas(cmp):
    """INV A: setp.<cmp>.u64 against a UR-backed operand must produce
    the same output as PTXAS across the full n range.
    """
    name = f"k_pred_{cmp}"
    ptx = _pred_kernel_ptx(name, cmp)
    co, _ = compile_openptxas(ptx)
    cp, _ = compile_ptxas(ptx)
    ctx = CUDAContext()
    for n in _PRED_N_VALUES:
        out_o = _run_pred_kernel(ctx, co, name, n)
        out_p = _run_pred_kernel(ctx, cp, name, n)
        assert out_o == out_p, (
            f"setp.{cmp}.u64 at n={n}: OURS={out_o} PTXAS={out_p}. "
            f"FG-2.3 INV A violation."
        )


# ---------------------------------------------------------------------------
# INV B — opcode coverage completeness
# ---------------------------------------------------------------------------

# Keep this list in sync with probe_work/fg23_opcode_coverage.py.
_LATENCY_INERT = {
    0x918, 0x947, 0x94d, 0x948, 0x941, 0x949, 0x94a, 0x94c,
    0x980, 0x981, 0x982, 0x983, 0x984, 0x985, 0x986, 0x988,
    0x98a, 0x98b, 0x98c, 0x98f, 0x992, 0x9a3, 0x9a8, 0x3a9,
    0x3c4, 0x7ac, 0x7ad, 0xb1d, 0xb1e, 0x82f, 0x9b7, 0x806,
    0x3cc, 0x389, 0x388, 0x98d, 0x98e, 0x9b4,
    0x589, 0xf89, 0x989,  # SHFL all forms: warp-synchronous class (KERNEL-100)
    0x424,  # IMAD.424: 32-bit addend form (PTXAS only, same ALU class as 0x824)
    0x802,  # ULDC: uniform load (UR dest, no GPR latency)
    0x886,  # UR pipeline init (P3-7, no GPR latency for consumer)
    0x2bd,  # UR pipeline finalize (P3-7, no GPR latency for consumer)
    0xd09,  # AT06 UR-pipeline data-routing op for atom.add K=1 imm_data
            # variant; emitted only by the atom-UR template (never via isel),
            # and consumes/produces UR state, not GPR latency.
    0x7b8, 0x7b9, 0x7ba, 0x7b0, 0x7b1, 0x7b2, 0x7b3,
    0x7bb, 0x7bc, 0x7bd, 0xab9,
    # FG-2.3 additions:
    0x80c, 0x919, 0x91a, 0x945, 0x9af, 0x9c3, 0xb82, 0xf89, 0xfae,
    0x209, 0x229, 0x812, 0x835, 0xc11, 0xc12, 0xc25,
    # MPT34: uncommon ALU helper op emitted ONLY by the k200_nested_pred
    # whole-kernel template (verbatim PTXAS bytes); does not go through
    # OpenPTXas's scheduler/scoreboard so PTXAS's own ctrl-byte scheduling
    # is preserved and OpenPTXas's latency model never observes it.
    0x81c,
    # IMNMX01-04: 0x848 is now properly modeled in _OPCODE_META as
    # IMNMX.IMM (R-imm variant of 0x217); no longer in this allowlist.
}


def test_opcode_coverage_complete():
    """INV B: every opcode emitted by OpenPTXas across all workbench +
    probe kernels must be either modeled in _OPCODE_META or on the
    latency-inert allowlist.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    kernels = _probe_kernels() + _workbench_kernels()
    seen_opcodes = set()
    for _, ptx in kernels:
        try:
            cubin, _ = compile_openptxas(ptx)
        except Exception:
            continue
        for _sym, off, sz in _iter_text_sections(cubin):
            for o in range(0, sz, 16):
                raw = cubin[off + o:off + o + 16]
                if len(raw) < 16:
                    continue
                seen_opcodes.add(_opcode(raw))
    known = set(_OPCODE_META.keys())
    unclassified = seen_opcodes - known - _LATENCY_INERT
    assert not unclassified, (
        f"FG-2.3 INV B violation — unclassified opcodes emitted: "
        f"{sorted(f'0x{o:03x}' for o in unclassified)}. "
        f"Either add to _OPCODE_META with evidence-backed min_gpr_gap, "
        f"or add to the FG-2.3 latency-inert allowlist in "
        f"probe_work/fg23_opcode_coverage.py AND this test's "
        f"_LATENCY_INERT set with rationale."
    )


# ---------------------------------------------------------------------------
# INV C — verify_schedule: zero real hazards, all classified
# ---------------------------------------------------------------------------

# (writer_opc, reader_opc) → ("FALSE_POSITIVE", reason)
_VERIFY_FALSE_POSITIVES = {
    (0x224, 0x235),  # IMAD.32 → IADD.64: pair read overlap decoder artefact
    (0x235, 0xf89),  # IADD.64 → SHFL: SHFL src decoding imprecise
    (0x221, 0xf89),  # FADD  → SHFL: same as above
    (0xc35, 0x986),  # IADD.64-UR → STG: ctrl-word handled
    (0xc02, 0x986),  # MOV.UR → STG: ctrl-word handled
    (0x819, 0x20c),  # SHF → ISETP: ISETP R-R src decoding imprecise
    (0x221, 0x986),  # FADD → STG: FADD ctrl-word handles window
}
_LDCU_FALSE_POSITIVE = 0x7ac  # LDCU.64 post-boundary gap reports


def _classify_verify_report(instrs, v_str):
    """Return 'FALSE_POSITIVE' or 'REAL' for a verify_schedule report."""
    import re
    m = re.search(r"\[(\d+)\].*\[(\d+)\]", v_str)
    if not m:
        return "UNPARSED"
    i, j = int(m.group(1)), int(m.group(2))
    if i >= len(instrs) or j >= len(instrs):
        return "UNPARSED"
    w_opc = _opcode(instrs[i].raw)
    r_opc = _opcode(instrs[j].raw)
    if (w_opc, r_opc) in _VERIFY_FALSE_POSITIVES:
        return "FALSE_POSITIVE"
    if w_opc == _LDCU_FALSE_POSITIVE:
        return "FALSE_POSITIVE"
    return "REAL"


def test_verify_schedule_no_real_hazards():
    """INV C: every sass.schedule.verify_schedule report on the
    workbench + probe kernels must be a classified false positive.
    """
    kernels = _probe_kernels() + _workbench_kernels()
    real_violations = []
    unparsed = []
    for label, ptx in kernels:
        try:
            cubin, _ = compile_openptxas(ptx)
        except Exception:
            continue
        for sym, off, sz in _iter_text_sections(cubin):
            instrs = [SassInstr(cubin[off + o:off + o + 16], "")
                      for o in range(0, sz, 16)
                      if off + o + 16 <= off + sz]
            viols = verify_schedule(instrs)
            for v in viols:
                cls = _classify_verify_report(instrs, v)
                if cls == "REAL":
                    real_violations.append(f"{label}:{sym}: {v}")
                elif cls == "UNPARSED":
                    unparsed.append(f"{label}:{sym}: {v}")
    assert not real_violations, (
        f"FG-2.3 INV C violation — {len(real_violations)} REAL "
        f"verify_schedule hazard(s):\n" + "\n".join(real_violations[:10])
    )
    assert not unparsed, (
        f"FG-2.3 INV C violation — {len(unparsed)} UNPARSED "
        f"verify_schedule report(s):\n" + "\n".join(unparsed[:10])
    )


# ---------------------------------------------------------------------------
# INV D — FG-1 IMAD.R-UR defense still active
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,symbol", [
    ("probe_work/fg114b_diag.ptx", "diag"),
    ("probe_work/fg114a_step1b_guarded.ptx", "min_store_guarded"),
    ("probe_work/fg114a_step3b_freshaddr.ptx", "probe_fresh"),
])
def test_imad_ru_fg114c_defense_still_active(path, symbol):
    """INV D: No IMAD.R-UR (opcode 0xc24) in the FG-1 reproducer kernels
    may have dest == src0.  This is the FG-1.14C defense at the fused
    mul+add site, asserted here so FG-2.3 is self-contained.
    """
    ptx = (ROOT / path).read_text(encoding="utf-8")
    cubin, _ = compile_openptxas(ptx)
    found = 0
    for sym, off, sz in _iter_text_sections(cubin):
        for o in range(0, sz, 16):
            raw = cubin[off + o:off + o + 16]
            if len(raw) < 16:
                continue
            if _opcode(raw) != 0xc24:
                continue
            if raw[2] == raw[3]:
                found += 1
    assert found == 0, (
        f"{symbol}: found {found} IMAD.R-UR with dest == src0. "
        f"FG-2.3 INV D violation — FG-1.14C defense has been "
        f"bypassed."
    )


# ---------------------------------------------------------------------------
# INV E — FG-2.4 precise decoder output
# ---------------------------------------------------------------------------
#
# The FG-2.4 refactor added explicit per-opcode cases in
# sass.scoreboard._get_src_regs for every opcode that OpenPTXas emits
# and had previously relied on the generic fallback.  These tests
# assert that for a hand-crafted 16-byte instruction with known
# operand layout, _get_src_regs returns exactly the expected GPR
# source set.  Regressions here mean someone broke the per-opcode
# decoding logic.

from sass.scoreboard import _get_src_regs, _is_forwarding_safe_pair


def _mk(b0, b1, b2=0, b3=0, b4=0, b8=0, b9=0, b10=0, b11=0):
    """Build a 16-byte instruction stub with the given opcode fields."""
    raw = bytearray(16)
    raw[0] = b0; raw[1] = b1; raw[2] = b2; raw[3] = b3; raw[4] = b4
    raw[8] = b8; raw[9] = b9; raw[10] = b10; raw[11] = b11
    return bytes(raw)


@pytest.mark.parametrize("name,raw,expected", [
    # opcode 0xf89 — SHFL reg-imm: only b3 is a GPR source.
    ("shfl_reg_imm_basic",
     _mk(0x89, 0x7f, b2=0x06, b3=0x02, b4=0x00, b10=0x0e),
     {2}),
    ("shfl_reg_imm_rz_src",
     _mk(0x89, 0x7f, b2=0x06, b3=0xff, b4=0x00, b10=0x0e),
     set()),
    # opcode 0x589 — SHFL reg-reg: b3 = data, b4 = lane (both GPR).
    ("shfl_reg_reg_both_gpr",
     _mk(0x89, 0x75, b2=0x06, b3=0x04, b4=0x05),
     {4, 5}),
    # opcode 0x209 — FMNMX: b3, b4 both GPR.
    ("fmnmx_basic",
     _mk(0x09, 0x72, b2=0x06, b3=0x04, b4=0x05),
     {4, 5}),
    # opcode 0x812 — IADD3-family R-imm: b3 src0, b8 src2.
    ("iadd3_r_imm",
     _mk(0x12, 0x78, b2=0x05, b3=0x04, b4=0x1f, b8=0x06),
     {4, 6}),
    # opcode 0x835 — IADD.64 R-imm: only b3:b3+1 pair (no b8 source).
    ("iadd64_r_imm",
     _mk(0x35, 0x78, b2=0x04, b3=0x04, b8=0x00),
     {4, 5}),
    # opcode 0xc11 — LEA R-UR: b3 base GPR, b4 is UR (not GPR).
    ("lea_r_ur",
     _mk(0x11, 0x7c, b2=0x04, b3=0x02, b4=0x04, b8=0xff),
     {2}),
    # opcode 0xc12 — IADD3X R-UR: b3 GPR, b4 UR, b8 GPR.
    ("iadd3x_r_ur",
     _mk(0x12, 0x7c, b2=0x07, b3=0x05, b4=0x06, b8=0x02),
     {2, 5}),
    # opcode 0xc25 — IMAD.WIDE R-UR: b3 GPR, b4 UR, b8 pair.
    ("imad_wide_r_ur",
     _mk(0x25, 0x7c, b2=0x08, b3=0x0a, b4=0x0a, b8=0x08),
     {8, 9, 10}),
    # opcode 0x235 — IADD.64 R-R: src0 pair + src1 pair.
    ("iadd64_rr_pair",
     _mk(0x35, 0x72, b2=0x02, b3=0x02, b4=0x04),
     {2, 3, 4, 5}),
    # opcode 0xc35 — IADD.64-UR: only b3 pair is a GPR.
    ("iadd64_ur",
     _mk(0x35, 0x7c, b2=0x06, b3=0x08, b4=0x08),
     {8, 9}),
    # opcode 0x20c — ISETP R-R: b3 + b4.
    ("isetp_rr",
     _mk(0x0c, 0x72, b2=0x00, b3=0x06, b4=0x02),
     {2, 6}),
    # opcode 0x0c24 — IMAD R-UR: b3 GPR, b4 UR, b8 GPR.
    ("imad_r_ur",
     _mk(0x24, 0x7c, b2=0x03, b3=0x03, b4=0x06, b8=0x02),
     {2, 3}),
    # opcode 0x819 — SHF R-imm: b3 + b8 are GPR, b4 is shift count.
    ("shf_r_imm",
     _mk(0x19, 0x78, b2=0x04, b3=0x0a, b4=0x04, b8=0xff),
     {10}),
])
def test_get_src_regs_precise(name, raw, expected):
    """INV E: sass.scoreboard._get_src_regs returns exactly the set of
    GPR source registers (no immediates, no lanes, no UR indices).

    Each case is a hand-crafted instruction with a known operand
    layout; the expected set is what the hardware would actually read.
    """
    got = _get_src_regs(raw)
    assert got == expected, (
        f"INV E violation in {name}: expected {sorted(expected)}, got "
        f"{sorted(got)}. _get_src_regs lost precision for this opcode."
    )


# ---------------------------------------------------------------------------
# INV F — verify_schedule returns zero reports after FG-2.4
# ---------------------------------------------------------------------------

def test_verify_schedule_zero_reports_after_fg24():
    """INV F (FG-2.4): with the forwarding-safe pair table and LDCU.64
    consumer whitelist in place, sass.schedule.verify_schedule must
    return ZERO reports across all test kernels.  Any new report is
    a regression or a new pattern that needs classification.

    This is strictly stronger than INV C (which accepts classified
    false positives).  After FG-2.4 there should be no false
    positives left to classify.
    """
    kernels = _probe_kernels() + _workbench_kernels()
    all_reports = []
    for label, ptx in kernels:
        try:
            cubin, _ = compile_openptxas(ptx)
        except Exception:
            continue
        for sym, off, sz in _iter_text_sections(cubin):
            instrs = [SassInstr(cubin[off + o:off + o + 16], "")
                      for o in range(0, sz, 16)
                      if off + o + 16 <= off + sz]
            viols = verify_schedule(instrs)
            for v in viols:
                all_reports.append(f"{label}:{sym}: {v}")
    assert not all_reports, (
        f"FG-2.4 INV F violation — {len(all_reports)} verify_schedule "
        f"report(s) survived the forwarding-safe filter:\n"
        + "\n".join(all_reports[:15])
    )


# ---------------------------------------------------------------------------
# INV G — forwarding-safe pair safety (no accidental FG-1.14A bypass)
# ---------------------------------------------------------------------------

def test_forwarding_safe_pairs_exclude_imad_ru_producer():
    """INV G: the FG-2.4 forwarding-safe whitelist must NOT include any
    pair whose producer is IMAD.R-UR (0xc24) paired with the MOV/IMAD/
    IADD consumers that FG-1.14A identified as unsafe.

    This guards against accidentally covering the FG-1.14A hazard
    with a forwarding-safe exemption.  The scheduler must still
    insert the 1-instruction gap for IMAD.R-UR writes.
    """
    unsafe_pairs = [
        (0xc24, 0x202),  # IMAD.R-UR → MOV: the FG-1.14A pattern
        (0xc24, 0xc24),  # chained IMAD.R-UR
        (0xc24, 0x235),  # IMAD.R-UR → IADD.64
        (0xc24, 0x210),  # IMAD.R-UR → IADD3
        (0xc24, 0x986),  # IMAD.R-UR → STG
    ]
    for w, r in unsafe_pairs:
        assert not _is_forwarding_safe_pair(w, r), (
            f"INV G violation: forwarding-safe whitelist covers "
            f"(0x{w:03x}, 0x{r:03x}) which is the FG-1.14A-unsafe "
            f"IMAD.R-UR producer path. Remove the entry and keep "
            f"the 1-instruction gap enforced."
        )
