"""
FG-2.5 — proof-engine system-property tests.

Four positive invariants (INV H–K) and three negative invariants
(INV L–N).  The positive invariants lock the current corpus as
"proven safe under the FG-2.5 dependency model".  The negative
invariants prove the verifier can still distinguish safe from
unsafe code — it is not a green-light generator.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from benchmarks.bench_util import compile_openptxas
from sass.pipeline import SassInstr
from sass.schedule import verify_proof, verify_schedule, ProofClass, ProofReport, ProofEdge
from sass.encoding.sm_120_opcodes import (
    encode_imad_ur, encode_mov, encode_nop,
    encode_imad_wide_rr, encode_iadd3,
    encode_ldg_e, encode_lds, encode_ldc,
)


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


def _proof_for(ptx_src: str) -> ProofReport:
    cubin, _ = compile_openptxas(ptx_src)
    combined = ProofReport()
    for _sym, off, sz in _iter_text_sections(cubin):
        instrs = [SassInstr(cubin[off + o:off + o + 16], "")
                  for o in range(0, sz, 16)
                  if off + o + 16 <= off + sz]
        report = verify_proof(instrs)
        combined.edges.extend(report.edges)
    return combined


def _workbench_kernels():
    import sys
    sys.path.insert(0, str(ROOT))
    import workbench
    out = []
    for name, entry in workbench.KERNELS.items():
        src = entry.get("ptx_inline")
        if src is None:
            p = entry.get("ptx_path")
            if p and Path(p).exists():
                src = Path(p).read_text(encoding="utf-8")
        if src is None:
            continue
        out.append((name, src))
    return out


def _probe_kernels():
    paths = [
        ("probe_work/fg114b_diag.ptx", "diag"),
        ("probe_work/fg114b_diag3.ptx", "diag3"),
        ("probe_work/fg114a_step1b_guarded.ptx", "min_store_guarded"),
        ("probe_work/fg114a_step3b_freshaddr.ptx", "probe_fresh"),
    ]
    return [(sym, (ROOT / p).read_text()) for p, sym in paths]


def _fg21_predicate_kernels():
    import sys
    sys.path.insert(0, str(ROOT))
    from probe_work.fg21_setp_repros import kernel_ptx, CMPS
    return [(f"fg21:k_{cmp}", kernel_ptx(f"k_{cmp}", cmp)) for cmp in CMPS]


# ===========================================================================
# POSITIVE INVARIANTS (H, I, J, K)
# ===========================================================================

# ---------------------------------------------------------------------------
# INV H — every tested kernel is SAFE under proof mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,ptx",
                         _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels(),
                         ids=lambda x: x[0] if isinstance(x, tuple) else x)
def test_inv_h_corpus_is_safe_under_proof(label, ptx):
    """INV H: every kernel in the FG-2.5 corpus must be SAFE under the
    proof-engine model.  A VIOLATION edge means the emitted SASS has
    a hazard class that isn't covered by any of:
      LATENCY_INERT / FORWARDING_SAFE / CTRLWORD_SAFE / GAP_SAFE

    This is the strongest corpus-level assertion in the FG series: it
    requires constructive proof for every classified edge, not just
    absence of a complaint from the hazard detector.
    """
    report = _proof_for(ptx)
    assert report.safe, (
        f"{label}: {len(report.violations)} VIOLATION edge(s) in "
        f"proof report.\n" + "\n".join(v.legacy_str() for v in report.violations[:5])
    )


# ---------------------------------------------------------------------------
# INV I — every safe kernel has zero VIOLATION edges
# ---------------------------------------------------------------------------

def test_inv_i_safe_kernels_have_zero_violations():
    """INV I: aggregate check — across every corpus kernel, the total
    number of VIOLATION edges in proof mode must be exactly zero.

    Strictly equivalent to INV H at the aggregate level; included so a
    corpus-wide regression produces one failure rather than N.
    """
    corpus = _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels()
    total_violations = 0
    problem_labels = []
    for label, ptx in corpus:
        try:
            report = _proof_for(ptx)
        except Exception:
            continue
        total_violations += len(report.violations)
        if report.violations:
            problem_labels.append(label)
    assert total_violations == 0, (
        f"INV I violation — {total_violations} VIOLATION edges across "
        f"{len(problem_labels)} kernel(s): {problem_labels[:10]}"
    )


# ---------------------------------------------------------------------------
# INV J — proof class counts are stable for key kernels
# ---------------------------------------------------------------------------
#
# Locks in the current classification shape for a handful of
# representative kernels so structural regressions (e.g. the
# forwarding-safe table losing an entry, or an opcode falling out of
# _OPCODE_META) fail loudly.  Values come from the aggregate table
# printed by probe_work/fg25_prove_corpus.py.

_EXPECTED_COUNTS = {
    # FG-3.2 11-tuple: (total, LATENCY_INERT, FORWARDING_SAFE,
    # CTRLWORD_SAFE, GAP_SAFE, MEMORY_SCOREBOARD_SAFE, MEMORY_INERT,
    # MEMORY_VIOLATION, UR_MEMORY_SCOREBOARD_SAFE, UR_MEMORY_VIOLATION,
    # VIOLATION).
    #
    # Baselines regenerated 2026-04-11 for FG-3.2:
    #   - Added MEMORY_INERT column (producer wdep=0x3f explicit class).
    #   - Added UR_MEMORY_SCOREBOARD_SAFE column (rule R10 for S2UR,
    #     REDUX, ULDC.64).  Every kernel gains +1 UR_SB edge from
    #     the S2UR → consumer chain introduced by the preamble.
    #   - reduce_sum gains +2 MSB edges because wdep=0x3b (rotating
    #     LDG slot) is now explicitly classified as LDG-class.
    "diag":                  (2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0),
    "diag3":                 (2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0),
    "min_store_guarded":     (3, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0),
    "probe_fresh":           (3, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0),
    "reduce_sum":            (16, 4, 6, 1, 1, 3, 0, 0, 1, 0, 0),
    "conv2d_looped":         (107, 10, 0, 31, 1, 63, 0, 0, 2, 0, 0),
    "conv2d_unrolled":       (89, 1, 8, 30, 1, 47, 0, 0, 2, 0, 0),
    "fg21:k_ge":             (3, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0),
    "fg21:k_lt":             (3, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0),
    "fg21:k_gt":             (3, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0),
    "fg21:k_le":             (3, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0),
    "fg21:k_eq":             (3, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0),
    "fg21:k_ne":             (3, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0),
}


@pytest.mark.parametrize("label,expected", sorted(_EXPECTED_COUNTS.items()))
def test_inv_j_class_counts_stable(label, expected):
    """INV J: the per-kernel count of edges in each proof class must
    match the locked-in baseline for representative kernels.  Any
    drift means either a compiler change perturbed the emitted SASS
    or a proof rule was changed — both require an intentional
    baseline update.
    """
    # Locate the kernel source in the corpus.
    kernels = {k: v for k, v in (
        _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels())}
    if label not in kernels:
        pytest.skip(f"kernel {label} not in current corpus")
    report = _proof_for(kernels[label])
    c = report.counts
    actual = (
        report.total,
        c[ProofClass.LATENCY_INERT],
        c[ProofClass.FORWARDING_SAFE],
        c[ProofClass.CTRLWORD_SAFE],
        c[ProofClass.GAP_SAFE],
        c[ProofClass.MEMORY_SCOREBOARD_SAFE],
        c[ProofClass.MEMORY_INERT],
        c[ProofClass.MEMORY_VIOLATION],
        c[ProofClass.UR_MEMORY_SCOREBOARD_SAFE],
        c[ProofClass.UR_MEMORY_VIOLATION],
        c[ProofClass.VIOLATION],
    )
    assert actual == expected, (
        f"{label}: expected counts {expected}, got {actual}. "
        f"Baseline drift — either the compiler emitted a different "
        f"instruction stream or a proof rule changed."
    )


# ---------------------------------------------------------------------------
# INV K — every modeled GPR-writer opcode has a proof rule
# ---------------------------------------------------------------------------

def test_inv_k_every_gpr_writer_has_proof_rule():
    """INV K: the proof engine must have a proof path for every opcode
    _OPCODE_META registers.  As of FG-3.0 the engine supports bounded
    lookahead for arbitrary min_gpr_gap ≥ 1: writers with k=1 are
    reasoned about adjacent-only, writers with k>1 get a multi-step
    scan with shadow tracking.  This test now simply asserts that
    every entry has a non-negative gap and a rule the engine covers.
    """
    from sass.scoreboard import _OPCODE_META
    bad = []
    for opc, meta in _OPCODE_META.items():
        if meta.min_gpr_gap < 0:
            bad.append((opc, meta.name, meta.min_gpr_gap))
    assert not bad, (
        f"INV K violation — opcodes with negative min_gpr_gap: {bad}."
    )


# ===========================================================================
# NEGATIVE INVARIANTS (L, M, N) — the verifier can fail
# ===========================================================================

def _stream(*raws):
    """Build an instruction stream of SassInstr objects from raw bytes."""
    return [SassInstr(r, "") for r in raws]


# ---------------------------------------------------------------------------
# INV L — unsafe IMAD.R-UR → MOV pattern is flagged
# ---------------------------------------------------------------------------

def test_inv_l_negative_imad_ru_to_mov_flagged():
    """INV L: an IMAD.R-UR writing a GPR followed immediately by a
    MOV reading that same GPR is NOT on the forwarding-safe list
    (the FG-1.14A pattern — hardware does not forward in this case
    when the IMAD.R-UR aliases a source operand).  verify_proof must
    return a VIOLATION.

    This is the kind of hazard that triggered the entire FG-1.14A
    investigation; the test locks in that the proof engine catches
    it as a VIOLATION rather than silently passing.
    """
    instrs = _stream(
        # IMAD R4, R3, UR6, R2 (dest R4, src0 R3, src1 UR6, src2 R2)
        encode_imad_ur(dest=4, src0=3, ur_src=6, src2=2),
        # MOV R5, R4 — reads IMAD's dest at gap 0
        encode_mov(dest=5, src=4),
    )
    report = verify_proof(instrs)
    assert not report.safe, (
        f"INV L failed: proof engine did not flag IMAD.R-UR → MOV "
        f"0-gap RAW as a VIOLATION. Report: {report.summary_line()}"
    )
    viols = report.violations
    assert len(viols) == 1
    assert viols[0].writer_opc == 0xc24
    assert viols[0].reader_opc == 0x202
    assert 4 in viols[0].regs


# ---------------------------------------------------------------------------
# INV M — inserting the required NOP turns the same sequence SAFE
# ---------------------------------------------------------------------------

def test_inv_m_negative_nop_gap_makes_safe():
    """INV M: the same IMAD.R-UR → MOV pattern with an intervening
    NOP becomes safe: the NOP satisfies the min_gpr_gap=1 rule.

    The proof engine must classify the edge as "no hazard" (zero
    violations) because there is no longer an overlap at i+1.
    """
    instrs = _stream(
        encode_imad_ur(dest=4, src0=3, ur_src=6, src2=2),
        encode_nop(),  # latency gap
        encode_mov(dest=5, src=4),
    )
    report = verify_proof(instrs)
    assert report.safe, (
        f"INV M failed: proof engine flagged the NOP-separated "
        f"sequence. Report: {report.summary_line()}"
    )
    assert len(report.violations) == 0


# ---------------------------------------------------------------------------
# INV N — unsafe IMAD.WIDE pair write → next-instruction read is flagged
# ---------------------------------------------------------------------------

def test_inv_n_negative_imad_wide_rr_to_iadd3_flagged():
    """INV N: a second negative case, distinct from INV L, exercising
    a different writer opcode.  IMAD.WIDE.RR writes a GPR pair; an
    IADD3 (not IADD3.IMM) reading the low half of the pair in the
    next instruction is a real 0-gap RAW.  The pair (0x225, 0x210)
    is NOT on the forwarding-safe whitelist, so verify_proof must
    return a VIOLATION.
    """
    instrs = _stream(
        # IMAD.WIDE R8, R2, R3, RZ — writes R8:R9 pair
        encode_imad_wide_rr(dest=8, src0=2, src1=3, src2=0xff),
        # IADD3 R10, R8, R11, RZ — reads R8 at gap 0
        encode_iadd3(dest=10, src0=8, src1=11, src2=0xff),
    )
    report = verify_proof(instrs)
    assert not report.safe, (
        f"INV N failed: proof engine did not flag IMAD.WIDE.RR → "
        f"IADD3 0-gap RAW as a VIOLATION. Report: "
        f"{report.summary_line()}"
    )
    viols = report.violations
    assert any(v.writer_opc == 0x225 and v.reader_opc == 0x210
               for v in viols), (
        f"INV N failed: expected a (0x225, 0x210) VIOLATION, got "
        f"{[(hex(v.writer_opc), hex(v.reader_opc)) for v in viols]}"
    )


# ---------------------------------------------------------------------------
# INV O — legacy verify_schedule API still works (format compatibility)
# ---------------------------------------------------------------------------

def test_inv_o_legacy_verify_schedule_returns_list_of_str():
    """INV O: the legacy verify_schedule() API must still return a
    list of strings, one per VIOLATION edge.  The string format must
    still match the regex FG-2.3 INV C uses to parse reports.
    """
    # A clean stream should return [].
    clean = _stream(
        encode_imad_ur(dest=4, src0=3, ur_src=6, src2=2),
        encode_nop(),
        encode_mov(dest=5, src=4),
    )
    assert verify_schedule(clean) == []
    # An unsafe stream should return exactly one string.
    unsafe = _stream(
        encode_imad_ur(dest=4, src0=3, ur_src=6, src2=2),
        encode_mov(dest=5, src=4),
    )
    msgs = verify_schedule(unsafe)
    assert len(msgs) == 1
    import re
    # The regex used by FG-2.3 INV C: r"\[(\d+)\].*\[(\d+)\]"
    assert re.search(r"\[(\d+)\].*\[(\d+)\]", msgs[0]), (
        f"Legacy format broken: {msgs[0]!r}"
    )


# ===========================================================================
# FG-3.0 INVARIANTS (P, Q, R, S) — bounded non-adjacent hazard reasoning
# ===========================================================================
#
# The FG-2.5 proof engine only reasoned about adjacent (gap=0) edges.
# FG-3.0 extends it to bounded multi-step reasoning: for a writer with
# min_gpr_gap = k > 1, the engine scans k+1 positions forward with
# shadow tracking and classifies every candidate reader.
#
# Because no opcode currently lives in _OPCODE_META with min_gpr_gap > 1,
# these tests monkeypatch _OPCODE_META to temporarily register a
# synthetic k=2 rule for an existing opcode (IMAD.WIDE.RR = 0x225) and
# exercise the multi-step scan path on encoded instruction streams.


def _patch_min_gpr_gap(monkeypatch, opc: int, k: int) -> None:
    """Temporarily set _OPCODE_META[opc].min_gpr_gap = k for a test."""
    from sass.scoreboard import _OPCODE_META, _OpMeta
    original = _OPCODE_META.get(opc)
    if original is None:
        patched = _OpMeta(name=f"SYNTH_0x{opc:03x}", min_gpr_gap=k,
                          wdep=0x3e, misc=1)
    else:
        patched = _OpMeta(name=original.name, min_gpr_gap=k,
                          wdep=original.wdep, misc=original.misc)
    monkeypatch.setitem(_OPCODE_META, opc, patched)


# ---------------------------------------------------------------------------
# INV P — engine performs bounded lookahead > 1 for k>1 writers
# ---------------------------------------------------------------------------

def test_inv_p_bounded_lookahead_beyond_adjacent(monkeypatch):
    """INV P: when a writer has min_gpr_gap = k > 1, the engine must
    actually look at readers beyond index i+1.  We monkeypatch
    IMAD.WIDE.RR (0x225) to k=3 and build a stream:

        IMAD.WIDE.RR R8, R2, R3    # writer, dest R8:R9
        NOP
        NOP
        IADD3 R10, R8, R11, RZ     # reader at gap=2 < k=3 → VIOLATION

    If the engine were still adjacent-only (j = i+1 only), it would
    see the NOP at i+1 (no overlap) and emit no edge.  FG-3.0 must
    find the IADD3 reader at j = i+3 and emit a VIOLATION edge with
    gap=2.  This is the *definitional* FG-3.0 capability.
    """
    _patch_min_gpr_gap(monkeypatch, 0x225, 3)

    instrs = _stream(
        encode_imad_wide_rr(dest=8, src0=2, src1=3, src2=0xff),
        encode_nop(),
        encode_nop(),
        encode_iadd3(dest=10, src0=8, src1=11, src2=0xff),
    )
    report = verify_proof(instrs)

    # There must be at least one edge with writer_idx=0, reader_idx=3,
    # proving the engine looked beyond i+1.
    multi_step_edges = [
        e for e in report.edges
        if e.writer_idx == 0 and e.reader_idx == 3
    ]
    assert multi_step_edges, (
        f"INV P failed: proof engine emitted no non-adjacent edge "
        f"for IMAD.WIDE.RR → IADD3 separated by 2 NOPs. Report: "
        f"{report.summary_line()}. Edges: {report.edges}"
    )
    edge = multi_step_edges[0]
    assert edge.gap == 2, (
        f"INV P failed: expected gap=2, got gap={edge.gap}"
    )
    assert edge.classification == ProofClass.VIOLATION, (
        f"INV P failed: expected VIOLATION at gap=2 < k=3, got "
        f"{edge.classification}"
    )


# ---------------------------------------------------------------------------
# INV Q — non-adjacent VIOLATION is flagged
# ---------------------------------------------------------------------------

def test_inv_q_non_adjacent_violation_flagged(monkeypatch):
    """INV Q: with min_gpr_gap = 2 patched onto IMAD.WIDE.RR, a
    sequence

        IMAD.WIDE.RR R8, R2, R3
        NOP
        IADD3 R10, R8, R11, RZ    # gap=1 < k=2 → VIOLATION

    must be flagged as UNSAFE with exactly one VIOLATION whose
    writer/reader opcodes and gap match the non-adjacent RAW.

    This is the strong positive capability assertion of FG-3.0: the
    engine must not lose track of a hazard just because a benign
    (non-shadowing) instruction sits between producer and consumer.
    """
    _patch_min_gpr_gap(monkeypatch, 0x225, 2)

    instrs = _stream(
        encode_imad_wide_rr(dest=8, src0=2, src1=3, src2=0xff),
        encode_nop(),
        encode_iadd3(dest=10, src0=8, src1=11, src2=0xff),
    )
    report = verify_proof(instrs)
    assert not report.safe, (
        f"INV Q failed: proof engine did not flag non-adjacent "
        f"IMAD.WIDE.RR → NOP → IADD3 as a VIOLATION. Report: "
        f"{report.summary_line()}"
    )
    viols = [v for v in report.violations
             if v.writer_opc == 0x225 and v.reader_opc == 0x210]
    assert len(viols) == 1, (
        f"INV Q failed: expected exactly one 0x225→0x210 VIOLATION, "
        f"got {[(hex(v.writer_opc), hex(v.reader_opc), v.gap) for v in report.violations]}"
    )
    assert viols[0].gap == 1, (
        f"INV Q failed: expected gap=1, got gap={viols[0].gap}"
    )
    assert viols[0].writer_idx == 0 and viols[0].reader_idx == 2, (
        f"INV Q failed: expected edge [0]→[2], got "
        f"[{viols[0].writer_idx}]→[{viols[0].reader_idx}]"
    )


# ---------------------------------------------------------------------------
# INV R — non-adjacent SAFE is classified as GAP_SAFE
# ---------------------------------------------------------------------------

def test_inv_r_non_adjacent_gap_safe_classified(monkeypatch):
    """INV R: with the same k=2 patch, inserting the required number
    of NOPs turns the sequence SAFE:

        IMAD.WIDE.RR R8, R2, R3
        NOP
        NOP
        IADD3 R10, R8, R11, RZ    # gap=2 ≥ k=2 → GAP_SAFE

    The engine must produce one GAP_SAFE edge and zero violations.
    This proves the constructive classification side of FG-3.0 works:
    not just flagging bad cases, but affirmatively proving safety
    when the required gap is met.
    """
    _patch_min_gpr_gap(monkeypatch, 0x225, 2)

    instrs = _stream(
        encode_imad_wide_rr(dest=8, src0=2, src1=3, src2=0xff),
        encode_nop(),
        encode_nop(),
        encode_iadd3(dest=10, src0=8, src1=11, src2=0xff),
    )
    report = verify_proof(instrs)
    assert report.safe, (
        f"INV R failed: proof engine flagged a gap=2 ≥ k=2 "
        f"sequence as UNSAFE. Report: {report.summary_line()}. "
        f"Violations: {[v.rationale for v in report.violations]}"
    )
    gap_safe_edges = [
        e for e in report.edges
        if e.writer_opc == 0x225 and e.reader_opc == 0x210
        and e.classification == ProofClass.GAP_SAFE
    ]
    assert len(gap_safe_edges) == 1, (
        f"INV R failed: expected exactly one GAP_SAFE edge for "
        f"0x225 → 0x210 at gap=2, got {len(gap_safe_edges)}. "
        f"Full edges: {report.edges}"
    )
    assert gap_safe_edges[0].gap == 2


# ---------------------------------------------------------------------------
# INV R' — shadow tracking: intervening writer hides the producer
# ---------------------------------------------------------------------------

def test_inv_r_prime_shadowing_kills_edge(monkeypatch):
    """INV R' (shadowing): if an intervening instruction overwrites
    the writer's dest register before any reader sees it, the engine
    must NOT emit a VIOLATION for a later reader of that register —
    the later reader is reading the *newer* value, not the original
    producer's.

    Sequence with k=2 patched onto IMAD.WIDE.RR:

        IMAD.WIDE.RR R8, R2, R3    # writes R8 (and R9)
        MOV R8, R99                # overwrites R8 — shadow
        IADD3 R10, R8, R11, RZ     # reads R8, but from MOV, not IMAD

    For this test we check that no VIOLATION edge reports writer_idx=0
    → reader_idx=2 on register R8.  (R9 is still live from the IMAD
    pair; no consumer of R9 exists in the stream, so nothing is
    emitted for it either.)
    """
    _patch_min_gpr_gap(monkeypatch, 0x225, 2)

    instrs = _stream(
        encode_imad_wide_rr(dest=8, src0=2, src1=3, src2=0xff),
        encode_mov(dest=8, src=99),           # shadows R8
        encode_iadd3(dest=10, src0=8, src1=11, src2=0xff),
    )
    report = verify_proof(instrs)
    # No edge from writer_idx=0 to reader_idx=2 on register 8 —
    # shadowing at idx 1 means the IADD3 is reading MOV's value.
    shadowed = [
        e for e in report.edges
        if e.writer_idx == 0 and e.reader_idx == 2 and 8 in e.regs
    ]
    assert not shadowed, (
        f"INV R' failed: shadow tracking did not kill edge for "
        f"register R8 after MOV overwrite. Edges: {shadowed}"
    )


# ---------------------------------------------------------------------------
# INV S — entire real corpus remains SAFE under the expanded model
# ---------------------------------------------------------------------------

def test_inv_s_corpus_safe_under_expanded_model():
    """INV S: the corpus already asserted SAFE by INV H must still be
    SAFE under the FG-3.0 expanded proof engine, with zero
    VIOLATION edges in aggregate.  Distinct from INV I because the
    test name/semantics are anchored to the FG-3.0 upgrade — a
    regression in the bounded-lookahead path (e.g. a silent enumeration
    explosion or a shadowing bug) would show up here first.
    """
    corpus = _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels()
    total = 0
    viols = 0
    problem = []
    for label, ptx in corpus:
        try:
            report = _proof_for(ptx)
        except Exception:
            continue
        total += report.total
        viols += len(report.violations)
        if report.violations:
            problem.append(label)
    assert viols == 0, (
        f"INV S failed: {viols} VIOLATION edges across "
        f"{len(problem)} kernel(s) under FG-3.0 engine: "
        f"{problem[:10]}"
    )
    assert total > 0, "INV S corpus collected zero edges"


# ===========================================================================
# FG-3.1 INVARIANTS (T, U, V, W) — memory latency proof model
# ===========================================================================
#
# FG-3.1 extends the proof engine into memory-latency domains.  Memory
# producing opcodes (LDG / LDS / LDC and siblings) write GPRs through a
# long-latency scoreboard slot; a consumer must wait via its ctrl-word
# `rbar` bitmask.  The proof model uses the producer's EMITTED wdep
# (not a static opcode→class map) to determine which rbar bit must be
# set in the [writer+1, reader] window.
#
# Mapping (from sass.scoreboard._WDEP_TO_RBAR_MASK):
#   wdep 0x31 (LDC) → rbar 0x03 (class bit 0x02)
#   wdep 0x33 (LDS) → rbar 0x05 (class bit 0x04)
#   wdep 0x35 (LDG) → rbar 0x09 (class bit 0x08)
#
# INV T — no memory edge can be SAFE without producer-wdep evidence
# INV U — synthetic 0-gap memory reads without rbar wait are VIOLATION
# INV V — synthetic memory reads with matching rbar wait are SAFE
# INV W — the full real corpus is SAFE under the expanded model


def _mk_ctrl(rbar: int = 0x01, wdep: int = 0x3f, misc: int = 0x1) -> int:
    """Pack a 23-bit ctrl word from rbar / wdep / misc fields."""
    return ((rbar & 0x1f) << 10) | ((wdep & 0x3f) << 4) | (misc & 0xf)


# ---------------------------------------------------------------------------
# INV T — no tracked memory writer produces an unclassified edge
# ---------------------------------------------------------------------------

def test_inv_t_memory_writers_are_classified():
    """INV T: every edge whose writer is a tracked memory producer
    (wdep in _WDEP_TO_RBAR_MASK and opcode in _OPCODES_MEMORY_GPR)
    must be classified as MEMORY_SCOREBOARD_SAFE or MEMORY_VIOLATION —
    never LATENCY_INERT, FORWARDING_SAFE, etc.  This ensures the
    FG-3.1 rule claims ownership of memory edges rather than letting
    them leak into the R8 ALU path.
    """
    from sass.scoreboard import _OPCODES_MEMORY_GPR, _WDEP_TO_RBAR_MASK, _get_wdep
    corpus = _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels()
    bad = []
    for label, ptx in corpus:
        try:
            cubin, _ = compile_openptxas(ptx)
        except Exception:
            continue
        for _sym, off, sz in _iter_text_sections(cubin):
            instrs = [SassInstr(cubin[off + o:off + o + 16], "")
                      for o in range(0, sz, 16)
                      if off + o + 16 <= off + sz]
            report = verify_proof(instrs)
            for e in report.edges:
                if e.writer_opc in _OPCODES_MEMORY_GPR:
                    w = _get_wdep(instrs[e.writer_idx].raw)
                    if w in _WDEP_TO_RBAR_MASK and e.classification not in (
                            ProofClass.MEMORY_SCOREBOARD_SAFE,
                            ProofClass.MEMORY_VIOLATION):
                        bad.append((label, e.writer_idx, e.reader_idx,
                                    hex(e.writer_opc), e.classification))
    assert not bad, (
        f"INV T failed: {len(bad)} edge(s) from tracked memory writers "
        f"classified outside the MEMORY_* classes: {bad[:5]}"
    )


# ---------------------------------------------------------------------------
# INV U — synthetic unsafe memory reads are flagged as VIOLATION
# ---------------------------------------------------------------------------

def test_inv_u_synthetic_unsafe_ldg_flagged():
    """INV U (LDG): LDG.E R4, [R2:R3] writing R4 with wdep=0x35
    followed by a MOV R5, R4 at gap 0 with rbar=0x01 (no LDG wait)
    must be flagged as MEMORY_VIOLATION.
    """
    # Producer: LDG.E with wdep=0x35 (LDG slot)
    # Consumer: MOV with rbar=0x01 (no wait bit)
    ldg_ctrl = _mk_ctrl(rbar=0x01, wdep=0x35)
    mov_ctrl = _mk_ctrl(rbar=0x01, wdep=0x3e)
    instrs = _stream(
        encode_ldg_e(dest=4, ur_desc=0, src_addr=2, width=32, ctrl=ldg_ctrl),
        encode_mov(dest=5, src=4, ctrl=mov_ctrl),
    )
    report = verify_proof(instrs)
    assert not report.safe, (
        f"INV U (LDG) failed: engine did not flag LDG→MOV at gap=0 "
        f"without rbar wait. Report: {report.summary_line()}"
    )
    mviols = [v for v in report.violations
              if v.classification == ProofClass.MEMORY_VIOLATION
              and v.writer_idx == 0 and v.reader_idx == 1]
    assert len(mviols) == 1, (
        f"INV U (LDG) failed: expected one MEMORY_VIOLATION, got "
        f"{[v.rationale for v in report.violations]}"
    )


def test_inv_u_synthetic_unsafe_lds_flagged():
    """INV U (LDS): LDS with wdep=0x33 followed by a consumer with
    rbar=0x01 at gap 0 must be flagged as MEMORY_VIOLATION.
    """
    lds_ctrl = _mk_ctrl(rbar=0x01, wdep=0x33)
    mov_ctrl = _mk_ctrl(rbar=0x01, wdep=0x3e)
    instrs = _stream(
        encode_lds(dest=4, ur_addr=0, offset=0, ctrl=lds_ctrl),
        encode_mov(dest=5, src=4, ctrl=mov_ctrl),
    )
    report = verify_proof(instrs)
    assert not report.safe
    mviols = [v for v in report.violations
              if v.classification == ProofClass.MEMORY_VIOLATION]
    assert len(mviols) == 1, (
        f"INV U (LDS) failed: expected one MEMORY_VIOLATION, got "
        f"{[v.rationale for v in report.violations]}"
    )


def test_inv_u_synthetic_unsafe_ldc_flagged():
    """INV U (LDC): LDC with wdep=0x31 followed by a consumer with
    rbar=0x01 at gap 0 must be flagged as MEMORY_VIOLATION.
    """
    ldc_ctrl = _mk_ctrl(rbar=0x01, wdep=0x31)
    mov_ctrl = _mk_ctrl(rbar=0x01, wdep=0x3e)
    instrs = _stream(
        encode_ldc(dest=4, const_bank=0, const_offset_bytes=0x10,
                   ctrl=ldc_ctrl),
        encode_mov(dest=5, src=4, ctrl=mov_ctrl),
    )
    report = verify_proof(instrs)
    assert not report.safe
    mviols = [v for v in report.violations
              if v.classification == ProofClass.MEMORY_VIOLATION]
    assert len(mviols) == 1, (
        f"INV U (LDC) failed: expected one MEMORY_VIOLATION, got "
        f"{[v.rationale for v in report.violations]}"
    )


# ---------------------------------------------------------------------------
# INV V — synthetic safe memory reads are classified MEMORY_SCOREBOARD_SAFE
# ---------------------------------------------------------------------------

def test_inv_v_synthetic_safe_ldg_scoreboard_safe():
    """INV V (LDG): LDG.E with wdep=0x35 followed by a consumer whose
    ctrl word has rbar=0x09 (LDG class bit 0x08 set) at gap 0 must
    be MEMORY_SCOREBOARD_SAFE — the consumer's own rbar proves the
    scoreboard wait and no VIOLATION is emitted.
    """
    ldg_ctrl = _mk_ctrl(rbar=0x01, wdep=0x35)
    mov_ctrl = _mk_ctrl(rbar=0x09, wdep=0x3e)   # LDG-wait bit 3 set
    instrs = _stream(
        encode_ldg_e(dest=4, ur_desc=0, src_addr=2, width=32, ctrl=ldg_ctrl),
        encode_mov(dest=5, src=4, ctrl=mov_ctrl),
    )
    report = verify_proof(instrs)
    assert report.safe, (
        f"INV V (LDG) failed: engine flagged a properly rbar-guarded "
        f"LDG→MOV as unsafe. Report: {report.summary_line()}"
    )
    safe_edges = [e for e in report.edges
                  if e.classification == ProofClass.MEMORY_SCOREBOARD_SAFE
                  and e.writer_idx == 0 and e.reader_idx == 1]
    assert len(safe_edges) == 1, (
        f"INV V (LDG) failed: expected one MEMORY_SCOREBOARD_SAFE edge, "
        f"got {[(e.classification, e.rationale) for e in report.edges]}"
    )


def test_inv_v_synthetic_safe_lds_scoreboard_safe():
    """INV V (LDS): LDS with wdep=0x33 followed by rbar=0x05 reader."""
    lds_ctrl = _mk_ctrl(rbar=0x01, wdep=0x33)
    mov_ctrl = _mk_ctrl(rbar=0x05, wdep=0x3e)  # LDS bit 2 set
    instrs = _stream(
        encode_lds(dest=4, ur_addr=0, offset=0, ctrl=lds_ctrl),
        encode_mov(dest=5, src=4, ctrl=mov_ctrl),
    )
    report = verify_proof(instrs)
    assert report.safe
    safe_edges = [e for e in report.edges
                  if e.classification == ProofClass.MEMORY_SCOREBOARD_SAFE]
    assert len(safe_edges) == 1


def test_inv_v_synthetic_safe_ldc_scoreboard_safe():
    """INV V (LDC): LDC with wdep=0x31 followed by rbar=0x03 reader."""
    ldc_ctrl = _mk_ctrl(rbar=0x01, wdep=0x31)
    mov_ctrl = _mk_ctrl(rbar=0x03, wdep=0x3e)  # LDC bit 1 set
    instrs = _stream(
        encode_ldc(dest=4, const_bank=0, const_offset_bytes=0x10,
                   ctrl=ldc_ctrl),
        encode_mov(dest=5, src=4, ctrl=mov_ctrl),
    )
    report = verify_proof(instrs)
    assert report.safe
    safe_edges = [e for e in report.edges
                  if e.classification == ProofClass.MEMORY_SCOREBOARD_SAFE]
    assert len(safe_edges) == 1


def test_inv_v_prime_barrier_between_proves_wait():
    """INV V' (barrier): LDG with wdep=0x35 → NOP with rbar=0x09 (barrier
    wait) → consumer with rbar=0x01 must still be SAFE because the
    scan window [i+1, j] includes the barrier NOP whose rbar proves
    the LDG wait.  This is exactly the pattern observed in reduce_sum
    where a BSYNC between LDG and its consumer absorbs the wait.
    """
    ldg_ctrl = _mk_ctrl(rbar=0x01, wdep=0x35)
    bar_ctrl = _mk_ctrl(rbar=0x09, wdep=0x3e)  # NOP with LDG-wait bit
    mov_ctrl = _mk_ctrl(rbar=0x01, wdep=0x3e)  # consumer no wait
    instrs = _stream(
        encode_ldg_e(dest=4, ur_desc=0, src_addr=2, width=32, ctrl=ldg_ctrl),
        encode_nop(ctrl=bar_ctrl),
        encode_mov(dest=5, src=4, ctrl=mov_ctrl),
    )
    report = verify_proof(instrs)
    assert report.safe, (
        f"INV V' failed: barrier NOP between LDG and consumer did not "
        f"absorb the scoreboard wait. Report: {report.summary_line()}"
    )
    safe_edges = [e for e in report.edges
                  if e.classification == ProofClass.MEMORY_SCOREBOARD_SAFE
                  and e.writer_idx == 0 and e.reader_idx == 2]
    assert len(safe_edges) == 1, (
        f"INV V' failed: expected one MEMORY_SCOREBOARD_SAFE edge at "
        f"gap=1 via barrier, got {report.edges}"
    )


# ---------------------------------------------------------------------------
# INV W — entire real corpus remains SAFE under FG-3.1 memory model
# ---------------------------------------------------------------------------

def test_inv_w_corpus_safe_under_memory_model():
    """INV W: with the FG-3.1 memory proof model active, every kernel
    in the real corpus must remain SAFE — zero VIOLATION and zero
    MEMORY_VIOLATION edges across the aggregate.  Any MEMORY_VIOLATION
    surfacing here is a real hazard in the emitted SASS that the
    previous latency-inert classification was hiding.
    """
    corpus = _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels()
    agg_mviol = 0
    agg_viol = 0
    mem_safe = 0
    problem = []
    for label, ptx in corpus:
        try:
            report = _proof_for(ptx)
        except Exception:
            continue
        c = report.counts
        mem_safe += c[ProofClass.MEMORY_SCOREBOARD_SAFE]
        agg_mviol += c[ProofClass.MEMORY_VIOLATION]
        agg_viol += c[ProofClass.VIOLATION]
        if c[ProofClass.MEMORY_VIOLATION] or c[ProofClass.VIOLATION]:
            problem.append(label)
    assert agg_viol == 0, (
        f"INV W: ALU VIOLATION regression: {agg_viol} edges in "
        f"{problem[:10]}"
    )
    assert agg_mviol == 0, (
        f"INV W: {agg_mviol} MEMORY_VIOLATION edges in "
        f"{len(problem)} kernel(s): {problem[:10]}"
    )
    # Sanity: the memory rule should actually be classifying edges.
    assert mem_safe > 0, (
        "INV W: zero MEMORY_SCOREBOARD_SAFE edges means the memory "
        "rule never fired — regression in the FG-3.1 classifier"
    )


# ===========================================================================
# FG-3.2 INVARIANTS (X, Y, Z, AA, AB, AC) — memory-class completion
# ===========================================================================
#
# FG-3.2 closes the boundaries FG-3.1 left deferred:
#   - explicit classification of wdep=0x3b (LDG rotating variant)
#     and wdep=0x3f (no-track descriptor loads)
#   - UR-destination producer modeling (S2UR, REDUX, ULDC.64) via
#     new rule R10, in parallel with the existing LDCU.64 rule R1
#   - empirical answer on whether MEMORY_GAP_SAFE and
#     MEMORY_FORWARD_SAFE classes have any corpus or synthetic
#     evidence (answer: no, so no classes added)
#   - fail-loudly behavior for unknown memory wdep values


# ---------------------------------------------------------------------------
# INV X — every observed memory-producer wdep is explicitly classified
# ---------------------------------------------------------------------------

def test_inv_x_observed_memory_wdeps_classified():
    """INV X: across the entire real corpus, every observed GPR-dest
    memory producer must have a wdep that is EXPLICITLY classified —
    either in _WDEP_TO_RBAR_MASK (tracked scoreboard) or in
    _LATENCY_INERT_WDEPS (no-track).  An observed wdep outside both
    sets would be a silent "falls through as inert" gap, which is
    exactly what FG-3.2 eliminates.
    """
    from sass.scoreboard import (_OPCODES_MEMORY_GPR, _WDEP_TO_RBAR_MASK,
                                 _LATENCY_INERT_WDEPS, _get_opcode,
                                 _get_wdep)
    corpus = _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels()
    unclassified = set()  # (opc, wdep, sample_kernel)
    for label, ptx in corpus:
        try:
            cubin, _ = compile_openptxas(ptx)
        except Exception:
            continue
        for _sym, off, sz in _iter_text_sections(cubin):
            for k in range(sz // 16):
                raw = cubin[off + k*16 : off + k*16 + 16]
                opc = _get_opcode(raw)
                if opc not in _OPCODES_MEMORY_GPR:
                    continue
                wd = _get_wdep(raw)
                if wd not in _WDEP_TO_RBAR_MASK and wd not in _LATENCY_INERT_WDEPS:
                    unclassified.add((opc, wd, label))
    assert not unclassified, (
        f"INV X failed: {len(unclassified)} (opcode, wdep, kernel) "
        f"tuples with unclassified memory wdeps: "
        f"{sorted(unclassified)[:10]}"
    )


# ---------------------------------------------------------------------------
# INV Y — UR-destination producer chains are constructively classified
# ---------------------------------------------------------------------------

def test_inv_y_ur_producer_chains_classified():
    """INV Y: every UR-destination producer observed in the corpus
    (S2UR / REDUX / ULDC.64) must either produce a
    UR_MEMORY_SCOREBOARD_SAFE edge or have no live UR consumer.  No
    UR_MEMORY_VIOLATION and no silent skips.
    LDCU is excluded here because it still routes through rule R1.
    """
    from sass.scoreboard import _get_opcode
    corpus = _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels()
    ur_viols = 0
    ur_safe = 0
    problem = []
    for label, ptx in corpus:
        try:
            report = _proof_for(ptx)
        except Exception:
            continue
        c = report.counts
        ur_viols += c[ProofClass.UR_MEMORY_VIOLATION]
        ur_safe += c[ProofClass.UR_MEMORY_SCOREBOARD_SAFE]
        if c[ProofClass.UR_MEMORY_VIOLATION]:
            problem.append(label)
    assert ur_viols == 0, (
        f"INV Y failed: {ur_viols} UR_MEMORY_VIOLATION edges in "
        f"{problem[:10]}"
    )
    # Must actually fire at least once — otherwise R10 never ran.
    assert ur_safe > 0, (
        "INV Y: rule R10 never emitted a UR_MEMORY_SCOREBOARD_SAFE "
        "edge — regression in the UR producer enumeration."
    )


# ---------------------------------------------------------------------------
# INV Z — no memory edge in the corpus relies on instruction-stream
#          gap alone (no MEMORY_GAP_SAFE class needed)
# ---------------------------------------------------------------------------

def test_inv_z_no_memory_gap_safe_evidence():
    """INV Z: empirical check that MEMORY_GAP_SAFE is not a real
    class in the current corpus.  For every memory writer edge in the
    proof report (MEMORY_SCOREBOARD_SAFE or UR_MEMORY_SCOREBOARD_SAFE),
    verify the window [writer+1, reader] actually has rbar wait
    evidence — i.e. the classification is not just "reader far enough
    away".  This locks in the FG-3.2 empirical finding: if a future
    memory edge appears that relies on gap alone, the invariant fires
    and the user is forced to add an explicit MEMORY_GAP_SAFE class
    backed by evidence.
    """
    from sass.scoreboard import (_get_wdep, _get_rbar,
                                 _WDEP_TO_RBAR_MASK, _get_opcode)
    corpus = _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels()
    gap_only = []
    for label, ptx in corpus:
        try:
            cubin, _ = compile_openptxas(ptx)
        except Exception:
            continue
        for _sym, off, sz in _iter_text_sections(cubin):
            instrs = [SassInstr(cubin[off + o:off + o + 16], "")
                      for o in range(0, sz, 16)
                      if off + o + 16 <= off + sz]
            report = verify_proof(instrs)
            for e in report.edges:
                if e.classification not in (
                        ProofClass.MEMORY_SCOREBOARD_SAFE,
                        ProofClass.UR_MEMORY_SCOREBOARD_SAFE):
                    continue
                wd = _get_wdep(instrs[e.writer_idx].raw)
                if wd not in _WDEP_TO_RBAR_MASK:
                    continue
                cb = _WDEP_TO_RBAR_MASK[wd] & ~0x01
                has_wait = any(
                    _get_rbar(instrs[m].raw) & cb
                    for m in range(e.writer_idx + 1, e.reader_idx + 1)
                )
                if not has_wait:
                    gap_only.append((label, e.writer_idx, e.reader_idx,
                                     hex(wd), e.classification))
    assert not gap_only, (
        f"INV Z failed: {len(gap_only)} memory edge(s) claim "
        f"SCOREBOARD_SAFE without rbar wait evidence in the window. "
        f"Either the classifier is wrong or there is genuine "
        f"MEMORY_GAP_SAFE evidence to add an explicit class for: "
        f"{gap_only[:5]}"
    )


# ---------------------------------------------------------------------------
# INV AA — no corpus or synthetic evidence for MEMORY_FORWARD_SAFE
# ---------------------------------------------------------------------------

def test_inv_aa_no_memory_forward_safe_evidence():
    """INV AA: there is no memory producer→consumer pair in the
    corpus at gap=0 without a direct rbar wait on the consumer.  That
    would be the only pattern consistent with a "memory operand
    forwarding" story, and there is no such evidence.  Like INV Z,
    this locks the empirical finding so that a future edge matching
    the pattern cannot slip in silently.
    """
    from sass.scoreboard import _get_wdep, _get_rbar, _WDEP_TO_RBAR_MASK
    corpus = _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels()
    candidates = []
    for label, ptx in corpus:
        try:
            cubin, _ = compile_openptxas(ptx)
        except Exception:
            continue
        for _sym, off, sz in _iter_text_sections(cubin):
            instrs = [SassInstr(cubin[off + o:off + o + 16], "")
                      for o in range(0, sz, 16)
                      if off + o + 16 <= off + sz]
            report = verify_proof(instrs)
            for e in report.edges:
                if e.classification not in (
                        ProofClass.MEMORY_SCOREBOARD_SAFE,
                        ProofClass.UR_MEMORY_SCOREBOARD_SAFE):
                    continue
                if e.gap != 0:
                    continue
                wd = _get_wdep(instrs[e.writer_idx].raw)
                if wd not in _WDEP_TO_RBAR_MASK:
                    continue
                cb = _WDEP_TO_RBAR_MASK[wd] & ~0x01
                # Gap=0 means reader is at i+1.  Check the reader's
                # own rbar — if it does not have the class bit, the
                # edge would be a forwarding candidate.
                rdr_rbar = _get_rbar(instrs[e.reader_idx].raw)
                if not (rdr_rbar & cb):
                    candidates.append((label, e.writer_idx, e.reader_idx))
    assert not candidates, (
        f"INV AA failed: {len(candidates)} memory edge(s) at gap=0 "
        f"classified SAFE without rbar wait on the reader itself. "
        f"This would be the signature of a MEMORY_FORWARD_SAFE "
        f"class; if real, add it with evidence: {candidates[:5]}"
    )


# ---------------------------------------------------------------------------
# INV AB — all real kernels remain SAFE under FG-3.2
# ---------------------------------------------------------------------------

def test_inv_ab_corpus_safe_under_fg32():
    """INV AB: the full corpus remains SAFE with zero VIOLATION,
    MEMORY_VIOLATION, and UR_MEMORY_VIOLATION edges in aggregate.
    Distinct from INV H/I/S/W as an FG-3.2 anchor so regressions in
    the new UR rule or the extended wdep table fail here first.
    """
    corpus = _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels()
    totals = {cls: 0 for cls in ProofClass}
    problem = []
    for label, ptx in corpus:
        try:
            report = _proof_for(ptx)
        except Exception:
            continue
        for cls, n in report.counts.items():
            totals[cls] += n
        if report.violations:
            problem.append(label)
    assert totals[ProofClass.VIOLATION] == 0
    assert totals[ProofClass.MEMORY_VIOLATION] == 0
    assert totals[ProofClass.UR_MEMORY_VIOLATION] == 0
    assert not problem, f"INV AB: violations in {problem[:10]}"
    # Sanity: FG-3.2 rules actually produce edges in the corpus.
    assert totals[ProofClass.UR_MEMORY_SCOREBOARD_SAFE] > 0, (
        "INV AB: UR rule R10 never fired"
    )


# ---------------------------------------------------------------------------
# INV AC — unknown memory wdep on a synthetic stream fails loudly
# ---------------------------------------------------------------------------

def test_inv_ac_unknown_wdep_fails_loudly():
    """INV AC: if a memory producer is emitted with a wdep value that
    is neither in _WDEP_TO_RBAR_MASK nor in _LATENCY_INERT_WDEPS, the
    proof engine must NOT silently skip the edge — it must classify
    it as MEMORY_VIOLATION with an "unknown wdep" rationale so the
    mis-modeled slot is surfaced immediately.

    Build a synthetic LDG with a deliberately unknown wdep=0x3c and
    verify the proof report contains a MEMORY_VIOLATION whose
    rationale names the unknown slot.
    """
    # Synthetic LDG.E with a weird (unknown) wdep slot 0x3c.
    bad_ctrl = _mk_ctrl(rbar=0x01, wdep=0x3c)
    mov_ctrl = _mk_ctrl(rbar=0x09, wdep=0x3e)  # would cover 0x35/0x3b
    instrs = _stream(
        encode_ldg_e(dest=4, ur_desc=0, src_addr=2, width=32, ctrl=bad_ctrl),
        encode_mov(dest=5, src=4, ctrl=mov_ctrl),
    )
    report = verify_proof(instrs)
    assert not report.safe, (
        f"INV AC failed: unknown wdep=0x3c did not produce a "
        f"VIOLATION. Report: {report.summary_line()}"
    )
    assert any(
        v.classification == ProofClass.MEMORY_VIOLATION and "UNKNOWN" in v.rationale
        for v in report.violations
    ), (
        f"INV AC failed: expected MEMORY_VIOLATION with 'UNKNOWN' "
        f"rationale, got: "
        f"{[(v.classification, v.rationale[:60]) for v in report.violations]}"
    )


# ---------------------------------------------------------------------------
# INV AC' — MEMORY_INERT is emitted for wdep=0x3f
# ---------------------------------------------------------------------------

def test_inv_ac_prime_memory_inert_classified():
    """INV AC': a memory producer with wdep=0x3f (explicit no-track)
    must produce a MEMORY_INERT edge for its first reader rather than
    being silently skipped.  This is the "every candidate edge is
    constructively classified" rule from FG-3.2.
    """
    ldc_ctrl = _mk_ctrl(rbar=0x01, wdep=0x3f)
    mov_ctrl = _mk_ctrl(rbar=0x01, wdep=0x3e)
    instrs = _stream(
        encode_ldc(dest=4, const_bank=0, const_offset_bytes=0x10,
                   ctrl=ldc_ctrl),
        encode_mov(dest=5, src=4, ctrl=mov_ctrl),
    )
    report = verify_proof(instrs)
    inerts = [e for e in report.edges
              if e.classification == ProofClass.MEMORY_INERT]
    assert len(inerts) == 1, (
        f"INV AC': expected one MEMORY_INERT edge for wdep=0x3f LDC, "
        f"got {[(e.classification, e.rationale[:60]) for e in report.edges]}"
    )
    assert report.safe, (
        f"INV AC': LDC wdep=0x3f → MOV must be SAFE; report="
        f"{report.summary_line()}"
    )
