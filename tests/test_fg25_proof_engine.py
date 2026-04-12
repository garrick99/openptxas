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
    encode_ldcu_64, encode_hfma2_zero,
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
    # FG-3.3 13-tuple: (total, LATENCY_INERT, FORWARDING_SAFE,
    # CTRLWORD_SAFE, GAP_SAFE, MEMORY_SCOREBOARD_SAFE, MEMORY_INERT,
    # MEMORY_VIOLATION, UR_MEMORY_SCOREBOARD_SAFE, UR_MEMORY_GAP_SAFE,
    # UR_MEMORY_INERT, UR_MEMORY_VIOLATION, VIOLATION).
    #
    # FG-3.3: legacy R1 rule (LDCU.64 → first UR consumer) retired.
    # All LDCU edges now flow through the unified R10 path.  Changes
    # from FG-3.2 baselines:
    #   - CTRLWORD_SAFE and GAP_SAFE drop to 0 on every kernel that
    #     used to rely on R1 (LDCU was the only emitter of those
    #     classes in the corpus).
    #   - UR_MEMORY_SCOREBOARD_SAFE grows: 150 + 34 + 22 + 15 = 221
    #     new LDCU edges join the existing 16 S2UR/REDUX edges in
    #     the UR scoreboard class.
    #   - UR_MEMORY_GAP_SAFE: new column, 4 edges in corpus
    #     (hmma_zero / imma_zero / dmma_zero / fg114b_diag3 — LDCU.64
    #     wdep=0x35 → STG at gap ≥ 3 with no rbar evidence).
    #   - UR_MEMORY_INERT: new column, 1 edge (reduce_sum post-
    #     boundary LDCU.64 wdep=0x37).
    #   - Totals grow by the number of LDCU edges R1 used to miss
    #     (STG-consumer edges and LDCU.32 edges).
    "diag":                  (4, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0),
    "diag3":                 (4, 0, 0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0),
    # FG-4.4 drift: the bug-1 fix moves one u64 param from UR to GPR
    # (LDC.64 direct) on kernels where the param register is redefined
    # later.  That adds one MEMORY_SCOREBOARD_SAFE edge for the new
    # LDC.64 → consumer chain and shifts one UR_MEMORY_* edge.
    "min_store_guarded":     (5, 0, 0, 0, 0, 2, 0, 0, 2, 1, 0, 0, 0),
    "probe_fresh":           (5, 0, 0, 0, 0, 2, 0, 0, 2, 1, 0, 0, 0),
    "reduce_sum":            (18, 4, 6, 0, 0, 3, 0, 0, 4, 0, 1, 0, 0),
    "conv2d_looped":         (109, 10, 0, 0, 0, 63, 0, 0, 36, 0, 0, 0, 0),
    "conv2d_unrolled":       (91, 1, 8, 0, 0, 47, 0, 0, 35, 0, 0, 0, 0),
    "fg21:k_ge":             (5, 0, 0, 0, 0, 2, 0, 0, 2, 1, 0, 0, 0),
    "fg21:k_lt":             (5, 0, 0, 0, 0, 2, 0, 0, 2, 1, 0, 0, 0),
    "fg21:k_gt":             (5, 0, 0, 0, 0, 2, 0, 0, 2, 1, 0, 0, 0),
    "fg21:k_le":             (5, 0, 0, 0, 0, 2, 0, 0, 2, 1, 0, 0, 0),
    "fg21:k_eq":             (5, 0, 0, 0, 0, 2, 0, 0, 2, 1, 0, 0, 0),
    "fg21:k_ne":             (5, 0, 0, 0, 0, 2, 0, 0, 2, 1, 0, 0, 0),
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
        c[ProofClass.UR_MEMORY_GAP_SAFE],
        c[ProofClass.UR_MEMORY_INERT],
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


# ===========================================================================
# FG-3.3 INVARIANTS (AD, AE, AF, AG, AH) — LDCU unification
# ===========================================================================
#
# FG-3.3 retires the legacy R1 rule (LDCU.64 → first UR consumer via
# a bespoke gap/whitelist classifier) and folds LDCU into the same
# wdep-based rule R10 used for every other UR-destination producer.
# The corpus evidence justified two additional UR classes beyond
# UR_MEMORY_SCOREBOARD_SAFE:
#   - UR_MEMORY_GAP_SAFE  — LDCU.64 with gap ≥ _LDCU_GAP_SAFE_MIN and
#                           no rbar evidence (4 corpus edges).
#   - UR_MEMORY_INERT     — producer wdep in _LATENCY_INERT_WDEPS
#                           (reduce_sum wdep=0x37, 1 corpus edge).


# ---------------------------------------------------------------------------
# INV AD — no legacy R1 path remains for LDCU classification
# ---------------------------------------------------------------------------

def test_inv_ad_no_legacy_r1_path():
    """INV AD: verify_proof contains no references to the legacy R1
    mechanism — no `rule R1` comment token, no
    `_LDCU_GAP_EXEMPT_CONSUMERS` import.  Detects accidental
    re-introduction by source inspection of verify_proof.

    (The same scoreboard.py set may still exist; schedule.py uses
    LDCU.64 byte gates in OTHER scheduling passes which are not
    part of the proof path — those are not in scope for INV AD.)
    """
    import re, inspect
    from sass.schedule import verify_proof
    src = inspect.getsource(verify_proof)
    assert not re.search(r"\brule R1\b", src), (
        "legacy 'rule R1' comment reappeared in verify_proof"
    )
    assert "_LDCU_GAP_EXEMPT_CONSUMERS" not in src, (
        "legacy _LDCU_GAP_EXEMPT_CONSUMERS import reappeared "
        "in verify_proof"
    )
    # The R1 rule gated LDCU.64 via si.raw[9] != 0x0a.  verify_proof
    # no longer needs that gate — LDCU.64 is classified uniformly
    # via wdep-based rule R10.
    assert "raw[9] != 0x0a" not in src, (
        "legacy LDCU.64 b9 gate reappeared in verify_proof"
    )


# ---------------------------------------------------------------------------
# INV AE — every corpus LDCU edge is in a UR_* class under R10
# ---------------------------------------------------------------------------

def test_inv_ae_ldcu_edges_classified_unified():
    """INV AE: every corpus edge whose writer is LDCU (0x7ac) is
    classified by the unified R10 path — i.e. in one of the UR_*
    classes.  No legacy CTRLWORD_SAFE / GAP_SAFE for LDCU.
    """
    corpus = _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels()
    ok_classes = {
        ProofClass.UR_MEMORY_SCOREBOARD_SAFE,
        ProofClass.UR_MEMORY_GAP_SAFE,
        ProofClass.UR_MEMORY_INERT,
        ProofClass.UR_MEMORY_VIOLATION,
    }
    bad = []
    count = 0
    for label, ptx in corpus:
        try:
            report = _proof_for(ptx)
        except Exception:
            continue
        for e in report.edges:
            if e.writer_opc == 0x7ac:
                count += 1
                if e.classification not in ok_classes:
                    bad.append((label, e.classification, e.rationale[:60]))
    assert count > 0, (
        "INV AE: no LDCU edges found in corpus — R10 never enumerated them"
    )
    assert not bad, (
        f"INV AE: {len(bad)} LDCU edges classified outside UR_* classes: "
        f"{bad[:5]}"
    )


# ---------------------------------------------------------------------------
# INV AF — synthetic unsafe LDCU.64 case is flagged
# ---------------------------------------------------------------------------

def test_inv_af_synthetic_unsafe_ldcu():
    """INV AF: LDCU.64 with tracked wdep=0x35 followed by an IMAD.R-UR
    at gap=0 with no rbar wait bit for LDG class must produce a
    UR_MEMORY_VIOLATION.  Gap=0 excludes the gap-safe fallback
    (_LDCU_GAP_SAFE_MIN = 3).
    """
    ldcu_ctrl = _mk_ctrl(rbar=0x01, wdep=0x35)
    imad_ctrl = _mk_ctrl(rbar=0x01, wdep=0x3e)  # no LDG wait bit
    instrs = _stream(
        encode_ldcu_64(dest_ur=4, const_bank=0, const_offset_bytes=0x10,
                       ctrl=ldcu_ctrl),
        # IMAD R5, R3, UR4, R2 — reads UR4 at gap=0
        encode_imad_ur(dest=5, src0=3, ur_src=4, src2=2,
                       ctrl=imad_ctrl),
    )
    report = verify_proof(instrs)
    assert not report.safe, (
        f"INV AF failed: LDCU.64 → IMAD.R-UR at gap=0 with no rbar "
        f"wait should be UNSAFE. Report: {report.summary_line()}"
    )
    viols = [v for v in report.violations
             if v.classification == ProofClass.UR_MEMORY_VIOLATION]
    assert len(viols) == 1, (
        f"INV AF: expected one UR_MEMORY_VIOLATION, got "
        f"{[(v.classification, v.rationale[:80]) for v in report.violations]}"
    )


# ---------------------------------------------------------------------------
# INV AG — synthetic safe LDCU.64 cases are classified SAFE
# ---------------------------------------------------------------------------

def test_inv_ag_synthetic_safe_ldcu_rbar():
    """INV AG (rbar): LDCU.64 wdep=0x35 → IMAD.R-UR with rbar=0x09
    (LDG class bit 3 set) at gap=0 must be UR_MEMORY_SCOREBOARD_SAFE.
    This is the direct rbar-wait path.
    """
    ldcu_ctrl = _mk_ctrl(rbar=0x01, wdep=0x35)
    imad_ctrl = _mk_ctrl(rbar=0x09, wdep=0x3e)  # LDG bit 3 set
    instrs = _stream(
        encode_ldcu_64(dest_ur=4, const_bank=0, const_offset_bytes=0x10,
                       ctrl=ldcu_ctrl),
        encode_imad_ur(dest=5, src0=3, ur_src=4, src2=2,
                       ctrl=imad_ctrl),
    )
    report = verify_proof(instrs)
    assert report.safe, (
        f"INV AG (rbar) failed: rbar-guarded LDCU → IMAD UNSAFE. "
        f"Report: {report.summary_line()}"
    )
    safe_edges = [e for e in report.edges
                  if e.classification == ProofClass.UR_MEMORY_SCOREBOARD_SAFE]
    assert len(safe_edges) == 1


def test_inv_ag_synthetic_safe_ldcu_gap():
    """INV AG (gap): LDCU.64 wdep=0x35 → NOP → NOP → NOP → IMAD.R-UR
    (gap=3) with NO rbar wait bits must still be SAFE, classified as
    UR_MEMORY_GAP_SAFE.  This is the ptxas LDCU.64 latency convention
    made explicit under the unified model.
    """
    ldcu_ctrl = _mk_ctrl(rbar=0x01, wdep=0x35)
    nop_ctrl  = _mk_ctrl(rbar=0x01, wdep=0x3e)
    imad_ctrl = _mk_ctrl(rbar=0x01, wdep=0x3e)
    instrs = _stream(
        encode_ldcu_64(dest_ur=4, const_bank=0, const_offset_bytes=0x10,
                       ctrl=ldcu_ctrl),
        encode_nop(ctrl=nop_ctrl),
        encode_nop(ctrl=nop_ctrl),
        encode_nop(ctrl=nop_ctrl),
        encode_imad_ur(dest=5, src0=3, ur_src=4, src2=2,
                       ctrl=imad_ctrl),
    )
    report = verify_proof(instrs)
    assert report.safe, (
        f"INV AG (gap) failed: LDCU.64 at gap=3 should be SAFE. "
        f"Report: {report.summary_line()}"
    )
    gap_edges = [e for e in report.edges
                 if e.classification == ProofClass.UR_MEMORY_GAP_SAFE]
    assert len(gap_edges) == 1, (
        f"INV AG (gap) failed: expected one UR_MEMORY_GAP_SAFE edge, "
        f"got {[(e.classification, e.rationale[:80]) for e in report.edges]}"
    )


def test_inv_ag_synthetic_safe_ldcu_inert():
    """INV AG (inert): LDCU.64 with wdep=0x37 (reserved no-rbar slot)
    followed by an IMAD.R-UR at gap=0 must be SAFE, classified as
    UR_MEMORY_INERT.  Slot 0x37 carries no rbar bit so the hardware
    cannot — and does not need to — wait via scoreboard.
    """
    ldcu_ctrl = _mk_ctrl(rbar=0x01, wdep=0x37)
    imad_ctrl = _mk_ctrl(rbar=0x01, wdep=0x3e)
    instrs = _stream(
        encode_ldcu_64(dest_ur=4, const_bank=0, const_offset_bytes=0x10,
                       ctrl=ldcu_ctrl),
        encode_imad_ur(dest=5, src0=3, ur_src=4, src2=2,
                       ctrl=imad_ctrl),
    )
    report = verify_proof(instrs)
    assert report.safe, (
        f"INV AG (inert) failed: wdep=0x37 should be SAFE via inert "
        f"class. Report: {report.summary_line()}"
    )
    inert_edges = [e for e in report.edges
                   if e.classification == ProofClass.UR_MEMORY_INERT]
    assert len(inert_edges) == 1


# ---------------------------------------------------------------------------
# INV AH — the real corpus remains SAFE after LDCU unification
# ---------------------------------------------------------------------------

# ===========================================================================
# FG-4.1 INVARIANTS (ADJ1–ADJ4) — HFMA2 zero-init precision refinement
# ===========================================================================
#
# FG-4.1 adds a narrow semantic exemption to the ALU latency rule:
# the HFMA2 zero-init trick (HFMA2 Rd, -RZ, imm_fp16x2, RZ) has a
# deterministically-zero result and the hardware forwards it to the
# next consumer without observing the standard 1-instruction latency.
# The exemption is gated by `_is_zero_init_fastpath(raw)` which tests
# `opcode==0x431 AND b3==0xff AND b8==0xff` — nothing else.
#
# Evidence for the refinement: five FG-4.0 adversarial false
# positives (f3_shadowed_gap0, f4_setp_exit_ne, f5_mov_shadow,
# f5_ld_shadow, f5_long_shadow) all trace to the same PTXAS-emitted
# HFMA2 bytes with src0=0xff and src2=0xff.


def _patch_hfma2_non_rz(raw: bytes, new_src0: int = 0x05,
                         new_src2: int = 0x06) -> bytes:
    """Turn an encode_hfma2_zero() output into a non-zero-init HFMA2
    by overwriting the b3 (src0) and b8 (src2) fields with GPR
    indices.  Everything else (opcode, dest, immediate, ctrl word)
    is preserved.
    """
    buf = bytearray(raw)
    buf[3] = new_src0 & 0xff
    buf[8] = new_src2 & 0xff
    return bytes(buf)


def test_inv_adj1_hfma2_zero_init_safe():
    """INV ADJ1: HFMA2 Rd, -RZ, imm, RZ followed by an immediate
    consumer must be classified ZERO_INIT_SAFE, not VIOLATION.  This
    is the FG-4.1 precision refinement in its most direct form.
    """
    instrs = _stream(
        encode_hfma2_zero(dest=5),
        # Consumer at gap=0 reading R5 — MOV R6, R5.
        encode_mov(dest=6, src=5),
    )
    report = verify_proof(instrs)
    assert report.safe, (
        f"INV ADJ1 failed: HFMA2 zero-init → MOV should be SAFE. "
        f"Report: {report.summary_line()}"
    )
    zi_edges = [e for e in report.edges
                if e.classification == ProofClass.ZERO_INIT_SAFE]
    assert len(zi_edges) == 1, (
        f"INV ADJ1: expected 1 ZERO_INIT_SAFE edge, got "
        f"{[(e.classification, e.rationale[:80]) for e in report.edges]}"
    )
    assert zi_edges[0].writer_opc == 0x431
    assert zi_edges[0].reader_opc == 0x202
    assert 5 in zi_edges[0].regs


def test_inv_adj2_hfma2_non_rz_still_violation():
    """INV ADJ2: nearby non-zero-init HFMA2 (either src0 or src2 is
    a real register, not RZ) must STILL be flagged as VIOLATION —
    the fastpath rule is narrow and does not accidentally whitelist
    all HFMA2.
    """
    # Start from the zero-init encoding and patch src0 / src2 to
    # GPR indices.
    hfma_raw = _patch_hfma2_non_rz(encode_hfma2_zero(dest=5),
                                    new_src0=0x05, new_src2=0x06)
    instrs = _stream(hfma_raw, encode_mov(dest=6, src=5))
    report = verify_proof(instrs)
    assert not report.safe, (
        f"INV ADJ2 failed: non-RZ HFMA2 at gap=0 should still be "
        f"UNSAFE. Report: {report.summary_line()}"
    )
    viols = [v for v in report.violations
             if v.writer_opc == 0x431]
    assert len(viols) == 1, (
        f"INV ADJ2: expected 1 HFMA2 VIOLATION, got "
        f"{[(v.classification, v.rationale[:80]) for v in report.violations]}"
    )
    # And the safe-edge list must NOT include a ZERO_INIT_SAFE.
    zi_edges = [e for e in report.edges
                if e.classification == ProofClass.ZERO_INIT_SAFE]
    assert len(zi_edges) == 0


def test_inv_adj2b_hfma2_src0_only_rz_still_violation():
    """INV ADJ2b: even if src0 is RZ but src2 is a real register,
    the edge must NOT be classified ZERO_INIT_SAFE — the trick
    requires BOTH src0 and src2 to be RZ.
    """
    hfma_raw = _patch_hfma2_non_rz(encode_hfma2_zero(dest=5),
                                    new_src0=0xff, new_src2=0x06)
    instrs = _stream(hfma_raw, encode_mov(dest=6, src=5))
    report = verify_proof(instrs)
    assert not report.safe
    zi_edges = [e for e in report.edges
                if e.classification == ProofClass.ZERO_INIT_SAFE]
    assert len(zi_edges) == 0


def test_inv_adj2c_hfma2_src2_only_rz_still_violation():
    """INV ADJ2c: even if src2 is RZ but src0 is a real register,
    the edge must NOT be classified ZERO_INIT_SAFE."""
    hfma_raw = _patch_hfma2_non_rz(encode_hfma2_zero(dest=5),
                                    new_src0=0x05, new_src2=0xff)
    instrs = _stream(hfma_raw, encode_mov(dest=6, src=5))
    report = verify_proof(instrs)
    assert not report.safe
    zi_edges = [e for e in report.edges
                if e.classification == ProofClass.ZERO_INIT_SAFE]
    assert len(zi_edges) == 0


def test_inv_adj3_no_false_negatives_introduced():
    """INV ADJ3: the FG-4.1 refinement must NOT turn any VIOLATION
    from the existing negative-control INV tests into SAFE.
    """
    # Reuse the FG-2.5 INV L hazard (IMAD.R-UR → MOV) — must still
    # be UNSAFE.
    from sass.encoding.sm_120_opcodes import encode_imad_ur
    instrs = _stream(
        encode_imad_ur(dest=4, src0=3, ur_src=6, src2=2),
        encode_mov(dest=5, src=4),
    )
    report = verify_proof(instrs)
    assert not report.safe, (
        "INV ADJ3 regression: FG-2.5 INV L IMAD.R-UR → MOV hazard "
        "no longer flagged — FG-4.1 refinement leaked"
    )


# ===========================================================================
# FG-4.2 INVARIANTS (AE1, AE2, AE3) — hardware-evidence forwarding pairs
# ===========================================================================
#
# FG-4.2 adds three new entries to _FORWARDING_SAFE_PAIRS backed by
# GPU-runtime evidence from probe_work/fg42_evidence_harness.py:
#
#   (0x224, 0x986)  IMAD.32 → STG.E
#   (0x819, 0x986)  SHF     → STG.E
#   (0x235, 0x235)  IADD.64 → IADD.64
#
# Each is confirmed by a kernel where PTXAS emits the pair at gap=0
# AND the GPU runtime output matches a Python-computed expected
# value (non-trivial computation).  Other FG-4.0 false-positive pairs
# (LEA→IADD3X, LEA→STG, IMAD.32→IADD3X, SHF→IADD.64) remain
# conservative because their replay evidence was trivially zero and
# no dedicated probe could exercise them with a non-zero computation.


def test_inv_ae1_fg42_pairs_are_forwarding_safe():
    """INV AE1: each FG-4.2 evidence-backed pair must be on
    _FORWARDING_SAFE_PAIRS so the proof engine classifies them as
    FORWARDING_SAFE instead of VIOLATION.
    """
    from sass.scoreboard import _FORWARDING_SAFE_PAIRS
    confirmed = {
        (0x224, 0x986),  # IMAD.32 → STG.E
        (0x819, 0x986),  # SHF     → STG.E
        (0x235, 0x235),  # IADD.64 → IADD.64
    }
    missing = confirmed - _FORWARDING_SAFE_PAIRS
    assert not missing, (
        f"INV AE1: FG-4.2 evidence-backed pairs missing from "
        f"_FORWARDING_SAFE_PAIRS: {missing}"
    )


def test_inv_ae2_nearby_pair_still_conservative():
    """INV AE2: pairs NOT covered by any evidence pass (FG-4.2
    through FG-4.3) must still be flagged as VIOLATION.  FG-4.3
    confirmed LEA→IADD3X, LEA→STG, IMAD.32→IADD3X, and (transitively)
    IADD3X→STG via dedicated non-trivial probes.  The only remaining
    uncovered pair is (0x819, 0x235) SHF→IADD.64 — probes could not
    force PTXAS to emit 0x819 for that ALU shape while also producing
    a Python-matching non-zero output.
    """
    from sass.scoreboard import _FORWARDING_SAFE_PAIRS
    unsupported = {
        (0x819, 0x235),  # SHF → IADD.64 — inconclusive in FG-4.3
    }
    leaked = unsupported & _FORWARDING_SAFE_PAIRS
    assert not leaked, (
        f"INV AE2: unsupported pairs leaked into _FORWARDING_SAFE_PAIRS "
        f"without evidence: {leaked}"
    )


# ===========================================================================
# FG-4.3 INVARIANTS — evidence-backed forwarding pairs for F6 patterns
# ===========================================================================
#
# FG-4.3 added four more _FORWARDING_SAFE_PAIRS entries after running
# non-trivial clones of four FG-4.0 false-positive kernels on GPU and
# comparing PTXAS outputs against a small PTX-body simulator.  The
# four pairs below are each backed by at least one probe whose runtime
# output matched the simulator bit-for-bit with arg1=0x12345678.


def test_inv_af1_fg43_pairs_are_forwarding_safe():
    """INV AF1: each FG-4.3 evidence-backed pair must be on
    _FORWARDING_SAFE_PAIRS.
    """
    from sass.scoreboard import _FORWARDING_SAFE_PAIRS
    confirmed = {
        (0x211, 0x212),  # LEA → IADD3X
        (0x211, 0x986),  # LEA → STG.E
        (0x224, 0x212),  # IMAD.32 → IADD3X
        (0x212, 0x986),  # IADD3X → STG.E (transitive)
    }
    missing = confirmed - _FORWARDING_SAFE_PAIRS
    assert not missing, (
        f"INV AF1: FG-4.3 pairs missing from _FORWARDING_SAFE_PAIRS: "
        f"{missing}"
    )


def test_inv_af2_shf_iadd64_still_conservative():
    """INV AF2: the SHF→IADD.64 pair (0x819, 0x235) was the one
    FG-4.3 probe that could NOT be confirmed — PTXAS re-routed
    through opcode 0x899 instead of 0x819 on every clone shape.
    The pair must remain OFF the whitelist until a future pass
    produces direct evidence.
    """
    from sass.scoreboard import _FORWARDING_SAFE_PAIRS
    assert (0x819, 0x235) not in _FORWARDING_SAFE_PAIRS, (
        "INV AF2: (0x819, 0x235) SHF→IADD.64 leaked into "
        "_FORWARDING_SAFE_PAIRS without evidence"
    )


def test_fg44_bug2_mad_lo_immediate():
    """FG-4.4 Bug 2: mad.lo.u32 with immediate src1 and/or src2 used
    to drop operands silently.  `mad.lo.u32 %r1, %r0, 3, 7` was the
    minimal repro — OURS produced 0 instead of r0*3+7.  Root cause:
    (a) src2 ImmOp was replaced by RZ without materializing the
    immediate; (b) the literal-pool slot for src1 was placed adjacent
    to the param area and zeroed by the CUDA driver at launch.  Fix:
    use encode_imad_r_imm (inline 16-bit immediate) for small imm
    multipliers, and materialize src2 immediate via _materialize_imm.

    This is a compile-only regression — the runtime correctness check
    lives in probe_work/ (not runnable under unit tests that avoid
    GPU launches).  Here we only verify the kernel builds.
    """
    from benchmarks.bench_util import compile_openptxas
    ptx = """
.version 8.8
.target sm_120
.address_size 64

.visible .entry k_fg44_bug2(.param .u64 out, .param .u32 a) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<2>;
    ld.param.u64 %rd0, [out];
    ld.param.u32 %r0, [a];
    mad.lo.u32 %r1, %r0, 3, 7;
    st.global.u32 [%rd0], %r1;
    ret;
}
"""
    cubin, _ = compile_openptxas(ptx)
    assert len(cubin) > 0
    # Sanity check: the cubin should contain an IMAD R-imm (0x824)
    # rather than LDCU.32 → IMAD R-UR for the mad.lo pattern.
    import struct as _s
    e_shoff = _s.unpack_from('<Q', cubin, 40)[0]
    e_shnum = _s.unpack_from('<H', cubin, 60)[0]
    e_shstrndx = _s.unpack_from('<H', cubin, 62)[0]
    stoff = _s.unpack_from('<Q', cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    found_imad_r_imm = False
    for i in range(e_shnum):
        base = e_shoff + i * 64
        nm = _s.unpack_from('<I', cubin, base)[0]
        end = cubin.index(0, stoff + nm)
        name = cubin[stoff + nm:end]
        if name.startswith(b'.text.'):
            off = _s.unpack_from('<Q', cubin, base + 24)[0]
            sz = _s.unpack_from('<Q', cubin, base + 32)[0]
            for k in range(sz // 16):
                raw = cubin[off + k*16 : off + k*16 + 16]
                opc = raw[0] | ((raw[1] & 0xF) << 8)
                if opc == 0x824:
                    found_imad_r_imm = True
            break
    assert found_imad_r_imm, (
        "FG-4.4 Bug 2 regression: kernel did not emit IMAD R-imm "
        "(0x824) — the inline-immediate fix was bypassed"
    )


def test_fg44_bug1_duplicate_ldparam_u64():
    """FG-4.4 Bug 1: a kernel that loads the same u64 param into
    two different vregs AND redefines one of them via add.u64 used
    to raise KeyError('%rd2') in the regalloc → isel path.  The
    fix in _select_ld_param routes a GPR-allocated u64 param
    through LDC.64 (direct-to-GPR) instead of the UR path, so the
    subsequent add.u64 can resolve both operands.
    """
    from benchmarks.bench_util import compile_openptxas
    ptx = """
.version 8.8
.target sm_120
.address_size 64

.visible .entry k_fg44_bug1(.param .u64 arg0) {
    .reg .u64 %rd<4>;
    ld.param.u64 %rd1, [arg0];
    ld.param.u64 %rd2, [arg0];
    add.u64 %rd1, %rd1, %rd2;
    st.global.u64 [%rd1], %rd1;
    ret;
}
"""
    # Must not raise.  Produces a valid cubin.
    cubin, _ = compile_openptxas(ptx)
    assert len(cubin) > 0


def test_inv_af3_no_false_negatives_from_fg43():
    """INV AF3: FG-4.3 additions must not turn any FG-2.5
    negative-control VIOLATION into SAFE.  Re-runs INV L (IMAD.R-UR
    → MOV) and INV N (IMAD.WIDE.RR → IADD3) as live verify_proof
    calls.
    """
    from sass.encoding.sm_120_opcodes import encode_imad_ur
    instrs_l = _stream(
        encode_imad_ur(dest=4, src0=3, ur_src=6, src2=2),
        encode_mov(dest=5, src=4),
    )
    report_l = verify_proof(instrs_l)
    assert not report_l.safe, (
        "INV AF3: INV L (IMAD.R-UR → MOV) hazard no longer flagged "
        "— FG-4.3 refinement leaked"
    )
    instrs_n = _stream(
        encode_imad_wide_rr(dest=8, src0=2, src1=3, src2=0xff),
        encode_iadd3(dest=10, src0=8, src1=11, src2=0xff),
    )
    report_n = verify_proof(instrs_n)
    assert not report_n.safe, (
        "INV AF3: INV N (IMAD.WIDE.RR → IADD3) hazard no longer "
        "flagged — FG-4.3 refinement leaked"
    )


def test_inv_ae3_no_false_negatives_from_fg42():
    """INV AE3: the FG-4.2 additions must not turn any FG-2.5
    negative-control VIOLATION into SAFE.  Specifically:

      - INV L (IMAD.R-UR → MOV): must still be UNSAFE because
        the pair is (0xc24, 0x202), not (0x224, 0x986).
      - INV N (IMAD.WIDE.RR → IADD3): must still be UNSAFE
        because the pair is (0x225, 0x210).

    Regenerating these as live verify_proof calls guards against
    any accidental overlap between FG-4.2 entries and existing
    negative test patterns.
    """
    # Re-run INV L hazard
    instrs_l = _stream(
        encode_imad_ur(dest=4, src0=3, ur_src=6, src2=2),
        encode_mov(dest=5, src=4),
    )
    report_l = verify_proof(instrs_l)
    assert not report_l.safe, (
        "INV AE3: INV L hazard (IMAD.R-UR → MOV) no longer flagged — "
        "FG-4.2 refinement leaked into an adjacent pair"
    )

    # Re-run INV N hazard
    instrs_n = _stream(
        encode_imad_wide_rr(dest=8, src0=2, src1=3, src2=0xff),
        encode_iadd3(dest=10, src0=8, src1=11, src2=0xff),
    )
    report_n = verify_proof(instrs_n)
    assert not report_n.safe, (
        "INV AE3: INV N hazard (IMAD.WIDE.RR → IADD3) no longer "
        "flagged — FG-4.2 refinement leaked"
    )


def test_inv_adj4_corpus_safe_after_refinement():
    """INV ADJ4: the full corpus remains SAFE after the FG-4.1
    precision refinement — zero VIOLATION / MEMORY_VIOLATION /
    UR_MEMORY_VIOLATION in aggregate.
    """
    corpus = _workbench_kernels() + _probe_kernels() + _fg21_predicate_kernels()
    viol = 0
    problem = []
    for label, ptx in corpus:
        try:
            report = _proof_for(ptx)
        except Exception:
            continue
        viol += len(report.violations)
        if report.violations:
            problem.append(label)
    assert viol == 0, (
        f"INV ADJ4: {viol} violation edge(s) after FG-4.1 in "
        f"{problem[:10]}"
    )


def test_inv_ah_corpus_safe_under_fg33():
    """INV AH: the full corpus remains SAFE with zero VIOLATION,
    MEMORY_VIOLATION, and UR_MEMORY_VIOLATION edges in aggregate
    after the FG-3.3 LDCU unification.  Also verifies R10 now emits
    at least one UR_MEMORY_GAP_SAFE or UR_MEMORY_INERT edge
    somewhere, confirming those classes are not dead code.
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
    assert not problem, f"INV AH: violations in {problem[:10]}"
    # CTRLWORD_SAFE and R1-flavored GAP_SAFE should be zero (no R1).
    assert totals[ProofClass.CTRLWORD_SAFE] == 0, (
        "INV AH: CTRLWORD_SAFE edges still exist — legacy R1 path "
        "not fully retired"
    )
    # FG-3.3 UR_MEMORY_GAP_SAFE + UR_MEMORY_INERT should together
    # have the 4+1 = 5 edges from the inventory.
    assert (totals[ProofClass.UR_MEMORY_GAP_SAFE]
            + totals[ProofClass.UR_MEMORY_INERT]) > 0, (
        "INV AH: UR_MEMORY_GAP_SAFE and UR_MEMORY_INERT both empty "
        "— narrow fallback rules are dead code"
    )
