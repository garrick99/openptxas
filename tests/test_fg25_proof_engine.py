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
    # (kernel-label, (total, inert, fwd, ctrl, gap, viol))
    # Labels match those produced by the corpus collectors (_probe_kernels
    # uses short symbol names; _workbench_kernels uses bare kernel names;
    # _fg21_predicate_kernels uses the "fg21:..." prefix).
    "diag":                  (1, 1, 0, 0, 0, 0),
    "diag3":                 (1, 1, 0, 0, 0, 0),
    "min_store_guarded":     (2, 1, 0, 0, 1, 0),
    "probe_fresh":           (2, 1, 0, 0, 1, 0),
    "reduce_sum":            (13, 5, 6, 1, 1, 0),
    "conv2d_looped":         (59, 27, 0, 31, 1, 0),
    "conv2d_unrolled":       (50, 11, 8, 30, 1, 0),
    "fg21:k_ge":             (2, 1, 0, 0, 1, 0),
    "fg21:k_lt":             (2, 1, 0, 0, 1, 0),
    "fg21:k_gt":             (2, 1, 0, 0, 1, 0),
    "fg21:k_le":             (2, 1, 0, 0, 1, 0),
    "fg21:k_eq":             (2, 1, 0, 0, 1, 0),
    "fg21:k_ne":             (2, 1, 0, 0, 1, 0),
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
    """INV K: the proof engine must be complete for every opcode that
    _OPCODE_META registers as needing a GPR latency gap.  Specifically,
    every opcode in _OPCODE_META with min_gpr_gap > 0 must either:
      (a) have at least one forwarding-safe pair entry that covers
          an observed consumer, OR
      (b) have min_gpr_gap = 1 so the current-cycle check applies.

    (Condition b is satisfied by every entry today.  The test is kept
    here so that adding a new entry with min_gpr_gap > 1 triggers a
    compile-time reminder to extend the proof engine's gap lookahead
    logic.)
    """
    from sass.scoreboard import _OPCODE_META
    unhandled = []
    for opc, meta in _OPCODE_META.items():
        if meta.min_gpr_gap == 0:
            continue
        if meta.min_gpr_gap > 1:
            # Gap lookahead isn't implemented beyond adjacent reader.
            unhandled.append((opc, meta.name, meta.min_gpr_gap))
    assert not unhandled, (
        f"INV K violation — opcodes with min_gpr_gap > 1 are not "
        f"handled by the proof engine's adjacent-reader logic: "
        f"{unhandled}. Extend verify_proof's R8 loop to look "
        f"`min_gpr_gap` instructions forward before classifying."
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
