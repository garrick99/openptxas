"""FG-4.0 — Harness tests.

Focused and fast tests for the adversarial validation harness.
These verify:

  1. Determinism — the same recipe yields the same kernel text
     across multiple calls.
  2. Classification logic — a known-safe kernel classifies as
     MODEL_CONFIRMED or MODEL_FALSE_POSITIVE (i.e. not INVALID /
     FALSE_NEGATIVE), and a kernel that PTXAS refuses is flagged
     as GENERATOR_INVALID.
  3. Corpus enumeration returns a non-empty deterministic list.

These tests intentionally do NOT run the full 51-kernel harness —
that corpus lives in probe_work and is exercised by the standalone
script.  Pytest keeps things small.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe_work.fg40_adversarial_harness import (
    ADVERSARIAL_FAMILIES,
    HarnessResult,
    MODEL_CONFIRMED,
    MODEL_FALSE_POSITIVE,
    MODEL_FALSE_NEGATIVE,
    NEW_BOUNDARY_FOUND,
    GENERATOR_INVALID,
    classify_kernel,
    enumerate_corpus,
    gen_f1_alias,
    gen_f6_random,
    summarize,
)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_f1_generator():
    """Same F1 recipe → same (kernel_id, ptx) on repeated calls."""
    recipe = {"pattern": "self_src", "op": "add.u32"}
    k1, p1 = gen_f1_alias(dict(recipe))
    k2, p2 = gen_f1_alias(dict(recipe))
    assert k1 == k2
    assert p1 == p2


def test_determinism_f6_random_generator():
    """F6 generator is a seeded LCG; same seed → same kernel."""
    recipe = {"seed": 0xDEADBEEF, "length": 6}
    k1, p1 = gen_f6_random(dict(recipe))
    k2, p2 = gen_f6_random(dict(recipe))
    assert k1 == k2
    assert p1 == p2
    # Different seed → different kernel (in general — with tiny
    # lengths there's a slim probability of collision, so use a
    # longer length).
    k3, p3 = gen_f6_random({"seed": 0xCAFEBABE, "length": 8})
    assert p3 != p1


def test_determinism_enumerate_corpus_stable():
    """enumerate_corpus() produces the same list across calls."""
    a = enumerate_corpus()
    b = enumerate_corpus()
    assert len(a) == len(b)
    for (fa, ka, _, pa), (fb, kb, _, pb) in zip(a, b):
        assert fa == fb and ka == kb and pa == pb


# ---------------------------------------------------------------------------
# Classification sanity
# ---------------------------------------------------------------------------

def test_classify_known_safe_kernel():
    """A minimal valid kernel should classify as MODEL_CONFIRMED or
    MODEL_FALSE_POSITIVE — never INVALID, NEW_BOUNDARY, or FALSE_NEGATIVE.
    """
    ptx = """
.version 8.8
.target sm_120
.address_size 64

.visible .entry k_safe(.param .u64 arg0, .param .u32 arg1) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<3>;
    ld.param.u64 %rd0, [arg0];
    ld.param.u32 %r0, [arg1];
    add.u32 %r1, %r0, 1;
    st.global.u32 [%rd0], %r1;
    ret;
}
"""
    r = classify_kernel("TEST", "k_safe", {}, ptx)
    assert r.classification in (MODEL_CONFIRMED, MODEL_FALSE_POSITIVE), (
        f"known-safe kernel classified as {r.classification}: {r.rationale}"
    )
    assert r.ours_ok
    assert r.ptxas_ok


def test_classify_ptxas_rejects_invalid():
    """A kernel PTXAS refuses (deliberately malformed) must
    classify as GENERATOR_INVALID.
    """
    ptx = """
.version 8.8
.target sm_120
.address_size 64

.visible .entry k_bogus(.param .u64 arg0) {
    this_is_not_valid_ptx %r0;
    ret;
}
"""
    r = classify_kernel("TEST", "k_bogus", {}, ptx)
    assert r.classification == GENERATOR_INVALID
    assert not r.ptxas_ok


def test_classify_result_has_all_fields():
    """HarnessResult should carry all fields set even when
    classification short-circuits."""
    ptx = "not valid ptx at all"
    r = classify_kernel("TEST", "k_garbage", {}, ptx)
    assert r.classification == GENERATOR_INVALID
    assert r.kernel_id == "k_garbage"
    assert r.family == "TEST"
    # Must still have string fields even if empty
    assert r.proof_verdict is not None
    assert r.rationale


# ---------------------------------------------------------------------------
# Enumeration sanity
# ---------------------------------------------------------------------------

def test_enumerate_corpus_non_empty():
    c = enumerate_corpus()
    assert len(c) >= 50, (
        f"corpus has {len(c)} kernels — FG-4.0 requires ≥ 50"
    )
    # Every family registered.
    fams = {fam for fam, _, _, _ in c}
    assert "F1" in fams and "F2" in fams and "F3" in fams
    assert "F4" in fams and "F5" in fams and "F6" in fams
    assert "NEG" in fams


def test_enumerate_corpus_unique_kernel_ids():
    c = enumerate_corpus()
    kids = [k for _, k, _, _ in c]
    assert len(kids) == len(set(kids)), (
        "duplicate kernel_ids in corpus — generator recipes collide"
    )


# ---------------------------------------------------------------------------
# Summary structure
# ---------------------------------------------------------------------------

def test_summarize_structure():
    fake_results = [
        HarnessResult(
            kernel_id="k1", family="F1", recipe={}, ptx="",
            classification=MODEL_CONFIRMED, proof_verdict="SAFE",
            proof_summary="", ours_ok=True, ours_error="",
            ours_stats={}, ptxas_ok=True, ptxas_error="",
            ptxas_stats={}, rationale="",
        ),
        HarnessResult(
            kernel_id="k2", family="F1", recipe={}, ptx="",
            classification=MODEL_FALSE_POSITIVE, proof_verdict="UNSAFE",
            proof_summary="", ours_ok=True, ours_error="",
            ours_stats={}, ptxas_ok=True, ptxas_error="",
            ptxas_stats={}, rationale="",
        ),
    ]
    s = summarize(fake_results)
    assert s["total"] == 2
    assert s["by_class"][MODEL_CONFIRMED] == 1
    assert s["by_class"][MODEL_FALSE_POSITIVE] == 1
    assert s["by_family"]["F1"][MODEL_CONFIRMED] == 1
    assert s["by_family"]["F1"][MODEL_FALSE_POSITIVE] == 1
