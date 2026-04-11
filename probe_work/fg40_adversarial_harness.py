"""FG-4.0 — Adversarial validation harness for the proof model.

This harness generates adversarial PTX kernels targeted at specific
weak points in the proof engine and classifies each case across three
axes:

    1. OpenPTXas assembly outcome
    2. PTXAS assembly outcome (ground-truth reference)
    3. Proof engine verdict (SAFE / VIOLATION)

Each generated kernel produces a `HarnessResult` whose `classification`
is one of:

    MODEL_CONFIRMED      — both OURS and PTXAS cubins are classified
                           SAFE by the proof engine.  This is the
                           expected case for the first adversarial
                           pass.
    MODEL_FALSE_POSITIVE — the PTXAS cubin is classified VIOLATION
                           by the proof engine.  Since PTXAS is
                           treated as the production reference, a
                           VIOLATION on PTXAS output means the
                           proof model is too strict — PTXAS would
                           not emit it if the hardware didn't accept
                           it.  This is the primary adversarial
                           signal detectable without GPU runtime.
    MODEL_FALSE_NEGATIVE — NOT DETECTABLE without GPU runtime.  A
                           false negative would require catching a
                           case where the proof says SAFE but the
                           kernel actually misbehaves at runtime.
                           This harness cannot run kernels, so it
                           cannot surface this class.  Documented
                           here for completeness; future runs that
                           add GPU execution can fill in the gap.
    NEW_BOUNDARY_FOUND   — any of:
                             * proof engine raised an exception
                             * proof engine flagged UNKNOWN wdep
                             * OURS assembler refused source PTXAS
                               accepted
                           These exercise corners the proof model
                           or OURS compiler explicitly declares
                           out-of-scope.  Expected during adversarial
                           expansion; must be logged individually.
    GENERATOR_INVALID    — PTXAS refused the generated PTX; harness
                           bug, not a model finding.

Family generators live in ADVERSARIAL_FAMILIES and are deterministic:
each generator takes a recipe dict and returns `(kernel_id, ptx)`.
Re-running the harness with the same recipe yields the same kernel.
"""
from __future__ import annotations

import json
import struct
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.bench_util import compile_openptxas, compile_ptxas, analyze_cubin
from sass.pipeline import SassInstr
from sass.schedule import verify_proof, ProofClass, ProofReport


# ---------------------------------------------------------------------------
# Classification enum (string values so JSON stays readable)
# ---------------------------------------------------------------------------

MODEL_CONFIRMED       = "MODEL_CONFIRMED"
MODEL_FALSE_POSITIVE  = "MODEL_FALSE_POSITIVE"
MODEL_FALSE_NEGATIVE  = "MODEL_FALSE_NEGATIVE"
NEW_BOUNDARY_FOUND    = "NEW_BOUNDARY_FOUND"
GENERATOR_INVALID     = "GENERATOR_INVALID"


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class HarnessResult:
    """One entry in the adversarial corpus report."""
    kernel_id: str
    family: str
    recipe: dict
    ptx: str
    classification: str
    proof_verdict: str
    proof_summary: str
    ours_ok: bool
    ours_error: str
    ours_stats: dict
    ptxas_ok: bool
    ptxas_error: str
    ptxas_stats: dict
    rationale: str


# ---------------------------------------------------------------------------
# F1 — ALU alias stress
# ---------------------------------------------------------------------------
#
# Generate minimal kernels where a single ALU op exhibits source/dest
# aliasing in various patterns.  The proof engine must NOT fabricate
# a hazard just because the same register appears as both producer
# and consumer in a single instruction — a single ALU op is atomic.

def gen_f1_alias(recipe: dict) -> tuple[str, str]:
    pattern = recipe["pattern"]
    op = recipe.get("op", "add.u32")
    kid = f"f1_{pattern}_{op.replace('.', '_')}"
    body = _F1_BODIES[pattern](op)
    ptx = _PTX_ENTRY.format(name=kid, body=body)
    return kid, ptx

_F1_BODIES: dict[str, Callable[[str], str]] = {
    # self-add: dest = src0 = src1 — no real hazard, just reuse.
    "self_src":
        lambda op: (f"    add.u32 %r1, %r0, %r0;\n"
                    f"    {op} %r2, %r1, %r1;\n"
                    f"    st.global.u32 [%rd0], %r2;\n"),
    # MAD accumulator alias: dest is the accumulator.
    "mad_acc":
        lambda op: (f"    add.u32 %r1, %r0, 1;\n"
                    f"    mad.lo.u32 %r1, %r0, %r0, %r1;\n"
                    f"    st.global.u32 [%rd0], %r1;\n"),
    # Three-register chain, reusing the first register as output.
    "overwrite":
        lambda op: (f"    add.u32 %r1, %r0, 1;\n"
                    f"    add.u32 %r2, %r1, 2;\n"
                    f"    add.u32 %r0, %r2, 3;\n"  # overwrites tid
                    f"    st.global.u32 [%rd0], %r0;\n"),
    # IADD.64 pair with partially-aliasing src/dst.
    "pair_lo":
        lambda op: (f"    cvt.u64.u32 %rd1, %r0;\n"
                    f"    add.u64 %rd1, %rd1, %rd1;\n"
                    f"    st.global.u64 [%rd0], %rd1;\n"),
    # SHF chain where the shift amount aliases the shifted value.
    "shf_alias":
        lambda op: (f"    shl.b32 %r1, %r0, 4;\n"
                    f"    and.b32 %r1, %r1, 15;\n"
                    f"    shr.u32 %r1, %r1, 1;\n"
                    f"    st.global.u32 [%rd0], %r1;\n"),
}

F1_RECIPES = [
    {"pattern": "self_src",  "op": "add.u32"},
    {"pattern": "self_src",  "op": "xor.b32"},
    {"pattern": "self_src",  "op": "and.b32"},
    {"pattern": "self_src",  "op": "or.b32"},
    {"pattern": "mad_acc",   "op": "mad.lo.u32"},
    {"pattern": "overwrite", "op": "add.u32"},
    {"pattern": "pair_lo",   "op": "add.u64"},
    {"pattern": "shf_alias", "op": "shl.b32"},
]


# ---------------------------------------------------------------------------
# F2 — UR/GPR boundary stress
# ---------------------------------------------------------------------------

def gen_f2_ur_gpr(recipe: dict) -> tuple[str, str]:
    pattern = recipe["pattern"]
    kid = f"f2_{pattern}"
    body = _F2_BODIES[pattern]()
    ptx = _PTX_ENTRY.format(name=kid, body=body)
    return kid, ptx

_F2_BODIES: dict[str, Callable[[], str]] = {
    "param_to_alu":
        lambda: ("    ld.param.u64 %rd1, [arg0];\n"
                 "    add.u64 %rd1, %rd1, 8;\n"
                 "    st.global.u32 [%rd0], %r0;\n"),
    "tid_to_mad":
        lambda: ("    mul.lo.u32 %r1, %r0, %r0;\n"
                 "    mad.lo.u32 %r1, %r0, %r0, %r1;\n"
                 "    st.global.u32 [%rd0], %r1;\n"),
    "const_to_ur":
        lambda: ("    ld.param.u32 %r1, [arg1];\n"
                 "    add.u32 %r1, %r1, 1;\n"
                 "    add.u32 %r1, %r1, %r0;\n"
                 "    st.global.u32 [%rd0], %r1;\n"),
    "multi_ur":
        lambda: ("    ld.param.u64 %rd1, [arg0];\n"
                 "    ld.param.u32 %r1, [arg1];\n"
                 "    cvt.u64.u32 %rd2, %r0;\n"
                 "    shl.b64 %rd2, %rd2, 2;\n"
                 "    add.u64 %rd2, %rd1, %rd2;\n"
                 "    st.global.u32 [%rd2], %r1;\n"),
    "param_chain":
        lambda: ("    ld.param.u64 %rd1, [arg0];\n"
                 "    ld.param.u64 %rd2, [arg0];\n"
                 "    add.u64 %rd1, %rd1, %rd2;\n"
                 "    st.global.u64 [%rd0], %rd1;\n"),
}

F2_RECIPES = [
    {"pattern": "param_to_alu"},
    {"pattern": "tid_to_mad"},
    {"pattern": "const_to_ur"},
    {"pattern": "multi_ur"},
    {"pattern": "param_chain"},
]


# ---------------------------------------------------------------------------
# F3 — Memory + ALU interleave
# ---------------------------------------------------------------------------

def gen_f3_mem_alu(recipe: dict) -> tuple[str, str]:
    pattern = recipe["pattern"]
    gap = recipe.get("gap", 0)
    kid = f"f3_{pattern}_gap{gap}"
    body = _F3_BODIES[pattern](gap)
    ptx = _PTX_ENTRY.format(name=kid, body=body)
    return kid, ptx

def _f3_global_load(gap: int) -> str:
    pad = "".join(f"    add.u32 %r{i+1}, %r{i+1}, 1;\n"
                  for i in range(gap))
    return ("    ld.param.u64 %rd1, [arg0];\n"
            "    cvt.u64.u32 %rd2, %r0;\n"
            "    shl.b64 %rd2, %rd2, 2;\n"
            "    add.u64 %rd2, %rd1, %rd2;\n"
            "    ld.global.u32 %r5, [%rd2];\n"
            f"{pad}"
            "    add.u32 %r5, %r5, 1;\n"
            "    st.global.u32 [%rd0], %r5;\n")

def _f3_shared_load(gap: int) -> str:
    # Shared-memory PTX syntax requires a .shared declaration at
    # module scope and uses [reg] addressing with the shared address.
    pad = "".join(f"    add.u32 %r{i+1}, %r{i+1}, 1;\n"
                  for i in range(gap))
    # We use cvta.shared to get a generic address to the shared buf
    # declared via the module-level .shared directive.
    return ("    mov.u32 %r1, smem_f3;\n"
            "    shl.b32 %r2, %r0, 2;\n"
            "    add.u32 %r2, %r1, %r2;\n"
            "    st.shared.u32 [%r2], %r0;\n"
            "    bar.sync 0;\n"
            "    ld.shared.u32 %r3, [%r2];\n"
            f"{pad}"
            "    add.u32 %r3, %r3, 1;\n"
            "    st.global.u32 [%rd0], %r3;\n")

def _f3_shadowed(gap: int) -> str:
    return ("    ld.param.u64 %rd1, [arg0];\n"
            "    cvt.u64.u32 %rd2, %r0;\n"
            "    shl.b64 %rd2, %rd2, 2;\n"
            "    add.u64 %rd2, %rd1, %rd2;\n"
            "    ld.global.u32 %r5, [%rd2];\n"
            "    mov.u32 %r5, 42;\n"  # shadow
            "    st.global.u32 [%rd0], %r5;\n")

_F3_BODIES = {
    "global":   _f3_global_load,
    "shared":   _f3_shared_load,
    "shadowed": _f3_shadowed,
}

F3_RECIPES = [
    {"pattern": "global",   "gap": 0},
    {"pattern": "global",   "gap": 1},
    {"pattern": "global",   "gap": 2},
    {"pattern": "global",   "gap": 3},
    {"pattern": "shared",   "gap": 0},
    {"pattern": "shared",   "gap": 1},
    {"pattern": "shared",   "gap": 2},
    {"pattern": "shadowed", "gap": 0},
]


# ---------------------------------------------------------------------------
# F4 — Predicate / control stress
# ---------------------------------------------------------------------------

def gen_f4_pred(recipe: dict) -> tuple[str, str]:
    pattern = recipe["pattern"]
    cmp = recipe.get("cmp", "lt")
    kid = f"f4_{pattern}_{cmp}"
    body = _F4_BODIES[pattern](cmp)
    ptx = _PTX_ENTRY.format(name=kid, body=body)
    return kid, ptx

_F4_BODIES: dict[str, Callable[[str], str]] = {
    "setp_bra":
        lambda cmp: (f"    setp.{cmp}.u32 %p0, %r0, 8;\n"
                     f"    @%p0 bra DONE;\n"
                     f"    add.u32 %r1, %r0, 1;\n"
                     f"    st.global.u32 [%rd0], %r1;\n"
                     f"DONE:\n"),
    "setp_exit":
        lambda cmp: (f"    setp.{cmp}.u32 %p0, %r0, 8;\n"
                     f"    @%p0 ret;\n"
                     f"    add.u32 %r1, %r0, 1;\n"
                     f"    st.global.u32 [%rd0], %r1;\n"),
    "setp_guarded":
        lambda cmp: (f"    setp.{cmp}.u32 %p0, %r0, 8;\n"
                     f"    add.u32 %r1, %r0, 1;\n"
                     f"    @%p0 st.global.u32 [%rd0], %r1;\n"),
    "chained_setp":
        lambda cmp: (f"    setp.{cmp}.u32 %p0, %r0, 8;\n"
                     f"    setp.{cmp}.u32 %p1, %r0, 16;\n"
                     f"    selp.b32 %r1, 1, 0, %p0;\n"
                     f"    selp.b32 %r2, 1, 0, %p1;\n"
                     f"    add.u32 %r1, %r1, %r2;\n"
                     f"    st.global.u32 [%rd0], %r1;\n"),
}

F4_RECIPES = [
    {"pattern": "setp_bra",     "cmp": "lt"},
    {"pattern": "setp_bra",     "cmp": "ge"},
    {"pattern": "setp_bra",     "cmp": "gt"},
    {"pattern": "setp_exit",    "cmp": "eq"},
    {"pattern": "setp_exit",    "cmp": "ne"},
    {"pattern": "setp_guarded", "cmp": "ne"},
    {"pattern": "setp_guarded", "cmp": "lt"},
    {"pattern": "chained_setp", "cmp": "lt"},
]


# ---------------------------------------------------------------------------
# F5 — Writer shadowing stress
# ---------------------------------------------------------------------------

def gen_f5_shadow(recipe: dict) -> tuple[str, str]:
    pattern = recipe["pattern"]
    kid = f"f5_{pattern}"
    body = _F5_BODIES[pattern]()
    ptx = _PTX_ENTRY.format(name=kid, body=body)
    return kid, ptx

_F5_BODIES: dict[str, Callable[[], str]] = {
    # producer → shadow → reader: the reader reads the shadow, not
    # the producer.  Proof engine must NOT flag the producer→reader
    # edge as a hazard because there is no such edge.
    "mov_shadow":
        lambda: ("    mul.lo.u32 %r1, %r0, %r0;\n"
                 "    mov.u32 %r1, 0;\n"
                 "    add.u32 %r2, %r1, 7;\n"
                 "    st.global.u32 [%rd0], %r2;\n"),
    # ld.global → shadow → reader: classic memory RAW masked by write.
    "ld_shadow":
        lambda: ("    ld.param.u64 %rd1, [arg0];\n"
                 "    ld.global.u32 %r1, [%rd1];\n"
                 "    mov.u32 %r1, 99;\n"
                 "    st.global.u32 [%rd0], %r1;\n"),
    # Partial pair shadow: producer writes pair, intermediate writes
    # only the low half, later reader reads low half.
    "pair_low_shadow":
        lambda: ("    ld.param.u64 %rd1, [arg0];\n"
                 "    ld.global.u64 %rd2, [%rd1];\n"
                 "    mov.b64 %rd2, 0;\n"
                 "    st.global.u64 [%rd0], %rd2;\n"),
    # Shadow after multiple intervening instructions — proof must
    # track shadow across a longer window, not just adjacent.
    "long_shadow":
        lambda: ("    ld.param.u64 %rd1, [arg0];\n"
                 "    ld.global.u32 %r1, [%rd1];\n"
                 "    add.u32 %r2, %r0, 1;\n"
                 "    add.u32 %r3, %r0, 2;\n"
                 "    mov.u32 %r1, 42;\n"
                 "    st.global.u32 [%rd0], %r1;\n"),
}

F5_RECIPES = [
    {"pattern": "mov_shadow"},
    {"pattern": "ld_shadow"},
    {"pattern": "pair_low_shadow"},
    {"pattern": "long_shadow"},
]


# ---------------------------------------------------------------------------
# F6 — Random-ish generator (seeded, constrained)
# ---------------------------------------------------------------------------
#
# Constrained random generation using a deterministic LCG seeded by
# the recipe.  Only generates sequences of whitelisted ALU ops, so
# every output is syntactically valid PTX.  Used to stress the proof
# engine with patterns I wouldn't hand-write.

def _lcg(seed: int):
    s = [seed & 0xFFFFFFFF]
    def nxt():
        s[0] = (s[0] * 1103515245 + 12345) & 0xFFFFFFFF
        return s[0]
    return nxt

def gen_f6_random(recipe: dict) -> tuple[str, str]:
    seed = recipe["seed"]
    length = recipe.get("length", 6)
    kid = f"f6_random_s{seed}_l{length}"
    rnd = _lcg(seed)
    # Allocate up to 8 registers (%r0..%r7); use tid.x at %r0
    # as starting live value.
    live = {0}
    lines = []
    ops = ["add", "sub", "mul", "shl", "shr", "xor", "and", "or"]
    for _ in range(length):
        opc = ops[rnd() % len(ops)]
        dst = 1 + (rnd() % 7)  # %r1..%r7
        src0 = rnd() % 8
        if src0 not in live:
            src0 = 0  # fall back to tid
        src1 = (rnd() % 8)
        if src1 not in live:
            src1 = 0
        if opc in ("shl", "shr"):
            ptx_op = f"{opc}.b32"
            lines.append(f"    {ptx_op} %r{dst}, %r{src0}, {1 + (rnd() % 7)};")
        else:
            ptx_op = {
                "add": "add.u32", "sub": "sub.u32", "mul": "mul.lo.u32",
                "xor": "xor.b32", "and": "and.b32", "or": "or.b32",
            }[opc]
            lines.append(f"    {ptx_op} %r{dst}, %r{src0}, %r{src1};")
        live.add(dst)
    # Pick a final register to store.
    final = max(live - {0}) if len(live) > 1 else 0
    body = "\n".join(lines) + f"\n    st.global.u32 [%rd0], %r{final};\n"
    ptx = _PTX_ENTRY.format(name=kid, body=body)
    return kid, ptx

F6_RECIPES = [
    {"seed": 0x1234, "length": 4},
    {"seed": 0x5678, "length": 6},
    {"seed": 0x9abc, "length": 8},
    {"seed": 0xdef0, "length": 10},
    {"seed": 0x1111, "length": 5},
    {"seed": 0x2222, "length": 7},
    {"seed": 0x3333, "length": 9},
    {"seed": 0x4444, "length": 12},
    {"seed": 0xa001, "length": 3},
    {"seed": 0xa002, "length": 4},
    {"seed": 0xa003, "length": 5},
    {"seed": 0xa004, "length": 6},
    {"seed": 0xb001, "length": 8},
    {"seed": 0xb002, "length": 10},
    {"seed": 0xb003, "length": 14},
    {"seed": 0xc001, "length": 16},
]


# ---------------------------------------------------------------------------
# PTX entry template
# ---------------------------------------------------------------------------

# Uses a consistent parameter signature so every generator can reuse
# %r0 = tid.x, %rd0 = output pointer, arg1 = u32 param.
_PTX_ENTRY = """
.version 8.8
.target sm_120
.address_size 64

.shared .align 4 .b8 smem_f3[256];

.visible .entry {name}(
    .param .u64 arg0,
    .param .u32 arg1)
{{
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .pred %p<4>;

    mov.u32 %r0, %tid.x;
    ld.param.u64 %rd0, [arg0];

{body}
    ret;
}}
"""


# Negative controls — intentionally unsafe patterns.  These kernels
# are used to verify the harness classifier is NOT biased toward
# "everything looks fine".  In the current model these are either
# syntactically invalid, produce proof violations, or match the
# known synthetic hazards from FG-2.5/FG-3.1.

def gen_neg_known_unsafe(recipe: dict) -> tuple[str, str]:
    """Known-unsafe patterns — deliberate negative controls.

    These come from the FG-2.5 / FG-3.1 regression tests as minimal
    reprouctions of hazards the proof engine is known to flag.  We
    emit them via inline-SASS if possible, otherwise via PTX that
    PTXAS would compile identically (with the hazard) if asked.

    For the harness we mostly emit plain PTX; PTXAS inserts its own
    scheduling so the hazard is typically absorbed.  The *PTX-level*
    classification mostly catches OURS bugs, not adversarial hazard
    discovery.  Additional SASS-level negative controls live in the
    FG-2.5 unit tests.
    """
    pattern = recipe["pattern"]
    kid = f"neg_{pattern}"
    body = _NEG_BODIES[pattern]()
    ptx = _PTX_ENTRY.format(name=kid, body=body)
    return kid, ptx

_NEG_BODIES: dict[str, Callable[[], str]] = {
    # Tight back-to-back IMAD dependency — the FG-1.14A pattern.
    # PTXAS will schedule around it; OURS should also schedule.
    "tight_imad":
        lambda: ("    ld.param.u32 %r1, [arg1];\n"
                 "    mad.lo.u32 %r2, %r1, %r0, %r1;\n"
                 "    mov.u32 %r3, %r2;\n"
                 "    st.global.u32 [%rd0], %r3;\n"),
    # IMAD.WIDE → IADD3 — same root cause family as INV N.
    "wide_imad_iadd":
        lambda: ("    ld.param.u32 %r1, [arg1];\n"
                 "    mul.wide.u32 %rd1, %r1, %r0;\n"
                 "    cvt.u32.u64 %r2, %rd1;\n"
                 "    add.u32 %r2, %r2, 1;\n"
                 "    st.global.u32 [%rd0], %r2;\n"),
}

NEG_RECIPES = [
    {"pattern": "tight_imad"},
    {"pattern": "wide_imad_iadd"},
]


# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------

ADVERSARIAL_FAMILIES = {
    "F1": (gen_f1_alias,    F1_RECIPES),
    "F2": (gen_f2_ur_gpr,   F2_RECIPES),
    "F3": (gen_f3_mem_alu,  F3_RECIPES),
    "F4": (gen_f4_pred,     F4_RECIPES),
    "F5": (gen_f5_shadow,   F5_RECIPES),
    "F6": (gen_f6_random,   F6_RECIPES),
    "NEG": (gen_neg_known_unsafe, NEG_RECIPES),
}


def enumerate_corpus() -> list[tuple[str, str, dict, str]]:
    """Yield (family, kernel_id, recipe, ptx) for every adversarial
    kernel in the registered families.  Deterministic — same ordering
    across runs.
    """
    out = []
    for fam, (gen, recipes) in ADVERSARIAL_FAMILIES.items():
        for recipe in recipes:
            kid, ptx = gen(dict(recipe))
            out.append((fam, kid, dict(recipe), ptx))
    return out


# ---------------------------------------------------------------------------
# Proof + assembly runners
# ---------------------------------------------------------------------------

def _iter_text(cubin):
    e_shoff = struct.unpack_from('<Q', cubin, 40)[0]
    e_shnum = struct.unpack_from('<H', cubin, 60)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 62)[0]
    stoff = struct.unpack_from('<Q', cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    for i in range(e_shnum):
        base = e_shoff + i * 64
        nm = struct.unpack_from('<I', cubin, base)[0]
        name_end = cubin.index(0, stoff + nm)
        name = cubin[stoff + nm:name_end]
        if name.startswith(b".text."):
            off = struct.unpack_from('<Q', cubin, base + 24)[0]
            sz = struct.unpack_from('<Q', cubin, base + 32)[0]
            yield name.decode()[len(".text."):], off, sz


def _proof_for_cubin(cubin) -> ProofReport:
    combined = ProofReport()
    for _sym, off, sz in _iter_text(cubin):
        instrs = [SassInstr(cubin[off + o:off + o + 16], "")
                  for o in range(0, sz, 16)
                  if off + o + 16 <= off + sz]
        try:
            r = verify_proof(instrs)
        except Exception:
            # Raise upstream so the harness can classify as
            # NEW_BOUNDARY_FOUND.
            raise
        combined.edges.extend(r.edges)
    return combined


def classify_kernel(family: str,
                    kernel_id: str,
                    recipe: dict,
                    ptx: str) -> HarnessResult:
    """Assemble `ptx` with both OURS and PTXAS, run the proof engine
    on BOTH cubins, and classify the outcome.

    Classification rules (documented in the module docstring):
      - PTXAS refused     → GENERATOR_INVALID
      - OURS refused      → NEW_BOUNDARY_FOUND
      - proof raised / UNKNOWN wdep on either cubin → NEW_BOUNDARY_FOUND
      - proof flags PTXAS cubin VIOLATION → MODEL_FALSE_POSITIVE
        (PTXAS is the production reference; if the model marks its
         output as unsafe, the model is too strict)
      - proof flags OURS cubin VIOLATION but PTXAS cubin SAFE →
        NEW_BOUNDARY_FOUND (OURS emitted a real hazard; separate
        pass needs to diagnose whether the hazard is real or a
        scoreboard bug)
      - both cubins SAFE → MODEL_CONFIRMED
    """
    ours_ok, ours_err, ours_stats = False, "", {}
    ptxas_ok, ptxas_err, ptxas_stats = False, "", {}
    ours_cubin = None
    ptxas_cubin = None

    try:
        ours_cubin, _ = compile_openptxas(ptx)
        ours_stats = analyze_cubin(ours_cubin, kernel_id)
        ours_ok = True
    except Exception as exc:
        ours_err = f"{type(exc).__name__}: {exc}"[:200]

    try:
        ptxas_cubin, _ = compile_ptxas(ptx)
        ptxas_stats = analyze_cubin(ptxas_cubin, kernel_id)
        ptxas_ok = True
    except Exception as exc:
        ptxas_err = f"{type(exc).__name__}: {exc}"[:200]

    # If PTXAS rejected the PTX, the generator produced invalid code.
    if not ptxas_ok:
        return HarnessResult(
            kernel_id=kernel_id, family=family, recipe=recipe, ptx=ptx,
            classification=GENERATOR_INVALID,
            proof_verdict="N/A",
            proof_summary="ptxas rejected source",
            ours_ok=ours_ok, ours_error=ours_err, ours_stats=ours_stats,
            ptxas_ok=ptxas_ok, ptxas_error=ptxas_err, ptxas_stats=ptxas_stats,
            rationale="PTXAS refused the generated PTX — generator bug or invalid pattern",
        )

    # If OURS rejected while PTXAS accepted, that's a boundary.
    if not ours_ok:
        return HarnessResult(
            kernel_id=kernel_id, family=family, recipe=recipe, ptx=ptx,
            classification=NEW_BOUNDARY_FOUND,
            proof_verdict="N/A",
            proof_summary="OURS refused source PTXAS accepted",
            ours_ok=ours_ok, ours_error=ours_err, ours_stats=ours_stats,
            ptxas_ok=ptxas_ok, ptxas_error=ptxas_err, ptxas_stats=ptxas_stats,
            rationale=f"OURS assembler failure while PTXAS succeeded: {ours_err}",
        )

    # Run proof engine on BOTH cubins.
    try:
        ours_report = _proof_for_cubin(ours_cubin)
        ptxas_report = _proof_for_cubin(ptxas_cubin)
    except Exception:
        tb = traceback.format_exc().splitlines()[-3:]
        return HarnessResult(
            kernel_id=kernel_id, family=family, recipe=recipe, ptx=ptx,
            classification=NEW_BOUNDARY_FOUND,
            proof_verdict="EXCEPTION",
            proof_summary="verify_proof raised",
            ours_ok=True, ours_error="", ours_stats=ours_stats,
            ptxas_ok=True, ptxas_error="", ptxas_stats=ptxas_stats,
            rationale="proof engine raised: " + " | ".join(tb),
        )

    # Detect UNKNOWN wdep VIOLATION edges on either cubin.
    def _unknown(r: ProofReport) -> bool:
        return any("UNKNOWN" in e.rationale for e in r.violations)

    if _unknown(ours_report) or _unknown(ptxas_report):
        side = "OURS" if _unknown(ours_report) else "PTXAS"
        return HarnessResult(
            kernel_id=kernel_id, family=family, recipe=recipe, ptx=ptx,
            classification=NEW_BOUNDARY_FOUND,
            proof_verdict=f"{side}_UNKNOWN_WDEP",
            proof_summary=f"ours={ours_report.summary_line()} | ptxas={ptxas_report.summary_line()}",
            ours_ok=True, ours_error="", ours_stats=ours_stats,
            ptxas_ok=True, ptxas_error="", ptxas_stats=ptxas_stats,
            rationale=f"UNKNOWN wdep flagged on {side} cubin — unmodeled scoreboard slot",
        )

    ours_safe = ours_report.safe
    ptxas_safe = ptxas_report.safe
    verdict = f"OURS={'SAFE' if ours_safe else 'UNSAFE'}/PTXAS={'SAFE' if ptxas_safe else 'UNSAFE'}"
    summary = f"ours={ours_report.summary_line()} | ptxas={ptxas_report.summary_line()}"

    if not ptxas_safe:
        cls = MODEL_FALSE_POSITIVE
        v = ptxas_report.violations[0]
        rat = (f"proof flagged PTXAS cubin UNSAFE "
               f"({len(ptxas_report.violations)} violation(s)); "
               f"first: {v.rationale[:120]}")
    elif not ours_safe:
        cls = NEW_BOUNDARY_FOUND
        v = ours_report.violations[0]
        rat = (f"OURS cubin UNSAFE while PTXAS SAFE; "
               f"first violation: {v.rationale[:120]}")
    else:
        cls = MODEL_CONFIRMED
        rat = "proof SAFE on both OURS and PTXAS cubins"

    return HarnessResult(
        kernel_id=kernel_id, family=family, recipe=recipe, ptx=ptx,
        classification=cls,
        proof_verdict=verdict,
        proof_summary=summary,
        ours_ok=True, ours_error="", ours_stats=ours_stats,
        ptxas_ok=True, ptxas_error="", ptxas_stats=ptxas_stats,
        rationale=rat,
    )


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------

def run_harness() -> list[HarnessResult]:
    results = []
    for family, kid, recipe, ptx in enumerate_corpus():
        r = classify_kernel(family, kid, recipe, ptx)
        results.append(r)
    return results


def summarize(results: list[HarnessResult]) -> dict:
    by_class = {k: 0 for k in (
        MODEL_CONFIRMED, MODEL_FALSE_POSITIVE, MODEL_FALSE_NEGATIVE,
        NEW_BOUNDARY_FOUND, GENERATOR_INVALID,
    )}
    by_family = {}
    for r in results:
        by_class[r.classification] = by_class.get(r.classification, 0) + 1
        fam = by_family.setdefault(r.family, {k: 0 for k in by_class})
        fam[r.classification] += 1
    return {
        "total": len(results),
        "by_class": by_class,
        "by_family": by_family,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    results = run_harness()
    summary = summarize(results)

    # Print summary table.
    print("FG-4.0 adversarial validation harness")
    print("=" * 78)
    print(f"total kernels    : {summary['total']}")
    for cls, n in summary["by_class"].items():
        print(f"  {cls:<22s}: {n}")
    print()
    print("By family:")
    for fam, counts in sorted(summary["by_family"].items()):
        row = "  ".join(f"{k}={v}" for k, v in counts.items() if v)
        print(f"  {fam}: {row}")
    print()

    # Escalate any MODEL_FALSE_NEGATIVE immediately.
    # NOTE: this class is not currently reachable without GPU runtime
    # — documented in the module docstring.  The check is kept for
    # forward-compatibility with a future harness that executes
    # kernels.
    negs = [r for r in results if r.classification == MODEL_FALSE_NEGATIVE]
    if negs:
        print("!!! MODEL_FALSE_NEGATIVE detected — first case:")
        r = negs[0]
        print(f"  kernel: {r.kernel_id}  family: {r.family}")
        print(f"  rationale: {r.rationale}")
        print(f"  PTX:\n{r.ptx}")

    pos = [r for r in results if r.classification == MODEL_FALSE_POSITIVE]
    if pos:
        print(f"MODEL_FALSE_POSITIVE: {len(pos)} case(s)")
        for r in pos[:5]:
            print(f"  {r.kernel_id}: {r.rationale}")
        print()

    boundaries = [r for r in results if r.classification == NEW_BOUNDARY_FOUND]
    if boundaries:
        print(f"NEW_BOUNDARY_FOUND: {len(boundaries)} case(s)")
        for r in boundaries[:10]:
            print(f"  {r.kernel_id}: {r.rationale}")
        print()

    invalid = [r for r in results if r.classification == GENERATOR_INVALID]
    if invalid:
        print(f"GENERATOR_INVALID: {len(invalid)} case(s)")
        for r in invalid[:5]:
            print(f"  {r.kernel_id}: {r.ptxas_error[:100]}")
        print()

    # Write JSON report.
    out_path = ROOT / "probe_work" / "fg40_adversarial_report.json"
    payload = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[OK] wrote {out_path.relative_to(ROOT)}")

    return 1 if negs else 0


if __name__ == "__main__":
    raise SystemExit(main())
