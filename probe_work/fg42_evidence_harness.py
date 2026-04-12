"""FG-4.2 — Hardware evidence harness for remaining forwarding candidates.

The FG-4.0 adversarial pass left 10 MODEL_FALSE_POSITIVE cases.  FG-4.1
closed 5 of them (HFMA2 zero-init) with a semantic fastpath rule backed
by encoder ground truth.  This harness uses GPU runtime + ctrl-word
analysis to decide the remaining 10, grouped into three clusters:

    Cluster A  ALU writer → STG reads producer's dest as byte-4 data
               (IMAD.32 / IADD.64 / SHF → STG at gap=0)
    Cluster B  IADD.64 → IADD.64 self-chain
    Cluster C  LEA → IADD3X chain

For each cluster we construct a MINIMAL kernel whose observable output
depends on the producer-consumer forwarding working correctly.  PTXAS
compiles each kernel at gap=0 (the production pattern).  OURS compiles
the same PTX, inserting NOPs per its current proof model (the
conservative pattern).  Both cubins are launched on GPU with fixed
inputs and the outputs are compared against a CPU-computed expected
value.

Classification per cluster:

    FORWARDING_CONFIRMED — PTXAS gap=0 output == expected value AND
                           ctrl-word analysis shows no obvious
                           scoreboard coverage.  The hardware
                           accepts the pattern, so the current
                           proof engine is too strict.

    LATENCY_REQUIRED     — PTXAS gap=0 output != expected value
                           (would indicate a ptxas bug; unlikely
                           but listed for completeness).

    INCONCLUSIVE         — runtime did not execute, or the kernel
                           design cannot distinguish the cases.
                           Must NOT be promoted to SAFE.

Output:
  probe_work/fg42_evidence_report.json with one entry per
  cluster pattern listing kernels tested, ctrl-word details,
  runtime outputs, and the final classification.
"""
from __future__ import annotations

import ctypes
import json
import struct
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.bench_util import (
    CUDAContext, compile_openptxas, compile_ptxas, analyze_cubin,
)
from sass.scoreboard import (
    _get_opcode, _get_wdep, _get_rbar, _get_src_regs, _get_dest_regs,
    _WDEP_TO_RBAR_MASK,
)


# ---------------------------------------------------------------------------
# Classification tokens
# ---------------------------------------------------------------------------

FORWARDING_CONFIRMED = "FORWARDING_CONFIRMED"
LATENCY_REQUIRED     = "LATENCY_REQUIRED"
INCONCLUSIVE         = "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Minimal cluster kernels
# ---------------------------------------------------------------------------
#
# Each kernel is a tiny PTX program with a provable expected output.
# We pass input values via .param so the compiler cannot constant-fold
# the computation away.  The output is stored to a u32 * out_ptr.
#
# If forwarding actually works at gap=0, the GPU run produces the
# expected value.  If the producer's result hasn't committed by the
# time the consumer reads, we get garbage (or the old value).

# -- Cluster A.1 — IMAD.32 → STG data (input * input + constant) ---------
CLUSTER_A_IMAD_STG = {
    "id": "A_imad_stg",
    "cluster": "A",
    "description": "IMAD.32 writes R5, STG reads R5 at data byte 4",
    "ptx": """
.version 8.8
.target sm_120
.address_size 64

.visible .entry A_imad_stg(
    .param .u64 arg0,
    .param .u32 arg1,
    .param .u32 arg2)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd0, [arg0];
    ld.param.u32 %r0, [arg1];
    ld.param.u32 %r1, [arg2];
    mad.lo.u32 %r2, %r0, %r1, 7;
    st.global.u32 [%rd0], %r2;
    ret;
}
""",
    "inputs": {"arg1": 13, "arg2": 17},
    "expected": lambda i: (i["arg1"] * i["arg2"] + 7) & 0xFFFFFFFF,
}

# -- Cluster A.2 — SHF → STG ---------------------------------------------
CLUSTER_A_SHF_STG = {
    "id": "A_shf_stg",
    "cluster": "A",
    "description": "SHF writes R5, STG reads R5 at data byte 4",
    "ptx": """
.version 8.8
.target sm_120
.address_size 64

.visible .entry A_shf_stg(
    .param .u64 arg0,
    .param .u32 arg1)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<3>;

    ld.param.u64 %rd0, [arg0];
    ld.param.u32 %r0, [arg1];
    shl.b32 %r1, %r0, 3;
    st.global.u32 [%rd0], %r1;
    ret;
}
""",
    "inputs": {"arg1": 0xABCD},
    "expected": lambda i: (i["arg1"] << 3) & 0xFFFFFFFF,
}

# -- Cluster A.3 — IADD → STG --------------------------------------------
CLUSTER_A_IADD_STG = {
    "id": "A_iadd_stg",
    "cluster": "A",
    "description": "IADD writes R5, STG reads R5",
    "ptx": """
.version 8.8
.target sm_120
.address_size 64

.visible .entry A_iadd_stg(
    .param .u64 arg0,
    .param .u32 arg1,
    .param .u32 arg2)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<3>;

    ld.param.u64 %rd0, [arg0];
    ld.param.u32 %r0, [arg1];
    ld.param.u32 %r1, [arg2];
    add.u32 %r2, %r0, %r1;
    st.global.u32 [%rd0], %r2;
    ret;
}
""",
    "inputs": {"arg1": 100, "arg2": 200},
    "expected": lambda i: (i["arg1"] + i["arg2"]) & 0xFFFFFFFF,
}

# -- Cluster B.1 — IADD.64 → IADD.64 self-chain --------------------------
#
# x + x followed by (x+x) + x via explicit 64-bit add chain.  The
# consumer reads the producer's dest pair as src0 of the next IADD.64.
CLUSTER_B_IADD64_CHAIN = {
    "id": "B_iadd64_chain",
    "cluster": "B",
    "description": "IADD.64 chain: (x+x)+x where src0 = producer's dest",
    "ptx": """
.version 8.8
.target sm_120
.address_size 64

.visible .entry B_iadd64_chain(
    .param .u64 arg0,
    .param .u64 arg1)
{
    .reg .u64 %rd<6>;

    ld.param.u64 %rd0, [arg0];
    ld.param.u64 %rd1, [arg1];
    add.u64 %rd2, %rd1, %rd1;
    add.u64 %rd3, %rd2, %rd1;
    st.global.u64 [%rd0], %rd3;
    ret;
}
""",
    "inputs": {"arg1": 0x123456789ABCDEF0},
    "expected": lambda i: (3 * i["arg1"]) & 0xFFFFFFFFFFFFFFFF,
}

# -- Cluster C.1 — LEA → IADD3X chain ------------------------------------
#
# This cluster is harder to reduce to a pure PTX pattern because LEA
# and IADD3X are SASS-level instructions not always reachable from PTX
# directly.  We approximate by issuing an address computation that
# ptxas lowers to LEA + IADD3X (tid-based array indexing into a
# 64-bit pointer).
CLUSTER_C_LEA_IADD3X = {
    "id": "C_lea_iadd3x",
    "cluster": "C",
    "description": "LEA → IADD3X address calculation chain",
    "ptx": """
.version 8.8
.target sm_120
.address_size 64

.visible .entry C_lea_iadd3x(
    .param .u64 arg0,
    .param .u32 arg1)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<6>;

    ld.param.u64 %rd0, [arg0];
    ld.param.u32 %r0, [arg1];
    mul.wide.u32 %rd1, %r0, 4;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r1, 42;
    st.global.u32 [%rd2], %r1;
    ret;
}
""",
    "inputs": {"arg1": 3},
    "out_slot_index": 3,  # we store 42 into out[3]
    "expected": lambda i: 42,
}


ALL_CLUSTER_KERNELS = [
    CLUSTER_A_IMAD_STG,
    CLUSTER_A_SHF_STG,
    CLUSTER_A_IADD_STG,
    CLUSTER_B_IADD64_CHAIN,
    CLUSTER_C_LEA_IADD3X,
]


# ---------------------------------------------------------------------------
# Dedicated probes with non-trivial expected outputs for the remaining
# FG-4.0 FP pairs.  Each probe is designed so that the STORED value
# depends on the producer → consumer edge functioning correctly.
# ---------------------------------------------------------------------------

# IMAD.32 → STG with arithmetic that exercises the STG data operand
PROBE_IMAD_STG_ND = {
    "id": "P_imad_stg_nd",
    "cluster": "A",
    "description": "IMAD.32 → STG with a non-trivial product (arg1*3 + 7)",
    "ptx": """
.version 8.8
.target sm_120
.address_size 64

.visible .entry P_imad_stg_nd(
    .param .u64 arg0,
    .param .u32 arg1)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<3>;

    ld.param.u64 %rd0, [arg0];
    ld.param.u32 %r0, [arg1];
    mad.lo.u32 %r1, %r0, 3, 7;
    st.global.u32 [%rd0], %r1;
    ret;
}
""",
    "inputs": {"arg1": 1001},
    "expected": lambda i: (i["arg1"] * 3 + 7) & 0xFFFFFFFF,
}

# LEA / IADD3X via array indexing: tid-scaled pointer arithmetic.
PROBE_LEA_INDEX = {
    "id": "P_lea_index",
    "cluster": "C",
    "description": "LEA + IADD3X-style pointer arithmetic out[arg1] = arg1*arg1",
    "ptx": """
.version 8.8
.target sm_120
.address_size 64

.visible .entry P_lea_index(
    .param .u64 arg0,
    .param .u32 arg1)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<6>;

    ld.param.u64 %rd0, [arg0];
    ld.param.u32 %r0, [arg1];
    mul.wide.u32 %rd1, %r0, 4;
    add.u64 %rd2, %rd0, %rd1;
    mul.lo.u32 %r1, %r0, %r0;
    st.global.u32 [%rd2], %r1;
    ret;
}
""",
    "inputs": {"arg1": 5},
    "out_slot_index": 5,
    "expected": lambda i: (i["arg1"] * i["arg1"]) & 0xFFFFFFFF,
}

# SHF → IADD.64: shift result fed into 64-bit add.
PROBE_SHF_IADD64 = {
    "id": "P_shf_iadd64",
    "cluster": "B",
    "description": "SHF → IADD.64 chain ((arg1 << 3) + const)",
    "ptx": """
.version 8.8
.target sm_120
.address_size 64

.visible .entry P_shf_iadd64(
    .param .u64 arg0,
    .param .u32 arg1)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd0, [arg0];
    ld.param.u32 %r0, [arg1];
    shl.b32 %r1, %r0, 3;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd1, 19;
    st.global.u64 [%rd0], %rd2;
    ret;
}
""",
    "inputs": {"arg1": 42},
    "expected": lambda i: ((i["arg1"] << 3) + 19) & 0xFFFFFFFFFFFFFFFF,
}


DEDICATED_PROBES = [
    PROBE_IMAD_STG_ND,
    PROBE_LEA_INDEX,
    PROBE_SHF_IADD64,
]


# ---------------------------------------------------------------------------
# Direct FG-4.0 kernel replay: run each false-positive kernel on GPU
# using BOTH OURS and PTXAS cubins, compare outputs.  If both produce
# the same output for the same input, the PTXAS gap=0 schedule is
# functionally equivalent to the OURS conservative (gapped) schedule
# — strong evidence the forwarding pattern is hardware-safe.
# ---------------------------------------------------------------------------


def replay_fg40_fp_kernels(ctx: CUDAContext | None) -> list[dict]:
    """Launch every FG-4.0 MODEL_FALSE_POSITIVE kernel on GPU using
    OURS and PTXAS cubins, compare outputs.  Returns a list of
    result dicts (opc_pair, ours_out, ptxas_out, match, kernel_id).
    """
    from probe_work.fg40_adversarial_harness import enumerate_corpus
    from sass.pipeline import SassInstr
    from sass.schedule import verify_proof

    results: list[dict] = []
    for fam, kid, rec, ptx in enumerate_corpus():
        # Compile and run proof engine on PTXAS cubin to find the
        # violation pair, if any.
        try:
            pcubin, _ = compile_ptxas(ptx)
        except Exception:
            continue
        for _sym, off, sz in _iter_text(pcubin):
            instrs = [SassInstr(pcubin[off+k*16:off+k*16+16], "")
                      for k in range(sz // 16)]
            report = verify_proof(instrs)
            if not report.violations:
                break
            v = report.violations[0]
            pair = (v.writer_opc, v.reader_opc)
            row = {
                "kernel_id": kid, "family": fam,
                "pair": pair, "gap": v.gap,
                "ours_out": None, "ptxas_out": None,
                "match": False, "ours_ok": False, "ptxas_ok": False,
            }
            if ctx is not None:
                try:
                    ocubin, _ = compile_openptxas(ptx)
                    row["ours_ok"] = True
                except Exception:
                    ocubin = None
                row["ptxas_ok"] = True

                # The FG-4.0 kernels all use the same signature:
                #   (.param .u64 arg0, .param .u32 arg1)
                # arg0 is the output pointer, arg1 is a u32 value
                # used inside the kernel body.  We launch with one
                # thread (grid=1, block=1) and a fixed input.
                inputs = {"arg1": 0x12345}
                try:
                    if ocubin is not None:
                        row["ours_out"] = _run_kernel(
                            ctx, ocubin, kid, inputs, out_nbytes=4)
                    row["ptxas_out"] = _run_kernel(
                        ctx, pcubin, kid, inputs, out_nbytes=4)
                    if (row["ours_out"] is not None
                            and row["ptxas_out"] is not None):
                        row["match"] = (row["ours_out"] == row["ptxas_out"])
                except Exception as exc:
                    row["error"] = str(exc)[:80]
            results.append(row)
            break
    return results


# ---------------------------------------------------------------------------
# Ctrl-word helpers
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


def analyze_ctrl_words(cubin):
    """For every instruction in the cubin's first text section, return
    a list of (index, opcode, wdep, rbar, dest_regs, src_regs).  Used
    to locate the producer→consumer gap and check whether the
    scoreboard covers it.
    """
    for _sym, off, sz in _iter_text(cubin):
        out = []
        for k in range(sz // 16):
            raw = cubin[off + k*16 : off + k*16 + 16]
            out.append({
                "idx": k,
                "opc": _get_opcode(raw),
                "wdep": _get_wdep(raw),
                "rbar": _get_rbar(raw),
                "dest": sorted(_get_dest_regs(raw)),
                "src": sorted(_get_src_regs(raw)),
                "raw": raw.hex(),
            })
        return out
    return []


def find_producer_consumer_gap(ctrl_info, producer_opc, consumer_opc):
    """Locate the first (producer, consumer) pair in ctrl_info and
    return (producer_idx, consumer_idx, gap, ctrl_covers_wait).

    `ctrl_covers_wait` is True iff any instruction in [producer+1,
    consumer] has rbar carrying the class bit for the producer's
    wdep slot.
    """
    for i, instr in enumerate(ctrl_info):
        if instr["opc"] != producer_opc:
            continue
        dest_i = set(instr["dest"])
        if not dest_i:
            continue
        for j in range(i + 1, len(ctrl_info)):
            cons = ctrl_info[j]
            if cons["opc"] != consumer_opc:
                continue
            overlap = dest_i & set(cons["src"])
            if not overlap:
                continue
            gap = j - i - 1
            class_bit = 0
            if instr["wdep"] in _WDEP_TO_RBAR_MASK:
                class_bit = _WDEP_TO_RBAR_MASK[instr["wdep"]] & ~0x01
            wait_covered = False
            if class_bit:
                for m in range(i + 1, j + 1):
                    if ctrl_info[m]["rbar"] & class_bit:
                        wait_covered = True
                        break
            return (i, j, gap, wait_covered, class_bit)
    return None


# ---------------------------------------------------------------------------
# GPU runner
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRun:
    kernel_id: str
    cluster: str
    description: str
    # Compilation
    ours_ok: bool = False
    ptxas_ok: bool = False
    ours_err: str = ""
    ptxas_err: str = ""
    # Runtime
    ours_output: int | None = None
    ptxas_output: int | None = None
    expected: int = 0
    runtime_ok_ours: bool = False
    runtime_ok_ptxas: bool = False
    # Ctrl-word analysis
    producer_idx: int = -1
    consumer_idx: int = -1
    producer_gap_ptxas: int = -1
    producer_wdep_ptxas: int = -1
    consumer_rbar_ptxas: int = -1
    ctrl_covers_wait: bool = False
    # Classification
    classification: str = INCONCLUSIVE
    rationale: str = ""


def _run_kernel(ctx: CUDAContext, cubin: bytes, kid: str,
                inputs: dict, out_nbytes: int = 8) -> int | None:
    """Launch `kid` from `cubin` with the given param inputs and
    return the first 4 or 8 bytes of out_ptr as an integer.

    All kernels in this harness have the signature:
        (u64 out_ptr, ...ptx params)
    where the additional params are u32 or u64 values from `inputs`.
    """
    if not ctx.load(cubin):
        return None
    try:
        func = ctx.get_func(kid)
    except AssertionError:
        return None
    out_ptr = ctx.alloc(max(out_nbytes, 8))
    ctx.memset_d8(out_ptr, 0, max(out_nbytes, 8))

    # Build ctypes param pack: [out_ptr, arg1, arg2, ...]
    holders = [ctypes.c_uint64(out_ptr)]
    for name in sorted(inputs.keys()):
        v = inputs[name]
        if v > 0xFFFFFFFF:
            holders.append(ctypes.c_uint64(v))
        else:
            holders.append(ctypes.c_uint32(v))
    ptrs = (ctypes.c_void_p * len(holders))(
        *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders]
    )

    err = ctx.launch(func, 1, 1, ptrs)
    if err != 0:
        ctx.free(out_ptr)
        return None
    if ctx.sync() != 0:
        ctx.free(out_ptr)
        return None

    buf = ctx.copy_from(out_ptr, max(out_nbytes, 8))
    ctx.free(out_ptr)
    if out_nbytes == 8:
        return struct.unpack('<Q', buf[:8])[0]
    return struct.unpack('<I', buf[:4])[0]


# ---------------------------------------------------------------------------
# Per-cluster runner
# ---------------------------------------------------------------------------

# Producer/consumer opcode pairs per cluster probe.  These are the
# SASS-level opcodes PTXAS is expected to emit for each cluster.
PRODUCER_CONSUMER_HINT = {
    "A_imad_stg":     (0x224, 0x986),   # IMAD.32 → STG
    "A_shf_stg":      (0x819, 0x986),   # SHF → STG
    "A_iadd_stg":     (0x235, 0x986),   # IADD.64 → STG (PTXAS promotes)
    "B_iadd64_chain": (0x235, 0x235),   # IADD.64 → IADD.64
    "C_lea_iadd3x":   (0x211, 0x212),   # LEA → IADD3X
    "P_imad_stg_nd":  (0x224, 0x986),   # IMAD.32 → STG
    "P_lea_index":    (0x211, 0x986),   # LEA → STG
    "P_shf_iadd64":   (0x819, 0x235),   # SHF → IADD.64
}


def classify(kernel: dict, ctx: CUDAContext | None) -> EvidenceRun:
    kid = kernel["id"]
    run = EvidenceRun(
        kernel_id=kid, cluster=kernel["cluster"],
        description=kernel["description"],
        expected=kernel["expected"](kernel["inputs"]),
    )

    # Compile both cubins.
    try:
        ours_cubin, _ = compile_openptxas(kernel["ptx"])
        run.ours_ok = True
    except Exception as exc:
        run.ours_err = str(exc)[:120]
        ours_cubin = None
    try:
        ptxas_cubin, _ = compile_ptxas(kernel["ptx"])
        run.ptxas_ok = True
    except Exception as exc:
        run.ptxas_err = str(exc)[:120]
        ptxas_cubin = None

    # Ctrl-word analysis on PTXAS cubin (the gap=0 reference).
    if ptxas_cubin is not None:
        info = analyze_ctrl_words(ptxas_cubin)
        hint = PRODUCER_CONSUMER_HINT.get(kid)
        if hint:
            found = find_producer_consumer_gap(info, hint[0], hint[1])
            if found:
                p, c, gap, covered, _cb = found
                run.producer_idx = p
                run.consumer_idx = c
                run.producer_gap_ptxas = gap
                run.producer_wdep_ptxas = info[p]["wdep"]
                run.consumer_rbar_ptxas = info[c]["rbar"]
                run.ctrl_covers_wait = covered

    # Runtime if available.
    if ctx is not None:
        out_nbytes = 8 if kernel["cluster"] == "B" else 4
        if ours_cubin is not None:
            try:
                o = _run_kernel(ctx, ours_cubin, kid,
                                kernel["inputs"], out_nbytes)
                run.ours_output = o
                run.runtime_ok_ours = (o == run.expected)
            except Exception as exc:
                run.ours_err = (run.ours_err or "") + f" RT:{exc}"[:60]
        if ptxas_cubin is not None:
            try:
                o = _run_kernel(ctx, ptxas_cubin, kid,
                                kernel["inputs"], out_nbytes)
                run.ptxas_output = o
                run.runtime_ok_ptxas = (o == run.expected)
            except Exception as exc:
                run.ptxas_err = (run.ptxas_err or "") + f" RT:{exc}"[:60]

    # Classify.
    if ctx is None:
        run.classification = INCONCLUSIVE
        run.rationale = "GPU runtime unavailable — cannot confirm forwarding"
        return run

    if not run.ptxas_ok:
        run.classification = INCONCLUSIVE
        run.rationale = f"ptxas refused the kernel: {run.ptxas_err}"
        return run

    if run.runtime_ok_ptxas:
        # PTXAS cubin produced the correct output at whatever scheduling
        # it emitted.  If its producer→consumer gap was 0 and the
        # scoreboard did not cover it, we have positive forwarding
        # evidence.
        if run.producer_gap_ptxas == 0 and not run.ctrl_covers_wait:
            run.classification = FORWARDING_CONFIRMED
            run.rationale = (
                f"PTXAS emitted {kid} with gap=0 between producer and "
                f"consumer; consumer's rbar=0x{run.consumer_rbar_ptxas:02x} "
                f"does not carry the class bit for producer wdep=0x"
                f"{run.producer_wdep_ptxas:02x}; GPU runtime output "
                f"matched the expected value ({run.expected}) — the "
                f"pattern forwards without scoreboard coverage"
            )
        elif run.producer_gap_ptxas == 0 and run.ctrl_covers_wait:
            run.classification = INCONCLUSIVE
            run.rationale = (
                f"PTXAS scoreboard covers the wait (rbar 0x"
                f"{run.consumer_rbar_ptxas:02x} matches producer "
                f"wdep 0x{run.producer_wdep_ptxas:02x}); "
                f"this is not a forwarding case — it is a rbar case"
            )
        elif run.producer_gap_ptxas > 0:
            run.classification = INCONCLUSIVE
            run.rationale = (
                f"PTXAS inserted gap={run.producer_gap_ptxas} between "
                f"producer and consumer — cannot distinguish from "
                f"latency-required"
            )
        else:
            run.classification = INCONCLUSIVE
            run.rationale = "no producer→consumer pair found in PTXAS cubin"
    else:
        # PTXAS output did not match the expected value.  Either the
        # probe is wrong or PTXAS itself emitted a hazard (unlikely).
        run.classification = INCONCLUSIVE
        run.rationale = (
            f"PTXAS runtime output {run.ptxas_output!r} != expected "
            f"{run.expected} — probe design issue, not a model signal"
        )

    return run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from benchmarks.bench_util import _CUDA
    ctx = None
    if _CUDA is not None:
        try:
            ctx = CUDAContext()
        except Exception as exc:
            print(f"[WARN] CUDAContext creation failed: {exc}")
            ctx = None

    runs = []
    for kernel in ALL_CLUSTER_KERNELS + DEDICATED_PROBES:
        runs.append(classify(kernel, ctx))

    # Direct replay of the 10 FG-4.0 FP kernels: run OURS and PTXAS
    # cubins and compare outputs.  Aggregate by (producer, reader)
    # opcode pair.  Matching outputs on a non-trivial computation is
    # strong evidence that the PTXAS gap=0 schedule is functionally
    # equivalent to the OURS conservative schedule.
    replay = replay_fg40_fp_kernels(ctx)

    if ctx is not None:
        ctx.close()

    # Print.
    print("FG-4.2 hardware evidence harness")
    print("=" * 78)
    for r in runs:
        print(f"  [{r.cluster}] {r.kernel_id:<18s} {r.classification}")
        print(f"       {r.rationale}")
        print(f"       expected=0x{r.expected:x}  "
              f"ours_out={r.ours_output!r} "
              f"ptxas_out={r.ptxas_output!r}")
        print(f"       producer[{r.producer_idx}] "
              f"wdep=0x{r.producer_wdep_ptxas:02x}  "
              f"consumer[{r.consumer_idx}] "
              f"rbar=0x{r.consumer_rbar_ptxas:02x}  "
              f"gap={r.producer_gap_ptxas}  "
              f"rbar_covers={r.ctrl_covers_wait}")
        print()

    # Replay summary.
    print()
    print("=== FG-4.0 FP kernel replay: OURS vs PTXAS runtime ===")
    pair_match: dict[tuple, list] = {}
    for r in replay:
        pair_match.setdefault(r["pair"], []).append(r)
    for pair, rows in sorted(pair_match.items()):
        matches = sum(1 for r in rows if r.get("match"))
        total = len(rows)
        ex = rows[0]
        w, rd = pair
        tag = "OK   " if matches == total else "DIFF "
        print(f"  {tag} (0x{w:03x}, 0x{rd:03x}): {matches}/{total} match "
              f"ours_out={ex['ours_out']!r} ptxas_out={ex['ptxas_out']!r}  "
              f"[{rows[0]['kernel_id']}]")

    # Write JSON.
    out_path = ROOT / "probe_work" / "fg42_evidence_report.json"
    payload = {
        "cluster_runs": [asdict(r) for r in runs],
        "fg40_replay": replay,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[OK] wrote {out_path.relative_to(ROOT)}")

    # Summary.
    print()
    print("=== Classification summary (cluster probes) ===")
    by_cls: dict[str, int] = {}
    for r in runs:
        by_cls[r.classification] = by_cls.get(r.classification, 0) + 1
    for cls, n in sorted(by_cls.items()):
        print(f"  {cls:<22s}: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
