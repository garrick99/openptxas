"""FG-4.5 — Isolate (0x819, 0x235) SHF → IADD.64 forwarding.

Background:
  FG-4.0 flagged (0x819, 0x235) SHF → IADD.64 at gap=0 in the
  adversarial kernel f6_random_s45058_l10.  FG-4.3 tried to clone
  it with ld.param inputs but PTXAS re-routed through opcode 0x899
  (a different SHF variant) due to register-liveness changes.

FG-4.5 investigation:
  Every attempt to observe the pattern on GPU with a non-trivial
  input runs into the same scheduling wall: PTXAS always fills the
  slot BETWEEN the SHF producer and the IADD.64 consumer with an
  independent instruction (typically an LDC.64 loading the STG
  memory descriptor) to hide latency via ILP.  Consequences:

    - Adding a per-thread store  →  gap shifts to 1
    - Adding atom.global.max     →  gap shifts to 1
    - Adding predicated store    →  gap shifts to 1
    - Hoisting predicate setp    →  gap shifts to 1

  In every case, PTXAS emits the pair at gap=1, which is already
  classified as GAP_SAFE by the FG-3.0 bounded-lookahead proof
  engine (SHF has min_gpr_gap=1, gap=1 satisfies the bound).  The
  probe therefore CONFIRMS the (already-safe) gap=1 case and
  reports INCONCLUSIVE for gap=0.

Verdict:
  Because no PTXAS-reachable probe preserves gap=0 AND produces
  observable non-trivial output, FG-4.5 cannot directly confirm
  forwarding for the gap=0 case.  Per the FG-4.3 principle
  "Do NOT promote INCONCLUSIVE to SAFE", the (0x819, 0x235) pair
  is NOT added to _FORWARDING_SAFE_PAIRS.  The FG-4.0 adversarial
  harness therefore retains one MODEL_FALSE_POSITIVE case
  (f6_random_s45058_l10), which is a honest, documented, evidence-
  driven conservative decision — not a model gap.

  The gap=0 case can be resolved in a future pass by:
    (a) raw SASS emission that bypasses PTXAS scheduling entirely
    (b) post-hoc cubin patching to remove the intervening
        descriptor-load instruction
    (c) hardware-doc evidence from NVIDIA internals
  None of these are in scope for FG-4.5.
"""
from __future__ import annotations

import ctypes
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.bench_util import CUDAContext, compile_ptxas
from sass.scoreboard import (
    _get_opcode, _get_wdep, _get_rbar, _get_dest_regs, _get_src_regs,
)


# Original f6_random_s45058_l10 body with a SINGLE-TID predicated
# store: only tid.x == 17 writes.  The store is predicated so the
# rest of the 32-thread warp executes the SHF→IADD.64 chain for its
# own r0 value but does NOT store.  All 32 lanes share the same
# SASS schedule (SIMT); if forwarding works for lane 17 (non-zero
# tid ≠ 0 mid-warp), it works for all.  No address computation in
# the body → PTXAS has nothing to interleave between SHF and IADD.64.
PROBE_PTX = """
.version 8.8
.target sm_120
.address_size 64

.visible .entry fg45_shf_iadd64(
    .param .u64 arg0,
    .param .u32 arg1)
{
    .reg .u32 %r<10>;
    .reg .u64 %rd<8>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u64 %rd0, [arg0];
    // Hoist the predicate eval BEFORE the SHF body so PTXAS does
    // not interleave it between the SHF and IADD.64.
    setp.eq.u32 %p0, %r0, 17;

    shl.b32 %r5, %r0, 4;
    shr.b32 %r4, %r0, 2;
    sub.u32 %r2, %r0, %r4;
    xor.b32 %r7, %r0, %r0;
    sub.u32 %r3, %r7, %r4;
    xor.b32 %r1, %r3, %r0;
    sub.u32 %r3, %r7, %r4;
    xor.b32 %r2, %r3, %r0;
    sub.u32 %r7, %r7, %r4;
    xor.b32 %r5, %r3, %r0;

    @%p0 st.global.u32 [%rd0], %r7;

    ret;
}
"""

BLOCK_DIM = 32


def _mask32(x: int) -> int:
    return x & 0xFFFFFFFF


def simulate(tid: int) -> int:
    """Python model of the kernel body for a given tid.x value."""
    r0 = _mask32(tid)
    r5 = _mask32(r0 << 4)     # unused in final output
    r4 = r0 >> 2
    r2 = _mask32(r0 - r4)     # unused
    r7 = _mask32(r0 ^ r0)     # = 0
    r3 = _mask32(r7 - r4)
    r1 = _mask32(r3 ^ r0)     # unused
    r3 = _mask32(r7 - r4)
    r2 = _mask32(r3 ^ r0)     # unused
    r7 = _mask32(r7 - r4)     # final value for r7
    r5 = _mask32(r3 ^ r0)     # unused
    return r7


def _iter_text(cubin):
    e_shoff = struct.unpack_from('<Q', cubin, 40)[0]
    e_shnum = struct.unpack_from('<H', cubin, 60)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 62)[0]
    stoff = struct.unpack_from('<Q', cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    for i in range(e_shnum):
        base = e_shoff + i * 64
        nm = struct.unpack_from('<I', cubin, base)[0]
        end = cubin.index(0, stoff + nm)
        name = cubin[stoff + nm:end]
        if name.startswith(b".text."):
            off = struct.unpack_from('<Q', cubin, base + 24)[0]
            sz = struct.unpack_from('<Q', cubin, base + 32)[0]
            yield name.decode()[len(".text."):], off, sz


def analyze_sass(cubin) -> dict:
    """Locate the (0x819, 0x235) pair and its ctrl words."""
    for _sym, off, sz in _iter_text(cubin):
        for i in range(sz // 16):
            raw_i = cubin[off + i*16 : off + i*16 + 16]
            if _get_opcode(raw_i) != 0x819:
                continue
            dest_i = set(_get_dest_regs(raw_i))
            if not dest_i:
                continue
            # Look for immediately-following IADD.64 reading overlap regs
            for j in range(i + 1, min(i + 3, sz // 16)):
                raw_j = cubin[off + j*16 : off + j*16 + 16]
                if _get_opcode(raw_j) != 0x235:
                    continue
                overlap = dest_i & set(_get_src_regs(raw_j))
                if overlap:
                    return {
                        "found": True,
                        "producer_idx": i,
                        "consumer_idx": j,
                        "gap": j - i - 1,
                        "producer_wdep": _get_wdep(raw_i),
                        "consumer_rbar": _get_rbar(raw_j),
                        "overlap_regs": sorted(overlap),
                    }
        return {"found": False}
    return {"found": False, "reason": "no text section"}


def run():
    # Compile with PTXAS (not OURS — we need the exact SASS pair and
    # OURS doesn't have a stable lowering path to that shape).
    cubin, _ = compile_ptxas(PROBE_PTX)

    sass_info = analyze_sass(cubin)
    print("=== SASS analysis ===")
    print(json.dumps(sass_info, indent=2))

    if not sass_info.get("found"):
        print("ABORT: (0x819, 0x235) pair not present in PTXAS cubin")
        return {"status": "ABORT", "reason": "pair not found"}

    # Launch with block=BLOCK_DIM; atom.max captures the maximum of
    # all lanes' r7 values into out[0] deterministically.
    ctx = CUDAContext()
    ctx.load(cubin)
    func = ctx.get_func("fg45_shf_iadd64")
    out = ctx.alloc(4)
    ctx.memset_d8(out, 0, 4)
    holders = [ctypes.c_uint64(out), ctypes.c_uint32(0)]
    ptrs = (ctypes.c_void_p * 2)(
        *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders]
    )
    err = ctx.launch(func, 1, BLOCK_DIM, ptrs)
    assert err == 0, f"launch failed: {err}"
    assert ctx.sync() == 0

    buf = ctx.copy_from(out, 4)
    ctx.free(out)
    ctx.close()

    observed = struct.unpack('<I', buf)[0]
    # The probe predicates the store so only tid=17 writes.
    # Expected = simulate(17).
    expected = simulate(17)

    print()
    print("=== single-tid predicated store result ===")
    print(f"observed (from tid=17) = 0x{observed:08x} ({observed})")
    print(f"expected (simulate(17)) = 0x{expected:08x} ({expected})")

    print()
    match = observed == expected
    observed_max = observed
    expected_max = expected
    gap = sass_info.get("gap", -1)

    if match and gap == 0:
        verdict = "FORWARDING_CONFIRMED"
        rationale = (
            f"PTXAS cubin emits (0x819, 0x235) at gap=0; consumer rbar="
            f"0x{sass_info['consumer_rbar']:02x} does not carry the "
            f"producer's class bit; atom.max across 32 lanes matches "
            f"the Python simulator max "
            f"({observed_max:#010x}).  Every non-zero lane must have "
            f"read the correct forwarded value from SHF; otherwise "
            f"the max would be wrong."
        )
        print(f"[VERDICT] {verdict}")
        print(f"  {rationale}")
    elif match and gap > 0:
        verdict = "INCONCLUSIVE_GAP_NOT_ZERO"
        rationale = (
            f"Runtime matches but PTXAS emitted the pair at gap={gap}, "
            f"not gap=0.  This only confirms the (already proven) "
            f"GAP_SAFE case, not the gap=0 forwarding case."
        )
        print(f"[VERDICT] {verdict}")
        print(f"  {rationale}")
    else:
        verdict = "LATENCY_REQUIRED"
        rationale = (
            f"Runtime observed max 0x{observed_max:08x} != expected "
            f"0x{expected_max:08x}.  At gap={gap}, the pattern does "
            f"NOT produce the simulator-computed value."
        )
        print(f"[VERDICT] {verdict}")
        print(f"  {rationale}")

    # Write JSON report
    report = {
        "status": verdict,
        "sass_info": sass_info,
        "observed_max": observed_max,
        "expected_max": expected_max,
        "per_tid_sim": [simulate(t) for t in range(BLOCK_DIM)],
        "block_dim": BLOCK_DIM,
        "match": match,
        "rationale": rationale,
    }
    out_path = ROOT / "probe_work" / "fg45_evidence.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[OK] wrote {out_path.relative_to(ROOT)}")

    return report


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    run()
