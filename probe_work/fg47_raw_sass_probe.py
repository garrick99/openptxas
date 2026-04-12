"""FG-4.7 — Raw SASS cubin-patch probe for (0x819, 0x235) SHF → IADD.64.

Forces the exact opcode pair at gap=0 in a patched cubin and launches
on real hardware with per-thread observable output.

Strategy:
  1. Compile a minimal base kernel via PTXAS (per-thread store of tid-
     based computation).
  2. Overwrite instructions [4]-[7] in the .text section with hand-
     crafted SASS bytes:
       [4] IMAD.WIDE R4, R7, 4, R2   — address calc (out + tid*4)
       [5] SHF.L     R0, RZ, 5, R7   — R0 = tid.x << 5  (producer)
       [6] IADD.64   R2:R3, R0:R1, R4:R5 — reads R0 at gap=0 (consumer)
       [7] STG       [R4:R5, UR4], R2    — store low half of IADD.64
  3. Launch with block=32, read out[0..31].
  4. Compare observed vs expected.

Non-trivial data design:
  If SHF forwards correctly:
    R0 = tid.x << 5 = tid * 32
    R2 = (tid * 32) + out_ptr_lo    (low half of 64-bit add)
  If SHF does NOT forward (R0 stale = hardware init):
    R2 = 0 + out_ptr_lo             (same for all threads)

  We compare gap=0 vs gap=1 variants.  If both produce the same
  per-thread values, forwarding is confirmed.
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
from sass.scoreboard import _get_opcode, _get_wdep, _get_rbar
from sass.encoding.sm_120_opcodes import (
    encode_imad_wide, encode_nop,
)
from sass.schedule import _patch_ctrl


# ---------------------------------------------------------------------------
# Base PTX (compiled by PTXAS; we only keep the cubin shell + preamble)
# ---------------------------------------------------------------------------

BASE_PTX = """
.version 8.8
.target sm_120
.address_size 64

.visible .entry fg47_base(
    .param .u64 out,
    .param .u32 n)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    mov.u32 %r0, %tid.x;
    ld.param.u64 %rd0, [out];
    mul.wide.u32 %rd1, %r0, 4;
    add.u64 %rd1, %rd0, %rd1;
    mul.lo.u32 %r1, %r0, 3;
    add.u32 %r1, %r1, 7;
    st.global.u32 [%rd1], %r1;
    ret;
}
"""

BLOCK = 32


# ---------------------------------------------------------------------------
# Instruction builders (raw 16-byte SASS)
# ---------------------------------------------------------------------------

def _mk_ctrl(rbar: int = 0x01, wdep: int = 0x3e, misc: int = 0x1) -> int:
    return ((rbar & 0x1f) << 10) | ((wdep & 0x3f) << 4) | (misc & 0xf)


def _encode_shf_l_u32(dest: int, src_funnel: int, shift_imm: int,
                       ctrl: int = 0) -> bytes:
    """SHF.L.U32 dest, RZ, shift_imm, src_funnel.

    Result: dest = src_funnel << shift_imm   (zero-extended from RZ).

    Ground truth from f6_random_s45058_l10 PTXAS cubin [4]:
      SHF.L R0, RZ, 4, R0 → 19 78 00 ff 04 00 00 00 00 16 01 00 ...
      b0=0x19, b1=0x78, b2=dest, b3=0xff(RZ), b4=shift_imm,
      b8=src_funnel, b9=0x16, b10=0x01, b11=0x00
    """
    if ctrl == 0:
        ctrl = _mk_ctrl()
    raw24 = (ctrl & 0x7FFFFF) << 1
    raw = bytearray(16)
    raw[0] = 0x19
    raw[1] = 0x78
    raw[2] = dest & 0xff
    raw[3] = 0xff            # src0 = RZ
    raw[4] = shift_imm & 0xff
    raw[8] = src_funnel & 0xff
    raw[9] = 0x16            # SHF.L modifier
    raw[10] = 0x01
    raw[11] = 0x00
    raw[13] = raw24 & 0xff
    raw[14] = (raw24 >> 8) & 0xff
    raw[15] = (raw24 >> 16) & 0xff
    return bytes(raw)


def _encode_iadd64_rr(dest: int, src0: int, src1: int,
                       ctrl: int = 0) -> bytes:
    """IADD.64 R-R: dest:dest+1 = src0:src0+1 + src1:src1+1.

    Opcode 0x235.  Ground truth from f6_random_s45058_l10 PTXAS [5]:
      IADD.64 R5:R6, R0:R1, R0:R1 → 35 72 05 00 00 00 00 00 00 00 8e 07 ...
      b0=0x35, b1=0x72, b2=dest, b3=src0, b4=src1,
      b9=0x00, b10=0x8e, b11=0x07
    """
    if ctrl == 0:
        ctrl = _mk_ctrl()
    raw24 = (ctrl & 0x7FFFFF) << 1
    raw = bytearray(16)
    raw[0] = 0x35
    raw[1] = 0x72
    raw[2] = dest & 0xff
    raw[3] = src0 & 0xff
    raw[4] = src1 & 0xff
    raw[9] = 0x00
    raw[10] = 0x8e
    raw[11] = 0x07
    raw[13] = raw24 & 0xff
    raw[14] = (raw24 >> 8) & 0xff
    raw[15] = (raw24 >> 16) & 0xff
    return bytes(raw)


def _encode_stg32(addr_pair_base: int, data_reg: int, ur_desc: int,
                  ctrl: int = 0) -> bytes:
    """STG.E.32 [addr_pair, UR_desc], data_reg.

    Ground truth from fg47_base PTXAS [7]:
      STG [R2:R3, UR4], R5 → 86 79 00 02 05 00 00 00 04 19 10 0c ...
      b0=0x86, b1=0x79, b2=0x00, b3=addr, b4=data,
      b8=ur_desc, b9=0x19, b10=0x10, b11=0x0c
    """
    if ctrl == 0:
        ctrl = _mk_ctrl(rbar=0x09, wdep=0x3f)
    raw24 = (ctrl & 0x7FFFFF) << 1
    raw = bytearray(16)
    raw[0] = 0x86
    raw[1] = 0x79
    raw[2] = 0x00
    raw[3] = addr_pair_base & 0xff
    raw[4] = data_reg & 0xff
    raw[8] = ur_desc & 0xff
    raw[9] = 0x19
    raw[10] = 0x10
    raw[11] = 0x0c
    raw[13] = raw24 & 0xff
    raw[14] = (raw24 >> 8) & 0xff
    raw[15] = (raw24 >> 16) & 0xff
    return bytes(raw)


# ---------------------------------------------------------------------------
# Cubin patcher
# ---------------------------------------------------------------------------

def _find_text_section(cubin: bytes) -> tuple[int, int]:
    """Return (file_offset, size) of the first .text.* section."""
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
            return off, sz
    raise RuntimeError("no .text.* section found")


def patch_cubin(cubin: bytes, gap: int) -> bytes:
    """Minimal cubin patching strategy:

    KEEP the original preamble [0]-[3] and the original IMAD.WIDE
    address calc at [6] and the EXIT/BRA at [8]/[9].

    gap=0: overwrite [4] and [5] with SHF and IADD.64; patch [7]
           STG's data register byte.
    gap>0: shift IMAD.WIDE and STG into later NOP slots to make room
           for gap NOPs between SHF and IADD.64.

    Register plan:
      R7 = tid.x  (from [1] S2R)
      R2:R3 = out base ptr  (from [2] LDC.64)
      [4] SHF R0, RZ, 5, R7 → R0 = tid.x << 5
      [5+gap] IADD.64 R0:R1, R0:R1, R2:R3 → R0 = (tid<<5)+out_lo
      [next] original IMAD.WIDE R2:R3, R7, 4, R2:R3 → addr calc
      [next] STG [R2:R3, UR4], R0 → store R0
      [next] EXIT + BRA (unchanged bytes)
    """
    off, sz = _find_text_section(cubin)
    buf = bytearray(cubin)

    # Preserve original IMAD.WIDE ([6]) and EXIT ([8]) / BRA ([9]) bytes.
    orig_imadw = bytes(buf[off + 6*16 : off + 6*16 + 16])
    orig_stg   = bytes(buf[off + 7*16 : off + 7*16 + 16])
    orig_exit  = bytes(buf[off + 8*16 : off + 8*16 + 16])
    orig_bra   = bytes(buf[off + 9*16 : off + 9*16 + 16])

    # Patch STG data-register (b4) from R5 to R0 — the IADD.64 result.
    stg_patched = bytearray(orig_stg)
    stg_patched[4] = 0x00  # data_reg = R0
    stg_patched = bytes(stg_patched)

    # Build the patched instruction sequence.
    shf = _encode_shf_l_u32(
        dest=0, src_funnel=7, shift_imm=5,
        # rbar=0x07: wait for S2R (wdep 0x31, rbar bit 1 = 0x02) AND
        # LDC.64 (wdep 0x33, rbar bit 2 = 0x04) to ensure R7 (tid.x)
        # and R2:R3 (output base) are both ready before the SHF reads
        # R7 and the subsequent IADD.64 reads R2:R3.
        ctrl=_mk_ctrl(rbar=0x07, wdep=0x3e, misc=0x1),
    )
    iadd64 = _encode_iadd64_rr(
        dest=0, src0=0, src1=2,
        # rbar=0x01: no extra wait (forwarding test — intentionally
        # NOT setting any class bit that could cover the SHF→IADD.64
        # edge via scoreboard).  wdep=0x3e: ALU.
        ctrl=_mk_ctrl(rbar=0x01, wdep=0x3e, misc=0x1),
    )
    nop = encode_nop(ctrl=_mk_ctrl(rbar=0x01, wdep=0x3e, misc=0x0))

    # Assemble sequence starting at slot [4].
    seq = [shf]
    for _ in range(gap):
        seq.append(nop)
    seq.append(iadd64)
    seq.append(orig_imadw)
    seq.append(stg_patched)
    seq.append(orig_exit)
    seq.append(orig_bra)

    # Overwrite from slot [4] onward.  Remaining slots fill with NOPs.
    slot = 4
    for instr in seq:
        o = off + slot * 16
        buf[o:o + 16] = instr
        slot += 1
    nop_fill = encode_nop(ctrl=_mk_ctrl(rbar=0x01, wdep=0x3e, misc=0x0))
    while slot < (sz // 16):
        o = off + slot * 16
        buf[o:o + 16] = nop_fill
        slot += 1

    return bytes(buf)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_variant(ctx: CUDAContext, cubin: bytes, gap: int) -> list[int]:
    """Launch the patched cubin with block=BLOCK threads and return
    the per-thread u32 outputs as a list."""
    ctx.load(cubin)
    func = ctx.get_func("fg47_base")
    out = ctx.alloc(BLOCK * 4)
    ctx.memset_d8(out, 0xEE, BLOCK * 4)
    holders = [ctypes.c_uint64(out), ctypes.c_uint32(0)]
    ptrs = (ctypes.c_void_p * 2)(
        *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders]
    )
    err = ctx.launch(func, 1, BLOCK, ptrs)
    if err != 0:
        ctx.free(out)
        return None
    if ctx.sync() != 0:
        ctx.free(out)
        return None
    buf = ctx.copy_from(out, BLOCK * 4)
    ctx.free(out)
    return [struct.unpack_from('<I', buf, t * 4)[0] for t in range(BLOCK)]


def dump_sass(cubin: bytes, slots: range):
    off, _ = _find_text_section(cubin)
    for k in slots:
        raw = cubin[off + k*16 : off + k*16 + 16]
        opc = _get_opcode(raw)
        wd = _get_wdep(raw)
        rb = _get_rbar(raw)
        print(f"  [{k:2d}] opc=0x{opc:03x} wd=0x{wd:02x} rb=0x{rb:02x} hex={raw.hex()}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    base_cubin, _ = compile_ptxas(BASE_PTX)
    ctx = CUDAContext()

    # Verify base cubin runs
    base_out = run_variant(ctx, base_cubin, gap=-1)
    print("=== Base cubin (unpatched) ===")
    if base_out is None:
        print("FAIL: base cubin did not launch")
        ctx.close()
        return
    base_expected = [t * 3 + 7 for t in range(BLOCK)]
    base_ok = base_out == base_expected
    print(f"  base output matches expected: {base_ok}")
    if not base_ok:
        print(f"  observed[0..4] = {base_out[:5]}")
        print(f"  expected[0..4] = {base_expected[:5]}")

    # Build patched variants
    results = {}
    for gap in (0, 1, 2):
        label = f"gap={gap}"
        patched = patch_cubin(base_cubin, gap)
        print(f"\n=== Patched variant: {label} ===")
        dump_sass(patched, range(4, 8))
        observed = run_variant(ctx, patched, gap)
        if observed is None:
            print(f"  LAUNCH FAILED for {label}")
            results[gap] = {"status": "LAUNCH_FAIL", "observed": None}
            continue
        # Compute deltas: observed[t] - observed[0]
        deltas = [o - observed[0] for o in observed]
        expected_deltas = [t * 32 for t in range(BLOCK)]
        match = deltas == expected_deltas
        print(f"  observed[0..7] = {[f'0x{o:08x}' for o in observed[:8]]}")
        print(f"  deltas[0..7]   = {deltas[:8]}")
        print(f"  expected deltas = {expected_deltas[:8]}")
        print(f"  deltas match expected: {match}")
        results[gap] = {
            "status": "OK" if match else "MISMATCH",
            "observed": observed,
            "deltas": deltas,
            "match": match,
        }

    ctx.close()

    # Classify
    print("\n=== Classification ===")
    g0 = results.get(0, {})
    g1 = results.get(1, {})
    g2 = results.get(2, {})

    if g0.get("match") and g1.get("match") and g2.get("match"):
        # All variants produce correct per-thread values.
        # For gap=0, scoreboard doesn't cover (rbar=0x01 on IADD.64) so
        # hardware forwarding is the only mechanism.
        verdict = "FORWARDING_CONFIRMED"
        rationale = (
            "gap=0, gap=1, gap=2 all produce correct per-thread values. "
            "IADD.64 at gap=0 reads R0 (from SHF) with rbar=0x01 (no "
            "scoreboard wait bits). The only mechanism making the result "
            "correct is hardware forwarding."
        )
    elif not g0.get("match") and g1.get("match"):
        verdict = "LATENCY_REQUIRED"
        rationale = (
            f"gap=0 deltas MISMATCH (first 4: {g0.get('deltas', [])[:4]}); "
            f"gap=1 deltas correct. Forwarding does NOT work at gap=0."
        )
    elif g0.get("status") == "LAUNCH_FAIL":
        verdict = "HARDWARE_INVALID"
        rationale = "gap=0 patched cubin refused to launch."
    else:
        verdict = "INCONCLUSIVE"
        rationale = f"unexpected combination: g0={g0.get('status')}, g1={g1.get('status')}"

    print(f"  VERDICT: {verdict}")
    print(f"  {rationale}")

    # Write report
    report = {
        "verdict": verdict,
        "rationale": rationale,
        "base_ok": base_ok,
        "gap_0": g0,
        "gap_1": g1,
        "gap_2": g2,
    }
    out_path = ROOT / "probe_work" / "fg47_evidence.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[OK] wrote {out_path.relative_to(ROOT)}")
    return verdict


if __name__ == "__main__":
    main()
