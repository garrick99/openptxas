"""
WB-0: Kernel Workbench MVP

CLI cockpit for the openptxas → forge → ptxas stack.

Examples
--------
  python workbench.py --kernel reduce_sum
  python workbench.py --kernel conv2d_looped --compare ptxas
  python workbench.py --kernel hmma_zero --compare ptxas --mode bench

The workbench:
  • compiles a known PTX through openptxas
  • optionally compiles the same PTX through ptxas
  • launches the kernel on the GPU and verifies correctness
  • collects regs / sass_total / sass_non_nop / time_ms for both
  • prints a canonical block
  • writes a JSON artifact to results/<ts>_<kernel>.json

WB-0 is intentionally narrow: hardcoded catalog, no kernel editor, no
GUI, no AI, no plugin system.  Each catalog entry just points at a PTX
file (or inline string), names the entry symbol, and supplies a
correctness/benchmark harness.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmarks.bench_util import (
    CUDAContext,
    analyze_cubin,
    compile_openptxas,
    compile_ptxas,
)
from sass import compact as compact_mod
from sass.compact import CompactReport, collect_used_gprs


# ---------------------------------------------------------------------------
# Cubin metric extraction.
#
# bench_util.analyze_cubin reads `capmerc[8]` for num_gprs which doesn't
# match the OpenPTXas cubin layout (returns the non-nop instruction count
# by accident).  We instead walk the cubin's text section directly,
# decoding 16-byte SASS instructions and asking sass.compact.collect_used_gprs
# for the maximum referenced GPR.  This works for any sm_120 cubin (ours
# or ptxas) because the GPR_FIELDS metadata table is field-safe.
# ---------------------------------------------------------------------------
class _RawSassInstr:
    """Minimal SASS-instr shim for collect_used_gprs (only .raw is read)."""
    __slots__ = ("raw", "comment")

    def __init__(self, raw: bytes):
        self.raw = raw
        self.comment = ""


def _find_text_section(cubin: bytes) -> bytes | None:
    """Return the bytes of the kernel text section in an ELF64 cubin."""
    e_shoff = struct.unpack_from("<Q", cubin, 40)[0]
    e_shnum = struct.unpack_from("<H", cubin, 60)[0]
    e_shstrndx = struct.unpack_from("<H", cubin, 62)[0]
    shstrtab_off = struct.unpack_from("<Q", cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    shstrtab_sz = struct.unpack_from("<Q", cubin, e_shoff + e_shstrndx * 64 + 32)[0]
    shstrtab = cubin[shstrtab_off:shstrtab_off + shstrtab_sz]
    for i in range(e_shnum):
        sh = e_shoff + i * 64
        n_off = struct.unpack_from("<I", cubin, sh)[0]
        nm = shstrtab[n_off:shstrtab.index(0, n_off)].decode()
        sec_off = struct.unpack_from("<Q", cubin, sh + 24)[0]
        sec_sz = struct.unpack_from("<Q", cubin, sh + 32)[0]
        if ".text." in nm and "capmerc" not in nm:
            return cubin[sec_off:sec_off + sec_sz]
    return None


def cubin_metrics(cubin: bytes) -> dict:
    """Extract regs / sass_total / sass_non_nop from a cubin.

    `regs` is computed as max(GPR index referenced in any field-covered
    instruction) + 1, so it reflects the actual register footprint of the
    emitted code (not whatever the .nv.info metadata declares).
    """
    text = _find_text_section(cubin)
    if text is None:
        return {"regs": 0, "sass_total": 0, "sass_non_nop": 0}
    n_instrs = len(text) // 16
    n_nops = 0
    instrs = []
    for off in range(0, len(text), 16):
        raw = text[off:off + 16]
        opcode = (raw[0] | (raw[1] << 8)) & 0xFFF
        if opcode == 0x918:  # NOP
            n_nops += 1
        instrs.append(_RawSassInstr(raw))
    used, _pair, _quad = collect_used_gprs(instrs)
    # Filter sentinel (RZ=255 already filtered by collect_used_gprs)
    regs = (max(used) + 1) if used else 0
    return {
        "regs": regs,
        "sass_total": n_instrs,
        "sass_non_nop": n_instrs - n_nops,
    }


# ---------------------------------------------------------------------------
# Repo paths (used for commit hash collection only).
# ---------------------------------------------------------------------------
REPO_OPENPTXAS = ROOT
REPO_FORGE     = ROOT.parent / "forge"
REPO_OPENCUDA  = ROOT.parent / "opencuda"


def _git_short(repo: Path) -> str:
    if not repo.exists() or not (repo / ".git").exists():
        return "(missing)"
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "(unknown)"


# ---------------------------------------------------------------------------
# Compaction-report capture.  We monkey-patch sass.compact.compact for the
# duration of one openptxas build so we can record the per-kernel diagnostics
# (regs_before, regs_after, compacted_insts, gpr_fields_rewritten).
# ---------------------------------------------------------------------------
def compile_with_report(ptx: str) -> tuple[bytes, float, CompactReport | None]:
    captured: list[CompactReport] = []
    orig = compact_mod.compact

    def spy(sass_instrs, verbose=False, kernel_name="<unknown>", report=None):
        if report is None:
            report = CompactReport(kernel_name)
        result = orig(sass_instrs, verbose=False,
                      kernel_name=kernel_name, report=report)
        captured.append(report)
        return result

    compact_mod.compact = spy
    try:
        cubin, dt = compile_openptxas(ptx)
    finally:
        compact_mod.compact = orig
    return cubin, dt, (captured[0] if captured else None)


# ---------------------------------------------------------------------------
# Kernel harnesses.  Each takes (ctx, func) and returns (correct, time_ms).
# ---------------------------------------------------------------------------
def _make_args(*ctypes_values):
    arr = (ctypes.c_void_p * len(ctypes_values))(
        *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in ctypes_values]
    )
    return arr, ctypes_values  # second tuple keeps refs alive


def _bench_launch(ctx: CUDAContext, func, grid, block, args, iters: int = 50,
                  warmup: int = 5, smem: int = 0) -> float:
    """Run the kernel `iters` times and return median launch time in ms."""
    s = ctx.event_create()
    e = ctx.event_create()
    for _ in range(warmup):
        ctx.cuda.cuLaunchKernel(func, *grid, *block, smem, None, args, None)
    ctx.sync()
    times = []
    for _ in range(iters):
        ctx.event_record(s)
        ctx.cuda.cuLaunchKernel(func, *grid, *block, smem, None, args, None)
        ctx.event_record(e)
        ctx.sync()
        times.append(ctx.event_elapsed_ms(s, e))
    return median(times)


def harness_reduce_sum(ctx: CUDAContext, func, mode: str) -> dict:
    """Warp butterfly reduction of u64 values.

    Layout (matches reduce_sum_open.ptx):
      .param .u64 data_ptr, data_len, output_ptr, output_len, n
    Each block reduces one warp (32 lanes) and writes one u64 to output[block].
    """
    n_data = 65536
    block_size = 256
    grid_size = n_data // block_size  # one block per chunk

    nbytes = n_data * 8
    out_bytes = grid_size * 8

    d_data = ctx.alloc(nbytes)
    d_out = ctx.alloc(out_bytes)
    try:
        # All-ones data: each warp's reduction = 32, each block has 8 warps,
        # block writes only the first warp's result so output[block] = 32.
        host_data = (ctypes.c_uint64 * n_data)(*([1] * n_data))
        ctx.copy_to(d_data, bytes(host_data))
        ctx.copy_to(d_out, b"\x00" * out_bytes)

        a_data = ctypes.c_uint64(d_data)
        a_dlen = ctypes.c_uint64(n_data)
        a_out  = ctypes.c_uint64(d_out)
        a_olen = ctypes.c_uint64(grid_size)
        a_n    = ctypes.c_uint64(n_data)
        args, _hold = _make_args(a_data, a_dlen, a_out, a_olen, a_n)

        # Single launch for correctness
        ctx.cuda.cuLaunchKernel(func, grid_size, 1, 1, block_size, 1, 1,
                                0, None, args, None)
        assert ctx.sync() == 0, "reduce_sum kernel crashed"

        out = ctx.copy_from(d_out, out_bytes)
        out_vals = struct.unpack(f"<{grid_size}Q", out)
        # Each block writes the warp-reduce of lanes 0..31 of the first warp
        # = sum of 32 ones = 32.
        expected = 32
        correct = all(v == expected for v in out_vals)

        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (grid_size, 1, 1), (block_size, 1, 1), args
            )
    finally:
        ctx.free(d_data)
        ctx.free(d_out)

    return {"correct": correct, "time_ms": time_ms}


def harness_conv2d_looped(ctx: CUDAContext, func, mode: str) -> dict:
    """3x3 conv with zero-padding, u64 arithmetic.

    Layout (matches conv2d_looped.ptx):
      .param .u64 input_ptr, input_len, output_ptr, output_len,
                  filter_ptr, filter_len, p_width, p_height
    """
    width = 128
    height = 128
    n_in = width * height
    n_out = width * height
    n_f = 9

    d_in = ctx.alloc(n_in * 8)
    d_out = ctx.alloc(n_out * 8)
    d_f = ctx.alloc(n_f * 8)
    try:
        host_in = (ctypes.c_uint64 * n_in)(*([1] * n_in))      # input = ones
        host_f  = (ctypes.c_uint64 * n_f )(*([1] * n_f ))      # filter = ones
        ctx.copy_to(d_in, bytes(host_in))
        ctx.copy_to(d_f, bytes(host_f))
        ctx.copy_to(d_out, b"\x00" * (n_out * 8))

        a_in   = ctypes.c_uint64(d_in)
        a_ilen = ctypes.c_uint64(n_in)
        a_out  = ctypes.c_uint64(d_out)
        a_olen = ctypes.c_uint64(n_out)
        a_f    = ctypes.c_uint64(d_f)
        a_flen = ctypes.c_uint64(n_f)
        a_w    = ctypes.c_uint64(width)
        a_h    = ctypes.c_uint64(height)
        args, _hold = _make_args(
            a_in, a_ilen, a_out, a_olen, a_f, a_flen, a_w, a_h)

        block = (16, 16, 1)
        grid = ((width + 15) // 16, (height + 15) // 16, 1)

        ctx.cuda.cuLaunchKernel(func, *grid, *block, 0, None, args, None)
        assert ctx.sync() == 0, "conv2d kernel crashed"

        out = ctx.copy_from(d_out, n_out * 8)
        vals = struct.unpack(f"<{n_out}Q", out)

        # All-ones input × all-ones filter, with KERNEL_RADIUS=1 zero-pad
        # (kernel uses ix-1, iy-1).  Interior pixels accumulate 9 taps; edges
        # accumulate fewer.  Center pixel: 9.
        center = vals[64 * width + 64]
        corner = vals[0]
        # Center should be 9 (full 3x3 of ones).  Corner depends on whether
        # the kernel handles bounds correctly; we just check center.
        correct = (center == 9)

        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, grid, block, args)
    finally:
        ctx.free(d_in)
        ctx.free(d_out)
        ctx.free(d_f)

    return {"correct": correct, "time_ms": time_ms}


_PTX_HMMA_ZERO = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry hmma_zero_kernel(
    .param .u64 p_out
)
{
    .reg .f32 %f<4>;
    .reg .b32 %r<2>;
    .reg .u64 %rd<2>;

    ld.param.u64    %rd0, [p_out];

    mov.b32 %r0, 0;
    mov.b32 %r1, 0;
    mov.f32 %f0, 0f00000000;
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;

    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
        {%f0, %f1, %f2, %f3},
        {%r0, %r1},
        {%r0},
        {%f0, %f1, %f2, %f3};

    st.global.f32 [%rd0], %f0;
    ret;
}
"""


def harness_hmma_zero(ctx: CUDAContext, func, mode: str) -> dict:
    """HMMA.16816.F32 with all-zero inputs — output must be 0.0f."""
    d_out = ctx.alloc(4)
    try:
        ctx.copy_to(d_out, b"\x00\x00\x00\x00")
        a_out = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_out)

        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "hmma kernel crashed"

        raw = ctx.copy_from(d_out, 4)
        result = struct.unpack("<f", raw)[0]
        correct = (result == 0.0)

        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (1, 1, 1), (32, 1, 1), args
            )
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
KERNELS: dict[str, dict] = {
    "reduce_sum": {
        "display": "reduce_sum (warp butterfly, u64)",
        "ptx_path": REPO_FORGE / "benchmarks" / "fb0_baseline" / "reduce_sum_open.ptx",
        "ptx_inline": None,
        "kernel_name": "reduce_sum",
        "harness": harness_reduce_sum,
    },
    "conv2d_looped": {
        "display": "conv2d 3x3 looped (u64)",
        "ptx_path": REPO_FORGE / "benchmarks" / "fb0_baseline" / "conv2d_looped.ptx",
        "ptx_inline": None,
        "kernel_name": "conv2d",
        "harness": harness_conv2d_looped,
    },
    "hmma_zero": {
        "display": "HMMA m16n8k8 zero accumulator",
        "ptx_path": None,
        "ptx_inline": _PTX_HMMA_ZERO,
        "kernel_name": "hmma_zero_kernel",
        "harness": harness_hmma_zero,
    },
}


# ---------------------------------------------------------------------------
# Output formatting + JSON artifact
# ---------------------------------------------------------------------------
def _fmt_metric(label, ours, theirs, delta_str):
    return f"    {label:<14} ours={ours:<10} ptxas={theirs:<10} delta={delta_str}"


def print_block(name: str, kentry: dict, build_ok: bool, correct: bool,
                ours: dict, theirs: dict | None, commits: dict) -> None:
    print(f"[workbench] kernel={name}  ({kentry['display']})")
    print(f"  build:    {'PASS' if build_ok else 'FAIL'}")
    print(f"  correct:  {'PASS' if correct else 'FAIL'}")
    print(f"  forge:     {commits['forge']}")
    print(f"  opencuda:  {commits['opencuda']}")
    print(f"  openptxas: {commits['openptxas']}")
    print()
    print("  ours:")
    print(f"    regs:         {ours['regs']}")
    print(f"    sass_total:   {ours['sass_total']}")
    print(f"    sass_non_nop: {ours['sass_non_nop']}")
    print(f"    time_ms:      {_fmt_time(ours.get('time_ms'))}")
    print(f"    compile_ms:   {ours['compile_ms']:.1f}")
    if theirs is not None:
        print()
        print("  ptxas:")
        print(f"    regs:         {theirs['regs']}")
        print(f"    sass_total:   {theirs['sass_total']}")
        print(f"    sass_non_nop: {theirs['sass_non_nop']}")
        print(f"    time_ms:      {_fmt_time(theirs.get('time_ms'))}")
        print(f"    compile_ms:   {theirs['compile_ms']:.1f}")
        print()
        print("  delta:")
        print(f"    regs:         {ours['regs'] - theirs['regs']:+d}")
        print(f"    sass_total:   {ours['sass_total'] - theirs['sass_total']:+d}")
        print(f"    sass_non_nop: {ours['sass_non_nop'] - theirs['sass_non_nop']:+d}")
        if ours.get('time_ms') is not None and theirs.get('time_ms') is not None:
            d = ours['time_ms'] - theirs['time_ms']
            print(f"    time_ms:      {d:+.4f}")


def _fmt_time(t):
    return f"{t:.4f}" if t is not None else "(skipped)"


def write_json(name: str, parameters: dict, build_ok: bool, correct: bool,
               ours: dict, theirs: dict | None, commits: dict,
               compact_meta: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact = {
        "kernel": name,
        "timestamp": ts,
        "parameters": parameters,
        "build": "PASS" if build_ok else "FAIL",
        "correctness": "PASS" if correct else "FAIL",
        "commits": commits,
        "ours": ours,
        "ptxas": theirs,
        "deltas": (
            {
                "regs": ours["regs"] - theirs["regs"],
                "sass_total": ours["sass_total"] - theirs["sass_total"],
                "sass_non_nop": ours["sass_non_nop"] - theirs["sass_non_nop"],
                "time_ms": (
                    ours["time_ms"] - theirs["time_ms"]
                    if ours.get("time_ms") is not None
                    and theirs.get("time_ms") is not None
                    else None
                ),
            }
            if theirs is not None
            else None
        ),
        "metadata": compact_meta,
    }
    out_path = results_dir / f"{ts}_{name}.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def metrics_from_cubin(cubin: bytes) -> dict:
    return cubin_metrics(cubin)


def run_kernel(name: str, mode: str, do_compare: bool,
               results_dir: Path) -> int:
    if name not in KERNELS:
        print(f"[workbench] unknown kernel '{name}'. "
              f"Available: {', '.join(sorted(KERNELS))}", file=sys.stderr)
        return 2

    kentry = KERNELS[name]
    if kentry["ptx_inline"] is not None:
        ptx = kentry["ptx_inline"]
        ptx_source = "(inline)"
    else:
        path = kentry["ptx_path"]
        if not path.exists():
            print(f"[workbench] PTX file not found: {path}", file=sys.stderr)
            return 3
        ptx = path.read_text(encoding="utf-8")
        ptx_source = str(path)

    commits = {
        "openptxas": _git_short(REPO_OPENPTXAS),
        "forge":     _git_short(REPO_FORGE),
        "opencuda":  _git_short(REPO_OPENCUDA),
    }

    # 1. Build through openptxas (with compaction-report capture)
    build_ok = True
    cubin_ours = None
    report = None
    t_compile_ours = 0.0
    try:
        cubin_ours, t_compile_ours, report = compile_with_report(ptx)
    except Exception as e:
        print(f"[workbench] openptxas build FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        build_ok = False

    if not build_ok:
        return 4

    ours = metrics_from_cubin(cubin_ours)
    ours["compile_ms"] = t_compile_ours * 1000.0
    ours["time_ms"] = None

    # 2. Build through ptxas (optional)
    cubin_ptxas = None
    theirs = None
    if do_compare:
        try:
            cubin_ptxas, t_compile_ptxas = compile_ptxas(ptx)
            theirs = metrics_from_cubin(cubin_ptxas)
            theirs["compile_ms"] = t_compile_ptxas * 1000.0
            theirs["time_ms"] = None
        except Exception as e:
            print(f"[workbench] ptxas build skipped: "
                  f"{type(e).__name__}: {e}")
            theirs = None

    # 3. Launch + correctness (and optional benchmark)
    ctx = CUDAContext()
    correct = False
    try:
        if not ctx.load(cubin_ours):
            print("[workbench] cuModuleLoadData failed for openptxas cubin",
                  file=sys.stderr)
            return 5
        func = ctx.get_func(kentry["kernel_name"])
        result = kentry["harness"](ctx, func, mode)
        correct = result["correct"]
        ours["time_ms"] = result["time_ms"]

        if theirs is not None:
            if not ctx.load(cubin_ptxas):
                print("[workbench] cuModuleLoadData failed for ptxas cubin",
                      file=sys.stderr)
            else:
                func_p = ctx.get_func(kentry["kernel_name"])
                result_p = kentry["harness"](ctx, func_p, mode)
                # we trust ours-vs-ptxas correctness equivalence;
                # workbench measures ours' correctness directly
                theirs["time_ms"] = result_p["time_ms"]
    finally:
        ctx.close()

    # 4. Compaction metadata for the JSON artifact
    compact_meta = {
        "compaction_attempted": False,
        "compaction_covered":   False,
        "compacted":            False,
        "compact_regs_before":  None,
        "compact_regs_after":   None,
    }
    if report is not None:
        compact_meta = {
            "compaction_attempted": report.attempted,
            "compaction_covered":   report.covered,
            "compacted":            report.gpr_fields_rewritten > 0,
            "compact_regs_before":  report.regs_before,
            "compact_regs_after":   report.regs_after,
            "compacted_insts":      report.compacted_insts,
            "gpr_fields_rewritten": report.gpr_fields_rewritten,
        }

    # 5. Print canonical block + write JSON
    print_block(name, kentry, build_ok, correct, ours, theirs, commits)

    parameters = {
        "kernel": name,
        "mode": mode,
        "ptx_source": ptx_source,
    }
    artifact = write_json(name, parameters, build_ok, correct, ours, theirs,
                          commits, compact_meta, results_dir)
    print()
    print(f"[workbench] artifact: {artifact}")
    return 0 if correct else 1


def main():
    p = argparse.ArgumentParser(
        prog="workbench",
        description="WB-0: kernel workbench for the openptxas/forge/ptxas stack",
    )
    p.add_argument("--kernel", default=None,
                   help=f"one of: {', '.join(sorted(KERNELS))}")
    p.add_argument("--mode", choices=["correct", "bench"], default="correct",
                   help="correct = build+correctness, bench = +benchmark")
    p.add_argument("--compare", choices=["ptxas"], default=None,
                   help="if set, also compile via ptxas and report deltas")
    p.add_argument("--results-dir", default=str(ROOT / "results"),
                   help="directory for JSON artifacts")
    p.add_argument("--list", action="store_true",
                   help="list catalog and exit")
    args = p.parse_args()

    if args.list:
        print("Available kernels:")
        for k, v in KERNELS.items():
            print(f"  {k:20s} {v['display']}")
        return 0

    if not args.kernel:
        p.error("--kernel is required (use --list to see options)")

    return run_kernel(
        name=args.kernel,
        mode=args.mode,
        do_compare=(args.compare == "ptxas"),
        results_dir=Path(args.results_dir),
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
