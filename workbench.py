"""
WB-0: Kernel Workbench MVP  (subcommand CLI as of WB-12.0)

CLI cockpit for the openptxas → forge → ptxas stack.

Examples
--------
  python workbench.py run --kernel reduce_sum
  python workbench.py run --kernel conv2d_looped --compare ptxas
  python workbench.py run --kernel hmma_zero --compare ptxas --mode bench
  python workbench.py run --suite all --compare ptxas --mode bench
  python workbench.py list

The workbench:
  • compiles a known PTX through openptxas
  • optionally compiles the same PTX through ptxas
  • launches the kernel on the GPU and verifies correctness
  • collects regs / sass_total / sass_non_nop / time_ms for both
  • prints a canonical block
  • writes a JSON artifact to results/<ts>_<kernel>.json

Subcommands
-----------
  run       run a kernel or suite (current behavior)
  list      list catalog and suites
  show      kernel detail (stub — WB-12.2)
  status    current leaderboard (stub — WB-12.1)
  dump      side-by-side SASS dump (stub — WB-12.3)
  history   metric history walk (stub — WB-12.4)
  diff      diff two artifacts (stub — WB-12.5)

WB-0 is intentionally narrow: hardcoded catalog, no kernel editor, no
GUI, no AI, no plugin system.  Each catalog entry just points at a PTX
file (or inline string), names the entry symbol, and supplies a
correctness/benchmark harness.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev

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
# WB-6 catalog expansion — broader stress kernels.
# ---------------------------------------------------------------------------

# conv2d_unrolled — 9 tap fully-unrolled 3x3 conv (structural contrast vs
# the looped variant; same semantics, different liveness shape).
_PTX_CONV2D_UNROLLED_PATH = REPO_FORGE / "benchmarks" / "fb0_baseline" / "conv2d_open.ptx"


def harness_conv2d_unrolled(ctx: CUDAContext, func, mode: str) -> dict:
    """Same harness as conv2d_looped — 128x128 u64 grid, all-ones input."""
    return harness_conv2d_looped(ctx, func, mode)


# vecadd_large — 1M-thread u64 vector add with bounds check (memory-bound).
# Stresses the multi-param + multi-LDG path.
_PTX_VECADD_LARGE = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry vecadd_large(
    .param .u64 p_out,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u32 p_n
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<10>;
    .reg .pred %p<2>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.u32 %r3, %r1, %r2, %r0;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 ret;

    shl.b32 %r5, %r3, 2;
    cvt.u64.u32 %rd0, %r5;

    ld.param.u64 %rd1, [p_a];
    add.u64 %rd2, %rd1, %rd0;
    ld.global.u32 %r6, [%rd2];

    ld.param.u64 %rd3, [p_b];
    add.u64 %rd4, %rd3, %rd0;
    ld.global.u32 %r7, [%rd4];

    add.u32 %r6, %r6, %r7;

    ld.param.u64 %rd5, [p_out];
    add.u64 %rd6, %rd5, %rd0;
    st.global.u32 [%rd6], %r6;
    ret;
}
"""


def harness_vecadd_large(ctx: CUDAContext, func, mode: str) -> dict:
    """1<<20 element u32 vector add: a[i]+b[i] -> c[i]; verify a sample."""
    N = 1 << 20
    block = 256
    grid = (N + block - 1) // block

    d_a = ctx.alloc(N * 4)
    d_b = ctx.alloc(N * 4)
    d_out = ctx.alloc(N * 4)
    try:
        a_host = (ctypes.c_uint32 * N)(*[i for i in range(N)])
        b_host = (ctypes.c_uint32 * N)(*[i * 2 for i in range(N)])
        ctx.copy_to(d_a, bytes(a_host))
        ctx.copy_to(d_b, bytes(b_host))
        ctx.copy_to(d_out, b"\x00" * (N * 4))

        a_out = ctypes.c_uint64(d_out)
        a_a   = ctypes.c_uint64(d_a)
        a_b   = ctypes.c_uint64(d_b)
        a_n   = ctypes.c_uint32(N)
        args, _hold = _make_args(a_out, a_a, a_b, a_n)
        ctx.cuda.cuLaunchKernel(func, grid, 1, 1, block, 1, 1,
                                0, None, args, None)
        assert ctx.sync() == 0, "vecadd_large kernel crashed"

        # Verify first 1024 + last 1024
        first_raw = ctx.copy_from(d_out, 1024 * 4)
        first = struct.unpack(f"<{1024}I", first_raw)
        correct = all(first[i] == i * 3 for i in range(1024))
        if correct:
            tail_off = (N - 1024) * 4
            last_raw = ctx.copy_from(d_out + tail_off, 1024 * 4)
            last = struct.unpack(f"<{1024}I", last_raw)
            correct = all(last[j] == (N - 1024 + j) * 3 for j in range(1024))

        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (grid, 1, 1), (block, 1, 1), args
            )
    finally:
        ctx.free(d_a)
        ctx.free(d_b)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# multi_ldg — load+add aliasing pattern (the canary that exposed
# FB-5.1's address-pair quarantine bug).
_PTX_MULTI_LDG = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry multi_ldg_test(.param .u64 pin, .param .u64 pout) {
    .reg .u32 %r<8>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd0, [pin];
    ld.param.u64 %rd1, [pout];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd2, %r1;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];
    add.u64 %rd4, %rd3, 4;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f0, %f1;
    add.u64 %rd5, %rd1, %rd2;
    st.global.f32 [%rd5], %f2;
    ret;
}
"""


def harness_multi_ldg(ctx: CUDAContext, func, mode: str) -> dict:
    """Each thread reads in[i] + in[i+1], writes to out[i]."""
    N = 4
    in_vals = [float(i + 1) for i in range(N + 1)]
    d_in = ctx.alloc((N + 1) * 4)
    d_out = ctx.alloc(N * 4)
    try:
        ctx.copy_to(d_in, struct.pack(f"<{N+1}f", *in_vals))
        ctx.copy_to(d_out, b"\x00" * (N * 4))
        a_in  = ctypes.c_uint64(d_in)
        a_out = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_in, a_out)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "multi_ldg kernel crashed"
        out = struct.unpack(f"<{N}f", ctx.copy_from(d_out, N * 4))
        correct = all(out[i] == in_vals[i] + in_vals[i + 1] for i in range(N))
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (N, 1, 1), args)
    finally:
        ctx.free(d_in)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# smem_exchange — shared memory write/barrier/read roundtrip.
_PTX_SMEM_EXCHANGE = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry smem_exchange(
    .param .u64 p_out
)
{
    .reg .u32 %r<12>;
    .reg .u64 %rd<6>;
    .shared .align 4 .b32 smem[256];

    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    add.u32 %r2, %r0, 1;
    st.shared.b32 [%r1], %r2;
    bar.sync 0;

    add.u32 %r3, %r1, 4;
    sub.u32 %r4, %r1, 4;
    ld.shared.b32 %r5, [%r1];
    ld.param.u64 %rd0, [p_out];
    add.u64 %rd0, %rd0, 0;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r5;
    ret;
}
"""


def harness_smem_exchange(ctx: CUDAContext, func, mode: str) -> dict:
    """32 threads write tid+1 to smem, barrier, read own slot back, store."""
    N = 32
    d_out = ctx.alloc(N * 4)
    try:
        ctx.copy_to(d_out, b"\x00" * (N * 4))
        a_out = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_out)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 1024, None, args, None)
        assert ctx.sync() == 0, "smem_exchange kernel crashed"
        out = struct.unpack(f"<{N}I", ctx.copy_from(d_out, N * 4))
        correct = all(out[i] == i + 1 for i in range(N))
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (1, 1, 1), (N, 1, 1), args, smem=1024,
            )
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# atomg_add — atom.global.add.u32 (different atomic class than atom_or).
_PTX_ATOMG_ADD = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry atomg_add_test(
    .param .u64 p_out,
    .param .u32 p_addend
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<2>;

    ld.param.u64 %rd0, [p_out];
    ld.param.u32 %r0, [p_addend];
    atom.global.add.u32 %r1, [%rd0], %r0;
    ret;
}
"""


def harness_atomg_add(ctx: CUDAContext, func, mode: str) -> dict:
    """32 threads each atomic-add 1 → counter = 32."""
    d = ctx.alloc(4)
    try:
        ctx.copy_to(d, struct.pack("<I", 0))
        a_out = ctypes.c_uint64(d)
        a_add = ctypes.c_uint32(1)
        args, _hold = _make_args(a_out, a_add)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "atomg_add kernel crashed"
        val = struct.unpack("<I", ctx.copy_from(d, 4))[0]
        correct = (val == 32)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (32, 1, 1), args)
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": time_ms}


# fmax_kernel — scalar ALU sanity (FMNMX path).
_PTX_FMAX = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry fmax_test(.param .u64 p_out, .param .u64 p_a, .param .u64 p_b) {
    .reg .u64 %rd<4>; .reg .f32 %f<4>;
    ld.param.u64 %rd0, [p_a]; ld.global.f32 %f0, [%rd0];
    ld.param.u64 %rd1, [p_b]; ld.global.f32 %f1, [%rd1];
    max.f32 %f2, %f0, %f1;
    ld.param.u64 %rd2, [p_out];
    st.global.f32 [%rd2], %f2;
    ret;
}
"""


def harness_fmax(ctx: CUDAContext, func, mode: str) -> dict:
    """max(a, b) for f32 scalars; expect b's larger value."""
    d_a = ctx.alloc(4)
    d_b = ctx.alloc(4)
    d_out = ctx.alloc(4)
    try:
        ctx.copy_to(d_a, struct.pack("<f", 3.5))
        ctx.copy_to(d_b, struct.pack("<f", 7.25))
        ctx.copy_to(d_out, b"\x00" * 4)
        a_out = ctypes.c_uint64(d_out)
        a_a   = ctypes.c_uint64(d_a)
        a_b   = ctypes.c_uint64(d_b)
        args, _hold = _make_args(a_out, a_a, a_b)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "fmax kernel crashed"
        result = struct.unpack("<f", ctx.copy_from(d_out, 4))[0]
        correct = (result == 7.25)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (1, 1, 1), args)
    finally:
        ctx.free(d_a)
        ctx.free(d_b)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# WB-9 frontier kernels — broader sampling of memory + atomic + sync
# patterns to find out whether vecadd_large is the *last* real GAP or
# just the last gap in the current 15-kernel catalog.
# ---------------------------------------------------------------------------

# smem_cycle — single-warp shared-memory write/barrier/read cycle.
# Different shape than smem_exchange (uses param-loaded base value).
_PTX_SMEM_CYCLE = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry smem_cycle(
    .param .u64 p_out,
    .param .u32 p_val
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<4>;
    .shared .align 4 .b32 smem[256];

    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;

    ld.param.u32 %r2, [p_val];
    add.u32 %r2, %r2, %r0;

    st.shared.b32 [%r1], %r2;
    bar.sync 0;

    ld.shared.b32 %r3, [%r1];

    ld.param.u64 %rd0, [p_out];
    add.u64 %rd0, %rd0, 0;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r3;
    ret;
}
"""


def harness_smem_cycle(ctx: CUDAContext, func, mode: str) -> dict:
    """Each thread writes (param+tid) to smem, barrier, reads back, stores."""
    N = 32
    base_val = 100
    d_out = ctx.alloc(N * 4)
    try:
        ctx.copy_to(d_out, b"\x00" * (N * 4))
        a_out = ctypes.c_uint64(d_out)
        a_val = ctypes.c_uint32(base_val)
        args, _hold = _make_args(a_out, a_val)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 256 * 4, None, args, None)
        assert ctx.sync() == 0, "smem_cycle kernel crashed"
        out = struct.unpack(f"<{N}I", ctx.copy_from(d_out, N * 4))
        correct = all(out[i] == base_val + i for i in range(N))
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (1, 1, 1), (N, 1, 1), args, smem=256 * 4,
            )
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# bar_ldc_xor — barrier + late LDC of param + XOR.
# Tests the LDC-after-bar-sync correctness path that historically was
# poison-prone (FB-3 era bug).
_PTX_BAR_LDC_XOR = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry bar_ldc_xor(
    .param .u64 p_out,
    .param .u32 p_n,
    .param .u32 p_mask
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<4>;
    .reg .pred %p0;
    .shared .align 4 .b32 smem[256];

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r5, [p_n];
    setp.ge.u32 %p0, %r0, %r5;
    @%p0 bra DONE;

    shl.b32 %r1, %r0, 2;
    add.u32 %r2, %r0, 42;
    st.shared.b32 [%r1], %r2;
    bar.sync 0;

    ld.param.u32 %r3, [p_mask];
    xor.b32 %r4, %r2, %r3;

    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
DONE:
    ret;
}
"""


def harness_bar_ldc_xor(ctx: CUDAContext, func, mode: str) -> dict:
    """tid+42 -> shared, barrier, then xor with mask, store. Verify."""
    N = 32
    mask = 0x55
    d_out = ctx.alloc(N * 4)
    try:
        ctx.copy_to(d_out, b"\x00" * (N * 4))
        a_out  = ctypes.c_uint64(d_out)
        a_n    = ctypes.c_uint32(N)
        a_mask = ctypes.c_uint32(mask)
        args, _hold = _make_args(a_out, a_n, a_mask)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 256 * 4, None, args, None)
        assert ctx.sync() == 0, "bar_ldc_xor kernel crashed"
        out = struct.unpack(f"<{N}I", ctx.copy_from(d_out, N * 4))
        correct = all(out[tid] == ((tid + 42) ^ mask) & 0xFFFFFFFF
                      for tid in range(N))
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (1, 1, 1), (N, 1, 1), args, smem=256 * 4,
            )
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# dual_ldg64_dadd — two LDG.E.64 -> DADD -> STG.E.64.
# FP64 sibling of multi_ldg.  Tests that the second LDG isn't zeroed
# by scoreboard collision (the FB-3 LDG dual-load bug).
_PTX_DUAL_LDG64_DADD = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry dual_ldg64_dadd(
    .param .u64 p_out,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u32 p_n
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<16>;
    .reg .f64 %fd<4>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [p_n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;

    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 3;

    ld.param.u64 %rd1, [p_a];
    add.u64 %rd2, %rd1, %rd0;
    ld.global.f64 %fd0, [%rd2];

    ld.param.u64 %rd3, [p_b];
    add.u64 %rd4, %rd3, %rd0;
    ld.global.f64 %fd1, [%rd4];

    add.f64 %fd2, %fd0, %fd1;

    ld.param.u64 %rd5, [p_out];
    add.u64 %rd6, %rd5, %rd0;
    st.global.f64 [%rd6], %fd2;
DONE:
    ret;
}
"""


def harness_dual_ldg64_dadd(ctx: CUDAContext, func, mode: str) -> dict:
    """Per-thread f64 a[i] + b[i]; verify."""
    N = 32
    a_vals = [float(i) * 1.5 + 100.0 for i in range(N)]
    b_vals = [float(i) * 2.5 + 200.0 for i in range(N)]
    d_a = ctx.alloc(N * 8)
    d_b = ctx.alloc(N * 8)
    d_out = ctx.alloc(N * 8)
    try:
        ctx.copy_to(d_a, struct.pack(f"<{N}d", *a_vals))
        ctx.copy_to(d_b, struct.pack(f"<{N}d", *b_vals))
        ctx.copy_to(d_out, b"\x00" * (N * 8))
        a_out = ctypes.c_uint64(d_out)
        a_a   = ctypes.c_uint64(d_a)
        a_b   = ctypes.c_uint64(d_b)
        a_n   = ctypes.c_uint32(N)
        args, _hold = _make_args(a_out, a_a, a_b, a_n)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "dual_ldg64_dadd kernel crashed"
        results = struct.unpack(f"<{N}d", ctx.copy_from(d_out, N * 8))
        correct = all(abs(results[i] - (a_vals[i] + b_vals[i])) < 1e-9
                      for i in range(N))
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (N, 1, 1), args)
    finally:
        ctx.free(d_a)
        ctx.free(d_b)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# multi_block_atomic — 64 blocks x 256 threads each atomic-add 1.
# Grid-wide contention pattern (vs the single-warp atom_or / atomg_add).
_PTX_MULTI_BLOCK_ATOMIC = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry multi_block_atomic(
    .param .u64 p_counter
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd0, [p_counter];
    add.u64 %rd0, %rd0, 0;

    mov.u32 %r0, 1;
    atom.global.add.u32 %r1, [%rd0], %r0;
    ret;
}
"""


def harness_multi_block_atomic(ctx: CUDAContext, func, mode: str) -> dict:
    """64 blocks * 256 threads atomic-add 1 -> counter == 16384."""
    num_blocks = 64
    block_size = 256
    expected = num_blocks * block_size
    d = ctx.alloc(4)
    try:
        ctx.copy_to(d, struct.pack("<I", 0))
        a_d = ctypes.c_uint64(d)
        args, _hold = _make_args(a_d)
        ctx.cuda.cuLaunchKernel(func, num_blocks, 1, 1, block_size, 1, 1,
                                0, None, args, None)
        assert ctx.sync() == 0, "multi_block_atomic kernel crashed"
        val = struct.unpack("<I", ctx.copy_from(d, 4))[0]
        correct = (val == expected)
        time_ms = None
        if mode == "bench":
            # Reset between runs is too expensive; just time a fresh launch.
            time_ms = _bench_launch(
                ctx, func, (num_blocks, 1, 1), (block_size, 1, 1), args
            )
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": time_ms}


# atom_cas64 — 64-bit compare-and-swap.  Distinct atomic class from
# atom.add / atom.or; exercises the CAS-64 encoding path.
_PTX_ATOM_CAS64 = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry atom_cas64_test(
    .param .u64 p_addr,
    .param .u64 p_cmp,
    .param .u64 p_new,
    .param .u64 p_out
)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<4>;

    ld.param.u64 %rd0, [p_addr];
    ld.param.u64 %rd1, [p_cmp];
    ld.param.u64 %rd2, [p_new];

    add.u64 %rd0, %rd0, 0;
    add.u64 %rd1, %rd1, 0;
    add.u64 %rd2, %rd2, 0;

    atom.global.cas.b64 %rd3, [%rd0], %rd1, %rd2;
    ld.param.u64 %rd4, [p_out];
    st.global.u64 [%rd4], %rd3;
    ret;
}
"""


def harness_atom_cas64(ctx: CUDAContext, func, mode: str) -> dict:
    """Successful CAS: returns old value, mem becomes new."""
    old_val = 0xDEADBEEFCAFEBABE
    cmp_val = 0xDEADBEEFCAFEBABE
    new_val = 0x1234567890ABCDEF
    d_addr = ctx.alloc(8)
    d_out = ctx.alloc(8)
    try:
        ctx.copy_to(d_addr, struct.pack("<Q", old_val))
        ctx.copy_to(d_out, struct.pack("<Q", 0))
        a_addr = ctypes.c_uint64(d_addr)
        a_cmp  = ctypes.c_uint64(cmp_val)
        a_new  = ctypes.c_uint64(new_val)
        a_out  = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_addr, a_cmp, a_new, a_out)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "atom_cas64 kernel crashed"
        returned = struct.unpack("<Q", ctx.copy_from(d_out, 8))[0]
        mem_now  = struct.unpack("<Q", ctx.copy_from(d_addr, 8))[0]
        correct = (returned == old_val and mem_now == new_val)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (1, 1, 1), args)
    finally:
        ctx.free(d_addr)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# redux_sum — REDUX.SYNC.ADD warp aggregation (vs shfl-based warp_reduce).
_PTX_REDUX_SUM = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry redux_sum_kernel(
    .param .u64 p_out,
    .param .u32 p_val
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<2>;

    ld.param.u64    %rd0, [p_out];
    ld.param.u32    %r0, [p_val];

    redux.sync.add.s32 %r1, %r0, 0xffffffff;

    st.global.u32 [%rd0], %r1;
    ret;
}
"""


def harness_redux_sum(ctx: CUDAContext, func, mode: str) -> dict:
    """Single thread: redux.sync.add.s32 of 1 lane == input value."""
    p_val = 42
    d_out = ctx.alloc(4)
    try:
        ctx.copy_to(d_out, b"\x00\x00\x00\x00")
        a_out = ctypes.c_uint64(d_out)
        a_val = ctypes.c_uint32(p_val)
        args, _hold = _make_args(a_out, a_val)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "redux_sum kernel crashed"
        result = struct.unpack("<I", ctx.copy_from(d_out, 4))[0]
        correct = (result == p_val)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (1, 1, 1), args)
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# IMMA / DMMA / QMMA zero kernels (sibling tensor cores).  Same all-zero
# input pattern as hmma_zero — exercises every tensor backend variant.
# ---------------------------------------------------------------------------
_PTX_IMMA_ZERO = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry imma_zero_kernel(.param .u64 p_out)
{
    .reg .s32 %r<6>;
    .reg .u64 %rd<1>;

    ld.param.u64    %rd0, [p_out];

    mov.b32 %r4, 0;
    mov.b32 %r5, 0;
    mov.b32 %r0, 0;
    mov.b32 %r1, 0;
    mov.b32 %r2, 0;
    mov.b32 %r3, 0;

    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r0, %r1, %r2, %r3},
        {%r0, %r1, %r2, %r3},
        {%r4, %r5},
        {%r0, %r1, %r2, %r3};

    st.global.u32 [%rd0], %r0;
    ret;
}
"""


def harness_imma_zero(ctx: CUDAContext, func, mode: str) -> dict:
    """IMMA.16832.S8 with all-zero inputs — output must be 0."""
    d_out = ctx.alloc(4)
    try:
        ctx.copy_to(d_out, b"\x00\x00\x00\x00")
        a_out = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_out)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "imma kernel crashed"
        result = struct.unpack("<i", ctx.copy_from(d_out, 4))[0]
        correct = (result == 0)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (32, 1, 1), args)
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


_PTX_DMMA_ZERO = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry dmma_zero_kernel(.param .u64 p_out)
{
    .reg .f64 %fd<4>;
    .reg .u64 %rd<1>;

    ld.param.u64    %rd0, [p_out];

    mov.f64 %fd0, 0d0000000000000000;
    mov.f64 %fd1, 0d0000000000000000;
    mov.f64 %fd2, 0d0000000000000000;
    mov.f64 %fd3, 0d0000000000000000;

    mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
        {%fd0, %fd1},
        {%fd2},
        {%fd3},
        {%fd0, %fd1};

    st.global.f64 [%rd0], %fd0;
    ret;
}
"""


def harness_dmma_zero(ctx: CUDAContext, func, mode: str) -> dict:
    """DMMA.8x8x4 with all-zero inputs — output must be 0.0."""
    d_out = ctx.alloc(8)
    try:
        ctx.copy_to(d_out, b"\x00" * 8)
        a_out = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_out)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "dmma kernel crashed"
        result = struct.unpack("<d", ctx.copy_from(d_out, 8))[0]
        correct = (result == 0.0)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (32, 1, 1), args)
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


_PTX_QMMA_ZERO = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry qmma_zero_kernel(.param .u64 p_out)
{
    .reg .b32 %r<8>;
    .reg .u64 %rd<1>;

    ld.param.u64    %rd0, [p_out];

    mov.b32 %r4, 0;
    mov.b32 %r5, 0;
    mov.b32 %r6, 0;
    mov.b32 %r7, 0;
    mov.b32 %r0, 0;
    mov.b32 %r1, 0;
    mov.b32 %r2, 0;
    mov.b32 %r3, 0;

    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
        {%r0, %r1, %r2, %r3},
        {%r0, %r1, %r2, %r3},
        {%r4, %r5},
        {%r0, %r1, %r2, %r3};

    st.global.u32 [%rd0], %r0;
    ret;
}
"""


def harness_qmma_zero(ctx: CUDAContext, func, mode: str) -> dict:
    """QMMA.16832.F32.E4M3.E4M3 with all-zero inputs — output must be 0.0f."""
    d_out = ctx.alloc(4)
    try:
        ctx.copy_to(d_out, b"\x00\x00\x00\x00")
        a_out = ctypes.c_uint64(d_out)
        args, _hold = _make_args(a_out)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "qmma kernel crashed"
        result = struct.unpack("<f", ctx.copy_from(d_out, 4))[0]
        correct = (result == 0.0)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (32, 1, 1), args)
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# cp.async — async global → shared copy with commit/wait, then broadcast.
# Exercises LDGSTS, BAR.SYNC, shared-memory load.
# ---------------------------------------------------------------------------
_PTX_CP_ASYNC = """
.version 8.7
.target sm_120
.address_size 64

.visible .entry cp_async_test(
    .param .u64 p_out,
    .param .u64 p_in
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .pred %p0;
    .shared .align 4 .b32 smem[256];

    mov.u32 %r0, %tid.x;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra SKIP_COPY;

    mov.u32 %r1, 0;
    ld.param.u64 %rd0, [p_in];
    cp.async.ca.shared.global [%r1], [%rd0], 4;

SKIP_COPY:
    cp.async.commit_group;
    cp.async.wait_group 0;
    bar.sync 0;

    mov.u32 %r2, 0;
    ld.shared.b32 %r3, [%r2];

    shl.b32 %r4, %r0, 2;
    cvt.u64.u32 %rd1, %r4;
    ld.param.u64 %rd2, [p_out];
    add.u64 %rd3, %rd2, %rd1;
    st.global.u32 [%rd3], %r3;
    ret;
}
"""


def harness_cp_async(ctx: CUDAContext, func, mode: str) -> dict:
    """cp.async copy of 4B from global to shared, broadcast across warp."""
    N = 32
    magic = 0xDEADBEEF
    d_in = ctx.alloc(4)
    d_out = ctx.alloc(N * 4)
    try:
        ctx.copy_to(d_in, struct.pack("<I", magic))
        ctx.copy_to(d_out, b"\x00" * (N * 4))
        a_out = ctypes.c_uint64(d_out)
        a_in  = ctypes.c_uint64(d_in)
        args, _hold = _make_args(a_out, a_in)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 1024, None, args, None)
        assert ctx.sync() == 0, "cp_async kernel crashed"
        results = struct.unpack(f"<{N}I", ctx.copy_from(d_out, N * 4))
        correct = all(v == magic for v in results)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(
                ctx, func, (1, 1, 1), (N, 1, 1), args, smem=1024,
            )
    finally:
        ctx.free(d_in)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# warp_reduce — fp32 warp-level butterfly via shfl.down (5 stages).
# Exercises SHFL scoreboard slots and the warp shuffle path.
# ---------------------------------------------------------------------------
_PTX_WARP_REDUCE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry warp_reduce(
    .param .u64 p_out, .param .u64 p_in, .param .u32 n)
{
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<4>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    cvt.u64.u32 %rd0, %r0; shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [p_in]; add.u64 %rd2, %rd1, %rd0;
    ld.global.f32 %f0, [%rd2];

    shfl.sync.down.b32 %f1, %f0, 16, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;
    shfl.sync.down.b32 %f1, %f0, 8, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;
    shfl.sync.down.b32 %f1, %f0, 4, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;
    shfl.sync.down.b32 %f1, %f0, 2, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;
    shfl.sync.down.b32 %f1, %f0, 1, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %f1;

    setp.ne.u32 %p0, %r0, 0; @%p0 ret;
    ld.param.u64 %rd3, [p_out];
    st.global.f32 [%rd3], %f0;
    ret;
}
"""


def harness_warp_reduce(ctx: CUDAContext, func, mode: str) -> dict:
    """Warp-level fp32 sum of [1.0, 2.0, ..., 32.0] = 528.0."""
    N = 32
    expected = float(N * (N + 1) // 2)  # 528
    vals = [float(i + 1) for i in range(N)]
    d_in = ctx.alloc(N * 4)
    d_out = ctx.alloc(4)
    try:
        ctx.copy_to(d_in, struct.pack(f"<{N}f", *vals))
        ctx.copy_to(d_out, b"\x00\x00\x00\x00")
        a_out = ctypes.c_uint64(d_out)
        a_in  = ctypes.c_uint64(d_in)
        a_n   = ctypes.c_uint32(N)
        args, _hold = _make_args(a_out, a_in, a_n)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "warp_reduce kernel crashed"
        result = struct.unpack("<f", ctx.copy_from(d_out, 4))[0]
        correct = (result == expected)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (N, 1, 1), args)
    finally:
        ctx.free(d_in)
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


# ---------------------------------------------------------------------------
# atom_or — single-warp atomic OR into a global location.
# Exercises ATOMG.E.OR.b32 path (non-tensor, non-shared).
# ---------------------------------------------------------------------------
_PTX_ATOM_OR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry atom_or(.param .u64 p_out) {
    .reg .u32 %r<4>; .reg .u64 %rd<2>;
    mov.u32 %r1, 0xFF;
    ld.param.u64 %rd0, [p_out];
    atom.global.or.b32 %r0, [%rd0], %r1;
    ret;
}
"""


def harness_atom_or(ctx: CUDAContext, func, mode: str) -> dict:
    """32 lanes each OR 0xFF into the same word — final value must be 0xFF."""
    d = ctx.alloc(4)
    try:
        ctx.copy_to(d, struct.pack("<I", 0))
        a = ctypes.c_uint64(d)
        args, _hold = _make_args(a)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, args, None)
        assert ctx.sync() == 0, "atom_or kernel crashed"
        val = struct.unpack("<I", ctx.copy_from(d, 4))[0]
        correct = (val == 0xFF)
        time_ms = None
        if mode == "bench":
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (32, 1, 1), args)
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": time_ms}


# ===========================================================================
# PERF-4: ILP benchmark PTX sources + harnesses
# ===========================================================================
# Each kernel has TWO or more independent instruction chains so the
# scheduler has opportunities to fill latency NOPs with useful work
# from the other chain.  All kernels write out[tid.x] = f(tid.x).

_PTX_ILP_DUAL_INT32 = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry ilp_dual_int32(
    .param .u64 p_out, .param .u32 n)
{
    .reg .u32 %r<12>;
    .reg .u64 %rd<4>;
    .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // Chain A: a = ((tid * 3) + 7) ^ 0xABCD
    mul.lo.u32 %r2, %r0, 3;
    // Chain B: b = ((tid * 5) + 13) ^ 0x1234  (independent of A)
    mul.lo.u32 %r5, %r0, 5;
    add.u32 %r3, %r2, 7;
    add.u32 %r6, %r5, 13;
    xor.b32 %r4, %r3, 0xABCD;
    xor.b32 %r7, %r6, 0x1234;
    // Merge
    add.u32 %r8, %r4, %r7;
    // Store out[tid]
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd1, %rd0, %rd1;
    st.global.u32 [%rd1], %r8;
    ret;
}
"""

def harness_ilp_dual_int32(ctx, func, _ptxas_func=None):
    N = 64; sz = N * 4
    d = ctx.alloc(sz); ctx.memset_d8(d, 0, sz)
    args, holders = _make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    time_ms = None
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, sz)
        correct = True
        for t in range(N):
            a = (((t * 3) + 7) ^ 0xABCD) & 0xFFFFFFFF
            b = (((t * 5) + 13) ^ 0x1234) & 0xFFFFFFFF
            expected = (a + b) & 0xFFFFFFFF
            got = struct.unpack_from('<I', buf, t * 4)[0]
            if got != expected:
                correct = False; break
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": time_ms}


_PTX_ILP_DUAL_INT64 = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry ilp_dual_int64(
    .param .u64 p_out, .param .u64 p_a, .param .u64 p_b, .param .u32 n)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<10>;
    .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    // Chain A: addr_a = p_a + tid*8, val_a = ld.global.u64 [addr_a]
    cvt.u64.u32 %rd3, %r0; shl.b64 %rd4, %rd3, 3;
    add.u64 %rd5, %rd1, %rd4;
    // Chain B: addr_b = p_b + tid*8 (independent)
    add.u64 %rd6, %rd2, %rd4;
    // Both loads independent
    ld.global.u64 %rd7, [%rd5];
    ld.global.u64 %rd8, [%rd6];
    // Merge: result = val_a + val_b
    add.u64 %rd9, %rd7, %rd8;
    // Store out[tid]
    add.u64 %rd5, %rd0, %rd4;
    st.global.u64 [%rd5], %rd9;
    ret;
}
"""

def harness_ilp_dual_int64(ctx, func, _ptxas_func=None):
    N = 32; sz8 = N * 8; sz_out = N * 8
    a_vals = [i * 100 + 7 for i in range(N)]
    b_vals = [i * 200 + 13 for i in range(N)]
    d_a = ctx.alloc(sz8); d_b = ctx.alloc(sz8); d_out = ctx.alloc(sz_out)
    ctx.copy_to(d_a, struct.pack(f'<{N}Q', *a_vals))
    ctx.copy_to(d_b, struct.pack(f'<{N}Q', *b_vals))
    ctx.memset_d8(d_out, 0, sz_out)
    args, holders = _make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_a),
                               ctypes.c_uint64(d_b), ctypes.c_uint32(N))
    time_ms = None
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d_out, sz_out)
        correct = True
        for t in range(N):
            expected = a_vals[t] + b_vals[t]
            got = struct.unpack_from('<Q', buf, t * 8)[0]
            if got != expected:
                correct = False; break
    finally:
        ctx.free(d_a); ctx.free(d_b); ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


_PTX_ILP_ALU_ADDR = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry ilp_alu_addr(
    .param .u64 p_out, .param .u32 n)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<4>;
    .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // Value chain (independent of address chain)
    mul.lo.u32 %r2, %r0, 7;
    add.u32 %r3, %r2, 42;
    xor.b32 %r4, %r3, 0xFF00;
    and.b32 %r5, %r4, 0xFFFF;
    // Address chain (independent of value chain)
    cvt.u64.u32 %rd1, %r0;
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    // Merge: store
    st.global.u32 [%rd2], %r5;
    ret;
}
"""

def harness_ilp_alu_addr(ctx, func, _ptxas_func=None):
    N = 64; sz = N * 4
    d = ctx.alloc(sz); ctx.memset_d8(d, 0, sz)
    args, holders = _make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    time_ms = None
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, sz)
        correct = True
        for t in range(N):
            expected = (((t * 7 + 42) ^ 0xFF00) & 0xFFFF) & 0xFFFFFFFF
            got = struct.unpack_from('<I', buf, t * 4)[0]
            if got != expected:
                correct = False; break
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": time_ms}


_PTX_ILP_UNROLLED_SUM4 = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry ilp_unrolled_sum4(
    .param .u64 p_out, .param .u64 p_data, .param .u32 n)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_data];
    // 4 independent loads (4 elements per thread, stride = n)
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.f32 %f0, [%rd3];
    add.u64 %rd4, %rd3, 256;
    ld.global.f32 %f1, [%rd4];
    add.u64 %rd5, %rd4, 256;
    ld.global.f32 %f2, [%rd5];
    add.u64 %rd6, %rd5, 256;
    ld.global.f32 %f3, [%rd6];
    // 4 independent accumulates (each add is independent)
    add.f32 %f4, %f0, %f1;
    add.f32 %f5, %f2, %f3;
    // Final merge
    add.f32 %f6, %f4, %f5;
    // Store
    add.u64 %rd7, %rd0, %rd2;
    st.global.f32 [%rd7], %f6;
    ret;
}
"""

def harness_ilp_unrolled_sum4(ctx, func, _ptxas_func=None):
    N = 64; STRIDE = 256 // 4  # 256 bytes = 64 floats
    total_elems = N + 3 * STRIDE
    data = [float(i % 100) for i in range(total_elems)]
    sz_data = total_elems * 4; sz_out = N * 4
    d_data = ctx.alloc(sz_data); d_out = ctx.alloc(sz_out)
    ctx.copy_to(d_data, struct.pack(f'<{total_elems}f', *data))
    ctx.memset_d8(d_out, 0, sz_out)
    args, holders = _make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_data),
                               ctypes.c_uint32(N))
    time_ms = None
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d_out, sz_out)
        correct = True
        for t in range(N):
            expected = data[t] + data[t + STRIDE] + data[t + 2*STRIDE] + data[t + 3*STRIDE]
            got = struct.unpack_from('<f', buf, t * 4)[0]
            if abs(got - expected) > 0.01:
                correct = False; break
    finally:
        ctx.free(d_data); ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


_PTX_ILP_PIPELINE_LOAD = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry ilp_pipeline_load(
    .param .u64 p_out, .param .u64 p_x, .param .u64 p_y, .param .u32 n)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<6>;
    .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_x];
    ld.param.u64 %rd2, [p_y];
    cvt.u64.u32 %rd3, %r0; shl.b64 %rd3, %rd3, 2;
    // Pipeline: issue load A, then load B, then compute A, compute B
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];     // load x[tid]
    add.u64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd5];     // load y[tid] (independent)
    // Compute on x (while y is in flight)
    mul.f32 %f2, %f0, 0f40400000;     // 3.0
    add.f32 %f3, %f2, 0f40E00000;     // 7.0
    // Compute on y (while x-compute runs)
    mul.f32 %f4, %f1, 0f40A00000;     // 5.0
    add.f32 %f5, %f4, 0f41500000;     // 13.0
    // Merge
    add.f32 %f2, %f3, %f5;
    // Store
    add.u64 %rd6, %rd0, %rd3;
    st.global.f32 [%rd6], %f2;
    ret;
}
"""

def harness_ilp_pipeline_load(ctx, func, _ptxas_func=None):
    N = 64; sz = N * 4
    x = [float(i) for i in range(N)]
    y = [float(i * 10) for i in range(N)]
    d_x = ctx.alloc(sz); d_y = ctx.alloc(sz); d_out = ctx.alloc(sz)
    ctx.copy_to(d_x, struct.pack(f'<{N}f', *x))
    ctx.copy_to(d_y, struct.pack(f'<{N}f', *y))
    ctx.memset_d8(d_out, 0, sz)
    args, holders = _make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_x),
                               ctypes.c_uint64(d_y), ctypes.c_uint32(N))
    time_ms = None
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d_out, sz)
        correct = True
        for t in range(N):
            expected = (x[t] * 3.0 + 7.0) + (y[t] * 5.0 + 13.0)
            got = struct.unpack_from('<f', buf, t * 4)[0]
            if abs(got - expected) > 0.1:
                correct = False; break
    finally:
        ctx.free(d_x); ctx.free(d_y); ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms}


_PTX_ILP_PRED_ALU = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry ilp_pred_alu(
    .param .u64 p_out, .param .u32 n)
{
    .reg .u32 %r<10>;
    .reg .u64 %rd<4>;
    .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // Chain A: val = tid * 7 + 42
    mul.lo.u32 %r2, %r0, 7;
    add.u32 %r3, %r2, 42;
    // Chain B: flag = (tid > 16)  (independent of A)
    setp.gt.u32 %p1, %r0, 16;
    // Chain C: bonus = tid * 3   (independent of A and B)
    mul.lo.u32 %r4, %r0, 3;
    // Merge: result = flag ? (val + bonus) : val
    // Use predicated add instead of selp (selp operand order varies)
    mov.u32 %r5, %r3;
    @%p1 add.u32 %r5, %r3, %r4;
    // Store
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r5;
    ret;
}
"""

def harness_ilp_pred_alu(ctx, func, _ptxas_func=None):
    N = 64; sz = N * 4
    d = ctx.alloc(sz); ctx.memset_d8(d, 0, sz)
    args, holders = _make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    time_ms = None
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, sz)
        correct = True
        for t in range(N):
            val = (t * 7 + 42) & 0xFFFFFFFF
            bonus = (t * 3) & 0xFFFFFFFF
            expected = ((val + bonus) & 0xFFFFFFFF) if t > 16 else val
            got = struct.unpack_from('<I', buf, t * 4)[0]
            if got != expected:
                correct = False; break
    finally:
        ctx.free(d)
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
    "imma_zero": {
        "display": "IMMA m16n8k32 S8 zero accumulator",
        "ptx_path": None,
        "ptx_inline": _PTX_IMMA_ZERO,
        "kernel_name": "imma_zero_kernel",
        "harness": harness_imma_zero,
    },
    "dmma_zero": {
        "display": "DMMA m8n8k4 F64 zero accumulator",
        "ptx_path": None,
        "ptx_inline": _PTX_DMMA_ZERO,
        "kernel_name": "dmma_zero_kernel",
        "harness": harness_dmma_zero,
    },
    "qmma_zero": {
        "display": "QMMA m16n8k32 E4M3 zero accumulator",
        "ptx_path": None,
        "ptx_inline": _PTX_QMMA_ZERO,
        "kernel_name": "qmma_zero_kernel",
        "harness": harness_qmma_zero,
    },
    "cp_async": {
        "display": "cp.async global->shared broadcast",
        "ptx_path": None,
        "ptx_inline": _PTX_CP_ASYNC,
        "kernel_name": "cp_async_test",
        "harness": harness_cp_async,
    },
    "warp_reduce": {
        "display": "warp_reduce fp32 shfl.down butterfly",
        "ptx_path": None,
        "ptx_inline": _PTX_WARP_REDUCE,
        "kernel_name": "warp_reduce",
        "harness": harness_warp_reduce,
    },
    "atom_or": {
        "display": "atom.global.or.b32",
        "ptx_path": None,
        "ptx_inline": _PTX_ATOM_OR,
        "kernel_name": "atom_or",
        "harness": harness_atom_or,
    },
    # WB-6 additions
    "conv2d_unrolled": {
        "display": "conv2d 3x3 fully-unrolled (u64)",
        "ptx_path": _PTX_CONV2D_UNROLLED_PATH,
        "ptx_inline": None,
        "kernel_name": "conv2d",
        "harness": harness_conv2d_unrolled,
    },
    "vecadd_large": {
        "display": "vecadd_large (1M-thread, 4-param, bounds check)",
        "ptx_path": None,
        "ptx_inline": _PTX_VECADD_LARGE,
        "kernel_name": "vecadd_large",
        "harness": harness_vecadd_large,
    },
    "multi_ldg": {
        "display": "multi_ldg aliased base (FB-5 canary)",
        "ptx_path": None,
        "ptx_inline": _PTX_MULTI_LDG,
        "kernel_name": "multi_ldg_test",
        "harness": harness_multi_ldg,
    },
    "smem_exchange": {
        "display": "shared-memory write/barrier/read",
        "ptx_path": None,
        "ptx_inline": _PTX_SMEM_EXCHANGE,
        "kernel_name": "smem_exchange",
        "harness": harness_smem_exchange,
    },
    "atomg_add": {
        "display": "atom.global.add.u32",
        "ptx_path": None,
        "ptx_inline": _PTX_ATOMG_ADD,
        "kernel_name": "atomg_add_test",
        "harness": harness_atomg_add,
    },
    "fmax": {
        "display": "max.f32 scalar (FMNMX)",
        "ptx_path": None,
        "ptx_inline": _PTX_FMAX,
        "kernel_name": "fmax_test",
        "harness": harness_fmax,
    },
    # WB-9 frontier additions
    "smem_cycle": {
        "display": "smem write/barrier/read cycle (param-base)",
        "ptx_path": None,
        "ptx_inline": _PTX_SMEM_CYCLE,
        "kernel_name": "smem_cycle",
        "harness": harness_smem_cycle,
    },
    "bar_ldc_xor": {
        "display": "bar.sync + LDC param + XOR",
        "ptx_path": None,
        "ptx_inline": _PTX_BAR_LDC_XOR,
        "kernel_name": "bar_ldc_xor",
        "harness": harness_bar_ldc_xor,
    },
    "dual_ldg64_dadd": {
        "display": "dual LDG.E.64 + DADD (FP64 multi-load)",
        "ptx_path": None,
        "ptx_inline": _PTX_DUAL_LDG64_DADD,
        "kernel_name": "dual_ldg64_dadd",
        "harness": harness_dual_ldg64_dadd,
    },
    "multi_block_atomic": {
        "display": "64-block atom.add scatter (grid contention)",
        "ptx_path": None,
        "ptx_inline": _PTX_MULTI_BLOCK_ATOMIC,
        "kernel_name": "multi_block_atomic",
        "harness": harness_multi_block_atomic,
    },
    "atom_cas64": {
        "display": "atom.global.cas.b64 (64-bit CAS)",
        "ptx_path": None,
        "ptx_inline": _PTX_ATOM_CAS64,
        "kernel_name": "atom_cas64_test",
        "harness": harness_atom_cas64,
    },
    "redux_sum": {
        "display": "redux.sync.add.s32 (warp REDUX)",
        "ptx_path": None,
        "ptx_inline": _PTX_REDUX_SUM,
        "kernel_name": "redux_sum_kernel",
        "harness": harness_redux_sum,
    },
    # =================================================================
    # PERF-4: ILP benchmark suite
    # =================================================================
    # Each kernel has multiple independent instruction chains to
    # create scheduling opportunities (body latency NOPs that a local
    # rescheduler could fill with independent work from another chain).
    # =================================================================
    "ilp_dual_int32": {
        "display": "ILP: dual independent u32 chains",
        "ptx_path": None,
        "ptx_inline": _PTX_ILP_DUAL_INT32,
        "kernel_name": "ilp_dual_int32",
        "harness": harness_ilp_dual_int32,
    },
    "ilp_dual_int64": {
        "display": "ILP: dual independent u64 add chains",
        "ptx_path": None,
        "ptx_inline": _PTX_ILP_DUAL_INT64,
        "kernel_name": "ilp_dual_int64",
        "harness": harness_ilp_dual_int64,
    },
    "ilp_alu_addr": {
        "display": "ILP: independent ALU value + address chains",
        "ptx_path": None,
        "ptx_inline": _PTX_ILP_ALU_ADDR,
        "kernel_name": "ilp_alu_addr",
        "harness": harness_ilp_alu_addr,
    },
    "ilp_unrolled_sum4": {
        "display": "ILP: 4-accumulator unrolled sum",
        "ptx_path": None,
        "ptx_inline": _PTX_ILP_UNROLLED_SUM4,
        "kernel_name": "ilp_unrolled_sum4",
        "harness": harness_ilp_unrolled_sum4,
    },
    "ilp_pipeline_load": {
        "display": "ILP: software-pipelined dual load+compute",
        "ptx_path": None,
        "ptx_inline": _PTX_ILP_PIPELINE_LOAD,
        "kernel_name": "ilp_pipeline_load",
        "harness": harness_ilp_pipeline_load,
    },
    "ilp_pred_alu": {
        "display": "ILP: independent scalar + predicate chains",
        "ptx_path": None,
        "ptx_inline": _PTX_ILP_PRED_ALU,
        "kernel_name": "ilp_pred_alu",
        "harness": harness_ilp_pred_alu,
    },
}


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    if len(values) == 1:
        v = values[0]
        return {"min": v, "max": v, "mean": v, "stddev": 0.0, "n": 1}
    return {
        "min":    min(values),
        "max":    max(values),
        "mean":   mean(values),
        "stddev": pstdev(values),
        "n":      len(values),
    }


def _fmt_time(t):
    return f"{t:.4f}" if t is not None else "(skipped)"


def _fmt_stat(s: dict | None) -> str:
    if s is None:
        return "(none)"
    if s["n"] == 1:
        return f"{s['mean']:.4f}"
    return (f"mean={s['mean']:.4f}  sd={s['stddev']:.4f}  "
            f"[{s['min']:.4f}..{s['max']:.4f}]  n={s['n']}")


# ---------------------------------------------------------------------------
# Per-kernel measurement.  Build openptxas + ptxas (each once), then run the
# correctness/benchmark harness `repeat` times and aggregate timing into
# stats.  Static metrics (regs, sass_total, sass_non_nop, compile_ms) come
# from the cubin and never vary across repeats.
# ---------------------------------------------------------------------------
def measure_kernel(name: str, mode: str, do_compare: bool,
                   repeat: int) -> dict:
    if name not in KERNELS:
        return {"kernel": name, "error": f"unknown kernel '{name}'"}

    kentry = KERNELS[name]
    if kentry["ptx_inline"] is not None:
        ptx = kentry["ptx_inline"]
        ptx_source = "(inline)"
    else:
        path = kentry["ptx_path"]
        if not path.exists():
            return {"kernel": name, "error": f"PTX file not found: {path}"}
        ptx = path.read_text(encoding="utf-8")
        ptx_source = str(path)

    result = {
        "kernel": name,
        "display": kentry["display"],
        "mode": mode,
        "repeat": repeat,
        "ptx_source": ptx_source,
        "build": "FAIL",
        "correctness": "FAIL",
        "ours": None,
        "ptxas": None,
        "deltas": None,
        "metadata": None,
    }

    # 1. Build through openptxas (with compaction-report capture)
    try:
        cubin_ours, t_compile_ours, report = compile_with_report(ptx)
    except Exception as e:
        result["error"] = f"openptxas build FAILED: {type(e).__name__}: {e}"
        return result

    ours = metrics_from_cubin(cubin_ours)
    ours["compile_ms"] = t_compile_ours * 1000.0
    ours["time_ms_runs"] = []
    result["ours"] = ours
    result["build"] = "PASS"

    # 2. Build through ptxas (optional)
    cubin_ptxas = None
    if do_compare:
        try:
            cubin_ptxas, t_compile_ptxas = compile_ptxas(ptx)
            theirs = metrics_from_cubin(cubin_ptxas)
            theirs["compile_ms"] = t_compile_ptxas * 1000.0
            theirs["time_ms_runs"] = []
            result["ptxas"] = theirs
        except Exception as e:
            result["ptxas_error"] = f"{type(e).__name__}: {e}"

    # 3. Launch + correctness (and optional benchmark) — repeat as requested
    ctx = CUDAContext()
    correct = True
    try:
        if not ctx.load(cubin_ours):
            result["error"] = "cuModuleLoadData failed for openptxas cubin"
            return result
        func = ctx.get_func(kentry["kernel_name"])
        for i in range(repeat):
            r = kentry["harness"](ctx, func, mode)
            if not r["correct"]:
                correct = False
            if r["time_ms"] is not None:
                ours["time_ms_runs"].append(r["time_ms"])

        if result["ptxas"] is not None and cubin_ptxas is not None:
            if ctx.load(cubin_ptxas):
                func_p = ctx.get_func(kentry["kernel_name"])
                for i in range(repeat):
                    rp = kentry["harness"](ctx, func_p, mode)
                    if rp["time_ms"] is not None:
                        result["ptxas"]["time_ms_runs"].append(rp["time_ms"])
            else:
                result["ptxas_error"] = "cuModuleLoadData failed for ptxas cubin"
    finally:
        ctx.close()

    result["correctness"] = "PASS" if correct else "FAIL"

    # 4. Stats + deltas
    ours["time_ms_stats"] = _stats(ours["time_ms_runs"])
    if result["ptxas"] is not None:
        result["ptxas"]["time_ms_stats"] = _stats(result["ptxas"]["time_ms_runs"])
        theirs = result["ptxas"]
        deltas = {
            "regs":         ours["regs"]         - theirs["regs"],
            "sass_total":   ours["sass_total"]   - theirs["sass_total"],
            "sass_non_nop": ours["sass_non_nop"] - theirs["sass_non_nop"],
        }
        if (ours["time_ms_stats"] is not None
                and theirs["time_ms_stats"] is not None):
            deltas["time_ms_mean"] = (ours["time_ms_stats"]["mean"]
                                      - theirs["time_ms_stats"]["mean"])
        result["deltas"] = deltas

    # 5. Compaction metadata
    if report is not None:
        result["metadata"] = {
            "compaction_attempted": report.attempted,
            "compaction_covered":   report.covered,
            "compacted":            report.gpr_fields_rewritten > 0,
            "compact_regs_before":  report.regs_before,
            "compact_regs_after":   report.regs_after,
            "compacted_insts":      report.compacted_insts,
            "gpr_fields_rewritten": report.gpr_fields_rewritten,
        }

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_block(result: dict, commits: dict) -> None:
    name = result["kernel"]
    print(f"[workbench] kernel={name}  ({result.get('display', '')})")
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    print(f"  build:    {result['build']}")
    print(f"  correct:  {result['correctness']}")
    print(f"  repeat:   {result['repeat']}")
    print(f"  forge:     {commits['forge']}")
    print(f"  opencuda:  {commits['opencuda']}")
    print(f"  openptxas: {commits['openptxas']}")

    ours = result["ours"]
    if ours is not None:
        print()
        print("  ours:")
        print(f"    regs:         {ours['regs']}")
        print(f"    sass_total:   {ours['sass_total']}")
        print(f"    sass_non_nop: {ours['sass_non_nop']}")
        print(f"    compile_ms:   {ours['compile_ms']:.1f}")
        print(f"    time_ms:      {_fmt_stat(ours.get('time_ms_stats'))}")

    theirs = result.get("ptxas")
    if theirs is not None:
        print()
        print("  ptxas:")
        print(f"    regs:         {theirs['regs']}")
        print(f"    sass_total:   {theirs['sass_total']}")
        print(f"    sass_non_nop: {theirs['sass_non_nop']}")
        print(f"    compile_ms:   {theirs['compile_ms']:.1f}")
        print(f"    time_ms:      {_fmt_stat(theirs.get('time_ms_stats'))}")
        print()
        print("  delta:")
        d = result["deltas"]
        print(f"    regs:         {d['regs']:+d}")
        print(f"    sass_total:   {d['sass_total']:+d}")
        print(f"    sass_non_nop: {d['sass_non_nop']:+d}")
        if "time_ms_mean" in d:
            print(f"    time_ms_mean: {d['time_ms_mean']:+.4f}")
    elif "ptxas_error" in result:
        print()
        print(f"  ptxas: skipped ({result['ptxas_error']})")


def write_kernel_json(result: dict, commits: dict,
                      results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact = {
        "schema": "workbench.kernel/v1",
        "timestamp": ts,
        "commits": commits,
        **result,
    }
    out_path = results_dir / f"{ts}_{result['kernel']}.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str))
    return out_path


# ---------------------------------------------------------------------------
# Suite mode + leaderboard
# ---------------------------------------------------------------------------
SUITES: dict[str, list[str]] = {
    "core":     ["reduce_sum", "conv2d_looped", "hmma_zero"],
    "tensor":   ["hmma_zero", "imma_zero", "dmma_zero", "qmma_zero"],
    "extended": [
        "reduce_sum", "conv2d_looped",
        "hmma_zero", "imma_zero", "dmma_zero", "qmma_zero",
        "cp_async", "warp_reduce", "atom_or",
    ],
    # WB-6 suites
    "stress": [
        "vecadd_large",   # memory bandwidth, multi-param, bounds check
        "smem_exchange",  # shared memory + barrier
        "multi_ldg",      # multi-LDG, aliased base address chains
    ],
    "contrast": [
        "conv2d_looped",
        "conv2d_unrolled",
        "warp_reduce",
        "hmma_zero",
    ],
    "wb6": [
        # Everything new in WB-6 (sanity that all build + correct + measure)
        "conv2d_unrolled", "vecadd_large", "multi_ldg",
        "smem_exchange", "atomg_add", "fmax",
    ],
    "ilp": [
        "ilp_dual_int32", "ilp_dual_int64", "ilp_alu_addr",
        "ilp_unrolled_sum4", "ilp_pipeline_load", "ilp_pred_alu",
    ],
    "all": [
        # Whole catalog (everything)
        "reduce_sum", "conv2d_looped", "conv2d_unrolled",
        "hmma_zero", "imma_zero", "dmma_zero", "qmma_zero",
        "cp_async", "warp_reduce",
        "atom_or", "atomg_add",
        "vecadd_large", "multi_ldg", "smem_exchange", "fmax",
        # WB-9 frontier
        "smem_cycle", "bar_ldc_xor", "dual_ldg64_dadd",
        "multi_block_atomic", "atom_cas64", "redux_sum",
        # PERF-4 ILP suite
        "ilp_dual_int32", "ilp_dual_int64", "ilp_alu_addr",
        "ilp_unrolled_sum4", "ilp_pipeline_load", "ilp_pred_alu",
    ],
    # WB-9: the 6 new kernels in isolation
    "frontier": [
        "smem_cycle", "bar_ldc_xor", "dual_ldg64_dadd",
        "multi_block_atomic", "atom_cas64", "redux_sum",
    ],
}

# KERNEL-100: corpus expansion registration
try:
    import workbench_expanded
    workbench_expanded.register(KERNELS, SUITES, _make_args)
except ImportError:
    pass  # expanded kernels not available (optional)


def classify_kernel(result: dict) -> str:
    """Bucket a kernel result into PARITY / NATIVE_WIN / GAP / NO_COMPARE."""
    if result.get("ptxas") is None or result.get("deltas") is None:
        return "NO_COMPARE"
    d = result["deltas"]
    # Compare static metrics first; then bench time if both present.
    metric_diffs = [d["regs"], d["sass_total"], d["sass_non_nop"]]
    if all(m == 0 for m in metric_diffs):
        # Static parity.  If we also have time stats and ours is faster
        # by ≥1 stddev of ptxas, count as a win; otherwise still parity.
        return "PARITY"
    if all(m <= 0 for m in metric_diffs) and any(m < 0 for m in metric_diffs):
        return "NATIVE_WIN"
    if all(m >= 0 for m in metric_diffs) and any(m > 0 for m in metric_diffs):
        return "GAP"
    return "MIXED"


def print_leaderboard(results: list[dict]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {
        "PARITY": [], "NATIVE_WIN": [], "GAP": [],
        "MIXED": [], "NO_COMPARE": [],
    }
    for r in results:
        if "error" in r:
            continue
        buckets[classify_kernel(r)].append(r["kernel"])

    print()
    print("=" * 64)
    print("LEADERBOARD")
    print("=" * 64)

    def _section(label: str, key: str, header: str):
        if not buckets[key]:
            return
        print()
        print(f"  {label}  ({len(buckets[key])} kernels)")
        print(f"    {header}")
        for r in results:
            if r["kernel"] not in buckets[key]:
                continue
            d = r.get("deltas") or {}
            print(f"    {r['kernel']:18s} "
                  f"regs={d.get('regs', 0):+d}  "
                  f"sass_total={d.get('sass_total', 0):+d}  "
                  f"sass_non_nop={d.get('sass_non_nop', 0):+d}")

    _section("A. EXACT PARITY (regs / sass / non-NOP all match ptxas)",
             "PARITY",
             "kernel             deltas")
    _section("B. NATIVE WINS (ours <= ptxas on every metric, < on at least one)",
             "NATIVE_WIN",
             "kernel             deltas")
    _section("C. REMAINING GAPS (ours >= ptxas on every metric, > on at least one)",
             "GAP",
             "kernel             deltas")
    _section("D. MIXED (some better, some worse)",
             "MIXED",
             "kernel             deltas")
    _section("X. NO COMPARE (ptxas unavailable / failed)",
             "NO_COMPARE",
             "kernel             deltas")
    print()
    return buckets


def write_suite_json(suite_name: str, results: list[dict],
                     buckets: dict[str, list[str]],
                     commits: dict, repeat: int, mode: str,
                     do_compare: bool, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    machine = {
        "platform": platform.platform(),
        "python":   platform.python_version(),
        "node":     platform.node(),
    }
    aggregate = {
        "kernels":     len(results),
        "parity":      len(buckets.get("PARITY", [])),
        "native_wins": len(buckets.get("NATIVE_WIN", [])),
        "gaps":        len(buckets.get("GAP", [])),
        "mixed":       len(buckets.get("MIXED", [])),
        "no_compare":  len(buckets.get("NO_COMPARE", [])),
    }
    artifact = {
        "schema":     "workbench.suite/v1",
        "suite":      suite_name,
        "timestamp":  ts,
        "mode":       mode,
        "repeat":     repeat,
        "compare":    "ptxas" if do_compare else None,
        "commits":    commits,
        "machine":    machine,
        "aggregate":  aggregate,
        "ranking":    buckets,
        "kernels":    results,
    }
    out_path = results_dir / f"{ts}_suite_{suite_name}.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def metrics_from_cubin(cubin: bytes) -> dict:
    return cubin_metrics(cubin)


def collect_commits() -> dict:
    return {
        "openptxas": _git_short(REPO_OPENPTXAS),
        "forge":     _git_short(REPO_FORGE),
        "opencuda":  _git_short(REPO_OPENCUDA),
    }


def run_kernel(name: str, mode: str, do_compare: bool, repeat: int,
               results_dir: Path) -> int:
    commits = collect_commits()
    result = measure_kernel(name, mode, do_compare, repeat)
    print_block(result, commits)
    if "error" in result:
        return 2
    artifact = write_kernel_json(result, commits, results_dir)
    print()
    print(f"[workbench] artifact: {artifact}")
    return 0 if result["correctness"] == "PASS" else 1


def run_suite(suite_name: str, mode: str, do_compare: bool, repeat: int,
              results_dir: Path) -> int:
    if suite_name not in SUITES:
        print(f"[workbench] unknown suite '{suite_name}'. "
              f"Available: {', '.join(sorted(SUITES))}", file=sys.stderr)
        return 2
    commits = collect_commits()
    print(f"[workbench] running suite '{suite_name}' ({len(SUITES[suite_name])} kernels)")
    print(f"  mode={mode}  repeat={repeat}  compare={'ptxas' if do_compare else 'none'}")
    print(f"  forge={commits['forge']}  opencuda={commits['opencuda']}  openptxas={commits['openptxas']}")
    print()

    results: list[dict] = []
    for kname in SUITES[suite_name]:
        print(f"--- {kname} ---")
        r = measure_kernel(kname, mode, do_compare, repeat)
        results.append(r)
        print_block(r, commits)
        print()

    buckets = print_leaderboard(results)
    artifact = write_suite_json(
        suite_name, results, buckets, commits, repeat, mode,
        do_compare, results_dir,
    )
    print(f"[workbench] suite artifact: {artifact}")

    # Exit code: non-zero if any kernel failed correctness or build
    bad = sum(1 for r in results
              if "error" in r or r.get("correctness") != "PASS")
    return 0 if bad == 0 else 1


def _cmd_run(args, parser):
    """Dispatch the `run` subcommand — body unchanged from the pre-WB-12.0
    flat-flag version.  Validation, dispatch, and return value all match
    the original `main()` exactly so the JSON artifact and stdout layout
    are byte-for-byte identical (modulo non-deterministic timing fields).
    """
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.kernel and args.suite:
        parser.error("--kernel and --suite are mutually exclusive")
    if not args.kernel and not args.suite:
        parser.error("one of --kernel / --suite is required (use `workbench list`)")

    do_compare = (args.compare == "ptxas")
    if args.suite:
        return run_suite(
            suite_name=args.suite,
            mode=args.mode,
            do_compare=do_compare,
            repeat=args.repeat,
            results_dir=Path(args.results_dir),
        )
    return run_kernel(
        name=args.kernel,
        mode=args.mode,
        do_compare=do_compare,
        repeat=args.repeat,
        results_dir=Path(args.results_dir),
    )


def _cmd_list(args):
    """Dispatch the `list` subcommand — body unchanged from the pre-WB-12.0
    `--list` flag handler.  Output is byte-for-byte identical.
    """
    print("Available kernels:")
    for k, v in KERNELS.items():
        print(f"  {k:20s} {v['display']}")
    print()
    print("Available suites:")
    for s, ks in SUITES.items():
        print(f"  {s:20s} ({len(ks)}) {', '.join(ks)}")
    return 0


def _cmd_stub(name: str, sub_id: str):
    """Stub for WB-12.1–12.5 subcommands.  Prints a not-yet-implemented
    notice and returns exit code 2 so callers can detect the unimplemented
    state without it being confused with a normal failure (exit 1).
    """
    print(f"workbench {name}: not yet implemented (WB-{sub_id} pending)")
    return 2


# ---------------------------------------------------------------------------
# WB-12.1: workbench status
# ---------------------------------------------------------------------------
# Snapshot of the latest (or --from-specified) suite_all artifact.  Pure
# replay — does not recompute deltas, does not classify kernels, does not
# call run/bench.  Bucket order, kernel order, and counts come straight
# from the artifact's `ranking` and `aggregate` fields.

# Display order for buckets in the status output.  NO_COMPARE is
# intentionally excluded — see WB-12.1 spec.
#
# Each tuple is (ranking_key, summary_label, leaderboard_label).
_STATUS_BUCKETS = [
    ("PARITY",     "PARITY",  "PARITY"),
    ("NATIVE_WIN", "WINS",    "NATIVE WIN"),
    ("GAP",        "GAPS",    "GAP"),
    ("MIXED",      "MIXED",   "MIXED"),
]

# Map ranking key → aggregate field name for the count.
_STATUS_AGG_KEY = {
    "PARITY":     "parity",
    "NATIVE_WIN": "native_wins",
    "GAP":        "gaps",
    "MIXED":      "mixed",
}


def _format_human_date(ts: str) -> str:
    """YYYYMMDD_HHMMSS → 'YYYY-MM-DD HH:MM:SS' for table output."""
    if len(ts) != 15 or ts[8] != "_":
        return ts
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"


def _format_iso_timestamp(ts: str) -> str:
    """YYYYMMDD_HHMMSS → 'YYYY-MM-DDTHH:MM:SS' (ISO 8601) for JSON."""
    if len(ts) != 15 or ts[8] != "_":
        return ts
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}"


def _resolve_suite_artifact(args, cmd_name: str) -> Path | None:
    """Find a suite_all artifact to read.  Returns Path on success,
    prints to stderr and returns None on any failure.

    Shared by `status` (WB-12.1), `show` (WB-12.2), and any future
    subcommand that needs to point at the latest (or --from-specified)
    suite_all artifact.  `cmd_name` is the user-facing subcommand name
    so error messages get prefixed correctly.
    """
    if args.from_path:
        path = Path(args.from_path)
        if not path.exists():
            print(f"workbench {cmd_name}: artifact not found: {path}",
                  file=sys.stderr)
            return None
        return path

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(
            f"workbench {cmd_name}: results dir not found: {results_dir}",
            file=sys.stderr,
        )
        return None
    candidates = sorted(results_dir.glob("*_suite_all.json"))
    if not candidates:
        print(
            f"workbench {cmd_name}: no suite_all artifact in {results_dir}",
            file=sys.stderr,
        )
        return None
    # Filenames are prefixed with YYYYMMDD_HHMMSS so lexical sort is chronological.
    return candidates[-1]


def _load_suite_artifact(path: Path, cmd_name: str) -> dict | None:
    """Load + schema-check a suite_all artifact.  Returns dict on success,
    None on any failure (with the error already printed to stderr).
    """
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"workbench {cmd_name}: malformed JSON in {path}: {e}",
              file=sys.stderr)
        return None
    if data.get("schema") != "workbench.suite/v1":
        print(
            f"workbench {cmd_name}: unexpected schema in {path}: "
            f"{data.get('schema')!r}",
            file=sys.stderr,
        )
        return None
    return data


def _cmd_status(args):
    """WB-12.1: snapshot the latest (or --from) suite_all artifact.

    Pure replay of the saved leaderboard.  No recomputation.
    """
    path = _resolve_suite_artifact(args, "status")
    if path is None:
        return 2
    data = _load_suite_artifact(path, "status")
    if data is None:
        return 2

    commit       = data.get("commits", {}).get("openptxas", "?")
    timestamp_raw = data.get("timestamp", "")
    aggregate    = data.get("aggregate", {})
    ranking      = data.get("ranking", {})

    if args.format == "json":
        out = {
            "commit":       commit,
            "timestamp":    _format_iso_timestamp(timestamp_raw),
            "kernel_count": aggregate.get("kernels", 0),
            "buckets": {
                rank_key: list(ranking.get(rank_key, []))
                for rank_key, _summary, _label in _STATUS_BUCKETS
            },
        }
        print(json.dumps(out, indent=2))
        return 0

    # Table mode
    print(f"{'commit:':<10s}{commit}")
    print(f"{'date:':<10s}{_format_human_date(timestamp_raw)}")
    print(f"{'kernels:':<10s}{aggregate.get('kernels', 0)}")
    print()
    for rank_key, summary_label, _disp_label in _STATUS_BUCKETS:
        count = aggregate.get(_STATUS_AGG_KEY[rank_key], 0)
        print(f"{summary_label + ':':<7s}{count:>5d}")
    print()
    print("leaderboard:")
    for rank_key, _summary, disp_label in _STATUS_BUCKETS:
        members = ranking.get(rank_key, [])
        if not members:
            continue
        print(f"  {disp_label}:")
        for k in members:
            print(f"    {k}")
    return 0


# ---------------------------------------------------------------------------
# WB-12.2: workbench show
# ---------------------------------------------------------------------------
# Drill-down into a single kernel record from a suite_all artifact.  Pure
# replay — pulls regs / sass_total / sass_non_nop / time_ms_stats.mean /
# deltas straight from the saved record.  Bucket label comes from the
# artifact's `ranking` field (cross-checks with `workbench status`).

# Bucket lookup order — matches WB-12.1's _STATUS_BUCKETS plus NO_COMPARE
# at the end so a kernel that ran without a ptxas compare is still
# locatable.
_SHOW_BUCKET_LOOKUP = ("PARITY", "NATIVE_WIN", "GAP", "MIXED", "NO_COMPARE")


def _show_metric_line(label: str, value) -> None:
    """Print a `  label:           value` line with the value column at
    column 18 (matching the WB-12.2 spec layout).
    """
    if value is None:
        return
    print(f"  {label + ':':<16s}{value}")


def _show_signed_int(label: str, value) -> None:
    if value is None:
        return
    print(f"  {label + ':':<16s}{value:+d}")


def _show_signed_float(label: str, value) -> None:
    if value is None:
        return
    print(f"  {label + ':':<16s}{value:+.4f}")


def _cmd_show(args):
    """WB-12.2: print a single-kernel record from the latest (or --from)
    suite_all artifact.  Pure replay; numbers come straight from the
    record's stored fields.
    """
    path = _resolve_suite_artifact(args, "show")
    if path is None:
        return 2
    data = _load_suite_artifact(path, "show")
    if data is None:
        return 2

    kernel_name = args.kernel
    record = None
    for r in data.get("kernels", []):
        if r.get("kernel") == kernel_name:
            record = r
            break
    if record is None:
        print(
            f"workbench show: kernel '{kernel_name}' not found in {path}",
            file=sys.stderr,
        )
        return 2

    # Bucket lookup — find which ranking list contains this kernel.
    ranking = data.get("ranking", {})
    bucket = None
    for b in _SHOW_BUCKET_LOOKUP:
        if kernel_name in ranking.get(b, []):
            bucket = b
            break
    if bucket is None:
        bucket = "?"

    ours    = record.get("ours") or {}
    ptxas   = record.get("ptxas") or {}
    deltas  = record.get("deltas") or {}

    if args.format == "json":
        out = {
            "kernel": kernel_name,
            "bucket": bucket,
            "ours":   ours,
            "ptxas":  ptxas if ptxas else None,
            "delta":  deltas if deltas else None,
        }
        print(json.dumps(out, indent=2))
        return 0

    # Table mode
    print(f"{'kernel:':<10s}{kernel_name}")
    print(f"{'bucket:':<10s}{bucket}")
    print()
    print("ours:")
    _show_metric_line("regs",         ours.get("regs"))
    _show_metric_line("sass_total",   ours.get("sass_total"))
    _show_metric_line("sass_non_nop", ours.get("sass_non_nop"))
    ours_mean = (ours.get("time_ms_stats") or {}).get("mean")
    if ours_mean is not None:
        print(f"  {'time_ms:':<16s}{ours_mean:.4f} (mean)")
    if ptxas:
        print()
        print("ptxas:")
        _show_metric_line("regs",         ptxas.get("regs"))
        _show_metric_line("sass_total",   ptxas.get("sass_total"))
        _show_metric_line("sass_non_nop", ptxas.get("sass_non_nop"))
        ptxas_mean = (ptxas.get("time_ms_stats") or {}).get("mean")
        if ptxas_mean is not None:
            print(f"  {'time_ms:':<16s}{ptxas_mean:.4f} (mean)")
    if deltas:
        print()
        print("delta:")
        _show_signed_int  ("regs",         deltas.get("regs"))
        _show_signed_int  ("sass_total",   deltas.get("sass_total"))
        _show_signed_int  ("sass_non_nop", deltas.get("sass_non_nop"))
        _show_signed_float("time_ms",      deltas.get("time_ms_mean"))
    return 0


# ---------------------------------------------------------------------------
# WB-12.3: workbench dump
# ---------------------------------------------------------------------------
# Raw passthrough of suite_all artifacts.  No parsing, no validation, no
# schema checks.  This is the "no interpretation" layer — anything that
# wants the original JSON bytes can pipe `workbench dump` and get them.
#
# Critical: byte-for-byte equality with the source file.  On Windows the
# artifacts are written with CRLF line endings (Path.write_text default),
# so the dump path uses Path.read_bytes + sys.stdout.buffer.write to bypass
# Python's text-mode CRLF translation entirely.

def _cmd_dump(args):
    """WB-12.3: raw passthrough of a suite_all artifact, or list mode."""
    # ---- --list mode ---------------------------------------------------
    if args.list:
        results_dir = Path(args.results_dir)
        if not results_dir.is_dir():
            print(f"workbench dump: results dir not found: {results_dir}",
                  file=sys.stderr)
            return 2
        candidates = sorted(results_dir.glob("*_suite_all.json"))
        if not candidates:
            print(
                f"workbench dump: no suite_all artifact in {results_dir}",
                file=sys.stderr,
            )
            return 2
        # Header is the basename of the results dir + "/" so the default
        # default-results-dir prints as "results/" per the WB-12.3 spec
        # example, regardless of whether the user passed an absolute path.
        print(f"{Path(args.results_dir).name}/")
        for c in candidates:
            print(f"  {c.name}")
        return 0

    # ---- dump (default / --latest / --from) ----------------------------
    # _resolve_suite_artifact handles both --from <path> and the
    # latest-in-results-dir fallback.  --latest is just an explicit no-op
    # selector for the same behavior — argparse already accepts it.
    path = _resolve_suite_artifact(args, "dump")
    if path is None:
        return 2

    try:
        data = Path(path).read_bytes()
    except OSError as e:
        print(f"workbench dump: cannot read {path}: {e}", file=sys.stderr)
        return 2

    # Binary write — bypass Windows text-mode CRLF translation so the
    # output bytes match the file bytes exactly.
    sys.stdout.buffer.write(data)
    return 0


# ---------------------------------------------------------------------------
# WB-12.4: workbench history
# ---------------------------------------------------------------------------
# Trend display across multiple suite_all artifacts.  Pure replay — every
# field comes from the artifacts as-is.  No smoothing, no averaging, no
# inference, no "best/worst" labels.

# Buckets we look at when locating a kernel in an artifact's `ranking`.
_HISTORY_BUCKETS = ("PARITY", "NATIVE_WIN", "GAP", "MIXED", "NO_COMPARE")


def _load_history_entries(results_dir: Path) -> list[dict] | None:
    """Scan results_dir for suite_all artifacts and load minimal fields
    from each one.  Returns the list (chronological), or None on a fatal
    error (already printed to stderr).

    Individual unreadable / wrong-schema artifacts are skipped silently
    so a single bad file doesn't take out the whole history view.
    """
    if not results_dir.is_dir():
        print(f"workbench history: results dir not found: {results_dir}",
              file=sys.stderr)
        return None
    candidates = sorted(results_dir.glob("*_suite_all.json"))
    if not candidates:
        print(f"workbench history: no suite_all artifact in {results_dir}",
              file=sys.stderr)
        return None

    entries: list[dict] = []
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("schema") != "workbench.suite/v1":
            continue
        entries.append({
            "path":      path,
            "timestamp": data.get("timestamp", ""),
            "commit":    data.get("commits", {}).get("openptxas", "?"),
            "aggregate": data.get("aggregate", {}),
            "ranking":   data.get("ranking", {}),
            "kernels":   data.get("kernels", []),
        })
    if not entries:
        print(
            f"workbench history: no valid suite_all artifacts in {results_dir}",
            file=sys.stderr,
        )
        return None
    return entries


def _history_default_view(entries: list[dict], fmt: str) -> int:
    """Default history view: one row per artifact with aggregate counts."""
    if fmt == "json":
        out = {
            "history": [
                {
                    "timestamp": e["timestamp"],
                    "commit":    e["commit"],
                    "aggregate": e["aggregate"],
                }
                for e in entries
            ]
        }
        print(json.dumps(out, indent=2))
        return 0

    print("history (latest last)")
    print()
    header = (
        f"{'timestamp':<18s}{'commit':<10s}{'kernels':<9s}"
        f"{'parity':<8s}{'wins':<6s}{'gaps':<6s}{'mixed':<5s}"
    )
    print(header)
    print("-" * len(header))
    for e in entries:
        agg = e["aggregate"]
        row = (
            f"{e['timestamp']:<18s}"
            f"{e['commit']:<10s}"
            f"{agg.get('kernels',     0):<9d}"
            f"{agg.get('parity',      0):<8d}"
            f"{agg.get('native_wins', 0):<6d}"
            f"{agg.get('gaps',        0):<6d}"
            f"{agg.get('mixed',       0):<5d}"
        )
        print(row.rstrip())
    return 0


def _history_kernel_view(entries: list[dict], kernel: str, fmt: str) -> int:
    """--kernel view: per-entry trend for one kernel.  Skip artifacts
    where the kernel isn't present (catalog grew over time).
    """
    rows: list[dict] = []
    for e in entries:
        record = None
        for r in e["kernels"]:
            if r.get("kernel") == kernel:
                record = r
                break
        if record is None:
            continue  # kernel not present in this artifact — skip silently
        bucket = "?"
        for b in _HISTORY_BUCKETS:
            if kernel in e["ranking"].get(b, []):
                bucket = b
                break
        deltas = record.get("deltas") or {}
        rows.append({
            "timestamp":     e["timestamp"],
            "commit":        e["commit"],
            "aggregate":     e["aggregate"],
            "bucket":        bucket,
            "non_nop_delta": deltas.get("sass_non_nop", 0),
            "record":        record,
        })

    if not rows:
        print(
            f"workbench history: kernel '{kernel}' not found in any artifact",
            file=sys.stderr,
        )
        return 2

    if fmt == "json":
        out = {
            "kernel": kernel,
            "history": [
                {
                    "timestamp": r["timestamp"],
                    "commit":    r["commit"],
                    "aggregate": r["aggregate"],
                    "kernel": {
                        "name":   kernel,
                        "bucket": r["bucket"],
                        "ours":   r["record"].get("ours"),
                        "ptxas":  r["record"].get("ptxas"),
                        "delta":  r["record"].get("deltas"),
                    },
                }
                for r in rows
            ]
        }
        print(json.dumps(out, indent=2))
        return 0

    print(f"kernel: {kernel}")
    print()
    header = f"{'timestamp':<18s}{'bucket':<11s}{'non_nop_delta':<13s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (
            f"{r['timestamp']:<18s}"
            f"{r['bucket']:<11s}"
            f"{r['non_nop_delta']:+d}"
        )
        print(line)
    return 0


def _cmd_history(args):
    """WB-12.4: trend display across all suite_all artifacts."""
    if args.limit is not None and args.limit < 1:
        print("workbench history: --limit must be >= 1", file=sys.stderr)
        return 2

    entries = _load_history_entries(Path(args.results_dir))
    if entries is None:
        return 2

    # --limit applies as a tail (most recent N).
    if args.limit is not None:
        entries = entries[-args.limit:]

    if args.kernel:
        return _history_kernel_view(entries, args.kernel, args.format)
    return _history_default_view(entries, args.format)


# ---------------------------------------------------------------------------
# WB-12.5: workbench diff
# ---------------------------------------------------------------------------
# Compare two suite_all artifacts (default: latest vs previous).  Pure
# field-level diff.  No inference, no scoring, no labels.

# Aggregate fields shown in the diff table.  (key, display_label).
_DIFF_AGG_FIELDS = [
    ("kernels",     "kernels"),
    ("parity",      "parity"),
    ("native_wins", "wins"),
    ("gaps",        "gaps"),
    ("mixed",       "mixed"),
]

# Kernel fields tracked for the per-kernel diff.  Order matters — it's
# the display order in the table when multiple fields differ.
_DIFF_KERNEL_FIELDS = ["bucket", "build", "correctness",
                       "regs", "sass_total", "sass_non_nop"]


def _kernel_fields_at(art: dict, kernel_name: str) -> dict | None:
    """Extract diffable fields for a kernel from a suite artifact.

    Returns dict {field: value} or None if the kernel isn't in the
    artifact at all.  `bucket` comes from `ranking`; numeric fields come
    from `deltas`; `build`/`correctness` from the kernel record itself.
    """
    rec = None
    for r in art.get("kernels", []):
        if r.get("kernel") == kernel_name:
            rec = r
            break
    if rec is None:
        return None
    bucket = "?"
    for b in _HISTORY_BUCKETS:
        if kernel_name in art.get("ranking", {}).get(b, []):
            bucket = b
            break
    deltas = rec.get("deltas") or {}
    return {
        "bucket":       bucket,
        "build":        rec.get("build"),
        "correctness":  rec.get("correctness"),
        "regs":         deltas.get("regs"),
        "sass_total":   deltas.get("sass_total"),
        "sass_non_nop": deltas.get("sass_non_nop"),
    }


def _fmt_diff_value(field: str, value) -> str:
    """Format a kernel-field value for the diff display.

    Numeric delta fields print signed (`+1`, `-1`, `+0`).  Strings
    print as-is.  None becomes `<none>` (used when a kernel was added
    or removed between artifacts).
    """
    if value is None:
        return "<none>"
    if field in ("regs", "sass_total", "sass_non_nop"):
        return f"{value:+d}"
    return str(value)


def _diff_resolve_artifacts(args) -> tuple[Path, Path] | None:
    """Resolve the (from, to) pair for diff.

    Either both --from and --to are given (explicit), or neither is
    (default to latest two artifacts in chronological order).
    """
    if args.from_path or args.to_path:
        if not (args.from_path and args.to_path):
            print(
                "workbench diff: --from and --to must be specified together",
                file=sys.stderr,
            )
            return None
        from_path = Path(args.from_path)
        to_path   = Path(args.to_path)
        if not from_path.exists():
            print(f"workbench diff: artifact not found: {from_path}",
                  file=sys.stderr)
            return None
        if not to_path.exists():
            print(f"workbench diff: artifact not found: {to_path}",
                  file=sys.stderr)
            return None
        return from_path, to_path

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"workbench diff: results dir not found: {results_dir}",
              file=sys.stderr)
        return None
    candidates = sorted(results_dir.glob("*_suite_all.json"))
    if len(candidates) < 2:
        print(
            f"workbench diff: need at least 2 suite_all artifacts, "
            f"got {len(candidates)} in {results_dir}",
            file=sys.stderr,
        )
        return None
    return candidates[-2], candidates[-1]


def _diff_default_view(from_data: dict, to_data: dict, fmt: str) -> int:
    """Default diff view: aggregate diff + per-kernel field changes."""
    from_ts  = from_data.get("timestamp", "")
    to_ts    = to_data.get("timestamp", "")
    from_agg = from_data.get("aggregate", {})
    to_agg   = to_data.get("aggregate", {})

    # Walk both kernel sets to compute changes / added / removed.
    from_names = {r["kernel"] for r in from_data.get("kernels", [])}
    to_names   = {r["kernel"] for r in to_data.get("kernels", [])}
    common     = from_names & to_names
    added      = sorted(to_names - from_names)
    removed    = sorted(from_names - to_names)

    # Walk in the to-artifact's stored order so changed kernels appear
    # in run order, not set/dict order.
    kernel_changes: list[dict] = []
    for r in to_data.get("kernels", []):
        name = r.get("kernel")
        if name not in common:
            continue
        ff = _kernel_fields_at(from_data, name)
        tf = _kernel_fields_at(to_data, name)
        diffs: dict = {}
        for field in _DIFF_KERNEL_FIELDS:
            if ff.get(field) != tf.get(field):
                diffs[field] = [ff.get(field), tf.get(field)]
        if diffs:
            kernel_changes.append({"kernel": name, "fields": diffs})

    if fmt == "json":
        out = {
            "from": from_ts,
            "to":   to_ts,
            "aggregate": {
                key: [from_agg.get(key, 0), to_agg.get(key, 0)]
                for key, _ in _DIFF_AGG_FIELDS
            },
            "kernel_changes": kernel_changes,
        }
        if added:
            out["added"] = added
        if removed:
            out["removed"] = removed
        print(json.dumps(out, indent=2))
        return 0

    # Table mode
    print(f"diff: {from_ts} → {to_ts}")
    print()
    print("aggregate:")
    for key, label in _DIFF_AGG_FIELDS:
        from_v = from_agg.get(key, 0)
        to_v   = to_agg.get(key, 0)
        delta  = to_v - from_v
        print(f"  {label + ':':<10s}{from_v:>2d} → {to_v:>2d}     ({delta:+d})")
    print()
    print("kernel changes:")
    if not kernel_changes:
        print("  (none)")
    else:
        for i, change in enumerate(kernel_changes):
            if i > 0:
                print()
            print(f"  {change['kernel']}:")
            for field, (old, new) in change["fields"].items():
                print(
                    f"    {field}: "
                    f"{_fmt_diff_value(field, old)} → "
                    f"{_fmt_diff_value(field, new)}"
                )
    if added:
        print()
        print("added kernels:")
        for n in added:
            print(f"  {n}")
    if removed:
        print()
        print("removed kernels:")
        for n in removed:
            print(f"  {n}")
    return 0


def _diff_kernel_view(from_data: dict, to_data: dict,
                      kernel: str, fmt: str) -> int:
    """--kernel view: focused per-kernel diff."""
    from_ts = from_data.get("timestamp", "")
    to_ts   = to_data.get("timestamp", "")

    from_fields = _kernel_fields_at(from_data, kernel)
    to_fields   = _kernel_fields_at(to_data, kernel)

    if from_fields is None and to_fields is None:
        print(
            f"workbench diff: kernel '{kernel}' not in either artifact",
            file=sys.stderr,
        )
        return 2

    # Build the field-by-field diff.  If the kernel was added or removed,
    # all fields contribute (with the missing side as None).
    diffs: dict = {}
    if from_fields is None:
        for k in _DIFF_KERNEL_FIELDS:
            diffs[k] = [None, to_fields.get(k)]
    elif to_fields is None:
        for k in _DIFF_KERNEL_FIELDS:
            diffs[k] = [from_fields.get(k), None]
    else:
        for k in _DIFF_KERNEL_FIELDS:
            if from_fields.get(k) != to_fields.get(k):
                diffs[k] = [from_fields.get(k), to_fields.get(k)]

    if fmt == "json":
        out = {
            "kernel": kernel,
            "from":   from_ts,
            "to":     to_ts,
            "fields": diffs,
        }
        print(json.dumps(out, indent=2))
        return 0

    print(f"kernel: {kernel}")
    print(f"{'from:':<6s}{from_ts}")
    print(f"{'to:':<6s}{to_ts}")
    print()
    if not diffs:
        print("(no changes)")
        return 0
    for field, (old, new) in diffs.items():
        print(
            f"{field}: "
            f"{_fmt_diff_value(field, old)} → "
            f"{_fmt_diff_value(field, new)}"
        )
    return 0


def _cmd_diff(args):
    """WB-12.5: compare two suite_all artifacts."""
    paths = _diff_resolve_artifacts(args)
    if paths is None:
        return 2
    from_path, to_path = paths

    from_data = _load_suite_artifact(from_path, "diff")
    if from_data is None:
        return 2
    to_data = _load_suite_artifact(to_path, "diff")
    if to_data is None:
        return 2

    if args.kernel:
        return _diff_kernel_view(from_data, to_data, args.kernel, args.format)
    return _diff_default_view(from_data, to_data, args.format)


# ===========================================================================
# FG-1: Forge integration
# ===========================================================================
#
# Pipeline (decided in FG-1 design phase, all 5 questions locked):
#
#     Forge .fg  →  Forge PTX backend  →  OpenPTXas  →  cubin  →  GPU
#
# - Forge already has its own PTX backend (lib/codegen/codegen_ptx.ml).
#   We do NOT route through OpenCUDA — Forge → PTX is direct.  OpenCUDA
#   commit hash is still recorded in artifacts for traceability but is
#   not part of the kernel execution path.
# - Each Forge target has its own per-target Python harness because
#   Forge param shapes differ from the hand-crafted reference kernels
#   (e.g. forge `reduce_sum` is 4 args + single global atomic result;
#   the hand-crafted reference is 5 args + per-block output array).
# - Forge runs in WSL (the binary is a Linux ELF).  Each `forge run`
#   shells out to `wsl.exe -- bash -c '...'`.
# - Forge writes its .ptx in-place inside the Forge tree.  We copy it
#   into results/<ts>_forge_<target>.ptx so workbench owns its inputs
#   and runs are reproducible.
# - Hard rule: NO silent fallback to the hand-crafted PTX path.  If
#   Forge fails, openptxas fails to assemble forge PTX, or the GPU
#   refuses to run the forge-emitted kernel — STOP and report.
#
# ---------------------------------------------------------------------
# FG-1.0: artifact schema (workbench.forge_run/v1)
# ---------------------------------------------------------------------
#
# Locked schema:
#
# {
#   "schema":      "workbench.forge_run/v1",
#   "timestamp":   "YYYYMMDD_HHMMSS",
#   "source_mode": "forge",
#   "ptx_source":  "forge",
#   "target":      "<logical name>",
#   "source": {
#     "fg_path":       "<relative to forge repo>",
#     "kernel_symbol": "<.entry name in PTX>",
#     "language":      "forge"
#   },
#   "commits": {
#     "forge":     "<short>",
#     "opencuda":  "<short>",   # recorded but not in execution path
#     "openptxas": "<short>"
#   },
#   "stages": [
#     {"name": "forge_compile",       "status": "PASS|FAIL", "duration_ms": ..., "exit_code": ..., "stdout_tail": [], "stderr_tail": []},
#     {"name": "openptxas_assemble",  "status": "PASS|FAIL", "duration_ms": ..., "error": ...},
#     {"name": "ptxas_compile",       "status": "PASS|FAIL", "duration_ms": ..., "error": ...},  # only if --compare ptxas
#     {"name": "gpu_correctness",     "status": "PASS|FAIL", "duration_ms": ..., "error": ...}
#   ],
#   "artifacts": {
#     "forge_cu_path":      "<absolute, may be null>",
#     "forge_ptx_source":   "<absolute path inside forge tree>",
#     "forge_ptx_cached":   "<absolute path inside results/>",
#     "ours_cubin_size":    int,
#     "ptxas_cubin_size":   int  # null if no compare
#   },
#   "build":       "PASS|FAIL",
#   "correctness": "PASS|FAIL",
#   "ours":        { ... same shape as workbench.kernel/v1 ours ... },
#   "ptxas":       { ... same shape as workbench.kernel/v1 ptxas ... }  | null,
#   "deltas":      { ... same shape as workbench.kernel/v1 deltas ... } | null,
#   "bucket":      "PARITY|NATIVE_WIN|GAP|MIXED|NO_COMPARE"
# }
#
# Distinct schema name from `workbench.kernel/v1` so consumers (status,
# show, history, diff) can reliably tell PTX-backed and Forge-backed
# runs apart.

_FORGE_SCHEMA_VERSION = "workbench.forge_run/v1"


# ---------------------------------------------------------------------
# FG-1.1: Forge target catalog
# ---------------------------------------------------------------------
# First-target choice: `reduce_step` from demos/205_gpu_reduce.fg.
#
# This is the simplest verified GPU kernel Forge has — pure global-memory
# dataflow loop with no warp shuffles, no special registers beyond the
# four supported (tid/ntid/ctaid/nctaid), no device function calls, and
# no atomics.  It exists specifically to validate the Forge → OpenPTXas
# → GPU pipeline plumbing without tripping any of the missing-feature
# bugs found during the FG-1.1 first attempt against `1017_gpu_warp_reduce.fg`.
#
# History:
#   - FG-1.1 first attempt used `reduce_sum` from 1017_gpu_warp_reduce.fg
#     and surfaced two real bugs:
#       (A) OpenPTXas missing `%laneid` in _SPECIAL_REGS (sass/isel.py)
#       (B) Forge PTX backend stubs device function calls
#           (warp_reduce_sum compiles to `mov 0` placeholder)
#   - Both findings are explicitly OUT OF SCOPE for FG-1.1 — see the
#     FG-1.1 stop report.  They become FG-1.5 (laneid) and FG-1.6
#     (Forge PTX backend) when their time comes.
#   - This catalog deliberately avoids any Forge target that uses warp
#     shuffles, device function calls, or %laneid until FG-1.5 / FG-1.6
#     resolve those gaps.

_FORGE_KERNELS: dict[str, dict] = {
    "reduce_step": {
        "display":       "reduce_step (forge-backed, single-threaded "
                         "in-place pair reduction, u64)",
        "fg_path":       "demos/205_gpu_reduce.fg",
        "kernel_symbol": "reduce_step",
        "harness":       None,  # set below after harness fn is defined
    },
    # FG-1.13A — TEMPORARY diagnostic target for %laneid isel coverage.
    # Minimal Forge kernel that reads lane_id() and stores it into
    # output[tid].  Forced by FG-1.13 to surface FG-1-A (OpenPTXas isel
    # missing %laneid in _SPECIAL_REGS).  The .fg source is a temp file
    # in forge/demos/ that should be removed after FG-1.14 completes.
    "laneid_trigger": {
        "display":       "laneid_trigger (FG-1.13A: minimal %laneid)",
        "fg_path":       "demos/1099_laneid_trigger.fg",
        "kernel_symbol": "laneid_trigger",
        "harness":       None,
    },
    # FG-1.13B — TEMPORARY diagnostic target for device function call
    # lowering.  Minimal Forge kernel that calls a user-defined helper
    # `double_it(x) = x + x` and writes the result.  Forced by FG-1.13
    # to surface FG-1-B (Forge PTX backend stubs device function calls
    # to `mov 0`).  The .fg source is a temp file in forge/demos/ that
    # should be removed after FG-1.14 completes.
    "devfn_trigger": {
        "display":       "devfn_trigger (FG-1.13B: minimal device fn call)",
        "fg_path":       "demos/1098_devfn_trigger.fg",
        "kernel_symbol": "devfn_trigger",
        "harness":       None,
    },
}


def _wsl_path(p: Path) -> str:
    """Convert a Windows path (C:\\Users\\...) to a WSL /mnt/c/users/... path."""
    s = str(p).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s


def _invoke_forge(fg_path: Path) -> dict:
    """FG-1.1: invoke the Forge compiler on a single .fg file via WSL.

    The Forge binary is a Linux ELF (`forge/_build/default/bin/main.exe`)
    so we shell out via `wsl.exe -- bash -c`.  No opam env needed — the
    prebuilt binary is self-contained.

    Returns a stage record matching the FG-1.0 schema:
        {
            "name":         "forge_compile",
            "status":       "PASS" | "FAIL",
            "duration_ms":  float,
            "exit_code":    int,
            "stdout_tail":  list[str],
            "stderr_tail":  list[str],
        }
    """
    forge_root_wsl = _wsl_path(REPO_FORGE)
    fg_rel = fg_path.relative_to(REPO_FORGE) if fg_path.is_absolute() else fg_path
    fg_rel_str = str(fg_rel).replace("\\", "/")

    cmd_str = (
        f"cd {forge_root_wsl} && "
        f"./_build/default/bin/main.exe build {fg_rel_str}"
    )

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            ["wsl.exe", "--", "bash", "-c", cmd_str],
            capture_output=True,
            timeout=180,
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0
    except subprocess.TimeoutExpired:
        return {
            "name":        "forge_compile",
            "status":      "FAIL",
            "duration_ms": (time.perf_counter() - t0) * 1000.0,
            "exit_code":   -1,
            "stdout_tail": [],
            "stderr_tail": ["timeout (180s)"],
        }
    except FileNotFoundError as e:
        return {
            "name":        "forge_compile",
            "status":      "FAIL",
            "duration_ms": (time.perf_counter() - t0) * 1000.0,
            "exit_code":   -1,
            "stdout_tail": [],
            "stderr_tail": [f"wsl.exe not found: {e}"],
        }

    stdout_lines = result.stdout.decode("utf-8", errors="replace").splitlines()
    stderr_lines = result.stderr.decode("utf-8", errors="replace").splitlines()

    return {
        "name":        "forge_compile",
        "status":      "PASS" if result.returncode == 0 else "FAIL",
        "duration_ms": duration_ms,
        "exit_code":   result.returncode,
        "stdout_tail": stdout_lines[-12:],
        "stderr_tail": stderr_lines[-12:],
    }


# ---------------------------------------------------------------------
# FG-1.1: per-target harnesses for Forge-backed kernels
# ---------------------------------------------------------------------

def harness_forge_reduce_step(ctx: CUDAContext, func, mode: str) -> dict:
    """Forge-emitted reduce_step (demos/205_gpu_reduce.fg).

    Forge param shape (4 args, span<u64> flattened to ptr+len):
        .param .u64 reduce_step_param_s_data
        .param .u64 reduce_step_param_s_len
        .param .u64 reduce_step_param_n
        .param .u64 reduce_step_param_stride

    Semantics — sequential single-pair reduction step:

        let mut i = 0
        while i + stride < n:
            s[i] = s[i] + s[i + stride]
            i += stride * 2

    With stride=1, n=N this writes pair sums into the even indices:
        s[0] = s[0]+s[1], s[2] = s[2]+s[3], ...

    *Crucial:* this is a single-threaded sequential algorithm.  Every
    thread runs the same loop on the same memory.  Launching with more
    than one thread/block would cause data races.  We launch (1,1,1) ×
    (1,1,1) — the kernel exists to validate the pipeline plumbing, not
    to demonstrate parallelism.
    """
    n = 16
    stride = 1

    # Input: 1..N
    host_in = (ctypes.c_uint64 * n)(*[i + 1 for i in range(n)])

    # Expected output: even indices hold s[i]+s[i+1], odd indices unchanged.
    expected = list(range(1, n + 1))
    i = 0
    while i + stride < n:
        expected[i] = expected[i] + expected[i + stride]
        i += stride * 2

    d_s = ctx.alloc(n * 8)
    try:
        ctx.copy_to(d_s, bytes(host_in))

        a_s_data = ctypes.c_uint64(d_s)
        a_s_len  = ctypes.c_uint64(n)
        a_n      = ctypes.c_uint64(n)
        a_stride = ctypes.c_uint64(stride)
        args, _hold = _make_args(a_s_data, a_s_len, a_n, a_stride)

        # Single thread, single block — no race over the shared loop state.
        ctx.cuda.cuLaunchKernel(
            func, 1, 1, 1, 1, 1, 1, 0, None, args, None
        )
        sync_rc = ctx.sync()
        if sync_rc != 0:
            return {"correct": False, "time_ms": None,
                    "error": f"sync failed: {sync_rc}"}

        out_bytes = ctx.copy_from(d_s, n * 8)
        actual = list(struct.unpack(f"<{n}Q", out_bytes))
        correct = (actual == expected)

        time_ms = None
        if mode == "bench":
            # Reset buffer between bench iterations so each launch sees
            # the same input and we measure the kernel, not accumulated
            # state from previous calls.
            ctx.copy_to(d_s, bytes(host_in))
            time_ms = _bench_launch(
                ctx, func, (1, 1, 1), (1, 1, 1), args
            )
    finally:
        ctx.free(d_s)

    return {"correct": correct, "time_ms": time_ms,
            "expected": expected, "actual": actual}


def harness_forge_laneid_trigger(ctx: CUDAContext, func, mode: str) -> dict:
    """FG-1.13A: read lane_id() into output[tid], verify against expected
    pattern [0, 1, 2, ..., block_size-1] for a single warp-shaped block.

    Param shape (3 args):
        .param .u64 laneid_trigger_param_output_data
        .param .u64 laneid_trigger_param_output_len
        .param .u64 laneid_trigger_param_n
    """
    n = 32  # one warp
    host_out = (ctypes.c_uint64 * n)(*([0] * n))
    expected = list(range(n))  # lane 0..31

    d_out = ctx.alloc(n * 8)
    try:
        ctx.copy_to(d_out, bytes(host_out))
        a_out_data = ctypes.c_uint64(d_out)
        a_out_len  = ctypes.c_uint64(n)
        a_n        = ctypes.c_uint64(n)
        args, _hold = _make_args(a_out_data, a_out_len, a_n)
        # Launch 1 block of n threads (one warp)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, n, 1, 1, 0, None, args, None)
        sr = ctx.sync()
        if sr != 0:
            return {"correct": False, "time_ms": None,
                    "error": f"sync failed: {sr}"}
        out = ctx.copy_from(d_out, n * 8)
        actual = list(struct.unpack(f"<{n}Q", out))
        correct = (actual == expected)
        time_ms = None
        if mode == "bench":
            ctx.copy_to(d_out, bytes(host_out))
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (n, 1, 1), args)
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms,
            "expected": expected, "actual": actual}


def harness_forge_devfn_trigger(ctx: CUDAContext, func, mode: str) -> dict:
    """FG-1.13B: call double_it(tid) and store result.  Expected output is
    [2*tid for tid in 0..n).  If Forge PTX backend stubs the device call
    to `mov 0`, actual output will be all zeros.

    Param shape (3 args):
        .param .u64 devfn_trigger_param_output_data
        .param .u64 devfn_trigger_param_output_len
        .param .u64 devfn_trigger_param_n
    """
    n = 16
    host_out = (ctypes.c_uint64 * n)(*([0xDEADBEEF] * n))  # sentinel
    expected = [2 * i for i in range(n)]

    d_out = ctx.alloc(n * 8)
    try:
        ctx.copy_to(d_out, bytes(host_out))
        a_out_data = ctypes.c_uint64(d_out)
        a_out_len  = ctypes.c_uint64(n)
        a_n        = ctypes.c_uint64(n)
        args, _hold = _make_args(a_out_data, a_out_len, a_n)
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, n, 1, 1, 0, None, args, None)
        sr = ctx.sync()
        if sr != 0:
            return {"correct": False, "time_ms": None,
                    "error": f"sync failed: {sr}"}
        out = ctx.copy_from(d_out, n * 8)
        actual = list(struct.unpack(f"<{n}Q", out))
        correct = (actual == expected)
        time_ms = None
        if mode == "bench":
            ctx.copy_to(d_out, bytes(host_out))
            time_ms = _bench_launch(ctx, func, (1, 1, 1), (n, 1, 1), args)
    finally:
        ctx.free(d_out)
    return {"correct": correct, "time_ms": time_ms,
            "expected": expected, "actual": actual}


# Bind harnesses to catalog entries now that the functions exist.
_FORGE_KERNELS["reduce_step"]["harness"] = harness_forge_reduce_step
_FORGE_KERNELS["laneid_trigger"]["harness"] = harness_forge_laneid_trigger
_FORGE_KERNELS["devfn_trigger"]["harness"] = harness_forge_devfn_trigger


# ---------------------------------------------------------------------
# FG-1.1: forge measurement + dispatch
# ---------------------------------------------------------------------

def _classify_forge_result(result: dict) -> str:
    """Classify a Forge run into PARITY / NATIVE_WIN / GAP / MIXED / NO_COMPARE.

    Same rules as classify_kernel for hand-crafted runs — uses regs +
    sass_total + sass_non_nop deltas.  Centralized here so the Forge
    artifact's `bucket` field is computed at write-time, not at view-time.
    """
    if result.get("error") or result.get("ptxas") is None:
        return "NO_COMPARE"
    d = result.get("deltas") or {}
    fields = [d.get("regs", 0), d.get("sass_total", 0), d.get("sass_non_nop", 0)]
    if all(f == 0 for f in fields):
        return "PARITY"
    if all(f <= 0 for f in fields) and any(f < 0 for f in fields):
        return "NATIVE_WIN"
    if all(f >= 0 for f in fields) and any(f > 0 for f in fields):
        return "GAP"
    return "MIXED"


def measure_forge_kernel(target: str, mode: str, do_compare: bool,
                         repeat: int, results_dir: Path) -> dict:
    """FG-1.1: full Forge → OpenPTXas → GPU pipeline for a single target.

    Mirrors `measure_kernel` but:
    - Stage 1 invokes Forge via WSL to compile .fg → .ptx
    - The Forge-emitted .ptx is copied into results/<ts>_forge_<target>.ptx
    - Each pipeline stage is recorded in `result["stages"]` with status,
      duration, and (on failure) error/stdout/stderr tail
    - On any stage failure, the function STOPS and returns a partial
      result so the caller can write a failure artifact
    """
    if target not in _FORGE_KERNELS:
        return {"target": target, "error": f"unknown forge target '{target}'",
                "stages": []}

    entry = _FORGE_KERNELS[target]
    fg_path = REPO_FORGE / entry["fg_path"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    result: dict = {
        "schema":       _FORGE_SCHEMA_VERSION,
        "timestamp":    ts,
        "source_mode":  "forge",
        "ptx_source":   "forge",
        "target":       target,
        "display":      entry["display"],
        "mode":         mode,
        "repeat":       repeat,
        "source": {
            "fg_path":       entry["fg_path"],
            "kernel_symbol": entry["kernel_symbol"],
            "language":      "forge",
        },
        "stages":       [],
        "artifacts":    {
            "forge_cu_path":     None,
            "forge_ptx_source":  None,
            "forge_ptx_cached":  None,
            "ours_cubin_size":   None,
            "ptxas_cubin_size":  None,
        },
        "build":        "FAIL",
        "correctness":  "FAIL",
        "ours":         None,
        "ptxas":        None,
        "deltas":       None,
        "bucket":       "NO_COMPARE",
    }

    if not fg_path.exists():
        result["error"] = f"forge source not found: {fg_path}"
        return result

    # ----- Stage 1: forge compile via WSL -----
    print(f"[forge] compiling {entry['fg_path']} ...", flush=True)
    forge_stage = _invoke_forge(fg_path)
    result["stages"].append(forge_stage)
    if forge_stage["status"] != "PASS":
        result["error"] = (
            f"forge compile failed (exit {forge_stage['exit_code']})"
        )
        return result

    # Capture forge outputs and copy ptx into results/.
    forge_cu_src  = fg_path.with_suffix(".cu")
    forge_ptx_src = fg_path.with_suffix(".ptx")
    if not forge_ptx_src.exists():
        result["error"] = (
            f"forge succeeded but no .ptx output at {forge_ptx_src}"
        )
        return result

    results_dir.mkdir(parents=True, exist_ok=True)
    cached_ptx = results_dir / f"{ts}_forge_{target}.ptx"
    cached_ptx.write_bytes(forge_ptx_src.read_bytes())
    result["artifacts"]["forge_cu_path"]    = str(forge_cu_src) if forge_cu_src.exists() else None
    result["artifacts"]["forge_ptx_source"] = str(forge_ptx_src)
    result["artifacts"]["forge_ptx_cached"] = str(cached_ptx)

    ptx_text = cached_ptx.read_text(encoding="utf-8")

    # ----- Stage 2: openptxas assemble -----
    print(f"[forge] assembling via openptxas ...", flush=True)
    t0 = time.perf_counter()
    cubin_ours: bytes | None = None
    report = None
    try:
        cubin_ours, t_compile_ours, report = compile_with_report(ptx_text)
        result["stages"].append({
            "name":        "openptxas_assemble",
            "status":      "PASS",
            "duration_ms": (time.perf_counter() - t0) * 1000.0,
        })
    except Exception as e:
        result["stages"].append({
            "name":        "openptxas_assemble",
            "status":      "FAIL",
            "duration_ms": (time.perf_counter() - t0) * 1000.0,
            "error":       f"{type(e).__name__}: {e}",
        })
        result["error"] = (
            f"openptxas refused forge PTX: {type(e).__name__}: {e}"
        )
        return result

    ours = metrics_from_cubin(cubin_ours)
    ours["compile_ms"] = t_compile_ours * 1000.0
    ours["time_ms_runs"] = []
    result["ours"] = ours
    result["build"] = "PASS"
    result["artifacts"]["ours_cubin_size"] = len(cubin_ours)

    # ----- Stage 2b (optional): ptxas compile for compare -----
    cubin_ptxas: bytes | None = None
    if do_compare:
        t0 = time.perf_counter()
        try:
            cubin_ptxas, t_compile_ptxas = compile_ptxas(ptx_text)
            result["stages"].append({
                "name":        "ptxas_compile",
                "status":      "PASS",
                "duration_ms": (time.perf_counter() - t0) * 1000.0,
            })
            theirs = metrics_from_cubin(cubin_ptxas)
            theirs["compile_ms"] = t_compile_ptxas * 1000.0
            theirs["time_ms_runs"] = []
            result["ptxas"] = theirs
            result["artifacts"]["ptxas_cubin_size"] = len(cubin_ptxas)
        except Exception as e:
            result["stages"].append({
                "name":        "ptxas_compile",
                "status":      "FAIL",
                "duration_ms": (time.perf_counter() - t0) * 1000.0,
                "error":       f"{type(e).__name__}: {e}",
            })
            result["ptxas_error"] = f"{type(e).__name__}: {e}"

    # ----- Stage 3: GPU correctness + benchmarking -----
    print(f"[forge] launching kernel on GPU ...", flush=True)
    ctx = CUDAContext()
    correct = True
    gpu_t0 = time.perf_counter()
    gpu_error: str | None = None
    try:
        if not ctx.load(cubin_ours):
            gpu_error = "cuModuleLoadData failed for openptxas cubin"
        else:
            try:
                func = ctx.get_func(entry["kernel_symbol"])
            except AssertionError as e:
                gpu_error = f"cuModuleGetFunction failed: {e}"

            if gpu_error is None:
                for _ in range(repeat):
                    r = entry["harness"](ctx, func, mode)
                    if not r.get("correct", False):
                        correct = False
                        if "error" in r:
                            gpu_error = r["error"]
                    if r.get("time_ms") is not None:
                        ours["time_ms_runs"].append(r["time_ms"])

                if (result["ptxas"] is not None and cubin_ptxas is not None
                        and gpu_error is None):
                    if ctx.load(cubin_ptxas):
                        func_p = ctx.get_func(entry["kernel_symbol"])
                        for _ in range(repeat):
                            rp = entry["harness"](ctx, func_p, mode)
                            if rp.get("time_ms") is not None:
                                result["ptxas"]["time_ms_runs"].append(
                                    rp["time_ms"]
                                )
                    else:
                        result["ptxas_error"] = (
                            "cuModuleLoadData failed for ptxas cubin"
                        )
    finally:
        ctx.close()

    result["stages"].append({
        "name":        "gpu_correctness",
        "status":      "PASS" if correct and gpu_error is None else "FAIL",
        "duration_ms": (time.perf_counter() - gpu_t0) * 1000.0,
        **({"error": gpu_error} if gpu_error else {}),
    })

    if gpu_error and not correct:
        result["error"] = gpu_error
        return result

    result["correctness"] = "PASS" if correct else "FAIL"

    # ----- Stats + deltas -----
    ours["time_ms_stats"] = _stats(ours["time_ms_runs"])
    if result["ptxas"] is not None:
        result["ptxas"]["time_ms_stats"] = _stats(result["ptxas"]["time_ms_runs"])
        theirs = result["ptxas"]
        deltas = {
            "regs":         ours["regs"]         - theirs["regs"],
            "sass_total":   ours["sass_total"]   - theirs["sass_total"],
            "sass_non_nop": ours["sass_non_nop"] - theirs["sass_non_nop"],
        }
        if (ours["time_ms_stats"] is not None
                and theirs["time_ms_stats"] is not None):
            deltas["time_ms_mean"] = (
                ours["time_ms_stats"]["mean"]
                - theirs["time_ms_stats"]["mean"]
            )
        result["deltas"] = deltas

    result["bucket"] = _classify_forge_result(result)

    if report is not None:
        result["metadata"] = {
            "compaction_attempted": report.attempted,
            "compaction_covered":   report.covered,
            "compacted":            report.gpr_fields_rewritten > 0,
            "compact_regs_before":  report.regs_before,
            "compact_regs_after":   report.regs_after,
            "compacted_insts":      report.compacted_insts,
            "gpr_fields_rewritten": report.gpr_fields_rewritten,
        }

    return result


def _print_forge_block(result: dict, commits: dict) -> None:
    """Print a human-readable summary of a Forge run."""
    print(f"[forge] target={result['target']}  ({result['display']})")
    for s in result.get("stages", []):
        marker = "PASS" if s["status"] == "PASS" else "FAIL"
        ms = s.get("duration_ms", 0.0)
        print(f"  {s['name']:22s} {marker}  ({ms:.1f} ms)")
        if s["status"] != "PASS":
            for line in s.get("stderr_tail", []):
                print(f"    ! {line}")
            for line in s.get("stdout_tail", []):
                print(f"    | {line}")
            if "error" in s:
                print(f"    error: {s['error']}")

    print(f"  build:       {result.get('build', 'FAIL')}")
    print(f"  correctness: {result.get('correctness', 'FAIL')}")
    print(f"  bucket:      {result.get('bucket', 'NO_COMPARE')}")
    print(f"  forge:     {commits.get('forge', '?')}")
    print(f"  opencuda:  {commits.get('opencuda', '?')}")
    print(f"  openptxas: {commits.get('openptxas', '?')}")

    if result.get("error"):
        print(f"  error: {result['error']}")
        return

    ours = result.get("ours") or {}
    if ours:
        print()
        print("  ours:")
        print(f"    regs:         {ours.get('regs', '?')}")
        print(f"    sass_total:   {ours.get('sass_total', '?')}")
        print(f"    sass_non_nop: {ours.get('sass_non_nop', '?')}")
        print(f"    compile_ms:   {ours.get('compile_ms', 0.0):.1f}")
        stats = ours.get("time_ms_stats")
        if stats:
            print(f"    time_ms:      {stats['mean']:.4f}")

    ptxas = result.get("ptxas") or {}
    if ptxas:
        print()
        print("  ptxas:")
        print(f"    regs:         {ptxas.get('regs', '?')}")
        print(f"    sass_total:   {ptxas.get('sass_total', '?')}")
        print(f"    sass_non_nop: {ptxas.get('sass_non_nop', '?')}")
        print(f"    compile_ms:   {ptxas.get('compile_ms', 0.0):.1f}")
        stats = ptxas.get("time_ms_stats")
        if stats:
            print(f"    time_ms:      {stats['mean']:.4f}")

    deltas = result.get("deltas") or {}
    if deltas:
        print()
        print("  delta:")
        print(f"    regs:         {deltas.get('regs', 0):+d}")
        print(f"    sass_total:   {deltas.get('sass_total', 0):+d}")
        print(f"    sass_non_nop: {deltas.get('sass_non_nop', 0):+d}")
        if "time_ms_mean" in deltas:
            print(f"    time_ms_mean: {deltas['time_ms_mean']:+.4f}")


def write_forge_kernel_json(result: dict, commits: dict,
                            results_dir: Path) -> Path:
    """Write a forge_run/v1 artifact next to the cached PTX."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = result["timestamp"]
    target = result["target"]
    artifact = dict(result)  # shallow copy — preserves field order
    artifact["commits"] = commits
    out_path = results_dir / f"{ts}_forge_{target}.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str))
    return out_path


def _cmd_forge_run(args):
    """FG-1.1: workbench forge run --target <name> [--compare ptxas] ..."""
    if args.repeat < 1:
        print("workbench forge run: --repeat must be >= 1", file=sys.stderr)
        return 2
    if args.target not in _FORGE_KERNELS:
        print(
            f"workbench forge run: unknown target '{args.target}'. "
            f"Try `workbench forge list`.",
            file=sys.stderr,
        )
        return 2

    do_compare = (args.compare == "ptxas")
    commits = collect_commits()
    results_dir = Path(args.results_dir)

    result = measure_forge_kernel(
        target=args.target,
        mode=args.mode,
        do_compare=do_compare,
        repeat=args.repeat,
        results_dir=results_dir,
    )

    _print_forge_block(result, commits)
    artifact = write_forge_kernel_json(result, commits, results_dir)
    print()
    print(f"[workbench] forge artifact: {artifact}")

    if "error" in result:
        return 1
    return 0 if result.get("correctness") == "PASS" else 1


def _cmd_forge_list(args):
    """FG-1.1: list available Forge-backed targets."""
    print("Available forge targets:")
    for k, v in _FORGE_KERNELS.items():
        print(f"  {k:20s} {v['display']}")
        print(f"  {'':20s}   source: {v['fg_path']}")
    return 0


# ---------------------------------------------------------------------------
# FG-2 B1: workbench explore
# ---------------------------------------------------------------------------
# One-shot summary of every catalogued kernel: name, class, last bucket,
# and headline metrics.  Pure replay from the most recent suite_all
# artifact plus the most recent per-kernel artifact, with a fallback to
# forge_* artifacts for Forge-backed kernels.

def _find_latest_kernel_record(results_dir: Path, kernel: str) -> dict | None:
    """Find the most recent artifact that contains metrics for `kernel`.

    Search order (newest first):
      1) per-kernel *_<kernel>.json single-kernel artifacts
      2) *_suite_all.json artifacts whose kernels[] list includes the name
    Returns a dict with fields {'bucket','regs','sass_total','sass_non_nop',
    'source','timestamp'} or None.
    """
    if not results_dir.exists():
        return None
    # Gather candidate files sorted by filename timestamp (newest first).
    candidates = sorted(results_dir.glob("*.json"), reverse=True)
    for p in candidates:
        name = p.name
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Case 1: single-kernel artifact (schema WB-0)
        if data.get("kernel") == kernel:
            ours = data.get("ours") or {}
            deltas = data.get("deltas") or {}
            return {
                "bucket":       data.get("bucket", "?"),
                "regs":         ours.get("regs"),
                "sass_total":   ours.get("sass_total"),
                "sass_non_nop": ours.get("sass_non_nop"),
                "source":       name,
                "timestamp":    data.get("timestamp", ""),
            }
        # Case 2: suite_all artifact
        if "kernels" in data and "ranking" in data:
            for rec in data.get("kernels", []):
                if rec.get("kernel") != kernel:
                    continue
                # Find bucket from ranking
                bucket = "?"
                for b, members in data.get("ranking", {}).items():
                    if kernel in members:
                        bucket = b
                        break
                ours = rec.get("ours") or {}
                return {
                    "bucket":       bucket,
                    "regs":         ours.get("regs"),
                    "sass_total":   ours.get("sass_total"),
                    "sass_non_nop": ours.get("sass_non_nop"),
                    "source":       name,
                    "timestamp":    data.get("timestamp", ""),
                }
        # Case 3: forge_run artifact
        if data.get("schema") == _FORGE_SCHEMA_VERSION and data.get("target") == kernel:
            ours = data.get("ours") or {}
            return {
                "bucket":       data.get("bucket", "NO_COMPARE"),
                "regs":         ours.get("regs"),
                "sass_total":   ours.get("sass_total"),
                "sass_non_nop": ours.get("sass_non_nop"),
                "source":       name,
                "timestamp":    data.get("timestamp", ""),
            }
    return None


def _cmd_explore(args):
    """FG-2 B1: enumerate every catalogued kernel with its last known
    bucket + headline metrics.  Includes both hand-crafted workbench
    kernels and Forge-backed kernels.
    """
    results_dir = Path(args.results_dir)

    rows = []
    for name in sorted(KERNELS.keys()):
        rec = _find_latest_kernel_record(results_dir, name)
        rows.append(("hand", name, rec))
    for name in sorted(_FORGE_KERNELS.keys()):
        rec = _find_latest_kernel_record(results_dir, name)
        rows.append(("forge", name, rec))

    def _fmt(v):
        return "-" if v is None else str(v)

    print(f"{'name':<22s} {'class':<6s} {'last bucket':<13s} "
          f"{'regs':>5s} {'sass':>5s} {'nop':>5s}  source")
    print("-" * 78)
    for kind, name, rec in rows:
        if rec is None:
            print(f"{name:<22s} {kind:<6s} {'(no runs)':<13s} "
                  f"{'-':>5s} {'-':>5s} {'-':>5s}  -")
            continue
        print(f"{name:<22s} {kind:<6s} {rec['bucket']:<13s} "
              f"{_fmt(rec['regs']):>5s} {_fmt(rec['sass_total']):>5s} "
              f"{_fmt(rec['sass_non_nop']):>5s}  {rec['source']}")
    print()
    print(f"Total: {len(rows)} kernels  "
          f"({sum(1 for _, _, r in rows if r)} with runs, "
          f"{sum(1 for _, _, r in rows if not r)} without)")
    return 0


# ---------------------------------------------------------------------------
# FG-2 B2: workbench kdiff (one-shot compile + SASS side-by-side)
# ---------------------------------------------------------------------------
def _decode_sass_line(raw: bytes) -> str:
    """Return a short text label for a 16-byte SASS instruction.

    Uses the scoreboard's opcode map for recognized opcodes and falls
    back to `OP_<hex>` for unknown ones.  Follows the convention used
    throughout the codebase (comment strings after each SassInstr).
    """
    if len(raw) < 16:
        return "<short>"
    opc = (raw[0] | (raw[1] << 8)) & 0xFFF
    labels = {
        0x918: 'NOP',     0x947: 'BRA',     0x94d: 'EXIT',
        0x919: 'S2R',     0x9c3: 'S2UR',    0x7b8: 'LDC',
        0xb82: 'LDC.alt', 0x7ac: 'LDCU',
        0x210: 'IADD3',   0x212: 'IADD3X',  0x810: 'IADD3.IMM',
        0x224: 'IMAD',    0x2a4: 'IMAD.RR', 0xc24: 'IMAD.RU',
        0x824: 'IMAD.I',  0x825: 'IMAD.WIDE.I', 0x225: 'IMAD.WIDE',
        0x235: 'IADD.64', 0xc35: 'IADD.64-UR',
        0x20c: 'ISETP',   0xc0c: 'ISETP.RU', 0x80c: 'ISETP.IMM',
        0x202: 'MOV',     0xc02: 'MOV.UR',
        0x986: 'STG',     0x981: 'LDG',
        0x308: 'MUFU',    0x221: 'FADD',    0x223: 'FFMA',
    }
    name = labels.get(opc, f'OP_{opc:03x}')
    return f"{raw.hex()}  {name}"


def _extract_sass_text(cubin: bytes, symbol: str) -> list[str]:
    """Walk .text.<symbol> and return a list of decoded 16-byte rows."""
    e_shoff = struct.unpack_from('<Q', cubin, 40)[0]
    e_shnum = struct.unpack_from('<H', cubin, 60)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 62)[0]
    stoff = struct.unpack_from('<Q', cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    target = b".text." + symbol.encode()
    for i in range(e_shnum):
        base = e_shoff + i * 64
        nm = struct.unpack_from('<I', cubin, base)[0]
        name_end = cubin.index(0, stoff + nm)
        if cubin[stoff + nm:name_end] != target:
            continue
        off = struct.unpack_from('<Q', cubin, base + 24)[0]
        sz = struct.unpack_from('<Q', cubin, base + 32)[0]
        out = []
        for o in range(0, sz, 16):
            out.append(_decode_sass_line(cubin[off + o:off + o + 16]))
        return out
    return []


def _cmd_kdiff(args):
    """FG-2 B2: one-shot compile of a catalogued kernel through both
    OpenPTXas and PTXAS, then print a side-by-side SASS diff plus the
    delta block.
    """
    name = args.kernel
    if name not in KERNELS:
        print(f"workbench kdiff: unknown kernel '{name}'. "
              f"Try `workbench list`.",
              file=sys.stderr)
        return 2
    entry = KERNELS[name]
    symbol = entry["kernel_name"]
    ptx = entry.get("ptx_inline")
    if ptx is None:
        path = entry.get("ptx_path")
        if path is None:
            print(f"workbench kdiff: no PTX source for '{name}'", file=sys.stderr)
            return 2
        ptx = Path(path).read_text(encoding="utf-8")

    try:
        cubin_o, _ = compile_openptxas(ptx)
    except Exception as exc:
        print(f"workbench kdiff: openptxas failed: {exc}", file=sys.stderr)
        return 1
    try:
        cubin_p, _ = compile_ptxas(ptx)
    except Exception as exc:
        print(f"workbench kdiff: ptxas failed: {exc}", file=sys.stderr)
        return 1

    info_o = analyze_cubin(cubin_o, kernel_name=symbol)
    info_p = analyze_cubin(cubin_p, kernel_name=symbol)
    regs_o = _num_gprs(cubin_o, symbol)
    regs_p = _num_gprs(cubin_p, symbol)

    print(f"kernel: {name}")
    print(f"symbol: {symbol}")
    print()
    print(f"{'metric':<14s} {'ours':>8s}  {'ptxas':>8s}  {'delta':>8s}")
    print("-" * 44)
    def _row(label, ov, pv):
        if ov is None or pv is None:
            print(f"{label:<14s} {str(ov):>8s}  {str(pv):>8s}  {'-':>8s}")
            return
        delta = ov - pv
        print(f"{label:<14s} {ov:>8d}  {pv:>8d}  {delta:>+8d}")
    _row("regs",         regs_o, regs_p)
    _row("sass_total",   info_o["n_instrs"], info_p["n_instrs"])
    _row("sass_non_nop", info_o["n_real"],   info_p["n_real"])
    print()

    sass_o = _extract_sass_text(cubin_o, symbol)
    sass_p = _extract_sass_text(cubin_p, symbol)

    print("side-by-side SASS  (! marks lines that differ):")
    print("=" * 92)
    width = 42
    max_len = max(len(sass_o), len(sass_p))
    for i in range(max_len):
        lo = sass_o[i] if i < len(sass_o) else ""
        lp = sass_p[i] if i < len(sass_p) else ""
        # Compare by opcode label (last token after hex)
        lo_op = lo.split("  ")[-1] if lo else ""
        lp_op = lp.split("  ")[-1] if lp else ""
        marker = "!" if lo_op != lp_op else " "
        # Show hex + opcode label, truncate to width
        def _cell(s):
            # Take everything after first double-space once
            if not s: return ""
            return s[:width]
        print(f"{marker} {_cell(lo):<{width}s} | {_cell(lp):<{width}s}")
    return 0


def _num_gprs(cubin: bytes, symbol: str) -> int | None:
    """Best-effort GPR count for a kernel in a cubin via the pipeline's
    helper (`analyze_cubin` returns it for OpenPTXas cubins but not for
    PTXAS ones — in which case we fall back to scanning the text).
    """
    try:
        info = analyze_cubin(cubin, kernel_name=symbol)
    except Exception:
        return None
    return info.get("num_gprs")


def main():
    # Force stdout to UTF-8 so non-ASCII characters in subcommand output
    # (e.g. WB-12.5 diff's `→` arrows) work on the Windows cp1252 console.
    # Safe for ASCII output (cp1252 and UTF-8 agree on ASCII bytes), so
    # WB-12.0's byte-equality lock for `run` is unaffected.  WB-12.3's
    # `dump` writes via sys.stdout.buffer and bypasses text mode entirely
    # so this reconfigure has no effect on it either.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    p = argparse.ArgumentParser(
        prog="workbench",
        description="WB-12.0: kernel workbench (subcommand CLI dashboard)",
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # ---- run ----
    p_run = sub.add_parser(
        "run",
        help="run a kernel or suite",
        description="Run a kernel or suite through openptxas + optional ptxas compare.",
    )
    p_run.add_argument("--kernel", default=None,
                       help=f"one of: {', '.join(sorted(KERNELS))}")
    p_run.add_argument("--suite", default=None,
                       help=f"one of: {', '.join(sorted(SUITES))}")
    p_run.add_argument("--mode", choices=["correct", "bench"], default="correct",
                       help="correct = build+correctness, bench = +benchmark")
    p_run.add_argument("--compare", choices=["ptxas"], default=None,
                       help="if set, also compile via ptxas and report deltas")
    p_run.add_argument("--repeat", type=int, default=1,
                       help="number of measurement repeats (default: 1)")
    p_run.add_argument("--results-dir", default=str(ROOT / "results"),
                       help="directory for JSON artifacts")

    # ---- list ----
    sub.add_parser(
        "list",
        help="list catalog and suites",
        description="List available kernels and suites.",
    )

    # ---- status (WB-12.1) ----
    p_status = sub.add_parser(
        "status",
        help="snapshot the latest suite_all artifact",
        description="Print a snapshot of the most recent suite_all artifact "
                    "(or the artifact specified via --from).  Pure replay — "
                    "does not recompute or rerun anything.",
    )
    p_status.add_argument("--from", dest="from_path", default=None,
                          metavar="ARTIFACT",
                          help="path to a specific suite_all.json (default: latest)")
    p_status.add_argument("--format", choices=["table", "json"], default="table",
                          help="output format (default: table)")
    p_status.add_argument("--results-dir", default=str(ROOT / "results"),
                          help="directory to scan for the latest suite_all.json")

    # ---- show (WB-12.2) ----
    p_show = sub.add_parser(
        "show",
        help="drill down into a single kernel record",
        description="Print the regs / sass / time / delta block for a "
                    "single kernel from the most recent suite_all artifact "
                    "(or the artifact specified via --from).  Pure replay.",
    )
    p_show.add_argument("--kernel", required=True,
                        help=f"one of: {', '.join(sorted(KERNELS))}")
    p_show.add_argument("--from", dest="from_path", default=None,
                        metavar="ARTIFACT",
                        help="path to a specific suite_all.json (default: latest)")
    p_show.add_argument("--format", choices=["table", "json"], default="table",
                        help="output format (default: table)")
    p_show.add_argument("--results-dir", default=str(ROOT / "results"),
                        help="directory to scan for the latest suite_all.json")

    # ---- dump (WB-12.3) ----
    p_dump = sub.add_parser(
        "dump",
        help="raw passthrough of a suite_all artifact",
        description="Print the bytes of a suite_all artifact verbatim. "
                    "No parsing, no validation, no schema checks. "
                    "Use --list to see available artifacts.",
    )
    _dump_mode = p_dump.add_mutually_exclusive_group()
    _dump_mode.add_argument("--latest", action="store_true",
                            help="print the most recent suite_all artifact (default)")
    _dump_mode.add_argument("--from", dest="from_path", default=None,
                            metavar="ARTIFACT",
                            help="print the bytes of a specific artifact")
    _dump_mode.add_argument("--list", action="store_true",
                            help="list available suite_all artifacts")
    p_dump.add_argument("--results-dir", default=str(ROOT / "results"),
                        help="directory to scan for suite_all.json files")

    # ---- history (WB-12.4) ----
    p_hist = sub.add_parser(
        "history",
        help="trend display across all suite_all artifacts",
        description="Walk results/*_suite_all.json in chronological order "
                    "and display aggregate counts per artifact (default), "
                    "or per-kernel trend (--kernel).  Pure replay — every "
                    "value comes straight from the saved artifacts.",
    )
    p_hist.add_argument("--limit", type=int, default=None,
                        help="show only the most recent N entries (default: all)")
    p_hist.add_argument("--kernel", default=None,
                        help="show per-kernel trend instead of aggregate counts")
    p_hist.add_argument("--format", choices=["table", "json"], default="table",
                        help="output format (default: table)")
    p_hist.add_argument("--results-dir", default=str(ROOT / "results"),
                        help="directory to scan for suite_all.json files")

    # ---- diff (WB-12.5) ----
    p_diff = sub.add_parser(
        "diff",
        help="compare two suite_all artifacts",
        description="Compare two suite_all artifacts (default: latest vs "
                    "previous).  Shows aggregate diff and per-kernel "
                    "field-level changes.  Pure replay.",
    )
    p_diff.add_argument("--from", dest="from_path", default=None,
                        metavar="ARTIFACT",
                        help="explicit `from` artifact (default: previous)")
    p_diff.add_argument("--to", dest="to_path", default=None,
                        metavar="ARTIFACT",
                        help="explicit `to` artifact (default: latest)")
    p_diff.add_argument("--kernel", default=None,
                        help="focus on a single kernel")
    p_diff.add_argument("--format", choices=["table", "json"], default="table",
                        help="output format (default: table)")
    p_diff.add_argument("--results-dir", default=str(ROOT / "results"),
                        help="directory to scan for suite_all.json files")

    # ---- forge (FG-1) ----
    p_forge = sub.add_parser(
        "forge",
        help="forge-backed kernel runs (Forge → OpenPTXas → GPU)",
        description="Run kernels through the live Forge → OpenPTXas → GPU "
                    "pipeline.  Forge is invoked via WSL on the .fg source; "
                    "the resulting PTX is cached into results/ and assembled "
                    "by OpenPTXas.",
    )
    forge_sub = p_forge.add_subparsers(dest="forge_cmd", required=True,
                                       metavar="<forge-command>")

    pf_run = forge_sub.add_parser(
        "run",
        help="run a forge-backed kernel through the full pipeline",
    )
    pf_run.add_argument("--target", required=True,
                        help=f"one of: {', '.join(sorted(_FORGE_KERNELS))}")
    pf_run.add_argument("--mode", choices=["correct", "bench"], default="correct",
                        help="correct = build+correctness, bench = +benchmark")
    pf_run.add_argument("--compare", choices=["ptxas"], default=None,
                        help="if set, also compile via ptxas and report deltas")
    pf_run.add_argument("--repeat", type=int, default=1,
                        help="number of measurement repeats (default: 1)")
    pf_run.add_argument("--results-dir", default=str(ROOT / "results"),
                        help="directory for forge artifacts")

    forge_sub.add_parser(
        "list",
        help="list available forge targets",
    )

    # ---- FG-2 B1: explore ----
    p_explore = sub.add_parser(
        "explore",
        help="enumerate every kernel with last-known bucket + metrics",
        description="FG-2 B1.  List every catalogued kernel (hand-crafted "
                    "and Forge-backed) with the most recent known bucket "
                    "and headline metrics (regs / sass_total / sass_non_nop).  "
                    "Pure replay from results/*.json.",
    )
    p_explore.add_argument("--results-dir", default=str(ROOT / "results"),
                           help="directory to scan for artifacts")

    # ---- FG-2 B2: kdiff ----
    p_kdiff = sub.add_parser(
        "kdiff",
        help="one-shot compile + side-by-side SASS diff OURS vs PTXAS",
        description="FG-2 B2.  Compile a single catalogued kernel through "
                    "both OpenPTXas and PTXAS, print the metric deltas, "
                    "and print a side-by-side SASS diff. Marks lines that "
                    "differ with a leading `!`.",
    )
    p_kdiff.add_argument("--kernel", required=True,
                         help=f"one of: {', '.join(sorted(KERNELS))}")

    # ---- FG-2 B3: leaderboard (alias for status) ----
    p_lb = sub.add_parser(
        "leaderboard",
        help="alias for `status` — bucket summary + per-bucket kernel list",
        description="FG-2 B3.  Print the PARITY / NATIVE WIN / GAP / MIXED "
                    "buckets with counts and kernel names from the most "
                    "recent suite_all artifact.  Pure replay.",
    )
    p_lb.add_argument("--from", dest="from_path", default=None,
                      metavar="ARTIFACT",
                      help="path to a specific suite_all.json (default: latest)")
    p_lb.add_argument("--format", choices=["table", "json"], default="table",
                      help="output format (default: table)")
    p_lb.add_argument("--results-dir", default=str(ROOT / "results"),
                      help="directory to scan for the latest suite_all.json")

    # ---- FG-2 top-level flag layer --------------------------------------
    # The task spec asks for four top-level flag forms that aren't native
    # argparse shapes (e.g. `python workbench.py --explore`).  Translate
    # them into the equivalent subcommand invocations before parse_args.
    # Valid rewrites:
    #   --explore            → explore
    #   --leaderboard        → leaderboard
    #   --history <kernel>   → history --kernel <kernel>
    #   --kernel <k> --diff ptxas → kdiff --kernel <k>
    argv = sys.argv[1:]
    if argv and argv[0] == "--explore":
        argv = ["explore"] + argv[1:]
    elif argv and argv[0] == "--leaderboard":
        argv = ["leaderboard"] + argv[1:]
    elif argv and argv[0] == "--history":
        if len(argv) >= 2 and not argv[1].startswith("-"):
            argv = ["history", "--kernel", argv[1]] + argv[2:]
        else:
            argv = ["history"] + argv[1:]
    elif (len(argv) >= 4
          and argv[0] == "--kernel"
          and argv[2] == "--diff"
          and argv[3] == "ptxas"):
        argv = ["kdiff", "--kernel", argv[1]] + argv[4:]

    args = p.parse_args(argv)

    if args.cmd == "run":
        return _cmd_run(args, p_run)
    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "status":
        return _cmd_status(args)
    if args.cmd == "show":
        return _cmd_show(args)
    if args.cmd == "dump":
        return _cmd_dump(args)
    if args.cmd == "history":
        return _cmd_history(args)
    if args.cmd == "diff":
        return _cmd_diff(args)
    if args.cmd == "forge":
        if args.forge_cmd == "run":
            return _cmd_forge_run(args)
        if args.forge_cmd == "list":
            return _cmd_forge_list(args)
        p.error(f"unknown forge subcommand: {args.forge_cmd}")
    if args.cmd == "explore":
        return _cmd_explore(args)
    if args.cmd == "kdiff":
        return _cmd_kdiff(args)
    if args.cmd == "leaderboard":
        # FG-2 B3: leaderboard is a thin alias over status, so it
        # replays the same saved suite_all artifact.
        return _cmd_status(args)
    p.error(f"unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main() or 0)
