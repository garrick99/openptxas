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
    ],
    # WB-9: the 6 new kernels in isolation
    "frontier": [
        "smem_cycle", "bar_ldc_xor", "dual_ldg64_dadd",
        "multi_block_atomic", "atom_cas64", "redux_sum",
    ],
}


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


def main():
    p = argparse.ArgumentParser(
        prog="workbench",
        description="WB-1: kernel workbench (multi-run + suite + leaderboard)",
    )
    p.add_argument("--kernel", default=None,
                   help=f"one of: {', '.join(sorted(KERNELS))}")
    p.add_argument("--suite", default=None,
                   help=f"one of: {', '.join(sorted(SUITES))}")
    p.add_argument("--mode", choices=["correct", "bench"], default="correct",
                   help="correct = build+correctness, bench = +benchmark")
    p.add_argument("--compare", choices=["ptxas"], default=None,
                   help="if set, also compile via ptxas and report deltas")
    p.add_argument("--repeat", type=int, default=1,
                   help="number of measurement repeats (default: 1)")
    p.add_argument("--results-dir", default=str(ROOT / "results"),
                   help="directory for JSON artifacts")
    p.add_argument("--list", action="store_true",
                   help="list catalog and exit")
    args = p.parse_args()

    if args.list:
        print("Available kernels:")
        for k, v in KERNELS.items():
            print(f"  {k:20s} {v['display']}")
        print()
        print("Available suites:")
        for s, ks in SUITES.items():
            print(f"  {s:20s} ({len(ks)}) {', '.join(ks)}")
        return 0

    if args.repeat < 1:
        p.error("--repeat must be >= 1")
    if args.kernel and args.suite:
        p.error("--kernel and --suite are mutually exclusive")
    if not args.kernel and not args.suite:
        p.error("one of --kernel / --suite is required (use --list)")

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


if __name__ == "__main__":
    sys.exit(main() or 0)
