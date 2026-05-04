"""
FB-1 end-to-end test: compile a FORGE-emitted CUDA C wrapper through the
fully open-source pipeline (OpenCUDA → OpenPTXas → cubin) and run the
resulting kernel on the GPU, validating that it produces correct output.

The kernel under test is `bit_reverse_qm31` — one of the 5 FORGE wrappers
that achieve byte-equivalent cubin to ptxas's output.  Byte-equivalence
proves the compilers agree; this test proves the kernel actually RUNS
correctly when launched via the CUDA driver API.

This is the smallest concrete demonstration of FB-1 working end-to-end:
no NVIDIA closed-source compiler in the build, just NVIDIA's GPU + driver
loading our cubin and executing it.
"""
from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, r'C:\Users\kraken\openptxas')
sys.path.insert(0, r'C:\Users\kraken\opencuda')
sys.path.insert(0, r'C:\Users\kraken\forge-workbench')

# Re-use OpenCUDA's CUDAContext (has cuModuleLoad / cuLaunchKernel glue).
from opencuda.tests.test_gpu_e2e import CUDAContext, _CUDA


gpu = pytest.mark.skipif(_CUDA is None, reason="No CUDA GPU available")


FORGE_WRAPPER = Path(r"C:\Users\kraken\VortexSTARK\cuda\forge\bit_reverse_qm31_forge.cu")
CUDA_INCLUDE = Path(r"C:\Users\kraken\VortexSTARK\cuda\include")


def _build_cubin_open() -> bytes:
    """Run OpenCUDA + OpenPTXas to produce a cubin from the FORGE wrapper."""
    with tempfile.TemporaryDirectory() as tmp:
        ptx_path = Path(tmp) / "k.ptx"
        r = subprocess.run(
            [sys.executable, "-m", "opencuda", str(FORGE_WRAPPER),
             "--emit-ptx", "--out", str(ptx_path),
             "-I", str(CUDA_INCLUDE)],
            cwd=r"C:\Users\kraken\opencuda",
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode == 0, f"OpenCUDA failed: {r.stderr}"
        from sass.pipeline import compile_ptx_source
        result = compile_ptx_source(ptx_path.read_text())
        assert "bit_reverse_qm31" in result, \
            f"kernel bit_reverse_qm31 not in OpenPTXas output: {list(result.keys())}"
        return result["bit_reverse_qm31"]


def _bit_reverse_u32(v: int, log_n: int) -> int:
    """Reference bit-reverse of a log_n-bit value (matches FORGE's
    bit_reverse_u32 device fn at cuda/forge/bit_reverse_qm31_forge.cu:24)."""
    r = 0
    x = v
    for _ in range(log_n):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


@gpu
def test_bit_reverse_qm31_open_toolchain(cuda_ctx):
    """End-to-end: compile + load + launch + verify FORGE-emitted kernel
    through the all-open-source path."""
    n = 8           # number of QM31 elements
    log_n = 3       # log2(n)
    qm31_size = 16  # 4 × u32 = 16 bytes per QM31

    # Build cubin through the open-source path.
    cubin = _build_cubin_open()
    print(f"\n[FB-1] cubin size: {len(cubin)} bytes")
    # Dump entry point names by parsing the ELF symtab — diagnostic.
    import struct as _s
    e_shoff = _s.unpack_from('<Q', cubin, 0x28)[0]
    e_shentsize = _s.unpack_from('<H', cubin, 0x3a)[0]
    e_shnum = _s.unpack_from('<H', cubin, 0x3c)[0]
    print(f"[FB-1] cubin sections:")
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        name_off = _s.unpack_from('<I', cubin, off)[0]
        sec_off = _s.unpack_from('<Q', cubin, off + 24)[0]
        sec_size = _s.unpack_from('<Q', cubin, off + 32)[0]
        # Get name from .shstrtab
        e_shstrndx = _s.unpack_from('<H', cubin, 0x3e)[0]
        shstr_off = _s.unpack_from('<Q', cubin, e_shoff + e_shstrndx * e_shentsize + 24)[0]
        p = shstr_off + name_off
        name_bytes = []
        while cubin[p] != 0:
            name_bytes.append(cubin[p]); p += 1
        name = bytes(name_bytes).decode('ascii', errors='ignore')
        if name and not name.startswith('.shstrtab') and not name.startswith('.strtab'):
            print(f"  section: {name} @ 0x{sec_off:x} size 0x{sec_size:x}")
    assert cuda_ctx.load(cubin), "cuModuleLoadData rejected our cubin"
    func = cuda_ctx.get_func("bit_reverse_qm31")
    print(f"[FB-1] func handle: {func.value:#x}")

    # Construct deterministic input: element i has QM31 = (i, i+100, i+200, i+300).
    # That makes it easy to verify which input went where after bit-reverse.
    input_words = []
    for i in range(n):
        input_words += [i, i + 100, i + 200, i + 300]
    input_bytes = struct.pack(f'<{len(input_words)}I', *input_words)
    output_bytes_zero = b'\x00' * len(input_bytes)

    d_in = cuda_ctx.alloc(len(input_bytes))
    d_out = cuda_ctx.alloc(len(input_bytes))
    cuda_ctx.copy_to(d_in, input_bytes)
    cuda_ctx.copy_to(d_out, output_bytes_zero)

    # Kernel signature (post-flatten, see project_fb1_status.md):
    #   .param .u64 in_buf_data
    #   .param .u64 in_buf_len     (u32 ELEMENT COUNT, NOT bytes — see host
    #                               wrapper at cuda/bit_reverse_qm31_forge.cu:21
    #                               which sets `len = n * 4u`, where n is the
    #                               QM31 count and 4u is the u32-per-QM31).
    #                               The kernel bounds-checks src_off+3<len in
    #                               u32 index space.
    #   .param .u64 out_buf_data
    #   .param .u64 out_buf_len    (u32 element count)
    #   .param .u64 n              (QM31 element count)
    #   .param .u32 log_n
    in_buf_len = n * 4    # n QM31s × 4 u32 = total u32 count
    out_buf_len = n * 4

    # Pack args with explicit types — CUDAContext.launch() infers types
    # from int magnitude (small ints become c_int32), but the kernel
    # expects u64 / u32 per the .param signature.  Build the arg array
    # manually with the right widths.
    holders = [
        ctypes.c_uint64(d_in),
        ctypes.c_uint64(in_buf_len),
        ctypes.c_uint64(d_out),
        ctypes.c_uint64(out_buf_len),
        ctypes.c_uint64(n),
        ctypes.c_uint32(log_n),
    ]
    ptrs = [ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders]
    args_arr = (ctypes.c_void_p * len(ptrs))(*ptrs)

    # Grid/block from the host wrapper at cuda/bit_reverse_qm31_forge.cu:
    #   threads = 256, blocks = ceil(n / 256).  For n=8, that's 1 block.
    threads = 256
    blocks = (n + threads - 1) // threads
    rc = cuda_ctx.cuda.cuLaunchKernel(
        func, blocks, 1, 1, threads, 1, 1, 0, None, args_arr, None)
    assert rc == 0, f"cuLaunchKernel failed: rc={rc}"
    assert cuda_ctx.sync() == 0, "cuCtxSynchronize reported a kernel error"

    out_bytes = cuda_ctx.copy_from(d_out, len(input_bytes))
    out_words = list(struct.unpack(f'<{len(input_words)}I', out_bytes))

    # Build the expected output by mirroring the kernel: for each input
    # index i, write QM31 i to output index bit_reverse_u32(i, log_n).
    expected = [0] * len(input_words)
    for i in range(n):
        j = _bit_reverse_u32(i, log_n)
        for k in range(4):
            expected[j * 4 + k] = input_words[i * 4 + k]

    assert out_words == expected, (
        f"FB-1 e2e mismatch.  This means the open-toolchain cubin is "
        f"semantically wrong, even though it byte-matches ptxas.\n"
        f"  expected: {expected}\n"
        f"  got:      {out_words}")

    cuda_ctx.free(d_in)
    cuda_ctx.free(d_out)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
