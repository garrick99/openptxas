"""
tests/test_ldg64_ur.py — Systematic test of 64-bit LDG with LDCU->UR addresses.

Tests the full path: LDCU.64 -> IADD.64-UR -> LDG.E.64 -> STG.E.64
with varying NOP gaps to determine minimum latency requirements.

The kernel copies a uint64 from in_ptr to out_ptr:
  1. S2R R0, SR_TID.X          (required for LDCU warm-up)
  2. LDCU.64 UR4, c[0][0x358]  (memory descriptor)
  3. NOP                        (warm-up gap)
  4. LDCU.64 UR6, c[0][0x380]  (out_ptr param)
  5. NOP                        (warm-up gap for UR6)
  6. LDCU.64 UR8, c[0][0x388]  (in_ptr param)
  7. NOP                        (warm-up gap for UR8)
  8. IADD.64-UR R2, RZ, UR8    (materialize in_ptr)
  9. [N NOPs — gap under test]
  10. LDG.E.64 R4, desc[UR4][R2.64]  (load from in_ptr)
  11. IADD.64-UR R6, RZ, UR6   (materialize out_ptr)
  12. STG.E.64 desc[UR4][R6.64], R4  (store to out_ptr)
  13. EXIT
"""
import ctypes
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass.encoding.sm_120_opcodes import (
    encode_nop, encode_exit, encode_s2r, encode_ldcu_64,
    encode_iadd64_ur, encode_ldg_e_64, encode_stg_e_64,
    SR_TID_X, RZ,
)
from cubin.emitter import emit_cubin, KernelDesc

# CUDA driver API
cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64

def check(err, msg=""):
    if err != 0:
        name = ctypes.c_char_p()
        desc = ctypes.c_char_p()
        cuda.cuGetErrorName(err, ctypes.byref(name))
        cuda.cuGetErrorString(err, ctypes.byref(desc))
        n = name.value.decode() if name.value else "?"
        d = desc.value.decode() if desc.value else "?"
        raise RuntimeError(f"CUDA ERROR {err}: {n} - {d} [{msg}]")


def build_ldg64_kernel(nop_gap: int = 0, nop_after_ldcu: int = 1) -> bytes:
    """Build a minimal LDG.E.64 copy kernel with configurable NOP gaps.

    Args:
        nop_gap: Number of NOPs between IADD.64-UR and LDG.E.64
        nop_after_ldcu: Number of NOPs after each LDCU.64 (warm-up)
    """
    instrs = bytearray()

    # S2R R0, SR_TID.X — required for LDCU warm-up on SM_120
    instrs.extend(encode_s2r(0, SR_TID_X))

    # LDCU.64 UR4, c[0][0x358] — memory descriptor
    instrs.extend(encode_ldcu_64(4, 0, 0x358))

    # Warm-up NOPs after descriptor load
    for _ in range(nop_after_ldcu):
        instrs.extend(encode_nop())

    # LDCU.64 UR8, c[0][0x388] — in_ptr (param 1)
    instrs.extend(encode_ldcu_64(8, 0, 0x388))

    # Warm-up NOPs
    for _ in range(nop_after_ldcu):
        instrs.extend(encode_nop())

    # LDCU.64 UR6, c[0][0x380] — out_ptr (param 0)
    instrs.extend(encode_ldcu_64(6, 0, 0x380))

    # Warm-up NOPs
    for _ in range(nop_after_ldcu):
        instrs.extend(encode_nop())

    # IADD.64-UR R2, RZ, UR8 — materialize in_ptr into GPRs
    instrs.extend(encode_iadd64_ur(2, RZ, 8))

    # Gap under test: NOPs between IADD.64-UR and LDG.E.64
    for _ in range(nop_gap):
        instrs.extend(encode_nop())

    # LDG.E.64 R4, desc[UR4][R2.64] — load 64 bits from in_ptr
    instrs.extend(encode_ldg_e_64(4, 4, 2))

    # IADD.64-UR R6, RZ, UR6 — materialize out_ptr into GPRs
    instrs.extend(encode_iadd64_ur(6, RZ, 6))

    # STG.E.64 desc[UR4][R6.64], R4 — store to out_ptr
    instrs.extend(encode_stg_e_64(4, 6, 4))

    # EXIT
    instrs.extend(encode_exit())

    sass = bytes(instrs)

    # Count GPRs used: R0-R7 (dest pairs R2-R3, R4-R5, R6-R7, plus R0 from S2R)
    num_gprs = 16  # conservative: use large capmerc template

    # Find EXIT offset and S2R offset
    exit_offset = len(sass) - 16  # EXIT is last
    s2r_offset = 0  # S2R is first

    kd = KernelDesc(
        name='ldg64_ur_test',
        sass_bytes=sass,
        num_gprs=num_gprs,
        num_params=2,
        param_sizes=[8, 8],
        param_offsets={'out_ptr': 0x380, 'in_ptr': 0x388},
        param_base=0x380,
        const0_size=0x390,
        exit_offset=exit_offset,
        s2r_offset=s2r_offset,
    )
    return emit_cubin(kd)


def run_variant(nop_gap: int, nop_after_ldcu: int, verbose: bool = False) -> str:
    """Test one configuration. Returns 'PASS', 'FAIL:reason', or 'CRASH:reason'."""
    try:
        cubin_data = build_ldg64_kernel(nop_gap=nop_gap, nop_after_ldcu=nop_after_ldcu)
    except Exception as e:
        return f"BUILD_ERR:{e}"

    # Write cubin to temp file
    cubin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'probe_work', f'ldg64_gap{nop_gap}_warm{nop_after_ldcu}.cubin')
    cubin_path = os.path.normpath(cubin_path)
    with open(cubin_path, 'wb') as f:
        f.write(cubin_data)

    if verbose:
        print(f"  Cubin: {len(cubin_data)} bytes, text section instructions: "
              f"{7 + nop_gap + nop_after_ldcu * 3}")

    # Load and execute
    try:
        check(cuda.cuInit(0), "cuInit")
        dev = CUdevice()
        check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        ctx = CUcontext()
        check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

        mod = CUmodule()
        err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin_data)
        if err != 0:
            name = ctypes.c_char_p()
            cuda.cuGetErrorName(err, ctypes.byref(name))
            n = name.value.decode() if name.value else "?"
            cuda.cuCtxDestroy_v2(ctx)
            return f"LOAD_FAIL:{n}(err={err})"

        func = CUfunction()
        err = cuda.cuModuleGetFunction(ctypes.byref(func), mod, b'ldg64_ur_test')
        if err != 0:
            cuda.cuModuleUnload(mod)
            cuda.cuCtxDestroy_v2(ctx)
            return f"GETFUNC_FAIL:err={err}"

        # Allocate device memory
        d_in = CUdeviceptr()
        d_out = CUdeviceptr()
        check(cuda.cuMemAlloc_v2(ctypes.byref(d_in), 8), "cuMemAlloc d_in")
        check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 8), "cuMemAlloc d_out")

        # Write test pattern
        pattern = 0xDEADBEEFCAFEBABE
        h_in = struct.pack('<Q', pattern)
        check(cuda.cuMemcpyHtoD_v2(d_in, h_in, 8), "memcpy in")
        check(cuda.cuMemcpyHtoD_v2(d_out, struct.pack('<Q', 0), 8), "memcpy out")

        # Launch: kernel(out_ptr, in_ptr)
        arg_out = ctypes.c_uint64(d_out.value)
        arg_in = ctypes.c_uint64(d_in.value)
        args = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
        )

        err = cuda.cuLaunchKernel(func, 1,1,1, 1,1,1, 0, None, args, None)
        if err != 0:
            name = ctypes.c_char_p()
            cuda.cuGetErrorName(err, ctypes.byref(name))
            n = name.value.decode() if name.value else "?"
            cuda.cuMemFree_v2(d_in)
            cuda.cuMemFree_v2(d_out)
            cuda.cuModuleUnload(mod)
            cuda.cuCtxDestroy_v2(ctx)
            return f"LAUNCH_FAIL:{n}"

        err = cuda.cuCtxSynchronize()
        if err != 0:
            name = ctypes.c_char_p()
            cuda.cuGetErrorName(err, ctypes.byref(name))
            n = name.value.decode() if name.value else "?"
            cuda.cuMemFree_v2(d_in)
            cuda.cuMemFree_v2(d_out)
            cuda.cuModuleUnload(mod)
            cuda.cuCtxDestroy_v2(ctx)
            return f"CRASH:{n}"

        # Read back
        h_out = ctypes.create_string_buffer(8)
        check(cuda.cuMemcpyDtoH_v2(h_out, d_out, 8), "memcpy out")
        out_val = struct.unpack('<Q', h_out.raw)[0]

        cuda.cuMemFree_v2(d_in)
        cuda.cuMemFree_v2(d_out)
        cuda.cuModuleUnload(mod)
        cuda.cuCtxDestroy_v2(ctx)

        if out_val == pattern:
            return "PASS"
        else:
            return f"WRONG:0x{out_val:016x}"

    except RuntimeError as e:
        return f"ERROR:{e}"
    except Exception as e:
        return f"EXCEPTION:{e}"


def main():
    print("=" * 70)
    print("LDG.E.64 with LDCU->UR Addresses -- NOP Gap Matrix")
    print("=" * 70)
    print()
    print("Tests LDCU.64->IADD.64-UR->[N NOPs]->LDG.E.64->STG.E.64")
    print("Expected: 0xDEADBEEFCAFEBABE copied from in_ptr to out_ptr")
    print()

    # Phase 1: Test LDCU warm-up NOPs with fixed gap
    print("Phase 1: LDCU warm-up variation (gap=4 fixed)")
    print("-" * 50)
    for warm in [0, 1, 2, 3, 4]:
        result = run_variant(nop_gap=4, nop_after_ldcu=warm, verbose=True)
        status = "OK" if result == "PASS" else result
        print(f"  warm={warm}  ->  {status}")
    print()

    # Phase 2: Test IADD->LDG gap with warm-up=1
    print("Phase 2: IADD.64-UR -> LDG.E.64 gap (warm=1 fixed)")
    print("-" * 50)
    for gap in [0, 1, 2, 4, 8, 12, 16]:
        result = run_variant(nop_gap=gap, nop_after_ldcu=1, verbose=True)
        status = "OK" if result == "PASS" else result
        print(f"  gap={gap:2d}  ->  {status}")
    print()

    # Phase 3: Warm=2, varying gap
    print("Phase 3: gap variation with warm=2")
    print("-" * 50)
    for gap in [0, 1, 2, 4, 8]:
        result = run_variant(nop_gap=gap, nop_after_ldcu=2, verbose=True)
        status = "OK" if result == "PASS" else result
        print(f"  gap={gap:2d}  ->  {status}")
    print()

    print("Done.")


if __name__ == '__main__':
    main()
