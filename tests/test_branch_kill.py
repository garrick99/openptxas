"""
Branch kill matrix: isolate whether taken conditional branches crash.

Test 1: N-sweep on canonical vector_add (all threads branch at N=0)
Test 2: Minimal branch-only kernel (no memory ops on either path)
Test 3: Compare ISETP+BRA encoding against ptxas ground truth
"""
import ctypes
import struct
import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass.encoding.sm_120_opcodes import (
    encode_nop, encode_exit, encode_s2r, encode_ldc, encode_ldcu_64,
    encode_ldcu_32, encode_iadd64_ur, encode_iadd3,
    encode_isetp_ge_and, encode_bra, patch_pred,
    SR_TID_X, SR_CTAID_X, RZ,
)
from sass.isel import SassInstr
from sass.scoreboard import assign_ctrl
from cubin.emitter import emit_cubin, KernelDesc


def build_branch_only_kernel():
    """Minimal kernel: just ISETP + predicated BRA + EXIT.

    Kernel takes one s32 param 'n'.
    If tid.x >= n, branch to EXIT (skip).
    Otherwise fall through to EXIT.
    Both paths just exit - no memory ops at all.

    This isolates pure branch mechanics.
    """
    instrs = bytearray()

    # S2R R0, SR_TID.X
    instrs.extend(encode_s2r(0, SR_TID_X))

    # LDCU.32 UR6, c[0][0x380]  -- load param 'n'
    instrs.extend(encode_ldcu_32(6, 0, 0x380))

    # ISETP.GE.AND P0, R0, UR6  -- P0 = (tid.x >= n)
    instrs.extend(encode_isetp_ge_and(0, 0, 6))

    # @P0 BRA +16  -- if tid >= n, skip to EXIT
    bra_raw = encode_bra(16)  # skip 1 instruction (the NOP)
    bra_raw = patch_pred(bra_raw, 0, neg=False)  # @P0
    instrs.extend(bra_raw)

    # NOP (fallthrough path - represents "do work")
    instrs.extend(encode_nop())

    # EXIT
    instrs.extend(encode_exit())

    return bytes(instrs)


def build_branch_negated_kernel():
    """Same but with negated predicate: @!P0 BRA."""
    instrs = bytearray()
    instrs.extend(encode_s2r(0, SR_TID_X))
    instrs.extend(encode_ldcu_32(6, 0, 0x380))
    instrs.extend(encode_isetp_ge_and(0, 0, 6))  # P0 = (tid >= n)

    # @!P0 BRA +16  -- if tid < n, skip
    bra_raw = encode_bra(16)
    bra_raw = patch_pred(bra_raw, 0, neg=True)  # @!P0
    instrs.extend(bra_raw)

    instrs.extend(encode_nop())
    instrs.extend(encode_exit())
    return bytes(instrs)


def build_unconditional_bra_kernel():
    """Just an unconditional BRA over a NOP to EXIT. No predicate at all."""
    instrs = bytearray()
    instrs.extend(encode_s2r(0, SR_TID_X))
    instrs.extend(encode_bra(16))  # skip NOP
    instrs.extend(encode_nop())
    instrs.extend(encode_exit())
    return bytes(instrs)


def build_no_branch_kernel():
    """Straight-line: S2R + NOP + EXIT. No branch at all."""
    instrs = bytearray()
    instrs.extend(encode_s2r(0, SR_TID_X))
    instrs.extend(encode_nop())
    instrs.extend(encode_exit())
    return bytes(instrs)


def emit(sass_bytes, kernel_name, num_params=1, param_sizes=None):
    if param_sizes is None:
        param_sizes = [4]  # single s32 param
    # Find all EXIT instruction offsets (opcode mask handles predicated exits too)
    exit_offsets = [i for i in range(0, len(sass_bytes), 16)
                    if (struct.unpack_from('<Q', sass_bytes, i)[0] & 0xFFF) == 0x94d]
    if not exit_offsets:
        exit_offsets = [len(sass_bytes) - 16]
    kd = KernelDesc(
        name=kernel_name,
        sass_bytes=sass_bytes,
        num_gprs=8,
        num_params=num_params,
        param_sizes=param_sizes,
        param_offsets={},
        param_base=0x380,
        const0_size=0x390,
        exit_offsets=exit_offsets,
        s2r_offset=0,
    )
    return emit_cubin(kd)


def run_kernel(cubin_data, kernel_name, n_val=0, block_size=32):
    """Launch kernel with one s32 param. Returns 'PASS' or error string."""
    cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
    CUdevice = ctypes.c_int
    CUcontext = ctypes.c_void_p
    CUmodule = ctypes.c_void_p
    CUfunction = ctypes.c_void_p

    cuda.cuInit(0)
    dev = CUdevice()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = CUcontext()
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

    mod = CUmodule()
    err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin_data)
    if err != 0:
        name = ctypes.c_char_p()
        cuda.cuGetErrorName(err, ctypes.byref(name))
        cuda.cuCtxDestroy_v2(ctx)
        return f"LOAD:{name.value.decode()}"

    func = CUfunction()
    err = cuda.cuModuleGetFunction(ctypes.byref(func), mod, kernel_name.encode())
    if err != 0:
        cuda.cuModuleUnload(mod)
        cuda.cuCtxDestroy_v2(ctx)
        return "GETFUNC_FAIL"

    arg_n = ctypes.c_int32(n_val)
    args = (ctypes.c_void_p * 1)(
        ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
    )

    err = cuda.cuLaunchKernel(func, 1,1,1, block_size,1,1, 0, None, args, None)
    if err != 0:
        name = ctypes.c_char_p()
        cuda.cuGetErrorName(err, ctypes.byref(name))
        result = f"LAUNCH:{name.value.decode()}"
    else:
        err = cuda.cuCtxSynchronize()
        if err != 0:
            name = ctypes.c_char_p()
            cuda.cuGetErrorName(err, ctypes.byref(name))
            result = f"CRASH:{name.value.decode()}"
        else:
            result = "PASS"

    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy_v2(ctx)
    return result


def main():
    print("=" * 60)
    print("Branch Kill Matrix")
    print("=" * 60)
    print()

    # Test 0: No branch at all
    print("Test 0: Straight-line (no branch)")
    sass = build_no_branch_kernel()
    cubin = emit(sass, 'test_nobranch')
    r = run_kernel(cubin, 'test_nobranch')
    print(f"  -> {r}")
    print()

    # Test 1: Unconditional BRA
    print("Test 1: Unconditional BRA (skip 1 NOP)")
    sass = build_unconditional_bra_kernel()
    cubin = emit(sass, 'test_uncond')
    r = run_kernel(cubin, 'test_uncond')
    print(f"  -> {r}")
    print()

    # Test 2: @P0 BRA (branch if tid >= n)
    print("Test 2: @P0 BRA (branch if tid >= n)")
    sass = build_branch_only_kernel()
    cubin = emit(sass, 'test_p0')
    for n in [0, 1, 16, 31, 32, 33]:
        r = run_kernel(cubin, 'test_p0', n_val=n)
        print(f"  N={n:3d}  -> {r}")
    print()

    # Test 3: @!P0 BRA (branch if tid < n)
    print("Test 3: @!P0 BRA (branch if tid < n)")
    sass = build_branch_negated_kernel()
    cubin = emit(sass, 'test_np0')
    for n in [0, 1, 16, 31, 32, 33]:
        r = run_kernel(cubin, 'test_np0', n_val=n)
        print(f"  N={n:3d}  -> {r}")
    print()

    # Test 4: Dump ISETP + BRA encoding for inspection
    print("Test 4: ISETP + BRA encoding dump")
    sass = build_branch_only_kernel()
    for i in range(0, len(sass), 16):
        raw = sass[i:i+16]
        opcode = struct.unpack_from('<Q', raw, 0)[0] & 0xFFF
        pred = (raw[1] >> 4) & 0xf
        names = {0x919: 'S2R', 0x7ac: 'LDCU', 0xc0c: 'ISETP', 0x947: 'BRA',
                 0x918: 'NOP', 0x94d: 'EXIT'}
        name = names.get(opcode, f'0x{opcode:03x}')
        pred_str = f'@{"!" if pred >= 8 else ""}P{pred & 7}' if pred != 7 else ''
        print(f"  +{i:3d}: {raw.hex()}  {pred_str:5s} {name}")
    print()

    print("Done.")


if __name__ == '__main__':
    main()
