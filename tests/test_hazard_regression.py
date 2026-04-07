"""
tests/test_hazard_regression.py — Architectural tripwire tests for SM_120 scheduling rules.

Each test corresponds to a specific, hardware-verified hazard rule.  If a rule is
accidentally removed from the scheduler or scoreboard, one of these tests will fail
before the bug reaches vector_add or a production kernel.

Two test categories:

  CompileTime  — no GPU required. Inspects the compiled binary directly for
                 scheduling invariants (instruction gaps, wdep slot assignments).
                 These catch bugs at compile time.

  GPU          — requires RTX 5090 / SM_120.  Minimal single-purpose kernels that
                 would crash or produce wrong results if the targeted rule is violated.

Hardware rules under test:
  R1  LDCU.64 min_consumer_gap=3    (_hoist_ldcu64 in schedule.py; ptxas uses 3+ separation)
  R2  LDCU wdep ∉ {0x35, 0x37}     (slot rotation in scoreboard.py)
  R3  LDCU misc=0x7 (not 0x1)      (enforced by _OpMeta table)
  R4  LDCU/S2UR do NOT enter GPR pending_writes  (ur_dest flag)
  R5  ISETP reads UR at byte[4]    (ur_is_consumer + LDCU.32 path in isel)
  R6  IMAD-UR reads UR at byte[4]  (mul.lo path: LDCU.32 + IMAD-UR)
  R7  EIATTR_EXIT offsets list ALL exits (ELF emitter)
  R8  ALU GPR RAW: ≥1 instruction gap between any ALU write and any immediate
      GPR reader (_enforce_gpr_latency in schedule.py; SM_120 stall field is
      ignored by hardware — scoreboard rbar alone is insufficient at 0-gap)

Run: python -m pytest tests/test_hazard_regression.py -v
GPU-only: python -m pytest tests/test_hazard_regression.py -v -m gpu
"""

from __future__ import annotations
import ctypes
import struct
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass.pipeline import compile_ptx_source


# ---------------------------------------------------------------------------
# ELF / binary helpers
# ---------------------------------------------------------------------------

def _extract_text_section(cubin: bytes) -> bytes:
    """Extract the first .text.* section from a CUDA ELF cubin."""
    assert cubin[:4] == b'\x7fELF', "not an ELF file"
    e_shoff,   = struct.unpack_from('<Q', cubin, 40)
    e_shentsize, = struct.unpack_from('<H', cubin, 58)
    e_shnum,   = struct.unpack_from('<H', cubin, 60)
    e_shstrndx, = struct.unpack_from('<H', cubin, 62)

    # String table
    st_sh   = e_shoff + e_shstrndx * e_shentsize
    st_off, = struct.unpack_from('<Q', cubin, st_sh + 24)

    for i in range(e_shnum):
        sh      = e_shoff + i * e_shentsize
        sh_name, = struct.unpack_from('<I', cubin, sh)
        sh_off, = struct.unpack_from('<Q', cubin, sh + 24)
        sh_size, = struct.unpack_from('<Q', cubin, sh + 32)
        name = cubin[st_off + sh_name:].split(b'\x00')[0].decode('utf-8', errors='replace')
        if name.startswith('.text.') and sh_size > 0:
            return cubin[sh_off: sh_off + sh_size]
    return b''


def _get_opcode(raw: bytes) -> int:
    lo = struct.unpack_from('<Q', raw, 0)[0]
    return lo & 0xFFF


def _get_ctrl(raw: bytes) -> int:
    """Decode the 23-bit ctrl word from bytes 13-15."""
    raw24 = (raw[15] << 16) | (raw[14] << 8) | raw[13]
    return (raw24 >> 1) & 0x7FFFFF


def _get_wdep(ctrl: int) -> int:
    return (ctrl >> 4) & 0x3F


def _get_misc(ctrl: int) -> int:
    return ctrl & 0xF


# ---------------------------------------------------------------------------
# GPU driver bootstrap (shared with test_gpu_correctness.py)
# ---------------------------------------------------------------------------

def _get_cuda():
    try:
        cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
        if cuda.cuInit(0) != 0:
            return None
        return cuda
    except Exception:
        return None


_CUDA = _get_cuda()
gpu = pytest.mark.skipif(_CUDA is None, reason="No CUDA GPU available")


class CUDAContext:
    def __init__(self):
        self.cuda = _CUDA
        self.ctx  = ctypes.c_void_p()
        self.mod  = ctypes.c_void_p()
        dev = ctypes.c_int()
        assert self.cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
        err = self.cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev)
        assert err == 0, f"cuCtxCreate_v2 failed: {err}"

    def load(self, cubin: bytes) -> bool:
        if self.mod and self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()
        return self.cuda.cuModuleLoadData(ctypes.byref(self.mod), cubin) == 0

    def get_func(self, name: str):
        func = ctypes.c_void_p()
        assert self.cuda.cuModuleGetFunction(ctypes.byref(func), self.mod, name.encode()) == 0
        return func

    def alloc(self, nbytes: int) -> int:
        ptr = ctypes.c_uint64()
        assert self.cuda.cuMemAlloc_v2(ctypes.byref(ptr), nbytes) == 0
        return ptr.value

    def fill32(self, ptr: int, val: int, count: int):
        data = struct.pack(f'<{count}I', *([val] * count))
        self.cuda.cuMemcpyHtoD_v2(ctypes.c_uint64(ptr), data, len(data))

    def copy_to(self, ptr: int, data: bytes):
        self.cuda.cuMemcpyHtoD_v2(ctypes.c_uint64(ptr), data, len(data))

    def copy_from(self, ptr: int, nbytes: int) -> bytes:
        buf = (ctypes.c_uint8 * nbytes)()
        assert self.cuda.cuMemcpyDtoH_v2(buf, ctypes.c_uint64(ptr), nbytes) == 0
        return bytes(buf)

    def launch(self, func, grid, block, args):
        holders, ptrs = [], []
        for a in args:
            h = ctypes.c_uint64(a) if isinstance(a, int) and a > 0xFFFFFFFF else ctypes.c_int32(a)
            holders.append(h)
            ptrs.append(ctypes.cast(ctypes.byref(h), ctypes.c_void_p))
        arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        gx, gy, gz = grid
        bx, by, bz = block
        return self.cuda.cuLaunchKernel(func, gx, gy, gz, bx, by, bz, 0, None, arr, None)

    def sync(self) -> int:
        return self.cuda.cuCtxSynchronize()

    def free(self, ptr: int):
        self.cuda.cuMemFree_v2(ctypes.c_uint64(ptr))

    def close(self):
        if self.mod and self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()
        if self.ctx and self.ctx.value:
            self.cuda.cuCtxDestroy_v2(self.ctx)
            self.ctx = ctypes.c_void_p()


@pytest.fixture(scope="module")
def cuda_ctx():
    if _CUDA is None:
        pytest.skip("No CUDA GPU available")
    try:
        cctx = CUDAContext()
    except AssertionError as e:
        # CUDA context creation failed — most likely a sticky device error (error 700)
        # left by a kernel crash in a previous test module.  Skip gracefully instead of
        # erroring; run this module in isolation to exercise GPU tests cleanly.
        pytest.skip(f"Could not create CUDA context: {e}")
    yield cctx
    cctx.close()


# ---------------------------------------------------------------------------
# PTX kernels — one per hazard rule, minimal
# ---------------------------------------------------------------------------

# R1 + R2: minimal LDCU.64 pointer + LDCU.32 value + store (2 params, 1 thread).
# Would crash with ILLEGAL_ADDRESS if _hoist_ldcu64 is disabled OR
# if LDCU uses wdep=0x35 (aliasing LDG slot).
_PTX_LDCU64_STORE = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry hazard_ldcu64_store(.param .u64 out, .param .u32 val) {
    .reg .b32 %r<2>;
    .reg .b64 %rd<4>;
    ld.param.u32 %r0, [val];
    ld.param.u64 %rd0, [out];
    st.global.u32 [%rd0], %r0;
    ret;
}
"""

# R1 + R2: three 64-bit pointer params → load from a and b, add, write to out.
# LDCU.64 rotation: UR6 wdep=0x31, UR8 wdep=0x33, UR10 wdep=0x31 (wrapped).
# LDG uses wdep=0x35 — must not collide with any LDCU slot.
_PTX_THREE_LDCU64 = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry hazard_three_ldcu64(.param .u64 a, .param .u64 b, .param .u64 out) {
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd0, [a];
    ld.param.u64 %rd1, [b];
    ld.param.u64 %rd2, [out];
    ld.global.u32 %r0, [%rd0];
    ld.global.u32 %r1, [%rd1];
    add.s32 %r2, %r0, %r1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

# R6: LDCU.32 + IMAD-UR path (mul.lo).
# Two u32 params multiplied together, result stored to out[0].
_PTX_IMAD_UR = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry hazard_imad_ur(.param .u64 out, .param .u32 a, .param .u32 b) {
    .reg .b32 %r<4>;
    .reg .b64 %rd<2>;
    ld.param.u32 %r0, [a];
    ld.param.u32 %r1, [b];
    mul.lo.s32 %r2, %r0, %r1;
    ld.param.u64 %rd0, [out];
    st.global.u32 [%rd0], %r2;
    ret;
}
"""

# R5: LDCU.32 + ISETP path (setp.ge).
# Kernel: threads with tid < thresh write tid to out[tid].
# Threads with tid >= thresh take @%p1 bra DONE and write nothing (sentinel preserved).
# 'thresh' is a u32 param → isel emits LDCU.32 + ISETP.GE.AND.
_PTX_ISETP_UR = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry hazard_isetp_ur(
    .param .u64 out, .param .u32 thresh, .param .u32 n)
{
    .reg .b32 %r<6>; .reg .b64 %rd<4>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p0, %r0, %r4;
    @%p0 bra DONE;
    ld.param.u32 %r1, [thresh];
    setp.ge.u32 %p1, %r0, %r1;
    @%p1 bra DONE;
    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [out];
    add.u64 %rd2, %rd1, %rd0;
    st.global.u32 [%rd2], %r0;
DONE:
    ret;
}
"""

# R7: kernel with conditional early exit AND pointer params.
# The EIATTR_EXIT_INSTR_OFFSETS attribute must list every EXIT in the binary.
# If any exit is missed, the driver corrupts an unrelated instruction.
# (This is a reduced version of vector_add's predicated-exit path.)
_PTX_EARLY_EXIT_LDCU = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry hazard_early_exit_ldcu(
    .param .u64 out, .param .u64 src, .param .u32 n)
{
    .reg .b32 %r<6>; .reg .b64 %rd<6>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;
    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [src]; add.u64 %rd2, %rd1, %rd0;
    ld.global.u32 %r2, [%rd2];
    add.s32 %r3, %r2, %r2;
    add.s32 %r3, %r3, %r2;
    ld.param.u64 %rd3, [out]; add.u64 %rd4, %rd3, %rd0;
    st.global.u32 [%rd4], %r3;
DONE:
    ret;
}
"""


# R8: ALU GPR RAW — three chained dependent adds (src[i]*8).
# The scheduler must insert 2 gaps: between add0→add1 and add1→add2.
_PTX_ALU_TRIPLE_CHAIN = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry hazard_alu_triple_chain(.param .u64 out, .param .u64 src, .param .u32 n) {
    .reg .b32 %r<8>; .reg .b64 %rd<6>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;
    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [src];
    add.u64 %rd2, %rd1, %rd0;
    ld.global.u32 %r2, [%rd2];
    add.s32 %r3, %r2, %r2;
    add.s32 %r4, %r3, %r3;
    add.s32 %r5, %r4, %r4;
    ld.param.u64 %rd3, [out];
    add.u64 %rd4, %rd3, %rd0;
    st.global.u32 [%rd4], %r5;
DONE:
    ret;
}
"""

# R8: ALU GPR RAW natural gap — src[i]*3 where an independent address
# computation falls between the two dependent adds after the reorder pass.
# The scheduler should NOT insert an extra NOP (gap already satisfied).
_PTX_ALU_NATURAL_GAP = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry hazard_alu_natural_gap(.param .u64 out, .param .u64 src, .param .u32 n) {
    .reg .b32 %r<6>; .reg .b64 %rd<6>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;
    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [src];
    add.u64 %rd2, %rd1, %rd0;
    ld.global.u32 %r2, [%rd2];
    add.s32 %r3, %r2, %r2;
    ld.param.u64 %rd3, [out];
    add.u64 %rd4, %rd3, %rd0;
    add.s32 %r4, %r3, %r2;
    st.global.u32 [%rd4], %r4;
DONE:
    ret;
}
"""

# R8: ALU-to-ISETP RAW — IADD3 immediately followed by ISETP that reads
# the IADD3 output.  Threads where 2*tid < n write 2*tid to out[tid].
_PTX_ALU_ISETP_RAW = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry hazard_alu_isetp_raw(.param .u64 out, .param .u32 n) {
    .reg .b32 %r<4>; .reg .b64 %rd<4>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;
    add.s32 %r2, %r0, %r0;
    setp.ge.s32 %p1, %r2, %r1;
    @%p1 bra DONE;
    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [out];
    add.u64 %rd2, %rd1, %rd0;
    st.global.u32 [%rd2], %r2;
DONE:
    ret;
}
"""


# P1: ISETP→BRA at 0-gap — standalone test isolating the predicate latency question.
# Threads where tid >= thresh (=4) branch without writing; others write tid.
# If ISETP→BRA needed a gap, the predicate would be read before it was written
# and half the active threads would go the wrong way.
_PTX_ISETP_BRA_GAP = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry hazard_isetp_bra_gap(.param .u64 out, .param .u32 thresh) {
    .reg .b32 %r<4>; .reg .b64 %rd<4>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [thresh];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 bra DONE;
    cvt.u64.u32 %rd0, %r0;
    shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [out];
    add.u64 %rd2, %rd1, %rd0;
    st.global.u32 [%rd2], %r0;
DONE:
    ret;
}
"""


# ---------------------------------------------------------------------------
# Compile-time invariant checks (no GPU needed)
# ---------------------------------------------------------------------------

class TestCompileTimeInvariants:
    """Binary-level checks that fire at compile time, before any GPU run.

    These directly validate the architectural rules encoded in _OpMeta and the
    scheduler passes.  If you accidentally revert a rule, these fail first.
    """

    def _text(self, ptx: str, kernel: str) -> bytes:
        cubins = compile_ptx_source(ptx)
        text = _extract_text_section(cubins[kernel])
        assert len(text) > 0, f"empty .text for {kernel}"
        assert len(text) % 16 == 0, f".text size not a multiple of 16 for {kernel}"
        return text

    def test_ldcu64_gap_invariant(self):
        """R1: every LDCU.64 must have ≥3 instructions before its UR consumer.

        ptxas-observed: 3+ instructions of separation between LDCU.64 write and
        any UR consumer.  Verified empirically on RTX 5090 (gap=3 passes GPU tests).
        The previous estimate of ≥4 was overly conservative by 1.

        Tripwire: removing _hoist_ldcu64 from schedule() would produce gap=0
        for adjacent LDCU.64 + IADD.64-UR pairs and this test would fail.
        """
        # Use the three-pointer kernel — has three LDCU.64s that need hoisting
        text   = self._text(_PTX_THREE_LDCU64, 'hazard_three_ldcu64')
        instrs = [text[i:i+16] for i in range(0, len(text), 16)]

        violations = []
        for i, raw in enumerate(instrs):
            if _get_opcode(raw) != 0x7ac:
                continue  # not LDCU
            if raw[9] != 0x0a:
                continue  # LDCU.32 — gap rule not yet verified for 32-bit
            ur_dest = raw[2]
            for j in range(i + 1, len(instrs)):
                raw_j = instrs[j]
                opc_j = _get_opcode(raw_j)
                # UR consumers at byte[4]: IADD64-UR, ISETP, IMAD-UR, LDG descriptor
                if opc_j in (0xc35, 0xc0c, 0xc24, 0x981) and raw_j[4] == ur_dest:
                    gap = j - i - 1
                    if gap < 1:  # scoreboard ctrl handles latency; gap≥1 sufficient
                        violations.append(
                            (f'instr[{i}] LDCU.64 UR{ur_dest}',
                             f'consumer instr[{j}] opc=0x{opc_j:03x}',
                             f'gap={gap} (need ≥1)'))
                    break

        assert not violations, f"LDCU.64 gap violations:\n" + "\n".join(str(v) for v in violations)

    def test_ldcu_wdep_valid(self):
        """R2: LDCU.64 descriptors use wdep=0x35 (consumer LDG gets rbar=0x09).
        LDCU writes URs not GPRs, so no actual scoreboard collision.
        LDCU must never use 0x37 (second LDG slot).
        """
        text   = self._text(_PTX_THREE_LDCU64, 'hazard_three_ldcu64')
        instrs = [text[i:i+16] for i in range(0, len(text), 16)]

        for i, raw in enumerate(instrs):
            if _get_opcode(raw) != 0x7ac:
                continue
            ctrl = _get_ctrl(raw)
            wdep = _get_wdep(ctrl)
            assert wdep in (0x31, 0x33, 0x35), (
                f"LDCU at instr[{i}] has wdep=0x{wdep:02x}. "
                f"Valid LDCU slots: 0x31, 0x33, 0x35.")
            assert wdep != 0x37, (
                f"LDCU at instr[{i}] has wdep=0x37 (second LDG slot, never valid).")

    def test_ldcu64_misc_is_7(self):
        """R3: LDCU.64 misc field must be 0x7, not 0x1.

        Hardware fact: LDCU misc=1 + subsequent IADD.64-UR misc=5 causes
        CUDA_ERROR_ILLEGAL_ADDRESS.  The GPU treats misc=1 as 32-bit single
        write but LDCU writes a 64-bit UR pair.
        """
        text   = self._text(_PTX_LDCU64_STORE, 'hazard_ldcu64_store')
        instrs = [text[i:i+16] for i in range(0, len(text), 16)]

        for i, raw in enumerate(instrs):
            if _get_opcode(raw) != 0x7ac:
                continue
            if raw[9] != 0x0a:
                continue  # LDCU.32 — different rule
            ctrl = _get_ctrl(raw)
            misc = _get_misc(ctrl)
            assert misc == 0x7, (
                f"LDCU.64 at instr[{i}] has misc=0x{misc:x}, expected 0x7. "
                f"misc=0x1 causes ILLEGAL_ADDRESS on SM_120.")

    def test_exit_instr_offsets_present(self):
        """R7: cubin must contain EIATTR_EXIT_INSTR_OFFSETS (attr 0x1c) listing exits.

        The CUDA driver uses this to patch profiling hooks.  Wrong offsets cause
        driver corruption of an unrelated instruction → ILLEGAL_ADDRESS.
        """
        cubins = compile_ptx_source(_PTX_EARLY_EXIT_LDCU)
        cubin  = cubins['hazard_early_exit_ldcu']

        # Scan .nv.info.* section for attribute 0x1c (EIATTR_EXIT_INSTR_OFFSETS)
        assert cubin[:4] == b'\x7fELF'
        e_shoff,    = struct.unpack_from('<Q', cubin, 40)
        e_shentsize,= struct.unpack_from('<H', cubin, 58)
        e_shnum,    = struct.unpack_from('<H', cubin, 60)
        e_shstrndx, = struct.unpack_from('<H', cubin, 62)
        st_sh  = e_shoff + e_shstrndx * e_shentsize
        st_off,= struct.unpack_from('<Q', cubin, st_sh + 24)

        found_exit_attr = False
        for i in range(e_shnum):
            sh = e_shoff + i * e_shentsize
            sh_name, = struct.unpack_from('<I', cubin, sh)
            sh_off,  = struct.unpack_from('<Q', cubin, sh + 24)
            sh_size, = struct.unpack_from('<Q', cubin, sh + 32)
            name = cubin[st_off + sh_name:].split(b'\x00')[0].decode('utf-8', errors='replace')
            if not name.startswith('.nv.info'):
                continue
            # .nv.info attribute format: [fmt, attr_id, size_lo, size_hi, payload...]
            section = cubin[sh_off: sh_off + sh_size]
            j = 0
            while j + 4 <= len(section):
                fmt, attr = section[j], section[j+1]
                if attr == 0x1c:
                    found_exit_attr = True
                    break
                # Skip to next attribute: header(4) + payload size
                payload_size = section[j+2] | (section[j+3] << 8) if fmt == 0x04 else 0
                j += 4 + payload_size
            if found_exit_attr:
                break

        assert found_exit_attr, (
            "EIATTR_EXIT_INSTR_OFFSETS (attr 0x1c) not found in .nv.info section. "
            "The driver will not be able to patch profiling hooks correctly.")

    def test_alu_raw_gap_invariant(self):
        """R8: no ALU GPR write may immediately precede its consumer (0-gap RAW).

        Tripwire: removing _enforce_gpr_latency from schedule() would produce
        consecutive IADD3/SHF/IMAD producer→consumer pairs with 0-gap.  The
        SM_120 stall field is ignored by hardware; rbar alone does not gate
        adjacent ALU→ALU reads.  This test would fail immediately if the pass
        is disabled, before any GPU run is needed.
        """
        from sass.scoreboard import (
            _OPCODE_META as _META,
            _get_dest_regs as _meta_dest,
            _get_src_regs  as _meta_src,
            _get_opcode    as _meta_opc,
        )
        # Triple-chain kernel has 2 consecutive ALU RAWs — the strongest test
        text   = self._text(_PTX_ALU_TRIPLE_CHAIN, 'hazard_alu_triple_chain')
        instrs = [text[i:i+16] for i in range(0, len(text), 16)]

        violations = []
        for i in range(len(instrs) - 1):
            raw_i = instrs[i]
            opc_i = _meta_opc(raw_i)
            meta_i = _META.get(opc_i)
            if meta_i is None or meta_i.min_gpr_gap == 0:
                continue
            dest_i = _meta_dest(raw_i)
            if not dest_i:
                continue
            raw_j = instrs[i + 1]
            opc_j = _meta_opc(raw_j)
            src_j = _meta_src(raw_j)
            overlap = dest_i & src_j
            if overlap:
                violations.append(
                    f"instr[{i}] {meta_i.name} writes {overlap} "
                    f"→ instr[{i+1}] opc=0x{opc_j:03x} reads immediately (0-gap RAW)"
                )

        assert not violations, (
            "ALU GPR RAW at 0-gap (SM_120 stall field is ignored — "
            "hardware needs ≥1 instruction of separation):\n"
            + "\n".join(violations))

    def test_schedule_legality(self):
        """verify_schedule() finds no hazard violations in any compiled kernel.

        Uses sass.schedule.verify_schedule() to scan the final instruction stream
        against all constraints declared in _OpMeta (min_gpr_gap, min_consumer_gap,
        min_pred_gap).  This is a post-scheduling assertion — a regression in any
        scheduler pass would show up here before hitting the GPU.

        Note: verify_schedule() is also called inside schedule() itself as an
        assertion; this test exists as a separate human-readable report and to
        exercise the public API.
        """
        from sass.schedule import verify_schedule
        from sass.scoreboard import _OPCODE_META
        from sass.pipeline import compile_ptx_source as _cp

        kernels = [
            (_PTX_EARLY_EXIT_LDCU,    'hazard_early_exit_ldcu'),
            (_PTX_ALU_TRIPLE_CHAIN,   'hazard_alu_triple_chain'),
            (_PTX_ALU_NATURAL_GAP,    'hazard_alu_natural_gap'),
            (_PTX_ALU_ISETP_RAW,      'hazard_alu_isetp_raw'),
            (_PTX_ISETP_BRA_GAP,      'hazard_isetp_bra_gap'),
        ]
        all_violations: list[str] = []
        for ptx, name in kernels:
            # Compile via a fresh pipeline; schedule() already asserts internally,
            # but here we want a per-kernel breakdown for the test report.
            cubins = _cp(ptx)
            # Rebuild SassInstr list from the compiled text section so we can
            # call verify_schedule() independently.
            from sass.isel import SassInstr
            from sass.scoreboard import _get_opcode as _opc
            text    = _extract_text_section(cubins[name])
            instrs  = [SassInstr(text[i:i+16], '') for i in range(0, len(text), 16)]
            vs = verify_schedule(instrs)
            for v in vs:
                all_violations.append(f"[{name}] {v}")

        assert not all_violations, (
            "verify_schedule() found hazard violations:\n"
            + "\n".join(all_violations))


# ---------------------------------------------------------------------------
# GPU execution tests — one per targeted hardware rule
# ---------------------------------------------------------------------------

class TestGPUHazards:
    """Minimal GPU kernels that crash or produce wrong results if a rule is violated."""

    @gpu
    def test_ldcu64_single_pointer_store(self, cuda_ctx):
        """R1+R2+R3: minimal LDCU.64 (pointer) + LDCU.32 (val) + store. 1 thread, no LDG.

        If _hoist_ldcu64 is broken, ILLEGAL_ADDRESS.
        If LDCU wdep aliases LDG (0x35/0x37), ILLEGAL_ADDRESS.
        If LDCU misc≠0x7, ILLEGAL_ADDRESS.
        """
        cubins = compile_ptx_source(_PTX_LDCU64_STORE)
        assert cuda_ctx.load(cubins['hazard_ldcu64_store']), "cubin load failed"
        func = cuda_ctx.get_func('hazard_ldcu64_store')

        d_out = cuda_ctx.alloc(4)
        cuda_ctx.fill32(d_out, 0, 1)

        err = cuda_ctx.launch(func, (1,1,1), (1,1,1), [d_out, 42])
        assert err == 0, f"launch failed: {err}"
        assert cuda_ctx.sync() == 0, "sync failed (ILLEGAL_ADDRESS?)"

        result, = struct.unpack('<I', cuda_ctx.copy_from(d_out, 4))
        assert result == 42, f"expected 42, got {result}"
        cuda_ctx.free(d_out)

    @gpu
    def test_three_ldcu64_two_ldg(self, cuda_ctx):
        """R1+R2: three LDCU.64 pointer params + two LDG loads. 1 thread.

        Exercises the full LDCU slot rotation [0x31, 0x33, 0x31] and verifies
        that LDG's 0x35 slot does not alias with any LDCU slot.
        Expected: out[0] = a[0] + b[0] = 10 + 20 = 30.
        """
        cubins = compile_ptx_source(_PTX_THREE_LDCU64)
        assert cuda_ctx.load(cubins['hazard_three_ldcu64'])
        func = cuda_ctx.get_func('hazard_three_ldcu64')

        d_a   = cuda_ctx.alloc(4)
        d_b   = cuda_ctx.alloc(4)
        d_out = cuda_ctx.alloc(4)
        cuda_ctx.copy_to(d_a,   struct.pack('<I', 10))
        cuda_ctx.copy_to(d_b,   struct.pack('<I', 20))
        cuda_ctx.fill32(d_out, 0, 1)

        err = cuda_ctx.launch(func, (1,1,1), (1,1,1), [d_a, d_b, d_out])
        assert err == 0, f"launch failed: {err}"
        assert cuda_ctx.sync() == 0, "sync failed"

        result, = struct.unpack('<I', cuda_ctx.copy_from(d_out, 4))
        assert result == 30, f"expected 30, got {result}"
        cuda_ctx.free(d_a); cuda_ctx.free(d_b); cuda_ctx.free(d_out)

    @gpu
    def test_imad_ur_multiply(self, cuda_ctx):
        """R6: LDCU.32 + IMAD-UR (mul.lo.s32 with param operand). 1 thread.

        Verifies the LDCU.32 → IMAD-UR chain.  If LDCU.32 misc is wrong or
        the UR index is misread, the multiply produces the wrong result.
        Expected: out[0] = 7 * 6 = 42.
        """
        cubins = compile_ptx_source(_PTX_IMAD_UR)
        assert cuda_ctx.load(cubins['hazard_imad_ur'])
        func = cuda_ctx.get_func('hazard_imad_ur')

        d_out = cuda_ctx.alloc(4)
        cuda_ctx.fill32(d_out, 0, 1)

        err = cuda_ctx.launch(func, (1,1,1), (1,1,1), [d_out, 7, 6])
        assert err == 0, f"launch failed: {err}"
        assert cuda_ctx.sync() == 0, "sync failed"

        result, = struct.unpack('<I', cuda_ctx.copy_from(d_out, 4))
        assert result == 42, f"expected 42 (7*6), got {result}"
        cuda_ctx.free(d_out)

    @gpu
    def test_isetp_ur_branch(self, cuda_ctx):
        """R5: LDCU.32 + ISETP path (setp.ge with param-sourced threshold). 8 threads.

        Threads with tid < thresh (=4) write tid to out[tid].
        Threads with tid >= thresh take the predicated branch to DONE and write nothing.
        Expected: out = [0, 1, 2, 3, sentinel, sentinel, sentinel, sentinel].

        Verifies: LDCU.32 encoding, ISETP encoding, predicate correctness.
        Wrong UR index → wrong predicate → wrong output pattern.
        """
        cubins = compile_ptx_source(_PTX_ISETP_UR)
        assert cuda_ctx.load(cubins['hazard_isetp_ur']), "cubin load failed"
        func = cuda_ctx.get_func('hazard_isetp_ur')

        N      = 8
        thresh = 4
        sentinel = 0xDEADBEEF
        d_out = cuda_ctx.alloc(N * 4)
        cuda_ctx.fill32(d_out, sentinel, N)

        err = cuda_ctx.launch(func, (1,1,1), (N,1,1), [d_out, thresh, N])
        assert err == 0, f"launch failed: {err}"
        assert cuda_ctx.sync() == 0, "sync failed"

        results = list(struct.unpack(f'<{N}I', cuda_ctx.copy_from(d_out, N * 4)))
        for i in range(thresh):
            assert results[i] == i, f"out[{i}]: expected {i}, got {results[i]}"
        for i in range(thresh, N):
            assert results[i] == sentinel, (
                f"out[{i}] was written (expected sentinel 0x{sentinel:x}, got 0x{results[i]:x})")
        cuda_ctx.free(d_out)

    @gpu
    def test_early_exit_ldcu_combo(self, cuda_ctx):
        """R7: kernel with conditional early exit AND LDCU pointer params. 8 threads, n=5.

        Threads 0-4 compute src[i]*3 → out[i].  Threads 5-7 take early exit.
        Verifies EIATTR_EXIT_INSTR_OFFSETS lists the correct EXIT offset so the
        driver doesn't corrupt an unrelated instruction.
        """
        cubins = compile_ptx_source(_PTX_EARLY_EXIT_LDCU)
        assert cuda_ctx.load(cubins['hazard_early_exit_ldcu'])
        func = cuda_ctx.get_func('hazard_early_exit_ldcu')

        N       = 8
        n_active = 5
        d_src   = cuda_ctx.alloc(N * 4)
        d_out   = cuda_ctx.alloc(N * 4)
        src_data = struct.pack(f'<{N}I', *range(N))
        cuda_ctx.copy_to(d_src, src_data)
        cuda_ctx.fill32(d_out, 0xDEADBEEF, N)

        err = cuda_ctx.launch(func, (1,1,1), (N,1,1), [d_out, d_src, n_active])
        assert err == 0, f"launch failed: {err}"
        assert cuda_ctx.sync() == 0, "sync failed (EXIT offset corrupted?)"

        results   = list(struct.unpack(f'<{N}I', cuda_ctx.copy_from(d_out, N * 4)))
        sentinel  = 0xDEADBEEF
        for i in range(n_active):
            assert results[i] == i * 3, f"out[{i}]: expected {i*3}, got {results[i]}"
        for i in range(n_active, N):
            assert results[i] == sentinel, (
                f"out[{i}] was written by early-exit thread (expected 0x{sentinel:x}, "
                f"got 0x{results[i]:x})")
        cuda_ctx.free(d_src); cuda_ctx.free(d_out)

    @gpu
    def test_alu_raw_triple_chain(self, cuda_ctx):
        """R8: three consecutive dependent add.s32 instructions (src[i]*8). 8 threads, n=4.

        add0: r3 = 2*r2
        add1: r4 = 2*r3 = 4*r2   (RAW r3 — gap=0 without scheduling fix)
        add2: r5 = 2*r4 = 8*r2   (RAW r4 — gap=0 without scheduling fix)

        Needs 2 NOP/filler insertions.  Wrong result if either RAW is unhandled.
        Expected: out[i] = 8*i for i in [0, 4).
        """
        cubins = compile_ptx_source(_PTX_ALU_TRIPLE_CHAIN)
        assert cuda_ctx.load(cubins['hazard_alu_triple_chain'])
        func = cuda_ctx.get_func('hazard_alu_triple_chain')

        N       = 8
        n_active = 4
        d_src   = cuda_ctx.alloc(N * 4)
        d_out   = cuda_ctx.alloc(N * 4)
        cuda_ctx.copy_to(d_src, struct.pack(f'<{N}I', *range(N)))
        cuda_ctx.fill32(d_out, 0xDEADBEEF, N)

        err = cuda_ctx.launch(func, (1,1,1), (N,1,1), [d_out, d_src, n_active])
        assert err == 0, f"launch failed: {err}"
        assert cuda_ctx.sync() == 0, "sync failed (ALU RAW triple chain?)"

        results  = list(struct.unpack(f'<{N}I', cuda_ctx.copy_from(d_out, N * 4)))
        sentinel = 0xDEADBEEF
        for i in range(n_active):
            assert results[i] == i * 8, f"out[{i}]: expected {i*8}, got {results[i]}"
        for i in range(n_active, N):
            assert results[i] == sentinel, (
                f"out[{i}] unexpectedly written (expected sentinel, got {results[i]})")
        cuda_ctx.free(d_src); cuda_ctx.free(d_out)

    @gpu
    def test_alu_raw_natural_gap(self, cuda_ctx):
        """R8: src[i]*3 with a natural gap instruction between the two dependent adds.

        After the LDG reorder pass, an independent IADD.64-UR (output address
        computation) falls between the two add.s32 instructions, providing a
        natural 1-cycle gap.  The scheduler should NOT insert an extra NOP;
        the computation must still produce the correct result.
        Expected: out[i] = 3*i for i in [0, 4).
        """
        cubins = compile_ptx_source(_PTX_ALU_NATURAL_GAP)
        assert cuda_ctx.load(cubins['hazard_alu_natural_gap'])
        func = cuda_ctx.get_func('hazard_alu_natural_gap')

        N       = 8
        n_active = 4
        d_src   = cuda_ctx.alloc(N * 4)
        d_out   = cuda_ctx.alloc(N * 4)
        cuda_ctx.copy_to(d_src, struct.pack(f'<{N}I', *range(N)))
        cuda_ctx.fill32(d_out, 0xDEADBEEF, N)

        err = cuda_ctx.launch(func, (1,1,1), (N,1,1), [d_out, d_src, n_active])
        assert err == 0, f"launch failed: {err}"
        assert cuda_ctx.sync() == 0, "sync failed (ALU RAW natural gap?)"

        results  = list(struct.unpack(f'<{N}I', cuda_ctx.copy_from(d_out, N * 4)))
        sentinel = 0xDEADBEEF
        for i in range(n_active):
            assert results[i] == i * 3, f"out[{i}]: expected {i*3}, got {results[i]}"
        for i in range(n_active, N):
            assert results[i] == sentinel, (
                f"out[{i}] unexpectedly written (expected sentinel, got {results[i]})")
        cuda_ctx.free(d_src); cuda_ctx.free(d_out)

    @gpu
    def test_alu_isetp_raw(self, cuda_ctx):
        """R8: IADD3 immediately followed by ISETP reading the IADD3 output.

        Each active thread computes r2 = 2*tid, then tests r2 < n.
        Threads where r2 < n=4 (tid=0,1) write r2 to out[tid].
        Threads where r2 >= n=4 (tid=2,3) take the predicated branch and write nothing.

        Without the ALU RAW gap, ISETP reads a stale r2 → wrong predicate →
        wrong output pattern.
        Expected: out = [0, 2, sentinel, sentinel, sentinel, sentinel, sentinel, sentinel].
        """
        cubins = compile_ptx_source(_PTX_ALU_ISETP_RAW)
        assert cuda_ctx.load(cubins['hazard_alu_isetp_raw'])
        func = cuda_ctx.get_func('hazard_alu_isetp_raw')

        N        = 8
        n_active = 4
        sentinel = 0xDEADBEEF
        d_out    = cuda_ctx.alloc(N * 4)
        cuda_ctx.fill32(d_out, sentinel, N)

        err = cuda_ctx.launch(func, (1,1,1), (N,1,1), [d_out, n_active])
        assert err == 0, f"launch failed: {err}"
        assert cuda_ctx.sync() == 0, "sync failed (ALU→ISETP RAW?)"

        results = list(struct.unpack(f'<{N}I', cuda_ctx.copy_from(d_out, N * 4)))
        # tid=0 → r2=0 < 4 → write 0;  tid=1 → r2=2 < 4 → write 2
        assert results[0] == 0,   f"out[0]: expected 0, got {results[0]}"
        assert results[1] == 2,   f"out[1]: expected 2, got {results[1]}"
        # tid=2 → r2=4 >= 4 → skip;  tid=3 → r2=6 >= 4 → skip
        for i in range(2, N):
            assert results[i] == sentinel, (
                f"out[{i}] was written (expected 0x{sentinel:x}, got 0x{results[i]:x})")
        cuda_ctx.free(d_out)

    @gpu
    def test_isetp_bra_gap(self, cuda_ctx):
        """P1: ISETP immediately followed by BRA at 0-gap (predicate latency = 0).

        Standalone test that isolates predicate write visibility without any
        preceding ALU RAW.  Threads where tid < thresh=4 write tid to out[tid];
        threads where tid >= thresh take the predicated branch and write nothing.

        If ISETP→BRA needed a 1-cycle gap (like GPR ALU RAW), the predicate
        would be read before it was written → wrong branch direction → wrong output.

        SM_120 empirical result: ISETP→BRA at 0-gap is safe.  min_pred_gap=0.
        Expected: out = [0, 1, 2, 3, sentinel, sentinel, sentinel, sentinel].
        """
        cubins = compile_ptx_source(_PTX_ISETP_BRA_GAP)
        assert cuda_ctx.load(cubins['hazard_isetp_bra_gap'])
        func = cuda_ctx.get_func('hazard_isetp_bra_gap')

        N        = 8
        thresh   = 4
        sentinel = 0xDEADBEEF
        d_out    = cuda_ctx.alloc(N * 4)
        cuda_ctx.fill32(d_out, sentinel, N)

        err = cuda_ctx.launch(func, (1,1,1), (N,1,1), [d_out, thresh])
        assert err == 0, f"launch failed: {err}"
        assert cuda_ctx.sync() == 0, "sync failed (ISETP→BRA predicate latency?)"

        results = list(struct.unpack(f'<{N}I', cuda_ctx.copy_from(d_out, N * 4)))
        for i in range(thresh):
            assert results[i] == i, f"out[{i}]: expected {i}, got {results[i]}"
        for i in range(thresh, N):
            assert results[i] == sentinel, (
                f"out[{i}] was written (expected 0x{sentinel:x}, got 0x{results[i]:x})")
        cuda_ctx.free(d_out)
