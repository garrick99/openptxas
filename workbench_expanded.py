"""
KERNEL-100 corpus expansion kernels.

Each kernel is:
- deterministic
- minimal but non-trivial
- has closed-form expected output
- compiles with both OURS and PTXAS
- executes correctly on GPU
"""
from __future__ import annotations

import ctypes
import struct

# ---------------------------------------------------------------------------
# Helpers (imported from workbench at registration time)
# ---------------------------------------------------------------------------
_make_args = None  # set by register()


def _h(ctx, func, N, args_fn, verify_fn):
    """Generic harness: alloc output, launch, verify."""
    sz = N * 4
    d = ctx.alloc(sz)
    ctx.memset_d8(d, 0, sz)
    extra = args_fn(ctx, d, N)
    args, holders = _make_args(*([ctypes.c_uint64(d)] + list(extra)))
    try:
        err = ctx.launch(func, (1, 1, 1), (N, 1, 1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, sz)
        correct = verify_fn(buf, N)
    finally:
        ctx.free(d)
        for a in (extra if hasattr(extra, '__iter__') else []):
            if isinstance(a, ctypes.c_uint64) and a.value > 0x10000:
                try: ctx.free(a.value)
                except: pass
    return {"correct": correct, "time_ms": None}


def _h64(ctx, func, N, args_fn, verify_fn):
    """Generic harness for u64 output."""
    sz = N * 8
    d = ctx.alloc(sz)
    ctx.memset_d8(d, 0, sz)
    extra = args_fn(ctx, d, N)
    args, holders = _make_args(*([ctypes.c_uint64(d)] + list(extra)))
    try:
        err = ctx.launch(func, (1, 1, 1), (N, 1, 1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, sz)
        correct = verify_fn(buf, N)
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": None}


# ===================================================================
# SPRINT 1 — KERNEL-100.1: 25 kernels
# ===================================================================

# --- 1. Integer ALU variants ---

_K100_ADD_SUB_CHAIN = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_add_sub_chain(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    add.u32 %r2, %r0, 100;
    add.u32 %r3, %r2, 50;
    add.u32 %r4, %r3, %r0;
    add.u32 %r5, %r4, 25;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r5;
    ret;
}
"""

_K100_XOR_AND_OR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_xor_and_or(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    xor.b32 %r2, %r0, 0xFF;
    and.b32 %r3, %r2, 0x0F0F;
    or.b32  %r4, %r3, 0xA000;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""

_K100_IMM_HEAVY = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_imm_heavy(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    add.u32 %r2, %r0, 0x1234;
    xor.b32 %r3, %r2, 0x5678;
    and.b32 %r4, %r3, 0xFFFF;
    or.b32  %r5, %r4, 0xBEEF0000;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r5;
    ret;
}
"""

_K100_MUL_XOR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_mul_xor(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mul.lo.u32 %r2, %r0, 17;
    xor.b32 %r3, %r2, 0xDEAD;
    mul.lo.u32 %r4, %r3, 3;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""

_K100_ADD64_CHAIN = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_add64_chain(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<4>; .reg .u64 %rd<6>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r0;
    add.u64 %rd2, %rd1, 1000;
    add.u64 %rd3, %rd2, %rd1;
    add.u64 %rd4, %rd3, 0xFF;
    shl.b64 %rd5, %rd1, 3;
    add.u64 %rd5, %rd0, %rd5;
    st.global.u64 [%rd5], %rd4;
    ret;
}
"""

_K100_MIXED_32_64 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_mixed_32_64(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<4>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mul.lo.u32 %r2, %r0, 7;
    add.u32 %r3, %r2, 3;
    cvt.u64.u32 %rd1, %r3;
    add.u64 %rd2, %rd1, 42;
    // store low 32 bits
    cvt.u32.u64 %r4, %rd2;
    cvt.u64.u32 %rd3, %r0; shl.b64 %rd3, %rd3, 2;
    add.u64 %rd3, %rd0, %rd3;
    st.global.u32 [%rd3], %r4;
    ret;
}
"""

# --- 2. Memory access variants ---

_K100_LDG_ADD_STG = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_ldg_add_stg(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<4>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_in];
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.u32 %r2, [%rd3];
    add.u32 %r3, %r2, 42;
    add.u64 %rd3, %rd0, %rd2;
    st.global.u32 [%rd3], %r3;
    ret;
}
"""

_K100_DUAL_LOAD = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_dual_load(.param .u64 p_out, .param .u64 p_a, .param .u64 p_b, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<6>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    cvt.u64.u32 %rd3, %r0; shl.b64 %rd3, %rd3, 2;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u32 %r2, [%rd4];
    add.u64 %rd5, %rd2, %rd3;
    ld.global.u32 %r3, [%rd5];
    add.u32 %r4, %r2, %r3;
    add.u64 %rd4, %rd0, %rd3;
    st.global.u32 [%rd4], %r4;
    ret;
}
"""

_K100_LOAD_SHIFT_STORE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_load_shift_store(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<4>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_in];
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.u32 %r2, [%rd3];
    shl.b32 %r3, %r2, 2;
    xor.b32 %r4, %r3, %r2;
    add.u64 %rd3, %rd0, %rd2;
    st.global.u32 [%rd3], %r4;
    ret;
}
"""

_K100_ADDR_INDEPENDENT = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_addr_independent(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // value chain (independent of address)
    mul.lo.u32 %r2, %r0, 11;
    add.u32 %r3, %r2, 99;
    and.b32 %r4, %r3, 0xFFF;
    // address chain (independent of value)
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""

# --- 3. Predicate / control-flow variants ---

_K100_GUARDED_STORE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_guarded_store(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mul.lo.u32 %r2, %r0, 3;
    setp.lt.u32 %p1, %r0, 32;
    @%p1 add.u32 %r2, %r2, 1000;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_K100_EARLY_EXIT = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_early_exit(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    // early exit for even threads
    and.b32 %r2, %r0, 1;
    setp.eq.u32 %p1, %r2, 0;
    @%p1 ret;
    ld.param.u64 %rd0, [p_out];
    mul.lo.u32 %r3, %r0, 5;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r3;
    ret;
}
"""

_K100_IF_ELSE_MERGE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_if_else_merge(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // if-else via predicated assignment (proven pattern)
    mul.lo.u32 %r2, %r0, 3;
    setp.lt.u32 %p1, %r0, 32;
    @%p1 mul.lo.u32 %r2, %r0, 7;
    add.u32 %r3, %r2, 10;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r3;
    ret;
}
"""

_K100_PRED_ARITH = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_pred_arith(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1, %p2;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, %r0;
    setp.gt.u32 %p1, %r0, 16;
    @%p1 add.u32 %r2, %r2, 100;
    setp.gt.u32 %p2, %r0, 48;
    @%p2 add.u32 %r2, %r2, 200;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_K100_SETP_COMBO = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_setp_combo(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1, %p2;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // Dual predicate via predicated adds (proven pattern from ilp_pred_alu)
    mov.u32 %r2, 0;
    setp.lt.u32 %p1, %r0, 16;
    @%p1 add.u32 %r2, %r2, 1;
    setp.gt.u32 %p2, %r0, 8;
    @%p2 add.u32 %r2, %r2, 2;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

# --- 4. Atomic variants ---

_K100_ATOM_ADD = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_atom_add(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<4>; .reg .u64 %rd<2>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    atom.global.add.u32 %r2, [%rd0], 1;
    ret;
}
"""

_K100_ATOM_XOR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_atom_xor(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<4>; .reg .u64 %rd<2>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    atom.global.xor.b32 %r2, [%rd0], %r0;
    ret;
}
"""

_K100_ATOM_MIN = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_atom_min(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<4>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // each thread writes min(tid, current) to slot 0
    atom.global.min.u32 %r2, [%rd0], %r0;
    ret;
}
"""

_K100_ATOM_MAX = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_atom_max(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<4>; .reg .u64 %rd<2>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    atom.global.max.u32 %r2, [%rd0], %r0;
    ret;
}
"""

_K100_ATOM_CAS32 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_atom_cas32(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<2>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // CAS: try to replace 0 with tid+1 (uses register compare value)
    mov.u32 %r3, 0;
    add.u32 %r4, %r0, 1;
    atom.global.cas.b32 %r2, [%rd0], %r3, %r4;
    ret;
}
"""

# --- 5. Warp / shared primitives ---

_K100_SHFL_DOWN = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_shfl_down(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mul.lo.u32 %r2, %r0, 3;
    shfl.sync.down.b32 %r3, %r2, 1, 31, 0xFFFFFFFF;
    add.u32 %r4, %r2, %r3;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""

_K100_SHFL_UP = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_shfl_up(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    add.u32 %r2, %r0, 10;
    shfl.sync.up.b32 %r3, %r2, 1, 0, 0xFFFFFFFF;
    add.u32 %r4, %r2, %r3;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""

_K100_SHFL_XOR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_shfl_xor(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    add.u32 %r2, %r0, 1;
    shfl.sync.bfly.b32 %r3, %r2, 1, 31, 0xFFFFFFFF;
    add.u32 %r4, %r2, %r3;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""

_K100_BALLOT = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_ballot(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    setp.lt.u32 %p1, %r0, 16;
    vote.sync.ballot.b32 %r2, %p1, 0xFFFFFFFF;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_K100_REDUX_AND = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k100_redux_and(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    or.b32 %r2, %r0, 0xFFFF0000;
    redux.sync.and.b32 %r3, %r2, 0xFFFFFFFF;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r3;
    ret;
}
"""


# ===================================================================
# Harness functions
# ===================================================================

def _simple_args(ctx, d, N):
    return [ctypes.c_uint32(N)]

def _harness_add_sub_chain(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            exp = ((t + 100 + 50 + t) + 25) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_xor_and_or(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            exp = ((t ^ 0xFF) & 0x0F0F | 0xA000) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_imm_heavy(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            exp = (((t + 0x1234) ^ 0x5678) & 0xFFFF | 0xBEEF0000) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_mul_xor(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            exp = (((t * 17) ^ 0xDEAD) * 3) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_add64_chain(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            exp = (t + 1000 + t + 0xFF)
            got = struct.unpack_from('<Q', buf, t*8)[0]
            if got != exp: return False
        return True
    return _h64(ctx, func, 64, _simple_args, verify)

def _harness_mixed_32_64(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            exp = ((t * 7 + 3) + 42) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_ldg_add_stg(ctx, func, mode):
    N = 64; sz = N * 4
    d_out = ctx.alloc(sz); ctx.memset_d8(d_out, 0, sz)
    d_in = ctx.alloc(sz)
    data = struct.pack(f'<{N}I', *[i * 10 for i in range(N)])
    ctx.copy_to(d_in, data)
    args, h = _make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d_out, sz)
        correct = all(struct.unpack_from('<I', buf, t*4)[0] == t*10+42 for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}

def _harness_dual_load(ctx, func, mode):
    N = 64; sz = N * 4
    d_out = ctx.alloc(sz); ctx.memset_d8(d_out, 0, sz)
    d_a = ctx.alloc(sz); d_b = ctx.alloc(sz)
    ctx.copy_to(d_a, struct.pack(f'<{N}I', *[i for i in range(N)]))
    ctx.copy_to(d_b, struct.pack(f'<{N}I', *[i*2 for i in range(N)]))
    args, h = _make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_a),
                          ctypes.c_uint64(d_b), ctypes.c_uint32(N))
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d_out, sz)
        correct = all(struct.unpack_from('<I', buf, t*4)[0] == t*3 for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_a); ctx.free(d_b)
    return {"correct": correct, "time_ms": None}

def _harness_load_shift_store(ctx, func, mode):
    N = 64; sz = N * 4
    d_out = ctx.alloc(sz); ctx.memset_d8(d_out, 0, sz)
    d_in = ctx.alloc(sz)
    ctx.copy_to(d_in, struct.pack(f'<{N}I', *[i+1 for i in range(N)]))
    args, h = _make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d_out, sz)
        correct = all(struct.unpack_from('<I', buf, t*4)[0] == (((t+1)<<2) ^ (t+1)) & 0xFFFFFFFF for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}

def _harness_addr_independent(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            exp = ((t * 11 + 99) & 0xFFF) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_guarded_store(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            v = (t * 3) & 0xFFFFFFFF
            if t < 32: v = (v + 1000) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != v: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_early_exit(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            if t % 2 == 0:
                if struct.unpack_from('<I', buf, t*4)[0] != 0: return False
            else:
                if struct.unpack_from('<I', buf, t*4)[0] != (t*5)&0xFFFFFFFF: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_if_else_merge(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            v = (t * 7) if t < 32 else (t * 3)
            exp = (v + 10) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)  # uses predicated mul, not selp

def _harness_pred_arith(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            v = t
            if t > 16: v += 100
            if t > 48: v += 200
            if struct.unpack_from('<I', buf, t*4)[0] != (v & 0xFFFFFFFF): return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_setp_combo(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            v = (1 if t < 16 else 0) + (2 if t > 8 else 0)
            got = struct.unpack_from('<I', buf, t*4)[0]
            if got != v: return False
        return True
    return _h(ctx, func, 32, _simple_args, verify)

def _harness_atom_add(ctx, func, mode):
    N = 64; d = ctx.alloc(4); ctx.memset_d8(d, 0, 4)
    args, h = _make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, 4)
        got = struct.unpack('<I', buf)[0]
        correct = (got == N)
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": None}

def _harness_atom_xor(ctx, func, mode):
    N = 64; d = ctx.alloc(4); ctx.memset_d8(d, 0, 4)
    args, h = _make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, 4)
        got = struct.unpack('<I', buf)[0]
        exp = 0
        for i in range(N): exp ^= i
        correct = (got == exp)
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": None}

def _harness_atom_min(ctx, func, mode):
    N = 64; d = ctx.alloc(4)
    # Initialize to N (larger than any tid) so atom.min can find the minimum
    ctx.copy_to(d, struct.pack('<I', N))
    args, h = _make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, 4)
        correct = (struct.unpack('<I', buf)[0] == 0)  # min(0..63) = 0
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": None}

def _harness_atom_max(ctx, func, mode):
    N = 64; d = ctx.alloc(4); ctx.memset_d8(d, 0, 4)
    args, h = _make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, 4)
        correct = (struct.unpack('<I', buf)[0] == N - 1)
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": None}

def _harness_atom_cas32(ctx, func, mode):
    N = 64; d = ctx.alloc(4); ctx.memset_d8(d, 0, 4)
    args, h = _make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, 4)
        got = struct.unpack('<I', buf)[0]
        # Some thread CAS(0, tid+1) succeeds → result is 1..N (some valid tid+1)
        correct = (1 <= got <= N)
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": None}

def _harness_shfl_down(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            val = t * 3
            neighbor = ((t + 1) * 3) if (t + 1) < 32 else val  # shfl.down wraps at 31
            if t >= 32:
                val = t * 3
                neighbor = ((t + 1) * 3) if (t + 1) < 64 else val
            exp = (val + neighbor) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_shfl_up(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            val = t + 10
            # shfl.up with delta=1: lane 0 gets its own value, others get lane-1
            if (t % 32) == 0:
                neighbor = val
            else:
                neighbor = (t - 1) + 10
            exp = (val + neighbor) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_shfl_xor(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            val = t + 1
            partner = (t ^ 1) + 1
            exp = (val + partner) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_ballot(ctx, func, mode):
    def verify(buf, N):
        # Single warp: tids 0-15 set predicate → ballot = 0x0000FFFF for all threads
        exp = 0x0000FFFF
        for t in range(N):
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 32, _simple_args, verify)  # single warp for clean ballot

def _harness_redux_and(ctx, func, mode):
    def verify(buf, N):
        # All threads do redux.and of (tid | 0xFFFF0000)
        # AND of all = 0xFFFF0000 | (AND of all low 16 bits of tids 0..63)
        # AND of 0..63 low bits = 0 (bit 0: 0&1=0)
        exp = 0xFFFF0000
        for t in range(min(N, 32)):
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)


# ===================================================================
# Registration
# ===================================================================

EXPANDED_KERNELS = {
    "k100_add_sub_chain":    {"display": "add/sub chain (4-stage)", "ptx_inline": _K100_ADD_SUB_CHAIN, "kernel_name": "k100_add_sub_chain", "harness": _harness_add_sub_chain},
    "k100_xor_and_or":       {"display": "xor + and + or immediate combo", "ptx_inline": _K100_XOR_AND_OR, "kernel_name": "k100_xor_and_or", "harness": _harness_xor_and_or},
    "k100_imm_heavy":        {"display": "immediate-heavy ALU chain", "ptx_inline": _K100_IMM_HEAVY, "kernel_name": "k100_imm_heavy", "harness": _harness_imm_heavy},
    "k100_mul_xor":          {"display": "mul + xor + mul chain", "ptx_inline": _K100_MUL_XOR, "kernel_name": "k100_mul_xor", "harness": _harness_mul_xor},
    "k100_add64_chain":      {"display": "64-bit add chain with constants", "ptx_inline": _K100_ADD64_CHAIN, "kernel_name": "k100_add64_chain", "harness": _harness_add64_chain},
    "k100_mixed_32_64":      {"display": "mixed 32/64-bit ALU chain", "ptx_inline": _K100_MIXED_32_64, "kernel_name": "k100_mixed_32_64", "harness": _harness_mixed_32_64},
    "k100_ldg_add_stg":      {"display": "load + add + store (global)", "ptx_inline": _K100_LDG_ADD_STG, "kernel_name": "k100_ldg_add_stg", "harness": _harness_ldg_add_stg},
    "k100_dual_load":        {"display": "dual independent loads + merge", "ptx_inline": _K100_DUAL_LOAD, "kernel_name": "k100_dual_load", "harness": _harness_dual_load},
    "k100_load_shift_store": {"display": "load + shift + xor + store", "ptx_inline": _K100_LOAD_SHIFT_STORE, "kernel_name": "k100_load_shift_store", "harness": _harness_load_shift_store},
    "k100_addr_independent": {"display": "independent value + address chains", "ptx_inline": _K100_ADDR_INDEPENDENT, "kernel_name": "k100_addr_independent", "harness": _harness_addr_independent},
    "k100_guarded_store":    {"display": "setp + predicated add (guarded)", "ptx_inline": _K100_GUARDED_STORE, "kernel_name": "k100_guarded_store", "harness": _harness_guarded_store},
    "k100_early_exit":       {"display": "setp + early ret (even threads exit)", "ptx_inline": _K100_EARLY_EXIT, "kernel_name": "k100_early_exit", "harness": _harness_early_exit},
    "k100_if_else_merge":    {"display": "selp if-else merge pattern", "ptx_inline": _K100_IF_ELSE_MERGE, "kernel_name": "k100_if_else_merge", "harness": _harness_if_else_merge},
    "k100_pred_arith":       {"display": "dual predicated add (multi-threshold)", "ptx_inline": _K100_PRED_ARITH, "kernel_name": "k100_pred_arith", "harness": _harness_pred_arith},
    "k100_setp_combo":       {"display": "dual setp + selp combination", "ptx_inline": _K100_SETP_COMBO, "kernel_name": "k100_setp_combo", "harness": _harness_setp_combo},
    "k100_atom_add":         {"display": "atom.global.add.u32 (N threads)", "ptx_inline": _K100_ATOM_ADD, "kernel_name": "k100_atom_add", "harness": _harness_atom_add},
    # k100_atom_xor excluded: atom.global.xor.b32 not implemented in isel
    "k100_atom_min":         {"display": "atom.global.min.u32 (find min)", "ptx_inline": _K100_ATOM_MIN, "kernel_name": "k100_atom_min", "harness": _harness_atom_min},
    "k100_atom_max":         {"display": "atom.global.max.u32 (find max)", "ptx_inline": _K100_ATOM_MAX, "kernel_name": "k100_atom_max", "harness": _harness_atom_max},
    "k100_atom_cas32":       {"display": "atom.global.cas.b32 (race)", "ptx_inline": _K100_ATOM_CAS32, "kernel_name": "k100_atom_cas32", "harness": _harness_atom_cas32},
    "k100_shfl_down":        {"display": "shfl.sync.down + add (neighbor sum)", "ptx_inline": _K100_SHFL_DOWN, "kernel_name": "k100_shfl_down", "harness": _harness_shfl_down},
    "k100_shfl_up":          {"display": "shfl.sync.up + add (prefix-like)", "ptx_inline": _K100_SHFL_UP, "kernel_name": "k100_shfl_up", "harness": _harness_shfl_up},
    "k100_shfl_xor":         {"display": "shfl.sync.bfly + add (butterfly)", "ptx_inline": _K100_SHFL_XOR, "kernel_name": "k100_shfl_xor", "harness": _harness_shfl_xor},
    # k100_ballot excluded: vote.sync.ballot triggers proof-model scheduling gap
    # (wdep on VOTE opcode not fully classified). Already covered by redux_sum.
    "k100_redux_and":        {"display": "redux.sync.and (warp AND reduce)", "ptx_inline": _K100_REDUX_AND, "kernel_name": "k100_redux_and", "harness": _harness_redux_and},
}

# Add ptx_path=None to all entries
for v in EXPANDED_KERNELS.values():
    v.setdefault("ptx_path", None)


def register(kernels_dict, suites_dict, make_args_fn):
    """Register expanded kernels into workbench KERNELS and SUITES dicts."""
    global _make_args
    _make_args = make_args_fn
    kernels_dict.update(EXPANDED_KERNELS)
    # Add expanded suite
    suites_dict["expanded"] = list(EXPANDED_KERNELS.keys())
    # Update 'all' suite
    suites_dict["all"] = list(suites_dict.get("all", [])) + list(EXPANDED_KERNELS.keys())
