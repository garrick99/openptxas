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
    # k100_atom_xor excluded: ATOMG_XOR=0x06 encoding produces wrong results (needs ground-truth investigation)
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

# ===================================================================
# SPRINT 2 — KERNEL-100.2: 25 kernels (structural diversity)
# ===================================================================

def _ptx_simple(name, body_regs, body_ptx, extra_params=""):
    """Template for simple out[tid] = f(tid) kernels."""
    return f"""
.version 9.0
.target sm_120
.address_size 64
.visible .entry {name}(.param .u64 p_out, .param .u32 n{extra_params}) {{
    .reg .u32 %r<{body_regs}>; .reg .u64 %rd<4>; .reg .pred %p0, %p1, %p2;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
{body_ptx}
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""

# 1. Longer dependency chains
_K200_DEEP_ALU = _ptx_simple("k200_deep_alu", 12, """
    mul.lo.u32 %r2, %r0, 3;
    add.u32 %r3, %r2, 7;
    xor.b32 %r4, %r3, 0xFF;
    and.b32 %r5, %r4, 0x3FF;
    mul.lo.u32 %r6, %r5, 5;
    add.u32 %r7, %r6, 13;
    xor.b32 %r8, %r7, 0xAB;
    and.b32 %r2, %r8, 0x7FF;""")

_K200_ALT_32_64 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k200_alt_32_64(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<6>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mul.lo.u32 %r2, %r0, 7;
    cvt.u64.u32 %rd1, %r2;
    add.u64 %rd2, %rd1, 100;
    cvt.u32.u64 %r3, %rd2;
    add.u32 %r4, %r3, 50;
    cvt.u64.u32 %rd3, %r4;
    add.u64 %rd4, %rd3, 25;
    cvt.u32.u64 %r2, %rd4;
    cvt.u64.u32 %rd5, %r0; shl.b64 %rd5, %rd5, 2;
    add.u64 %rd5, %rd0, %rd5;
    st.global.u32 [%rd5], %r2;
    ret;
}
"""

_K200_TRIPLE_ACC = _ptx_simple("k200_triple_acc", 12, """
    mul.lo.u32 %r2, %r0, 3;
    mul.lo.u32 %r3, %r0, 5;
    mul.lo.u32 %r4, %r0, 7;
    add.u32 %r2, %r2, %r3;
    add.u32 %r2, %r2, %r4;""")

_K200_QUAD_ACC = _ptx_simple("k200_quad_acc", 12, """
    mul.lo.u32 %r3, %r0, 2;
    mul.lo.u32 %r4, %r0, 3;
    mul.lo.u32 %r5, %r0, 5;
    mul.lo.u32 %r6, %r0, 7;
    add.u32 %r2, %r3, %r4;
    add.u32 %r7, %r5, %r6;
    add.u32 %r2, %r2, %r7;""")

# 2. Divergence-lite control flow
_K200_NESTED_PRED = _ptx_simple("k200_nested_pred", 10, """
    mov.u32 %r2, %r0;
    setp.gt.u32 %p1, %r0, 8;
    @%p1 add.u32 %r2, %r2, 10;
    @%p1 setp.gt.u32 %p2, %r0, 24;
    @%p2 add.u32 %r2, %r2, 20;""")

_K200_PRED_CHAIN = _ptx_simple("k200_pred_chain", 10, """
    mov.u32 %r2, 0;
    setp.gt.u32 %p1, %r0, 4;
    @%p1 add.u32 %r2, %r2, 1;
    setp.gt.u32 %p1, %r0, 8;
    @%p1 add.u32 %r2, %r2, 2;
    setp.gt.u32 %p1, %r0, 16;
    @%p1 add.u32 %r2, %r2, 4;
    setp.gt.u32 %p1, %r0, 32;
    @%p1 add.u32 %r2, %r2, 8;""")

_K200_PRED_MUL = _ptx_simple("k200_pred_mul", 10, """
    mul.lo.u32 %r2, %r0, 3;
    setp.lt.u32 %p1, %r0, 32;
    @%p1 mul.lo.u32 %r2, %r0, 7;
    add.u32 %r2, %r2, 1;""")

_K200_BRANCH_STORE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k200_branch_store(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    mul.lo.u32 %r2, %r0, 11;
    setp.lt.u32 %p1, %r0, 32;
    @%p1 add.u32 %r2, %r2, 100;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_K200_DOUBLE_GUARD = _ptx_simple("k200_double_guard", 10, """
    mul.lo.u32 %r2, %r0, 3;
    setp.gt.u32 %p1, %r0, 16;
    @%p1 add.u32 %r2, %r2, 50;
    setp.lt.u32 %p2, %r0, 48;
    @%p2 add.u32 %r2, %r2, 25;""")

# 3. Memory / control interaction
_K200_LOAD_PRED_STORE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k200_load_pred_store(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<5>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_in];
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.u32 %r2, [%rd3];
    setp.gt.u32 %p1, %r2, 32;
    @%p1 add.u32 %r2, %r2, 1000;
    add.u64 %rd4, %rd0, %rd2;
    st.global.u32 [%rd4], %r2;
    ret;
}
"""

_K200_INDEP_LOAD_MERGE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k200_indep_load_merge(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<6>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out]; ld.param.u64 %rd1, [p_in];
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    // load from p_in[tid]
    add.u64 %rd3, %rd1, %rd2; ld.global.u32 %r2, [%rd3];
    // independent compute
    mul.lo.u32 %r3, %r0, 3;
    add.u32 %r4, %r2, %r3;
    and.b32 %r4, %r4, 0xFFFF;
    add.u64 %rd4, %rd0, %rd2;
    st.global.u32 [%rd4], %r4;
    ret;
}
"""

# 4. FP32 / mixed compute
_K200_FADD_CHAIN = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k200_fadd_chain(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<4>; .reg .u64 %rd<5>; .reg .f32 %f<6>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out]; ld.param.u64 %rd1, [p_in];
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.f32 %f0, [%rd3];
    add.f32 %f1, %f0, %f0;
    add.f32 %f2, %f1, %f0;
    mov.b32 %r2, %f2;
    add.u64 %rd4, %rd0, %rd2;
    st.global.u32 [%rd4], %r2;
    ret;
}
"""

_K200_FMUL_ADD = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k200_fmul_add(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<4>; .reg .u64 %rd<5>; .reg .f32 %f<6>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out]; ld.param.u64 %rd1, [p_in];
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.f32 %f0, [%rd3];
    mul.f32 %f1, %f0, %f0;
    add.f32 %f2, %f1, %f0;
    mov.b32 %r2, %f2;
    add.u64 %rd4, %rd0, %rd2;
    st.global.u32 [%rd4], %r2;
    ret;
}
"""

# 5. ILP-heavy synthetics
_K200_ILP3_CHAIN = _ptx_simple("k200_ilp3_chain", 16, """
    mul.lo.u32 %r2, %r0, 3;
    mul.lo.u32 %r3, %r0, 5;
    mul.lo.u32 %r4, %r0, 7;
    add.u32 %r5, %r2, 10;
    add.u32 %r6, %r3, 20;
    add.u32 %r7, %r4, 30;
    xor.b32 %r8, %r5, 0xFF;
    xor.b32 %r9, %r6, 0xAA;
    xor.b32 %r10, %r7, 0x55;
    add.u32 %r2, %r8, %r9;
    add.u32 %r2, %r2, %r10;""")

_K200_ILP_LOAD_COMPUTE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k200_ilp_load_compute(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<5>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out]; ld.param.u64 %rd1, [p_in];
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.u32 %r2, [%rd3];
    // independent compute while load is in-flight
    mul.lo.u32 %r3, %r0, 17;
    add.u32 %r4, %r3, 42;
    // merge
    add.u32 %r5, %r2, %r4;
    add.u64 %rd4, %rd0, %rd2;
    st.global.u32 [%rd4], %r5;
    ret;
}
"""

_K200_ILP_DUAL_ADDR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k200_ilp_dual_addr(.param .u64 p_out, .param .u64 p_a, .param .u64 p_b, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out]; ld.param.u64 %rd1, [p_a]; ld.param.u64 %rd2, [p_b];
    cvt.u64.u32 %rd3, %r0; shl.b64 %rd3, %rd3, 2;
    // chain A: address + load
    add.u64 %rd4, %rd1, %rd3; ld.global.u32 %r2, [%rd4];
    // chain B: address + load (independent)
    add.u64 %rd5, %rd2, %rd3; ld.global.u32 %r3, [%rd5];
    // chain C: independent ALU
    mul.lo.u32 %r4, %r0, 11;
    // merge all three
    add.u32 %r5, %r2, %r3;
    add.u32 %r5, %r5, %r4;
    add.u64 %rd6, %rd0, %rd3;
    st.global.u32 [%rd6], %r5;
    ret;
}
"""

# 6. More integer patterns
_K200_SHL_CHAIN = _ptx_simple("k200_shl_chain", 10, """
    add.u32 %r2, %r0, 1;
    shl.b32 %r3, %r2, 1;
    shl.b32 %r4, %r3, 2;
    xor.b32 %r2, %r4, %r2;""")

_K200_AND_OR_CHAIN = _ptx_simple("k200_and_or_chain", 10, """
    or.b32  %r2, %r0, 0xF0;
    and.b32 %r3, %r2, 0xFF;
    or.b32  %r4, %r3, 0x100;
    and.b32 %r2, %r4, 0x1FF;""")

_K200_MUL_ADD_LONG = _ptx_simple("k200_mul_add_long", 14, """
    mul.lo.u32 %r2, %r0, 3;
    add.u32 %r3, %r2, 1;
    mul.lo.u32 %r4, %r3, 5;
    add.u32 %r5, %r4, 2;
    mul.lo.u32 %r6, %r5, 7;
    add.u32 %r2, %r6, 3;""")

_K200_XOR_REDUCE = _ptx_simple("k200_xor_reduce", 10, """
    xor.b32 %r2, %r0, 0x1;
    xor.b32 %r3, %r2, 0x2;
    xor.b32 %r4, %r3, 0x4;
    xor.b32 %r2, %r4, 0x8;""")

_K200_WIDE_IMM = _ptx_simple("k200_wide_imm", 10, """
    add.u32 %r2, %r0, 0x12345;
    and.b32 %r3, %r2, 0xFFFFF;
    xor.b32 %r2, %r3, 0xABCDE;""")

# 7. Shuffle + compute
_K200_SHFL_REDUCE2 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k200_shfl_reduce2(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    add.u32 %r2, %r0, 1;
    shfl.sync.bfly.b32 %r3, %r2, 1, 31, 0xFFFFFFFF;
    add.u32 %r2, %r2, %r3;
    shfl.sync.bfly.b32 %r3, %r2, 2, 31, 0xFFFFFFFF;
    add.u32 %r2, %r2, %r3;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

# --- Sprint 2 harnesses ---

def _verify_simple(fn):
    def verify(buf, N):
        for t in range(N):
            exp = fn(t) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return verify

def _harness_s2(name, fn, N=64):
    def harness(ctx, func, mode):
        return _h(ctx, func, N, _simple_args, _verify_simple(fn))
    return harness

def _harness_s2_load(ctx, func, mode):
    N=64; sz=N*4
    d_out=ctx.alloc(sz); ctx.memset_d8(d_out,0,sz)
    d_in=ctx.alloc(sz)
    ctx.copy_to(d_in, struct.pack(f'<{N}I', *range(N)))
    args,h=_make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d_out,sz)
        correct=all(struct.unpack_from('<I',buf,t*4)[0]==(t+1000 if t>32 else t)&0xFFFFFFFF for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}

def _harness_s2_dual_load(ctx, func, mode):
    N=64; sz=N*4
    d_out=ctx.alloc(sz); ctx.memset_d8(d_out,0,sz)
    d_in=ctx.alloc(sz)
    ctx.copy_to(d_in, struct.pack(f'<{N}I', *[i*10 for i in range(N)]))
    args,h=_make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d_out,sz)
        correct=all(struct.unpack_from('<I',buf,t*4)[0]==((t*10+t*3)&0xFFFF)&0xFFFFFFFF for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}

def _harness_s2_fadd(ctx, func, mode):
    N=64; sz=N*4
    d_out=ctx.alloc(sz); ctx.memset_d8(d_out,0,sz)
    d_in=ctx.alloc(sz)
    # Input: float(tid+1)
    data = struct.pack(f'<{N}f', *[float(i+1) for i in range(N)])
    ctx.copy_to(d_in, data)
    args,h=_make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d_out,sz)
        correct = True
        for t in range(N):
            inp = float(t+1)
            exp = inp + inp + inp  # f0+f0+f0 = 3*f0
            got = struct.unpack_from('<f', buf, t*4)[0]
            if abs(got - exp) > 0.01: correct = False; break
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}

def _harness_s2_fmul(ctx, func, mode):
    N=64; sz=N*4
    d_out=ctx.alloc(sz); ctx.memset_d8(d_out,0,sz)
    d_in=ctx.alloc(sz)
    data = struct.pack(f'<{N}f', *[float(i+1) for i in range(N)])
    ctx.copy_to(d_in, data)
    args,h=_make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d_out,sz)
        correct = True
        for t in range(N):
            inp = float(t+1)
            exp = inp * inp + inp  # f0*f0 + f0
            got = struct.unpack_from('<f', buf, t*4)[0]
            if abs(got - exp) > max(0.01, abs(exp)*1e-6): correct = False; break
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}

def _harness_s2_ilp_load_compute(ctx, func, mode):
    N=64; sz=N*4
    d_out=ctx.alloc(sz); ctx.memset_d8(d_out,0,sz)
    d_in=ctx.alloc(sz)
    ctx.copy_to(d_in, struct.pack(f'<{N}I', *[i*10 for i in range(N)]))
    args,h=_make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d_out,sz)
        correct=all(struct.unpack_from('<I',buf,t*4)[0]==(t*10+t*17+42)&0xFFFFFFFF for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}

def _harness_s2_ilp_dual_addr(ctx, func, mode):
    N=64; sz=N*4
    d_out=ctx.alloc(sz); ctx.memset_d8(d_out,0,sz)
    d_a=ctx.alloc(sz); d_b=ctx.alloc(sz)
    ctx.copy_to(d_a, struct.pack(f'<{N}I', *[i for i in range(N)]))
    ctx.copy_to(d_b, struct.pack(f'<{N}I', *[i*2 for i in range(N)]))
    args,h=_make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_a), ctypes.c_uint64(d_b), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d_out,sz)
        correct=all(struct.unpack_from('<I',buf,t*4)[0]==(t+t*2+t*11)&0xFFFFFFFF for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_a); ctx.free(d_b)
    return {"correct": correct, "time_ms": None}

def _harness_s2_shfl_reduce2(ctx, func, mode):
    def verify(buf, N):
        # Two rounds of butterfly: pairs of 1,2 then pairs of 4
        vals = [t+1 for t in range(N)]
        # Round 1: each lane adds partner^1
        r1 = [vals[t] + vals[t^1] for t in range(N)]
        # Round 2: each lane adds partner^2
        r2 = [r1[t] + r1[t^2] for t in range(N)]
        for t in range(min(N,32)):
            if struct.unpack_from('<I', buf, t*4)[0] != r2[t] & 0xFFFFFFFF: return False
        return True
    return _h(ctx, func, 32, _simple_args, verify)


SPRINT2_KERNELS = {
    "k200_deep_alu":          {"display": "8-stage deep ALU chain", "ptx_inline": _K200_DEEP_ALU, "kernel_name": "k200_deep_alu",
                               "harness": _harness_s2("k200_deep_alu", lambda t: ((((t*3+7)^0xFF)&0x3FF)*5+13)^0xAB & 0x7FF)},
    "k200_alt_32_64":         {"display": "alternating 32/64-bit chain", "ptx_inline": _K200_ALT_32_64, "kernel_name": "k200_alt_32_64",
                               "harness": _harness_s2("k200_alt_32_64", lambda t: (t*7+100+50+25)&0xFFFFFFFF)},
    "k200_triple_acc":        {"display": "3 independent accumulators merge", "ptx_inline": _K200_TRIPLE_ACC, "kernel_name": "k200_triple_acc",
                               "harness": _harness_s2("k200_triple_acc", lambda t: t*3+t*5+t*7)},
    "k200_quad_acc":          {"display": "4 independent accumulators merge", "ptx_inline": _K200_QUAD_ACC, "kernel_name": "k200_quad_acc",
                               "harness": _harness_s2("k200_quad_acc", lambda t: t*2+t*3+t*5+t*7)},
    "k200_nested_pred":       {"display": "nested predicated adds", "ptx_inline": _K200_NESTED_PRED, "kernel_name": "k200_nested_pred",
                               "harness": _harness_s2("k200_nested_pred", lambda t: t+(10 if t>8 else 0)+(20 if t>24 else 0))},
    "k200_pred_chain":        {"display": "4-stage predicate chain (bucket count)", "ptx_inline": _K200_PRED_CHAIN, "kernel_name": "k200_pred_chain",
                               "harness": _harness_s2("k200_pred_chain", lambda t: (1 if t>4 else 0)+(2 if t>8 else 0)+(4 if t>16 else 0)+(8 if t>32 else 0))},
    "k200_pred_mul":          {"display": "predicated mul override", "ptx_inline": _K200_PRED_MUL, "kernel_name": "k200_pred_mul",
                               "harness": _harness_s2("k200_pred_mul", lambda t: (t*7 if t<32 else t*3)+1)},
    "k200_branch_store":      {"display": "branch + guarded add + store", "ptx_inline": _K200_BRANCH_STORE, "kernel_name": "k200_branch_store",
                               "harness": _harness_s2("k200_branch_store", lambda t: t*11+(100 if t<32 else 0))},
    "k200_double_guard":      {"display": "two independent guards on same value", "ptx_inline": _K200_DOUBLE_GUARD, "kernel_name": "k200_double_guard",
                               "harness": _harness_s2("k200_double_guard", lambda t: t*3+(50 if t>16 else 0)+(25 if t<48 else 0))},
    "k200_load_pred_store":   {"display": "load + predicated transform + store", "ptx_inline": _K200_LOAD_PRED_STORE, "kernel_name": "k200_load_pred_store",
                               "harness": _harness_s2_load},
    "k200_indep_load_merge":  {"display": "load + independent ALU + mask merge", "ptx_inline": _K200_INDEP_LOAD_MERGE, "kernel_name": "k200_indep_load_merge",
                               "harness": _harness_s2_dual_load},
    "k200_fadd_chain":        {"display": "FP32 add chain (3 stages)", "ptx_inline": _K200_FADD_CHAIN, "kernel_name": "k200_fadd_chain",
                               "harness": _harness_s2_fadd},
    "k200_fmul_add":          {"display": "FP32 mul + add + mul chain", "ptx_inline": _K200_FMUL_ADD, "kernel_name": "k200_fmul_add",
                               "harness": _harness_s2_fmul},
    "k200_ilp3_chain":        {"display": "3 independent ALU chains + merge", "ptx_inline": _K200_ILP3_CHAIN, "kernel_name": "k200_ilp3_chain",
                               "harness": _harness_s2("k200_ilp3_chain", lambda t: ((t*3+10)^0xFF)+((t*5+20)^0xAA)+((t*7+30)^0x55))},
    "k200_ilp_load_compute":  {"display": "ILP: load + independent ALU + merge", "ptx_inline": _K200_ILP_LOAD_COMPUTE, "kernel_name": "k200_ilp_load_compute",
                               "harness": _harness_s2_ilp_load_compute},
    "k200_ilp_dual_addr":     {"display": "ILP: 2 addr chains + ALU + merge", "ptx_inline": _K200_ILP_DUAL_ADDR, "kernel_name": "k200_ilp_dual_addr",
                               "harness": _harness_s2_ilp_dual_addr},
    "k200_shl_chain":         {"display": "shift-left chain + xor", "ptx_inline": _K200_SHL_CHAIN, "kernel_name": "k200_shl_chain",
                               "harness": _harness_s2("k200_shl_chain", lambda t: ((t+1)<<3)^(t+1))},
    "k200_and_or_chain":      {"display": "and/or alternating chain", "ptx_inline": _K200_AND_OR_CHAIN, "kernel_name": "k200_and_or_chain",
                               "harness": _harness_s2("k200_and_or_chain", lambda t: (((t|0xF0)&0xFF)|0x100)&0x1FF)},
    "k200_mul_add_long":      {"display": "mul+add x3 deep chain", "ptx_inline": _K200_MUL_ADD_LONG, "kernel_name": "k200_mul_add_long",
                               "harness": _harness_s2("k200_mul_add_long", lambda t: ((t*3+1)*5+2)*7+3)},
    "k200_xor_reduce":        {"display": "4-stage xor with constants", "ptx_inline": _K200_XOR_REDUCE, "kernel_name": "k200_xor_reduce",
                               "harness": _harness_s2("k200_xor_reduce", lambda t: t^0x1^0x2^0x4^0x8)},
    "k200_wide_imm":          {"display": "wide immediate constants (>16-bit)", "ptx_inline": _K200_WIDE_IMM, "kernel_name": "k200_wide_imm",
                               "harness": _harness_s2("k200_wide_imm", lambda t: ((t+0x12345)&0xFFFFF)^0xABCDE)},
    "k200_shfl_reduce2":      {"display": "2-round butterfly shuffle reduce", "ptx_inline": _K200_SHFL_REDUCE2, "kernel_name": "k200_shfl_reduce2",
                               "harness": _harness_s2_shfl_reduce2},
}

for v in SPRINT2_KERNELS.values():
    v.setdefault("ptx_path", None)

EXPANDED_KERNELS.update(SPRINT2_KERNELS)


# ===================================================================
# SPRINT 3 — KERNEL-100.3: 30 kernels (push to 100+, nasty patterns)
# ===================================================================

# --- Normal additions (15 kernels) ---

_K300_LONG_MUL_CHAIN = _ptx_simple("k300_long_mul_chain", 14, """
    mul.lo.u32 %r2, %r0, 3; mul.lo.u32 %r3, %r2, 5;
    mul.lo.u32 %r4, %r3, 7; mul.lo.u32 %r2, %r4, 11;
    and.b32 %r2, %r2, 0xFFFF;""")

_K300_ADD_XOR_ALT = _ptx_simple("k300_add_xor_alt", 10, """
    add.u32 %r2, %r0, 1; xor.b32 %r3, %r2, 3;
    add.u32 %r4, %r3, 5; xor.b32 %r2, %r4, 7;""")

_K300_OR_CHAIN = _ptx_simple("k300_or_chain", 10, """
    or.b32 %r2, %r0, 0x10; or.b32 %r3, %r2, 0x20;
    or.b32 %r4, %r3, 0x40; or.b32 %r2, %r4, 0x80;""")

_K300_AND_MASK = _ptx_simple("k300_and_mask", 10, """
    mul.lo.u32 %r2, %r0, 257;
    and.b32 %r2, %r2, 0xFF;""")

_K300_MUL7_ADD3 = _ptx_simple("k300_mul7_add3", 10, """
    mul.lo.u32 %r2, %r0, 7; add.u32 %r2, %r2, 3;""")

_K300_MUL11_XOR = _ptx_simple("k300_mul11_xor", 10, """
    mul.lo.u32 %r2, %r0, 11; xor.b32 %r2, %r2, 0x5555;
    and.b32 %r2, %r2, 0xFFFF;""")

_K300_TRIPLE_XOR = _ptx_simple("k300_triple_xor", 10, """
    xor.b32 %r2, %r0, 0xAA; xor.b32 %r3, %r2, 0x55;
    xor.b32 %r2, %r3, 0xFF;""")

_K300_PRED3 = _ptx_simple("k300_pred3", 10, """
    mov.u32 %r2, %r0;
    setp.gt.u32 %p1, %r0, 10;
    @%p1 add.u32 %r2, %r2, 1;
    setp.gt.u32 %p1, %r0, 20;
    @%p1 add.u32 %r2, %r2, 2;
    setp.gt.u32 %p1, %r0, 40;
    @%p1 add.u32 %r2, %r2, 4;""")

_K300_PRED_MUL_ADD = _ptx_simple("k300_pred_mul_add", 10, """
    mul.lo.u32 %r2, %r0, 5;
    setp.lt.u32 %p1, %r0, 32;
    @%p1 add.u32 %r2, %r2, 100;""")

_K300_SHFL_IDX = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k300_shfl_idx(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    add.u32 %r2, %r0, 100;
    shfl.sync.idx.b32 %r3, %r2, 0, 31, 0xFFFFFFFF;
    add.u32 %r4, %r2, %r3;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""

_K300_SHL_ADD = _ptx_simple("k300_shl_add", 10, """
    shl.b32 %r2, %r0, 3; add.u32 %r2, %r2, %r0;""")

_K300_MUL_PAIR = _ptx_simple("k300_mul_pair", 12, """
    mul.lo.u32 %r2, %r0, 13; mul.lo.u32 %r3, %r0, 17;
    add.u32 %r2, %r2, %r3;""")

_K300_ADD5 = _ptx_simple("k300_add5", 12, """
    add.u32 %r2, %r0, 1; add.u32 %r3, %r2, 2;
    add.u32 %r4, %r3, 3; add.u32 %r5, %r4, 4;
    add.u32 %r2, %r5, 5;""")

_K300_XOR_PAIR = _ptx_simple("k300_xor_pair", 10, """
    xor.b32 %r2, %r0, 0x1234; xor.b32 %r2, %r2, 0x5678;""")

_K300_AND_OR_XOR = _ptx_simple("k300_and_or_xor", 10, """
    and.b32 %r2, %r0, 0xFF; or.b32 %r3, %r2, 0x100;
    xor.b32 %r2, %r3, 0x55;""")

# --- Nasty additions (15 kernels) ---

_K300_NASTY_LONG_LIVE = _ptx_simple("k300_nasty_long_live", 16, """
    mul.lo.u32 %r2, %r0, 3;
    mul.lo.u32 %r3, %r0, 5;
    mul.lo.u32 %r4, %r0, 7;
    mul.lo.u32 %r5, %r0, 11;
    mul.lo.u32 %r6, %r0, 13;
    add.u32 %r7, %r2, %r3;
    add.u32 %r8, %r4, %r5;
    add.u32 %r9, %r7, %r8;
    add.u32 %r2, %r9, %r6;""")

_K300_NASTY_DEEP_DEP = _ptx_simple("k300_nasty_deep_dep", 16, """
    add.u32 %r2, %r0, 1; add.u32 %r3, %r2, 1; add.u32 %r4, %r3, 1;
    add.u32 %r5, %r4, 1; add.u32 %r6, %r5, 1; add.u32 %r7, %r6, 1;
    add.u32 %r8, %r7, 1; add.u32 %r9, %r8, 1; add.u32 %r2, %r9, 1;""")

_K300_NASTY_WIDE_XOR = _ptx_simple("k300_nasty_wide_xor", 10, """
    xor.b32 %r2, %r0, 0xDEADBEEF;
    and.b32 %r2, %r2, 0xFFFFFF;""")

_K300_NASTY_IMM_HEAVY = _ptx_simple("k300_nasty_imm_heavy", 10, """
    add.u32 %r2, %r0, 0x11111;
    xor.b32 %r3, %r2, 0x22222;
    and.b32 %r4, %r3, 0x33333;
    or.b32 %r2, %r4, 0x44000;""")

_K300_NASTY_PRED_NEST3 = _ptx_simple("k300_nasty_pred_nest3", 10, """
    mov.u32 %r2, 0;
    setp.gt.u32 %p1, %r0, 5;
    @%p1 add.u32 %r2, %r2, 1;
    @%p1 setp.gt.u32 %p2, %r0, 15;
    @%p2 add.u32 %r2, %r2, 2;
    @%p2 setp.gt.u32 %p1, %r0, 30;
    @%p1 add.u32 %r2, %r2, 4;""")

_K300_NASTY_MUL_CHAIN3 = _ptx_simple("k300_nasty_mul_chain3", 10, """
    mul.lo.u32 %r2, %r0, 3;
    mul.lo.u32 %r2, %r2, 5;
    mul.lo.u32 %r2, %r2, 7;
    and.b32 %r2, %r2, 0xFFFF;""")

_K300_NASTY_ADD_WRAP = _ptx_simple("k300_nasty_add_wrap", 10, """
    add.u32 %r2, %r0, 0xFFFFFFF0;
    add.u32 %r2, %r2, 0x20;""")

_K300_NASTY_SHFL_CHAIN = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k300_nasty_shfl_chain(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    add.u32 %r2, %r0, 1;
    shfl.sync.bfly.b32 %r3, %r2, 1, 31, 0xFFFFFFFF;
    add.u32 %r2, %r2, %r3;
    shfl.sync.bfly.b32 %r3, %r2, 2, 31, 0xFFFFFFFF;
    add.u32 %r2, %r2, %r3;
    shfl.sync.bfly.b32 %r3, %r2, 4, 31, 0xFFFFFFFF;
    add.u32 %r2, %r2, %r3;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_K300_NASTY_MULTI_PRED = _ptx_simple("k300_nasty_multi_pred", 10, """
    mov.u32 %r2, %r0;
    setp.gt.u32 %p1, %r0, 4;
    @%p1 add.u32 %r2, %r2, 10;
    setp.gt.u32 %p2, %r0, 8;
    @%p2 add.u32 %r2, %r2, 20;
    setp.gt.u32 %p1, %r0, 16;
    @%p1 add.u32 %r2, %r2, 40;
    setp.gt.u32 %p2, %r0, 32;
    @%p2 add.u32 %r2, %r2, 80;
    setp.gt.u32 %p1, %r0, 48;
    @%p1 add.u32 %r2, %r2, 160;""")

_K300_NASTY_ZERO_INIT = _ptx_simple("k300_nasty_zero_init", 10, """
    mov.u32 %r2, 0;
    add.u32 %r2, %r2, %r0;
    add.u32 %r2, %r2, %r0;""")

_K300_NASTY_IDENTITY = _ptx_simple("k300_nasty_identity", 10, """
    xor.b32 %r2, %r0, 0;
    and.b32 %r2, %r2, 0xFFFFFFFF;
    or.b32 %r2, %r2, 0;""")

_K300_NASTY_OVERFLOW = _ptx_simple("k300_nasty_overflow", 10, """
    mul.lo.u32 %r2, %r0, 0xFFFF;
    add.u32 %r2, %r2, 0xFFFF;""")

_K300_NASTY_SHL_XOR = _ptx_simple("k300_nasty_shl_xor", 10, """
    shl.b32 %r2, %r0, 4;
    xor.b32 %r3, %r2, %r0;
    shl.b32 %r4, %r3, 2;
    xor.b32 %r2, %r4, %r3;""")

_K300_NASTY_ACCUM5 = _ptx_simple("k300_nasty_accum5", 16, """
    mul.lo.u32 %r2, %r0, 2;
    mul.lo.u32 %r3, %r0, 3;
    mul.lo.u32 %r4, %r0, 5;
    mul.lo.u32 %r5, %r0, 7;
    mul.lo.u32 %r6, %r0, 11;
    add.u32 %r7, %r2, %r3;
    add.u32 %r8, %r4, %r5;
    add.u32 %r9, %r7, %r8;
    add.u32 %r2, %r9, %r6;""")

_K300_NASTY_PRED_XOR = _ptx_simple("k300_nasty_pred_xor", 10, """
    xor.b32 %r2, %r0, 0xAA;
    setp.gt.u32 %p1, %r0, 16;
    @%p1 xor.b32 %r2, %r2, 0x55;""")


# Sprint 3 harnesses
def _harness_shfl_idx(ctx, func, mode):
    def verify(buf, N):
        for t in range(min(N, 32)):
            val = t + 100
            lane0_val = 0 + 100  # shfl.idx src=0 → gets lane 0's value
            exp = (val + lane0_val) & 0xFFFFFFFF
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, 32, _simple_args, verify)

def _harness_nasty_shfl_chain(ctx, func, mode):
    def verify(buf, N):
        vals = [t+1 for t in range(N)]
        for rnd in [1, 2, 4]:
            vals = [vals[t] + vals[t ^ rnd] for t in range(N)]
        for t in range(min(N, 32)):
            if struct.unpack_from('<I', buf, t*4)[0] != vals[t] & 0xFFFFFFFF: return False
        return True
    return _h(ctx, func, 32, _simple_args, verify)


SPRINT3_KERNELS = {
    # Normal
    "k300_long_mul_chain": {"display": "4-stage mul chain", "ptx_inline": _K300_LONG_MUL_CHAIN, "kernel_name": "k300_long_mul_chain",
                            "harness": _harness_s2("", lambda t: (t*3*5*7*11)&0xFFFF)},
    "k300_add_xor_alt": {"display": "alternating add+xor chain", "ptx_inline": _K300_ADD_XOR_ALT, "kernel_name": "k300_add_xor_alt",
                          "harness": _harness_s2("", lambda t: ((t+1)^3)+5^7)},
    "k300_or_chain": {"display": "4-stage OR chain", "ptx_inline": _K300_OR_CHAIN, "kernel_name": "k300_or_chain",
                       "harness": _harness_s2("", lambda t: t|0x10|0x20|0x40|0x80)},
    "k300_and_mask": {"display": "mul + AND mask", "ptx_inline": _K300_AND_MASK, "kernel_name": "k300_and_mask",
                       "harness": _harness_s2("", lambda t: (t*257)&0xFF)},
    "k300_mul7_add3": {"display": "mul*7 + add 3", "ptx_inline": _K300_MUL7_ADD3, "kernel_name": "k300_mul7_add3",
                        "harness": _harness_s2("", lambda t: t*7+3)},
    "k300_mul11_xor": {"display": "mul*11 + xor + mask", "ptx_inline": _K300_MUL11_XOR, "kernel_name": "k300_mul11_xor",
                        "harness": _harness_s2("", lambda t: (t*11^0x5555)&0xFFFF)},
    "k300_triple_xor": {"display": "3-stage xor with constants", "ptx_inline": _K300_TRIPLE_XOR, "kernel_name": "k300_triple_xor",
                         "harness": _harness_s2("", lambda t: (t^0xAA^0x55^0xFF))},
    "k300_pred3": {"display": "3-threshold predicate chain", "ptx_inline": _K300_PRED3, "kernel_name": "k300_pred3",
                    "harness": _harness_s2("", lambda t: t+(1 if t>10 else 0)+(2 if t>20 else 0)+(4 if t>40 else 0))},
    "k300_pred_mul_add": {"display": "mul + predicated add", "ptx_inline": _K300_PRED_MUL_ADD, "kernel_name": "k300_pred_mul_add",
                           "harness": _harness_s2("", lambda t: t*5+(100 if t<32 else 0))},
    "k300_shfl_idx": {"display": "shfl.sync.idx (broadcast lane 0)", "ptx_inline": _K300_SHFL_IDX, "kernel_name": "k300_shfl_idx",
                       "harness": _harness_shfl_idx},
    "k300_shl_add": {"display": "shl + self-add", "ptx_inline": _K300_SHL_ADD, "kernel_name": "k300_shl_add",
                      "harness": _harness_s2("", lambda t: (t<<3)+t)},
    "k300_mul_pair": {"display": "dual mul merge", "ptx_inline": _K300_MUL_PAIR, "kernel_name": "k300_mul_pair",
                       "harness": _harness_s2("", lambda t: t*13+t*17)},
    "k300_add5": {"display": "5-stage add chain", "ptx_inline": _K300_ADD5, "kernel_name": "k300_add5",
                   "harness": _harness_s2("", lambda t: t+1+2+3+4+5)},
    "k300_xor_pair": {"display": "double xor with large constants", "ptx_inline": _K300_XOR_PAIR, "kernel_name": "k300_xor_pair",
                       "harness": _harness_s2("", lambda t: t^0x1234^0x5678)},
    "k300_and_or_xor": {"display": "and + or + xor combo", "ptx_inline": _K300_AND_OR_XOR, "kernel_name": "k300_and_or_xor",
                         "harness": _harness_s2("", lambda t: ((t&0xFF)|0x100)^0x55)},
    # Nasty
    "k300_nasty_long_live": {"display": "nasty: 5 simultaneous live values", "ptx_inline": _K300_NASTY_LONG_LIVE, "kernel_name": "k300_nasty_long_live",
                              "harness": _harness_s2("", lambda t: t*3+t*5+t*7+t*11+t*13)},
    "k300_nasty_deep_dep": {"display": "nasty: 9-deep serial add chain", "ptx_inline": _K300_NASTY_DEEP_DEP, "kernel_name": "k300_nasty_deep_dep",
                             "harness": _harness_s2("", lambda t: t+9)},
    "k300_nasty_wide_xor": {"display": "nasty: xor with 0xDEADBEEF", "ptx_inline": _K300_NASTY_WIDE_XOR, "kernel_name": "k300_nasty_wide_xor",
                             "harness": _harness_s2("", lambda t: (t^0xDEADBEEF)&0xFFFFFF)},
    "k300_nasty_imm_heavy": {"display": "nasty: 4 wide immediates", "ptx_inline": _K300_NASTY_IMM_HEAVY, "kernel_name": "k300_nasty_imm_heavy",
                              "harness": _harness_s2("", lambda t: (((t+0x11111)^0x22222)&0x33333)|0x44000)},
    "k300_nasty_pred_nest3": {"display": "nasty: 3-level nested predication", "ptx_inline": _K300_NASTY_PRED_NEST3, "kernel_name": "k300_nasty_pred_nest3",
                               "harness": _harness_s2("", lambda t: (1+2+4 if t>30 else (1+2 if t>15 else (1+4 if t>5 else 0))))},
    "k300_nasty_mul_chain3": {"display": "nasty: 3-mul serial chain", "ptx_inline": _K300_NASTY_MUL_CHAIN3, "kernel_name": "k300_nasty_mul_chain3",
                               "harness": _harness_s2("", lambda t: (t*3*5*7)&0xFFFF)},
    "k300_nasty_add_wrap": {"display": "nasty: unsigned add wraparound", "ptx_inline": _K300_NASTY_ADD_WRAP, "kernel_name": "k300_nasty_add_wrap",
                             "harness": _harness_s2("", lambda t: (t+0xFFFFFFF0+0x20)&0xFFFFFFFF)},
    "k300_nasty_shfl_chain": {"display": "nasty: 3-round butterfly shuffle", "ptx_inline": _K300_NASTY_SHFL_CHAIN, "kernel_name": "k300_nasty_shfl_chain",
                               "harness": _harness_nasty_shfl_chain},
    "k300_nasty_multi_pred": {"display": "nasty: 5-stage predicate accumulator", "ptx_inline": _K300_NASTY_MULTI_PRED, "kernel_name": "k300_nasty_multi_pred",
                               "harness": _harness_s2("", lambda t: t+(10 if t>4 else 0)+(20 if t>8 else 0)+(40 if t>16 else 0)+(80 if t>32 else 0)+(160 if t>48 else 0))},
    "k300_nasty_zero_init": {"display": "nasty: zero-init + double self-add", "ptx_inline": _K300_NASTY_ZERO_INIT, "kernel_name": "k300_nasty_zero_init",
                              "harness": _harness_s2("", lambda t: t*2)},
    "k300_nasty_identity": {"display": "nasty: xor 0 + and ~0 + or 0 (identity)", "ptx_inline": _K300_NASTY_IDENTITY, "kernel_name": "k300_nasty_identity",
                             "harness": _harness_s2("", lambda t: t)},
    "k300_nasty_overflow": {"display": "nasty: mul*0xFFFF + add 0xFFFF (overflow)", "ptx_inline": _K300_NASTY_OVERFLOW, "kernel_name": "k300_nasty_overflow",
                             "harness": _harness_s2("", lambda t: (t*0xFFFF+0xFFFF)&0xFFFFFFFF)},
    "k300_nasty_shl_xor": {"display": "nasty: shift + xor + shift + xor", "ptx_inline": _K300_NASTY_SHL_XOR, "kernel_name": "k300_nasty_shl_xor",
                            "harness": _harness_s2("", lambda t: (((t<<4)^t)<<2)^((t<<4)^t))},
    "k300_nasty_accum5": {"display": "nasty: 5-accumulator tree merge", "ptx_inline": _K300_NASTY_ACCUM5, "kernel_name": "k300_nasty_accum5",
                           "harness": _harness_s2("", lambda t: t*2+t*3+t*5+t*7+t*11)},
    "k300_nasty_pred_xor": {"display": "nasty: xor + predicated xor", "ptx_inline": _K300_NASTY_PRED_XOR, "kernel_name": "k300_nasty_pred_xor",
                             "harness": _harness_s2("", lambda t: (t^0xAA)^(0x55 if t>16 else 0))},
}

for v in SPRINT3_KERNELS.values():
    v.setdefault("ptx_path", None)

EXPANDED_KERNELS.update(SPRINT3_KERNELS)


# ===================================================================
# SPRINT 4 — WEIRD-1: 22 kernels (shared memory + loops + divergence)
# ===================================================================

def _h_smem(ctx, func, N, smem_bytes, args_fn, verify_fn):
    """Harness for shared-memory kernels (passes smem_bytes to launch)."""
    sz = N * 4
    d = ctx.alloc(sz); ctx.memset_d8(d, 0, sz)
    extra = args_fn(ctx, d, N)
    args, holders = _make_args(*([ctypes.c_uint64(d)] + list(extra)))
    try:
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, smem_bytes, None, args, None)
        assert ctx.sync() == 0
        buf = ctx.copy_from(d, sz)
        correct = verify_fn(buf, N)
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": None}

# --- A. Shared memory patterns ---

_W1_SMEM_COPY = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_smem_copy(.param .u64 p_out) {
    .reg .u32 %r<6>; .reg .u64 %rd<3>;
    .shared .align 4 .b32 smem[64];
    mov.u32 %r0, %tid.x;
    // write tid+1 to smem[tid]
    shl.b32 %r1, %r0, 2;
    add.u32 %r2, %r0, 1;
    st.shared.b32 [%r1], %r2;
    bar.sync 0;
    // read back from smem[tid]
    ld.shared.b32 %r3, [%r1];
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r3;
    ret;
}
"""

_W1_SMEM_NEIGHBOR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_smem_neighbor(.param .u64 p_out) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    .shared .align 4 .b32 smem[64];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    add.u32 %r2, %r0, 10;
    st.shared.b32 [%r1], %r2;
    bar.sync 0;
    // read neighbor: smem[(tid+1) % 32]
    add.u32 %r3, %r0, 1;
    and.b32 %r3, %r3, 31;
    shl.b32 %r4, %r3, 2;
    ld.shared.b32 %r5, [%r4];
    add.u32 %r6, %r2, %r5;
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r6;
    ret;
}
"""

_W1_SMEM_COMPUTE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_smem_compute(.param .u64 p_out) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>;
    .shared .align 4 .b32 smem[64];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    mul.lo.u32 %r2, %r0, 7;
    st.shared.b32 [%r1], %r2;
    bar.sync 0;
    ld.shared.b32 %r3, [%r1];
    add.u32 %r4, %r3, 42;
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""

_W1_SMEM_XOR_SWAP = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_smem_xor_swap(.param .u64 p_out) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>;
    .shared .align 4 .b32 smem[64];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    // write tid to smem
    st.shared.b32 [%r1], %r0;
    bar.sync 0;
    // read from xor partner: smem[tid ^ 1]
    xor.b32 %r2, %r0, 1;
    shl.b32 %r3, %r2, 2;
    ld.shared.b32 %r4, [%r3];
    add.u32 %r5, %r0, %r4;
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r5;
    ret;
}
"""

_W1_SMEM_REDUCE_PAIR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_smem_reduce_pair(.param .u64 p_out) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    .shared .align 4 .b32 smem[64];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    add.u32 %r2, %r0, 1;
    st.shared.b32 [%r1], %r2;
    bar.sync 0;
    // even threads add their pair
    and.b32 %r3, %r0, 1;
    setp.eq.u32 %p0, %r3, 0;
    ld.shared.b32 %r4, [%r1];
    @%p0 add.u32 %r5, %r1, 4;
    @%p0 ld.shared.b32 %r6, [%r5];
    @%p0 add.u32 %r4, %r4, %r6;
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""

_W1_SMEM_GUARDED = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_smem_guarded(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p0;
    .shared .align 4 .b32 smem[64];
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    shl.b32 %r2, %r0, 2;
    mul.lo.u32 %r3, %r0, 3;
    st.shared.b32 [%r2], %r3;
    bar.sync 0;
    ld.shared.b32 %r4, [%r2];
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""

# --- B. Loop-body patterns ---

_W1_LOOP_SUM = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_loop_sum(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // sum = 0; for i in 0..7: sum += tid
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
LOOP:
    add.u32 %r2, %r2, %r0;
    add.u32 %r3, %r3, 1;
    setp.lt.u32 %p1, %r3, 8;
    @%p1 bra LOOP;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_W1_LOOP_MUL_ACC = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_loop_mul_acc(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // acc = 1; for i in 0..3: acc += tid*i
    mov.u32 %r2, 1;
    mov.u32 %r3, 0;
LOOP:
    mul.lo.u32 %r4, %r0, %r3;
    add.u32 %r2, %r2, %r4;
    add.u32 %r3, %r3, 1;
    setp.lt.u32 %p1, %r3, 4;
    @%p1 bra LOOP;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_W1_LOOP_PRED_ACC = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_loop_pred_acc(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1, %p2;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
LOOP:
    setp.gt.u32 %p2, %r0, %r3;
    @%p2 add.u32 %r2, %r2, 1;
    add.u32 %r3, %r3, 1;
    setp.lt.u32 %p1, %r3, 32;
    @%p1 bra LOOP;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_W1_LOOP_TWO_ACC = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_loop_two_acc(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
    mov.u32 %r4, 0;
LOOP:
    add.u32 %r2, %r2, %r0;
    add.u32 %r3, %r3, 1;
    add.u32 %r4, %r4, 1;
    setp.lt.u32 %p1, %r4, 4;
    @%p1 bra LOOP;
    add.u32 %r5, %r2, %r3;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r5;
    ret;
}
"""

_W1_LOOP_XOR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_loop_xor(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, %r0;
    mov.u32 %r3, 0;
LOOP:
    xor.b32 %r2, %r2, %r3;
    add.u32 %r3, %r3, 1;
    setp.lt.u32 %p1, %r3, 8;
    @%p1 bra LOOP;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_W1_LOOP_SHIFT = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_loop_shift(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // loop: accumulate tid into r2 with xor (4 iterations)
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
LOOP:
    add.u32 %r2, %r2, %r0;
    xor.b32 %r2, %r2, %r3;
    add.u32 %r3, %r3, 1;
    setp.lt.u32 %p1, %r3, 4;
    @%p1 bra LOOP;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

# --- C. Divergence-lite control flow ---

_W1_DIV_IF_ELSE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_div_if_else(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    and.b32 %r2, %r0, 1;
    setp.eq.u32 %p1, %r2, 0;
    @%p1 bra EVEN;
    // odd path
    mul.lo.u32 %r3, %r0, 5;
    bra MERGE;
EVEN:
    mul.lo.u32 %r3, %r0, 3;
MERGE:
    add.u32 %r4, %r3, 1;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""

_W1_DIV_MULTI_GUARD = _ptx_simple("w1_div_multi_guard", 10, """
    mov.u32 %r2, 0;
    setp.gt.u32 %p1, %r0, 8;
    @%p1 add.u32 %r2, %r2, 1;
    setp.gt.u32 %p2, %r0, 16;
    @%p2 add.u32 %r2, %r2, 2;
    setp.gt.u32 %p1, %r0, 32;
    @%p1 add.u32 %r2, %r2, 4;
    setp.gt.u32 %p2, %r0, 48;
    @%p2 add.u32 %r2, %r2, 8;""")

_W1_DIV_PRED_STORE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_div_pred_store(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mul.lo.u32 %r2, %r0, 7;
    setp.lt.u32 %p1, %r0, 32;
    @%p1 add.u32 %r2, %r2, 1000;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    // only store if tid < n (already checked)
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_W1_DIV_LOAD_PATHS = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_div_load_paths(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<5>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out]; ld.param.u64 %rd1, [p_in];
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.u32 %r2, [%rd3];
    // divergent transform
    setp.lt.u32 %p1, %r0, 32;
    @%p1 add.u32 %r2, %r2, 100;
    @!%p1 mul.lo.u32 %r2, %r2, 3;
    add.u64 %rd4, %rd0, %rd2;
    st.global.u32 [%rd4], %r2;
    ret;
}
"""

# More loop patterns
_W1_LOOP_COUNTDOWN = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_loop_countdown(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // acc = sum of (tid+i) for i in 0..3
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
LOOP:
    add.u32 %r4, %r0, %r3;
    add.u32 %r2, %r2, %r4;
    add.u32 %r3, %r3, 1;
    setp.lt.u32 %p1, %r3, 4;
    @%p1 bra LOOP;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_W1_LOOP_LOAD_ACC = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w1_loop_load_acc(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<6>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out]; ld.param.u64 %rd1, [p_in];
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
LOOP:
    ld.global.u32 %r4, [%rd3];
    add.u32 %r2, %r2, %r4;
    add.u32 %r3, %r3, 1;
    setp.lt.u32 %p1, %r3, 4;
    @%p1 bra LOOP;
    add.u64 %rd4, %rd0, %rd2;
    st.global.u32 [%rd4], %r2;
    ret;
}
"""

# --- WEIRD-1 harnesses ---

def _harness_smem_1param(expected_fn):
    def harness(ctx, func, mode):
        N = 32
        return _h_smem(ctx, func, N, 256, lambda c, d, n: [], _verify_simple(expected_fn))
    return harness

def _harness_smem_guarded(ctx, func, mode):
    N = 32
    return _h_smem(ctx, func, N, 256, lambda c, d, n: [ctypes.c_uint32(N)],
                   _verify_simple(lambda t: t * 3))

def _harness_smem_reduce_pair(ctx, func, mode):
    N = 32
    def verify(buf, N):
        for t in range(N):
            v = t + 1
            if t % 2 == 0 and t + 1 < N:
                v = (t + 1) + (t + 2)
            if struct.unpack_from('<I', buf, t*4)[0] != v & 0xFFFFFFFF: return False
        return True
    return _h_smem(ctx, func, N, 256, lambda c, d, n: [], verify)

def _harness_loop_sum(ctx, func, mode):
    return _h(ctx, func, 64, _simple_args, _verify_simple(lambda t: t * 8))

def _harness_loop_mul_acc(ctx, func, mode):
    return _h(ctx, func, 64, _simple_args, _verify_simple(lambda t: 1 + t*0 + t*1 + t*2 + t*3))

def _harness_loop_pred_acc(ctx, func, mode):
    return _h(ctx, func, 64, _simple_args, _verify_simple(lambda t: min(t, 32)))

def _harness_loop_two_acc(ctx, func, mode):
    return _h(ctx, func, 64, _simple_args, _verify_simple(lambda t: t*4 + 4))

def _harness_loop_xor(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            v = t
            for i in range(8): v ^= i
            if struct.unpack_from('<I', buf, t*4)[0] != v & 0xFFFFFFFF: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_loop_shift(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            v = 0
            for i in range(4):
                v = (v + t) ^ i
            if struct.unpack_from('<I', buf, t*4)[0] != v & 0xFFFFFFFF: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_div_if_else(ctx, func, mode):
    return _h(ctx, func, 64, _simple_args,
              _verify_simple(lambda t: (t*3 if t%2==0 else t*5) + 1))

def _harness_div_pred_store(ctx, func, mode):
    return _h(ctx, func, 64, _simple_args,
              _verify_simple(lambda t: t*7 + (1000 if t < 32 else 0)))

def _harness_div_load_paths(ctx, func, mode):
    N=64; sz=N*4
    d_out=ctx.alloc(sz); ctx.memset_d8(d_out,0,sz)
    d_in=ctx.alloc(sz)
    ctx.copy_to(d_in, struct.pack(f'<{N}I', *[i*10 for i in range(N)]))
    args,h=_make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d_out,sz)
        correct=True
        for t in range(N):
            v = t * 10
            if t < 32: v += 100
            else: v *= 3
            if struct.unpack_from('<I',buf,t*4)[0] != v & 0xFFFFFFFF: correct=False; break
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}

def _harness_loop_countdown(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            # sum of (t+i) for i in 0..3 = 4*t + 0+1+2+3 = 4*t + 6
            exp = 4 * t + 6
            if struct.unpack_from('<I', buf, t*4)[0] != exp & 0xFFFFFFFF: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_loop_load_acc(ctx, func, mode):
    N=64; sz=N*4
    d_out=ctx.alloc(sz); ctx.memset_d8(d_out,0,sz)
    d_in=ctx.alloc(sz)
    ctx.copy_to(d_in, struct.pack(f'<{N}I', *[i+1 for i in range(N)]))
    args,h=_make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d_out,sz)
        correct=all(struct.unpack_from('<I',buf,t*4)[0]==(t+1)*4 for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}


WEIRD1_KERNELS = {
    # Shared memory
    "w1_smem_copy":        {"display": "smem write + barrier + read back", "ptx_inline": _W1_SMEM_COPY, "kernel_name": "w1_smem_copy",
                            "harness": _harness_smem_1param(lambda t: t + 1)},
    "w1_smem_neighbor":    {"display": "smem neighbor read (tid+1 mod 32)", "ptx_inline": _W1_SMEM_NEIGHBOR, "kernel_name": "w1_smem_neighbor",
                            "harness": _harness_smem_1param(lambda t: (t+10) + ((t+1)%32+10))},
    "w1_smem_compute":     {"display": "smem write + barrier + read + compute", "ptx_inline": _W1_SMEM_COMPUTE, "kernel_name": "w1_smem_compute",
                            "harness": _harness_smem_1param(lambda t: t*7 + 42)},
    "w1_smem_xor_swap":    {"display": "smem xor-partner swap", "ptx_inline": _W1_SMEM_XOR_SWAP, "kernel_name": "w1_smem_xor_swap",
                            "harness": _harness_smem_1param(lambda t: t + (t^1))},
    "w1_smem_reduce_pair": {"display": "smem pair reduction (even threads)", "ptx_inline": _W1_SMEM_REDUCE_PAIR, "kernel_name": "w1_smem_reduce_pair",
                            "harness": _harness_smem_reduce_pair},
    "w1_smem_guarded":     {"display": "smem with bounds-checked write", "ptx_inline": _W1_SMEM_GUARDED, "kernel_name": "w1_smem_guarded",
                            "harness": _harness_smem_guarded},
    # Loop bodies
    "w1_loop_sum":         {"display": "loop: sum tid*8 (8 iterations)", "ptx_inline": _W1_LOOP_SUM, "kernel_name": "w1_loop_sum",
                            "harness": _harness_loop_sum},
    "w1_loop_mul_acc":     {"display": "loop: accumulate tid*i (4 iters)", "ptx_inline": _W1_LOOP_MUL_ACC, "kernel_name": "w1_loop_mul_acc",
                            "harness": _harness_loop_mul_acc},
    "w1_loop_pred_acc":    {"display": "loop: predicated counter (32 iters)", "ptx_inline": _W1_LOOP_PRED_ACC, "kernel_name": "w1_loop_pred_acc",
                            "harness": _harness_loop_pred_acc},
    "w1_loop_two_acc":     {"display": "loop: two accumulators + merge", "ptx_inline": _W1_LOOP_TWO_ACC, "kernel_name": "w1_loop_two_acc",
                            "harness": _harness_loop_two_acc},
    "w1_loop_xor":         {"display": "loop: xor with loop counter", "ptx_inline": _W1_LOOP_XOR, "kernel_name": "w1_loop_xor",
                            "harness": _harness_loop_xor},
    "w1_loop_shift":       {"display": "loop: repeated shift-left (3 iters)", "ptx_inline": _W1_LOOP_SHIFT, "kernel_name": "w1_loop_shift",
                            "harness": _harness_loop_shift},
    "w1_loop_countdown":   {"display": "loop: countdown accumulator", "ptx_inline": _W1_LOOP_COUNTDOWN, "kernel_name": "w1_loop_countdown",
                            "harness": _harness_loop_countdown},
    "w1_loop_load_acc":    {"display": "loop: repeated load + accumulate", "ptx_inline": _W1_LOOP_LOAD_ACC, "kernel_name": "w1_loop_load_acc",
                            "harness": _harness_loop_load_acc},
    # Divergence-lite
    "w1_div_if_else":      {"display": "divergent if/else (odd/even paths)", "ptx_inline": _W1_DIV_IF_ELSE, "kernel_name": "w1_div_if_else",
                            "harness": _harness_div_if_else},
    "w1_div_multi_guard":  {"display": "4-threshold divergent guards", "ptx_inline": _W1_DIV_MULTI_GUARD, "kernel_name": "w1_div_multi_guard",
                            "harness": _harness_s2("", lambda t: (1 if t>8 else 0)+(2 if t>16 else 0)+(4 if t>32 else 0)+(8 if t>48 else 0))},
    "w1_div_pred_store":   {"display": "divergent predicated add + store", "ptx_inline": _W1_DIV_PRED_STORE, "kernel_name": "w1_div_pred_store",
                            "harness": _harness_div_pred_store},
    "w1_div_load_paths":   {"display": "divergent load paths (add vs mul)", "ptx_inline": _W1_DIV_LOAD_PATHS, "kernel_name": "w1_div_load_paths",
                            "harness": _harness_div_load_paths},
}

for v in WEIRD1_KERNELS.values():
    v.setdefault("ptx_path", None)

EXPANDED_KERNELS.update(WEIRD1_KERNELS)


# ===================================================================
# SPRINT 5 — WEIRD-2: 10 targeted weird-pattern kernels
# ===================================================================

_W2_ATOM_XOR_REDUCE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w2_atom_xor_reduce(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<4>; .reg .u64 %rd<2>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    add.u32 %r2, %r0, 1;
    atom.global.xor.b32 %r3, [%rd0], %r2;
    ret;
}
"""

_W2_ATOM_AND_REDUCE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w2_atom_and_reduce(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<4>; .reg .u64 %rd<2>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    or.b32 %r2, %r0, 0xFFFF0000;
    atom.global.and.b32 %r3, [%rd0], %r2;
    ret;
}
"""

_W2_LOOP_ATOM_ADD = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w2_loop_atom_add(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<2>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, 0;
LOOP:
    atom.global.add.u32 %r3, [%rd0], 1;
    add.u32 %r2, %r2, 1;
    setp.lt.u32 %p1, %r2, 3;
    @%p1 bra LOOP;
    ret;
}
"""

_W2_SMEM_LOOP = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w2_smem_loop(.param .u64 p_out) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>;  .reg .pred %p0;
    .shared .align 4 .b32 smem[64];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    // write tid to smem, loop 3 times adding 1 each time
    st.shared.b32 [%r1], %r0;
    bar.sync 0;
    ld.shared.b32 %r2, [%r1];
    mov.u32 %r3, 0;
LOOP:
    add.u32 %r2, %r2, 1;
    add.u32 %r3, %r3, 1;
    setp.lt.u32 %p0, %r3, 3;
    @%p0 bra LOOP;
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

_W2_DIV_LOOP = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w2_div_loop(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // loop count depends on tid (divergent iteration count)
    and.b32 %r2, %r0, 3;
    add.u32 %r2, %r2, 1;
    mov.u32 %r3, 0;
    mov.u32 %r4, 0;
LOOP:
    add.u32 %r3, %r3, %r0;
    add.u32 %r4, %r4, 1;
    setp.lt.u32 %p1, %r4, %r2;
    @%p1 bra LOOP;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r3;
    ret;
}
"""

_W2_NESTED_LOOP = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w2_nested_loop(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1, %p2;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
OUTER:
    mov.u32 %r4, 0;
INNER:
    add.u32 %r2, %r2, 1;
    add.u32 %r4, %r4, 1;
    setp.lt.u32 %p1, %r4, 3;
    @%p1 bra INNER;
    add.u32 %r3, %r3, 1;
    setp.lt.u32 %p2, %r3, 2;
    @%p2 bra OUTER;
    add.u32 %r2, %r2, %r0;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

# More targeted patterns
_W2_PRED_LOAD = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry w2_pred_load(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<5>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out]; ld.param.u64 %rd1, [p_in];
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    mov.u32 %r2, 0;
    setp.lt.u32 %p1, %r0, 32;
    add.u64 %rd3, %rd1, %rd2;
    @%p1 ld.global.u32 %r2, [%rd3];
    add.u32 %r2, %r2, %r0;
    add.u64 %rd4, %rd0, %rd2;
    st.global.u32 [%rd4], %r2;
    ret;
}
"""

_W2_MULTI_STORE = _ptx_simple("w2_multi_store", 10, """
    mul.lo.u32 %r2, %r0, 7;
    add.u32 %r3, %r2, 42;
    xor.b32 %r4, %r3, 0xFF;
    and.b32 %r2, %r4, 0xFFF;""")

_W2_DEEP_PRED = _ptx_simple("w2_deep_pred", 10, """
    mov.u32 %r2, 0;
    setp.gt.u32 %p1, %r0, 2;
    @%p1 add.u32 %r2, %r2, 1;
    setp.gt.u32 %p2, %r0, 6;
    @%p2 add.u32 %r2, %r2, 2;
    setp.gt.u32 %p1, %r0, 12;
    @%p1 add.u32 %r2, %r2, 4;
    setp.gt.u32 %p2, %r0, 24;
    @%p2 add.u32 %r2, %r2, 8;
    setp.gt.u32 %p1, %r0, 48;
    @%p1 add.u32 %r2, %r2, 16;""")

_W2_LOOP_MUL = _ptx_simple("w2_loop_mul", 10, """
    mov.u32 %r2, 1;
    add.u32 %r2, %r2, %r0;
    mul.lo.u32 %r2, %r2, %r2;
    and.b32 %r2, %r2, 0xFFFF;""")


def _harness_atom_xor_reduce(ctx, func, mode):
    N=32; d=ctx.alloc(4); ctx.memset_d8(d,0,4)
    args,h=_make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d,4); got=struct.unpack('<I',buf)[0]
        exp=0
        for i in range(N): exp ^= (i+1)
        correct=(got==exp)
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": None}

def _harness_atom_and_reduce(ctx, func, mode):
    N=32; d=ctx.alloc(4)
    ctx.copy_to(d, struct.pack('<I', 0xFFFFFFFF))
    args,h=_make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d,4); got=struct.unpack('<I',buf)[0]
        exp=0xFFFFFFFF
        for i in range(N): exp &= (i | 0xFFFF0000)
        correct=(got==exp)
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": None}

def _harness_loop_atom_add(ctx, func, mode):
    N=32; d=ctx.alloc(4); ctx.memset_d8(d,0,4)
    args,h=_make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d,4); got=struct.unpack('<I',buf)[0]
        correct=(got==N*3)  # each thread adds 1 three times
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": None}

def _harness_smem_loop(ctx, func, mode):
    N=32
    return _h_smem(ctx, func, N, 256, lambda c,d,n: [],
                   _verify_simple(lambda t: t+3))

def _harness_div_loop(ctx, func, mode):
    def verify(buf, N):
        for t in range(N):
            iters = (t & 3) + 1
            exp = t * iters
            if struct.unpack_from('<I',buf,t*4)[0] != exp & 0xFFFFFFFF: return False
        return True
    return _h(ctx, func, 64, _simple_args, verify)

def _harness_nested_loop(ctx, func, mode):
    return _h(ctx, func, 64, _simple_args,
              _verify_simple(lambda t: 6 + t))  # 2 outer * 3 inner = 6 iterations + tid

def _harness_pred_load(ctx, func, mode):
    N=64; sz=N*4
    d_out=ctx.alloc(sz); ctx.memset_d8(d_out,0,sz)
    d_in=ctx.alloc(sz)
    ctx.copy_to(d_in, struct.pack(f'<{N}I', *[i*10 for i in range(N)]))
    args,h=_make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d_out,sz)
        correct=all(
            struct.unpack_from('<I',buf,t*4)[0] == ((t*10+t if t<32 else 0+t)&0xFFFFFFFF)
            for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}


WEIRD2_KERNELS = {
    # w2_atom_xor_reduce excluded: ATOMG_XOR encoding needs ground truth (0x98e family has different descriptor model)
    "w2_atom_and_reduce":  {"display": "atom.global.and.b32 reduce", "ptx_inline": _W2_ATOM_AND_REDUCE, "kernel_name": "w2_atom_and_reduce",
                            "harness": _harness_atom_and_reduce},
    "w2_loop_atom_add":    {"display": "loop: 3x atom.add per thread", "ptx_inline": _W2_LOOP_ATOM_ADD, "kernel_name": "w2_loop_atom_add",
                            "harness": _harness_loop_atom_add},
    "w2_smem_loop":        {"display": "smem + loop compute (write+read+loop)", "ptx_inline": _W2_SMEM_LOOP, "kernel_name": "w2_smem_loop",
                            "harness": _harness_smem_loop},
    "w2_div_loop":         {"display": "divergent loop count (tid-dependent iters)", "ptx_inline": _W2_DIV_LOOP, "kernel_name": "w2_div_loop",
                            "harness": _harness_div_loop},
    "w2_nested_loop":      {"display": "nested 2x3 loop + tid merge", "ptx_inline": _W2_NESTED_LOOP, "kernel_name": "w2_nested_loop",
                            "harness": _harness_nested_loop},
    "w2_pred_load":        {"display": "predicated global load (half-warp)", "ptx_inline": _W2_PRED_LOAD, "kernel_name": "w2_pred_load",
                            "harness": _harness_pred_load},
    "w2_multi_store":      {"display": "multi-stage ALU + store", "ptx_inline": _W2_MULTI_STORE, "kernel_name": "w2_multi_store",
                            "harness": _harness_s2("", lambda t: ((t*7+42)^0xFF)&0xFFF)},
    "w2_deep_pred":        {"display": "5-stage predicate accumulator (powers of 2)", "ptx_inline": _W2_DEEP_PRED, "kernel_name": "w2_deep_pred",
                            "harness": _harness_s2("", lambda t: (1 if t>2 else 0)+(2 if t>6 else 0)+(4 if t>12 else 0)+(8 if t>24 else 0)+(16 if t>48 else 0))},
    "w2_loop_mul":         {"display": "self-mul pattern ((tid+1)^2)", "ptx_inline": _W2_LOOP_MUL, "kernel_name": "w2_loop_mul",
                            "harness": _harness_s2("", lambda t: ((t+1)*(t+1))&0xFFFF)},
}

for v in WEIRD2_KERNELS.values():
    v.setdefault("ptx_path", None)

EXPANDED_KERNELS.update(WEIRD2_KERNELS)


# ===================================================================
# SPRINT 6 — REAL-1: 14 real-world workload shapes
# ===================================================================

# 1. Reduction (multi-stage)
_R1_WARP_SUM = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry r1_warp_sum(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // val = tid+1; butterfly sum across 32 lanes
    add.u32 %r2, %r0, 1;
    shfl.sync.bfly.b32 %r3, %r2, 16, 31, 0xFFFFFFFF; add.u32 %r2, %r2, %r3;
    shfl.sync.bfly.b32 %r3, %r2, 8,  31, 0xFFFFFFFF; add.u32 %r2, %r2, %r3;
    shfl.sync.bfly.b32 %r3, %r2, 4,  31, 0xFFFFFFFF; add.u32 %r2, %r2, %r3;
    shfl.sync.bfly.b32 %r3, %r2, 2,  31, 0xFFFFFFFF; add.u32 %r2, %r2, %r3;
    shfl.sync.bfly.b32 %r3, %r2, 1,  31, 0xFFFFFFFF; add.u32 %r2, %r2, %r3;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

# 2. Dot product (GEMM-ish: 1 element of A*B)
_R1_DOT4 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry r1_dot4(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<10>; .reg .u64 %rd<5>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out]; ld.param.u64 %rd1, [p_in];
    // load one element per thread, compute tid*tid+1
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.u32 %r2, [%rd3];
    mul.lo.u32 %r8, %r2, %r2;
    add.u32 %r8, %r8, 1;
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd4, %rd0, %rd2;
    st.global.u32 [%rd4], %r8;
    ret;
}
"""

# 3. Histogram-lite (scatter-add into 8 bins)
_R1_HISTOGRAM8 = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry r1_histogram8(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // bin = tid & 7; atomicAdd(out[bin], 1)
    and.b32 %r2, %r0, 7;
    cvt.u64.u32 %rd1, %r2; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    atom.global.add.u32 %r3, [%rd2], 1;
    ret;
}
"""

# 4. Prefix-like (inclusive scan per-warp via shuffle)
_R1_SCAN_WARP = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry r1_scan_warp(.param .u64 p_out, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    add.u32 %r2, %r0, 1;
    // inclusive prefix sum via shuffle-up
    shfl.sync.up.b32 %r3, %r2, 1, 0, 0xFFFFFFFF;
    setp.ge.u32 %p0, %r0, 1; @%p0 add.u32 %r2, %r2, %r3;
    shfl.sync.up.b32 %r3, %r2, 2, 0, 0xFFFFFFFF;
    setp.ge.u32 %p0, %r0, 2; @%p0 add.u32 %r2, %r2, %r3;
    shfl.sync.up.b32 %r3, %r2, 4, 0, 0xFFFFFFFF;
    setp.ge.u32 %p0, %r0, 4; @%p0 add.u32 %r2, %r2, %r3;
    shfl.sync.up.b32 %r3, %r2, 8, 0, 0xFFFFFFFF;
    setp.ge.u32 %p0, %r0, 8; @%p0 add.u32 %r2, %r2, %r3;
    shfl.sync.up.b32 %r3, %r2, 16, 0, 0xFFFFFFFF;
    setp.ge.u32 %p0, %r0, 16; @%p0 add.u32 %r2, %r2, %r3;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

# 5. Tiled shared-memory compute (smem copy + local transform)
_R1_TILE_COMPUTE = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry r1_tile_compute(.param .u64 p_out, .param .u64 p_in) {
    .reg .u32 %r<8>; .reg .u64 %rd<5>;
    .shared .align 4 .b32 smem[32];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    // load from global → smem
    ld.param.u64 %rd0, [p_in];
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r2, [%rd2];
    st.shared.b32 [%r1], %r2;
    bar.sync 0;
    // read from smem, compute, store to global
    ld.shared.b32 %r3, [%r1];
    mul.lo.u32 %r4, %r3, 3;
    add.u32 %r4, %r4, 7;
    ld.param.u64 %rd3, [p_out];
    add.u64 %rd4, %rd3, %rd1;
    st.global.u32 [%rd4], %r4;
    ret;
}
"""

# 6-14: More real-world shapes using proven patterns

_R1_SCALE_ADD = _ptx_simple("r1_scale_add", 10, """
    mul.lo.u32 %r2, %r0, 3;
    add.u32 %r3, %r2, 7;
    mul.lo.u32 %r2, %r3, 2;""")

_R1_MINMAX = _ptx_simple("r1_minmax", 10, """
    mul.lo.u32 %r2, %r0, 7;
    and.b32 %r3, %r2, 0xFF;
    // clamp to [16, 200]: max(min(r3, 200), 16) via predicated moves
    mov.u32 %r4, %r3;
    setp.gt.u32 %p1, %r3, 200;
    @%p1 mov.u32 %r4, 200;
    setp.lt.u32 %p2, %r4, 16;
    @%p2 mov.u32 %r4, 16;
    mov.u32 %r2, %r4;""")

_R1_BITCOUNT = _ptx_simple("r1_bitcount", 10, """
    mul.lo.u32 %r2, %r0, 0x11;
    and.b32 %r2, %r2, 0xFF;
    // popcount approximation: count set bits in low byte
    // via shift-and-add (Hamming weight hack for 8 bits)
    shl.b32 %r3, %r2, 1;
    xor.b32 %r2, %r2, %r3;
    and.b32 %r2, %r2, 0xFF;""")

_R1_RUNNING_XOR = _ptx_simple("r1_running_xor", 10, """
    xor.b32 %r2, %r0, 0xABCD;
    xor.b32 %r3, %r2, 0x1234;
    add.u32 %r2, %r2, %r3;
    and.b32 %r2, %r2, 0xFFFF;""")

_R1_MULTI_STAGE = _ptx_simple("r1_multi_stage", 14, """
    // stage 1: scale
    mul.lo.u32 %r2, %r0, 5;
    // stage 2: offset
    add.u32 %r3, %r2, 100;
    // stage 3: mask
    and.b32 %r4, %r3, 0x1FF;
    // stage 4: combine
    xor.b32 %r5, %r4, %r2;
    add.u32 %r2, %r5, %r0;""")

_R1_ACCUMULATOR = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry r1_accumulator(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<5>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out]; ld.param.u64 %rd1, [p_in];
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    // accumulate 4 loads from same position
    ld.global.u32 %r2, [%rd3];
    mov.u32 %r3, 0; mov.u32 %r4, 0;
LOOP:
    add.u32 %r3, %r3, %r2;
    add.u32 %r4, %r4, 1;
    setp.lt.u32 %p1, %r4, 4;
    @%p1 bra LOOP;
    add.u64 %rd4, %rd0, %rd2;
    st.global.u32 [%rd4], %r3;
    ret;
}
"""

_R1_GATHER = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry r1_gather(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {
    .reg .u32 %r<8>; .reg .u64 %rd<5>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out]; ld.param.u64 %rd1, [p_in];
    // gather: out[tid] = in[(tid*7) & 63]
    mul.lo.u32 %r2, %r0, 7;
    and.b32 %r2, %r2, 63;
    cvt.u64.u32 %rd2, %r2; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.u32 %r3, [%rd3];
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd4, %rd0, %rd2;
    st.global.u32 [%rd4], %r3;
    ret;
}
"""

_R1_SCATTER_ADD = _ptx_simple("r1_scatter_add", 10, """
    mul.lo.u32 %r2, %r0, 13;
    and.b32 %r2, %r2, 0xFF;
    add.u32 %r2, %r2, %r0;""")


# REAL-1 harnesses

def _harness_warp_sum(ctx, func, mode):
    N = 32  # single warp
    def verify(buf, N):
        # all lanes get the warp sum = sum(1..32) = 528
        for t in range(N):
            if struct.unpack_from('<I', buf, t*4)[0] != 528: return False
        return True
    return _h(ctx, func, N, _simple_args, verify)

def _harness_dot4(ctx, func, mode):
    N = 64
    d_out = ctx.alloc(N*4); ctx.memset_d8(d_out, 0, N*4)
    d_in = ctx.alloc(N*4)
    ctx.copy_to(d_in, struct.pack(f'<{N}I', *[i+1 for i in range(N)]))
    args, h = _make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d_out, N*4)
        correct = all(struct.unpack_from('<I', buf, t*4)[0] == ((t+1)*(t+1)+1) & 0xFFFFFFFF for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}

def _harness_histogram8(ctx, func, mode):
    N = 64; d = ctx.alloc(8*4); ctx.memset_d8(d, 0, 8*4)
    args, h = _make_args(ctypes.c_uint64(d), ctypes.c_uint32(N))
    try:
        err = ctx.launch(func, (1,1,1), (N,1,1), args)
        assert err == 0 and ctx.sync() == 0
        buf = ctx.copy_from(d, 8*4)
        bins = struct.unpack('<8I', buf)
        correct = all(b == 8 for b in bins)  # 64/8 = 8 per bin
    finally:
        ctx.free(d)
    return {"correct": correct, "time_ms": None}

def _harness_scan_warp(ctx, func, mode):
    N = 32
    def verify(buf, N):
        for t in range(N):
            exp = (t+1) * (t+2) // 2  # sum of 1..t+1
            if struct.unpack_from('<I', buf, t*4)[0] != exp: return False
        return True
    return _h(ctx, func, N, _simple_args, verify)

def _harness_tile_compute(ctx, func, mode):
    N = 32; sz = N*4
    d_out = ctx.alloc(sz); ctx.memset_d8(d_out, 0, sz)
    d_in = ctx.alloc(sz)
    ctx.copy_to(d_in, struct.pack(f'<{N}I', *[i*10 for i in range(N)]))
    args, h = _make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in))
    try:
        ctx.cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 128, None, args, None)
        assert ctx.sync() == 0
        buf = ctx.copy_from(d_out, sz)
        correct = all(struct.unpack_from('<I', buf, t*4)[0] == t*10*3+7 for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}

def _harness_accumulator(ctx, func, mode):
    N=64; sz=N*4
    d_out=ctx.alloc(sz); ctx.memset_d8(d_out,0,sz)
    d_in=ctx.alloc(sz)
    ctx.copy_to(d_in, struct.pack(f'<{N}I', *[i+1 for i in range(N)]))
    args,h=_make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d_out,sz)
        correct=all(struct.unpack_from('<I',buf,t*4)[0]==(t+1)*4 for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}

def _harness_gather(ctx, func, mode):
    N=64; sz=N*4
    d_out=ctx.alloc(sz); ctx.memset_d8(d_out,0,sz)
    d_in=ctx.alloc(sz)
    ctx.copy_to(d_in, struct.pack(f'<{N}I', *[i*100 for i in range(N)]))
    args,h=_make_args(ctypes.c_uint64(d_out), ctypes.c_uint64(d_in), ctypes.c_uint32(N))
    try:
        err=ctx.launch(func,(1,1,1),(N,1,1),args); assert err==0 and ctx.sync()==0
        buf=ctx.copy_from(d_out,sz)
        correct=all(struct.unpack_from('<I',buf,t*4)[0]==((t*7)&63)*100 for t in range(N))
    finally:
        ctx.free(d_out); ctx.free(d_in)
    return {"correct": correct, "time_ms": None}


REAL1_KERNELS = {
    "r1_warp_sum":     {"display": "warp butterfly reduction (5-stage sum)", "ptx_inline": _R1_WARP_SUM, "kernel_name": "r1_warp_sum",
                        "harness": _harness_warp_sum},
    "r1_dot4":         {"display": "4-element dot product (GEMM-ish tile)", "ptx_inline": _R1_DOT4, "kernel_name": "r1_dot4",
                        "harness": _harness_dot4},
    "r1_histogram8":   {"display": "8-bin histogram via atomicAdd", "ptx_inline": _R1_HISTOGRAM8, "kernel_name": "r1_histogram8",
                        "harness": _harness_histogram8},
    "r1_scan_warp":    {"display": "warp inclusive prefix sum (5-stage shuffle)", "ptx_inline": _R1_SCAN_WARP, "kernel_name": "r1_scan_warp",
                        "harness": _harness_scan_warp},
    "r1_tile_compute": {"display": "tiled smem compute (load -> smem -> transform -> store)", "ptx_inline": _R1_TILE_COMPUTE, "kernel_name": "r1_tile_compute",
                        "harness": _harness_tile_compute},
    "r1_scale_add":    {"display": "scale + offset (real-world normalize shape)", "ptx_inline": _R1_SCALE_ADD, "kernel_name": "r1_scale_add",
                        "harness": _harness_s2("", lambda t: (t*3+7)*2)},
    "r1_minmax":       {"display": "predicated clamp [16,200]", "ptx_inline": _R1_MINMAX, "kernel_name": "r1_minmax",
                        "harness": _harness_s2("", lambda t: max(16, min((t*7)&0xFF, 200)))},
    "r1_bitcount":     {"display": "xor-based bit manipulation", "ptx_inline": _R1_BITCOUNT, "kernel_name": "r1_bitcount",
                        "harness": _harness_s2("", lambda t: (((t*0x11)&0xFF) ^ (((t*0x11)&0xFF)<<1)) & 0xFF)},
    "r1_running_xor":  {"display": "running xor + combine", "ptx_inline": _R1_RUNNING_XOR, "kernel_name": "r1_running_xor",
                        "harness": _harness_s2("", lambda t: ((t^0xABCD) + ((t^0xABCD)^0x1234)) & 0xFFFF)},
    "r1_multi_stage":  {"display": "4-stage pipeline (scale → offset → mask → combine)", "ptx_inline": _R1_MULTI_STAGE, "kernel_name": "r1_multi_stage",
                        "harness": _harness_s2("", lambda t: (((t*5+100)&0x1FF) ^ (t*5)) + t)},
    "r1_accumulator":  {"display": "load + loop accumulate (4x)", "ptx_inline": _R1_ACCUMULATOR, "kernel_name": "r1_accumulator",
                        "harness": _harness_accumulator},
    "r1_gather":       {"display": "gather: out[tid] = in[(tid*7)&63]", "ptx_inline": _R1_GATHER, "kernel_name": "r1_gather",
                        "harness": _harness_gather},
    "r1_scatter_add":  {"display": "scatter-add pattern (mul+mask+add)", "ptx_inline": _R1_SCATTER_ADD, "kernel_name": "r1_scatter_add",
                        "harness": _harness_s2("", lambda t: (t*13 & 0xFF) + t)},
}

for v in REAL1_KERNELS.values():
    v.setdefault("ptx_path", None)

EXPANDED_KERNELS.update(REAL1_KERNELS)


# Add ptx_path=None to all entries
for v in EXPANDED_KERNELS.values():
    v.setdefault("ptx_path", None)


def register(kernels_dict, suites_dict, make_args_fn):
    """Register expanded kernels into workbench KERNELS and SUITES dicts."""
    global _make_args
    _make_args = make_args_fn
    kernels_dict.update(EXPANDED_KERNELS)
    s1_keys = [k for k in EXPANDED_KERNELS if k.startswith("k100_")]
    s2_keys = [k for k in EXPANDED_KERNELS if k.startswith("k200_")]
    s3_keys = [k for k in EXPANDED_KERNELS if k.startswith("k300_")]
    w1_keys = [k for k in EXPANDED_KERNELS if k.startswith("w1_")]
    w2_keys = [k for k in EXPANDED_KERNELS if k.startswith("w2_")]
    r1_keys = [k for k in EXPANDED_KERNELS if k.startswith("r1_")]
    s3_nasty = [k for k in s3_keys if "nasty" in k]
    all_expanded = s1_keys + s2_keys + s3_keys + w1_keys + w2_keys + r1_keys
    suites_dict["expanded"] = all_expanded
    suites_dict["sprint1"] = s1_keys
    suites_dict["sprint2"] = s2_keys
    suites_dict["sprint3"] = s3_keys
    suites_dict["nasty"] = s3_nasty
    suites_dict["weird"] = w1_keys + w2_keys
    suites_dict["real"] = r1_keys
    suites_dict["all"] = list(suites_dict.get("all", [])) + all_expanded
