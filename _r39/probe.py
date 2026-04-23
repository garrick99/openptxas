"""R39: probe post-EXIT S2R hazard for multiple consumer classes.

For each probe:
  * Post-EXIT S2R of SR_CTAID_X (ctaid = 0 in grid(1,1,1))
  * Immediate consumer op reads the S2R dest
  * Store the consumer's result to `out` and verify against expected

If the hazard is general to "any post-EXIT S2R consumer without gap",
non-IMAD.SHL.U32 consumers will also produce garbage.  If R38 is
already sufficient, those consumers must still PASS because the HW
only has the hazard for IMAD.SHL.U32.

Probes use a SCALAR-LDG input param so the compilation is as close to
s2_fail as possible (same pipeline phases: R29.1 direct_ldc_params,
R31, R32', R38).
"""
from __future__ import annotations
import ctypes, struct, subprocess, sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')
from ptx.parser import parse
from sass.pipeline import compile_function


# Each probe: (PTX, inputs, expected output value).
# `in` holds a known value that is LDG-loaded into %r0 and used as the
# base for the consumer's computation.  That way the bad value in %rY
# is observable as a wrong output.
#
# Structure is chosen to match s2_fail (scalar-LDG + predicated EXIT +
# post-EXIT S2R of SR_CTAID_X + consumer).  We DON'T do the full
# pair-build / IADD.64 / STG chain — we just store %rY directly so we
# can see the hazard as a wrong stored VALUE rather than a crash.
#
# Thread 0 (tid=0, ctaid=0) should produce deterministic output.
PROBES = {
# Control 1: S2R CTAID -> IMAD.SHL.U32 (the R38-fixed class)
'c1_shl': ('''
.version 9.0
.target sm_120
.address_size 64
.visible .entry c1_shl(.param .u64 in, .param .u64 out) {
    .reg .b32 %r<5>; .reg .b64 %rd<3>; .reg .pred %p<1>;
    ld.param.u64 %rd0, [in]; ld.param.u64 %rd1, [out];
    ld.global.u32 %r0, [%rd0];
    mov.u32 %r1, %tid.x; setp.eq.u32 %p0, %r1, 0; @!%p0 ret;
    mov.u32 %r2, %ctaid.x;
    shl.b32 %r3, %r2, 2;        // S2R -> IMAD.SHL.U32
    add.u32 %r4, %r0, %r3;
    st.global.u32 [%rd1], %r4; ret;
}''', 777),

# Probe 2: S2R CTAID -> IADD3 (add.u32 with imm)
'p2_add_imm': ('''
.version 9.0
.target sm_120
.address_size 64
.visible .entry p2_add_imm(.param .u64 in, .param .u64 out) {
    .reg .b32 %r<5>; .reg .b64 %rd<3>; .reg .pred %p<1>;
    ld.param.u64 %rd0, [in]; ld.param.u64 %rd1, [out];
    ld.global.u32 %r0, [%rd0];
    mov.u32 %r1, %tid.x; setp.eq.u32 %p0, %r1, 0; @!%p0 ret;
    mov.u32 %r2, %ctaid.x;
    add.u32 %r3, %r2, 42;       // S2R -> IADD3.IMM
    add.u32 %r4, %r0, %r3;
    st.global.u32 [%rd1], %r4; ret;
}''', 777 + 42),

# Probe 3: S2R CTAID -> IMAD (non-SHL, via mul.lo.u32)
'p3_mul_lo': ('''
.version 9.0
.target sm_120
.address_size 64
.visible .entry p3_mul_lo(.param .u64 in, .param .u64 out) {
    .reg .b32 %r<6>; .reg .b64 %rd<3>; .reg .pred %p<1>;
    ld.param.u64 %rd0, [in]; ld.param.u64 %rd1, [out];
    ld.global.u32 %r0, [%rd0];
    mov.u32 %r1, %tid.x; setp.eq.u32 %p0, %r1, 0; @!%p0 ret;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, 7;
    mul.lo.u32 %r4, %r2, %r3;   // S2R -> IMAD (non-SHL)
    add.u32 %r5, %r0, %r4;
    st.global.u32 [%rd1], %r5; ret;
}''', 777),

# Probe 4: S2R CTAID -> LOP3 (bitwise op)
'p4_or': ('''
.version 9.0
.target sm_120
.address_size 64
.visible .entry p4_or(.param .u64 in, .param .u64 out) {
    .reg .b32 %r<5>; .reg .b64 %rd<3>; .reg .pred %p<1>;
    ld.param.u64 %rd0, [in]; ld.param.u64 %rd1, [out];
    ld.global.u32 %r0, [%rd0];
    mov.u32 %r1, %tid.x; setp.eq.u32 %p0, %r1, 0; @!%p0 ret;
    mov.u32 %r2, %ctaid.x;
    or.b32 %r3, %r2, 99;        // S2R -> LOP3
    add.u32 %r4, %r0, %r3;
    st.global.u32 [%rd1], %r4; ret;
}''', 777 + 99),
}


def _dump_s2r_next(cubin: bytes, kname: str) -> tuple[int, int, int, int]:
    """Return (post_exit_s2r_opc, next_opc, next_b3, s2r_dest) from final SASS."""
    e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]
    def sh(i): return struct.unpack_from('<IIQQQQIIQQ', cubin, e_shoff + i*64)
    _, _, _, _, so, ss, *_ = sh(e_shstrndx)
    shs = cubin[so:so+ss]
    target = f'.text.{kname}'.encode()
    for i in range(e_shnum):
        nm, ty, _, _, off, sz, *_ = sh(i)
        end = shs.index(b'\x00', nm)
        if shs[nm:end] == target and ty == 1:
            text = cubin[off:off+sz]
            break
    seen_pexit = False
    for a in range(0, len(text), 16):
        r = text[a:a+16]
        opc = (r[0] | (r[1]<<8)) & 0xFFF
        guard = (r[1] >> 4) & 0xF
        if opc == 0x94d and guard != 0x7:
            seen_pexit = True
            continue
        if seen_pexit and opc == 0x919 and r[9] == 0x25:  # S2R CTAID_X
            s2r_dest = r[2]
            nr = text[a+16:a+32]
            next_opc = (nr[0] | (nr[1]<<8)) & 0xFFF
            return opc, next_opc, nr[3], s2r_dest
    return 0, 0, 0, 0


def run(kname):
    ptx, expected = PROBES[kname]
    cuda = ctypes.WinDLL('nvcuda'); cuda.cuInit(0)
    dev = ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p(); cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    try:
        cubin = compile_function(parse(ptx).functions[0], verbose=False, sm_version=120)
        _, next_opc, next_b3, s2r_dest = _dump_s2r_next(cubin, kname)
        mod = ctypes.c_void_p(); cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
        func = ctypes.c_void_p(); cuda.cuModuleGetFunction(ctypes.byref(func), mod, kname.encode())
        d_in = ctypes.c_uint64(); cuda.cuMemAlloc_v2(ctypes.byref(d_in), 4)
        d_out = ctypes.c_uint64(); cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4)
        cuda.cuMemcpyHtoD_v2(d_in, struct.pack('<I', 777), 4)
        cuda.cuMemcpyHtoD_v2(d_out, struct.pack('<I', 0), 4)
        a_in = ctypes.c_uint64(d_in.value); a_out = ctypes.c_uint64(d_out.value)
        argv = (ctypes.c_void_p*2)(ctypes.cast(ctypes.byref(a_in), ctypes.c_void_p),
                                    ctypes.cast(ctypes.byref(a_out), ctypes.c_void_p))
        cuda.cuLaunchKernel(func, 1,1,1, 32,1,1, 0, None, argv, None)
        err = cuda.cuCtxSynchronize()
        buf = ctypes.create_string_buffer(4); cuda.cuMemcpyDtoH_v2(buf, d_out, 4)
        val = struct.unpack('<I', buf.raw)[0]
        ok = 'PASS' if err == 0 and val == expected else 'FAIL'
        print(f'[{kname:12s}] next_opc=0x{next_opc:03x} next_b3=R{next_b3} s2r_dest=R{s2r_dest}  '
              f'sync={err} out={val} exp={expected} {ok}')
    finally:
        cuda.cuCtxDestroy_v2(ctx)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        for k in PROBES:
            subprocess.run([sys.executable, __file__, k])
