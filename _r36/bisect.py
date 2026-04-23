"""R36: bisect on compact pass — run s2_fail with and without compact.

Strategy: monkey-patch `sass.compact.compact` to be a no-op, rebuild,
and run. Compare against the normal path.
"""
from __future__ import annotations
import ctypes, struct, sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')


_PTX = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry s2_fail(.param .u64 in, .param .u64 out) {
    .reg .b32 %r<4>; .reg .b64 %rd<3>; .reg .pred %p<1>;
    ld.param.u64 %rd0, [in]; ld.param.u64 %rd1, [out];
    ld.global.u32 %r0, [%rd0]; mov.u32 %r1, %tid.x; setp.eq.u32 %p0, %r1, 0; @!%p0 ret;
    mov.u32 %r2, %ctaid.x; shl.b32 %r3, %r2, 2; cvt.u64.u32 %rd2, %r3;
    add.u64 %rd1, %rd1, %rd2; st.global.u32 [%rd1], %r0; ret;
}
"""


def compile_variant(bypass_compact: bool) -> bytes:
    # Fresh-import the pipeline so our monkey-patch applies cleanly.
    for m in list(sys.modules):
        if m.startswith('sass.') or m == 'sass':
            del sys.modules[m]
    import sass.compact as _sc
    _orig_compact = _sc.compact
    if bypass_compact:
        def _noop_compact(instrs, verbose=False, kernel_name=''):
            return instrs, 0
        _sc.compact = _noop_compact
    try:
        from ptx.parser import parse
        from sass.pipeline import compile_function
        cubin = compile_function(parse(_PTX).functions[0],
                                 verbose=False, sm_version=120)
        return cubin
    finally:
        _sc.compact = _orig_compact


def run_cubin(cubin: bytes, label: str) -> tuple[int, int]:
    cuda = ctypes.WinDLL('nvcuda'); cuda.cuInit(0)
    dev = ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    try:
        mod = ctypes.c_void_p()
        err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
        if err != 0:
            print(f'[{label}] cuModuleLoadData={err}')
            return err, 0
        func = ctypes.c_void_p()
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, b's2_fail')
        d_in = ctypes.c_uint64(); cuda.cuMemAlloc_v2(ctypes.byref(d_in), 4)
        d_out = ctypes.c_uint64(); cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4)
        cuda.cuMemcpyHtoD_v2(d_in, struct.pack('<I', 777), 4)
        cuda.cuMemcpyHtoD_v2(d_out, struct.pack('<I', 0), 4)
        a_in = ctypes.c_uint64(d_in.value)
        a_out = ctypes.c_uint64(d_out.value)
        argv = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.byref(a_in), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(a_out), ctypes.c_void_p))
        cuda.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, None, argv, None)
        err = cuda.cuCtxSynchronize()
        buf = ctypes.create_string_buffer(4)
        cuda.cuMemcpyDtoH_v2(buf, d_out, 4)
        val = struct.unpack('<I', buf.raw)[0]
        ok = 'PASS' if err == 0 and val == 777 else 'FAIL'
        print(f'[{label}] sync={err} out={val} expect=777 {ok}')
        return err, val
    finally:
        cuda.cuCtxDestroy_v2(ctx)


if __name__ == '__main__':
    import subprocess
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        bypass = (mode == 'bypass')
        cubin = compile_variant(bypass_compact=bypass)
        # Also dump SASS for the .text.s2_fail section
        e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
        e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
        e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]
        def sh(i): return struct.unpack_from('<IIQQQQIIQQ', cubin, e_shoff + i * 64)
        _, _, _, _, so, ss, *_ = sh(e_shstrndx)
        shs = cubin[so:so + ss]
        for i in range(e_shnum):
            nm, ty, _, _, off, sz, *_ = sh(i)
            end = shs.index(b'\x00', nm)
            if shs[nm:end] == b'.text.s2_fail' and ty == 1:
                text = cubin[off:off + sz]
                break
        print(f'=== s2_fail SASS ({mode}) text size={len(text)} ===')
        OPC = {0xb82: 'LDC', 0x7ac: 'LDCU', 0x919: 'S2R', 0x981: 'LDG.E',
               0x986: 'STG.E', 0xc35: 'IADD.64.RUR', 0x210: 'IADD3',
               0x80c: 'ISETP.IMM', 0x947: 'BRA', 0x94d: 'EXIT',
               0x918: 'NOP', 0x824: 'IMAD.SHL', 0x810: 'IADD3.IMM'}
        for i in range(0, len(text), 16):
            r = text[i:i + 16]
            opc = (r[0] | (r[1] << 8)) & 0xFFF
            name = OPC.get(opc, f'op={opc:#05x}')
            print(f'{i//16:2d} {name:14s} b2={r[2]:3d} b3={r[3]:3d} b4={r[4]:3d} '
                  f'b8={r[8]:3d} b9={r[9]:#04x}')
        run_cubin(cubin, f's2_fail[{mode}]')
    else:
        # Run both variants in separate subprocesses to avoid cache poisoning.
        print('--- WITH COMPACT (normal path) ---')
        subprocess.run([sys.executable, __file__, 'normal'])
        print()
        print('--- WITHOUT COMPACT (bypassed) ---')
        subprocess.run([sys.executable, __file__, 'bypass'])
