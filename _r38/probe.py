"""R38: sync-primitive probe for s2_fail.

Insert a synchronization primitive at selected positions in the
post-EXIT body and test each variant.  In-place shift technique (eats
trailing padding NOPs so total .text size stays constant).

Primitives probed:
  * BSYNC (opc 0x941 — ptxas's RECONVERGENT variant)
  * WARPSYNC (opc 0x948 with mask=0xffffffff — warpsync.sync)
  * MEMBAR.SC.CTA (opc 0x992)

Placements probed:
  * A: immediately BEFORE IADD.64 R-UR (opc 0xc35)
  * B: immediately AFTER post-EXIT S2R CTAID (opc 0x919, b9=0x25)
"""
from __future__ import annotations
import ctypes, struct, subprocess, sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')

from ptx.parser import parse
from sass.pipeline import compile_function


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


NOP   = bytes.fromhex('18790000000000000000000000c00f00')
# BSYNC.RECONVERGENT (from isel.py line 3734)
BSYNC = bytes.fromhex('41790000000000000002800300ea1f00')
# MEMBAR.SC.CTA (scope=0x00 in b9, per encode_membar)
MEMBAR_CTA = bytes.fromhex('92790000000000000000000000ea1f00')
# MEMBAR.SC.GPU (scope=0x20 in b9)
MEMBAR_GPU = bytes.fromhex('92790000000000000020000000ea1f00')
# WARPSYNC — opcode 0x948 with full mask
# (try a plausible encoding with mask = all-ones in b4-7; if unsupported
#  the HW will reject and we'll see a load error)
WARPSYNC_ALL = bytes.fromhex('4879ff00ffffffff0000000000ea1f00')


def _find_text_section(cubin, name):
    e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]

    def sh(i):
        return struct.unpack_from('<IIQQQQIIQQ', cubin, e_shoff + i * 64)

    _, _, _, _, so, ss, *_ = sh(e_shstrndx)
    shs = cubin[so:so + ss]
    target = f'.text.{name}'.encode()
    for i in range(e_shnum):
        nm, ty, _, _, off, sz, *_ = sh(i)
        end = shs.index(b'\x00', nm)
        if shs[nm:end] == target and ty == 1:
            return i, off, sz
    raise AssertionError(f'no .text.{name}')


def build_variant(primitive_bytes: bytes | None, placement: str) -> bytes:
    """placement: 'before-iadd' or 'after-s2r-ctaid' or 'none'."""
    cubin = bytearray(compile_function(parse(_PTX).functions[0],
                                        verbose=False, sm_version=120))
    if primitive_bytes is None:
        return bytes(cubin)

    _, sec_off, sec_sz = _find_text_section(cubin, 's2_fail')
    text = bytearray(cubin[sec_off:sec_off + sec_sz])
    n = len(text) // 16
    instrs = [bytes(text[i * 16:(i + 1) * 16]) for i in range(n)]

    # Target instruction index.
    tgt = None
    if placement == 'before-iadd':
        # Insert AT the IADD.64 R-UR position (shifts IADD+following by 1).
        for i, ins in enumerate(instrs):
            if ins[0] == 0x35 and ins[1] == 0x7c:  # IADD.64 R-UR
                tgt = i
                break
    elif placement == 'after-s2r-ctaid':
        # Insert right AFTER post-EXIT S2R CTAID_X.
        for i, ins in enumerate(instrs):
            opc = (ins[0] | (ins[1] << 8)) & 0xFFF
            if opc == 0x919 and ins[9] == 0x25:
                tgt = i + 1
                break
    else:
        raise ValueError(f'unknown placement: {placement}')

    if tgt is None:
        raise AssertionError(f'placement {placement}: target instruction not found')

    # Count trailing padding NOPs.
    trailing = 0
    for i in range(n - 1, -1, -1):
        if instrs[i] == NOP:
            trailing += 1
        else:
            break
    if trailing < 1:
        raise AssertionError('no trailing NOP to consume')

    # Shift [tgt, n - trailing) down by 1, primitive at tgt.
    body = instrs[tgt : n - trailing]
    new_instrs = (
        instrs[:tgt]
        + [primitive_bytes]
        + body
        + instrs[n - trailing + 1:]
    )
    assert len(new_instrs) == n, f'{len(new_instrs)} vs {n}'

    new_text = b''.join(new_instrs)
    new_cubin = bytearray(cubin)
    new_cubin[sec_off:sec_off + sec_sz] = new_text
    return bytes(new_cubin)


def run(cubin: bytes, label: str) -> None:
    cuda = ctypes.WinDLL('nvcuda'); cuda.cuInit(0)
    dev = ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    try:
        mod = ctypes.c_void_p()
        err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
        if err != 0:
            print(f'[{label:40s}] cuModuleLoadData={err}')
            return
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
        print(f'[{label:40s}] sync={err} out=0x{val:08x} {ok}')
    finally:
        cuda.cuCtxDestroy_v2(ctx)


VARIANTS = [
    ('baseline',                     None,         'none'),
    ('BSYNC    before IADD.64',      BSYNC,        'before-iadd'),
    ('BSYNC    after S2R-CTAID',     BSYNC,        'after-s2r-ctaid'),
    ('MEMBAR.CTA before IADD.64',    MEMBAR_CTA,   'before-iadd'),
    ('MEMBAR.CTA after S2R-CTAID',   MEMBAR_CTA,   'after-s2r-ctaid'),
    ('MEMBAR.GPU before IADD.64',    MEMBAR_GPU,   'before-iadd'),
    ('WARPSYNC(all) before IADD.64', WARPSYNC_ALL, 'before-iadd'),
    ('WARPSYNC(all) after S2R-CTAID',WARPSYNC_ALL, 'after-s2r-ctaid'),
]


if __name__ == '__main__':
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        label, prim, place = VARIANTS[idx]
        cubin = build_variant(prim, place)
        run(cubin, label)
    else:
        for i in range(len(VARIANTS)):
            subprocess.run([sys.executable, __file__, str(i)])
