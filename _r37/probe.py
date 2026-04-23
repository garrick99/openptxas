"""R37: gap sensitivity probe for post-EXIT pair-build -> IADD.64 R-UR hazard.

Strategy: compile s2_fail normally, then cubin-level patch:
  - locate the IADD.64 R-UR (opc 0xc35) in .text.s2_fail
  - insert N extra NOP instructions (16 bytes each) IMMEDIATELY BEFORE it
  - grow the .text section size, shift any following section offsets, and
    patch the BRA trap-loop offset (it's self-relative but we keep it
    after IADD.64 so relative offsets from BRA still reach the preceding
    EXIT correctly)

Run each variant single-thread (block(1,1,1)) to rule out any warp
divergence masking, and compare sync + output value.
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


def _nop_bytes() -> bytes:
    # Standard NOP instruction (observed in existing cubins).
    return bytes.fromhex('18790000000000000000000000c00f00')


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


def build_variant(num_extra_nops: int) -> bytes:
    """In-place shift: move [IADD.64, NOP, STG.E, EXIT, BRA] down by N
    and replace the freed slots before IADD.64 with NOPs.  Eats trailing
    padding NOPs so the total .text section size is unchanged (no ELF
    structure changes required)."""
    cubin = bytearray(compile_function(parse(_PTX).functions[0],
                                        verbose=False, sm_version=120))
    if num_extra_nops == 0:
        return bytes(cubin)

    sec_i, sec_off, sec_sz = _find_text_section(cubin, 's2_fail')
    text = bytearray(cubin[sec_off:sec_off + sec_sz])

    # Find IADD.64 R-UR (opc 0xc35) — b0=0x35, b1=0x7c.
    iadd_idx = None
    for i in range(0, len(text), 16):
        if text[i] == 0x35 and text[i + 1] == 0x7c:
            iadd_idx = i // 16
            break
    if iadd_idx is None:
        raise AssertionError('no IADD.64 R-UR in s2_fail')

    # Decompose into 16-byte instructions.
    n = len(text) // 16
    instrs = [bytes(text[i * 16:(i + 1) * 16]) for i in range(n)]
    nop = _nop_bytes()

    # Trailing padding NOPs count (from end of text).
    trailing = 0
    for i in range(n - 1, -1, -1):
        if instrs[i] == nop:
            trailing += 1
        else:
            break
    if trailing < num_extra_nops:
        raise AssertionError(
            f'not enough trailing NOPs ({trailing}) to absorb '
            f'{num_extra_nops} insertions')

    # Shift [iadd_idx, n - trailing) down by num_extra_nops, replacing
    # [iadd_idx, iadd_idx + num_extra_nops) with NOPs.
    body = instrs[iadd_idx : n - trailing]
    new_instrs = (
        instrs[:iadd_idx]
        + [nop] * num_extra_nops
        + body
        + instrs[n - trailing + num_extra_nops:]
    )
    assert len(new_instrs) == n, \
        f'length mismatch: {len(new_instrs)} vs {n}'

    # Note: BRA trap-loop offset is self-relative (BRA $).  When we shift
    # the BRA by num_extra_nops, its PC moves by +num_extra_nops*16 but
    # its target-of-self moves by the same amount, so the encoded
    # PC-relative offset stays valid.
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
            print(f'[{label}] cuModuleLoadData={err}')
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
        print(f'[{label}] sync={err} out=0x{val:08x} expect=0x309 {ok}')
    finally:
        cuda.cuCtxDestroy_v2(ctx)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        cubin = build_variant(n)
        run(cubin, f'+{n}NOP')
    else:
        for n in (0, 1, 2, 3, 4):
            subprocess.run([sys.executable, __file__, str(n)])
