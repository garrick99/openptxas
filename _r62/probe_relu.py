"""Probe the relu crash by inserting NOPs between instructions."""
import sys, ctypes, struct
sys.path.insert(0, 'C:/Users/kraken/openptxas')
sys.path.insert(0, 'C:/Users/kraken/openptxas/benchmarks')
import relu_vs_nvidia as bm
from ptx.parser import parse
from sass.pipeline import compile_function

NOP = bytes.fromhex('18790000000000000000000000c00f00')


def find_text(cubin):
    e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]
    def sh(i): return struct.unpack_from('<IIQQQQIIQQ', cubin, e_shoff + i*64)
    _, _, _, _, so, ss, *_ = sh(e_shstrndx)
    shs = cubin[so:so+ss]
    for i in range(e_shnum):
        nm, ty, _, _, off, sz, *_ = sh(i)
        end = shs.index(b'\0', nm)
        if shs[nm:end].startswith(b'.text.') and ty == 1:
            return off, sz
    return None, None


def run(cubin, label):
    cuda = ctypes.WinDLL('nvcuda'); cuda.cuInit(0)
    dev = ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p(); cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    try:
        mod = ctypes.c_void_p()
        err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
        if err:
            print(f'[{label}] load_err={err}'); return
        func = ctypes.c_void_p()
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, b'relu')
        N = 64; nbytes = N*4
        d_x = ctypes.c_uint64(); cuda.cuMemAlloc_v2(ctypes.byref(d_x), nbytes)
        d_out = ctypes.c_uint64(); cuda.cuMemAlloc_v2(ctypes.byref(d_out), nbytes)
        x_data = struct.pack(f'<{N}f', *[(-1.0 if i&1 else 1.0) * (i+1) for i in range(N)])
        cuda.cuMemcpyHtoD_v2(d_x, x_data, nbytes)
        cuda.cuMemsetD8_v2(d_out, 0, nbytes)
        a_x = ctypes.c_uint64(d_x.value); a_out = ctypes.c_uint64(d_out.value); a_n = ctypes.c_uint32(N)
        argv = (ctypes.c_void_p*3)(
            ctypes.cast(ctypes.byref(a_x), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(a_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(a_n), ctypes.c_void_p))
        cuda.cuLaunchKernel(func, 1,1,1, N,1,1, 0, None, argv, None)
        err = cuda.cuCtxSynchronize()
        if err == 0:
            buf = ctypes.create_string_buffer(nbytes)
            cuda.cuMemcpyDtoH_v2(buf, d_out, nbytes)
            vals = struct.unpack(f'<{N}f', buf.raw)
            expected = [max(v, 0.0) for v in struct.unpack(f'<{N}f', x_data)]
            wrong = sum(1 for i in range(N) if abs(vals[i] - expected[i]) > 1e-6)
            print(f'[{label}] sync={err}  {wrong}/{N} wrong  first8={[round(v,2) for v in vals[:8]]}')
        else:
            print(f'[{label}] sync_err={err}')
    finally:
        cuda.cuCtxDestroy_v2(ctx)


def patch_insert_nop(cubin, insert_at_pos):
    """Insert NOP at instruction position `insert_at_pos`, eating one trailing NOP."""
    sec_off, sec_sz = find_text(cubin)
    text = bytearray(cubin[sec_off:sec_off+sec_sz])
    n = len(text)//16
    instrs = [bytes(text[i*16:(i+1)*16]) for i in range(n)]
    # find trailing NOPs to eat
    trailing = 0
    for i in range(n-1, -1, -1):
        if instrs[i] == NOP: trailing += 1
        else: break
    if trailing < 1:
        return None
    # Build new list: insert NOP at insert_at_pos, drop one trailing NOP
    new_instrs = (
        instrs[:insert_at_pos]
        + [NOP]
        + instrs[insert_at_pos : n - 1]
    )
    # Patch BRA offsets that cross insert_at_pos
    for i, ins in enumerate(new_instrs):
        opc = (ins[0] | (ins[1] << 8)) & 0xFFF
        if opc == 0x947:
            # Decode 18-bit signed offset (total_bytes) from b2, b4, b10
            total = ins[2] | (((ins[4] >> 2) & 0x3F) << 8) | ((ins[10] & 0x03) << 16)
            if ins[10] & 0x02: total |= 0xC000; total -= (1 << 18)
            offset_instrs = total // 16  # each instr is 16B? No, total is bytes/4 actually
            # Per pipeline.py: total = offset_instrs * 4
            offset_instrs = total // 4
            # next_pc for BRA at position i = (i+1)
            # target = next_pc + offset_instrs
            next_pc = i + 1
            target = next_pc + offset_instrs
            # adjust if either endpoint crosses insert
            # for original kernel: BRA at pos N-2 targets itself (.L_x_0:  BRA `(.L_x_0)).
            # Both pos and target shifted equally if both >= insert_at_pos — no change.
            # Only need adjustment if one side crosses.
            # For self-BRA at end: target = self, both shifted. No change.
            # So keep as-is.
            pass
    assert len(new_instrs) == n
    new_text = b''.join(new_instrs)
    new_cubin = bytearray(cubin)
    new_cubin[sec_off:sec_off+sec_sz] = new_text
    return bytes(new_cubin)


if __name__ == '__main__':
    base = compile_function(parse(bm.RELU_PTX).functions[0], verbose=False, sm_version=120)
    run(base, 'baseline')
    # Try inserting NOP at various positions (by instr index)
    # Key instrs in OURS:
    #  pos  7 = ISETP (0x080 / 16 = 8, so pos 8)
    #  pos  9 = NOP (R48 gap)
    #  pos 10 = @P0 EXIT
    #  pos 11 = IMAD.SHL (0xb0)
    #  pos 12 = SHF.R.U32.HI (0xc0)
    #  pos 13 = NOP
    #  pos 14 = IADD.64 (0xe0) - address compute
    #  pos 15 = NOP
    #  pos 16 = LDG (0x100)
    #  pos 17 = FMNMX (0x110)
    #  pos 18 = LDC.64 (0x120) - p_out
    #  pos 19 = IADD3 (0x130)
    #  pos 20 = IADD3.X (0x140)
    #  pos 21 = NOP
    #  pos 22 = STG (0x160)
    for pos in [14, 15, 16, 17, 18, 19, 20, 21, 22]:
        patched = patch_insert_nop(base, pos)
        if patched:
            run(patched, f'nop_at_pos_{pos}')
