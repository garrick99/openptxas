"""Probe relu by patching the raw .text bytes and fixing up the BRA offset."""
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


def patch(cubin, insert_at_instr):
    """Insert a NOP at the given instruction index, preserving text size by
    eating one trailing NOP.  Also fix self-loop BRA offset."""
    sec_off, sec_sz = find_text(cubin)
    text = bytearray(cubin[sec_off:sec_off+sec_sz])
    n = len(text)//16
    instrs = [bytes(text[i*16:(i+1)*16]) for i in range(n)]
    # count trailing NOPs
    trailing = 0
    for i in range(n-1, -1, -1):
        if instrs[i] == NOP: trailing += 1
        else: break
    if trailing < 1:
        return None
    # Build: insert NOP, drop one trailing NOP. Same n.
    new_instrs = instrs[:insert_at_instr] + [NOP] + instrs[insert_at_instr:n-1]
    assert len(new_instrs) == n
    # Fix up self-loop BRA — find opcode 0x947 (BRA). If target == self, adjust nothing
    # (both shifted equally). If target crosses, adjust offset_instrs.
    for i, ins in enumerate(new_instrs):
        opc = (ins[0] | (ins[1] << 8)) & 0xFFF
        if opc == 0x947:
            # find target by decoding offset
            b = bytearray(ins)
            # Reconstruct signed 18-bit "total" field per encoding
            total = b[2] | (((b[4] >> 2) & 0x3F) << 8) | ((b[10] & 0x03) << 16)
            if b[10] & 0x02:
                total |= 0xC000
                total -= (1 << 18)
            offset_bytes = total  # in "quarter-bytes" — total is offset_instrs * 4 in units where 1 instr = 4
            # Actually: total units = bytes_offset / 4. So offset_bytes = total * 4.
            next_pc = (i + 1) * 16
            target_abs = next_pc + total * 4
            target_instr = target_abs // 16
            # Did insertion cross the (i, target_instr) range?
            # original: BRA at old_pos = (i if i < insert else i-1), target at old_target = (target_instr if target_instr < insert else target_instr-1)
            # Self-loop case: target_instr == old_pos of BRA. Both shifted equally → no change.
            # If BRA at i and target at target_instr are both >= insert_at: both shift +1 equally.
            # If neither shift: total unchanged.
            # If one shifts and not other: need adjustment.
            old_i = i if i < insert_at_instr else i - 1
            old_target = target_instr if target_instr < insert_at_instr else target_instr - 1
            old_total = ((old_target - (old_i + 1)) * 16) // 4
            # New total computed above — if equal, no change needed
            # Otherwise re-encode.  For self-loop BRA this is always equal.
            pass
    new_text = b''.join(new_instrs)
    new_cubin = bytearray(cubin)
    new_cubin[sec_off:sec_off+sec_sz] = new_text
    return bytes(new_cubin)


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
            wrong = sum(1 for i in range(N) if abs(vals[i] - max(-1.0 if i&1 else 1.0, 0.0) * (i+1)) > 1e-6)
            print(f'[{label}] sync=0 wrong={wrong}/{N} first4={[round(v,2) for v in vals[:4]]}')
        else:
            print(f'[{label}] sync_err={err}')
    finally:
        cuda.cuCtxDestroy_v2(ctx)


if __name__ == '__main__':
    base = compile_function(parse(bm.RELU_PTX).functions[0], verbose=False, sm_version=120)
    run(base, 'baseline')
    # Instruction positions (16-byte indices):
    # 0:LDC, 1:S2R, 2-3:LDCUs, 4:S2UR, 5:LDC R5, 6:IMAD, 7:LDCU, 8:ISETP, 9:NOP,
    # 10:@P0 EXIT, 11:IMAD.SHL, 12:SHF, 13:NOP, 14:IADD.64, 15:NOP, 16:LDG,
    # 17:FMNMX, 18:LDC.64, 19:IADD3, 20:IADD3.X, 21:NOP, 22:STG, 23:EXIT
    for pos in [20, 16, 15, 14, 22, 19]:
        patched = patch(base, pos)
        if patched:
            run(patched, f'nop_before_pos_{pos}')
