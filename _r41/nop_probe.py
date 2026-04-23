"""R41 probe: is the loop bug latency-sensitive? Insert NOPs between
ISETP and BRA in the compiled cubin and see if iteration count changes."""
import ctypes, struct, sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')
import workbench_expanded as we
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


def build_variant(num_nops_before_bra):
    cubin = bytearray(compile_function(parse(we._W1_LOOP_SUM).functions[0], verbose=False, sm_version=120))
    sec_off, sec_sz = find_text(cubin)
    text = bytearray(cubin[sec_off:sec_off+sec_sz])
    n = len(text) // 16
    instrs = [bytes(text[i*16:(i+1)*16]) for i in range(n)]

    # Find BRA (opc 0x947) — the loop back-branch, which comes BEFORE the
    # final trap-loop BRA.  Loop BRA has guard != 0x7 (predicated).
    bra_idx = None
    for i, ins in enumerate(instrs):
        opc = (ins[0] | (ins[1]<<8)) & 0xFFF
        guard = (ins[1] >> 4) & 0xF
        if opc == 0x947 and guard != 0x7:
            bra_idx = i
            break
    if bra_idx is None:
        raise AssertionError('no predicated BRA found')
    # Count trailing padding NOPs
    trailing = 0
    for i in range(n - 1, -1, -1):
        if instrs[i] == NOP:
            trailing += 1
        else:
            break
    if trailing < num_nops_before_bra:
        raise AssertionError(f'only {trailing} trailing NOPs, need {num_nops_before_bra}')
    # Shift [bra_idx, n-trailing) by num_nops, put NOPs before BRA.
    body = instrs[bra_idx : n - trailing]
    new_instrs = (
        instrs[:bra_idx]
        + [NOP] * num_nops_before_bra
        + body
        + instrs[n - trailing + num_nops_before_bra:]
    )
    assert len(new_instrs) == n
    # Patch BRA offset: moved back by num_nops, so offset decreases by num_nops
    # BRA at new position: bra_idx + num_nops_before_bra
    # Target: same LOOP position (not shifted — LOOP is before BRA)
    # Old BRA next_pc = bra_idx+1, target = bra_idx+1 + old_off_instrs
    # New BRA next_pc = bra_idx+num_nops+1, target unchanged
    # old_off_instrs = target - (bra_idx+1)
    # new_off_instrs = target - (bra_idx+num_nops+1) = old_off_instrs - num_nops
    if num_nops_before_bra > 0:
        bra_raw = bytearray(new_instrs[bra_idx + num_nops_before_bra])
        # Decode existing total: reverse of the encoding
        b2 = bra_raw[2]; b4 = bra_raw[4]; b10 = bra_raw[10]
        # total[7:0]=b2, total[13:8]=b4>>2 (6 bits), total[17:16]=b10&3 (2 bits)
        old_total = (b2) | ((b4 >> 2) & 0x3F) << 8 | (b10 & 3) << 16
        if old_total & (1 << 17):
            old_total -= (1 << 18)
        # new_total = old_total - num_nops * 4
        new_total = old_total - num_nops_before_bra * 4
        new_total_enc = new_total & 0x3FFFF
        bra_raw[2] = new_total_enc & 0xFF
        bra_raw[4] = ((new_total_enc >> 8) << 2) & 0xFF
        # se
        se = 0xFF if bra_raw[4] >= 0x80 else 0x00
        bra_raw[5] = se; bra_raw[6] = se; bra_raw[7] = se
        bra_raw[8] = se; bra_raw[9] = se
        bra_raw[10] = 0x80 | ((new_total_enc >> 16) & 0x03)
        new_instrs[bra_idx + num_nops_before_bra] = bytes(bra_raw)
    new_text = b''.join(new_instrs)
    new_cubin = bytearray(cubin)
    new_cubin[sec_off:sec_off+sec_sz] = new_text
    return bytes(new_cubin)


def run_one(cubin, label):
    cuda = ctypes.WinDLL('nvcuda'); cuda.cuInit(0)
    dev = ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p(); cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    try:
        mod = ctypes.c_void_p(); err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
        if err:
            print(f'[{label}] load_err={err}'); return
        func = ctypes.c_void_p(); cuda.cuModuleGetFunction(ctypes.byref(func), mod, b'w1_loop_sum')
        N = 8
        d_out = ctypes.c_uint64(); cuda.cuMemAlloc_v2(ctypes.byref(d_out), N*4)
        cuda.cuMemcpyHtoD_v2(d_out, b'\0'*N*4, N*4)
        a_out = ctypes.c_uint64(d_out.value); a_n = ctypes.c_uint32(N)
        argv = (ctypes.c_void_p*2)(ctypes.cast(ctypes.byref(a_out), ctypes.c_void_p),
                                    ctypes.cast(ctypes.byref(a_n), ctypes.c_void_p))
        cuda.cuLaunchKernel(func, 1,1,1, N,1,1, 0, None, argv, None)
        err = cuda.cuCtxSynchronize()
        buf = ctypes.create_string_buffer(N*4); cuda.cuMemcpyDtoH_v2(buf, d_out, N*4)
        vals = struct.unpack(f'<{N}I', buf.raw)
        print(f'[{label}] sync={err} vals={vals}  (expect t*8: {tuple(t*8 for t in range(N))})')
    finally:
        cuda.cuCtxDestroy_v2(ctx)


if __name__ == '__main__':
    import subprocess
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        cubin = build_variant(n)
        run_one(cubin, f'+{n}NOP')
    else:
        for n in (0, 1, 2, 3, 4):
            subprocess.run([sys.executable, __file__, str(n)])
