"""R43 BSSY probe: manually insert BSSY.RECONVERGENT before the loop in
an OpenPTXas-compiled w1_loop_sum cubin and see if the loop now iterates."""
import ctypes, struct, sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')
import workbench_expanded as we
from ptx.parser import parse
from sass.pipeline import compile_function


NOP = bytes.fromhex('18790000000000000000000000c00f00')


# From ptxas ground truth:
# BSSY.RECONVERGENT B0, `(.L_x_0) ;
#   0x0000011000007945
#   0x000fe60003800200
# Bytes: 45 79 00 00 11 01 00 00 00 00 00 02 00 cc 1f 00
# Actually let me reconstruct from the ptxas output:
# "0x0000011000007945"  => little-endian = 45 79 00 00 11 01 00 00
# "0x000fe60003800200"  => little-endian = 00 02 80 03 00 e6 0f 00
# So BSSY with offset=0x111 bytes = 17 instructions forward from next PC.
#
# For our probe we want BSSY with offset = (post_loop_label_byte - bssy_next_pc).
# In w1_loop_sum SASS:
#   * LOOP is at pos 9 (byte 144)
#   * BRA is at pos 13 (byte 208), going back to LOOP
#   * post-loop continues at pos 14+ (byte 224+)
# We want BSSY before LOOP, with offset to the post-loop reconverge point.
#
# Simplest placement: insert BSSY at position 8 (just before LOOP), with
# target = pos 14 (just after BRA, post-loop region begins).


def build_template_bssy(offset_bytes: int) -> bytes:
    """Build a BSSY with given byte offset to target."""
    raw = bytearray(16)
    raw[0] = 0x45
    raw[1] = 0x79  # opc=0x945, guard=PT (unconditional)
    # Offset in b4-b7 (signed 32-bit from ptxas ground truth)
    off = offset_bytes & 0xFFFFFFFF
    raw[4] = off & 0xFF
    raw[5] = (off >> 8) & 0xFF
    raw[6] = (off >> 16) & 0xFF
    raw[7] = (off >> 24) & 0xFF
    raw[11] = 0x02  # fixed modifier
    # Ctrl bytes 13-15 from ptxas: 00 e6 0f 00 -> bytes 13,14,15 = 00, e6, 0f ?
    # Actually the ptxas hi word is 0x000fe60003800200, little-endian bytes:
    # byte 8 = 0x00, 9=0x02, 10=0x80, 11=0x03, 12=0x00, 13=0xe6, 14=0x0f, 15=0x00
    raw[8] = 0x00; raw[9] = 0x02; raw[10] = 0x80; raw[11] = 0x03
    raw[12] = 0x00; raw[13] = 0xe6; raw[14] = 0x0f; raw[15] = 0x00
    return bytes(raw)


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


BSYNC = bytes.fromhex('41790000000000000002800300ea0f00')


def build(with_bsync: bool):
    """Insert BSSY before LOOP; optionally insert BSYNC right after the
    back-edge BRA (at the reconverge point .L_x_1)."""
    cubin = bytearray(compile_function(parse(we._W1_LOOP_SUM).functions[0], verbose=False, sm_version=120))
    sec_off, sec_sz = find_text(cubin)
    text = bytearray(cubin[sec_off:sec_off+sec_sz])
    n = len(text) // 16
    instrs = [bytes(text[i*16:(i+1)*16]) for i in range(n)]

    loop_idx = None
    for i, ins in enumerate(instrs):
        opc = (ins[0] | (ins[1]<<8)) & 0xFFF
        if opc == 0x210 and ins[2] == 0x04 and ins[3] == 0x04:
            loop_idx = i
            break
    if loop_idx is None:
        raise AssertionError('no LOOP IADD3 R4,R4,... found')

    bra_idx = None
    for i in range(loop_idx + 1, n):
        opc = (instrs[i][0] | (instrs[i][1]<<8)) & 0xFFF
        guard = (instrs[i][1]>>4) & 0xF
        if opc == 0x947 and guard != 0x7:
            bra_idx = i
            break
    if bra_idx is None:
        raise AssertionError('no back-edge BRA found')

    trailing = 0
    for i in range(n - 1, -1, -1):
        if instrs[i] == NOP:
            trailing += 1
        else:
            break
    need = 2 if with_bsync else 1
    if trailing < need:
        raise AssertionError(f'only {trailing} trailing NOPs, need {need}')

    # Insert BSSY at loop_idx; optionally BSYNC at bra_idx+1 (after shift +1 for BSSY).
    if with_bsync:
        # After both insertions, layout is:
        #   old[:loop_idx]  ; BSSY ; old[loop_idx:bra_idx+1] ; BSYNC ; old[bra_idx+1:n-trailing] ; old[n-trailing+2:]
        # Total shift for code after BSSY = +1 before bra_idx+1, +2 after that.
        # BSSY at new_idx = loop_idx.  Target = post-BSYNC position.
        #   LOOP new pos = loop_idx + 1
        #   BRA new pos = bra_idx + 1
        #   BSYNC new pos = bra_idx + 2
        #   post-BSYNC new pos = bra_idx + 3
        bssy_next_pc = (loop_idx + 1) * 16
        target_byte = (bra_idx + 3) * 16  # skip BSYNC to post-BSYNC
        bssy_raw = build_template_bssy(target_byte - bssy_next_pc)
        new_instrs = (
            instrs[:loop_idx]
            + [bssy_raw]
            + instrs[loop_idx : bra_idx + 1]
            + [BSYNC]
            + instrs[bra_idx + 1 : n - trailing]
            + instrs[n - trailing + 2:]
        )
    else:
        new_bra_idx = bra_idx + 1
        new_post_loop_idx = bra_idx + 2
        bssy_next_pc = (loop_idx + 1) * 16
        target_byte = new_post_loop_idx * 16
        bssy_raw = build_template_bssy(target_byte - bssy_next_pc)
        new_instrs = (
            instrs[:loop_idx]
            + [bssy_raw]
            + instrs[loop_idx : n - trailing]
            + instrs[n - trailing + 1:]
        )
    assert len(new_instrs) == n
    new_text = b''.join(new_instrs)
    new_cubin = bytearray(cubin)
    new_cubin[sec_off:sec_off+sec_sz] = new_text
    return bytes(new_cubin)


def run(cubin, label):
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
        expected = tuple(t*8 for t in range(N))
        print(f'[{label}] sync={err} vals={vals} expect={expected}  '
              f'{"PASS" if vals==expected else "FAIL"}')
    finally:
        cuda.cuCtxDestroy_v2(ctx)


if __name__ == '__main__':
    import subprocess
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'bssy_only':
            cubin = build(with_bsync=False)
        elif mode == 'bssy_bsync':
            cubin = build(with_bsync=True)
        else:
            raise ValueError(mode)
        run(cubin, mode)
    else:
        for m in ('bssy_only', 'bssy_bsync'):
            subprocess.run([sys.executable, __file__, m])
