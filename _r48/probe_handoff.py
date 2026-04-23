"""R48: probe ISETP -> @P1 LOP3 predicate handoff in k300_nasty_pred_xor.

Apply the Bug-1 fix (FG56 LOP3 b3 rename) first, then probe ctrl-word
variants on the @P1 LOP3 to see which combination lets tids 17-31 pass.
"""
import ctypes, struct, sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')
import workbench_expanded as we
from ptx.parser import parse
from sass.pipeline import compile_function


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


def apply_bug1_rename(cubin):
    """Apply FG56 LOP3.LUT(0x812) src0 rename R3->R0 for instrs whose b3==3."""
    sec_off, sec_sz = find_text(cubin)
    text = bytearray(cubin[sec_off:sec_off+sec_sz])
    n = len(text)//16
    for i in range(n):
        b = text[i*16:(i+1)*16]
        opc = (b[0] | (b[1]<<8)) & 0xFFF
        if opc == 0x812 and b[3] == 3:
            text[i*16+3] = 0
    new_cubin = bytearray(cubin)
    new_cubin[sec_off:sec_off+sec_sz] = bytes(text)
    return bytes(new_cubin)


def locate_pred_lop3(text):
    """Find the @P1 LOP3.LUT (guard=1, opc=0x812). Return idx."""
    n = len(text)//16
    for i in range(n):
        b = text[i*16:(i+1)*16]
        opc = (b[0] | (b[1]<<8)) & 0xFFF
        guard = (b[1] >> 4) & 0xF
        if opc == 0x812 and guard == 1:  # @P1
            return i
    return None


def locate_isetp_pred(text):
    """Find the ISETP.IMM that writes P1 (opc=0x80c, pred_dest at b10 bits[3:1]==1)."""
    n = len(text)//16
    hits = []
    for i in range(n):
        b = text[i*16:(i+1)*16]
        opc = (b[0] | (b[1]<<8)) & 0xFFF
        if opc == 0x80c:
            pred_dst = (b[10] >> 1) & 0x7
            hits.append((i, pred_dst, b[10]))
    return hits


def patch_ctrl(text, idx, *, wdep=None, rbar=None, stall=None):
    """Patch the ctrl hi bytes (13, 14, 15) of instr idx.
    ctrl24 = (b15<<16)|(b14<<8)|b13; ctrl = ctrl24 >> 1.
    wdep is ctrl bits[9:4], rbar is ctrl bits[14:10], stall is bits[22:17].
    """
    b = bytearray(text[idx*16:(idx+1)*16])
    ctrl24 = (b[15] << 16) | (b[14] << 8) | b[13]
    ctrl = ctrl24 >> 1
    if wdep is not None:
        ctrl = (ctrl & ~(0x3f << 4)) | ((wdep & 0x3f) << 4)
    if rbar is not None:
        ctrl = (ctrl & ~(0x1f << 10)) | ((rbar & 0x1f) << 10)
    if stall is not None:
        ctrl = (ctrl & ~(0x3f << 17)) | ((stall & 0x3f) << 17)
    new_ctrl24 = (ctrl << 1) | (ctrl24 & 1)
    b[13] = new_ctrl24 & 0xFF
    b[14] = (new_ctrl24 >> 8) & 0xFF
    b[15] = (new_ctrl24 >> 16) & 0xFF
    text[idx*16:(idx+1)*16] = bytes(b)


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
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, b'k300_nasty_pred_xor')
        N = 32
        d_out = ctypes.c_uint64(); cuda.cuMemAlloc_v2(ctypes.byref(d_out), N*4)
        cuda.cuMemcpyHtoD_v2(d_out, b'\x00'*N*4, N*4)
        a_out = ctypes.c_uint64(d_out.value); a_n = ctypes.c_uint32(N)
        argv = (ctypes.c_void_p*2)(
            ctypes.cast(ctypes.byref(a_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(a_n), ctypes.c_void_p))
        cuda.cuLaunchKernel(func, 1,1,1, N,1,1, 0, None, argv, None)
        cuda.cuCtxSynchronize()
        buf = ctypes.create_string_buffer(N*4); cuda.cuMemcpyDtoH_v2(buf, d_out, N*4)
        vals = struct.unpack(f'<{N}I', buf.raw)
        wrong_lo = sum(1 for t in range(17) if vals[t] != (t ^ 0xAA))
        wrong_hi = sum(1 for t in range(17, N) if vals[t] != ((t ^ 0xAA) ^ 0x55))
        print(f'[{label}] lo_wrong={wrong_lo}/17  hi_wrong={wrong_hi}/15')
    finally:
        cuda.cuCtxDestroy_v2(ctx)


def build_variant(variant):
    base = compile_function(parse(we._K300_NASTY_PRED_XOR).functions[0], verbose=False, sm_version=120)
    fixed = apply_bug1_rename(base)
    sec_off, sec_sz = find_text(fixed)
    text = bytearray(fixed[sec_off:sec_off+sec_sz])

    isetps = locate_isetp_pred(text)
    p1_idx = locate_pred_lop3(text)
    if p1_idx is None:
        print('no @P1 LOP3 found'); return None
    # find the P1-producing ISETP (pred_dst==1)
    isetp_p1 = [i for i,pd,_ in isetps if pd == 1]
    isetp_p1_idx = isetp_p1[0] if isetp_p1 else None

    if variant == 'baseline':
        pass  # no ctrl changes
    elif variant == 'consumer_wdep_3e':
        # Set @P1 LOP3 wdep: 0x3f -> 0x3e (match ptxas: wait slot 0)
        patch_ctrl(text, p1_idx, wdep=0x3e)
    elif variant == 'consumer_wdep_0':
        # Wait everything
        patch_ctrl(text, p1_idx, wdep=0x00)
    elif variant == 'consumer_rbar_2':
        # Consumer wait rbar 2 (producer-set barrier)
        patch_ctrl(text, p1_idx, rbar=0x02)
    elif variant == 'consumer_stall_8':
        # Extra stall
        patch_ctrl(text, p1_idx, stall=0x08)
    elif variant == 'producer_rbar_2':
        # Producer ISETP: set rbar=2 so consumer can wait on it
        if isetp_p1_idx is not None:
            patch_ctrl(text, isetp_p1_idx, rbar=0x02)
    elif variant == 'producer_wdep_3e':
        # Producer ISETP wdep 0x3e
        if isetp_p1_idx is not None:
            patch_ctrl(text, isetp_p1_idx, wdep=0x3e)

    new_cubin = bytearray(fixed)
    new_cubin[sec_off:sec_off+sec_sz] = bytes(text)
    return bytes(new_cubin)


if __name__ == '__main__':
    import subprocess
    VARIANTS = [
        'baseline',          # bug1 fix only, no ctrl change
        'consumer_wdep_3e',  # ptxas match
        'consumer_wdep_0',   # wait all
        'consumer_rbar_2',   # wait rbar 2 (needs producer setter)
        'consumer_stall_8',  # extra stall
        'producer_rbar_2',   # producer sets rbar 2
        'producer_wdep_3e',  # producer waits slot 0
    ]
    if len(sys.argv) > 1:
        v = sys.argv[1]
        cubin = build_variant(v)
        if cubin: run(cubin, v)
    else:
        for v in VARIANTS:
            subprocess.run([sys.executable, __file__, v])
