"""
Compare ptxas vs OpenPTXas vector_add instruction sequences.
"""
import sys, os, struct, ctypes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass.pipeline import compile_ptx_source

_PTX = """
.version 9.0
.target sm_120
.address_size 64

.visible .entry vector_add(
    .param .u64 out, .param .u64 a, .param .u64 b, .param .u32 n)
{
    .reg .b32 %r<8>; .reg .b64 %rd<8>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.s32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;
    cvt.u64.u32 %rd0, %r3;
    shl.b64 %rd0, %rd0, 2;
    ld.param.u64 %rd1, [a]; add.u64 %rd2, %rd1, %rd0;
    ld.param.u64 %rd3, [b]; add.u64 %rd4, %rd3, %rd0;
    ld.global.u32 %r5, [%rd2];
    ld.global.u32 %r6, [%rd4];
    add.s32 %r7, %r5, %r6;
    ld.param.u64 %rd5, [out]; add.u64 %rd6, %rd5, %rd0;
    st.global.u32 [%rd6], %r7;
DONE:
    ret;
}
"""

PTXAS_CUBIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'probe_work', 'vector_add_ptxas.cubin')


def decode_ctrl(raw16: bytes) -> dict:
    b13 = raw16[13]; b14 = raw16[14]; b15 = raw16[15]
    raw24 = ((b15 & ~0x04) << 16) | (b14 << 8) | b13
    ctrl = raw24 >> 1
    stall = (ctrl >> 17) & 0x3f
    rbar  = (ctrl >> 10) & 0x1f
    wdep  = (ctrl >>  4) & 0x3f
    misc  = ctrl & 0xf
    return {'stall': stall, 'rbar': rbar, 'wdep': wdep, 'misc': misc}


def get_text(cubin: bytes, kernel: str) -> bytes:
    e_shoff    = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shentsize= struct.unpack_from('<H', cubin, 0x3a)[0]
    e_shnum    = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]
    sh = e_shoff + e_shstrndx * e_shentsize
    shstr_off  = struct.unpack_from('<Q', cubin, sh + 24)[0]
    shstr_size = struct.unpack_from('<Q', cubin, sh + 32)[0]
    shstr = cubin[shstr_off:shstr_off + shstr_size]
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        name_off = struct.unpack_from('<I', cubin, sh)[0]
        end = shstr.index(b'\x00', name_off)
        name = shstr[name_off:end].decode()
        if name == f'.text.{kernel}':
            off  = struct.unpack_from('<Q', cubin, sh + 24)[0]
            size = struct.unpack_from('<Q', cubin, sh + 32)[0]
            return cubin[off:off+size]
    raise RuntimeError(f'.text.{kernel} not found')


def show_text(label: str, text: bytes):
    n = len(text) // 16
    print(f"\n=== {label} ({n} instructions) ===")
    print(f"{'idx':>3}  {'opc':>7}  stall  rbar    wdep  misc  raw_bytes")
    print('-' * 90)
    for i in range(n):
        raw = text[i*16:(i+1)*16]
        opcode = struct.unpack_from('<Q', raw)[0] & 0xFFF
        ctrl = decode_ctrl(raw)
        raw_hex = raw.hex()
        stall = ctrl['stall']; rbar = ctrl['rbar']; wdep = ctrl['wdep']; misc = ctrl['misc']
        print(f"[{i:2d}] 0x{opcode:03x}  stall={stall:2d}  rbar=0x{rbar:02x}  "
              f"wdep=0x{wdep:02x}  misc=0x{misc:x}  {raw_hex}")


# Load ptxas cubin
with open(PTXAS_CUBIN, 'rb') as f:
    ptxas_data = f.read()
ptxas_text = get_text(ptxas_data, 'vector_add')

# Compile with OpenPTXas
our_cubins = compile_ptx_source(_PTX)
our_data = our_cubins['vector_add']
our_text = get_text(our_data, 'vector_add')

show_text("ptxas", ptxas_text)
show_text("OpenPTXas", our_text)

# Diff: find first diverging instruction
n_ptxas = len(ptxas_text) // 16
n_our   = len(our_text) // 16
print(f"\n=== Instruction count: ptxas={n_ptxas}, ours={n_our} ===")

# Also run our cubin on GPU
print("\n=== GPU execution test ===")
try:
    cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
    if cuda.cuInit(0) != 0:
        print("CUDA init failed")
    else:
        dev = ctypes.c_int()
        cuda.cuDeviceGet(ctypes.byref(dev), 0)
        ctx = ctypes.c_void_p()
        cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

        N = 32
        a = list(range(1, N+1))
        b = list(range(100, 100+N))

        def run_cubin(label, cubin_bytes):
            mod = ctypes.c_void_p()
            err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin_bytes)
            if err != 0:
                name = ctypes.c_char_p()
                cuda.cuGetErrorName(err, ctypes.byref(name))
                print(f"  {label}: LOAD_FAIL:{name.value.decode()}")
                return

            func = ctypes.c_void_p()
            cuda.cuModuleGetFunction(ctypes.byref(func), mod, b'vector_add')

            d_a = ctypes.c_uint64(); d_b = ctypes.c_uint64(); d_out = ctypes.c_uint64()
            cuda.cuMemAlloc_v2(ctypes.byref(d_a), N*4)
            cuda.cuMemAlloc_v2(ctypes.byref(d_b), N*4)
            cuda.cuMemAlloc_v2(ctypes.byref(d_out), N*4)

            import struct as st
            cuda.cuMemcpyHtoD_v2(d_a, st.pack(f'<{N}i', *a), N*4)
            cuda.cuMemcpyHtoD_v2(d_b, st.pack(f'<{N}i', *b), N*4)
            cuda.cuMemcpyHtoD_v2(d_out, st.pack(f'<{N}i', *([0]*N)), N*4)

            args = [d_out.value, d_a.value, d_b.value, N]
            arg_holders = []
            ptrs = []
            for av in args:
                if av > 0xFFFFFFFF:
                    h = ctypes.c_uint64(av)
                else:
                    h = ctypes.c_int32(av)
                arg_holders.append(h)
                ptrs.append(ctypes.cast(ctypes.byref(h), ctypes.c_void_p))
            args_arr = (ctypes.c_void_p * 4)(*ptrs)

            err = cuda.cuLaunchKernel(func, 1,1,1, N,1,1, 0, None, args_arr, None)
            if err != 0:
                name = ctypes.c_char_p()
                cuda.cuGetErrorName(err, ctypes.byref(name))
                print(f"  {label}: LAUNCH_FAIL:{name.value.decode()}")
                cuda.cuMemFree_v2(d_a); cuda.cuMemFree_v2(d_b); cuda.cuMemFree_v2(d_out)
                cuda.cuModuleUnload(mod)
                return

            err = cuda.cuCtxSynchronize()
            if err != 0:
                name = ctypes.c_char_p()
                cuda.cuGetErrorName(err, ctypes.byref(name))
                print(f"  {label}: CRASH:{name.value.decode()}")
                cuda.cuMemFree_v2(d_a); cuda.cuMemFree_v2(d_b); cuda.cuMemFree_v2(d_out)
                cuda.cuModuleUnload(mod)
                return

            buf = (ctypes.c_uint8 * (N*4))()
            cuda.cuMemcpyDtoH_v2(buf, d_out, N*4)
            results = list(st.unpack(f'<{N}i', bytes(buf)))
            expected = [a[i]+b[i] for i in range(N)]
            if results == expected:
                print(f"  {label}: PASS")
            else:
                mismatches = [i for i in range(N) if results[i] != expected[i]]
                print(f"  {label}: WRONG at idx {mismatches[0]}: got {results[mismatches[0]]}, expected {expected[mismatches[0]]}")
                print(f"    first 8 results: {results[:8]}")

            cuda.cuMemFree_v2(d_a); cuda.cuMemFree_v2(d_b); cuda.cuMemFree_v2(d_out)
            cuda.cuModuleUnload(mod)

        run_cubin("ptxas", ptxas_data)
        run_cubin("OpenPTXas", our_data)

        cuda.cuCtxDestroy_v2(ctx)
except Exception as e:
    print(f"GPU test failed: {e}")
