"""Test if LDC.64 at c[0x388] returns the right param.

Kernel: store p_out (via LDC.64) and p_in (via LDC.64) into output.
If LDC.64 works, we should see the actual pointer values at offsets 0..7.
"""
import sys, pathlib, ctypes
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from sass.pipeline import compile_ptx_source

# This PTX dumps [p_in, p_out, p_in, p_out] into buf[0..3]
ptx = """.version 9.0
.target sm_120
.address_size 64
.visible .entry dump(.param .u64 buf, .param .u64 p_a, .param .u64 p_b) {
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd0, [buf];
    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    cvt.u32.u64 %r0, %rd1;
    shr.u64 %rd3, %rd1, 32;
    cvt.u32.u64 %r1, %rd3;
    cvt.u32.u64 %r2, %rd2;
    shr.u64 %rd4, %rd2, 32;
    cvt.u32.u64 %r3, %rd4;
    st.global.u32 [%rd0], %r0;
    add.u64 %rd5, %rd0, 4;
    st.global.u32 [%rd5], %r1;
    add.u64 %rd6, %rd0, 8;
    st.global.u32 [%rd6], %r2;
    add.u64 %rd7, %rd0, 12;
    st.global.u32 [%rd7], %r3;
    ret;
}
"""

cub = compile_ptx_source(ptx)['dump']
cubin_path = '/tmp/_dump.cubin'
open(cubin_path, 'wb').write(cub)

cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
def ck(e):
    if e != 0:
        name = ctypes.c_char_p(); cuda.cuGetErrorName(e, ctypes.byref(name))
        raise RuntimeError(f'err={e} {name.value.decode() if name.value else ""}')

ck(cuda.cuInit(0))
dev = ctypes.c_int(); ck(cuda.cuDeviceGet(ctypes.byref(dev), 0))
cctx = ctypes.c_void_p(); ck(cuda.cuCtxCreate_v2(ctypes.byref(cctx), 0, dev))

mod = ctypes.c_void_p()
ck(cuda.cuModuleLoad(ctypes.byref(mod), cubin_path.encode()))
fn = ctypes.c_void_p()
ck(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b'dump'))

d_buf = ctypes.c_uint64()
ck(cuda.cuMemAlloc_v2(ctypes.byref(d_buf), 64))
ck(cuda.cuMemsetD8_v2(d_buf, 0, 64))
fake_a = 0x1111222233334444
fake_b = 0x5555666677778888

arg_buf = ctypes.c_uint64(d_buf.value)
arg_a = ctypes.c_uint64(fake_a)
arg_b = ctypes.c_uint64(fake_b)
args = (ctypes.c_void_p * 3)(
    ctypes.cast(ctypes.pointer(arg_buf), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(arg_a), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(arg_b), ctypes.c_void_p),
)
extra = ctypes.c_void_p(0)
err = cuda.cuLaunchKernel(fn, 1,1,1, 1,1,1, 0, 0, args, extra)
sync_err = cuda.cuCtxSynchronize()

host = (ctypes.c_uint32 * 4)()
ck(cuda.cuMemcpyDtoH_v2(host, d_buf, 16))
print(f"launch={err} sync={sync_err}")
print(f"Expected p_a_lo=0x33334444, p_a_hi=0x11112222, p_b_lo=0x77778888, p_b_hi=0x55556666")
print(f"Got     p_a_lo=0x{host[0]:08x}, p_a_hi=0x{host[1]:08x}, p_b_lo=0x{host[2]:08x}, p_b_hi=0x{host[3]:08x}")
