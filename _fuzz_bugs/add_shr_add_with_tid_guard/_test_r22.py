"""Test the R22 test kernel on GPU with the WB-8 exemption.
R22 was added to prevent an alleged CUDA_ERROR_ILLEGAL_ADDRESS from IADD.64 R-UR
on misaligned u64 param.  With my exemption, UR-bound p_out is now used for this
shape — so let's see if ILLEGAL_ADDRESS actually fires."""
import sys, pathlib, ctypes
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from sass.pipeline import compile_ptx_source

PTX = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k_misaligned_addr_u64(.param .u64 in, .param .u64 out)
{
    .reg .b32 %r<5>;
    .reg .b64 %rd<5>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [in];
    ld.param.u64 %rd1, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r3, %r1, %r2;
    add.u32 %r4, %r3, %r0;
    cvt.u64.u32 %rd2, %r4;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.u32 %r0, [%rd4];
    setp.eq.u32 %p0, %r0, 0;
    @!%p0 ret;
    add.u64 %rd0, %rd1, %rd3;
    st.global.u32 [%rd0], %r0;
    ret;
}
"""

cub = compile_ptx_source(PTX)['k_misaligned_addr_u64']
open('/tmp/_r22.cubin', 'wb').write(cub)

cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
def ck(e):
    if e != 0:
        name = ctypes.c_char_p(); cuda.cuGetErrorName(e, ctypes.byref(name))
        desc = ctypes.c_char_p(); cuda.cuGetErrorString(e, ctypes.byref(desc))
        raise RuntimeError(f'err={e} {name.value.decode()} {desc.value.decode()}')

ck(cuda.cuInit(0))
dev = ctypes.c_int(); ck(cuda.cuDeviceGet(ctypes.byref(dev), 0))
cctx = ctypes.c_void_p(); ck(cuda.cuCtxCreate_v2(ctypes.byref(cctx), 0, dev))

mod = ctypes.c_void_p()
ck(cuda.cuModuleLoad(ctypes.byref(mod), b'/tmp/_r22.cubin'))
fn = ctypes.c_void_p()
ck(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b'k_misaligned_addr_u64'))

N = 32
d_in = ctypes.c_uint64(); d_out = ctypes.c_uint64()
ck(cuda.cuMemAlloc_v2(ctypes.byref(d_in), 4*N))
ck(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4*N))

# Input: 0,0,0,1,0,0,0,0,... lane 3 is zero, rest non-zero
buf_in = (ctypes.c_uint32 * N)()
for i in range(N):
    buf_in[i] = 0 if i < 4 else 99
ck(cuda.cuMemcpyHtoD_v2(d_in, buf_in, 4*N))
ck(cuda.cuMemsetD8_v2(d_out, 0xCD, 4*N))  # sentinel

arg_in = ctypes.c_uint64(d_in.value)
arg_out = ctypes.c_uint64(d_out.value)
args = (ctypes.c_void_p * 2)(
    ctypes.cast(ctypes.pointer(arg_in), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(arg_out), ctypes.c_void_p),
)
extra = ctypes.c_void_p(0)
launch = cuda.cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, args, extra)
sync = cuda.cuCtxSynchronize()

host = (ctypes.c_uint32 * N)()
ck(cuda.cuMemcpyDtoH_v2(host, d_out, 4*N))
print(f"launch={launch} sync={sync}")
if sync == 700:
    print("ILLEGAL_ADDRESS — R22 defense was legitimate")
else:
    print(f"Lanes 0..3 (zero input) output: {list(host[:4])}  expected [0,0,0,0]")
    print(f"Lanes 4..31 (nonzero, skipped): {list(host[4:8])} expected [0xcdcdcdcd]*")
