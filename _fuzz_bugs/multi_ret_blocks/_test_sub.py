"""Test sub.f32 5 - 3. Expect 2.0 if correct, -2.0 if inverted."""
import sys, pathlib, ctypes, struct
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from sass.pipeline import compile_ptx_source

ptx = """.version 8.8
.target sm_120
.address_size 64

.visible .entry subtest(.param .u64 out, .param .f32 a, .param .f32 b)
{
    .reg .f32 %f<4>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd0, [out];
    ld.param.f32 %f0, [a];
    ld.param.f32 %f1, [b];
    sub.f32 %f2, %f0, %f1;
    st.global.f32 [%rd0], %f2;
    ret;
}
"""

cub = compile_ptx_source(ptx)['subtest']
cubin_path = str(pathlib.Path(__file__).with_name('_subtest.cubin'))
open(cubin_path, 'wb').write(cub)

cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')

def ck(e, tag=''):
    if e != 0:
        raise RuntimeError(f'{tag}: err={e}')

ck(cuda.cuInit(0))
dev = ctypes.c_int()
ck(cuda.cuDeviceGet(ctypes.byref(dev), 0))
cctx = ctypes.c_void_p()
ck(cuda.cuCtxCreate_v2(ctypes.byref(cctx), 0, dev))

mod = ctypes.c_void_p()
ck(cuda.cuModuleLoad(ctypes.byref(mod), cubin_path.encode()))
fn = ctypes.c_void_p()
ck(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b'subtest'))

d_out = ctypes.c_uint64()
ck(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4))
ck(cuda.cuMemsetD8_v2(d_out, 0, 4))

arg_out = ctypes.c_uint64(d_out.value)
arg_a = ctypes.c_float(5.0)
arg_b = ctypes.c_float(3.0)
args = (ctypes.c_void_p * 3)(
    ctypes.cast(ctypes.pointer(arg_out), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(arg_a), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(arg_b), ctypes.c_void_p),
)
extra = ctypes.c_void_p(0)
ck(cuda.cuLaunchKernel(fn, 1,1,1, 1,1,1, 0, 0, args, extra))
ck(cuda.cuCtxSynchronize())

host = (ctypes.c_float * 1)()
ck(cuda.cuMemcpyDtoH_v2(host, d_out, 4))
print(f"sub.f32(5, 3) = {host[0]:.3f}  (expected 2.0)")
