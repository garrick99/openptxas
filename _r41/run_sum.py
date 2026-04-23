"""R41: run w1_loop_sum directly and dump per-thread output."""
import ctypes, struct, sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')
import workbench_expanded as we
from ptx.parser import parse
from sass.pipeline import compile_function


def run(ptx, kname, N=8):
    cubin = compile_function(parse(ptx).functions[0], verbose=False, sm_version=120)
    cuda = ctypes.WinDLL('nvcuda'); cuda.cuInit(0)
    dev = ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p(); cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    try:
        mod = ctypes.c_void_p(); cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
        func = ctypes.c_void_p(); cuda.cuModuleGetFunction(ctypes.byref(func), mod, kname.encode())
        d_out = ctypes.c_uint64(); cuda.cuMemAlloc_v2(ctypes.byref(d_out), N * 4)
        cuda.cuMemcpyHtoD_v2(d_out, b'\xaa' * (N * 4), N * 4)
        a_out = ctypes.c_uint64(d_out.value); a_n = ctypes.c_uint32(N)
        argv = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.byref(a_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(a_n), ctypes.c_void_p))
        cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 0, None, argv, None)
        err = cuda.cuCtxSynchronize()
        buf = ctypes.create_string_buffer(N * 4); cuda.cuMemcpyDtoH_v2(buf, d_out, N * 4)
        vals = struct.unpack(f'<{N}I', buf.raw)
        print(f'[{kname}] sync={err}')
        for t, v in enumerate(vals):
            print(f'  tid={t:2d}: out=0x{v:08x} ({v})')
    finally:
        cuda.cuCtxDestroy_v2(ctx)


if __name__ == '__main__':
    run(we._W1_LOOP_SUM, 'w1_loop_sum', N=8)
