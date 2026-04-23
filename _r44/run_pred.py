"""R44: run each of the 6 predicate-cluster kernels, dump actual vs
expected output for each tid."""
import ctypes, struct, sys, subprocess
sys.path.insert(0, 'C:/Users/kraken/openptxas')
import workbench as wb
import workbench_expanded as we
from ptx.parser import parse
from sass.pipeline import compile_function


# Map kernel name to (ptx, num_params, expected_fn, has_input)
KERNELS = {
'ilp_pred_alu': (
    wb._PTX_ILP_PRED_ALU,
    # out = (tid>16) ? (tid*7+42 + tid*3) : (tid*7+42)
    lambda t: (t*7+42 + t*3) & 0xFFFFFFFF if t > 16 else (t*7+42) & 0xFFFFFFFF,
    False,
),
'k100_early_exit': (
    we._K100_EARLY_EXIT,
    # odd threads store tid*5; even threads skip (out stays 0)
    lambda t: (t*5) & 0xFFFFFFFF if (t & 1) else 0,
    False,
),
'k200_load_pred_store': (
    we._K200_LOAD_PRED_STORE,
    # in[tid] = tid*3 (test input). if in[tid] > 32: out[tid] = in[tid]+1000 else out[tid] = in[tid]
    lambda t: ((t*3 + 1000) if (t*3 > 32) else (t*3)) & 0xFFFFFFFF,
    True,
),
'k200_xor_reduce': (
    we._K200_XOR_REDUCE,
    lambda t: t ^ 0x1 ^ 0x2 ^ 0x4 ^ 0x8,
    False,
),
'k300_nasty_pred_xor': (
    we._K300_NASTY_PRED_XOR,
    # r2 = tid ^ 0xAA; if tid > 16: r2 ^= 0x55
    lambda t: ((t ^ 0xAA) ^ 0x55) if t > 16 else (t ^ 0xAA),
    False,
),
'w1_div_if_else': (
    we._W1_DIV_IF_ELSE,
    # even: tid*3 + 1;  odd: tid*5 + 1
    lambda t: (t*3 + 1) if (t & 1) == 0 else (t*5 + 1),
    False,
),
}


def run(kname, N=32):
    ptx, expected_fn, has_input = KERNELS[kname]
    cuda = ctypes.WinDLL('nvcuda'); cuda.cuInit(0)
    dev = ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p(); cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    try:
        cubin = compile_function(parse(ptx).functions[0], verbose=False, sm_version=120)
        mod = ctypes.c_void_p()
        err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
        if err:
            print(f'[{kname}] load_err={err}')
            return
        func = ctypes.c_void_p()
        cuda.cuModuleGetFunction(ctypes.byref(func), mod, kname.encode())

        d_out = ctypes.c_uint64(); cuda.cuMemAlloc_v2(ctypes.byref(d_out), N*4)
        cuda.cuMemcpyHtoD_v2(d_out, b'\x00'*N*4, N*4)

        if has_input:
            d_in = ctypes.c_uint64(); cuda.cuMemAlloc_v2(ctypes.byref(d_in), N*4)
            inp = b''.join(struct.pack('<I', t*3) for t in range(N))
            cuda.cuMemcpyHtoD_v2(d_in, inp, N*4)
            a_out = ctypes.c_uint64(d_out.value); a_in = ctypes.c_uint64(d_in.value); a_n = ctypes.c_uint32(N)
            argv = (ctypes.c_void_p*3)(
                ctypes.cast(ctypes.byref(a_out), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(a_in), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(a_n), ctypes.c_void_p))
        else:
            a_out = ctypes.c_uint64(d_out.value); a_n = ctypes.c_uint32(N)
            argv = (ctypes.c_void_p*2)(
                ctypes.cast(ctypes.byref(a_out), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(a_n), ctypes.c_void_p))

        cuda.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 0, None, argv, None)
        err = cuda.cuCtxSynchronize()
        buf = ctypes.create_string_buffer(N*4); cuda.cuMemcpyDtoH_v2(buf, d_out, N*4)
        vals = struct.unpack(f'<{N}I', buf.raw)
        print(f'\n[{kname}] sync={err}')
        mismatches = 0
        show = min(N, 8)
        for t in range(show):
            exp = expected_fn(t) & 0xFFFFFFFF
            ok = 'OK' if vals[t] == exp else 'XX'
            print(f'  tid={t:2d}: got=0x{vals[t]:08x} exp=0x{exp:08x} {ok}')
            if vals[t] != exp:
                mismatches += 1
        # full count
        total_wrong = sum(1 for t in range(N) if vals[t] != (expected_fn(t) & 0xFFFFFFFF))
        print(f'  ... total {total_wrong}/{N} wrong')
    finally:
        cuda.cuCtxDestroy_v2(ctx)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        for k in KERNELS:
            subprocess.run([sys.executable, __file__, k])
