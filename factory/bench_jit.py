"""Benchmark: ptxas subprocess vs driver JIT.

Both compile the same PTX.  Subprocess path is what differ_d.py does
today (spawn ptxas.exe, write temp file, read cubin).  Driver-JIT path
uses cuLinkCreate + cuLinkAddData(CU_JIT_INPUT_PTX) + cuLinkComplete,
which runs ptxas in-process through libcuda.
"""
import ctypes, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fuzzer.oracle import compile_theirs
from factory.supervisor import _BUG2_PTX as PTX

# ---- Driver-JIT compile via cuLink* ----
cuda = ctypes.WinDLL('nvcuda')
cuda.cuInit(0)
dev = ctypes.c_int(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p(); cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

# cuLinkCreate prototype:
#   CUresult cuLinkCreate(unsigned int numOptions, CUjit_option* options,
#                         void** optionValues, CUlinkState* stateOut);
# cuLinkAddData:
#   CUresult cuLinkAddData(CUlinkState state, CUjitInputType type,
#                          void* data, size_t size, const char* name,
#                          unsigned int numOptions, CUjit_option* options,
#                          void** optionValues);
# cuLinkComplete:
#   CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut);

CU_JIT_INPUT_PTX = 1
CU_JIT_TARGET    = 9      # jit target compute capability
CU_JIT_INFO_LOG_BUFFER = 3
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6


def compile_theirs_jit(ptx: str, sm: int = 120) -> tuple:
    ptx_bytes = ptx.encode() + b'\0'
    # Options: target sm_120 + error log buffer
    err_buf = ctypes.create_string_buffer(4096)
    opts = (ctypes.c_int * 3)(CU_JIT_TARGET,
                               CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
    vals = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.c_void_p(sm), ctypes.c_void_p),
        ctypes.cast(err_buf, ctypes.c_void_p),
        ctypes.cast(ctypes.c_void_p(len(err_buf)), ctypes.c_void_p))
    state = ctypes.c_void_p()
    rc = cuda.cuLinkCreate_v2(3, opts, vals, ctypes.byref(state))
    if rc != 0: return None, f'cuLinkCreate rc={rc}'
    rc = cuda.cuLinkAddData_v2(state, CU_JIT_INPUT_PTX,
                                ctypes.c_char_p(ptx_bytes),
                                ctypes.c_size_t(len(ptx_bytes)),
                                ctypes.c_char_p(b'k.ptx'),
                                0, None, None)
    if rc != 0:
        cuda.cuLinkDestroy(state)
        return None, err_buf.value.decode('utf-8', errors='replace')[:200] or f'cuLinkAddData rc={rc}'
    cubin_ptr = ctypes.c_void_p()
    cubin_sz = ctypes.c_size_t()
    rc = cuda.cuLinkComplete(state, ctypes.byref(cubin_ptr), ctypes.byref(cubin_sz))
    if rc != 0:
        cuda.cuLinkDestroy(state)
        return None, err_buf.value.decode('utf-8', errors='replace')[:200] or f'cuLinkComplete rc={rc}'
    cubin = ctypes.string_at(cubin_ptr, cubin_sz.value)
    cuda.cuLinkDestroy(state)
    return cubin, None


# ---- Benchmark ----
N = 100

# Warmup
compile_theirs(PTX)
compile_theirs_jit(PTX)

t0 = time.time()
for _ in range(N):
    c, e = compile_theirs(PTX)
    if c is None: print('subprocess err:', e); break
subp_dur = time.time() - t0

t0 = time.time()
for _ in range(N):
    c, e = compile_theirs_jit(PTX)
    if c is None: print('jit err:', e); break
jit_dur = time.time() - t0

print(f'subprocess: {N} calls in {subp_dur:.3f}s  =>  {N/subp_dur:6.1f}/sec  ({subp_dur*1000/N:.1f} ms/call)')
print(f'driver JIT: {N} calls in {jit_dur:.3f}s  =>  {N/jit_dur:6.1f}/sec  ({jit_dur*1000/N:.1f} ms/call)')
print(f'speedup:    {subp_dur/jit_dur:.1f}x')

# Sanity: same cubin?
cs, _ = compile_theirs(PTX)
cj, _ = compile_theirs_jit(PTX)
print(f'cubin sizes: subprocess={len(cs)}  jit={len(cj)}  equal={cs == cj}')
