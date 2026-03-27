"""Binary search: run OpenCUDA kernel truncated to N instructions."""
import ctypes, struct, sys, pickle
sys.path.insert(0, '.')
from sass.encoding.sm_120_opcodes import encode_exit
from sass.isel import SassInstr
from sass.scoreboard import assign_ctrl
from cubin.emitter import emit_cubin, KernelDesc

n_instrs = int(sys.argv[1])

with open('probe_work/_opencuda_body.pkl', 'rb') as f:
    preamble, body, num_gprs, param_offsets = pickle.load(f)

# Truncate body to n_instrs, always add unpredicated EXIT at the end
truncated = body[:n_instrs]
# Always append an unpredicated EXIT (even if last is @P EXIT)
truncated.append(SassInstr(encode_exit(), 'EXIT'))

# Apply scoreboard ctrl to body (skip preamble)
truncated = assign_ctrl(truncated)
sass_instrs = preamble + truncated
sass_bytes = b''.join(si.raw for si in sass_instrs)

exit_offset = 0
for i in range(0, len(sass_bytes), 16):
    if sass_bytes[i:i+2] == bytes([0x4d, 0x79]):
        exit_offset = i; break
s2r_offset = 0x10
for i in range(0, len(sass_bytes), 16):
    if sass_bytes[i:i+2] == bytes([0x19, 0x79]):
        s2r_offset = i; break

cubin = emit_cubin(KernelDesc(
    name='vector_add', sass_bytes=sass_bytes, num_gprs=max(num_gprs, 16),
    num_params=4, param_sizes=[8, 8, 8, 4], param_offsets=param_offsets,
    const0_size=0x39c, exit_offset=exit_offset, s2r_offset=s2r_offset,
))

cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
CUdevice = ctypes.c_int; CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p; CUfunction = ctypes.c_void_p; CUdeviceptr = ctypes.c_uint64

cuda.cuInit(0); dev = CUdevice(); cuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = CUcontext(); cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
mod = CUmodule(); err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin)
if err != 0:
    name = ctypes.c_char_p(); cuda.cuGetErrorName(err, ctypes.byref(name))
    print(f"LOAD:{name.value.decode()}"); sys.exit(0)
func = CUfunction(); cuda.cuModuleGetFunction(ctypes.byref(func), mod, b'vector_add')
d = CUdeviceptr(); cuda.cuMemAlloc_v2(ctypes.byref(d), 128)
arg = ctypes.c_uint64(d.value); arg_n = ctypes.c_int32(32)
args = (ctypes.c_void_p * 4)(
    ctypes.cast(ctypes.byref(arg), ctypes.c_void_p),
    ctypes.cast(ctypes.byref(arg), ctypes.c_void_p),
    ctypes.cast(ctypes.byref(arg), ctypes.c_void_p),
    ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
)
err = cuda.cuLaunchKernel(func, 1,1,1, 32,1,1, 0, None, args, None)
if err != 0:
    name = ctypes.c_char_p(); cuda.cuGetErrorName(err, ctypes.byref(name))
    print(f"LAUNCH:{name.value.decode()}"); sys.exit(0)
err = cuda.cuCtxSynchronize()
if err != 0:
    name = ctypes.c_char_p(); cuda.cuGetErrorName(err, ctypes.byref(name))
    print(f"CRASH:{name.value.decode()}")
else:
    print("PASS")
cuda.cuMemFree_v2(d); cuda.cuModuleUnload(mod); cuda.cuCtxDestroy_v2(ctx)
