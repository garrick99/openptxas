"""Check if OURS writes to wrong address — dump d_in after launch."""
import sys, pathlib, ctypes, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from sass.pipeline import compile_ptx_source

here = pathlib.Path(__file__).parent
ptx = (here / 'minimal.ptx').read_text()
cub_ours = compile_ptx_source(ptx)['fuzz']
open(here / '_ours.cubin', 'wb').write(cub_ours)

cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
def ck(e):
    if e != 0:
        name = ctypes.c_char_p(); cuda.cuGetErrorName(e, ctypes.byref(name))
        raise RuntimeError(f'err={e} {name.value.decode() if name.value else ""}')

ck(cuda.cuInit(0))
dev = ctypes.c_int(); ck(cuda.cuDeviceGet(ctypes.byref(dev), 0))
cctx = ctypes.c_void_p(); ck(cuda.cuCtxCreate_v2(ctypes.byref(cctx), 0, dev))

mod = ctypes.c_void_p()
ck(cuda.cuModuleLoad(ctypes.byref(mod), str(here/'_ours.cubin').encode()))
fn = ctypes.c_void_p()
ck(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b'fuzz'))

# Allocate big buffer, put in + out at specific known offsets
N = 4
d_mem = ctypes.c_uint64(); ck(cuda.cuMemAlloc_v2(ctypes.byref(d_mem), 4096))
# Sentinel the whole thing
sent = (ctypes.c_uint32 * 1024)(*[0xDEADBEEF] * 1024)
ck(cuda.cuMemcpyHtoD_v2(d_mem, sent, 4096))
# In buf at +0, Out buf at +2048
d_in = d_mem.value
d_out = d_mem.value + 2048
buf_in = (ctypes.c_uint32 * N)(*[i * 4 + 5 for i in range(N)])
ck(cuda.cuMemcpyHtoD_v2(ctypes.c_uint64(d_in), buf_in, 4*N))

arg_p_in = ctypes.c_uint64(d_in)
arg_p_out = ctypes.c_uint64(d_out)
arg_n = ctypes.c_uint32(N)
args = (ctypes.c_void_p * 3)(
    ctypes.cast(ctypes.pointer(arg_p_in), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(arg_p_out), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(arg_n), ctypes.c_void_p),
)
extra = ctypes.c_void_p(0)
err = cuda.cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, args, extra)
sync_err = cuda.cuCtxSynchronize()

# Read back entire buffer
host = (ctypes.c_uint32 * 1024)()
ck(cuda.cuMemcpyDtoH_v2(host, d_mem, 4096))

# Find any changes from sentinel
changes = []
for i in range(1024):
    if host[i] != 0xDEADBEEF:
        changes.append((i, host[i]))

print(f"launch={err} sync={sync_err}")
print(f"d_in  = 0x{d_in:x} (offset 0)")
print(f"d_out = 0x{d_out:x} (offset 512)")
print(f"\n{len(changes)} changed cells:")
for i, v in changes[:40]:
    offset = i * 4
    region = "IN  " if 0 <= offset < 2048 else ("OUT " if 2048 <= offset < 2048+256 else "????")
    print(f"  offset=0x{offset:04x} (word {i:3}): 0x{v:x}  [{region}]")
