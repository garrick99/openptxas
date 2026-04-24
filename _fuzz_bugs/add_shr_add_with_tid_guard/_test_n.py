"""Try N=1, 2, 4, 8, 16 to see which lanes actually execute the store."""
import sys, pathlib, ctypes
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

for test_n in [1, 2, 4, 8, 16, 32]:
    d_mem = ctypes.c_uint64(); ck(cuda.cuMemAlloc_v2(ctypes.byref(d_mem), 4096))
    sent = (ctypes.c_uint32 * 1024)(*[0xDEADBEEF] * 1024)
    ck(cuda.cuMemcpyHtoD_v2(d_mem, sent, 4096))
    d_in = d_mem.value
    d_out = d_mem.value + 2048
    # Distinct inputs: buf_in[i] = i * 4 + 5
    buf_in = (ctypes.c_uint32 * 32)(*[i * 4 + 5 for i in range(32)])
    ck(cuda.cuMemcpyHtoD_v2(ctypes.c_uint64(d_in), buf_in, 4*32))

    arg_p_in = ctypes.c_uint64(d_in)
    arg_p_out = ctypes.c_uint64(d_out)
    arg_n = ctypes.c_uint32(test_n)
    args = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.pointer(arg_p_in), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(arg_p_out), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(arg_n), ctypes.c_void_p),
    )
    extra = ctypes.c_void_p(0)
    err = cuda.cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, args, extra)
    sync_err = cuda.cuCtxSynchronize()

    # Readback
    host = (ctypes.c_uint32 * 1024)()
    ck(cuda.cuMemcpyDtoH_v2(host, d_mem, 4096))

    changes = []
    for i in range(1024):
        if host[i] != 0xDEADBEEF and host[i] != buf_in[i] if i < 32 else True:
            if host[i] != 0xDEADBEEF:
                changes.append((i * 4, host[i]))

    print(f"n={test_n:3d}: {len(changes)} changes")
    # Show first 10 "stores" (not input reads)
    expected = {((i+3)>>2)+256: i for i in range(32) for _ in [buf_in[i]]}
    for offset, v in changes[:10]:
        region = "IN " if offset < 2048 else ("OUT" if offset < 2048+128 else "???")
        which_lane = next((tid for tid, bv in enumerate(buf_in) if ((bv+3)>>2)+256 == v), -1)
        print(f"    +0x{offset:04x}: 0x{v:x}  [{region}]  lane={which_lane}")

    ck(cuda.cuMemFree_v2(d_mem))
