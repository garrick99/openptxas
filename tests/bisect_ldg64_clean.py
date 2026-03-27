"""
Clean bisect: apply each ctrl from ldg64_test_fresh.cubin onto ldg64_default_ctrl.cubin
one at a time, each in a fresh subprocess, to find which ctrl word(s) cause ILLEGAL_ADDRESS.
"""
import struct
import subprocess
import sys
import os
import tempfile

FRESH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'probe_work', 'ldg64_test_fresh.cubin'))
DEFAULT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'probe_work', 'ldg64_default_ctrl.cubin'))
WORK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', 'probe_work', 'bisect_work.cubin'))

RUNNER = r'''
import ctypes, struct, sys

cuda = ctypes.cdll.LoadLibrary("nvcuda.dll")
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64

def check(err, msg=""):
    if err != 0:
        name = ctypes.c_char_p()
        desc = ctypes.c_char_p()
        cuda.cuGetErrorName(err, ctypes.byref(name))
        cuda.cuGetErrorString(err, ctypes.byref(desc))
        n = name.value.decode() if name.value else "?"
        d = desc.value.decode() if desc.value else "?"
        raise RuntimeError(f"CUDA {err}: {n} - {d} [{msg}]")

path = sys.argv[1]
with open(path, "rb") as f:
    cubin_data = f.read()

check(cuda.cuInit(0), "cuInit")
dev = CUdevice()
check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ctx = CUcontext()
check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

mod = CUmodule()
err = cuda.cuModuleLoadData(ctypes.byref(mod), cubin_data)
if err != 0:
    name = ctypes.c_char_p()
    cuda.cuGetErrorName(err, ctypes.byref(name))
    print(f"LOAD_FAIL:{name.value.decode() if name.value else err}")
    cuda.cuCtxDestroy_v2(ctx)
    sys.exit(1)

func = CUfunction()
check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"ldg64_min"), "GetFunction")

d_in = CUdeviceptr()
d_out = CUdeviceptr()
check(cuda.cuMemAlloc_v2(ctypes.byref(d_in), 8), "alloc d_in")
check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 8), "alloc d_out")

pattern = 0xDEADBEEFCAFEBABE
check(cuda.cuMemcpyHtoD_v2(d_in, struct.pack("<Q", pattern), 8), "H2D in")
check(cuda.cuMemcpyHtoD_v2(d_out, struct.pack("<Q", 0), 8), "H2D out")

arg_out = ctypes.c_uint64(d_out.value)
arg_in  = ctypes.c_uint64(d_in.value)
args = (ctypes.c_void_p * 2)(
    ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
    ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
)

err = cuda.cuLaunchKernel(func, 1,1,1, 1,1,1, 0, None, args, None)
if err != 0:
    name = ctypes.c_char_p()
    cuda.cuGetErrorName(err, ctypes.byref(name))
    print(f"LAUNCH_FAIL:{name.value.decode() if name.value else err}")
    cuda.cuCtxDestroy_v2(ctx)
    sys.exit(1)

err = cuda.cuCtxSynchronize()
if err != 0:
    name = ctypes.c_char_p()
    cuda.cuGetErrorName(err, ctypes.byref(name))
    print(f"CRASH:{name.value.decode() if name.value else err}")
    cuda.cuCtxDestroy_v2(ctx)
    sys.exit(1)

h_out = ctypes.create_string_buffer(8)
check(cuda.cuMemcpyDtoH_v2(h_out, d_out, 8), "D2H out")
val = struct.unpack("<Q", h_out.raw)[0]
cuda.cuMemFree_v2(d_in)
cuda.cuMemFree_v2(d_out)
cuda.cuModuleUnload(mod)
cuda.cuCtxDestroy_v2(ctx)

if val == pattern:
    print("PASS")
else:
    print(f"WRONG:0x{val:016x}")
    sys.exit(1)
'''

RUNNER_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', 'probe_work', '_runner.py'))
with open(RUNNER_PATH, 'w') as f:
    f.write(RUNNER)


def get_text_info(data):
    """Return (text_offset, text_size) for .text.ldg64_min section."""
    e_shoff = struct.unpack_from('<Q', data, 0x28)[0]
    e_shentsize = struct.unpack_from('<H', data, 0x3a)[0]
    e_shnum = struct.unpack_from('<H', data, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', data, 0x3e)[0]
    sh = e_shoff + e_shstrndx * e_shentsize
    sh_offset = struct.unpack_from('<Q', data, sh + 24)[0]
    sh_size = struct.unpack_from('<Q', data, sh + 32)[0]
    shstr = data[sh_offset:sh_offset + sh_size]
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        name_off = struct.unpack_from('<I', data, sh)[0]
        end = shstr.index(b'\x00', name_off)
        name = shstr[name_off:end].decode()
        if name == '.text.ldg64_min':
            return (struct.unpack_from('<Q', data, sh + 24)[0],
                    struct.unpack_from('<Q', data, sh + 32)[0])
    raise RuntimeError('section not found')


def extract_ctrl(raw16):
    """Extract ctrl word (23 bits) from 16-byte instruction."""
    raw24 = ((raw16[15] & ~0x04) << 16) | (raw16[14] << 8) | raw16[13]
    return raw24 >> 1


def patch_ctrl(raw16, ctrl):
    """Return new 16-byte instruction with ctrl patched in."""
    buf = bytearray(raw16)
    raw24 = (ctrl & 0x7FFFFF) << 1
    buf[13] = raw24 & 0xFF
    buf[14] = (raw24 >> 8) & 0xFF
    buf[15] = ((raw24 >> 16) & 0xFF) | (buf[15] & 0x04)
    return bytes(buf)


def run_cubin(cubin_bytes):
    """Run cubin in a fresh subprocess. Returns 'PASS' or error string."""
    with open(WORK, 'wb') as f:
        f.write(cubin_bytes)
    result = subprocess.run(
        [sys.executable, RUNNER_PATH, WORK],
        capture_output=True, text=True, timeout=30
    )
    out = (result.stdout + result.stderr).strip()
    return out if out else f'EXIT:{result.returncode}'


def main():
    with open(FRESH, 'rb') as f:
        fresh_data = bytearray(f.read())
    with open(DEFAULT, 'rb') as f:
        default_data = bytearray(f.read())

    fresh_text_off, fresh_text_size = get_text_info(bytes(fresh_data))
    default_text_off, _ = get_text_info(bytes(default_data))

    n_instr = fresh_text_size // 16
    opnames = {0xb82: 'LDC', 0x919: 'S2R', 0x7ac: 'LDCU.64', 0xc35: 'IADD64-UR',
               0x981: 'LDG.E.64', 0x918: 'NOP', 0x986: 'STG.E.64',
               0x94d: 'EXIT', 0x947: 'BRA'}

    # First verify baseline passes
    print('Testing baseline (default ctrl)...', end=' ', flush=True)
    r = run_cubin(bytes(default_data))
    print(r)

    print()
    print(f'Bisecting {n_instr} instructions (applying our ctrl one at a time):')
    print('-' * 70)

    failures = []
    for i in range(n_instr):
        f_off = fresh_text_off + i * 16
        d_off = default_text_off + i * 16
        raw_fresh = bytes(fresh_data[f_off:f_off + 16])
        raw_def   = bytes(default_data[d_off:d_off + 16])

        opc = struct.unpack_from('<Q', raw_fresh, 0)[0] & 0xFFF
        name = opnames.get(opc, f'0x{opc:03x}')
        fresh_ctrl = extract_ctrl(raw_fresh)
        def_ctrl   = extract_ctrl(raw_def)

        if fresh_ctrl == def_ctrl:
            # No change for this instruction
            rbar = (fresh_ctrl >> 10) & 0x1f
            wdep = (fresh_ctrl >> 4) & 0x3f
            print(f'  [{i:2d}] +0x{i*16:03x} {name:12s}  rbar=0x{rbar:02x} wdep=0x{wdep:02x}  UNCHANGED')
            continue

        # Apply our ctrl to the default cubin (only this instruction)
        test_data = bytearray(default_data)
        test_data[d_off:d_off + 16] = patch_ctrl(raw_def, fresh_ctrl)
        r = run_cubin(bytes(test_data))

        rbar_f = (fresh_ctrl >> 10) & 0x1f
        wdep_f = (fresh_ctrl >> 4) & 0x3f
        misc_f = fresh_ctrl & 0xf
        rbar_d = (def_ctrl >> 10) & 0x1f
        wdep_d = (def_ctrl >> 4) & 0x3f
        status = 'OK' if r == 'PASS' else f'FAIL({r})'
        print(f'  [{i:2d}] +0x{i*16:03x} {name:12s}'
              f'  rbar=0x{rbar_f:02x} wdep=0x{wdep_f:02x} misc=0x{misc_f:x}'
              f'  (was rbar=0x{rbar_d:02x} wdep=0x{wdep_d:02x})'
              f'  -> {status}')
        if r != 'PASS':
            failures.append((i, name, fresh_ctrl, def_ctrl, r))

    print()
    if failures:
        print(f'FAILURES ({len(failures)}):')
        for i, name, fc, dc, r in failures:
            print(f'  [{i:2d}] {name}: {r}')
    else:
        print('All individual ctrl changes PASS — issue is combinatorial.')


if __name__ == '__main__':
    main()
