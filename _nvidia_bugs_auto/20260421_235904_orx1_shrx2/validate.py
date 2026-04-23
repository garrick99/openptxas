"""Standalone cross-arch validator (auto-generated).

Uses only ptxas + libcuda.so.  Compiles the minimal PTX, launches on
the GPU with a deterministic 128-byte input, compares the ptxas output
against the spec-expected output baked into this file.
"""
import sys, ctypes, struct, subprocess, os, tempfile

TARGET = sys.argv[1] if len(sys.argv) > 1 else 'sm_89'

PTX = """.version 9.0
.target sm_120
.address_size 64
.visible .entry fuzz(.param .u64 p_in, .param .u64 p_out, .param .u32 n) {
    .reg .b32 %r<32>;
    .reg .b64 %rd<8>;
    .reg .pred %p<2>;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 ret;
    ld.param.u64 %rd0, [p_in];
    cvt.u64.u32 %rd1, %r0;
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r3, [%rd2];
    or.b32 %r4, %r3, 2147483648;
    shr.s32 %r5, %r4, 1;
    shr.u32 %r6, %r5, 31;
    ld.param.u64 %rd3, [p_out];
    add.u64 %rd4, %rd3, %rd1;
    st.global.u32 [%rd4], %r6;
    ret;
}
"""

INPUTS = bytes([0,0,0,0,255,255,255,255,170,170,170,170,85,85,85,85,0,0,0,128,255,255,255,127,1,0,0,0,254,255,255,255,239,190,173,222,190,186,254,202,120,86,52,18,33,67,101,135,255,0,0,0,0,0,0,255,15,15,15,15,240,240,240,240,255,0,255,0,0,255,0,255,51,51,51,51,204,204,204,204,0,1,0,0,0,4,0,0,0,0,0,64,0,0,0,32,103,69,35,1,239,205,171,137,152,186,220,254,16,50,84,118,255,255,255,63,0,0,0,192,32,32,32,32,165,165,165,165])
EXPECTED = bytes([1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0])

for libname in ('libcuda.so.1', 'libcuda.so', 'nvcuda'):
    try:
        cuda = ctypes.CDLL(libname); break
    except OSError: continue
else:
    print('ERROR: libcuda not found'); sys.exit(1)

def chk(rc, where):
    if rc != 0:
        msg = ctypes.c_char_p()
        cuda.cuGetErrorString(rc, ctypes.byref(msg))
        raise RuntimeError(f'{where}: rc={rc} {msg.value!r}')

cuda.cuInit(0)
dev = ctypes.c_int(); chk(cuda.cuDeviceGet(ctypes.byref(dev), 0), 'cuDeviceGet')
ctx = ctypes.c_void_p(); chk(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), 'cuCtxCreate')

with tempfile.TemporaryDirectory() as tmp:
    ptx_p = os.path.join(tmp, 'k.ptx'); cubin_p = os.path.join(tmp, 'k.cubin')
    # Patch target and downgrade PTX version so older toolkits can parse.
    ptx_patched = PTX.replace('.target sm_120', f'.target {TARGET}') \
                     .replace('.target sm_89',  f'.target {TARGET}') \
                     .replace('.version 9.0',   '.version 7.8')
    open(ptx_p, 'w').write(ptx_patched)
    r = subprocess.run(['ptxas', f'-arch={TARGET}', ptx_p, '-o', cubin_p],
                        capture_output=True, text=True)
    if r.returncode != 0:
        print(f'ptxas failed: {r.stderr[:200]}'); sys.exit(1)
    cubin = open(cubin_p, 'rb').read()

mod = ctypes.c_void_p(); chk(cuda.cuModuleLoadData(ctypes.byref(mod), cubin), 'load')
fn = ctypes.c_void_p(); chk(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b'fuzz'), 'getfn')
d_in = ctypes.c_uint64(); chk(cuda.cuMemAlloc_v2(ctypes.byref(d_in), 128), 'alloc in')
d_out = ctypes.c_uint64(); chk(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 128), 'alloc out')
chk(cuda.cuMemcpyHtoD_v2(d_in, INPUTS, 128), 'HtoD')
chk(cuda.cuMemsetD8_v2(d_out, 0xAB, 128), 'memset')
ai = ctypes.c_uint64(d_in.value); ao = ctypes.c_uint64(d_out.value); an = ctypes.c_uint32(32)
argv = (ctypes.c_void_p*3)(ctypes.cast(ctypes.byref(ai), ctypes.c_void_p),
                            ctypes.cast(ctypes.byref(ao), ctypes.c_void_p),
                            ctypes.cast(ctypes.byref(an), ctypes.c_void_p))
chk(cuda.cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, None, argv, None), 'launch')
chk(cuda.cuCtxSynchronize(), 'sync')
buf = ctypes.create_string_buffer(128); chk(cuda.cuMemcpyDtoH_v2(buf, d_out, 128), 'DtoH')
out = bytes(buf.raw)

wrong = 0
print(f'ptxas target: {TARGET}')
print(f'lane | input      | expected   | ptxas      | bug?')
print(f'-----|------------|------------|------------|-----')
for i in range(32):
    inp_w = struct.unpack_from('<I', INPUTS, i*4)[0]
    exp_w = struct.unpack_from('<I', EXPECTED, i*4)[0]
    got_w = struct.unpack_from('<I', out, i*4)[0]
    if got_w != exp_w:
        wrong += 1
        if wrong <= 5 or i == 31:
            print(f' {i:3d} | {inp_w:#010x} | {exp_w:#010x} | {got_w:#010x} | YES')
print()
print(f'ptxas wrong on {wrong}/32 lanes')
sys.exit(1 if wrong > 0 else 0)
