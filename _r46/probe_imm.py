"""R46 probe: does the imm-dropping bug in k200_xor_reduce depend on the
specific imm value (0x4), the register reuse pattern, or the chain length?"""
import sys, tempfile, os, subprocess
sys.path.insert(0, 'C:/Users/kraken/openptxas')
from ptx.parser import parse
from sass.pipeline import compile_function

NVDISASM = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvdisasm.exe'

def build_ptx(body, name='k'):
    return f"""
.version 9.0
.target sm_120
.address_size 64
.visible .entry {name}(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<10>; .reg .u64 %rd<4>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
{body}
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""

def disasm_body(cubin):
    with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
        f.write(cubin); path = f.name
    try:
        r = subprocess.run([NVDISASM, '-c', '-hex', path], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            s = line.strip()
            if s.startswith('/*0') and ('LOP3' in s or 'IADD' in s or 'IMAD' in s or 'XOR' in s or 'S2R' in s):
                print('   ', s[:120])
    finally:
        os.unlink(path)

# Variant A: original
print('\n[A] original xor chain (baseline k200_xor_reduce):')
body = """    xor.b32 %r2, %r0, 0x1;
    xor.b32 %r3, %r2, 0x2;
    xor.b32 %r4, %r3, 0x4;
    xor.b32 %r2, %r4, 0x8;"""
disasm_body(compile_function(parse(build_ptx(body, 'kA')).functions[0], verbose=False, sm_version=120))

# Variant B: change imm 0x4 -> 0x7 (not a pow-of-2)
print('\n[B] swap imm 0x4 -> 0x7:')
body = """    xor.b32 %r2, %r0, 0x1;
    xor.b32 %r3, %r2, 0x2;
    xor.b32 %r4, %r3, 0x7;
    xor.b32 %r2, %r4, 0x8;"""
disasm_body(compile_function(parse(build_ptx(body, 'kB')).functions[0], verbose=False, sm_version=120))

# Variant C: no register reuse (%r2 not redefined)
print('\n[C] no reuse of %r2 (dest=%r5 final):')
body = """    xor.b32 %r2, %r0, 0x1;
    xor.b32 %r3, %r2, 0x2;
    xor.b32 %r4, %r3, 0x4;
    xor.b32 %r5, %r4, 0x8;
    mov.u32 %r2, %r5;"""
disasm_body(compile_function(parse(build_ptx(body, 'kC')).functions[0], verbose=False, sm_version=120))

# Variant D: 3-xor chain (short)
print('\n[D] 3-xor chain, with reuse:')
body = """    xor.b32 %r2, %r0, 0x1;
    xor.b32 %r3, %r2, 0x2;
    xor.b32 %r2, %r3, 0x4;"""
disasm_body(compile_function(parse(build_ptx(body, 'kD')).functions[0], verbose=False, sm_version=120))

# Variant E: use IADD (not XOR)
print('\n[E] add chain with reuse:')
body = """    add.u32 %r2, %r0, 0x1;
    add.u32 %r3, %r2, 0x2;
    add.u32 %r4, %r3, 0x4;
    add.u32 %r2, %r4, 0x8;"""
disasm_body(compile_function(parse(build_ptx(body, 'kE')).functions[0], verbose=False, sm_version=120))
