"""R49: dump full SASS of k200_xor_reduce final."""
import sys, struct
sys.path.insert(0, 'C:/Users/kraken/openptxas')
import workbench_expanded as we
from ptx.parser import parse
from sass.pipeline import compile_function

cubin = compile_function(parse(we._K200_XOR_REDUCE).functions[0], verbose=False, sm_version=120)
e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]
def sh(i): return struct.unpack_from('<IIQQQQIIQQ', cubin, e_shoff + i*64)
_, _, _, _, so, ss, *_ = sh(e_shstrndx)
shs = cubin[so:so+ss]
text = b''
for i in range(e_shnum):
    nm, ty, _, _, off, sz, *_ = sh(i)
    end = shs.index(b'\0', nm)
    if shs[nm:end].startswith(b'.text.') and ty == 1:
        text = cubin[off:off+sz]; break
for i in range(0, len(text), 16):
    r = text[i:i+16]
    opc = (r[0] | (r[1] << 8)) & 0xFFF
    full = (r[0] | (r[1] << 8)) & 0xFFFF
    print(f'  0x{i:03x} full=0x{full:04x} opc=0x{opc:03x} raw={r.hex()}')
