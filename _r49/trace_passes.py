"""R49: monkey-patch key passes to dump LOP3(0x812) instrs at pass
boundaries, to find where imm 0x4 becomes 0x0."""
import sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')
from ptx.parser import parse
import workbench_expanded as we
import sass.pipeline as P
import sass.compact as C
from sass.pipeline import SassInstr


def _dump(label, instrs):
    print(f'--- {label} ---')
    for i, si in enumerate(instrs):
        r = si.raw if hasattr(si, 'raw') else si
        if len(r) != 16: continue
        opc = (r[0] | (r[1] << 8)) & 0xFFF
        if opc == 0x812:
            imm = r[4] | (r[5]<<8) | (r[6]<<16) | (r[7]<<24)
            print(f'  [{i:3d}] LOP3.LUT dst=R{r[2]} src0=R{r[3]} imm=0x{imm:x} -- raw={r.hex()}')


# Wrap compact pass
_orig_compact = None
try:
    import sass.compact as _cm
    for name in dir(_cm):
        if callable(getattr(_cm, name, None)) and 'compact' in name.lower():
            print(f'compact func: {name}')
except Exception as e:
    print(e)

# Actually, the cleanest approach: run compile_function and insert a hook
# via instrumenting SassInstr construction.
_ORIG = SassInstr


class _Hook(_ORIG):
    """Wraps construction to record a stack-trace -> imm trail."""
    pass

# Just run the full pipeline and capture stdout — also manually re-run
# intermediate stages.
cubin = P.compile_function(parse(we._K200_XOR_REDUCE).functions[0], verbose=False, sm_version=120)

# Extract .text from cubin
import struct
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
        text = cubin[off:off+sz]
        break

print('\n=== FINAL .text ===')
for i in range(0, len(text), 16):
    r = text[i:i+16]
    opc = (r[0] | (r[1] << 8)) & 0xFFF
    if opc == 0x812:
        imm = r[4] | (r[5]<<8) | (r[6]<<16) | (r[7]<<24)
        print(f'  0x{i:03x} LOP3.LUT dst=R{r[2]} src0=R{r[3]} imm=0x{imm:x}  raw={r.hex()}')
    elif opc in (0x210, 0x810, 0x212):
        print(f'  0x{i:03x} opc=0x{opc:03x} dst=R{r[2]} src0=R{r[3]}  raw={r.hex()}')
