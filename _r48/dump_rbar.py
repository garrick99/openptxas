"""R48: dump every instruction's ctrl decoding for k300_nasty_pred_xor after
Bug-1 rename, to see who sets/waits on which rbar."""
import sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')
sys.path.insert(0, 'C:/Users/kraken/openptxas/_r48')
from probe_handoff import find_text, apply_bug1_rename
from ptx.parser import parse
from sass.pipeline import compile_function
import workbench_expanded as we

def decode(b):
    opc = (b[0] | (b[1]<<8)) & 0xFFF
    guard = (b[1] >> 4) & 0xF
    ctrl24 = (b[15] << 16) | (b[14] << 8) | b[13]
    ctrl = ctrl24 >> 1
    misc = ctrl & 0xF
    wdep = (ctrl >> 4) & 0x3F
    rbar = (ctrl >> 10) & 0x1F
    stall = (ctrl >> 17) & 0x3F
    return opc, guard, misc, wdep, rbar, stall

OPC_NAME = {
    0x7b82:'LDC', 0x919:'S2R', 0x77ac:'LDCU/LDCU.64', 0x80c:'ISETP.IMM',
    0xc0c:'ISETP.R-UR', 0x94d:'EXIT', 0x812:'LOP3.LUT', 0x7c11:'LEA',
    0x7986:'STG.E', 0x7947:'BRA', 0x7918:'NOP',
}

def name_of(opc, full_opc):
    if full_opc in OPC_NAME: return OPC_NAME[full_opc]
    return OPC_NAME.get(opc, f'op{opc:03x}')

if __name__ == '__main__':
    base = compile_function(parse(we._K300_NASTY_PRED_XOR).functions[0], verbose=False, sm_version=120)
    fixed = apply_bug1_rename(base)
    sec_off, sec_sz = find_text(fixed)
    text = fixed[sec_off:sec_off+sec_sz]
    n = len(text)//16
    print(f'{"pos":>4} {"name":<12} {"guard":<5} {"misc":<4} {"wdep":<4} {"rbar":<4} {"stall":<5}')
    for i in range(n):
        b = text[i*16:(i+1)*16]
        full_opc = (b[0] | (b[1]<<8)) & 0xFFFF
        opc = full_opc & 0xFFF
        _, guard, misc, wdep, rbar, stall = decode(b)
        name = name_of(opc, full_opc)
        print(f'{i*16:4d} {name:<12} P{guard:<4} 0x{misc:02x} 0x{wdep:02x} 0x{rbar:02x} 0x{stall:02x}')
