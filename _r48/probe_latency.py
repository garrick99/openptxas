"""R48: is the @P1 LOP3 hazard a pure-latency issue?  Probe:
 - insert N NOPs between ISETP and @P1 LOP3
 - vs bump stall on @P1 LOP3
 - vs bump stall on ISETP
"""
import sys, subprocess
sys.path.insert(0, 'C:/Users/kraken/openptxas')
sys.path.insert(0, 'C:/Users/kraken/openptxas/_r48')
from probe_handoff import (find_text, apply_bug1_rename, locate_pred_lop3,
                             locate_isetp_pred, patch_ctrl, run)
from ptx.parser import parse
from sass.pipeline import compile_function
import workbench_expanded as we

NOP = bytes.fromhex('18790000000000000000000000c00f00')


def build_nop(num_nops):
    base = compile_function(parse(we._K300_NASTY_PRED_XOR).functions[0], verbose=False, sm_version=120)
    fixed = apply_bug1_rename(base)
    sec_off, sec_sz = find_text(fixed)
    text = bytearray(fixed[sec_off:sec_off+sec_sz])
    p1_idx = locate_pred_lop3(text)
    instrs = [bytes(text[i*16:(i+1)*16]) for i in range(len(text)//16)]
    # count trailing NOPs
    trailing = 0
    for i in range(len(instrs)-1, -1, -1):
        if instrs[i] == NOP: trailing += 1
        else: break
    if trailing < num_nops:
        return None
    # insert num_nops before p1_idx; remove equal number from trailing
    new = instrs[:p1_idx] + [NOP]*num_nops + instrs[p1_idx:len(instrs)-num_nops]
    new_text = b''.join(new)
    new_cubin = bytearray(fixed)
    new_cubin[sec_off:sec_off+sec_sz] = new_text
    return bytes(new_cubin)


def build_stall(stall_on_consumer, val):
    base = compile_function(parse(we._K300_NASTY_PRED_XOR).functions[0], verbose=False, sm_version=120)
    fixed = apply_bug1_rename(base)
    sec_off, sec_sz = find_text(fixed)
    text = bytearray(fixed[sec_off:sec_off+sec_sz])
    if stall_on_consumer:
        idx = locate_pred_lop3(text)
    else:
        # stall on the P1-producing ISETP
        isetps = locate_isetp_pred(text)
        idx = [i for i, pd, _ in isetps if pd == 1][0]
    patch_ctrl(text, idx, stall=val)
    new_cubin = bytearray(fixed)
    new_cubin[sec_off:sec_off+sec_sz] = bytes(text)
    return bytes(new_cubin)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode.startswith('nop'):
            n = int(mode[3:])
            c = build_nop(n); run(c, f'nop{n}') if c else print(f'[nop{n}] not enough trailing NOPs')
        elif mode.startswith('cs'):
            n = int(mode[2:])
            c = build_stall(True, n); run(c, f'cons_stall{n}')
        elif mode.startswith('ps'):
            n = int(mode[2:])
            c = build_stall(False, n); run(c, f'prod_stall{n}')
    else:
        # Sweep
        for n in [1, 2, 4, 8, 16]:
            subprocess.run([sys.executable, __file__, f'nop{n}'])
        for n in [1, 2, 4, 8, 15, 32]:
            subprocess.run([sys.executable, __file__, f'cs{n}'])
        for n in [1, 2, 4, 8, 15, 32]:
            subprocess.run([sys.executable, __file__, f'ps{n}'])
