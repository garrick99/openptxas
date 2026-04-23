"""R48 follow-up: sweep rbar values on @P1 LOP3 consumer to identify which
barriers fix the handoff."""
import sys, subprocess
sys.path.insert(0, 'C:/Users/kraken/openptxas')
sys.path.insert(0, 'C:/Users/kraken/openptxas/_r48')
from probe_handoff import build_variant, run, find_text, locate_pred_lop3, patch_ctrl, apply_bug1_rename
from ptx.parser import parse
from sass.pipeline import compile_function
import workbench_expanded as we

def build_rbar_variant(rbar_val):
    base = compile_function(parse(we._K300_NASTY_PRED_XOR).functions[0], verbose=False, sm_version=120)
    fixed = apply_bug1_rename(base)
    sec_off, sec_sz = find_text(fixed)
    text = bytearray(fixed[sec_off:sec_off+sec_sz])
    p1_idx = locate_pred_lop3(text)
    patch_ctrl(text, p1_idx, rbar=rbar_val)
    new_cubin = bytearray(fixed)
    new_cubin[sec_off:sec_off+sec_sz] = bytes(text)
    return bytes(new_cubin)


if __name__ == '__main__':
    import ctypes, struct
    if len(sys.argv) > 1:
        v = int(sys.argv[1])
        cubin = build_rbar_variant(v)
        run(cubin, f'rbar={v}')
    else:
        for v in range(0, 8):
            subprocess.run([sys.executable, __file__, str(v)])
