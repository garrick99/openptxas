"""R46: dump OpenPTXas SASS for the 3 ALU cluster kernels, compare to ptxas."""
import sys, subprocess, tempfile, os, struct
sys.path.insert(0, 'C:/Users/kraken/openptxas')
import workbench as wb
import workbench_expanded as we
from ptx.parser import parse
from sass.pipeline import compile_function

NVDISASM = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvdisasm.exe'
PTXAS    = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\ptxas.exe'

KERNELS = {
    'ilp_pred_alu':         wb._PTX_ILP_PRED_ALU,
    'k200_xor_reduce':      we._K200_XOR_REDUCE,
    'k300_nasty_pred_xor':  we._K300_NASTY_PRED_XOR,
    # canary: passing deep ALU chain
    'k200_deep_alu':        we._K200_DEEP_ALU,
}

def disasm(cubin_bytes, label):
    with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
        f.write(cubin_bytes); path = f.name
    try:
        r = subprocess.run([NVDISASM, '-c', '-hex', path], capture_output=True, text=True)
        print(f'\n=== {label} ===')
        # keep only lines in the kernel body
        in_body = False
        for line in r.stdout.splitlines():
            if '.text.' in line: in_body = True
            if in_body:
                print(line)
    finally:
        os.unlink(path)

def ptxas_cubin(ptx_str, kname):
    with tempfile.NamedTemporaryFile(suffix='.ptx', delete=False, mode='w') as f:
        f.write(ptx_str); ptx_path = f.name
    cubin_path = ptx_path.replace('.ptx', '.cubin')
    try:
        r = subprocess.run([PTXAS, '-arch=sm_120', ptx_path, '-o', cubin_path],
                            capture_output=True, text=True)
        if r.returncode != 0:
            print(f'[{kname}] ptxas err: {r.stderr}'); return None
        with open(cubin_path, 'rb') as f: return f.read()
    finally:
        os.unlink(ptx_path)
        if os.path.exists(cubin_path): os.unlink(cubin_path)

for kname, ptx in KERNELS.items():
    ours = compile_function(parse(ptx).functions[0], verbose=False, sm_version=120)
    theirs = ptxas_cubin(ptx, kname)
    disasm(ours, f'OURS {kname}')
    if theirs: disasm(theirs, f'PTXAS {kname}')
