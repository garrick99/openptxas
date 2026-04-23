"""R49: trace where imm 0x4 disappears in k200_xor_reduce chain.

Plan: patch compile_function to intercept after each pass and dump any
LOP3.LUT(0x812) instruction bytes 4-7 (the imm field), to pinpoint the
pass that zeroes the imm."""
import sys, io, contextlib, re
sys.path.insert(0, 'C:/Users/kraken/openptxas')
import workbench_expanded as we
from ptx.parser import parse
from sass.pipeline import compile_function

# Simple approach: run with verbose=True and parse the stdout to see the
# lowering/compact passes printing instruction lists.
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    cubin = compile_function(parse(we._K200_XOR_REDUCE).functions[0], verbose=True, sm_version=120)
out = buf.getvalue()

# Find every stage that prints LOP3.LUT lines or any stage containing
# 0x1, 0x2, 0x4, 0x8 immediates in sequence.
lines = out.splitlines()
for i, line in enumerate(lines):
    if 'LOP3' in line or 'lop3' in line:
        print(f'{i:4d}: {line}')
