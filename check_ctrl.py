#!/usr/bin/env python3
"""Decode all ctrl words and flag suspicious misc values."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../opencuda')

ptx_path = r'C:\Users\kraken\forge\demos\1019_fp64_bench.ptx'
with open(ptx_path) as f:
    ptx_src = f.read()

from ptx.parser import parse
from sass.regalloc import allocate
from sass.isel import select_function
from sass.pipeline import _sink_param_loads, _if_convert
from sass.scoreboard import assign_ctrl, _get_opcode, _OPCODE_MISC

mod = parse(ptx_src)
fn = mod.functions[0]

_sink_param_loads(fn)
_if_convert(fn)

alloc = allocate(fn, param_base=0x380, has_capmerc=True, sm_version=120)
raw_instrs = select_function(fn, alloc, sm_version=120)
scored = assign_ctrl(raw_instrs)

print(f"{'off':>6}  {'opc':>6}  {'stl':>4}  {'rbar':>6}  {'wdep':>6}  {'misc':>5}  note")
print("-" * 85)

for i, si in enumerate(scored):
    raw = si.raw
    opcode = _get_opcode(raw)
    b13, b14, b15 = raw[13], raw[14], raw[15]
    raw24 = (b15 & 0xFB) << 16 | b14 << 8 | b13
    ctrl = raw24 >> 1
    stall = (ctrl >> 17) & 0x3f
    rbar  = (ctrl >> 10) & 0x7f
    wdep  = (ctrl >>  4) & 0x3f
    misc  = ctrl & 0xF

    expected_misc = _OPCODE_MISC.get(opcode, None)
    flag = ''
    if expected_misc is not None and misc != expected_misc:
        flag = f'*** WRONG: expected {expected_misc}'
    elif expected_misc is None and misc not in (0, 1, 2):
        flag = f'<<< UNUSUAL misc={misc}'

    comment = si.comment[:38] if si.comment else ''
    print(f"{i*16:>6}  0x{opcode:03x}   {stall:>4}  0x{rbar:02x}    0x{wdep:02x}    {misc:>5}  {comment}  {flag}")
