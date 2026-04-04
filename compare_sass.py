#!/usr/bin/env python3
"""Compare ptxas vs OpenPTXas raw SASS bytes for fp64_bench."""
import sys, struct
sys.path.insert(0, '.')

from tests.test_pipeline import ELF64

def get_sass(cubin_path):
    with open(cubin_path, 'rb') as f:
        cubin = f.read()
    elf = ELF64(cubin)
    return elf.section_data('.text.fp64_bench')

ptxas_sass = get_sass(r'C:\Users\kraken\openptxas\fp64_bench_ptxas.cubin')
openptxas_sass = get_sass(r'C:\Users\kraken\openptxas\fp64_bench.cubin')

def print_sass(sass, label):
    print(f"\n{label}: {len(sass)} bytes, {len(sass)//16} instrs")
    for i in range(0, len(sass), 16):
        if i + 16 > len(sass): break
        raw = sass[i:i+16]
        opcode = struct.unpack_from('<Q', raw, 0)[0] & 0xFFF
        b13, b14, b15 = raw[13], raw[14], raw[15]
        raw24 = (b15 & 0xFB) << 16 | b14 << 8 | b13
        ctrl = raw24 >> 1
        stall = (ctrl >> 17) & 0x3f
        rbar  = (ctrl >> 10) & 0x7f
        wdep  = (ctrl >>  4) & 0x3f
        misc  = ctrl & 0xF
        print(f"  +{i:4d}: {raw.hex()}  opc=0x{opcode:03x} stl={stall} rbar=0x{rbar:02x} wdep=0x{wdep:02x} misc={misc}")

print_sass(ptxas_sass, "ptxas SASS")
print_sass(openptxas_sass, "OpenPTXas SASS")
