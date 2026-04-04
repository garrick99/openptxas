#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../opencuda')

ptx_path = r'C:\Users\kraken\forge\demos\1019_fp64_bench.ptx'
with open(ptx_path) as f:
    ptx_src = f.read()

from sass.pipeline import compile_ptx_source
results = compile_ptx_source(ptx_src, verbose=True)
cubin = results['fp64_bench']
print(f'cubin: {len(cubin)} bytes')

# Save new cubin
with open('fp64_bench.cubin', 'wb') as f:
    f.write(cubin)
print('Saved fp64_bench.cubin')
