"""Verify: does driver-JIT cubin launch produce the same output as subprocess-ptxas cubin?"""
import ctypes, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fuzzer.oracle import compile_theirs, CudaRunner
from factory.bench_jit import compile_theirs_jit
from factory.supervisor import _BUG1_PTX, _BUG2_PTX
from factory.differ_d import _canonical_input

inp = _canonical_input()

runner = CudaRunner()

for name, ptx in [('bug1', _BUG1_PTX), ('bug2', _BUG2_PTX)]:
    c_sub, _ = compile_theirs(ptx)
    c_jit, _ = compile_theirs_jit(ptx)
    out_sub, e_sub = runner.run_cubin(c_sub, inp, 32)
    out_jit, e_jit = runner.run_cubin(c_jit, inp, 32)
    print(f'{name}:')
    print(f'  subprocess err={e_sub} cubin={len(c_sub)}B  out[0..3]={[hex(struct.unpack_from("<I",out_sub,i*4)[0]) for i in range(4)]}')
    print(f'  driverJIT  err={e_jit} cubin={len(c_jit)}B  out[0..3]={[hex(struct.unpack_from("<I",out_jit,i*4)[0]) for i in range(4)]}')
    print(f'  outputs equal? {out_sub == out_jit}')
