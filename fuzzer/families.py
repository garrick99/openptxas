"""Family registry: name -> generate function.

Each generator module exports `generate(seed) -> (ptx_text, last_reg_idx)`.
All families share the same fuzz-kernel shape (one b32 input, one b32
store) so the oracle and minimizer are family-agnostic.
"""
from fuzzer import generator as _alu_int
from fuzzer import generator_warp as _warp
from fuzzer import generator_bitmanip as _bitmanip

REGISTRY = {
    'alu_int':  _alu_int.generate,
    'warp':     _warp.generate,
    'bitmanip': _bitmanip.generate,
}

DEFAULT = 'alu_int'


def generate(family: str, seed: int):
    if family not in REGISTRY:
        raise ValueError(f'unknown family {family!r}; known: {sorted(REGISTRY)}')
    return REGISTRY[family](seed)


def all_families() -> list[str]:
    return list(REGISTRY.keys())
