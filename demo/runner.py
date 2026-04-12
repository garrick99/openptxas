"""GPU execution and correctness verification via workbench."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import workbench


def run_full(kernel_name: str) -> dict | None:
    """Run kernel through workbench: compile both, run GPU, return all metrics.

    Returns dict with:
        'correctness': 'PASS'|'FAIL'
        'ours': {'regs', 'sass_total', 'sass_non_nop', 'compile_ms', ...}
        'ptxas': {'regs', 'sass_total', 'sass_non_nop', 'compile_ms', ...}
        'error': str|None
    """
    if kernel_name not in workbench.KERNELS:
        return None
    return workbench.measure_kernel(kernel_name, mode='correct',
                                     do_compare=True, repeat=1)


def get_ptx(kernel_name: str) -> str | None:
    """Get the PTX source for a workbench kernel."""
    if kernel_name not in workbench.KERNELS:
        return None
    entry = workbench.KERNELS[kernel_name]
    ptx = entry.get('ptx_inline')
    if ptx is None:
        path = entry.get('ptx_path')
        if path and Path(path).exists():
            ptx = Path(path).read_text(encoding='utf-8')
    return ptx
