"""
run_cubin.py -- load a precompiled cubin and run a workbench kernel's
harness against it.  Used to test a specific cubin (regardless of which
SASS emitter produced it) on whatever GPU this script runs on.

Usage:
    python run_cubin.py <kernel_name> <cubin_path>

Exit code:
    0 = harness PASS
    1 = harness FAIL or load failure
    2 = bad arguments
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import workbench  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print("usage: run_cubin.py <kernel_name> <cubin_path>")
        return 2

    name = sys.argv[1]
    cubin_path = Path(sys.argv[2])

    if name not in workbench.KERNELS:
        print(f"unknown kernel: {name}")
        return 1
    if not cubin_path.exists():
        print(f"cubin not found: {cubin_path}")
        return 1

    kentry = workbench.KERNELS[name]
    cubin = cubin_path.read_bytes()

    print(f"kernel:       {name}")
    print(f"symbol:       {kentry['kernel_name']}")
    print(f"cubin path:   {cubin_path}")
    print(f"cubin size:   {len(cubin)} bytes")

    # Quick metrics so we can sanity-check the cubin content
    try:
        m = workbench.metrics_from_cubin(cubin)
        print(f"metrics:      regs={m.get('regs')} sass_total={m.get('sass_total')}"
              f" sass_non_nop={m.get('sass_non_nop')}")
    except Exception as e:  # noqa: BLE001
        print(f"metrics:      failed to extract -- {type(e).__name__}: {e}")

    print("loading via cuModuleLoadData...", flush=True)
    ctx = workbench.CUDAContext()
    try:
        if not ctx.load(cubin):
            print("FAIL: cuModuleLoadData failed")
            return 1
        func = ctx.get_func(kentry["kernel_name"])
        print("launching harness...", flush=True)
        r = kentry["harness"](ctx, func, "correct")
        if r.get("correct"):
            print("PASS")
            return 0
        print(f"FAIL: harness returned {r}")
        return 1
    finally:
        ctx.close()


if __name__ == "__main__":
    sys.exit(main())
