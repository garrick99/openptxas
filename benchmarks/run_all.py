"""
Run all OpenPTXas vs NVIDIA ptxas benchmarks and print a summary table.

Usage: python benchmarks/run_all.py
"""
import importlib
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BENCHMARKS = [
    ("vecadd",    "vecadd_vs_nvidia",    "GB/s"),
    ("saxpy",     "saxpy_vs_nvidia",     "GB/s"),
    ("memcpy",    "memcpy_vs_nvidia",    "GB/s"),
    ("scale",     "scale_vs_nvidia",     "GB/s"),
    ("stencil",   "stencil_vs_nvidia",   "GB/s"),
    ("relu",      "relu_vs_nvidia",      "GB/s"),
    ("fma_chain", "fmachain_vs_nvidia",  "GFLOPS"),
]

# vecadd doesn't use bench_util; it returns via its own path. Handle it specially.


def capture_main(mod):
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = mod.main()
    return buf.getvalue(), res


def main():
    print("\n" + "#" * 68)
    print("  OpenPTXas Benchmark Suite")
    print("  GPU: RTX 5090 (SM_120)")
    print("#" * 68)

    results = []
    for name, modname, unit in BENCHMARKS:
        print(f"\n>>> Running {name} <<<")
        mod = importlib.import_module(modname)
        mod.main.__module__ = modname  # sanity
        try:
            out, ret = capture_main(mod)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((name, None, None, False, unit))
            continue
        print(out)
        if ret is None:
            # vecadd's main returns None; parse from output instead
            ours = nvid = 0.0
            correct = "identical" in out
            for line in out.splitlines():
                if "Mem bandwidth" in line:
                    parts = line.split()
                    try:
                        ours = float(parts[2])
                        nvid = float(parts[4])
                    except (ValueError, IndexError):
                        pass
        else:
            ours, nvid, correct = ret
        results.append((name, ours, nvid, correct, unit))

    # Summary
    print("\n" + "#" * 68)
    print("  SUMMARY — OpenPTXas vs NVIDIA ptxas (RTX 5090, SM_120)")
    print("#" * 68)
    fmt = "  {:<12} {:>12}  {:>12}  {:>8}  {:>8}"
    print(fmt.format("Benchmark", "OpenPTXas", "NVIDIA", "Ratio", "Status"))
    print("-" * 68)
    for name, ours, nvid, correct, unit in results:
        if ours is None:
            print(fmt.format(name, "—", "—", "—", "FAIL"))
            continue
        ours_s = f"{ours:.1f} {unit}"
        nvid_s = f"{nvid:.1f} {unit}"
        ratio = f"{ours/max(nvid,0.001):.2f}x"
        status = "PASS" if correct else "WRONG"
        print(fmt.format(name, ours_s, nvid_s, ratio, status))
    print("-" * 68)

    # Geomean
    import math
    ratios = [ours/nvid for _, ours, nvid, c, _ in results
              if ours is not None and nvid is not None and c]
    if ratios:
        geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        print(f"  Geomean perf ratio (passing): {geo:.3f}x of NVIDIA ptxas")
    print("#" * 68)


if __name__ == '__main__':
    main()
