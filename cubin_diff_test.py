"""
cubin_diff_test.py -- on a single machine, compile a kernel with both
OpenPTXas and ptxas, save both cubins, and run each in an isolated
subprocess so a hang in one doesn't poison the test of the other.

The point is to control for hardware: same GPU, two SASS streams.  If
both cubins hang -> hardware is suspect.  If only the OpenPTXas cubin
hangs -> backend (SASS-emission) bug, not silicon.

Between subprocess runs we PnP-cycle the GPU if it's left in a stuck
state (high util / no processes).  Requires admin context for the cycle.

Usage:
    python cubin_diff_test.py <kernel_name> [--timeout 15]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import workbench  # noqa: E402
from benchmarks.bench_util import compile_ptxas  # noqa: E402

PNP_INSTANCE_ID = (
    r"PCI\VEN_10DE&DEV_2B85&SUBSYS_416F1458&REV_A1\4E77A905F92DB04800"
)


def gpu_is_stuck() -> bool:
    """Returns True if the GPU is reporting high utilization with no
    processes attached (the post-hang stuck-state signature)."""
    try:
        u = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        p = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        util_str, mem_str = [x.strip() for x in u.stdout.split(",")]
        util = int(util_str)
        mem  = int(mem_str)
        n_procs = len([l for l in p.stdout.splitlines() if l.strip()])
        return util > 50 and mem == 0 and n_procs == 0
    except Exception:
        return False


def cycle_gpu() -> bool:
    print("    [GPU stuck -- cycling via PnP disable/enable]", flush=True)
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f'Disable-PnpDevice -InstanceId "{PNP_INSTANCE_ID}" -Confirm:$false'],
            capture_output=True, timeout=30,
        )
        time.sleep(3)
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f'Enable-PnpDevice -InstanceId "{PNP_INSTANCE_ID}" -Confirm:$false'],
            capture_output=True, timeout=30,
        )
        time.sleep(5)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    cycle FAILED: {type(e).__name__}: {e}", flush=True)
        return False


def run_cubin_subprocess(kernel: str, cubin_path: Path,
                         timeout_s: float, label: str) -> dict:
    """Spawn run_cubin.py in a fresh subprocess.  Capture result."""
    cmd = [sys.executable, str(ROOT / "run_cubin.py"),
           kernel, str(cubin_path)]
    print(f"\n=== run {label} cubin ===", flush=True)
    print(f"cmd: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout_s, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"*** {label}: HUNG after {elapsed:.1f}s (subprocess timeout) ***",
              flush=True)
        # Subprocess force-killed by Python's timeout; the CUDA work that
        # was hung is now orphaned and the GPU may be stuck.
        return {"label": label, "status": "HANG",
                "elapsed_s": round(elapsed, 2), "rc": -1,
                "stdout": "", "stderr": ""}

    elapsed = time.perf_counter() - t0
    print(f"{label}: rc={p.returncode}  elapsed={elapsed:.2f}s", flush=True)
    print(f"--- stdout ---\n{p.stdout.strip()}", flush=True)
    if p.stderr.strip():
        print(f"--- stderr ---\n{p.stderr.strip()[:800]}", flush=True)
    status = "PASS" if (p.returncode == 0 and "PASS" in p.stdout) else \
             "FAIL" if "FAIL" in p.stdout else \
             "UNKNOWN"
    return {"label": label, "status": status,
            "elapsed_s": round(elapsed, 2), "rc": p.returncode,
            "stdout": p.stdout, "stderr": p.stderr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kernel", help="kernel name (must exist in workbench.KERNELS)")
    p.add_argument("--timeout", type=float, default=15.0,
                   help="per-cubin subprocess timeout in seconds (default: 15)")
    args = p.parse_args()

    if args.kernel not in workbench.KERNELS:
        print(f"unknown kernel: {args.kernel}")
        sys.exit(2)

    kentry = workbench.KERNELS[args.kernel]
    if kentry["ptx_inline"] is not None:
        ptx = kentry["ptx_inline"]
    else:
        ptx = kentry["ptx_path"].read_text(encoding="utf-8")

    print(f"kernel: {args.kernel}")
    print(f"timeout: {args.timeout}s per cubin")

    print("\n[1] compiling with OpenPTXas...", flush=True)
    cubin_ours, t_ours, _ = workbench.compile_with_report(ptx)
    out_dir = ROOT / "stress_runs"
    out_dir.mkdir(exist_ok=True)
    cubin_ours_path = out_dir / f"_cubin_{args.kernel}_ours.cubin"
    cubin_ours_path.write_bytes(cubin_ours)
    print(f"    wrote {len(cubin_ours)} bytes to {cubin_ours_path}  ({t_ours*1000:.1f}ms compile)")

    print("\n[2] compiling with ptxas...", flush=True)
    cubin_ptxas, t_ptxas = compile_ptxas(ptx)
    cubin_ptxas_path = out_dir / f"_cubin_{args.kernel}_ptxas.cubin"
    cubin_ptxas_path.write_bytes(cubin_ptxas)
    print(f"    wrote {len(cubin_ptxas)} bytes to {cubin_ptxas_path}  ({t_ptxas*1000:.1f}ms compile)")

    results = []

    # Run OpenPTXas cubin first
    r1 = run_cubin_subprocess(args.kernel, cubin_ours_path,
                              args.timeout, "OpenPTXas")
    results.append(r1)

    # If GPU is stuck, cycle before testing ptxas cubin
    if r1["status"] == "HANG" or gpu_is_stuck():
        cycle_gpu()
        time.sleep(2)
        if gpu_is_stuck():
            print("    [GPU still stuck after cycle; aborting ptxas test]")
            results.append({"label": "ptxas", "status": "SKIPPED",
                            "elapsed_s": 0.0, "rc": -2, "stdout": "", "stderr": ""})
        else:
            r2 = run_cubin_subprocess(args.kernel, cubin_ptxas_path,
                                      args.timeout, "ptxas")
            results.append(r2)
    else:
        r2 = run_cubin_subprocess(args.kernel, cubin_ptxas_path,
                                  args.timeout, "ptxas")
        results.append(r2)

    # Final summary
    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['label']:<10} : {r['status']:<8}  {r['elapsed_s']:>6.2f}s  rc={r['rc']}")

    # Diagnostic verdict
    s_ours  = next((r["status"] for r in results if r["label"] == "OpenPTXas"), "?")
    s_ptxas = next((r["status"] for r in results if r["label"] == "ptxas"),     "?")

    print("\n=== Verdict ===")
    if s_ours == "HANG" and s_ptxas == "PASS":
        print("  ONLY OpenPTXas cubin hangs.")
        print("  -> Hardware appears healthy.  Bug is in OpenPTXas SASS emission.")
    elif s_ours == "HANG" and s_ptxas == "HANG":
        print("  BOTH cubins hang on this GPU.")
        print("  -> Strong hardware-fault signal: even ptxas-emitted SASS doesn't")
        print("     run cleanly here.  Cross-machine confirmation recommended.")
    elif s_ours == "PASS" and s_ptxas == "PASS":
        print("  Both cubins PASS.")
        print("  -> Could not reproduce the hang in this run.  Try multiple iterations.")
    elif s_ours == "PASS" and s_ptxas == "HANG":
        print("  Only ptxas cubin hangs (unusual).")
        print("  -> Worth investigating; may be ptxas/driver interaction.")
    else:
        print(f"  Inconclusive: OpenPTXas={s_ours}, ptxas={s_ptxas}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
