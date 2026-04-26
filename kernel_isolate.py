"""
kernel_isolate.py -- run each catalogued kernel in its own Python
subprocess with a hard timeout to identify any kernel that hangs in
isolation.

Each kernel runs via `python workbench.py run --kernel <name>` as a
brand-new process with a fresh CUDA context.  A wall-clock timeout
catches kernels that hang inside the driver (which the in-process
threading approach in stress_runner.py couldn't recover from -- a
worker thread blocked in a CUDA call holds the executor open
indefinitely).

If a hang is detected, the runner can optionally cycle the GPU via
PnP disable/enable to recover before moving to the next kernel.
This keeps the loop going so we can survey all kernels rather than
stopping at the first hang.

Output:
    stress_runs/isolate_<ts>.csv   -- per-kernel result row

Usage:
    python kernel_isolate.py
    python kernel_isolate.py --timeout 20 --max-hangs 5 --no-gpu-cycle
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import workbench  # noqa: E402

# 5090 PnP instance ID on this machine; cycling unbinds the driver and
# rebinds it, which clears CUDA state without a full reboot.
PNP_INSTANCE_ID = (
    r"PCI\VEN_10DE&DEV_2B85&SUBSYS_416F1458&REV_A1\4E77A905F92DB04800"
)


def cycle_gpu(verbose: bool = True) -> bool:
    """Run Disable-PnpDevice + Enable-PnpDevice for the 5090.  Requires
    admin context.  Returns True on success."""
    if verbose:
        print("    cycling GPU ...", end=" ", flush=True)
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
        if verbose:
            print("ok")
        return True
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"FAILED: {type(e).__name__}: {e}")
        return False


def run_one_kernel(name: str, timeout_s: float) -> dict:
    """Spawn a subprocess that runs the kernel, with a wall-clock
    timeout.  Parses workbench's printed output for PASS/FAIL."""
    cmd = [
        sys.executable, str(ROOT / "workbench.py"),
        "run", "--kernel", name, "--mode", "correct",
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return {
            "kernel":     name,
            "status":     "HANG",
            "elapsed_s":  round(timeout_s, 2),
            "returncode": -1,
            "error":      f"subprocess timeout after {timeout_s:.1f}s",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "kernel":     name,
            "status":     "ERROR",
            "elapsed_s":  round(time.perf_counter() - t0, 2),
            "returncode": -1,
            "error":      f"{type(e).__name__}: {str(e)[:200]}",
        }

    elapsed = time.perf_counter() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    # Workbench's `run` command prints lines like:
    #     build:    PASS
    #     correct:  PASS
    # We match those.  (Also tolerate "correctness:" in case the
    # output format diverges in a future workbench version.)
    if re.search(r"\bcorrect(?:ness)?\s*[:=]\s*PASS\b", out):
        status = "PASS"
    elif re.search(r"\bcorrect(?:ness)?\s*[:=]\s*FAIL\b", out):
        status = "FAIL"
    elif re.search(r"\bbuild\s*[:=]\s*FAIL\b", out):
        status = "MISCOMPILE"
    elif proc.returncode != 0:
        status = "STARTUP_FAIL"
    else:
        status = "UNKNOWN"

    err_excerpt = ""
    if status != "PASS":
        # Try to surface a useful one-line error excerpt
        for pat in (r"error[:\s][^\n]+", r"FAILED[^\n]+",
                    r"Traceback[^\n]+\n[^\n]+"):
            m = re.search(pat, out, re.IGNORECASE)
            if m:
                err_excerpt = m.group(0)[:200].replace("\n", " ")
                break

    return {
        "kernel":     name,
        "status":     status,
        "elapsed_s":  round(elapsed, 2),
        "returncode": proc.returncode,
        "error":      err_excerpt,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=float, default=30.0,
                   help="per-kernel subprocess timeout in seconds (default: 30)")
    p.add_argument("--max-hangs", type=int, default=10,
                   help="abort after this many hangs total (default: 10)")
    p.add_argument("--no-gpu-cycle", action="store_true",
                   help="don't PnP-cycle the GPU after a hang")
    p.add_argument("--kernels", default=None,
                   help="comma-separated subset (default: all PTX-backed)")
    p.add_argument("--start-at", default=None,
                   help="resume from this kernel name")
    args = p.parse_args()

    if args.kernels:
        names = [n.strip() for n in args.kernels.split(",") if n.strip()]
    else:
        names = list(workbench.KERNELS.keys())

    if args.start_at:
        if args.start_at not in names:
            print(f"--start-at {args.start_at!r} not in kernel list")
            return 1
        names = names[names.index(args.start_at):]

    out_dir = ROOT / "stress_runs"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"isolate_{ts}.csv"

    print(f"[isolate] start    : {datetime.now().isoformat(timespec='seconds')}")
    print(f"[isolate] kernels  : {len(names)}")
    print(f"[isolate] timeout  : {args.timeout:.1f}s per kernel")
    print(f"[isolate] gpu cycle: {'no' if args.no_gpu_cycle else 'yes (after each hang)'}")
    print(f"[isolate] csv      : {csv_path}")
    print()

    hangs:    list[str] = []
    fails:    list[str] = []
    miscomp:  list[str] = []
    passes:   list[str] = []
    other:    list[tuple[str, str]] = []  # kernel, status
    aborted = False

    fields = ["idx", "kernel", "status", "elapsed_s", "returncode", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        fh.flush()

        for i, name in enumerate(names, 1):
            print(f"[{i:>3}/{len(names)}] {name:<32}", end=" ", flush=True)
            r = run_one_kernel(name, args.timeout)
            row = {"idx": i, **r}
            w.writerow(row)
            fh.flush()

            mark = {
                "PASS":         "PASS",
                "FAIL":         "FAIL",
                "MISCOMPILE":   "MCMP",
                "HANG":         "HANG",
                "ERROR":        " ERR",
                "STARTUP_FAIL": "STRT",
                "UNKNOWN":      " ???",
            }.get(r["status"], " ???")
            err_tag = f"  [{r['error'][:60]}]" if r["error"] else ""
            print(f"{mark}  {r['elapsed_s']:>5.1f}s{err_tag}")

            if r["status"] == "HANG":
                hangs.append(name)
                if not args.no_gpu_cycle:
                    cycle_gpu()
                if len(hangs) >= args.max_hangs:
                    print(f"\n[isolate] hit --max-hangs={args.max_hangs}; aborting")
                    aborted = True
                    break
            elif r["status"] == "FAIL":
                fails.append(name)
            elif r["status"] == "MISCOMPILE":
                miscomp.append(name)
            elif r["status"] == "PASS":
                passes.append(name)
            else:
                other.append((name, r["status"]))

    print()
    print(f"[isolate] === Summary ===")
    print(f"  total processed : {len(passes) + len(fails) + len(miscomp) + len(hangs) + len(other)}")
    print(f"  PASS            : {len(passes)}")
    print(f"  FAIL            : {len(fails)}")
    print(f"  MISCOMPILE      : {len(miscomp)}")
    print(f"  HANG            : {len(hangs)}")
    print(f"  STARTUP_FAIL/?? : {len(other)}")
    print(f"  aborted early   : {aborted}")
    if hangs:
        print(f"\n  Hung kernels (in isolation -- single subprocess):")
        for k in hangs:
            print(f"    - {k}")
    if miscomp:
        print(f"\n  Miscompile (build failed):")
        for k in miscomp:
            print(f"    - {k}")
    print(f"\n  CSV: {csv_path}")
    return 0 if not hangs else 1


if __name__ == "__main__":
    sys.exit(main())
