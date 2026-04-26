"""
stress_runner.py -- single-machine GPU stress + correctness loop.

Runs the workbench's catalogued kernels (PTX-backed and optionally
Forge-backed) in a loop, watching for:

  - Kernel status FLIPs: kernel that PASSed in pass 1 fails in a later
    pass, with byte-identical SASS between the runs (hardware suspect).
  - ECC counter increments during the run (memory hardware issue).
  - Thermal throttle / power-cap excursions correlated with failures.

Default cadence is serial, single-worker.  No concurrent CUDA contexts.
"""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# nvidia-smi sidecar query -- one row per second.  Field names vary by
# nvidia-smi version; this set is the intersection that's reliably
# supported on driver 590+ (which we care about).  Throttle reasons used
# to live at "throttle_reasons.active" but the canonical name is
# "clocks_throttle_reasons.active" on some builds and missing on others;
# we drop it from the core query and add it conditionally below.
_TELEMETRY_QUERY = (
    "timestamp,index,name,"
    "ecc.errors.corrected.volatile.total,"
    "ecc.errors.uncorrected.volatile.total,"
    "temperature.gpu,"
    "clocks.gr,"
    "clocks.mem,"
    "power.draw,"
    "utilization.gpu,"
    "memory.used,"
    "memory.total"
)


def _start_telemetry(out_csv: Path, interval_s: int = 1):
    """Launch nvidia-smi as a background loop writing CSV every interval_s."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nvidia-smi",
        f"--query-gpu={_TELEMETRY_QUERY}",
        "--format=csv,nounits",
        "-l", str(interval_s),
    ]
    try:
        fh = open(out_csv, "w", newline="", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.DEVNULL)
        return proc, fh
    except FileNotFoundError:
        # nvidia-smi missing (rare on a CUDA host, but be graceful)
        print("[stress] WARN: nvidia-smi not found; telemetry disabled", flush=True)
        return None, None


def _stop_telemetry(proc, fh) -> None:
    if proc is not None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass


def _run_one_kernel(wb, name: str, pass_idx: int, results_dir: Path,
                    is_forge: bool, per_kernel_timeout_s: float = 10.0) -> dict:
    """Drive measure_kernel / measure_forge_kernel for a single kernel.

    Returns a record like:
      {"pass": 4, "kernel": "reduce_sum", "is_forge": False,
       "status": "PASS", "elapsed_s": 0.42}
    Possible status values:
      PASS         -- harness verified correctness
      FAIL         -- harness ran but output was wrong
      MISCOMPILE   -- compile/build failed
      RUNTIME      -- CUDA error / exception during launch
    """
    record = {
        "pass":     pass_idx,
        "kernel":   name,
        "is_forge": is_forge,
        "wall_ts":  datetime.now().isoformat(timespec="seconds"),
        "status":   "PENDING",
    }
    def _do_work():
        if is_forge:
            # Forge harness compiles via WSL and caches the .ptx into
            # results_dir; needs the path argument.
            return wb.measure_forge_kernel(
                target=name, mode="correct", do_compare=False,
                repeat=1, results_dir=results_dir,
            )
        # PTX-backed measure_kernel is a four-arg function in the
        # current workbench; it does not take a results_dir kwarg.
        return wb.measure_kernel(
            name=name, mode="correct", do_compare=False, repeat=1,
        )

    t0 = time.perf_counter()
    try:
        # Run inside a thread pool so a hung CUDA launch can't freeze the
        # whole loop.  The hung worker thread will continue running in the
        # background until process exit; the main loop moves on and records
        # the timeout as a RUNTIME event.  ctypes-based CUDA calls release
        # the GIL while blocked, so timeout enforcement actually works here.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"stress-{name}"
        ) as exe:
            future = exe.submit(_do_work)
            try:
                r = future.result(timeout=per_kernel_timeout_s)
            except concurrent.futures.TimeoutError:
                record["status"] = "RUNTIME"
                record["elapsed_s"] = round(time.perf_counter() - t0, 4)
                record["error"] = (
                    f"timeout after {per_kernel_timeout_s:.1f}s "
                    f"(kernel did not return; CUDA launch likely hung)"
                )
                # Don't wait for the hung worker; abandon the executor.
                # shutdown(wait=False, cancel_futures=True) lets the worker
                # die when the interpreter exits.
                exe.shutdown(wait=False, cancel_futures=True)
                return record
        elapsed = time.perf_counter() - t0
        correctness = (r.get("correctness") or "FAIL").upper()
        build = (r.get("build") or "FAIL").upper()
        record["status"] = correctness if build == "PASS" else "MISCOMPILE"
        record["build"]  = build
        record["elapsed_s"] = round(elapsed, 4)
        if r.get("error"):
            record["error"] = str(r["error"])[:300]
    except Exception as e:  # noqa: BLE001
        record["status"] = "RUNTIME"
        record["elapsed_s"] = round(time.perf_counter() - t0, 4)
        record["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return record


def _run_one_pass(wb, kernel_names: list[str], pass_idx: int,
                  results_dir: Path, include_forge: bool,
                  per_kernel_timeout_s: float = 10.0) -> list[dict]:
    """Run every selected kernel once in serial.  Returns per-kernel records."""
    forge_set = set(getattr(wb, "_FORGE_KERNELS", {}).keys())
    records: list[dict] = []
    for name in kernel_names:
        is_forge = name in forge_set
        if is_forge and not include_forge:
            continue
        records.append(_run_one_kernel(
            wb, name, pass_idx, results_dir, is_forge, per_kernel_timeout_s
        ))
    return records


def stress_loop(wb, kernel_names: list[str], out_dir: Path,
                duration_s: Optional[float] = None,
                max_passes: Optional[int] = None,
                include_forge: bool = False,
                bail_on_fail: bool = False,
                telemetry_interval_s: int = 1,
                per_kernel_timeout_s: float = 10.0) -> dict:
    """Main loop.  Runs until duration_s elapses, max_passes hit, or
    --bail-on-fail triggers a stop on first kernel-status flip.

    `wb` is the imported workbench module (so we can call its
    `measure_kernel` and `measure_forge_kernel` directly without
    re-implementing the harness pipeline).

    Writes:
      stress_<ts>.jsonl           -- one line per kernel-pass + per-pass summaries
      stress_<ts>_telemetry.csv   -- nvidia-smi sidecar output
      stress_<ts>_summary.json    -- final aggregate

    Returns the final summary dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path    = out_dir / f"stress_{ts}.jsonl"
    telemetry_csv = out_dir / f"stress_{ts}_telemetry.csv"
    summary_path  = out_dir / f"stress_{ts}_summary.json"

    forge_set = set(getattr(wb, "_FORGE_KERNELS", {}).keys())
    n_forge_in_run = sum(1 for k in kernel_names
                         if k in forge_set and include_forge)

    print(f"[stress] start    : {datetime.now().isoformat(timespec='seconds')}",
          flush=True)
    print(f"[stress] kernels  : {len(kernel_names)} catalogued, "
          f"{n_forge_in_run} forge-backed in this run", flush=True)
    print(f"[stress] log      : {jsonl_path}", flush=True)
    print(f"[stress] telemetry: {telemetry_csv}", flush=True)
    if duration_s:
        print(f"[stress] duration : {duration_s:.0f}s", flush=True)
    if max_passes:
        print(f"[stress] max pass : {max_passes}", flush=True)
    if bail_on_fail:
        print(f"[stress] mode     : bail-on-flip", flush=True)
    print("", flush=True)

    baseline: dict[str, str] = {}
    flips:    list[dict] = []
    failures_post_baseline: list[dict] = []
    pass_count = 0
    start_time = time.perf_counter()

    tel_proc, tel_fh = _start_telemetry(telemetry_csv, telemetry_interval_s)

    try:
        with open(jsonl_path, "w", encoding="utf-8") as log:
            # Header line
            log.write(json.dumps({
                "type":          "stress_header",
                "started":       datetime.now().isoformat(timespec="seconds"),
                "kernels":       kernel_names,
                "include_forge": include_forge,
                "duration_s":    duration_s,
                "max_passes":    max_passes,
                "bail_on_fail":  bail_on_fail,
            }) + "\n")
            log.flush()

            while True:
                pass_count += 1
                pass_t0 = time.perf_counter()
                records = _run_one_pass(wb, kernel_names, pass_count,
                                        out_dir, include_forge,
                                        per_kernel_timeout_s)
                pass_elapsed = time.perf_counter() - pass_t0

                pass_pass = sum(1 for r in records if r["status"] == "PASS")
                pass_fail = len(records) - pass_pass

                pass_flips: list[dict] = []
                if pass_count == 1:
                    for r in records:
                        baseline[r["kernel"]] = r["status"]
                else:
                    for r in records:
                        base = baseline.get(r["kernel"], "?")
                        if base == "PASS" and r["status"] != "PASS":
                            flip = {
                                "pass":     pass_count,
                                "kernel":   r["kernel"],
                                "baseline": base,
                                "now":      r["status"],
                                "error":    r.get("error"),
                                "wall_ts":  r["wall_ts"],
                            }
                            flips.append(flip)
                            pass_flips.append(flip)
                            failures_post_baseline.append(flip)
                        elif r["status"] != "PASS" and base != "PASS":
                            # never-passed kernel; not a hardware-flip signal
                            pass

                # Write per-kernel records + a pass summary
                for r in records:
                    log.write(json.dumps(r) + "\n")
                log.write(json.dumps({
                    "type":        "pass_summary",
                    "pass":        pass_count,
                    "elapsed_s":   round(pass_elapsed, 2),
                    "pass_count":  pass_pass,
                    "fail_count":  pass_fail,
                    "flips":       len(pass_flips),
                    "wall_ts":     datetime.now().isoformat(timespec="seconds"),
                }) + "\n")
                log.flush()

                line = (f"[stress] pass {pass_count:>4}: "
                        f"{pass_pass:>3}/{len(records):<3} PASS, "
                        f"{pass_fail} non-PASS, "
                        f"{len(pass_flips)} flips, "
                        f"{pass_elapsed:.1f}s")
                print(line, flush=True)

                for f in pass_flips:
                    msg = (f"  [FLIP] kernel={f['kernel']}  "
                           f"{f['baseline']} -> {f['now']}")
                    if f.get("error"):
                        msg += f"  err={f['error'][:120]}"
                    print(msg, flush=True)

                if bail_on_fail and pass_flips:
                    print(f"[stress] --bail-on-fail tripped on pass {pass_count}; "
                          f"stopping.", flush=True)
                    break
                if max_passes is not None and pass_count >= max_passes:
                    break
                if duration_s is not None and (time.perf_counter() - start_time) >= duration_s:
                    break
    finally:
        _stop_telemetry(tel_proc, tel_fh)

    total_elapsed = time.perf_counter() - start_time

    # Did baseline have any non-PASS kernels?  Note them so the summary
    # distinguishes "never passed" from "passed then failed."
    baseline_failures = [k for k, s in baseline.items() if s != "PASS"]

    summary = {
        "started":              datetime.now().isoformat(timespec="seconds"),
        "duration_s":           round(total_elapsed, 1),
        "passes":               pass_count,
        "kernels":              kernel_names,
        "include_forge":        include_forge,
        "baseline_pass_count":  sum(1 for s in baseline.values() if s == "PASS"),
        "baseline_failures":    baseline_failures,
        "n_flips":              len(flips),
        "flips":                flips[:200],
        "n_failures_post_baseline": len(failures_post_baseline),
        "verdict":              "ANOMALY" if flips else "CLEAN",
        "log":                  str(jsonl_path),
        "telemetry_csv":        str(telemetry_csv),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("", flush=True)
    print(f"[stress] === Summary ===", flush=True)
    print(f"  passes              : {pass_count}", flush=True)
    print(f"  duration            : {total_elapsed:.1f}s", flush=True)
    print(f"  baseline PASS count : {summary['baseline_pass_count']} / {len([k for k in kernel_names if (k in (getattr(wb, '_FORGE_KERNELS', {}) or {})) <= include_forge or (k not in (getattr(wb, '_FORGE_KERNELS', {}) or {})) ])}", flush=True)
    if baseline_failures:
        print(f"  baseline failures   : {', '.join(baseline_failures)}", flush=True)
    print(f"  status flips        : {len(flips)}", flush=True)
    print(f"  verdict             : {summary['verdict']}", flush=True)
    if flips:
        print(f"\n  Flip log:", flush=True)
        for f in flips[:20]:
            err = f"  ({f['error'][:80]})" if f.get("error") else ""
            print(f"    pass {f['pass']:>4}  {f['kernel']:<32} {f['baseline']:>8} -> {f['now']:<10}{err}",
                  flush=True)
        if len(flips) > 20:
            print(f"    ... and {len(flips) - 20} more (full list in summary JSON)", flush=True)

    return summary
