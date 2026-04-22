"""Outer driver for differential fuzzing.  Runs N parallel workers;
respawns each on crash; prints periodic stats; logs summary on
Ctrl-C or time limit.

Usage:
    python -m fuzzer.run_fuzz [--seconds 60] [--workers 4]
"""
import argparse, ctypes, json, os, queue, random, signal, subprocess, sys, threading, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fuzzer.monitor import gpu_stats


# Windows priority classes.  IDLE = only runs when no other process wants
# CPU.  Use this so gaming / dev work is never even slightly contended.
IDLE_PRIORITY_CLASS = 0x00000040
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

# Core affinity: cap how many CPU cores the fuzzer can use so background
# work is never starved.  bit N set = core N usable.  0x3F = cores 0..5.
_CPU_AFFINITY_MASK = 0x3F  # 6 cores out of 24


def _nice_self():
    if os.name != 'nt':
        try: os.nice(19)
        except Exception: pass
        return
    try:
        kernel32 = ctypes.WinDLL('kernel32')
        h = kernel32.GetCurrentProcess()
        kernel32.SetPriorityClass(h, IDLE_PRIORITY_CLASS)
        kernel32.SetProcessAffinityMask(h, ctypes.c_size_t(_CPU_AFFINITY_MASK))
    except Exception:
        pass

_nice_self()


def spawn_worker(seed_start: int, iters: int, family: str = 'alu_int'):
    """Spawn worker as subprocess with IDLE priority + limited affinity."""
    flags = IDLE_PRIORITY_CLASS if os.name == 'nt' else 0
    p = subprocess.Popen(
        [sys.executable, '-m', 'fuzzer.worker',
         str(seed_start), str(iters), family],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, creationflags=flags)
    # Cap worker's affinity too
    if os.name == 'nt':
        try:
            kernel32 = ctypes.WinDLL('kernel32')
            PROCESS_ALL_ACCESS = 0x1F0FFF
            h = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, p.pid)
            if h:
                kernel32.SetProcessAffinityMask(h, ctypes.c_size_t(_CPU_AFFINITY_MASK))
                kernel32.CloseHandle(h)
        except Exception:
            pass
    return p


def _worker_loop(worker_id, seed_box, iters_per_spawn, deadline, event_q, stop,
                  family='alu_int'):
    """One thread: spawn-and-drain workers until deadline."""
    while time.time() < deadline and not stop.is_set():
        with seed_box['lock']:
            seed_start = seed_box['cursor']
            seed_box['cursor'] += iters_per_spawn
        w = spawn_worker(seed_start, iters_per_spawn, family=family)
        for line in w.stdout:
            if stop.is_set(): break
            line = line.strip()
            if not line: continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            ev['worker'] = worker_id
            event_q.put(ev)
            if time.time() >= deadline:
                break
        try: w.wait(timeout=2)
        except subprocess.TimeoutExpired: w.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=int, default=60)
    ap.add_argument('--workers', type=int, default=1,
                     help='parallel worker subprocesses. KEEP AT 1 ON BIGDADDY — '
                          'parallel CUDA contexts on Windows WDDM hard-hang the box.')
    ap.add_argument('--iters-per-worker', type=int, default=200,
                     help='iters each worker runs before respawn')
    ap.add_argument('--seed-base', type=int, default=random.randint(0, 1<<30))
    ap.add_argument('--family', default='alu_int',
                     help='generator family (alu_int|warp|bitmanip)')
    args = ap.parse_args()

    start = time.time()
    deadline = start + args.seconds
    seed_box = {'cursor': args.seed_base, 'lock': threading.Lock()}
    counts = Counter()
    new_findings = Counter()
    total_iters = 0
    event_q = queue.Queue()
    stop = threading.Event()

    print(f'[fuzz] starting  family={args.family}  seed_base={args.seed_base}  '
          f'budget={args.seconds}s  workers={args.workers}')
    print(f'[fuzz] priority: IDLE  |  affinity mask: 0x{_CPU_AFFINITY_MASK:X}  '
          f'(cores {bin(_CPU_AFFINITY_MASK).count("1")}/{os.cpu_count()})')
    print()

    threads = []
    for wid in range(args.workers):
        t = threading.Thread(target=_worker_loop,
                              args=(wid, seed_box, args.iters_per_worker,
                                    deadline, event_q, stop, args.family),
                              daemon=True)
        t.start()
        threads.append(t)

    last_print = start
    last_iter_at_print = 0
    try:
        while time.time() < deadline:
            try:
                ev = event_q.get(timeout=0.5)
            except queue.Empty:
                continue
            status = ev.get('status', '?')
            counts[status] += 1
            if ev.get('new'):
                new_findings[status] += 1
                s = gpu_stats()
                print(f'  [!] NEW {status}  seed={ev.get("seed")}  '
                      f'GPU={s.get("util_pct","?")}%  '
                      f'T={s.get("temp_c","?")}°C  '
                      f'P={s.get("power_w","?")}W')
            total_iters += 1
            now = time.time()
            if now - last_print >= 3.0:
                rate = (total_iters - last_iter_at_print) / (now - last_print)
                s = gpu_stats()
                elapsed = int(now - start)
                print(f'  [{elapsed:4d}s] iters={total_iters:>6d}  '
                      f'{rate:>5.0f}/s  ok={counts["ok"]}  '
                      f'div={counts["divergence"]}  '
                      f'se_ours={counts["sync_err_ours"]}  '
                      f'ctx_dead={counts["ctx_dead"]}  '
                      f'GPU={s.get("util_pct","?")}%  '
                      f'T={s.get("temp_c","?")}°C  '
                      f'P={s.get("power_w","?")}W')
                last_print = now
                last_iter_at_print = total_iters
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=3)

    # Final summary
    elapsed = time.time() - start
    print()
    print(f'[fuzz] done  {elapsed:.1f}s  {total_iters} iterations  '
           f'{total_iters/elapsed:.1f}/sec  workers={args.workers}')
    print(f'[fuzz] status breakdown:')
    for k in sorted(counts):
        n = counts[k]
        newn = new_findings[k]
        marker = f'  ({newn} NEW)' if newn else ''
        print(f'    {k:20s} {n:>6d}{marker}')
    if new_findings:
        print()
        print(f'[fuzz] artifacts at {Path(__file__).resolve().parent.parent}/_fuzz/artifacts/')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[fuzz] interrupted')
