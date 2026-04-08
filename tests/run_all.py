#!/usr/bin/env python
"""Run all test files, isolating GPU tests in separate processes.

SM_120 UR register cache leaks across module loads within a single
CUDA context, causing false failures when multiple GPU test modules
share one process.  This runner executes each GPU test file in its
own subprocess, guaranteeing clean UR state.

Usage:  python tests/run_all.py
"""
import subprocess
import sys
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)

# Non-GPU test files — safe to run together in one invocation
NON_GPU = [
    'test_capmerc_gen.py',
    'test_new_encoders.py',
    'test_nvdisasm_roundtrip.py',
    'test_opcodes.py',
    'test_parser.py',
    'test_patcher.py',
    'test_pipeline.py',
    'test_regalloc.py',
    'test_rotate_pass.py',
    'test_scoreboard_regression.py',
    'test_tex_surf.py',
    'test_tma_parser.py',
]

# GPU test files — each gets its own subprocess
GPU_FILES = [
    'test_gpu_coverage.py',
    'test_gpu_coverage2.py',
    'test_gpu_phase1.py',
    'test_gpu_phase2.py',
    'test_gpu_phase4.py',
    'test_gpu_phase6.py',
    'test_gpu_cvt_encoders.py',
    'test_hazard_regression.py',
    'test_bugfix_benchmark.py',
    'test_fsetp_negated_pred_regression.py',
    'test_tma_gpu.py',
]

total_passed = 0
total_failed = 0
total_errors = 0
total_skipped = 0
total_xfailed = 0
failures = []


def run(label, args):
    global total_passed, total_failed, total_errors, total_skipped, total_xfailed
    result = subprocess.run(
        [sys.executable, '-m', 'pytest'] + args + ['-q', '--tb=line'],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    # Parse summary line
    for line in result.stdout.splitlines()[-3:]:
        if 'passed' in line or 'failed' in line or 'error' in line:
            import re
            m = re.search(r'(\d+) passed', line)
            if m: total_passed += int(m.group(1))
            m = re.search(r'(\d+) failed', line)
            if m:
                n = int(m.group(1))
                total_failed += n
                if n > 0: failures.append(label)
            m = re.search(r'(\d+) error', line)
            if m: total_errors += int(m.group(1))
            m = re.search(r'(\d+) skipped', line)
            if m: total_skipped += int(m.group(1))
            m = re.search(r'(\d+) xfailed', line)
            if m: total_xfailed += int(m.group(1))
    # Show result
    status = 'PASS' if result.returncode == 0 else 'FAIL'
    summary = result.stdout.splitlines()[-1] if result.stdout.strip() else 'no output'
    print(f'  [{status}] {label}: {summary}')
    if result.returncode != 0 and '--tb=line' in args:
        for line in result.stdout.splitlines():
            if 'FAILED' in line or 'ERROR' in line:
                print(f'         {line.strip()}')


print('=== Non-GPU tests (single process) ===')
non_gpu_paths = [os.path.join('tests', f) for f in NON_GPU]
run('non-gpu', non_gpu_paths)

print()
print('=== GPU tests (isolated processes) ===')
for f in GPU_FILES:
    fpath = os.path.join('tests', f)
    if os.path.exists(os.path.join(ROOT, fpath)):
        run(f, [fpath])
    else:
        print(f'  [SKIP] {f}: not found')

print()
print(f'=== TOTAL: {total_passed} passed, {total_failed} failed, '
      f'{total_errors} errors, {total_skipped} skipped, {total_xfailed} xfailed ===')
if failures:
    print(f'Failures in: {", ".join(failures)}')
sys.exit(1 if total_failed > 0 or total_errors > 0 else 0)
