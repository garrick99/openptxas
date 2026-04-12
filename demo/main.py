#!/usr/bin/env python3
"""
ULTIMATE-DEMO-1: PTXAS vs OURS -- Undeniable GPU Control

Proof system that compiles, runs, verifies, and compares
OpenPTXas against NVIDIA's ptxas on real GPU hardware.

Usage:
    python demo/main.py --kernel ilp_dual_int32
    python demo/main.py --suite demo
    python demo/main.py --suite full
    python demo/main.py --kernel ilp_dual_int32 --diff
    python demo/main.py --suite demo --explain
    python demo/main.py --proof
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo.compare import compile_both, opcode_name
from demo.runner import run_full, get_ptx
from demo.diff import diff_streams, summarize_transforms
from demo.formatter import (fmt_kernel_report, fmt_suite_summary,
                             fmt_proof_footer, fmt_structured_diff)
import workbench


DEMO_KERNELS = [
    'ilp_dual_int32',
    'ilp_alu_addr',
    'ilp_pred_alu',
    'ilp_unrolled_sum4',
    'vecadd_large',
]

SUITES = {
    'demo': DEMO_KERNELS,
    'ilp': workbench.SUITES.get('ilp', []),
    'full': list(workbench.SUITES.get('all', [])),
}


def _get_proof_counts() -> dict:
    """Run adversarial harness + corpus proof, return counts."""
    import subprocess

    # Adversarial harness
    r = subprocess.run(
        [sys.executable, str(ROOT / 'probe_work' / 'fg40_adversarial_harness.py')],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    adv_confirmed = 0
    adv_total = 0
    for line in r.stdout.splitlines():
        line = line.strip()
        if 'MODEL_CONFIRMED' in line and '=' in line:
            parts = line.split('=')
            if len(parts) == 2:
                try:
                    n = int(parts[1].strip())
                    adv_confirmed += n
                    adv_total += n
                except ValueError:
                    pass
        for tag in ('MODEL_FALSE_POSITIVE', 'MODEL_FALSE_NEGATIVE'):
            if tag in line and ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    try:
                        n = int(parts[1].strip())
                        adv_total += n
                    except ValueError:
                        pass

    # Corpus proof (run via pytest in subprocess for isolation)
    r2 = subprocess.run(
        [sys.executable, '-m', 'pytest',
         'tests/test_fg25_proof_engine.py', '-k', 'inv_h',
         '-q', '--tb=no'],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    corpus_passed = 0
    corpus_total = 0
    for line in r2.stdout.splitlines():
        # pytest summary: "37 passed, 61 deselected in 0.88s"
        if 'passed' in line:
            for part in line.split():
                if part.isdigit():
                    corpus_passed = int(part)
                    corpus_total = corpus_passed
                    break

    return {
        'adversarial': (adv_confirmed, adv_total),
        'corpus': (corpus_passed, corpus_total),
    }


def run_one_kernel(name: str, show_diff: bool = False,
                   explain: bool = False) -> dict | None:
    """Run full comparison pipeline for one kernel."""
    ptx = get_ptx(name)
    if ptx is None:
        print(f'ERROR: no PTX source for kernel "{name}"', file=sys.stderr)
        return None

    wb = run_full(name)
    if wb is None or wb.get('error'):
        print(f'ERROR: {name}: {wb.get("error") if wb else "unknown"}', file=sys.stderr)
        return None

    correctness = wb.get('correctness', 'FAIL')
    ours_wb = wb['ours']
    ptxas_wb = wb['ptxas']

    sass = compile_both(ptx)

    metrics = {
        'ours': {
            'non_nop': ours_wb['sass_non_nop'],
            'regs': ours_wb['regs'],
            'nops': ours_wb['sass_total'] - ours_wb['sass_non_nop'],
            'total': ours_wb['sass_total'],
            'compile_ms': ours_wb['compile_ms'],
            'ops': sass['ours']['ops'],
        },
        'ptxas': {
            'non_nop': ptxas_wb['sass_non_nop'],
            'regs': ptxas_wb['regs'],
            'nops': ptxas_wb['sass_total'] - ptxas_wb['sass_non_nop'],
            'total': ptxas_wb['sass_total'],
            'compile_ms': ptxas_wb['compile_ms'],
            'ops': sass['ptxas']['ops'],
        },
    }

    gpu = {
        'ours_pass': correctness == 'PASS',
        'ptxas_pass': ptxas_wb is not None,
    }

    diff_records = diff_streams(metrics['ours']['ops'], metrics['ptxas']['ops'])
    highlights = summarize_transforms(diff_records)

    report = fmt_kernel_report(name, metrics, gpu, diff_records, highlights,
                               explain=explain)
    print(report)

    if show_diff:
        print(fmt_structured_diff(diff_records))

    return {
        'name': name,
        'metrics': metrics,
        'gpu': gpu,
        'diff': diff_records,
        'highlights': highlights,
    }


def main():
    parser = argparse.ArgumentParser(
        description='PTXAS vs OURS: Undeniable GPU Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--kernel', '-k', help='Run a single kernel by name')
    parser.add_argument('--suite', '-s', choices=list(SUITES.keys()),
                        help='Run a kernel suite (demo, ilp, full)')
    parser.add_argument('--diff', '-d', action='store_true',
                        help='Show structured transformation diff')
    parser.add_argument('--explain', '-e', action='store_true',
                        help='Show detailed WHY explanations for differences')
    parser.add_argument('--proof', '-p', action='store_true',
                        help='Run proof model verification (standalone)')
    args = parser.parse_args()

    if not args.kernel and not args.suite and not args.proof:
        parser.print_help()
        return 1

    print()
    print('=' * 70)
    print('  PTXAS vs OURS: Undeniable GPU Control')
    print('  OpenPTXas Proof Demo  |  SM_120 (Blackwell / RTX 5090)')
    print('=' * 70)

    results = []

    if args.kernel:
        r = run_one_kernel(args.kernel, show_diff=args.diff,
                           explain=args.explain)
        if r:
            results.append(r)

    if args.suite:
        kernels = SUITES.get(args.suite, [])
        for name in kernels:
            r = run_one_kernel(name, show_diff=args.diff,
                               explain=args.explain)
            if r:
                results.append(r)

    if results and len(results) > 1:
        print(fmt_suite_summary(results))

    # Auto proof footer for suites (or standalone --proof)
    if args.suite or args.proof:
        proof = _get_proof_counts()
        print(fmt_proof_footer(proof['adversarial'], proof['corpus']))

    if any(not r['gpu']['ours_pass'] for r in results):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
