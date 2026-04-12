#!/usr/bin/env python3
"""
ULTIMATE-DEMO-1: PTXAS vs OURS -- Undeniable GPU Control

Proof system that compiles, runs, verifies, and compares
OpenPTXas against NVIDIA's ptxas on real GPU hardware.

Usage:
    python demo/main.py --kernel ilp_dual_int32
    python demo/main.py --suite ilp
    python demo/main.py --suite demo
    python demo/main.py --suite full
    python demo/main.py --kernel vecadd_large --diff
    python demo/main.py --proof
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.bench_util import compile_openptxas, compile_ptxas
from demo.compare import compile_both, opcode_name
from demo.runner import run_full, get_ptx
from demo.diff import diff_streams, summarize_transforms
from demo.formatter import fmt_kernel_report, fmt_suite_summary, fmt_proof_status
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


def run_one_kernel(name: str, show_diff: bool = False) -> dict | None:
    """Run full comparison pipeline for one kernel."""
    ptx = get_ptx(name)
    if ptx is None:
        print(f'ERROR: no PTX source for kernel "{name}"', file=sys.stderr)
        return None

    # Run workbench: compile both, GPU correctness, metrics
    wb = run_full(name)
    if wb is None or wb.get('error'):
        print(f'ERROR: {name}: {wb.get("error") if wb else "unknown"}', file=sys.stderr)
        return None

    correctness = wb.get('correctness', 'FAIL')
    ours_wb = wb['ours']
    ptxas_wb = wb['ptxas']

    # Compile both for SASS disassembly (cheap -- already cached in memory)
    sass = compile_both(ptx)

    # Build unified metrics dict from workbench (authoritative source)
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

    # Diff analysis
    diff_records = diff_streams(metrics['ours']['ops'], metrics['ptxas']['ops'])
    highlights = summarize_transforms(diff_records)

    # Print report
    report = fmt_kernel_report(name, metrics, gpu, diff_records, highlights)
    print(report)

    # Optional detailed diff
    if show_diff:
        print('[DETAILED INSTRUCTION DIFF]')
        for r in diff_records:
            if r['type'] == 'match':
                opc = opcode_name(r['ours']['opcode'])
                print(f'    [{r["ours"]["idx"]:3d}]  {opc:<16s}  =  [{r["ptxas"]["idx"]:3d}] {opc}')
            elif r['type'] == 'ours_only':
                opc = opcode_name(r['ours']['opcode'])
                tag = '  <<< ' + r['explanation'] if r['explanation'] else ''
                print(f'  + [{r["ours"]["idx"]:3d}]  {opc:<16s}     {"---":>5s} ---{tag}')
            elif r['type'] == 'ptxas_only':
                opc = opcode_name(r['ptxas']['opcode'])
                print(f'  - {"---":>5s}  {"---":<16s}     [{r["ptxas"]["idx"]:3d}] {opc}')
        print()

    return {
        'name': name,
        'metrics': metrics,
        'gpu': gpu,
        'diff': diff_records,
        'highlights': highlights,
    }


def run_proof_summary():
    """Run adversarial harness and print proof status."""
    import subprocess
    r = subprocess.run(
        [sys.executable, str(ROOT / 'probe_work' / 'fg40_adversarial_harness.py')],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    confirmed = 0
    total_checked = 0
    for line in r.stdout.splitlines():
        line = line.strip()
        if 'MODEL_CONFIRMED' in line and '=' in line:
            parts = line.split('=')
            if len(parts) == 2:
                try:
                    n = int(parts[1].strip())
                    confirmed += n
                    total_checked += n
                except ValueError:
                    pass
        for tag in ('MODEL_FALSE_POSITIVE', 'MODEL_FALSE_NEGATIVE'):
            if tag in line and ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    try:
                        n = int(parts[1].strip())
                        total_checked += n
                    except ValueError:
                        pass
    print()
    print(fmt_proof_status((confirmed, total_checked)))
    print()


def main():
    parser = argparse.ArgumentParser(
        description='PTXAS vs OURS: Undeniable GPU Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--kernel', '-k', help='Run a single kernel by name')
    parser.add_argument('--suite', '-s', choices=list(SUITES.keys()),
                        help='Run a kernel suite (demo, ilp, full)')
    parser.add_argument('--diff', '-d', action='store_true',
                        help='Show detailed instruction-level diff')
    parser.add_argument('--proof', '-p', action='store_true',
                        help='Run proof model verification')
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
        r = run_one_kernel(args.kernel, show_diff=args.diff)
        if r:
            results.append(r)

    if args.suite:
        kernels = SUITES.get(args.suite, [])
        for name in kernels:
            r = run_one_kernel(name, show_diff=args.diff)
            if r:
                results.append(r)

    if results and len(results) > 1:
        print(fmt_suite_summary(results))

    if args.proof:
        run_proof_summary()

    if any(not r['gpu']['ours_pass'] for r in results):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
