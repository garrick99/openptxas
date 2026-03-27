"""
Diagnose ldg64_min pipeline failure vs hand-crafted success.
Runs each test in a subprocess to isolate CUDA crashes.
"""
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ldg64_worker.py')


def run_variant(label, variant, num_gprs=None):
    """Run one test variant as a subprocess."""
    cmd = [sys.executable, WORKER, variant]
    if num_gprs is not None:
        cmd.append(str(num_gprs))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output = r.stdout.strip()
        if r.returncode != 0 and not output:
            output = r.stderr.strip()[:200] if r.stderr else f"exit={r.returncode}"
        print(f"  [{label}] {output}")
    except subprocess.TimeoutExpired:
        print(f"  [{label}] TIMEOUT")


def main():
    print("=== LDG64 Pipeline vs Hand-Crafted Diagnosis ===")
    print()

    print("Test 1: Pipeline-compiled (num_gprs from regalloc)")
    run_variant('pipeline', 'pipeline')
    print()

    print("Test 2: Hand-crafted, same instr sequence, num_gprs=16 (large capmerc)")
    run_variant('hand-16', 'hand', 16)
    print()

    print("Test 3: Hand-crafted, same instr sequence, num_gprs=8 (small capmerc)")
    run_variant('hand-8', 'hand', 8)
    print()

    print("Test 4: Hand-crafted, same instr sequence, num_gprs=6")
    run_variant('hand-6', 'hand', 6)
    print()

    print("Test 5: Pipeline-compiled, force num_gprs=16")
    run_variant('pipeline-16', 'pipeline16')
    print()

    print("If pipeline fails but hand-6 passes -> ctrl/scoreboard issue.")
    print("If hand-6 fails but hand-16 passes -> capmerc sizing issue.")
    print("If pipeline-16 passes but pipeline fails -> capmerc sizing issue in pipeline.")


if __name__ == '__main__':
    main()
