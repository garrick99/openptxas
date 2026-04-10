"""
FB-4.5: Full benchmark sweep — quantify compaction impact across all
PTX kernels exercised by the test suite.

No code changes. Pure measurement.

Approach:
  1. Extract every `_PTX_*` triple-quoted literal from tests/test_*.py.
  2. Compile each with compaction ENABLED, capturing every per-kernel
     CompactReport via a monkey-patched spy on sass.compact.compact.
  3. CompactReport already tracks regs_before (baseline / no rewrite)
     and regs_after (post-compaction), so a single ON pass yields both
     compaction-OFF and compaction-ON metrics for each kernel.
  4. Aggregate, print summary, top gains, no-op covered cases, and
     skip cases.

The full pytest suite is run separately at the end to confirm 421/421.
"""
from __future__ import annotations

import glob
import re
import sys
import io
import contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sass import compact as compact_mod
from sass.compact import CompactReport
from sass.pipeline import compile_ptx_source


def extract_ptx_blocks(test_dir: Path) -> list[tuple[str, str, str]]:
    """Return list of (file, var_name, ptx_source) tuples."""
    out = []
    for path in sorted(test_dir.glob("test_*.py")):
        txt = path.read_text(encoding="utf-8")
        for m in re.finditer(
            r'(_PTX_\w+)\s*=\s*"""(.*?)"""', txt, re.DOTALL
        ):
            var = m.group(1)
            ptx = m.group(2)
            if ".visible" in ptx and ".entry" in ptx:
                out.append((path.name, var, ptx))
    return out


def run_sweep() -> list[CompactReport]:
    """Compile every PTX block, capturing all CompactReports."""
    blocks = extract_ptx_blocks(ROOT / "tests")
    print(f"[fb-4.5] Discovered {len(blocks)} PTX blocks across test files")

    captured: list[CompactReport] = []
    seen_kernels: set[str] = set()

    orig_compact = compact_mod.compact

    def spy(sass_instrs, verbose=False, kernel_name="<unknown>", report=None):
        if report is None:
            report = CompactReport(kernel_name)
        # Force compaction ON, but run silently
        result = orig_compact(
            sass_instrs, verbose=False, kernel_name=kernel_name, report=report
        )
        if kernel_name not in seen_kernels:
            seen_kernels.add(kernel_name)
            captured.append(report)
        return result

    compact_mod.compact = spy

    failed_blocks = 0
    for fname, var, ptx in blocks:
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                compile_ptx_source(ptx)
        except Exception as e:
            failed_blocks += 1
            print(f"[fb-4.5] WARN: failed to compile {fname}::{var}: "
                  f"{type(e).__name__}: {e}")

    compact_mod.compact = orig_compact
    print(f"[fb-4.5] Compiled {len(blocks) - failed_blocks}/{len(blocks)} blocks")
    print(f"[fb-4.5] Captured {len(captured)} unique kernel reports")
    return captured


def aggregate(reports: list[CompactReport]) -> dict:
    total = len(reports)
    covered = [r for r in reports if r.covered]
    compacted = [r for r in covered if r.gpr_fields_rewritten > 0]
    no_op_covered = [r for r in covered if r.gpr_fields_rewritten == 0]
    uncovered = [r for r in reports if not r.covered]

    deltas = []
    improved = 0
    unchanged = 0
    worse = 0
    sass_deltas = []

    for r in reports:
        delta = r.regs_before - r.regs_after  # positive = improvement
        deltas.append(delta)
        if delta > 0:
            improved += 1
        elif delta == 0:
            unchanged += 1
        else:
            worse += 1
        sass_deltas.append(r.sass_before - r.sass_after)

    total_regs_saved = sum(d for d in deltas if d > 0)
    avg_regs_saved = (total_regs_saved / len(compacted)) if compacted else 0.0
    max_gain = max(deltas) if deltas else 0
    min_delta = min(deltas) if deltas else 0
    meaningful = [r for r in compacted
                  if (r.regs_before - r.regs_after) >= 2]

    return {
        "total": total,
        "covered": len(covered),
        "compacted": len(compacted),
        "no_op_covered": no_op_covered,
        "uncovered": uncovered,
        "improved": improved,
        "unchanged": unchanged,
        "worse": worse,
        "total_regs_saved": total_regs_saved,
        "avg_regs_saved": avg_regs_saved,
        "max_gain": max_gain,
        "min_delta": min_delta,
        "meaningful": len(meaningful),
        "total_sass_delta": sum(sass_deltas),
        "all_reports": reports,
    }


def print_report(agg: dict) -> None:
    print()
    print("=" * 60)
    print("FB-4.5 Summary")
    print("=" * 60)
    print(f"  kernels:               {agg['total']}")
    print(f"  covered:               {agg['covered']}  "
          f"({agg['covered'] * 100 // max(agg['total'], 1)}%)")
    print(f"  compacted:             {agg['compacted']}  "
          f"({agg['compacted'] * 100 // max(agg['covered'], 1)}% of covered)")
    print(f"  meaningful (>=2 regs): {agg['meaningful']}  "
          f"({agg['meaningful'] * 100 // max(agg['compacted'], 1)}% of compacted)")
    print()
    print(f"  total regs saved:      {agg['total_regs_saved']}")
    print(f"  avg regs saved (compacted only): {agg['avg_regs_saved']:.2f}")
    print(f"  max gain:              {agg['max_gain']} regs")
    print(f"  min delta:             {agg['min_delta']} regs")
    print()
    print(f"  improved:              {agg['improved']}")
    print(f"  unchanged:             {agg['unchanged']}")
    print(f"  worse (regressions):   {agg['worse']}")
    print()
    print(f"  total SASS delta:      {agg['total_sass_delta']} "
          f"(should be 0 — compaction never adds/removes insts)")

    # Top 5 gains
    print()
    print("-" * 60)
    print("Top 5 kernels by register reduction")
    print("-" * 60)
    sorted_reports = sorted(
        agg["all_reports"],
        key=lambda r: r.regs_before - r.regs_after,
        reverse=True,
    )
    for r in sorted_reports[:5]:
        delta = r.regs_before - r.regs_after
        if delta <= 0:
            break
        print(f"  {r.kernel_name:40s} "
              f"{r.regs_before:3d} -> {r.regs_after:3d}  "
              f"(-{delta} regs, {r.gpr_fields_rewritten} fields, "
              f"{r.compacted_insts} insts)")

    # No-op covered cases
    print()
    print("-" * 60)
    print(f"Covered but no-op ({len(agg['no_op_covered'])} kernels)")
    print("-" * 60)
    if agg["no_op_covered"]:
        names = sorted(r.kernel_name for r in agg["no_op_covered"])
        # Print in columns
        for i in range(0, len(names), 3):
            row = names[i:i + 3]
            print("  " + "  ".join(f"{n:30s}" for n in row))
    else:
        print("  (none)")

    # Skip cases
    print()
    print("-" * 60)
    print(f"Uncovered / skipped ({len(agg['uncovered'])} kernels)")
    print("-" * 60)
    if agg["uncovered"]:
        for r in agg["uncovered"]:
            ops = ", ".join(f"0x{op:03x}" for op in sorted(r.uncovered)[:6])
            more = f" +{len(r.uncovered) - 6} more" if len(r.uncovered) > 6 else ""
            print(f"  {r.kernel_name:40s} uncovered: [{ops}{more}]")
    else:
        print("  (none — full coverage)")


def main():
    reports = run_sweep()
    agg = aggregate(reports)
    print_report(agg)


if __name__ == "__main__":
    main()
