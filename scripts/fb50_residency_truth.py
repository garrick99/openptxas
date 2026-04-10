"""
FB-5.0 — Residency Truth Table (read-only, no behavior change)

For each of the 36 compactable kernels (from FB-4.5), compare:
  - allocator's view: every vreg → physical GPR assignment
  - SASS view: which physical GPRs actually appear in the emitted byte stream
  - delta: phantom GPRs the allocator reserved that the SASS does not need

Buckets every "phantom" allocation into one of:
  - phantom_pair       : 64-bit allocation where the SASS uses neither
                         lo nor hi (or only one half)
  - phantom_quad       : tensor-core 4-register reservation where the
                         SASS does not use one or more of the 4 slots
  - ur_backed          : vreg analyzed as UR-bound (LDCU.64 path), but
                         allocator still gave it a GPR pair anyway
  - dead_at_emit       : 32-bit allocation that simply never appears in
                         the final SASS (e.g. eliminated by ISel)
  - real               : control class — appears in SASS, kept by compaction

This is a pure instrumentation pass: it monkey-patches `allocate` and
`compact` from the outside to capture state, runs each test PTX through
the pipeline once, and reconciles after the fact.

NOTHING in sass/ is modified.
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sass import compact as compact_mod
from sass import regalloc as regalloc_mod
from sass.compact import CompactReport, collect_used_gprs
from sass.pipeline import compile_ptx_source


# ---------------------------------------------------------------------------
# PTX block discovery (same as fb45_sweep.py)
# ---------------------------------------------------------------------------
def extract_ptx_blocks(test_dir: Path) -> list[tuple[str, str, str]]:
    out = []
    for path in sorted(test_dir.glob("test_*.py")):
        txt = path.read_text(encoding="utf-8")
        for m in re.finditer(
            r'(_PTX_\w+)\s*=\s*"""(.*?)"""', txt, re.DOTALL
        ):
            var, ptx = m.group(1), m.group(2)
            if ".visible" in ptx and ".entry" in ptx:
                out.append((path.name, var, ptx))
    return out


# ---------------------------------------------------------------------------
# Recompute UR / quad sets from a Function (mirrors regalloc.py logic)
# ---------------------------------------------------------------------------
def compute_ur_and_quad_sets(fn) -> dict:
    """Re-derive the residency-classification sets the allocator computes
    internally but doesn't expose.  Mirrors lines 200-304 of regalloc.py."""
    from ptx.ir import RegOp, MemOp

    all_instrs = []
    for bb in fn.blocks:
        all_instrs.extend(bb.instructions)

    # u64 params bound to UR (only consumed by add/sub/mul/shl on 64-bit)
    param_u64_names: set[str] = set()
    for inst in all_instrs:
        if (inst.op == 'ld' and 'param' in inst.types
                and any(t in ('u64', 's64', 'b64') for t in inst.types)
                and isinstance(inst.dest, RegOp)):
            param_u64_names.add(inst.dest.name)
    ur_param_regs: set[str] = set()
    for pname in param_u64_names:
        only_add64 = True
        for inst in all_instrs:
            for src in inst.srcs:
                src_name = (src.name if isinstance(src, RegOp) else
                            (src.base if isinstance(src, MemOp) and
                             isinstance(src.base, str) else None))
                if src_name == pname:
                    if not (inst.op in ('add', 'sub', 'mul', 'shl')
                            and any(t in ('u64', 's64', 'b64') for t in inst.types)):
                        only_add64 = False
                        break
            if not only_add64:
                break
        if only_add64:
            ur_param_regs.add(pname)

    # f64 params bound to UR (only consumed by f64 mul/add/fma/mov)
    f64_param_names: set[str] = set()
    for inst in all_instrs:
        if (inst.op == 'ld' and 'param' in inst.types
                and 'f64' in inst.types
                and isinstance(inst.dest, RegOp)):
            f64_param_names.add(inst.dest.name)
    ur_only_f64_regs: set[str] = set()
    for pname in f64_param_names:
        safe = True
        for inst in all_instrs:
            for src in inst.srcs:
                src_name = (src.name if isinstance(src, RegOp) else
                            (src.base if isinstance(src, MemOp) and
                             isinstance(src.base, str) else None))
                if src_name == pname:
                    if not (inst.op in ('mul', 'add', 'fma', 'mov')
                            and any(t == 'f64' for t in inst.types)):
                        safe = False
                        break
            if not safe:
                break
        if safe:
            ur_only_f64_regs.add(pname)

    # Quad-aligned MMA dest + 3 followers in same RegDecl
    from ptx.ir import ScalarKind
    quad_align_regs: set[str] = set()
    for inst in all_instrs:
        if inst.op == 'mma' and 'sync' in inst.types and inst.dest is not None:
            if isinstance(inst.dest, RegOp):
                quad_align_regs.add(inst.dest.name)
    quad_follow_regs: set[str] = set()
    for rd in fn.reg_decls:
        if rd.type.kind == ScalarKind.PRED:
            continue
        for i, nm in enumerate(rd.names):
            if nm in quad_align_regs:
                for j in range(1, 4):
                    if i + j < len(rd.names):
                        quad_follow_regs.add(rd.names[i + j])

    return {
        'ur_param_regs': ur_param_regs,
        'ur_only_f64_regs': ur_only_f64_regs,
        'quad_align_regs': quad_align_regs,
        'quad_follow_regs': quad_follow_regs,
    }


def vreg_kind(fn, name: str, sets: dict) -> str:
    """Return one of: u32, u64, f64, pred, tensor_quad."""
    from ptx.ir import ScalarKind
    if name in sets['quad_align_regs'] or name in sets['quad_follow_regs']:
        return 'tensor_quad'
    for rd in fn.reg_decls:
        if name in rd.names:
            if rd.type.kind == ScalarKind.PRED:
                return 'pred'
            if rd.type.width >= 64:
                if rd.type.kind == ScalarKind.F:
                    return 'f64'
                return 'u64'
            return 'u32'
    return 'u32'


def producer_opcode(fn, name: str) -> str:
    from ptx.ir import RegOp
    for bb in fn.blocks:
        for inst in bb.instructions:
            if isinstance(inst.dest, RegOp) and inst.dest.name == name:
                t = '.'.join(inst.types) if inst.types else ''
                return f'{inst.op}.{t}' if t else inst.op
    return '<none>'


def consumer_opcodes(fn, name: str) -> list[str]:
    from ptx.ir import RegOp, MemOp
    out = []
    for bb in fn.blocks:
        for inst in bb.instructions:
            for src in inst.srcs:
                hit = False
                if isinstance(src, RegOp) and src.name == name:
                    hit = True
                elif isinstance(src, MemOp) and isinstance(src.base, str):
                    base = src.base if src.base.startswith('%') else f'%{src.base}'
                    if base == name:
                        hit = True
                if hit:
                    t = '.'.join(inst.types) if inst.types else ''
                    out.append(f'{inst.op}.{t}' if t else inst.op)
    return out


# ---------------------------------------------------------------------------
# Per-kernel introspection
# ---------------------------------------------------------------------------
class VRegRow:
    __slots__ = ('name', 'kind', 'phys', 'first_def', 'last_use',
                 'alloc_class', 'sass_residency', 'bucket',
                 'producer', 'consumers')


def classify_kernel(fn, alloc_result, sass_instrs) -> tuple[list[VRegRow], dict]:
    """Build per-vreg truth table for one kernel.

    Returns (rows, kernel_summary).
    """
    from ptx.ir import RegOp, MemOp

    sets = compute_ur_and_quad_sets(fn)
    int_regs: dict[str, int] = alloc_result.ra.int_regs

    # Liveness (mirror regalloc lines 142-160)
    all_instrs = []
    for bb in fn.blocks:
        all_instrs.extend(bb.instructions)
    first_def: dict[str, int] = {}
    last_use: dict[str, int] = {}
    for idx, inst in enumerate(all_instrs):
        if inst.dest and isinstance(inst.dest, RegOp):
            n = inst.dest.name
            if n not in first_def:
                first_def[n] = idx
        for src in inst.srcs:
            if isinstance(src, RegOp):
                last_use[src.name] = idx
            if isinstance(src, MemOp) and src.base:
                bn = src.base if src.base.startswith('%') else f'%{src.base}'
                last_use[bn] = idx

    # SASS-referenced GPR set (post-isel, pre-compaction view)
    sass_used, sass_pair_bases, sass_quad_bases = collect_used_gprs(sass_instrs)

    rows: list[VRegRow] = []
    bucket_counts = Counter()

    for name, phys in int_regs.items():
        kind = vreg_kind(fn, name, sets)
        is_64 = kind in ('u64', 'f64')
        is_quad = kind == 'tensor_quad'

        # Allocator's footprint (which physical slots it claimed)
        if is_quad:
            footprint = {phys, phys + 1, phys + 2, phys + 3}
            alloc_class = 'quad'
        elif is_64:
            footprint = {phys, phys + 1}
            alloc_class = 'pair'
        else:
            footprint = {phys}
            alloc_class = 'GPR'

        # SASS view: which slots in this footprint actually show up
        sass_seen = footprint & sass_used

        # UR-backed test (UR-classified by allocator analysis but still given GPRs)
        is_ur_backed = (name in sets['ur_param_regs'] or
                        name in sets['ur_only_f64_regs'])

        # Residency classification
        if is_ur_backed:
            residency = 'UR'
        elif sass_seen == footprint:
            residency = 'GPR'  # fully present
        elif sass_seen:
            residency = 'GPR_partial'
        else:
            residency = 'absent'

        # Bucket assignment
        if is_ur_backed:
            bucket = 'ur_backed'
        elif is_quad and sass_seen != footprint:
            bucket = 'phantom_quad'
        elif is_64 and sass_seen != footprint:
            bucket = 'phantom_pair'
        elif not is_64 and not is_quad and not sass_seen:
            bucket = 'dead_at_emit'
        else:
            bucket = 'real'

        bucket_counts[bucket] += 1

        row = VRegRow()
        row.name = name
        row.kind = kind
        row.phys = phys
        row.first_def = first_def.get(name, -1)
        row.last_use = last_use.get(name, -1)
        row.alloc_class = alloc_class
        row.sass_residency = residency
        row.bucket = bucket
        row.producer = producer_opcode(fn, name)
        row.consumers = consumer_opcodes(fn, name)
        rows.append(row)

    return rows, dict(bucket_counts)


# ---------------------------------------------------------------------------
# Sweep driver — patches allocate + compact, walks all PTX blocks
# ---------------------------------------------------------------------------
def run_sweep():
    blocks = extract_ptx_blocks(ROOT / "tests")
    print(f"[fb-5.0] Discovered {len(blocks)} PTX blocks")

    captured_alloc: dict[str, tuple[object, object]] = {}  # kernel_name -> (fn, alloc)
    captured_sass: dict[str, list] = {}                    # kernel_name -> sass_instrs
    captured_reports: dict[str, CompactReport] = {}

    orig_allocate = regalloc_mod.allocate
    orig_compact = compact_mod.compact

    # The allocate() function takes a Function as its first arg.
    # We capture (fn, alloc_result) keyed by fn.name.
    def alloc_spy(fn, *args, **kwargs):
        result = orig_allocate(fn, *args, **kwargs)
        captured_alloc[fn.name] = (fn, result)
        return result

    # compact() is called by pipeline._compile_kernel right after isel.
    # We capture the sass_instrs list at that point.
    def compact_spy(sass_instrs, verbose=False, kernel_name="<unknown>", report=None):
        if report is None:
            report = CompactReport(kernel_name)
        if kernel_name not in captured_sass:
            # Snapshot the pre-compaction SASS for later reconciliation
            captured_sass[kernel_name] = list(sass_instrs)
        result = orig_compact(sass_instrs, verbose=False,
                              kernel_name=kernel_name, report=report)
        if kernel_name not in captured_reports:
            captured_reports[kernel_name] = report
        return result

    # The pipeline imports allocate as a name-binding from sass.regalloc.
    # We need to patch BOTH the source module and the pipeline's local binding.
    import sass.pipeline as pipeline_mod
    regalloc_mod.allocate = alloc_spy
    pipeline_mod.allocate = alloc_spy
    compact_mod.compact = compact_spy

    failed = 0
    for fname, var, ptx in blocks:
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                compile_ptx_source(ptx)
        except Exception as e:
            failed += 1
            print(f"[fb-5.0] WARN failed {fname}::{var}: {type(e).__name__}: {e}")

    regalloc_mod.allocate = orig_allocate
    pipeline_mod.allocate = orig_allocate
    compact_mod.compact = orig_compact

    print(f"[fb-5.0] Captured {len(captured_alloc)} allocator snapshots, "
          f"{len(captured_sass)} SASS snapshots, "
          f"{len(captured_reports)} compact reports")
    return captured_alloc, captured_sass, captured_reports


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------
def main():
    captured_alloc, captured_sass, captured_reports = run_sweep()

    # FB-4.5 told us 36 kernels are compactable. Filter to those.
    compactable_names = sorted(
        name for name, r in captured_reports.items()
        if r.gpr_fields_rewritten > 0
    )
    print(f"[fb-5.0] Compactable kernels (from FB-4.5): {len(compactable_names)}")

    bucket_totals = Counter()
    per_kernel_summaries = []
    pattern_counter = Counter()  # producer/consumer-shape patterns

    for kname in compactable_names:
        if kname not in captured_alloc or kname not in captured_sass:
            print(f"  [skip] {kname}: missing allocator or SASS capture")
            continue
        fn, alloc = captured_alloc[kname]
        rows, kbuckets = classify_kernel(fn, alloc, captured_sass[kname])
        report = captured_reports[kname]

        for k, v in kbuckets.items():
            bucket_totals[k] += v

        per_kernel_summaries.append({
            'name': kname,
            'regs_before': report.regs_before,
            'regs_after': report.regs_after,
            'phantom_pairs': kbuckets.get('phantom_pair', 0),
            'phantom_quads': kbuckets.get('phantom_quad', 0),
            'ur_backed': kbuckets.get('ur_backed', 0),
            'dead_at_emit': kbuckets.get('dead_at_emit', 0),
            'real': kbuckets.get('real', 0),
        })

        # Patterns: for each phantom row, record (kind, producer, top consumer)
        for r in rows:
            if r.bucket in ('phantom_pair', 'phantom_quad', 'ur_backed', 'dead_at_emit'):
                top_cons = r.consumers[0] if r.consumers else '<none>'
                pattern = f"{r.kind:12s}  {r.producer:25s}  ->  {top_cons}"
                pattern_counter[pattern] += 1

    # ---------- Aggregate table ----------
    print()
    print("=" * 64)
    print("FB-5.0 Residency Truth")
    print("=" * 64)
    print(f"  compactable kernels:       {len(compactable_names)}")
    print(f"  phantom pair cases:        {bucket_totals['phantom_pair']}")
    print(f"  phantom quad cases:        {bucket_totals['phantom_quad']}")
    print(f"  UR-backed phantom cases:   {bucket_totals['ur_backed']}")
    print(f"  dead-at-emit cases:        {bucket_totals['dead_at_emit']}")
    print(f"  real (control):            {bucket_totals['real']}")
    print()
    total_phantom = (bucket_totals['phantom_pair']
                     + bucket_totals['phantom_quad']
                     + bucket_totals['ur_backed']
                     + bucket_totals['dead_at_emit'])
    print(f"  total phantom:             {total_phantom}")
    total_all = total_phantom + bucket_totals['real']
    if total_all:
        print(f"  phantom rate:              "
              f"{total_phantom * 100 // total_all}% of all vregs in compactable kernels")

    # ---------- Per-kernel summaries ----------
    print()
    print("-" * 64)
    print("Per-kernel compactable summary")
    print("-" * 64)
    for s in per_kernel_summaries:
        print(f"kernel={s['name']}")
        print(f"  regs_before: {s['regs_before']}")
        print(f"  regs_after:  {s['regs_after']}")
        print(f"  phantom_pairs: {s['phantom_pairs']}")
        print(f"  phantom_quads: {s['phantom_quads']}")
        print(f"  ur_backed:     {s['ur_backed']}")
        print(f"  dead_at_emit:  {s['dead_at_emit']}")
        print(f"  real:          {s['real']}")

    # ---------- Top recurring patterns ----------
    print()
    print("-" * 64)
    print("Top 5 recurring producer/consumer shapes (phantom only)")
    print("-" * 64)
    print(f"  {'count':>6s}  {'kind':12s}  {'producer':25s}      {'top consumer'}")
    for pattern, count in pattern_counter.most_common(5):
        print(f"  {count:6d}  {pattern}")


if __name__ == "__main__":
    main()
