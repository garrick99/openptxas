"""Continuous fuzz + triage + minimize loop.

Runs run_fuzz.py in short batches, ingests new artifacts into a SQLite
DB, classifies each by op-family signature, and spawns minimization
subprocesses for well-formed divergences.

Single worker per batch — BigDaddy rule; parallel CUDA contexts on
Windows WDDM hard-hang the box.

Usage:
    python -m fuzzer.loop run [--minutes N] [--batch-seconds S]
    python -m fuzzer.loop status          # dump DB summary
    python -m fuzzer.loop clusters [--n 20]  # top minimal-signature clusters
"""
import argparse, json, random, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fuzzer.oracle import ARTIFACT_DIR
from fuzzer.classify import signature, family_signature
from fuzzer.minimize import _split_body, _is_well_formed, _resolve
from fuzzer.db import BugDB
from fuzzer.families import all_families

_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _ROOT / '_fuzz' / 'bugs.db'
_REPORT_DIR = _ROOT / '_fuzz' / 'report'


def _meta(artifact_dir: Path) -> dict:
    f = artifact_dir / 'meta.json'
    if not f.exists(): return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def _artifact_tag(name: str) -> str:
    # name like 'divergence_018c45304eed2497' or 'sync_err_ours_abc...'
    parts = name.rsplit('_', 1)
    return parts[0]


def scan_and_ingest(db: BugDB) -> tuple[int, int]:
    """Walk ARTIFACT_DIR; insert any new artifacts.  Returns (new, total)."""
    new = total = 0
    if not ARTIFACT_DIR.exists(): return (0, 0)
    for d in ARTIFACT_DIR.iterdir():
        if not d.is_dir(): continue
        sha = d.name.rsplit('_', 1)[-1]
        tag = _artifact_tag(d.name)
        f = d / 'input.ptx'
        if not f.exists(): continue
        total += 1
        ptx = f.read_text()
        parts = _split_body(ptx)
        body_lines = len(parts[1]) if parts else 0
        wf = _is_well_formed(ptx)
        fs = signature(ptx)
        ffs = family_signature(ptx)
        meta = _meta(d)
        seed = meta.get('seed')
        family = meta.get('family')
        if db.upsert_artifact(sha, tag, body_lines, wf, fs, ffs, seed,
                               family=family):
            new += 1
    return (new, total)


def run_batch(seed_base: int, seconds: int, iters_per_worker: int = 50,
               family: str = 'alu_int') -> int:
    """Spawn one fuzz batch as a subprocess.  Returns subprocess rc."""
    cmd = [sys.executable, '-m', 'fuzzer.run_fuzz',
           '--workers', '1',
           '--seconds', str(seconds),
           '--seed-base', str(seed_base),
           '--iters-per-worker', str(iters_per_worker),
           '--family', family]
    print(f'\n[loop] batch family={family} seed_base={seed_base} for {seconds}s ...')
    r = subprocess.run(cmd, cwd=str(_ROOT))
    return r.returncode


def minimize_one(sha: str, db: BugDB) -> None:
    """Run minimize.py --one in a subprocess, then update DB from its output."""
    d = None
    for cand in ARTIFACT_DIR.iterdir():
        if cand.is_dir() and cand.name.endswith('_' + sha):
            d = cand; break
    if d is None:
        print(f'  [min] {sha}: artifact dir missing — skipping')
        return
    r = subprocess.run(
        [sys.executable, '-m', 'fuzzer.minimize', '--one', str(d)],
        cwd=str(_ROOT), capture_output=True, text=True, timeout=300)
    # minimal.ptx (if written) + status parsing
    minf = d / 'minimal.ptx'
    outcome = 'unknown'
    for ln in r.stdout.splitlines():
        if ln.startswith('[' + d.name + ']'):
            # '[divergence_abc] 12 -> 7 body lines  (fixed_point)'
            if '(' in ln and ')' in ln:
                outcome = ln[ln.rfind('(')+1:ln.rfind(')')]
    if minf.exists():
        min_ptx = minf.read_text()
        parts = _split_body(min_ptx)
        mbl = len(parts[1]) if parts else 0
        ms = signature(min_ptx)
        mfs = family_signature(min_ptx)
        db.record_minimization(sha, outcome, mbl, ms, mfs)
        print(f'  [min] {d.name}: {outcome}  body={mbl}  sig={mfs}')
    else:
        db.record_minimization(sha, outcome, -1, '', '')
        print(f'  [min] {d.name}: {outcome}  (no minimal.ptx written)')


def print_status(db: BugDB, per_family: bool = False) -> None:
    if per_family:
        print('\n[loop] family x tag counts:')
        for fam, tag, n, wf in db.family_status():
            print(f'  {str(fam):10s}  {tag:20s}  n={n:6d}  wf={(wf or 0):6d}')
    else:
        counts = db.status_counts()
        print('\n[loop] artifact tags:')
        for tag in sorted(counts):
            c = counts[tag]
            print(f'  {tag:20s}  unique={c["unique"]:6d}  well_formed={c["well_formed"]:6d}')
    clusters = db.top_minimal_signatures(limit=10)
    if clusters:
        print('\n[loop] top minimal-PTX signature clusters (all families):')
        for fam_sig, n, min_body in clusters:
            print(f'  n={n:4d}  min_body={min_body:2d}  {fam_sig}')


def cmd_run(args):
    db = BugDB(_DB_PATH)
    seed_base = args.seed_base if args.seed_base is not None \
                else random.randint(0, 1 << 30)
    cid = db.start_campaign(seed_base=seed_base)

    families = all_families() if args.families == 'all' \
               else [f.strip() for f in args.families.split(',')]
    db.update_campaign(cid, families=','.join(families))

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    batch_n = 0
    rnd = random.Random(seed_base)

    # Per-family plateau tracking: consecutive-zero-new-artifact streak.
    streak = {f: 0 for f in families}
    last_counts = {f: db.count_artifacts(family=f) for f in families}
    plateaued = set()

    try:
        round_idx = 0
        while True:
            if deadline and time.time() >= deadline:
                break
            # All families plateaued -> end.
            active = [f for f in families if f not in plateaued]
            if not active:
                print('[loop] all families plateaued — ending run')
                break
            fam = active[round_idx % len(active)]
            round_idx += 1
            batch_n += 1

            cur_seed = rnd.randint(0, 1 << 30)
            rc = run_batch(cur_seed, args.batch_seconds,
                            iters_per_worker=args.iters_per_worker,
                            family=fam)

            new, total = scan_and_ingest(db)
            cur_count = db.count_artifacts(family=fam)
            delta = cur_count - last_counts[fam]
            last_counts[fam] = cur_count
            if delta == 0:
                streak[fam] += 1
            else:
                streak[fam] = 0
            if streak[fam] >= args.plateau_streak:
                plateaued.add(fam)
                print(f'[loop] family {fam} plateaued after {streak[fam]} '
                       f'zero-delta batches')

            print(f'[loop] batch {batch_n}  family={fam}  rc={rc}  '
                   f'new={new}  delta[{fam}]={delta}  streak[{fam}]={streak[fam]}')

            # Minimize pending for THIS family first, then any family.
            pending = db.pending_minimizations(limit=args.minimize_per_cycle,
                                                family=fam)
            if len(pending) < args.minimize_per_cycle:
                pending += db.pending_minimizations(
                    limit=args.minimize_per_cycle - len(pending))
            if pending:
                print(f'[loop] minimizing {len(pending)} fresh divergences ...')
                for sha in pending:
                    try:
                        minimize_one(sha, db)
                    except subprocess.TimeoutExpired:
                        print(f'  [min] {sha}: TIMEOUT — skipping')

            db.update_campaign(cid, last_seed=cur_seed, iters=batch_n)
            print_status(db, per_family=True)
    except KeyboardInterrupt:
        print('\n[loop] interrupted')
    finally:
        db.end_campaign(cid)
        db.close()


def cmd_status(args):
    db = BugDB(_DB_PATH)
    print_status(db, per_family=args.per_family)
    db.close()


def cmd_clusters(args):
    db = BugDB(_DB_PATH)
    clusters = db.minimal_clusters(family=args.family, limit=args.n)
    if not clusters:
        print('(no minimized clusters yet)')
    else:
        for fam_sig, n, min_body, example_sha in clusters:
            print(f'n={n:4d}  min_body={min_body:2d}  sha={example_sha}  {fam_sig}')
    db.close()


def cmd_report(args):
    """Generate a per-family markdown report under _fuzz/report/."""
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    db = BugDB(_DB_PATH)
    families = all_families()
    index_lines = ['# Fuzzer bug surface report',
                   '',
                   f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}',
                   '',
                   '## Families', '']
    for fam in families:
        rows = db.family_status()
        fam_rows = [r for r in rows if r[0] == fam]
        clusters = db.minimal_clusters(family=fam, limit=50)
        path = _REPORT_DIR / f'{fam}.md'
        lines = [f'# Family: `{fam}`',
                 '',
                 '## Tag counts', '',
                 '| tag | artifacts | well-formed |',
                 '|-----|-----------|-------------|']
        total = 0; wf_total = 0
        for _, tag, n, wf in fam_rows:
            lines.append(f'| {tag} | {n} | {wf or 0} |')
            total += n; wf_total += (wf or 0)
        lines.append(f'| **total** | **{total}** | **{wf_total}** |')
        lines.append('')
        lines.append('## Minimal-PTX clusters (fixed-point minimizations)')
        lines.append('')
        if not clusters:
            lines.append('_none yet — run minimizer longer_')
        else:
            lines.append('| n | min-body | family signature | example sha |')
            lines.append('|---|----------|------------------|-------------|')
            for fam_sig, n, min_body, sha in clusters:
                lines.append(f'| {n} | {min_body} | `{fam_sig}` | `{sha}` |')
        path.write_text('\n'.join(lines))
        index_lines.append(f'- [{fam}]({fam}.md) — {total} artifacts, {wf_total} well-formed')
    (_REPORT_DIR / 'index.md').write_text('\n'.join(index_lines))
    print(f'[report] wrote {_REPORT_DIR}')
    db.close()


def main():
    ap = argparse.ArgumentParser(prog='fuzzer.loop')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('run', help='continuous fuzz + triage + minimize')
    p.add_argument('--minutes', type=int, default=None,
                   help='stop after N minutes (default: run until plateau or Ctrl-C)')
    p.add_argument('--batch-seconds', type=int, default=45)
    p.add_argument('--iters-per-worker', type=int, default=100)
    p.add_argument('--minimize-per-cycle', type=int, default=2)
    p.add_argument('--seed-base', type=int, default=None)
    p.add_argument('--families', default='all',
                   help="comma-separated families or 'all' (default: all)")
    p.add_argument('--plateau-streak', type=int, default=4,
                   help='N consecutive zero-delta batches -> family marked plateaued')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('status', help='dump DB summary')
    p.add_argument('--per-family', action='store_true')
    p.set_defaults(func=cmd_status)

    p = sub.add_parser('clusters', help='top minimal-PTX signature clusters')
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--family', default=None)
    p.set_defaults(func=cmd_clusters)

    p = sub.add_parser('report', help='generate markdown report per family')
    p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
