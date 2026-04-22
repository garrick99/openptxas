"""Factory supervisor.

Launches each daemon as a subprocess, monitors heartbeats, restarts on
crash, shuts cleanly on Ctrl+C.  Usage:

    python -m factory.supervisor run [--seconds N]
    python -m factory.supervisor status
    python -m factory.supervisor seed --ptx <file>     # inject a known PTX
    python -m factory.supervisor seed-bug2             # inject the or+shr Bug 2
    python -m factory.supervisor seed-bug1             # inject the bfe.s32 Bug 1

The differ daemon owns the GPU serially (BigDaddy rule — no parallel
CUDA contexts).  Generator, classifier, oracle, reporter all run in
parallel with the differ; they don't touch the GPU.
"""
import argparse, signal, subprocess, sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from factory import db

_DAEMONS = [
    ('generator',  'factory.generator_d'),
    ('differ',     'factory.differ_d'),
    ('classifier', 'factory.classifier_d'),
    ('minimizer',  'factory.minimizer_d'),
    ('oracle',     'factory.oracle_d'),
    ('reporter',   'factory.reporter_d'),
]


def _launch(mod_name: str, extra_args):
    return subprocess.Popen(
        [sys.executable, '-m', mod_name, *extra_args],
        cwd=str(_REPO),
        stdout=sys.stdout, stderr=sys.stderr)


def run(seconds: float = None):
    extra = []
    if seconds is not None:
        extra = ['--seconds', str(seconds)]
    procs = {}
    for name, mod in _DAEMONS:
        print(f'[supervisor] launching {name}')
        procs[name] = _launch(mod, extra)

    stopped = False
    def _h(sig, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, _h)

    started = time.time()
    try:
        while not stopped:
            # Restart any that died (unless we're winding down)
            for name, mod in _DAEMONS:
                p = procs[name]
                rc = p.poll()
                if rc is not None:
                    if seconds is not None and time.time() - started > seconds:
                        continue
                    print(f'[supervisor] {name} exited rc={rc}; restarting')
                    procs[name] = _launch(mod, extra)
            if seconds is not None and time.time() - started > seconds + 5:
                break
            time.sleep(2.0)
    finally:
        print('[supervisor] stopping all daemons')
        for name, p in procs.items():
            try: p.terminate()
            except Exception: pass
        # Give them a moment
        for name, p in procs.items():
            try: p.wait(timeout=10)
            except Exception:
                try: p.kill()
                except Exception: pass


def status():
    conn = db.connect()
    s = db.summary(conn)
    db_bytes = db.DB_PATH.stat().st_size if db.DB_PATH.exists() else 0
    wal_path = db.DB_PATH.with_name(db.DB_PATH.name + '-wal')
    wal_bytes = wal_path.stat().st_size if wal_path.exists() else 0
    print('=== Factory status ===')
    print(f'  db_size:       {db_bytes/1e6:9.1f} MB')
    print(f'  wal_size:      {wal_bytes/1e6:9.1f} MB')
    for k, v in s.items():
        if k == 'daemons': continue
        print(f'  {k}: {v}')
    print()
    print('Daemons:')
    now = int(time.time())
    for d in s['daemons']:
        age = now - d['last_tick']
        print(f'  {d["name"]:12s} state={d["state"]:20s} '
              f'items={d["items_processed"]:6d}  last_tick={age}s ago')
    # Top classes
    print()
    print('Top classes:')
    for r in conn.execute(
        'SELECT sig, specimen_count, spec_verdict, escalated, reported '
        'FROM classes ORDER BY specimen_count DESC LIMIT 15'):
        print(f'  n={r["specimen_count"]:4d}  verdict={r["spec_verdict"] or "-":14s} '
              f'esc={r["escalated"]} rep={r["reported"]}  {r["sig"]}')


def seed_ptx_file(path: str):
    from factory.generator_d import seed_program
    ptx = Path(path).read_text()
    pid = seed_program(ptx, family='seed', source='seed')
    print(f'inserted seed program id={pid}')


_BUG2_PTX = """.version 9.0
.target sm_120
.address_size 64
.visible .entry fuzz(.param .u64 p_in, .param .u64 p_out, .param .u32 n) {
    .reg .b32 %r<32>;
    .reg .b64 %rd<8>;
    .reg .pred %p<2>;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 ret;
    ld.param.u64 %rd0, [p_in];
    cvt.u64.u32 %rd1, %r0;
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r3, [%rd2];
    or.b32 %r4, %r3, 2147483648;
    shr.s32 %r5, %r4, 2;
    shr.u32 %r6, %r5, 30;
    ld.param.u64 %rd3, [p_out];
    add.u64 %rd4, %rd3, %rd1;
    st.global.u32 [%rd4], %r6;
    ret;
}
"""

_BUG1_PTX = """.version 9.0
.target sm_120
.address_size 64
.visible .entry fuzz(.param .u64 p_in, .param .u64 p_out, .param .u32 n) {
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    .reg .pred %p<2>;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n];
    setp.ge.u32 %p0, %r0, %r1;
    @%p0 ret;
    ld.param.u64 %rd0, [p_in];
    cvt.u64.u32 %rd1, %r0;
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r3, [%rd2];
    bfe.s32 %r2, %r3, 32, 1;
    ld.param.u64 %rd3, [p_out];
    add.u64 %rd4, %rd3, %rd1;
    st.global.u32 [%rd4], %r2;
    ret;
}
"""


def coverage_gaps(limit: int = 30):
    """List unsupported opcodes by frequency — the spec simulator's TODO list."""
    conn = db.connect()
    print(f'Top {limit} unsupported opcodes blocking the spec oracle:')
    print()
    print(f'  count | opcode')
    print(f'  ------|-------')
    rows = conn.execute(
        "SELECT substr(spec_verdict, 13) AS op, COUNT(*) AS n "
        "FROM programs WHERE spec_verdict LIKE 'unsupported:%' "
        "GROUP BY op ORDER BY n DESC LIMIT ?", (limit,)).fetchall()
    for r in rows:
        print(f'  {r["n"]:5d} | {r["op"]}')
    if not rows:
        print('  (none — spec simulator handles every PTX seen so far)')


def watch_bugs():
    """Block, polling the reports table, alert on each new report.

    Run in a separate terminal while `supervisor run` is going on.
    """
    conn = db.connect()
    seen = set(r['id'] for r in conn.execute('SELECT id FROM reports'))
    print(f'[watch-bugs] polling factory DB; starting with {len(seen)} existing '
          f'reports.  Ctrl+C to stop.')
    try:
        while True:
            rows = conn.execute(
                'SELECT r.*, c.specimen_count FROM reports r '
                'LEFT JOIN classes c ON c.sig = r.class_sig '
                'ORDER BY r.id DESC').fetchall()
            for r in rows:
                if r['id'] in seen:
                    continue
                seen.add(r['id'])
                bar = '=' * 72
                print(f'\a\n{bar}\nNEW PTXAS BUG CANDIDATE')
                print(f'  at:        {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}')
                print(f'  class:     {r["class_sig"]}')
                print(f'  specimens: {r["specimen_count"]}')
                print(f'  dossier:   {r["path"]}')
                print(bar, flush=True)
            time.sleep(5.0)
    except KeyboardInterrupt:
        print('\n[watch-bugs] stopped')


def seed_known(which: str):
    from factory.generator_d import seed_program
    templates = []
    if which == 'bug2':
        # N in {1, 2, 3} all trigger the ptxas fold; 3 distinct ptx_sha,
        # same family signature.
        for N in (1, 2, 3):
            t = _BUG2_PTX.replace(
                'shr.s32 %r5, %r4, 2;', f'shr.s32 %r5, %r4, {N};').replace(
                'shr.u32 %r6, %r5, 30;', f'shr.u32 %r6, %r5, {32-N};')
            templates.append(t)
    elif which == 'bug1':
        # 3 distinct (start, len) pairs in the OOR region.
        for s, l in ((32, 1), (32, 16), (33, 1)):
            t = _BUG1_PTX.replace('bfe.s32 %r2, %r3, 32, 1;',
                                    f'bfe.s32 %r2, %r3, {s}, {l};')
            templates.append(t)
    else:
        raise ValueError(which)
    for t in templates:
        pid = seed_program(t, family='seed', source='seed')
        print(f'inserted known-{which} seed id={pid}')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    pr = sub.add_parser('run')
    pr.add_argument('--seconds', type=float, default=None)
    sub.add_parser('status')
    sub.add_parser('watch-bugs')
    pc = sub.add_parser('coverage-gaps')
    pc.add_argument('--limit', type=int, default=30)
    ps = sub.add_parser('seed')
    ps.add_argument('--ptx', required=True)
    sub.add_parser('seed-bug1')
    sub.add_parser('seed-bug2')
    args = p.parse_args()

    if args.cmd == 'run':
        run(seconds=args.seconds)
    elif args.cmd == 'status':
        status()
    elif args.cmd == 'watch-bugs':
        watch_bugs()
    elif args.cmd == 'coverage-gaps':
        coverage_gaps(limit=args.limit)
    elif args.cmd == 'seed':
        seed_ptx_file(args.ptx)
    elif args.cmd == 'seed-bug1':
        seed_known('bug1')
    elif args.cmd == 'seed-bug2':
        seed_known('bug2')


if __name__ == '__main__':
    main()
