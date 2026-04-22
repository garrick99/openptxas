"""Factory shared store.

One central `programs` table with stage gates.  Each daemon polls the rows
pending its stage, processes them, advances the gate.  WAL mode lets
multiple daemon processes write concurrently.
"""
import hashlib, sqlite3, time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
# The factory DB is high-write.  On WSL the repo lives on the /mnt/c
# 9P filesystem which is slow and has weak SQLite-WAL guarantees.  Set
# FACTORY_DB_PATH to a native-Linux path when running from WSL (e.g.
# ~/factory/factory.db) to avoid corruption under concurrent writers.
import os as _os
DB_PATH = Path(_os.environ.get('FACTORY_DB_PATH', str(_ROOT / 'factory' / 'factory.db')))
REPORT_DIR = Path(_os.environ.get('FACTORY_NVIDIA_BUGS_DIR',
                                    str(_ROOT / '_nvidia_bugs_auto')))
OPENPTXAS_REPORT_DIR = Path(_os.environ.get('FACTORY_OPENPTXAS_BUGS_DIR',
                                    str(_ROOT / '_openptxas_bugs_auto')))
BOTH_WRONG_REPORT_DIR = Path(_os.environ.get('FACTORY_BOTH_WRONG_DIR',
                                    str(_ROOT / '_both_wrong_auto')))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS programs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ptx_sha       TEXT NOT NULL UNIQUE,
    ptx           TEXT NOT NULL,
    seed          INTEGER,
    family        TEXT,
    source        TEXT,         -- 'random' | 'truth_table' | 'minimized' | 'seed'
    parent_id     INTEGER,      -- for 'minimized': id of the program we shrunk
    created_at    INTEGER NOT NULL,

    -- Stage gates (non-NULL timestamp = done)
    differ_done   INTEGER,
    differ_state  TEXT,         -- 'ok' | 'divergence' | 'sync_err_ours' | 'sync_err_theirs'
                                -- 'compile_err_ours' | 'compile_err_theirs' | 'timeout'
    differ_note   TEXT,

    minimize_done INTEGER,
    minimal_id    INTEGER,      -- id of the minimized version (another programs row)

    classify_done INTEGER,
    class_sig     TEXT,

    spec_checked  INTEGER,
    spec_verdict  TEXT,         -- 'ours_correct' | 'theirs_correct'
                                -- | 'both_correct' | 'both_wrong' | 'unsupported'
    spec_expected BLOB,         -- 128-byte expected output per spec

    report_done   INTEGER,
    report_path   TEXT
);

CREATE INDEX IF NOT EXISTS idx_programs_differ_pending
    ON programs(differ_done) WHERE differ_done IS NULL;
CREATE INDEX IF NOT EXISTS idx_programs_classify_pending
    ON programs(classify_done, differ_state)
    WHERE classify_done IS NULL AND differ_state = 'divergence';
CREATE INDEX IF NOT EXISTS idx_programs_minimize_pending
    ON programs(minimize_done, differ_state)
    WHERE minimize_done IS NULL AND differ_state = 'divergence';
CREATE INDEX IF NOT EXISTS idx_programs_spec_pending
    ON programs(spec_checked, differ_state)
    WHERE spec_checked IS NULL AND differ_state = 'divergence';
CREATE INDEX IF NOT EXISTS idx_programs_class_sig
    ON programs(class_sig);
CREATE INDEX IF NOT EXISTS idx_programs_spec_verdict
    ON programs(spec_verdict);

CREATE TABLE IF NOT EXISTS differences (
    program_id    INTEGER PRIMARY KEY REFERENCES programs(id),
    inputs_blob   BLOB NOT NULL,        -- 128 bytes fed to both cubins
    out_ours      BLOB,                 -- 128 bytes or NULL on error
    out_theirs    BLOB,
    diff_lanes    INTEGER,
    ours_err      TEXT,
    theirs_err    TEXT
);

CREATE TABLE IF NOT EXISTS classes (
    sig              TEXT PRIMARY KEY,
    specimen_count   INTEGER NOT NULL DEFAULT 0,
    first_seen       INTEGER NOT NULL,
    last_seen        INTEGER NOT NULL,
    canonical_id     INTEGER,                -- minimal program in the class
    spec_verdict     TEXT,                   -- verdict for the canonical specimen
    escalated        INTEGER DEFAULT 0,      -- 1 = reporter should emit dossier
    reported         INTEGER DEFAULT 0       -- 1 = dossier already emitted
);

CREATE TABLE IF NOT EXISTS daemons (
    name             TEXT PRIMARY KEY,
    last_tick        INTEGER NOT NULL,
    items_processed  INTEGER DEFAULT 0,
    state            TEXT NOT NULL           -- 'running' | 'idle' | 'error:...'
);

CREATE TABLE IF NOT EXISTS reports (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    class_sig        TEXT NOT NULL,
    program_id       INTEGER NOT NULL,
    path             TEXT NOT NULL,
    created_at       INTEGER NOT NULL
);

-- Small persistent key/value store (used for the generator's seed
-- high-water mark so we can DELETE ok-program rows without losing it).
CREATE TABLE IF NOT EXISTS kv (
    k TEXT PRIMARY KEY,
    v INTEGER NOT NULL
);

-- Running totals that survive differ-side row deletion (so
-- `supervisor status` shows real throughput, not just current
-- live rows).
CREATE TABLE IF NOT EXISTS counters (
    k TEXT PRIMARY KEY,
    v INTEGER NOT NULL DEFAULT 0
);
"""


def ptx_sha(ptx: str) -> str:
    return hashlib.sha256(ptx.encode()).hexdigest()[:16]


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA busy_timeout=30000')
    # Trigger a WAL checkpoint every 1000 committed frames so the WAL
    # file can't run away the way it did the first time around.
    conn.execute('PRAGMA wal_autocheckpoint=1000')
    conn.executescript(_SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def checkpoint(conn):
    """Fold WAL back into main DB and truncate.  Cheap no-op if WAL tiny."""
    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

def insert_program(conn, ptx: str, *, seed: Optional[int] = None,
                   family: str = 'random', source: str = 'random',
                   parent_id: Optional[int] = None) -> Optional[int]:
    """Insert a program.  Returns id, or None if ptx_sha already exists."""
    sha = ptx_sha(ptx)
    try:
        cur = conn.execute(
            'INSERT INTO programs(ptx_sha, ptx, seed, family, source, '
            'parent_id, created_at) VALUES (?,?,?,?,?,?,?)',
            (sha, ptx, seed, family, source, parent_id, int(time.time())))
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None  # duplicate ptx_sha


def claim_next(conn, gate_col: str, where: str = '') -> Optional[sqlite3.Row]:
    """Return one row with NULL at gate_col, or None.  Caller should update
    the gate promptly to avoid another daemon re-claiming."""
    q = (f'SELECT * FROM programs WHERE {gate_col} IS NULL ' +
         (f'AND {where} ' if where else '') +
         'ORDER BY id ASC LIMIT 1')
    row = conn.execute(q).fetchone()
    return row


def set_gate(conn, program_id: int, gate_col: str, **fields):
    fields[gate_col] = int(time.time())
    sets = ', '.join(f'{k}=?' for k in fields)
    vals = list(fields.values()) + [program_id]
    conn.execute(f'UPDATE programs SET {sets} WHERE id=?', vals)


# ---------------------------------------------------------------------------
# Differences
# ---------------------------------------------------------------------------

def record_difference(conn, program_id: int, inputs: bytes,
                      out_ours: Optional[bytes], out_theirs: Optional[bytes],
                      diff_lanes: Optional[int],
                      ours_err: Optional[str], theirs_err: Optional[str]):
    conn.execute(
        'INSERT OR REPLACE INTO differences(program_id, inputs_blob, out_ours, '
        'out_theirs, diff_lanes, ours_err, theirs_err) VALUES (?,?,?,?,?,?,?)',
        (program_id, inputs, out_ours, out_theirs, diff_lanes, ours_err, theirs_err))


def get_difference(conn, program_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        'SELECT * FROM differences WHERE program_id=?', (program_id,)).fetchone()


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

import os as _os
CLASS_ESCALATION_THRESHOLD = int(_os.environ.get('FACTORY_CLASS_THRESHOLD', '3'))


def bump_class(conn, sig: str, program_id: int,
               threshold: int = CLASS_ESCALATION_THRESHOLD) -> sqlite3.Row:
    """Increment specimen_count for a class; set canonical to the smallest
    program body seen so far.  Escalates when count crosses threshold."""
    now = int(time.time())
    row = conn.execute('SELECT * FROM classes WHERE sig=?', (sig,)).fetchone()
    if row is None:
        conn.execute(
            'INSERT INTO classes(sig, specimen_count, first_seen, last_seen, '
            'canonical_id) VALUES (?, 1, ?, ?, ?)',
            (sig, now, now, program_id))
    else:
        # Pick canonical = shortest ptx in the class (prefer minimized).
        canonical_id = row['canonical_id']
        cand = conn.execute(
            'SELECT id, length(ptx) AS L FROM programs WHERE id IN (?, ?) '
            'ORDER BY L ASC LIMIT 1', (canonical_id, program_id)).fetchone()
        canonical_id = cand['id']
        new_count = row['specimen_count'] + 1
        escalated = 1 if new_count >= threshold else row['escalated']
        conn.execute(
            'UPDATE classes SET specimen_count=?, last_seen=?, canonical_id=?, '
            'escalated=? WHERE sig=?',
            (new_count, now, canonical_id, escalated, sig))
    return conn.execute('SELECT * FROM classes WHERE sig=?', (sig,)).fetchone()


def set_class_verdict(conn, sig: str, verdict: str):
    conn.execute('UPDATE classes SET spec_verdict=? WHERE sig=?', (verdict, sig))


def claim_unreported_class(conn) -> Optional[sqlite3.Row]:
    """Any escalated, not-yet-reported class with a definitive verdict.
    Caller branches on spec_verdict to route to the right template/dir."""
    return conn.execute(
        "SELECT * FROM classes WHERE escalated=1 AND reported=0 "
        "AND spec_verdict IN ('theirs_wrong', 'ours_wrong', 'both_wrong') "
        "ORDER BY sig ASC LIMIT 1").fetchone()


def mark_class_reported(conn, sig: str, path: str):
    conn.execute('UPDATE classes SET reported=1 WHERE sig=?', (sig,))


# ---------------------------------------------------------------------------
# Daemon heartbeats
# ---------------------------------------------------------------------------

def kv_get(conn, key: str, default: int = 0) -> int:
    row = conn.execute('SELECT v FROM kv WHERE k=?', (key,)).fetchone()
    return row['v'] if row else default


def kv_set(conn, key: str, val: int):
    conn.execute('INSERT OR REPLACE INTO kv(k, v) VALUES (?, ?)', (key, val))


def counter_add(conn, key: str, delta: int = 1):
    conn.execute('INSERT INTO counters(k, v) VALUES (?, ?) '
                 'ON CONFLICT(k) DO UPDATE SET v = v + excluded.v',
                 (key, delta))


def counter_get(conn, key: str) -> int:
    row = conn.execute('SELECT v FROM counters WHERE k=?', (key,)).fetchone()
    return row['v'] if row else 0


def heartbeat(conn, name: str, state: str, items_processed: int = 0):
    now = int(time.time())
    conn.execute(
        'INSERT INTO daemons(name, last_tick, state, items_processed) '
        'VALUES (?,?,?,?) ON CONFLICT(name) DO UPDATE SET '
        'last_tick=excluded.last_tick, state=excluded.state, '
        'items_processed=daemons.items_processed + excluded.items_processed',
        (name, now, state, items_processed))


# ---------------------------------------------------------------------------
# Summary (for status CLI)
# ---------------------------------------------------------------------------

def summary(conn) -> dict:
    s = {}
    # Lifetime totals (via counters — survive row deletion of 'ok' programs)
    s['programs_generated'] = counter_get(conn, 'programs_generated')
    s['programs_differ_ok'] = counter_get(conn, 'programs_differ_ok')
    s['programs_differ_err'] = counter_get(conn, 'programs_differ_err')
    # Live rows (divergences not yet deleted, plus anything in flight)
    s['live_rows'] = conn.execute('SELECT COUNT(*) FROM programs').fetchone()[0]
    s['differ_pending'] = conn.execute(
        'SELECT COUNT(*) FROM programs WHERE differ_done IS NULL').fetchone()[0]
    s['divergences'] = conn.execute(
        "SELECT COUNT(*) FROM programs WHERE differ_state='divergence'").fetchone()[0]
    s['minimized'] = conn.execute(
        "SELECT COUNT(*) FROM programs WHERE source='minimized'").fetchone()[0]
    s['classified'] = conn.execute(
        'SELECT COUNT(*) FROM programs WHERE classify_done IS NOT NULL').fetchone()[0]
    s['classes'] = conn.execute('SELECT COUNT(*) FROM classes').fetchone()[0]
    s['escalated_classes'] = conn.execute(
        'SELECT COUNT(*) FROM classes WHERE escalated=1').fetchone()[0]
    s['ptxas_wrong_classes'] = conn.execute(
        "SELECT COUNT(*) FROM classes WHERE spec_verdict='theirs_wrong'").fetchone()[0]
    s['openptxas_wrong_classes'] = conn.execute(
        "SELECT COUNT(*) FROM classes WHERE spec_verdict='ours_wrong'").fetchone()[0]
    s['reports'] = conn.execute('SELECT COUNT(*) FROM reports').fetchone()[0]
    s['daemons'] = conn.execute('SELECT * FROM daemons').fetchall()
    return s
