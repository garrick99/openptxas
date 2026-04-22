"""SQLite-backed bug database for the fuzz loop.

One row per unique PTX-hash artifact.  Campaign table logs each
invocation of the loop (start/end, seeds covered, aggregate counts).
"""
import json, sqlite3, time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    sha                 TEXT PRIMARY KEY,
    tag                 TEXT NOT NULL,
    first_seen          INTEGER NOT NULL,
    last_seen           INTEGER NOT NULL,
    hits                INTEGER NOT NULL DEFAULT 1,
    body_lines          INTEGER,
    well_formed         INTEGER,
    full_sig            TEXT,
    full_family_sig     TEXT,
    minimal_sig         TEXT,
    minimal_family_sig  TEXT,
    minimal_body_lines  INTEGER,
    minimized_outcome   TEXT,
    seed                INTEGER
);

CREATE INDEX IF NOT EXISTS idx_tag                ON artifacts(tag);
CREATE INDEX IF NOT EXISTS idx_wf                 ON artifacts(well_formed);
CREATE INDEX IF NOT EXISTS idx_minimal_sig        ON artifacts(minimal_sig);
CREATE INDEX IF NOT EXISTS idx_minimal_family_sig ON artifacts(minimal_family_sig);

CREATE TABLE IF NOT EXISTS campaigns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started       INTEGER NOT NULL,
    ended         INTEGER,
    seed_base     INTEGER,
    last_seed     INTEGER,
    iters         INTEGER DEFAULT 0,
    ok            INTEGER DEFAULT 0,
    divergence    INTEGER DEFAULT 0,
    sync_err_ours INTEGER DEFAULT 0,
    ctx_dead      INTEGER DEFAULT 0,
    other_json    TEXT
);
"""


class BugDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        """Add columns introduced after initial release."""
        cur = self.conn.execute('PRAGMA table_info(artifacts)')
        cols = {r[1] for r in cur.fetchall()}
        if 'family' not in cols:
            self.conn.execute('ALTER TABLE artifacts ADD COLUMN family TEXT')
        cur = self.conn.execute('PRAGMA table_info(campaigns)')
        cols = {r[1] for r in cur.fetchall()}
        if 'families' not in cols:
            self.conn.execute('ALTER TABLE campaigns ADD COLUMN families TEXT')

    def start_campaign(self, seed_base: int) -> int:
        cur = self.conn.execute(
            'INSERT INTO campaigns(started, seed_base, last_seed) VALUES (?,?,?)',
            (int(time.time()), seed_base, seed_base))
        self.conn.commit()
        return cur.lastrowid

    def update_campaign(self, cid: int, **fields):
        if not fields: return
        sets = ', '.join(f'{k}=?' for k in fields)
        vals = list(fields.values()) + [cid]
        self.conn.execute(f'UPDATE campaigns SET {sets} WHERE id=?', vals)
        self.conn.commit()

    def end_campaign(self, cid: int):
        self.conn.execute('UPDATE campaigns SET ended=? WHERE id=?',
                          (int(time.time()), cid))
        self.conn.commit()

    def upsert_artifact(self, sha: str, tag: str, body_lines: int,
                        well_formed: bool, full_sig: str,
                        full_family_sig: str, seed: int | None,
                        family: str | None = None) -> bool:
        """Insert new row or bump hits.  Returns True if newly inserted."""
        now = int(time.time())
        cur = self.conn.execute('SELECT sha FROM artifacts WHERE sha=?', (sha,))
        if cur.fetchone() is not None:
            self.conn.execute(
                'UPDATE artifacts SET hits=hits+1, last_seen=? WHERE sha=?',
                (now, sha))
            self.conn.commit()
            return False
        self.conn.execute(
            '''INSERT INTO artifacts
               (sha, tag, first_seen, last_seen, body_lines, well_formed,
                full_sig, full_family_sig, seed, family)
               VALUES (?,?,?,?,?,?,?,?,?,?)''',
            (sha, tag, now, now, body_lines, int(well_formed),
             full_sig, full_family_sig, seed, family))
        self.conn.commit()
        return True

    def record_minimization(self, sha: str, outcome: str,
                             min_body_lines: int, min_sig: str,
                             min_family_sig: str):
        self.conn.execute(
            '''UPDATE artifacts
               SET minimized_outcome=?, minimal_body_lines=?,
                   minimal_sig=?, minimal_family_sig=?
               WHERE sha=?''',
            (outcome, min_body_lines, min_sig, min_family_sig, sha))
        self.conn.commit()

    def pending_minimizations(self, limit: int = 5,
                               family: str | None = None) -> list[str]:
        """Well-formed divergences not yet minimized, shortest body first."""
        if family:
            cur = self.conn.execute(
                '''SELECT sha FROM artifacts
                   WHERE tag='divergence' AND well_formed=1
                     AND minimized_outcome IS NULL
                     AND family=?
                   ORDER BY body_lines ASC, first_seen ASC
                   LIMIT ?''', (family, limit))
        else:
            cur = self.conn.execute(
                '''SELECT sha FROM artifacts
                   WHERE tag='divergence' AND well_formed=1
                     AND minimized_outcome IS NULL
                   ORDER BY body_lines ASC, first_seen ASC
                   LIMIT ?''', (limit,))
        return [r[0] for r in cur.fetchall()]

    def count_artifacts(self, family: str | None = None) -> int:
        if family:
            cur = self.conn.execute(
                'SELECT COUNT(*) FROM artifacts WHERE family=?', (family,))
        else:
            cur = self.conn.execute('SELECT COUNT(*) FROM artifacts')
        return cur.fetchone()[0]

    def family_status(self) -> list[tuple]:
        """Return (family, tag, count, well_formed_count) tuples."""
        cur = self.conn.execute(
            '''SELECT COALESCE(family, '(none)'), tag, COUNT(*), SUM(well_formed)
               FROM artifacts GROUP BY family, tag
               ORDER BY family, tag''')
        return cur.fetchall()

    def minimal_clusters(self, family: str | None = None, limit: int = 20):
        """Per-family clusters by minimal-family-signature."""
        if family:
            cur = self.conn.execute(
                '''SELECT minimal_family_sig, COUNT(*) as n,
                          MIN(minimal_body_lines) as min_body,
                          MIN(sha) as example_sha
                   FROM artifacts
                   WHERE minimal_family_sig IS NOT NULL
                     AND minimized_outcome='fixed_point'
                     AND family=?
                   GROUP BY minimal_family_sig
                   ORDER BY n DESC LIMIT ?''', (family, limit))
        else:
            cur = self.conn.execute(
                '''SELECT minimal_family_sig, COUNT(*) as n,
                          MIN(minimal_body_lines) as min_body,
                          MIN(sha) as example_sha
                   FROM artifacts
                   WHERE minimal_family_sig IS NOT NULL
                     AND minimized_outcome='fixed_point'
                   GROUP BY minimal_family_sig
                   ORDER BY n DESC LIMIT ?''', (limit,))
        return cur.fetchall()

    def status_counts(self) -> dict:
        cur = self.conn.execute(
            '''SELECT tag, COUNT(*), SUM(well_formed)
               FROM artifacts GROUP BY tag''')
        out = {}
        for tag, n, nwf in cur.fetchall():
            out[tag] = {'unique': n, 'well_formed': nwf or 0}
        return out

    def top_minimal_signatures(self, limit: int = 10) -> list[tuple]:
        cur = self.conn.execute(
            '''SELECT minimal_family_sig, COUNT(*) as n,
                      MIN(minimal_body_lines) as min_body
               FROM artifacts
               WHERE minimal_family_sig IS NOT NULL
                 AND minimized_outcome='fixed_point'
               GROUP BY minimal_family_sig
               ORDER BY n DESC
               LIMIT ?''', (limit,))
        return cur.fetchall()

    def close(self):
        self.conn.close()
