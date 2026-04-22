"""Classifier daemon.

For each divergent program, compute a canonical op-family signature
and upsert the classes row.  When a class crosses the specimen-count
threshold, mark it escalated so the reporter picks it up.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from factory import db
from factory.daemon import Daemon, cli_main
from fuzzer.classify import family_signature


class ClassifierDaemon(Daemon):
    NAME = 'classifier'
    IDLE_SLEEP_SEC = 3.0

    def tick(self) -> bool:
        row = self.conn.execute(
            "SELECT * FROM programs WHERE classify_done IS NULL "
            "AND differ_state='divergence' "
            "ORDER BY id ASC LIMIT 1").fetchone()
        if row is None:
            return False

        sig = family_signature(row['ptx'])
        db.bump_class(self.conn, sig, row['id'])
        db.set_gate(self.conn, row['id'], 'classify_done', class_sig=sig)

        # If this program was already oracle'd, propagate its verdict to
        # the class.  (Handles the race where oracle runs before classifier.)
        prog = self.conn.execute(
            'SELECT spec_verdict FROM programs WHERE id=?',
            (row['id'],)).fetchone()
        if prog and prog['spec_verdict']:
            class_verdict = {
                'ours_correct':   'theirs_wrong',
                'theirs_correct': 'ours_wrong',
                'both_wrong':     'both_wrong',
                'both_correct':   'both_correct',
            }.get(prog['spec_verdict'])
            if class_verdict:
                db.set_class_verdict(self.conn, sig, class_verdict)
        return True


if __name__ == '__main__':
    cli_main(ClassifierDaemon)
