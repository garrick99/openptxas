"""Minimizer daemon.

For each divergent program that hasn't been minimized, delta-debug the
body to the smallest chain that still reproduces the divergence.
Insert the minimized version as a new programs row with
source='minimized' and parent_id set; it will flow through the differ
(and agree, since it reproduces by definition) and then the classifier
picks up the shorter signature.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from factory import db
from factory.daemon import Daemon, cli_main
from fuzzer.oracle import CudaRunner
from fuzzer.minimize import minimize as _minimize


class MinimizerDaemon(Daemon):
    NAME = 'minimizer'
    IDLE_SLEEP_SEC = 5.0

    def __init__(self, **kw):
        super().__init__(**kw)
        self.runner = CudaRunner()

    def tick(self) -> bool:
        row = self.conn.execute(
            "SELECT p.*, d.inputs_blob FROM programs p "
            "LEFT JOIN differences d ON d.program_id = p.id "
            "WHERE p.minimize_done IS NULL "
            "AND p.differ_state = 'divergence' "
            "AND p.source != 'minimized' "
            "ORDER BY p.id ASC LIMIT 1").fetchone()
        if row is None:
            return False

        pid = row['id']
        inputs = row['inputs_blob']
        if inputs is None:
            db.set_gate(self.conn, pid, 'minimize_done')
            return True

        try:
            minimal, outcome = _minimize(row['ptx'], inputs, self.runner,
                                          verbose=False)
        except Exception as e:
            db.set_gate(self.conn, pid, 'minimize_done')
            return True

        # Only insert as a new program if it actually shrank.
        min_id = None
        if minimal and minimal != row['ptx']:
            min_id = db.insert_program(
                self.conn, minimal, family=row['family'],
                source='minimized', parent_id=pid)

        db.set_gate(self.conn, pid, 'minimize_done', minimal_id=min_id)
        return True


if __name__ == '__main__':
    cli_main(MinimizerDaemon)
