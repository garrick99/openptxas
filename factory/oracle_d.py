"""Spec oracle daemon.

For each divergent program, run `factory.spec.simulate` on the same
inputs to get the spec-correct output, then decide:

    'ours_correct'    — out_ours == spec != out_theirs
    'theirs_correct'  — out_theirs == spec != out_ours
    'both_correct'    — both equal spec (shouldn't happen given divergence)
    'both_wrong'      — neither matches spec
    'unsupported'     — body contains an opcode the simulator doesn't handle

When a program gets 'ours_correct', that's a ptxas bug candidate.  The
daemon also updates the class's spec_verdict (if class is already
assigned) so the reporter can pick up the class as a whole.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from factory import db
from factory.daemon import Daemon, cli_main
from factory.spec import simulate, Unsupported


class OracleDaemon(Daemon):
    NAME = 'oracle'
    IDLE_SLEEP_SEC = 2.0

    def tick(self) -> bool:
        row = self.conn.execute(
            "SELECT p.*, d.inputs_blob, d.out_ours, d.out_theirs "
            "FROM programs p JOIN differences d ON d.program_id = p.id "
            "WHERE p.spec_checked IS NULL "
            "AND p.differ_state = 'divergence' "
            "ORDER BY p.id ASC LIMIT 1").fetchone()
        if row is None:
            return False

        pid = row['id']
        inputs = row['inputs_blob']
        out_ours = row['out_ours']
        out_theirs = row['out_theirs']

        try:
            spec = simulate(row['ptx'], inputs)
        except Unsupported as e:
            # Record the exact opcode so we can build a coverage-gap
            # frequency table via SQL.
            op = str(e).strip()[:40] or 'unknown'
            db.set_gate(self.conn, pid, 'spec_checked',
                        spec_verdict=f'unsupported:{op}')
            return True
        except Exception as e:
            db.set_gate(self.conn, pid, 'spec_checked',
                        spec_verdict=f'sim_err:{type(e).__name__}')
            return True

        if out_ours == spec and out_theirs == spec:
            verdict = 'both_correct'
        elif out_ours == spec:
            verdict = 'ours_correct'        # ptxas is wrong
        elif out_theirs == spec:
            verdict = 'theirs_correct'      # we are wrong
        else:
            verdict = 'both_wrong'

        db.set_gate(self.conn, pid, 'spec_checked',
                    spec_verdict=verdict, spec_expected=spec)

        # Propagate to class if already classified
        if row['class_sig'] is not None:
            # Translate per-program verdict to class-level label
            class_verdict = {
                'ours_correct': 'theirs_wrong',   # ptxas bug
                'theirs_correct': 'ours_wrong',   # our bug
                'both_wrong': 'both_wrong',
                'both_correct': 'both_correct',
            }.get(verdict)
            if class_verdict:
                db.set_class_verdict(self.conn, row['class_sig'], class_verdict)

        return True


if __name__ == '__main__':
    cli_main(OracleDaemon)
