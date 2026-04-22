"""Differ daemon.

Pops the oldest program with differ_done IS NULL, compiles with both
OpenPTXas and ptxas, launches both on the GPU with a deterministic
128-byte input, diffs lane-by-lane, writes differences row + updates
program stage gate.
"""
import struct, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from factory import db
from factory.daemon import Daemon, cli_main
from fuzzer.oracle import compile_ours, compile_theirs, CudaRunner
# JIT path exists in factory.jit but has a context-handoff bug on this
# driver+Python combo that I haven't root-caused.  Sticking with the
# subprocess path until then.
def _jit_set_context(_): pass

N_THREADS = 32


def _canonical_input() -> bytes:
    """128 bytes = 32 u32 values with a spread of sign bits and magnitudes."""
    # Mix of canonical bit patterns: 0, -1, pow2, neg, alternating, small, large.
    vals = [
        0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555,
        0x80000000, 0x7FFFFFFF, 0x00000001, 0xFFFFFFFE,
        0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x87654321,
        0x000000FF, 0xFF000000, 0x0F0F0F0F, 0xF0F0F0F0,
        0x00FF00FF, 0xFF00FF00, 0x33333333, 0xCCCCCCCC,
        0x00000100, 0x00000400, 0x40000000, 0x20000000,
        0x01234567, 0x89ABCDEF, 0xFEDCBA98, 0x76543210,
        0x3FFFFFFF, 0xC0000000, 0x20202020, 0xA5A5A5A5,
    ]
    return b''.join(struct.pack('<I', v) for v in vals)


class DifferDaemon(Daemon):
    NAME = 'differ'
    IDLE_SLEEP_SEC = 1.0

    def __init__(self, **kw):
        super().__init__(**kw)
        self.runner = CudaRunner()
        _jit_set_context(self.runner.ctx)
        self.inputs = _canonical_input()
        self._consec_sync_errs = 0
    # Once this many consecutive programs fail with sync_err (on either
    # side), assume the driver's per-process state is poisoned beyond
    # reset's ability to recover.  Exit cleanly; the supervisor will
    # spawn a fresh process (= fresh driver state).
    #
    # Chosen via a WSL2 smoke test: POISON_THRESHOLD=3 tripped within 0.7s
    # on legitimate gen_danger fault-prone programs (it's a 50/50 coin to
    # see a 3-in-a-row danger streak under round-robin).  POISON_THRESHOLD=20
    # (the original) let WDDM rot during the 2026-04-22 crash.  10 distinguishes
    # an unlucky run of fault-prone generated PTX from driver-wide poisoning,
    # and still catches real poisoning within a few seconds.
    POISON_THRESHOLD = 10

    def tick(self) -> bool:
        row = db.claim_next(self.conn, 'differ_done')
        if row is None:
            return False

        pid = row['id']
        ptx = row['ptx']

        # Compile both sides.
        cubin_ours, err_ours = compile_ours(ptx)
        cubin_theirs, err_theirs = compile_theirs(ptx)
        out_ours = out_theirs = None
        sync_o = sync_t = None

        # Early outs — compile errors short-circuit the launch.  Mark the
        # row, count the error, delete it (only divergences stay alive).
        def _compile_early_out(state: str, note: str) -> bool:
            db.set_gate(self.conn, pid, 'differ_done',
                        differ_state=state, differ_note=note[:200])
            self.conn.execute('DELETE FROM programs WHERE id=?', (pid,))
            db.counter_add(self.conn, 'programs_differ_err', 1)
            db.counter_add(self.conn, f'err_{state}', 1)
            return True

        if err_ours is not None and err_theirs is not None:
            return _compile_early_out('compile_err_both',
                f'ours={err_ours[:50]} theirs={err_theirs[:50]}')
        if err_ours is not None:
            return _compile_early_out('compile_err_ours', err_ours)
        if err_theirs is not None:
            return _compile_early_out('compile_err_theirs', err_theirs)

        # Launch both on GPU.  CUDA LAUNCH_FAILED (719) poisons the whole
        # process — cuCtxDestroy+Create in-process inherits the sticky
        # error (verified by _wsl_reset_probe.sh on 2026-04-22).  The only
        # recovery is to exit; supervisor respawns with a fresh process.
        #
        # Before exiting, we MUST mark the offending program as done —
        # otherwise the next respawn claims the same poison-trigger from
        # the top of the pending queue and we infinite-loop.
        def _poison_exit(side: str, reset_err):
            print(f'[differ] reset after {side} sync_err failed: {reset_err}; '
                  f'flagging pid={pid} and exiting for respawn', flush=True)
            try:
                db.set_gate(self.conn, pid, 'differ_done',
                            differ_state=f'sync_err_poison_{side}',
                            differ_note=(f'{sync_o or ""}|{sync_t or ""}|'
                                         f'reset_err={reset_err}')[:200])
                db.record_difference(
                    self.conn, pid, self.inputs,
                    out_ours, out_theirs, None,
                    str(sync_o) if sync_o else None,
                    str(sync_t) if sync_t else None)
                db.counter_add(self.conn, 'programs_differ_poison', 1)
                self.conn.commit()
            except Exception as db_err:
                print(f'[differ] failed to flag poison trigger pid={pid}: '
                      f'{db_err}', flush=True)
            db.heartbeat(self.conn, self.NAME, 'reset_failed', 0)
            sys.exit(2)

        out_ours, sync_o = self.runner.run_cubin(cubin_ours, self.inputs, N_THREADS)
        if sync_o:
            try:
                self.runner.reset()
                _jit_set_context(self.runner.ctx)
            except Exception as e:
                _poison_exit('ours', e)
        out_theirs, sync_t = self.runner.run_cubin(cubin_theirs, self.inputs, N_THREADS)
        if sync_t:
            try:
                self.runner.reset()
                _jit_set_context(self.runner.ctx)
            except Exception as e:
                _poison_exit('theirs', e)

        # Outcome
        if sync_o and sync_t:
            state = 'sync_err_both'
            note = f'{sync_o}|{sync_t}'[:200]
        elif sync_o:
            state = 'sync_err_ours'; note = str(sync_o)[:200]
        elif sync_t:
            state = 'sync_err_theirs'; note = str(sync_t)[:200]
        else:
            # Both ran — compare
            diff_lanes = sum(
                1 for i in range(N_THREADS)
                if out_ours[i*4:(i+1)*4] != out_theirs[i*4:(i+1)*4])
            if diff_lanes == 0:
                state = 'ok'; note = None
            else:
                state = 'divergence'; note = f'diff_lanes={diff_lanes}'
                db.record_difference(self.conn, pid, self.inputs,
                                      out_ours, out_theirs, diff_lanes,
                                      None, None)

        # Record ran-but-one-side-errored cases with the outputs we have
        if state.startswith('sync_err'):
            db.record_difference(self.conn, pid, self.inputs,
                                  out_ours, out_theirs, None,
                                  str(sync_o) if sync_o else None,
                                  str(sync_t) if sync_t else None)

        db.set_gate(self.conn, pid, 'differ_done',
                    differ_state=state, differ_note=note)

        # Pruning + counting FIRST (always runs, even if we're about to
        # exit on poison).  Previously the sys.exit below caused one
        # orphaned row per poison event; this ordering guarantees every
        # non-divergence row is either deleted or persisted deliberately.
        if state != 'divergence':
            self.conn.execute('DELETE FROM programs WHERE id=?', (pid,))
            if state == 'ok':
                db.counter_add(self.conn, 'programs_differ_ok', 1)
            else:
                db.counter_add(self.conn, 'programs_differ_err', 1)
                db.counter_add(self.conn, f'err_{state}', 1)

        # Track consecutive sync_err streaks and exit if poisoned.
        if state.startswith('sync_err'):
            self._consec_sync_errs += 1
            if self._consec_sync_errs >= self.POISON_THRESHOLD:
                print(f'[differ] {self._consec_sync_errs} consecutive sync_errs; '
                      f'context appears poisoned — exiting so supervisor spawns a '
                      f'fresh process', flush=True)
                sys.exit(2)  # supervisor respawns
        else:
            self._consec_sync_errs = 0

        return True


if __name__ == '__main__':
    cli_main(DifferDaemon)
