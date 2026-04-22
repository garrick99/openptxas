"""Base class for a poll-loop daemon.

Subclass and override tick() to do one unit of work.  Return a truthy
value if work was done (so we avoid sleeping) or falsy if idle.
"""
import signal, sys, time
from pathlib import Path
from typing import Optional

from factory import db


class Daemon:
    NAME: str = 'base'
    IDLE_SLEEP_SEC: float = 2.0
    BUSY_SLEEP_SEC: float = 0.0       # no sleep while work is flowing
    STOP_ON_ERROR: bool = False       # default: log and keep going

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or db.DB_PATH
        self.conn = db.connect(self.db_path)
        self._stopped = False
        self.items_processed = 0

    def tick(self) -> bool:
        """Do one unit of work.  Return True if work was done."""
        raise NotImplementedError

    def _install_signal(self):
        def _h(signum, frame):
            self._stopped = True
            print(f'[{self.NAME}] caught signal {signum}; stopping')
        try:
            signal.signal(signal.SIGINT, _h)
            signal.signal(signal.SIGTERM, _h)
        except (ValueError, AttributeError):
            pass  # not main thread, or platform doesn't support

    def run(self, max_iters: Optional[int] = None,
            max_seconds: Optional[float] = None):
        self._install_signal()
        started = time.time()
        db.heartbeat(self.conn, self.NAME, 'running', 0)
        iters = 0
        _last_checkpoint = started
        while not self._stopped:
            if max_iters is not None and iters >= max_iters:
                break
            if max_seconds is not None and time.time() - started >= max_seconds:
                break
            iters += 1
            try:
                did_work = self.tick()
            except Exception as e:
                msg = f'error: {type(e).__name__}: {e}'
                print(f'[{self.NAME}] {msg}', file=sys.stderr)
                db.heartbeat(self.conn, self.NAME, msg[:100], 0)
                if self.STOP_ON_ERROR:
                    raise
                time.sleep(self.IDLE_SLEEP_SEC)
                continue
            if did_work:
                self.items_processed += 1
                db.heartbeat(self.conn, self.NAME, 'running', 1)
                if self.BUSY_SLEEP_SEC:
                    time.sleep(self.BUSY_SLEEP_SEC)
            else:
                db.heartbeat(self.conn, self.NAME, 'idle', 0)
                time.sleep(self.IDLE_SLEEP_SEC)
        db.heartbeat(self.conn, self.NAME, 'stopped', 0)


def cli_main(daemon_cls):
    """Standard entry point: python -m factory.<daemon> [--iters N] [--seconds N]"""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=None)
    p.add_argument('--seconds', type=float, default=None)
    args = p.parse_args()
    d = daemon_cls()
    try:
        d.run(max_iters=args.iters, max_seconds=args.seconds)
    finally:
        print(f'[{d.NAME}] processed {d.items_processed} items')
