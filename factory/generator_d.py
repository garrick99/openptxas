"""Generator daemon: emit PTX programs into the factory queue.

Sources:
  - random: seeded random kernels via fuzzer.generator.generate
  - random-bitmanip: fuzzer.generator_bitmanip
  - random-warp:     fuzzer.generator_warp
  - truth_bfe_s32:   systematic (start, len, input) sweep for bfe.s32
  - seed:            user-supplied PTX (for smoke tests)
"""
import sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from factory import db
from factory.daemon import Daemon, cli_main
from fuzzer.generator import generate as gen_alu
from fuzzer.generator_bitmanip import generate as gen_bit
from fuzzer.generator_warp import generate as gen_warp
from factory.generator_danger import generate as gen_danger


class GeneratorDaemon(Daemon):
    NAME = 'generator'
    IDLE_SLEEP_SEC = 0.5   # sleep a bit when backpressured

    # Backpressure: pause when the differ queue is this deep, resume
    # when it drops below the low-water mark.  Without this, the
    # generator (~3k/sec) massively outpaces the differ (~50/sec) and
    # balloons the pending-row count — which is the same failure mode
    # as the first WAL blowup, just slower.
    PENDING_HIGH = 2000
    PENDING_LOW  = 500

    # Cycle through generators to get coverage.  Danger appears THREE
    # times in the rotation so ~60% of emitted programs are bug-adjacent
    # while still getting regular coverage of the uniform-random space
    # (which occasionally surfaces fresh bug families outside our
    # existing templates).
    _GENS = [
        ('random_danger',   gen_danger, 0),
        ('random_danger',   gen_danger, 0),
        ('random_danger',   gen_danger, 0),
        ('random_alu',      gen_alu,    14),
        ('random_bitmanip', gen_bit,    14),
        ('random_warp',     gen_warp,   14),
    ]

    def __init__(self, seed_base: int = 1_000_000, **kw):
        super().__init__(**kw)
        # Seed high-water mark lives in the kv table (survives the
        # differ-side delete of 'ok' program rows).
        self._next_seed = db.kv_get(self.conn, 'next_random_seed', seed_base)
        self._which = 0
        self._last_checkpoint = time.time()

    def tick(self) -> bool:
        # Periodic WAL checkpoint runs BEFORE the backpressure check so
        # it still fires during backpressured idle — otherwise the WAL
        # grows unbounded while the generator is paused.
        if time.time() - self._last_checkpoint > 60:
            try: db.checkpoint(self.conn)
            except Exception: pass
            self._last_checkpoint = time.time()

        # Backpressure check — if the differ is behind, stop filling queue.
        pending = self.conn.execute(
            'SELECT COUNT(*) FROM programs WHERE differ_done IS NULL'
        ).fetchone()[0]
        if pending > self.PENDING_HIGH:
            self._backpressured = True
        elif pending < self.PENDING_LOW:
            self._backpressured = False
        if getattr(self, '_backpressured', False):
            return False  # idle-sleep

        name, fn, n = self._GENS[self._which]
        self._which = (self._which + 1) % len(self._GENS)
        seed = self._next_seed
        self._next_seed += 1
        try:
            ptx, _ = fn(seed, n)
        except Exception as e:
            # Generators sometimes throw on weird seed-dependent paths; skip.
            print(f'[generator] gen {name} seed={seed} err: {e}', file=sys.stderr)
            return True
        family = name.split('_', 1)[1]
        pid = db.insert_program(self.conn, ptx,
                                 seed=seed, family=family, source='random')
        # Advance the persistent seed high-water mark unconditionally
        # (whether or not the INSERT deduplicated on ptx_sha).
        db.kv_set(self.conn, 'next_random_seed', self._next_seed)
        if pid is not None:
            db.counter_add(self.conn, 'programs_generated', 1)
        return pid is not None


def seed_program(ptx: str, family: str = 'manual', source: str = 'seed') -> int:
    """Insert a specific PTX as a seed.  For smoke tests."""
    conn = db.connect()
    pid = db.insert_program(conn, ptx, family=family, source=source)
    conn.close()
    return pid


if __name__ == '__main__':
    cli_main(GeneratorDaemon)
