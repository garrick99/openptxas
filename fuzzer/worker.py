"""One fuzzer worker process.  Runs N iterations or until a sync_err
corrupts the CUDA context, then exits cleanly.  The outer run_fuzz.py
respawns workers forever.

Communication with the outer process is a single JSON line per iteration
on stdout:
    {"seed": 1234, "status": "ok"}
    {"seed": 1235, "status": "divergence", "new": true}
    {"seed": 1236, "status": "sync_err_ours", "new": false}
    ...

On fatal state (ctx reset fails), worker exits with status 2.
"""
import json, random, struct, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fuzzer.oracle import CudaRunner, test_ptx, N_THREADS
from fuzzer.families import generate as family_generate, DEFAULT as DEFAULT_FAMILY


# Cornerstone u32 values.  Several known ptxas codegen bugs only fire on
# sign-boundary inputs (e.g., 0x80000000 with signed ops), so every input
# buffer includes these values sprinkled across threads.
_CORNERSTONES = [
    0x00000000, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF,
    0x00000001, 0xFFFFFFFE, 0xAAAAAAAA, 0x55555555,
    0xDEADBEEF, 0xCAFEBABE, 0x00000100, 0x0000FFFF,
]


def make_input(seed: int, n: int) -> bytes:
    """Build input buffer: cornerstone values plus random, shuffled deterministically."""
    rnd = random.Random(seed ^ 0xDEADBEEF)
    k = min(len(_CORNERSTONES), n)
    words = list(_CORNERSTONES[:k]) + [rnd.randrange(0, 1 << 32) for _ in range(n - k)]
    rnd.shuffle(words)
    return b''.join(struct.pack('<I', w) for w in words)


def main():
    # CLI: worker.py <seed_start> <max_iters> [family]
    seed_start = int(sys.argv[1])
    max_iters = int(sys.argv[2])
    family = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_FAMILY
    runner = CudaRunner()
    try:
        for i in range(max_iters):
            seed = seed_start + i
            inp = make_input(seed, N_THREADS)
            try:
                ptx, _ = family_generate(family, seed)
                status, new = test_ptx(ptx, seed, runner, inp, family=family)
            except RuntimeError as e:
                print(json.dumps({'seed': seed, 'status': 'ctx_dead',
                                   'err': str(e)[:200]}))
                sys.stdout.flush()
                sys.exit(2)
            print(json.dumps({'seed': seed, 'status': status, 'new': new,
                               'family': family}))
            sys.stdout.flush()
    finally:
        try: runner.close()
        except: pass


if __name__ == '__main__':
    main()
