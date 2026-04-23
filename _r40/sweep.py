"""R40: run each workbench kernel in isolation via subprocess, collect
pass/fail, and for failing kernels dump the post-EXIT SASS to check
whether any failure fits the post-EXIT S2R -> unknown-opcode hazard
class (opcodes outside R39's {0x824, 0x835, 0x812}).
"""
from __future__ import annotations
import subprocess, sys, struct

KERNELS = open('_r40_kernels.txt').read().split()


def _extract_post_exit_window(cubin: bytes, kname: str) -> str:
    """Return a short readable trace of the post-EXIT region (up to 4
    instructions after the first predicated EXIT), useful for seeing
    whether a failure kernel hits a new S2R -> ALU hazard."""
    try:
        e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
        e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
        e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]
        def sh(i): return struct.unpack_from('<IIQQQQIIQQ', cubin, e_shoff + i * 64)
        _, _, _, _, so, ss, *_ = sh(e_shstrndx)
        shs = cubin[so:so + ss]
        target = f'.text.{kname}'.encode()
        text = b''
        for i in range(e_shnum):
            nm, ty, _, _, off, sz, *_ = sh(i)
            end = shs.index(b'\x00', nm)
            if shs[nm:end] == target and ty == 1:
                text = cubin[off:off + sz]
                break
        if not text:
            # try any .text.*
            for i in range(e_shnum):
                nm, ty, _, _, off, sz, *_ = sh(i)
                end = shs.index(b'\x00', nm)
                if shs[nm:end].startswith(b'.text.') and ty == 1:
                    text = cubin[off:off + sz]
                    break
        seen_pexit = False
        post = []
        for a in range(0, len(text), 16):
            r = text[a:a + 16]
            opc = (r[0] | (r[1] << 8)) & 0xFFF
            guard = (r[1] >> 4) & 0xF
            if opc == 0x94d and guard != 0x7:
                seen_pexit = True
                continue
            if seen_pexit:
                post.append((opc, r[2], r[3], r[9]))
                if len(post) >= 6:
                    break
        return ' | '.join(f'op=0x{o:03x} dst=R{d} b3=R{b3} b9=0x{b9:02x}'
                          for o, d, b3, b9 in post)
    except Exception as e:
        return f'<dump-error: {e}>'


def run_one(kname):
    try:
        out = subprocess.run(
            [sys.executable, 'workbench.py', 'run', '--kernel', kname],
            cwd='C:/Users/kraken/openptxas',
            capture_output=True, text=True, timeout=45)
        stdout = out.stdout
        # Parse workbench output for build/correct
        build = 'PASS' if 'build:    PASS' in stdout else ('FAIL' if 'build:' in stdout else 'UNKNOWN')
        correct = 'PASS' if 'correct:  PASS' in stdout else ('FAIL' if 'correct:' in stdout else 'UNKNOWN')
        if 'kernel crashed' in stdout + out.stderr:
            correct = 'FAIL'
        if out.returncode != 0 and correct == 'UNKNOWN':
            correct = 'ERROR'
        return build, correct
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', 'TIMEOUT'
    except Exception as e:
        return 'ERROR', f'{type(e).__name__}'


def main():
    fails = []
    passes = []
    for k in KERNELS:
        build, correct = run_one(k)
        if correct == 'PASS':
            passes.append(k)
        else:
            fails.append((k, build, correct))
            print(f'[FAIL] {k:30s} build={build} correct={correct}')
    print()
    print(f'Total kernels: {len(KERNELS)}')
    print(f'Pass: {len(passes)}')
    print(f'Fail: {len(fails)}')
    if fails:
        print()
        print('Failure list:')
        for k, b, c in fails:
            print(f'  {k:30s} build={b} correct={c}')


if __name__ == '__main__':
    main()
