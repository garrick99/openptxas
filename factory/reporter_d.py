"""Report generator daemon.

For each escalated class with spec_verdict='theirs_wrong', emit a
PSIRT-format REPORT.md and a standalone validate.py in a new directory
under _nvidia_bugs_auto/.  Mark the class reported.

The dossier follows the tight 5-section format we've been using:
Title / Minimal repro / Expected vs Actual / Proof / Environment.
"""
import struct, subprocess, sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from factory import db
from factory.daemon import Daemon, cli_main

_VALIDATE_TEMPLATE = '''"""Standalone cross-arch validator (auto-generated).

Uses only ptxas + libcuda.so.  Compiles the minimal PTX, launches on
the GPU with a deterministic 128-byte input, compares the ptxas output
against the spec-expected output baked into this file.
"""
import sys, ctypes, struct, subprocess, os, tempfile

TARGET = sys.argv[1] if len(sys.argv) > 1 else 'sm_89'

# Try common ptxas locations; fall back to whatever's on PATH.
_PTXAS_CANDIDATES = [
    '/usr/local/cuda-13.2/bin/ptxas',
    '/usr/local/cuda/bin/ptxas',
    r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2\\bin\\ptxas.exe',
    'ptxas',
]
PTXAS_BIN = None
for _p in _PTXAS_CANDIDATES:
    if _p == 'ptxas' or os.path.exists(_p):
        PTXAS_BIN = _p; break
if PTXAS_BIN is None:
    print('ERROR: ptxas not found'); sys.exit(1)

PTX = """__PTX__"""

INPUTS = bytes([__INPUTS__])
EXPECTED = bytes([__EXPECTED__])

for libname in ('libcuda.so.1', 'libcuda.so', 'nvcuda'):
    try:
        cuda = ctypes.CDLL(libname); break
    except OSError: continue
else:
    print('ERROR: libcuda not found'); sys.exit(1)

def chk(rc, where):
    if rc != 0:
        msg = ctypes.c_char_p()
        cuda.cuGetErrorString(rc, ctypes.byref(msg))
        raise RuntimeError(f'{where}: rc={rc} {msg.value!r}')

cuda.cuInit(0)
dev = ctypes.c_int(); chk(cuda.cuDeviceGet(ctypes.byref(dev), 0), 'cuDeviceGet')
ctx = ctypes.c_void_p(); chk(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), 'cuCtxCreate')

with tempfile.TemporaryDirectory() as tmp:
    ptx_p = os.path.join(tmp, 'k.ptx'); cubin_p = os.path.join(tmp, 'k.cubin')
    # Patch target.  Only downgrade PTX version for older arches that
    # CUDA 12.x understands (<= sm_90).  sm_120 needs PTX >= 8.6.
    ptx_patched = PTX.replace('.target sm_120', f'.target {TARGET}') \\
                     .replace('.target sm_89',  f'.target {TARGET}')
    _sm = int(TARGET.replace('sm_', '')) if TARGET.startswith('sm_') else 120
    if _sm < 100:
        ptx_patched = ptx_patched.replace('.version 9.0', '.version 7.8')
    open(ptx_p, 'w').write(ptx_patched)
    r = subprocess.run([PTXAS_BIN, f'-arch={TARGET}', ptx_p, '-o', cubin_p],
                        capture_output=True, text=True)
    if r.returncode != 0:
        print(f'ptxas failed: {r.stderr[:200]}'); sys.exit(1)
    cubin = open(cubin_p, 'rb').read()

mod = ctypes.c_void_p(); chk(cuda.cuModuleLoadData(ctypes.byref(mod), cubin), 'load')
fn = ctypes.c_void_p(); chk(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b'fuzz'), 'getfn')
d_in = ctypes.c_uint64(); chk(cuda.cuMemAlloc_v2(ctypes.byref(d_in), 128), 'alloc in')
d_out = ctypes.c_uint64(); chk(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 128), 'alloc out')
chk(cuda.cuMemcpyHtoD_v2(d_in, INPUTS, 128), 'HtoD')
chk(cuda.cuMemsetD8_v2(d_out, 0xAB, 128), 'memset')
ai = ctypes.c_uint64(d_in.value); ao = ctypes.c_uint64(d_out.value); an = ctypes.c_uint32(32)
argv = (ctypes.c_void_p*3)(ctypes.cast(ctypes.byref(ai), ctypes.c_void_p),
                            ctypes.cast(ctypes.byref(ao), ctypes.c_void_p),
                            ctypes.cast(ctypes.byref(an), ctypes.c_void_p))
chk(cuda.cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, None, argv, None), 'launch')
chk(cuda.cuCtxSynchronize(), 'sync')
buf = ctypes.create_string_buffer(128); chk(cuda.cuMemcpyDtoH_v2(buf, d_out, 128), 'DtoH')
out = bytes(buf.raw)

wrong = 0
print(f'ptxas target: {TARGET}')
print(f'lane | input      | expected   | ptxas      | bug?')
print(f'-----|------------|------------|------------|-----')
for i in range(32):
    inp_w = struct.unpack_from('<I', INPUTS, i*4)[0]
    exp_w = struct.unpack_from('<I', EXPECTED, i*4)[0]
    got_w = struct.unpack_from('<I', out, i*4)[0]
    if got_w != exp_w:
        wrong += 1
        if wrong <= 5 or i == 31:
            print(f' {i:3d} | {inp_w:#010x} | {exp_w:#010x} | {got_w:#010x} | YES')
print()
print(f'ptxas wrong on {wrong}/32 lanes')
sys.exit(1 if wrong > 0 else 0)
'''


def _hex_bytes(b: bytes) -> str:
    return ','.join(str(x) for x in b)


def _render_ptxas_report(sig: str, program_row, diff_row, expected: bytes) -> str:
    """PSIRT-format dossier for a ptxas bug (theirs_wrong)."""
    ptx = program_row['ptx']
    out_ours = diff_row['out_ours']
    out_theirs = diff_row['out_theirs']
    inputs = diff_row['inputs_blob']

    # Find first divergent lane
    first = None
    for i in range(32):
        if out_theirs[i*4:(i+1)*4] != expected[i*4:(i+1)*4]:
            first = i; break
    if first is None: first = 0

    in_w = struct.unpack_from('<I', inputs, first*4)[0]
    exp_w = struct.unpack_from('<I', expected, first*4)[0]
    got_w = struct.unpack_from('<I', out_theirs, first*4)[0]

    # Extract body line count for title
    body_lines = sum(1 for ln in ptx.splitlines() if ln.strip().startswith(
        ('add.', 'sub.', 'mul.', 'shl.', 'shr.', 'and.', 'or.',
         'xor.', 'min.', 'max.', 'bfe.', 'bfi.', 'cvt.', 'selp.',
         'mov.u32 %r', 'setp.')) and not ln.strip().startswith('setp.ge.u32 %p0'))

    return f"""# ptxas miscompile — class `{sig}`

## Minimal repro

```ptx
{ptx.strip()}
```

## Expected vs Actual

| input `%r3` | expected output | ptxas output | mismatch |
|-------------|----------------:|-------------:|:--------:|
| {in_w:#010x}  | {exp_w:#010x}      | {got_w:#010x}   | YES      |

ptxas disagrees with the PTX-spec-computed output on this specimen.
OpenPTXas (an independent PTX->SASS compiler) produces the
spec-correct output.

## Proof

Validator: [`validate.py`](validate.py) — standalone, uses only
`ptxas` and `libcuda.so`, no third-party Python deps.

```
python3 validate.py sm_89       # default target
python3 validate.py sm_120      # or any other ptxas -arch target
```

The validator bakes in the minimal PTX, the 32-lane input buffer, and
the spec-expected output.  It compiles with ptxas, launches on the
GPU, and reports any lane where ptxas disagrees with spec.

## Environment

- Discovered via differential testing of PTX against an independent
  PTX->SASS compiler on NVIDIA RTX 5090 (sm_120), driver 595.79,
  CUDA 13.2.
- Body line count: {body_lines}
- Class signature: `{sig}`
- Auto-generated by factory at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
"""


def _render_openptxas_report(sig: str, program_row, diff_row, expected: bytes) -> str:
    """Internal backlog entry for an OpenPTXas bug (ours_wrong)."""
    ptx = program_row['ptx']
    out_ours = diff_row['out_ours']
    out_theirs = diff_row['out_theirs']
    inputs = diff_row['inputs_blob']

    # First divergent lane
    first = None
    for i in range(32):
        if out_ours[i*4:(i+1)*4] != expected[i*4:(i+1)*4]:
            first = i; break
    if first is None: first = 0
    in_w = struct.unpack_from('<I', inputs, first*4)[0]
    exp_w = struct.unpack_from('<I', expected, first*4)[0]
    ours_w = struct.unpack_from('<I', out_ours, first*4)[0]
    theirs_w = struct.unpack_from('<I', out_theirs, first*4)[0]

    return f"""# OpenPTXas bug — class `{sig}`

**This is a bug in OpenPTXas.**  ptxas produces the spec-correct result;
OpenPTXas does not.  Filed automatically by `factory/reporter_d.py`.

## Minimal repro

```ptx
{ptx.strip()}
```

## Outputs on lane {first}

| source    | value       | spec-correct |
|-----------|------------:|:------------:|
| spec      | {exp_w:#010x}   | (reference)  |
| ptxas     | {theirs_w:#010x} | {'YES' if theirs_w == exp_w else 'NO'}          |
| OpenPTXas | {ours_w:#010x}  | {'YES' if ours_w == exp_w else 'NO (bug)'}     |

Input `%r3` on that lane: `{in_w:#010x}`.  See `validate.py` for a full
32-lane reproducer with the exact input buffer baked in.

## Fixing this

1. Read the PTX above and the SASS OpenPTXas currently emits:
   ```
   python -c "from ptx.parser import parse; from sass.pipeline import compile_function; \\
              c = compile_function(parse(open('{program_row['id']}.ptx').read()).functions[0], verbose=True, sm_version=120)"
   ```
2. Compare against ptxas output for the same PTX.
3. Fix the lowering / encoding in `sass/isel.py` or `sass/encoding/sm_120_opcodes.py`.
4. Re-run this class's canonical program; verdict should flip to `both_correct`.

## Class signature

`{sig}` (n={program_row['id'] and 'multi' or '1'} specimens at the time of reporting)

Auto-generated at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.
"""


def _render_both_wrong_report(sig: str, program_row, diff_row, expected: bytes) -> str:
    """Both compilers disagree with spec — most suspicious class: either a
    real double-bug or (more commonly) a spec/simulator gap.  Dossier
    documents the three-way disagreement so a human can classify."""
    ptx = program_row['ptx']
    out_ours = diff_row['out_ours']
    out_theirs = diff_row['out_theirs']
    inputs = diff_row['inputs_blob']

    first = None
    for i in range(32):
        if (out_ours[i*4:(i+1)*4] != expected[i*4:(i+1)*4]
                or out_theirs[i*4:(i+1)*4] != expected[i*4:(i+1)*4]):
            first = i; break
    if first is None: first = 0

    in_w = struct.unpack_from('<I', inputs, first*4)[0]
    exp_w = struct.unpack_from('<I', expected, first*4)[0]
    ours_w = struct.unpack_from('<I', out_ours, first*4)[0]
    theirs_w = struct.unpack_from('<I', out_theirs, first*4)[0]
    compilers_agree = (ours_w == theirs_w)

    return f"""# both_wrong — class `{sig}`

Both ptxas and OpenPTXas disagree with the spec simulator.  Most
likely explanations, in order:

1. **Spec simulator gap** (`factory/spec.py` missing an opcode or
   edge-case).  Extend `spec.py` and re-oracle.
2. **Shared miscompile** (both compilers have the same bug — rare but
   possible; flag for deeper investigation).
3. **PTX undefined behavior** (the program exercises UB; spec should
   be updated to also return whatever the compilers compute, or the
   generator should avoid the UB pattern).

## Minimal repro

```ptx
{ptx.strip()}
```

## Outputs on lane {first}

| source    | value       | equals spec? | agrees with other compiler? |
|-----------|------------:|:------------:|:---------------------------:|
| spec      | {exp_w:#010x}   | (reference)  | —                           |
| ptxas     | {theirs_w:#010x} | {'YES' if theirs_w == exp_w else 'NO'}          | {'YES' if compilers_agree else 'NO'}                       |
| OpenPTXas | {ours_w:#010x}  | {'YES' if ours_w == exp_w else 'NO'}          | {'YES' if compilers_agree else 'NO'}                       |

Input `%r3` on that lane: `{in_w:#010x}`.

**Compilers agree with each other: {'YES' if compilers_agree else 'NO'}.**
{'If both compilers produce the same wrong value, likelihood favors spec gap or UB over a real double-miscompile.' if compilers_agree else 'If compilers disagree AND both differ from spec, this is either a spec gap plus a real bug, or two different bugs.'}

## Triage checklist

- [ ] Grep the PTX body for opcodes — is any of them missing from
      `factory/spec.py`?  (Symptom in the oracle: verdict would have
      been `unsupported:<opcode>` — if you see this dossier, the spec
      believes it implemented them all.)
- [ ] Re-run the spec simulator by hand on the input and compare.
- [ ] If spec is correct and compilers agree, test on an older arch
      (sm_89) to see if both NVIDIA generations miscompile — suggests
      a fundamental ptxas issue.
- [ ] If spec is correct and compilers disagree, split into an
      OpenPTXas dossier + a ptxas dossier.

## Environment

- Class signature: `{sig}`
- Canonical program id: {program_row['id']}
- Auto-generated at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
"""


class ReporterDaemon(Daemon):
    NAME = 'reporter'
    IDLE_SLEEP_SEC = 10.0

    def tick(self) -> bool:
        cls = db.claim_unreported_class(self.conn)
        if cls is None:
            return False

        sig = cls['sig']
        canonical_id = cls['canonical_id']

        # Fetch canonical program + its difference + spec expected
        row = self.conn.execute(
            "SELECT p.*, d.inputs_blob, d.out_ours, d.out_theirs "
            "FROM programs p JOIN differences d ON d.program_id = p.id "
            "WHERE p.id = ?", (canonical_id,)).fetchone()
        if row is None:
            # Can't report; mark reported to skip.
            self.conn.execute(
                'UPDATE classes SET reported=1 WHERE sig=?', (sig,))
            return True

        expected = row['spec_expected']
        if expected is None:
            # Not yet oracle'd; skip this round.
            return False

        safe_sig = sig.replace(',', '_').replace('*', 'x').replace(' ', '')
        ts = time.strftime('%Y%m%d_%H%M%S')
        verdict = cls['spec_verdict']
        if verdict == 'theirs_wrong':
            root = db.REPORT_DIR
            report = _render_ptxas_report(sig, row, row, expected)
            track_label = 'PTXAS BUG'
        elif verdict == 'ours_wrong':
            root = db.OPENPTXAS_REPORT_DIR
            report = _render_openptxas_report(sig, row, row, expected)
            track_label = 'OpenPTXas BUG'
        elif verdict == 'both_wrong':
            root = db.BOTH_WRONG_REPORT_DIR
            report = _render_both_wrong_report(sig, row, row, expected)
            track_label = 'BOTH_WRONG (spec-gap suspect)'
        else:
            # Shouldn't happen given the SELECT predicate, but be safe
            self.conn.execute(
                'UPDATE classes SET reported=1 WHERE sig=?', (sig,))
            return True

        outdir = root / f'{ts}_{safe_sig[:80]}'
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / 'REPORT.md').write_text(report, encoding='utf-8')

        validator = (_VALIDATE_TEMPLATE
                     .replace('__PTX__', row['ptx'].replace('"""', '\\"\\"\\"'))
                     .replace('__INPUTS__', _hex_bytes(row['inputs_blob']))
                     .replace('__EXPECTED__', _hex_bytes(expected)))
        (outdir / 'validate.py').write_text(validator, encoding='utf-8')

        # Record + mark reported
        self.conn.execute(
            'INSERT INTO reports(class_sig, program_id, path, created_at) '
            'VALUES (?, ?, ?, ?)',
            (sig, canonical_id, str(outdir), int(time.time())))
        self.conn.execute(
            'UPDATE classes SET reported=1 WHERE sig=?', (sig,))

        # Loud banner (survives stdout interleaving with other daemons)
        bar = '=' * 72
        print(f'\a\n{bar}\n[reporter] NEW {track_label} CANDIDATE\n'
              f'  class:    {sig}\n  specimens: {cls["specimen_count"]}\n'
              f'  dossier:   {outdir}\n{bar}\n', flush=True)

        # Best-effort Windows toast notification (no-op on non-Windows)
        if sys.platform.startswith('win'):
            try:
                msg = f'New {track_label} class: {sig[:60]}'
                subprocess.Popen([
                    'powershell', '-NoProfile', '-Command',
                    "Add-Type -AssemblyName System.Windows.Forms; "
                    "$n = New-Object System.Windows.Forms.NotifyIcon; "
                    "$n.Icon = [System.Drawing.SystemIcons]::Information; "
                    "$n.Visible = $true; "
                    f"$n.ShowBalloonTip(10000, 'ptxas-bug-factory', '{msg}', "
                    "[System.Windows.Forms.ToolTipIcon]::Info); "
                    "Start-Sleep -Seconds 10; $n.Dispose()"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

        return True


if __name__ == '__main__':
    cli_main(ReporterDaemon)
