"""Delta-debug one fuzz artifact down to a minimal diverging PTX.

Greedy 1-line reduction: for each body instruction, try dropping it; if
both compilers still accept + both cubins still run + outputs still
differ, keep the drop.  Repeat until fixed-point.

Usage:
    python -m fuzzer.minimize <artifact-name-or-path> [<more>...]
    python -m fuzzer.minimize --top 3          # minimize 3 shortest
    python -m fuzzer.minimize --all            # minimize every divergence

Writes 'minimal.ptx' into each artifact dir.  Prints the minimal PTX.

Does NOT attempt: coalescing multiple simultaneous removals (classical
ddmin-2), immediate simplification, register renaming.  Those are cheap
follow-ups once the 1-line pass exposes the bug family.
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fuzzer.oracle import (
    compile_ours, compile_theirs, CudaRunner, N_THREADS, ARTIFACT_DIR,
)


_REG_RE = re.compile(r'%r\d+|%rd\d+')
_PRED_RE = re.compile(r'%p\d+')

# Registers defined by the fuzz kernel's prologue (before the body).
_PROLOGUE_DEFINED = frozenset(
    ['%r0', '%r1', '%r3', '%rd0', '%rd1', '%rd2'])

# Predicates defined by the prologue (only %p0 for the bounds guard).
_PROLOGUE_PRED = frozenset(['%p0'])


def _store_src_reg(ptx: str) -> str | None:
    for ln in ptx.splitlines():
        s = ln.strip()
        if s.startswith('st.global.u32'):
            m = re.search(r',\s*(%r\d+)\s*;', s)
            if m: return m.group(1)
    return None


def _parse_instr(line: str):
    """Return (dst, srcs, predicated) for a writing PTX instruction, or
    None for lines that don't define a general register (st/setp/ret/..)."""
    s = line.strip()
    if not s: return None
    predicated = False
    if s.startswith('@'):
        parts = s.split(None, 1)
        if len(parts) < 2: return None
        s = parts[1]
        predicated = True
    toks = s.split(None, 1)
    if len(toks) < 2: return None
    op = toks[0]
    # Skip ops that don't write a general register.
    if (op.startswith('st.') or op.startswith('setp.')
            or op in ('ret', 'bra', 'bar.sync')):
        return None
    regs = _REG_RE.findall(toks[1])
    if not regs: return None
    return regs[0], regs[1:], predicated


def _reaching_defs(body_lines: list[str]) -> set[str]:
    """Forward-propagate: which regs are guaranteed-defined after the body?
    Rules:
      - unpredicated op: dst ∈ defined iff all srcs ∈ defined
      - predicated op: may or may not execute at runtime, so dst can only
        *remain* defined (not newly-become defined)
      - a write whose srcs are undefined makes the dst undefined"""
    defined = set(_PROLOGUE_DEFINED)
    for ln in body_lines:
        parsed = _parse_instr(ln)
        if parsed is None: continue
        dst, srcs, predicated = parsed
        srcs_ok = all(r in defined for r in srcs)
        if predicated:
            if not srcs_ok:
                defined.discard(dst)
            # else: dst keeps its prior state (already in or not in defined)
        else:
            if srcs_ok:
                defined.add(dst)
            else:
                defined.discard(dst)
    return defined


def _is_well_formed(ptx: str) -> bool:
    """Reject kernels with ANY use-before-def of a register or predicate.

    The store source has to be reaching-def (old check), but we also
    require that every individual instruction's source registers AND
    predicate guard are already defined when it executes.  Reads of
    undefined registers — even in dead code — produce
    implementation-defined SASS (the register allocator's choice of
    physical reg bleeds into runtime state and can interact with the
    scoreboard), which shows up as spurious divergence between OURS
    and ptxas.  Those aren't bugs in either compiler, just reads of
    garbage from whatever happened to be in the coloring slot.
    """
    parts = _split_body(ptx)
    if parts is None:
        return True
    _, body, _ = parts
    defined = set(_PROLOGUE_DEFINED)
    pred_defined = set(_PROLOGUE_PRED)
    for ln in body:
        s = ln.strip()
        if not s:
            continue
        # Predicate guard check: any `@%pN` / `@!%pN` prefix requires
        # %pN to be defined already.
        if s.startswith('@'):
            tok = s.split(None, 1)[0]  # '@%p1' or '@!%p1'
            pn = tok.lstrip('@').lstrip('!')
            if pn not in pred_defined:
                return False
            parts2 = s.split(None, 1)
            s = parts2[1] if len(parts2) > 1 else ''
        toks = s.split(None, 1)
        if len(toks) < 2:
            continue
        op = toks[0]
        rhs = toks[1]
        parsed = _parse_instr(ln)
        if parsed is None:
            # st.* / setp.* / other non-gpr-dest ops.  All %r/%rd
            # tokens must be defined; setp writes a predicate.
            if op.startswith('st.') or op.startswith('setp.'):
                for r in _REG_RE.findall(rhs):
                    if r not in defined:
                        return False
                if op.startswith('setp.'):
                    # setp dest is the first %pN in rhs.
                    m = _PRED_RE.search(rhs)
                    if m:
                        pred_defined.add(m.group(0))
            continue
        dst, srcs, predicated = parsed
        srcs_ok = all(r in defined for r in srcs)
        # Also check predicate sources (e.g. selp reads %pN).
        for pn in _PRED_RE.findall(rhs):
            if pn not in pred_defined:
                return False
        if not srcs_ok:
            return False
        if predicated:
            # Predicated write keeps dst's prior state.
            pass
        else:
            defined.add(dst)
    # Final check — the store's source register must be defined.
    reg = _store_src_reg(ptx)
    if reg is None:
        return True
    return reg in defined


def _still_diverges(ptx: str, runner: CudaRunner, input_bytes: bytes) -> bool:
    """True iff both compilers accept, both cubins run, outputs differ,
    AND the store source register is actually written somewhere."""
    if not _is_well_formed(ptx):
        return False
    ours, e_o = compile_ours(ptx)
    if e_o: return False
    theirs, e_t = compile_theirs(ptx)
    if e_t: return False
    out_o, se_o = runner.run_cubin(ours, input_bytes, N_THREADS)
    if se_o is not None:
        runner.reset()
        return False
    out_t, se_t = runner.run_cubin(theirs, input_bytes, N_THREADS)
    if se_t is not None:
        runner.reset()
        return False
    return out_o != out_t


def _split_body(ptx: str):
    """Return (prologue_lines, body_lines, epilogue_lines) or None if the
    kernel doesn't match the expected fuzz shape."""
    lines = ptx.splitlines()
    bstart = bend = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if bstart is None and s.startswith('ld.global.u32 %r3'):
            bstart = i + 1
        elif bstart is not None and s.startswith('ld.param.u64 %rd3'):
            bend = i
            break
    if bstart is None or bend is None:
        return None
    return lines[:bstart], lines[bstart:bend], lines[bend:]


def minimize(ptx: str, input_bytes: bytes, runner: CudaRunner,
              verbose: bool = True) -> tuple[str, str]:
    """Return (minimized_ptx, outcome).  outcome in {'fixed_point', 'ctx_dead',
    'not_divergent'}."""
    parts = _split_body(ptx)
    if parts is None:
        if verbose: print('[minimize] cannot locate body; returning as-is')
        return ptx, 'fixed_point'
    prologue, body, epilogue = parts

    try:
        original_diverges = _still_diverges(ptx, runner, input_bytes)
    except RuntimeError:
        return ptx, 'ctx_dead'
    if not original_diverges:
        if verbose: print('[minimize] ORIGINAL does not reproduce — skipping')
        return ptx, 'not_divergent'

    if verbose:
        print(f'[minimize] start: {len(body)} body lines')
    changed = True
    while changed:
        changed = False
        for i in range(len(body)):
            cand_body = body[:i] + body[i+1:]
            cand = '\n'.join(prologue + cand_body + epilogue)
            try:
                keep = _still_diverges(cand, runner, input_bytes)
            except RuntimeError:
                if verbose:
                    print(f'  ! ctx unrecoverable — stopping here ({len(body)} lines)')
                return '\n'.join(prologue + body + epilogue), 'ctx_dead'
            if keep:
                if verbose:
                    print(f'  - dropped:  {body[i].strip()}   ({len(body)-1} left)')
                body = cand_body
                changed = True
                break

    if verbose:
        print(f'[minimize] final: {len(body)} body lines')
    return '\n'.join(prologue + body + epilogue), 'fixed_point'


def _find_divergences(limit: int | None = None, well_formed_only: bool = True):
    """Return list of divergence artifact dirs, shortest-body first.
    By default, only include artifacts whose store reads a guaranteed-
    defined register (avoids wasting cycles on uninitialized-read noise)."""
    out = []
    skipped_ill = 0
    for d in sorted(ARTIFACT_DIR.glob('divergence_*')):
        f = d / 'input.ptx'
        if not f.exists() or not (d / 'input.bin').exists():
            continue
        ptx = f.read_text()
        parts = _split_body(ptx)
        if parts is None: continue
        if well_formed_only and not _is_well_formed(ptx):
            skipped_ill += 1
            continue
        out.append((len(parts[1]), d))
    if skipped_ill:
        print(f'[minimize] skipped {skipped_ill} ill-formed artifacts '
              f'(store reads undefined register)')
    out.sort(key=lambda x: x[0])
    dirs = [d for _, d in out]
    return dirs if limit is None else dirs[:limit]


def _resolve(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    p = ARTIFACT_DIR / arg
    if p.exists():
        return p
    sys.exit(f'[minimize] not found: {arg}')


def _run_one(artifact_path: str):
    """Minimize a single artifact in this process.  Called via subprocess
    from the multi-target driver so a dead CUDA context only kills one
    artifact's session, not the whole batch."""
    d = Path(artifact_path)
    ptx = (d / 'input.ptx').read_text()
    input_bytes = (d / 'input.bin').read_bytes()
    runner = CudaRunner()
    try:
        minimized, outcome = minimize(ptx, input_bytes, runner)
    finally:
        try: runner.close()
        except: pass
    (d / 'minimal.ptx').write_text(minimized)
    orig_body = len(_split_body(ptx)[1])
    min_body = len(_split_body(minimized)[1])
    print(f'[{d.name}] {orig_body} -> {min_body} body lines  ({outcome})')
    print('---')
    print(minimized)


def main():
    import subprocess
    argv = sys.argv[1:]
    if not argv:
        sys.exit('usage: python -m fuzzer.minimize <artifact>... | --top N | --all | --one <path>')

    # Internal single-artifact entry point used by subprocess dispatch.
    if argv[0] == '--one':
        return _run_one(argv[1])

    if argv[0] == '--all':
        targets = _find_divergences()
    elif argv[0] == '--top':
        n = int(argv[1])
        targets = _find_divergences(limit=n)
    else:
        targets = [_resolve(a) for a in argv]

    if not targets:
        print('[minimize] no targets found')
        return

    # Spawn one subprocess per artifact to isolate CUDA context death.
    for d in targets:
        print(f'\n=== {d.name} ===')
        r = subprocess.run(
            [sys.executable, '-m', 'fuzzer.minimize', '--one', str(d)],
            capture_output=False)
        if r.returncode != 0:
            print(f'[{d.name}] subprocess exit {r.returncode}')


if __name__ == '__main__':
    main()
