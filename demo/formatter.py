"""Clean terminal output formatting."""
from __future__ import annotations

from demo.compare import opcode_name


def fmt_kernel_report(name: str, metrics: dict, gpu: dict,
                      diff_records: list[dict], highlights: list[str]) -> str:
    """Format a complete kernel comparison report."""
    ours = metrics['ours']
    ptxas = metrics['ptxas']
    lines = []

    lines.append('')
    lines.append(f'=== KERNEL: {name} ===')
    lines.append('')

    # -- Correctness --
    lines.append('[CORRECTNESS]')
    lines.append(f'  PTXAS:  {"PASS" if gpu["ptxas_pass"] else "FAIL"}')
    lines.append(f'  OURS:   {"PASS" if gpu["ours_pass"] else "FAIL"}')
    lines.append('')

    # -- Performance --
    lines.append('[PERFORMANCE]')
    lines.append(f'  {"Metric":<20s} {"PTXAS":>8s} {"OURS":>8s} {"Delta":>8s}')
    lines.append(f'  {"-" * 46}')

    def _row(label, pv, ov):
        d = ov - pv
        ds = f'{d:+d}' if d != 0 else '0'
        lines.append(f'  {label:<20s} {pv:>8d} {ov:>8d} {ds:>8s}')

    _row('Instructions', ptxas['non_nop'], ours['non_nop'])
    _row('Registers', ptxas['regs'], ours['regs'])
    _row('NOPs', ptxas['nops'], ours['nops'])
    _row('Total (incl NOPs)', ptxas['total'], ours['total'])
    lines.append('')

    # -- Highlights --
    if highlights:
        lines.append('[HIGHLIGHTS]')
        for h in highlights:
            lines.append(f'  - {h}')
        lines.append('')

    # -- Diff (compact) --
    ours_only = [r for r in diff_records if r['type'] == 'ours_only']
    ptxas_only = [r for r in diff_records if r['type'] == 'ptxas_only']
    if ours_only or ptxas_only:
        lines.append('[INSTRUCTION DIFF]')
        if ours_only:
            ops = [opcode_name(r['ours']['opcode']) for r in ours_only]
            lines.append(f'  OURS unique ({len(ops)}):  {" ".join(ops)}')
        if ptxas_only:
            ops = [opcode_name(r['ptxas']['opcode']) for r in ptxas_only]
            lines.append(f'  PTXAS unique ({len(ops)}): {" ".join(ops)}')
        # WHY section for explained differences
        explained = [r for r in diff_records
                     if r['type'] == 'ours_only' and r['explanation']]
        if explained:
            lines.append('')
            lines.append('  WHY:')
            seen = set()
            for r in explained:
                if r['explanation'] not in seen:
                    seen.add(r['explanation'])
                    opc = opcode_name(r['ours']['opcode'])
                    lines.append(f'    {opc}: {r["explanation"]}')
        lines.append('')

    # -- Verdict --
    instr_delta = ours['non_nop'] - ptxas['non_nop']
    reg_delta = ours['regs'] - ptxas['regs']
    lines.append('[VERDICT]')
    if instr_delta < 0:
        lines.append(f'  OURS WINS by {-instr_delta} instruction(s)')
    elif instr_delta == 0 and reg_delta <= 0:
        lines.append('  OURS MATCHES PTXAS')
    elif instr_delta == 0:
        lines.append(f'  PARITY on instructions (+{reg_delta} registers)')
    else:
        lines.append(f'  PTXAS leads by {instr_delta} instruction(s) (bounded residual)')
    lines.append('')

    return '\n'.join(lines)


def fmt_suite_summary(results: list[dict]) -> str:
    """Format a summary table across multiple kernels."""
    lines = []
    lines.append('')
    lines.append('=' * 70)
    lines.append('SUITE SUMMARY')
    lines.append('=' * 70)
    lines.append('')
    lines.append(f'  {"Kernel":<24s} {"Correct":>8s} {"Instrs":>8s} {"Regs":>8s} {"Verdict"}')
    lines.append(f'  {"-" * 66}')

    total_ours = 0
    total_ptxas = 0
    wins = 0; parity = 0; gaps = 0

    for r in results:
        name = r['name']
        m = r['metrics']
        gpu = r['gpu']
        oi = m['ours']['non_nop']
        pi = m['ptxas']['non_nop']
        total_ours += oi
        total_ptxas += pi
        di = oi - pi
        dr = m['ours']['regs'] - m['ptxas']['regs']
        correct = 'PASS' if (gpu['ours_pass'] and gpu['ptxas_pass']) else 'FAIL'

        if di < 0:
            verdict = 'OURS WINS'
            wins += 1
        elif di == 0:
            verdict = 'PARITY'
            parity += 1
        else:
            verdict = f'+{di} gap'
            gaps += 1

        di_s = f'{di:+d}' if di != 0 else '0'
        dr_s = f'{dr:+d}' if dr != 0 else '0'
        lines.append(f'  {name:<24s} {correct:>8s} {di_s:>8s} {dr_s:>8s}   {verdict}')

    lines.append(f'  {"-" * 66}')
    td = total_ours - total_ptxas
    lines.append(f'  {"TOTAL":<24s} {"":>8s} {td:>+8d} {"":>8s}   '
                 f'OURS {total_ours} vs PTXAS {total_ptxas}')
    lines.append('')
    lines.append(f'  Wins: {wins}  |  Parity: {parity}  |  Bounded gaps: {gaps}')
    lines.append('')

    return '\n'.join(lines)


def fmt_proof_status(adversarial: tuple[int, int],
                     corpus: tuple[int, int] | None = None) -> str:
    """Format proof model status line."""
    a_pass, a_total = adversarial
    parts = [f'Proof: {a_pass}/{a_total} adversarial CONFIRMED']
    if corpus:
        c_pass, c_total = corpus
        parts.append(f'{c_pass}/{c_total} corpus SAFE')
    return '  '.join(parts)
