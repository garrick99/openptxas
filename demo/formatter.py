"""Clean terminal output formatting."""
from __future__ import annotations

from demo.compare import opcode_name


# Bounded gap explanations — why a specific kernel has a residual
_GAP_CATEGORY = {
    'ilp_unrolled_sum4': ('struct', 'chained stride addresses require allocator rework'),
    'ilp_pipeline_load': ('minor', 'scheduling style difference, diminishing returns'),
    'vecadd_large':      ('sched', 'address generation style difference'),
}


def fmt_kernel_report(name: str, metrics: dict, gpu: dict,
                      diff_records: list[dict], highlights: list[str],
                      explain: bool = False) -> str:
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

    # -- Highlights (shown when --explain or always if concise) --
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
        if explained and explain:
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
        cat, reason = _GAP_CATEGORY.get(name, ('', ''))
        tag = f' ({cat})' if cat else ''
        lines.append(f'  PTXAS leads by {instr_delta} instruction(s){tag}')
        if reason and explain:
            lines.append(f'  Reason: {reason}')
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
            cat, _ = _GAP_CATEGORY.get(name, ('', ''))
            tag = f' ({cat})' if cat else ''
            verdict = f'+{di} gap{tag}'
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


def fmt_proof_footer(adversarial: tuple[int, int],
                     corpus: tuple[int, int] | None = None) -> str:
    """Format proof footer for suite output."""
    a_pass, a_total = adversarial
    parts = [f'{a_pass}/{a_total} adversarial CONFIRMED']
    if corpus and corpus[0] > 0:
        c_pass, c_total = corpus
        parts.append(f'{c_pass}/{c_total} corpus SAFE')
    lines = []
    lines.append(f'  Proof: {" | ".join(parts)}')
    lines.append('')
    return '\n'.join(lines)


def fmt_structured_diff(diff_records: list[dict]) -> str:
    """Format transformation-grouped diff output."""
    lines = []
    lines.append('[STRUCTURED DIFF]')
    lines.append('')

    # Group contiguous ours_only / ptxas_only into transformation blocks
    transforms = []
    current_block = {'ours': [], 'ptxas': []}

    for r in diff_records:
        if r['type'] == 'match':
            # Flush current block if non-empty
            if current_block['ours'] or current_block['ptxas']:
                transforms.append(current_block)
                current_block = {'ours': [], 'ptxas': []}
        elif r['type'] == 'ours_only':
            current_block['ours'].append(r)
        elif r['type'] == 'ptxas_only':
            current_block['ptxas'].append(r)

    if current_block['ours'] or current_block['ptxas']:
        transforms.append(current_block)

    if not transforms:
        lines.append('  Instruction streams are identical.')
        lines.append('')
        return '\n'.join(lines)

    for idx, block in enumerate(transforms, 1):
        # Determine transformation name
        name = _classify_transform(block)
        lines.append(f'  [TRANSFORMATION {idx}: {name}]')

        if block['ptxas']:
            ops = [opcode_name(r['ptxas']['opcode']) for r in block['ptxas']]
            idxs = [f'[{r["ptxas"]["idx"]:d}]' for r in block['ptxas']]
            lines.append(f'  PTXAS ({len(ops)}): {" ".join(f"{i} {o}" for i, o in zip(idxs, ops))}')

        if block['ours']:
            ops = [opcode_name(r['ours']['opcode']) for r in block['ours']]
            idxs = [f'[{r["ours"]["idx"]:d}]' for r in block['ours']]
            lines.append(f'  OURS  ({len(ops)}): {" ".join(f"{i} {o}" for i, o in zip(idxs, ops))}')

        # Effect
        delta = len(block['ours']) - len(block['ptxas'])
        if delta < 0:
            lines.append(f'  Effect: {len(block["ptxas"])} -> {len(block["ours"])} (saved {-delta})')
        elif delta > 0:
            lines.append(f'  Effect: {len(block["ptxas"])} -> {len(block["ours"])} (+{delta})')
        else:
            lines.append(f'  Effect: {len(block["ptxas"])} -> {len(block["ours"])} (neutral)')

        # Explanation from any annotated record
        for r in block['ours']:
            if r.get('explanation'):
                lines.append(f'  Why: {r["explanation"]}')
                break

        lines.append('')

    return '\n'.join(lines)


def _classify_transform(block: dict) -> str:
    """Classify a diff block into a named transformation."""
    ours_ops = [opcode_name(r['ours']['opcode']) for r in block['ours']]
    ptxas_ops = [opcode_name(r['ptxas']['opcode']) for r in block['ptxas']]
    all_ops = ours_ops + ptxas_ops

    # Check for explained records first
    for r in block['ours']:
        if r.get('explanation'):
            if 'LOP3' in r['explanation']:
                return 'LOP3 inline immediate'
            if 'IMAD.WIDE' in r['explanation']:
                return 'IMAD.WIDE address fusion'
            if 'IADD.64' in r['explanation']:
                return 'IADD.64 R-UR address add'
            if 'predicated' in r['explanation']:
                return 'predicated IMAD fusion'

    # Pattern-based classification
    if any('LEA' in o for o in ptxas_ops):
        return 'address generation (LEA vs IMAD.WIDE+IADD.64)'
    if any('IMAD.424' in o for o in ptxas_ops):
        return 'IMAD mul+add form (PTXAS 0x424 vs OURS IMAD.Ri+IADD3)'
    if any('HFMA2' in o for o in ptxas_ops):
        return 'constant materialization (HFMA2 vs LDCU/S2R)'
    if any('SHF' in o for o in ptxas_ops):
        return 'stride address calc (SHF vs IADD3 chain)'
    if any('ISETP.RUR' in o for o in ptxas_ops) or any('ISETP' in o for o in ours_ops):
        return 'setup / bounds check style'
    if any('LDCU' in o for o in all_ops):
        return 'parameter loading'

    return 'instruction selection difference'
