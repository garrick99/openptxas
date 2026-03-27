"""
Cumulative bisect: add our ctrl changes one-by-one from index 0 upward.
Find at which point the combination starts failing.
Also tests subsets to identify the minimal failing pair.
"""
import struct
import subprocess
import sys
import os

FRESH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'probe_work', 'ldg64_test_fresh.cubin'))
DEFAULT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'probe_work', 'ldg64_default_ctrl.cubin'))
WORK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', 'probe_work', 'bisect_work.cubin'))
RUNNER_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', 'probe_work', '_runner.py'))


def get_text_off(data):
    e_shoff = struct.unpack_from('<Q', data, 0x28)[0]
    e_shentsize = struct.unpack_from('<H', data, 0x3a)[0]
    e_shnum = struct.unpack_from('<H', data, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', data, 0x3e)[0]
    sh = e_shoff + e_shstrndx * e_shentsize
    sh_offset = struct.unpack_from('<Q', data, sh + 24)[0]
    sh_size = struct.unpack_from('<Q', data, sh + 32)[0]
    shstr = data[sh_offset:sh_offset + sh_size]
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        name_off = struct.unpack_from('<I', data, sh)[0]
        end = shstr.index(b'\x00', name_off)
        if shstr[name_off:end].decode() == '.text.ldg64_min':
            return (struct.unpack_from('<Q', data, sh + 24)[0],
                    struct.unpack_from('<Q', data, sh + 32)[0])
    raise RuntimeError('section not found')


def extract_ctrl(raw16):
    raw24 = ((raw16[15] & ~0x04) << 16) | (raw16[14] << 8) | raw16[13]
    return raw24 >> 1


def patch_ctrl(raw16, ctrl):
    buf = bytearray(raw16)
    raw24 = (ctrl & 0x7FFFFF) << 1
    buf[13] = raw24 & 0xFF
    buf[14] = (raw24 >> 8) & 0xFF
    buf[15] = ((raw24 >> 16) & 0xFF) | (buf[15] & 0x04)
    return bytes(buf)


def run_cubin(cubin_bytes):
    with open(WORK, 'wb') as f:
        f.write(cubin_bytes)
    result = subprocess.run(
        [sys.executable, RUNNER_PATH, WORK],
        capture_output=True, text=True, timeout=30
    )
    return (result.stdout + result.stderr).strip() or f'EXIT:{result.returncode}'


def apply_changes(default_data, default_text_off, fresh_data, fresh_text_off,
                  n_instr, change_set):
    """Apply a specific set of instruction indices (by ctrl change) to default cubin."""
    test = bytearray(default_data)
    for i in change_set:
        f_off = fresh_text_off + i * 16
        d_off = default_text_off + i * 16
        raw_fresh = bytes(fresh_data[f_off:f_off + 16])
        raw_def = bytes(test[d_off:d_off + 16])
        fc = extract_ctrl(raw_fresh)
        dc = extract_ctrl(raw_def)
        if fc != dc:
            test[d_off:d_off + 16] = patch_ctrl(raw_def, fc)
    return bytes(test)


def main():
    with open(FRESH, 'rb') as f:
        fresh_data = bytearray(f.read())
    with open(DEFAULT, 'rb') as f:
        default_data = bytearray(f.read())

    fresh_text_off, fresh_text_size = get_text_off(bytes(fresh_data))
    default_text_off, _ = get_text_off(bytes(default_data))
    n_instr = fresh_text_size // 16

    opnames = {0xb82: 'LDC', 0x919: 'S2R', 0x7ac: 'LDCU.64', 0xc35: 'IADD64-UR',
               0x981: 'LDG.E.64', 0x918: 'NOP', 0x986: 'STG.E.64',
               0x94d: 'EXIT', 0x947: 'BRA'}

    # Find which instructions actually have different ctrl
    changed = []
    for i in range(n_instr):
        f_off = fresh_text_off + i * 16
        d_off = default_text_off + i * 16
        fc = extract_ctrl(bytes(fresh_data[f_off:f_off + 16]))
        dc = extract_ctrl(bytes(default_data[d_off:d_off + 16]))
        if fc != dc:
            opc = struct.unpack_from('<Q', fresh_data, f_off)[0] & 0xFFF
            rbar = (fc >> 10) & 0x1f
            wdep = (fc >> 4) & 0x3f
            misc = fc & 0xf
            changed.append((i, opc, opnames.get(opc, f'0x{opc:03x}'), fc, dc,
                             rbar, wdep, misc))

    print(f'Instructions with changed ctrl: {len(changed)}')
    for i, opc, name, fc, dc, rbar, wdep, misc in changed:
        rbar_d = (dc >> 10) & 0x1f
        wdep_d = (dc >> 4) & 0x3f
        print(f'  [{i:2d}] {name:12s}  rbar=0x{rbar:02x} wdep=0x{wdep:02x} misc=0x{misc:x}'
              f'  (was rbar=0x{rbar_d:02x} wdep=0x{wdep_d:02x})')

    print()
    print('Cumulative bisect (add changes from 0 upward):')
    print('-' * 60)

    cumulative_set = []
    first_fail = None
    for item in changed:
        i = item[0]
        cumulative_set.append(i)
        cb = apply_changes(bytes(default_data), default_text_off,
                           bytes(fresh_data), fresh_text_off,
                           n_instr, cumulative_set)
        r = run_cubin(cb)
        idx_names = [opnames.get(struct.unpack_from('<Q', fresh_data,
                     fresh_text_off + x * 16)[0] & 0xFFF, f'0x{struct.unpack_from("<Q", fresh_data, fresh_text_off + x*16)[0] & 0xFFF:03x}')
                     for x in cumulative_set]
        status = 'OK' if r == 'PASS' else f'FAIL({r})'
        print(f'  After adding [{i:2d}]: {status}  (set={cumulative_set})')
        if r != 'PASS' and first_fail is None:
            first_fail = (i, list(cumulative_set))

    if first_fail is None:
        print()
        print('All cumulative combinations PASS! Issue is non-monotonic.')
    else:
        print()
        print(f'First failure at instruction [{first_fail[0]}]')
        print(f'Minimal failing set (so far): {first_fail[1]}')

    # Phase 2: Try all pairs from the changed set
    if first_fail is not None:
        print()
        print('Testing all pairs from changed set:')
        print('-' * 60)
        failing_pairs = []
        indices = [item[0] for item in changed]
        for ai in range(len(indices)):
            for bi in range(ai + 1, len(indices)):
                pair = [indices[ai], indices[bi]]
                cb = apply_changes(bytes(default_data), default_text_off,
                                   bytes(fresh_data), fresh_text_off,
                                   n_instr, pair)
                r = run_cubin(cb)
                if r != 'PASS':
                    na = opnames.get(struct.unpack_from('<Q', fresh_data,
                         fresh_text_off + indices[ai] * 16)[0] & 0xFFF, '?')
                    nb = opnames.get(struct.unpack_from('<Q', fresh_data,
                         fresh_text_off + indices[bi] * 16)[0] & 0xFFF, '?')
                    print(f'  FAIL: [{indices[ai]}]{na} + [{indices[bi]}]{nb} -> {r}')
                    failing_pairs.append(pair)

        if not failing_pairs:
            print('  No failing pairs found — need 3+ instruction interaction.')
        else:
            print(f'  {len(failing_pairs)} failing pair(s) found!')


if __name__ == '__main__':
    main()
