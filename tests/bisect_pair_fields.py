"""
Drill into the LDCU.64[3] + IADD64-UR[4] failing pair.
Test each individual field change to find the minimal trigger.
"""
import struct, subprocess, sys, os

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
            return struct.unpack_from('<Q', data, sh + 24)[0]
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


def make_ctrl(rbar, wdep, misc, stall=0):
    return (stall << 17) | (rbar << 10) | (wdep << 4) | misc


def run_cubin(cubin_bytes):
    with open(WORK, 'wb') as f:
        f.write(cubin_bytes)
    r = subprocess.run([sys.executable, RUNNER_PATH, WORK],
                       capture_output=True, text=True, timeout=30)
    return (r.stdout + r.stderr).strip() or f'EXIT:{r.returncode}'


def test_pair(default_data, text_off, idx_a, idx_b, ctrl_a, ctrl_b):
    """Apply ctrl_a at idx_a and ctrl_b at idx_b, run."""
    test = bytearray(default_data)
    off_a = text_off + idx_a * 16
    off_b = text_off + idx_b * 16
    raw_a = bytes(test[off_a:off_a+16])
    raw_b = bytes(test[off_b:off_b+16])
    test[off_a:off_a+16] = patch_ctrl(raw_a, ctrl_a)
    test[off_b:off_b+16] = patch_ctrl(raw_b, ctrl_b)
    r = run_cubin(bytes(test))
    return 'OK' if r == 'PASS' else f'FAIL({r})'


def main():
    with open(DEFAULT, 'rb') as f:
        default_data = bytearray(f.read())
    with open(FRESH, 'rb') as f:
        fresh_data = bytearray(f.read())

    d_off = get_text_off(bytes(default_data))
    f_off_base = get_text_off(bytes(fresh_data))

    # Our ctrl values for instructions 3 and 4 (first failing pair: LDCU + IADD64-UR)
    # LDCU.64 [3]:   rbar=0x01 wdep=0x31 misc=0x1  (default: rbar=0x01 wdep=0x3e misc=0x0)
    # IADD64-UR [4]: rbar=0x03 wdep=0x3e misc=0x5  (default: rbar=0x01 wdep=0x3e misc=0x0)

    # Default ctrl values (all same for these positions)
    DEF = make_ctrl(0x01, 0x3e, 0x0)

    # Our ctrl values
    LDCU_OURS  = make_ctrl(0x01, 0x31, 0x1)  # wdep=0x31, misc=1
    IADD_OURS  = make_ctrl(0x03, 0x3e, 0x5)  # rbar=0x03, misc=5

    print('Pair [3]=LDCU + [4]=IADD64-UR field decomposition')
    print('=' * 65)

    # Test subsets of field changes for LDCU
    ldcu_variants = [
        ('default',           make_ctrl(0x01, 0x3e, 0x0)),
        ('wdep=0x31',         make_ctrl(0x01, 0x31, 0x0)),  # only wdep change
        ('misc=1',            make_ctrl(0x01, 0x3e, 0x1)),  # only misc change
        ('wdep=0x31+misc=1',  make_ctrl(0x01, 0x31, 0x1)),  # full ours
    ]
    iadd_variants = [
        ('default',            make_ctrl(0x01, 0x3e, 0x0)),
        ('rbar=0x03',          make_ctrl(0x03, 0x3e, 0x0)),  # only rbar change
        ('misc=5',             make_ctrl(0x01, 0x3e, 0x5)),  # only misc change
        ('rbar=0x03+misc=5',   make_ctrl(0x03, 0x3e, 0x5)),  # full ours
    ]

    print(f'{"LDCU":30s}  {"IADD":30s}  Result')
    print('-' * 75)
    for lname, lctrl in ldcu_variants:
        for iname, ictrl in iadd_variants:
            r = test_pair(bytes(default_data), d_off, 3, 4, lctrl, ictrl)
            marker = '  <<<' if 'FAIL' in r else ''
            print(f'  LDCU:{lname:25s}  IADD:{iname:25s}  {r}{marker}')
        print()

    # Now test the same for pair [7]+[8]
    print()
    print('Pair [7]=LDCU + [8]=IADD64-UR field decomposition')
    print('=' * 65)
    print(f'{"LDCU":30s}  {"IADD":30s}  Result')
    print('-' * 75)
    for lname, lctrl in ldcu_variants:
        for iname, ictrl in iadd_variants:
            r = test_pair(bytes(default_data), d_off, 7, 8, lctrl, ictrl)
            marker = '  <<<' if 'FAIL' in r else ''
            print(f'  LDCU:{lname:25s}  IADD:{iname:25s}  {r}{marker}')
        print()


if __name__ == '__main__':
    main()
