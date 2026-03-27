"""
Test candidate misc fixes for LDCU.64 and IADD64-UR.
The failing pair is LDCU misc=1 + IADD64-UR misc=5.
"""
import struct, subprocess, sys, os

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


def patch_ctrl(raw16, ctrl):
    buf = bytearray(raw16)
    raw24 = (ctrl & 0x7FFFFF) << 1
    buf[13] = raw24 & 0xFF
    buf[14] = (raw24 >> 8) & 0xFF
    buf[15] = ((raw24 >> 16) & 0xFF) | (buf[15] & 0x04)
    return bytes(buf)


def make_ctrl(rbar, wdep, misc, stall=0):
    return (stall << 17) | (rbar << 10) | (wdep << 4) | misc


def run_variant(cubin_bytes):
    with open(WORK, 'wb') as f:
        f.write(cubin_bytes)
    r = subprocess.run([sys.executable, RUNNER_PATH, WORK],
                       capture_output=True, text=True, timeout=30)
    return (r.stdout + r.stderr).strip() or f'EXIT:{r.returncode}'


def build_test(default_data, text_off,
               s2r_misc, ldcu4_misc, ldcu6_misc, ldcu6_wdep,
               iadd4_rbar, iadd4_misc,
               ldg_rbar, ldg_wdep,
               ldcu8_misc, ldcu8_wdep,
               iadd8_rbar, iadd8_misc,
               stg_rbar):
    """Build a test cubin with specified ctrl for each instruction."""
    test = bytearray(default_data)
    off = text_off

    def patch(idx, ctrl):
        o = off + idx * 16
        test[o:o+16] = patch_ctrl(bytes(test[o:o+16]), ctrl)

    # [0] LDC: keep default
    # [1] S2R
    patch(1, make_ctrl(0x01, 0x31, s2r_misc))
    # [2] LDCU UR4
    patch(2, make_ctrl(0x01, 0x31, ldcu4_misc))
    # [3] LDCU UR6
    patch(3, make_ctrl(0x01, ldcu6_wdep, ldcu6_misc))
    # [4] IADD64-UR R2←UR6
    patch(4, make_ctrl(iadd4_rbar, 0x3e, iadd4_misc))
    # [5] LDG
    patch(5, make_ctrl(ldg_rbar, ldg_wdep, 0x1))
    # [6] NOP: keep default
    # [7] LDCU UR8
    patch(7, make_ctrl(0x01, ldcu8_wdep, ldcu8_misc))
    # [8] IADD64-UR R2←UR8
    patch(8, make_ctrl(iadd8_rbar, 0x3e, iadd8_misc))
    # [9] STG
    patch(9, make_ctrl(stg_rbar, 0x3f, 0x1))
    # [10] EXIT
    patch(10, make_ctrl(0x01, 0x3f, 0x5))

    return bytes(test)


def main():
    with open(DEFAULT, 'rb') as f:
        default_data = bytearray(f.read())
    text_off = get_text_off(bytes(default_data))

    print('Testing misc combinations for LDCU.64 and IADD64-UR')
    print('All other ctrl fields match our scoreboard values (rbar, wdep)')
    print('=' * 70)

    # The full "our ctrl" settings (what scoreboard.py generates):
    # S2R: misc=1, LDCU UR4: misc=1, LDCU UR6: wdep=0x31 misc=1,
    # IADD4: rbar=0x03 misc=5, LDG: rbar=0x03 wdep=0x35,
    # LDCU UR8: wdep=0x31 misc=1, IADD8: rbar=0x03 misc=5, STG: rbar=0x0b

    print('\n--- Varying LDCU misc and IADD64-UR misc (full scoreboard rbar/wdep) ---')
    ldcu_misc_vals = [0, 1, 2, 7]
    iadd_misc_vals = [0, 1, 2, 5]

    for ldcu_m in ldcu_misc_vals:
        for iadd_m in iadd_misc_vals:
            cb = build_test(bytes(default_data), text_off,
                            s2r_misc=1,
                            ldcu4_misc=ldcu_m, ldcu6_misc=ldcu_m, ldcu6_wdep=0x31,
                            iadd4_rbar=0x03, iadd4_misc=iadd_m,
                            ldg_rbar=0x03, ldg_wdep=0x35,
                            ldcu8_misc=ldcu_m, ldcu8_wdep=0x31,
                            iadd8_rbar=0x03, iadd8_misc=iadd_m,
                            stg_rbar=0x0b)
            r = run_variant(cb)
            marker = '  <<<' if r != 'PASS' else ''
            print(f'  LDCU misc={ldcu_m}  IADD misc={iadd_m}  ->  {r}{marker}')
        print()

    print('\n--- All correct: LDCU misc=7 (ptxas style), IADD misc=? ---')
    for iadd_m in range(16):
        cb = build_test(bytes(default_data), text_off,
                        s2r_misc=1,
                        ldcu4_misc=7, ldcu6_misc=7, ldcu6_wdep=0x31,
                        iadd4_rbar=0x03, iadd4_misc=iadd_m,
                        ldg_rbar=0x03, ldg_wdep=0x35,
                        ldcu8_misc=7, ldcu8_wdep=0x31,
                        iadd8_rbar=0x03, iadd8_misc=iadd_m,
                        stg_rbar=0x0b)
        r = run_variant(cb)
        marker = '  <<<' if r != 'PASS' else ''
        print(f'  IADD misc=0x{iadd_m:x}  ->  {r}{marker}')


if __name__ == '__main__':
    main()
