"""Compare SASS ctrl words between two cubins instruction-by-instruction."""
import struct, sys

def find_text_section(data):
    """Find .text section offset and size in ELF cubin."""
    e_shoff = struct.unpack_from('<Q', data, 40)[0]
    e_shnum = struct.unpack_from('<H', data, 60)[0]
    e_shstrndx = struct.unpack_from('<H', data, 62)[0]

    # Get shstrtab
    sh_base = e_shoff + e_shstrndx * 64
    shstrtab_off = struct.unpack_from('<Q', data, sh_base + 24)[0]
    shstrtab_size = struct.unpack_from('<Q', data, sh_base + 32)[0]
    shstrtab = data[shstrtab_off:shstrtab_off + shstrtab_size]

    results = []
    for i in range(e_shnum):
        sh = e_shoff + i * 64
        name_off = struct.unpack_from('<I', data, sh)[0]
        sh_type = struct.unpack_from('<I', data, sh + 4)[0]
        sh_off = struct.unpack_from('<Q', data, sh + 24)[0]
        sh_size = struct.unpack_from('<Q', data, sh + 32)[0]

        end = shstrtab.find(b'\x00', name_off)
        name = shstrtab[name_off:end].decode('utf-8', errors='replace')

        if sh_type == 1 and name.startswith('.text.'):  # SHT_PROGBITS + .text
            results.append((name, sh_off, sh_size))
    return results


def decode_ctrl(raw16):
    """Decode 23-bit ctrl from bytes 13-15 of a 16-byte instruction."""
    raw24 = (raw16[15] << 16) | (raw16[14] << 8) | raw16[13]
    ctrl = raw24 >> 1
    stall = (ctrl >> 17) & 0x3f
    yld   = (ctrl >> 16) & 1
    wbar  = (ctrl >> 15) & 1
    rbar  = (ctrl >> 10) & 0x1f
    wdep  = (ctrl >> 4) & 0x3f
    misc  = ctrl & 0xf
    return ctrl, stall, yld, wbar, rbar, wdep, misc


def get_opcode_label(raw16):
    """Get a short label for the instruction based on the opcode."""
    lo = struct.unpack_from('<Q', raw16, 0)[0]
    opc = lo & 0xFFF
    labels = {
        0xb82: 'LDC', 0x7ac: 'LDCU', 0x919: 'S2R', 0x9c3: 'S2UR',
        0x981: 'LDG', 0x986: 'STG', 0x918: 'NOP', 0x94d: 'EXIT',
        0x947: 'BRA', 0x210: 'IADD3', 0x221: 'FADD', 0x235: 'IADD64',
        0xc35: 'IADD64U', 0x824: 'IMAD', 0x224: 'IMAD2',
        0xc0c: 'ISETP_UR', 0x20c: 'ISETP',
    }
    name = labels.get(opc, f'opc{opc:03x}')
    dest = raw16[2]
    return f'{name}(R{dest})'


def check_validity(ctrl, rbar, wdep, misc):
    """Check ctrl word validity rules."""
    issues = []
    # TABLES_opex_0: odd wdep → misc must be non-zero
    if (wdep & 1) and misc == 0:
        issues.append(f'TABLES_opex_0: wdep=0x{wdep:02x} odd but misc=0')
    # rbar must have bit 0 set (always-valid bit)
    if rbar != 0 and not (rbar & 1):
        issues.append(f'rbar=0x{rbar:02x} missing bit0')
    return issues


def dump_cubin(path):
    data = open(path, 'rb').read()
    sections = find_text_section(data)
    results = {}
    for name, off, size in sections:
        instrs = []
        for i in range(0, size, 16):
            raw = data[off + i:off + i + 16]
            if len(raw) < 16:
                break
            ctrl, stall, yld, wbar, rbar, wdep, misc = decode_ctrl(raw)
            label = get_opcode_label(raw)
            instrs.append({
                'offset': i,
                'raw': raw.hex(),
                'label': label,
                'ctrl': ctrl,
                'stall': stall, 'yld': yld, 'wbar': wbar,
                'rbar': rbar, 'wdep': wdep, 'misc': misc,
                'body': raw[:12].hex(),
                'ctrlbytes': raw[12:].hex(),
            })
        results[name] = instrs
    return results


def compare(path_a, path_b):
    a = dump_cubin(path_a)
    b = dump_cubin(path_b)

    # Find matching text sections
    a_keys = [k for k in a if '.text.' in k]
    b_keys = [k for k in b if '.text.' in k]

    if not a_keys or not b_keys:
        print("No .text sections found!")
        return

    a_instrs = a[a_keys[0]]
    b_instrs = b[b_keys[0]]

    print(f"A: {path_a} ({a_keys[0]}), {len(a_instrs)} instrs")
    print(f"B: {path_b} ({b_keys[0]}), {len(b_instrs)} instrs")
    print()

    # Print header
    print(f"{'off':>5} {'label':>14}  {'ctrl_A':>8} {'ctrl_B':>8}  "
          f"{'stall':>5} {'rbar':>4} {'wdep':>4} {'misc':>4}  "
          f"{'body_A':>24} {'body_B':>24}  {'issues'}")
    print('-' * 140)

    max_len = max(len(a_instrs), len(b_instrs))
    ctrl_diffs = 0
    body_diffs = 0

    for i in range(max_len):
        if i >= len(a_instrs):
            ia = None
        else:
            ia = a_instrs[i]
        if i >= len(b_instrs):
            ib = None
        else:
            ib = b_instrs[i]

        if ia is None:
            print(f"{i*16:5x} [A missing] // {ib['label']}")
            continue
        if ib is None:
            print(f"{i*16:5x} [B missing] // {ia['label']}")
            continue

        label = ia['label']
        ctrl_match = ia['ctrl'] == ib['ctrl']
        body_match = ia['body'] == ib['body']

        if not ctrl_match:
            ctrl_diffs += 1
        if not body_match:
            body_diffs += 1

        issues = check_validity(ia['ctrl'], ia['rbar'], ia['wdep'], ia['misc'])
        issues_b = check_validity(ib['ctrl'], ib['rbar'], ib['wdep'], ib['misc'])
        all_issues = issues + [f'B:{x}' for x in issues_b]

        # Print if there are differences or validity issues
        if not ctrl_match or not body_match or all_issues:
            ctrl_a_str = f'0x{ia["ctrl"]:06x}'
            ctrl_b_str = f'0x{ib["ctrl"]:06x}'
            body_diff_marker = '!!' if not body_match else '  '
            ctrl_diff_marker = '!!' if not ctrl_match else '  '
            stall_str = f'{ia["stall"]}/{ib["stall"]}'
            rbar_str = f'{ia["rbar"]:02x}/{ib["rbar"]:02x}'
            wdep_str = f'{ia["wdep"]:02x}/{ib["wdep"]:02x}'
            misc_str = f'{ia["misc"]:x}/{ib["misc"]:x}'
            issue_str = ', '.join(all_issues) if all_issues else ''
            print(f'{i*16:5x} {label:>14} {ctrl_diff_marker}{ctrl_a_str:>8} {ctrl_b_str:>8}  '
                  f'{stall_str:>5} {rbar_str:>4} {wdep_str:>4} {misc_str:>4} '
                  f'{body_diff_marker}{ia["body"]:>24} {ib["body"]:>24}  {issue_str}')

    print()
    print(f'Ctrl differences: {ctrl_diffs}/{max_len}')
    print(f'Body differences: {body_diffs}/{max_len}')

    # Also dump ALL ctrl words from A with validity check
    print()
    print("=== Full ctrl dump for A (showing all) ===")
    print(f"{'off':>5} {'label':>14} {'ctrl':>8} {'stall':>5} {'rbar':>4} {'wdep':>4} {'misc':>4}  ctrl_bytes  issues")
    print('-' * 100)
    for ia in a_instrs:
        issues = check_validity(ia['ctrl'], ia['rbar'], ia['wdep'], ia['misc'])
        issue_str = ', '.join(issues) if issues else ''
        flag = '**' if issues else '  '
        print(f'{ia["offset"]:5x} {ia["label"]:>14} {flag}0x{ia["ctrl"]:06x} '
              f'{ia["stall"]:>5} {ia["rbar"]:>4x} {ia["wdep"]:>4x} {ia["misc"]:>4x}  '
              f'{ia["ctrlbytes"]}  {issue_str}')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        compare(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        data = dump_cubin(sys.argv[1])
        for sec, instrs in data.items():
            print(f"Section: {sec}, {len(instrs)} instructions")
            for ia in instrs:
                issues = check_validity(ia['ctrl'], ia['rbar'], ia['wdep'], ia['misc'])
                issue_str = ', '.join(issues) if issues else ''
                flag = '**' if issues else '  '
                print(f'  {ia["offset"]:5x} {ia["label"]:>14} {flag}0x{ia["ctrl"]:06x} '
                      f'stall={ia["stall"]} rbar=0x{ia["rbar"]:02x} wdep=0x{ia["wdep"]:02x} misc=0x{ia["misc"]:x}  '
                      f'{ia["ctrlbytes"]}  {issue_str}')
    else:
        print(f"Usage: {sys.argv[0]} <cubin_a> [cubin_b]")
