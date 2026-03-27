"""Dump .nv.info section contents from a cubin."""
import struct, sys

def find_sections(data):
    e_shoff = struct.unpack_from('<Q', data, 40)[0]
    e_shnum = struct.unpack_from('<H', data, 60)[0]
    e_shstrndx = struct.unpack_from('<H', data, 62)[0]
    sh_base = e_shoff + e_shstrndx * 64
    shstrtab_off = struct.unpack_from('<Q', data, sh_base + 24)[0]
    shstrtab_size = struct.unpack_from('<Q', data, sh_base + 32)[0]
    shstrtab = data[shstrtab_off:shstrtab_off + shstrtab_size]
    results = {}
    for i in range(e_shnum):
        sh = e_shoff + i * 64
        name_off = struct.unpack_from('<I', data, sh)[0]
        sh_off = struct.unpack_from('<Q', data, sh + 24)[0]
        sh_size = struct.unpack_from('<Q', data, sh + 32)[0]
        end = shstrtab.find(b'\x00', name_off)
        name = shstrtab[name_off:end].decode('utf-8', errors='replace')
        if sh_size > 0 and ('nv.info' in name or 'nv.compat' in name):
            results[name] = data[sh_off:sh_off + sh_size]
    return results


ATTR_NAMES = {
    0x04: 'CTAID_DIMS',
    0x0a: 'EXTERNS',
    0x0b: 'REQNTID',
    0x11: 'FILTER',
    0x12: 'BINARYTYPE',
    0x17: 'PARAM_INFO',
    0x19: 'S2RCTAIDX',
    0x1b: 'CBANK_PARAM_SIZE',
    0x1c: 'MAX_REG_COUNT',
    0x2f: 'CUDA_API_VER',
    0x36: 'ATTR_36',
    0x37: 'REGCOUNT',
    0x3a: 'ATTR_3a',
    0x4a: 'CTAID_DIMS2',
    0x50: 'PARAM_CBANK',
    0x5f: 'ATTR_5f',
}


def parse_attr_stream(data):
    """Parse EIATTR stream."""
    pos = 0
    attrs = []
    while pos < len(data):
        if pos + 2 > len(data):
            break
        fmt = data[pos]
        tag = data[pos + 1]
        name = ATTR_NAMES.get(tag, f'0x{tag:02x}')

        if fmt == 0x02:  # 2-byte payload
            payload = data[pos+2:pos+4]
            val = struct.unpack_from('<H', payload)[0]
            attrs.append((name, fmt, val, payload))
            pos += 4
        elif fmt == 0x03:  # 2-byte inline
            val = (data[pos+2] << 8) | data[pos+3] if len(data) > pos+3 else data[pos+2]
            val = struct.unpack_from('<H', data, pos+2)[0]
            attrs.append((name, fmt, val, data[pos+2:pos+4]))
            pos += 4
        elif fmt == 0x04:  # word payload
            size = struct.unpack_from('<H', data, pos+2)[0]
            payload = data[pos+4:pos+4+size]
            attrs.append((name, fmt, size, payload))
            pos += 4 + size
        else:
            attrs.append((f'UNKNOWN_FMT_{fmt:02x}_{name}', fmt, 0, b''))
            pos += 2
            break  # can't continue safely
    return attrs


def dump(path):
    data = open(path, 'rb').read()
    sections = find_sections(data)
    for sec_name in sorted(sections):
        sec_data = sections[sec_name]
        print(f"\n=== {sec_name} ({len(sec_data)} bytes) ===")
        print(f"  Raw: {sec_data.hex()}")
        attrs = parse_attr_stream(sec_data)
        for name, fmt, val, payload in attrs:
            if fmt == 0x04 and name == 'PARAM_INFO':
                # Parse param info entries
                print(f"  [{name}] (fmt=0x{fmt:02x}, size={val}): {payload.hex()}")
                if len(payload) >= 8:
                    ordinal = struct.unpack_from('<H', payload, 4)[0]
                    offset = struct.unpack_from('<H', payload, 6)[0]
                    flags = struct.unpack_from('<H', payload, 8)[0] if len(payload) >= 10 else 0
                    size_ind = (flags >> 8) & 0xff
                    size_bytes = 8 if size_ind == 0x21 else 4 if size_ind == 0x11 else size_ind
                    print(f"    ordinal={ordinal}, cbank_offset=0x{offset:04x} ({offset}), flags=0x{flags:04x}, size={size_bytes}B")
            elif fmt == 0x04:
                print(f"  [{name}] (fmt=0x{fmt:02x}, size={val}): {payload.hex()}")
            elif fmt == 0x03:
                print(f"  [{name}] (fmt=0x{fmt:02x}): 0x{val:04x}")
            elif fmt == 0x02:
                print(f"  [{name}] (fmt=0x{fmt:02x}): 0x{val:04x}")
            else:
                print(f"  [{name}] (fmt=0x{fmt:02x}): 0x{val:x}")


if __name__ == '__main__':
    dump(sys.argv[1])
