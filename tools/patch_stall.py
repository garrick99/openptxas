"""Patch all ctrl words in a cubin to add maximum stall counts.

If stall=63 fixes ILLEGAL_INSTRUCTION → timing/dependency issue.
If it still crashes → encoding issue.
"""
import struct, sys

def find_text_section(data):
    e_shoff = struct.unpack_from('<Q', data, 40)[0]
    e_shnum = struct.unpack_from('<H', data, 60)[0]
    e_shstrndx = struct.unpack_from('<H', data, 62)[0]

    sh_base = e_shoff + e_shstrndx * 64
    shstrtab_off = struct.unpack_from('<Q', data, sh_base + 24)[0]
    shstrtab_size = struct.unpack_from('<Q', data, sh_base + 32)[0]
    shstrtab = data[shstrtab_off:shstrtab_off + shstrtab_size]

    for i in range(e_shnum):
        sh = e_shoff + i * 64
        name_off = struct.unpack_from('<I', data, sh)[0]
        sh_type = struct.unpack_from('<I', data, sh + 4)[0]
        sh_off = struct.unpack_from('<Q', data, sh + 24)[0]
        sh_size = struct.unpack_from('<Q', data, sh + 32)[0]

        end = shstrtab.find(b'\x00', name_off)
        name = shstrtab[name_off:end].decode('utf-8', errors='replace')

        if sh_type == 1 and name.startswith('.text.'):
            return sh_off, sh_size
    return None, None


def patch_stall(input_path, output_path, stall=15):
    data = bytearray(open(input_path, 'rb').read())
    text_off, text_size = find_text_section(data)
    if text_off is None:
        print("No .text section found!")
        return

    for i in range(0, text_size, 16):
        base = text_off + i
        # Read current ctrl from bytes 13-15
        b13 = data[base + 13]
        b14 = data[base + 14]
        b15 = data[base + 15]
        raw24 = (b15 << 16) | (b14 << 8) | b13
        ctrl = raw24 >> 1

        # Extract current fields
        old_stall = (ctrl >> 17) & 0x3f
        yld   = (ctrl >> 16) & 1
        wbar  = (ctrl >> 15) & 1
        rbar  = (ctrl >> 10) & 0x1f
        wdep  = (ctrl >> 4) & 0x3f
        misc  = ctrl & 0xf

        # Patch: set stall to desired value, keep other fields
        new_ctrl = (stall << 17) | (yld << 16) | (wbar << 15) | (rbar << 10) | (wdep << 4) | misc
        new_raw24 = (new_ctrl & 0x7FFFFF) << 1
        data[base + 13] = new_raw24 & 0xFF
        data[base + 14] = (new_raw24 >> 8) & 0xFF
        data[base + 15] = ((new_raw24 >> 16) & 0xFF) | (data[base + 15] & 0x04)

    with open(output_path, 'wb') as f:
        f.write(data)
    print(f"Written {output_path} with stall={stall} on all instructions")


if __name__ == '__main__':
    stall = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    patch_stall(sys.argv[1], sys.argv[2], stall)
