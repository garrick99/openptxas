"""Patch LDCU.64 instructions in our cubin to use wdep=0x35 (LDG slot).

Hypothesis: LDCU.64 must use wdep=0x35 on SM_120, not 0x31/0x33.
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


def get_ctrl(raw16):
    raw24 = (raw16[15] << 16) | (raw16[14] << 8) | raw16[13]
    return raw24 >> 1


def set_ctrl(data, base, ctrl):
    raw24 = (ctrl & 0x7FFFFF) << 1
    data[base + 13] = raw24 & 0xFF
    data[base + 14] = (raw24 >> 8) & 0xFF
    data[base + 15] = ((raw24 >> 16) & 0xFF) | (data[base + 15] & 0x04)


def patch(input_path, output_path, new_wdep=0x35):
    data = bytearray(open(input_path, 'rb').read())
    text_off, text_size = find_text_section(data)
    if text_off is None:
        print("No .text section found!")
        return

    patched = 0
    for i in range(0, text_size, 16):
        base = text_off + i
        raw = data[base:base + 16]

        # Identify LDCU.64: opcode=0x7ac (b0=0xac, b1=0x77), b9=0x0a
        lo = struct.unpack_from('<Q', bytes(raw), 0)[0]
        opc = lo & 0xFFF
        if opc == 0x7ac and raw[9] == 0x0a:
            ctrl = get_ctrl(raw)
            stall = (ctrl >> 17) & 0x3f
            yld   = (ctrl >> 16) & 1
            wbar  = (ctrl >> 15) & 1
            rbar  = (ctrl >> 10) & 0x1f
            old_wdep = (ctrl >> 4) & 0x3f
            misc  = ctrl & 0xf
            # Update wdep and set misc=1 (non-zero for odd wdep 0x35)
            new_misc = 1  # odd wdep requires non-zero misc
            new_ctrl = (stall << 17) | (yld << 16) | (wbar << 15) | (rbar << 10) | (new_wdep << 4) | new_misc
            set_ctrl(data, base, new_ctrl)
            print(f"  Patched LDCU.64 at +0x{i:04x}: wdep 0x{old_wdep:02x} -> 0x{new_wdep:02x}, ctrl 0x{ctrl:06x} -> 0x{new_ctrl:06x}")
            patched += 1

    with open(output_path, 'wb') as f:
        f.write(data)
    print(f"Patched {patched} LDCU.64 instructions -> {output_path}")


if __name__ == '__main__':
    patch(sys.argv[1], sys.argv[2])
