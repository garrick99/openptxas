"""Test: patch misc field for specific wdep patterns.

ptxas uses misc=5 for real ALU instructions (wdep=0x3e) and misc=0 only for NOP/BRA.
Our scoreboard uses misc=0 for ALL even-wdep instructions.
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


NOP_OPC  = 0x918
BRA_OPC  = 0x947
EXIT_OPC = 0x94d


def patch(input_path, output_path, misc_mode):
    """
    misc_mode:
      'force1'  — set misc=1 for ALL instructions
      'alu5'    — set misc=5 for wdep=0x3e non-NOP/BRA/EXIT instructions
      'all_nonzero' — set misc=1 for wdep=0x3e non-NOP/BRA instructions only
    """
    data = bytearray(open(input_path, 'rb').read())
    text_off, text_size = find_text_section(data)

    for i in range(0, text_size, 16):
        base = text_off + i
        raw = bytes(data[base:base+16])
        lo = struct.unpack_from('<Q', raw, 0)[0]
        opc = lo & 0xFFF

        b13 = raw[13]; b14 = raw[14]; b15 = raw[15]
        raw24 = (b15 << 16) | (b14 << 8) | b13
        ctrl = raw24 >> 1

        stall = (ctrl >> 17) & 0x3f
        yld   = (ctrl >> 16) & 1
        wbar  = (ctrl >> 15) & 1
        rbar  = (ctrl >> 10) & 0x1f
        wdep  = (ctrl >> 4) & 0x3f
        misc  = ctrl & 0xf

        new_misc = misc
        if misc_mode == 'force1':
            new_misc = 1
        elif misc_mode == 'alu5':
            if wdep == 0x3e and opc not in (NOP_OPC, BRA_OPC, EXIT_OPC):
                new_misc = 5
        elif misc_mode == 'all_nonzero':
            if misc == 0 and opc not in (NOP_OPC, BRA_OPC):
                new_misc = 1

        if new_misc != misc:
            new_ctrl = (stall << 17) | (yld << 16) | (wbar << 15) | (rbar << 10) | (wdep << 4) | new_misc
            new_raw24 = (new_ctrl & 0x7FFFFF) << 1
            data[base + 13] = new_raw24 & 0xFF
            data[base + 14] = (new_raw24 >> 8) & 0xFF
            data[base + 15] = ((new_raw24 >> 16) & 0xFF) | (raw[15] & 0x04)

    with open(output_path, 'wb') as f:
        f.write(data)
    print(f"Written {output_path} (mode={misc_mode})")


if __name__ == '__main__':
    patch(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'alu5')
