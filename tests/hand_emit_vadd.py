#!/usr/bin/env python3
"""Hand-emit the ptxas vector_add SASS sequence using our encoders.
This proves whether our encoders produce correct bytes."""

import sys, struct
sys.path.insert(0, 'C:/users/kraken/openptxas')

from sass.encoding.sm_120_opcodes import (
    encode_ldc, encode_ldc_64, encode_s2r, encode_s2ur,
    encode_ldcu_32, encode_ldcu_64,
    encode_imad_ur, encode_imad_shl_u32,
    encode_isetp_ur,
    encode_iadd64_ur, encode_ldg_e, encode_stg_e,
    encode_fadd, encode_exit, encode_bra, encode_nop,
    SR_TID_X, SR_CTAID_X, ISETP_GE,
    patch_pred,
)
from sass.encoding.sm_120_encode import encode_shf_r_s32_hi
from sass.isel import SassInstr

RZ = 0xFF

# Match ptxas register allocation:
# R1 = stack, R2 = gid/byte_offset, R3 = tid.x/sign, R4:R5 = addr pair a
# R6:R7 = addr pair b, R9 = FADD result
# UR4 = ctaid.x (then descriptor), UR5 = n, UR6:UR7 = b_ptr
# UR8:UR9 = out_ptr, UR10:UR11 = a_ptr (from LDCU.128)

instrs = []

# [0] LDC R1, c[0][0x37c]  — stack frame
instrs.append(SassInstr(encode_ldc(1, 0, 0x37c), 'LDC R1'))

# [1] S2R R3, SR_TID.X
instrs.append(SassInstr(encode_s2r(3, SR_TID_X), 'S2R R3'))

# [2] S2UR UR4, SR_CTAID.X
instrs.append(SassInstr(encode_s2ur(4, SR_CTAID_X), 'S2UR UR4'))

# [3] LDCU UR5, c[0][0x398]  — n param
instrs.append(SassInstr(encode_ldcu_32(5, 0, 0x398), 'LDCU UR5, n'))

# [4] LDC R2, c[0][0x360]  — ntid.x
instrs.append(SassInstr(encode_ldc(2, 0, 0x360), 'LDC R2, ntid.x'))

# [5] IMAD R2, R2, UR4, R3  — gid = ntid.x * ctaid.x + tid.x
instrs.append(SassInstr(encode_imad_ur(2, 2, 4, 3), 'IMAD R2=R2*UR4+R3'))

# [6] ISETP.GE P0, R2, UR5  — P0 = (gid >= n)
instrs.append(SassInstr(encode_isetp_ur(0, 2, 5, cmp=ISETP_GE), 'ISETP.GE P0, R2, UR5'))

# [7] @P0 EXIT
exit_raw = patch_pred(encode_exit(), pred=0, neg=False)
instrs.append(SassInstr(exit_raw, '@P0 EXIT'))

# [8] LDCU.128 UR8, c[0][0x380]  — out+a params (16 bytes)
# LDCU.128: same as LDCU.64 but with b9=0x0c
ldcu128_raw = bytearray(encode_ldcu_64(8, 0, 0x380))
ldcu128_raw[9] = 0x0c  # 128-bit mode
instrs.append(SassInstr(bytes(ldcu128_raw), 'LDCU.128 UR8'))

# [9] IMAD.SHL R2, R2, 4, RZ  — byte_offset = gid * 4
instrs.append(SassInstr(encode_imad_shl_u32(2, 2, 2), 'IMAD.SHL R2=R2*4'))

# [10] LDCU.64 UR6, c[0][0x390]  — b param
instrs.append(SassInstr(encode_ldcu_64(6, 0, 0x390), 'LDCU.64 UR6, b'))

# [11] SHF.R.S32.HI R3, RZ, 31, R2  — sign extend
instrs.append(SassInstr(encode_shf_r_s32_hi(3, 2, 31), 'SHF R3=sign(R2)'))

# [12] LDCU.64 UR4, c[0][0x358]  — descriptor
instrs.append(SassInstr(encode_ldcu_64(4, 0, 0x358), 'LDCU.64 UR4, desc'))

# [13] IADD.64 R4, R2, UR10  — addr_a = offset + a_ptr
instrs.append(SassInstr(encode_iadd64_ur(4, 2, 10), 'IADD.64 R4=R2+UR10'))

# [14] IADD.64 R6, R2, UR6  — addr_b = offset + b_ptr
instrs.append(SassInstr(encode_iadd64_ur(6, 2, 6), 'IADD.64 R6=R2+UR6'))

# [15] LDG.E R4, desc[UR4][R4.64]  — a[gid]
instrs.append(SassInstr(encode_ldg_e(4, 4, 4, width=32), 'LDG R4'))

# [16] LDG.E R7, desc[UR4][R6.64]  — b[gid]
instrs.append(SassInstr(encode_ldg_e(7, 4, 6, width=32), 'LDG R7'))

# [17] IADD.64 R2, R2, UR8  — addr_out = offset + out_ptr
instrs.append(SassInstr(encode_iadd64_ur(2, 2, 8), 'IADD.64 R2=R2+UR8'))

# [18] FADD R9, R4, R7
instrs.append(SassInstr(encode_fadd(9, 4, 7), 'FADD R9=R4+R7'))

# [19] STG.E desc[UR4][R2.64], R9
instrs.append(SassInstr(encode_stg_e(4, 2, 9, width=32), 'STG [R2]=R9'))

# [20] EXIT
instrs.append(SassInstr(encode_exit(), 'EXIT'))

# [21] BRA (trap loop)
instrs.append(SassInstr(encode_bra(-16), 'BRA -1'))

# Pad to 32 instructions (512 bytes) with NOPs
while len(instrs) < 32:
    instrs.append(SassInstr(encode_nop(), 'NOP'))

# Concatenate SASS bytes
sass_bytes = b''.join(si.raw for si in instrs)
print(f'Generated {len(instrs)} instructions ({len(sass_bytes)} bytes)')

# Now compare against ptxas byte-by-byte
ref_data = open('C:/users/kraken/opencuda/tests/_ref.cubin', 'rb').read()
# ptxas .text at offset 1792
ref_sass = ref_data[1792:1792+512]

print('\nByte-by-byte comparison (first 22 instructions):')
diffs = 0
for j in range(22):
    our = sass_bytes[j*16:j*16+16]
    ref = ref_sass[j*16:j*16+16]
    match = (our == ref)
    if not match:
        diffs += 1
        print(f'  [{j:2d}] DIFF:')
        for b in range(16):
            if our[b] != ref[b]:
                print(f'       b{b}: ours={our[b]:#04x} ptxas={ref[b]:#04x}')
    else:
        # Check just the operational bytes (0-12)
        op_match = our[:13] == ref[:13]
        if not op_match:
            diffs += 1
            print(f'  [{j:2d}] OP DIFF (ctrl OK)')
        # else: fully matching, skip

print(f'\n{diffs} instructions differ out of 22')
print(f'{22-diffs} instructions match exactly')

# Write the hand-emitted cubin using our ELF template
oc_data = bytearray(open('C:/users/kraken/opencuda/tests/simple_vadd.cubin', 'rb').read())
e_shoff = struct.unpack_from('<Q', oc_data, 40)[0]
e_shnum = struct.unpack_from('<H', oc_data, 60)[0]
e_shentsize = struct.unpack_from('<H', oc_data, 58)[0]
e_shstrndx = struct.unpack_from('<H', oc_data, 62)[0]
sh_off = e_shoff + e_shstrndx * e_shentsize
str_offset = struct.unpack_from('<Q', oc_data, sh_off + 24)[0]
strtab = oc_data[str_offset:]
for i in range(e_shnum):
    off = e_shoff + i * e_shentsize
    sh_name_idx = struct.unpack_from('<I', oc_data, off)[0]
    sname = strtab[sh_name_idx:strtab.index(0, sh_name_idx)].decode('ascii', errors='replace')
    sh_offset_val = struct.unpack_from('<Q', oc_data, off + 24)[0]
    sh_size = struct.unpack_from('<Q', oc_data, off + 32)[0]
    sh_flags = struct.unpack_from('<Q', oc_data, off + 8)[0]
    if '.text.' in sname and sh_flags & 4:
        # Pad our SASS to match the section size
        padded = sass_bytes + (encode_nop() * ((sh_size - len(sass_bytes)) // 16))
        oc_data[sh_offset_val:sh_offset_val+sh_size] = padded[:sh_size]
        print(f'\nInjected {len(sass_bytes)} bytes into .text ({sh_size} byte section)')

open('C:/users/kraken/opencuda/tests/_hand.cubin', 'wb').write(oc_data)
print('Wrote _hand.cubin')
