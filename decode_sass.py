#!/usr/bin/env python3
"""Decode ctrl words from verbose SASS dump."""

# Paste the verbose hex lines here
sass_lines = [
    ("827b01ff00df00000008000000e20f00", "LDC R1 frame ptr"),
    ("19790600000000000021000000220e00", "S2R R6 tid.x"),
    ("ac770cff00720000000a000800ae0e00", "LDCU.64 UR12 n_iters"),
    ("ac7704ff006b0000000a0008002e0e00", "LDCU.64 UR4 mem desc"),
    ("ac7708ff00700000000a0008006e0e00", "LDCU.64 UR8 out_data"),
    ("ac770aff00710000000a0008002e0e00", "LDCU.64 UR10 out_len"),
    ("827b02ff00e6000000080000002a0e00", "LDC R2 b.lo"),
    ("827b03ff00e7000000080000002c0e00", "LDC R3 b.hi"),
    ("827b04ff00e8000000080000002e0e00", "LDC R4 c.lo"),
    ("827b05ff00e900000008000000300e00", "LDC R5 c.hi"),
    ("c3790600000000000025000000320e00", "S2UR UR6 ctaid.x"),
    ("827b07ff00d800000008000000340e00", "LDC R7 ntid.x"),
    ("247c07070600000006028e0f00c21f00", "IMAD R7=ctaid*ntid+tid"),
    ("10780aff00000000ffe0ff0700d80f00", "IADD3 R10=0 fd10.lo"),
    ("10780bff0000f03fffe0ff0700da0f00", "IADD3 R11=3ff00000 fd10.hi"),
    ("10780cff00000000ffe0ff0700dc0f00", "IADD3 R12=0 fd11.lo"),
    ("10780dff00000040ffe0ff0700de0f00", "IADD3 R13=40000000 fd11.hi"),
    ("10780eff00000000ffe0ff0700c00f00", "IADD3 R14=0 fd12.lo"),
    ("10780fff00000840ffe0ff0700c20f00", "IADD3 R15=40080000 fd12.hi"),
    ("107810ff00000000ffe0ff0700c40f00", "IADD3 R16=0 fd13.lo"),
    ("107811ff00001040ffe0ff0700c60f00", "IADD3 R17=40100000 fd13.hi"),
    ("107812ff00000000ffe0ff0700c80f00", "IADD3 R18=0 rd14.lo"),
    ("107813ff00000000ffe0ff0700ca0f00", "IADD3 R19=0 rd14.hi"),
    ("0c7c00120c0000007060f00b00c05f00", "ISETP loop guard"),
    ("4709fc00fcffffff1700800300e21f00", "@P0 BRA loop_end"),
    ("287c1a0a020000000000000800641e00", "DMUL R26,R10,R2 a0*b"),
    ("297e1c1a040000000000000800643e00", "DADD R28,R26,R4 +c"),
    ("287c1a0c020000000000000800641e00", "DMUL R26,R12,R2 a1*b"),
    ("297e1e1a040000000000000800643e00", "DADD R30,R26,R4 +c"),
    ("287c1a0e020000000000000800641e00", "DMUL R26,R14,R2 a2*b"),
    ("297e201a040000000000000800643e00", "DADD R32,R26,R4 +c"),
    ("287c1a10020000000000000800641e00", "DMUL R26,R16,R2 a3*b"),
    ("297e221a040000000000000800643e00", "DADD R34,R26,R4 +c"),
    ("10781aff01000000ffe0ff0700c00f00", "IADD3 R26=1 rd25.lo"),
    ("10781bff00000000ffe0ff0700c20f00", "IADD3 R27=0 rd25.hi"),
    ("18790000000000000000000000c00f00", "NOP"),
    ("357224121a00000000028e0700c61f00", "IADD.64 R36=R18+R26"),
    ("10720a1cff000000ffe0f10700c82f00", "MOV R10=R28 fd10.lo"),
    ("10720b1dff000000ffe0f10700ca2f00", "MOV R11=R29 fd10.hi"),
    ("10720c1eff000000ffe0f10700cc2f00", "MOV R12=R30 fd11.lo"),
    ("10720d1fff000000ffe0f10700ce2f00", "MOV R13=R31 fd11.hi"),
    ("10720e20ff000000ffe0f10700d02f00", "MOV R14=R32 fd12.lo"),
    ("10720f21ff000000ffe0f10700d22f00", "MOV R15=R33 fd12.hi"),
    ("10721022ff000000ffe0f10700d42f00", "MOV R16=R34 fd13.lo"),
    ("10721123ff000000ffe0f10700d62f00", "MOV R17=R35 fd13.hi"),
    ("10721224ff000000ffe0f10700d81f00", "MOV R18=R36 rd14.lo"),
    ("10721325ff000000ffe0f10700da1f00", "MOV R19=R37 rd14.hi"),
    ("4779fc00fcffffffe7ff830300e20f00", "BRA loop_back"),
    ("107808ff00000000ffe0ff0700de0f00", "IADD3 R8=0 post-loop"),
    ("10720207ff000000ffe0f10700c01f00", "MOV R2=R7 tid.lo"),
    ("107203ffff000000ffe0f10700c20f00", "MOV R3=RZ tid.hi"),
    ("0c7c00020a0000007060f00b00c01f00", "ISETP bounds check"),
    ("298e040a0c0000000000000800641e00", "@!P0 DADD R4,R10,R12"),
    ("298e12040e0000000000000800643e00", "@!P0 DADD R18,R4,R14"),
    ("298e1812100000000000000800643e00", "@!P0 DADD R24,R18,R16"),
    ("10881eff08000000ffe0ff0700cc0f00", "@!P0 IADD3 R30=8"),
    ("10881fff00000000ffe0ff0700ce0f00", "@!P0 IADD3 R31=0"),
    ("258220021e000000ff028e0700c21f00", "@!P0 IMAD.WIDE R32=R2*R30"),
    ("258228021f000000ff028e0700c21f00", "@!P0 IMAD.WIDE R40=R2*R31 cross"),
    ("18790000000000000000000000c00f00", "NOP"),
    ("1082212128000000ffe0f10700d61f00", "@!P0 IADD3 R33+=R40"),
    ("258228031e000000ff028e0700c21f00", "@!P0 IMAD.WIDE R40=R3*R30 cross"),
    ("18790000000000000000000000c00f00", "NOP"),
    ("1082212128000000ffe0f10700dc1f00", "@!P0 IADD3 R33+=R40"),
    ("358c22200800000000028e0f00ca2f00", "@!P0 IADD.64 R34=R32+UR8"),
    ("18790000000000000000000000c00f00", "NOP"),
    ("8689002218000000041b100c00e20f00", "@!P0 STG.E.64"),
    ("108806ff00000000ffe0ff0700c40f00", "@!P0 IADD3 R6=0"),
    ("18790000000000000000000000c00f00", "NOP"),
    ("10820706ff000000ffe0f10700c81f00", "@!P0 MOV R7=R6"),
    ("100807ff00000000ffe0ff0700ca1f00", "@P0 IADD3 R7=0"),
    ("107806ff00000000ffe0ff0700cc0f00", "IADD3 R6=0"),
    ("4d790000000000000000800300ea0f00", "EXIT"),
    ("4779fc00fcffffffffff830300c00f00", "BRA $ trap"),
]

OPCODE_MISC = {
    0x918: 0,   # NOP
    0x947: 0,   # BRA
    0xe29: 2,   # DADD
    0xc28: 2,   # DMUL
    0xc2b: 2,   # DFMA
    0xc35: 5,   # IADD.64-UR
    0xc0c: 0,   # ISETP R-UR
    0x20c: 0,   # ISETP R-R
    0x80a: 5,   # FSEL.step
    0x986: 1,   # STG.E
    0x988: 4,   # STS.E
    0x225: 1,   # IMAD.WIDE R-R (fixed)
    0x825: 1,   # IMAD.WIDE R-imm (fixed)
    0xc24: 1,   # IMAD R-UR (fixed)
    0x94d: 5,   # EXIT
}

def get_opcode(raw):
    return ((raw[10] & 0xF0) >> 4) | (raw[11] << 4)

print(f"{'off':>5}  {'opc':>6}  {'stl':>4}  {'rbar':>5}  {'wdep':>5}  {'misc':>5}  note")
print("-" * 85)

offset = 0
for hexstr, comment in sass_lines:
    raw = bytes.fromhex(hexstr)
    opcode = get_opcode(raw)
    b13, b14, b15 = raw[13], raw[14], raw[15]
    raw24 = (b15 & 0xFB) << 16 | b14 << 8 | b13
    ctrl = raw24 >> 1
    stall = (ctrl >> 17) & 0x3f
    rbar  = (ctrl >> 10) & 0x7f
    wdep  = (ctrl >>  4) & 0x3f
    misc  = ctrl & 0xF

    expected_misc = OPCODE_MISC.get(opcode, None)
    flag = ''
    if expected_misc is not None and misc != expected_misc:
        flag = f'  *** WRONG misc! expected={expected_misc}'
    elif expected_misc is None and misc not in (0, 1, 2):
        flag = f'  <<< unusual misc={misc}'

    print(f"{offset:>5}  0x{opcode:03x}   {stall:>4}  0x{rbar:02x}   0x{wdep:02x}    {misc:>5}  {comment}{flag}")
    offset += 16
