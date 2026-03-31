#!/usr/bin/env python3
"""
capmerc_analysis.py — Reverse-engineer the .nv.capmerc.text.{kernel} record format
for SM_120 Blackwell GPUs.

This script:
1. Parses .text and .nv.capmerc sections from multiple ptxas-generated cubins
2. Decodes each SASS instruction (opcode, registers, control word)
3. Parses capmerc header, body records (type 01/02), and trailer
4. Attempts to correlate capmerc records with instruction groups / scheduling barriers

No files are modified — pure read-only analysis.
"""

import struct
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# SM_120 opcode table (bits 11:0 of the 128-bit instruction)
# ---------------------------------------------------------------------------
OPCODES = {
    # Memory
    0x381: 'LDG.E',    0x385: 'STG.E',
    0x389: 'LDS',      0x38d: 'STS',
    0x30c: 'LDSM',     0x30e: 'ATOMG',
    0x182: 'LDCU',     0x189: 'LDC',
    0x431: 'LDCU.64?', # guess — 64-bit const uniform load
    # Integer ALU
    0x810: 'IADD3',    0x824: 'IMAD',     0x825: 'IMAD.WIDE',
    0x819: 'SHF',      0x816: 'LOP3',
    0x86c: 'ISETP',
    # Float ALU
    0x221: 'FADD',     0x220: 'FMUL',     0x223: 'FFMA',
    0x235: 'IADD3?',   # appears in many cubins as ALU
    # MMA
    0xb23: 'HMMA',     0xb24: 'IMMA',
    # Special
    0x910: 'S2R',      0x919: 'S2UR',     0x918: 'CS2R',
    0x802: 'MOV',      0x807: 'MOV32I',
    0x984: 'NOP',      0x94d: 'EXIT',     0x947: 'BRA',
    0x363: 'BAR',
    # Discovered in cubins
    0xb82: 'LDCU.UR?', # uniform register const load
    0x7ac: 'ULDC.64',  # uniform load constant 64-bit
    0x981: 'IADD3.X?', # extended add
    0x986: 'STG.E.64?',# 64-bit global store
    0xc0c: 'ISETP.GE?',# set predicate
    0xc11: 'IMAD.W?',  # wide multiply-add
}


@dataclass
class SassInstr:
    offset: int
    raw: bytes       # 16 bytes
    opcode: int
    op_name: str
    pred: int        # 0-6=P0-P6, 7=PT, 8-14=!P0-!P6
    dest: int        # register index (255=RZ)
    src0: int
    src1: int
    src2: int
    ctrl: int        # 23-bit control word
    stall: int
    yld: int
    wbar: int
    rbar: int
    wdep: int
    misc: int

    @property
    def is_padding(self):
        """CS2R with ctrl=0x7e0 is NOP padding."""
        return self.opcode == 0x918 and self.ctrl == 0x7e0

    @property
    def is_exit(self):
        return self.opcode == 0x94d

    @property
    def is_bra(self):
        return self.opcode == 0x947

    @property
    def max_reg(self):
        """Highest non-RZ register referenced."""
        m = -1
        for r in [self.dest, self.src0, self.src1, self.src2]:
            if r != 255 and r < 128 and r > m:  # cap at 128 to exclude SR codes
                m = r
        return m

    def reg_str(self, r):
        if r == 255:
            return 'RZ'
        return f'R{r}'


@dataclass
class CapmercRecord:
    offset: int      # byte offset within capmerc body
    rec_type: int    # 0x01 or 0x02
    raw: bytes       # full record bytes
    # Parsed fields (tentative)
    fields: dict = field(default_factory=dict)


def decode_sass(text_bytes: bytes) -> list[SassInstr]:
    """Decode all 128-bit SASS instructions from .text section."""
    instrs = []
    for i in range(0, len(text_bytes), 16):
        raw = text_bytes[i:i+16]
        lo = struct.unpack_from('<Q', raw, 0)[0]
        hi = struct.unpack_from('<Q', raw, 8)[0]

        opcode = lo & 0xFFF
        pred = (lo >> 12) & 0xF
        dest = (lo >> 16) & 0xFF
        src0 = (lo >> 24) & 0xFF
        src1 = (lo >> 32) & 0xFF
        src2 = (hi >> 0) & 0xFF

        # Control word from bytes 13-15
        raw24 = raw[13] | (raw[14] << 8) | (raw[15] << 16)
        ctrl = raw24 >> 1
        stall = (ctrl >> 17) & 0x3F
        yld = (ctrl >> 16) & 1
        wbar = (ctrl >> 15) & 1
        rbar = (ctrl >> 10) & 0x1F
        wdep = (ctrl >> 4) & 0x3F
        misc = ctrl & 0xF

        op_name = OPCODES.get(opcode, f'UNK_{opcode:#05x}')
        instrs.append(SassInstr(
            offset=i, raw=raw, opcode=opcode, op_name=op_name,
            pred=pred, dest=dest, src0=src0, src1=src1, src2=src2,
            ctrl=ctrl, stall=stall, yld=yld, wbar=wbar,
            rbar=rbar, wdep=wdep, misc=misc
        ))
    return instrs


def parse_capmerc(data: bytes) -> dict:
    """Parse capmerc header, body records, and trailer."""
    result = {
        'raw': data,
        'size': len(data),
        'header': {},
        'records': [],
        'trailer': None,
        'filler': [],   # 0x41-prefixed filler runs
    }

    if len(data) < 16:
        return result

    # Header: 16 bytes
    hdr = data[:16]
    result['header'] = {
        'magic': hdr[:8],
        'magic_hex': hdr[:8].hex(),
        'reg_count': hdr[8],
        'field_9': hdr[9],
        'field_10': hdr[10],
        'field_11': hdr[11],
        'capability_mask': struct.unpack_from('<I', hdr, 12)[0],
        'capability_hex': hdr[12:16].hex(),
    }

    # Trailer: last 2 bytes
    if len(data) >= 18:
        result['trailer'] = {
            'byte0': data[-2],
            'byte1': data[-1],
            'hex': data[-2:].hex(),
        }

    # Body: bytes 16 to len-2
    body = data[16:-2] if len(data) > 18 else b''
    pos = 0
    rec_idx = 0

    while pos < len(body):
        rec_type = body[pos]

        if rec_type == 0x01:
            # Type 01 record: 16 bytes
            if pos + 16 > len(body):
                break
            rec_bytes = body[pos:pos+16]
            rec = CapmercRecord(offset=16+pos, rec_type=0x01, raw=rec_bytes)
            rec.fields = parse_type01(rec_bytes)
            result['records'].append(rec)
            pos += 16
            rec_idx += 1

        elif rec_type == 0x02:
            # Type 02 record: 32 bytes (two 16-byte halves)
            if pos + 32 > len(body):
                break
            rec_bytes = body[pos:pos+32]
            rec = CapmercRecord(offset=16+pos, rec_type=0x02, raw=rec_bytes)
            rec.fields = parse_type02(rec_bytes)
            result['records'].append(rec)
            pos += 32
            rec_idx += 1

        elif rec_type == 0x41:
            # Filler/padding (5 bytes each, repeating pattern)
            # These appear as 41 0c 50 04 41 0c 50 04 ...
            # Try to consume all 0x41-prefixed 5-byte chunks
            filler_start = pos
            while pos < len(body) and body[pos] == 0x41:
                chunk = body[pos:pos+5]
                result['filler'].append({
                    'offset': 16+pos,
                    'hex': chunk.hex() if len(chunk) == 5 else body[pos:].hex(),
                })
                pos += 5
                if pos > len(body):
                    break
            # Actually the 41 pattern seems to be 4-byte units
            # Let me re-examine: 41 0c 50 04 repeats
            # Reset and try 4-byte chunks
            result['filler'] = []
            pos = filler_start
            while pos < len(body) and body[pos] == 0x41:
                end = min(pos + 4, len(body))
                chunk = body[pos:end]
                result['filler'].append({
                    'offset': 16+pos,
                    'hex': chunk.hex(),
                })
                pos += 4

        else:
            # Unknown byte — likely part of a record we didn't parse correctly
            # Try to detect the structure
            print(f"    WARNING: Unknown record type 0x{rec_type:02x} at body offset {pos} (file offset {16+pos})")
            pos += 1

    return result


def parse_type01(rec: bytes) -> dict:
    """Parse a 16-byte type-01 capmerc record.

    Layout hypothesis:
      [0]    = 0x01 (type)
      [1]    = flags/subtype (0x0b common)
      [2]    = field A (register-related?)
      [3]    = field B (0x0a common)
      [4]    = field C (0xf8 or 0xfa — bitmask?)
      [5]    = field D (0x00)
      [6]    = field E (0x04 or 0x05)
      [7]    = 0x00
      [8:10] = 0x0000
      [10]   = field F (varies)
      [11]   = field G (varies)
      [12]   = field H (varies, often 0x00 or register-related)
      [13]   = field I (varies)
      [14:16]= 0x0000
    """
    return {
        'type': rec[0],
        'subtype': rec[1],
        'byte2': rec[2],
        'byte3': rec[3],
        'byte4_mask': rec[4],
        'byte5': rec[5],
        'byte6': rec[6],
        'byte7': rec[7],
        'word8_9': struct.unpack_from('<H', rec, 8)[0],
        'byte10': rec[10],
        'byte11': rec[11],
        'byte12': rec[12],
        'byte13': rec[13],
        'byte14_15': struct.unpack_from('<H', rec, 14)[0],
    }


def parse_type02(rec: bytes) -> dict:
    """Parse a 32-byte type-02 capmerc record.

    Layout hypothesis:
      [0]     = 0x02 (type)
      [1]     = subtype (0x22 or 0x38)
      [2]     = field A
      [3]     = field B
      [4]     = bitmask (0xf8 or 0xfa)
      [5]     = 0x00
      [6]     = field C (0x42, 0x52, 0x62, 0x50)
      [7]     = field D (0x00 or 0x11)
      [8:10]  = 0x0000
      [10]    = field E
      [11]    = field F
      [12]    = field G (0x40 common)
      [13]    = field H (0x00)
      [14]    = field I (0x02)
      [15]    = field J (0x00)
      [16:32] = second half (zeros or continuation data)
    """
    return {
        'type': rec[0],
        'subtype': rec[1],
        'byte2': rec[2],
        'byte3': rec[3],
        'byte4_mask': rec[4],
        'byte5': rec[5],
        'byte6_mode': rec[6],
        'byte7': rec[7],
        'word8_9': struct.unpack_from('<H', rec, 8)[0],
        'byte10': rec[10],
        'byte11': rec[11],
        'byte12': rec[12],
        'byte13': rec[13],
        'byte14': rec[14],
        'byte15': rec[15],
        'second_half': rec[16:].hex(),
        'second_half_nonzero': any(b != 0 for b in rec[16:]),
    }


def extract_elf_sections(path: str) -> dict:
    """Extract named sections from an ELF cubin."""
    with open(path, 'rb') as f:
        data = f.read()

    e_shoff = struct.unpack_from('<Q', data, 40)[0]
    e_shentsize = struct.unpack_from('<H', data, 58)[0]
    e_shnum = struct.unpack_from('<H', data, 60)[0]
    e_shstrndx = struct.unpack_from('<H', data, 62)[0]

    shstr_off = e_shoff + e_shstrndx * e_shentsize
    shstr_offset = struct.unpack_from('<Q', data, shstr_off + 24)[0]
    shstr_size = struct.unpack_from('<Q', data, shstr_off + 32)[0]
    shstrtab = data[shstr_offset:shstr_offset + shstr_size]

    sections = {}
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_name = struct.unpack_from('<I', data, off)[0]
        sh_offset = struct.unpack_from('<Q', data, off + 24)[0]
        sh_size = struct.unpack_from('<Q', data, off + 32)[0]
        if sh_name < len(shstrtab):
            name_end = shstrtab.index(0, sh_name)
            sname = shstrtab[sh_name:name_end].decode('ascii', errors='replace')
        else:
            sname = f'<idx_{sh_name}>'
        if sh_size > 0:
            sections[sname] = data[sh_offset:sh_offset + sh_size]
    return sections


def find_scheduling_groups(instrs: list[SassInstr]) -> list[list[int]]:
    """Identify scheduling groups — contiguous instruction runs between barriers/exits.

    A new group starts after:
    - EXIT instruction
    - BRA instruction
    - Any instruction with wbar=1 (write barrier set)
    """
    groups = []
    current = []
    for i, ins in enumerate(instrs):
        if ins.is_padding:
            continue
        current.append(i)
        if ins.is_exit or ins.is_bra:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def analyze_cubin(path: str):
    """Full analysis of a single cubin."""
    basename = os.path.basename(path)
    sections = extract_elf_sections(path)

    # Find .text and .capmerc sections
    text_key = None
    capmerc_key = None
    for k in sections:
        if k.startswith('.text.'):
            text_key = k
        if 'capmerc' in k and 'text' in k:
            capmerc_key = k

    if not text_key or not capmerc_key:
        print(f"SKIP {basename}: missing .text or .capmerc section")
        return None

    text_data = sections[text_key]
    capmerc_data = sections[capmerc_key]

    print("=" * 100)
    print(f"CUBIN: {basename}")
    print(f"  .text size: {len(text_data)} bytes ({len(text_data)//16} instructions)")
    print(f"  capmerc size: {len(capmerc_data)} bytes")
    print()

    # --- Decode SASS ---
    instrs = decode_sass(text_data)
    real_instrs = [ins for ins in instrs if not ins.is_padding]
    pad_instrs = [ins for ins in instrs if ins.is_padding]

    max_reg_used = max((ins.max_reg for ins in instrs), default=0)

    print(f"  SASS Instructions ({len(real_instrs)} real + {len(pad_instrs)} padding):")
    print(f"  Max register used: R{max_reg_used}")
    print()

    for ins in instrs:
        if ins.is_padding:
            continue
        pred_str = '' if ins.pred == 7 else f'@P{ins.pred} ' if ins.pred < 8 else f'@!P{ins.pred-8} '
        regs = ins.reg_str(ins.dest)
        if ins.op_name not in ('NOP', 'EXIT', 'BRA', 'BAR'):
            regs += f', {ins.reg_str(ins.src0)}, {ins.reg_str(ins.src1)}, {ins.reg_str(ins.src2)}'

        barrier_info = ''
        if ins.wbar:
            barrier_info += ' [WBAR]'
        if ins.stall > 0:
            barrier_info += f' [STALL={ins.stall}]'

        print(f"    [{ins.offset//16:3d}] 0x{ins.offset:04x}: {pred_str}{ins.op_name:14s} {regs:32s}"
              f"  ctrl=0x{ins.ctrl:06x} stall={ins.stall} rbar={ins.rbar:2d} wdep={ins.wdep:2d}{barrier_info}")
    print()

    # --- Parse capmerc ---
    cm = parse_capmerc(capmerc_data)
    hdr = cm['header']

    print(f"  CAPMERC Header (16 bytes):")
    print(f"    Magic:      {hdr['magic_hex']}  (expected: 0c000000010000c0)")
    print(f"    Byte[8]:    {hdr['reg_count']:3d}  (register count)")
    print(f"    Byte[9]:    0x{hdr['field_9']:02x}")
    print(f"    Byte[10]:   0x{hdr['field_10']:02x}")
    print(f"    Byte[11]:   0x{hdr['field_11']:02x}")
    print(f"    Cap mask:   0x{hdr['capability_mask']:08x}  ({hdr['capability_hex']})")
    print()

    # Print body records
    print(f"  CAPMERC Records ({len(cm['records'])} records, {len(cm['filler'])} filler blocks):")
    for idx, rec in enumerate(cm['records']):
        f = rec.fields
        print(f"    Record #{idx} @ offset {rec.offset}: type=0x{rec.rec_type:02x} ({rec.rec_type})")
        hex_str = ' '.join(f'{b:02x}' for b in rec.raw)
        print(f"      Raw: {hex_str}")

        if rec.rec_type == 0x01:
            print(f"      subtype=0x{f['subtype']:02x}  byte2=0x{f['byte2']:02x}  "
                  f"byte3=0x{f['byte3']:02x}  mask=0x{f['byte4_mask']:02x}  "
                  f"byte6=0x{f['byte6']:02x}")
            print(f"      byte10=0x{f['byte10']:02x}  byte11=0x{f['byte11']:02x}  "
                  f"byte12=0x{f['byte12']:02x}  byte13=0x{f['byte13']:02x}")
        elif rec.rec_type == 0x02:
            print(f"      subtype=0x{f['subtype']:02x}  byte2=0x{f['byte2']:02x}  "
                  f"byte3=0x{f['byte3']:02x}  mask=0x{f['byte4_mask']:02x}  "
                  f"mode=0x{f['byte6_mode']:02x}  byte7=0x{f['byte7']:02x}")
            print(f"      byte10=0x{f['byte10']:02x}  byte11=0x{f['byte11']:02x}  "
                  f"byte12=0x{f['byte12']:02x}  byte13=0x{f['byte13']:02x}  "
                  f"byte14=0x{f['byte14']:02x}")
            if f['second_half_nonzero']:
                print(f"      2nd half: {f['second_half']}")
        print()

    if cm['filler']:
        print(f"    Filler blocks ({len(cm['filler'])}):")
        for fb in cm['filler']:
            print(f"      @ offset {fb['offset']}: {fb['hex']}")
        print()

    if cm['trailer']:
        t = cm['trailer']
        print(f"  CAPMERC Trailer: 0x{t['byte0']:02x} 0x{t['byte1']:02x}  ({t['hex']})")
        print(f"    Trailer byte0: 0x{t['byte0']:02x} (0xd0 = standard end marker, 0x50 = alt)")
        print(f"    Trailer byte1: 0x{t['byte1']:02x}")
    print()

    # --- Scheduling groups ---
    groups = find_scheduling_groups(instrs)
    print(f"  Scheduling Groups ({len(groups)}):")
    for gi, grp in enumerate(groups):
        grp_instrs = [instrs[j] for j in grp]
        max_r = max((ins.max_reg for ins in grp_instrs), default=-1)
        regs_used = set()
        for ins in grp_instrs:
            for r in [ins.dest, ins.src0, ins.src1, ins.src2]:
                if r != 255 and r < 128:
                    regs_used.add(r)
        print(f"    Group {gi}: instrs [{grp[0]}-{grp[-1]}] ({len(grp)} instrs)"
              f"  max_reg=R{max_r}  regs={sorted(regs_used)}")
    print()

    return {
        'name': basename,
        'instrs': instrs,
        'real_instrs': real_instrs,
        'capmerc': cm,
        'max_reg': max_reg_used,
        'groups': groups,
    }


def cross_cubin_analysis(results: list[dict]):
    """Compare capmerc patterns across multiple cubins."""
    print("\n" + "=" * 100)
    print("CROSS-CUBIN CORRELATION ANALYSIS")
    print("=" * 100)

    # 1. Header byte[8] vs max register
    print("\n--- Header byte[8] (reg_count) vs actual max register ---")
    for r in results:
        if r is None:
            continue
        hdr = r['capmerc']['header']
        print(f"  {r['name']:45s}  capmerc_reg={hdr['reg_count']:3d}  "
              f"actual_max=R{r['max_reg']:3d}  #real_instrs={len(r['real_instrs']):3d}  "
              f"#records={len(r['capmerc']['records']):3d}  "
              f"capmerc_size={r['capmerc']['size']:4d}")

    # 2. Record count vs instruction count
    print("\n--- Record count vs instruction metrics ---")
    for r in results:
        if r is None:
            continue
        n_recs = len(r['capmerc']['records'])
        n_real = len(r['real_instrs'])
        n_groups = len(r['groups'])
        n_type01 = sum(1 for rec in r['capmerc']['records'] if rec.rec_type == 1)
        n_type02 = sum(1 for rec in r['capmerc']['records'] if rec.rec_type == 2)
        n_filler = len(r['capmerc']['filler'])
        print(f"  {r['name']:45s}  records={n_recs:2d} (type01={n_type01} type02={n_type02}) "
              f"filler={n_filler}  real_instrs={n_real:2d}  groups={n_groups:2d}")

    # 3. Capability mask analysis
    print("\n--- Capability mask (bytes 12-15) ---")
    for r in results:
        if r is None:
            continue
        hdr = r['capmerc']['header']
        cap = hdr['capability_mask']
        cap_hex = hdr['capability_hex']
        print(f"  {r['name']:45s}  cap=0x{cap:08x} ({cap_hex})  "
              f"bits={cap:032b}")

    # 4. Trailer analysis
    print("\n--- Trailer bytes ---")
    for r in results:
        if r is None:
            continue
        t = r['capmerc']['trailer']
        if t:
            print(f"  {r['name']:45s}  trailer=0x{t['byte0']:02x} 0x{t['byte1']:02x}  "
                  f"max_reg=R{r['max_reg']}")

    # 5. Type-02 record byte[6] (mode byte) patterns
    print("\n--- Type-02 record mode byte (byte[6]) ---")
    for r in results:
        if r is None:
            continue
        for rec in r['capmerc']['records']:
            if rec.rec_type == 2:
                f = rec.fields
                print(f"  {r['name']:35s}  rec@{rec.offset:3d}: "
                      f"sub=0x{f['subtype']:02x} b2=0x{f['byte2']:02x} "
                      f"b3=0x{f['byte3']:02x} mode=0x{f['byte6_mode']:02x} "
                      f"b7=0x{f['byte7']:02x} b10=0x{f['byte10']:02x} "
                      f"b11=0x{f['byte11']:02x} b12=0x{f['byte12']:02x}")

    # 6. Type-01 vs Type-02 byte-by-byte comparison
    print("\n--- Type-01 record field patterns ---")
    for r in results:
        if r is None:
            continue
        for rec in r['capmerc']['records']:
            if rec.rec_type == 1:
                f = rec.fields
                print(f"  {r['name']:35s}  rec@{rec.offset:3d}: "
                      f"sub=0x{f['subtype']:02x} b2=0x{f['byte2']:02x} "
                      f"b3=0x{f['byte3']:02x} mask=0x{f['byte4_mask']:02x} "
                      f"b6=0x{f['byte6']:02x} b10=0x{f['byte10']:02x} "
                      f"b11=0x{f['byte11']:02x} b12=0x{f['byte12']:02x} "
                      f"b13=0x{f['byte13']:02x}")

    # 7. Deep hypothesis: do type-02 records correspond to scheduling groups?
    print("\n--- HYPOTHESIS: type-02 records per scheduling group ---")
    for r in results:
        if r is None:
            continue
        n_type02 = sum(1 for rec in r['capmerc']['records'] if rec.rec_type == 2)
        n_groups = len(r['groups'])
        match = "MATCH" if n_type02 == n_groups else f"MISMATCH ({n_type02} vs {n_groups})"
        print(f"  {r['name']:45s}  type02={n_type02}  groups={n_groups}  {match}")

    # 8. Try mapping: type-01 records encode per-instruction metadata?
    print("\n--- HYPOTHESIS: type-01 records per real instruction ---")
    for r in results:
        if r is None:
            continue
        n_type01 = sum(1 for rec in r['capmerc']['records'] if rec.rec_type == 1)
        n_real = len(r['real_instrs'])
        print(f"  {r['name']:45s}  type01={n_type01}  real_instrs={n_real}")

    # 9. Byte[2] of type-01 records — does it encode max register per scheduling group?
    print("\n--- Type-01 byte[2] — register index hypothesis ---")
    for r in results:
        if r is None:
            continue
        t01_b2 = [rec.fields['byte2'] for rec in r['capmerc']['records'] if rec.rec_type == 1]
        if t01_b2:
            print(f"  {r['name']:45s}  type01 byte[2] values: {[f'0x{v:02x}' for v in t01_b2]}  "
                  f"max_reg=R{r['max_reg']}")

    # 10. Type-02 byte[10:11] bitmap hypothesis
    print("\n--- Type-02 bytes[10:11] — register liveness bitmap hypothesis ---")
    for r in results:
        if r is None:
            continue
        for idx, rec in enumerate(r['capmerc']['records']):
            if rec.rec_type == 2:
                b10 = rec.fields['byte10']
                b11 = rec.fields['byte11']
                bitmap = (b11 << 8) | b10
                active_regs = [i for i in range(16) if bitmap & (1 << i)]
                print(f"  {r['name']:35s}  type02 #{idx}: "
                      f"bytes[10:11]=0x{b10:02x}{b11:02x}  "
                      f"bitmap={bitmap:016b}  active_regs={active_regs}")


def main():
    probe_dir = os.path.dirname(os.path.abspath(__file__))

    # Selected cubins: varying capmerc sizes and instruction counts
    cubins = [
        'ldg64_min_ptxas.cubin',     # 146 bytes capmerc, 16 instrs (simplest ptxas)
        'ptxas_0x90_r8.cubin',       # 146 bytes, 24 instrs
        'bra_test_ptxas.cubin',      # 162 bytes, 24 instrs (has branch)
        'force_highreg.cubin',       # 166 bytes, 24 instrs (high regs)
        'decomposed.cubin',          # 182 bytes, 24 instrs
        '_var_shl_gt.cubin',         # 178 bytes, 24 instrs
    ]

    # Also check for larger cubins
    for extra in ['opencuda_vecadd_test.cubin', '_bsearch_tmp2.cubin',
                  'ldg64_gap0_warm1.cubin', 'mma_ptx.cubin',
                  'probe_k32.cubin', 'imad_gt_ptxas.cubin',
                  'idx_test_ptxas.cubin', 'r14test.cubin']:
        cubins.append(extra)

    results = []
    for name in cubins:
        path = os.path.join(probe_dir, name)
        if os.path.exists(path):
            results.append(analyze_cubin(path))
        else:
            print(f"SKIP: {name} not found")

    valid_results = [r for r in results if r is not None]
    if valid_results:
        cross_cubin_analysis(valid_results)

    # Final summary
    print("\n" + "=" * 100)
    print("SUMMARY OF FINDINGS")
    print("=" * 100)
    print("""
Key observations to investigate:
1. Header byte[8] = GPR allocation count (confirmed from emitter.py)
2. Header bytes[12:15] = capability bitmask (varies per kernel)
3. Body records: type-01 (16B) and type-02 (32B)
4. Type-01 records may encode per-instruction or per-scheduling-group register metadata
5. Type-02 records may encode scheduling group boundaries with register liveness
6. 0x41-prefixed filler blocks pad high-register kernels
7. Trailer byte 0xd0 vs 0x50 may indicate register range (low vs high)
8. Trailer byte[1] correlates with kernel complexity
""")


if __name__ == '__main__':
    main()
