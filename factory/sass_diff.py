"""SASS byte-diff + fingerprint for clustering OpenPTXas bugs.

For each (ours_cubin, theirs_cubin) pair, extract the .text.fuzz section
bytes, align the two 16-byte instruction streams, find the first
divergent instruction pair, and classify it into a "fingerprint" that
groups all dossiers with the same underlying bug shape.

Fingerprint kinds:
  CTRL_DIFF   — bytes 0..12 identical, ctrl bytes 13..15 differ
                  => scoreboard / stall / rbar issue
  OPCODE_DIFF — bytes 0..1 differ
                  => wrong opcode selected (isel bug)
  OPERAND_DIFF — bytes 0..1 identical, bytes 2..12 differ
                  => right opcode, wrong operands (encoder / regalloc)
  IMM_DIFF     — opcode matches AND suggests immediate-field mismatch
                  (bytes 4..7 differ while 2..3 and 8..12 match)
  STRUCTURAL   — instruction counts differ or offsets don't align
                  => different isel path (e.g. LDCU+IMAD vs IMAD.IMM)
"""
from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass(frozen=True)
class InstrDiff:
    """Summary of the first differing instruction pair."""
    kind: str                 # CTRL_DIFF | OPCODE_DIFF | OPERAND_DIFF | IMM_DIFF | STRUCTURAL
    offset: int               # virtual offset into .text section (multiple of 16)
    ours_bytes: bytes         # 16 bytes
    theirs_bytes: bytes       # 16 bytes
    ours_opcode: int          # 12-bit opcode
    theirs_opcode: int        # 12-bit opcode
    note: str = ''

    def fingerprint(self) -> str:
        """Stable short key for clustering."""
        if self.kind == 'STRUCTURAL':
            return f'{self.kind}:our={self.ours_opcode:03x}/theirs={self.theirs_opcode:03x}'
        if self.kind == 'OPCODE_DIFF':
            return f'OPCODE:{self.ours_opcode:03x}->{self.theirs_opcode:03x}'
        if self.kind == 'CTRL_DIFF':
            # Which ctrl fields differ?
            fields = _ctrl_field_diff(self.ours_bytes[13:16], self.theirs_bytes[13:16])
            return f'CTRL[{self.ours_opcode:03x}]:{fields}'
        if self.kind == 'IMM_DIFF':
            return f'IMM[{self.ours_opcode:03x}]:ours_zero={int(self.ours_bytes[4:8] == b"\\x00\\x00\\x00\\x00")}'
        if self.kind == 'OPERAND_DIFF':
            # Which byte indices differ?
            diff_idx = [i for i in range(2, 13)
                        if self.ours_bytes[i] != self.theirs_bytes[i]]
            return f'OPERAND[{self.ours_opcode:03x}]:b{"_".join(str(i) for i in diff_idx)}'
        return f'{self.kind}:{self.ours_opcode:03x}'


def _opcode(b: bytes) -> int:
    """12-bit SM_120 opcode from bytes 0..1."""
    return b[0] | ((b[1] & 0x0F) << 8)


def _decode_ctrl(raw3: bytes) -> dict:
    v = raw3[0] | (raw3[1] << 8) | (raw3[2] << 16)
    ctrl = v >> 1
    return dict(
        stall=(ctrl >> 17) & 0x3F,
        yield_=(ctrl >> 16) & 0x1,
        wbar=(ctrl >> 15) & 0x1,
        rbar=(ctrl >> 10) & 0x1F,
        wdep=(ctrl >> 4) & 0x3F,
        misc=ctrl & 0xF,
    )


def _ctrl_field_diff(o: bytes, t: bytes) -> str:
    od = _decode_ctrl(o); td = _decode_ctrl(t)
    diffs = sorted(k for k in od if od[k] != td[k])
    return '_'.join(diffs) or 'unknown'


def extract_text_fuzz(cubin: bytes) -> Optional[bytes]:
    """Return the raw bytes of .text.fuzz, or None if not found."""
    if len(cubin) < 64 or cubin[:4] != b'\x7fELF':
        return None
    e_shoff = struct.unpack_from('<Q', cubin, 0x28)[0]
    e_shentsize = struct.unpack_from('<H', cubin, 0x3a)[0]
    e_shnum = struct.unpack_from('<H', cubin, 0x3c)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 0x3e)[0]
    sh_off = e_shoff + e_shstrndx * e_shentsize
    shstr_off = struct.unpack_from('<Q', cubin, sh_off + 0x18)[0]
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        name_off = struct.unpack_from('<I', cubin, off)[0]
        name_addr = shstr_off + name_off
        nend = cubin.find(b'\x00', name_addr)
        name = cubin[name_addr:nend].decode('ascii', errors='replace')
        if name == '.text.fuzz':
            s_off = struct.unpack_from('<Q', cubin, off + 0x18)[0]
            s_size = struct.unpack_from('<Q', cubin, off + 0x20)[0]
            return cubin[s_off:s_off+s_size]
    return None


def _find_ldg_offset(text: bytes) -> Optional[int]:
    """Byte offset of the first LDG (opcode 0x981) — marks end of prologue."""
    for v in range(0, len(text), 16):
        if _opcode(text[v:v+2]) == 0x981:
            return v
    return None


def first_diff(ours_cubin: bytes, theirs_cubin: bytes,
                skip_regalloc: bool = True) -> Optional[InstrDiff]:
    """Find the first *semantic* divergent instruction pair IN THE BODY.

    Skips the param-loading prologue by starting the comparison from the
    first LDG instruction (opcode 0x981) in each cubin — the LDG is the
    SASS image of `ld.global.u32 %r3, [%rd2]` which begins the fuzz body
    in every generator-emitted kernel.  This way, prologue-level
    divergences in register allocation / param loading don't swamp the
    real body-level divergence we care about.

    When skip_regalloc=True, same-opcode pairs with only register-number
    or immediate differences (no opcode or ctrl change) are skipped —
    those are cosmetic; the true bug is usually a downstream opcode
    mismatch or ctrl-word mismatch.
    """
    ours_text = extract_text_fuzz(ours_cubin)
    theirs_text = extract_text_fuzz(theirs_cubin)
    if ours_text is None or theirs_text is None:
        return None

    # Lock to the LDG as a common anchor — prologues are free to be different
    o_start = _find_ldg_offset(ours_text) or 0
    t_start = _find_ldg_offset(theirs_text) or 0

    # Walk from each side's LDG.  If one is longer in body bytes we'll
    # eventually STRUCTURAL-out on length mismatch.
    o_body = ours_text[o_start:]
    t_body = theirs_text[t_start:]

    min_n = min(len(o_body), len(t_body)) // 16
    for idx in range(min_n):
        v = idx * 16
        o = o_body[v:v+16]
        t = t_body[v:v+16]
        if o == t:
            continue

        oop = _opcode(o)
        top = _opcode(t)

        if oop != top:
            kind = 'OPCODE_DIFF'
        elif o[:13] == t[:13] and o[13:16] != t[13:16]:
            kind = 'CTRL_DIFF'
        elif o[2:4] == t[2:4] and o[8:13] == t[8:13] and o[4:8] != t[4:8]:
            kind = 'IMM_DIFF'
        else:
            kind = 'OPERAND_DIFF'

        # Skip non-structural differences when looking for the real
        # semantic divergence.  OPERAND_DIFF alone can be pure regalloc
        # renaming — not a bug in itself.  IMM_DIFF may or may not be
        # semantic (e.g. different target registers vs different
        # constants); keep if no downstream OPCODE_DIFF appears.
        if skip_regalloc and kind in ('OPERAND_DIFF', 'IMM_DIFF'):
            continue

        return InstrDiff(kind=kind, offset=v, ours_bytes=o, theirs_bytes=t,
                         ours_opcode=oop, theirs_opcode=top)

    # One body longer than the other
    if len(o_body) != len(t_body):
        v = min_n * 16
        ours_tail = o_body[v:v+16] if v < len(o_body) else b'\0' * 16
        theirs_tail = t_body[v:v+16] if v < len(t_body) else b'\0' * 16
        return InstrDiff(kind='STRUCTURAL', offset=v,
                         ours_bytes=ours_tail, theirs_bytes=theirs_tail,
                         ours_opcode=_opcode(ours_tail) if ours_tail != b'\0'*16 else 0,
                         theirs_opcode=_opcode(theirs_tail) if theirs_tail != b'\0'*16 else 0,
                         note=f'len ours={len(o_body)} vs theirs={len(t_body)}')
    return None  # identical


# Human-readable opcode-to-name map (SM_120 subset we care about)
OPCODE_NAMES = {
    0x210: 'IADD3.RR', 0x212: 'IADD3X', 0x810: 'IADD3.IMM',
    0x819: 'LEA.IMM',  0x848: 'IMNMX.IMM',
    0x224: 'IMAD.R',   0x2a4: 'IMAD.RR', 0x824: 'IMAD.IMM',
    0x225: 'IMAD.WIDE.RR', 0x825: 'IMAD.WIDE.IMM',
    0xc24: 'IMAD.R-UR',
    0xc35: 'IADD.64-UR', 0x235: 'IADD.64',
    0x248: 'VIMNMX.RR',
    0x919: 'S2R', 0x9c3: 'S2UR',
    0xb82: 'LDC', 0x7ac: 'LDCU',
    0x981: 'LDG', 0x986: 'STG.E',
    0x94d: 'EXIT', 0x947: 'BRA',
    0x918: 'NOP',
    0xc0c: 'ISETP.UR', 0x20c: 'ISETP.RR', 0x80c: 'ISETP.IMM',
    0x300: 'FLO', 0x309: 'POPC', 0x310: 'BREV', 0x812: 'LOP3.IMM',
    0x806: 'VOTE',
    0x816: 'PRMT', 0xc11: 'IADD3.R-UR',
}


def pretty_op(op: int) -> str:
    return OPCODE_NAMES.get(op, f'op_{op:03x}')
