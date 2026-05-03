"""OURS vs PTXAS compilation and metric extraction."""
from __future__ import annotations

import struct
from pathlib import Path

from benchmarks.bench_util import compile_openptxas, compile_ptxas
from sass.scoreboard import _get_opcode


def _extract_text(cubin: bytes):
    """Extract .text.* section(s): list of (name, offset, size)."""
    e_shoff = struct.unpack_from('<Q', cubin, 40)[0]
    e_shnum = struct.unpack_from('<H', cubin, 60)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 62)[0]
    stoff = struct.unpack_from('<Q', cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    sections = []
    for i in range(e_shnum):
        base = e_shoff + i * 64
        nm = struct.unpack_from('<I', cubin, base)[0]
        ne = cubin.index(0, stoff + nm)
        name = cubin[stoff + nm:ne].decode()
        if name.startswith('.text.'):
            off = struct.unpack_from('<Q', cubin, base + 24)[0]
            sz = struct.unpack_from('<Q', cubin, base + 32)[0]
            sections.append((name, off, sz))
    return sections


def _extract_capmerc_gprs(cubin: bytes) -> int:
    """Read num_gprs from .nv.capmerc.text.* section byte[8]."""
    e_shoff = struct.unpack_from('<Q', cubin, 40)[0]
    e_shnum = struct.unpack_from('<H', cubin, 60)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 62)[0]
    stoff = struct.unpack_from('<Q', cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    for i in range(e_shnum):
        base = e_shoff + i * 64
        nm = struct.unpack_from('<I', cubin, base)[0]
        ne = cubin.index(0, stoff + nm)
        name = cubin[stoff + nm:ne].decode()
        if name.startswith('.nv.capmerc.text.'):
            off = struct.unpack_from('<Q', cubin, base + 24)[0]
            return cubin[off + 8]
    return -1


def disassemble(cubin: bytes) -> list[dict]:
    """Disassemble the first .text section into instruction records."""
    secs = _extract_text(cubin)
    if not secs:
        return []
    _, off, sz = secs[0]
    instrs = []
    for j in range(0, sz, 16):
        raw = cubin[off + j:off + j + 16]
        opc = _get_opcode(raw)
        instrs.append({
            'idx': j // 16,
            'opcode': opc,
            'raw': raw,
            'is_nop': opc == 0x918,
        })
    return instrs


# Opcode name table (partial, covers the common SM_120 opcodes)
_OPC_NAME = {
    0xb82: 'S2R', 0x919: 'S2UR', 0x7ac: 'LDCU.64', 0x9c3: 'S2UR.9c3',
    0x20c: 'ISETP', 0xc0c: 'ISETP.RUR', 0x80c: 'ISETP.IMM',
    0x94d: 'EXIT', 0x947: 'BRA', 0x918: 'NOP',
    0x824: 'IMAD.Ri', 0x825: 'IMAD.WIDE', 0x224: 'IMAD.RR',
    0x424: 'IMAD.424', 0x2a4: 'IMAD.RR2', 0xc24: 'IMAD.RUR',
    0x810: 'IADD3.IMM', 0x812: 'LOP3.IMM', 0x210: 'IADD3',
    0x212: 'LOP3', 0x986: 'STG', 0xc35: 'IADD64.RUR',
    0x235: 'IADD64', 0xc11: 'LEA.RUR', 0x211: 'LEA',
    0x811: 'LEA.IMM', 0x431: 'HFMA2', 0x802: 'ULDC',
    0x981: 'LDG', 0x221: 'FADD', 0x223: 'FFMA', 0x820: 'FMUL.IMM',
    0x819: 'SHF', 0x219: 'SHF.VAR', 0x308: 'MUFU',
    0x202: 'MOV', 0x207: 'SEL', 0x248: 'VIMNMX',
    0x309: 'POPC', 0x301: 'BREV', 0x300: 'FLO',
    0x226: 'IDP4A', 0x227: 'IMAD.HI',
    0x23c: 'HMMA', 0x237: 'IMMA', 0x23f: 'DMMA', 0x27a: 'QMMA', 0x47f: 'OMMA',
    0x806: 'VOTE', 0x589: 'SHFL', 0xf89: 'SHFL.ri', 0x989: 'SHFL.ii',
    0x416: 'PRMT', 0x209: 'FMNMX', 0x20b: 'FSETP',
    0x624: 'IMAD.MOV', 0xa24: 'IMAD.cb', 0xa10: 'IADD3.cb',
    0xa12: 'LOP3.cb', 0xa19: 'SHF.89',
    0x306: 'I2FP', 0x305: 'F2I', 0x310: 'F2F', 0x245: 'I2FP.32',
    0x225: 'IMAD.WIDE.RR', 0x823: 'FFMA.IMM',
    0x80a: 'FSEL.STEP', 0x808: 'FSEL.IMM',
    0x299: 'SHF.VAR2', 0x81a: 'BFE',
    0x213: 'IABS', 0x848: 'VIMNMX.i',
    0xc02: 'MOV.UR', 0x203: 'P2R', 0x204: 'R2P',
}


def opcode_name(opc: int) -> str:
    return _OPC_NAME.get(opc, f'0x{opc:03x}')


def compile_both(ptx: str) -> dict:
    """Compile PTX with both OURS and PTXAS, return metrics."""
    cubin_o, ms_o = compile_openptxas(ptx)
    cubin_p, ms_p = compile_ptxas(ptx)

    def metrics(cubin, compile_ms):
        instrs = disassemble(cubin)
        total = len(instrs)
        non_nop = sum(1 for i in instrs if not i['is_nop'])
        nops = total - non_nop
        regs = _extract_capmerc_gprs(cubin)
        ops = [i for i in instrs if not i['is_nop']]
        return {
            'cubin': cubin,
            'instrs': instrs,
            'ops': ops,
            'total': total,
            'non_nop': non_nop,
            'nops': nops,
            'regs': regs,
            'compile_ms': compile_ms * 1000,
        }

    return {
        'ours': metrics(cubin_o, ms_o),
        'ptxas': metrics(cubin_p, ms_p),
    }
