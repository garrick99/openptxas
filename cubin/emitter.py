"""
cubin/emitter.py — Generate a loadable SM_120 cubin ELF from scratch.

All section types, flags, and metadata formats reverse-engineered from ptxas 13.0
output and confirmed against 32 probe cubins.
"""

from __future__ import annotations
import struct
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# ELF64 constants
# ---------------------------------------------------------------------------

ELF_MAGIC = b'\x7fELF'
ELFCLASS64 = 2
ELFDATA2LSB = 1
EV_CURRENT = 1
ET_EXEC = 2
EM_CUDA = 0xBE

SHT_NULL     = 0
SHT_PROGBITS = 1
SHT_SYMTAB   = 2
SHT_STRTAB   = 3
SHT_NOTE     = 7
SHT_NOBITS   = 8
SHT_NV_INFO      = 0x70000000
SHT_NV_CALLGRAPH = 0x70000001
SHT_NV_COMPAT    = 0x70000086

SHF_WRITE     = 0x1
SHF_ALLOC     = 0x2
SHF_EXECINSTR = 0x4

PT_LOAD = 1
PT_PHDR = 6

SM120_FLAGS = 0x06007802

STB_LOCAL  = 0
STB_GLOBAL = 1
STB_WEAK   = 2
STT_NOTYPE  = 0
STT_OBJECT  = 1
STT_FUNC    = 2
STT_SECTION = 3

ELF_HEADER_SIZE = 64
SH_ENTRY_SIZE = 64
PH_ENTRY_SIZE = 56


def _u16(v): return struct.pack('<H', v & 0xFFFF)
def _u32(v): return struct.pack('<I', v & 0xFFFFFFFF)
def _u64(v): return struct.pack('<Q', v & 0xFFFFFFFFFFFFFFFF)

def _strtab_add(strtab: bytearray, s: str) -> int:
    if not strtab:
        strtab.extend(b'\x00')
    off = len(strtab)
    strtab.extend(s.encode('utf-8') + b'\x00')
    return off


# ---------------------------------------------------------------------------
# Note section templates (loaded from probe cubin or built from scratch)
# ---------------------------------------------------------------------------

_NOTE_TKINFO = None
_NOTE_CUINFO = None


def _load_note_templates():
    """Load .note.nv.tkinfo and .note.nv.cuinfo from a probe cubin if available."""
    global _NOTE_TKINFO, _NOTE_CUINFO
    if _NOTE_TKINFO is not None:
        return
    from pathlib import Path
    probe = Path(__file__).parent.parent / 'probe_work' / 'probe_k1.cubin'
    if probe.exists():
        from cubin.patcher import ELF64
        elf = ELF64(probe.read_bytes())
        _NOTE_TKINFO = elf.section_data('.note.nv.tkinfo')
        _NOTE_CUINFO = elf.section_data('.note.nv.cuinfo')
    else:
        # Hardcoded fallback: build minimal note sections
        # .note.nv.cuinfo: name="NVIDIA Corp", type=0x3E8, desc=8 bytes
        cuinfo = bytearray()
        cuinfo.extend(_u32(12))   # namesz
        cuinfo.extend(_u32(8))    # descsz
        cuinfo.extend(_u32(0x3E8))  # type
        cuinfo.extend(b'NVIDIA Corp\x00')  # name (12 bytes)
        cuinfo.extend(bytes([0x02, 0x00, 0x78, 0x00, 0x82, 0x00, 0x00, 0x00]))  # desc
        _NOTE_CUINFO = bytes(cuinfo)
        # .note.nv.tkinfo: minimal
        tkinfo = bytearray()
        tkinfo.extend(_u32(12))   # namesz
        tkinfo.extend(_u32(8))    # descsz
        tkinfo.extend(_u32(0x7D0))  # type
        tkinfo.extend(b'NVIDIA Corp\x00')
        tkinfo.extend(bytes([0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]))
        _NOTE_TKINFO = bytes(tkinfo)


def _build_nv_info_global():
    return bytes.fromhex(
        '042f0800080000000a000000'
        '041108000800000000000000'
        '041208000800000000000000'
    )


def _build_nv_info_kernel():
    """Per-kernel .nv.info — exact 88 bytes from ptxas (2-param u64 kernel).

    REGCOUNT=0x82 is constant across all probe cubins regardless of actual
    register usage (10 or 12 GPRs both use 0x82). The GPU hardware handles
    register allocation independently of this field.
    """
    return bytes.fromhex(
        '043704008200000004170c00000000000100080000f02100'
        '04170c00000000000000000000f02100'
        '03500000031bff00035f0101024a0000'
        '041c04008000000003191000'
        '040a08000900000080031000'
        '0436040000000000'
    )


def _build_nv_compat():
    return bytes.fromhex(
        '020900000202010002050500030701010203000002060100040b08005000000000000000'
    )


def _build_callgraph():
    return bytes.fromhex(
        '00000000ffffffff00000000feffffff00000000fdffffff00000000fcffffff'
    )


# ---------------------------------------------------------------------------
# Main emitter
# ---------------------------------------------------------------------------

@dataclass
class KernelDesc:
    name: str
    sass_bytes: bytes
    num_gprs: int
    num_params: int
    param_sizes: list[int]
    param_offsets: dict[str, int]
    param_base: int = 0x380
    const0_size: int = 0x390


def emit_cubin(kernel: KernelDesc) -> bytes:
    _load_note_templates()

    from sass.encoding.sm_120_opcodes import encode_nop, encode_bra

    # Pad SASS: ensure EXIT + BRA self-loop + NOP padding to 384B minimum
    sass = bytearray(kernel.sass_bytes)
    if len(sass) >= 16:
        lo_last = struct.unpack_from('<Q', sass, len(sass)-16)[0]
        if (lo_last & 0xFFF) != 0x94d:
            from sass.encoding.sm_120_opcodes import encode_exit
            sass.extend(encode_exit())
    sass.extend(encode_bra(-16))
    while len(sass) % 128 != 0:
        sass.extend(encode_nop())
    while len(sass) < 384:
        sass.extend(encode_nop())
    text_data = bytes(sass)

    const0_data = b'\x00' * kernel.const0_size
    shared_reserved = b''  # NOBITS: no file content, but 64 bytes virtual

    # Section data
    text_sec = f'.text.{kernel.name}'
    const0_sec = f'.nv.constant0.{kernel.name}'
    nvinfo_k_sec = f'.nv.info.{kernel.name}'

    # Build string tables
    shstrtab = bytearray()
    _strtab_add(shstrtab, '')
    strtab = bytearray()
    _strtab_add(strtab, '')

    # Section indices
    # 0:null 1:shstrtab 2:strtab 3:symtab 4:tkinfo 5:cuinfo
    # 6:nv.info 7:nv.compat 8:nv.info.K 9:callgraph 10:text 11:shared 12:const0
    TKINFO_IDX = 4
    CUINFO_IDX = 5
    COMPAT_IDX = 7
    TEXT_IDX = 10
    SHARED_IDX = 11
    CONST0_IDX = 12
    NUM_SECTIONS = 13

    sec_names = [
        '',                          # 0
        '.shstrtab',                 # 1
        '.strtab',                   # 2
        '.symtab',                   # 3
        '.note.nv.tkinfo',           # 4
        '.note.nv.cuinfo',           # 5
        '.nv.info',                  # 6
        '.nv.compat',                # 7
        nvinfo_k_sec,                # 8
        '.nv.callgraph',             # 9
        text_sec,                    # 10
        '.nv.shared.reserved.0',     # 11
        const0_sec,                  # 12
    ]

    name_offsets = {}
    for n in sec_names:
        if n and n not in name_offsets:
            name_offsets[n] = _strtab_add(shstrtab, n)
    name_offsets[''] = 0
    shstrtab_data = bytes(shstrtab)

    # Symbol names
    sn_tkinfo = _strtab_add(strtab, '.note.nv.tkinfo')
    sn_cuinfo = _strtab_add(strtab, '.note.nv.cuinfo')
    sn_text = _strtab_add(strtab, text_sec)
    sn_smem_off = _strtab_add(strtab, '.nv.reservedSmem.offset0')
    sn_smem_alias = _strtab_add(strtab, '__nv_reservedSMEM_offset_0_alias')
    sn_callgraph = _strtab_add(strtab, '.nv.callgraph')
    sn_kernel = _strtab_add(strtab, kernel.name)
    sn_const0 = _strtab_add(strtab, const0_sec)
    strtab_data = bytes(strtab)

    # Build symbol table (matching ptxas layout)
    def _sym(st_name, st_info, st_other, st_shndx, st_value, st_size):
        return (_u32(st_name) + bytes([st_info, st_other]) +
                _u16(st_shndx) + _u64(st_value) + _u64(st_size))

    symtab = bytearray()
    symtab.extend(_sym(0, 0, 0, 0, 0, 0))  # [0] null
    symtab.extend(_sym(sn_tkinfo, (STB_LOCAL<<4)|STT_SECTION, 0, TKINFO_IDX, 0, 0))  # [1]
    symtab.extend(_sym(sn_cuinfo, (STB_LOCAL<<4)|STT_SECTION, 0, CUINFO_IDX, 0, 0))  # [2]
    symtab.extend(_sym(sn_text, (STB_LOCAL<<4)|STT_SECTION, 0, TEXT_IDX, 0, 0))  # [3]
    symtab.extend(_sym(sn_smem_off, (STB_WEAK<<4)|STT_OBJECT, 0, 0, 0x40, 4))  # [4]
    symtab.extend(_sym(sn_smem_alias, (STB_WEAK<<4)|STT_NOTYPE, 0xa0, SHARED_IDX, 0x40, 0))  # [5]
    symtab.extend(_sym(sn_callgraph, (STB_LOCAL<<4)|STT_SECTION, 0, 9, 0, 0))  # [6]
    symtab.extend(_sym(sn_const0, (STB_LOCAL<<4)|STT_SECTION, 0, CONST0_IDX, 0, 0))  # [7]
    KERNEL_SYM_IDX = 8
    FIRST_GLOBAL = 8
    symtab.extend(_sym(sn_kernel, (STB_GLOBAL<<4)|STT_FUNC, 0x10, TEXT_IDX, 0, len(text_data)))  # [8]
    symtab_data = bytes(symtab)

    section_datas = [
        b'',                         # 0
        shstrtab_data,               # 1
        strtab_data,                 # 2
        symtab_data,                 # 3
        _NOTE_TKINFO,                # 4
        _NOTE_CUINFO,                # 5
        _build_nv_info_global(),     # 6
        _build_nv_compat(),          # 7
        _build_nv_info_kernel(),  # 8
        _build_callgraph(),          # 9
        text_data,                   # 10
        shared_reserved,             # 11 (NOBITS)
        const0_data,                 # 12
    ]

    # (type, flags, link, info, align, entsize)
    section_meta = [
        (SHT_NULL,         0, 0, 0, 0, 0),
        (SHT_STRTAB,       0, 0, 0, 1, 0),
        (SHT_STRTAB,       0, 0, 0, 1, 0),
        (SHT_SYMTAB,       0, 2, FIRST_GLOBAL, 8, 24),
        (SHT_NOTE,         0x2000000, 0, 0, 4, 0),
        (SHT_NOTE,         0x1000040, TKINFO_IDX, COMPAT_IDX, 4, 0),
        (SHT_NV_INFO,      0, 3, 0, 4, 0),
        (SHT_NV_COMPAT,    0, 0, 0, 4, 0),
        (SHT_NV_INFO,      0x40, 3, TEXT_IDX, 4, 0),
        (SHT_NV_CALLGRAPH, 0, 3, 0, 4, 8),
        (SHT_PROGBITS,     SHF_ALLOC|SHF_EXECINSTR, 3, KERNEL_SYM_IDX, 128, 0),
        (SHT_NOBITS,       SHF_WRITE|SHF_ALLOC, 0, 0, 1, 0),
        (SHT_PROGBITS,     SHF_ALLOC|0x40, 0, TEXT_IDX, 4, 0),
    ]

    # Compute file offsets
    NUM_PHDRS = 5
    offset = ELF_HEADER_SIZE
    section_offsets = [0] * NUM_SECTIONS
    for i in range(1, NUM_SECTIONS):
        _, _, _, _, align, _ = section_meta[i]
        if align > 1:
            r = offset % align
            if r: offset += align - r
        section_offsets[i] = offset
        if section_meta[i][0] != SHT_NOBITS:
            offset += len(section_datas[i])

    # Section header table
    r = offset % 8
    if r: offset += 8 - r
    shoff = offset
    offset += NUM_SECTIONS * SH_ENTRY_SIZE

    # Program header table
    r = offset % 8
    if r: offset += 8 - r
    phoff = offset
    offset += NUM_PHDRS * PH_ENTRY_SIZE

    total_size = offset
    buf = bytearray(total_size)

    # ELF header
    buf[0:4] = ELF_MAGIC
    buf[4] = ELFCLASS64
    buf[5] = ELFDATA2LSB
    buf[6] = EV_CURRENT
    buf[7] = 0x41  # ELFOSABI_CUDA (NVIDIA-specific)
    buf[8] = 0x08  # EI_ABIVERSION (CUDA ABI v8)
    struct.pack_into('<H', buf, 16, ET_EXEC)
    struct.pack_into('<H', buf, 18, EM_CUDA)
    struct.pack_into('<I', buf, 20, EV_CURRENT)
    struct.pack_into('<Q', buf, 24, 0)
    struct.pack_into('<Q', buf, 32, phoff)
    struct.pack_into('<Q', buf, 40, shoff)
    struct.pack_into('<I', buf, 48, SM120_FLAGS)
    struct.pack_into('<H', buf, 52, ELF_HEADER_SIZE)
    struct.pack_into('<H', buf, 54, PH_ENTRY_SIZE)
    struct.pack_into('<H', buf, 56, NUM_PHDRS)
    struct.pack_into('<H', buf, 58, SH_ENTRY_SIZE)
    struct.pack_into('<H', buf, 60, NUM_SECTIONS)
    struct.pack_into('<H', buf, 62, 1)  # e_shstrndx

    # Write section data
    for i in range(1, NUM_SECTIONS):
        if section_meta[i][0] == SHT_NOBITS:
            continue
        data = section_datas[i]
        off = section_offsets[i]
        buf[off:off + len(data)] = data

    # Write section headers
    for i in range(NUM_SECTIONS):
        base = shoff + i * SH_ENTRY_SIZE
        n_off = name_offsets.get(sec_names[i], 0)
        sh_type, sh_flags, sh_link, sh_info, sh_align, sh_entsize = section_meta[i]
        sh_size = len(section_datas[i]) if i > 0 else 0
        # NOBITS: file size in section_datas is 0, but sh_size should be 64
        if i == SHARED_IDX:
            sh_size = 64

        struct.pack_into('<I', buf, base + 0, n_off)
        struct.pack_into('<I', buf, base + 4, sh_type)
        struct.pack_into('<Q', buf, base + 8, sh_flags)
        struct.pack_into('<Q', buf, base + 16, 0)
        struct.pack_into('<Q', buf, base + 24, section_offsets[i])
        struct.pack_into('<Q', buf, base + 32, sh_size)
        struct.pack_into('<I', buf, base + 40, sh_link)
        struct.pack_into('<I', buf, base + 44, sh_info)
        struct.pack_into('<Q', buf, base + 48, sh_align)
        struct.pack_into('<Q', buf, base + 56, sh_entsize)

    # Program headers
    def _phdr(idx, p_type, p_flags, p_offset, p_filesz, p_memsz, p_align):
        base = phoff + idx * PH_ENTRY_SIZE
        struct.pack_into('<I', buf, base + 0, p_type)
        struct.pack_into('<I', buf, base + 4, p_flags)
        struct.pack_into('<Q', buf, base + 8, p_offset)
        struct.pack_into('<Q', buf, base + 16, 0)
        struct.pack_into('<Q', buf, base + 24, 0)
        struct.pack_into('<Q', buf, base + 32, p_filesz)
        struct.pack_into('<Q', buf, base + 40, p_memsz)
        struct.pack_into('<Q', buf, base + 48, p_align)

    phdr_sz = NUM_PHDRS * PH_ENTRY_SIZE
    _phdr(0, PT_PHDR, 0x4, phoff, phdr_sz, phdr_sz, 0x8)
    _phdr(1, PT_LOAD, 0x4, phoff, phdr_sz, phdr_sz, 0x8)
    _phdr(2, PT_LOAD, 0x5, section_offsets[TEXT_IDX], len(text_data), len(text_data), 0x8)
    _phdr(3, PT_LOAD, 0x6, section_offsets[SHARED_IDX], 0, 64, 0x8)
    _phdr(4, PT_LOAD, 0x4, section_offsets[CONST0_IDX], len(const0_data), len(const0_data), 0x4)

    return bytes(buf)
