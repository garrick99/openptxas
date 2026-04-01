"""
cubin/emitter_sm89.py — Generate a loadable SM_89 (Ada Lovelace) cubin ELF.

SM_89 is simpler than SM_120: no capmerc/merc sections, 12 sections total,
3 program headers. Reverse-engineered from ptxas 12.x output for sm_89.
"""

from __future__ import annotations
import struct


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
SHT_RELA     = 4
SHT_REL      = 9
SHT_NV_INFO      = 0x70000000
SHT_NV_CALLGRAPH = 0x70000001

SHF_WRITE     = 0x1
SHF_ALLOC     = 0x2
SHF_EXECINSTR = 0x4

PT_LOAD = 1
PT_PHDR = 6

SM89_E_FLAGS = 0x00590559

STB_LOCAL  = 0
STB_GLOBAL = 1
STT_NOTYPE  = 0
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
# .nv.info (global CUDA attributes)
# ---------------------------------------------------------------------------

def _build_nv_info_global(num_sections: int = 12):
    """Global .nv.info: section count, MINCTASM, MAXNTID."""
    buf = bytearray()
    # Tag 0x2f: section count (format 0x04, 8 bytes payload)
    buf.extend(bytes([0x04, 0x2f, 0x08, 0x00]))
    buf.extend(_u32(num_sections))
    buf.extend(_u32(0x0a))
    # Tag 0x12: MINCTASM (format 0x04, 8 bytes)
    buf.extend(bytes([0x04, 0x12, 0x08, 0x00]))
    buf.extend(bytes(8))
    # Tag 0x11: MAXNTID (format 0x04, 8 bytes)
    buf.extend(bytes([0x04, 0x11, 0x08, 0x00]))
    buf.extend(bytes(8))
    return bytes(buf)


# ---------------------------------------------------------------------------
# .nv.info.{kernel} (per-kernel attributes)
# ---------------------------------------------------------------------------

def _build_nv_info_kernel(num_gprs: int, num_params: int,
                          param_sizes: list[int],
                          param_offsets: list[int],
                          cbank_param_size: int,
                          exit_offsets: list[int],
                          s2r_offset: int = 0x10):
    """Per-kernel .nv.info attributes for SM_89."""
    buf = bytearray()

    # EIATTR_REGCOUNT (0x37): format 0x04, 4 bytes payload
    # SM_89: ptxas always uses 0x7c regardless of actual GPR count
    buf.extend(bytes([0x04, 0x37, 0x04, 0x00]))
    buf.extend(_u32(0x7c))

    # EIATTR_MAXREG (0x0a): format 0x04, 8 bytes payload
    # ptxas SM_89: 0x02000000 + text_offset<<8 | 0x1c<<0
    buf.extend(bytes([0x04, 0x0a, 0x08, 0x00]))
    buf.extend(bytes([0x02, 0x00, 0x00, 0x00]))
    # Second dword: encodes text section reference + S2R offset
    buf.extend(bytes([0x60, 0x01, 0x1c, 0x00]))

    # EIATTR_S2RCTX (0x19): format 0x03, 2-byte value
    # ptxas SM_89 uses 0x1c for vecadd
    buf.extend(bytes([0x03, 0x19, 0x1c, 0x00]))

    # EIATTR_PARAM_INFO (0x17): 12 bytes per param, reverse order
    for i in range(num_params - 1, -1, -1):
        buf.extend(bytes([0x04, 0x17, 0x0c, 0x00]))
        buf.extend(bytes([0x00, 0x00, 0x00, 0x00]))  # padding
        buf.extend(bytes([i & 0xFF, 0x00]))            # param ordinal
        off = param_offsets[i]
        buf.extend(bytes([off & 0xFF, (off >> 8) & 0xFF]))  # offset in cbank
        # Flags: 0xf011 for 32-bit, 0xf021 for 64-bit
        size_ind = 0x11 if param_sizes[i] <= 4 else 0x21
        buf.extend(bytes([0x00, 0xf0, size_ind, 0x00]))

    # EIATTR_CBANK_SIZE (0x1b): format 0x03, value = 0xFF (ptxas SM_89 always uses 0xFF)
    cb_val = 0xFF
    buf.extend(bytes([0x03, 0x1b, cb_val & 0xFF, 0x00]))

    # EIATTR_EXIT_OFFSETS (0x1c): format 0x04, list of EXIT byte offsets
    if not exit_offsets:
        exit_offsets = [0x10]
    payload = b''.join(struct.pack('<I', off) for off in exit_offsets)
    buf.extend(bytes([0x04, 0x1c]))
    buf.extend(struct.pack('<H', len(payload)))
    buf.extend(payload)

    return bytes(buf)


# ---------------------------------------------------------------------------
# Fixed-content helper sections
# ---------------------------------------------------------------------------

def _build_callgraph():
    """32 bytes of zeros for .nv.callgraph."""
    return b'\x00' * 32


def _build_rel_action():
    """16-byte placeholder for .nv.rel.action."""
    return b'\x00' * 16


def _build_rel_debug_frame():
    """16-byte placeholder for .rel.debug_frame."""
    return b'\x00' * 16


def _build_debug_frame():
    """Minimal empty .debug_frame placeholder."""
    return b'\x00' * 4  # 4 bytes, CIE terminator


# ---------------------------------------------------------------------------
# Main emitter
# ---------------------------------------------------------------------------

def emit_cubin_sm89(kernel_name: str,
                    sass_bytes: bytes,
                    num_gprs: int,
                    num_params: int,
                    param_sizes: list[int],
                    param_offsets: list[int],
                    const0_size: int,
                    const0_init_data: bytes | None = None,
                    exit_offsets: list[int] | None = None) -> bytes:
    """
    Generate a complete SM_89 cubin ELF binary.

    Parameters
    ----------
    kernel_name : str
        Kernel function name (e.g. "vector_add").
    sass_bytes : bytes
        Assembled SASS machine code.
    num_gprs : int
        Number of general-purpose registers used.
    num_params : int
        Number of kernel parameters.
    param_sizes : list[int]
        Size in bytes of each parameter (4 for u32, 8 for u64/ptr).
    param_offsets : list[int]
        Byte offset of each parameter within the constant bank param area.
    const0_size : int
        Total size of .nv.constant0 section. Param area starts at 0x160.
    const0_init_data : bytes, optional
        Initial data for constant bank. If None, filled with zeros.
    exit_offsets : list[int], optional
        Byte offsets of EXIT instructions in the SASS. Auto-detected if None.

    Returns
    -------
    bytes
        Complete ELF cubin ready for cuModuleLoadData().
    """

    # --- Prepare text data (pad to 128-byte alignment, minimum 128 bytes) ---
    text_data = bytearray(sass_bytes)
    while len(text_data) % 128 != 0:
        text_data.extend(b'\x00' * 16)  # NOP-sized padding
    if len(text_data) < 128:
        text_data.extend(b'\x00' * (128 - len(text_data)))
    text_data = bytes(text_data)

    # Auto-scan for EXIT instructions if not provided
    if exit_offsets is None:
        exit_offsets = [
            byte_off for byte_off in range(0, len(text_data), 16)
            if len(text_data) >= byte_off + 8 and
               (struct.unpack_from('<Q', text_data, byte_off)[0] & 0xFFF) == 0x94d
        ]
        if not exit_offsets:
            exit_offsets = [0x10]

    # --- Prepare constant0 data ---
    if const0_init_data is not None:
        raw = bytearray(const0_size)
        raw[:len(const0_init_data)] = const0_init_data[:const0_size]
        const0_data = bytes(raw)
    else:
        const0_data = b'\x00' * const0_size

    # --- Section names and indices ---
    # SM_89: 12 sections total
    #  0: (null)
    #  1: .shstrtab
    #  2: .strtab
    #  3: .symtab
    #  4: .debug_frame
    #  5: .nv.info
    #  6: .nv.info.{kernel}
    #  7: .nv.callgraph
    #  8: .nv.rel.action
    #  9: .rel.debug_frame
    # 10: .nv.constant0.{kernel}
    # 11: .text.{kernel}

    NUM_SECTIONS = 12
    NUM_PHDRS = 3

    text_sec = f'.text.{kernel_name}'
    const0_sec = f'.nv.constant0.{kernel_name}'
    nvinfo_k_sec = f'.nv.info.{kernel_name}'

    TEXT_IDX = 11
    CONST0_IDX = 10
    KERNEL_SYM_IDX = 2  # index in symtab
    FIRST_GLOBAL = 2

    sec_names = [
        '',                      # 0
        '.shstrtab',             # 1
        '.strtab',               # 2
        '.symtab',               # 3
        '.debug_frame',          # 4
        '.nv.info',              # 5
        nvinfo_k_sec,            # 6
        '.nv.callgraph',         # 7
        '.nv.rel.action',        # 8
        '.rel.debug_frame',      # 9
        const0_sec,              # 10
        text_sec,                # 11
    ]

    # --- Build string tables ---
    shstrtab = bytearray()
    _strtab_add(shstrtab, '')
    strtab = bytearray()
    _strtab_add(strtab, '')

    name_offsets = {}
    for n in sec_names:
        if n and n not in name_offsets:
            name_offsets[n] = _strtab_add(shstrtab, n)
    name_offsets[''] = 0
    shstrtab_data = bytes(shstrtab)

    # Symbol string table entries
    sn_kernel = _strtab_add(strtab, kernel_name)
    sn_text = _strtab_add(strtab, text_sec)
    strtab_data = bytes(strtab)

    # --- Build symbol table ---
    def _sym(st_name, st_info, st_other, st_shndx, st_value, st_size):
        return (_u32(st_name) + bytes([st_info, st_other]) +
                _u16(st_shndx) + _u64(st_value) + _u64(st_size))

    symtab = bytearray()
    symtab.extend(_sym(0, 0, 0, 0, 0, 0))                                         # [0] null
    symtab.extend(_sym(sn_text, (STB_LOCAL << 4) | STT_SECTION, 0, TEXT_IDX, 0, 0))  # [1] .text section
    symtab.extend(_sym(sn_kernel, (STB_GLOBAL << 4) | STT_FUNC, 0x10, TEXT_IDX, 0, len(text_data)))  # [2] kernel
    symtab_data = bytes(symtab)

    # --- Build section data ---
    cbank_param_size = sum(param_sizes) if param_sizes else 0

    section_datas = [
        b'',                                                          # 0: null
        shstrtab_data,                                                # 1: .shstrtab
        strtab_data,                                                  # 2: .strtab
        symtab_data,                                                  # 3: .symtab
        _build_debug_frame(),                                         # 4: .debug_frame
        _build_nv_info_global(NUM_SECTIONS),                          # 5: .nv.info
        _build_nv_info_kernel(num_gprs, num_params, param_sizes,      # 6: .nv.info.{k}
                              param_offsets, cbank_param_size,
                              exit_offsets),
        _build_callgraph(),                                           # 7: .nv.callgraph
        _build_rel_action(),                                          # 8: .nv.rel.action
        _build_rel_debug_frame(),                                     # 9: .rel.debug_frame
        const0_data,                                                  # 10: .nv.constant0.{k}
        text_data,                                                    # 11: .text.{k}
    ]

    # Section metadata: (sh_type, sh_flags, sh_link, sh_info, sh_addralign, sh_entsize)
    section_meta = [
        (SHT_NULL,         0,                       0, 0,              0,   0),   # 0
        (SHT_STRTAB,       0,                       0, 0,              1,   0),   # 1
        (SHT_STRTAB,       0,                       0, 0,              1,   0),   # 2
        (SHT_SYMTAB,       0,                       2, FIRST_GLOBAL,   8,  24),   # 3
        (SHT_PROGBITS,     0,                       0, 0,              1,   0),   # 4: .debug_frame
        (SHT_NV_INFO,      0,                       3, 0,              4,   0),   # 5: .nv.info
        (SHT_NV_INFO,      0x40,                    3, TEXT_IDX,       4,   0),   # 6: .nv.info.{k}
        (SHT_NV_CALLGRAPH, 0,                       3, 0,              4,   8),   # 7: .nv.callgraph
        (SHT_PROGBITS,     0,                       0, 0,              4,   0),   # 8: .nv.rel.action
        (SHT_REL,          0,                       3, 4,              8,  16),   # 9: .rel.debug_frame
        (SHT_PROGBITS,     SHF_ALLOC | 0x40,        0, TEXT_IDX,       4,   0),   # 10: .nv.constant0
        (SHT_PROGBITS,     SHF_ALLOC | SHF_EXECINSTR, 3, KERNEL_SYM_IDX, 128, 0),  # 11: .text
    ]

    # --- Compute file offsets ---
    offset = ELF_HEADER_SIZE
    section_offsets = [0] * NUM_SECTIONS
    for i in range(1, NUM_SECTIONS):
        _, _, _, _, align, _ = section_meta[i]
        if align > 1:
            r = offset % align
            if r:
                offset += align - r
        section_offsets[i] = offset
        offset += len(section_datas[i])

    # Section header table (8-byte aligned)
    r = offset % 8
    if r:
        offset += 8 - r
    shoff = offset
    offset += NUM_SECTIONS * SH_ENTRY_SIZE

    # Program header table (8-byte aligned)
    r = offset % 8
    if r:
        offset += 8 - r
    phoff = offset
    offset += NUM_PHDRS * PH_ENTRY_SIZE

    total_size = offset
    buf = bytearray(total_size)

    # --- ELF header ---
    buf[0:4] = ELF_MAGIC
    buf[4] = ELFCLASS64
    buf[5] = ELFDATA2LSB
    buf[6] = EV_CURRENT
    buf[7] = 0x33  # ELFOSABI_CUDA (SM_89: 0x33, SM_120: 0x41)
    buf[8] = 0x07  # EI_ABIVERSION (CUDA ABI v7 for SM_89)
    struct.pack_into('<H', buf, 16, ET_EXEC)
    struct.pack_into('<H', buf, 18, EM_CUDA)
    struct.pack_into('<I', buf, 20, 124)  # CUDA toolchain version (ptxas 12.4 = 124)
    struct.pack_into('<Q', buf, 24, 0)          # e_entry
    struct.pack_into('<Q', buf, 32, phoff)      # e_phoff
    struct.pack_into('<Q', buf, 40, shoff)      # e_shoff
    struct.pack_into('<I', buf, 48, SM89_E_FLAGS)
    struct.pack_into('<H', buf, 52, ELF_HEADER_SIZE)
    struct.pack_into('<H', buf, 54, PH_ENTRY_SIZE)
    struct.pack_into('<H', buf, 56, NUM_PHDRS)
    struct.pack_into('<H', buf, 58, SH_ENTRY_SIZE)
    struct.pack_into('<H', buf, 60, NUM_SECTIONS)
    struct.pack_into('<H', buf, 62, 1)          # e_shstrndx

    # --- Write section data ---
    for i in range(1, NUM_SECTIONS):
        data = section_datas[i]
        off = section_offsets[i]
        buf[off:off + len(data)] = data

    # --- Write section headers ---
    for i in range(NUM_SECTIONS):
        base = shoff + i * SH_ENTRY_SIZE
        n_off = name_offsets.get(sec_names[i], 0)
        sh_type, sh_flags, sh_link, sh_info, sh_align, sh_entsize = section_meta[i]
        sh_size = len(section_datas[i]) if i > 0 else 0

        struct.pack_into('<I', buf, base + 0,  n_off)
        struct.pack_into('<I', buf, base + 4,  sh_type)
        struct.pack_into('<Q', buf, base + 8,  sh_flags)
        struct.pack_into('<Q', buf, base + 16, 0)               # sh_addr
        struct.pack_into('<Q', buf, base + 24, section_offsets[i])
        struct.pack_into('<Q', buf, base + 32, sh_size)
        struct.pack_into('<I', buf, base + 40, sh_link)
        struct.pack_into('<I', buf, base + 44, sh_info)
        struct.pack_into('<Q', buf, base + 48, sh_align)
        struct.pack_into('<Q', buf, base + 56, sh_entsize)

    # --- Program headers ---
    def _phdr(idx, p_type, p_flags, p_offset, p_filesz, p_memsz, p_align):
        base = phoff + idx * PH_ENTRY_SIZE
        struct.pack_into('<I', buf, base + 0,  p_type)
        struct.pack_into('<I', buf, base + 4,  p_flags)
        struct.pack_into('<Q', buf, base + 8,  p_offset)
        struct.pack_into('<Q', buf, base + 16, 0)   # p_vaddr
        struct.pack_into('<Q', buf, base + 24, 0)   # p_paddr
        struct.pack_into('<Q', buf, base + 32, p_filesz)
        struct.pack_into('<Q', buf, base + 40, p_memsz)
        struct.pack_into('<Q', buf, base + 48, p_align)

    phdr_sz = NUM_PHDRS * PH_ENTRY_SIZE
    _phdr(0, PT_PHDR, 0x4, phoff, phdr_sz, phdr_sz, 0x8)
    _phdr(1, PT_LOAD, 0x5, section_offsets[TEXT_IDX], len(text_data), len(text_data), 0x8)
    _phdr(2, PT_LOAD, 0x4, section_offsets[CONST0_IDX], len(const0_data), len(const0_data), 0x4)

    return bytes(buf)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Generate a minimal test cubin with a dummy kernel
    kernel_name = 'test_kernel'
    # Minimal SASS: 128 bytes of zeros (placeholder instructions)
    sass = b'\x00' * 128
    num_gprs = 8
    num_params = 2
    param_sizes = [8, 8]        # two u64 params
    param_offsets = [0, 8]      # offsets within param area
    const0_size = 0x170         # 0x160 param base + 16 bytes for 2 params

    cubin = emit_cubin_sm89(
        kernel_name=kernel_name,
        sass_bytes=sass,
        num_gprs=num_gprs,
        num_params=num_params,
        param_sizes=param_sizes,
        param_offsets=param_offsets,
        const0_size=const0_size,
        exit_offsets=[0x10],
    )

    # Verify ELF header
    assert cubin[:4] == b'\x7fELF', 'Bad ELF magic'
    assert cubin[4] == 2, 'Not ELFCLASS64'
    assert cubin[5] == 1, 'Not little-endian'
    assert cubin[7] == 0x33, 'Not ELFOSABI_CUDA (SM_89)'

    e_machine = struct.unpack_from('<H', cubin, 18)[0]
    assert e_machine == 0xBE, f'e_machine={e_machine:#x}, expected 0xBE'

    e_flags = struct.unpack_from('<I', cubin, 48)[0]
    assert e_flags == SM89_E_FLAGS, f'e_flags={e_flags:#x}, expected {SM89_E_FLAGS:#x}'

    e_phnum = struct.unpack_from('<H', cubin, 56)[0]
    assert e_phnum == 3, f'e_phnum={e_phnum}, expected 3'

    e_shnum = struct.unpack_from('<H', cubin, 60)[0]
    assert e_shnum == 12, f'e_shnum={e_shnum}, expected 12'

    e_shstrndx = struct.unpack_from('<H', cubin, 62)[0]
    assert e_shstrndx == 1, f'e_shstrndx={e_shstrndx}, expected 1'

    print(f'SM_89 cubin generated: {len(cubin)} bytes')
    print(f'  e_machine  = {e_machine:#06x}')
    print(f'  e_flags    = {e_flags:#010x}')
    print(f'  e_phnum    = {e_phnum}')
    print(f'  e_shnum    = {e_shnum}')
    print(f'  e_shstrndx = {e_shstrndx}')
    print('All ELF header checks passed.')
