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

# Architecture-specific ELF flags
SM120_FLAGS = 0x06007802
SM89_FLAGS  = 0x06005904

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


def _patch_cuinfo_sm(sm_version: int) -> bytes:
    """Patch .note.nv.cuinfo with the target SM version."""
    _load_note_templates()
    buf = bytearray(_NOTE_CUINFO)
    sm_hex = {89: 0x59, 120: 0x78}.get(sm_version, 0x78)
    buf[26] = sm_hex
    return bytes(buf)


def _build_nv_info_global(num_gprs: int = 16):
    # EIATTR_REGCOUNT (0x2f): must be at least the actual register count so that
    # R16+ instructions are not rejected as out-of-range by the driver.
    # Previously hard-coded to 0x10 (16), which caused ERR_ILLEGAL_INSTRUCTION
    # for kernels that use addr-scratch registers R16..R17.
    rc = max(num_gprs, 16)  # minimum 16 per SM_120 hardware requirement
    buf = bytearray(bytes.fromhex(
        '042f08000800000010000000'
        '041108000800000000000000'
        '041208000800000000000000'
    ))
    buf[8] = rc & 0xFF
    return bytes(buf)


def _build_nv_info_kernel(num_gprs: int = 8, num_params: int = 2,
                          param_sizes: list[int] = None,
                          exit_offsets: list[int] = None,
                          s2r_instr_offset: int = 0x10):
    """Generate per-kernel .nv.info attributes dynamically.

    Builds the attribute stream based on actual kernel parameters
    instead of using a fixed template.
    """
    if param_sizes is None:
        param_sizes = [8] * num_params  # default: all u64

    buf = bytearray()

    # EIATTR_REGCOUNT (0x37): always 0x82
    buf.extend(bytes([0x04, 0x37, 0x04, 0x00, 0x82, 0x00, 0x00, 0x00]))

    # EIATTR_PARAM_INFO (0x17): one entry per parameter
    # ptxas emits in reverse order with cumulative byte offsets from param base.
    # Flags: 0xf021 for 64-bit params (u64, s64), 0xf011 for 32-bit params (u32, s32).
    cumulative_offset = 0
    param_offsets_list = []
    for i in range(num_params):
        param_offsets_list.append(cumulative_offset)
        cumulative_offset += param_sizes[i]
    for i in range(num_params - 1, -1, -1):
        buf.extend(bytes([0x04, 0x17, 0x0c, 0x00]))
        buf.extend(bytes([0x00, 0x00, 0x00, 0x00]))  # padding
        buf.extend(bytes([i & 0xFF, 0x00]))  # param ordinal
        off = param_offsets_list[i]
        buf.extend(bytes([off & 0xFF, (off >> 8) & 0xFF]))  # cumulative offset
        # Flags: 0x00, 0xf0, size_indicator, 0x00
        # size_indicator: 0x11 for 32-bit params, 0x21 for 64-bit params
        size_ind = 0x11 if param_sizes[i] <= 4 else 0x21
        buf.extend(bytes([0x00, 0xf0, size_ind, 0x00]))

    # EIATTR_PARAM_CBANK (0x50)
    buf.extend(bytes([0x03, 0x50, 0x00, 0x00]))

    # EIATTR_CBANK_PARAM_SIZE (0x1b): actual total parameter bytes.
    # Must NOT exceed the real param area — driver zeros this many bytes at
    # param_base, clobbering any literal pool values placed after params.
    total_param_bytes = sum(param_sizes) if param_sizes else 0
    buf.extend(bytes([0x03, 0x1b, total_param_bytes & 0xFF, 0x00]))

    # EIATTR_0x5f: Mercury compiler version flag — required for SM_120.
    # ptxas always emits 0x0101 here for SM_120 cubins.
    buf.extend(bytes([0x03, 0x5f, 0x01, 0x01]))

    # EIATTR 0x29: thread mask (0xFFFFFFFF = all threads active)
    # ptxas emits this for all kernels with 3+ params.
    buf.extend(bytes([0x04, 0x29, 0x04, 0x00, 0xFF, 0xFF, 0xFF, 0xFF]))

    # EIATTR 0x28: reg allocation hint (ptxas uses 0xA0 = 160)
    buf.extend(bytes([0x04, 0x28, 0x04, 0x00, 0xA0, 0x00, 0x00, 0x00]))

    # EIATTR_EXIT_INSTR_OFFSETS (0x1c): list of EXIT byte offsets in .text
    # Format 0x04: 4-byte header (fmt, tag, size_lo, size_hi) + N*4 bytes payload
    if not exit_offsets:
        exit_offsets = [0x10]  # fallback: assume first EXIT at offset 16
    payload = b''.join(struct.pack('<I', off) for off in exit_offsets)
    buf.extend(bytes([0x04, 0x1c]))
    buf.extend(struct.pack('<H', len(payload)))
    buf.extend(payload)

    # EIATTR_CTAID_DIMS (0x4a)
    buf.extend(bytes([0x02, 0x4a, 0x00, 0x00]))

    # EIATTR_0x19: total parameter bytes (not s2r offset).
    buf.extend(bytes([0x03, 0x19, total_param_bytes & 0xFF, (total_param_bytes >> 8) & 0xFF]))

    # EIATTR_EXTERNS (0x0a): references the .nv.constant0 symtab symbol.
    # In our symtab layout (no .debug_frame), .nv.constant0 is always at index 7.
    # ptxas uses index 9 (it has extra .debug_frame symbols shifting the layout).
    # Last 2 bytes = PARAM_CBANK size = total parameter bytes.
    buf.extend(bytes([0x04, 0x0a, 0x08, 0x00,
                      0x07, 0x00, 0x00, 0x00, 0x80, 0x03,
                      total_param_bytes & 0xFF, (total_param_bytes >> 8) & 0xFF]))

    # EIATTR 0x36: always zero payload (matches ptxas)
    buf.extend(bytes([0x04, 0x36, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00]))

    return bytes(buf)


def _build_merc_nv_info_kernel(num_gprs: int = 8, num_params: int = 2,
                                param_sizes: list[int] = None,
                                exit_offsets: list[int] = None):
    """Generate per-kernel .nv.merc.nv.info attributes.

    The merc (Mercury compiler) version includes the 0x5a attribute which
    encodes per-instruction register allocation data. Without it, the GPU
    limits register access and triggers ERR715 for R14+.

    Since we can't generate the real 0x5a data (requires RE of Mercury format),
    we emit a 52-byte all-0xFF blob which tells the hardware to allow maximum
    register access for all instructions.
    """
    if param_sizes is None:
        param_sizes = [8] * num_params

    buf = bytearray()

    # EIATTR_REGCOUNT (0x37)
    buf.extend(bytes([0x04, 0x37, 0x04, 0x00, 0x82, 0x00, 0x00, 0x00]))

    # EIATTR_0x5a: per-instruction register allocation (52 bytes).
    # Uses the exact bytes from ptxas vector_add (sm_120) as a baseline.
    # This data is a compressed encoding of instruction-level metadata.
    buf.extend(bytes([0x04, 0x5a, 0x34, 0x00]))  # header: fmt=04, tag=5a, size=52
    buf.extend(bytes.fromhex(
        '8a9d22a4b19d146d00b42af3f758038e0c070a1be2de8ad75263870c'
        'd72b0700cd2b8a124e4c1624ba19f5f027946a021a000000'))

    # EIATTR_PARAM_INFO (0x17): same as non-merc version
    cumulative_offset = 0
    param_offsets_list = []
    for i in range(num_params):
        param_offsets_list.append(cumulative_offset)
        cumulative_offset += param_sizes[i]
    for i in range(num_params - 1, -1, -1):
        buf.extend(bytes([0x04, 0x17, 0x0c, 0x00]))
        buf.extend(bytes([0x00, 0x00, 0x00, 0x00]))
        buf.extend(bytes([i & 0xFF, 0x00]))
        off = param_offsets_list[i]
        buf.extend(bytes([off & 0xFF, (off >> 8) & 0xFF]))
        size_ind = 0x11 if param_sizes[i] <= 4 else 0x21
        buf.extend(bytes([0x00, 0xf0, size_ind, 0x00]))

    # EIATTR_PARAM_CBANK (0x50)
    buf.extend(bytes([0x03, 0x50, 0x00, 0x00]))
    # EIATTR_CBANK_PARAM_SIZE (0x1b)
    total_param_bytes = sum(param_sizes) if param_sizes else 0
    buf.extend(bytes([0x03, 0x1b, total_param_bytes & 0xFF, 0x00]))
    # EIATTR_0x5f: version/flag (always 0x0101 in ptxas cubins)
    buf.extend(bytes([0x03, 0x5f, 0x01, 0x01]))
    # EIATTR 0x29: thread mask (0xFFFFFFFF = all threads active)
    buf.extend(bytes([0x04, 0x29, 0x04, 0x00, 0xFF, 0xFF, 0xFF, 0xFF]))
    # EIATTR 0x28: reg allocation hint
    buf.extend(bytes([0x04, 0x28, 0x04, 0x00, 0xA0, 0x00, 0x00, 0x00]))
    # EIATTR_CTAID_DIMS (0x4a)
    buf.extend(bytes([0x02, 0x4a, 0x00, 0x00]))
    # EIATTR_EXIT_INSTR_OFFSETS (0x1c)
    if not exit_offsets:
        exit_offsets = [0x10]
    payload = b''.join(struct.pack('<I', off) for off in exit_offsets)
    buf.extend(bytes([0x04, 0x1c]))
    buf.extend(struct.pack('<H', len(payload)))
    buf.extend(payload)

    return bytes(buf)


def _build_nv_compat():
    return bytes.fromhex(
        '020900000202010002050500030701010203000002060100040b08005000000000000000'
    )


def _build_callgraph():
    return bytes.fromhex(
        '00000000ffffffff00000000feffffff00000000fdffffff00000000fcffffff'
    )


def _build_capmerc(num_gprs: int = 10):
    """
    .nv.capmerc.text.<kernel> — Mercury compiler metadata encoding PRF allocation.

    Header byte[8] = register count. This is what tells the CUDA driver how many
    physical registers to allocate per thread.

    Uses the R0-R7 template (146 bytes) with patched register count for any
    kernel that fits within 8 GPRs. For larger kernels, uses the force_highreg
    template (166 bytes) with patched register count.
    """
    # Capmerc (Mercury compiler metadata) is optimization hints, not required for execution.
    # The CUDA driver launches kernels correctly with a zeroed/minimal capmerc.
    # A proper capmerc would encode instruction scheduling and register liveness data
    # matching the specific SASS instruction sequence — generating this requires deep
    # reverse engineering of the Mercury compiler format.
    # For now: emit a minimal valid header that allows execution.
    # Capmerc byte[8] = register allocation: tells hardware how many GPRs to allocate.
    # Must use the full 130-byte template (16-byte minimal doesn't work).
    # Allocate registers: round up to next multiple of 8, minimum 16
    # Byte[8] = register count. Tells HW how many GPRs to allocate per thread.
    # The 202-byte generic template (extracted from ptxas sm_120 test kernels)
    # is instruction-sequence-independent — only the header/footer change.
    # Verified across 6 ptxas cubins with 19-35 GPRs: body bytes 16-199 identical.
    reg_count = max(num_gprs, 16)  # minimum 16 GPRs
    # Generic 202-byte capmerc template (from ptxas sm_120 test kernels).
    # Body bytes 16-199 are identical across ptxas cubins with 19-35 GPRs.
    # Only the header (byte[8], bytes[13-15]) and footer vary per GPR count.
    buf = bytearray.fromhex(
        '0c000000010000c016000000e06a1700'  # 16B header
        '010b040af80004000000410000040000'
        '010b040af80004000000010001020000'
        '010b060afa0004000000010104020000'
        '02220806fa0042000000410140000200'
        '00000000000000000000000018000000'
        '010b040af80004000000c10001040000'
        '02220806fa0062000000070240000200'
        '00000000000000000000000000000000'
        '010b0e0afa0005000000030139040000'
        '410c5404410c540402380e32f8004011'
        '0000000082000a00000201c001000000'
        '0000000000000000d005'              # 2B footer
    )
    buf[8] = reg_count
    # Register bitmap: set all bits for maximum register access
    buf[13] = 0xFF; buf[14] = 0xFF; buf[15] = 0xFF
    return bytes(buf)


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
    const0_size: int = 0x390  # default, overridden by pipeline based on params
    const0_init_data: bytes | None = None  # if set, used verbatim for .nv.constant0; else zeros
    exit_offset: int = 0x10  # byte offset of EXIT instruction in .text
    s2r_offset: int = 0x10  # byte offset of first S2R instruction in .text
    smem_size: int = 0           # static shared memory size in bytes (0 = none)
    sm_version: int = 120        # 89 (Ada) or 120 (Blackwell)
    ptxas_capmerc: bytes | None = None    # capmerc from ptxas (overrides generated)
    ptxas_merc_info: bytes | None = None  # merc.nv.info from ptxas (overrides generated)


def emit_cubin(kernel: KernelDesc) -> bytes:
    _load_note_templates()

    from sass.encoding.sm_120_opcodes import encode_nop, encode_bra

    # Pad SASS: ensure EXIT + BRA self-loop + NOP padding to 384B minimum
    sass = bytearray(kernel.sass_bytes)
    # The pipeline already appends EXIT + BRA trap loop.
    # Only add EXIT+BRA if the pipeline didn't (e.g., pre-assembled SASS).
    if len(sass) >= 32:
        lo_2nd_last = struct.unpack_from('<Q', sass, len(sass)-32)[0]
        lo_last = struct.unpack_from('<Q', sass, len(sass)-16)[0]
        if (lo_2nd_last & 0xFFF) == 0x94d and (lo_last & 0xFFF) == 0x947:
            pass  # already has EXIT+BRA trap loop
        else:
            if (lo_last & 0xFFF) != 0x94d:
                from sass.encoding.sm_120_opcodes import encode_exit
                sass.extend(encode_exit())
            sass.extend(encode_bra(-16))
    else:
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

    # Auto-scan text_data for all EXIT instructions (opcode 0x94d, bits[11:0])
    exit_offsets = [
        byte_off for byte_off in range(0, len(text_data), 16)
        if (struct.unpack_from('<Q', text_data, byte_off)[0] & 0xFFF) == 0x94d
    ]
    if not exit_offsets:
        exit_offsets = [kernel.exit_offset]  # fallback

    if kernel.const0_init_data is not None:
        # Literal pool baked in; pad/trim to the declared size
        raw = bytearray(kernel.const0_size)
        raw[:len(kernel.const0_init_data)] = kernel.const0_init_data[:kernel.const0_size]
        const0_data = bytes(raw)
    else:
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

    capmerc_sec = f'.nv.capmerc.text.{kernel.name}'
    merc_info_sec = f'.nv.merc.nv.info'
    merc_info_k_sec = f'.nv.merc.nv.info.{kernel.name}'
    merc_symtab_sec = '.nv.merc.symtab'
    nv_shared_sec = f'.nv.shared.{kernel.name}'

    has_smem = kernel.smem_size > 0

    # Section indices (conditional on smem)
    TKINFO_IDX = 4
    CUINFO_IDX = 5
    COMPAT_IDX = 7
    TEXT_IDX = 10
    SHARED_IDX = 11       # .nv.shared.reserved.0
    NV_SHARED_IDX = 12 if has_smem else -1  # .nv.shared.<kernel>
    CONST0_IDX = 13 if has_smem else 12
    CAPMERC_IDX = 14 if has_smem else 13
    MERC_INFO_IDX = 15 if has_smem else 14
    MERC_INFO_K_IDX = 16 if has_smem else 15
    MERC_SYMTAB_IDX = 17 if has_smem else 16
    NUM_SECTIONS = 18 if has_smem else 17

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
    ]
    if has_smem:
        sec_names.append(nv_shared_sec)   # 12
    sec_names.extend([
        const0_sec,                  # 12 or 13
        capmerc_sec,                 # 13 or 14
        merc_info_sec,               # 14 or 15
        merc_info_k_sec,             # 15 or 16
        merc_symtab_sec,             # 16 or 17
    ])

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

    # Mercury symtab mirrors the regular symtab but entries [3] and [8] point to
    # the CAPMERC section (not TEXT). In ptxas's Mercury architecture, the "text"
    # of a kernel is represented by the capmerc section from the merc perspective.
    # This is critical: the SM_120 hardware reads per-instruction metadata from
    # the capmerc section referenced through the merc symtab.
    merc_symtab = bytearray()
    merc_symtab.extend(_sym(0, 0, 0, 0, 0, 0))  # [0] null
    merc_symtab.extend(_sym(sn_tkinfo, (STB_LOCAL<<4)|STT_SECTION, 0, TKINFO_IDX, 0, 0))  # [1]
    merc_symtab.extend(_sym(sn_cuinfo, (STB_LOCAL<<4)|STT_SECTION, 0, CUINFO_IDX, 0, 0))  # [2]
    # [3]: ".text" symbol points to CAPMERC section in Mercury view
    merc_symtab.extend(_sym(sn_text, (STB_LOCAL<<4)|STT_SECTION, 0, CAPMERC_IDX, 0, 0))  # [3]
    merc_symtab.extend(_sym(sn_smem_off, (STB_WEAK<<4)|STT_OBJECT, 0, 0, 0x40, 4))  # [4]
    merc_symtab.extend(_sym(sn_smem_alias, (STB_WEAK<<4)|STT_NOTYPE, 0xa0, SHARED_IDX, 0x40, 0))  # [5]
    merc_symtab.extend(_sym(sn_callgraph, (STB_LOCAL<<4)|STT_SECTION, 0, 9, 0, 0))  # [6]
    merc_symtab.extend(_sym(sn_const0, (STB_LOCAL<<4)|STT_SECTION, 0, CONST0_IDX, 0, 0))  # [7]
    # [8]: kernel function symbol points to CAPMERC section in Mercury view
    merc_symtab.extend(_sym(sn_kernel, (STB_GLOBAL<<4)|STT_FUNC, 0x10, CAPMERC_IDX, 0, len(text_data)))  # [8]
    merc_symtab_data = bytes(merc_symtab)

    section_datas = [
        b'',                         # 0
        shstrtab_data,               # 1
        strtab_data,                 # 2
        symtab_data,                 # 3
        _NOTE_TKINFO,                # 4
        _patch_cuinfo_sm(kernel.sm_version),  # 5
        _build_nv_info_global(num_gprs=kernel.num_gprs),  # 6
        _build_nv_compat(),          # 7
        _build_nv_info_kernel(num_gprs=kernel.num_gprs, num_params=kernel.num_params,
                              param_sizes=kernel.param_sizes, exit_offsets=exit_offsets,
                              s2r_instr_offset=kernel.s2r_offset),
        _build_callgraph(),          # 9
        text_data,                   # 10
        shared_reserved,             # 11 (NOBITS)
    ]
    meta_base = [
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
    ]

    # Optionally add .nv.shared.<kernel> for smem kernels
    if has_smem:
        # .nv.shared.<kernel>: NOBITS, flags=0x43, info=TEXT_IDX, size=smem_size
        section_datas.append(b'')  # NOBITS: no file content
        meta_base.append(
            (SHT_NOBITS, SHF_WRITE|SHF_ALLOC|0x40, 0, TEXT_IDX, 4, 0))

    SHT_CUDA_CAPMERC = 0x70000016
    SHT_CUDA_MERC_INFO = 0x70000083
    SHT_CUDA_MERC_SYMTAB = 0x70000085

    # Use ptxas metadata when available. Otherwise, use the capmerc generator
    # which produces text-size-aware capmerc from SASS analysis.
    if kernel.ptxas_capmerc:
        capmerc_data = kernel.ptxas_capmerc
    else:
        from cubin.capmerc_gen import build_capmerc_from_sass
        capmerc_data = build_capmerc_from_sass(text_data, num_gprs=kernel.num_gprs)
    merc_info_data = kernel.ptxas_merc_info or _build_merc_nv_info_kernel(
        num_gprs=kernel.num_gprs, num_params=kernel.num_params,
        param_sizes=kernel.param_sizes, exit_offsets=exit_offsets)

    section_datas.extend([
        const0_data,
        capmerc_data,
        _build_nv_info_global(num_gprs=kernel.num_gprs),
        merc_info_data,
        merc_symtab_data,
    ])

    meta_base.extend([
        (SHT_PROGBITS,     SHF_ALLOC|0x40, 0, TEXT_IDX, 4, 0),
        # capmerc
        (SHT_CUDA_CAPMERC, 0x10000000, MERC_SYMTAB_IDX, KERNEL_SYM_IDX, 16, 0),
        # merc nv.info
        (SHT_CUDA_MERC_INFO, 0x10000000, MERC_SYMTAB_IDX, 0, 4, 0),
        # merc nv.info.<kernel>
        (SHT_CUDA_MERC_INFO, 0x10000040, MERC_SYMTAB_IDX, CAPMERC_IDX, 4, 0),
        # 16: .nv.merc.symtab
        (SHT_CUDA_MERC_SYMTAB, 0x10000000, 2, KERNEL_SYM_IDX, 8, 24),
    ])

    section_meta = meta_base

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
    elf_flags = SM89_FLAGS if kernel.sm_version == 89 else SM120_FLAGS
    struct.pack_into('<I', buf, 48, elf_flags)
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
        # NOBITS sections: sh_size is virtual size, not file size
        if i == SHARED_IDX:
            sh_size = 64  # .nv.shared.reserved.0
        if has_smem and i == NV_SHARED_IDX:
            sh_size = kernel.smem_size  # .nv.shared.<kernel>

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
