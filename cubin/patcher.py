"""
cubin/patcher.py — Template-based cubin instruction patcher.

Strategy: load an existing ptxas-compiled cubin, locate the .text.{kernel}
section(s), and overwrite individual instructions at known byte offsets with
corrected encodings.

This is the fastest path to a loadable, driver-runnable cubin that doesn't
contain the ptxas miscompilation bug.  The ELF headers, .nv.info, .nv.constant0,
and all other sections are left intact — only the SASS instruction bytes change.

Typical workflow:
    1. Compile PTX → cubin with ptxas (buggy output)
    2. Identify miscompiled instruction offsets via tools/re_probe.py
    3. Use CubinPatcher to replace each bad instruction with correct bytes
    4. Write the patched cubin; load it with cuModuleLoad

Usage example:
    from cubin.patcher import CubinPatcher
    from sass.encoding.sm_120_encode import encode_shf_l_w_u32_hi
    from sass.encoding.sm_120_opcodes import encode_iadd3, RZ

    p = CubinPatcher('buggy.cubin')
    # Replace offset 80 in kernel 'rotate_test' with correct IADD3
    p.patch_instruction('rotate_test', offset=80, new_bytes=encode_iadd3(RZ, RZ, 4, RZ))
    p.write('fixed.cubin')
"""

from __future__ import annotations
import struct
import copy
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Minimal ELF64 parser/patcher (no external deps — pyelftools for reading,
# but we patch raw bytes so we only need the section offsets).
# ---------------------------------------------------------------------------

class ELF64:
    """
    Minimal ELF64 reader that maps section names → (file_offset, size).
    We use this to locate .text.{kernel} sections by byte offset in the file,
    then patch them directly in the raw byte buffer.
    """

    ELF_MAGIC = b'\x7fELF'

    # ELF header field offsets
    _E_SHOFF  = 40   # section header table offset (uint64)
    _E_SHENTSIZE = 58  # section header entry size (uint16)
    _E_SHNUM  = 60   # number of section headers (uint16)
    _E_SHSTRNDX = 62 # section name string table index (uint16)

    # Section header field offsets (within each 64-byte SHdr)
    _SH_NAME   = 0   # uint32: offset into shstrtab
    _SH_TYPE   = 4   # uint32
    _SH_FLAGS  = 8   # uint64
    _SH_ADDR   = 16  # uint64
    _SH_OFFSET = 24  # uint64: file offset of section data
    _SH_SIZE   = 32  # uint64: section size in bytes

    def __init__(self, data: bytes):
        if data[:4] != self.ELF_MAGIC:
            raise ValueError("Not an ELF file")
        if data[4] != 2:
            raise ValueError("Expected ELF64 (EI_CLASS=2)")
        if data[5] != 1:
            raise ValueError("Expected little-endian ELF (EI_DATA=1)")
        self._data = bytearray(data)
        self._parse()

    def _u16(self, off: int) -> int:
        return struct.unpack_from('<H', self._data, off)[0]

    def _u64(self, off: int) -> int:
        return struct.unpack_from('<Q', self._data, off)[0]

    def _parse(self):
        shoff    = self._u64(self._E_SHOFF)
        shentsz  = self._u16(self._E_SHENTSIZE)
        shnum    = self._u16(self._E_SHNUM)
        shstrndx = self._u16(self._E_SHSTRNDX)

        # Read the section name string table
        sh_start = shoff + shstrndx * shentsz
        strtab_off  = self._u64(sh_start + self._SH_OFFSET)
        strtab_size = self._u64(sh_start + self._SH_SIZE)
        strtab = self._data[strtab_off : strtab_off + strtab_size]

        def _get_name(name_off: int) -> str:
            end = strtab.index(b'\x00', name_off)
            return strtab[name_off:end].decode('utf-8', errors='replace')

        # Build section index: name → (file_offset, size, shdr_offset)
        self._sections: dict[str, tuple[int, int, int]] = {}
        for i in range(shnum):
            sh_base = shoff + i * shentsz
            name_off = struct.unpack_from('<I', self._data, sh_base + self._SH_NAME)[0]
            name = _get_name(name_off)
            sec_off  = self._u64(sh_base + self._SH_OFFSET)
            sec_size = self._u64(sh_base + self._SH_SIZE)
            self._sections[name] = (sec_off, sec_size, sh_base)

    def section_names(self) -> list[str]:
        return list(self._sections.keys())

    def section_data(self, name: str) -> bytes:
        off, size, _ = self._sections[name]
        return bytes(self._data[off : off + size])

    def patch_bytes(self, section_name: str, section_offset: int,
                    new_bytes: bytes) -> None:
        """
        Overwrite bytes within a named section.

        Args:
            section_name:   ELF section name (e.g. '.text.rotate_test')
            section_offset: Byte offset from start of section
            new_bytes:      Replacement bytes (must fit within section)
        """
        if section_name not in self._sections:
            raise KeyError(f"Section {section_name!r} not found; "
                           f"available: {self.section_names()}")
        sec_off, sec_size, _ = self._sections[section_name]
        end = section_offset + len(new_bytes)
        if end > sec_size:
            raise ValueError(
                f"Patch [{section_offset}:{end}] extends past section "
                f"end (size={sec_size})")
        file_pos = sec_off + section_offset
        self._data[file_pos : file_pos + len(new_bytes)] = new_bytes

    def to_bytes(self) -> bytes:
        return bytes(self._data)


# ---------------------------------------------------------------------------
# High-level patcher
# ---------------------------------------------------------------------------

class CubinPatcher:
    """
    Load a ptxas-compiled cubin and patch specific SASS instructions.

    Each patch targets a kernel (by name) and an instruction byte offset
    within that kernel's .text.{name} section.

    Example::

        p = CubinPatcher('buggy.cubin')
        p.patch_instruction('rotate_test', offset=80,
                            new_bytes=encode_iadd3(RZ, RZ, 4, RZ))
        p.write('fixed.cubin')
    """

    def __init__(self, cubin_path: str | Path):
        self._path = Path(cubin_path)
        raw = self._path.read_bytes()
        self._elf = ELF64(raw)
        self._patches: list[tuple[str, int, bytes]] = []

    def kernel_names(self) -> list[str]:
        """Return all kernel names found in the cubin (.text.{name} sections)."""
        return [
            s[len('.text.'):] for s in self._elf.section_names()
            if s.startswith('.text.') and s != '.text.'
        ]

    def text_section_name(self, kernel: str) -> str:
        return f'.text.{kernel}'

    def get_instruction(self, kernel: str, offset: int) -> bytes:
        """
        Read 16 bytes of SASS at instruction byte offset within a kernel.

        Args:
            kernel: kernel function name
            offset: byte offset within .text.{kernel} (must be 16-byte aligned)
        """
        if offset % 16 != 0:
            raise ValueError(f"offset={offset} is not 16-byte aligned")
        sec = self.text_section_name(kernel)
        data = self._elf.section_data(sec)
        return data[offset : offset + 16]

    def patch_instruction(self, kernel: str, offset: int,
                          new_bytes: bytes) -> None:
        """
        Queue a 16-byte instruction patch.

        The patch is applied to the in-memory ELF image immediately.
        Call write() to persist.

        Args:
            kernel:    kernel function name
            offset:    byte offset within .text.{kernel} (must be multiple of 16)
            new_bytes: exactly 16 replacement bytes
        """
        if offset % 16 != 0:
            raise ValueError(f"offset={offset} is not 16-byte aligned")
        if len(new_bytes) != 16:
            raise ValueError(f"new_bytes must be 16 bytes, got {len(new_bytes)}")
        sec = self.text_section_name(kernel)
        self._elf.patch_bytes(sec, offset, new_bytes)
        self._patches.append((kernel, offset, new_bytes))

    def patch_instructions(self, kernel: str,
                           patches: list[tuple[int, bytes]]) -> None:
        """
        Apply multiple instruction patches to a single kernel.

        Args:
            kernel:  kernel function name
            patches: list of (offset, 16_bytes) pairs
        """
        for offset, new_bytes in patches:
            self.patch_instruction(kernel, offset, new_bytes)

    def patch_summary(self) -> str:
        lines = [f"CubinPatcher: {len(self._patches)} patch(es) to {self._path.name}"]
        for kernel, offset, new_bytes in self._patches:
            lines.append(f"  .text.{kernel} +{offset}: {new_bytes.hex()}")
        return '\n'.join(lines)

    def write(self, out_path: str | Path | None = None) -> Path:
        """
        Write the patched cubin to disk.

        Args:
            out_path: output path (default: add '_patched' suffix to input name)

        Returns:
            Path of written file.
        """
        if out_path is None:
            stem = self._path.stem
            suffix = self._path.suffix
            out_path = self._path.with_name(f'{stem}_patched{suffix}')
        out_path = Path(out_path)
        out_path.write_bytes(self._elf.to_bytes())
        return out_path


# ---------------------------------------------------------------------------
# Convenience: locate a byte pattern within a kernel's .text section
# ---------------------------------------------------------------------------

def find_instruction_offset(cubin_path: str | Path, kernel: str,
                             pattern: bytes) -> Optional[int]:
    """
    Search for a 16-byte instruction pattern in a kernel's .text section.

    Returns the byte offset of the first match, or None if not found.
    Only checks at 16-byte aligned positions.

    Useful for locating a specific miscompiled instruction to patch.
    """
    raw = Path(cubin_path).read_bytes()
    elf = ELF64(raw)
    sec_name = f'.text.{kernel}'
    data = elf.section_data(sec_name)
    for off in range(0, len(data) - 15, 16):
        if data[off:off+16] == pattern:
            return off
    return None


def find_all_instruction_offsets(cubin_path: str | Path, kernel: str,
                                  pattern: bytes) -> list[int]:
    """
    Find all 16-byte aligned offsets where pattern appears in .text.{kernel}.
    """
    raw = Path(cubin_path).read_bytes()
    elf = ELF64(raw)
    data = elf.section_data(f'.text.{kernel}')
    return [off for off in range(0, len(data) - 15, 16)
            if data[off:off+16] == pattern]


def disassemble_text(cubin_path: str | Path, kernel: str) -> list[tuple[int, bytes]]:
    """
    Return all 16-byte instructions from .text.{kernel} as (offset, bytes) pairs.
    """
    raw = Path(cubin_path).read_bytes()
    elf = ELF64(raw)
    data = elf.section_data(f'.text.{kernel}')
    return [(off, data[off:off+16]) for off in range(0, len(data), 16)]
