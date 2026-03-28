"""
End-to-end pipeline tests: PTX source → cubin bytes.

Validates that the generated cubin is a valid ELF with correct structure.
"""

import struct
import pytest

from sass.pipeline import compile_ptx_source
from cubin.patcher import ELF64


SIMPLE_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry simple_add(
    .param .u64 out_ptr,
    .param .u64 in_ptr)
{
    .reg .b64   %rd<4>;
    ld.param.u64    %rd0, [in_ptr];
    ld.param.u64    %rd1, [out_ptr];
    add.u64         %rd2, %rd0, %rd1;
    ret;
}
"""

ROTATE_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry rotate_add(
    .param .u64 out_ptr,
    .param .u64 in_ptr)
{
    .reg .b64   %rd<8>;
    ld.param.u64    %rd0, [in_ptr];
    shl.b64         %rd2, %rd0, 1;
    shr.u64         %rd3, %rd0, 63;
    add.u64         %rd4, %rd2, %rd3;
    ld.param.u64    %rd5, [out_ptr];
    ret;
}
"""

BUGGY_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry sub_not_rotate(
    .param .u64 out_ptr,
    .param .u64 in_ptr)
{
    .reg .b64   %rd<8>;
    ld.param.u64    %rd0, [in_ptr];
    shl.b64         %rd2, %rd0, 8;
    shr.u64         %rd3, %rd0, 56;
    sub.u64         %rd4, %rd2, %rd3;
    ld.param.u64    %rd5, [out_ptr];
    ret;
}
"""


def test_compile_simple():
    """Basic compilation produces bytes."""
    results = compile_ptx_source(SIMPLE_KERNEL)
    assert "simple_add" in results
    cubin = results["simple_add"]
    assert len(cubin) > 0
    assert cubin[:4] == b'\x7fELF'


def test_cubin_elf_valid():
    """Generated cubin is a valid ELF64 that our patcher can parse."""
    results = compile_ptx_source(SIMPLE_KERNEL)
    cubin = results["simple_add"]
    elf = ELF64(cubin)
    names = elf.section_names()
    assert '.text.simple_add' in names
    assert '.nv.constant0.simple_add' in names


def test_cubin_text_section():
    """Text section contains 16-byte-aligned SASS instructions."""
    results = compile_ptx_source(SIMPLE_KERNEL)
    cubin = results["simple_add"]
    elf = ELF64(cubin)
    text = elf.section_data('.text.simple_add')
    assert len(text) > 0
    assert len(text) % 16 == 0
    # Padded to 128-byte boundary
    assert len(text) % 128 == 0


def test_cubin_sm120_flags():
    """ELF e_flags contains SM_120 version."""
    results = compile_ptx_source(SIMPLE_KERNEL)
    cubin = results["simple_add"]
    e_flags = struct.unpack_from('<I', cubin, 48)[0]
    sm_ver = (e_flags >> 8) & 0xFF
    assert sm_ver == 120, f"Expected SM_120 (0x78), got SM_{sm_ver}"


def test_cubin_has_symbols():
    """Symbol table contains the kernel function symbol."""
    results = compile_ptx_source(SIMPLE_KERNEL)
    cubin = results["simple_add"]
    elf = ELF64(cubin)
    symtab = elf.section_data('.symtab')
    strtab = elf.section_data('.strtab')
    # Search for "simple_add" in symbol names
    assert b'simple_add' in strtab


def test_constant_bank_size():
    """.nv.constant0 is 0x390 bytes (912)."""
    results = compile_ptx_source(SIMPLE_KERNEL)
    cubin = results["simple_add"]
    elf = ELF64(cubin)
    const0 = elf.section_data('.nv.constant0.simple_add')
    assert len(const0) == 0x390


def test_compile_rotate():
    """Kernel with valid rotate pattern compiles."""
    results = compile_ptx_source(ROTATE_KERNEL)
    assert "rotate_add" in results
    cubin = results["rotate_add"]
    assert cubin[:4] == b'\x7fELF'


def test_compile_buggy_sub():
    """Kernel with sub-based 'fake rotate' compiles (correctly as subtraction)."""
    results = compile_ptx_source(BUGGY_KERNEL)
    assert "sub_not_rotate" in results
    cubin = results["sub_not_rotate"]
    assert cubin[:4] == b'\x7fELF'


def test_text_has_exit():
    """Text section ends with EXIT instruction (opcode 0x94d)."""
    results = compile_ptx_source(SIMPLE_KERNEL)
    cubin = results["simple_add"]
    elf = ELF64(cubin)
    text = elf.section_data('.text.simple_add')
    # Find EXIT (opcode 0x94d at bits[11:0])
    found_exit = False
    for off in range(0, len(text), 16):
        lo = struct.unpack_from('<Q', text, off)[0]
        opcode = lo & 0xFFF
        if opcode == 0x94d:
            found_exit = True
            break
    assert found_exit, "No EXIT instruction found in text section"


def test_program_headers():
    """Cubin has program headers."""
    results = compile_ptx_source(SIMPLE_KERNEL)
    cubin = results["simple_add"]
    e_phnum = struct.unpack_from('<H', cubin, 56)[0]
    assert e_phnum >= 2, f"Expected >=2 program headers, got {e_phnum}"


# ---------------------------------------------------------------------------
# Literal pool tests
# ---------------------------------------------------------------------------

IMM_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry imm_kernel(
    .param .u64 out,
    .param .u32 n)
{
    .reg .b32   %r<4>;
    .reg .b64   %rd<2>;
    .reg .pred  %p0;
    mov.u32     %r0, 42;
    setp.lt.u32 %p0, %r0, 100;
    ld.param.u64 %rd0, [out];
    @!%p0 st.global.u32 [%rd0], %r0;
    ret;
}
"""


def test_mov_immediate_compiles():
    """mov.u32 with immediate literal compiles to LDC (not a TODO NOP)."""
    results = compile_ptx_source(IMM_KERNEL)
    cubin = results["imm_kernel"]
    elf = ELF64(cubin)
    text = elf.section_data('.text.imm_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    # LDC opcode 0xb82 must appear in body (beyond preamble LDC at offset 0)
    ldc_count = opcodes.count(0xb82)
    assert ldc_count >= 2, f"Expected >=2 LDC instructions (preamble + literal), got {ldc_count}"


def test_mov_immediate_bakes_constant():
    """.nv.constant0 contains the literal 42 (0x2a) at the pool offset."""
    results = compile_ptx_source(IMM_KERNEL)
    cubin = results["imm_kernel"]
    elf = ELF64(cubin)
    const0 = elf.section_data('.nv.constant0.imm_kernel')
    # 42 should appear as a 4-byte LE word somewhere in the constant bank
    assert b'\x2a\x00\x00\x00' in const0, "Literal 42 not found in constant bank"


def test_setp_immediate_compiles():
    """setp.lt.u32 with immediate 100 compiles to LDCU.32 + ISETP R-UR."""
    results = compile_ptx_source(IMM_KERNEL)
    cubin = results["imm_kernel"]
    elf = ELF64(cubin)
    text = elf.section_data('.text.imm_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    # ISETP R-UR opcode 0xc0c must appear
    assert 0xc0c in opcodes, "ISETP R-UR (0xc0c) not found — setp with immediate failed"
    # LDCU.32 opcode 0x7ac must appear (at least one for setp imm, possibly more for params)
    assert 0x7ac in opcodes, "LDCU.32 (0x7ac) not found — setp immediate literal load missing"
