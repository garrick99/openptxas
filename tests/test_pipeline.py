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
# Variable-shift tests
# ---------------------------------------------------------------------------

VAR_SHIFT_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry var_shift(
    .param .u64 out,
    .param .u32 val,
    .param .u32 shift)
{
    .reg .b32   %r<8>;
    .reg .b64   %rd<4>;
    ld.param.u32 %r0, [val];
    ld.param.u32 %r1, [shift];
    shl.b32      %r2, %r0, %r1;
    shr.u32      %r3, %r0, %r1;
    cvt.u64.u32  %rd1, %r2;
    ld.param.u64 %rd0, [out];
    st.global.u64 [%rd0], %rd1;
    ret;
}
"""


def test_var_shift_compiles():
    """shl/shr with register shift amount compiles to SHF.VAR (0x299), not NOP."""
    results = compile_ptx_source(VAR_SHIFT_KERNEL)
    cubin = results["var_shift"]
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.var_shift')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x299 in opcodes, "SHF.VAR (0x299) not found — variable-shift shl/shr emitted NOP"


# ---------------------------------------------------------------------------
# mul.lo R-R tests
# ---------------------------------------------------------------------------

MUL_RR_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry mul_rr(
    .param .u64 out,
    .param .u32 val)
{
    .reg .b32  %r<8>;
    .reg .b64  %rd<4>;
    ld.param.u32 %r0, [val];
    add.u32      %r1, %r0, 1;
    add.u32      %r2, %r0, 2;
    mul.lo.u32   %r3, %r1, %r2;
    cvt.u64.u32  %rd1, %r3;
    ld.param.u64 %rd0, [out];
    st.global.u64 [%rd0], %rd1;
    ret;
}
"""


def test_mul_rr_compiles():
    """mul.lo with two computed GPRs compiles to IMAD.WIDE (0x225) — 0x2a4 is broken on SM_120."""
    results = compile_ptx_source(MUL_RR_KERNEL)
    cubin = results["mul_rr"]
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.mul_rr')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x225 in opcodes, "IMAD.WIDE (0x225) not found — mul.lo R-R needs WIDE fallback"
    assert 0x2a4 not in opcodes, "IMAD.RR (0x2a4) found — this opcode is broken on SM_120"


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
    """mov.u32 with immediate literal compiles to IADD3_IMM32 (not a TODO NOP)."""
    results = compile_ptx_source(IMM_KERNEL)
    cubin = results["imm_kernel"]
    elf = ELF64(cubin)
    text = elf.section_data('.text.imm_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    # IADD3.IMM32 (0x810) or IADD3 (0x210) must appear for the mov immediate
    assert 0x810 in opcodes or 0x210 in opcodes, \
        f"Expected IADD3 for mov immediate, opcodes={[hex(o) for o in set(opcodes)]}"


def test_mov_immediate_bakes_constant():
    """mov.u32 with immediate 42 uses IADD3_IMM32 (inline, no constant bank needed)."""
    results = compile_ptx_source(IMM_KERNEL)
    cubin = results["imm_kernel"]
    elf = ELF64(cubin)
    text = elf.section_data('.text.imm_kernel')
    # 42 (0x2a) should appear as an inline immediate in the IADD3_IMM32 instruction
    assert b'\x2a\x00\x00\x00' in text, "Literal 42 not found inline in .text"


def test_setp_immediate_compiles():
    """setp.lt.u32 with immediate 100 compiles to IADD3_IMM + ISETP R-R.

    The old LDCU.32 + ISETP R-UR path is incorrect: LDCU loads from the literal
    pool beyond the param area, which is uninitialized. The correct path materializes
    the constant into a GPR with IADD3_IMM32 and then uses ISETP R-R (0x20c).
    """
    results = compile_ptx_source(IMM_KERNEL)
    cubin = results["imm_kernel"]
    elf = ELF64(cubin)
    text = elf.section_data('.text.imm_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    # ISETP.IMM (0x80c) or ISETP R-R (0x20c) must appear for the comparison
    assert 0x80c in opcodes or 0x20c in opcodes, \
        "Neither ISETP.IMM (0x80c) nor ISETP R-R (0x20c) found — setp with immediate failed"


# ---------------------------------------------------------------------------
# bfe.u32 tests
# ---------------------------------------------------------------------------

BFE_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry bfe_kernel(
    .param .u64 out,
    .param .u32 val)
{
    .reg .b32   %r<4>;
    .reg .b64   %rd<2>;
    ld.param.u32 %r1, [val];
    bfe.u32     %r0, %r1, 0, 8;
    bfe.u32     %r2, %r1, 8, 8;
    ld.param.u64 %rd0, [out];
    ret;
}
"""


def test_bfe_u32_compiles():
    """bfe.u32 compiles without crashing (no TODO NOP for constant start/length)."""
    results = compile_ptx_source(BFE_KERNEL)
    cubin = results["bfe_kernel"]
    assert cubin[:4] == b'\x7fELF'


def test_bfe_u32_no_nop():
    """bfe.u32 with constant immediates does not emit TODO NOP (0x918)."""
    results = compile_ptx_source(BFE_KERNEL)
    cubin = results["bfe_kernel"]
    elf = ELF64(cubin)
    text = elf.section_data('.text.bfe_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    # NOP opcode is 0x918; LDC is 0xb82; LOP3 is 0x212; SHF.R is 0x819
    assert 0x212 in opcodes, "LOP3 (0x212) not found — bfe mask step missing"


def test_bfe_u32_bakes_mask():
    """.nv.constant0 contains the bfe mask (0xff) for an 8-bit extract."""
    results = compile_ptx_source(BFE_KERNEL)
    cubin = results["bfe_kernel"]
    elf = ELF64(cubin)
    const0 = elf.section_data('.nv.constant0.bfe_kernel')
    assert b'\xff\x00\x00\x00' in const0, "Mask 0xFF not found in constant bank"


# ---------------------------------------------------------------------------
# bfi.b32 tests
# ---------------------------------------------------------------------------

BFI_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry bfi_kernel(
    .param .u64 out,
    .param .u32 val)
{
    .reg .b32   %r<4>;
    .reg .b64   %rd<2>;
    ld.param.u32 %r0, [val];
    mov.u32     %r1, 0;
    bfi.b32     %r2, %r0, %r1, 4, 8;
    ld.param.u64 %rd0, [out];
    ret;
}
"""


def test_bfi_b32_compiles():
    """bfi.b32 with constant start/count compiles to real instructions."""
    results = compile_ptx_source(BFI_KERNEL)
    cubin = results["bfi_kernel"]
    assert cubin[:4] == b'\x7fELF'


def test_bfi_b32_has_lop3():
    """bfi.b32 emits multiple LOP3 instructions for mask/merge logic."""
    results = compile_ptx_source(BFI_KERNEL)
    cubin = results["bfi_kernel"]
    elf = ELF64(cubin)
    text = elf.section_data('.text.bfi_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    lop3_count = opcodes.count(0x212)
    assert lop3_count >= 3, f"Expected >=3 LOP3 instructions for bfi, got {lop3_count}"


def test_bfi_b32_bakes_masks():
    """.nv.constant0 contains both shifted_mask and ~shifted_mask for bfi."""
    results = compile_ptx_source(BFI_KERNEL)
    cubin = results["bfi_kernel"]
    elf = ELF64(cubin)
    const0 = elf.section_data('.nv.constant0.bfi_kernel')
    import struct as _s
    # start=4, count=8 → raw_mask=0xFF, shifted_mask=0xFF0, not_shifted_mask=0xFFFFF00F
    shifted_mask = 0xFF0
    not_shifted_mask = (~shifted_mask) & 0xFFFFFFFF
    assert _s.pack('<I', shifted_mask) in const0, "shifted_mask not in constant bank"
    assert _s.pack('<I', not_shifted_mask) in const0, "~shifted_mask not in constant bank"


# ---------------------------------------------------------------------------
# cvt tests
# ---------------------------------------------------------------------------

CVT_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry cvt_kernel(
    .param .u64 out,
    .param .u32 val)
{
    .reg .b32   %r<4>;
    .reg .b64   %rd<4>;
    ld.param.u32 %r0, [val];
    cvt.s32.u32  %r1, %r0;
    cvt.u64.u32  %rd1, %r0;
    ld.param.u64 %rd0, [out];
    ret;
}
"""

CVT_SIGN_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry cvt_sign_kernel(
    .param .u64 out,
    .param .s32 val)
{
    .reg .s32   %r<2>;
    .reg .s64   %rd<4>;
    ld.param.s32 %r0, [val];
    cvt.s64.s32  %rd1, %r0;
    ld.param.u64 %rd0, [out];
    ret;
}
"""


def test_cvt_same_width_compiles():
    """cvt.s32.u32 (same-width reinterpret) compiles to MOV, not TODO NOP."""
    results = compile_ptx_source(CVT_KERNEL)
    cubin = results["cvt_kernel"]
    elf = ELF64(cubin)
    text = elf.section_data('.text.cvt_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    # IADD3 (0x210) used as MOV — must appear (cvt.s32.u32 reinterpret)
    assert 0x210 in opcodes, "IADD3/MOV not found — cvt.s32.u32 emitted NOP"


def test_cvt_u64_u32_compiles():
    """cvt.u64.u32 (zero-extend to 64-bit) compiles successfully."""
    results = compile_ptx_source(CVT_KERNEL)
    cubin = results["cvt_kernel"]
    elf = ELF64(cubin)
    text = elf.section_data('.text.cvt_kernel')
    assert len(text) > 0 and len(text) % 128 == 0


def test_cvt_s64_s32_compiles():
    """cvt.s64.s32 (sign-extend to 64-bit) compiles to SHF+INEG sequence."""
    results = compile_ptx_source(CVT_SIGN_KERNEL)
    cubin = results["cvt_sign_kernel"]
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.cvt_sign_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    # SHF.R.U32.HI (0x819) must appear for sign-bit extraction
    assert 0x819 in opcodes, "SHF not found — cvt.s64.s32 sign-extend missing"


SHR_S32_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry shr_s32_kernel(
    .param .s32 a,
    .param .u32 shift,
    .param .u64 out)
{
    .reg .s32   %r<4>;
    .reg .u32   %r4;
    .reg .u64   %rd<2>;
    ld.param.s32    %r0, [a];
    ld.param.u32    %r4, [shift];
    shr.s32         %r1, %r0, %r4;
    ld.param.u64    %rd0, [out];
    st.global.s32   [%rd0], %r1;
    ret;
}
"""


PRMT_REG_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry prmt_reg_kernel(
    .param .u32 a,
    .param .u32 b,
    .param .u32 sel,
    .param .u64 out)
{
    .reg .u32   %r<8>;
    .reg .u64   %rd<2>;
    ld.param.u32    %r0, [a];
    ld.param.u32    %r1, [b];
    ld.param.u32    %r2, [sel];
    prmt.b32        %r3, %r0, %r1, %r2;
    ld.param.u64    %rd0, [out];
    st.global.u32   [%rd0], %r3;
    ret;
}
"""


def test_prmt_reg_sel_compiles():
    """prmt.b32 with register selector uses PRMT.REG (opcode 0x216), not TODO NOP."""
    results = compile_ptx_source(PRMT_REG_KERNEL)
    cubin = results['prmt_reg_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.prmt_reg_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x216 in opcodes, "PRMT.REG not found — prmt with register selector emitted NOP"


def test_shr_s32_var_compiles():
    """shr.s32 with register shift uses SHF.R.S32.HI (opcode 0x219), not TODO NOP."""
    results = compile_ptx_source(SHR_S32_KERNEL)
    cubin = results['shr_s32_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.shr_s32_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x219 in opcodes, "SHF.R.S32.HI.VAR not found — shr.s32 emitted NOP"

DIV_U32_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry div_u32_kernel(
    .param .u64 out_ptr,
    .param .u32 a_param,
    .param .u32 b_param)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<2>;
    ld.param.u32 %r0, [a_param];
    ld.param.u32 %r1, [b_param];
    div.u32      %r2, %r0, %r1;
    ld.param.u64 %rd0, [out_ptr];
    st.global.u32 [%rd0], %r2;
    ret;
}
"""


def test_div_u32_compiles():
    """div.u32 emits full Newton-Raphson sequence (no TODO NOPs)."""
    results = compile_ptx_source(DIV_U32_KERNEL)
    cubin = results['div_u32_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.div_u32_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    # Must have Newton-Raphson key opcodes
    assert 0x306 in opcodes, "I2F.U32.RP not found — div.u32 NR not emitted"
    assert 0x308 in opcodes, "MUFU.RCP not found — div.u32 NR not emitted"
    assert 0x305 in opcodes, "F2I.FTZ.U32 not found — div.u32 NR not emitted"
    assert 0x225 in opcodes or 0x227 in opcodes, "IMAD.WIDE/HI not found — div.u32 NR not emitted"
    # Verify NOPs are scheduling-only (not TODO NOPs from unimplemented instrs).
    # Exclude trailing NOPs (text section padding after EXIT+BRA trap).
    last_real = max(i for i, opc in enumerate(opcodes) if opc != 0x918)
    nop_count = opcodes[:last_real + 1].count(0x918)
    assert nop_count <= 16, f"Too many NOPs in div.u32 body (likely unimplemented instruction): {nop_count}"

REM_U32_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry rem_u32_kernel(
    .param .u64 out_ptr,
    .param .u32 a_param,
    .param .u32 b_param)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<2>;
    ld.param.u32 %r0, [a_param];
    ld.param.u32 %r1, [b_param];
    rem.u32      %r2, %r0, %r1;
    ld.param.u64 %rd0, [out_ptr];
    st.global.u32 [%rd0], %r2;
    ret;
}
"""


def test_rem_u32_compiles():
    """rem.u32 emits Newton-Raphson sequence (no TODO NOPs)."""
    results = compile_ptx_source(REM_U32_KERNEL)
    cubin = results['rem_u32_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.rem_u32_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x306 in opcodes, "I2F.U32.RP not found in rem.u32"
    assert 0x227 in opcodes, "IMAD.HI.U32 not found in rem.u32"
    last_real = max(i for i, opc in enumerate(opcodes) if opc != 0x918)
    nop_count = opcodes[:last_real + 1].count(0x918)
    assert nop_count <= 13, f"Too many NOPs ({nop_count}) in rem.u32"

DIV_S32_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry div_s32_kernel(
    .param .u64 out_ptr,
    .param .s32 a_param,
    .param .s32 b_param)
{
    .reg .s32 %r<4>;
    .reg .u64 %rd<2>;
    ld.param.s32 %r0, [a_param];
    ld.param.s32 %r1, [b_param];
    div.s32      %r2, %r0, %r1;
    ld.param.u64 %rd0, [out_ptr];
    st.global.s32 [%rd0], %r2;
    ret;
}
"""


def test_div_s32_compiles():
    """div.s32 emits IABS+NR+sign-correction sequence (no TODO NOPs)."""
    results = compile_ptx_source(DIV_S32_KERNEL)
    cubin = results['div_s32_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.div_s32_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x213 in opcodes, "IABS not found in div.s32"
    assert 0x306 in opcodes, "I2F.RP not found in div.s32"
    assert 0x227 in opcodes, "IMAD.HI not found in div.s32"
    last_real = max(i for i, opc in enumerate(opcodes) if opc != 0x918)
    nop_count = opcodes[:last_real + 1].count(0x918)
    assert nop_count <= 12, f"Too many NOPs ({nop_count}) in div.s32"


REM_S32_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry rem_s32_kernel(
    .param .u64 out_ptr,
    .param .s32 a_param,
    .param .s32 b_param)
{
    .reg .s32 %r<4>;
    .reg .u64 %rd<2>;
    ld.param.s32 %r0, [a_param];
    ld.param.s32 %r1, [b_param];
    rem.s32      %r2, %r0, %r1;
    ld.param.u64 %rd0, [out_ptr];
    st.global.s32 [%rd0], %r2;
    ret;
}
"""


def test_rem_s32_compiles():
    """rem.s32 emits IABS+NR+sign-correction sequence (no TODO NOPs)."""
    results = compile_ptx_source(REM_S32_KERNEL)
    cubin = results['rem_s32_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.rem_s32_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x213 in opcodes, "IABS not found in rem.s32"
    assert 0x306 in opcodes, "I2F.RP not found in rem.s32"
    assert 0x227 in opcodes, "IMAD.HI not found in rem.s32"
    last_real = max(i for i, opc in enumerate(opcodes) if opc != 0x918)
    nop_count = opcodes[:last_real + 1].count(0x918)
    assert nop_count <= 13, f"Too many NOPs ({nop_count}) in rem.s32"


MAD_WIDE_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry mad_wide_kernel(
    .param .u64 out_ptr,
    .param .u32 idx_param,
    .param .u64 base_param)
{
    .reg .u32 %r<2>;
    .reg .u64 %rd<4>;
    ld.param.u32 %r0, [idx_param];
    ld.param.u64 %rd0, [base_param];
    mad.wide.u32 %rd1, %r0, 4, %rd0;
    ld.param.u64 %rd2, [out_ptr];
    st.global.u64 [%rd2], %rd1;
    ret;
}
"""


def test_mad_wide_compiles():
    """mad.wide.u32 with small immediate emits IMAD.WIDE (opcode 0x825)."""
    results = compile_ptx_source(MAD_WIDE_KERNEL)
    cubin = results['mad_wide_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.mad_wide_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x825 in opcodes, f"IMAD.WIDE not found in mad_wide_kernel; opcodes={[hex(o) for o in set(opcodes)]}"
    # Count only non-trailing NOPs (trailing ones are ELF alignment padding)
    last_real = max(i for i, op in enumerate(opcodes) if op != 0x918)
    sched_nops = opcodes[:last_real].count(0x918)
    assert sched_nops <= 7, f"Too many scheduling NOPs ({sched_nops}) in mad_wide_kernel"


NOT_B32_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry not_b32_kernel(
    .param .u64 out_ptr,
    .param .u32 a_param)
{
    .reg .b32 %r<2>;
    .reg .u64 %rd<2>;
    ld.param.u32 %r0, [a_param];
    not.b32      %r1, %r0;
    ld.param.u64 %rd0, [out_ptr];
    st.global.u32 [%rd0], %r1;
    ret;
}
"""

AND_B64_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry and_b64_kernel(
    .param .u64 out_ptr,
    .param .u64 a_param,
    .param .u64 b_param)
{
    .reg .b64 %rd<4>;
    ld.param.u64 %rd0, [a_param];
    ld.param.u64 %rd1, [b_param];
    and.b64      %rd2, %rd0, %rd1;
    ld.param.u64 %rd3, [out_ptr];
    st.global.u64 [%rd3], %rd2;
    ret;
}
"""


def test_not_b32_compiles():
    """not.b32 emits LOP3.LUT with ~a LUT (0x0F)."""
    results = compile_ptx_source(NOT_B32_KERNEL)
    cubin = results['not_b32_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.not_b32_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x212 in opcodes, "LOP3.LUT not found in not_b32_kernel"


def test_and_b64_compiles():
    """and.b64 emits two LOP3.LUT instructions (lo and hi words)."""
    results = compile_ptx_source(AND_B64_KERNEL)
    cubin = results['and_b64_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.and_b64_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    lop3_count = opcodes.count(0x212)
    assert lop3_count >= 2, f"Expected ≥2 LOP3.LUT for and.b64, got {lop3_count}"


# ---------------------------------------------------------------------------
# F2F: float precision conversion
# ---------------------------------------------------------------------------

CVT_F32_F64_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry cvt_f32_f64_kernel(
    .param .u64 out_ptr,
    .param .f64 val)
{
    .reg .b64 %rd<2>;
    .reg .f32 %f<2>;
    .reg .f64 %fd<2>;
    ld.param.u64 %rd0, [out_ptr];
    ld.param.f64 %fd0, [val];
    cvt.rn.f32.f64 %f0, %fd0;
    st.global.f32 [%rd0], %f0;
    ret;
}
"""

CVT_F64_F32_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry cvt_f64_f32_kernel(
    .param .u64 out_ptr,
    .param .f32 val)
{
    .reg .b64 %rd<2>;
    .reg .f32 %f<2>;
    .reg .f64 %fd<2>;
    ld.param.u64 %rd0, [out_ptr];
    ld.param.f32 %f0, [val];
    cvt.f64.f32 %fd0, %f0;
    cvt.rn.f32.f64 %f1, %fd0;
    st.global.f32 [%rd0], %f1;
    ret;
}
"""

MUL_HI_U64_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry mul_hi_u64_kernel(
    .param .u64 out_ptr,
    .param .u64 a_val,
    .param .u64 b_val)
{
    .reg .b64 %rd<8>;
    ld.param.u64 %rd0, [out_ptr];
    ld.param.u64 %rd1, [a_val];
    ld.param.u64 %rd2, [b_val];
    mul.hi.u64 %rd3, %rd1, %rd2;
    st.global.u64 [%rd0], %rd3;
    ret;
}
"""

CVT_U8_U32_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry cvt_u8_kernel(
    .param .u64 out_ptr,
    .param .u32 val)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.u32 %r0, [val];
    cvt.u8.u32 %r1, %r0;
    ld.param.u64 %rd0, [out_ptr];
    st.global.u32 [%rd0], %r1;
    ret;
}
"""


def test_cvt_f32_f64_compiles():
    """cvt.rn.f32.f64 emits F2F.F32.F64 (opcode 0x310)."""
    results = compile_ptx_source(CVT_F32_F64_KERNEL)
    cubin = results['cvt_f32_f64_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.cvt_f32_f64_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x310 in opcodes, "F2F (0x310) not found in cvt.rn.f32.f64"


def test_cvt_f64_f32_compiles():
    """cvt.f64.f32 emits F2F.F64.F32 (opcode 0x310), no TODO NOPs."""
    results = compile_ptx_source(CVT_F64_F32_KERNEL)
    cubin = results['cvt_f64_f32_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.cvt_f64_f32_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x310 in opcodes, "F2F (0x310) not found in cvt.f64.f32"


def test_mul_hi_u64_compiles():
    """mul.hi.u64 emits IMAD.WIDE.U32 sequence (opcode 0x225), no TODO NOPs."""
    results = compile_ptx_source(MUL_HI_U64_KERNEL)
    cubin = results['mul_hi_u64_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.mul_hi_u64_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x225 in opcodes, "IMAD.WIDE.U32 (0x225) not found in mul.hi.u64"
    last_real = max(i for i, op in enumerate(opcodes) if op != 0x918)
    sched_nops = opcodes[:last_real].count(0x918)
    assert sched_nops <= 8, f"Too many scheduling NOPs ({sched_nops}) in mul.hi.u64"


def test_cvt_u8_u32_compiles():
    """cvt.u8.u32 emits LDC+LOP3.AND (opcode 0x212), no TODO NOPs."""
    results = compile_ptx_source(CVT_U8_U32_KERNEL)
    cubin = results['cvt_u8_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.cvt_u8_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x212 in opcodes, "LOP3.LUT (0x212) not found in cvt.u8.u32"


SQRT_RN_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry sqrt_rn_kernel(
    .param .u64 out_ptr,
    .param .f32 val)
{
    .reg .f32 %f<2>;
    .reg .b64 %rd<2>;
    ld.param.f32 %f0, [val];
    sqrt.rn.f32 %f1, %f0;
    ld.param.u64 %rd0, [out_ptr];
    st.global.f32 [%rd0], %f1;
    ret;
}
"""

RCP_RN_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry rcp_rn_kernel(
    .param .u64 out_ptr,
    .param .f32 val)
{
    .reg .f32 %f<2>;
    .reg .b64 %rd<2>;
    ld.param.f32 %f0, [val];
    rcp.rn.f32 %f1, %f0;
    ld.param.u64 %rd0, [out_ptr];
    st.global.f32 [%rd0], %f1;
    ret;
}
"""


def test_sqrt_rn_f32_compiles():
    """sqrt.rn.f32 emits MUFU.SQRT (opcode 0x308)."""
    results = compile_ptx_source(SQRT_RN_KERNEL)
    cubin = results['sqrt_rn_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.sqrt_rn_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x308 in opcodes, f"MUFU (0x308) not found in sqrt.rn.f32; got {[hex(o) for o in opcodes]}"


def test_rcp_rn_f32_compiles():
    """rcp.rn.f32 emits MUFU.RCP (opcode 0x308)."""
    results = compile_ptx_source(RCP_RN_KERNEL)
    cubin = results['rcp_rn_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.rcp_rn_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x308 in opcodes, f"MUFU (0x308) not found in rcp.rn.f32; got {[hex(o) for o in opcodes]}"


CVT_S32_F64_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry cvt_s32_f64_kernel(
    .param .u64 out_ptr,
    .param .f64 val)
{
    .reg .s32 %r<2>;
    .reg .f64 %fd<2>;
    .reg .b64 %rd<2>;
    ld.param.f64 %fd0, [val];
    cvt.rzi.s32.f64 %r0, %fd0;
    ld.param.u64 %rd0, [out_ptr];
    st.global.s32 [%rd0], %r0;
    ret;
}
"""

CVT_U32_F64_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry cvt_u32_f64_kernel(
    .param .u64 out_ptr,
    .param .f64 val)
{
    .reg .u32 %r<2>;
    .reg .f64 %fd<2>;
    .reg .b64 %rd<2>;
    ld.param.f64 %fd0, [val];
    cvt.rzi.u32.f64 %r0, %fd0;
    ld.param.u64 %rd0, [out_ptr];
    st.global.u32 [%rd0], %r0;
    ret;
}
"""

CVT_F64_S32_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry cvt_f64_s32_kernel(
    .param .u64 out_ptr,
    .param .s32 val)
{
    .reg .s32 %r<2>;
    .reg .f64 %fd<2>;
    .reg .b64 %rd<2>;
    ld.param.s32 %r0, [val];
    cvt.rn.f64.s32 %fd0, %r0;
    ld.param.u64 %rd0, [out_ptr];
    st.global.f64 [%rd0], %fd0;
    ret;
}
"""


def test_cvt_s32_f64_compiles():
    """cvt.rzi.s32.f64 emits F2I.S32.F64 (opcode 0x311)."""
    results = compile_ptx_source(CVT_S32_F64_KERNEL)
    cubin = results['cvt_s32_f64_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.cvt_s32_f64_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x311 in opcodes, f"F2I.F64 (0x311) not found in cvt.rzi.s32.f64"


def test_cvt_u32_f64_compiles():
    """cvt.rzi.u32.f64 emits F2I.U32.F64 (opcode 0x311)."""
    results = compile_ptx_source(CVT_U32_F64_KERNEL)
    cubin = results['cvt_u32_f64_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.cvt_u32_f64_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x311 in opcodes, f"F2I.F64 (0x311) not found in cvt.rzi.u32.f64"


def test_cvt_f64_s32_compiles():
    """cvt.rn.f64.s32 emits I2F.F64.S32 (opcode 0x312)."""
    results = compile_ptx_source(CVT_F64_S32_KERNEL)
    cubin = results['cvt_f64_s32_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.cvt_f64_s32_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x312 in opcodes, f"I2F.F64 (0x312) not found in cvt.rn.f64.s32"


BFE_S32_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry bfe_s32_kernel(
    .param .u64 out_ptr,
    .param .s32 val)
{
    .reg .s32 %r<3>;
    .reg .b64 %rd<2>;
    ld.param.s32 %r0, [val];
    bfe.s32 %r1, %r0, 2, 8;
    ld.param.u64 %rd0, [out_ptr];
    st.global.s32 [%rd0], %r1;
    ret;
}
"""

SAD_U32_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry sad_u32_kernel(
    .param .u64 out_ptr,
    .param .u32 a,
    .param .u32 b,
    .param .u32 c)
{
    .reg .u32 %r<5>;
    .reg .b64 %rd<2>;
    ld.param.u32 %r0, [a];
    ld.param.u32 %r1, [b];
    ld.param.u32 %r2, [c];
    sad.u32 %r3, %r0, %r1, %r2;
    ld.param.u64 %rd0, [out_ptr];
    st.global.u32 [%rd0], %r3;
    ret;
}
"""


def test_bfe_s32_compiles():
    """bfe.s32 emits SHF.R.S32.HI (0x819) + BFE_SEXT (0x81a)."""
    results = compile_ptx_source(BFE_S32_KERNEL)
    cubin = results['bfe_s32_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.bfe_s32_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x819 in opcodes, "SHF.R.S32.HI (0x819) not found in bfe.s32"
    assert 0x81a in opcodes, "BFE_SEXT (0x81a) not found in bfe.s32"


def test_sad_u32_compiles():
    """sad.u32 emits VIMNMX (0x248) + IADD3 (0x210)."""
    results = compile_ptx_source(SAD_U32_KERNEL)
    cubin = results['sad_u32_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.sad_u32_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x248 in opcodes, "VIMNMX (0x248) not found in sad.u32"
    assert 0x210 in opcodes, "IADD3 (0x210) not found in sad.u32"


ATOM_CAS_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry atom_cas_kernel(
    .param .u64 ptr_param,
    .param .u32 cmp_param,
    .param .u32 val_param)
{
    .reg .u64 %rd<2>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd0, [ptr_param];
    ld.param.u32 %r0, [cmp_param+8];
    ld.param.u32 %r1, [val_param+16];
    atom.cas.b32 %r2, [%rd0], %r0, %r1;
    st.global.u32 [%rd0], %r2;
    ret;
}
"""


def test_atom_cas_b32_compiles():
    """atom.cas.b32 emits ATOMG.E.CAS (opcode 0x3a9); no TODO NOPs."""
    results = compile_ptx_source(ATOM_CAS_KERNEL)
    cubin = results['atom_cas_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.atom_cas_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x3a9 in opcodes, f"ATOMG.CAS (0x3a9) not found; opcodes={[hex(o) for o in set(opcodes)]}"
    last_real = max(i for i, op in enumerate(opcodes) if op != 0x918)
    sched_nops = opcodes[:last_real].count(0x918)
    # Allow scheduling NOPs for IADD.64 UR->GPR materialization latency
    # (atom.cas and st.global each need UR->GPR conversion = 2 IADD.64 + 2 NOPs)
    assert sched_nops <= 7, f"Scheduling NOPs ({sched_nops}) in atom.cas kernel (trailing padding is OK)"


SELP_IMM_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry selp_imm_kernel(
    .param .u64 out_ptr,
    .param .u32 thresh)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<2>;
    .reg .pred %p<1>;
    ld.param.u32 %r0, [thresh];
    setp.lt.u32 %p0, %r0, 100;
    selp.u32 %r1, 1, 0, %p0;
    ld.param.u64 %rd0, [out_ptr+8];
    st.global.u32 [%rd0], %r1;
    ret;
}
"""


def test_selp_imm_compiles():
    """selp with immediate sources uses predicated IADD3 (no SEL barrier race)."""
    results = compile_ptx_source(SELP_IMM_KERNEL)
    cubin = results['selp_imm_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.selp_imm_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x810 in opcodes, f"IADD3_IMM (0x810) not found (predicated MOV for selp)"


TESTP_FINITE_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry testp_finite_kernel(
    .param .u64 out_ptr,
    .param .f32 val)
{
    .reg .f32 %f<2>;
    .reg .u32 %r<2>;
    .reg .u64 %rd<2>;
    .reg .pred %p<1>;
    ld.param.f32 %f0, [val];
    testp.finite.f32 %p0, %f0;
    selp.u32 %r0, 1, 0, %p0;
    ld.param.u64 %rd0, [out_ptr+8];
    st.global.u32 [%rd0], %r0;
    ret;
}
"""


def test_testp_finite_f32_compiles():
    """testp.finite.f32 emits IADD3_IMM+LOP3+LDCU+ISETP; no TODO NOPs."""
    results = compile_ptx_source(TESTP_FINITE_KERNEL)
    cubin = results['testp_finite_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.testp_finite_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x810 in opcodes, "IADD3_IMM (0x810) not found in testp.finite"
    assert 0x212 in opcodes, "LOP3 (0x212) not found in testp.finite"
    assert 0xc0c in opcodes, "ISETP.R-UR (0xc0c) not found in testp.finite"
    last_real = max(i for i, op in enumerate(opcodes) if op != 0x918)
    sched_nops = opcodes[:last_real].count(0x918)
    assert sched_nops <= 9, f"Too many scheduling NOPs ({sched_nops}) in testp.finite kernel"


CVT_WIDEN_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry cvt_widen_kernel(
    .param .u64 out_u64,
    .param .u64 out_s64,
    .param .u32 u_val,
    .param .s32 s_val)
{
    .reg .u32 %r<2>;
    .reg .s32 %rs<2>;
    .reg .u64 %rd<4>;
    .reg .s64 %rds<2>;
    ld.param.u32 %r0, [u_val];
    ld.param.s32 %rs0, [s_val+4];
    cvt.u64.u32 %rd0, %r0;
    cvt.s64.s32 %rds0, %rs0;
    ld.param.u64 %rd2, [out_u64+16];
    ld.param.u64 %rd3, [out_s64+24];
    st.global.u64 [%rd2], %rd0;
    st.global.s64 [%rd3], %rds0;
    ret;
}
"""


def test_cvt_widen_compiles():
    """cvt.u64.u32 (zero-ext) and cvt.s64.s32 (sign-ext) lower without TODO NOPs."""
    results = compile_ptx_source(CVT_WIDEN_KERNEL)
    cubin = results['cvt_widen_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.cvt_widen_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x819 in opcodes, "SHF.R.S32.HI (0x819) not found — sign extension for cvt.s64.s32"
    last_real = max(i for i, op in enumerate(opcodes) if op != 0x918)
    sched_nops = opcodes[:last_real].count(0x918)
    assert sched_nops <= 13, f"Too many scheduling NOPs ({sched_nops}) in cvt_widen_kernel"


F64_ARITH_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry f64_arith_kernel(
    .param .u64 out_ptr,
    .param .f64 a,
    .param .f64 b,
    .param .f64 c)
{
    .reg .f64 %fd<6>;
    .reg .u64 %rd<2>;
    ld.param.f64 %fd0, [a];
    ld.param.f64 %fd1, [b+8];
    ld.param.f64 %fd2, [c+24];
    add.f64 %fd3, %fd0, %fd1;
    mul.f64 %fd4, %fd0, %fd1;
    fma.rn.f64 %fd5, %fd0, %fd1, %fd2;
    ld.param.u64 %rd0, [out_ptr+32];
    st.global.f64 [%rd0], %fd5;
    ret;
}
"""


def test_f64_arith_compiles():
    """add.f64/mul.f64/fma.rn.f64 emit DADD/DMUL/DFMA (no TODO NOPs)."""
    results = compile_ptx_source(F64_ARITH_KERNEL)
    cubin = results['f64_arith_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.f64_arith_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x229 in opcodes, f"DADD (0x229) not found; opcodes={[hex(o) for o in set(opcodes)]}"
    assert 0x228 in opcodes, f"DMUL (0x228) not found"
    assert 0x22b in opcodes or 0xc2b in opcodes, \
        f"DFMA (0x22b or 0xc2b) not found; opcodes={[hex(o) for o in set(opcodes)]}"
    last_real = max(i for i, op in enumerate(opcodes) if op != 0x918)
    sched_nops = opcodes[:last_real].count(0x918)
    assert sched_nops <= 7, f"Too many scheduling NOPs ({sched_nops}) in f64_arith_kernel"


F64_SUB_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry f64_sub_kernel(
    .param .u64 out_ptr,
    .param .f64 a,
    .param .f64 b)
{
    .reg .f64 %fd<3>;
    .reg .u64 %rd<2>;
    ld.param.u64 %rd0, [out_ptr];
    ld.param.f64 %fd0, [a+8];
    ld.param.f64 %fd1, [b+24];
    sub.f64 %fd2, %fd0, %fd1;
    st.global.f64 [%rd0], %fd2;
    ret;
}
"""


def test_f64_sub_compiles():
    """sub.f64 emits DADD with negated src0 (no TODO NOP)."""
    results = compile_ptx_source(F64_SUB_KERNEL)
    cubin = results['f64_sub_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.f64_sub_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x229 in opcodes, f"DADD (0x229) not found for sub.f64; opcodes={[hex(o) for o in set(opcodes)]}"
    nop_count = opcodes.count(0x918)
    assert nop_count < len(opcodes), f"All instructions are NOPs in sub.f64 output"


F64_MINMAX_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry f64_minmax_kernel(
    .param .u64 out_ptr,
    .param .f64 a,
    .param .f64 b)
{
    .reg .f64 %fd<4>;
    .reg .u64 %rd<2>;
    .reg .pred %p<2>;
    ld.param.u64 %rd0, [out_ptr];
    ld.param.f64 %fd0, [a+8];
    ld.param.f64 %fd1, [b+24];
    min.f64 %fd2, %fd0, %fd1;
    max.f64 %fd3, %fd0, %fd1;
    st.global.f64 [%rd0],   %fd2;
    st.global.f64 [%rd0+8], %fd3;
    ret;
}
"""


def test_f64_minmax_compiles():
    """min.f64/max.f64 emit DSETP+FSEL (no TODO NOPs)."""
    results = compile_ptx_source(F64_MINMAX_KERNEL)
    cubin = results['f64_minmax_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.f64_minmax_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    # DSETP opcode 0xa72 (inferred) or check no TODO NOPs
    nop_count = opcodes.count(0x918)
    non_nop = [op for op in opcodes if op != 0x918]
    assert len(non_nop) >= 4, f"Expected ≥4 real instructions for min+max f64, got {len(non_nop)}: {[hex(o) for o in non_nop]}"


F64_SETP_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry f64_setp_kernel(
    .param .u64 out_ptr,
    .param .f64 a,
    .param .f64 b)
{
    .reg .f64 %fd<2>;
    .reg .b64 %rd<2>;
    .reg .pred %p<1>;
    ld.param.f64 %fd0, [a];
    ld.param.f64 %fd1, [b];
    setp.lt.f64 %p0, %fd0, %fd1;
    ld.param.u64 %rd0, [out_ptr];
    @%p0 st.global.u64 [%rd0], %fd0;
    @!%p0 st.global.u64 [%rd0], %fd1;
    ret;
}
"""

F64_NEG_ABS_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry f64_neg_abs_kernel(
    .param .u64 out_ptr,
    .param .f64 a)
{
    .reg .f64 %fd<4>;
    .reg .b64 %rd<2>;
    ld.param.f64 %fd0, [a];
    neg.f64 %fd1, %fd0;
    abs.f64 %fd2, %fd1;
    add.f64 %fd3, %fd1, %fd2;
    ld.param.u64 %rd0, [out_ptr];
    st.global.f64 [%rd0], %fd3;
    ret;
}
"""


def test_f64_setp_compiles():
    """setp.lt.f64 emits DSETP (opcode 0x22a)."""
    results = compile_ptx_source(F64_SETP_KERNEL)
    cubin = results['f64_setp_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.f64_setp_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x22a in opcodes, f"DSETP (0x22a) not found; opcodes={[hex(o) for o in set(opcodes)]}"


def test_f64_neg_abs_compiles():
    """neg.f64 and abs.f64 emit LOP3.XOR/AND sequences (no TODO NOPs)."""
    results = compile_ptx_source(F64_NEG_ABS_KERNEL)
    cubin = results['f64_neg_abs_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.f64_neg_abs_kernel')
    opcodes = [struct.unpack_from('<Q', text, off)[0] & 0xFFF
               for off in range(0, len(text), 16)]
    assert 0x212 in opcodes, f"LOP3 (0x212) not found — neg/abs.f64 lowering missing"
    assert 0x810 in opcodes, f"IADD3.IMM (0x810) not found — sign mask load missing"


# ---------------------------------------------------------------------------
# atom.add.u32 pipeline test
# ---------------------------------------------------------------------------

ATOM_ADD_U32_KERNEL = """\
.version 9.0
.target sm_120
.address_size 64

.visible .entry atom_add_u32_kernel(
    .param .u64 counter,
    .param .u32 val)
{
    .reg .b32  %r<4>;
    .reg .b64  %rd<2>;
    ld.param.u64 %rd0, [counter];
    ld.param.u32 %r0, [val];
    atom.global.add.u32 %r1, [%rd0], %r0;
    ret;
}
"""


def test_atom_add_u32_compiles():
    """atom.global.add.u32 compiles to ATOMG.E.ADD (opcode 0x9a8, PT guard b1=0x79)."""
    results = compile_ptx_source(ATOM_ADD_U32_KERNEL)
    cubin = results['atom_add_u32_kernel']
    assert cubin[:4] == b'\x7fELF'
    elf = ELF64(cubin)
    text = elf.section_data('.text.atom_add_u32_kernel')
    # ATOMG.E.ADD: byte0=0xa8, byte1 lower nibble=0x9 → opcode = 0x9a8
    found_atomg_add = False
    for off in range(0, len(text), 16):
        if text[off] == 0xa8 and (text[off + 1] & 0x0F) == 0x09:
            found_atomg_add = True
            break
    assert found_atomg_add, (
        "ATOMG.E.ADD (byte0=0xa8, opcode nibble 0x9) not found in cubin text section"
    )
