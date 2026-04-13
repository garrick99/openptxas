"""Extract PTXAS-faithful templates from compiled cubins.

Supported domain: atom.global.xor.b32 with uniform SR-derived data,
kernel signature (.u64 p_out, .u32 n).
"""
from __future__ import annotations

import struct
from .spec import TemplateSpec, TemplateInstruction, ParamField

# Opcode semantic labels for the atom.xor activation region.
_ROLE_TABLE = {
    0xb82: "S2R",
    0x919: "S2UR",
    0x7ac: "LDCU",
    0xc0c: "ISETP_RUR",
    0x94d: "EXIT",
    0x3c4: "UMOV",
    0x886: "UR_PIPE_INIT",
    0x2bd: "UR_PIPE_FINAL",
    0xc02: "MOV_UR",
    0x835: "UIADD",
    0x98e: "ATOMG_XOR",
    0x947: "BRA",
    0x918: "NOP",
}


def _get_opcode(raw: bytes) -> int:
    return int.from_bytes(raw[:2], "little") & 0xFFF


def _iter_text_instructions(cubin: bytes):
    """Yield (index, raw_16_bytes) for every instruction in the first .text section."""
    e_shoff = struct.unpack_from("<Q", cubin, 40)[0]
    e_shnum = struct.unpack_from("<H", cubin, 60)[0]
    e_shstrndx = struct.unpack_from("<H", cubin, 62)[0]
    stoff = struct.unpack_from("<Q", cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    for i in range(e_shnum):
        base = e_shoff + i * 64
        nm = struct.unpack_from("<I", cubin, base)[0]
        ne = cubin.index(0, stoff + nm)
        name = cubin[stoff + nm : ne]
        if not name.startswith(b".text."):
            continue
        off = struct.unpack_from("<Q", cubin, base + 24)[0]
        sz = struct.unpack_from("<Q", cubin, base + 32)[0]
        for j in range(0, sz, 16):
            raw = cubin[off + j : off + j + 16]
            if len(raw) == 16:
                yield j // 16, bytes(raw)
        return  # first .text section only


def _refine_role(opc: int, raw: bytes, seq_idx: int, has_uiadd: bool) -> str:
    """Produce a more specific role label from opcode + context."""
    base = _ROLE_TABLE.get(opc, f"UNK_0x{opc:03x}")
    if opc == 0x919:
        ur_dest = raw[2]
        sr_src = raw[9]
        if ur_dest == 0 and sr_src == 0x21:
            return "S2UR_UR0_TID"
        if ur_dest == 2 and sr_src == 0x00:
            return "S2UR_UR2_LANEID"
        return f"S2UR_UR{ur_dest}_SR{sr_src:#x}"
    if opc == 0x7ac:
        ur_dest = raw[2]
        offset = raw[5]
        if offset == 0x6b:
            return f"LDCU_UR{ur_dest}_DESC"
        return f"LDCU_UR{ur_dest}_PARAM_0x{offset:02x}"
    if opc == 0xb82:
        dest = raw[2]
        offset = raw[5]
        if dest == 1:
            return "S2R_R1_PREAMBLE"
        return f"S2R_R{dest}_CBANK_0x{offset:02x}"
    if opc == 0xc0c:
        return "ISETP_RUR_BOUNDS" if seq_idx < 5 else "ISETP_RUR_FLUSH"
    if opc == 0x3c4:
        return f"UMOV_UR{raw[2]}_UR{raw[3]}"
    if opc == 0xc02:
        return f"MOV_UR_R{raw[2]}_UR{raw[4]}"
    if opc == 0x835:
        imm = raw[4] | (raw[5] << 8) | (raw[6] << 16)
        return f"UIADD_K{imm}"
    if opc == 0x98e:
        return f"ATOMG_XOR_R{raw[2]}_R{raw[3]}_data{raw[4]}_desc_UR{raw[8]}"
    return base


def extract_atom_xor_template(cubin: bytes) -> TemplateSpec:
    """Extract an atom.xor template from a PTXAS-compiled cubin.

    Returns a TemplateSpec with all instructions from the first .text section
    (excluding trailing NOPs), with the UIADD immediate marked as parameterized
    if present.
    """
    instrs = list(_iter_text_instructions(cubin))

    # Detect variant: Variant B has UIADD (0x835)
    has_uiadd = any(_get_opcode(raw) == 0x835 for _, raw in instrs)
    variant = "tid_plus_constant" if has_uiadd else "direct_sr"

    template_instrs = []
    for idx, raw in instrs:
        opc = _get_opcode(raw)
        if opc == 0x918:  # skip NOP padding
            continue
        role = _refine_role(opc, raw, idx, has_uiadd)
        params = []
        if opc == 0x835:
            # UIADD immediate K is at bytes 4-6
            params.append(ParamField(
                name="add_imm_K",
                byte_offset=4,
                byte_length=3,
                description="UIADD immediate constant (tid + K)",
            ))
        ti = TemplateInstruction(
            index=len(template_instrs),
            opcode=opc,
            role=role,
            raw_bytes=raw,
            invariant=(len(params) == 0),
            params=params,
        )
        template_instrs.append(ti)

    name = f"atom_xor_uniform_{variant}"
    desc = (
        "PTXAS-faithful template for atom.global.xor.b32 with "
        + ("direct SR data" if variant == "direct_sr" else "tid + constant data")
        + ", kernel signature (.u64 p_out, .u32 n)."
    )
    selector = (
        "ur_activation_add == 0" if variant == "direct_sr"
        else "ur_activation_add != 0"
    )

    return TemplateSpec(
        name=name,
        variant=variant,
        description=desc,
        instructions=template_instrs,
        selector_condition=selector,
    )
