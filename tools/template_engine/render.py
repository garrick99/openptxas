"""Render a TemplateSpec back to executable instruction bytes.

Given a template spec and parameter values, produces the exact byte sequence
that should be spliced into a cubin's .text section.
"""
from __future__ import annotations

from .spec import TemplateSpec


def render_template(spec: TemplateSpec, params: dict[str, int] | None = None) -> list[bytes]:
    """Render a template spec into a list of 16-byte instruction blocks.

    Parameters
    ----------
    spec : TemplateSpec
        The template to render.
    params : dict
        Parameter name -> value mapping.  For Variant A (direct_sr), no
        parameters are needed.  For Variant B (tid_plus_constant), provide
        ``{"add_imm_K": <int>}``.

    Returns
    -------
    list[bytes]
        One 16-byte entry per instruction, in order.
    """
    params = params or {}
    result = []
    for instr in spec.instructions:
        raw = bytearray(instr.raw_bytes)
        for p in instr.params:
            if p.name not in params:
                raise ValueError(
                    f"Missing parameter '{p.name}' for instruction "
                    f"[{instr.index}] {instr.role}"
                )
            val = params[p.name]
            for i in range(p.byte_length):
                raw[p.byte_offset + i] = (val >> (8 * i)) & 0xFF
        result.append(bytes(raw))
    return result


def render_to_block(spec: TemplateSpec, params: dict[str, int] | None = None) -> bytes:
    """Render and concatenate into a single contiguous byte block."""
    return b"".join(render_template(spec, params))
