"""
sass/encoding/ — Architecture-specific SASS instruction encoders.

Supported targets:
- SM_120 (Blackwell / RTX 5090): sass/encoding/sm_120_opcodes.py
- SM_89  (Ada Lovelace / RTX 4090): sass/encoding/sm_89_opcodes.py

Usage:
    from sass.encoding import get_encoder
    enc = get_encoder(120)  # or 89
    raw = enc.encode_nop()
"""


def get_target_from_ptx(ptx_src: str) -> int:
    """Extract SM version from PTX source (.target sm_NNN)."""
    import re
    m = re.search(r'\.target\s+sm_(\d+)', ptx_src)
    if m:
        return int(m.group(1))
    return 120  # default to Blackwell


def get_encoder(sm_version: int):
    """Return the encoding module for the given SM version."""
    if sm_version == 89:
        from sass.encoding import sm_89_opcodes
        return sm_89_opcodes
    elif sm_version >= 120:
        from sass.encoding import sm_120_opcodes
        return sm_120_opcodes
    else:
        raise ValueError(f"Unsupported SM version: sm_{sm_version}")
