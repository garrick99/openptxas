"""
sass/arch.py — Architecture-specific parameters for SM_89 and SM_120.

Encapsulates the differences between Ada Lovelace (4090) and Blackwell (5090)
so the rest of the pipeline can be architecture-agnostic.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ArchConfig:
    """Architecture-specific configuration."""
    sm_version: int          # 89 or 120
    sm_hex: int              # 0x59 or 0x78
    e_flags: int             # ELF e_flags
    param_base: int          # constant bank offset for kernel params
    uses_descriptor_ldg: bool  # True for SM_120 (desc[UR][R]), False for SM_89 ([R])
    uses_ldc: bool           # True for SM_120 (LDC), False for SM_89 (MOV/IMAD from cbank)
    frame_ptr_offset: int    # c[0][offset] for frame pointer (R1)


SM_89 = ArchConfig(
    sm_version=89,
    sm_hex=0x59,
    e_flags=0x6005904,
    param_base=0x160,
    uses_descriptor_ldg=False,
    uses_ldc=False,
    frame_ptr_offset=0x28,
)

SM_120 = ArchConfig(
    sm_version=120,
    sm_hex=0x78,
    e_flags=0x6007802,
    param_base=0x380,
    uses_descriptor_ldg=True,
    uses_ldc=True,
    frame_ptr_offset=0x37c,
)

ARCHITECTURES = {
    'sm_89': SM_89,
    'sm_120': SM_120,
}


def get_arch(target: str) -> ArchConfig:
    """Look up architecture config by target string."""
    target = target.lower().strip()
    if target in ARCHITECTURES:
        return ARCHITECTURES[target]
    raise ValueError(f"Unsupported target: {target}. Supported: {list(ARCHITECTURES.keys())}")
