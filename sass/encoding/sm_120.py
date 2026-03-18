"""
SM_120 (Blackwell consumer, RTX 5090) SASS instruction encoding tables.

All fields confirmed against actual cubin disassembly from repro_sub_bug.sm_120.cubin
using tools/re_probe.py.  See sm_120_encoding_tables.json for raw data.

Instruction format: 128 bits (16 bytes) per instruction.
  bits [127:105] = control/scheduling word (23 bits)
  bits [104:  0] = instruction encoding (105 bits, stored in lo[63:0] + hi[40:0])

Confirmed field layout (from RE, 2026-03-17):
  bits [ 11:  0] = opcode family
  bits [ 23: 16] = destination register index
  bits [ 31: 24] = source register 0 (src0) index
  bits [ 39: 32] = immediate / shift amount (for SHF, MOV imm, etc.)
  hi word        = src1 register + modifiers + control bits

Status: EARLY — opcode table seeded from 2 cubin files, 28 unique opcodes.
        Field positions confirmed for SHF, MOV, IADD3, IMAD families.
        Encoding (re-assembly) not yet implemented — discovery phase.
"""

from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Confirmed opcode constants for SM_120
# Source: re_probe.py against repro_sub_bug.sm_120.cubin + repro_new.cubin
# ---------------------------------------------------------------------------

OPCODES: dict[str, int] = {
    # Integer arithmetic
    "IADD3":          0x210,   # 3-input add (replaces IADD on Volta+)
    "IADD":           0x235,   # 2-input add (used for 64-bit with .64/.X variants)
    "IMAD":           0xC24,   # integer multiply-add
    "IMAD.WIDE":      0x825,   # widening multiply-add (32x32->64)
    "LEA.HI":         0x211,   # load-effective-address, high part
    "LOP3.LUT":       0x212,   # 3-input logic op (LUT-based)

    # Shifts — all SHF variants share the same base opcode 0x819
    # Modifiers (.L/.R, .W, .U32/.U64.HI) are encoded in bits above the opcode
    "SHF":            0x819,   # funnel shift (base)

    # Data movement
    "MOV":            0x202,   # register move
    "S2R":            0x919,   # system register to register (e.g. S2R %r, SR_CTAID.X)
    "S2UR":           0x9C3,   # system register to uniform register
    "LDC":            0xB82,   # load from constant bank
    "LDCU":           0x7AC,   # load from constant bank (uniform)
    "LDG":            0x981,   # load from global memory
    "STG":            0x986,   # store to global memory

    # FP
    "HFMA2":          0x431,   # half-precision FMA (FP16x2)

    # Control flow
    "BRA":            0x947,   # branch
    "EXIT":           0x94D,   # exit warp
    "NOP":            0x918,   # no-op (scheduling filler)

    # Comparison
    "ISETP":          0xC0C,   # integer set predicate (ISETP.GE.AND = 0xC0C)
}

# ---------------------------------------------------------------------------
# Confirmed field positions
# ---------------------------------------------------------------------------

# For integer register instructions (IADD3, SHF, IMAD, MOV, etc.)
FIELD_OPCODE   = (11,  0)    # bits [11:0]
FIELD_DEST     = (23, 16)    # bits [23:16]  — destination register index
FIELD_SRC0     = (31, 24)    # bits [31:24]  — first source register index
FIELD_IMM8     = (39, 32)    # bits [39:32]  — 8-bit immediate / shift amount
# src1 register is in the hi word; exact field position TBD from further RE

# Control field (upper 23 bits of 128-bit instruction, in hi word)
FIELD_CTRL     = (127, 105)  # scheduling / stall / yield / etc.


# ---------------------------------------------------------------------------
# SHF modifier bit patterns (confirmed from RE)
# All SHF variants: opcode bits [11:0] = 0x819
# Modifier bits distinguish .L/.R, .W (wrap), .U32/.U64.HI
# ---------------------------------------------------------------------------

# Observed hi-word patterns for SHF variants (from re_probe output):
#   SHF.L.U32:       hi ≈ 0x...06ff  (wrap=0, .U32, no .HI)
#   SHF.L.U64.HI:    hi ≈ 0x...0203  (wrap=0, .U64.HI)
#   SHF.L.W.U32.HI:  hi ≈ 0x...0e02 or 0x...0e03 (wrap=1, .U32.HI)
# Full modifier field analysis pending more RE probing.

SHF_MODIFIERS = {
    # (wrap, u64_hi, hi_mode) -> partial hi-word bits (placeholder until fully decoded)
    "SHF.L.U32":      {"wrap": 0, "u64": 0, "hi": 0},
    "SHF.L.U64.HI":   {"wrap": 0, "u64": 1, "hi": 1},
    "SHF.L.W.U32.HI": {"wrap": 1, "u64": 0, "hi": 1},
    "SHF.R.U32.HI":   {"wrap": 0, "u64": 0, "hi": 1},   # right shift variant
}


# ---------------------------------------------------------------------------
# Observed encoding samples (ground truth for validation)
# Each entry: (disasm_text, lo_hex, hi_hex)
# ---------------------------------------------------------------------------

SAMPLES: dict[str, list[tuple[str, int, int]]] = {
    "SHF.L.W.U32.HI": [
        # R5, R3, 0x1f, R2  (dest=R5, src0=R3, K=0x1f=31, src1=R2)
        ("R5, R3, 0x1f, R2",  0x1f03057819, 0x8fe40000010e02),
        # R4, R2, 0x1f, R3  (dest=R4, src0=R2, K=0x1f=31, src1=R3)
        ("R4, R2, 0x1f, R3",  0x1f02047819, 0x00fca0000010e03),
        # R5, R3, 0x8, R2   (dest=R5, src0=R3, K=0x8=8,  src1=R2)
        ("R5, R3, 0x8, R2",   0x0803057819, 0x8fe40000010e02),
        # R4, R2, 0x8, R3   (dest=R4, src0=R2, K=0x8=8,  src1=R3)
        ("R4, R2, 0x8, R3",   0x0802047819, 0x00fca0000010e03),
    ],
    "SHF.L.U32": [
        # R9, R2, 0x1f, RZ
        ("R9, R2, 0x1f, RZ",  0x1f02097819, 0x00fc800000006ff),
        # R9, R2, 0x8, RZ
        ("R9, R2, 0x8, RZ",   0x0802097819, 0x00fc800000006ff),
        # R8, R2, 0x1f, RZ
        ("R8, R2, 0x1f, RZ",  0x1f02087819, 0x00fe200000006ff),
    ],
    "SHF.L.U64.HI": [
        # R9, R2.reuse, 0x1f, R3
        ("R9, R2.reuse, 0x1f, R3", 0x1f02097819, 0x40fe40000010203),
        # R9, R2.reuse, 0x8, R3
        ("R9, R2.reuse, 0x8, R3",  0x0802097819, 0x40fe40000010203),
    ],
    "IADD3": [
        # (sample from re_probe — exact operands TBD from further analysis)
    ],
    "MOV": [
        # (sample from re_probe)
    ],
}


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def extract_bits(value: int, high: int, low: int) -> int:
    mask = (1 << (high - low + 1)) - 1
    return (value >> low) & mask


def decode_sample(lo: int, hi: int) -> dict:
    """Decode a 128-bit instruction into known field values."""
    full = lo | (hi << 64)
    return {
        "opcode":   extract_bits(full, 11,  0),
        "dest":     extract_bits(full, 23, 16),
        "src0":     extract_bits(full, 31, 24),
        "imm8":     extract_bits(full, 39, 32),
        "ctrl":     extract_bits(full, 127, 105),
        "lo":       hex(lo),
        "hi":       hex(hi),
    }


def verify_samples():
    """
    Verify that our field positions correctly decode the ground-truth samples.
    Call this to confirm RE findings are consistent.
    """
    print("SM_120 SHF field verification:")
    for opcode_str, instances in SAMPLES.items():
        print(f"\n  {opcode_str}:")
        for disasm, lo, hi in instances:
            d = decode_sample(lo, hi)
            print(f"    disasm: {disasm}")
            print(f"    opcode=0x{d['opcode']:03x}  dest=R{d['dest']}  "
                  f"src0=R{d['src0']}  K={d['imm8']} (0x{d['imm8']:02x})")
            # Validate against disasm text
            parts = [p.strip() for p in disasm.split(",")]
            if len(parts) >= 3:
                expected_dest = int(parts[0].replace("R", "")) if parts[0].startswith("R") else -1
                expected_src0 = int(parts[1].replace("R", "").split(".")[0]) if parts[1].startswith("R") else -1
                expected_k    = int(parts[2], 16) if parts[2].startswith("0x") else -1
                ok_dest  = d['dest'] == expected_dest
                ok_src0  = d['src0'] == expected_src0
                ok_k     = d['imm8'] == expected_k
                print(f"    dest OK={ok_dest}  src0 OK={ok_src0}  K OK={ok_k}")


if __name__ == "__main__":
    verify_samples()
