"""UR-aware instruction selection policy derived from PTXAS comparison.

TEMPLATE-ENGINE-8A: evidence-based policy rules for choosing between
UR-path and GPR-path instruction variants.

Derived from aggregate opcode balance across 115 non-byte-exact kernels.
"""

# ── Policy rules ─────────────────────────────────────────────────────────
#
# Rule 1: UNIFORM COMPARISON (highest leverage: -278 delta)
#   PTXAS: ISETP.R-UR (0xc0c) for GPR vs uniform param
#          ISETP.UR-UR (0xc11) for uniform vs uniform
#   OURS:  ISETP.R-R (0x20c) for everything
#   Evidence: PTXAS +164 ISETP.UR-UR, +114 ISETP.R-UR; OURS +102 ISETP.R-R
#   Rule: if one or both operands are uniform (param-derived), use UR variant
#   Confidence: HIGH (covers 278 instruction substitutions)
#
# Rule 2: UNIFORM ADD-IMMEDIATE (leverage: -93 delta)
#   PTXAS: UIADD (0x835) for uniform + immediate
#   OURS:  IADD3.IMM (0x810) for everything
#   Evidence: PTXAS +93 UIADD; OURS +154 IADD3.IMM
#   Rule: if add-immediate operand is uniform, use UIADD
#   Confidence: HIGH
#
# Rule 3: UNIFORM CONSTANT LOAD (leverage: -98 delta)
#   PTXAS: LDCU (0x7ac) for uniform constant loads
#   OURS:  S2R/LDC (0xb82) + IADD.64-UR (0xc35) for param address
#   Evidence: PTXAS +98 LDCU; OURS +99 S2R/LDC, +91 IADD.64-UR
#   Rule: prefer LDCU for param loads when uniform provenance is known
#   Confidence: MEDIUM (our LDCU path already exists for 32-bit params;
#     the gap is in how 64-bit param addresses are loaded)
#
# Rule 4: WIDE ADDRESS COMPUTATION (leverage: +102 delta)
#   PTXAS: uses LDCU to UR + UIADD for address offsets
#   OURS:  IMAD.WIDE (0x825) for 32->64 bit address computation
#   Evidence: OURS +102 IMAD.WIDE (PTXAS uses 0)
#   Rule: replace IMAD.WIDE with UR-based address computation when safe
#   Confidence: LOW (IMAD.WIDE is deeply integrated in address gen)
#
# ── Implementation priority ──────────────────────────────────────────────
#
# Phase 1 (TE8-B): Rules 1 + 2 (uniform comparison + uniform add)
#   Expected impact: ~370 instruction substitutions across 115 kernels
#   Risk: LOW (local substitutions, operand provenance already tracked)
#
# Phase 2 (future): Rule 3 (LDCU routing)
#   Expected impact: ~98 substitutions
#   Risk: MEDIUM (touches param loading infrastructure)
#
# Phase 3 (future): Rule 4 (address gen)
#   Expected impact: ~102 substitutions
#   Risk: HIGH (requires rethinking address computation path)

POLICY_RULES = [
    {
        'id': 'UNIFORM_COMPARE',
        'ptxas_opcodes': [0xc0c, 0xc11],
        'ours_opcodes': [0x20c],
        'delta': -278,
        'description': 'Use ISETP.R-UR or ISETP.UR-UR when operand is uniform',
        'confidence': 'HIGH',
        'priority': 1,
    },
    {
        'id': 'UNIFORM_ADD_IMM',
        'ptxas_opcodes': [0x835],
        'ours_opcodes': [0x810],
        'delta': -93,
        'description': 'Use UIADD instead of IADD3.IMM for uniform + immediate',
        'confidence': 'HIGH',
        'priority': 2,
    },
    {
        'id': 'UNIFORM_CONST_LOAD',
        'ptxas_opcodes': [0x7ac],
        'ours_opcodes': [0xb82, 0xc35],
        'delta': -98,
        'description': 'Prefer LDCU for uniform constant bank loads',
        'confidence': 'MEDIUM',
        'priority': 3,
    },
    {
        'id': 'WIDE_ADDR_GEN',
        'ptxas_opcodes': [],
        'ours_opcodes': [0x825],
        'delta': +102,
        'description': 'Replace IMAD.WIDE with UR-based address computation',
        'confidence': 'LOW',
        'priority': 4,
    },
]
