"""
sass/scoreboard.py — SM_120 scoreboard emulator for ctrl word generation.

Generates correct dependency barrier (ctrl) values for any instruction stream.
Replaces the manual ptxas-matching approach with automated tracking.

SM_120 ctrl word (23 bits):
  [22:17] stall  — cycle count to wait (0 for barrier-based scheduling)
  [16]    yield  — yield hint
  [15]    wbar   — write-after-read barrier flag
  [14:10] rbar   — read barrier mask (which scoreboard slots to wait for)
  [9:4]   wdep   — write dependency slot (scoreboard slot for this result)
  [3:0]   misc   — instruction sequence counter (wraps mod 8)

Scoreboard slots (from ptxas RE):
  0x31 = LDC/LDCU result slot
  0x33 = LDS result slot
  0x35 = LDG result slot (all LDGs share; ptxas-verified 2026-04-04)
  0x3e = ALU result slot (SHF, IADD, FADD, etc.)
  0x3f = no write tracking (EXIT, BRA, STG, BAR)

Read barrier encoding (BITMASK — combine with OR, not max):
  bit 0 (0x01) = always set (base)
  bit 1 (0x02) = wait for LDC/LDCU scoreboard slot
  bit 2 (0x04) = wait for LDS scoreboard slot
  bit 3 (0x08) = wait for LDG scoreboard slot
  bit 4 (0x10) = reserved (slot 0x37 does NOT have an rbar bit — never use wdep=0x37)
  Common values: 0x01=no wait, 0x03=LDC, 0x05=LDS, 0x09=LDG, 0x0B=LDC+LDG
"""

from __future__ import annotations
import struct
from typing import NamedTuple
from sass.isel import SassInstr


class _OpMeta(NamedTuple):
    name: str
    min_gpr_gap: int  # minimum instruction gap between write and immediate GPR reader (0 = no constraint)
    wdep: int         # scoreboard write-dep slot (0x3e=ALU, 0x3f=no-track, etc.)
    misc: int         # ctrl misc nibble value (hardware-verified per opcode)


# SM_120 ALU instructions that require ≥1 instruction gap before a GPR consumer.
# The stall field is ignored by hardware; rbar alone does not gate adjacent ALU reads.
_OPCODE_META: dict[int, _OpMeta] = {
    0x210: _OpMeta('IADD3',      1, 0x3e, 1),
    0x212: _OpMeta('IADD3X',     1, 0x3e, 1),
    0x224: _OpMeta('IMAD.32',    1, 0x3e, 1),
    0x2a4: _OpMeta('IMAD.RR',   1, 0x3e, 1),   # R-R-R multiply (opcode 0x2a4, SM_120 validated)
    0x824: _OpMeta('IMAD',       1, 0x3e, 1),
    0x825: _OpMeta('IMAD.WIDE',  1, 0x3e, 1),   # IMAD.WIDE R-imm (64-bit result)
    0x225: _OpMeta('IMAD.WIDE.RR', 1, 0x3e, 1),  # IMAD.WIDE R-R
    0x819: _OpMeta('SHF',        1, 0x3e, 1),
    0x219: _OpMeta('SHF.R.VAR', 1, 0x3e, 1),  # SHF.R (variable shift: U64/S64/U32.HI/S32.HI)
    0xa19: _OpMeta('SHF.89',     1, 0x3e, 1),  # SM_89 SHF
    0xa10: _OpMeta('IADD3.89',   1, 0x3e, 1),  # SM_89 IADD3 cbuf
    0xa24: _OpMeta('IMAD.89',    1, 0x3e, 1),  # SM_89 IMAD cbuf
    0x624: _OpMeta('IMAD.MOV',   0, 0x3f, 1),  # SM_89 param load (like LDC)
    0xa02: _OpMeta('MOV.cb',     0, 0x3e, 1),  # SM_89 MOV from cbuf
    0xab9: _OpMeta('ULDC.64',    0, 0x31, 7),  # SM_89 uniform const load
    0xa0c: _OpMeta('ISETP.89',   0, 0x3e, 0),  # SM_89 ISETP cbuf
    0x807: _OpMeta('SEL.89',     0, 0x3e, 1),  # SM_89 SEL immediate
    0xa12: _OpMeta('LOP3.89',    1, 0x3e, 1),  # SM_89 LOP3.LUT cbuf
    0x299: _OpMeta('SHF.VAR',   1, 0x3e, 1),   # variable-shift SHF (opcode 0x7299)
    0x219: _OpMeta('SHF.R.S32.HI.VAR', 1, 0x3e, 1),  # SHF.R.S32.HI variable-shift (shr.s32)
    0x221: _OpMeta('FADD',       1, 0x3e, 1),
    0x223: _OpMeta('FFMA',       1, 0x3e, 1),
    0x308: _OpMeta('MUFU',       1, 0x3e, 1),  # MUFU (SFU: RCP, SQRT, SIN, COS, EX2, LG2)
    0x309: _OpMeta('POPC',       1, 0x3e, 1),  # POPC (population count)
    0x301: _OpMeta('BREV',       1, 0x3e, 1),  # BREV (bit reverse)
    0x300: _OpMeta('FLO',        1, 0x3e, 1),  # FLO (find leading one)
    0x820: _OpMeta('FMUL.IMM',   1, 0x3e, 1),  # FMUL with 32-bit float immediate
    0x823: _OpMeta('FFMA.IMM',   1, 0x3e, 1),  # FFMA with 32-bit float immediate
    0x80a: _OpMeta('FSEL.STEP',  1, 0x3e, 5),  # Combined float compare+select (misc=5, ptxas-verified)
    0x235: _OpMeta('IADD.64',    1, 0x3e, 1),
    0xc35: _OpMeta('IADD.64-UR', 1, 0x3e, 5),  # misc=5 per hardware bisect 2026-03-25
    0x202: _OpMeta('MOV',        0, 0x3e, 1),
    0x20c: _OpMeta('ISETP.RR',   0, 0x3e, 0),  # ISETP R-R: misc=0 (SM_120 predicate)
    0xc0c: _OpMeta('ISETP.RU',   0, 0x3e, 0),  # ISETP R-UR: misc=0 on SM_120
    0x431: _OpMeta('HFMA2',      1, 0x3e, 1),  # HFMA2 (half-precision FMA2, used as zero-init in div.u32)
    0x810: _OpMeta('IADD3.IMM',  1, 0x3e, 1),  # IADD3 with 32-bit immediate operand
    0x306: _OpMeta('I2F.U32.RP', 1, 0x3e, 1),  # I2F unsigned int to float, round toward +inf
    0x305: _OpMeta('F2I.FTZ.U32',1, 0x3e, 1),  # F2I float to unsigned int, truncate
    0x310: _OpMeta('F2F',        1, 0x33, 1),  # F2F float-to-float precision conversion (F32↔F64), long-latency wdep=0x33
    0x311: _OpMeta('F2I.F64',   1, 0x3e, 1),  # F2I.F64 float64-to-int32 conversion
    0x312: _OpMeta('I2F.F64',   1, 0x3e, 1),  # I2F.F64 int32-to-float64 conversion (writes pair)
    0x81a: _OpMeta('BFE_SEXT',  1, 0x3e, 1),  # BFE sign-extension step (bfe.s32 lowering)
    0x22a: _OpMeta('DSETP',     0, 0x3e, 0),  # DSETP FP64 compare → predicate (misc=0, like ISETP)
    # Tensor core MMA: wdep=0x3e (ALU), min_gpr_gap=1, misc from ptxas (2 for HMMA/DMMA)
    0x23c: _OpMeta('HMMA',      1, 0x3e, 2),  # HMMA FP16/BF16/TF32 MMA (m16n8k*) misc=2 ptxas-observed
    0x237: _OpMeta('IMMA',      1, 0x3e, 2),  # IMMA INT8 MMA (m16n8k32)
    0x23f: _OpMeta('DMMA',      1, 0x3e, 2),  # DMMA FP64 MMA (m8n8k4)
    0x27a: _OpMeta('QMMA',      1, 0x3e, 2),  # QMMA FP8 E4M3/E5M2 MMA (m16n8k32)
    0x83b: _OpMeta('LDSM',      1, 0x33, 2),  # LDSM load shared→matrix regs (wdep=LDS slot)
    0x3c4: _OpMeta('REDUX',     0, 0x3f, 0),  # REDUX warp reduction → UR (no GPR dest)
    0xc02: _OpMeta('MOV.UR',   1, 0x3e, 1),  # MOV R, UR — copy uniform reg to GPR
    0x226: _OpMeta('IDP.4A',   1, 0x3e, 1),  # IDP.4A dp4a (integer dot product)
    # Phase 3 opcodes
    0x211: _OpMeta('LEA',      1, 0x3e, 1),  # LEA load effective address
    0x811: _OpMeta('LEA.IMM',  1, 0x3e, 1),  # LEA with immediate index
    0x217: _OpMeta('IMNMX',    1, 0x3e, 1),  # IMNMX integer min/max
    0x203: _OpMeta('P2R',      1, 0x3e, 1),  # P2R predicate to register
    0x204: _OpMeta('R2P',      0, 0x3f, 0),  # R2P register to predicate (no GPR dest)
    0x21b: _OpMeta('BMSK',     1, 0x3e, 1),  # BMSK bitmask generation
    0x21a: _OpMeta('SGXT',     1, 0x3e, 1),  # SGXT sign extend (register form)
    0x21e: _OpMeta('PLOP3',    0, 0x3f, 0),  # PLOP3 predicate LOP3 (pred dest only)
    0x239: _OpMeta('I2IP',     1, 0x3e, 1),  # I2IP integer pack
    0x822: _OpMeta('FSWZADD',  1, 0x3e, 1),  # FSWZADD float swizzle-add
    # TMA (Tensor Memory Accelerator) — SM_120 Blackwell
    0x5b2: _OpMeta('SYNCS.EXCH',   0, 0x03, 2),  # mbarrier.init (wdep varies per context; 0x03/0x15 observed)
    0x9a7: _OpMeta('SYNCS.ARRIVE', 0, 0x3f, 1),  # mbarrier.arrive (no GPR dest)
    0x5a7: _OpMeta('SYNCS.TRYWAIT',0, 0x3f, 1),  # mbarrier.try_wait (pred dest only)
    0x3ba: _OpMeta('UBLKCP',       0, 0x0e, 12), # bulk copy S↔G (wdep=0x0e for load, 0x1f for store)
    0x5b4: _OpMeta('UTMALDG',      0, 0x0e, 12), # TMA tensor load (wdep=0x0e, misc=12)
    0x3b5: _OpMeta('UTMASTG',      0, 0x1f, 1),  # TMA tensor store (wdep=0x1f, misc=1)
    0x9b7: _OpMeta('UTMACMDFLUSH', 0, 0x0f, 1),  # TMA command flush (wdep=0x0f)
    0x82f: _OpMeta('ELECT',        0, 0x3f, 1),  # elect leader thread (no GPR dest)
    0x98f: _OpMeta('CCTL',         0, 0x3f, 1),  # cache control invalidate all
    # Rare opcodes batch (2026-04-04)
    0x9ab: _OpMeta('ERRBAR',       0, 0x3f, 0),  # error barrier (no GPR dest)
    0x5ab: _OpMeta('CGAERRBAR',    0, 0x3f, 0),  # CGA error barrier (no GPR dest)
    0x801: _OpMeta('PMTRIG',       0, 0x3f, 0),  # performance monitor trigger (no GPR dest)
    0x944: _OpMeta('CALL.REL',     0, 0x3f, 1),  # relative function call (control flow)
    0x950: _OpMeta('RET.REL',      0, 0x3f, 1),  # return from function (control flow)
    0x547: _OpMeta('BRA.U',        0, 0x3f, 1),  # uniform branch (control flow)
    0x882: _OpMeta('UMOV',         0, 0x3f, 1),  # uniform register move (writes UR, not GPR)
    0x890: _OpMeta('UIADD3',       0, 0x3f, 1),  # uniform 3-input add (writes UR)
    0x28c: _OpMeta('UISETP',       0, 0x3f, 0),  # uniform integer set predicate (writes UP)
    0x887: _OpMeta('USEL',         0, 0x3f, 1),  # uniform select (writes UR)
    0x31c: _OpMeta('B2R',          1, 0x3e, 1),  # barrier result to register/predicate
    0x853: _OpMeta('UFSETP',       0, 0x3f, 0),  # uniform FP set predicate (writes UP)
    0x856: _OpMeta('UFMUL',        0, 0x3f, 1),  # uniform FP multiply (writes UR)
    0xc26: _OpMeta('IDP.4A.UR',    1, 0x3e, 1),  # IDP.4A with UR source
    # Cluster operations (2026-04-04)
    0x9c7: _OpMeta('UCGABAR_ARV',  0, 0x3f, 0),  # cluster barrier arrive (no GPR dest)
    0xdc7: _OpMeta('UCGABAR_WAIT', 0, 0x3f, 0),  # cluster barrier wait (no GPR dest)
    0x291: _OpMeta('ULEA',         0, 0x3f, 1),  # uniform LEA (writes UR)
    0xc82: _OpMeta('UMOV.RR',      0, 0x3f, 1),  # uniform reg-reg move (writes UR)
    # Texture/surface opcodes (identified, encoders TBD)
    0xf60: _OpMeta('TEX',          1, 0x35, 2),  # texture fetch (long-latency like LDG)
    0xf63: _OpMeta('TLD4',         1, 0x35, 2),  # texture gather
    0xf66: _OpMeta('TLD',          1, 0x35, 2),  # texture load (TLD.LZ) (long-latency like LDG)
    0xf6f: _OpMeta('TXQ',          1, 0x35, 2),  # texture query
    0xf99: _OpMeta('SULD',         1, 0x35, 2),  # surface load
    0xf9d: _OpMeta('SUST',         0, 0x3f, 2),  # surface store (no GPR dest)
    # Additional rare opcodes (2026-04-04 batch 2)
    0x3a1: _OpMeta('MATCH',        1, 0x3e, 1),  # warp match (any/all)
    0x95d: _OpMeta('NANOSLEEP',    0, 0x3f, 1),  # thread sleep
}


# Opcode classification (includes both SM_120 and SM_89 variants)
_OPCODES_LDG = {0x981, 0xf60, 0xf63, 0xf66, 0xf6f, 0xf99}  # LDG, TEX, TLD, TLD4, TXQ, SULD
_OPCODES_ATOMG = {0x3a9,   # ATOMG.E.CAS.b32 / CAS.b64
                 0x9a8,   # ATOMG.E.{ADD|MIN|MAX|EXCH}.u32
                 0x9a3}   # ATOMG.E.ADD.F32
_OPCODES_LDC = {0xb82, 0x7ac, 0x919, 0x9c3,  # SM_120: LDC, LDCU, S2R, S2UR
                0x624, 0xab9, 0xa02}           # SM_89: IMAD.MOV.U32(cbuf), ULDC.64, MOV(cbuf)
_OPCODES_LDS = {0x984, 0x83b}  # LDS, LDSM (load shared to matrix)
_OPCODES_STG = {0x986, 0xf9d}  # STG, SUST
_OPCODES_STS = {0x988}
_OPCODES_BAR = {0xb1d}
_OPCODES_DFPU  = {0x229, 0x228, 0x22b, 0xc2b}  # DADD, DMUL, DFMA (R-R b1=0x72), DFMA-UR-UR (b1=0x7c)
_OPCODES_DSETP = {0x22a}                 # DSETP (FP64 compare → predicate; reads pairs, no GPR dest)
_OPCODES_F2F   = {0x310}                 # F2F (float-to-float precision conversion; long-latency, wdep=0x33)
_OPCODES_CTRL = {0x94d, 0x947, 0x918, 0x91a, 0x992,  # EXIT, BRA, NOP, DEPBAR.LE, MEMBAR
                 0x9ab, 0x5ab, 0x801,                 # ERRBAR, CGAERRBAR, PMTRIG
                 0x944, 0x950, 0x547,                  # CALL.REL, RET.REL, BRA.U
                 0x9c7, 0xdc7,                          # UCGABAR_ARV, UCGABAR_WAIT
                 0x95d}                                # NANOSLEEP
_OPCODES_LDGSTS = {0xfae}  # LDGSTS.E (cp.async global→shared)
_OPCODES_LDGDEPBAR = {0x9af}  # LDGDEPBAR (cp.async commit)
_OPCODES_REDUX = {0x3c4}  # REDUX.SUM (warp reduction → UR)
_OPCODES_TMA = {0x5b2, 0x9a7, 0x5a7,  # SYNCS (mbarrier init/arrive/trywait)
                0x3ba, 0x5b4, 0x3b5,  # UBLKCP, UTMALDG, UTMASTG
                0x9b7, 0x82f, 0x98f}  # UTMACMDFLUSH, ELECT, CCTL
_OPCODES_ALU = {
    # Integer arithmetic (SM_120 + SM_89)
    0x210,        # IADD3 (SM_120 R-R / SM_89 R-R)
    0xa10,        # IADD3 (SM_89 cbuf form)
    0x235,        # IADD.64
    0x202,        # IADD3.X (with carry)
    0x224, 0x2a4, 0xc24,  # IMAD R-R (old), IMAD R-R (validated), IMAD R-UR
    0xa24,                # IMAD (SM_89 cbuf/UR form)
    0x624,                # IMAD.MOV.U32 (SM_89 param load from cbuf)
    0x824, 0x825, 0x225, # IMAD.SHL.U32, IMAD.WIDE (imm), IMAD.WIDE (R-R)
    0x227,        # IMAD.HI.U32
    0x213,        # IABS
    0x248, 0x848, # VIMNMX R-R, R-imm (integer min/max)
    0x309, 0x301, # POPC, BREV
    0x300,        # FLO
    0x226,        # IDP.4A (dp4a)
    # Float arithmetic
    0x221,        # FADD
    0x223,        # FMUL / FFMA
    0x820,        # FMUL with float immediate
    0x823,        # FFMA with float immediate
    0x80a,        # FSEL.STEP (combined compare+select)
    0x209,        # FMNMX (float min/max)
    0x308,        # MUFU (RCP, SQRT, SIN, COS, EX2, LG2)
    # Type conversion
    0x245,        # I2FP.F32.U32
    0x305,        # F2I.U32
    # Logic
    0x212,        # LOP3.LUT (SM_120)
    0xa12,        # LOP3.LUT (SM_89 cbuf form)
    0x819,        # SHF (SM_120, all variants)
    0xa19,        # SHF (SM_89)
    0x299,        # SHF.VAR (variable-shift SHF, shift amount in register)
    0x219,        # SHF.R.S32.HI.VAR (arithmetic right shift, variable amount)
    # Select / predicate
    0x207,        # SEL (SM_120)
    0x807,        # SEL (SM_89, imm form)
    0xa0c,        # ISETP (SM_89 cbuf form)
    0x208,        # FSEL
    0x20b,        # FSETP
    0x20c,        # ISETP R-R
    0xc0c,        # ISETP R-UR
    # Permute / misc
    0x416,        # PRMT (immediate selector, opc=0x416)
    0x216,        # PRMT.REG (register selector, opc=0x216)
    0x589, 0xf89, 0x989,  # SHFL (reg-reg, reg-imm, imm-imm)
    0x806,        # VOTE.ANY
    # Matrix multiply (HMMA, IMMA, DMMA, QMMA)
    0x23c, 0x237, 0x23f, 0x27a,
    # Predicate ↔ register moves
    0x203,        # P2R (predicate-to-register move)
    0x204,        # R2P (register-to-predicate move)
    # Address calculation
    0x211,        # LEA (address calculation)
    0x811,        # LEA.IMM (address calc with immediate)
    # Integer min/max & sign-extend & bitmask
    0x217,        # IMNMX (integer min/max)
    0x21a,        # SGXT (sign extend)
    0x21b,        # BMSK (bitmask generation)
    # Predicate logic
    0x21e,        # PLOP3 (predicate logic op)
    # Integer pack
    0x239,        # I2IP (integer pack)
    # Float swizzle-add
    0x822,        # FSWZADD (float swizzle-add)
    # 64-bit add with UR
    0xc35,        # IADD.64-UR (64-bit add with UR)
    # Miscellaneous / div.u32 helpers
    0x431,        # HFMA2 (zero-init trick)
    0x810,        # IADD3 immediate form
    0x306,        # I2F.U32.RP
    0x305,        # F2I.FTZ.U32.TRUNC
    # Float precision conversion / integer↔float F64
    # NOTE: F2F (0x310) moved to _OPCODES_F2F — long-latency, wdep=0x33
    0x311,        # F2I.F64 (F64→int32)
    0x312,        # I2F.F64 (int32→F64)
    # BFE helpers
    0x81a,        # BFE_SEXT (bfe.s32 sign-extend step)
    # FP64 comparison
    0x22a,        # DSETP (FP64 compare → predicate; reads pairs, no GPR dest)
    # FP conversion (decoded 2026-04-01)
    0x23e,        # F2FP.F16.F32 (FP32→packed FP16)
    # Warp reduction (decoded 2026-04-01) — writes UR, not GPR
    0x3c4,        # REDUX.SUM (warp sum → UR)
    # MOV R, UR — copy uniform register to GPR (after REDUX)
    0xc02,        # MOV R, UR
    # Rare opcodes batch (2026-04-04)
    0xc26,        # IDP.4A with UR source
    0x31c,        # B2R (barrier result to register)
    0x807,        # SEL (immediate form, already in SM_89 set)
    0x28c,        # UISETP (uniform integer set predicate)
    0x3a1,        # MATCH (warp match any/all)
}
# Note: IADD.64-UR (0xc35) uses wdep=0x3f (no tracking) + stall=1.
# The 1-cycle stall ensures the result is ready for the subsequent LDG/STG.
_OPCODES_IADD64_UR = {0xc35}
_OPCODES_SMEM_SETUP = {0x9c3, 0x882, 0x291,  # S2UR, UMOV, ULEA
                       0x890, 0x887, 0x853, 0x856,  # UIADD3, USEL, UFSETP, UFMUL
                       0x291, 0xc82}               # ULEA, UMOV.RR

# SM_89-specific opcodes that map to SM_120 equivalents in ALU set
# ISETP (SM_89 cbuf): 0xa0c — already handled if in _OPCODES_ALU
# SHF (SM_89): 0xa19
# SEL (SM_89): 0x807
# MOV cbuf (SM_89): 0xa02 — already in _OPCODES_LDC


def _get_opcode(raw: bytes) -> int:
    return struct.unpack_from('<Q', raw, 0)[0] & 0xFFF


def _get_dest_reg(raw: bytes) -> int:
    """Get the destination register index, or -1 if none."""
    opcode = _get_opcode(raw)
    # LDCU/S2UR write UR registers, not GPR
    if opcode in (_OPCODES_CTRL | _OPCODES_STG | _OPCODES_STS | _OPCODES_BAR | _OPCODES_TMA | {0x7ac, 0x9c3}):
        return -1  # no GPR dest (TMA instructions operate on UR, not GPR)
    return raw[2]


def _get_src_regs(raw: bytes) -> set[int]:
    """Get source register indices this instruction reads from GPRs."""
    opcode = _get_opcode(raw)
    regs = set()

    if opcode in _OPCODES_LDG:
        # LDG: src_addr at b3
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
    elif opcode in _OPCODES_ATOMG:
        # All ATOMG ops: addr at b3 (64-bit pair), data at b4
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
        if raw[4] < 255: regs.add(raw[4])
        if opcode == 0x3a9:
            # ATOMG.CAS: also reads new_val at b8
            if raw[8] < 255: regs.add(raw[8])
            # CAS.64 (b9=0xe5): compare and new_val are 64-bit pairs
            if raw[9] == 0xe5:
                if raw[4] < 255: regs.add(raw[4]+1)
                if raw[8] < 255: regs.add(raw[8]+1)
    elif opcode in _OPCODES_DFPU:
        # DADD (0x229, b1=0x72): src0=b3(pair), src1=b8(pair) — ptxas-verified 2026-04-04
        # DMUL (0x228, b1=0x72): src0=b3(pair), src1=b4(pair)
        # DFMA (0x22b, b1=0x72): src0=b3(pair), src1=b4(pair), src2=b8(pair) — all GPR
        # DFMA-UR-UR (0xc2b, b1=0x7c): src0=b3(pair), b4=UR, b8=UR — only src0 is GPR
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
        if opcode == 0x229:  # DADD: src1 at b8 (NOT b4)
            if raw[8] < 255: regs |= {raw[8], raw[8]+1}
        elif opcode == 0xc2b:  # DFMA-UR-UR: b4/b8 are UR indices, not GPR
            pass  # only src0 (b3) is a GPR
        else:
            if raw[4] < 255: regs |= {raw[4], raw[4]+1}  # DMUL/DFMA src1 in b4
            if opcode == 0x22b and raw[8] < 255: regs |= {raw[8], raw[8]+1}  # DFMA R-R src2
    elif opcode in _OPCODES_DSETP:
        # DSETP: src0(b3, 64-bit pair), src1(b4, 64-bit pair); no GPR dest
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
        if raw[4] < 255: regs |= {raw[4], raw[4]+1}
    elif opcode in _OPCODES_STG:
        # STG: addr at b3, data at b4 (NOT b8 — b8 is UR descriptor)
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
        if raw[4] < 255: regs.add(raw[4])
    elif opcode in _OPCODES_STS:
        # STS: data at b4
        if raw[4] < 255: regs.add(raw[4])
    elif opcode in _OPCODES_IADD64_UR:
        # IADD.64-UR: GPR src pair at b3 (b4 is UR, not tracked here)
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
    elif opcode in _OPCODES_ALU:
        # ALU: src0 at b3, src1 at b4, src2 at b8 (varies by opcode)
        # Unary ops (MUFU, POPC, BREV, FLO, IABS): src at b4, b3=0x00 (not a real source)
        if opcode in (0x308, 0x309, 0x301, 0x300, 0x213):
            # MUFU/POPC/BREV/FLO/IABS: single src at b4 only
            if raw[4] < 255: regs.add(raw[4])
        elif opcode in (0x210, 0x212, 0x810):  # IADD3/LOP3/IADD3.IMM: src0=b3, src1=b4, src2=b8
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255: regs.add(raw[8])
        elif opcode in (0x207, 0x20b, 0x416, 0x216):  # SEL/FSETP/PRMT/PRMT.REG: src0=b3, src1=b4
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
            if opcode == 0x216 and raw[8] < 255: regs.add(raw[8])  # PRMT.REG also reads b8
        elif opcode == 0x235:  # IADD.64: src0=b3 pair, src1=b4 pair
            if raw[3] < 255: regs |= {raw[3], raw[3]+1}
            if raw[4] < 255: regs |= {raw[4], raw[4]+1}
        elif opcode in (0x819,):  # SHF (const): src0=b3, K=b4(imm), src1=b8
            if raw[3] < 255: regs.add(raw[3])
            if raw[8] < 255: regs.add(raw[8])
        elif opcode in (0x299,):  # SHF.VAR: src0=b3, k_reg=b4(reg), src1=b8
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])   # shift-amount register
            if raw[8] < 255: regs.add(raw[8])
        elif opcode in (0x23c, 0x237, 0x27a):  # HMMA/IMMA/QMMA: a=b3(4 regs), b=b4(2), c=b8(4)
            for r in range(4): regs.add(raw[3]+r)
            for r in range(2): regs.add(raw[4]+r)
            if raw[8] < 255:
                for r in range(4): regs.add(raw[8]+r)
        elif opcode in (0x820, 0x823, 0x80a):  # FMUL.IMM/FFMA.IMM/FSEL: src0=b3, b4-b7=imm
            if raw[3] < 255: regs.add(raw[3])
            if opcode == 0x823 and raw[8] < 255: regs.add(raw[8])  # FFMA addend
        elif opcode in (0x221, 0x223):  # FADD/FFMA: src0=b3, src1=b4, src2=b8
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255 and opcode == 0x223: regs.add(raw[8])
        elif opcode in (0x825, 0x225):  # IMAD.WIDE (R-imm 0x825, R-R 0x225): src2 is 64-bit pair
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255: regs |= {raw[8], raw[8]+1}
        elif opcode in (0x824, 0x224, 0x2a4):  # IMAD non-wide variants: src0=b3, src1=b4, src2=b8
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255: regs.add(raw[8])
        elif opcode == 0xc24:  # IMAD R-UR: src0=b3 (GPR), src1=b4 (UR, not GPR), src2=b8 (GPR)
            if raw[3] < 255: regs.add(raw[3])
            if raw[8] < 255: regs.add(raw[8])
        elif opcode == 0x20c:  # ISETP R-R: src0=b3, src1=b4
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
        elif opcode == 0x226:  # IDP.4A: src_a=b3, src_b=b4, src_c=b8
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255: regs.add(raw[8])
        else:
            # Default: src0 at b3 (generic ALU)
            if raw[3] < 255: regs.add(raw[3])
    elif opcode in _OPCODES_F2F:
        # F2F: src at b4. F2F.F32.F64 (b9=0x10, narrowing) reads f64 pair;
        # F2F.F64.F32 (b9=0x18, widening) reads single f32.
        if raw[4] < 255:
            regs.add(raw[4])
            if raw[9] == 0x10:  # F2F.F32.F64: src is f64 pair
                regs.add(raw[4] + 1)
    return regs


def _get_dest_regs(raw: bytes) -> set[int]:
    """Get destination register indices this instruction writes."""
    opcode = _get_opcode(raw)
    if opcode in (0x7ac, 0x9c3, 0x3c4):  # LDCU, S2UR, REDUX: write UR bank, not GPR
        return set()
    if opcode in _OPCODES_DSETP:  # DSETP: writes predicate, not GPR
        return set()
    dest = raw[2]
    regs = set()

    if opcode in _OPCODES_LDG:
        if dest < 255:
            regs.add(dest)
            # Check if 64-bit or 128-bit load
            b9 = raw[9]
            if b9 in (0x1b, 0x9b):  # LDG.E.64
                regs.add(dest+1)
            elif b9 in (0x1d, 0x9d):  # LDG.E.128
                regs |= {dest+1, dest+2, dest+3}
            elif b9 in (0x19, 0x99):  # LDG.E.32 (single register)
                pass  # already added dest
            else:
                regs.add(dest+1)  # unknown width — assume 64-bit
    elif opcode in _OPCODES_ATOMG:
        # ATOMG: writes dest (b2) — the old value read from memory
        if dest < 255:
            regs.add(dest)
            # CAS.64 (b9=0xe5): writes 64-bit dest pair
            if opcode == 0x3a9 and raw[9] == 0xe5:
                regs.add(dest+1)
    elif opcode in _OPCODES_DFPU:
        # DADD/DMUL/DFMA: writes 64-bit dest pair (b2, b2+1)
        if dest < 255: regs |= {dest, dest+1}
    elif opcode in _OPCODES_LDC:
        if dest < 255:
            regs.add(dest)
            if raw[9] == 0x0a:  # 64-bit
                regs.add(dest+1)
    elif opcode in _OPCODES_LDS:
        if dest < 255: regs.add(dest)
    elif opcode in (0x235, 0xc35):  # IADD.64 / IADD.64-UR: writes GPR pair
        if dest < 255: regs |= {dest, dest+1}
    elif opcode in (0x23c, 0x237, 0x27a):  # HMMA/IMMA/QMMA: writes 4 regs
        if dest < 255: regs |= {dest, dest+1, dest+2, dest+3}
    elif opcode in (0x825, 0x225):  # IMAD.WIDE: writes dest pair
        if dest < 255: regs |= {dest, dest+1}
    elif opcode == 0x310:  # F2F: F2F.F64.F32 (b9=0x18) writes pair; F2F.F32.F64 writes single
        if dest < 255:
            regs.add(dest)
            if raw[9] == 0x18:  # F2F.F64.F32: dest is f64 pair
                regs.add(dest + 1)
    elif opcode == 0x312:  # I2F.F64: always writes dest pair
        if dest < 255: regs |= {dest, dest + 1}
    elif opcode in _OPCODES_ALU:
        if dest < 255: regs.add(dest)
    return regs


_ldcu_slot_counter = [0]  # mutable counter for rotating LDCU wdep slots
_ldc_slot_counter = [0]   # mutable counter for rotating LDC wdep slots

def _wdep_for_opcode(opcode: int, raw: bytes = None) -> int:
    """Assign the scoreboard write-dependency slot for an opcode."""
    if opcode == 0x7ac:  # LDCU
        if raw is not None and raw[9] == 0x0a:  # LDCU.64
            # First LDCU.64 (descriptor, c[0][0x358]) uses 0x35 so LDG gets rbar=0x09.
            # Subsequent LDCU.64s (pointer params) use 0x31/0x33 rotation to avoid
            # aliasing LDG's 0x35 scoreboard slot — if LDCU.64 appears between LDG and
            # its consumer, a 0x35 write would clear LDG's barrier prematurely.
            if _ldcu_slot_counter[0] == 0:
                _ldcu_slot_counter[0] += 1
                return 0x35  # first LDCU.64 = descriptor
            slots = [0x31, 0x33]
            slot = slots[(_ldcu_slot_counter[0] - 1) % len(slots)]
            _ldcu_slot_counter[0] += 1
            return slot
        # LDCU.32: always use 0x31 (LDC/LDCU scoreboard slot).
        # rbar bit2 (slot 0x33) stalls for LDCU.64 correctly, but NOT for
        # LDCU.32 on SM_120 — hardware asymmetry verified by test_imad_chain.
        # Consumer IMAD R-UR uses rbar bit1 (0x02) which correctly gates on 0x31.
        _ldcu_slot_counter[0] += 1
        return 0x31
    if opcode == 0x918:  # NOP: even wdep (misc=0 paired with 0x3e is safe)
        return 0x3e
    if opcode in _OPCODES_LDC:
        return 0x31
    if opcode in _OPCODES_LDS | _OPCODES_DFPU | _OPCODES_DSETP | _OPCODES_F2F:
        return 0x33  # DSETP, F2F also post via slot 0x33 (ptxas SM_120 ground truth)
    if opcode in _OPCODES_LDG | _OPCODES_ATOMG:
        return 0x35
    if opcode == 0x3c4:  # REDUX: writes to UR, posts to slot 0x31 like LDCU (ptxas-verified)
        return 0x31
    if opcode in _OPCODES_LDGSTS:
        return 0x3f  # LDGSTS: async copy writes to shared mem, not GPR — no scoreboard slot
    if opcode in _OPCODES_LDGDEPBAR:
        return 0x31  # LDGDEPBAR: commit group, posts to LDC slot (ptxas-verified)
    if opcode in _OPCODES_IADD64_UR:
        return 0x3e  # ALU slot — consumer LDG/STG gets rbar via pending_writes
    if opcode in _OPCODES_ALU | _OPCODES_SMEM_SETUP:
        return 0x3e
    # No write tracking for control flow (EXIT/BRA), stores, barriers
    return 0x3f


# Opcode-specific misc nibble overrides.
# Hybrid model: some opcodes have strict hardware requirements (override applied),
# others use the sequential instruction counter (no override).
#
# Counter model (verified against ptxas fp64_bench ground truth):
#   - misc_counter increments for EVERY instruction regardless of override
#   - For each instruction: misc = override if in _OPCODE_MISC, else misc_counter & 0xF
#   - Verified sequence for fp64 preamble:
#       S2R(ctr=0,ovr=1), LDCU×4(ctr=1-4,ovr=7), LDC R2(ctr=5)=5✓,
#       LDC R3(6)=6✓, LDC R4(7)=7✓, LDC R5(8)=8✓, S2UR(9)=9✓, LDC R7(10)=10✓,
#       IMAD(11,ovr=1), IADD3×10(12-21,ovr=1), ISETP(22,ovr=0)✓, BRA(23,ovr=1)✓
#
# Strict-requirement opcodes (hardware rejects wrong values):
#   LDCU (0x7ac): misc MUST be 7. SM_120 LDCU = SM_89 ULDC.64 which also requires 7.
#     Absent from all prior attempts → ILLEGAL_INSTRUCTION at first LDCU in preamble.
#   ISETP R-UR (0xc0c): misc 1-12 → WRONG PREDICATE on SM_120 (see encode_isetp_ur
#     docstring). misc=6 from counter at ISETP position (instruction 22) causes
#     inverted predicate → wrong BRA direction → DMUL/DADD execute → ILLEGAL_INSTRUCTION.
#   DMUL/DADD/DFMA: misc MUST be 2 (hardware-probed on RTX 5090 2026-03-27).
#
# Counter-based opcodes (no override — hardware accepts any counter value):
#   LDC (0xb82): ptxas uses counter values 5,6,7,8,10 in fp64 preamble.
#   S2UR (0x9c3): ptxas uses counter value 9 in fp64 preamble.
#   MOV (0x202): ptxas uses counter value (e.g., misc=4 at position 36).
_OPCODE_MISC: dict[int, int] = {
    0x918: 0,   # NOP: misc=0 (ptxas-verified)
    0x947: 1,   # BRA: misc=1 (ptxas-verified: @P0 BRA and loop-back BRA both use 1)
    0x94d: 5,   # EXIT: misc=5 (ptxas-verified: same for predicated and unconditional)
    0x981: 6,   # LDG.E: misc=6 (hardware-verified SM_120)
    0x7ac: 7,   # LDCU: misc=7 — CRITICAL FIX. SM_120 LDCU requires misc=7 (same as
                #   SM_89 ULDC.64). All 4 LDCU.64 in fp64 preamble use misc=7 per ptxas.
                #   counter gives 1,2,3,4 → ILLEGAL_INSTRUCTION at first LDCU.
    0x3a9: 4,   # ATOMG.CAS: misc=4 (from RTX 5090 probe 2026-03-27)
    0x9a8: 4,   # ATOMG.{ADD|MIN|MAX|EXCH}.u32: misc=4 (global-memory category)
    0x9a3: 4,   # ATOMG.ADD.F32: misc=4 (global-memory category)
    0x992: 0,   # MEMBAR: misc=0 (control/fence instruction)
    0x229: 2,   # DADD R-R: misc=2 (ptxas-verified: b1=0x72, src1 at b8)
    0x228: 2,   # DMUL: misc=2
    0x22b: 2,   # DFMA R-R: misc=2
    0xc2b: 2,   # DFMA R-UR-UR: misc=2 (b1=0x7c form)
    0x22a: 2,   # DSETP: misc=2 (ptxas ground truth — all DSETP variants use 2)
    0xc35: 5,   # IADD.64-UR: misc=5 (wide ALU result)
    0xc0c: 0,   # ISETP R-UR: misc=0 (SM_120: misc 1-12 → wrong predicate; see
                #   encode_isetp_ur docstring. Counter value 6 at position 22 → wrong pred.)
    0x20c: 0,   # ISETP R-R: misc=0 (same SM_120 predicate correctness requirement)
    0x80a: 5,   # FSEL.step: misc=5 (ptxas-verified)
    0x986: 1,   # STG.E: misc=1 (from ptxas ground truth)
    0x988: 4,   # STS.E: misc=4
    0x225: 1,   # IMAD.WIDE R-R: misc=1
    0x825: 1,   # IMAD.WIDE R-imm: misc=1
    0xc24: 1,   # IMAD R-UR: misc=1
    0x810: 1,   # IADD3.IMM: misc=1 (ptxas: all 10 IADD3.IMM in fp64 preamble use 1)
    0x919: 1,   # S2R: misc=1 (ptxas-verified; counter=0 at body start would give 0)
    0xfae: 4,   # LDGSTS.E: misc=4 (ptxas-verified: async global→shared copy)
    0x9af: 1,   # LDGDEPBAR: misc=1 (ptxas-verified: cp.async commit group)
    # Tensor core MMA: misc=2 (ptxas-verified for SM_120 HMMA, IMMA, DMMA)
    0x23c: 2,   # HMMA (FP16/BF16/TF32)
    0x237: 2,   # IMMA (INT8)
    0x23f: 2,   # DMMA (FP64)
    0x27a: 2,   # QMMA (FP8 E4M3/E5M2)
    # LDC (0xb82) and S2UR (0x9c3) intentionally omitted — use counter for correct values
}

# All opcodes recognised by assign_ctrl.  Unknown opcodes raise ValueError.
_ALL_KNOWN_OPCODES: frozenset = frozenset(
    _OPCODES_LDG | _OPCODES_LDC | _OPCODES_LDS |
    _OPCODES_STG | _OPCODES_STS | _OPCODES_BAR |
    _OPCODES_CTRL | _OPCODES_ALU | _OPCODES_IADD64_UR |
    _OPCODES_SMEM_SETUP | _OPCODES_ATOMG | _OPCODES_DFPU |
    _OPCODES_F2F | _OPCODES_LDGSTS | _OPCODES_LDGDEPBAR
)


def _patch_ctrl(raw: bytes, ctrl: int) -> bytes:
    buf = bytearray(raw)
    raw24 = (ctrl & 0x7FFFFF) << 1
    buf[13] = raw24 & 0xFF
    buf[14] = (raw24 >> 8) & 0xFF
    buf[15] = ((raw24 >> 16) & 0xFF) | (buf[15] & 0x04)  # preserve SHF reuse flag
    return bytes(buf)


def assign_ctrl(instrs: list[SassInstr]) -> list[SassInstr]:
    """
    Assign ctrl values to an instruction stream using scoreboard emulation.

    Tracks which registers are written by long-latency ops (LDG, LDC, LDS)
    and sets rbar on consumer instructions to wait for the result.
    """
    # Track which registers have pending long-latency writes
    # Maps reg_index → (slot_index, wdep_slot) for pending writes
    pending_writes: dict[int, tuple[int, int]] = {}
    # Track UR writes separately (LDCU destinations)
    pending_ur_writes: dict[int, tuple[int, int]] = {}  # ur_index → (slot_index, wdep)
    # Track predicate writes from ISETP/SETP instructions
    pending_pred_writes: dict[int, tuple[int, int]] = {}  # pred_reg → (slot_index, wdep)

    # rbar encoding: maps wdep_slot → rbar bit pattern
    _WDEP_TO_RBAR = {
        0x31: 0x03,   # LDC/LDCU slot → rbar=0x03
        0x33: 0x05,   # LDS/LDCU.32 slot → rbar=0x05
        0x35: 0x09,   # LDG slot → rbar=0x09 (all LDGs share this slot, ptxas-verified)
        0x3e: 0x03,   # ALU → rbar=0x03
    }

    misc_counter = 0
    _ldcu_slot_counter[0] = 0  # reset per kernel
    _ldc_slot_counter[0] = 0   # reset per kernel
    result = []

    for i, si in enumerate(instrs):
        opcode = _get_opcode(si.raw)
        if opcode not in _ALL_KNOWN_OPCODES:
            raise ValueError(f"assign_ctrl: unrecognized opcode 0x{opcode:03x} at instruction {i}")

        # Determine wdep for this instruction
        wdep = _wdep_for_opcode(opcode, si.raw)

        # All LDG instructions share wdep=0x35 (ptxas-verified: both LDG.E.64
        # in dual-load patterns use wdep=0x35). The hardware scoreboard tracks
        # the LAST write to a slot; consumer rbar=0x09 waits until the final
        # LDG posting to slot 0x35 completes. FIFO ordering in the load pipeline
        # guarantees earlier LDGs also complete. Using wdep=0x37 for the second
        # LDG is WRONG — slot 0x37 has no rbar bit, so consumers never wait for it.

        # Determine rbar: check if any source register has a pending long-latency write
        rbar = 0x01  # default: no wait
        src_regs = _get_src_regs(si.raw)
        for r in src_regs:
            if r in pending_writes:
                _, pending_wdep = pending_writes[r]
                candidate_rbar = _WDEP_TO_RBAR.get(pending_wdep, 0x01)
                # For LDG consumers: first consumer gets 0x09, subsequent get 0x03
                if pending_wdep == 0x35:
                    candidate_rbar = 0x09
                rbar = rbar | candidate_rbar

        # WAW (write-after-write) hazard: if this instruction writes to a register
        # that has a pending long-latency write, we must wait for the prior write
        # to complete before overwriting. Otherwise the prior slot may post its
        # result *after* this one, clobbering the new value. RAW handling above
        # only covers cases where this instruction also reads the register as a
        # source — it misses pure-overwrite cases (e.g. reload LDG → different
        # destination, or ALU write to a register that previously held an LDG
        # result that's still in flight).
        dest_regs_now = _get_dest_regs(si.raw)
        for r in dest_regs_now:
            if r in pending_writes:
                _, pending_wdep = pending_writes[r]
                # ALU writes (wdep=0x3e) retire in-order so no WAW wait needed
                # against another ALU write. Only long-latency prior writes matter.
                if pending_wdep in _WDEP_TO_RBAR and pending_wdep != 0x3e:
                    candidate_rbar = _WDEP_TO_RBAR[pending_wdep]
                    if pending_wdep == 0x35:
                        candidate_rbar = 0x09
                    rbar = rbar | candidate_rbar

        # STS needs rbar=0x09 if writing data that came from LDG
        if opcode in _OPCODES_STS:
            for r in src_regs:
                if r in pending_writes:
                    rbar = 0x09

        # STG: let the general rbar computation handle all source dependencies.
        # Both the address register (b3) and data register (b4) are included in
        # src_regs via _get_src_regs, so the loop above already computes the
        # correct rbar for ALU (wdep=0x3e→rbar=0x03), DFPU (wdep=0x33→rbar=0x05),
        # and LDG (wdep=0x35→rbar=0x09) dependencies.
        # Special case: LDG data (wdep=0x35) must use rbar=0x09, not the table default.
        if opcode in _OPCODES_STG:
            data_reg = si.raw[4]
            if data_reg in pending_writes:
                _, pw = pending_writes[data_reg]
                if pw == 0x35:
                    rbar = rbar | 0x09

        # ATOMG needs rbar=0x03 (memory ordering, same as STG)
        if opcode in _OPCODES_ATOMG:
            rbar = rbar | 0x03

        # MOV R, UR (0xc02): reads UR source from raw[4]; wait for REDUX/LDCU writes
        if opcode == 0xc02:
            ur_src = si.raw[4]
            if ur_src in pending_ur_writes:
                _, pw = pending_ur_writes[ur_src]
                if pw in _WDEP_TO_RBAR:
                    rbar = rbar | _WDEP_TO_RBAR[pw]
        # LDCU consumers: any instruction using UR operands needs rbar for LDCU
        # Check byte 4 for UR source in R-UR instructions
        if opcode in (0xc35, 0xc0c, 0xc24):  # IADD.64-UR, ISETP R-UR, IMAD R-UR
            ur_src = si.raw[4]
            if ur_src in pending_ur_writes:
                _, pw = pending_ur_writes[ur_src]
                if pw in _WDEP_TO_RBAR:
                    rbar = rbar | _WDEP_TO_RBAR[pw]
        # LDG/STG use descriptor from UR (LDG: b4=UR, STG: b8=UR)
        # NOTE: UR4 descriptor is loaded via LDCU in the preamble with
        # hardcoded ctrl. Body LDCUs (for pointer params) also track in
        # pending_ur_writes. Only apply rbar if UR4 is in pending_ur_writes
        # (i.e., LDCU UR4 went through the scoreboard in the body).
        if opcode in _OPCODES_LDG:
            ur_desc = si.raw[4]
            if ur_desc in pending_ur_writes:
                _, pw = pending_ur_writes[ur_desc]
                if pw in _WDEP_TO_RBAR:
                    rbar = rbar | _WDEP_TO_RBAR[pw]
        # STG UR descriptor: ptxas uses rbar=1 for STG, relying on instruction
        # scheduling to ensure the descriptor is available. Don't override rbar
        # for the UR descriptor — it's guaranteed ready by the time STG executes.
        # (LDG DOES need rbar for the descriptor — see above.)

        # Check predicate-register hazards: any instruction guarded by @Px or @!Px
        # must wait for the instruction that wrote Px to complete.
        # Strip the negation bit (bit 3) to get the pred reg index 0-6.
        guard = (si.raw[1] >> 4) & 0xF
        pred_idx = guard & 0x7
        if guard != 0x7 and pred_idx in pending_pred_writes:  # 0x7 = PT (unconditional)
            _, pw = pending_pred_writes[pred_idx]
            candidate = _WDEP_TO_RBAR.get(pw, 0x01)
            rbar = rbar | candidate

        # BAR.SYNC: all threads synchronize, so all prior memory operations
        # are guaranteed complete. Clear pending_writes so post-barrier
        # instructions start with a clean scoreboard. Without this, stale
        # pending_writes entries from pre-barrier LDG/LDC could cause
        # post-barrier consumers to wait on already-resolved slots, or worse,
        # confuse the scoreboard when a post-barrier LDG reuses slot 0x35
        # that was tracked for a pre-barrier LDG.
        if opcode in _OPCODES_BAR:
            wdep = 0x3f
            rbar = 0x01
            pending_writes.clear()
            pending_ur_writes.clear()
            pending_pred_writes.clear()
        if opcode == 0x94d:  # EXIT — always wdep=0x3f, misc=5 (ptxas-verified)
            # ptxas uses identical ctrl for both predicated and unconditional EXIT
            rbar = 0x01
            wdep = 0x3f

        # Build ctrl — bits[22:17] = OPEX (instruction extension / hardware modifier).
        # These are NOT stall counters on SM_120. Each opcode has a fixed OPEX value
        # determined by reverse-engineering ptxas output. Wrong OPEX → hardware
        # misinterprets the instruction (e.g., LDG without OPEX=15 crashes).
        # SM_120: OPEX bits (ctrl[22:17]) are ALWAYS 0 for all opcodes.
        # Verified by extracting ctrl from multiple ptxas SM_120 cubins.
        # Non-zero OPEX corrupts the instruction encoding.
        stall = 0
        if opcode == 0x94d:  # EXIT: if predicated (@Px EXIT)
            guard = (si.raw[1] >> 4) & 0xF
            if guard != 0x7:  # 0x7 = PT (unconditional); any other guard = @Px
                # Predicated EXIT: reset LDCU slot counter so post-branch LDCU
                # instructions start from slot 0 again, matching ptxas behavior.
                _ldcu_slot_counter[0] = 0
        # BRA (opcode 0x947) with a non-PT guard also resets the LDCU counter.
        if opcode == 0x947:
            guard = (si.raw[1] >> 4) & 0xF
            if guard != 0x7:
                _ldcu_slot_counter[0] = 0
        # Misc nibble: hybrid counter+override model.
        # misc_counter increments for every instruction regardless of override.
        # Opcodes in _OPCODE_MISC get the override value; all others use the counter.
        # This matches ptxas ground truth for fp64_bench preamble exactly.
        misc = _OPCODE_MISC.get(opcode, misc_counter & 0xF)
        ctrl = (stall << 17) | (rbar << 10) | (wdep << 4) | misc
        misc_counter += 1

        # Track this instruction's writes for future consumers.
        dest_regs = _get_dest_regs(si.raw)
        if wdep != 0x3f:
            for r in dest_regs:
                pending_writes[r] = (i, wdep)
        # Track UR writes: LDCU (0x7ac), S2UR (0x9c3), REDUX (0x3c4)
        if opcode == 0x7ac:
            ur_dest = si.raw[2]  # UR destination index
            pending_ur_writes[ur_dest] = (i, wdep)
            # Also track UR+1 for 64-bit pairs
            pending_ur_writes[ur_dest + 1] = (i, wdep)
        elif opcode == 0x9c3:  # S2UR: writes single UR (dest at byte 2)
            ur_dest = si.raw[2]
            pending_ur_writes[ur_dest] = (i, wdep)
        elif opcode == 0x3c4:  # REDUX: writes result to UR dest (raw[2]), wdep=0x31
            ur_dest = si.raw[2]
            pending_ur_writes[ur_dest] = (i, wdep)
        # ALU writes (wdep=0x3e) DO need to be tracked for GPR-gap enforcement.
        # The min_gpr_gap ensures ≥1 instruction between ALU write and consumer.
        # Do NOT clear pending_writes here — consumers need the dependency info.
        # Track predicate writes from ISETP/FSETP: pred dest location varies by opcode
        if opcode == 0xc0c:  # ISETP R-UR: pred_dest at raw[2]
            pred_dest = si.raw[2]  # destination predicate index (0..6)
            pending_pred_writes[pred_dest] = (i, wdep)
        elif opcode == 0x20c:  # ISETP R-R: pred_dest at (raw[10]>>1) & 0x7
            pred_dest = (si.raw[10] >> 1) & 0x7
            pending_pred_writes[pred_dest] = (i, wdep)
        elif opcode == 0x20b:  # FSETP: pred_dest at raw[9] & 0x7
            pred_dest = si.raw[9] & 0x7
            pending_pred_writes[pred_dest] = (i, wdep)
        # DSETP: predicate write is long-latency (wdep=0x33, same slot as LDS).
        # pred_dest encoded in raw[10] as 0xf0|(pred<<1); extract the 0-6 index.
        if opcode in _OPCODES_DSETP:
            pred_dest = (si.raw[10] >> 1) & 0x7
            pending_pred_writes[pred_dest] = (i, 0x33)
        # Note: we do NOT clear pending_writes for consumed SOURCE registers.
        # Multiple instructions may read from the same LDG output, and each
        # needs the rbar wait independently.

        patched = _patch_ctrl(si.raw, ctrl)
        result.append(SassInstr(patched, si.comment))

    return result
