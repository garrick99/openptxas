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
    0xc24: _OpMeta('IMAD.RU',    1, 0x3e, 1),  # IMAD R-UR (FG-1 closeout): same ALU class
                                                # as the other IMAD variants; 1-instruction
                                                # GPR latency is required before any reader
                                                # of the dest.  Previously absent from the
                                                # scoreboard model → _enforce_gpr_latency
                                                # did not insert a NOP after fused mul+add,
                                                # producing the FG-1.14A/B/C anomalies.
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
    0x223: _OpMeta('FFMA',       0, 0x00, 1),  # FFMA: wdep=0 (ptxas pattern, pipeline handles ordering)
    0x308: _OpMeta('MUFU',       1, 0x3e, 1),  # MUFU (SFU: RCP, SQRT, SIN, COS, EX2, LG2)
    0x309: _OpMeta('POPC',       1, 0x31, 1),  # POPC (population count) — long-latency, LDC-class slot per ptxas
    0x301: _OpMeta('BREV',       1, 0x3e, 1),  # BREV (bit reverse)
    0x300: _OpMeta('FLO',        1, 0x31, 1),  # FLO/CLZ — long-latency, LDC-class slot per ptxas
    0x816: _OpMeta('PRMT.IMM',   1, 0x3e, 6),  # PRMT imm-selector: needs 1-cycle gap before dependent consumer (FLO, etc.)
    0x416: _OpMeta('PRMT.IMM.L', 1, 0x3e, 6),  # PRMT legacy imm-selector form — matches 0x816 behavior
    0x216: _OpMeta('PRMT.REG',   1, 0x3e, 6),  # PRMT reg-selector
    0x820: _OpMeta('FMUL.IMM',   1, 0x3e, 1),  # FMUL with 32-bit float immediate
    0x823: _OpMeta('FFMA.IMM',   0, 0x00, 1),  # FFMA.IMM: wdep=0, min_gpr_gap=0 (matches plain FFMA; ptxas emits back-to-back with no NOPs)
    0x80a: _OpMeta('FSEL.STEP',  1, 0x3e, 5),  # Combined float compare+select (misc=5, ptxas-verified)
    0x235: _OpMeta('IADD.64',    1, 0x3e, 1),
    # Phase 2 (edge_87): 32-bit IADD shares opcode 0x235 with IADD.64 but
    # selects the 32-bit ALU path via byte[9]=0x00 (vs 0x02 for IADD.64).
    # Looked up via the synthetic key 0x1235 by `_disc_opcode(raw)` so its
    # meta + forwarding rules can differ from IADD.64.  Base min_gpr_gap=1
    # (default for safety); forwarding-safe consumers are listed in
    # _FORWARDING_SAFE_PAIRS.  See _harvest/prompts/iadd_probes/REPORT.md
    # for empirical evidence.
    0x1235: _OpMeta('IADD',     1, 0x3e, 1),
    0xc35: _OpMeta('IADD.64-UR', 1, 0x3e, 5),  # misc=5 per hardware bisect 2026-03-25
    0x202: _OpMeta('MOV',        0, 0x3e, 1),
    0x802: _OpMeta('MOV.IMM',    0, 0x3e, 1),  # GPR-immediate move (b0=0x02, b1=0x78); same scheduling as 0x202.
    0x20b: _OpMeta('FSETP.RR',   0, 0x3c, 5),  # FSETP R-R: misc=5, wdep=0x3c (ptxas-verified from forced_match)
    0x808: _OpMeta('FSEL.IMM',   1, 0x3e, 1),  # FSEL with 32-bit immediate (float select)
    0x20c: _OpMeta('ISETP.RR',   0, 0x3e, 0),  # ISETP R-R: misc=0 (SM_120 predicate)
    0xc0c: _OpMeta('ISETP.RU',   0, 0x3e, 0),  # ISETP R-UR: misc=0 on SM_120
    0x431: _OpMeta('HFMA2',      1, 0x3e, 1),  # HFMA2 (half-precision FMA2, used as zero-init in div.u32)
    0x810: _OpMeta('IADD3.IMM',  1, 0x3e, 1),  # IADD3 with 32-bit immediate operand
    0x812: _OpMeta('LOP3.IMM',   1, 0x3e, 1),  # LOP3 with 32-bit immediate (IMAD-FUSE-1)
    0x306: _OpMeta('I2F.U32.RP', 1, 0x3e, 1),  # I2F unsigned int to float, round toward +inf
    0x305: _OpMeta('F2I.FTZ.U32',1, 0x3e, 1),  # F2I float to unsigned int, truncate
    0x310: _OpMeta('F2F',        1, 0x33, 1),  # F2F float-to-float precision conversion (F32↔F64), long-latency wdep=0x33
    0x311: _OpMeta('F2I.F64',   1, 0x3e, 1),  # F2I.F64 float64-to-int32 conversion
    0x312: _OpMeta('I2F.F64',   1, 0x3e, 1),  # I2F.F64 int32-to-float64 conversion (writes pair)
    0x81a: _OpMeta('BFE_SEXT',  1, 0x3e, 1),  # BFE sign-extension step (bfe.s32 lowering)
    0x207: _OpMeta('SEL',       0, 0x3e, 1),  # SEL: register select (P3-2)
    0x22a: _OpMeta('DSETP',     0, 0x3e, 0),  # DSETP FP64 compare → predicate (misc=0, like ISETP)
    # Tensor core MMA: dedicated wdep slot 0x32 with rbar bit 0x11 (bit 4 +
    # gate bit 0).  Previously used 0x3e (ALU class), which was too short a
    # latency class — consumer of HMMA dest read stale data.  The new slot
    # is registered in _WDEP_TO_RBAR so consumers properly wait.  Surfaced
    # by mower hmma probe (2026-04-29).
    0x23c: _OpMeta('HMMA',      1, 0x32, 2),  # HMMA FP16/BF16/TF32 MMA (m16n8k*)
    0x237: _OpMeta('IMMA',      1, 0x32, 2),  # IMMA INT8 MMA (m16n8k32)
    0x23f: _OpMeta('DMMA',      1, 0x32, 2),  # DMMA FP64 MMA (m8n8k4)
    0x27a: _OpMeta('QMMA',      1, 0x32, 2),  # QMMA FP8 E4M3/E5M2 MMA (m16n8k32)
    0x83b: _OpMeta('LDSM',      1, 0x33, 2),  # LDSM load shared→matrix regs (wdep=LDS slot)
    0x3c4: _OpMeta('REDUX',     0, 0x3f, 0),  # REDUX warp reduction → UR (no GPR dest)
    0xc02: _OpMeta('MOV.UR',   1, 0x3e, 1),  # MOV R, UR — copy uniform reg to GPR
    0x226: _OpMeta('IDP.4A',   1, 0x3e, 1),  # IDP.4A dp4a (integer dot product)
    # Phase 3 opcodes
    0x211: _OpMeta('LEA',      1, 0x3e, 1),  # LEA load effective address
    0x811: _OpMeta('LEA.IMM',  1, 0x3e, 1),  # LEA with immediate index
    0x217: _OpMeta('IMNMX',    1, 0x3e, 1),  # IMNMX integer min/max
    0x248: _OpMeta('VIMNMX.RR',1, 0x3e, 1),  # VIMNMX R-R (3-operand integer min/max).
                                             # Same ALU class + min_gpr_gap=1 as IMNMX.
                                             # Was missing → gap enforcement skipped →
                                             # VIMNMX reading a back-to-back-materialized
                                             # immediate source (e.g. IADD3.IMM R9=7
                                             # directly before VIMNMX R4=max(R5,R9))
                                             # read R9 stale, returned the wrong operand
                                             # for inputs where R5 < the materialized
                                             # constant.  Observed in the fuzz-discovered
                                             # popc+bfe+bfi+max+bfi minimal (7/32 wrong
                                             # post-POPC-fix); all residuals had popc(input)<7.
    0x848: _OpMeta('IMNMX.IMM',1, 0x3e, 1),  # IMNMX integer min/max with immediate
                                              # (parallel to IMAD R-R 0x224 / R-imm 0x824).
                                              # Same ALU class as 0x217.  Emitted by PTXAS
                                              # only when it recognizes the clamp idiom
                                              # (mov + setp gt + @P mov + setp lt + @P mov).
                                              # IMNMX01-04 evidence: r1_minmax PTXAS cubin
                                              # at [9],[10]; runtime GPU correctness matches
                                              # PTX-body Python simulation.
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
    0xc0b: _OpMeta('FSETP_UR',     0, 0x3c, 10), # FSETP R-UR (ptxas-verified ctrl)
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
                 0x9a8,   # ATOMG.E.{ADD|MIN|MAX|EXCH|OR|AND}.u32
                 0x98e,   # ATOMG.E.XOR.b32 (BREAK-1A, 0x98e family)
                 0x9a3}   # ATOMG.E.ADD.F32
_OPCODES_LDC = {0xb82, 0x7ac, 0x919, 0x9c3,  # SM_120: LDC, LDCU, S2R, S2UR
                0x624, 0xab9, 0xa02}           # SM_89: IMAD.MOV.U32(cbuf), ULDC.64, MOV(cbuf)
_OPCODES_LDS = {0x984, 0x83b}  # LDS, LDSM (load shared to matrix)
_OPCODES_STG = {0x986, 0xf9d}  # STG, SUST
_OPCODES_STS = {0x988, 0x388}
_OPCODES_BAR = {0xb1d, 0x941}  # BAR.SYNC + BSYNC
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
    0x812,        # LOP3.LUT.IMM (SM_120, 32-bit immediate form)
    0xa12,        # LOP3.LUT (SM_89 cbuf form)
    0x819,        # SHF (SM_120, all variants)
    0xa19,        # SHF (SM_89)
    0x299,        # SHF.VAR (variable-shift SHF, shift amount in register)
    0x219,        # SHF.R.S32.HI.VAR (arithmetic right shift, variable amount)
    # Select / predicate
    0x207,        # SEL (SM_120)
    0x807,        # SEL (SM_89, imm form)
    0xa0c,        # ISETP (SM_89 cbuf form)
    0x208,        # FSEL (register)
    0x808,        # FSEL (immediate)
    0x20b,        # FSETP
    0xc0b,        # FSETP R-UR

    0x20c,        # ISETP R-R
    0x80c,        # ISETP IMM (32-bit immediate)
    0xc0c,        # ISETP R-UR
    # Permute / misc
    0x816,        # PRMT (immediate selector, opc=0x816, ptxas-verified sm_120)
    0x416,        # PRMT (legacy imm-selector 0x416 — kept for backward compat)
    0x216,        # PRMT.REG (register selector, opc=0x216)
    0x589, 0xf89, 0x989,  # SHFL (reg-reg, reg-imm, imm-imm)
    # 0x806 VOTE removed from ALU — uses wdep=0x3F (see _wdep_for_opcode)
    # Matrix multiply (HMMA, IMMA, DMMA, QMMA)
    0x23c, 0x237, 0x23f, 0x27a,
    # Predicate ↔ register moves
    0x203,        # P2R (predicate-to-register move)
    0x204,        # R2P (register-to-predicate move)
    # Address calculation
    0x211,        # LEA (address calculation)
    0x811,        # LEA.IMM (address calc with immediate)
    0x835,        # UIADD: uniform add (P3-3, dual GPR+UR write, no _OPCODE_META entry)
    0x886,        # UR pipeline init (P3-5, descriptor setup for 0x98e ATOMG)
    0x2bd,        # UR pipeline finalize (P3-5, descriptor finalize for 0x98e ATOMG)
    0xd09,        # AT06 UR-pipeline data-routing op for atom.add K=1 imm_data variant
                  # (template-only; never emitted by isel directly outside the
                  # imm_data_K1 atom-UR template path)
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
    0xc11,        # TE12: IADD3.R-UR (carry-chain 64-bit address add)
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
    0x304,        # CVT.F16.F32 (FP32→FP16, low 16 bits)
    # Warp reduction (decoded 2026-04-01) — writes UR, not GPR
    0x3c4,        # REDUX.SUM (warp sum → UR)
    # MOV R, UR — copy uniform register to GPR (after REDUX)
    0xc02,        # MOV R, UR
    0x802,        # MOV.IMM (32-bit immediate → GPR; replaces IADD3-imm-as-MOV)
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


# Phase 2 (edge_87): discriminating-opcode lookup.  Same as _get_opcode for
# everything except IADD: opcode 0x235 with byte[9]=0x00 is 32-bit IADD
# (synthetic key 0x1235), byte[9]=0x02 is IADD.64 (kept as 0x235).  The two
# share the same raw opcode but use different ALU paths and different
# scoreboard rules per Phase 0 empirical study (_harvest/prompts/iadd_probes/
# REPORT.md).  Pass the result of this function (not _get_opcode) to
# _OPCODE_META.get() and _is_forwarding_safe_pair when distinguishing the
# two variants matters.
def _disc_opcode(raw: bytes) -> int:
    opc = struct.unpack_from('<Q', raw, 0)[0] & 0xFFF
    if opc == 0x235 and len(raw) >= 10 and raw[9] == 0x00:
        return 0x1235
    return opc


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

    # FG-2.4: explicit per-opcode cases for opcodes that are emitted by
    # OpenPTXas but not in any of the opcode-class sets below.  These
    # are dispatched first so the generic set-based chain cannot
    # silently route them through a fallback that returns the wrong
    # source set.
    if opcode == 0x812:
        # LOP3.IMM / IADD3 R-imm: b3 = src0 GPR, b4..b7 = 32-bit imm, b8 = src2 GPR.
        if raw[3] < 255: regs.add(raw[3])
        if raw[8] < 255: regs.add(raw[8])
        return regs
    if opcode == 0x835:
        # IADD.64 R-imm: b3:b3+1 = src0 pair, b4..b7 = 32-bit imm.
        # Unlike IADD3.IMM the 64-bit form has no src2 at b8 — empirical
        # evidence from smem_exchange/bar_ldc_xor PARITY kernels shows
        # b8 is a reserved/padding byte (typically 0x00) that the
        # hardware does not treat as a source register.
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
        return regs
    if opcode == 0xc11:
        # LEA R-UR: b3 = base GPR, b4 = UR (not a GPR source), b9 = shift.
        if raw[3] < 255: regs.add(raw[3])
        return regs
    if opcode == 0xc12:
        # IADD3X R-UR: b3 = src0 GPR, b4 = UR (not a GPR), b8 = src2 GPR.
        if raw[3] < 255: regs.add(raw[3])
        if raw[8] < 255: regs.add(raw[8])
        return regs
    if opcode == 0xc25:
        # IMAD.WIDE R-UR: b3 = src0 GPR, b4 = UR, b8:b8+1 = src2 pair.
        if raw[3] < 255: regs.add(raw[3])
        if raw[8] < 255: regs |= {raw[8], raw[8]+1}
        return regs
    if opcode == 0x202:
        # MOV (register-to-register copy).
        # Encoder ground truth (sm_120_opcodes.py line 198):
        #   b2 = dest, b3 = 0x00 (fixed), b4 = src.
        # The src is at b4 — NOT at b3 — so the generic ALU fallback
        # that returns {b3} produced {R0} regardless of the real source.
        # FG-2.5 fix: return {b4} explicitly.
        if raw[4] < 255: regs.add(raw[4])
        return regs
    if opcode == 0x802:
        # MOV.IMM — 32-bit immediate to GPR; b4..b7 = imm32 (not a source).
        # No GPR sources.
        return regs

    if opcode in _OPCODES_LDG:
        # LDG: src_addr at b3
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
    elif opcode in _OPCODES_ATOMG:
        # ATOMG ops: addr at b3 (64-bit pair)
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
        # All ATOMG families: data at b4
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
        # STS: addr_reg at b3, data at b4 (UR variant has addr via UR descriptor + R addr)
        if raw[3] < 255: regs.add(raw[3])
        if raw[4] < 255: regs.add(raw[4])
    elif opcode in _OPCODES_LDS:
        # LDS: addr_reg at b3 (when GPR-addressed via encode_lds_r); immediate-only variants
        # set b3=0xFF.
        if raw[3] < 255: regs.add(raw[3])
    elif opcode in _OPCODES_IADD64_UR:
        # IADD.64-UR: GPR src pair at b3 (b4 is UR, not tracked here)
        if raw[3] < 255: regs |= {raw[3], raw[3]+1}
    elif opcode in _OPCODES_ALU:
        # ALU: src0 at b3, src1 at b4, src2 at b8 (varies by opcode)
        # Unary ops (MUFU, POPC, BREV, FLO, IABS): src at b4, b3=0x00 (not a real source)
        if opcode in (0x308, 0x309, 0x301, 0x300, 0x213,
                       0x245, 0x305, 0x306, 0x311, 0x312):
            # MUFU/POPC/BREV/FLO/IABS/I2FP/F2I (both F32/F64): single src at b4 only
            if raw[4] < 255: regs.add(raw[4])
        elif opcode in (0x210, 0x212, 0x810):  # IADD3/LOP3/IADD3.IMM: src0=b3, src1=b4, src2=b8
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255: regs.add(raw[8])
        elif opcode in (0x207, 0x20b, 0xc0b):  # SEL/FSETP/FSETP-UR: src0=b3, src1=b4
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
        elif opcode in (0x816, 0x416):  # PRMT imm: src0=b3, selector in b4-b7 (imm), src1=b8
            if raw[3] < 255: regs.add(raw[3])
            if raw[8] < 255: regs.add(raw[8])
        elif opcode == 0x216:  # PRMT.REG: src0=b3, sel_reg=b4, src1=b8
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255: regs.add(raw[8])
        elif opcode == 0x235:
            # Phase 2 (edge_87): 32-bit IADD (b9=0x00) reads single GPR
            # sources; IADD.64 (b9=0x02) reads 64-bit pair sources.
            if raw[9] == 0x00:
                if raw[3] < 255: regs.add(raw[3])
                if raw[4] < 255: regs.add(raw[4])
            else:
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
        elif opcode == 0x304:  # F2F.F16.F32: src at both b3 and b4 (same register)
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
        elif opcode in (0x820, 0x823, 0x80a, 0x808, 0x80c):  # FMUL.IMM/FFMA.IMM/FSEL.step/FSEL.imm/ISETP.IMM: src0=b3, b4-b7=imm
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
        elif opcode == 0x248:  # VIMNMX R-R: src0=b3, src1=b4 (integer min/max 32-bit)
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
        elif opcode == 0x226:  # IDP.4A: src_a=b3, src_b=b4, src_c=b8
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
            if raw[8] < 255: regs.add(raw[8])
        # FG-2.4: precise per-opcode cases added to replace the generic
        # fallback below for every opcode OpenPTXas is known to emit.
        # Each entry documents the actual GPR source layout per the
        # encoder in sass.encoding.sm_120_opcodes.
        elif opcode == 0xf89:
            # SHFL reg-imm (SHFL.DOWN / SHFL.BFLY form).  Encoder:
            #   raw[3] = src (GPR to shuffle)
            #   raw[4..7] = immediate lane + clamp + mode, NO GPR
            #   raw[10] = predicate dest (PT encoded), not a GPR source
            # Only b3 is a real GPR source.
            if raw[3] < 255: regs.add(raw[3])
        elif opcode == 0x589:
            # SHFL reg-reg form (shuffle lane in a GPR).  Encoder uses
            # b3 for data source and b4 for lane source.
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
        elif opcode == 0x989:
            # SHFL imm-imm form.  Same as 0xf89 — b3 is the single GPR
            # source; remaining fields are immediates.
            if raw[3] < 255: regs.add(raw[3])
        elif opcode == 0x209:
            # FMNMX float min/max (R-R).  src0=b3, src1=b4.
            if raw[3] < 255: regs.add(raw[3])
            if raw[4] < 255: regs.add(raw[4])
        elif opcode == 0x812:
            # IADD3-family R-imm variant.  src0=b3; b4..b7 are the
            # 32-bit immediate; b8 is the third source (usually RZ).
            if raw[3] < 255: regs.add(raw[3])
            if raw[8] < 255: regs.add(raw[8])
        elif opcode == 0x835:
            # IADD.64 R-imm variant.  src0 pair at b3:b3+1; b4..b7 imm;
            # b8 is an additional operand (often RZ).
            if raw[3] < 255: regs |= {raw[3], raw[3]+1}
            if raw[8] < 255: regs.add(raw[8])
        elif opcode == 0xc11:
            # LEA R-UR form (dest = base + (index << scale)).
            # b3 = base GPR, b4 = UR index (NOT a GPR), b9 = shift imm.
            if raw[3] < 255: regs.add(raw[3])
        elif opcode == 0xc12:
            # IADD3X R-UR variant.  b3 = src0 GPR, b4 = UR (not GPR),
            # b8 = src2 GPR.
            if raw[3] < 255: regs.add(raw[3])
            if raw[8] < 255: regs.add(raw[8])
        elif opcode == 0xc25:
            # IMAD.WIDE R-UR variant.  Like encode_imad_ur but writes
            # a GPR pair.  Source layout: b3 = src0 GPR, b4 = UR,
            # b8 = src2 (pair, like 0x825/0x225).
            if raw[3] < 255: regs.add(raw[3])
            if raw[8] < 255: regs |= {raw[8], raw[8]+1}
        else:
            # Default: src0 at b3 (generic ALU).
            # FG-2.4 kept as safety net; every opcode currently emitted
            # by OpenPTXas has an explicit case above.
            if raw[3] < 255: regs.add(raw[3])
    elif opcode in _OPCODES_F2F:
        # F2F: src at b4. F2F.F32.F64 (b9=0x10, narrowing) reads f64 pair;
        # F2F.F64.F32 (b9=0x18, widening) reads single f32.
        if raw[4] < 255:
            regs.add(raw[4])
            if raw[9] == 0x10:  # F2F.F32.F64: src is f64 pair
                regs.add(raw[4] + 1)
    return regs


# ---------------------------------------------------------------------------
# FG-2.4: forwarding-safe reader / producer-consumer pair model
# ---------------------------------------------------------------------------
#
# The _OPCODE_META.min_gpr_gap rule is a coarse "any reader of the
# writer's dest needs ≥1 instruction of gap" policy.  It is correct
# in the worst case but conservative for many producer-consumer
# pairs where hardware operand-forwarding handles the RAW without
# a NOP.  PTXAS ground truth across the 21-kernel workbench shows
# it emitting certain 0-gap ALU→ALU and ALU→consumer patterns
# repeatedly in byte-for-byte PARITY with OpenPTXas, proving those
# patterns are hardware-safe.
#
# Rather than making the decoder return incomplete source sets
# (which would introduce false negatives in novel kernels), we
# keep `_get_src_regs` faithful and model the forwarding-safe
# pairs here as a separate policy consulted by `verify_schedule`.
#
# Each entry is a (writer_opc, reader_opc) tuple where:
#  * The hardware 1-cycle ALU result latency is covered by
#    operand forwarding into the reader's read stage, OR
#  * The reader's ctrl-word (rbar/wdep) scoreboard slot covers the
#    RAW, OR
#  * The reader is a SHFL/STG family instruction whose pipeline
#    reads the source operand in a stage that naturally sees the
#    preceding ALU writer's forwarded value.
#
# Pairs NOT on this list are still subject to the standard
# min_gpr_gap rule, so this list is strictly a "permission to
# skip the 0-gap check for these specific pairs" — false
# negatives require a producer opcode whose hardware truly needs
# a wait-state AND a corresponding entry here that exempts it.
#
# Evidence: every pair in this set is found in a PARITY workbench
# kernel that byte-matches PTXAS.  Removing an entry loses nothing
# but conservatism; adding an entry without PTXAS evidence is
# forbidden (see tests/test_fg23_model_complete.py::INV F).
_FORWARDING_SAFE_PAIRS: set[tuple[int, int]] = {
    # Integer ALU → integer ALU (most common path).
    # IMAD.32 / IMAD.RR writing its accumulator (dest == src2) feeds
    # directly into IADD.64 / IADD3 / next IMAD.  Observed in
    # conv2d_looped, conv2d_unrolled, reduce_sum, etc.
    (0x224, 0x235),   # IMAD.32 → IADD.64
    (0x2a4, 0x235),   # IMAD.RR → IADD.64
    (0x210, 0x235),   # IADD3   → IADD.64
    (0x212, 0x235),   # IADD3X  → IADD.64
    (0x224, 0x210),   # IMAD.32 → IADD3
    (0x235, 0x210),   # IADD.64 → IADD3
    # ALU → SHFL (warp shuffle): SHFL's reg-imm form reads the source
    # GPR in its shuffle pipeline stage, which sees the forwarded
    # value from the previous cycle.
    (0x235, 0xf89),   # IADD.64 → SHFL
    (0x221, 0xf89),   # FADD    → SHFL
    (0xf89, 0xf89),   # SHFL    → SHFL (chained shuffles in reduce_sum)
    # ALU → STG: store addr/data pair read covered by STG's own
    # read barrier, not instruction-stream gap.
    (0xc35, 0x986),   # IADD.64-UR → STG.E
    (0xc11, 0x986),   # TE21: IADD3.R-UR → STG.E (carry-chain addr → store)
    (0xc11, 0x981),   # TE21: IADD3.R-UR → LDG.E (carry-chain addr → load)
    (0x812, 0x812),   # TE28: LOP3→LOP3 (13 PTXAS gap=0 instances)
    (0x812, 0x824),   # TE28: LOP3→IMAD (2 PTXAS gap=0 instances)
    (0x812, 0x235),   # TPL13: LOP3.IMM → IADD.64 (PTXAS gap=0 in r1_running_xor template at [8]→[9])
    (0x235, 0x812),   # TPL13: IADD.64 → LOP3.IMM (PTXAS gap=0 in r1_running_xor template at [9]→[10])
    # IMNMX01-04: clamp-idiom RAW chain in r1_minmax template.
    # PTXAS evidence: r1_minmax SASS [8] LOP3 → [9] IMNMX.IMM at gap=0,
    # then [9] → [10] IMNMX.IMM → IMNMX.IMM at gap=0.  Both pairs verified
    # GPU-correct by PTXAS cubin runtime output matching PTX-body simulation.
    (0x812, 0x848),   # IMNMX01: LOP3.IMM → IMNMX.IMM (clamp-idiom entry)
    (0x848, 0x848),   # IMNMX01: IMNMX.IMM → IMNMX.IMM (clamp-idiom self-chain)
    (0xc11, 0xc11),   # TE21: IADD3.R-UR → IADD3.R-UR (lo → hi carry chain)
    (0xc11, 0x235),   # TE21: IADD3.R-UR → IADD.64
    (0xc02, 0x986),   # MOV.UR     → STG.E
    (0x221, 0x986),   # FADD       → STG.E
    (0x235, 0x986),   # IADD.64    → STG.E
    # FG-4.2 additions: GPU-runtime evidence from the FG-4.0
    # adversarial harness false-positive replay + dedicated
    # microbenchmarks in probe_work/fg42_evidence_harness.py.
    # Each entry is confirmed by BOTH a PTXAS cubin that emits the
    # pair at gap=0 AND a GPU runtime output that matches the
    # Python-computed expected value (non-trivial computation).
    (0x224, 0x986),   # IMAD.32 → STG.E
                      # Evidence: f1_mad_acc_mad_lo_u32 replay —
                      # PTXAS emits the pair at gap=0, OURS and
                      # PTXAS both produce correct output 1 from
                      # `mad.lo.u32 r1, r0, r0, r1`.
    (0x819, 0x986),   # SHF → STG.E
                      # Evidence: A_shf_stg dedicated probe —
                      # PTXAS emits the pair at gap=0, consumer
                      # rbar=0x05 does not carry ALU class bit,
                      # runtime output 0xABCD << 3 = 351848 matches
                      # expected.
    # SHF → ISETP R-R: SHF's output forwarded into ISETP compare stage.
    (0x819, 0x20c),   # SHF (R-imm) → ISETP R-R
    # FG-4.8: SHF → IADD.64.
    # Evidence: in-place cubin-mutation forensics on the original
    # f6_random_s45058_l10 PTXAS cubin (which has SHF at [4] and
    # IADD.64 at [5] at gap=0).  Replaced S2R with MOV R0=0x42,
    # then compared three variants:
    #   SHF shift=2 present:  out = -(0x42>>2)  = 0xFFFFFFF0
    #   SHF shift=5 present:  out = -(0x42>>5)  = 0xFFFFFFFE
    #   SHF replaced by NOP:  out = -(0x42)     = 0xFFFFFFBE
    # The first two values differ from the NOP control — proving the
    # IADD.64 at gap=0 reads the SHF's shifted R0, not the stale
    # pre-SHF value.  Ctrl words: SHF wdep=0x3e, IADD.64 rbar=0x01
    # (no class bit covers the dependency).  Hardware forwarding is
    # the only mechanism.
    (0x819, 0x235),   # SHF → IADD.64
    # FG-4.2 addition: IADD.64 self-chain.
    # Evidence: B_iadd64_chain dedicated probe — PTXAS emits two
    # consecutive IADD.64 instructions at gap=0 where the second
    # reads the first's dest pair; consumer rbar=0x01 carries no
    # class bits; runtime output 3*arg1 matches expected for
    # arg1 = 0x123456789ABCDEF0.
    (0x235, 0x235),   # IADD.64 → IADD.64
    # FG-4.3 additions: clones of three of the four remaining FG-4.0
    # false-positive F6 random kernels were built by substituting
    # `mov.u32 %r0, %tid.x` → `ld.param.u32 %r0, [arg1]` so the
    # computation flows through a non-trivial constant
    # (arg1=0x12345678).  Expected outputs were computed by a
    # small PTX-body simulator in probe_work/fg42_evidence_harness.py
    # and compared to the GPU runtime output of the PTXAS cubin.
    # Three of the four probes passed with bit-exact matches.  The
    # fourth probe (F43_shf_iadd64) did NOT confirm its pair
    # because the ld.param version caused PTXAS to pick a different
    # shift opcode (0x899 instead of 0x819).  Each entry below
    # carries its own probe + evidence citation.
    (0x211, 0x212),   # LEA → IADD3X
                      # Evidence: F43_lea_iadd3x probe — clone of
                      # f6_random_s13107_l9 with ld.param arg1=
                      # 0x12345678; PTXAS SASS has 0x211→0x212 at
                      # gap=0, consumer rbar=0x01, runtime output
                      # 3794166319 (0xe226622f) matches PTX-body
                      # simulator.
    (0x211, 0x986),   # LEA → STG.E
                      # Evidence: F43_lea_stg probe — clone of
                      # f6_random_s45059_l14 with ld.param arg1=
                      # 0x12345678; PTXAS SASS has 0x211→0x986 at
                      # gap=0, consumer rbar=0x05 (LDS class, NOT
                      # the producer's ALU class bit), runtime
                      # output 3684127504 (0xdb975310) matches
                      # simulator.
    (0x224, 0x212),   # IMAD.32 → IADD3X
                      # Evidence: F43_imad_iadd3x probe — clone of
                      # f6_random_s49153_l16 with ld.param arg1=
                      # 0x12345678; PTXAS SASS has 0x224→0x212 at
                      # gap=0, consumer rbar=0x01, runtime output
                      # 271863872 (0x10345040) matches simulator.
    (0x212, 0x986),   # IADD3X → STG.E
                      # Evidence: transitively proven by F43_lea_iadd3x
                      # and F43_imad_iadd3x probes.  Both probe cubins
                      # have a SASS chain of ALU producer → IADD3X →
                      # STG.E at gap=0 at every step.  The LEA→IADD3X
                      # and IMAD.32→IADD3X links are already confirmed
                      # above.  If IADD3X→STG.E were broken, the
                      # runtime outputs (0xe226622f and 0x10345040)
                      # would not match their Python-simulated values;
                      # since they do, every link in the chain —
                      # including IADD3X→STG.E — is forwarding-safe.
    # FG-2.5: surfaced by constructive proof engine.
    # IADD.64-UR → IADD3.IMM: IADD.64-UR writes a pair; immediate
    # IADD3 adds a constant to the low half as the second phase of
    # a 64-bit accumulator update.  Observed in Forge reduce_step
    # (passes gpu_correctness) and OURS's multi_block_atomic.
    # IADD.64-UR's ctrl word (wdep=0x3e) broadcasts the result to
    # the dependency network; the subsequent IADD3.IMM's rbar
    # covers the forwarding window.
    (0xc35, 0x810),   # IADD.64-UR → IADD3.IMM
    # P2-5: IMAD family forwarding pairs needed by smem kernels.
    # Evidence: same ALU pipeline class as (0x824, 0x210) which is
    # proven by FG-4.2. All IMAD variants (0x824, 0x825, 0x810, 0x812)
    # share the integer ALU pipeline and forward at gap=0.
    # GPU correctness verified for all affected smem kernels.
    (0x825, 0x824),   # IMAD.WIDE → IMAD.Ri (smem_neighbor address chain)
    (0x825, 0x810),   # IMAD.WIDE → IADD3.IMM (smem address offset)
    (0x824, 0x824),   # IMAD.Ri → IMAD.Ri (chained multiply)
    (0x810, 0x824),   # IADD3.IMM → IMAD.Ri (address→multiply)
    (0x810, 0x812),   # IADD3.IMM → LOP3.IMM (integer→logic)
    (0x812, 0x984),   # LOP3.IMM → LDS (logic→shared load)
    (0x824, 0x984),   # IMAD.Ri → LDS (multiply→shared load)
    (0x810, 0x984),   # IADD3.IMM → LDS (offset→shared load)
    (0x984, 0x210),   # LDS → IADD3 (shared load→ALU)
    (0x984, 0x824),   # LDS → IMAD.Ri (shared load→multiply)
    (0x824, 0x388),   # IMAD.Ri → STS (multiply→shared store)
    (0x824, 0x202),   # IMAD.Ri → MOV (multiply→register move)
    (0x824, 0x802),   # IMAD.Ri → MOV.IMM (multiply→imm move; same forwarding window as 0x202)
    (0xb82, 0x835),   # S2R → IADD.64.IMM (special reg→64-bit add, large gap OK)
    # ------------------------------------------------------------------
    # Phase 2 (edge_87): 32-bit IADD (synthetic key 0x1235; opcode 0x235
    # with byte[9]=0x00) → consumer pairs.  Empirically 0-gap forwarding-
    # safe per Phase 0 study (_harvest/prompts/iadd_probes/REPORT.md).
    # Each pair was probed via byte-patched cubin with the consumer's
    # wdep/stall/rbar stripped — register-file forwarding alone delivered
    # the value correctly to the consumer at 128 threads.  QMMA / HMMA
    # and 64-bit IADD producers are NOT covered (see REPORT.md §"Pairs
    # to retain min_gpr_gap=1").
    (0x1235, 0x210),   # IADD-32 → IADD3 (probed: 2*tid+1 ✓)
    (0x1235, 0x810),   # IADD-32 → IADD3.IMM (analogous to 0x210)
    (0x1235, 0x824),   # IADD-32 → IMAD (probed: 4*tid+2 ✓)
    (0x1235, 0x224),   # IADD-32 → IMAD.32 (same ALU class as 0x824)
    (0x1235, 0x2a4),   # IADD-32 → IMAD.RR (same ALU class)
    (0x1235, 0x819),   # IADD-32 → SHF (probed: (2*tid+1)>>1 ✓)
    (0x1235, 0x223),   # IADD-32 → FFMA (probed: f32 roundtrip ✓)
    (0x1235, 0x823),   # IADD-32 → FFMA.IMM (same path as 0x223)
    (0x1235, 0x245),   # IADD-32 → I2FP/FSEL family (probed)
    (0x1235, 0x20c),   # IADD-32 → ISETP.RR (probed: 1 (R != 0) ✓)
    (0x1235, 0xc0c),   # IADD-32 → ISETP.RU (same scoreboard class)
    (0x1235, 0x208),   # IADD-32 → FSEL (probed: predicate-select ✓)
    (0x1235, 0x808),   # IADD-32 → FSEL.IMM (same class)
    (0x1235, 0x80a),   # IADD-32 → FSEL.STEP (same class)
    (0x1235, 0x309),   # IADD-32 → POPC (probed: popcount(2*tid+1) ✓)
    (0x1235, 0x986),   # IADD-32 → STG.E (probed: data-port forwarding ✓)
    (0x1235, 0xf89),   # IADD-32 → SHFL (mirror of (0x235, 0xf89))
    (0x1235, 0x812),   # IADD-32 → LOP3.IMM (analogous to IADD.64→LOP3.IMM)
    (0x1235, 0x202),   # IADD-32 → MOV (ptxas folded probe; safe by class)
    (0x1235, 0x802),   # IADD-32 → MOV.IMM (same class)
    (0x1235, 0x1235),  # IADD-32 → IADD-32 (self-chain)
    (0x1235, 0x235),   # IADD-32 → IADD.64 (data path forwards by analogy)
    (0x1235, 0x212),   # IADD-32 → IADD3X (carry-extended add; same ALU class)
    # NOT added: (0x1235, 0x27a) → QMMA, (0x1235, 0x23c) → HMMA,
    # (0x1235, 0x237) → IMMA, (0x1235, 0x23f) → DMMA — Phase 0 explicitly
    # carved these out; v2 transcript confirms removing the gap on opcode
    # 0x235 broke test_qmma_e4m3_zero_inputs.  Tensor-core consumers need
    # ≥1 instruction of separation from any IADD producer.
    # ------------------------------------------------------------------
    # Phase 2 reader-side: when 32-bit IADD is the *consumer*, it accepts
    # forwarded values from the same producers that already feed IADD.64
    # at gap=0 (mirror entries with reader_opc=0x235 above).  Both variants
    # share the integer ALU pipeline; the 32-bit consumer's read ports see
    # forwarded values identically to the 64-bit consumer's lo half.
    (0x224, 0x1235),  # IMAD.32 → IADD-32  (mirror of (0x224, 0x235))
    (0x2a4, 0x1235),  # IMAD.RR → IADD-32  (mirror of (0x2a4, 0x235))
    (0x210, 0x1235),  # IADD3   → IADD-32  (mirror of (0x210, 0x235))
    (0x212, 0x1235),  # IADD3X  → IADD-32  (mirror of (0x212, 0x235))
    (0x812, 0x1235),  # LOP3.IMM → IADD-32 (mirror of (0x812, 0x235))
    (0x819, 0x1235),  # SHF     → IADD-32  (mirror of (0x819, 0x235))
    (0x235, 0x1235),  # IADD.64 → IADD-32  (mirror of (0x235, 0x235))
    (0xc11, 0x1235),  # IADD3.R-UR → IADD-32 (mirror of (0xc11, 0x235))
    (0x824, 0x1235),  # IMAD    → IADD-32  (IMAD-family pipeline)
    (0xb82, 0x1235),  # S2R     → IADD-32  (mirror of (0xb82, 0x835))
}


def _is_forwarding_safe_pair(writer_opc: int, reader_opc: int) -> bool:
    """Return True if a 0-gap RAW between the producer and consumer
    is covered by hardware operand forwarding / ctrl-word scoreboard,
    per the FG-2.4 forwarding-safe pair table.

    NOTE: the IADD-32 / IADD.64 split (Phase 2 / edge_87) is keyed via
    `_disc_opcode(raw)` — pass the *discriminating* opcode (0x1235 for
    IADD-32, 0x235 for IADD.64), not the bare 12-bit opcode.
    """
    return (writer_opc, reader_opc) in _FORWARDING_SAFE_PAIRS


# ---------------------------------------------------------------------------
# FG-2.4: LDCU.64 consumer whitelist for verify_schedule
# ---------------------------------------------------------------------------
#
# The LDCU.64 "≥3 instructions before consumer" rule was designed for
# the fp64 preamble path where LDCU.64 is followed by a DMMA or similar
# heavy consumer.  For simple kernels (fmax, cp_async, atom_or,
# atomg_add, multi_block_atomic) the consumer is an IADD.64-UR (0xc35)
# or LDG.E (0x981) that PTXAS freely emits at gap=0/1/2 and which is
# safe because the ctrl word of the consumer (wdep / rbar) handles the
# wait directly.  The gap rule is therefore skipped when the consumer
# is on this whitelist.
_LDCU_GAP_EXEMPT_CONSUMERS: set[int] = {
    0xc35,  # IADD.64-UR: wdep/rbar on the add covers the LDCU latency
    0x981,  # LDG.E.64: LDG's own read barrier covers the addr pair
    0x986,  # TE29: STG.E: rbar on store covers descriptor LDCU latency
    0xc11,  # TE29: IADD3.R-UR: rbar covers LDCU latency (like 0xc35)
}


# ---------------------------------------------------------------------------
# FG-3.1: memory-producer latency proof model
# ---------------------------------------------------------------------------
#
# A memory-producing opcode writes a GPR result that is NOT ready on the
# next cycle; the hardware scoreboard mechanism uses the consumer's ctrl
# word `rbar` field (bits[14:10] of the 23-bit ctrl word) as a bitmask
# telling the fetch unit which scoreboard slot classes to wait on before
# issuing the instruction.
#
# Class → rbar bit:
#   LDC / LDCU / ULDC / MOV cbuf / IMAD.MOV cbuf  → bit 1 (0x02)
#   LDS / LDSM                                    → bit 2 (0x04)
#   LDG / LDG.E / ATOMG / TEX / TLD / TLD4 /
#   TXQ / SULD                                    → bit 3 (0x08)
#
# Scoreboard model:
#   For a memory writer at i with GPR dest set D:
#     1. Shadow-walk forward until first reader j of D (or until D is
#        fully shadowed by intervening writes).
#     2. Scan instructions in [i+1, j] inclusive and check rbar.  If
#        ANY instruction in that window has the writer's class bit set
#        in its rbar, the scoreboard has proven the wait — edge is
#        MEMORY_SCOREBOARD_SAFE.
#     3. Otherwise the edge is MEMORY_VIOLATION — gap=0/1/… reads of
#        a memory-loaded register with no rbar wait evidence is a
#        real latency hazard.
#
# LDCU.64 (UR dest) and ULDC.64 (UR dest) already have their own R1
# rule in verify_proof; they are excluded from the GPR-dest memory rule.
# S2R / S2UR are hardware special-register reads with zero latency and
# are not considered memory producers.

# Memory-producing opcodes with GPR dest (excluding LDCU/ULDC/S2UR
# which write the UR bank and are handled by rules R1 / R10).
_OPCODES_MEMORY_GPR: set[int] = (
    _OPCODES_LDG
    | _OPCODES_ATOMG
    | _OPCODES_LDS
    | {0xb82,          # LDC (SM_120 GPR dest)
       0x624, 0xa02}   # SM_89 IMAD.MOV cbuf, MOV cbuf
)


# FG-3.2: UR-destination memory / system producers.
# These opcodes write the UR register bank (not GPR).  Consumers read
# the UR via byte 4 on UR-consuming opcodes.  LDCU.64 is handled by
# the older LDCU-specific rule R1; all others go through R10.
_OPCODES_MEMORY_UR: set[int] = {
    0x7ac,   # LDCU / LDCU.64 — handled by R1 (.64 variant) and R10 (.32)
    0x9c3,   # S2UR (special register → UR)
    0x3c4,   # REDUX.SUM (warp reduction → UR)
    0xab9,   # ULDC.64 (SM_89 uniform const load)
}


# UR-consumer opcodes: opcodes whose byte 4 (b4) is a UR index
# reference.  Used to find UR consumers during rule R1 / R10.
_UR_CONSUMER_OPCODES: set[int] = {
    0xc35,   # IADD.64-UR
    0xc0c,   # ISETP.R-UR
    0xc24,   # IMAD.R-UR
    0xc25,   # IMAD.WIDE.R-UR
    0xc11,   # LEA.R-UR
    0xc12,   # IADD3X.R-UR
    0xc0b,   # FSETP.R-UR
    0xc02,   # MOV R, UR
    0x981,   # LDG.E (UR descriptor)
    0x984,   # LDS (UR base)
    0xb82,   # LDC (UR base)
    0x7ac,   # LDCU (UR base)
    0x986,   # STG.E (UR descriptor at b8, not b4 — see _get_ur_src)
}


# Module-level version of the scheduler's _WDEP_TO_RBAR table.
# Maps producer wdep slot → required consumer rbar bit mask (the value
# the scheduler sets when a consumer reads a register produced in that
# slot).  Used by the FG-3.1/3.2 memory proof model; kept in sync
# with the duplicate inside `assign_ctrl` below.
#
#   0x31 → 0x03 : LDC/LDCU slot         (class bit 0x02)
#   0x33 → 0x05 : LDS/LDCU.32 slot      (class bit 0x04)
#   0x35 → 0x09 : LDG/ATOMG slot        (class bit 0x08)
#   0x3b → 0x09 : LDG rotating variant  (class bit 0x08) — FG-3.2
#                   Evidence: reduce_sum LDG [15] emits wdep=0x3b and
#                   the following BSYNC at [16] has rbar=0x09
#                   (LDG-class wait bit 3).  Same class as 0x35.
#   0x3e → 0x03 : ALU slot              (class bit 0x02)
#
# `class_bit = required_rbar & ~0x01` — rbar bit 0 (0x01) is the
# "always-present" base bit and does not encode a class.
_WDEP_TO_RBAR_MASK: dict[int, int] = {
    0x31: 0x03,
    0x33: 0x05,
    0x35: 0x09,
    0x3b: 0x09,  # FG-3.2: LDG rotating variant (reduce_sum)
    0x3e: 0x03,
}


# FG-3.2 + FG-3.3: explicitly-declared no-track wdep slots.
# A producer with wdep ∈ _LATENCY_INERT_WDEPS is "not scoreboard-
# tracked" — the hardware does not track its result via an rbar
# slot.  A consumer therefore CANNOT wait on the producer via rbar;
# it must rely on either instruction-stream gap distance or a
# post-segment-boundary scoreboard reset.
#
#   0x3f — explicit no-tracking (descriptor loads, cp_async LDS,
#          short-latency const fetches).
#   0x37 — reserved LDCU/LDC slot with NO rbar bit (header comment
#          line 27: "never use wdep=0x37").  ptxas emits this on a
#          post-segment-boundary LDCU.64 (observed in reduce_sum);
#          the hardware clears the scoreboard at the boundary, so
#          the consumer does not need to wait at all.
_LATENCY_INERT_WDEPS: set[int] = {0x37, 0x3f}


# FG-3.3: LDCU.64 latency in instruction-stream slots.
# This is the ptxas convention: a gap of >= 3 instructions between
# an LDCU.64 producer and its UR consumer is sufficient to cover the
# LDCU load latency even when the consumer's rbar does not carry the
# producer's class bit.
#
# Evidence in the corpus (4 edges):
#   hmma_zero   [1]→[5]  gap=3  LDCU.64 wdep=0x35 → STG, no rbar 0x08
#   imma_zero   [1]→[5]  gap=3  (same pattern)
#   dmma_zero   [1]→[5]  gap=3  (same pattern)
#   fg114b_diag3[2]→[15] gap=12 LDCU.64 wdep=0x35 → STG, no rbar 0x08
#
# The original FG-2.4 R1 rule also used gap >= 3 as a safety class.
# FG-3.3 preserves that empirical convention as a narrow, data-driven
# rule folded into the unified R10 path instead of a legacy special
# case.
_LDCU_GAP_SAFE_MIN: int = 3


def _get_rbar(raw: bytes) -> int:
    """Extract the 5-bit rbar field (ctrl bits [14:10]) from a raw
    128-bit SASS instruction.  ctrl is packed into bytes 13–15 via
    `_patch_ctrl`: raw24 = ctrl << 1; byte[13..15] = raw24 little-endian.
    """
    raw24 = raw[13] | (raw[14] << 8) | ((raw[15] & 0x7f) << 16)
    ctrl = raw24 >> 1
    return (ctrl >> 10) & 0x1f


def _get_wdep(raw: bytes) -> int:
    """Extract the 6-bit wdep field (ctrl bits [9:4]) from a raw
    128-bit SASS instruction.  Used by the FG-3.1 memory proof model
    to determine which scoreboard slot the producer is tracked in.
    """
    raw24 = raw[13] | (raw[14] << 8) | ((raw[15] & 0x7f) << 16)
    ctrl = raw24 >> 1
    return (ctrl >> 4) & 0x3f


def _is_memory_gpr_producer(opc: int) -> bool:
    """Return True if `opc` is a memory-loading opcode with a GPR dest
    (i.e. eligible for the FG-3.1 scoreboard proof model).
    """
    return opc in _OPCODES_MEMORY_GPR


# ---------------------------------------------------------------------------
# PERF-2: operand-role-aware forwarding
# ---------------------------------------------------------------------------
#
# The flat _FORWARDING_SAFE_PAIRS set proves that a (writer, reader)
# opcode pair is safe at gap=0 when the overlap feeds the reader's
# DATA operand.  However, the SAME opcode pair is NOT safe when the
# overlap feeds the reader's ADDRESS operand — the address pipeline
# has a different timing requirement from the data pipeline.
#
# Evidence: PERF-1 found that removing the NOP between IADD.64-UR
# and STG.E is safe when STG's DATA register (b4) overlaps the
# writer's dest, but BREAKS when STG's ADDRESS pair (b3) overlaps
# (test_4ptr_add: produced stale-address store, got 1.0 instead of
# 111.0).
#
# The operand-role check extracts which bytes of the reader
# instruction carry the overlapping register index, then classifies
# the overlap as DATA (b4 for STG/ATOMG, b3 for ALU consumers) or
# ADDRESS (b3 for STG/LDG/ATOMG when the reader is a memory op).
#
# A forwarding-safe pair is only safe to elide the NOP if EVERY
# overlapping register lands in a DATA role — never in an ADDRESS
# role.

# Opcodes where b3 is a memory ADDRESS pair (not a generic ALU src).
_MEMORY_ADDR_OPCODES: set[int] = (
    _OPCODES_STG          # STG: b3 = store address pair
    | _OPCODES_LDG        # LDG: b3 = load address pair
    | _OPCODES_ATOMG      # ATOMG: b3 = atomic address pair
    | _OPCODES_LDS        # LDS: b3 = shared-memory address (when GPR-addressed)
    | _OPCODES_STS        # STS: b3 = shared-memory address
)


def _overlap_is_data_only(reader_raw: bytes, overlap_regs: set[int]) -> bool:
    """PERF-2: return True iff EVERY overlapping register feeds only
    a DATA operand of the reader, never an ADDRESS operand.

    For memory instructions (STG, LDG, ATOMG, LDS, STS):
      b3 = address pair base (b3, b3+1)  → ADDRESS role
      b4 = data register                 → DATA role
      b8 = UR descriptor (not a GPR)     → ignored

    For ALU instructions:
      all operand bytes are ALU sources — DATA role by definition.

    Conservative: if the reader opcode is unrecognized, return False.
    """
    opc = _get_opcode(reader_raw)
    if opc not in _MEMORY_ADDR_OPCODES:
        # ALU consumer — all GPR sources are data. Safe.
        return True

    # Memory consumer: check if any overlap reg is in the address pair.
    addr_base = reader_raw[3]
    if addr_base >= 0xff:
        # RZ as address — no real address dependency.
        return True
    addr_regs = {addr_base, addr_base + 1}
    if overlap_regs & addr_regs:
        # At least one overlapping register is in the address pair.
        return False

    # Overlap is in data (b4) or src2 (b8) or something else — DATA role.
    return True


def _is_zero_init_fastpath(raw: bytes) -> bool:
    """FG-4.1: semantic predicate for the HFMA2 zero-init trick.

    ptxas emits `HFMA2 Rd, -RZ, imm_fp16x2, RZ` as a register-zero
    init primitive: the computation is (-0.0) * imm + 0.0 = 0.0 for
    any finite FP16x2 immediate, so the result is deterministically
    zero and the hardware can forward it to a consumer on the next
    cycle regardless of the normal 1-instruction ALU GPR latency.

    Encoding ground truth (sass/encoding/sm_120_opcodes.py line 3109):
        HFMA2 R2, -RZ, RZ, 0, 0
        b0=0x31, b1=0x74     → opcode 0x431
        b2 = dest
        b3 = 0xff            → src0 = -RZ (negated zero-latency src)
        b4 = 0               → FP16x2 immediate (0.0 in encoder,
                               but ptxas observed to emit non-zero
                               values like 0x2a, 0x07; irrelevant
                               because src0 = -RZ makes the product 0)
        b8 = 0xff            → src2 = RZ (zero accumulator)

    The predicate is intentionally narrow: BOTH b3 and b8 must be
    0xff.  A non-RZ source on either side is a real FMA and retains
    the standard 1-instruction latency.  The b4 immediate does not
    constrain the predicate because src0 = RZ forces the product
    regardless of its value.

    This predicate is the only exemption FG-4.1 adds to the ALU
    latency rule; see sass/schedule.py verify_proof rule R8 for the
    consumer.
    """
    if len(raw) < 16:
        return False
    if _get_opcode(raw) != 0x431:
        return False
    return raw[3] == 0xff and raw[8] == 0xff


def _is_memory_ur_producer(opc: int) -> bool:
    """Return True if `opc` writes the UR register bank and is
    eligible for the FG-3.2 UR memory proof model (rule R10).
    LDCU.64 still routes through the older rule R1.
    """
    return opc in _OPCODES_MEMORY_UR


def _get_ur_dest(raw: bytes) -> int:
    """Extract the UR destination register index for a UR-writing
    opcode (LDCU / ULDC / S2UR / REDUX).  Returns -1 if not a
    UR producer.  The UR dest is at byte 2 for all modeled opcodes.
    """
    opc = _get_opcode(raw)
    if opc not in _OPCODES_MEMORY_UR:
        return -1
    return raw[2]


def _get_ur_src(raw: bytes) -> int:
    """Extract the UR source register index for a UR-consuming opcode.
    Most UR-consuming opcodes place the UR index at byte 4; STG.E
    (0x986) uses byte 8 for its UR descriptor.  Returns -1 if the
    opcode does not consume UR at a known byte position.
    """
    opc = _get_opcode(raw)
    if opc not in _UR_CONSUMER_OPCODES:
        return -1
    if opc == 0x986:  # STG.E: UR descriptor at b8
        return raw[8]
    return raw[4]


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
    elif opcode == 0x235:
        # Phase 2 (edge_87): 32-bit IADD (b9=0x00) writes a single GPR;
        # IADD.64 (b9=0x02) writes a 64-bit pair.  Differentiate via
        # byte[9] so the scheduler doesn't claim a phantom dest+1 write
        # for 32-bit IADD (which would introduce false WAW conflicts).
        if dest < 255:
            if raw[9] == 0x00:
                regs.add(dest)
            else:
                regs |= {dest, dest+1}
    elif opcode == 0xc35:  # IADD.64-UR: writes GPR pair
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
    elif opcode == 0x806:  # VOTE.BALLOT: writes b2 as single GPR
        # Previously VOTE was absent from dest tracking because it was
        # removed from _OPCODES_ALU when we thought wdep=0x3f (no
        # scoreboard).  With wdep=0x3e (ALU) the consumer needs to find
        # VOTE's dest in pending_writes to pick up the implicit-ALU
        # rbar contribution.  Observed on b24a5fa6: without this tracking
        # the downstream IADD3 rbar missed bit 1 and read R7 stale.
        if dest < 255: regs.add(dest)
    return regs


_ldcu_slot_counter = [0]  # mutable counter for rotating LDCU wdep slots
_ldc_slot_counter = [0]   # mutable counter for rotating LDC wdep slots

def _wdep_for_opcode(opcode: int, raw: bytes = None) -> int:
    """Assign the scoreboard write-dependency slot for an opcode."""
    if opcode == 0x7ac:  # LDCU
        if raw is not None and raw[9] == 0x0a:  # LDCU.64
            # WB-wdep-audit (2026-04-28, refined): ptxas rotates
            # LDCU.64 wdep through [0x31, 0x33].  The first LDCU.64
            # gets 0x31, the second 0x33, third 0x31, etc.  Verified
            # by per-kernel kdiff evidence across the 100-kernel
            # majority (conv2d/tensor exceptions handled separately).
            # Earlier flat-0x31 rule produced systemic STG.E rbar
            # mismatches because consumers waiting for the
            # 2nd-LDCU-loaded UR were waiting on the wrong slot.
            slots = [0x31, 0x33]
            slot = slots[_ldcu_slot_counter[0] % 2]
            _ldcu_slot_counter[0] += 1
            return slot
        # LDCU.32: WB-wdep-audit (2026-04-28): aligned with ptxas, which
        # uses 0x31 flat for LDCU.32 across the corpus.  Earlier rotation
        # (0x31/0x33) didn't match ptxas evidence and produced 13 audit
        # discrepancies.  Flat 0x31 closes them.
        # NB: do NOT increment _ldcu_slot_counter — that counter tracks
        # LDCU.64 rotation only.  Mixing the two would shift LDCU.64
        # phase based on LDCU.32 presence.
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
    if opcode in (0x589, 0xf89, 0x989):  # SHFL (warp shuffle)
        # ptxas-verified (sm_120): SHFL posts to slot 0x31 (same as LDC/LDCU).
        # SHFL has variable latency; consumers wait via rbar=0x03 for this slot.
        # Treating SHFL as wdep=0x3e (ALU in-order) was incorrect — consumers
        # read stale data because SHFL actually takes many cycles.
        return 0x31
    if opcode == 0x309:  # POPC — long-latency on SM_120; ptxas posts to LDC
        # slot 0x31, NOT ALU slot 0x3e.  Consumer rbar must include bit 1
        # (via _WDEP_TO_RBAR[0x31]=0x03) or the scoreboard treats POPC as
        # in-order ALU and consumers read pre-retirement partial results —
        # low 6 bits correct, upper bits still holding the input register's
        # prior content.  Observed: ours r16 = popc(r3) | (r3 & 0xF00) |
        # (r3 & 0x80000000) for input-dependent stale-bit patterns.
        return 0x31
    if opcode == 0x300:  # FLO/CLZ — long-latency, same LDC-class as POPC.
        # ptxas byte-diff confirms wdep=0x31 (2026-04-20 clz_bytediff.py).
        return 0x31
    if opcode == 0x806:  # VOTE
        # Byte-diff against ptxas sm_120 on b24a5fa6 (vote + IADD3 consumer):
        # ptxas actually emits wdep=0x3e (ALU) so downstream ALU consumers
        # can wait on the vote's result via implicit ALU ordering.
        # The older "0x3f / no tracking" assumption produced scoreboard-less
        # vote output; consumers read stale R<dest> before the warp-sync
        # posted the ballot mask.  Changing to 0x3e matches ptxas for this
        # kernel; the earlier ISETP+predicated-EXIT concern noted as a
        # reason for 0x3f did not reproduce with 0x3e on SM_120.
        return 0x3e
    if opcode in _OPCODES_LDGSTS:
        return 0x3f  # LDGSTS: async copy writes to shared mem, not GPR — no scoreboard slot
    if opcode in _OPCODES_LDGDEPBAR:
        return 0x31  # LDGDEPBAR: commit group, posts to LDC slot (ptxas-verified)
    # FFMA/FFMA.IMM: standard ALU wdep=0x3e (ptxas uses same as other ALU ops)
    # Earlier analysis incorrectly decoded ctrl (forgot <<1 shift in _ctrl_to_bytes)
    if opcode in _OPCODES_IADD64_UR:
        return 0x3e  # ALU slot — consumer LDG/STG gets rbar via pending_writes
    # WB-wdep-audit (2026-04-28): ptxas leaves these ops untracked
    # (wdep=0x3f).  Each closes a wdep-audit bucket once aligned:
    #   0x812 LOP3.IMM     — 17 occurrences
    #   0xc11 IADD3.UR     — 12 occurrences (postamble carry-chain
    #                        addr-pair; STG consumer waits via natural
    #                        ALU latency, not scoreboard)
    if opcode in (0x812, 0xc11):
        return 0x3f
    # Tensor-core MMA family: dedicated long-latency slot 0x32.
    # Consumers wait via _WDEP_TO_RBAR[0x32]=0x11 (bit 4).  Surfaced by
    # mower hmma probe — was returning ALU slot 0x3e via the fallback
    # below, which gave too-short a latency window for the consumer to
    # see the tensor write retire.
    if opcode in (0x23c, 0x237, 0x23f, 0x27a):
        return 0x32
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
    0xb1d: 6,   # BAR.SYNC: misc=6 (ptxas-verified 2026-04-08 from bar_probe)
    0x941: 5,   # BSYNC: misc=5 (ptxas-verified 2026-04-09 from shared_copy)
    0x221: 1,   # FADD: misc=1 (ptxas-verified 2026-04-08 bar_probe; counter-misc causes ERR715)
    0x388: 1,   # STS (store shared): misc=1 (ptxas-verified 2026-04-08)
    0x984: 2,   # LDS (load shared): misc=2 (ptxas-verified 2026-04-08)
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
    0xc11: 5,   # TE21: IADD3.R-UR carry chain (fixed misc, not in _OPCODE_META)
    0xc0c: 0,   # ISETP R-UR: misc=0 default (overridden for VOTE-adjacent context)
    0x20c: 0,   # ISETP R-R: misc=0 default (overridden for VOTE-adjacent context)
    0x80c: 0,   # ISETP IMM: misc=0 default (overridden for VOTE-adjacent context)
    0x20b: 5,   # FSETP R-R: misc=5 (ptxas-verified from forced_match kernel)
    0xc0b: 5,   # FSETP R-UR: misc=5 (ptxas-verified, decoded with <<1 shift)
    0x80a: 5,   # FSEL.step: misc=5 (ptxas-verified)
    0x223: 4,   # FFMA R-R-R: misc=4 (ptxas-verified for FMA chains on SM_120)
    0x823: 4,   # FFMA.IMM: misc=4 (same as FFMA R-R-R, ptxas ground truth after <<1 correction)
    0x806: 5,   # VOTE: misc=5 (ptxas-verified on b24a5fa6 vote+IADD3 kernel)
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
    # SHFL: ptxas uses misc=2 (ground truth SM_120)
    0x589: 2,   # SHFL R-R
    0xf89: 2,   # SHFL R-imm
    0x989: 2,   # SHFL imm-imm
    # FLO/CLZ (0x300): wdep=0x31 → opex=0x10|misc.  Counter-based misc can hit
    # 0xd/0xe/0xf → opex 0x1d/0x1e/0x1f which are undefined for this decoder
    # class (same rule as LDC).  ptxas byte-diff (2026-04-20 clz_bytediff + clz_prmt
    # minimal) shows misc=2 is constant across kernel positions.
    0x300: 2,
    0x816: 6,  # PRMT imm-selector (ptxas-observed form 2026-04-20): misc=6
    0x416: 6,  # PRMT legacy imm-selector form: same misc
    0x216: 6,  # PRMT reg-selector:  matched for consistency (probe pending)
}

# All opcodes recognised by assign_ctrl.  Unknown opcodes raise ValueError.
_OPCODES_VOTE = {0x806}  # VOTE.BALLOT — warp-level, wdep=0x3F (no ALU tracking)

_ALL_KNOWN_OPCODES: frozenset = frozenset(
    _OPCODES_LDG | _OPCODES_LDC | _OPCODES_LDS |
    _OPCODES_STG | _OPCODES_STS | _OPCODES_BAR |
    _OPCODES_CTRL | _OPCODES_ALU | _OPCODES_IADD64_UR |
    _OPCODES_SMEM_SETUP | _OPCODES_ATOMG | _OPCODES_DFPU |
    _OPCODES_F2F | _OPCODES_LDGSTS | _OPCODES_LDGDEPBAR |
    _OPCODES_VOTE
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
        0x32: 0x11,   # HMMA/IMMA/DMMA/QMMA tensor-core slot → rbar=0x11 (bit 4 + gate)
        0x33: 0x05,   # LDS/LDCU.32 slot → rbar=0x05
        0x35: 0x09,   # LDG slot → rbar=0x09 (all LDGs share this slot, ptxas-verified)
        0x3b: 0x09,   # FG-3.2: LDG rotating variant, same class as 0x35
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
        if opcode in (0xc35, 0xc0c, 0xc24, 0xc0b, 0xc11):  # IADD.64-UR, ISETP R-UR, IMAD R-UR, FSETP R-UR, IADD3.R-UR
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
        # TE29: STG UR descriptor at b8 — needs rbar for LDCU that loaded it
        if opcode in _OPCODES_STG:
            ur_desc_stg = si.raw[8]
            if ur_desc_stg in pending_ur_writes:
                _, pw = pending_ur_writes[ur_desc_stg]
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

        # SEL/FSEL read a predicate as operand, not as a guard.
        # Must also wait for the ISETP that wrote that predicate.
        # SEL R-R (0x207): pred at b8. SEL.IMM (0x807): pred at b11 & 0x07.
        # FSEL (0x208): pred at b8. FSEL.IMM (0x808): pred at b11 & 0x07.
        if opcode in (0x207, 0x208):
            sel_pred = si.raw[8] & 0x07
            if sel_pred in pending_pred_writes:
                _, pw = pending_pred_writes[sel_pred]
                candidate = _WDEP_TO_RBAR.get(pw, 0x01)
                rbar = rbar | candidate
        elif opcode in (0x807, 0x808):
            sel_pred = si.raw[11] & 0x07
            if sel_pred in pending_pred_writes:
                _, pw = pending_pred_writes[sel_pred]
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
        # Misc nibble: counter-based default with per-opcode overrides.
        # Most ALU instructions use misc = (counter & 0xF). Specific opcodes
        # have fixed overrides in _OPCODE_MISC (BAR=6, STS=1, LDS=2, etc.).
        misc = _OPCODE_MISC.get(opcode, misc_counter & 0xF)
        # PTXAS-R23A.1 (FB-1 Phase A / Family A completion fix): the 5-bit
        # `opex` field that the SM_120 instruction decoder reads for LDC
        # is (wdep[0] << 4) | misc.  LDC's wdep is 0x31 (bit 0 set), so
        # ctrl bit 4 is 1 and opex = 0x10 | misc.  `nvdisasm` and the
        # runtime decoder both reject opex == 0x1d (i.e. misc == 0xd) for
        # LDC with `Opclass 'ldc__RaRZ', undefined value 0x1d for table
        # TABLES_opex_0`; empirically, opex 0x1d/0x1e/0x1f (misc
        # 0xd/0xe/0xf, 13/14/15) are all undefined for this LDC class.
        # ptxas's observed misc range for LDC (per the counter-model
        # comment below) is 0..10 — a strict subset of the decoder's
        # valid range.  Clamp misc to 0..7 for LDC whenever the counter
        # would produce an invalid opex; this keeps the counter pattern
        # for kernels where the counter was already < 8 (no change) and
        # re-maps the overflow range 13-15 into 5-7 (valid, matches
        # ptxas's observed range).  Only LDC is affected — LDCU, S2R,
        # S2UR, ALU, etc. are untouched.
        if opcode == 0xb82 and misc >= 0xb:
            # ptxas's observed misc range for LDC is 0..10 (0xa).  Empirically,
            # misc 0xc also produces an undefined opex at runtime (opex 0x1c
            # causes LAUNCH_FAILED/719) — confirmed by clz_prmt/bfx minimal
            # 0e6dcc8b on HEAD eea3523 (2026-04-20 patch_ldc_misc.py).  The
            # earlier >= 0xd threshold missed 0xb/0xc.  Remap >= 0xb into 0..7
            # via `& 0x7`, which stays within ptxas's observed safe range.
            misc = misc & 0x7
        # ISETP misc: context-sensitive. Default misc=0 (from _OPCODE_MISC).
        # When within 3 instructions of VOTE (0x806), use counter-based misc
        # instead. ptxas uses counter-based misc for ISETP near VOTE.
        # SM_120 rule #24: ISETP misc=0 near VOTE causes ERR715.
        if opcode in (0x20c, 0xc0c, 0x80c):
            # Check if VOTE is within +/- 3 instructions
            vote_nearby = False
            for k in range(max(0, i-3), min(len(instrs), i+4)):
                if k != i:
                    k_op = (instrs[k].raw[0] | (instrs[k].raw[1] << 8)) & 0xFFF
                    if k_op == 0x806:  # VOTE
                        vote_nearby = True
                        break
            if vote_nearby:
                misc = misc_counter & 0xF  # use counter, not override
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
        elif opcode in (0x20c, 0x80c):  # ISETP R-R/IMM: pred_dest at (raw[10]>>1) & 0x7
            pred_dest = (si.raw[10] >> 1) & 0x7
            pending_pred_writes[pred_dest] = (i, wdep)
        elif opcode in (0x20b, 0xc0b):  # FSETP/FSETP-UR: pred_dest at raw[9] & 0x7
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

        # SM_120 ISETP.UR b9 vote-path override:
        # When ISETP.UR feeds VOTE and source register is in the %4==2 group
        # (R2, R6, R10, R14), ptxas uses b9=0x42 instead of 0x60.
        # This is a narrow surgical fix — only applies to the vote window.
        if opcode == 0xc0c:
            vote_feeds = False
            for k in range(i+1, min(len(instrs), i+4)):
                k_op = (instrs[k].raw[0] | (instrs[k].raw[1] << 8)) & 0xFFF
                if k_op == 0x806:  # VOTE
                    vote_feeds = True
                    break
                if k_op != 0x918:  # non-NOP between ISETP and VOTE
                    break
            if vote_feeds:
                src_reg = patched[3]
                if src_reg % 4 == 2:  # R2, R6, R10, R14
                    patched = bytearray(patched)
                    patched[9] = 0x42
                    patched = bytes(patched)

        result.append(SassInstr(patched, si.comment))

    return result
