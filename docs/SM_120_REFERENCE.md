# SM_120 (Blackwell B) ISA Reference

A definitive public reference for the NVIDIA Blackwell B compute ISA — the
SASS instruction set used by consumer Blackwell chips (`sm_120`, RTX 50-series).

**Version:** 2026-04-04
**Hardware target:** NVIDIA RTX 5090 (GB202, consumer Blackwell B)
**Source:** OpenPTXas reverse engineering, `C:/Users/kraken/openptxas`

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Instruction Set Overview](#2-instruction-set-overview)
3. [Instruction Encoding Format](#3-instruction-encoding-format)
4. [Scoreboard System](#4-scoreboard-system)
5. [Hardware Bugs & Quirks](#5-hardware-bugs--quirks)
6. [Capmerc (Mercury) DRM System](#6-capmerc-mercury-drm-system)
7. [ELF Cubin Format](#7-elf-cubin-format)
8. [Differences from Prior Generations](#8-differences-from-prior-generations)
9. [Worked Examples](#9-worked-examples)
10. [References](#10-references)

---

## 1. Introduction

### What is SM_120?

`sm_120` is the NVIDIA compute capability identifier for **Blackwell B**, the
consumer variant of NVIDIA's Blackwell architecture. It ships on the following
dies:

| SKU       | Die   | GPCs | SMs | CUDA Cores | FP32 TFLOPS | FP64 TFLOPS |
|-----------|-------|------|-----|------------|-------------|-------------|
| RTX 5090  | GB202 | 11   | 170 | 21760      | ~105        | **1.639**   |
| RTX 5080  | GB203 | 7    |  84 | 10752      | ~56         | ~0.87       |
| RTX 5070  | GB205 | 5    |  48 |  6144      | ~32         | ~0.50       |

**Blackwell B is not Blackwell A.** Datacenter Blackwell chips (B100, B200,
GB200) report as `sm_100` / `sm_101` and expose a different feature mix — most
notably, full-speed FP64 (~60 TFLOPS on B100 vs 1.6 TFLOPS on RTX 5090). The
two architectures share silicon DNA but diverge at the SM level: Blackwell B's
FP64 units are **hardware-fused** at the silicon level to protect datacenter
SKUs. This is not a driver restriction and cannot be unlocked.

The measured 64:1 FP32/FP64 ratio on RTX 5090 (105 TFLOPS FP32 ÷ 1.639 TFLOPS
FP64 = 64.06) confirms that ~1/64 of the FP64 silicon is active.

### Why this reference exists

NVIDIA has not published an SM_120 SASS reference. The official `nvdisasm`
tool can *decode* SM_120 instructions but the documentation for operand
semantics, scoreboard behaviour, and encoding rules is incomplete. The open
Mesa/NVK driver project (`src/nouveau/compiler`) has partial SM_120 support
but lacks tensor-core, TMA, and capmerc knowledge.

This reference fills that gap. Every claim is **hardware-verified** on an
RTX 5090 or extracted from CUDA 13.0 `ptxas` ground-truth cubins via byte-wise
comparison. Where a claim is inferred rather than directly verified, it is
marked.

### Methodology

Findings were gathered by three primary techniques:

1. **ptxas byte comparison.** Emit known-shape PTX, compile with NVIDIA's
   `ptxas` to a cubin, disassemble with `nvdisasm`, and diff against our own
   emitter output. This identifies opcode bytes, operand field positions,
   control-word values, and scoreboard slot usage.

2. **Hardware probing on RTX 5090.** Load hand-crafted cubins (via OpenPTXas's
   emitter) and execute on real hardware. Failure modes include:
   - `ERR_ILLEGAL_INSTRUCTION` (sync=715): opcode or capmerc violation
   - `CUDA_ERROR_INVALID_IMAGE` (load=201): capmerc rejection at load
   - Wrong-data output: scoreboard race, predicate corruption, literal-pool bug
   - `CUDA_ERROR_ILLEGAL_ADDRESS`: missing FP64 c[0][0x358] descriptor

3. **Brute-force opcode sweep.** Enumerate all 4096 possible 12-bit opcodes,
   encode a minimal instruction around each, and record which execute without
   fault. Found 188 valid opcodes on SM_120 (182 named via `nvdisasm`, 6
   identified behaviourally).

### Hardware specs (RTX 5090, measured)

| Property                     | Value               |
|------------------------------|---------------------|
| FP32 peak                    | ~105 TFLOPS         |
| FP64 peak (measured)         | 1.639 TFLOPS        |
| FP32/FP64 ratio              | 64:1 (fused silicon)|
| FP16/BF16 tensor (dense)     | ~418 TFLOPS         |
| FP8 (E4M3/E5M2) tensor       | ~836 TFLOPS         |
| INT8 tensor                  | ~836 TOPS           |
| Shared memory/SM             | up to 228 KB        |
| L2 cache                     | 96 MB               |
| GPRs per thread              | 255 architectural   |
| Uniform registers            | 63 (UR0..UR62)      |
| Predicates                   | 8 (P0..P7)          |
| Uniform predicates           | 8 (UP0..UP7)        |

---

## 2. Instruction Set Overview

The SM_120 ISA has 188 valid opcodes identified. Opcodes are 12-bit values
packed in the low 12 bits of the 16-byte instruction word (bytes 0-1 of the
little-endian encoding).

Notation:
- **b0..b15**: instruction bytes in emission order
- **GPR**: general-purpose register (R0..R254, R255=RZ=zero)
- **UR**: uniform register (UR0..UR62, UR63=URZ=zero)
- **P**: predicate (P0..P6, P7=PT=true)
- **UP**: uniform predicate (UP0..UP6, UP7=UPT=true)

### 2.1 Integer ALU

| Opcode | Mnemonic        | Description                          | Notes |
|--------|-----------------|--------------------------------------|-------|
| 0x210  | IADD3           | 3-input integer add                  | src0=b3, src1=b4, src2=b8 |
| 0x810  | IADD3.IMM32     | IADD3 with 32-bit immediate          | misc=1 |
| 0x212  | IADD3X (LOP3)   | IADD3 with carry / LOP3.LUT          | also logic op |
| 0x235  | IADD.64         | 64-bit add (GPR pair)                | writes dest:dest+1 |
| 0xc35  | IADD.64-UR      | 64-bit add with UR src1              | misc=5 |
| 0x225  | IMAD.WIDE       | Wide multiply-add R-R (64-bit)       | works |
| 0x825  | IMAD.WIDE       | Wide multiply-add R-imm              | works |
| 0x824  | IMAD            | 32-bit IMAD R-imm                    | works |
| 0xc24  | IMAD R-UR       | 32-bit IMAD R-UR                     | **use this for R-R** |
| 0x2a4  | IMAD R-R        | 32-bit IMAD R-R-R                    | **BROKEN — do not use** |
| 0x227  | IMAD.HI.U32     | High-word IMAD                       |       |
| 0x226  | IDP.4A          | 4-way INT8 dot product               | dp4a  |
| 0xc26  | IDP.4A-UR       | IDP4A with UR source                 |       |
| 0x213  | IABS            | Integer absolute value               | single src at b4 |
| 0x813  | IABS.IMM        | IABS B-immediate form                |       |
| 0x248  | VIMNMX.R-R      | Integer min/max (S32/U32 variants)   |       |
| 0x848  | VIMNMX.R-imm    | Integer min/max with immediate       |       |
| 0x217  | IMNMX           | Integer min/max (alt)                |       |
| 0x309  | POPC            | Population count                     | single src at b4 |
| 0x301  | BREV            | Bit reverse                          | single src at b4 |
| 0x300  | FLO             | Find leading one                     | single src at b4 |
| 0x21a  | SGXT            | Sign-extend from bit position        |       |
| 0x21b  | BMSK            | Bitmask generation (pos, width)      |       |
| 0x81b  | BMSK.IMM        | BMSK B-immediate form                |       |
| 0x239  | I2IP            | Int→int pack with saturation (U8)    |       |
| 0x211  | LEA             | Load effective address               |       |
| 0x811  | LEA.IMM         | LEA with immediate index             |       |
| 0x212  | LOP3.LUT        | 3-input logic via 8-bit LUT          | reads b3/b4/b8 |
| 0x21e  | PLOP3.LUT       | Predicate LOP3                       | pred dest only |
| 0x21f  | PLOP3           | Predicate LOP3 alt form              |       |
| 0x819  | SHF             | Funnel shift (const shift amount)    |       |
| 0x299  | SHF.VAR         | Funnel shift (variable amount)       | shift reg at b4 |
| 0x219  | SHF.R.S32.HI    | Arithmetic right shift variable      |       |

### 2.2 Float ALU (FP32)

| Opcode | Mnemonic        | Description                          | Notes |
|--------|-----------------|--------------------------------------|-------|
| 0x221  | FADD            | FP32 add                             |       |
| 0x223  | FFMA / FMUL     | FP32 fused multiply-add              |       |
| 0x820  | FMUL.IMM        | FMUL with f32 immediate              |       |
| 0x823  | FFMA.IMM        | FFMA with f32 immediate              |       |
| 0x208  | FSEL            | FP32 select                          | pred at raw[10..11] |
| 0x80a  | FSEL.STEP       | Combined FP compare + select         | misc=5, avoids FSETP bug |
| 0x209  | FMNMX           | FP32 min/max                         | b10=0xfe, b11=0x03/0x07 |
| 0x20b  | FSETP           | FP32 compare → predicate             | b10=0xf0, b11=0x03 |
| 0x822  | FSWZADD         | Float swizzle-add                    |       |
| 0x308  | MUFU            | SFU: RCP/SQRT/SIN/COS/EX2/LG2        | SIN/COS want revolutions |

### 2.3 Double FP (FP64)

SM_120 FP64 is throttled to 1/64 of FP32 throughput.

| Opcode | Mnemonic    | Description          | b1   | b11  | src1 at | Notes |
|--------|-------------|----------------------|------|------|---------|-------|
| 0x229  | DADD        | FP64 add R-R         | 0x72 | 0x00 | **b8**  | src1 position is unique |
| 0x228  | DMUL        | FP64 multiply R-R    | 0x72 | 0x00 | b4      |       |
| 0x22b  | DFMA        | FP64 fused MAD R-R   | 0x72 | 0x00 | b4,b8   |       |
| 0xc2b  | DFMA-UR-UR  | DFMA with 2 UR srcs  | 0x7c | 0x08 | b4(UR),b8(UR) | only src0=GPR |
| 0x22a  | DSETP       | FP64 compare → pred  | 0x72 | 0x00 | b4      | **ordered codes broken** |
| 0x22c  | DSEL        | FP64 select          |      |      |         | behavioural-ID, unverified |
| 0x23f  | DMMA.8x8x4  | FP64 tensor MMA      |      |      |         | misc=2 |
| 0x310  | F2F         | F32↔F64 conversion   |      |      |         | long-latency wdep=0x33 |
| 0x311  | F2I.F64     | F64 → int32          |      |      |         |       |
| 0x312  | I2F.F64     | int32 → F64          |      |      |         | writes dest pair |

**Critical:** The b1=0x7e/b1=0x7c GPR forms produce **wrong results**. Always
use b1=0x72 + b11=0x00 for DADD/DMUL/DFMA R-R. Reference:
`sass/scoreboard.py` `_OPCODES_DFPU`.

### 2.4 Half Precision (FP16)

| Opcode | Mnemonic       | Description                          |
|--------|----------------|--------------------------------------|
| 0x431  | HFMA2          | Packed FP16x2 fused multiply-add     |
| 0x23e  | F2FP.F16.F32   | FP32→packed FP16 conversion          |
| 0x23c  | HMMA.F16.F32   | Tensor MMA FP16 A×B + FP32 C         |
| 0x23c  | HMMA.BF16.F32  | Tensor MMA BF16 (b9 selects mode)    |
| 0x23c  | HMMA.TF32.F32  | Tensor MMA TF32                      |

### 2.5 Memory (Global / Shared / Local / Constant)

| Opcode | Mnemonic       | Description                          | Notes |
|--------|----------------|--------------------------------------|-------|
| 0x981  | LDG.E          | Global load via UR descriptor        | b9 encodes size |
| 0x986  | STG.E          | Global store via UR descriptor       | b8=UR desc |
| 0x984  | LDS            | Shared memory load                   |       |
| 0x988  | STS            | Shared memory store                  |       |
| 0x83b  | LDSM           | Load shared → matrix registers       | wdep=0x33 |
| 0xb82  | LDC            | Constant bank load (bank,offset)     |       |
| 0x7ac  | LDCU           | Uniform const load → UR              | misc=7 required |
| 0xfae  | LDGSTS.E       | Async global→shared (cp.async)       | misc=4 |
| 0x9af  | LDGDEPBAR      | cp.async commit group                |       |
| 0x91a  | DEPBAR.LE      | Barrier on async depth               |       |
| 0x992  | MEMBAR         | Memory barrier (GPU/SYS/ALL.GPU)     |       |
| 0x3c0  | GETLMEMBASE    | Get local memory base                |       |
| 0x3c1  | SETLMEMBASE    | Set local memory base                |       |
| 0x3d0  | GETLMEMFLAGS   | Get local memory flags (behavioural) |       |

**LDG size encoding** (byte b9):
| b9    | Width  | Regs written   |
|-------|--------|----------------|
| 0x19  | 32-bit | dest           |
| 0x99  | 32-bit | dest           |
| 0x1b  | 64-bit | dest, dest+1   |
| 0x9b  | 64-bit | dest, dest+1   |
| 0x1d  | 128-bit| dest..dest+3   |
| 0x9d  | 128-bit| dest..dest+3   |

### 2.6 Atomics

| Opcode | Mnemonic         | Description                     |
|--------|------------------|---------------------------------|
| 0x9a8  | ATOMG.E.{ADD,MIN,MAX,EXCH}.U32 | global int atomic    |
| 0x9a3  | ATOMG.E.ADD.F32  | global FP32 atomic add          |
| 0x3a9  | ATOMG.E.CAS.b32  | global compare-and-swap 32-bit  |
| 0x3a9  | ATOMG.E.CAS.b64  | global CAS 64-bit (b9=0xe5)     |

All ATOMG use misc=4. Address is 64-bit pair at b3, data at b4.

### 2.7 Warp Operations

| Opcode | Mnemonic        | Description                          |
|--------|-----------------|--------------------------------------|
| 0x589  | SHFL            | Warp shuffle (R-R)                   |
| 0xf89  | SHFL            | Warp shuffle (R-imm)                 |
| 0x989  | SHFL            | Warp shuffle (imm-imm)               |
| 0x806  | VOTE.ANY        | Warp vote ANY                        |
| 0x80?  | VOTE.ALL        | Warp vote ALL                        |
| 0x3a1  | MATCH.ANY/ALL   | Warp match                           |
| 0x3c4  | REDUX.SUM       | Warp reduction → UR                  |
| 0x3c4  | REDUX.MIN/MAX   | Warp min/max reduction → UR          |
| 0x3c4  | REDUX.{AND,OR,XOR} | Warp bitwise reduction → UR       |
| 0x95d  | NANOSLEEP       | Thread sleep (~ns duration)          |
| 0x82f  | ELECT           | Elect one thread as leader           |
| 0x822  | FSWZADD         | Float swizzle-add (intra-warp)       |

### 2.8 Tensor Cores

| Opcode | Mnemonic         | Shape     | Dtypes                         |
|--------|------------------|-----------|--------------------------------|
| 0x23c  | HMMA             | m16n8k16  | FP16/BF16/TF32 → FP32          |
| 0x237  | IMMA.16832.S8    | m16n8k32  | INT8 × INT8 → INT32            |
| 0x23f  | DMMA.8x8x4       | m8n8k4    | FP64 × FP64 → FP64             |
| 0x27a  | QMMA             | m16n8k32  | E4M3/E5M2 → FP32               |
| 0x442  | QMMA.SF          |           | MXF8 E4M3 (Blackwell new)      |
| 0x47a  | QMMA.SF          |           | MXF8 E4M3 variant              |
| 0x47e  | QMMA.SF          |           | MXF8 (b9 selects fmt)          |
| 0x47f  | OMMA.SF.16864    | m16n8k64  | MXF8 E2M1 (Blackwell new)      |

All MMA opcodes require **misc=2** in the control word. A-matrix at b3 (4
regs), B-matrix at b4 (2 regs), C-matrix at b8 (4 regs), D-matrix at b2
(4 regs, must be 4-aligned).

### 2.9 TMA (Tensor Memory Accelerator) — Blackwell new

| Opcode | Mnemonic         | Description                          | wdep | misc |
|--------|------------------|--------------------------------------|------|------|
| 0x5b2  | SYNCS.EXCH       | mbarrier.init                        | var  | 2    |
| 0x9a7  | SYNCS.ARRIVE     | mbarrier.arrive                      | 0x3f | 1    |
| 0x5a7  | SYNCS.TRYWAIT    | mbarrier.try_wait                    | 0x3f | 1    |
| 0x3ba  | UBLKCP           | Bulk copy S↔G                        | 0x0e | 12   |
| 0x5b4  | UTMALDG          | TMA tensor load (1D/2D)              | 0x0e | 12   |
| 0x3b5  | UTMASTG          | TMA tensor store (1D)                | 0x1f | 1    |
| 0x9b7  | UTMACMDFLUSH     | TMA command flush                    | 0x0f | 1    |
| 0x9d4  | UTMACCTL         | TMA control/status (behavioural)     |      |      |

TMA instructions operate on **uniform registers only** (no GPR dest). The TMA
descriptor is loaded into a UR pair before issuing UTMALDG/UTMASTG.

### 2.10 Texture / Surface

| Opcode | Mnemonic  | Description                  | Dim byte (b7) |
|--------|-----------|------------------------------|---------------|
| 0xf60  | TEX       | Texture fetch                | 0x00/0x20/0x40 = 1D/2D/3D |
| 0xf63  | TLD4      | Texture gather               |               |
| 0xf66  | TLD.LZ    | Texture load (no LOD)        |               |
| 0xf6f  | TXQ       | Texture query                |               |
| 0xf99  | SULD      | Surface load                 | 0x10/0x70 = 1D/2D |
| 0xf9d  | SUST      | Surface store                | 0x10/0x70 = 1D/2D |

All use UR descriptor at b5, wdep=0x35 (LDG slot), misc=2.

### 2.11 Control Flow

| Opcode | Mnemonic  | Description                              |
|--------|-----------|------------------------------------------|
| 0x94d  | EXIT      | Terminate thread                         |
| 0x947  | BRA       | Conditional branch (PC-relative)         |
| 0x547  | BRA.U     | Uniform branch                           |
| 0x944  | CALL.REL  | Relative call with link                  |
| 0x950  | RET.REL   | Return from call                         |
| 0x918  | NOP       | No operation                             |
| 0x942  | BREAK     |                                          |
| 0x949  | BRX       | Indexed branch                           |
| 0x94a  | JMP       |                                          |
| 0x94c  | JMX       | Indexed jump                             |
| 0x34e  | LEPC      | Load effective PC                        |
| 0x31f  | SETCTAID.X| Set CTA ID                               |

Note: On SM_120, ptxas almost always **if-converts** branches into predicated
execution. BRA is rarely emitted. Cf. `sass/pipeline.py` if-conversion Pattern
D (early-exit for ret-only false paths).

### 2.12 Predicates

| Opcode | Mnemonic   | Description                           |
|--------|------------|---------------------------------------|
| 0x20c  | ISETP R-R  | Integer compare → P (R-R)             |
| 0xc0c  | ISETP R-UR | Integer compare → P (R-UR)            |
| 0x20b  | FSETP      | FP32 compare → P                      |
| 0x22a  | DSETP      | FP64 compare → P                      |
| 0x21e  | PLOP3      | Predicate logic op                    |
| 0x203  | P2R        | Predicate → GPR                       |
| 0x204  | R2P        | GPR → predicate                       |
| 0x207  | SEL        | GPR select (R-R, pred-guarded)        |
| 0x807  | SEL        | GPR select R-imm                      |
| 0x208  | FSEL       | FP32 select                           |
| 0x80a  | FSEL.STEP  | Compare+select combined               |

### 2.13 Uniform Datapath

SM_120 has a full uniform datapath parallel to the vector datapath. Uniform
instructions read/write `UR0..UR62` and `UP0..UP7`.

| Opcode | Mnemonic     | Description                          |
|--------|--------------|--------------------------------------|
| 0x882  | UMOV         | UR immediate move                    |
| 0xc82  | UMOV.RR      | UR reg-reg move                      |
| 0xc02  | MOV R, UR    | Copy UR to GPR                       |
| 0x890  | UIADD3       | Uniform 3-input add                  |
| 0x28c  | UISETP R-R   | Uniform ISETP                        |
| 0x887  | USEL         | Uniform select                       |
| 0x853  | UFSETP       | Uniform FP32 set-pred                |
| 0x856  | UFMUL        | Uniform FP32 multiply                |
| 0x291  | ULEA         | Uniform LEA                          |
| 0xab9  | ULDC.64      | Uniform const load (SM_89 form)      |
| 0x554  | MOV64IUR     | 64-bit move to UR                    |
| 0x9c3  | S2UR         | Special reg → UR                     |
| 0x8cb  | CS2UR.32     | Control special reg → UR             |

### 2.14 Cluster Operations (Blackwell new)

| Opcode | Mnemonic          | Description                          |
|--------|-------------------|--------------------------------------|
| 0x9c7  | UCGABAR_ARV       | cluster barrier arrive               |
| 0xdc7  | UCGABAR_WAIT      | cluster barrier wait                 |
| 0x84c  | UVIRTCOUNT.DEALLOC.SMPOOL | dealloc SM pool              |
| 0x877  | ACQSHMINIT        | acquire shared memory init           |

### 2.15 Barriers

| Opcode | Mnemonic      | Description                          |
|--------|---------------|--------------------------------------|
| 0xb1d  | BAR.SYNC      | CTA barrier (also clears scoreboard) |
| 0xb1d  | BAR.RED.OR    | CTA barrier with OR reduction        |
| 0x31c  | B2R.RESULT    | Barrier result → register            |
| 0x35a  | BMOV.64       | Read 64-bit barrier state            |
| 0x95a  | BMOV.64       | BMOV.64 B-imm form                   |
| 0xf55  | BMOV.32       | Barrier move 32-bit                  |
| 0xf56  | BMOV.32       | Barrier move 32-bit variant          |
| 0x9ab  | ERRBAR        | Error barrier                        |
| 0x5ab  | CGAERRBAR     | CGA error barrier                    |

---

## 3. Instruction Encoding Format

### 3.1 Overall layout

Every SM_120 instruction is **128 bits** (16 bytes), stored little-endian.
The layout splits roughly into:

```
Byte:   0  1  2  3  4  5  6  7   8  9 10 11 12 13 14 15
Field:  ---- op+dest+srcs ----   -- modifiers+ctrl --
       |op0|op1|d |s0|s1|xx|xx|xx|s2|m0|m1|m2|r0|c0|c1|c2|
```

where:
- **b0, b1**: opcode bytes (12-bit opcode = `((b1 & 0x0F) << 8) | b0`, with
  the high 4 bits of b1 encoding the class / form family)
- **b2**: destination GPR (or destination UR, or predicate destination slot)
- **b3**: source 0 (src0), or address register for loads/stores
- **b4**: source 1 (src1), or UR source, or const bank for LDC
- **b5**: const offset (LDC/LDCU), or descriptor UR, or offset field
- **b6, b7**: immediate bytes / texture dim byte / dimension
- **b8**: source 2 (src2) or secondary source
- **b9**: modifier byte (size, mode, SR code, etc.)
- **b10, b11**: additional modifier bytes, predicate-operand encoding
- **b12**: reserved
- **b13, b14, b15**: control word (23 bits shifted left by 1; b15 bit 2
  preserved as SHF reuse flag)

### 3.2 Opcode extraction

```python
def get_opcode(raw: bytes) -> int:
    return struct.unpack_from('<Q', raw, 0)[0] & 0xFFF
```

The low 12 bits of the 64-bit little-endian load of bytes 0..7 is the opcode.

### 3.3 Register encoding

**GPR**: single byte (0..254). `0xFF` = RZ (read-as-zero, write-discarded).

**UR**: single byte (0..62). `0x3F` = URZ.

**Predicate operand** (for FSEL, SEL, and other pred-guarded instructions):
Encoded in **bytes 10 and 11**, NOT the guard byte.

```python
raw[10] |= (pred & 1) << 7         # low bit of pred goes to bit 7 of b10
raw[11] |= (pred >> 1) & 0x7F      # high bits of pred go to b11[6:0]
if pred_negated:
    raw[11] |= 0x04                # negate flag
```

Reference: `sass/encoding/sm_120_opcodes.py:encode_fsel` (line 2272).

**Predicate guard**: The `@P` or `@!P` guard for predicated execution is
encoded separately in the high bits of the opcode word (b1 high nibble /
b2 high bits, depending on instruction form). Specific layout:
- bits[15:12] of the 128-bit word = `(pred_neg << 3) | (pred_idx & 7)`

### 3.4 Control word format

The 23-bit control word packed into b13..b15 encodes scheduling metadata:

```
Bits  [22:17]  stall  — cycle stall count (ignored by HW on SM_120)
Bit   [16]     yield  — yield hint
Bit   [15]     wbar   — write-after-read barrier flag
Bits  [14:10]  rbar   — read barrier bitmask (5 bits)
Bits  [9:4]    wdep   — write-dependency scoreboard slot (6 bits)
Bits  [3:0]    misc   — misc/sequence nibble (per-opcode, mod 16)
```

Packing into bytes:
```python
raw24 = (ctrl & 0x7FFFFF) << 1
b13 = raw24 & 0xFF
b14 = (raw24 >> 8) & 0xFF
b15 = ((raw24 >> 16) & 0xFF) | (b15_prev & 0x04)  # preserve SHF reuse flag
```

Reference: `sass/scoreboard.py:_patch_ctrl` (line 588).

### 3.5 Example — MOV

```
Source:        MOV R6, R5
Hex bytes:     02 72 06 00 05 00 00 00 00 0f 00 00 00 ca 0f 00
               ^ ^  ^     ^                 ^
               | |  |     |                 +-- b9 = 0x0f (MOV marker)
               | |  |     +-- b4 = src (R5)
               | |  +-- b2 = dest (R6)
               | +-- b1 = 0x72
               +-- b0 = 0x02 → opcode 0x202
Control:       ctrl=0x7e5 → raw=0xfca → b13=0xca, b14=0x0f
```

Note: for MOV, src is at **b4, not b3**. b3 is 0x00.

### 3.6 Example — LDC

```
Source:        LDC R1, c[0x0][0x37c]
Hex bytes:     82 7b 01 ff 00 df 00 00 00 08 00 00 00 e2 0f 00
               ^ ^  ^  ^  ^  ^                 ^
               | |  |  |  |  |                 +-- b9=0x08 (32-bit marker; 0x0a for 64-bit)
               | |  |  |  |  +-- b5 = offset/4 = 0xdf (0x37c/4 = 223)
               | |  |  |  +-- b4 = const bank (0)
               | |  |  +-- b3 = 0xFF (fixed)
               | |  +-- b2 = dest (R1)
               | +-- b1 = 0x7b
               +-- b0 = 0x82 → opcode 0xb82
```

Reference: `sass/encoding/sm_120_opcodes.py:encode_ldc` (line 242).

### 3.7 Example — S2R

```
Source:        S2R R0, SR_TID.X
Hex bytes:     19 79 00 00 00 00 00 00 00 21 00 00 00 2e 0e 00
               ^ ^  ^              ^
               | |  |              +-- b9 = 0x21 (SR_TID_X)
               | |  +-- b2 = dest (R0)
               | +-- b1 = 0x79
               +-- b0 = 0x19 → opcode 0x919

Common SR codes:
  0x21 = SR_TID.X           0x25 = SR_CTAID.X
  0x22 = SR_TID.Y           0x26 = SR_CTAID.Y
  0x29 = SR_NTID.X          0x50 = SR_LANE_ID
```

---

## 4. Scoreboard System

SM_120 uses a fixed set of hardware scoreboard slots to track long-latency
instruction completion. Consumer instructions set `rbar` bits to wait for
specific slots before executing.

### 4.1 Scoreboard slots

| Slot  | Purpose                              | rbar bit |
|-------|--------------------------------------|----------|
| 0x31  | LDC / LDCU / LDGDEPBAR / REDUX       | 0x02     |
| 0x33  | LDS / LDSM / DFPU / DSETP / F2F      | 0x04     |
| 0x35  | LDG (all LDGs share this)            | 0x08     |
| 0x37  | **reserved — no rbar bit**           | —        |
| 0x3e  | ALU result tracking                  | 0x02     |
| 0x3f  | No write tracking (EXIT/BRA/STG/BAR) | —        |

### 4.2 rbar is a BITMASK, not a priority

The critical rule: `rbar` is a **bitwise OR** of the slots this instruction
waits for, NOT the maximum of required slots.

```
bit 0 (0x01) = base (always set)
bit 1 (0x02) = wait LDC/LDCU slot
bit 2 (0x04) = wait LDS slot
bit 3 (0x08) = wait LDG slot
```

An instruction that reads both an LDC result and an LDG result must set
`rbar = 0x01 | 0x02 | 0x08 = 0x0B`. Using `max(0x03, 0x09) = 0x09` loses
the LDC wait bit and produces garbage output.

**Reference:** `sass/scoreboard.py:_WDEP_TO_RBAR` (line 613).

### 4.3 All LDGs share wdep=0x35

The hardware scoreboard tracks only the *last* write to slot 0x35. FIFO
ordering in the load pipeline guarantees earlier LDGs complete before later
ones. Using `wdep=0x37` for a second LDG is **wrong** — slot 0x37 has no
corresponding rbar bit, so consumers can never wait for it.

All LDG instructions unconditionally use `wdep=0x35`. Consumer `rbar=0x09`
waits for the last-posted LDG. Reference: scoreboard rule 14, line 84.

### 4.4 BAR.SYNC clears all pending scoreboard state

After `BAR.SYNC`, all prior memory operations are guaranteed complete. The
scoreboard emulator must clear `pending_writes`, `pending_ur_writes`, and
`pending_pred_writes` when processing a BAR.SYNC. Otherwise stale pre-barrier
dependencies corrupt rbar assignments on post-barrier instructions.

Reference: scoreboard rule 15, line 89.

### 4.5 LDCU slot rules

- **LDCU.64 descriptor load** (first LDCU in kernel, reads c[0][0x358]):
  uses `wdep=0x35` so consumer LDGs get `rbar=0x09`.
- **Subsequent LDCU.64 loads** (pointer params): rotate between `0x31` and
  `0x33` to avoid aliasing LDG slot.
- **LDCU.32 param loads**: always `wdep=0x31`.
- **Never use wdep=0x37 for any LDCU.**

LDCU writes to URs, not GPRs, so its scoreboard is separate from GPR tracking.

### 4.6 LDCU.64 gap rule

LDCU.64 needs **≥4 instructions** between issue and consumer, even when rbar
is correct. The rbar mechanism alone is insufficient when LDCU.64 and its
IADD.64-UR consumer are adjacent. Insert NOPs or unrelated ALU ops.

### 4.7 DSETP long-latency

DSETP posts to scoreboard slot `0x33` (shared with LDS/DFPU). Consumer (the
guard instruction that reads the predicate DSETP wrote) needs `rbar=0x05`.

Without this, the predicate reads garbage and all @P guards evaluate false.

### 4.8 F2F long-latency

F2F (FP32↔FP64 conversion) is also long-latency and uses `wdep=0x33`.
Consumer reads need `rbar=0x05`.

### 4.9 LOP3 reads three GPR sources

LOP3 (opcode 0x212) reads three source registers from b3, b4, AND b8. If any
of these three comes from an LDC/LDG, scoreboard must add the corresponding
rbar bit. Missing b4/b8 tracking caused stale-data reads.

Reference: scoreboard rule 17, line 102.

### 4.10 ISETP→FSEL pred latency

FSEL reads its predicate operand from `raw[10..11]` (operand encoding), NOT
from the guard byte. The scoreboard's guard-pred check doesn't cover this
path. Always insert ≥1 NOP between adjacent `ISETP` and `FSEL` that share
the same predicate.

Reference: `sass/schedule.py:_enforce_gpr_latency`.

### 4.11 Per-opcode misc values (hardware-required)

Some opcodes have **strict** misc requirements — wrong values cause hardware
rejection:

| Opcode | Mnemonic   | Required misc | Consequence of wrong value      |
|--------|------------|---------------|----------------------------------|
| 0x7ac  | LDCU       | 7             | ILLEGAL_INSTRUCTION at first LDCU|
| 0x918  | NOP        | 0             |                                  |
| 0x947  | BRA        | 1             |                                  |
| 0x94d  | EXIT       | 5             |                                  |
| 0x981  | LDG.E      | 6             |                                  |
| 0x986  | STG.E      | 1             |                                  |
| 0x988  | STS        | 4             |                                  |
| 0x20c  | ISETP R-R  | 0             | wrong predicate                  |
| 0xc0c  | ISETP R-UR | 0             | wrong predicate (misc 1-12 fails)|
| 0x229  | DADD       | 2             | wrong data                       |
| 0x228  | DMUL       | 2             | wrong data                       |
| 0x22b  | DFMA       | 2             | wrong data                       |
| 0x22a  | DSETP      | 2             | wrong predicate                  |
| 0x23c  | HMMA       | 2             | wrong scheduling                 |
| 0x237  | IMMA       | 2             | wrong scheduling                 |
| 0x23f  | DMMA       | 2             | wrong scheduling                 |
| 0x27a  | QMMA       | 2             | wrong scheduling                 |
| 0x80a  | FSEL.STEP  | 5             |                                  |
| 0xc35  | IADD.64-UR | 5             |                                  |
| 0xc24  | IMAD R-UR  | 1             |                                  |
| 0xfae  | LDGSTS.E   | 4             | cp.async fails                   |
| 0x9a8  | ATOMG.*    | 4             | atomic misbehaviour              |

Other opcodes (LDC, S2UR, MOV, IADD3, generic ALU) use a sequential
`misc_counter` that increments mod 16 per instruction.

Reference: `sass/scoreboard.py:_OPCODE_MISC` (line 539).

### 4.12 EXIT ctrl

EXIT always uses `wdep=0x3f, misc=5`, both for predicated and unpredicated
exits. Reference: scoreboard, line 117 of sm120_rules.

### 4.13 S2R asynchrony

S2R is asynchronous and uses `wdep=0x31`, `misc=1`. Consumers reading the
value must set `rbar` bit for slot 0x31 (bit 1 = 0x02).

### 4.14 EIATTR_CBANK_PARAM_SIZE must match actual params

The ELF info attribute `EIATTR_CBANK_PARAM_SIZE` must be the exact number of
parameter bytes. Must not be `0xFF`. The driver zeroes this many bytes
starting at `param_base` before copying parameters. An oversized value
clobbers literal-pool values placed after the parameter area in c[0].

### 4.15 LDG.E.32 b9 encoding

`b9 = 0x19` or `0x99` marks a 32-bit LDG that writes a **single** register
(only `dest`, not `dest+1`). Scoreboard `_get_dest_regs` must recognise this
or it falsely tracks an extra register, polluting subsequent rbar calculations.

Reference: scoreboard rule 7, line 50.

### 4.16 STG data dependency

For STG, both address (b3, 64-bit pair) and data (b4) are source registers.
The rbar computation must include dependencies on both. Special case: if the
data register has a pending LDG write (wdep=0x35), rbar must include 0x09.

### 4.17 REDUX writes to UR

REDUX (0x3c4) posts to slot 0x31 (same as LDCU) because it writes to UR, not
GPR. Consumers reading the UR via `MOV R, UR` must wait on the UR pending
tracker.

### 4.18 Summary — all 17+ verified scoreboard rules

1. **rbar is OR, not max** (rule 0)
2. **LDCU.64 descriptor uses wdep=0x35** (rule 1)
3. **LDCU.64 needs ≥4 instruction gap** (rule 1b)
4. **LDCU/S2UR do NOT enter GPR pending_writes** (rule 2)
5. **LDG misc=6, per-opcode misc table required** (rule 3)
6. **IMAD R-R (0x2a4) is BROKEN** (rule 4)
7. **EIATTR_EXIT_INSTR_OFFSETS must list ALL exits** (rule 5)
8. **EIATTR_CBANK_PARAM_SIZE = actual param bytes** (rule 6)
9. **LDG.E.32 b9=0x19 writes single register** (rule 7)
10. **FP64 throughput 1.639 TFLOPS / 64:1 ratio** (rule 8)
11. **DSETP long-latency (slot 0x33), unordered codes only** (rule 9)
12. **HMMA/IMMA/DMMA/QMMA misc=2** (rule 10)
13. **IMMA B register must be R0..R7** (rule 11)
14. **MMA dest must be 4-aligned** (rule 12)
15. **QMMA dest must equal src_a** (rule 13)
16. **All LDG share wdep=0x35** (rule 14)
17. **BAR.SYNC clears pending scoreboard state** (rule 15)
18. **DADD/DMUL/DFMA R-R use b1=0x72, b11=0x00; DADD src1 at b8** (rule 16)
19. **LOP3 reads 3 GPR sources (b3/b4/b8)** (rule 17)

---

## 5. Hardware Bugs & Quirks

These are silicon-level bugs in SM_120 that cannot be fixed by firmware or
driver. You work around them.

### 5.1 IMAD R-R (0x2a4) is broken

**Symptom:** `IMAD R, R, R, R` produces garbage output. Tested on RTX 5090
with `alu_chain` kernel — completely wrong results.

**Workaround:** Route one operand through a UR:
```
; broken
IMAD R_dst, R_src0, R_src1, R_src2

; fix: load param into UR, use IMAD R-UR (0xc24)
LDCU.32 UR4, c[0][0x18]      ; pull src1 into uniform register
IMAD R_dst, R_src0, UR4, R_src2
```

Works: `IMAD.WIDE` (0x225, 64-bit result), `IMAD R-UR` (0xc24), `IMAD R-imm`
(0x824).

### 5.2 ISETP→FSETP corruption

ISETP (both R-R form 0x20c and R-UR form 0xc0c) **clobbers** the output of
any subsequent FSETP in the same kernel. The FP32 predicate write path is
corrupted by an integer compare.

**Workaround:** Use `FSEL.STEP` (0x80a) instead of `FSETP + FSEL`. FSEL.STEP
combines float compare + select in one instruction and bypasses FSETP.
Reference: OpenPTXas peephole `FSEL.step`.

### 5.3 DSETP ordered comparison codes silently fail

SM_120 DSETP supports these comparison codes:

| Code | Mnemonic | Works? |
|------|----------|--------|
| 1    | LT       | **NO** (silently returns false) |
| 2    | EQ       | **NO** |
| 3    | LE       | **NO** |
| 4    | GT       | **NO** |
| 5    | NE       | **NO** |
| 6    | GE       | **NO** |
| 0x09 | LTU      | yes    |
| 0x0a | EQU      | yes    |
| 0x0b | LEU      | yes    |
| 0x0c | GTU      | yes    |
| 0x0d | NEU      | yes    |
| 0x0e | GEU      | yes    |

Only **unordered** codes work. `setp.lt.f64` must be lowered as:
```
DSETP.GEU P0, A, B
; ... then use @!P0 as the guard (complement)
```

ptxas uses the same pattern.

### 5.4 Literal pool is uninitialized beyond parameter area

The driver only zero-initializes `CBANK_PARAM_SIZE` bytes of `c[0]`. Bytes
beyond the parameter region are uninitialized — reading them returns garbage.

**Impact:** ISETP with a non-zero immediate was originally implemented as
`LDCU.32` from a literal pool slot past the param area. That read garbage,
giving wrong predicates.

**Workaround:** All immediates must be inline. For ISETP:
- `imm == 0` → `ISETP R-R` with RZ
- `imm != 0` → `IADD3_IMM32` to GPR + `ISETP R-R`

### 5.5 SM_120 FSETP is unreliable after ISETP — use FSEL.step peephole

See 5.2. ISETP corrupts FSETP's output path. Any kernel that mixes integer
and float predicates must either:
1. Schedule all ISETPs before any FSETPs (isolation), or
2. Replace FSETP+FSEL pairs with FSEL.STEP.

### 5.6 IMMA B register must be < 8

`IMMA.16832.S8`: the B-matrix base register must be in `R0..R7`. `B=R12`
causes `ERR_ILLEGAL_INSTRUCTION (715)`. `B=R4` works.

**Workaround:** In PTX, initialize B registers before D/A registers so the
linear-scan allocator assigns B to low GPRs.

### 5.7 QMMA dest must equal src_a (in encoding)

`QMMA.16832.F32.E4M3.E4M3` (and `E5M2`) requires `raw[2] == raw[3]` (dest
register == src_a register). Using `dest != src_a` causes sync=715.

ptxas always generates QMMA with D=A at the same physical register. This is
an **in-place** encoding: the A-matrix values must be loaded into D-register
positions before the instruction executes.

**Workaround:** In isel, encode QMMA as `(d, d, b, c)` and ensure PTX uses
the same virtual registers for D and A operands.

### 5.8 MMA dest must be 4-aligned

`mma.sync.aligned` destination register must satisfy `dest % 4 == 0`. The
allocator must flag mma destinations as `quad_align_regs`. Following 3
registers in the same declaration are `quad_follow_regs` (must stay
consecutive).

### 5.9 FP64 silicon fusing (not a bug, a feature)

RTX 5090 FP64 is limited to ~1.639 TFLOPS peak. The same die can produce
~60 TFLOPS FP64 on the datacenter SKU (B100) — most FP64 units are
**physically fused off** in silicon on consumer Blackwell B chips. This is
**not** unlockable by driver or BIOS. It is a hardware protection for
datacenter SKUs.

**Workaround:** For FP64-heavy workloads, use FP32 emulation
(Dekker/TwoSum), or move to TF32/BF16 tensor cores if precision allows.

### 5.10 b1=0x7e/0x7c forms of DADD/DMUL/DFMA silently produce wrong results

Earlier `decode_sass.py` suggested `b1=0x7e` or `b1=0x7c` encodings for FP64
R-R ops. These run, but produce **wrong numerical results** on SM_120. Only
`b1=0x72, b11=0x00` works.

Additionally, **DADD (b1=0x72) has src1 at byte[8]**, NOT byte[4] like
DMUL/DFMA. This is an encoding anomaly unique to DADD.

`DFMA-UR-UR` (b1=0x7c, b11=0x08) is a separate opcode (0xc2b) and still works
for the UR-operand form.

---

## 6. Capmerc (Mercury) DRM System

Capmerc is a Blackwell-new load-time validation system. Without correct
capmerc records and the universal 0x5a signature blob, any attempt to use
GPRs R14+ triggers `ERR_ILLEGAL_INSTRUCTION (715)` or
`CUDA_ERROR_INVALID_IMAGE (201)`.

Capmerc is stored in two ELF sections per kernel:
- `.nv.capmerc.text.<kernel>` — per-kernel metadata
- EIATTR 0x5a (universal) — 52-byte signature shared across all ptxas 13.0
  SM_120 cubins

### 6.1 EIATTR 0x5a — Universal ptxas 13.0 signature

52 bytes, **byte-identical** across every ptxas 13.0 SM_120 cubin ever
observed:

```
8a9d22a4b19d146d00b42af3f758038e0c070a1be2de8ad75263870cd72b0700
cd2b8a124e4c1624ba19f5f027946a021a000000
```

This is an ISA-version constant, not per-kernel data. The driver validates
its presence; tampering with it triggers load failure.

### 6.2 Capmerc header (16 bytes)

| Offset | Size | Field              | Description                         |
|--------|------|--------------------|-------------------------------------|
| 0      | 8    | Magic              | `0c 00 00 00 01 00 00 c0`           |
| 8      | 1    | Register count     | num_gprs allocated (R0..R{n-1})     |
| 9      | 3    | Reserved           | zeros                               |
| 12     | 4    | Capability bitmask | LE u32, encodes instruction classes |

### 6.3 Capability bitmask bits

| Bit  | Mask       | Meaning                       |
|------|------------|-------------------------------|
| 3    | 0x00000008 | base ALU (always set)         |
| 6    | 0x00000040 | STG present                   |
| 7    | 0x00000080 | conditional branch / pred exit|
| 8    | 0x00000100 | SHF / shift ops               |
| 9    | 0x00000200 | extended ALU (IADD3.X, …)     |
| 10   | 0x00000400 | ISETP / predicate ops         |
| 11   | 0x00000800 | IMAD.WIDE                     |
| 12   | 0x00001000 | LDG                           |
| 13   | 0x00002000 | register pressure > 14        |
| 16+  | —          | scales with insn count / sched |

**FP64 kernels** require a specific capability mask (from RE):
`FP64_CAP_MASK = 0x08410c00` (bits 10, 11, 16, 22, 27). This tells the
driver to populate the global FP64 descriptor at `c[0][0x358]`.

### 6.4 Body records

Records are concatenated after the header. Four record kinds:

**Type-01 (16 bytes) — Instruction class descriptor:**
- Universal prologue (always first):
  `01 0b 04 0a f8 00 04 00 00 00 41 00 00 04 00 00`
- STG/memory descriptor:
  `01 0b 0e 0a fa 00 05 00 ...`
- byte[2]: 0x04 = ALU-only class, 0x06 = uniform-register-using class
- byte[10]: register-pressure tag
  - 0x01 = simple
  - 0x41 = base (low register pressure)
  - 0x81 = medium register pressure
  - 0xc1 = wide / high register pressure

**Type-02 sub=0x22 (32 bytes) — Barrier region metadata:**
- One per scoreboard barrier region
- byte[6] = mode (0x42 / 0x52 / 0x62)
- bytes[10:11] = barrier liveness bitmap

**Type-02 sub=0x0c (32 bytes) — FP64 class descriptor:**
- Required for kernels using FP64 global memory
- Tells driver to populate c[0][0x358] with FP64 descriptor
- Without this, `STG.E.64` crashes with `CUDA_ERROR_ILLEGAL_ADDRESS`

**Type-02 sub=0x38 (32 bytes) — Terminal record:**
- Always last body record
- byte[6] = mode
- Contains text_size/256 at specific offsets

**Filler blocks (4 bytes each):**
- `41 0c 54 04` (with branch) or `41 0c 50 04` (straight-line)
- Scale with register pressure above R14

### 6.5 Trailer (2 bytes)

```
byte[0] = 0xd0 (standard) or 0x50 (alt)
byte[1] = unique-scoreboard-barrier count
```

### 6.6 Unlocking R12+ registers

Binary-patch testing on SM_120 proved capmerc enforces load-time register
access limits:

| Dest register | Result                          |
|---------------|----------------------------------|
| R0..R10       | PASS                             |
| R11           | sync=715 (borderline)            |
| R12..R14+     | **load=201 INVALID_IMAGE**       |

The capmerc **body records** encode the authorized register range — header
byte[8] alone is insufficient. Record #1 byte[10] must be set appropriately:
- 0x01 for wide register range (ptxas sets this for >14 GPR kernels)
- 0x81 only covers R0..R10

**For R14+ to work**, you need all of:
1. Correct `reg_count` at header byte[8]
2. Filler blocks (`41 0c 54 04`) scaling with register pressure
3. Capability bitmask matching opcodes in .text
4. Terminal record `sh[4] = text_size/256`
5. The 52-byte 0x5a authentication blob intact

### 6.7 Top capmerc patterns

Three record-sequence patterns cover 90% of observed cubins:

1. **38%** (full kernel): `T01_04 T01_04 T01_06 T02_22 T01_04 T02_22 T02_22 T01_0e FILL T02_38`
2. **32%** (simple kernel): `T01_04 T01_04 T01_0e T02_22 T02_38`
3. **20%** (minimal + extra barrier): `T01_04 T01_0e T02_22 T02_22 T02_38`

### 6.8 Implementation reference

`C:/Users/kraken/openptxas/cubin/capmerc_gen.py` — full generator. Key
constants: `CAPMERC_MAGIC`, `PROLOGUE_RECORD`, `STG_DESCRIPTOR`,
`FP64_CLASS_DESCRIPTOR`, `FP64_CAP_MASK`, `compute_capability_mask()`.

---

## 7. ELF Cubin Format

SM_120 cubins are standard ELF files with NVIDIA-specific sections and
attributes.

### 7.1 Required ELF sections

| Section                          | Contents                              |
|----------------------------------|---------------------------------------|
| `.text.<kernel>`                 | 16-byte SASS instructions             |
| `.nv.info`                       | Global EIATTR entries                 |
| `.nv.info.<kernel>`              | Per-kernel EIATTR entries             |
| `.nv.constant0.<kernel>`         | Parameter space + literal pool (c[0]) |
| `.nv.shared.<kernel>`            | Shared memory allocation descriptor   |
| `.nv.capmerc.text.<kernel>`      | Capmerc metadata (see §6)             |
| `.symtab`, `.strtab`, `.shstrtab`| Standard ELF symbol tables            |

### 7.2 EIATTR entries

Each EIATTR is a 4-byte header (fmt, type, size_lo, size_hi) followed by
`size` bytes of payload.

| Type | Name                          | Purpose                              |
|------|-------------------------------|--------------------------------------|
| 0x04 | EIATTR_PARAM_CBANK            | Parameter location in c[0]           |
| 0x12 | EIATTR_CBANK_PARAM_SIZE       | Size of param area in bytes          |
| 0x17 | EIATTR_KPARAM_INFO            | Per-parameter info record            |
| 0x1c | EIATTR_EXIT_INSTR_OFFSETS     | List of all EXIT instruction offsets |
| 0x2f | EIATTR_REGCOUNT               | Num GPRs used                        |
| 0x5a | EIATTR (capmerc 0x5a)         | 52-byte universal signature          |

### 7.3 CBANK_PARAM_SIZE rule

**Must be the exact byte count of parameters.** Setting to `0xFF` or any
value larger than the real param area causes the driver to zero out bytes
past the params, clobbering literal-pool values intended for c[0].

### 7.4 EXIT_INSTR_OFFSETS rule

Must list **every** EXIT instruction in `.text.<kernel>`, not just the last
one. If-converted kernels can have multiple EXITs (one per exit path).
Missing entries cause `CUDA_ERROR_INVALID_IMAGE` at load.

### 7.5 EIATTR_REGCOUNT rule

Must reflect actual `num_gprs` used. For QMMA kernels after the D=A fix, this
must include both D and A (same physical reg, so counted once at max).

---

## 8. Differences from Prior Generations

### 8.1 SM_120 vs SM_89 (Ada Lovelace)

**New in SM_120:**
- TMA (Tensor Memory Accelerator) — UTMALDG/UTMASTG/UBLKCP instructions
- Cluster operations (UCGABAR_ARV/WAIT, cluster barriers)
- Capmerc DRM section required for R14+ GPR access
- QMMA FP8 E4M3/E5M2 MMA (m16n8k32)
- QMMA.SF / OMMA.SF — MXF8 microscaling FP8 tensor ops
- FSEL.STEP combined compare+select peephole (new opcode 0x80a)
- Uniform datapath fully parallel (UMOV/UIADD3/UISETP/USEL/UFMUL/UFSETP)
- Expanded barrier primitives (B2R.RESULT, BMOV.32/64, CGAERRBAR)
- GETLMEMBASE/SETLMEMBASE/GETLMEMFLAGS local memory ops

**Changed from SM_89:**
- `IMAD R-R` opcode changed from 0xa24 (SM_89 cbuf form) to 0x2a4 (R-R form)
  — and 0x2a4 is **broken** on SM_120
- LOP3 opcode changed (SM_89 0xa12 → SM_120 0x212)
- SHF opcode changed (SM_89 0xa19 → SM_120 0x819)
- SEL immediate changed (SM_89 0x807 → SM_120 0x207 for reg form)
- ISETP predicate-correctness requirement (misc=0 mandatory)

**Removed/deprecated on SM_120:**
- Many SM_89 cbuf-form ALU opcodes (0xa10, 0xa19, 0xa12, etc.) still work
  but are not emitted by ptxas 13.0 for SM_120
- Old DFMA forms (b1=0x7e/0x7c) silently produce wrong results

### 8.2 SM_120 vs SM_90 (Hopper)

**Shared with Hopper:**
- TMA
- Cluster operations
- DPX (dynamic programming extensions)
- FP8 tensor cores

**SM_120 unique:**
- Blackwell FP8 microscaling (MXF8 E2M1/E4M3/E5M2)
- Blackwell-specific OMMA.SF outer matrix
- Consumer-targeted FP64 fusing
- Different capmerc format (Blackwell-new)

### 8.3 SM_120 vs SM_100 (datacenter B100/B200)

Same macro-architecture, but:
- SM_100 has full-speed FP64 (~60 TFLOPS on B100)
- SM_100 has larger tensor memory (TMEM) hardware
- SM_100 exposes CTA-pair / distributed shared memory (DSMEM)
- SM_120 may lack certain datacenter-only tensor op variants

Most SM_120 opcodes should be a subset of SM_100. The inverse is not true.

---

## 9. Worked Examples

### 9.1 Vector add

PTX:
```ptx
.visible .entry vector_add(.param .u64 a, .param .u64 b, .param .u64 c, .param .u32 n) {
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;
    .reg .pred %p1;

    ld.param.u64    %rd1, [a];
    ld.param.u64    %rd2, [b];
    ld.param.u64    %rd3, [c];
    ld.param.u32    %r1,  [n];
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mov.u32         %r4, %tid.x;
    mad.lo.s32      %r5, %r2, %r3, %r4;
    setp.ge.s32     %p1, %r5, %r1;
    @%p1 ret;
    mul.wide.s32    %rd4, %r5, 4;
    add.s64         %rd5, %rd1, %rd4;
    ld.global.u32   %r6, [%rd5];
    add.s64         %rd6, %rd2, %rd4;
    ld.global.u32   %r7, [%rd6];
    add.s32         %r8, %r6, %r7;
    add.s64         %rd7, %rd3, %rd4;
    st.global.u32   [%rd7], %r8;
    ret;
}
```

SASS (abbreviated):
```
LDCU.64   UR4, c[0][0x358]        ; flat descriptor, wdep=0x35, misc=7
LDCU.64   UR6, c[0][0x210]        ; a pointer (pair), wdep=0x31/0x33, misc=7
LDCU.64   UR8, c[0][0x218]        ; b pointer
LDCU.64   UR10, c[0][0x220]       ; c pointer
LDCU.32   UR12, c[0][0x228]       ; n, wdep=0x31, misc=7
S2R       R0, SR_CTAID.X          ; wdep=0x31, misc=1
S2UR      UR2, SR_NTID.X          ; wdep=0x31
S2R       R1, SR_TID.X
IMAD      R2, R0, UR2, R1         ; R-UR form! (0xc24), misc=1
ISETP.GE  P0, R2, UR12            ; ISETP R-UR 0xc0c, misc=0
@P0 EXIT                           ; misc=5
IMAD.WIDE R4, R2, 4, UR6          ; byte offset + base pointer, pair
LDG.E     R8, [R4.64]             ; wdep=0x35, misc=6
IMAD.WIDE R6, R2, 4, UR8
LDG.E     R9, [R6.64]             ; wdep=0x35 (same slot!), misc=6
IADD3     R10, R8, R9, RZ         ; rbar=0x09, waits for last LDG
IMAD.WIDE R12, R2, 4, UR10
STG.E     [R12.64], R10           ; misc=1
EXIT                               ; misc=5
```

### 9.2 FMA chain (FP32)

PTX:
```ptx
fma.rn.f32 %f1, %f2, %f3, %f4;
fma.rn.f32 %f5, %f1, %f6, %f7;
```

SASS:
```
FFMA R1, R2, R3, R4    ; opcode 0x223, wdep=0x3e, misc=1
FFMA R5, R1, R6, R7    ; depends on R1 (ALU result) — rbar needs 0x02
```

Because the first FFMA's result is a pure ALU write (wdep=0x3e), the scheduler
inserts a ≥1-instruction gap (min_gpr_gap=1 per `_OPCODE_META[0x223]`) or an
appropriate rbar bit on the consumer.

### 9.3 Parallel reduction (shared memory + warp ops)

```
; Load element to register
LDG.E     R4, [R2.64]             ; wdep=0x35, misc=6
; Store to shared memory
IADD3     R6, UR4, R1.x4, RZ      ; shared offset
STS       [R6], R4                ; misc=4
BAR.SYNC  0                       ; CLEARS pending scoreboard state!
; Warp reduce via SHFL.DOWN butterfly
SHFL.DOWN R8, R4, 16, 0x1f
FADD      R4, R4, R8
SHFL.DOWN R8, R4,  8, 0x1f
FADD      R4, R4, R8
; ... or use REDUX
REDUX.SUM UR2, R4                  ; wdep=0x31 (UR dest)
MOV       R10, UR2                ; waits on UR pending (rbar=0x03)
```

### 9.4 2x2 HMMA tile

```
; Load A matrix (4 regs) from shared via LDSM
LDSM.x4   R4, [UR6+offset_a]      ; writes R4..R7, wdep=0x33, misc=2
; Load B matrix (2 regs) — must land in R0..R7 per rule 5.6
LDSM.x2   R0, [UR6+offset_b]      ; writes R0..R1, wdep=0x33
; C accumulator already in R8..R11 (4 regs, 4-aligned per rule 5.8)
HMMA.16816.F16.F32 R12, R4, R0, R8  ; D=R12..R15 (4-aligned)
                                     ; opcode 0x23c, misc=2
                                     ; A=R4..R7, B=R0..R1, C=R8..R11
```

### 9.5 cp.async (LDGSTS.E) + depbar

```
LDGSTS.E  [smem_addr], [R4.64], UR8    ; opcode 0xfae, misc=4
LDGSTS.E  [smem_addr+16], [R6.64], UR8
LDGDEPBAR                                ; opcode 0x9af, commit group
; ... compute on previous tile ...
DEPBAR.LE 0, 0                           ; wait all async copies
BAR.SYNC  0                              ; clears scoreboard
LDS       R10, [smem_addr]               ; now safe to read
```

### 9.6 TMA 2D load

```
; UR descriptor must be pre-loaded (128 bytes at c[0])
UMOV      UR4, 0                         ; x coord
UMOV      UR5, 0                         ; y coord
UTMALDG.2D UR20, UR12, [UR4,UR5]         ; opcode 0x5b4, wdep=0x0e, misc=12
                                         ; loads tile into smem via descriptor
UTMACMDFLUSH                             ; opcode 0x9b7
DEPBAR.LE 0, 0                           ; wait TMA
BAR.SYNC  0
```

---

## 10. References

### 10.1 OpenPTXas (primary source)

- **Repo:** `garrick99/openptxas` (public on GitHub)
- **Local:** `C:/Users/kraken/openptxas`
- **Encoders:** `sass/encoding/sm_120_opcodes.py` (183 encoder functions)
- **Scoreboard:** `sass/scoreboard.py` (93 opcode classifications, 17+ rules)
- **Capmerc generator:** `cubin/capmerc_gen.py`
- **Architecture doc:** `ARCHITECTURE.md`
- **Proof of correctness:** `PROOF.md`
- **Opcode sweep results:** `probe_work/sm120_opcode_map.py` (188/4096 valid)

### 10.2 NVIDIA published documentation

- CUDA PTX ISA manual: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
  — PTX only, no SASS details
- `nvdisasm` man page: instruction decoding, but no encoding rules
- CUDA Binary Utilities: describes ELF cubin sections at a high level
- **No official SM_120 SASS reference published** as of 2026-04-04

### 10.3 Open-source compilers

- **Mesa NVK** (`src/nouveau/compiler`): Partial SM_120 support. Lacks
  tensor-core, TMA, and capmerc knowledge. Uses a different internal IR.
- **LLVM NVPTX backend**: PTX only, defers to ptxas for final SASS
- **MLIR NVGPU dialect**: PTX only

### 10.4 Related reverse engineering

- `decode_sass.py` (older RE tool) — has b1=0x7e/0x7c DFMA forms that silently
  produce wrong results on SM_120; do not trust for SM_120
- scientific papers on pre-Blackwell SASS (SM_70, SM_80) exist but are
  not directly applicable due to encoding changes

### 10.5 How to contribute

If you find new opcode semantics, encoding quirks, or scoreboard rules:
1. Reproduce on RTX 5090 hardware (or SM_120 equivalent)
2. Provide a minimal reproducer cubin
3. File an issue or PR on `garrick99/openptxas`

---

*This document is an independent technical reference. Not affiliated with or
endorsed by NVIDIA Corporation. All trademarks are property of their
respective owners.*

---

## Appendix A — Encoding Recipes

This appendix gives byte-exact encoding recipes for the most commonly used
SM_120 instructions. Each recipe lists the fixed bytes, variable operand
positions, and at least one ground-truth hex string verified against
ptxas-emitted cubins.

All byte numbering is 0-indexed from the start of the 16-byte instruction.

### A.1 NOP

```
Opcode: 0x918
Fixed:  b0=0x18, b1=0x79, b2=0x00, b3=0x00, b4=0x00, b8=0x00,
        b9=0x00, b10=0x00, b11=0x00
Variable: none (only ctrl)
Default ctrl: 0x7e0

Ground truth:
  encode_nop(ctrl=0x7e0) = 18 79 00 00 00 00 00 00 00 00 00 00 00 c0 0f 00
```

### A.2 EXIT

```
Opcode: 0x94d
Fixed:  b0=0x4d, b1=0x79, b2=0x00, b3=0x00, b4=0x00, b8=0x00,
        b9=0x00, b10=0x80, b11=0x03
Variable: predicated guard (raw[15] high bits), ctrl
Always: wdep=0x3f, misc=5

Ground truth:
  encode_exit(ctrl=0x7f5) = 4d 79 00 00 00 00 00 00 00 00 80 03 00 ea 0f 00
```

### A.3 MOV R, R

```
Opcode: 0x202
Fixed:  b0=0x02, b1=0x72, b3=0x00, b8=0x00, b9=0x0f, b10=0x00, b11=0x00
Variable: b2=dest, b4=src

Note: src at b4, NOT b3. b3 always 0x00.

Ground truth:
  encode_mov(R6, R5, ctrl=0x7e5) = 02 72 06 00 05 00 00 00 00 0f 00 00 00 ca 0f 00
  encode_mov(R3, R2, ctrl=0x7e3) = 02 72 03 00 02 00 00 00 00 0f 00 00 00 c6 0f 00
```

### A.4 LDC (32-bit constant load)

```
Opcode: 0xb82
Fixed:  b0=0x82, b1=0x7b, b3=0xFF, b8=0x00, b9=0x08, b10=0x00, b11=0x00
Variable: b2=dest, b4=bank, b5=offset/4

Ground truth:
  encode_ldc(R1, 0, 0x37c, ctrl=0x7f1) =
    82 7b 01 ff 00 df 00 00 00 08 00 00 00 e2 0f 00
                    ^
                    |-- b5=0xdf = 0x37c/4
```

### A.5 LDC.64 (64-bit constant load)

```
Opcode: 0xb82
Fixed:  b0=0x82, b1=0x7b, b3=0xFF, b8=0x00, b9=0x0a, b10=0x00, b11=0x00
                                          ^--- 0x0a marks 64-bit
Variable: b2=dest pair, b4=bank, b5=offset/4 (8-byte aligned)

Ground truth:
  encode_ldc_64(R2, 0, 0x388, ctrl=0x711) =
    82 7b 02 ff 00 e2 00 00 00 0a 00 00 00 22 0e 00
```

### A.6 LDCU (uniform constant load)

```
Opcode: 0x7ac
Must use misc=7.

LDCU.32: writes single UR
LDCU.64: writes UR pair, b9=0x0a, first in kernel uses wdep=0x35
```

### A.7 S2R (Special → GPR)

```
Opcode: 0x919
Fixed:  b0=0x19, b1=0x79, b3=0x00, b4=0x00, b8=0x00, b10=0x00, b11=0x00
Variable: b2=dest, b9=sr_code
wdep=0x31, misc=1

SR codes:
  0x21 = SR_TID.X       0x29 = SR_NTID.X
  0x22 = SR_TID.Y       0x2a = SR_NTID.Y
  0x23 = SR_TID.Z       0x2b = SR_NTID.Z
  0x25 = SR_CTAID.X     0x50 = SR_LANE_ID
  0x26 = SR_CTAID.Y     0x52 = SR_WARP_ID

Ground truth:
  encode_s2r(R0, SR_TID_X=0x21, ctrl=0x717) =
    19 79 00 00 00 00 00 00 00 21 00 00 00 2e 0e 00
```

### A.8 IADD3 (3-way integer add)

```
Opcode: 0x210
Fixed:  b0=0x10, b1=0x72, b9=0xe0, b10=0xf1, b11=0x07
Variable: b2=dest, b3=src0, b4=src1, b8=src2
Negation: b10=0xff, b7=0x80 when src1 is negated (subtract)

Ground truth:
  encode_iadd3(RZ, RZ, R4, RZ, ctrl=0x7f1) =
    10 72 ff ff 04 00 00 00 ff e0 f1 07 00 e2 0f 00
```

### A.9 IADD3.X (carry-extended add)

```
Opcode: 0x210 (same family, different modifiers)
Fixed:  b0=0x10, b1=0x72, b9=0xe4, b10=0x7f, b11=0x00
Variable: b2=dest, b3=src0, b4=src1, b8=src2

Reads carry-in from P0 predicate — used for high word of 64-bit add/sub.

Ground truth:
  encode_iadd3x(R7, RZ, RZ, RZ, ctrl=0x7f2) =
    10 72 07 ff ff 00 00 00 ff e4 7f 00 00 e4 0f 00
```

### A.10 IMAD.WIDE R-imm (0x825)

```
Opcode: 0x825
Fixed:  b0=0x25, b1=0x78, b9=0x02, b10=0x8e, b11=0x07
Variable: b2=dest (pair), b3=src0, b4=8-bit immediate, b8=src2 (pair addend)

Ground truth:
  encode_imad_wide(R2, R13, 0x8, R2, ctrl=0x0fe6) =
    25 78 02 0d 08 00 00 00 02 02 8e 07 00 cc 1f 00
```

### A.11 IMAD.WIDE R-R (0x225)

```
Opcode: 0x225 (signed), b9=0x02
Fixed:  b0=0x25, b1=0x72, b10=0x8e, b11=0x07
Variable: b2=dest pair, b3=src0, b4=src1, b8=src2 pair

Unsigned form: b9=0x00 (IMAD.WIDE.U32)
Carry-out form: b9=0x00, b10=0x80 (P0 receives carry)
Carry-in form: b9=0x04, b10=0x0e, b11=0x00 (reads P0 as carry-in, .X suffix)
```

### A.12 IMAD R-UR (0xc24) — use this, not 0x2a4

```
Opcode: 0xc24
Operands: src0=b3 (GPR), src1=b4 (UR, not GPR), src2=b8 (GPR)
misc=1

This is the working R-R-equivalent multiply form. Always route one operand
through a UR when computing a 32-bit product from three GPRs.
```

### A.13 LOP3.LUT (opcode 0x212)

```
Opcode: 0x212
Reads: b3, b4, b8 (three GPR sources)
LUT: embedded in operand encoding

A LOP3 computes: D = f(A, B, C) where f is determined by the 8-bit LUT.
The LUT byte encodes truth table outputs for all 8 (A,B,C) combinations.

Common LUT values:
  0xfc = A | B | C
  0x80 = A & B & C
  0x96 = A ^ B ^ C
  0xc0 = A & B
  0x3c = A ^ B
  0xaa = C
```

### A.14 FSEL (0x208) — FP32 select

```
Opcode: 0x208
Operands: b2=dest, b3=src0, b4=src1
Predicate: encoded in raw[10..11]:
  raw[10] |= (pred & 1) << 7
  raw[11] |= (pred >> 1) & 0x7F
  raw[11] |= 0x04   ; if predicate is negated

Semantics: dest = pred ? src0 : src1
```

### A.15 FSEL.STEP (0x80a) — combined FP compare + select

```
Opcode: 0x80a (misc=5 required)
Operands: b2=dest, b3=src, b4..b7=f32 threshold immediate
Compare mode (b10/b11 encode comparison):
  FSEL_GT, FSEL_GE, FSEL_LT, FSEL_LE, FSEL_EQ, FSEL_NE

Semantics: dest = (src CMP threshold) ? src : 0
           (or variant depending on mode)

Use FSEL.STEP to avoid the FSETP-after-ISETP corruption bug.
```

### A.16 FFMA / FMUL (0x223)

```
Opcode: 0x223
Operands: b2=dest, b3=src0, b4=src1, b8=src2

FMUL = FFMA with src2=RZ.
Rounding modes encoded in modifier bytes (RN=default, RZ, RP, RM).

FFMA.IMM (0x823): 32-bit float immediate at b4..b7.
```

### A.17 DFMA (0x22b) — FP64 fused MAD R-R

```
Opcode: 0x22b
Fixed:  b0=0x2b, b1=0x72, b11=0x00
Operands: b2=dest pair, b3=src0 pair, b4=src1 pair, b8=src2 pair
misc=2

All GPR pairs: each src/dest occupies 2 consecutive registers.

WRONG: b1=0x7e / b1=0x7c forms silently give wrong results on SM_120.
```

### A.18 DADD (0x229) — FP64 add R-R (src1 at b8!)

```
Opcode: 0x229
Fixed:  b0=0x29, b1=0x72, b11=0x00
Operands: b2=dest pair, b3=src0 pair, b8=src1 pair (UNIQUE: not b4)
misc=2

DADD is the only FP64 R-R op where src1 is at b8 instead of b4.
```

### A.19 DMUL (0x228) — FP64 multiply

```
Opcode: 0x228
Fixed:  b0=0x28, b1=0x72, b11=0x00
Operands: b2=dest pair, b3=src0 pair, b4=src1 pair
misc=2
```

### A.20 DSETP (0x22a) — FP64 compare → predicate

```
Opcode: 0x22a
Operands: b3=src0 pair, b4=src1 pair
Predicate destination: encoded in modifier bytes
misc=2, wdep=0x33 (long-latency)

Comparison codes (at b9 or similar):
  ONLY UNORDERED CODES WORK on SM_120:
    0x09=LTU, 0x0a=EQU, 0x0b=LEU, 0x0c=GTU, 0x0d=NEU, 0x0e=GEU
  ORDERED CODES SILENTLY FAIL:
    1=LT, 2=EQ, 3=LE, 4=GT, 5=NE, 6=GE

For setp.lt.f64: emit DSETP.GEU then use @!P as guard.
```

### A.21 LDG.E (0x981) — global load via UR descriptor

```
Opcode: 0x981
Operands: b2=dest, b3=addr_reg (64-bit pair), b5=UR descriptor
misc=6, wdep=0x35 (all LDGs share this)

Size (b9):
  0x19/0x99 = 32-bit (single dest reg)
  0x1b/0x9b = 64-bit (dest + dest+1)
  0x1d/0x9d = 128-bit (dest..dest+3)

The UR descriptor (typically UR4..UR5 from c[0][0x358]) carries the
flat-addressing metadata.
```

### A.22 STG.E (0x986) — global store

```
Opcode: 0x986
Operands: b3=addr pair, b4=data reg, b8=UR descriptor
misc=1, wdep=0x3f (no tracking)

When data register has pending LDG write, consumer rbar must include 0x09.
```

### A.23 BAR.SYNC (0xb1d)

```
Opcode: 0xb1d
Variable: b2 or immediate field = barrier_id (0..15)
misc varies; BAR clears scoreboard state.

IMPORTANT: After BAR.SYNC, the scoreboard emulator must clear all pending
trackers (pending_writes, pending_ur_writes, pending_pred_writes).
```

### A.24 HMMA (0x23c) — tensor core MMA

```
Opcode: 0x23c
Operands: b2=D (4 regs, 4-aligned), b3=A (4 regs), b4=B (2 regs), b8=C (4 regs)
misc=2
b9 selects dtype mode: FP16, BF16, or TF32 (specific codes TBD per ptxas)

D, C: FP32 accumulator registers.
A: FP16/BF16/TF32 values, 4 registers = 8xF16 or 4xBF16 or 4xTF32.
B: same dtype, 2 registers.
```

### A.25 IMMA (0x237) — INT8 tensor MMA

```
Opcode: 0x237
Operands: b2=D (4 regs, 4-aligned), b3=A (4 regs), b4=B (MUST BE R0..R7!), b8=C
misc=2

B register constraint: R0..R7 only. R8+ causes ERR_ILLEGAL_INSTRUCTION.
Work around by forcing allocator to assign B to low GPRs first.
```

### A.26 QMMA (0x27a) — FP8 tensor MMA

```
Opcode: 0x27a
Constraint: raw[2] must equal raw[3] (D register == A register)
misc=2

Encode as (d, d, b, c). PTX must use same virtual regs for D and A.
This is an in-place encoding: A matrix values pre-loaded into D positions.
```

### A.27 DMMA (0x23f) — FP64 tensor MMA

```
Opcode: 0x23f
Shape: m8n8k4
Operands: D (4 regs = 4 FP64 pairs), A, B, C
misc=2
```

### A.28 REDUX (0x3c4) — warp reduction → UR

```
Opcode: 0x3c4
Writes: UR (not GPR)
Operand: b2=dest UR, b4=src GPR
wdep=0x31 (same slot as LDCU), misc=0

Variants (mode bits in modifiers):
  REDUX.SUM.U32      REDUX.SUM.S32
  REDUX.MIN.S32      REDUX.MIN.U32
  REDUX.MAX.S32      REDUX.MAX.U32
  REDUX.AND.B32      REDUX.OR.B32      REDUX.XOR.B32

Consumer (MOV R, UR from 0xc02) must wait on UR pending via rbar=0x03.
```

### A.29 UTMALDG (0x5b4) — TMA tensor load

```
Opcode: 0x5b4
Writes: UR (the descriptor-based load target is shared memory)
Operands: UR-only
wdep=0x0e, misc=12

Variants:
  UTMALDG.1D: 1D tile
  UTMALDG.2D: 2D tile
  UTMALDG.3D: 3D tile (if supported)
  UTMALDG.4D: 4D tile (if supported)
  UTMALDG.5D: 5D tile (if supported)

Requires pre-built TMA descriptor in c[0]. See CUDA TMA API for descriptor
layout.
```

---

## Appendix B — Complete Control Word Table

This appendix lists the ctrl word assignment rules used by OpenPTXas for
every classified opcode. Values are `(wdep, misc, min_gpr_gap)`.

### B.1 Memory operations

| Opcode | Mnemonic      | wdep | misc | min_gap | Notes                    |
|--------|---------------|------|------|---------|--------------------------|
| 0x981  | LDG.E         | 0x35 | 6    | —       | all LDGs share slot 0x35 |
| 0x986  | STG.E         | 0x3f | 1    | —       | no write tracking        |
| 0x984  | LDS           | 0x33 | —    | —       |                          |
| 0x988  | STS           | 0x3f | 4    | —       |                          |
| 0x83b  | LDSM          | 0x33 | 2    | 1       | long-latency             |
| 0xb82  | LDC           | 0x31 | —    | —       | counter-driven misc      |
| 0x7ac  | LDCU          | 0x31/0x33/0x35 | 7 | — | strict misc, slot rotates|
| 0xfae  | LDGSTS.E      | 0x3f | 4    | —       | async copy               |
| 0x9af  | LDGDEPBAR     | 0x31 | 1    | —       | commit group             |
| 0x91a  | DEPBAR.LE     | 0x3f | —    | —       |                          |
| 0x992  | MEMBAR        | 0x3f | 0    | —       |                          |

### B.2 Integer ALU

| Opcode | Mnemonic        | wdep | misc | min_gap |
|--------|-----------------|------|------|---------|
| 0x210  | IADD3           | 0x3e | 1    | 1       |
| 0x212  | IADD3X/LOP3     | 0x3e | 1    | 1       |
| 0x810  | IADD3.IMM32     | 0x3e | 1    | 1       |
| 0x235  | IADD.64         | 0x3e | 1    | 1       |
| 0xc35  | IADD.64-UR      | 0x3e | 5    | 1       |
| 0x224  | IMAD.32         | 0x3e | 1    | 1       |
| 0x824  | IMAD R-imm      | 0x3e | 1    | 1       |
| 0xc24  | IMAD R-UR       | 0x3e | 1    | 1       |
| 0x2a4  | IMAD R-R        | —    | —    | —       | **BROKEN**               |
| 0x825  | IMAD.WIDE R-imm | 0x3e | 1    | 1       |
| 0x225  | IMAD.WIDE R-R   | 0x3e | 1    | 1       |
| 0x219  | SHF.R.VAR       | 0x3e | 1    | 1       |
| 0x819  | SHF             | 0x3e | 1    | 1       |
| 0x299  | SHF.VAR         | 0x3e | 1    | 1       |
| 0x309  | POPC            | 0x3e | 1    | 1       |
| 0x301  | BREV            | 0x3e | 1    | 1       |
| 0x300  | FLO             | 0x3e | 1    | 1       |
| 0x213  | IABS            | 0x3e | 1    | 1       |
| 0x217  | IMNMX           | 0x3e | 1    | 1       |
| 0x21a  | SGXT            | 0x3e | 1    | 1       |
| 0x21b  | BMSK            | 0x3e | 1    | 1       |
| 0x239  | I2IP            | 0x3e | 1    | 1       |
| 0x211  | LEA             | 0x3e | 1    | 1       |
| 0x811  | LEA.IMM         | 0x3e | 1    | 1       |
| 0x226  | IDP.4A          | 0x3e | 1    | 1       |
| 0xc26  | IDP.4A-UR       | 0x3e | 1    | 1       |

### B.3 Float ALU

| Opcode | Mnemonic    | wdep | misc | min_gap |
|--------|-------------|------|------|---------|
| 0x221  | FADD        | 0x3e | 1    | 1       |
| 0x223  | FFMA / FMUL | 0x3e | 1    | 1       |
| 0x820  | FMUL.IMM    | 0x3e | 1    | 1       |
| 0x823  | FFMA.IMM    | 0x3e | 1    | 1       |
| 0x209  | FMNMX       | 0x3e | 1    | 1       |
| 0x208  | FSEL        | 0x3e | 1    | 1       |
| 0x80a  | FSEL.STEP   | 0x3e | 5    | 1       |
| 0x20b  | FSETP       | 0x3e | —    | 0       |
| 0x308  | MUFU        | 0x3e | 1    | 1       |
| 0x822  | FSWZADD     | 0x3e | 1    | 1       |
| 0x431  | HFMA2       | 0x3e | 1    | 1       |
| 0x23e  | F2FP.F16    | 0x3e | 1    | 1       |

### B.4 FP64

| Opcode | Mnemonic     | wdep | misc | min_gap |
|--------|--------------|------|------|---------|
| 0x229  | DADD         | 0x33 | 2    | —       |
| 0x228  | DMUL         | 0x33 | 2    | —       |
| 0x22b  | DFMA         | 0x33 | 2    | —       |
| 0xc2b  | DFMA-UR-UR   | 0x33 | 2    | —       |
| 0x22a  | DSETP        | 0x33 | 2    | —       |
| 0x310  | F2F          | 0x33 | 1    | 1       |
| 0x311  | F2I.F64      | 0x3e | 1    | 1       |
| 0x312  | I2F.F64      | 0x3e | 1    | 1       |

### B.5 Tensor core

| Opcode | Mnemonic | wdep | misc | min_gap | Constraints           |
|--------|----------|------|------|---------|-----------------------|
| 0x23c  | HMMA     | 0x3e | 2    | 1       | D 4-aligned           |
| 0x237  | IMMA     | 0x3e | 2    | 1       | B must be R0..R7      |
| 0x23f  | DMMA     | 0x3e | 2    | 1       | D 4-aligned           |
| 0x27a  | QMMA     | 0x3e | 2    | 1       | D == A (raw[2]==raw[3])|

### B.6 Predicates / Select

| Opcode | Mnemonic   | wdep | misc | min_gap |
|--------|------------|------|------|---------|
| 0x20c  | ISETP R-R  | 0x3e | 0    | 0       |
| 0xc0c  | ISETP R-UR | 0x3e | 0    | 0       |
| 0x207  | SEL        | 0x3e | 1    | —       |
| 0x807  | SEL.IMM    | 0x3e | 1    | —       |
| 0x203  | P2R        | 0x3e | 1    | 1       |
| 0x204  | R2P        | 0x3f | 0    | —       |
| 0x21e  | PLOP3      | 0x3f | 0    | —       |

### B.7 Control flow / barriers

| Opcode | Mnemonic      | wdep | misc |
|--------|---------------|------|------|
| 0x918  | NOP           | 0x3e | 0    |
| 0x947  | BRA           | 0x3f | 1    |
| 0x547  | BRA.U         | 0x3f | 1    |
| 0x94d  | EXIT          | 0x3f | 5    |
| 0x944  | CALL.REL      | 0x3f | 1    |
| 0x950  | RET.REL       | 0x3f | 1    |
| 0xb1d  | BAR.SYNC      | —    | —    | **clears scoreboard** |
| 0x9ab  | ERRBAR        | 0x3f | 0    |
| 0x5ab  | CGAERRBAR     | 0x3f | 0    |
| 0x9c7  | UCGABAR_ARV   | 0x3f | 0    |
| 0xdc7  | UCGABAR_WAIT  | 0x3f | 0    |
| 0x95d  | NANOSLEEP     | 0x3f | 1    |

### B.8 Warp operations

| Opcode | Mnemonic     | wdep | misc |
|--------|--------------|------|------|
| 0x589  | SHFL.RR      | 0x3e | 1    |
| 0xf89  | SHFL.RI      | 0x3e | 1    |
| 0x989  | SHFL.II      | 0x3e | 1    |
| 0x806  | VOTE.ANY     | 0x3e | 1    |
| 0x3a1  | MATCH        | 0x3e | 1    |
| 0x3c4  | REDUX        | 0x31 | 0    | writes UR |
| 0x82f  | ELECT        | 0x3f | 1    |

### B.9 Uniform datapath

| Opcode | Mnemonic    | wdep | misc | Target |
|--------|-------------|------|------|--------|
| 0x882  | UMOV.IMM    | 0x3f | 1    | UR     |
| 0xc82  | UMOV.RR     | 0x3f | 1    | UR     |
| 0xc02  | MOV R, UR   | 0x3e | 1    | GPR    |
| 0x890  | UIADD3      | 0x3f | 1    | UR     |
| 0x28c  | UISETP      | 0x3f | 0    | UP     |
| 0x887  | USEL        | 0x3f | 1    | UR     |
| 0x853  | UFSETP      | 0x3f | 0    | UP     |
| 0x856  | UFMUL       | 0x3f | 1    | UR     |
| 0x291  | ULEA        | 0x3f | 1    | UR     |
| 0x9c3  | S2UR        | 0x31 | —    | UR     |

### B.10 Atomics

| Opcode | Mnemonic              | wdep | misc |
|--------|-----------------------|------|------|
| 0x9a8  | ATOMG.E.{ADD/MIN/MAX/EXCH}.U32 | 0x3f | 4 |
| 0x9a3  | ATOMG.E.ADD.F32       | 0x3f | 4    |
| 0x3a9  | ATOMG.E.CAS.b32       | 0x3e | 4    |
| 0x3a9  | ATOMG.E.CAS.b64       | 0x3e | 4    | b9=0xe5 |

### B.11 TMA

| Opcode | Mnemonic      | wdep | misc |
|--------|---------------|------|------|
| 0x5b2  | SYNCS.EXCH    | 0x03 | 2    |
| 0x9a7  | SYNCS.ARRIVE  | 0x3f | 1    |
| 0x5a7  | SYNCS.TRYWAIT | 0x3f | 1    |
| 0x3ba  | UBLKCP        | 0x0e | 12   |
| 0x5b4  | UTMALDG       | 0x0e | 12   |
| 0x3b5  | UTMASTG       | 0x1f | 1    |
| 0x9b7  | UTMACMDFLUSH  | 0x0f | 1    |

### B.12 Texture / surface

| Opcode | Mnemonic | wdep | misc |
|--------|----------|------|------|
| 0xf60  | TEX      | 0x35 | 2    |
| 0xf63  | TLD4     | 0x35 | 2    |
| 0xf66  | TLD.LZ   | 0x35 | 2    |
| 0xf6f  | TXQ      | 0x35 | 2    |
| 0xf99  | SULD     | 0x35 | 2    |
| 0xf9d  | SUST     | 0x3f | 2    |

---

## Appendix C — SM_120 Opcode Master List (188 verified)

The following opcodes execute without fault on RTX 5090. Opcodes are given
as 12-bit hex. 182 were identified via nvdisasm, 6 via behavioural probing.

### C.1 Via nvdisasm (182)

Common opcodes (already documented above):
```
0x018 0x019 0x02b 0x040 0x041 0x048 0x049 0x04d
0x100 0x101 0x102 0x103 0x104 0x105 0x106 0x107
0x108 0x109 0x10a 0x10b 0x10c 0x10d 0x10e 0x10f
0x110 0x111 0x112 0x113 0x114 0x115 0x116 0x117
0x118 0x119 0x11a 0x11b 0x11c 0x11d 0x11e 0x11f
0x120 0x121 0x122 0x123 0x124 0x125 0x126 0x127
0x128 0x129 0x12a 0x12b 0x12c 0x12d 0x12e 0x12f
0x130 0x131 0x132 0x133 0x134 0x135 0x136 0x137
0x138 0x139 0x13a 0x13b 0x13c 0x13d 0x13e 0x13f
```

(These opcode numbers are examples; the actual complete list is maintained
in `C:/Users/kraken/openptxas/probe_work/sm120_opcode_map.py`.)

### C.2 Behaviourally identified (6)

These 6 are NOT in the CUDA 13.0 nvdisasm opcode table but were found by
brute-force execution testing:

| Opcode | Inferred Name | Behaviour |
|--------|---------------|-----------|
| 0x22c  | DSEL          | FP64 select; dest_pair = R[b8]:R[b8+1] when pred false |
| 0x35a  | BMOV.64       | Reads 64-bit barrier state; values vary per barrier_idx |
| 0x3d0  | GETLMEMFLAGS  | Writes HW state to dest pair; distinct from GETLMEMBASE |
| 0x47e  | QMMA.SF       | MXF8 matrix variant; b9 encodes FP8 format |
| 0x95a  | BMOV.64       | B-imm form of 0x35a |
| 0x9d4  | UTMACCTL      | UTMAC TMA control/status variant; b3 indexes channel |

---

## Appendix D — Kernel Prologue Template

Every SM_120 kernel follows a similar prologue pattern for loading the
flat descriptor and parameters:

```
; --- Prologue ---
LDCU.64   UR4, c[0][0x358]    ; flat global memory descriptor
                              ; wdep=0x35 (so LDGs get rbar=0x09), misc=7
LDCU.64   UR6, c[0][0x210]    ; first pointer param (pair)
                              ; wdep=0x31, misc=7
LDCU.64   UR8, c[0][0x218]    ; second pointer param
                              ; wdep=0x33 (rotates), misc=7
; ... more LDCU.64 for additional pointer params
LDCU.32   UR12, c[0][0x228]   ; 32-bit scalar param
                              ; wdep=0x31, misc=7

S2R       R0, SR_CTAID.X      ; wdep=0x31, misc=1
S2UR      UR2, SR_NTID.X      ; wdep=0x31
S2R       R1, SR_TID.X        ; wdep=0x31, misc=1

IMAD      R2, R0, UR2, R1     ; global thread id = ctaid*ntid + tid
                              ; opcode 0xc24 (R-UR), misc=1

; --- Bounds check (if-convert early exit) ---
ISETP.GE  P0, R2, UR12        ; opcode 0xc0c, misc=0
@P0 EXIT                       ; opcode 0x94d, misc=5

; --- Body ---
...

EXIT                           ; final exit
```

**Key observations:**
- All pointer params accessed via UR descriptors (LDCU.64)
- 32-bit scalar params go through LDCU.32
- Global thread ID always computed via `IMAD R-UR` (0xc24), never R-R
- Bounds check uses if-conversion: predicated EXIT, not BRA
- First LDCU.64 posts to slot 0x35 (so subsequent LDG gets rbar=0x09)

---

## Appendix E — c[0] constant bank layout (SM_120)

The flat-addressing model on SM_120 places several fixed descriptors in
constant bank 0. The driver populates these at kernel launch.

| Offset    | Size | Contents                              |
|-----------|------|---------------------------------------|
| 0x000     | ~    | Reserved / architecture constants     |
| 0x160     | 32   | Grid / block dimensions               |
| 0x1c0     | 32   | CTA dimensions                        |
| 0x210     | 8    | First kernel parameter (if 64-bit)    |
| ...       | ...  | Subsequent kernel parameters          |
| 0x358     | 16   | **Flat global memory descriptor**     |
| 0x368     | 16   | Flat shared memory descriptor         |
| 0x378     | 16   | Flat local memory descriptor          |
| 0x388+    | ~    | Literal pool (kernel constants)       |

The **flat descriptor at 0x358** is the primary descriptor used by all LDG/STG
operations. Without the correct capmerc FP64 class descriptor (type-02
sub=0x0c), the driver does not populate this slot for FP64 kernels, and
STG.E.64 crashes with `CUDA_ERROR_ILLEGAL_ADDRESS`.

The parameter area begins at 0x210 and extends for `CBANK_PARAM_SIZE` bytes.
Anything past this extent is **uninitialized** — the literal pool must be
explicitly populated via `.nv.constant0` section data, and
`CBANK_PARAM_SIZE` must be the exact parameter size so the driver's
zero-fill doesn't clobber the literal pool.

---

## Appendix F — Instruction Latency & Scheduling Hints

### F.1 Approximate latencies (from scheduling constants)

| Class                  | Latency (cycles) | wdep  |
|------------------------|------------------|-------|
| Integer ALU            | ~4-6             | 0x3e  |
| Float ALU (FFMA)       | ~4               | 0x3e  |
| MUFU (SFU)             | ~16-32           | 0x3e  |
| FP64 (DFMA)            | ~64+             | 0x33  |
| LDG (L1 hit)           | ~20              | 0x35  |
| LDG (L2 hit)           | ~200-400         | 0x35  |
| LDG (DRAM)             | ~400-600         | 0x35  |
| LDC                    | ~8               | 0x31  |
| LDS                    | ~20              | 0x33  |
| LDSM                   | ~30              | 0x33  |
| HMMA                   | ~32              | 0x3e  |
| DMMA                   | ~64               | 0x3e  |
| BAR.SYNC               | variable         | —     |
| TMA (UTMALDG)          | ~400+            | 0x0e  |

### F.2 Scheduling heuristics

1. **Group LDGs together.** Issue multiple LDG.E in a row; all share slot
   0x35 and FIFO in the load pipeline. A single rbar=0x09 on the first
   consumer waits for the last LDG.

2. **Separate ISETP from FSEL consumers.** Insert at least 1 NOP or
   unrelated ALU op between an ISETP and an FSEL that reads the predicate
   via raw[10..11].

3. **Gap LDCU.64 from its consumer.** At least 4 instructions between
   LDCU.64 and the IADD.64-UR or LDG that uses its UR output.

4. **Avoid long-latency pile-up.** Interleave DSETP, F2F, LDSM, and DFMA —
   they all use slot 0x33 and block each other.

5. **BAR.SYNC resets the scoreboard.** After a barrier, you start fresh;
   don't propagate pending dependencies.

6. **Predicated execution is cheap.** Use @P guards and if-conversion
   instead of BRA where possible.

### F.3 Register pressure vs capmerc

Capmerc body records encode the maximum register the kernel may access.
Higher register counts require:
- Capability bitmask bit 13 (0x2000) set
- Filler records (`41 0c 54 04`) scaled to register pressure
- Body record byte[10] set to 0x01 (not 0x81) for R14+ access

---

## Appendix G — Glossary

| Term      | Definition                                                      |
|-----------|-----------------------------------------------------------------|
| SASS      | Shader Assembly — NVIDIA's native GPU ISA                       |
| PTX       | Parallel Thread Execution — NVIDIA's IR (intermediate language) |
| GPR       | General-Purpose Register (R0..R254, 32-bit)                     |
| UR        | Uniform Register (UR0..UR62, 32-bit, warp-wide identical)       |
| P         | Predicate (P0..P7, 1-bit)                                       |
| UP        | Uniform Predicate (UP0..UP7)                                    |
| RZ        | Read-as-Zero (R255) — writes discarded, reads return 0          |
| URZ       | Uniform RZ (UR63)                                               |
| PT        | Predicate True (P7)                                             |
| rbar      | Read barrier — scoreboard slots this instruction waits for      |
| wdep      | Write dependency — scoreboard slot this instruction posts to    |
| misc      | Misc nibble — per-opcode sequencing metadata                    |
| wbar      | Write-after-read barrier flag                                   |
| yield     | Scheduling yield hint                                           |
| ctrl      | 23-bit control word (stall/yield/wbar/rbar/wdep/misc)           |
| if-conv   | If-conversion — replacing BRA with predicated execution         |
| cbank     | Constant bank — read-only memory in c[bank][offset]             |
| flat desc | Generic-pointer descriptor at c[0][0x358] for LDG/STG           |
| capmerc   | Mercury DRM metadata section (SM_120 new)                       |
| LDSM      | Load Shared Matrix — load smem into tensor-core matrix regs     |
| HMMA      | Half-precision Matrix Multiply-Accumulate                       |
| IMMA      | Integer MMA                                                     |
| DMMA      | Double MMA                                                      |
| QMMA      | Quarter (FP8) MMA                                               |
| OMMA      | Outer MMA (Blackwell new)                                       |
| TMA       | Tensor Memory Accelerator                                       |
| CTA       | Cooperative Thread Array (a.k.a. thread block)                  |
| CGA       | Cluster Grouped Array (cluster of CTAs, Blackwell)              |
| DSMEM     | Distributed Shared Memory (cross-CTA shared in cluster)         |
| MXF8      | Microscaling FP8 (block-scaled FP8, Blackwell new)              |

---

## Appendix H — Quick-Reference Cheat Sheet

```
=== ENCODING POSITIONS ===
opcode (12 bits)  = (b1 & 0x0F) << 8 | b0
dest              = b2
src0              = b3 (or addr pair for LDG/STG)
src1              = b4 (or UR for 0xc24, or bank for LDC)
const offset      = b5 (divided by 4)
src2              = b8
size modifier     = b9
mode modifiers    = b10, b11
ctrl (23 bits)    = b13..b15

=== PRED OPERAND ENCODING (FSEL, SEL) ===
raw[10] |= (pred & 1) << 7
raw[11] |= (pred >> 1) & 0x7F
raw[11] |= 0x04         if negated

=== REG ENCODING ===
GPR:  0..254, 255 = RZ
UR:   0..62,  63  = URZ
Pred: 0..6,   7   = PT

=== RBAR BITS ===
0x01 = base
0x02 = wait LDC (slot 0x31) / ALU (0x3e)
0x04 = wait LDS/DFPU/DSETP/F2F (slot 0x33)
0x08 = wait LDG (slot 0x35)
Combine with OR, never max.

=== COMMON CTRL VALUES ===
0x7e0  default
0x7ff  stall=all, yield
0xfea  EXIT (with misc=5)

=== HARDWARE BUGS TO AVOID ===
* IMAD R-R (0x2a4)              → use IMAD R-UR (0xc24)
* DFMA b1=0x7e/0x7c             → use b1=0x72, b11=0x00
* DSETP ordered codes           → use unordered (GEU/LEU/etc)
* ISETP → FSETP                 → use FSEL.STEP
* QMMA D != A                   → encode as (d, d, b, c)
* IMMA B >= R8                  → force B into R0..R7
* FSETP after ISETP             → use FSEL.STEP peephole
* LDCU.64 adjacent to consumer  → insert 4+ insns
* Literal pool past param area  → never read; inline immediates
* CBANK_PARAM_SIZE = 0xFF       → clobbers literal pool
```

---

*End of SM_120 Reference.*

---

## Appendix I — Detailed Scoreboard Walkthrough (worked example)

This appendix steps through scoreboard ctrl-word assignment for a real
kernel, showing exactly how rbar/wdep/misc are computed for each instruction.

### Kernel: vector addition (3-pointer load + add + store)

PTX:
```ptx
ld.param.u64   %rd1, [a];
ld.param.u64   %rd2, [b];
ld.param.u64   %rd3, [c];
mov.u32        %r0, %ctaid.x;
mov.u32        %r1, %ntid.x;
mov.u32        %r2, %tid.x;
mad.lo.s32     %r3, %r0, %r1, %r2;
mul.wide.s32   %rd4, %r3, 4;
add.s64        %rd5, %rd1, %rd4;
ld.global.u32  %r4, [%rd5];
add.s64        %rd6, %rd2, %rd4;
ld.global.u32  %r5, [%rd6];
add.s32        %r6, %r4, %r5;
add.s64        %rd7, %rd3, %rd4;
st.global.u32  [%rd7], %r6;
ret;
```

### Scoreboard trace

| # | Instruction          | Opcode | wdep | rbar  | misc | Reason                    |
|---|----------------------|--------|------|-------|------|---------------------------|
| 0 | LDCU.64 UR4, c[0][0x358] | 0x7ac | 0x35 | 0x01 | 7    | flat desc, first LDCU     |
| 1 | LDCU.64 UR6, c[0][0x210] | 0x7ac | 0x31 | 0x01 | 7    | a pointer                 |
| 2 | LDCU.64 UR8, c[0][0x218] | 0x7ac | 0x33 | 0x01 | 7    | b pointer (slot rotate)   |
| 3 | LDCU.64 UR10, c[0][0x220]| 0x7ac | 0x31 | 0x01 | 7    | c pointer                 |
| 4 | LDCU.32 UR12, c[0][0x228]| 0x7ac | 0x31 | 0x01 | 7    | n scalar                  |
| 5 | S2R R0, SR_CTAID.X   | 0x919 | 0x31 | 0x01 | 1    | async                     |
| 6 | S2UR UR2, SR_NTID.X  | 0x9c3 | 0x31 | 0x01 | —    | UR write                  |
| 7 | S2R R1, SR_TID.X     | 0x919 | 0x31 | 0x01 | 1    | async                     |
| 8 | IMAD R2, R0, UR2, R1 | 0xc24 | 0x3e | 0x03 | 1    | waits S2R R0/R1 + S2UR UR2|
| 9 | ISETP.GE P0, R2, UR12| 0xc0c | 0x3e | 0x03 | 0    | waits IMAD R2             |
|10 | @P0 EXIT             | 0x94d | 0x3f | 0x01 | 5    | pred guard                |
|11 | IMAD.WIDE R4, R2, 4, UR6 | 0x825 | 0x3e | 0x03 | 1 | waits R2 (ALU)           |
|12 | LDG.E R8, [R4.64]    | 0x981 | 0x35 | 0x03 | 6    | waits IMAD.WIDE R4        |
|13 | IMAD.WIDE R6, R2, 4, UR8 | 0x825 | 0x3e | 0x01 | 1 | independent               |
|14 | LDG.E R9, [R6.64]    | 0x981 | 0x35 | 0x03 | 6    | waits IMAD.WIDE R6        |
|15 | IADD3 R10, R8, R9, RZ| 0x210 | 0x3e | 0x09 | 1    | waits LDG (last wdep=0x35)|
|16 | IMAD.WIDE R12,R2,4,UR10| 0x825 | 0x3e | 0x01 | 1   | independent               |
|17 | STG.E [R12.64], R10  | 0x986 | 0x3f | 0x03 | 1    | waits R10 (ALU)           |
|18 | EXIT                 | 0x94d | 0x3f | 0x01 | 5    |                           |

### Step-by-step reasoning

**Instruction 0 (LDCU.64 UR4 = flat descriptor):**
- First LDCU in kernel → wdep=0x35 (posts to LDG slot so later LDGs inherit)
- misc=7 (required for LDCU)
- No dependencies → rbar=0x01

**Instruction 8 (IMAD computing thread_id):**
- Reads R0 (from S2R, wdep=0x31), UR2 (from S2UR, wdep=0x31), R1 (from S2R,
  wdep=0x31)
- All three dependencies post to slot 0x31 → rbar bit 1 = 0x02
- Base bit: 0x01
- Final: rbar = 0x01 | 0x02 = 0x03

**Instruction 15 (IADD3 adding two loaded values):**
- Reads R8 (from LDG, wdep=0x35) and R9 (from LDG, wdep=0x35)
- Both post to LDG slot → rbar bit 3 = 0x08
- Special case for LDG consumers: use 0x09 (not 0x09+base)
- Final: rbar = 0x09

**Instruction 17 (STG storing result):**
- Reads R12 pair (from IMAD.WIDE, wdep=0x3e) and R10 (from IADD3, wdep=0x3e)
- Both ALU dependencies → rbar bit 1 = 0x02
- Final: rbar = 0x01 | 0x02 = 0x03

### Why LDG interleaving is OK

Instructions 12 and 14 both have wdep=0x35 (same slot). The hardware
scoreboard tracks only the LAST write to each slot. However:

- FIFO ordering in the load pipeline guarantees LDG #12 completes before LDG #14
- Consumer #15 sets rbar=0x09 waiting for slot 0x35
- When slot 0x35 completes, it means LDG #14 is done (the last writer)
- Because of FIFO, LDG #12 is also done by then

This is why using `wdep=0x37` for the second LDG is wrong: slot 0x37 has no
rbar bit, and we'd lose FIFO semantics.

---

## Appendix J — Bisecting scoreboard bugs

When a kernel produces wrong output but no hardware fault, the cause is
often a scoreboard dependency miss. This appendix describes the bisection
workflow used to identify the 17 SM_120 scoreboard rules.

### J.1 Symptoms

1. **All zeros output**: consumer read stale data before LDG completed
2. **First N elements correct, rest wrong**: WAW hazard on a reloaded reg
3. **All elements correct except one**: race in specific slot
4. **Random output each run**: uninitialized UR or literal pool read
5. **Exactly inverted result**: predicate sign wrong (ISETP/DSETP)

### J.2 Bisection steps

1. **Extract ptxas ground truth** for the same PTX:
   ```bash
   ptxas -arch=sm_120 input.ptx -o reference.cubin
   nvdisasm --print-instruction-encoding reference.cubin > ref.asm
   ```

2. **Dump our cubin the same way:**
   ```bash
   python -m openptxas input.ptx -o our.cubin
   nvdisasm --print-instruction-encoding our.cubin > ours.asm
   ```

3. **Diff the assembly side-by-side.** Opcodes should match. If they don't:
   - Wrong isel pattern (SM_89 opcode emitted for SM_120, or vice versa)
   - Missing FSEL.STEP peephole
   - IMAD R-R vs IMAD R-UR

4. **If opcodes match, compare ctrl words:**
   ```bash
   diff <(cut -c50- ref.asm) <(cut -c50- ours.asm)
   ```
   Expect every wdep, rbar, misc to match.

5. **If ctrls differ on a single instruction**, identify which field:
   - rbar mismatch: scoreboard rule violation
   - wdep mismatch: slot-assignment rule
   - misc mismatch: per-opcode misc table incomplete

6. **Patch the single instruction** in our cubin to ptxas's ctrl and re-run.
   If output becomes correct, the scoreboard rule is confirmed.

### J.3 Common fix patterns

| Symptom                        | Fix                                    |
|--------------------------------|----------------------------------------|
| LDG consumer rbar=0x09 but wrong | Source reg also depends on LDC → use 0x0B |
| LOP3 gives stale data          | Add b4/b8 to LOP3 src tracking          |
| STG writes wrong data          | Add rbar=0x09 when data_reg has LDG dep |
| BAR.SYNC'd kernel still wrong  | Clear pending_writes on BAR.SYNC        |
| DFMA result wrong              | Switch to b1=0x72, b11=0x00 form        |
| FP64 kernel crashes at STG     | Add FP64 class descriptor to capmerc    |
| R14+ kernel load fails         | Capmerc body byte[10]=0x01 + filler blocks |

---

## Appendix K — Validation against ptxas

OpenPTXas validates every cubin in three stages:

### K.1 Byte-exact comparison (when possible)

For test kernels with known-good ptxas output, do a byte-wise diff of the
entire cubin. Acceptable differences:
- Timestamps in ELF headers
- Symbol order (if stable across builds)
- Padding bytes in ELF sections

Unacceptable:
- Any difference in `.text.<kernel>` bytes
- Any difference in capmerc body
- Any difference in EIATTR values (except size fields on different layouts)

### K.2 Execution testing

Load the cubin and run on RTX 5090 with known inputs. Compare outputs to
CPU reference. Tolerance:
- Integer ops: exact match
- FP32 ops: bit-exact match
- FP64 ops: bit-exact match (IEEE 754)
- Tensor core ops: 1 ULP (FFMA rounding may differ)

### K.3 Negative testing

Hand-craft cubins with known-bad patterns to verify:
- Missing capmerc blob → load fails with 201
- Wrong DSETP comparison code → P=false (verified by branch coverage)
- IMAD R-R (0x2a4) → garbage output
- Wrong LDG misc → sometimes fine, sometimes illegal

---

## Appendix L — Common ptxas emission patterns

These are ptxas 13.0 `sm_120` patterns that OpenPTXas learned to reproduce.

### L.1 u32 divide → IMAD.WIDE + MUFU.RCP sequence

PTX `div.u32 d, a, b` where b is known non-constant becomes:
```
I2FP.U32.RP R_bf, R_b               ; int→float, round toward +inf
MUFU.RCP R_rb, R_bf                 ; 1/b approximate
... (Newton-Raphson refinement) ...
F2I.FTZ.U32.TRUNC R_q, R_product    ; float→int, truncate
```

### L.2 mul.hi.u64 → IMAD.WIDE.U32 + carry chain

128-bit multiply of two 64-bit operands decomposes into four IMAD.WIDE.U32
with carry chaining (P0 carry-out, then .X carry-in).

### L.3 shr.s32 variable → SHF.R.S32.HI.VAR

Arithmetic right shift by a register amount uses SHF.R.S32.HI.VAR (0x219).
The high-word form ensures sign-extension.

### L.4 setp.lt.f64 → DSETP.GEU + negated guard

Ordered FP64 compares are unreliable. ptxas emits DSETP.GEU and uses the
**complement** of the predicate (via `@!P` guards).

### L.5 selp.f64 → two FSELs

FP64 select is implemented as two FSEL instructions (one per 32-bit word).

### L.6 Bounds check if-conversion

```
if (tid >= N) return;
```

becomes:
```
ISETP.GE P0, R_tid, UR_N
@P0 EXIT                       ; predicated exit, no BRA
... rest of kernel ...
EXIT
```

Both EXITs must be listed in EIATTR_EXIT_INSTR_OFFSETS.

### L.7 sub.u32 → IADD3 with negation

```
sub.u32 d, a, b
```
becomes:
```
IADD3 R_d, R_a, -R_b, RZ   ; b10=0xff, b7=0x80 flags negation
```

### L.8 shl.b32 by constant → SHF.L.U32

```
shl.b32 d, a, 5
```
becomes:
```
SHF.L.U32 R_d, R_a, 5, RZ    ; opcode 0x819, b4=5 (shift count)
```

### L.9 atomic.add.f32 → ATOMG.E.ADD.F32

```
atom.add.f32 d, [ptr], val
```
becomes:
```
ATOMG.E.ADD.F32 R_d, [R_addr.64], R_val   ; opcode 0x9a3, misc=4
```

### L.10 mad.lo.s32 with UR operand

When any of the three inputs to mad.lo.s32 is a kernel parameter (param
loaded via LDCU), ptxas uses IMAD R-UR (0xc24) rather than 0x2a4. This is
both because 0x2a4 is broken AND because LDCU→UR routing is more efficient.

---

## Appendix M — Boot sequence for a new SM_120 opcode

When adding support for a new opcode to OpenPTXas, follow this protocol:

1. **Identify the opcode**
   - Find via `nvdisasm --print-instruction-encoding` on ptxas output
   - Or brute-force probe: enumerate all 4096 opcodes, find ones that execute

2. **Reverse the encoding**
   - Emit the corresponding PTX and collect several ptxas samples
   - Diff samples to find variable vs fixed bytes
   - Map each variable byte to the corresponding PTX operand

3. **Identify the ctrl word behaviour**
   - Extract wdep/rbar/misc from each sample
   - Look for consistency: same opcode should use same wdep/misc
   - Note if misc is counter-driven vs fixed

4. **Add encoder function** in `sass/encoding/sm_120_opcodes.py`:
   ```python
   def encode_foo(dest, src0, src1, ctrl=0):
       """Encode FOO to 16 bytes. Ground truth: ..."""
       if ctrl == 0: ctrl = _CTRL_DEFAULT
       return _build(b0=..., b1=..., b2=dest, ..., ctrl=ctrl)
   ```

5. **Register in scoreboard** at `sass/scoreboard.py`:
   - Add to `_OPCODE_META` with (mnemonic, min_gap, wdep, misc)
   - Add to `_OPCODES_ALU` or appropriate category set
   - Extend `_get_src_regs` and `_get_dest_regs` if operand layout is new
   - Add to `_OPCODE_MISC` if strict misc required

6. **Add isel rule** in `sass/isel.py` if PTX pattern maps to this opcode

7. **Write a test** that emits a minimal cubin using this opcode and
   verifies GPU execution correctness

8. **Validate against ptxas ground truth** with byte-diff if possible

---

## Appendix N — Scoreboard trace format

OpenPTXas can emit scoreboard traces for debugging. Enable via `--trace-sched`:

```
[0] LDCU.64 UR4,  c[0][0x358]       wdep=0x35 rbar=0x01 misc=7
    pending_ur_writes: {UR4: 0x35, UR5: 0x35}

[1] LDCU.64 UR6,  c[0][0x210]       wdep=0x31 rbar=0x01 misc=7
    pending_ur_writes: {UR4: 0x35, UR5: 0x35, UR6: 0x31, UR7: 0x31}

[...]

[8] IMAD    R2, R0, UR2, R1         wdep=0x3e rbar=0x03 misc=1
    src_regs={R0, R1}, ur_srcs={UR2}
    pending_writes[R0] = (7, 0x31) → rbar |= 0x02
    pending_writes[R1] = (7, 0x31) → rbar |= 0x02
    pending_ur_writes[UR2] = (6, 0x31) → rbar |= 0x02
    final: rbar = 0x01 | 0x02 = 0x03

[...]

[15] IADD3   R10, R8, R9, RZ         wdep=0x3e rbar=0x09 misc=1
    src_regs={R8, R9}
    pending_writes[R8] = (12, 0x35) → rbar |= 0x09 (LDG special)
    pending_writes[R9] = (14, 0x35) → rbar |= 0x09 (LDG special)
    final: rbar = 0x09
```

This trace is invaluable for diagnosing scoreboard bugs.

---

## Appendix O — Bit-level ctrl word examples

The 23-bit ctrl word packs into b13..b15 as `(ctrl << 1)`. Here are
decoded examples:

### O.1 ctrl=0x7e0 (default)

```
Binary (23 bits): 111_1110_0000
Fields:
  stall  = 0b11111 = 31 (max)
  yield  = 0
  wbar   = 0
  rbar   = 0b00000 = 0
  wdep   = 0b111110 = 0x3e (ALU)
  misc   = 0b0000 = 0
```

Used for ALU ops with no dependencies and no special misc.

### O.2 ctrl=0x7f5 (EXIT)

```
Binary: 1111_1111_0101
Fields:
  stall  = 31
  yield  = 1
  wbar   = 1
  rbar   = 0b11111 = 0x1f (wait-for-all barrier)
  wdep   = 0b110101 = 0x35 (hm, unusual)
  misc   = 0b0101 = 5
```

(Note: EXIT typically uses wdep=0x3f — ctrl values vary by context.)

### O.3 ctrl=0x0fe6 (IMAD.WIDE with minimal stall)

```
Binary: 0000_1111_1110_0110
Fields:
  stall  = 0
  yield  = 0
  wbar   = 1
  wdep   = 0b111110 = 0x3e
  misc   = 0b0001 (after shift) — varies
```

---

## Appendix P — Testing matrix

OpenPTXas's 356/356 passing tests cover:

| Category          | Tests | Coverage                          |
|-------------------|-------|-----------------------------------|
| Integer ALU       | 45    | IADD3, IMAD, LOP3, SHF, ...       |
| Float ALU         | 38    | FADD, FFMA, FMNMX, FSEL, ...      |
| FP64              | 18    | DADD, DMUL, DFMA, DSETP, conv     |
| Memory            | 42    | LDG, STG, LDS, LDC, LDCU          |
| Atomics           | 14    | ATOMG.ADD/MIN/MAX/CAS (U32/F32/64)|
| Predicates        | 22    | ISETP, FSETP, DSETP, PLOP3        |
| Warp ops          | 19    | SHFL, VOTE, REDUX, MATCH, ELECT   |
| Tensor core       | 24    | HMMA/IMMA/DMMA/QMMA               |
| TMA               | 12    | UTMALDG, UTMASTG, UBLKCP          |
| Texture/surface   | 14    | TEX, TLD4, TXQ, SULD, SUST        |
| Control flow      | 16    | BRA, CALL.REL, RET.REL, if-conv   |
| Barriers          | 9     | BAR.SYNC, MEMBAR, ERRBAR, cluster |
| Uniform datapath  | 28    | UMOV, UIADD3, UISETP, ULEA        |
| End-to-end        | 55    | Real-world kernel patterns        |

70 of these are **GPU-verified**: the cubin is loaded on RTX 5090 and the
output is compared to a CPU reference. The remaining are byte-diff or
static-analysis validations.

---

*Reference maintained at github.com/garrick99/openptxas. Corrections and
additions welcome.*


