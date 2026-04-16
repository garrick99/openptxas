# IM01 — IMAD/UIADD coordinated frontier clustering

## 01.1 — Subcluster table (38 STRUCTURAL kernels with UR-family gap)

Top clusters by size, with cleanliness gates (HFMA2 / atom / loop):

| cluster | size | clean | delta | miss (UR) | extra (R) | representative kernels |
|---|---:|---:|---:|---|---|---|
| **C00** | **3** | **3** | **+0** | **IADD.64** | **IADD3** | k100_dual_load, r1_running_xor, r1_scatter_add |
| C01 | 3 | 2 | +1 | IADD.64 | IADD3+2 | ilp_dual_int64, k200_ilp_dual_addr, vecadd_large(HFMA2) |
| C02 | 2 | 2 | +1 | UIADD+3 | IADD3.IMM+5 | k200_pred_chain, w1_div_multi_guard *(MP02 multi-pred)* |
| C03 | 2 | 0 | +5 | IADD.64 | IADD3.IMM+3 | r1_accumulator, w1_loop_load_acc *(both LOOP)* |
| C05 | 1 | 1 | +0 | UIADD+2 | IADD3, IADD3.IMM+2 | k100_pred_arith *(MP02 multi-pred)* |
| C07 | 1 | 1 | +0 | UIADD+5 | IADD3, IADD3.IMM+5 | k300_nasty_multi_pred *(MP02)* |
| C08 | 1 | 1 | +0 | UIADD+3 | IADD3, IADD3.IMM+3 | k300_pred3 *(MP02)* |
| C11 | 1 | 1 | +1 | IADD.64 | IADD3, LOP3 | r1_multi_stage |
| C13 | 1 | 1 | +1 | UIADD+4 | IADD3.IMM+6 | w2_deep_pred *(MP02)* |
| C16 | 1 | 1 | +2 | IADD.64 | IADD3+2, IADD3.IMM | k300_nasty_zero_init |
| (22 size-1 size clusters) | | | | | | |

## 01.2 — Delta=0 prioritization (clean only)

| cluster | count | delta=0? | opcode-only? | clean (no HFMA2/atom/loop)? | priority | reason |
|---|---:|:-:|:-:|:-:|---|---|
| **C00** | **3** | ✓ | ✓ (1:1 IADD3 → IADD.64) | ✓ | **HIGHEST** | largest clean delta=0; same substitution shape across all 3 kernels |
| C05  | 1 | ✓ | ✓ + IADD3 shift | MP02 multi-pred | low | prohibited (MP02 expansion) |
| C07  | 1 | ✓ | ✓ + IADD3 shift | MP02 multi-pred | low | prohibited |
| C08  | 1 | ✓ | ✓ + IADD3 shift | MP02 multi-pred | low | prohibited |
| C04  | 1 | ✓ | mixed (IADD3.UR + IMAD shuffle) | HFMA2 | low | prohibited (HFMA2) |

## 01.3 — Coordination proof for C00

The 3 C00 kernels each have `add.u32 rd, rs1, rs2` (register-register,
no immediate, no carry) in their PTX:

* `k100_dual_load`: `add.u32 %r4, %r2, %r3;`
* `r1_running_xor`:  `add.u32 %r2, %r2, %r3;`
* `r1_scatter_add`:  `add.u32 %r2, %r2, %r0;`

OURS lowers each as `IADD3 R, R1, R2, RZ` (0x210, 3-source with explicit
RZ third source). PTXAS lowers each as `IADD.64 R, R1, R2` (0x235,
2-source 64-bit pair add — lo half is the same as IADD3's 32-bit
result; hi half writes R+1 with garbage).

PTXAS-evidence positional alignment (from kdiff dump):

| kernel | OURS site                    | PTXAS site                                    |
|---|---|---|
| r1_running_xor | `IADD3 R0,R0,R5,RZ`  | `IADD.64 R0,R0,R3` (lo = `add.u32` result)    |
| r1_scatter_add | `IADD3 R5,R0,R3,RZ`  | `IADD.64 R5,R0,R5` (lo = `add.u32` result)    |
| k100_dual_load | `IADD3 R11,R6,R10,RZ`| `IADD.64 R9,R4,R7`                            |

In all 3 cases the dest's HI half (R+1) is not read by any subsequent
instruction — the dest is consumed only by an STG (32-bit store) that
follows after a small address-compute chain (cvt + shl + add.u64).

| pattern | required proof | already implemented? | missing piece |
|---|---|:-:|---|
| `add.u32 rd, rs1, rs2` with both srcs RegOp | already isel-handled via `encode_iadd3(d, a, b, RZ)` | ✓ | — |
| dest+1 is dead at IADD3 site | look-ahead within bb: dest is consumed by exactly one STG-of-dest reachable through cvt/shl/add.u64 chain | ✗ | **new bounded look-ahead** in the `add.u32 reg-reg` isel branch |
| no HFMA2 / SHF / loop / atom contamination | PTX-level scan for `bra` (loop), and verify no HFMA2 / SHF / atom op in same kernel | ✗ | per-kernel exclusion gate |
| no scheduler dependency change | IADD.64 has same wdep slot (0x3e ALU) as IADD3 per scoreboard.py | ✓ | — |
| encoder correctness | `encode_iadd64(dest, src0, src1)` exists and is byte-verified against PTXAS | ✓ | — |

**Forbidden forms (IM02 safety obligations)**:
* `add.u32 rd, rs, ImmOp` — keep IADD3.IMM (0x810) path; immediate has no IADD.64 equivalent.
* `add.u32 rd, rs, rs` (self-modify) — must still admit if pattern matches; included in the C00 set (r1_running_xor and r1_scatter_add are self-modify).
* MP02 multi-pred kernels — gate must require no `@P` predicate guard on the add.
* HFMA2/SHF-touching kernels — gate must require no HFMA2 (0x431) or SHF (0x819) emission elsewhere in the kernel.
* Loop bodies — gate must require no `bra` in the function.
* Atom-family kernels — gate must require no atom op in the kernel.

## 01.4 — Exact first target

| chosen subcluster | kernels | why chosen | why others deferred |
|---|---|---|---|
| **C00 (IADD3 → IADD.64 for register-register add.u32 with dead HI half)** | k100_dual_load, r1_running_xor, r1_scatter_add | largest clean delta=0 cluster (3 kernels); single substitution shape (1:1); IADD.64 encoder exists and is proven; same wdep/scoreboard slot as IADD3; HI-half-dead provable via single-bb look-ahead; no MP02 / HFMA2 / SHF / atom / loop overlap | C02/C05/C07/C08/C13 are all MP02 multi-pred (prohibited); C01/C04/C06/C10/C22-30 mostly contaminated by HFMA2 / SHF / atom / loop; C11/C16/C18 are size-1 with delta≠0 |

## Proof obligations carried into IM02

1. New isel-level dead-HI-half look-ahead in the `add.u32 reg-reg`
   path: scan forward in the same basic block for an STG that reads
   `rd`; only allow cvt/shl/add.u64 instructions between.
2. Exclusion of HFMA2 / SHF / atom / loop kernels via PTX-level scan.
3. No emission change unless ALL gates pass (default keeps IADD3).
4. Existing BYTE_EXACT kernels using IADD3 with RZ third source must
   not silently switch to IADD.64 unless they meet the same gates.
5. AT07 lesson: GPU harness mandatory before declaring IM03 clean —
   semantic regression risk is highest here (HI-half corruption if
   liveness is wrong).
