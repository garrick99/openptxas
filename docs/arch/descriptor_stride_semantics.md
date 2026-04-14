# SM_120 Descriptor Stride Semantics

**Date:** 2026-04-14
**Status:** GPU-VERIFIED

## Finding

SM_120 descriptor-based STG uses **element-index addressing**.

The GPR address register holds an element index (not a byte offset).
The descriptor's element size field scales the index to produce the
byte address.  For u32 stores, the hardware multiplies the index by 4.

## Evidence

1. PTXAS compiles `out[tid] = val` using `0xc11: R_addr = tid.x + base_lo`
   (raw element index, no multiply by 4).
2. 128 threads all write to correct positions (byte offsets 0, 4, 8, ..., 508).
3. If 0xc11 high half added tid to upper 32 bits, threads >= 1 would write
   to 4GB-spaced addresses and crash.  They don't.

## 0xc11 High Half Semantics

Mode 0x14 (carry-in): `R_dest = carry + UR_src`

The GPR source at b3 is **IGNORED** by the hardware despite being present
in the encoding.  Only the carry flag from the low half and the UR source
contribute to the result.

## Implication for OpenPTXas

Our current path computes explicit byte offsets (tid * 4 via IMAD.WIDE)
and uses byte-addressed descriptors.  PTXAS uses element-indexed
descriptors and raw element indices.

Both approaches produce correct results because the descriptors are
configured differently by the CUDA driver based on .nv.info metadata.

To use the PTXAS-style 0xc11 element-index path, OpenPTXas would need
to emit .nv.info metadata that tells the driver to use element-indexed
descriptors (including attr=0x66 which PTXAS emits but we don't).

## What This Rules Out

The 0xc11 carry-chain path **cannot** simply replace IMAD.WIDE + IADD.64-UR
in non-byte-exact kernels by using raw tid.x.  The descriptor stride mode
must also be changed, which requires .nv.info attr=0x66 support.

The current 13 byte-exact kernels work because our output is identical to
PTXAS (including the nv.info from our PTXAS-matched template), so the
descriptor is configured identically by the driver.
