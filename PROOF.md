# OpenPTXas — Canonical Proof

**Date:** 2026-03-20
**Claim:** OpenPTXas compiles and executes multi-block GPU kernels on RTX 5090 without NVIDIA's compiler toolchain.

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 5090 |
| Architecture | SM_120 (Blackwell) |
| Driver | 595.79 |
| Compute Capability | 12.0 |
| OS | Windows 11 (Build 26200) |
| Python | 3.14.0 |
| NVIDIA Compiler | **NOT USED** |

## Results

```
71/71 unit tests passed

GPU: Rotate-Left (probe_k1)
  Input:    0x0123456789abcdef
  Output:   0x02468acf13579bde
  Expected: 0x02468acf13579bde
  PASS

GPU: Vector Add (32 elements, 1 block)
  PASS: 32 elements verified correct

GPU: Vector Add (128 elements, 4 blocks)
  PASS: 128 elements verified correct

GPU: Vector Add (1024 elements, 32 blocks)
  PASS: 1024 elements verified correct
```

## Pipeline

```
PTX source → OpenPTXas → cubin → RTX 5090 → correct output
```

No nvcc. No ptxas. No NVIDIA compiler at any stage.

## Artifact Checksums (MD5)

```
9b135d559c86e93dc1c83c2a3669ebec  vector_add.ptx
99c6b2196c1974b724919294a65255fd  vector_add.cubin
739a4b26e5cfc028d3d0df2338db5e24  probe_k1.cubin
```

## Reproduce

```bash
cd openptxas
python -m pytest tests/ -q                    # 71/71 pass
python tests/gpu_vecadd_test.py \
  probe_work/vector_add.cubin vector_add 1024 # 1024 elements correct
```
