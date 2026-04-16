# TPL08 — Non-atom template frontier recompute and continuation

Evidence for commit `1113396` (TPL07), built on TPL06's `432595c`.

## 08.1 — Frontier delta

| metric                              | before TPL05 | after TPL08 |
|-------------------------------------|-------------:|------------:|
| Corpus BYTE_EXACT                   | 51           | **52**      |
| Corpus STRUCTURAL                   | 93           | **92**      |
| MIXED / REG_AND_CTRL / errors       | 0 / 0 / 0    | 0 / 0 / 0   |
| pytest                              | 865/865      | 865/865     |
| GPU PASS / FAIL / RUN_EXC           | 126 / 11 / 7 | 126 / 11 / 7|

| family | before | after |
|---|---:|---:|
| atom-template kernels BYTE_EXACT | 6 | 6 (unchanged) |
| **non-atom-template kernels BYTE_EXACT** | 1 | **2** (+k300_nasty_zero_init) |

Cumulative non-atom template harvest (TPL01 + TPL05): **2 kernels**,
**+2 BYTE_EXACT**, zero regressions across both sprint chains.

## 08.2 — Continuation ranking

3 remaining template-direction candidates (post-TPL05):

| kernel | exact-shape match to any prior template? | next blocker | recommendation |
|---|:-:|---|---|
| r1_scatter_add | NO (1 param, no LDG, mul + and + add chain) | own per-kernel template (~13 PTX ops) | TPL09-style sprint chain — simplest of the 3 remaining |
| r1_running_xor | NO (1 param, no LDG, XOR + XOR + add + AND chain) | own per-kernel template (~14 PTX ops) | TPL13-style sprint chain |
| r1_multi_stage | NO (1 param, no LDG, mul + add + and + xor + add chain) | own per-kernel template (~15 PTX ops, longest body) | TPL17-style sprint chain |

**Each of the 3 needs its own JSON template** (different PTX shapes →
different PTXAS byte sequences).  The TPL01–TPL08 mechanism — now
formalized as the `_TPL_NON_ATOM_REGISTRY` list in `pipeline.py`
plus a per-kernel JSON file — generalises cleanly: each new template
adds one JSON file (~30–50 lines) and one registry entry (1 line).

Reachable BYTE_EXACT gain in this template direction: **+3 kernels**
if all 3 land cleanly.

## 08.3 — One next move

> **Next move: TPL09 — bounded whole-kernel template for `r1_scatter_add`.**
>
> **Why this and not the other 2 remaining**:
>
> * Same risk profile as TPL01–TPL08 (one-kernel template, kernel-name
>   admission gate, GPU validation per AT07 lesson).
> * `r1_scatter_add` has the shortest body of the 3 remaining template-
>   direction candidates (~13 PTX ops vs 14 / 15 for the others).
> * Body op set is well-trodden ground: `mul.lo.u32` + `and.b32` +
>   `add.u32(reg, reg)` + standard tid-addr chain. Each of these
>   ops is already template-coverable in some other context (atom
>   templates, TPL01, TPL05) — the encoder bytes are well known.
> * Workbench-driven byte extraction is a 1-line `compile_ptxas` call
>   away, exactly as it was for TPL01 and TPL05.
>
> **Why NOT pivot away from templates yet**:
>
> * TPL01 and TPL05 both landed cleanly with zero regressions — the
>   non-atom template machinery is now proven across 2 distinct
>   kernel shapes.
> * 3 more bounded BYTE_EXACT kernels are reachable in this exact
>   direction, each its own ~3-file commit (JSON + 1 registry entry +
>   validation doc).
> * Allocator-aware precursor remains explicitly out of bounded scope.
> * SHF was conclusively shown by MEGA-01 to be allocator/scheduler
>   bound — no clean bounded slice today.

After TPL09–TPL12 (the next sprint chain), if r1_scatter_add lands,
the corpus would be at **53 BYTE_EXACT / 91 STRUCTURAL / 0 MIXED**.
Two more sprint chains after that (TPL13–TPL16 for r1_running_xor,
TPL17–TPL20 for r1_multi_stage) would close out this direction at
**55 BYTE_EXACT / 89 STRUCTURAL** — the bounded ceiling for the
template-direction candidates identified in MEGA-01.

## Honest assessment

The `_TPL_NON_ATOM_REGISTRY` pattern is now **proven and stable**:
TPL05 added a second entry without any infrastructure changes — only
a new JSON file plus a one-line registry entry. The post-EXIT `b9=0x0c`
rewrite skip generalisation (`'TPL01'` → `'TPL'`) means future
templates need no further `pipeline.py` patches.

Per-sprint leverage stays at **+1 BYTE_EXACT** (small but consistent),
and per-sprint risk stays at **AT06-class** (low). The cumulative
gain over a 4-sprint-chain pace is meaningful.

The wider 92-kernel STRUCTURAL frontier remains gated by subsystems
explicitly outside the template path: predicate-body work (MP02
expansion), HFMA2 subsystem, allocator-aware IADD.64 substitution,
loop / CAS / divergent-if subsystems. Templates do not unblock those —
they harvest specific exact-shape kernels one at a time.

## Commit table for this sprint chain

| phase | commit | pushed |
|---|---|:-:|
| TPL05 | `a859f67` | ✓ |
| TPL06 | `432595c` | ✓ |
| TPL07 | `1113396` | ✓ |
| TPL08 | (this doc) | pending commit |
