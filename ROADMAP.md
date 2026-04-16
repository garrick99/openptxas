# OpenPTXas — Roadmap (MP04)

Evidence-driven roadmap recomputed after MP02 (multi-predicate fix) and MP03
(packaging baseline). Dated alongside commit `e32b14e`.

All numbers below come from `python scripts/health.py` executed against
commit `e32b14e`.

---

## Frontier state

Corpus: 144 registered kernels.

| Metric                     | Before MP02 | After MP03 |
|----------------------------|------------:|-----------:|
| BYTE_EXACT                 | 40          | **46**     |
| STRUCTURAL                 | 101         | **98**     |
| MIXED                      | 0           | **0**      |
| REG_AND_CTRL               | 0           | 0          |
| errors                     | 0           | 0          |
| GPU PASS (harness)         | 118         | **126**    |
| GPU FAIL (harness)         | 19          | **11**     |
| GPU RUN_EXC                | 7           | 7          |
| pytest                     | 865 / 865   | 865 / 865  |

Net MP02 delta: +6 BYTE_EXACT, +8 GPU-correct, 0 regressions. The 18 remaining
GPU failures are categorised in `PACKAGING.md` under *Known Unsupported Families*.

## Dominant STRUCTURAL blockers

Opcode accounting across the 98 STRUCTURAL kernels (delta = # instances
PTXAS emits minus # instances OURS emits; only top entries shown).

### Opcodes OURS is missing (PTXAS uses, OURS does not)

| Opcode  | Mnemonic    | Missing count | Family                          |
|---------|-------------|--------------:|---------------------------------|
| 0x835   | UIADD       | 70            | **UR-pipeline (dominant)**      |
| 0x80c   | ISETP.I     | 35            | predicate / UR-mixed            |
| 0x431   | HFMA2       | 24            | FP integer zero-init trick      |
| 0x235   | IADD.64     | 22            | 64-bit address arithmetic       |
| 0x424   | FMUL.I      | 19            | FP integer trick (pairs HFMA2)  |
| 0xc11   | IADD3.UR    | 14            | UR-pipeline                     |
| 0xc0c   | ISETP.UR    | 9             | UR-pipeline                     |

### Opcodes OURS emits in excess

| Opcode  | Mnemonic    | Extra count | Substitute for                  |
|---------|-------------|------------:|---------------------------------|
| 0x810   | IADD3.I     | 99          | UIADD (0x835)                   |
| 0x210   | IADD3       | 69          | UIADD (0x835)                   |
| 0x824   | IMAD.I      | 39          | UR-path IMAD / UIADD            |
| 0x835   | UIADD       | 32          | wrong count / wrong positions   |
| 0x7ac   | LDCU        | 18          | UR-pipeline LDCU placement      |
| 0xc35   | IADD.64-UR  | 16          | IADD.64 + UIADD blend           |

**Single-sentence reading**: PTXAS lowers many ALU operations through the
uniform-register pipeline (UIADD / IADD3.UR / ISETP.UR / LDCU placement); we
lower the same operations through the general-purpose-register pipeline
(IADD3.I / IADD3 / IMAD.I). The net missing-UR opcode count is **~120 across
the 98 STRUCTURAL kernels** — roughly 3× any other single family.

### Instruction-delta distribution (ours_count − ptxas_count)

| Delta | Kernels |
|------:|--------:|
|  −57  | 1       |
|  −20  | 1       |
|  −5   | 4       |
|  −4   | 3       |
|  −3   | 3       |
|  −2   | 1       |
|  −1   | 5       |
|   0   | **24**  |
|  +1   | 23      |
|  +2   | 8       |
|  +3   | 6       |
|  +4   | 9       |
|  +5   | 5       |
|  ≥+6  | 5       |

**24 STRUCTURAL kernels already have the PTXAS instruction count** — same
count, different opcodes. These are the shortest path to BYTE_EXACT and are
almost all UR-pipeline swaps (IADD3.I ↔ UIADD substitution).

## Family ranking

| Rank | Family                       | STRUCTURAL count* | Dominant blocker                          | Recommended subsystem |
|-----:|------------------------------|------------------:|-------------------------------------------|-----------------------|
| 1    | UR-pipeline (UIADD + IADD3.UR + ISETP.UR) | ~120 | isel picks R-form instead of UR-form when sources are uniform | **UIADD isel-level subsystem** |
| 2    | HFMA2 + FMUL.I (FP-for-int trick) | 43 | no isel substitution for the ptxas zero-init idiom | HFMA2 isel-level subsystem |
| 3    | IADD.64 (0x235)              | 22                | address pair lowering uses IADD3 pair vs ptxas 64-bit single | 64-bit add widening |
| 4    | SHF family                    | 4 emitted, many still missing | bottom-up byte encoders for remaining shifts | finish SHF harvest (FG67-70) |

*count = instances across STRUCTURAL corpus.

## Roadmap answers

**1. Is multi-predicate correctness now closed?**
**Yes.** 8/8 previously-failing multi-predicate kernels now GPU-pass, 6/6
single-predicate BYTE_EXACT controls unchanged, frontier MIXED remains 0,
pytest 865/865 green. MP02 fix committed (`a1a05ea`), validated end-to-end
through `scripts/health.py`.

**2. What is the next highest-leverage backend subsystem?**
**UIADD / UR-pipeline isel-level integration.** Evidence: ~120 missing
UR-family opcodes across the STRUCTURAL corpus — 3× any other missing family.
The 24 delta-0 kernels are dominated by IADD3.I→UIADD substitutions and would
flip to BYTE_EXACT directly with a correct uniformity-tracking isel pass.

**3. Is SHF harvest safe to resume?**
**Yes.** MP02 did not touch SHF (FG67-70 remains the last SHF work). The
frontier recompute shows SHF at only 4 extra emissions in OURS; blocker is
byte-level encoder completeness for variants we do not yet emit. Resumption
carries no MP02 interaction risk.

**4. What is still missing before backend packaging is "solid"?**
- The 18 GPU-failing kernels (loops, divergent if/else, accumulators, atom+reduce).
  Per-family fixes, each smaller in scope than UIADD.
- A documented "release" procedure (tag, artifact bundle, changelog). Not a
  correctness issue — purely process.
- Cross-host validation: current numbers are single-host (RTX 5090). No
  second-platform validation exists.

**5. Should Forge / OpenCUDA wiring remain deferred?**
**Yes.** PACKAGING.md explicitly scopes them out. Those live in external
repos and their wiring is unrelated to PTX → cubin correctness or coverage.

## One next move

> **Next move: UIADD isel-level subsystem** — a uniformity-tracking isel pass
> that selects UIADD (0x835) / IADD3.UR (0xc11) / ISETP.UR (0xc0c) when both
> sources are uniform (kernel parameter, constant, or SR-derived unchanged).
>
> **Because**: the opcode-delta ledger shows ~120 missing UR-family instances
> versus 24 STRUCTURAL kernels already at zero instruction-count delta — the
> largest single bucket and the closest convergence bucket are the same
> family. No other subsystem moves this many kernels for this little work.
>
> **Constraint from FG65/66 HARD BAIL**: the subsystem must integrate at
> isel-level, not as a post-scheduling byte rewrite. Post-scheduling rewrites
> corrupt the scoreboard (lesson preserved in `sass/scoreboard.py` comments).
