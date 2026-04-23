# OpenPTXas miscompile: UR4 clobber via FG26 + @Px EXIT

## Symptom

A minimal `load → add → shr → add → store` kernel with the standard
fuzz-kernel `@%p0 ret` tid-bounded guard writes **sentinel-untouched
bytes** to the output buffer for every lane (verified with a `0xABCDEF01`
pre-fill — all 32 lanes come back unwritten). Same kernel without the
guard writes correct values.

Surfaced by `factory/` differential fuzz 2026-04-22. `theirs_correct`
verdict on programs pid=68, pid=103 in the smoke DB (both have the
standard guard + short ALU chain + store). Remaining `theirs_correct`
cases (pid=22, pid=64, pid=106) likely the same family.

## Minimal repro

See [minimal.ptx](minimal.ptx) (5 body instructions) and
[noguard.ptx](noguard.ptx) (same body, guard removed).

Expected output per lane: `((input + 3) >> 2) + 256`.

## Root cause

**Not a scoreboard bug** (as first suspected from ctrl-byte diff).
The ctrl-byte difference was a symptom of a reordered SASS stream.
The real cause is in the uniform-register (UR) allocator:

FG26 admission in `pipeline.py` (line ~1234, condition on
`alloc.addr_pair_colocated && _fg26_setp_ok && !_has_ctaid_ntid &&
!_fg26_te10_blocked`) sets `ctx._next_ur = 4` so the setp param `n`
gets loaded into UR4 — matching ptxas's pattern of reusing UR4 for
both n and the memory descriptor.

But OpenPTXas emits the loads in the wrong order:

```
LDCU.64 UR4, c[0][0x358]   // mem desc (inserted at s2r_pos by preamble logic)
LDCU.64 UR6, c[0][0x380]   // p_in (body)
LDCU   UR4, c[0][0x390]    // n (body) -- CLOBBERS mem desc in UR4 !!
ISETP  .GE.U32.AND P0, R3, UR4, PT  // reads UR4 = n
@P0 EXIT
...
LDG.E R6, desc[UR4][R4.64] // reads UR4, but it's still "n", not mem desc
STG.E desc[UR4][R2.64], R7 // same — store goes to arbitrary address
```

ptxas emits:
```
LDCU UR4, c[0][0x390]      // n FIRST (no prior mem desc in UR4)
ISETP ... UR4 ...
@P0 EXIT                   // early-exit while UR4 still holds n
LDCU.128 UR8, c[0][0x380]  // p_in + p_out combined
LDCU.64 UR4, c[0][0x358]   // mem desc — reloads UR4 AFTER n is no longer needed
```

ptxas's invariant: UR4 sequentially holds `n` then `mem desc`, never
aliased. OpenPTXas violates this by doing the mem-desc load BEFORE
n's load in the same UR.

## Attempted fix (REVERTED)

Moving the mem-desc LDCU to after the predicated EXIT makes `UR4 = n
→ EXIT → UR4 = mem desc`, which looks right. But it breaks the LDCU
slot-rotation invariant in `scoreboard.py::_wdep_for_opcode`: the
first LDCU.64 after S2R gets slot `0x35` (= the LDG wdep slot), and
subsequent LDCU.64s rotate between `0x31` and `0x33`. After the
move, `LDCU.64 UR6 (p_in)` becomes the first LDCU.64 and takes slot
`0x35`, aliasing with the LDG scoreboard and corrupting its wait
chain. Sentinel test confirms: kernel still writes nothing.

The move also introduces a label-map adjustment problem for any kernel
with labels between the original mem-desc position and the EXIT. Not a
concern for this minimal kernel (single BB) but would matter in
production.

## What a real fix needs to do

The fix has to coordinate two invariants simultaneously:

1. **UR4 allocation**: under FG26, `n` goes into UR4 first, then
   mem desc reloads UR4 after it's no longer needed.
2. **LDCU slot rotation**: the first LDCU.64 (descriptor class) must
   get slot `0x35`, subsequent ones rotate `0x31`/`0x33`.

ptxas gets both right because it consistently emits n → EXIT → p_* →
mem_desc, so the first LDCU.64 IS the mem desc, which naturally gets
slot `0x35`.

Candidate approaches:
- **(a)** Move the preamble mem-desc LDCU to after `@Px EXIT`, AND
  reset `_ldcu_slot_counter` to 0 at the move point so the moved
  mem-desc gets slot `0x35` fresh. Also adjust label_map for
  instructions in the affected range.
- **(b)** Keep mem-desc at s2r_pos (preamble) but allocate `n` to a
  different UR (say UR6) when FG26 fires together with a prior body
  LDG/STG that needs UR4. Lose BYTE_EXACT parity with ptxas on those
  kernels.
- **(c)** Insert a *reload* LDCU.64 UR4 after the `@Px EXIT`, leaving
  the preamble copy in place. The first copy still takes slot `0x35`;
  the reload takes whatever rotation slot. Consumer LDG waits for the
  reload's slot via scoreboard tracking. Cheapest in label-map terms
  (just one inserted instruction to account for).

Option (a) is most faithful to ptxas but requires careful label-map
surgery. Option (c) is the smallest diff.

### Attempted option (c), REVERTED 2026-04-22

Implemented in `sass/pipeline.py` gated on `_fg26_ur4_start && not
has_bar_in_body && not _skip_ur4 && predicated_exit_present` — insert
a second `LDCU.64 UR4, c[0][0x358]` right after the `@Px EXIT`, with
label_map / _bra_fixups adjusted for the +1 instruction.

Result on the minimal repro: **works** — OURS and THEIRS byte-
identical for the 32-lane canonical input. Corpus health: **144/144
GPU PASS preserved**, BYTE_EXACT count unchanged at 50, but 16
CTRL_ONLY kernels moved to STRUCTURAL (extra instruction counted as
a structural divergence from ptxas).

Factory 60s smoke, however, regressed: 15 divergences → **48** (+33
new `theirs_correct` programs). These are random-fuzz kernels that
previously produced correct output and now miscompile with the reload
in place. Mechanism suspected: the inserted LDCU shifts the
misc-counter progression for the rest of the body, moving downstream
ALU ops into scoreboard misc-values that interact badly with specific
RAW chains the simpler corpus kernels don't exercise.

Net regression on the fuzz surface — reverted.

Next attempt should probably be **option (a)** (move + slot-counter
reset). Preserving the single-LDCU structure ptxas uses avoids
perturbing misc-counter alignment on kernels that already compile
correctly. The label_map surgery is the cost; it should be scoped
narrowly to labels in the (mem_desc_orig_pos, exit_pos] range.

### Attempted option (a), DOES NOT FIX THE REPRO

Implemented as a pop+insert that relocates the mem-desc `LDCU.64 UR4`
from s2r_pos to right after the `@Px EXIT`, with `label_map` and
`_bra_fixups` shifted for instructions in the (mem_desc, exit] range.

SASS post-move looks correct by inspection — moved LDCU.64 gets
`wdep=0x35` (scoreboard's `@Px EXIT` handling resets `_ldcu_slot_counter`,
so the post-EXIT LDCU.64 gets the descriptor slot fresh). The pre-EXIT
LDCU.64 for p_in also gets slot `0x35` (as the first overall LDCU.64),
so slot 0x35 has dual posters — but LDG/STG consumers should wait for
both to drain, which is correct-and-slow.

**But**: 32/32 lanes come back unwritten on the sentinel test. The
store doesn't fire. The working option (c) attempt (which keeps the
preamble mem-desc AND adds a reload) works on the same SASS, so
something in having-exactly-one-LDCU-64-UR4 is structurally different
from having-two.

Leading hypothesis: when the preamble LDCU.64 is absent, UR5 (the
high half of the mem-desc pair) is undefined between S2R and the
moved LDCU.64. The `LDCU UR4, n` between them writes only UR4 (low),
so nothing sets UR5 before the move point. Even though the moved
LDCU.64 does set UR5 later, there may be a scoreboard-tracking gap
where UR5's prior-undefined state propagates into some latched
dependency the rbar on later consumers doesn't cover. The dual-LDCU
arrangement of option (c) initializes UR5 early and keeps the
invariant that "UR5 always holds mem_desc_hi once the preamble
completes" unbroken.

### Attempted option (c) narrowed, REGRESSION PERSISTS

Added a narrowing gate to option (c): only insert the post-EXIT
reload when the body actually contains `LDCU UR4` (the 32-bit load
that performs the clobber). Hypothesis: maybe some FG26-admitted
kernels don't actually have n live in UR4 at runtime, and the
unconditional reload was breaking them.

Narrowed fix: minimal repro still passes (0/32 unwritten), fuzz smoke
still regresses by the same +33 programs. So the regression isn't
caused by irrelevant kernels — it's the same LDCU-UR4-in-body kernel
shape that the fix targets, producing now-different output.

### Remaining options

The fix probably can't live at "insert after EXIT" — the scoreboard
perturbation is too broad. Real fix likely needs one of:

1. **UR allocator**: when FG26 admits AND the body will have a post-
   EXIT `desc[UR4]` memory op, allocate `n` to a different UR (UR6+)
   so UR4 is never clobbered. Costs BYTE_EXACT parity with ptxas on
   those kernels but avoids the clobber entirely.
2. **LDCU slot assignment**: when the mem-desc LDCU.64 and p_in
   LDCU.64 both pre-date the EXIT, force p_in to slot `0x31`/`0x33`
   instead of `0x35` (break the rotation rule for this pattern).
   Would require `_wdep_for_opcode` awareness of whether the current
   LDCU.64 is the "canonical descriptor" vs a "param".
3. **Study ptxas's exact emission path** for this family — ptxas
   gets it right with a single LDCU.64 mem-desc post-EXIT, so the
   scoreboard interaction must work. Comparing the full ctrl-byte
   sequence ptxas emits vs what OpenPTXas emits in option (a) should
   identify the missing scoreboard bit.

## Regression test

Once fixed: `PTX_FILE=_fuzz_bugs/add_shr_add_with_tid_guard/minimal.ptx
bash _run_cubin.sh` should report OURS == THEIRS == `0x100` for lane 0.
The `noguard.ptx` variant already passes; keep it in the dossier as the
"known-good baseline" for this pipeline shape.
