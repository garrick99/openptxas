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

## Regression test

Once fixed: `PTX_FILE=_fuzz_bugs/add_shr_add_with_tid_guard/minimal.ptx
bash _run_cubin.sh` should report OURS == THEIRS == `0x100` for lane 0.
The `noguard.ptx` variant already passes; keep it in the dossier as the
"known-good baseline" for this pipeline shape.
