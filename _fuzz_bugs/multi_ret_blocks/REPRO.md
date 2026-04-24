# OpenPTXas miscompile: multi-ret basic blocks

## Symptom

PTX kernel with multiple basic blocks each terminated by `ret` (no
shared merge block before the function exit) compiles cleanly through
OpenPTXas but fires `sync_err=700` (CUDA_ERROR_ILLEGAL_ADDRESS) at
launch. The same PTX assembled by nvidia `ptxas` runs correctly.

## Minimal repro

See [minimal.ptx](minimal.ptx) — a diamond-style guarded compute
kernel. The shape:

```
setp.lt.s32 %p0, %r5, %r0;
@!%p0 ret;                 // outer bounds guard, one ret site
...
@%p0 bra if_true_5;
bra if_false_6;
if_true_5:
    <store, mul*2.0>
    ret;                   // arm-A ret
if_false_6:
    <store, mul*0.5>
    ret;                   // arm-B ret
```

Three separate `ret` sites: one outer guard, one per arm of the
inner diamond. No post-dominating merge block.

Reproduced 2026-04-23 during OpenCUDA cleanup work: an OpenCUDA IR
pass that folded empty `RetTerm` blocks started generating this shape
for any kernel with an if-else diamond where both arms returned
directly. Eleven OpenCUDA GPU E2E tests regressed (test_cond,
test_ballot, test_atomic_counter, test_atomic_minmax, test_shared_copy,
test_scan, test_stencil, test_matvec, test_maxreduce, test_histogram,
test_redux). All eleven compile cleanly through `ptxas` and execute
correctly — only OpenPTXas misbehaves on the multi-ret layout.

## Verification

```
# Via OpenCUDA smoke (fires bug):
python3 -m opencuda minimal.cu --emit-ptx --out /tmp/k.ptx
# Via ptxas: compiles + runs correctly
ptxas -arch=sm_120 minimal.ptx -o /tmp/k_theirs.cubin
# Via OpenPTXas: compiles, sync_err=700 at launch
python3 -c 'from sass.pipeline import compile_function; ...'
```

## Workaround

Keep OpenCUDA codegen emitting the classic "bra merge_label; merge_label: ret"
tail instead of collapsing it to inline `ret`. The OpenCUDA pass
`fold_empty_ret_blocks` was drafted then reverted (2026-04-23) after
discovering this OpenPTXas bug; `opencuda/ir/optimize.py` currently
contains no such pass. Restoring a narrow version that fires only
when the target block has exactly one BrTerm predecessor (and no
CondBr refs) would avoid creating new multi-ret patterns, but measured
0 fires across the 82-kernel test corpus — i.e. dead code.

## Likely root cause

OpenPTXas's pipeline somewhere assumes every function has a single
kernel-exit point (one post-dominating ret). Candidate investigation
sites:

- `sass/pipeline.py` — look for BRA fixup / exit-label logic that
  might rewrite only the first ret and leave others pointing at
  garbage offsets
- `sass/schedule.py` — reorder pass may treat the second `ret`
  basic block as unreachable-after-first-ret and drop or mis-order
  it
- `sass/scoreboard.py` — predicated @!pX ret doesn't reset scoreboard
  state the way an unconditional ret does; subsequent blocks' rbar
  might be computed off the wrong baseline

First place to look: is the emitted SASS even correct syntactically
for the second arm? Dump cubin, disassemble, compare against ptxas's
output for the same PTX.
