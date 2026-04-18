"""PTXAS-R29.3 regression — regalloc pair safety for direct `LDC.64`
scalar-LDG lowering.

R29.1 introduced a coordinated classification that routes scalar-LDG
params (`%rd` whose only use is a zero-offset `ld.global` base) through
a body `LDC.64 R_pair, c[0][param_off]` straight into the regalloc-
assigned GPR pair.  The scheduler hoists `S2R R, %tid.x` (and peer
special-reg S2Rs) to the top of the body to hide latency, so the S2R's
physical write executes BEFORE the `LDC.64` even though PTX source
order has the S2R defined LATER than the param vreg.

Without the pair-safety fix, the linear-scan allocator reused the
high half of an already-dead direct-LDC.64 pair as the S2R
destination, producing

    pos 1:  S2R  R3, SR_TID.X           // writes R3 = tid.x
    pos 4:  LDC.64 R2:R3, c[0][...]     // OVERWRITES R3 with ptr_hi
    pos 9:  ISETP.IMM R3, 0             // reads ptr_hi, not tid.x
    @!P0 EXIT                           // fires on ALL threads
                                        // output stays zero (G5 bug)

R29.3 extends `reg_first_def` of every S2R-destined single reg back
to 0 whenever `direct_ldc_params` is non-empty.  This forces the
linear scan to allocate S2R-dest single regs BEFORE any direct-LDC.64
pair, guaranteeing no overlap with a pair half.  Scoped narrowly to
`direct_ldc_params`-active kernels so pre-R29.1 register layouts are
unchanged for all other kernels.

The tests below verify the invariant structurally (without needing a
GPU) and confirm the old interfering assignment pattern is absent.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptx.parser import parse
from sass.regalloc import allocate


_PTX_DUAL_SCALAR_LDG = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k_r293_dual(.param .u64 a, .param .u64 b, .param .u64 out)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    .reg .pred %p<1>;
    ld.param.u64 %rd0, [a];
    ld.param.u64 %rd1, [b];
    ld.param.u64 %rd2, [out];
    ld.global.u32 %r0, [%rd0];
    ld.global.u32 %r1, [%rd1];
    add.u32 %r2, %r0, %r1;
    mov.u32 %r3, %tid.x;
    setp.eq.u32 %p0, %r3, 0;
    @!%p0 ret;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""


_PTX_NON_SCALAR_LDG = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry k_r293_offset(.param .u64 in, .param .u64 out)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<5>;
    .reg .pred %p<1>;
    ld.param.u64 %rd0, [in];
    ld.param.u64 %rd1, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r3, %r1, %r2;
    add.u32 %r4, %r3, %r0;
    cvt.u64.u32 %rd2, %r4;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.u32 %r5, [%rd4];
    st.global.u32 [%rd1], %r5;
    ret;
}
"""


# ---------------------------------------------------------------------------
# Test 1 — the direct-LDC.64 pair allocated for a scalar-LDG u64 param does
#          NOT overlap any S2R-destined single-reg allocation.
# ---------------------------------------------------------------------------

def test_direct_ldc_pair_does_not_overlap_s2r_dest():
    fn = parse(_PTX_DUAL_SCALAR_LDG).functions[0]
    res = allocate(fn, sm_version=120)

    # R29.1 should classify both %rd0 and %rd1 as scalar-LDG-only
    assert '%rd0' in res.direct_ldc_params, (
        f"R29.1 classification missing for %rd0; "
        f"direct_ldc_params={res.direct_ldc_params}")
    assert '%rd1' in res.direct_ldc_params, (
        f"R29.1 classification missing for %rd1; "
        f"direct_ldc_params={res.direct_ldc_params}")

    # Collect every reg index covered by a direct_ldc_params pair.
    pair_regs: set[int] = set()
    for pn in res.direct_ldc_params:
        if pn in res.ra.int_regs:
            lo = res.ra.int_regs[pn]
            pair_regs.add(lo)
            pair_regs.add(lo + 1)

    # %r3 is produced by `mov.u32 %r3, %tid.x`, i.e. S2R-destined.
    s2r_dest_phys = res.ra.int_regs['%r3']
    assert s2r_dest_phys not in pair_regs, (
        f"R29.3 violation: S2R-destined %r3 allocated at R{s2r_dest_phys}, "
        f"which overlaps a direct-LDC.64 pair half in {sorted(pair_regs)}. "
        f"This is the exact regalloc interference that crashes the G5 "
        f"Family-B multi-scalar-LDG kernel at runtime.")


# ---------------------------------------------------------------------------
# Test 2 — the old interfering pattern (S2R lands on hi-half of a direct-
#          LDC.64 pair) is ABSENT across the specific G5-class kernel.
# ---------------------------------------------------------------------------

def test_no_s2r_on_direct_ldc_pair_high_half():
    fn = parse(_PTX_DUAL_SCALAR_LDG).functions[0]
    res = allocate(fn, sm_version=120)

    high_halves: set[int] = set()
    for pn in res.direct_ldc_params:
        if pn in res.ra.int_regs:
            high_halves.add(res.ra.int_regs[pn] + 1)

    # Gather every S2R-destined single reg's physical allocation.
    # S2R destinations in this kernel: %r3 ← %tid.x.
    s2r_physes: list[tuple[str, int]] = []
    for inst in fn.blocks[0].instructions:
        if (inst.op == 'mov'
                and hasattr(inst.dest, 'name')
                and inst.srcs
                and hasattr(inst.srcs[0], 'name')
                and inst.srcs[0].name in {
                    '%tid.x', '%tid.y', '%tid.z',
                    '%ctaid.x', '%ctaid.y', '%ctaid.z',
                    '%ntid.x', '%ntid.y', '%ntid.z',
                    '%laneid',
                }):
            dn = inst.dest.name
            if dn in res.ra.int_regs:
                s2r_physes.append((dn, res.ra.int_regs[dn]))

    for vreg, phys in s2r_physes:
        assert phys not in high_halves, (
            f"R29.3 regression: S2R-dest {vreg} → R{phys} lands on a "
            f"direct-LDC.64 pair high half {sorted(high_halves)}. "
            f"Scheduler will hoist the S2R above the LDC.64 and the LDC.64 "
            f"will then overwrite the S2R's value.")


# ---------------------------------------------------------------------------
# Test 3 — R29.3 is NARROW: it must not activate on kernels with no
#          direct-LDC.64 (scalar-LDG) class.  Non-scalar-LDG kernels keep
#          their original register layout and GPR count.
# ---------------------------------------------------------------------------

def test_r293_narrow_scope_no_effect_on_offset_form_kernels():
    fn = parse(_PTX_NON_SCALAR_LDG).functions[0]
    res = allocate(fn, sm_version=120)

    # All u64 params are offset-form (used in add.u64 / as STG base) —
    # none should be scalar-LDG-only → direct_ldc_params must stay empty.
    assert res.direct_ldc_params == set(), (
        f"R29.1/R29.3 scope violation: non-scalar-LDG kernel acquired "
        f"direct_ldc_params={res.direct_ldc_params}; R29.3 must only fire "
        f"when a scalar-LDG class is actually present.  The R29.3 live-range "
        f"extension is gated by `if direct_ldc_params:` — this assertion "
        f"locks that gate in place so future refactors cannot broaden its "
        f"effect into kernels that do not need it.")
