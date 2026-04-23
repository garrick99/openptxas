"""R35 probe: dump PTX->phys regalloc mapping + live ranges for s2_fail,
check for IMAD.SHL dest overlap with other live vregs.

Specifically check:
  - IMAD.SHL dest (%r3 post-compact)
  - MOV source (the reg %r2 reads in cvt)
  - S2R dests
  - Any other vreg that might share the phys reg across the IMAD->MOV
    dependency window.
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, 'C:/Users/kraken/openptxas')

from ptx.ir import RegOp, MemOp
from ptx.parser import parse
from sass.pipeline import _if_convert, _sink_param_loads, _r31_rename_inplace_u64_redefine_across_exit
from sass.regalloc import allocate


_PTX = """
.version 9.0
.target sm_120
.address_size 64
.visible .entry s2_fail(.param .u64 in, .param .u64 out) {
    .reg .b32 %r<4>; .reg .b64 %rd<3>; .reg .pred %p<1>;
    ld.param.u64 %rd0, [in]; ld.param.u64 %rd1, [out];
    ld.global.u32 %r0, [%rd0]; mov.u32 %r1, %tid.x; setp.eq.u32 %p0, %r1, 0; @!%p0 ret;
    mov.u32 %r2, %ctaid.x; shl.b32 %r3, %r2, 2; cvt.u64.u32 %rd2, %r3;
    add.u64 %rd1, %rd1, %rd2; st.global.u32 [%rd1], %r0; ret;
}
"""


def main():
    fn = parse(_PTX).functions[0]
    _if_convert(fn)
    _sink_param_loads(fn)
    _r31_rename_inplace_u64_redefine_across_exit(fn)

    # Flatten into a single linear instruction list (matches regalloc's view).
    all_instrs = []
    for bb in fn.blocks:
        all_instrs.extend(bb.instructions)

    # Compute live ranges the same way regalloc does.
    reg_first_def: dict[str, int] = {}
    reg_last_use: dict[str, int] = {}
    for idx, inst in enumerate(all_instrs):
        if inst.dest and isinstance(inst.dest, RegOp):
            n = inst.dest.name
            if n not in reg_first_def:
                reg_first_def[n] = idx
        for src in inst.srcs:
            if isinstance(src, RegOp):
                reg_last_use[src.name] = idx
            if isinstance(src, MemOp) and isinstance(src.base, str):
                bn = src.base if src.base.startswith('%') else f'%{src.base}'
                reg_last_use[bn] = idx

    # Allocate.
    res = allocate(fn, sm_version=120)

    # PTX instruction listing with indices.
    print('=== PTX instruction list (post-R31 rename) ===')
    for idx, inst in enumerate(all_instrs):
        print(f'  [{idx:2d}] {inst}')
    print()

    # Phys-reg mapping.
    print('=== PTX vreg -> phys reg ===')
    rows = []
    for vname, phys in sorted(res.ra.int_regs.items()):
        fd = reg_first_def.get(vname, None)
        lu = reg_last_use.get(vname, None)
        rows.append((phys, vname, fd, lu))
    rows.sort()
    for phys, vname, fd, lu in rows:
        print(f'  R{phys:<3d}  {vname:<22s}  first_def={fd}  last_use={lu}')
    print()
    print(f'  direct_ldc_params = {res.direct_ldc_params}')
    print(f'  r31 force_ur      = {getattr(fn, "_r31_force_ur_params", set())}')
    print()

    # Build reverse map: phys -> [vname, ...]
    by_phys: dict[int, list[tuple[str, int, int]]] = {}
    for vname, phys in res.ra.int_regs.items():
        fd = reg_first_def.get(vname)
        lu = reg_last_use.get(vname)
        for p in (phys, phys + 1) if _is_64(fn, vname) else (phys,):
            by_phys.setdefault(p, []).append((vname, fd, lu))

    # Check the critical range: S2R CTAID (%r2 first_def) through IADD.64
    # consumer.  Look for overlaps at phys(%r3) and phys(%r2).
    print('=== Phys reg -> vreg(s) assigned ===')
    for phys in sorted(by_phys):
        entries = by_phys[phys]
        if len(entries) == 1:
            vname, fd, lu = entries[0]
            print(f'  R{phys:<3d}  {vname:<22s}  [{fd}..{lu}]')
        else:
            print(f'  R{phys:<3d}  SHARED:')
            for vname, fd, lu in entries:
                print(f'          {vname:<22s}  [{fd}..{lu}]')
    print()

    # Identify the critical chain vregs and check for overlap.
    # Chain in PTX after the predicated EXIT:
    #   mov.u32 %r2, %ctaid.x       <- %r2 (S2R CTAID_X dest)
    #   shl.b32 %r3, %r2, 2          <- %r3 (IMAD.SHL dest), reads %r2
    #   cvt.u64.u32 %rd2, %r3        <- %rd2 (cvt dest), reads %r3
    #   add.u64 %__r31_rd1_0, %rd1, %rd2  <- reads %rd2 (MOV chain in SASS)
    print('=== Critical chain (post-EXIT offset computation) ===')
    for vname in ('%r2', '%r3', '%rd2', '%__r31_rd1_0', '%r0', '%r1', '%rd0', '%rd1'):
        if vname in res.ra.int_regs:
            phys = res.ra.int_regs[vname]
            is_64 = _is_64(fn, vname)
            fd = reg_first_def.get(vname)
            lu = reg_last_use.get(vname)
            slot_str = f'R{phys}' + (f':R{phys+1}' if is_64 else '')
            print(f'  {vname:<22s} -> {slot_str:<12s} live=[{fd}..{lu}]')
        else:
            print(f'  {vname:<22s} NOT IN int_regs (may be UR / dead / etc.)')
    print()

    # Focused overlap check for IMAD dest (%r3) and MOV source (%r3 itself) —
    # but the real aliasing concern is %r3 vs %r1 (tid.x, lives through the
    # ISETP at idx 4) across R29.3's S2R live-range extension.
    print('=== R29.3 S2R live-range extension check ===')
    _SPECIAL_SRC = {
        '%tid.x', '%tid.y', '%tid.z',
        '%ctaid.x', '%ctaid.y', '%ctaid.z',
        '%ntid.x', '%ntid.y', '%ntid.z',
        '%nctaid.x', '%nctaid.y', '%nctaid.z',
        '%laneid',
    }
    s2r_dests = []
    for inst in all_instrs:
        if (inst.op == 'mov'
                and isinstance(inst.dest, RegOp)
                and inst.srcs
                and isinstance(inst.srcs[0], RegOp)
                and inst.srcs[0].name in _SPECIAL_SRC):
            dname = inst.dest.name
            s2r_dests.append((dname, inst.srcs[0].name))
    for dname, special in s2r_dests:
        phys = res.ra.int_regs.get(dname)
        print(f'  S2R-dest {dname} <- {special}  phys=R{phys}  '
              f'(R29.3 extended first_def to 0 because direct_ldc_params={bool(res.direct_ldc_params)})')
    print()

    # The bug hypothesis: R29.3 extended %r1 (tid.x) first_def to 0, so %r1
    # was allocated early.  Did %r2 (ctaid.x, also R29.3-extended) get
    # allocated to the SAME phys as %r3 (IMAD.SHL dest)?  Or did %r1 / %r2
    # collide with IMAD's dest or MOV's source?
    #
    # Specifically look at: does any vreg's live range end AT or after the
    # IMAD.SHL, share phys with IMAD.SHL dest (%r3)?
    if '%r3' in res.ra.int_regs:
        r3_phys = res.ra.int_regs['%r3']
        r3_fd = reg_first_def.get('%r3', -1)
        overlapping = []
        for vname, phys in res.ra.int_regs.items():
            if vname == '%r3':
                continue
            if phys != r3_phys and (not _is_64(fn, vname) or phys + 1 != r3_phys):
                # Also check if vname is a 64-bit pair covering r3_phys
                continue
            vfd = reg_first_def.get(vname, -1)
            vlu = reg_last_use.get(vname, -1)
            # Overlap if intervals intersect
            if vfd is not None and vlu is not None and vfd <= r3_fd <= vlu:
                overlapping.append((vname, vfd, vlu))
            elif r3_fd <= vfd and vfd <= reg_last_use.get('%r3', r3_fd):
                overlapping.append((vname, vfd, vlu))
        print('=== %r3 (IMAD.SHL dest) phys-reg overlap check ===')
        print(f'  %r3 at R{r3_phys}, live=[{r3_fd}..{reg_last_use.get("%r3", "?")}]')
        if overlapping:
            print(f'  OVERLAP DETECTED:')
            for v, f, l in overlapping:
                print(f'    {v} also at R{r3_phys}, live=[{f}..{l}]')
        else:
            print(f'  No overlap with other vregs.')
    print()


def _is_64(fn, vname):
    from ptx.ir import ScalarKind
    bare = vname.lstrip('%')
    for rd in fn.reg_decls:
        if bare in [n.lstrip('%') for n in rd.names]:
            return rd.type.width >= 64
    return False


if __name__ == '__main__':
    main()
