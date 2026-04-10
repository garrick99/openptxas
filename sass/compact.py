"""
FB-4.2: Field-safe register compaction.

Opcode-indexed metadata table describing which bytes encode GPR operands
for each instruction form. Only opcodes with explicit metadata are eligible
for compaction. Kernels using any non-covered GPR-touching opcode are
skipped entirely.

Coverage target (Pass 1): opcodes appearing in reduce_sum.
"""
from __future__ import annotations

# Each entry is a list of (byte_position, field_name) for GPR operand slots.
# Non-GPR operands (UR, immediate, predicate) are omitted.
#
# opcode = low 12 bits of (byte[0] | byte[1] << 8)
#
# field_name semantics:
#   "dst", "src0", "src1", "src2" = 32-bit GPR field
#   "dst64", "src0_64", etc. = 64-bit pair base (lo, lo+1)
#   "addr64" = 64-bit address pair base
#   "data64" = 64-bit data pair base
#
# Pair bases must be even-aligned and remapped together with their hi half.
GPR_FIELDS: dict[int, list[tuple[int, str]]] = {
    # --- Arithmetic ---
    0x210: [(2, 'dst'), (3, 'src0'), (4, 'src1'), (8, 'src2')],  # IADD3 / IADD3.X / MOV (via IADD3)
    0x810: [(2, 'dst'), (3, 'src0'), (8, 'src2')],               # IADD3.IMM32 (src1 = immediate, not GPR)
    0x212: [(2, 'dst'), (3, 'src0'), (4, 'src1'), (8, 'src2')],  # LOP3.LUT
    0x819: [(2, 'dst'), (3, 'src0'), (8, 'src2')],               # SHF.R.U32 / SHF.R.U32.HI (src1 = shift imm)

    # --- Compare ---
    0x20c: [(3, 'src0'), (4, 'src1')],                            # ISETP.*.R-R (dst is predicate)
    0x80c: [(3, 'src0')],                                         # ISETP.*.IMM (src0 is GPR, src1 is imm)
    0xc0c: [(3, 'src0')],                                         # ISETP.*.R-UR (src1 is UR, not GPR)

    # --- Multiply (WIDE writes 64-bit pair to dst) ---
    0x825: [(2, 'dst64'), (3, 'src0'), (8, 'src2_64')],          # IMAD.WIDE imm: dst is pair, src2 is pair (accumulator)
    0x225: [(2, 'dst64'), (3, 'src0'), (4, 'src1'), (8, 'src2_64')],  # IMAD.WIDE R-R-R-R
    0x824: [(2, 'dst'), (3, 'src0'), (8, 'src2')],               # IMAD.SHL.U32 (32-bit, src1 is shift imm)
    0xc24: [(2, 'dst'), (3, 'src0'), (8, 'src2')],               # IMAD R-R-UR-R (src1 is UR)

    # --- 64-bit ops with UR operand ---
    0xc35: [(2, 'dst64'), (3, 'src0_64')],                       # IADD.64 R-UR: both are pairs

    # --- Shuffle (32-bit only) ---
    0xf89: [(2, 'dst'), (3, 'src')],                              # SHFL imm-imm variant

    # --- Memory ---
    # LDG/STG dst/data width depends on byte[9]: 0x19=32, 0x1b=64, 0x1d=128
    # We treat addr as always 64-bit (it's an address pair).
    0x981: [(2, 'dst_var'), (3, 'addr64')],                       # LDG.E (dst width from byte[9])
    0x986: [(3, 'addr64'), (4, 'data_var')],                      # STG.E (data width from byte[9])

    # --- Special reg read ---
    0x919: [(2, 'dst')],                                          # S2R (dst GPR, src is SR code)

    # --- Const bank read ---
    0xb82: [(2, 'dst')],                                          # LDC (src is constant bank offset)

    # --- Pass-through (no GPR fields) ---
    0x918: [],  # NOP
    0x94d: [],  # EXIT
    0x947: [],  # BRA
    0x7ac: [],  # LDCU.64 (dst is UR, not GPR)
    0x9c3: [],  # S2UR (dst is UR, not GPR)
}

# Opcodes known to touch GPRs but NOT covered — kernels using these should
# NOT be compacted (we'd miss some register fields).
# Any opcode used in SASS that isn't in GPR_FIELDS and isn't in this
# exclusion set will cause coverage gating to reject the kernel.


def opcode_of(raw: bytes) -> int:
    """Extract the 12-bit opcode from a 16-byte SASS instruction."""
    return (raw[0] | (raw[1] << 8)) & 0xFFF


def kernel_is_compactable(sass_instrs: list) -> tuple[bool, int | None]:
    """Check if all instructions in the kernel have field metadata.

    Returns (True, None) if fully covered, else (False, blocking_opcode).
    """
    for si in sass_instrs:
        op = opcode_of(si.raw)
        if op not in GPR_FIELDS:
            return False, op
    return True, None


def _is_64bit_field(name: str) -> bool:
    return '64' in name


def _ldg_stg_data_is_64(raw: bytes) -> bool:
    """For LDG/STG opcodes, byte[9] indicates data width: 0x19=32, 0x1b=64, 0x1d=128."""
    b9 = raw[9]
    return b9 in (0x1b, 0x1d)  # 64-bit or 128-bit


def collect_used_gprs(sass_instrs: list) -> tuple[set[int], set[int]]:
    """Collect all GPR indices actually referenced.

    Returns (used_regs, pair_bases) where pair_bases is the set of register
    indices that are even-aligned bases of 64-bit pairs (their hi half is
    implicitly used at base+1).
    """
    used: set[int] = set()
    pair_bases: set[int] = set()
    for si in sass_instrs:
        op = opcode_of(si.raw)
        fields = GPR_FIELDS.get(op, [])
        for byte_pos, name in fields:
            reg = si.raw[byte_pos]
            if reg >= 254:
                continue
            # Determine if this field is 64-bit
            is_64 = False
            if _is_64bit_field(name):
                is_64 = True
            elif name in ('dst_var', 'data_var') and op in (0x981, 0x986):
                is_64 = _ldg_stg_data_is_64(si.raw)
            used.add(reg)
            if is_64:
                used.add(reg + 1)  # implicit hi half
                if reg % 2 == 0:
                    pair_bases.add(reg)
    return used, pair_bases


def build_dense_remap(used_regs: set[int],
                      pair_bases: set[int],
                      preserve: set[int] = None) -> dict[int, int]:
    """Build a remap from old register indices to dense new indices.

    R0 and R1 are always preserved.
    Pair bases (from opcode metadata) are kept consecutive and even-aligned.
    """
    if preserve is None:
        preserve = {0, 1}

    remap: dict[int, int] = {r: r for r in preserve}
    to_compact = sorted(r for r in used_regs if r not in preserve)

    next_slot = 2
    assigned: set[int] = set()
    for r in to_compact:
        if r in assigned:
            continue
        if r in pair_bases:
            # Pair: assign even-aligned, take both lo and hi
            if next_slot % 2 != 0:
                next_slot += 1
            remap[r] = next_slot
            if (r + 1) in used_regs:
                remap[r + 1] = next_slot + 1
                assigned.add(r + 1)
            assigned.add(r)
            next_slot += 2
        elif (r - 1) in pair_bases and (r - 1) in assigned:
            # Already handled as pair hi
            continue
        else:
            # Singleton: take next slot
            remap[r] = next_slot
            assigned.add(r)
            next_slot += 1
    return remap


def apply_remap(sass_instrs: list, remap: dict[int, int]) -> tuple[list, int]:
    """Rewrite register fields in each instruction using the remap table.

    Returns (new_sass_instrs, num_changed).
    """
    from sass.isel import SassInstr
    import re

    new_instrs = []
    n_changed = 0
    for si in sass_instrs:
        op = opcode_of(si.raw)
        fields = GPR_FIELDS.get(op, [])
        if not fields:
            new_instrs.append(si)
            continue

        raw = bytearray(si.raw)
        changed = False
        for byte_pos, _name in fields:
            old = raw[byte_pos]
            if old in remap and remap[old] != old:
                raw[byte_pos] = remap[old]
                changed = True

        if not changed:
            new_instrs.append(si)
            continue

        # Update comment to reflect new register numbers
        comment = si.comment or ''
        # Replace in descending order of old reg id to avoid substring collisions
        for old in sorted(remap.keys(), reverse=True):
            new = remap[old]
            if new != old:
                comment = re.sub(rf'R{old}(?!\d)', f'R{new}', comment)

        new_instrs.append(SassInstr(bytes(raw), comment))
        n_changed += 1

    return new_instrs, n_changed


def compact(sass_instrs: list, verbose: bool = False) -> tuple[list, int]:
    """Run field-safe register compaction on a kernel.

    Returns (new_sass_instrs, new_num_gprs).
    If coverage gating fails, returns (sass_instrs, current_max + 1) unchanged.
    """
    ok, blocking = kernel_is_compactable(sass_instrs)
    if not ok:
        if verbose:
            print(f"[compact] coverage gate blocked: opcode 0x{blocking:03x}")
        used, _ = collect_used_gprs(sass_instrs)
        return sass_instrs, (max(used) + 1 if used else 0)

    used, pair_bases = collect_used_gprs(sass_instrs)
    if not used:
        return sass_instrs, 0

    remap = build_dense_remap(used, pair_bases)
    max_before = max(used)
    max_after = max(remap.values())

    if max_after == max_before:
        return sass_instrs, max_before + 1

    new_instrs, n_changed = apply_remap(sass_instrs, remap)
    if verbose:
        print(f"[compact] used={len(used)} pairs={len(pair_bases)} "
              f"max {max_before}->{max_after} (changed {n_changed} instrs)")
    return new_instrs, max_after + 1
