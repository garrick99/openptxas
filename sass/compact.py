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

    # --- Shared memory (32-bit forms) ---
    # LDS opcode 0x984 has two address modes:
    #   - encode_lds: byte[3]=0xFF (filtered by < 254), byte[4]=ur_addr (UR)
    #   - encode_lds_r: byte[3]=addr_reg (GPR), byte[4]=0
    # Both forms safely handled by declaring byte[2]=dst, byte[3]=addr (RZ filtered).
    0x984: [(2, 'dst'), (3, 'addr')],                             # LDS / LDS.R (32-bit shared load)

    # STS opcode 0x988 (UR-addressed): byte[3]=0xFF, byte[4]=data, byte[8]=ur_addr
    # STS opcode 0x388 (GPR-addressed): byte[3]=addr, byte[4]=data
    0x988: [(4, 'data')],                                         # STS UR-addressed
    0x388: [(3, 'addr'), (4, 'data')],                            # STS R-addressed

    # --- Barriers (no GPR fields) ---
    0xb1d: [],  # BAR.SYNC
    0x941: [],  # BSYNC

    # --- Float ALU (32-bit) ---
    0x221: [(2, 'dst'), (3, 'src0'), (4, 'src1')],                # FADD
    0x223: [(2, 'dst'), (3, 'src0'), (4, 'src1'), (8, 'src2')],   # FFMA / FMUL (FFMA with src2)
    0x209: [(2, 'dst'), (3, 'src0'), (4, 'src1')],                # FMNMX
    0x20b: [(3, 'src0'), (4, 'src1')],                            # FSETP (predicate dst)
    0x820: [(2, 'dst'), (3, 'src0')],                             # FMUL.IMM (src1 is imm, no src2)
    0x808: [(2, 'dst'), (3, 'src0')],                             # FSEL.imm (src1 is imm)
    0x80a: [(2, 'dst'), (3, 'src0')],                             # FSEL.step (src1 is imm)

    # --- Integer ALU extras ---
    0x224: [(2, 'dst'), (3, 'src0'), (4, 'src1'), (8, 'src2')],   # IMAD R-R-R-R
    0x207: [(2, 'dst'), (3, 'src0'), (4, 'src1')],                # SEL (selp)
    0x248: [(2, 'dst'), (3, 'src0'), (4, 'src1')],                # VIMNMX (imax/imin)
    0x226: [(2, 'dst'), (3, 'src0'), (4, 'src1'), (8, 'src2')],   # IDP.4A

    # --- FP64 ---
    0x208: [(2, 'dst'), (3, 'src0'), (4, 'src1')],                # FSEL (f64 selp path)
    0x228: [(2, 'dst'), (3, 'src0'), (4, 'src1')],                # DMUL
    0x229: [(2, 'dst'), (3, 'src0'), (8, 'src1')],                # DADD (src1 at byte[8])
    0x22a: [(3, 'src0'), (4, 'src1')],                            # DSETP (src1 at byte[4])

    # --- Single-source converts / special functions (src at byte[4], byte[3] unused) ---
    0x245: [(2, 'dst'), (4, 'src0')],                             # I2FP
    0x304: [(2, 'dst'), (4, 'src0')],                             # CVT.F16.F32
    0x305: [(2, 'dst'), (4, 'src0')],                             # F2I
    0x306: [(2, 'dst'), (4, 'src0')],                             # I2F
    0x308: [(2, 'dst'), (4, 'src0')],                             # MUFU
    0x309: [(2, 'dst'), (4, 'src0')],                             # POPC
    0x300: [(2, 'dst'), (4, 'src0')],                             # FLO
    0x301: [(2, 'dst'), (4, 'src0')],                             # BREV
    0x310: [(2, 'dst'), (4, 'src0')],                             # F2F.F64.F32

    # PRMT has imm at byte[4-7]; sources at byte[3] and byte[8]
    0x416: [(2, 'dst'), (3, 'src0'), (8, 'src2')],                # PRMT

    # --- Predicated MOV / MOV R,UR ---
    # @P MOV R,R: byte[2]=dst, byte[4]=src (byte[3] unused)
    0x202: [(2, 'dst'), (4, 'src0')],                             # @P MOV
    0xc02: [(2, 'dst')],                                          # MOV R, UR

    # --- Vote / redux ---
    0x806: [(2, 'dst')],                                          # VOTE (src is predicate)
    0x3c4: [(3, 'src0')],                                         # REDUX (dst is UR)

    # --- Memory barriers (no GPR fields) ---
    0x91a: [],  # DEPBAR.LE
    0x992: [],  # MEMBAR
    0x9af: [],  # LDGDEPBAR

    # --- Atomics ---
    0x9a3: [(2, 'dst'), (3, 'addr64'), (4, 'data')],              # ATOMG.E.ADD
    0x9a8: [(2, 'dst'), (3, 'addr64'), (4, 'data')],              # ATOMG.E.MIN/MAX
    0x3a9: [(2, 'dst64'), (3, 'addr64'), (4, 'data64'), (8, 'swap64')],  # ATOMG.E.CAS.64

    # --- BRA variants (no GPR fields) ---
    0x547: [],  # BRA (predicated/relative variant)

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


def kernel_is_compactable(sass_instrs: list) -> tuple[bool, set[int]]:
    """Check if all instructions in the kernel have field metadata.

    Returns (covered, uncovered_opcodes).
    covered is True iff uncovered_opcodes is empty.
    """
    uncovered: set[int] = set()
    for si in sass_instrs:
        op = opcode_of(si.raw)
        if op not in GPR_FIELDS:
            uncovered.add(op)
    return (len(uncovered) == 0), uncovered


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
    n_insts_changed = 0
    n_fields_rewritten = 0
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
                n_fields_rewritten += 1

        if not changed:
            new_instrs.append(si)
            continue

        # Update comment to reflect new register numbers
        comment = si.comment or ''
        for old in sorted(remap.keys(), reverse=True):
            new = remap[old]
            if new != old:
                comment = re.sub(rf'R{old}(?!\d)', f'R{new}', comment)

        new_instrs.append(SassInstr(bytes(raw), comment))
        n_insts_changed += 1

    return new_instrs, n_insts_changed, n_fields_rewritten


class CompactReport:
    """Per-kernel compaction report (Phase 0 of FB-4.3)."""
    def __init__(self, kernel_name: str):
        self.kernel_name = kernel_name
        self.attempted = False      # did we try compaction?
        self.covered = False        # all opcodes have metadata?
        self.uncovered: set[int] = set()
        self.regs_before = 0
        self.regs_after = 0
        self.sass_before = 0
        self.sass_after = 0
        self.total_insts = 0
        self.compacted_insts = 0
        self.gpr_fields_rewritten = 0
        self.correct: str = 'unknown'  # 'PASS', 'FAIL', 'unknown'

    def format(self) -> str:
        lines = [f"[compact] kernel={self.kernel_name}"]
        lines.append(f"  attempted: {'yes' if self.attempted else 'no'}")
        lines.append(f"  covered: {'yes' if self.covered else 'no'}")
        if self.uncovered:
            opcode_list = ', '.join(f'0x{op:03x}' for op in sorted(self.uncovered))
            lines.append(f"  uncovered: [{opcode_list}]")
        else:
            lines.append(f"  uncovered: []")
        lines.append(f"  regs: {self.regs_before} -> {self.regs_after}")
        lines.append(f"  sass: {self.sass_before} -> {self.sass_after}")
        lines.append(f"  total_insts: {self.total_insts}")
        lines.append(f"  compacted_insts: {self.compacted_insts}")
        lines.append(f"  gpr_fields_rewritten: {self.gpr_fields_rewritten}")
        lines.append(f"  correct: {self.correct}")
        return '\n'.join(lines)


def compact(sass_instrs: list,
            verbose: bool = False,
            kernel_name: str = '<unknown>',
            report: 'CompactReport | None' = None) -> tuple[list, int]:
    """Run field-safe register compaction on a kernel.

    Returns (new_sass_instrs, new_num_gprs).
    If coverage gating fails, returns sass_instrs unchanged.

    If `report` is provided, fills it with per-kernel diagnostics.
    """
    if report is None:
        report = CompactReport(kernel_name)
    else:
        report.kernel_name = kernel_name

    report.total_insts = len(sass_instrs)
    report.sass_before = len(sass_instrs)
    report.sass_after = len(sass_instrs)  # compaction doesn't change SASS count

    # Coverage gating
    covered, uncovered = kernel_is_compactable(sass_instrs)
    report.covered = covered
    report.uncovered = uncovered

    # Baseline register count (what we'd have without compaction)
    used_pre, _ = collect_used_gprs(sass_instrs)
    report.regs_before = (max(used_pre) + 1) if used_pre else 0
    report.regs_after = report.regs_before  # default: no change

    if not covered:
        report.attempted = False
        if verbose:
            print(report.format())
        return sass_instrs, report.regs_before

    # Covered — attempt compaction
    report.attempted = True
    used, pair_bases = collect_used_gprs(sass_instrs)
    if not used:
        if verbose:
            print(report.format())
        return sass_instrs, 0

    remap = build_dense_remap(used, pair_bases)
    max_before = max(used)
    max_after = max(remap.values())

    if max_after == max_before:
        # Already dense — no rewrite needed. Return 0 to signal "no rewrite"
        # so the pipeline uses allocator high-water (compaction's view may
        # miss registers in opcodes without explicit metadata).
        report.regs_after = max_before + 1
        if verbose:
            print(report.format())
        return sass_instrs, 0

    new_instrs, n_insts, n_fields = apply_remap(sass_instrs, remap)
    report.compacted_insts = n_insts
    report.gpr_fields_rewritten = n_fields
    report.regs_after = max_after + 1

    if verbose:
        print(report.format())
    return new_instrs, max_after + 1
