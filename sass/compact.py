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
    # F2F has width-dispatched operands via byte[9]:
    #   raw[9] = 0x18 → F2F.F64.F32 (widening): dst is f64 pair, src is f32
    #   raw[9] = 0x10 → F2F.F32.F64 (narrowing): dst is f32, src is f64 pair
    # Use named fields handled in collect_used_gprs.
    0x310: [(2, 'f2f_dst'), (4, 'f2f_src')],

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

    # --- Async copy (cp.async) ---
    # LDGSTS.E [smem_addr], desc[UR][glob_addr.64]
    # byte[2] = smem_addr (32-bit GPR), byte[3] = glob_addr (64-bit pair base)
    # byte[8] = ur_desc (UR, not GPR). bytes[4-7] = unused.
    0xfae: [(2, 'smem_addr'), (3, 'addr64')],                     # LDGSTS.E (cp.async)

    # --- Tensor core: HMMA (FP16 matrix multiply-accumulate) ---
    # Opcode 0x23c covers multiple shapes:
    #   byte[9] = 0x10 → m16n8k8 (A: 2 regs, B: 1 reg)
    #   byte[9] = 0x18 → m16n8k16 (A: 4 regs, B: 2 regs)
    # byte[2] = dst (4-register QUAD group: dst..dst+3)
    # byte[3] = src_a (k8: 2 regs, k16: 4 regs) — dispatched via byte[9]
    # byte[4] = src_b (k8: 1 reg, k16: 2 regs) — dispatched via byte[9]
    # byte[8] = src_c (4-register QUAD group: src_c..src_c+3)
    # Width-dependent fields use special names handled in collect_used_gprs.
    0x23c: [(2, 'dst_quad'), (3, 'hmma_a'), (4, 'hmma_b'), (8, 'src_c_quad')],

    # --- Tensor core: IMMA (INT8 matrix multiply-accumulate) ---
    # Opcode 0x237 covers IMMA.16832.S8.S8 (only encoded form):
    #   m16n8k32, A: 4 INT8 regs, B: 2 INT8 regs, C/D: 4 INT32 regs
    #   byte[9]=0x5c (shape+type), byte[10]=0x40 (signed flag)
    # byte[2] = dst (4-register QUAD group: dst..dst+3)
    # byte[3] = src_a (4-register QUAD group: src_a..src_a+3, INT8 packed)
    # byte[4] = src_b (2-register PAIR: src_b..src_b+1, INT8 packed)
    # byte[8] = src_c (4-register QUAD group: src_c..src_c+3)
    # All groups have fixed widths (no shape dispatch).
    0x237: [(2, 'dst_quad'), (3, 'src_a_quad'), (4, 'src_b_pair'), (8, 'src_c_quad')],

    # --- Tensor core: DMMA (FP64 matrix multiply-accumulate) ---
    # Opcode 0x23f covers DMMA.8x8x4 (m8n8k4, single encoded form on SM_120):
    #   A: 1 .f64 = 2 GPRs (pair), B: 1 .f64 = 2 GPRs (pair)
    #   C/D: 2 .f64 = 4 GPRs (quad)
    #   raw[9]=0x00, raw[10]=0x00 (no shape modifier, single form)
    # byte[2] = dst   (4-register QUAD, 4-aligned)
    # byte[3] = src_a (2-register PAIR, 2-aligned)
    # byte[4] = src_b (2-register PAIR, 2-aligned)
    # byte[8] = src_c (4-register QUAD, 4-aligned)
    # Ground truth: DMMA.8x8x4 R8, R2, R4, R8
    0x23f: [(2, 'dst_quad'), (3, 'src_a_pair'), (4, 'src_b_pair'), (8, 'src_c_quad')],

    # --- Tensor core: QMMA (FP8 matrix multiply-accumulate, SM_120 only) ---
    # Opcode 0x27a covers QMMA.16832.F32 (m16n8k32 FP8 → FP32 accumulate):
    #   byte[9] = 0x2c → E4M3.E4M3
    #   byte[9] = 0xec → E5M2.E5M2
    # Both formats have identical operand widths (no shape dispatch).
    # Hardware constraints:
    #   D: 4-register quad, 4-aligned
    #   A: 4-register quad, 4-aligned
    #   B: 2-register pair, base must be < 8
    #   C: 4-register quad (or RZ=255 for no accumulation, filtered by reg<254)
    # byte[2] = dst   (4-register QUAD, 4-aligned)
    # byte[3] = src_a (4-register QUAD, 4-aligned)
    # byte[4] = src_b (2-register PAIR, 2-aligned)
    # byte[8] = src_c (4-register QUAD, 4-aligned) [or RZ]
    0x27a: [(2, 'dst_quad'), (3, 'src_a_quad'), (4, 'src_b_pair'), (8, 'src_c_quad')],

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
    """Field is a 2-register group: 64-bit lo/hi pair OR vector pair (e.g., MMA src_b)."""
    return '64' in name or 'pair' in name


def _is_quad_field(name: str) -> bool:
    return 'quad' in name


def _ldg_stg_data_is_64(raw: bytes) -> bool:
    """For LDG/STG opcodes, byte[9] indicates data width: 0x19=32, 0x1b=64, 0x1d=128."""
    b9 = raw[9]
    return b9 in (0x1b, 0x1d)  # 64-bit or 128-bit


def _hmma_field_width(raw: bytes, name: str) -> int:
    """Return the number of registers for an HMMA field based on shape.

    HMMA opcode 0x23c has two shape variants:
      byte[9] = 0x10 → m16n8k8  (A: 2 regs, B: 1 reg)
      byte[9] = 0x18 → m16n8k16 (A: 4 regs, B: 2 regs)
    """
    b9 = raw[9]
    if name == 'hmma_a':
        return 4 if b9 == 0x18 else 2
    if name == 'hmma_b':
        return 2 if b9 == 0x18 else 1
    return 1


def _f2f_field_width(raw: bytes, name: str) -> int:
    """Return the number of registers for an F2F field based on shape.

    F2F opcode 0x310 has two width variants:
      byte[9] = 0x18 → F2F.F64.F32 (widening): dst is f64 pair, src is f32
      byte[9] = 0x10 → F2F.F32.F64 (narrowing): dst is f32, src is f64 pair
    """
    b9 = raw[9]
    if name == 'f2f_dst':
        return 2 if b9 == 0x18 else 1
    if name == 'f2f_src':
        return 2 if b9 == 0x10 else 1
    return 1


def collect_used_gprs(sass_instrs: list) -> tuple[set[int], set[int], set[int]]:
    """Collect all GPR indices actually referenced.

    Returns (used_regs, pair_bases, quad_bases):
      - used_regs: all registers referenced (including implicit hi halves and
        quad-group members)
      - pair_bases: even-aligned bases of 64-bit pairs (implicit +1)
      - quad_bases: 4-aligned bases of 4-register groups (implicit +1, +2, +3)
    """
    used: set[int] = set()
    pair_bases: set[int] = set()
    quad_bases: set[int] = set()
    for si in sass_instrs:
        op = opcode_of(si.raw)
        fields = GPR_FIELDS.get(op, [])
        for byte_pos, name in fields:
            reg = si.raw[byte_pos]
            if reg >= 254:
                continue
            used.add(reg)

            # Quad group: 4 consecutive registers
            if _is_quad_field(name):
                used.add(reg + 1)
                used.add(reg + 2)
                used.add(reg + 3)
                if reg % 4 == 0:
                    quad_bases.add(reg)
                continue

            # 64-bit pair
            is_64 = False
            if _is_64bit_field(name):
                is_64 = True
            elif name in ('dst_var', 'data_var') and op in (0x981, 0x986):
                is_64 = _ldg_stg_data_is_64(si.raw)
            if is_64:
                used.add(reg + 1)
                if reg % 2 == 0:
                    pair_bases.add(reg)
                continue

            # HMMA width-dependent fields (hmma_a, hmma_b)
            if name in ('hmma_a', 'hmma_b') and op == 0x23c:
                width = _hmma_field_width(si.raw, name)
                for i in range(1, width):
                    used.add(reg + i)
                # Track as a group so the base stays aligned
                # (A and B don't require strict quad alignment, but keeping
                # the group consecutive is essential)
                if width >= 2:
                    pair_bases.add(reg)  # at least pair alignment
                if width == 4 and reg % 4 == 0:
                    quad_bases.add(reg)
                continue

            # F2F width-dependent fields (f2f_dst, f2f_src)
            if name in ('f2f_dst', 'f2f_src') and op == 0x310:
                width = _f2f_field_width(si.raw, name)
                if width == 2:
                    used.add(reg + 1)
                    if reg % 2 == 0:
                        pair_bases.add(reg)
                continue
    return used, pair_bases, quad_bases


def build_dense_remap(used_regs: set[int],
                      pair_bases: set[int],
                      quad_bases: set[int] = None,
                      preserve: set[int] = None) -> dict[int, int]:
    """Build a remap from old register indices to dense new indices.

    R0 and R1 are always preserved.
    Quad bases are kept as 4 consecutive regs, 4-aligned.
    Pair bases are kept as 2 consecutive regs, 2-aligned.
    """
    if quad_bases is None:
        quad_bases = set()
    if preserve is None:
        preserve = {0, 1}

    remap: dict[int, int] = {r: r for r in preserve}
    to_compact = sorted(r for r in used_regs if r not in preserve)

    # Precompute: which registers are hi-halves / quad-followers
    # so we skip them (they get assigned together with their base).
    followers: set[int] = set()
    for base in quad_bases:
        for i in (1, 2, 3):
            if (base + i) in used_regs:
                followers.add(base + i)
    for base in pair_bases:
        if base in quad_bases:
            continue  # already handled above
        if (base + 1) in used_regs:
            followers.add(base + 1)

    next_slot = 2
    for r in to_compact:
        if r in remap or r in followers:
            continue
        if r in quad_bases:
            # Align to 4
            if next_slot % 4 != 0:
                next_slot = (next_slot + 3) & ~3
            remap[r] = next_slot
            for i in (1, 2, 3):
                if (r + i) in used_regs:
                    remap[r + i] = next_slot + i
            next_slot += 4
        elif r in pair_bases:
            # Align to 2
            if next_slot % 2 != 0:
                next_slot += 1
            remap[r] = next_slot
            if (r + 1) in used_regs:
                remap[r + 1] = next_slot + 1
            next_slot += 2
        else:
            # Singleton
            remap[r] = next_slot
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
    used_pre, _, _ = collect_used_gprs(sass_instrs)
    report.regs_before = (max(used_pre) + 1) if used_pre else 0
    report.regs_after = report.regs_before  # default: no change

    if not covered:
        report.attempted = False
        if verbose:
            print(report.format())
        return sass_instrs, report.regs_before

    # Covered — attempt compaction
    report.attempted = True
    used, pair_bases, quad_bases = collect_used_gprs(sass_instrs)
    if not used:
        if verbose:
            print(report.format())
        return sass_instrs, 0

    remap = build_dense_remap(used, pair_bases, quad_bases)
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
