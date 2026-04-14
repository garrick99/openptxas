"""
sass/schedule.py — Instruction scheduling and ctrl word assignment for SM_120.

The ctrl word (23 bits) in each 128-bit instruction encodes dependency barriers:
  bits[22:17] = stall count (0..63 cycles)
  bits[16]    = yield hint
  bits[15]    = write-after-read barrier
  bits[14:10] = read-after-write barrier mask (rbar, 5 bits)
  bits[9:4]   = write dependency slot (wdep, 6 bits)
  bits[3:0]   = scoreboard set (misc, 4 bits)

Key observations from ptxas RE:
  - LDG sets a scoreboard entry (via misc field) when it starts loading
  - ALL consumers of LDG output registers need rbar=0x09 to wait for the load
  - LDC/LDCU between LDG and consumers use rbar=0x01 (no LDG wait needed)
  - STG needs rbar=0x03
  - wdep encodes write-dependency tracking (varies by instruction)
"""

from __future__ import annotations
import struct
from dataclasses import dataclass, field
from enum import Enum

from sass.isel import SassInstr
from sass.encoding.sm_120_opcodes import encode_nop as _encode_nop


# ---------------------------------------------------------------------------
# FG-2.5 — proof object model for verify_schedule.
#
# The verifier used to return a list of "I noticed a hazard" strings, with
# suppressions for known false positives.  That model could prove
# "nothing looks wrong" but could never prove "this is safe under the
# model."  FG-2.5 upgrades it to constructive classification: every
# candidate producer→consumer edge is positively tagged with WHY it is
# safe (or flagged as a VIOLATION).  `verify_schedule` preserves its
# legacy list-of-strings API by formatting the VIOLATION edges from a
# ProofReport.
# ---------------------------------------------------------------------------

class ProofClass(str, Enum):
    """Classification of a single producer→consumer edge."""
    LATENCY_INERT          = "LATENCY_INERT"          # writer has no min_gpr_gap in model
    FORWARDING_SAFE        = "FORWARDING_SAFE"        # pair on _FORWARDING_SAFE_PAIRS list
    CTRLWORD_SAFE          = "CTRLWORD_SAFE"          # reserved / legacy (unused after FG-3.3)
    GAP_SAFE               = "GAP_SAFE"               # ALU: instruction-stream gap ≥ min_gpr_gap
    ZERO_INIT_SAFE         = "ZERO_INIT_SAFE"         # FG-4.1: semantic fastpath (HFMA2 zero-init)
    # FG-3.1 memory latency classes (GPR memory producers)
    MEMORY_SCOREBOARD_SAFE = "MEMORY_SCOREBOARD_SAFE"  # rbar wait bit proven in window
    MEMORY_GAP_SAFE        = "MEMORY_GAP_SAFE"         # P3-2: gap exceeds producer latency
    MEMORY_INERT           = "MEMORY_INERT"            # producer wdep no-track (FG-3.2)
    MEMORY_VIOLATION       = "MEMORY_VIOLATION"        # memory read with no rbar wait
    # FG-3.2 / FG-3.3 UR-destination memory / system producers (rule R10)
    UR_MEMORY_SCOREBOARD_SAFE = "UR_MEMORY_SCOREBOARD_SAFE"
    UR_MEMORY_GAP_SAFE        = "UR_MEMORY_GAP_SAFE"      # FG-3.3: LDCU.64 gap >= _LDCU_GAP_SAFE_MIN
    UR_MEMORY_INERT           = "UR_MEMORY_INERT"         # FG-3.3: producer wdep no-track
    UR_MEMORY_VIOLATION       = "UR_MEMORY_VIOLATION"
    VIOLATION              = "VIOLATION"               # ALU / general violation


@dataclass(frozen=True)
class ProofEdge:
    """A single classified producer→consumer dependency edge.

    Fields are deliberately minimal so ProofEdge is cheap to
    construct and compare.  `regs` is a frozenset so edges are
    hashable for set membership tests.
    """
    writer_idx: int
    reader_idx: int
    writer_opc: int
    reader_opc: int
    regs: frozenset[int]
    gap: int
    classification: ProofClass
    rationale: str

    @property
    def writer_name(self) -> str:
        """Pretty name for the writer, from _OPCODE_META if known."""
        from sass.scoreboard import _OPCODE_META
        meta = _OPCODE_META.get(self.writer_opc)
        if meta is not None:
            return meta.name
        return f"opc=0x{self.writer_opc:03x}"

    @property
    def reader_name(self) -> str:
        from sass.scoreboard import _OPCODE_META
        meta = _OPCODE_META.get(self.reader_opc)
        if meta is not None:
            return meta.name
        return f"opc=0x{self.reader_opc:03x}"

    def legacy_str(self) -> str:
        """Format for backward-compatible verify_schedule output.

        Mirrors the pre-FG-2.5 string format so existing callers and
        regex parsers (e.g. FG-2.3 INV C) keep working.
        """
        if self.writer_opc == 0x7ac:
            # LDCU.64 pre-boundary gap violation
            return (f"LDCU.64 UR{self.regs and min(self.regs) or '?'} "
                    f"at [{self.writer_idx}] → consumer "
                    f"opc=0x{self.reader_opc:03x} at [{self.reader_idx}]: "
                    f"gap={self.gap} (need ≥3)")
        return (f"{self.writer_name} at [{self.writer_idx}] writes "
                f"{set(self.regs)} → opc=0x{self.reader_opc:03x} "
                f"at [{self.reader_idx}] reads immediately (0-gap RAW)")


@dataclass
class ProofReport:
    """Aggregate proof output for one instruction stream.

    Holds every classified edge plus a per-class count and the final
    safe/unsafe verdict.  `violations` is a convenience view of the
    edges whose classification is VIOLATION.
    """
    edges: list[ProofEdge] = field(default_factory=list)

    @property
    def counts(self) -> dict[ProofClass, int]:
        c = {cls: 0 for cls in ProofClass}
        for e in self.edges:
            c[e.classification] += 1
        return c

    @property
    def violations(self) -> list[ProofEdge]:
        return [e for e in self.edges
                if e.classification in (ProofClass.VIOLATION,
                                        ProofClass.MEMORY_VIOLATION,
                                        ProofClass.UR_MEMORY_VIOLATION)]

    @property
    def safe(self) -> bool:
        return len(self.violations) == 0

    @property
    def total(self) -> int:
        return len(self.edges)

    def verdict(self) -> str:
        return "SAFE" if self.safe else "UNSAFE"

    def summary_line(self, label: str = "") -> str:
        c = self.counts
        body = (
            f"total={self.total}  "
            f"INERT={c[ProofClass.LATENCY_INERT]}  "
            f"FWD={c[ProofClass.FORWARDING_SAFE]}  "
            f"GAP={c[ProofClass.GAP_SAFE]}  "
            f"ZI={c[ProofClass.ZERO_INIT_SAFE]}  "
            f"MSB={c[ProofClass.MEMORY_SCOREBOARD_SAFE]}  "
            f"MIN={c[ProofClass.MEMORY_INERT]}  "
            f"MVIOL={c[ProofClass.MEMORY_VIOLATION]}  "
            f"URSB={c[ProofClass.UR_MEMORY_SCOREBOARD_SAFE]}  "
            f"URGAP={c[ProofClass.UR_MEMORY_GAP_SAFE]}  "
            f"URIN={c[ProofClass.UR_MEMORY_INERT]}  "
            f"URVIOL={c[ProofClass.UR_MEMORY_VIOLATION]}  "
            f"VIOL={c[ProofClass.VIOLATION]}"
        )
        prefix = f"{label}: " if label else ""
        return f"{prefix}{self.verdict()}  {body}"


# Ctrl templates from ptxas ground truth
CTRL_LDC       = 0x7f1    # LDC.32
CTRL_LDCU      = 0x717    # LDCU.64
CTRL_LDC_64    = 0x712    # LDC.64 (default)
CTRL_LDC_64_AFTER_LDG = 0x711  # LDC.64 in the LDG latency-hiding slot
CTRL_LDG       = 0xf56    # LDG.E.64
CTRL_STG       = 0xff1    # STG.E.64
CTRL_EXIT      = 0x7f5    # EXIT
CTRL_BRA       = 0x7e0    # BRA
CTRL_NOP       = 0x7e0    # NOP

# Compute instructions: use 0x7e0 (matches ptxas NOP slots — safe default).
# The preamble's LDG ctrl sets proper barriers; compute just executes sequentially.
CTRL_COMPUTE = 0x7e0


def _get_opcode(raw: bytes) -> int:
    lo = struct.unpack_from('<Q', raw, 0)[0]
    return lo & 0xFFF


def _get_src_regs(raw: bytes) -> set[int]:
    """Extract source register indices that this instruction reads."""
    lo = struct.unpack_from('<Q', raw, 0)[0]
    opcode = lo & 0xFFF
    regs = set()
    src0 = raw[3]   # byte[3] = src0
    src1 = raw[8]   # byte[8] = src1
    b4 = raw[4]     # byte[4] = src1/imm depending on opcode

    if opcode == 0x981:  # LDG: src0 = address register
        if src0 < 255:
            regs.add(src0)
            regs.add(src0 + 1)  # 64-bit pair
    elif opcode == 0x986:  # STG: src0 = address (b3), data = b4 (NOT b8)
        if src0 < 255:
            regs.add(src0)
            regs.add(src0 + 1)
        if b4 < 255:
            regs.add(b4)
    elif opcode == 0x819:  # SHF: src0=byte[3], src1=byte[8]
        if src0 < 255:
            regs.add(src0)
        if src1 < 255:
            regs.add(src1)
    elif opcode == 0x235:  # IADD.64: src0=byte[3], src1=byte[4]
        if src0 < 255:
            regs.add(src0)
            regs.add(src0 + 1)
        if b4 < 255:
            regs.add(b4)
            regs.add(b4 + 1)
    elif opcode == 0x202:  # MOV: src=byte[4]
        if b4 < 255:
            regs.add(b4)
    return regs


def _get_dest_regs(raw: bytes) -> set[int]:
    """Extract destination register indices that this instruction writes."""
    lo = struct.unpack_from('<Q', raw, 0)[0]
    opcode = lo & 0xFFF
    dest = raw[2]
    regs = set()

    if opcode == 0x981:  # LDG — always 64-bit in our usage
        if dest < 255:
            regs.add(dest)
            regs.add(dest + 1)
    elif opcode == 0xb82:  # LDC
        b9 = raw[9]
        if dest < 255:
            regs.add(dest)
            if b9 == 0x0a:  # 64-bit
                regs.add(dest + 1)
    elif opcode == 0x235:  # IADD.64
        if dest < 255:
            regs.add(dest)
            regs.add(dest + 1)
    elif opcode in (0x819, 0x202):  # SHF, MOV
        if dest < 255:
            regs.add(dest)
    return regs


def _patch_ctrl(raw: bytes, ctrl: int) -> bytes:
    buf = bytearray(raw)
    raw24 = (ctrl & 0x7FFFFF) << 1
    buf[13] = raw24 & 0xFF
    buf[14] = (raw24 >> 8) & 0xFF
    # Preserve SHF.L.U64.HI reuse flag in bit 2 of byte[15]
    buf[15] = ((raw24 >> 16) & 0xFF) | (buf[15] & 0x04)
    return bytes(buf)


def _reorder_after_ldg(instrs: list[SassInstr]) -> list[SassInstr]:
    """Move independent LDC after LDG to hide latency.

    Only moves an LDC if its destination doesn't conflict with the LDG output
    OR with any instruction between LDG and the LDC's original position.
    If no moveable LDC is found, insert a NOP to provide minimum latency gap.
    """
    from sass.encoding.sm_120_opcodes import encode_nop
    result = list(instrs)
    i = 0
    while i < len(result) - 1:
        if _get_opcode(result[i].raw) == 0x981:  # LDG
            ldg_dests = _get_dest_regs(result[i].raw)
            if _get_opcode(result[i + 1].raw) != 0xb82:  # next isn't LDC
                moved = False
                for j in range(i + 2, len(result)):
                    if _get_opcode(result[j].raw) == 0xb82:
                        ldc_dests = _get_dest_regs(result[j].raw)
                        if ldc_dests & ldg_dests:
                            continue
                        # Also check: LDC dest must not conflict with any
                        # instruction between LDG+1 and LDC's original pos.
                        # Those instructions may write to the same registers.
                        conflict = False
                        for k in range(i + 1, j):
                            between_dests = _get_dest_regs(result[k].raw)
                            between_srcs = _get_src_regs(result[k].raw)
                            if ldc_dests & (between_dests | between_srcs):
                                conflict = True
                                break
                        if conflict:
                            continue
                        m = result.pop(j)
                        result.insert(i + 1, m)
                        moved = True
                        break
                if not moved:
                    # No moveable LDC — insert NOP for latency gap
                    result.insert(i + 1, SassInstr(encode_nop(), 'NOP  // LDG latency'))
        i += 1
    return result


def _hoist_ldcu64(instrs: list[SassInstr]) -> list[SassInstr]:
    """Hoist pre-boundary LDCU.64s to position 1; pad post-boundary ones with NOPs.

    SM_120 requires ≥3-instruction gap between each LDCU.64 and its first UR consumer
    (any opcode in {IADD.64-UR, ISETP R-UR, IMAD R-UR, LDG} that reads UR at byte[4]).

    The stream is split at the first predicated BRA (0x947) or EXIT (0x94d):
      • Pre-boundary LDCU.64s: hoisted to position 1 after the first instruction.
        The preamble LDC R1 + S2R + LDC-n + ISETP + BRA give ≥3 instructions of
        warm-up before the first consumer in the active-thread section.
      • Post-boundary LDCU.64s: cannot be hoisted across the boundary because
        pipeline.py BRA fixup uses pre-scheduling body indices and would patch the
        wrong instruction.  Instead, NOP latency fillers are inserted immediately
        after each such LDCU.64 so that gap ≥ 3.  These NOPs have "latency" in
        their comment and are transparent to _body_idx_to_abs (skipped in count).
    """
    _UR_CONSUMER_OPCODES = {0xc35, 0xc0c, 0xc24, 0x981}

    # Find first predicated BRA (0x947) or EXIT (0x94d) — segment boundary.
    # Guard nibble is bits[7:4] of raw[1]; 0x7 = @PT = unconditional.
    boundary_idx = len(instrs)
    for i, si in enumerate(instrs):
        opc = _get_opcode(si.raw)
        guard = (si.raw[1] >> 4) & 0xF
        if opc in (0x947, 0x94d) and guard != 0x7:
            boundary_idx = i
            break

    pre_boundary = instrs[:boundary_idx]
    post_boundary = instrs[boundary_idx:]

    # --- Pre-boundary: hoist LDCU.64s to position 1 ---
    # Skip deferred-param LDCU.64s — they must stay at their point of use
    # (inline LDCU.64 UR6 + IADD.64 pairs emitted by deferred param handling).
    ldcu64s: list[SassInstr] = []
    remaining: list[SassInstr] = []
    for si in pre_boundary:
        if (_get_opcode(si.raw) == 0x7ac and si.raw[9] == 0x0a
                and 'deferred' not in si.comment):
            ldcu64s.append(si)
        else:
            remaining.append(si)

    def _consumer_pos(ldcu_si: SassInstr) -> int:
        ur_dest = ldcu_si.raw[2]
        for i, si in enumerate(remaining):
            if _get_opcode(si.raw) in _UR_CONSUMER_OPCODES and si.raw[4] == ur_dest:
                return i
        return len(remaining)  # no consumer in pre-boundary → sort last

    # Also hoist S2R before any ALU that reads the S2R dest register.
    # S2R (0x919) writes GPRs asynchronously; if it comes AFTER an IMAD
    # that reads the same GPR, IMAD gets uninitialized data → crash.
    # Hoist S2R to right after position 0 (LDC R1), before LDCU.64s.
    # IMPORTANT: only hoist the FIRST S2R for each dest register.
    # If the same dest has multiple S2R writes (register reuse),
    # only the first must be early; the second must stay in its
    # original position to avoid clobbering the first value.
    s2r_instrs = []
    non_s2r_remaining = []
    s2r_dests_seen = set()
    for si in (remaining if not ldcu64s else remaining):
        if _get_opcode(si.raw) == 0x919:
            dest_reg = si.raw[2]
            if dest_reg not in s2r_dests_seen:
                s2r_instrs.append(si)
                s2r_dests_seen.add(dest_reg)
            else:
                # Second write to same dest — keep in place (register reuse:
                # e.g., ctaid→multiply→tid pattern where %r1 is reused).
                non_s2r_remaining.append(si)
        else:
            non_s2r_remaining.append(si)

    # ALSO detect SM_89 IMAD.MOV.U32 (0x7624) as "S2R-like" — they read from
    # constant bank and should also be hoisted for latency hiding.
    # (Skip for now — IMAD.MOV.U32 doesn't have the same async latency as S2R.)

    if ldcu64s:
        ldcu64s.sort(key=_consumer_pos)
        if non_s2r_remaining:
            # TE26-B: family-aware LDCU.64 placement.
            # For STG-only kernels: move LDCU.64 to post-boundary (after EXIT),
            # matching PTXAS layout.  LDG/ATOMG kernels keep pre-boundary.
            _is_stg_only = (
                any(_get_opcode(si.raw) == 0x986 for si in post_boundary)  # has STG
                and not any(_get_opcode(si.raw) in (0x981, 0x98e, 0x9a8, 0x9ae, 0x3a9)
                            for si in post_boundary)  # no LDG/ATOMG
            )
            if _is_stg_only:
                # STG-only: LDCU.64 goes post-EXIT with latency NOP padding
                pre_result = non_s2r_remaining[:1] + s2r_instrs + non_s2r_remaining[1:]
                _padded = []
                for _lsi in ldcu64s:
                    _padded.append(_lsi)
                    _ur_d = _lsi.raw[2]
                    _need_pad = any(
                        _get_opcode(post_boundary[_pi].raw) in _UR_CONSUMER_OPCODES
                        and post_boundary[_pi].raw[4] in (_ur_d, _ur_d + 1)
                        for _pi in range(min(3, len(post_boundary)))
                    )
                    if _need_pad:
                        for _ in range(3):
                            _padded.append(SassInstr(_encode_nop(), 'NOP  // TE26 latency'))
                post_boundary = _padded + list(post_boundary)
            else:
                # LDG/ATOMG: keep LDCU.64 pre-boundary (original safe order)
                pre_result = non_s2r_remaining[:1] + s2r_instrs + ldcu64s + non_s2r_remaining[1:]
        else:
            pre_result = s2r_instrs + ldcu64s
    else:
        if s2r_instrs and non_s2r_remaining:
            # No LDCU.64 to hoist, but still ensure hoisted S2Rs are early.
            # Use non_s2r_remaining (already excludes hoisted S2Rs but keeps
            # duplicate-dest S2Rs in their original position).
            pre_result = non_s2r_remaining[:1] + s2r_instrs + non_s2r_remaining[1:]
        else:
            pre_result = pre_boundary

    # Post-boundary LDCU.64s: SM_120 requires ≥3 physical instructions between
    # LDCU.64 and its first UR consumer. Insert latency NOPs with 'latency'
    # in the comment so the BRA fixup skips them in position counting.
    padded_post = []
    for idx, si in enumerate(post_boundary):
        padded_post.append(si)
        if _get_opcode(si.raw) == 0x7ac and si.raw[9] == 0x0a:  # LDCU.64
            ur_dest = si.raw[2]
            needs_gap = False
            for k in range(1, 4):
                if idx + k < len(post_boundary):
                    next_si = post_boundary[idx + k]
                    next_opc = _get_opcode(next_si.raw)
                    if next_opc in _UR_CONSUMER_OPCODES and next_si.raw[4] == ur_dest:
                        needs_gap = True
                        break
            if needs_gap:
                for _ in range(3):
                    padded_post.append(SassInstr(_encode_nop(), 'NOP  // LDCU.64 latency'))
    post_boundary = padded_post

    # Inline deferred LDCU.64s (in pre_result) also need latency padding.
    # These were kept in-place (not hoisted) and may be adjacent to their consumer.
    padded_pre = []
    for idx, si in enumerate(pre_result):
        padded_pre.append(si)
        if (_get_opcode(si.raw) == 0x7ac and si.raw[9] == 0x0a
                and 'deferred' in si.comment):
            ur_dest = si.raw[2]
            needs_gap = False
            for k in range(1, 4):
                if idx + k < len(pre_result):
                    next_si = pre_result[idx + k]
                    next_opc = _get_opcode(next_si.raw)
                    if next_opc in _UR_CONSUMER_OPCODES and next_si.raw[4] == ur_dest:
                        needs_gap = True
                        break
            if needs_gap:
                for _ in range(3):
                    padded_pre.append(SassInstr(_encode_nop(), 'NOP  // LDCU.64 latency'))
    pre_result = padded_pre

    return pre_result + post_boundary


def _hoist_bounds_check(instrs: list[SassInstr]) -> list[SassInstr]:
    """TE24-B: move bounds-check (ISETP.RUR + predicated EXIT) before LDCU.64s.

    After _hoist_ldcu64, the stream has:
      S2R, S2UR, [LDCU.64s], LDCU.32, ISETP.RUR, EXIT, ...body

    PTXAS layout is:
      S2R, S2UR, LDCU.32, ISETP.RUR, EXIT, [LDCU.64s], ...body

    This pass finds LDCU.64 instructions that are between S2UR and
    ISETP.RUR+EXIT, and moves them to AFTER the EXIT.  The ISETP+EXIT
    effectively "jump over" the LDCU.64s.

    Safety: LDCU.64 writes to UR; ISETP.RUR reads a different UR (the
    LDCU.32 param, not the LDCU.64 descriptor/param).  No dependency.
    """
    result = list(instrs)

    # Find the ISETP.RUR + predicated EXIT pair
    isetp_pos = -1
    for i in range(len(result) - 1):
        if (_get_opcode(result[i].raw) == 0xc0c and
                _get_opcode(result[i + 1].raw) == 0x94d and
                (result[i + 1].raw[1] >> 4) & 0xF != 0x7):
            isetp_pos = i
            break
    if isetp_pos < 0:
        return result

    # Find LDCU.64 instructions BEFORE the ISETP that can be moved after EXIT
    # These are at positions between S2UR (pos ~1) and ISETP (pos isetp_pos)
    isetp_ur = result[isetp_pos].raw[4]  # UR source of the comparison
    movable_ldcu64 = []
    keep = []
    for i in range(isetp_pos):
        si = result[i]
        opc = _get_opcode(si.raw)
        if (opc == 0x7ac and si.raw[9] == 0x0a  # LDCU.64
                and si.raw[2] != isetp_ur          # doesn't write the ISETP's UR
                and si.raw[2] + 1 != isetp_ur):    # pair doesn't either
            movable_ldcu64.append(si)
        else:
            keep.append(si)

    if not movable_ldcu64:
        return result  # nothing to move

    # Reconstruct: keep[...] + ISETP + EXIT + movable_ldcu64[...] + rest
    exit_pos = isetp_pos + 1
    rest = result[exit_pos + 1:]  # everything after EXIT
    new_result = keep + [result[isetp_pos], result[exit_pos]] + movable_ldcu64 + rest
    return new_result


def _enforce_gpr_latency(instrs: list[SassInstr]) -> list[SassInstr]:
    """Insert NOPs between adjacent ALU instructions with GPR read-after-write hazards.

    SM_120 ignores the stall field; rbar alone does not gate ALU→ALU reads.
    Any ALU opcode listed in _OPCODE_META with min_gpr_gap=1 must have at least
    one intervening instruction before a consumer that reads the same GPR.
    """
    from sass.encoding.sm_120_opcodes import encode_nop
    from sass.scoreboard import (
        _OPCODE_META,
        _get_dest_regs as _sc_dest,
        _get_src_regs  as _sc_src,
        _get_opcode    as _sc_opc,
    )

    _ISETP_OPCODES = {0x20c, 0xc0c}  # ISETP R-R, ISETP R-UR
    _FSEL_OPCODE   = 0x208

    def _fsel_pred(raw: bytes) -> int:
        """Extract pred index from FSEL raw[10..11] operand field."""
        return ((raw[10] >> 7) & 1) | ((raw[11] & 0x7F) << 1)

    result = list(instrs)
    i = 0
    while i < len(result) - 1:
        opc_i = _sc_opc(result[i].raw)

        # ISETP → FSEL pred hazard: ISETP writes pred at raw[2];
        # FSEL reads pred from raw[10..11]. Needs ≥1 intervening instruction.
        if opc_i in _ISETP_OPCODES:
            opc_j = _sc_opc(result[i + 1].raw)
            if opc_j == _FSEL_OPCODE:
                isetp_pred = result[i].raw[2]
                fsel_pred  = _fsel_pred(result[i + 1].raw)
                if isetp_pred == fsel_pred:
                    result.insert(i + 1, SassInstr(encode_nop(), 'NOP  // ISETP pred latency'))
                    i += 2
                    continue

        meta_i = _OPCODE_META.get(opc_i)
        # TE21: 0xc11 (IADD3.R-UR) needs min_gpr_gap=1 like 0xc35, but
        # is NOT in _OPCODE_META (avoids global misc counter shift).
        if meta_i is None and opc_i == 0xc11:
            class _M: min_gpr_gap = 1
            meta_i = _M()
        if meta_i is None or meta_i.min_gpr_gap == 0:
            i += 1
            continue
        dest_i = _sc_dest(result[i].raw)
        if not dest_i:
            i += 1
            continue
        src_j = _sc_src(result[i + 1].raw)
        if dest_i & src_j:
            result.insert(i + 1, SassInstr(encode_nop(), 'NOP  // ALU GPR latency'))
            i += 2  # skip the NOP; re-examine the original i+1 at i+2
        else:
            i += 1
    return result


def verify_proof(instrs: list[SassInstr]) -> ProofReport:
    """Constructive proof of schedule safety (FG-2.5 → FG-3.3).

    Enumerate every candidate producer→consumer dependency edge in
    `instrs` and classify each one into exactly one of:

      LATENCY_INERT              — ALU writer with no min_gpr_gap in
                                   the model; dest-reading readers
                                   are safe by opcode class.
      FORWARDING_SAFE            — (writer, reader) is on
                                   sass.scoreboard._FORWARDING_SAFE_PAIRS;
                                   hardware operand forwarding covers
                                   the 1-cycle latency.
      GAP_SAFE                   — ALU writer with min_gpr_gap = k
                                   and reader at gap >= k (FG-3.0
                                   bounded lookahead).
      MEMORY_SCOREBOARD_SAFE     — GPR memory writer; consumer window
                                   carries the rbar class bit for the
                                   producer's wdep slot (FG-3.1).
      MEMORY_INERT               — GPR memory writer with wdep in
                                   _LATENCY_INERT_WDEPS (FG-3.2).
      MEMORY_VIOLATION           — GPR memory writer without rbar
                                   evidence / unknown wdep.
      UR_MEMORY_SCOREBOARD_SAFE  — UR memory writer (LDCU/S2UR/REDUX
                                   /ULDC.64) with rbar evidence in
                                   window (FG-3.2; FG-3.3 folds LDCU
                                   in).
      UR_MEMORY_GAP_SAFE         — LDCU.64 with gap ≥
                                   _LDCU_GAP_SAFE_MIN and no rbar
                                   evidence; narrow data-driven
                                   fallback (FG-3.3).
      UR_MEMORY_INERT            — UR memory writer with wdep in
                                   _LATENCY_INERT_WDEPS (FG-3.3).
      UR_MEMORY_VIOLATION        — UR memory writer without safe
                                   evidence.
      VIOLATION                  — generic ALU violation fallthrough.

    The returned ProofReport holds every recorded edge plus counts
    and a final safe/unsafe verdict.  Its `safe` property is False
    iff any VIOLATION / MEMORY_VIOLATION / UR_MEMORY_VIOLATION edge
    exists.

    Callers that only need the legacy list-of-strings output should
    use `verify_schedule`, which is now a thin wrapper over this
    function.
    """
    from sass.scoreboard import (
        _OPCODE_META,
        _get_dest_regs as _sc_dest,
        _get_src_regs  as _sc_src,
        _get_opcode    as _sc_opc,
        _is_forwarding_safe_pair,
        _is_zero_init_fastpath,
        _WDEP_TO_RBAR_MASK,
        _LATENCY_INERT_WDEPS,
        _LDCU_GAP_SAFE_MIN,
        _get_rbar,
        _get_wdep,
        _is_memory_gpr_producer,
        _is_memory_ur_producer,
        _get_ur_dest,
        _get_ur_src,
        _UR_CONSUMER_OPCODES,
    )

    report = ProofReport()

    # FG-3.3: the legacy R1 (LDCU.64 → first-UR-consumer) rule has
    # been retired.  LDCU now flows through the unified R10 path
    # below alongside S2UR / REDUX / ULDC.64 — classified by the
    # same wdep-based scoreboard proof mechanism as every other
    # memory producer.  See FG-3.3 commit message for the inventory
    # and the evidence for the narrow UR_MEMORY_GAP_SAFE rule.

    # ---- FG-3.1 memory-producer → GPR consumer (rule R9) -----------------
    # Memory-producing opcodes (LDG / LDS / LDC and siblings) are
    # long-latency from the GPR register file's point of view.  Unlike
    # ALU ops, their latency is not covered by instruction-stream gap
    # alone — the hardware waits for them via the consumer's ctrl-word
    # `rbar` bitmask, which names the scoreboard slot class to wait on.
    #
    # The scheduler assigns each memory producer a wdep scoreboard slot
    # (typically 0x31 for LDC, 0x33 for LDS, 0x35 for LDG/ATOMG) but may
    # cross-slot when avoiding collisions — an LDS can be assigned to
    # the LDG slot, for example.  The proof model must classify each
    # edge by the producer's ACTUALLY EMITTED wdep, not a static
    # opcode→class map.  The class bit required in the consumer's rbar
    # is derived from sass.scoreboard._WDEP_TO_RBAR_MASK.
    #
    # Rule:
    #   For each memory-producing opcode at index i with a non-empty
    #   GPR dest set D:
    #     1. If the producer's wdep is not on the tracked list
    #        (typically wdep=0x3f, meaning no-track), the producer is
    #        MEMORY_LATENCY_INERT for this rule — skip edge emission.
    #     2. Otherwise let required_rbar = _WDEP_TO_RBAR_MASK[wdep]
    #        and class_bit = required_rbar & ~0x01 (bit 0 is base).
    #     3. Shadow-walk forward for each reader of a live dest reg.
    #        For each reader j found, scan [i+1, j] inclusive and
    #        check if any instruction's rbar has class_bit set.  If
    #        yes → MEMORY_SCOREBOARD_SAFE; else → MEMORY_VIOLATION.
    #     4. Stop tracking a register once a reader has consumed it
    #        (emit at most one edge per writer-register pair) or once
    #        an intervening writer shadows it.
    #
    # This rule runs BEFORE the R8 ALU loop; the R8 loop skips any
    # memory-producing writer so the same pair is not double-classified.
    for i in range(len(instrs)):
        opc_i = _sc_opc(instrs[i].raw)
        if not _is_memory_gpr_producer(opc_i):
            continue
        dest_i = _sc_dest(instrs[i].raw)
        if not dest_i:
            continue
        wdep_i = _get_wdep(instrs[i].raw)

        # FG-3.2: explicit classification of memory wdeps.  Every
        # observed wdep is constructively classified:
        #   - tracked in _WDEP_TO_RBAR_MASK → scoreboard proof below
        #   - in _LATENCY_INERT_WDEPS (0x3f) → MEMORY_INERT edge
        #   - otherwise → MEMORY_VIOLATION with "unknown wdep" rationale
        #     (fails loudly so new wdep values in future emissions
        #     cannot silently slip through as inert)
        mode_inert = wdep_i in _LATENCY_INERT_WDEPS
        mode_scoreboard = wdep_i in _WDEP_TO_RBAR_MASK
        mode_unknown = not (mode_inert or mode_scoreboard)

        if mode_scoreboard:
            required_rbar = _WDEP_TO_RBAR_MASK[wdep_i]
            class_bit = required_rbar & ~0x01
        else:
            class_bit = 0

        live_dest = set(dest_i)
        for j in range(i + 1, len(instrs)):
            if not live_dest:
                break
            src_j = _sc_src(instrs[j].raw)
            overlap = live_dest & src_j
            if overlap:
                opc_j = _sc_opc(instrs[j].raw)
                gap = j - i - 1

                if mode_inert:
                    cls = ProofClass.MEMORY_INERT
                    rat = (
                        f"memory writer opc=0x{opc_i:03x} "
                        f"(wdep=0x{wdep_i:02x}) → reader "
                        f"opc=0x{opc_j:03x} at gap={gap}: producer "
                        f"wdep=0x3f is explicitly no-track; hardware "
                        f"makes the result immediately available"
                    )
                elif mode_unknown:
                    cls = ProofClass.MEMORY_VIOLATION
                    rat = (
                        f"memory writer opc=0x{opc_i:03x} "
                        f"(wdep=0x{wdep_i:02x}) → reader "
                        f"opc=0x{opc_j:03x} at gap={gap}: UNKNOWN "
                        f"wdep — not in _WDEP_TO_RBAR_MASK and not "
                        f"in _LATENCY_INERT_WDEPS (FG-3.2 INV AC: "
                        f"fails loudly until explicitly classified)"
                    )
                else:
                    safe_by_rbar = False
                    wait_idx = -1
                    for m in range(i + 1, j + 1):
                        if _get_rbar(instrs[m].raw) & class_bit:
                            safe_by_rbar = True
                            wait_idx = m
                            break
                    # P3-2: gap-aware latency exemption.
                    # S2R (0xb82) reads a special register with ~1 cycle latency.
                    # If the gap exceeds 4 instructions, the result is guaranteed
                    # available regardless of rbar coverage.
                    _GAP_SAFE_PRODUCERS = {0xb82}  # S2R only (hardware-confirmed)
                    _GAP_SAFE_THRESHOLD = 4
                    gap_safe = (opc_i in _GAP_SAFE_PRODUCERS and gap >= _GAP_SAFE_THRESHOLD)

                    if safe_by_rbar:
                        cls = ProofClass.MEMORY_SCOREBOARD_SAFE
                        rat = (
                            f"memory writer opc=0x{opc_i:03x} "
                            f"(wdep=0x{wdep_i:02x}) → reader "
                            f"opc=0x{opc_j:03x} at gap={gap}: "
                            f"ctrl rbar class bit 0x{class_bit:02x} "
                            f"set on instr [{wait_idx}] proves the "
                            f"scoreboard wait"
                        )
                    elif gap_safe:
                        cls = ProofClass.MEMORY_GAP_SAFE
                        rat = (
                            f"memory writer opc=0x{opc_i:03x} "
                            f"(wdep=0x{wdep_i:02x}) → reader "
                            f"opc=0x{opc_j:03x} at gap={gap}: "
                            f"gap-safe (S2R latency ≤2, gap≥{_GAP_SAFE_THRESHOLD} "
                            f"is sufficient — P3-2 evidence-based rule)"
                        )
                    else:
                        cls = ProofClass.MEMORY_VIOLATION
                        rat = (
                            f"memory writer opc=0x{opc_i:03x} "
                            f"(wdep=0x{wdep_i:02x}) → reader "
                            f"opc=0x{opc_j:03x} at gap={gap}: no "
                            f"rbar class-bit 0x{class_bit:02x} wait "
                            f"evidence in [{i + 1}, {j}]"
                        )
                report.edges.append(ProofEdge(
                    writer_idx=i,
                    reader_idx=j,
                    writer_opc=opc_i,
                    reader_opc=opc_j,
                    regs=frozenset(overlap),
                    gap=gap,
                    classification=cls,
                    rationale=rat,
                ))
                live_dest -= overlap
            dest_j = _sc_dest(instrs[j].raw)
            if dest_j:
                live_dest -= dest_j

    # ---- Unified UR-memory producer → UR consumer (rule R10, FG-3.3) -----
    # Single rule covering every UR-destination memory/system producer:
    #     LDCU / LDCU.64 (0x7ac)     — FG-3.3: folded into R10,
    #                                   replacing the legacy R1 path
    #     S2UR  (0x9c3)              — FG-3.2
    #     REDUX (0x3c4)              — FG-3.2
    #     ULDC.64 (0xab9)            — FG-3.2 (SM_89, forward cover)
    #
    # The classification algorithm is identical to R9 for GPR memory
    # producers.  The producer's actually emitted wdep slot determines
    # the required consumer-side rbar class bit via
    # _WDEP_TO_RBAR_MASK.  Any instruction in [writer+1, reader]
    # whose rbar carries that class bit proves the scoreboard wait.
    #
    # Four additional classes cover cases that have NO rbar evidence
    # but are still safe by construction:
    #   - wdep in _LATENCY_INERT_WDEPS (0x37, 0x3f) →
    #     UR_MEMORY_INERT.  Slot 0x37 has no rbar bit at all (header
    #     comment at scoreboard.py:27); slot 0x3f is explicit
    #     no-track.  In both cases the consumer cannot — and need not
    #     — wait via rbar.
    #   - Producer is LDCU.64 and instruction-stream gap is
    #     ≥ _LDCU_GAP_SAFE_MIN (3) → UR_MEMORY_GAP_SAFE.
    #     This is the ptxas convention for LDCU.64 latency; evidence
    #     comes from four corpus edges (hmma_zero, imma_zero,
    #     dmma_zero, fg114b_diag3) where the consumer's rbar does
    #     not carry the LDG class bit yet the kernel runs correctly
    #     because the hardware load completes inside three slots.
    #     It is LDCU.64-specific and intentionally narrow.
    #   - wdep unknown (not in _WDEP_TO_RBAR_MASK, not inert) →
    #     UR_MEMORY_VIOLATION with "UNKNOWN wdep" rationale, so new
    #     slot values fail loudly instead of leaking through.
    #   - otherwise (wdep tracked but no rbar in window and not a
    #     gap-safe LDCU.64) → UR_MEMORY_VIOLATION.
    for i in range(len(instrs)):
        opc_i = _sc_opc(instrs[i].raw)
        if not _is_memory_ur_producer(opc_i):
            continue
        ur_dest = _get_ur_dest(instrs[i].raw)
        if ur_dest < 0 or ur_dest >= 0xff:
            continue
        wdep_i = _get_wdep(instrs[i].raw)
        raw_i = instrs[i].raw
        is_ldcu_64 = (opc_i == 0x7ac and raw_i[9] == 0x0a)

        mode_inert = wdep_i in _LATENCY_INERT_WDEPS
        mode_scoreboard = wdep_i in _WDEP_TO_RBAR_MASK
        mode_unknown = not (mode_inert or mode_scoreboard)

        if mode_scoreboard:
            required_rbar = _WDEP_TO_RBAR_MASK[wdep_i]
            class_bit = required_rbar & ~0x01
        else:
            class_bit = 0

        # Find first UR consumer of ur_dest, stopping on UR shadow.
        for j in range(i + 1, len(instrs)):
            opc_j = _sc_opc(instrs[j].raw)
            # Shadow: another producer overwrites this UR index.
            if _is_memory_ur_producer(opc_j):
                other_dest = _get_ur_dest(instrs[j].raw)
                if other_dest == ur_dest:
                    break
            if opc_j not in _UR_CONSUMER_OPCODES:
                continue
            if _get_ur_src(instrs[j].raw) != ur_dest:
                continue

            gap = j - i - 1
            if mode_inert:
                cls = ProofClass.UR_MEMORY_INERT
                rat = (
                    f"UR memory writer opc=0x{opc_i:03x} "
                    f"(wdep=0x{wdep_i:02x}) → UR consumer "
                    f"opc=0x{opc_j:03x} at gap={gap}: producer wdep "
                    f"is on _LATENCY_INERT_WDEPS (slot has no rbar "
                    f"bit or is explicit no-track); the scoreboard "
                    f"does not need to be waited on"
                )
            elif mode_unknown:
                cls = ProofClass.UR_MEMORY_VIOLATION
                rat = (
                    f"UR memory writer opc=0x{opc_i:03x} "
                    f"(wdep=0x{wdep_i:02x}) → UR consumer "
                    f"opc=0x{opc_j:03x} at gap={gap}: UNKNOWN wdep "
                    f"(FG-3.2 INV AC)"
                )
            else:
                safe_by_rbar = False
                wait_idx = -1
                for m in range(i + 1, j + 1):
                    if _get_rbar(instrs[m].raw) & class_bit:
                        safe_by_rbar = True
                        wait_idx = m
                        break
                if safe_by_rbar:
                    cls = ProofClass.UR_MEMORY_SCOREBOARD_SAFE
                    rat = (
                        f"UR memory writer opc=0x{opc_i:03x} "
                        f"(wdep=0x{wdep_i:02x}) → UR consumer "
                        f"opc=0x{opc_j:03x} at gap={gap}: ctrl rbar "
                        f"class bit 0x{class_bit:02x} set on instr "
                        f"[{wait_idx}] proves the scoreboard wait"
                    )
                elif is_ldcu_64 and gap >= _LDCU_GAP_SAFE_MIN:
                    # FG-3.3: narrow, data-driven gap-safe rule for
                    # LDCU.64.  Corpus evidence: hmma_zero / imma_zero
                    # / dmma_zero / fg114b_diag3.
                    cls = ProofClass.UR_MEMORY_GAP_SAFE
                    rat = (
                        f"UR memory writer LDCU.64 (wdep=0x{wdep_i:02x}"
                        f") → UR consumer opc=0x{opc_j:03x} at gap="
                        f"{gap}: no rbar evidence in window, but gap "
                        f"≥ {_LDCU_GAP_SAFE_MIN} (LDCU.64 latency "
                        f"convention, FG-3.3)"
                    )
                else:
                    cls = ProofClass.UR_MEMORY_VIOLATION
                    rat = (
                        f"UR memory writer opc=0x{opc_i:03x} "
                        f"(wdep=0x{wdep_i:02x}) → UR consumer "
                        f"opc=0x{opc_j:03x} at gap={gap}: no rbar "
                        f"class-bit 0x{class_bit:02x} wait evidence "
                        f"in [{i + 1}, {j}]"
                    )
            report.edges.append(ProofEdge(
                writer_idx=i,
                reader_idx=j,
                writer_opc=opc_i,
                reader_opc=opc_j,
                regs=frozenset({ur_dest}),  # UR index tagged
                gap=gap,
                classification=cls,
                rationale=rat,
            ))
            break  # at most one edge per UR producer

    # ---- ALU GPR writer → reader (rule R8, FG-3.0 bounded lookahead) -------
    # For every writer at index i with a non-empty GPR dest, enumerate
    # candidate reader edges in a bounded forward window.  The window
    # size is derived from _OPCODE_META.min_gpr_gap (call it k):
    #
    #   * k == 0 or no meta entry: writer is LATENCY_INERT.  Only the
    #     adjacent reader (j = i+1) gets an edge — informational —
    #     preserving FG-2.5 semantics for the existing corpus.
    #
    #   * k == 1: only j = i+1 can be unsafe; anything at j ≥ i+2 is
    #     trivially GAP_SAFE because gap ≥ 1 = k.  The engine emits
    #     an edge only for j = i+1, preserving FG-2.5 counts exactly.
    #
    #   * k >  1: scan j in [i+1, i+k+1] (inclusive of first
    #     gap == k position).  Each reader in the window gets a
    #     classified edge.  Shadowing is tracked: if an intervening
    #     instruction at j' overwrites one of the live dest regs, that
    #     reg is removed from the live-dest set and no further readers
    #     of it generate edges from this writer.  When the live-dest
    #     set becomes empty the scan stops early.
    #
    # Classification inside the window:
    #   gap == j-i-1, compare against k:
    #     gap >= k                              → GAP_SAFE
    #     gap == 0 and (opc_i,opc_j) forward    → FORWARDING_SAFE
    #     otherwise                             → VIOLATION
    # where "forward" means the pair is on FG-2.4 _FORWARDING_SAFE_PAIRS.
    for i in range(len(instrs)):
        opc_i = _sc_opc(instrs[i].raw)
        # FG-3.1: memory producers are handled by rule R9 above.
        # Skip them here so the same edge is not double-classified.
        if _is_memory_gpr_producer(opc_i):
            continue
        dest_i = _sc_dest(instrs[i].raw)
        if not dest_i:
            continue
        meta_i = _OPCODE_META.get(opc_i)
        k = 0 if (meta_i is None or meta_i.min_gpr_gap == 0) else meta_i.min_gpr_gap

        if k <= 1:
            # Adjacent-only emission — preserves FG-2.5 behavior exactly.
            if i + 1 >= len(instrs):
                continue
            src_j = _sc_src(instrs[i + 1].raw)
            overlap = dest_i & src_j
            if not overlap:
                continue
            opc_j = _sc_opc(instrs[i + 1].raw)
            if k == 0:
                cls = ProofClass.LATENCY_INERT
                rat = (f"writer opc=0x{opc_i:03x} has no min_gpr_gap entry; "
                       f"opcode class is latency-inert for ALU RAW rule")
            elif _is_forwarding_safe_pair(opc_i, opc_j):
                cls = ProofClass.FORWARDING_SAFE
                rat = (f"pair (0x{opc_i:03x}, 0x{opc_j:03x}) in FG-2.4 "
                       f"_FORWARDING_SAFE_PAIRS: hardware operand "
                       f"forwarding covers the 1-cycle latency")
            elif _is_zero_init_fastpath(instrs[i].raw):
                # FG-4.1: narrow semantic exemption — the HFMA2
                # zero-init trick (HFMA2 Rd, -RZ, imm, RZ) produces a
                # deterministic zero that the hardware can forward
                # without observing the 1-instruction ALU latency.
                # Scoped to exactly the all-RZ source pattern; any
                # HFMA2 with non-RZ sources retains the standard
                # VIOLATION classification.
                cls = ProofClass.ZERO_INIT_SAFE
                rat = (f"writer HFMA2 Rd, -RZ, imm, RZ — zero-init "
                       f"fastpath (result is deterministically 0; "
                       f"hardware forwards without ALU latency)")
            else:
                cls = ProofClass.VIOLATION
                rat = (f"0-gap RAW with no forwarding-safe exemption: "
                       f"writer {meta_i.name} needs ≥{meta_i.min_gpr_gap} "
                       f"instruction(s) before a reader of its dest")
            report.edges.append(ProofEdge(
                writer_idx=i,
                reader_idx=i + 1,
                writer_opc=opc_i,
                reader_opc=opc_j,
                regs=frozenset(overlap),
                gap=0,
                classification=cls,
                rationale=rat,
            ))
            continue

        # k > 1: bounded multi-step scan with shadow tracking.
        live_dest = set(dest_i)
        # Upper bound is exclusive; +k+2 so we include the first
        # j with gap == k (which will classify as GAP_SAFE) and stop.
        j_max = min(len(instrs), i + k + 2)
        for j in range(i + 1, j_max):
            if not live_dest:
                break
            src_j = _sc_src(instrs[j].raw)
            overlap = live_dest & src_j
            if overlap:
                opc_j = _sc_opc(instrs[j].raw)
                gap = j - i - 1
                if gap >= k:
                    cls = ProofClass.GAP_SAFE
                    rat = (f"instruction-stream gap {gap} ≥ "
                           f"min_gpr_gap={k} for writer {meta_i.name}")
                elif gap == 0 and _is_forwarding_safe_pair(opc_i, opc_j):
                    cls = ProofClass.FORWARDING_SAFE
                    rat = (f"pair (0x{opc_i:03x}, 0x{opc_j:03x}) in "
                           f"FG-2.4 _FORWARDING_SAFE_PAIRS: hardware "
                           f"operand forwarding covers the 1-cycle "
                           f"latency")
                else:
                    cls = ProofClass.VIOLATION
                    rat = (f"non-adjacent RAW at gap={gap} < "
                           f"min_gpr_gap={k} for writer {meta_i.name}")
                report.edges.append(ProofEdge(
                    writer_idx=i,
                    reader_idx=j,
                    writer_opc=opc_i,
                    reader_opc=opc_j,
                    regs=frozenset(overlap),
                    gap=gap,
                    classification=cls,
                    rationale=rat,
                ))
            # Shadow tracking: if instr j overwrites part of live_dest,
            # subsequent readers of those regs see the newer write, not
            # this producer's.  Remove them from the live set.
            dest_j = _sc_dest(instrs[j].raw)
            if dest_j:
                live_dest -= dest_j

    return report


def verify_schedule(instrs: list[SassInstr]) -> list[str]:
    """Legacy hazard-check API.

    Returns a list of human-readable violation strings.  Empty list
    means no violations.  Internally this is a thin formatter over
    `verify_proof`; the two functions agree on classification in
    every case.
    """
    report = verify_proof(instrs)
    return [e.legacy_str() for e in report.violations]


def _hoist_isetp_past_vote(instrs: list[SassInstr]) -> list[SassInstr]:
    """Hoist ISETP past VOTE when operands are independent.

    SM_120 rule #23: ISETP+VOTE+ISETP (same pred, misc=0) in sequence causes
    ERR715. ptxas avoids this by putting all ISETPs before the VOTE.
    This pass detects VOTE followed by ISETP and hoists the ISETP above
    the VOTE when the ISETP doesn't read the VOTE's output register.
    """
    result = list(instrs)
    _isetp_opcodes = {0x20c, 0xc0c}
    _vote_opcodes = {0x806}
    changed = True
    while changed:
        changed = False
        for i in range(len(result) - 1):
            op_i = (result[i].raw[0] | (result[i].raw[1] << 8)) & 0xFFF
            op_next = (result[i+1].raw[0] | (result[i+1].raw[1] << 8)) & 0xFFF
            if op_i in _vote_opcodes and op_next in _isetp_opcodes:
                # VOTE at i, ISETP at i+1
                # Check: does ISETP read VOTE's dest register?
                vote_dest = result[i].raw[2]  # VOTE dest at byte[2]
                isetp_src0 = result[i+1].raw[3]
                isetp_src1 = result[i+1].raw[4]
                if vote_dest not in (isetp_src0, isetp_src1):
                    # Safe to swap: ISETP doesn't depend on VOTE result
                    result[i], result[i+1] = result[i+1], result[i]
                    changed = True
                    break
            # Also handle VOTE...NOP...ISETP pattern
            if (op_i in _vote_opcodes and i + 2 < len(result)
                    and ((result[i+1].raw[0] | (result[i+1].raw[1] << 8)) & 0xFFF) == 0x918):
                op_next2 = (result[i+2].raw[0] | (result[i+2].raw[1] << 8)) & 0xFFF
                if op_next2 in _isetp_opcodes:
                    vote_dest = result[i].raw[2]
                    isetp_src0 = result[i+2].raw[3]
                    isetp_src1 = result[i+2].raw[4]
                    if vote_dest not in (isetp_src0, isetp_src1):
                        # Move ISETP before VOTE: [VOTE, NOP, ISETP] → [ISETP, VOTE, NOP]
                        isetp = result.pop(i+2)
                        result.insert(i, isetp)
                        changed = True
                        break
    return result


def _fill_ldcu_latency(instrs: list[SassInstr]) -> list[SassInstr]:
    """Hoist deferred LDCU.64 upward to fill latency window with useful work.

    Pattern: find [useful, useful, ..., LDCU.64(deferred), NOP, NOP, NOP, IADD.64]
    and move LDCU.64 earlier so preceding instructions fill the latency slots,
    reducing or eliminating NOPs.

    Only moves deferred LDCU.64 (identified by 'deferred' in comment).
    Safe if: no UR6 write/read conflict in the instructions we move past.
    """
    _UR_CONSUMER_OPCODES = {0xc35, 0xc0c, 0xc24, 0x981}
    result = list(instrs)
    changed = True
    while changed:
        changed = False
        for i in range(len(result)):
            si = result[i]
            opc = (si.raw[0] | (si.raw[1] << 8)) & 0xFFF
            if opc != 0x7ac or si.raw[9] != 0x0a or 'deferred' not in si.comment:
                continue
            ur_dest = si.raw[2]  # UR6 typically
            if i == 0:
                continue

            # Find how far back we can hoist: skip NOPs, stop at barriers
            hoist_to = i
            for j in range(i - 1, -1, -1):
                sj = result[j]
                opc_j = (sj.raw[0] | (sj.raw[1] << 8)) & 0xFFF
                # Skip NOPs (we'll clean up excess NOPs after)
                if opc_j == 0x918:
                    hoist_to = j
                    continue
                # Stop at another LDCU.64 (avoid reordering loads)
                if opc_j == 0x7ac:
                    break
                # Stop at UR consumer (previous IADD.64 reading UR6)
                if opc_j in _UR_CONSUMER_OPCODES and sj.raw[4] == ur_dest:
                    break
                # Stop at branch/exit
                if opc_j in (0x947, 0x94d):
                    break
                # Stop at label-tagged instructions (BRA fixup needs them in position)
                if sj.comment.startswith('// ') and ':' in sj.comment[:20]:
                    break
                # This instruction is safe to move past
                hoist_to = j
            if hoist_to < i:
                ldcu = result.pop(i)
                result.insert(hoist_to, ldcu)
                changed = True
                break

    # Clean up: remove excess LDCU.64 latency NOPs after hoisting.
    # Only remove NOPs explicitly tagged "LDCU.64 latency" when the total
    # instruction gap (including all instructions) between the LDCU.64 and
    # its UR consumer is already ≥3.
    cleaned = []
    for i, si in enumerate(result):
        opc = (si.raw[0] | (si.raw[1] << 8)) & 0xFFF
        if opc == 0x918 and 'LDCU.64 latency' in si.comment:
            # Find the preceding LDCU.64
            ldcu_pos = -1
            for j in range(i - 1, -1, -1):
                oj = (result[j].raw[0] | (result[j].raw[1] << 8)) & 0xFFF
                if oj == 0x7ac and result[j].raw[9] == 0x0a:
                    ldcu_pos = j
                    break
                # Stop at UR consumers EXCEPT LDG.E (which uses UR4 descriptor,
                # not the deferred param's UR6). LDG blocking the scan caused
                # excess NOPs to be retained for ReLU/FMA (the v1.6.1 fix).
                if oj in _UR_CONSUMER_OPCODES and oj != 0x981:
                    break
            if ldcu_pos >= 0:
                # Find the consumer IADD.64 after this NOP
                consumer_pos = -1
                ur_dest = result[ldcu_pos].raw[2]
                for k in range(i + 1, min(i + 6, len(result))):
                    ok = (result[k].raw[0] | (result[k].raw[1] << 8)) & 0xFFF
                    if ok in _UR_CONSUMER_OPCODES and result[k].raw[4] == ur_dest:
                        consumer_pos = k
                        break
                if consumer_pos >= 0:
                    # Don't clean up if DFPU instructions are in the window
                    # (DADD/DMUL/DFMA need extra latency, NOPs may be required)
                    _DFPU = {0x229, 0x228, 0x22b, 0xc2b, 0x22a}
                    has_dfpu = any(
                        (result[k].raw[0] | (result[k].raw[1] << 8)) & 0xFFF in _DFPU
                        for k in range(ldcu_pos + 1, consumer_pos))
                    if not has_dfpu:
                        current_gap = consumer_pos - ldcu_pos - 1
                        if current_gap > 3:
                            continue  # Excess NOP — skip it
        cleaned.append(si)
    return cleaned


def _remove_forwarding_safe_nops(instrs: list[SassInstr]) -> list[SassInstr]:
    """PERF-1: remove NOPs that separate proven forwarding-safe pairs.

    After all scheduling passes have run, scan for the pattern:

        [i]   ALU writer (opc in _OPCODE_META, min_gpr_gap > 0)
        [i+1] NOP (// ALU GPR latency)
        [i+2] ALU reader (reads writer's dest regs)

    If (writer_opc, reader_opc) is on _FORWARDING_SAFE_PAIRS or the
    writer is a ZERO_INIT_SAFE HFMA2, the NOP is unnecessary — remove
    it.  The hardware forwards the result without a gap.

    This runs as the LAST scheduling pass so all prior reordering and
    NOP insertion is already finalized.
    """
    from sass.scoreboard import (
        _OPCODE_META,
        _get_dest_regs as _sc_dest,
        _get_src_regs  as _sc_src,
        _get_opcode    as _sc_opc,
        _is_forwarding_safe_pair,
        _is_zero_init_fastpath,
        _overlap_is_data_only,
    )
    nop_opc = 0x918  # NOP opcode
    result = list(instrs)
    i = 0
    while i < len(result) - 2:
        opc_i = _sc_opc(result[i].raw)
        meta_i = _OPCODE_META.get(opc_i)
        if meta_i is None or meta_i.min_gpr_gap == 0:
            i += 1
            continue
        # Check if [i+1] is a NOP
        if _sc_opc(result[i + 1].raw) != nop_opc:
            i += 1
            continue
        # Check if [i+2] reads [i]'s dest (the pair that the NOP guards)
        dest_i = _sc_dest(result[i].raw)
        if not dest_i:
            i += 1
            continue
        src_k = _sc_src(result[i + 2].raw)
        overlap = dest_i & src_k
        if not overlap:
            i += 1
            continue
        opc_k = _sc_opc(result[i + 2].raw)
        # PERF-2: can we remove the NOP?  Three conditions required:
        #   1. (writer_opc, reader_opc) is forwarding-safe OR writer
        #      is a zero-init fastpath
        #   2. The overlap feeds only the reader's DATA operand, NOT
        #      its ADDRESS operand (operand-role check)
        pair_safe = (_is_forwarding_safe_pair(opc_i, opc_k)
                     or _is_zero_init_fastpath(result[i].raw))
        role_safe = _overlap_is_data_only(result[i + 2].raw, overlap)
        if pair_safe and role_safe:
            result.pop(i + 1)
            # Don't advance i — re-check at the same position
        else:
            i += 1
    return result


def schedule(instrs: list[SassInstr]) -> list[SassInstr]:
    """
    Reorder and pad instructions for correct SM_120 execution.

    Passes (in order):
      1. _hoist_ldcu64        — move LDCU.64 early (≥3-cycle gap before UR consumers)
      2. _enforce_gpr_latency — insert NOPs for ALU GPR RAW hazards
      3. _fill_ldcu_latency   — hoist deferred LDCU.64 to fill latency with useful work
      4. _enforce_gpr_latency — re-check after fill reordering
      5. _remove_forwarding_safe_nops — PERF-1: remove NOPs where hardware forwarding
                                        is proven safe (23 evidence-backed pairs from FG-4)

    Ctrl assignment is handled separately by sass.scoreboard.assign_ctrl().
    """
    instrs = _hoist_ldcu64(instrs)
    # TE24: _hoist_bounds_check disabled — moving LDCU.64 after EXIT
    # violates the ≥3-instruction latency constraint between LDCU.64
    # and its UR consumer.  The PTXAS layout works because PTXAS
    # interleaves LDCU.64 with enough instructions after EXIT to
    # satisfy the gap.  Our body layout doesn't have those fillers.
    # instrs = _hoist_bounds_check(instrs)
    instrs = _enforce_gpr_latency(instrs)
    instrs = _fill_ldcu_latency(instrs)
    instrs = _enforce_gpr_latency(instrs)  # re-check after fill reordering
    # PERF-2/PERF-3: NOP removal and local rescheduling passes are
    # structurally correct but produce zero safe transformations on
    # the current 21-kernel workbench.  Root causes:
    #
    #   1. DUAL-PURPOSE NOPs (PERF-2): body NOPs often guard both an
    #      explicit ALU RAW edge AND an implicit memory-load
    #      scoreboard wait.  Single-edge NOP removal is unsafe.
    #
    #   2. NO FILL CANDIDATES (PERF-3): every body NOP sits
    #      immediately before a terminal memory op (STG/ATOMG/LDG),
    #      and every candidate instruction in the [-3,+3] window is
    #      part of the dependency chain feeding the consumer.  No
    #      independent instruction exists to fill the NOP slot
    #      without creating a new RAW/WAW hazard.
    #
    #   3. KERNELS TOO LINEAR: the workbench kernels have minimal
    #      instruction-level parallelism — each body instruction is
    #      on the critical path.  There are no "free" instructions
    #      to reorder for latency hiding.
    #
    # The infrastructure (_remove_forwarding_safe_nops with operand-
    # role checks) is preserved for future kernels with more ILP.
    # instrs = _remove_forwarding_safe_nops(instrs)
    # Skip LDG latency NOPs — rely on ctrl wdep for LDG (ptxas pattern)
    # instrs = _reorder_after_ldg(instrs)

    return instrs
