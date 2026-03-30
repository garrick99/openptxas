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

from sass.isel import SassInstr
from sass.encoding.sm_120_opcodes import encode_nop as _encode_nop


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
    ldcu64s: list[SassInstr] = []
    remaining: list[SassInstr] = []
    for si in pre_boundary:
        if _get_opcode(si.raw) == 0x7ac and si.raw[9] == 0x0a:
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
    s2r_instrs = []
    non_s2r_remaining = []
    for si in (remaining if not ldcu64s else remaining):
        if _get_opcode(si.raw) == 0x919:
            s2r_instrs.append(si)
        else:
            non_s2r_remaining.append(si)

    if ldcu64s:
        ldcu64s.sort(key=_consumer_pos)
        if non_s2r_remaining:
            # Order: LDC R1, S2R(s), LDCU.64(s), rest
            pre_result = non_s2r_remaining[:1] + s2r_instrs + ldcu64s + non_s2r_remaining[1:]
        else:
            pre_result = s2r_instrs + ldcu64s
    else:
        if s2r_instrs and remaining:
            # No LDCU.64 to hoist, but still ensure S2R is early
            pre_result = remaining[:1] + s2r_instrs + [si for si in remaining[1:] if _get_opcode(si.raw) != 0x919]
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
    return pre_result + post_boundary


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

    result = list(instrs)
    i = 0
    while i < len(result) - 1:
        opc_i = _sc_opc(result[i].raw)
        meta_i = _OPCODE_META.get(opc_i)
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


def verify_schedule(instrs: list[SassInstr]) -> list[str]:
    """Check a final instruction stream for scheduling hazard violations.

    Returns a list of human-readable violation strings.  Empty list = no violations.

    Checks:
      • LDCU.64 minimum consumer gap ≥3 (R1)
      • ALU GPR 0-gap RAW (R8) — for opcodes with min_gpr_gap > 0 in _OPCODE_META
    """
    from sass.scoreboard import (
        _OPCODE_META,
        _get_dest_regs as _sc_dest,
        _get_src_regs  as _sc_src,
        _get_opcode    as _sc_opc,
    )

    _UR_CONSUMER_OPCODES = {0xc35, 0xc0c, 0xc24, 0x981}
    violations: list[str] = []

    # Find first predicated BRA/EXIT — segment boundary.
    # Post-boundary LDCU.64s satisfy their gap via scoreboard ctrl (rbar/wdep),
    # not instruction ordering, so they are excluded from the gap check.
    boundary_idx = len(instrs)
    for k, sk in enumerate(instrs):
        opc_k = _get_opcode(sk.raw)
        guard_k = (sk.raw[1] >> 4) & 0xF
        if opc_k in (0x947, 0x94d) and guard_k != 0x7:
            boundary_idx = k
            break

    # R1: LDCU.64 must have ≥3 instructions before its first UR consumer.
    # Only enforced for pre-boundary LDCU.64s (preamble section).
    for i, si in enumerate(instrs):
        if i >= boundary_idx:
            break
        if _get_opcode(si.raw) != 0x7ac or si.raw[9] != 0x0a:
            continue
        ur_dest = si.raw[2]
        for j in range(i + 1, len(instrs)):
            opc_j = _get_opcode(instrs[j].raw)
            if opc_j in _UR_CONSUMER_OPCODES and instrs[j].raw[4] == ur_dest:
                gap = j - i - 1
                if gap < 3:
                    violations.append(
                        f"LDCU.64 UR{ur_dest} at [{i}] → consumer opc=0x{opc_j:03x} "
                        f"at [{j}]: gap={gap} (need ≥3)")
                break

    # R8: ALU GPR 0-gap RAW
    for i in range(len(instrs) - 1):
        opc_i = _sc_opc(instrs[i].raw)
        meta_i = _OPCODE_META.get(opc_i)
        if meta_i is None or meta_i.min_gpr_gap == 0:
            continue
        dest_i = _sc_dest(instrs[i].raw)
        if not dest_i:
            continue
        src_j = _sc_src(instrs[i + 1].raw)
        overlap = dest_i & src_j
        if overlap:
            opc_j = _sc_opc(instrs[i + 1].raw)
            violations.append(
                f"{meta_i.name} at [{i}] writes {overlap} → "
                f"opc=0x{opc_j:03x} at [{i+1}] reads immediately (0-gap RAW)")

    return violations


def schedule(instrs: list[SassInstr]) -> list[SassInstr]:
    """
    Reorder and pad instructions for correct SM_120 execution.

    Passes (in order):
      1. _hoist_ldcu64        — move LDCU.64 early (≥3-cycle gap before UR consumers)
      2. _enforce_gpr_latency — insert NOPs for ALU GPR RAW hazards
      3. _reorder_after_ldg  — hide LDG latency by filling the slot with LDC/NOP

    Ctrl assignment is handled separately by sass.scoreboard.assign_ctrl().
    """
    instrs = _hoist_ldcu64(instrs)
    instrs = _enforce_gpr_latency(instrs)
    instrs = _reorder_after_ldg(instrs)
    return instrs
