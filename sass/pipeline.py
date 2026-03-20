"""
sass/pipeline.py — End-to-end PTX → cubin pipeline.

Orchestrates: parse → regalloc → isel → cubin emit.

Usage:
    from sass.pipeline import compile_ptx
    cubin_bytes = compile_ptx("kernel.ptx")
    # or
    cubin_bytes = compile_ptx_source(ptx_string)
"""

from __future__ import annotations
import struct
from pathlib import Path

from ptx.parser import parse, parse_file
from ptx.ir import Module, Function
from ptx.passes.rotate import run as rotate_run
from sass.regalloc import allocate
from sass.isel import ISelContext, select_function, SassInstr
from sass.encoding.sm_120_opcodes import encode_bra, encode_ldcu_64
from sass.schedule import schedule
from sass.scoreboard import assign_ctrl
from cubin.emitter import emit_cubin, KernelDesc


def compile_function(fn: Function, verbose: bool = False) -> bytes:
    """
    Compile a single PTX function/kernel to a cubin.

    Returns raw cubin bytes ready for cuModuleLoad.
    """
    # 1. Register allocation
    alloc = allocate(fn)

    if verbose:
        print(f"[pipeline] {fn.name}: {alloc.num_gprs} GPRs, "
              f"{len(fn.params)} params")
        for p in fn.params:
            off = alloc.param_offsets.get(p.name, -1)
            print(f"  param {p.name}: c[0][0x{off:x}]")

    # 2. Emit kernel preamble — use EXACT bytes from ptxas for the setup
    # instructions that configure frame pointer and memory descriptors.
    # These ctrl/barrier values are critical for correct GPU execution.
    preamble = [
        # LDC R1, c[0][0x37c] — frame pointer (first instruction)
        SassInstr(bytes.fromhex('827b01ff00df00000008000000e20f00'),
                  'LDC R1, c[0][0x37c]  // frame ptr'),
        # LDCU.64 UR4 — memory descriptor with ptxas-matched ctrl
        # ctrl=0x717: wdep=0x31, rbar=0x01, misc=7
        SassInstr(encode_ldcu_64(4, 0, 0x358, ctrl=0x717),
                  'LDCU.64 UR4, c[0][0x358]  // mem desc'),
    ]

    # 3. Instruction selection
    ctx = ISelContext(
        ra=alloc.ra,
        param_offsets=alloc.param_offsets,
        ur_desc=4,  # UR4 for memory descriptors (ptxas convention)
    )
    body_instrs = select_function(fn, ctx)

    # 4. Schedule: reorder for LDG latency, then assign ctrl via scoreboard
    raw_instrs = preamble + body_instrs
    reordered = schedule(raw_instrs)
    # The preamble instructions (LDC R1, LDCU UR4) have hardcoded ctrl from ptxas.
    # Only assign scoreboard ctrl to body instructions (after preamble).
    n_preamble = len(preamble)
    preamble_instrs = reordered[:n_preamble]
    body_scheduled = assign_ctrl(reordered[n_preamble:])
    sass_instrs = preamble_instrs + body_scheduled

    # 5. BRA offset fixup: resolve branch targets AFTER scheduling
    # (scheduler may insert NOPs that shift instruction positions)
    if hasattr(ctx, '_bra_fixups'):
        # Rebuild label map: scan for BRA placeholder targets in comments
        # and find the actual instruction index after scheduling
        n_total = len(sass_instrs)
        for i in range(n_total):
            comment = sass_instrs[i].comment
            # Labels are emitted as comments like "LABEL: <label_name>"
            # But we don't have label markers in the output. Instead, recalculate
            # offsets by mapping from old body indices to new post-schedule indices.
        # Simple approach: find BRA instructions and recalculate based on scanning
        # for the EXIT or target label pattern
        for bra_idx, target_label in ctx._bra_fixups:
            # Find the BRA instruction in the post-schedule output
            # The bra_idx is body-relative, add preamble offset
            abs_bra_idx = n_preamble + bra_idx
            # Account for scheduler-inserted NOPs before this BRA
            # Count NOPs inserted before bra_idx in the body
            nops_before_bra = 0
            for j in range(n_preamble, abs_bra_idx + nops_before_bra + 1):
                if j < len(sass_instrs) and 'latency' in sass_instrs[j].comment.lower():
                    if j <= abs_bra_idx + nops_before_bra:
                        nops_before_bra += 1
            actual_bra_idx = abs_bra_idx + nops_before_bra

            # Find target: count NOPs before the target label's original position
            if target_label in ctx.label_map:
                orig_target_body_byte = ctx.label_map[target_label]
                orig_target_body_idx = orig_target_body_byte // 16
                abs_target_idx = n_preamble + orig_target_body_idx
                nops_before_target = 0
                for j in range(n_preamble, abs_target_idx + nops_before_target + 1):
                    if j < len(sass_instrs) and 'latency' in sass_instrs[j].comment.lower():
                        if j <= abs_target_idx + nops_before_target:
                            nops_before_target += 1
                actual_target_idx = abs_target_idx + nops_before_target
                actual_target_byte = actual_target_idx * 16
                actual_bra_byte = (actual_bra_idx + 1) * 16
                rel_offset = actual_target_byte - actual_bra_byte
                if actual_bra_idx < len(sass_instrs):
                    # Preserve predicate from original BRA encoding
                    old_raw = sass_instrs[actual_bra_idx].raw
                    old_pred_byte = old_raw[1] & 0xF0
                    new_raw = bytearray(encode_bra(rel_offset))
                    new_raw[1] = (new_raw[1] & 0x0F) | old_pred_byte
                    old_comment = sass_instrs[actual_bra_idx].comment
                    pred_prefix = old_comment.split('BRA')[0] if 'BRA' in old_comment else ''
                    sass_instrs[actual_bra_idx] = SassInstr(
                        bytes(new_raw),
                        f'{pred_prefix}BRA {target_label} (offset={rel_offset})')

    if verbose:
        print(f"[pipeline] {len(sass_instrs)} SASS instructions:")
        for i, si in enumerate(sass_instrs):
            print(f"  +{i*16:4d}: {si.hex()}  // {si.comment}")

    # 3. Concatenate SASS bytes
    sass_bytes = b''.join(si.raw for si in sass_instrs)

    # 4. Build param size list
    param_sizes = []
    for p in fn.params:
        if p.type.width >= 64:
            param_sizes.append(8)
        elif p.type.width >= 32:
            param_sizes.append(4)
        else:
            param_sizes.append(p.type.width // 8)

    # 5. Emit cubin
    # Compute constant bank size: param_base + sum of all param sizes, rounded up to 4
    total_param_bytes = sum(param_sizes)
    const0_size = ((0x380 + total_param_bytes + 3) // 4) * 4  # 4-byte aligned

    # Find EXIT and S2R instruction offsets in the SASS byte stream
    exit_offset = 0
    s2r_offset = 0x10  # default
    for i in range(0, len(sass_bytes), 16):
        if sass_bytes[i:i+2] == bytes([0x4d, 0x79]):  # EXIT opcode
            exit_offset = i
            break
    # Find S2R that reads CTAID.X (SR code 0x25 in byte 9)
    # Falls back to any S2R if no CTAID S2R is found
    for i in range(0, len(sass_bytes), 16):
        if sass_bytes[i:i+2] == bytes([0x19, 0x79]) and sass_bytes[i+9] == 0x25:
            s2r_offset = i
            break
    else:
        # No CTAID S2R — find first S2R of any type
        for i in range(0, len(sass_bytes), 16):
            if sass_bytes[i:i+2] == bytes([0x19, 0x79]):
                s2r_offset = i
                break

    desc = KernelDesc(
        name=fn.name,
        sass_bytes=sass_bytes,
        num_gprs=alloc.num_gprs,
        num_params=len(fn.params),
        param_sizes=param_sizes,
        param_offsets=alloc.param_offsets,
        const0_size=const0_size,
        exit_offset=exit_offset,
        s2r_offset=s2r_offset,
    )
    return emit_cubin(desc)


def compile_ptx_source(ptx_src: str, verbose: bool = False) -> dict[str, bytes]:
    """
    Compile PTX source text. Returns {kernel_name: cubin_bytes} for each kernel.
    """
    mod = parse(ptx_src)
    mod, rotate_groups = rotate_run(mod)

    results = {}
    for fn in mod.functions:
        if fn.is_kernel:
            results[fn.name] = compile_function(fn, verbose=verbose)

    return results


def compile_ptx(ptx_path: str, verbose: bool = False) -> dict[str, bytes]:
    """
    Compile a PTX file. Returns {kernel_name: cubin_bytes} for each kernel.
    """
    src = Path(ptx_path).read_text(encoding='utf-8')
    return compile_ptx_source(src, verbose=verbose)
