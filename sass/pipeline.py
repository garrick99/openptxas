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
from pathlib import Path

from ptx.parser import parse, parse_file
from ptx.ir import Module, Function
from ptx.passes.rotate import run as rotate_run
from sass.regalloc import allocate
from sass.isel import ISelContext, select_function, SassInstr
from sass.schedule import schedule
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
        # LDC R1, c[0][0x37c] — exact ptxas bytes (ctrl=0x7f1)
        SassInstr(bytes.fromhex('827b01ff00df00000008000000e20f00'),
                  'LDC R1, c[0][0x37c]  // frame ptr'),
        # LDCU.64 UR4, c[0][0x358] — exact ptxas bytes (ctrl=0x717)
        SassInstr(bytes.fromhex('ac7704ff006b0000000a0008002e0e00'),
                  'LDCU.64 UR4, c[0][0x358]  // mem desc'),
    ]

    # 3. Instruction selection
    ctx = ISelContext(
        ra=alloc.ra,
        param_offsets=alloc.param_offsets,
        ur_desc=4,  # UR4 for memory descriptors (ptxas convention)
    )
    raw_instrs = preamble + select_function(fn, ctx)
    sass_instrs = schedule(raw_instrs)

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
    desc = KernelDesc(
        name=fn.name,
        sass_bytes=sass_bytes,
        num_gprs=alloc.num_gprs,
        num_params=len(fn.params),
        param_sizes=param_sizes,
        param_offsets=alloc.param_offsets,
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
