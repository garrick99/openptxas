"""CLI for template engine: extract, render, and round-trip test."""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.template_engine.extract import extract_atom_xor_template
from tools.template_engine.render import render_template, render_to_block
from benchmarks.bench_util import compile_ptxas
import workbench


def _round_trip_one(name: str) -> dict:
    """Run extract -> render -> compare for one kernel."""
    k = workbench.KERNELS[name]
    src = k.get("ptx_inline")
    if src is None:
        return {"kernel": name, "status": "SKIP", "reason": "no ptx_inline"}

    cubin, _ = compile_ptxas(src)
    spec = extract_atom_xor_template(cubin)

    # Determine params for rendering
    params = {}
    for instr in spec.instructions:
        for p in instr.params:
            if p.name == "add_imm_K":
                # Read original value from the raw bytes
                val = 0
                for i in range(p.byte_length):
                    val |= instr.raw_bytes[p.byte_offset + i] << (8 * i)
                params[p.name] = val

    rendered = render_template(spec, params)

    # Compare
    original_active = [
        instr.raw_bytes for instr in spec.instructions
    ]
    match = all(r == o for r, o in zip(rendered, original_active))

    mismatches = []
    if not match:
        for i, (r, o) in enumerate(zip(rendered, original_active)):
            if r != o:
                mismatches.append({
                    "index": i,
                    "role": spec.instructions[i].role,
                    "original": o.hex(),
                    "rendered": r.hex(),
                })

    return {
        "kernel": name,
        "variant": spec.variant,
        "total_instructions": len(spec.instructions),
        "parameterized_bytes": spec.total_parameterized_bytes(),
        "exact_match": match,
        "mismatches": mismatches,
        "status": "PASS" if match else "FAIL",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PTXAS template engine")
    sub = parser.add_subparsers(dest="cmd")

    p_extract = sub.add_parser("extract", help="Extract template from PTXAS cubin")
    p_extract.add_argument("kernel", help="Workbench kernel name")

    p_roundtrip = sub.add_parser("roundtrip", help="Round-trip test")
    p_roundtrip.add_argument("--kernel", nargs="*",
                             default=["k100_atom_xor", "w2_atom_xor_reduce"])

    args = parser.parse_args()

    if args.cmd == "extract":
        k = workbench.KERNELS[args.kernel]
        cubin, _ = compile_ptxas(k.get("ptx_inline"))
        spec = extract_atom_xor_template(cubin)
        print(json.dumps(spec.to_dict(), indent=2))

    elif args.cmd == "roundtrip":
        all_pass = True
        for name in args.kernel:
            result = _round_trip_one(name)
            status = result["status"]
            variant = result.get("variant", "?")
            n_instr = result.get("total_instructions", 0)
            p_bytes = result.get("parameterized_bytes", 0)
            print(f"{name}: variant={variant} instrs={n_instr} "
                  f"param_bytes={p_bytes} -> {status}")
            if result.get("mismatches"):
                for m in result["mismatches"]:
                    print(f"  MISMATCH [{m['index']}] {m['role']}")
                    print(f"    orig: {m['original']}")
                    print(f"    rend: {m['rendered']}")
            if status != "PASS":
                all_pass = False
        sys.exit(0 if all_pass else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
