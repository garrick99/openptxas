"""
openptxas CLI entry point.

Usage:
    python -m openptxas <file.ptx> [--arch sm_120] [--out file.cubin]
    python -m openptxas --check <file.ptx>   # check for ptxas bug patterns
    python -m openptxas --dump-ir <file.ptx>  # parse and dump IR
"""

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        prog="openptxas",
        description="Open-source PTX assembler/optimizer — SM_120 focused",
    )
    ap.add_argument("ptx_file", nargs="?", help="Input .ptx file")
    ap.add_argument("--arch",  default="sm_120", help="Target SM architecture")
    ap.add_argument("--out",   default=None,     help="Output .cubin (default: <input>.cubin)")
    ap.add_argument("--check", action="store_true",
                    help="Check for ptxas-miscompiled patterns and exit")
    ap.add_argument("--dump-ir", action="store_true",
                    help="Parse PTX and dump the IR, then exit")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Verbose output (show SASS instructions)")
    args = ap.parse_args()

    if args.ptx_file is None:
        ap.print_help()
        sys.exit(0)

    from ptx.parser import parse_file
    from ptx.passes.rotate import run as rotate_run

    print(f"[openptxas] Parsing {args.ptx_file}...")
    try:
        mod = parse_file(args.ptx_file)
    except Exception as e:
        print(f"[openptxas] Parse error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[openptxas] PTX {mod.version[0]}.{mod.version[1]}  "
          f"target={mod.target}  "
          f"{len(mod.functions)} function(s)")

    if args.dump_ir:
        print(mod.dump())
        sys.exit(0)

    # Run optimization passes
    print("[openptxas] Running passes...")
    mod, rotate_groups = rotate_run(mod)

    if rotate_groups:
        print(f"[openptxas] {len(rotate_groups)} rotate-left group(s) will use SHF.L.W")

    if args.check:
        print("[openptxas] --check mode: static analysis complete.")
        sys.exit(0)

    # Full compilation: PTX → cubin
    from sass.pipeline import compile_function
    from sass.regalloc import allocate

    ptx_path = Path(args.ptx_file)
    kernels = [fn for fn in mod.functions if fn.is_kernel]

    if not kernels:
        print("[openptxas] No .entry kernels found — nothing to compile.")
        sys.exit(0)

    for fn in kernels:
        default_out = str(ptx_path.with_name(ptx_path.stem + '_openptxas.cubin'))
        out_path = args.out or default_out
        print(f"[openptxas] Compiling kernel: {fn.name}")

        try:
            cubin_bytes = compile_function(fn, verbose=args.verbose)
        except Exception as e:
            print(f"[openptxas] Compilation error in {fn.name}: {e}",
                  file=sys.stderr)
            sys.exit(1)

        Path(out_path).write_bytes(cubin_bytes)
        print(f"[openptxas] Wrote {out_path} ({len(cubin_bytes)} bytes)")

    print("[openptxas] Done.")


if __name__ == "__main__":
    main()
