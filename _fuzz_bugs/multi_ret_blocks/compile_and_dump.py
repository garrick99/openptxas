"""Compile minimal.ptx through OpenPTXas, dump cubin + disasm via cuobjdump."""
import os, sys, subprocess, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from sass.pipeline import compile_ptx_source

here = pathlib.Path(__file__).parent
ptx_text = (here / "minimal.ptx").read_text()

cubins = compile_ptx_source(ptx_text)
name, cubin = next(iter(cubins.items()))
print(f"kernel: {name}")
out = here / "_ours.cubin"
out.write_bytes(cubin)
print(f"cubin written: {out} ({len(cubin)} bytes)")

# Try cuobjdump for disasm
for exe in ("cuobjdump", "cuobjdump.exe"):
    try:
        r = subprocess.run([exe, "--dump-sass", str(out)], capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            print(r.stdout)
            break
        else:
            print("stderr:", r.stderr[:500], file=sys.stderr)
    except FileNotFoundError:
        continue
