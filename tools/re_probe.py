"""
SM_120 SASS Encoding Reverse Engineering Probe.

Workflow:
    1. Load a compiled .cubin or .obj file (ELF container with SASS text sections)
    2. Parse the ELF to locate .text.{kernel} sections
    3. Run cuobjdump -sass to get the disassembly (opcode text + byte offsets)
    4. Correlate each disassembled instruction with its raw 16-byte encoding
    5. Decode the bit fields (opcode, dest, src0, src1, imm, modifiers, predicate)
    6. Build / update encoding tables in JSON

SM_75+ instruction format: 128 bits (16 bytes) per instruction.
  Bits [127:105] = control bits (scheduling, stall, yield, etc.)
  Bits [104:0]   = instruction encoding proper

Usage:
    python -m openptxas.tools.re_probe <file.cubin> [--arch sm_120] [--out tables.json]
    python -m openptxas.tools.re_probe --probe-ptx <file.ptx> [--arch sm_120]
    python -m openptxas.tools.re_probe --batch <dir_with_cubins/>
"""

from __future__ import annotations
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# ELF parsing (minimal — enough to find .text sections)
# ---------------------------------------------------------------------------

ELF_MAGIC = b"\x7fELF"

@dataclass
class ElfSection:
    name:   str
    offset: int      # byte offset into file
    size:   int      # byte size
    data:   bytes


def _read_elf_sections(data: bytes) -> list[ElfSection]:
    """Parse ELF64 section headers and return sections with their data."""
    if data[:4] != ELF_MAGIC:
        raise ValueError("Not an ELF file")
    if data[4] != 2:
        raise ValueError("Expected ELF64 (class=2)")

    # ELF64 header
    e_shoff     = struct.unpack_from("<Q", data, 0x28)[0]  # section header offset
    e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]  # section header entry size
    e_shnum     = struct.unpack_from("<H", data, 0x3C)[0]  # number of sections
    e_shstrndx  = struct.unpack_from("<H", data, 0x3E)[0]  # string table section index

    # Parse section headers
    sh_list = []
    for i in range(e_shnum):
        base = e_shoff + i * e_shentsize
        sh_name   = struct.unpack_from("<I", data, base + 0x00)[0]
        sh_offset = struct.unpack_from("<Q", data, base + 0x18)[0]
        sh_size   = struct.unpack_from("<Q", data, base + 0x20)[0]
        sh_list.append((sh_name, sh_offset, sh_size))

    # Extract string table
    strtab_off, strtab_size = sh_list[e_shstrndx][1], sh_list[e_shstrndx][2]
    strtab = data[strtab_off: strtab_off + strtab_size]

    def get_name(off: int) -> str:
        end = strtab.index(b"\x00", off)
        return strtab[off:end].decode("utf-8", errors="replace")

    sections = []
    for sh_name, sh_offset, sh_size in sh_list:
        if sh_size == 0:
            continue
        name    = get_name(sh_name)
        content = data[sh_offset: sh_offset + sh_size]
        sections.append(ElfSection(name=name, offset=sh_offset,
                                   size=sh_size, data=content))
    return sections


def find_text_sections(elf_data: bytes) -> dict[str, bytes]:
    """Return {kernel_name: raw_bytes} for all .text.* sections."""
    result = {}
    for sec in _read_elf_sections(elf_data):
        if sec.name.startswith(".text."):
            kernel = sec.name[len(".text."):]
            result[kernel] = sec.data
        elif sec.name == ".text" and sec.size > 0:
            result["__default__"] = sec.data
    return result


# ---------------------------------------------------------------------------
# SASS disassembly parsing
# ---------------------------------------------------------------------------

# cuobjdump -sass output line pattern:
#   /*0060*/              SHF.L.W.U32.HI R7, R4, 0x8, R5;
_SASS_LINE_RE = re.compile(
    r"/\*([0-9a-fA-F]+)\*/\s+"   # byte offset in hex
    r"(@!?P\d+\s+)?"             # optional predicate guard
    r"([A-Z][A-Z0-9_.]+)"        # opcode (all-caps SASS)
    r"(.*?);"                    # operands
)

@dataclass
class SassInst:
    offset:   int        # byte offset in .text section
    pred:     str        # predicate guard or ""
    opcode:   str        # e.g. "SHF.L.W.U32.HI"
    operands: str        # raw operand string (not parsed further yet)
    raw:      bytes = field(default_factory=bytes)  # 16 raw bytes (filled later)

    def encoding_hex(self) -> str:
        return self.raw.hex()

    def encoding_int(self) -> int:
        return int.from_bytes(self.raw, "little")

    def control_bits(self) -> int:
        """Upper 23 bits of 128-bit instruction (SM_75+ control word)."""
        v = self.encoding_int()
        return (v >> 105) & 0x7FFFFF

    def instruction_bits(self) -> int:
        """Lower 105 bits — the actual SASS encoding."""
        v = self.encoding_int()
        return v & ((1 << 105) - 1)


def disassemble_file(path: str, arch: str = "sm_120") -> str:
    """Run cuobjdump -sass and return stdout."""
    # Try cuobjdump first, then nvdisasm
    for tool in ["cuobjdump", "nvdisasm"]:
        try:
            if tool == "cuobjdump":
                cmd = ["cuobjdump", "-sass", path]
            else:
                cmd = ["nvdisasm", "-hex", path]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return result.stdout
        except FileNotFoundError:
            continue
    raise RuntimeError(
        "Neither cuobjdump nor nvdisasm found. "
        "Ensure CUDA toolkit bin is in PATH."
    )


def parse_sass_dump(sass_text: str) -> dict[str, list[SassInst]]:
    """
    Parse cuobjdump -sass output.
    Returns {kernel_name: [SassInst, ...]}.
    """
    result: dict[str, list[SassInst]] = {}
    current_kernel = "__default__"
    result[current_kernel] = []

    for line in sass_text.splitlines():
        # Kernel header line
        func_match = re.match(r"\s*Function\s*:\s*(\S+)", line)
        if func_match:
            current_kernel = func_match.group(1)
            result.setdefault(current_kernel, [])
            continue

        m = _SASS_LINE_RE.search(line)
        if not m:
            continue

        offset   = int(m.group(1), 16)
        pred     = (m.group(2) or "").strip()
        opcode   = m.group(3).strip()
        operands = m.group(4).strip()

        result[current_kernel].append(
            SassInst(offset=offset, pred=pred, opcode=opcode, operands=operands)
        )

    return {k: v for k, v in result.items() if v}


def correlate(text_bytes: bytes, insts: list[SassInst],
              inst_size: int = 16) -> list[SassInst]:
    """Fill SassInst.raw from the raw .text section bytes."""
    for inst in insts:
        start = inst.offset
        end   = start + inst_size
        if end <= len(text_bytes):
            inst.raw = text_bytes[start:end]
        else:
            inst.raw = b"\x00" * inst_size
    return insts


# ---------------------------------------------------------------------------
# Bit field extraction
# ---------------------------------------------------------------------------

def extract_bits(value: int, high: int, low: int) -> int:
    """Extract bits [high:low] inclusive from value."""
    mask = (1 << (high - low + 1)) - 1
    return (value >> low) & mask


@dataclass
class DecodedInst:
    """
    Decoded bit fields for an SM_75+ SASS instruction.
    Field positions are based on Turing/Ampere/Ada/Blackwell common layout.
    Some fields are opcode-specific and require per-opcode tables.
    """
    raw_lo:    int       # bits [63:0]
    raw_hi:    int       # bits [127:64]

    # Common fields (same position across most integer instructions)
    opcode:    int       # bits [11:0]  (rough opcode field)
    pred_reg:  int       # bits [15:12] predicate register
    pred_neg:  bool      # bit [16]     predicate negated
    dest_reg:  int       # bits [23:16] destination register
    src0_reg:  int       # bits [39:32] first source register
    src1_reg:  int       # bits [55:48] second source register (or imm low bits)
    imm:       int       # varies       immediate value (if applicable)

    @classmethod
    def from_raw(cls, raw: bytes) -> "DecodedInst":
        if len(raw) < 16:
            raw = raw.ljust(16, b"\x00")
        lo = int.from_bytes(raw[0:8],  "little")
        hi = int.from_bytes(raw[8:16], "little")
        full = lo | (hi << 64)

        return cls(
            raw_lo   = lo,
            raw_hi   = hi,
            opcode   = extract_bits(full, 11, 0),
            pred_reg = extract_bits(full, 15, 12),
            pred_neg = bool(extract_bits(full, 16, 16)),
            dest_reg = extract_bits(full, 23, 16),
            src0_reg = extract_bits(full, 39, 32),
            src1_reg = extract_bits(full, 55, 48),
            imm      = extract_bits(full, 63, 48),  # rough
        )

    def bits(self, high: int, low: int) -> int:
        full = self.raw_lo | (self.raw_hi << 64)
        return extract_bits(full, high, low)


# ---------------------------------------------------------------------------
# Encoding table builder
# ---------------------------------------------------------------------------

@dataclass
class OpcodeEntry:
    """
    Encoding information for a single SASS opcode.
    Built by observing multiple instances.
    """
    opcode_str:       str
    instances:        list[dict]   = field(default_factory=list)
    opcode_bits:      Optional[int] = None   # the opcode field value (bits [11:0] or subrange)
    # Known modifiers and their bit positions (discovered empirically)
    modifier_bits:    dict[str, tuple[int,int]] = field(default_factory=dict)

    def add_instance(self, inst: SassInst, decoded: DecodedInst):
        self.instances.append({
            "offset":    inst.offset,
            "operands":  inst.operands,
            "hex":       inst.encoding_hex(),
            "lo":        hex(decoded.raw_lo),
            "hi":        hex(decoded.raw_hi),
            "opcode_f":  hex(decoded.opcode),
            "pred_reg":  decoded.pred_reg,
            "pred_neg":  decoded.pred_neg,
            "dest_reg":  decoded.dest_reg,
            "src0_reg":  decoded.src0_reg,
            "src1_reg":  decoded.src1_reg,
            "control":   hex(inst.control_bits()),
        })
        # Track consistent opcode field
        if self.opcode_bits is None:
            self.opcode_bits = decoded.opcode
        elif self.opcode_bits != decoded.opcode:
            # opcode field isn't at [11:0] for this instruction — mark as varied
            self.opcode_bits = -1

    def to_dict(self) -> dict:
        return {
            "opcode_str":    self.opcode_str,
            "opcode_bits":   hex(self.opcode_bits) if self.opcode_bits is not None and self.opcode_bits >= 0 else "?",
            "instance_count": len(self.instances),
            "instances":     self.instances,
            "modifier_bits": self.modifier_bits,
        }


class EncodingTableBuilder:
    """Accumulates instances across multiple cubins and builds encoding tables."""

    def __init__(self):
        self.entries: dict[str, OpcodeEntry] = {}

    def ingest(self, sass_text: str, text_sections: dict[str, bytes]):
        """Process one cubin's disassembly + raw section bytes."""
        kernel_insts = parse_sass_dump(sass_text)
        for kernel, insts in kernel_insts.items():
            raw = text_sections.get(kernel, text_sections.get("__default__", b""))
            correlate(raw, insts)

            for inst in insts:
                if not inst.raw or len(inst.raw) < 16:
                    continue
                decoded = DecodedInst.from_raw(inst.raw)
                key = inst.opcode
                if key not in self.entries:
                    self.entries[key] = OpcodeEntry(opcode_str=key)
                self.entries[key].add_instance(inst, decoded)

    def diff_pair(self, opcode: str, idx_a: int, idx_b: int) -> dict:
        """
        Compare two instances of the same opcode to isolate which bits
        change between them. Critical for identifying field positions.
        """
        entry = self.entries.get(opcode)
        if not entry or len(entry.instances) <= max(idx_a, idx_b):
            return {}

        a = int(entry.instances[idx_a]["hex"], 16)
        b = int(entry.instances[idx_b]["hex"], 16)
        diff = a ^ b

        changed_bits = []
        for bit in range(128):
            if (diff >> bit) & 1:
                changed_bits.append(bit)

        # Group into contiguous fields
        fields = []
        if changed_bits:
            start = changed_bits[0]
            prev  = changed_bits[0]
            for bit in changed_bits[1:]:
                if bit == prev + 1:
                    prev = bit
                else:
                    fields.append((start, prev))
                    start = bit
                    prev  = bit
            fields.append((start, prev))

        return {
            "opcode": opcode,
            "instance_a": entry.instances[idx_a],
            "instance_b": entry.instances[idx_b],
            "xor_hex":    hex(diff),
            "changed_bit_ranges": [f"[{hi}:{lo}]" for lo, hi in fields],
        }

    def dump(self) -> dict:
        return {k: v.to_dict() for k, v in sorted(self.entries.items())}

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.dump(), f, indent=2)
        print(f"[re_probe] Saved encoding tables to {path}")

    def report(self):
        """Print a human-readable summary."""
        print(f"\n{'='*70}")
        print(f"  SM_120 SASS Encoding Analysis — {len(self.entries)} unique opcodes")
        print(f"{'='*70}")
        for opcode, entry in sorted(self.entries.items()):
            n = len(entry.instances)
            ob = hex(entry.opcode_bits) if entry.opcode_bits and entry.opcode_bits >= 0 else "?"
            print(f"  {opcode:<30} {n:>4} instances  opcode_field={ob}")
        print()

        # Highlight our target opcodes
        targets = {"SHF.L.W.U32.HI", "SHF.L.W.U32", "SHF.L.W",
                   "IADD3", "SHF.R.U32.HI", "SHF.R.U32"}
        found = targets & set(self.entries.keys())
        if found:
            print(f"  [!] Target opcodes found: {', '.join(sorted(found))}")
            print()

    def analyze_shf(self):
        """
        Specific analysis of SHF (shift funnel) instructions.
        These are the bug-carrying opcodes.
        Shows which bits differ between left/right and between
        different shift amounts — that's how we find the shift amount field.
        """
        shf_ops = [k for k in self.entries if k.startswith("SHF")]
        if not shf_ops:
            print("[re_probe] No SHF instructions found.")
            return

        print(f"\n  SHF instruction analysis:")
        for op in sorted(shf_ops):
            entry = self.entries[op]
            print(f"\n  {op} ({len(entry.instances)} instances):")
            for i, inst in enumerate(entry.instances[:8]):  # show first 8
                print(f"    [{i}] {inst['operands']:<30}  "
                      f"lo={inst['lo']}  hi={inst['hi']}")

            # Diff first two instances if available
            if len(entry.instances) >= 2:
                diff = self.diff_pair(op, 0, 1)
                if diff.get("changed_bit_ranges"):
                    print(f"    Bits changing [0]->[1]: {diff['changed_bit_ranges']}")


# ---------------------------------------------------------------------------
# PTX -> cubin compilation helper
# ---------------------------------------------------------------------------

def compile_ptx(ptx_src: str, arch: str = "sm_120",
                cuda_home: Optional[str] = None) -> bytes:
    """
    Compile a PTX string to a cubin using ptxas.
    Returns the raw cubin bytes.
    """
    # Find ptxas
    cuda_paths = []
    if cuda_home:
        cuda_paths.append(Path(cuda_home) / "bin" / "ptxas")
    # Common locations
    for prefix in [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin",
        "/usr/local/cuda/bin",
        "/usr/bin",
    ]:
        cuda_paths.append(Path(prefix) / "ptxas")

    ptxas_path = None
    for p in cuda_paths:
        if p.exists():
            ptxas_path = str(p)
            break

    if ptxas_path is None:
        # Try from PATH
        ptxas_path = "ptxas"

    with tempfile.NamedTemporaryFile(suffix=".ptx", mode="w",
                                     delete=False, encoding="utf-8") as f:
        f.write(ptx_src)
        ptx_file = f.name

    cubin_file = ptx_file.replace(".ptx", ".cubin")
    try:
        result = subprocess.run(
            [ptxas_path, f"-arch={arch}", "-m64", ptx_file, "-o", cubin_file],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            raise RuntimeError(f"ptxas failed:\n{result.stderr}")
        with open(cubin_file, "rb") as f:
            return f.read()
    finally:
        for fn in (ptx_file, cubin_file):
            try:
                os.unlink(fn)
            except OSError:
                pass


def probe_single_instruction(
    ptx_template: str,
    variants: list[dict],
    arch: str = "sm_120",
) -> list[SassInst]:
    """
    Compile PTX with varying parameters, extract the target instruction's
    encoding for each variant.  Used to isolate field positions.

    ptx_template: PTX with {placeholder} fields, e.g.:
        ".version 8.0\\n.target {arch}\\n..."
    variants: list of dicts to format into the template
    """
    results = []
    builder = EncodingTableBuilder()

    for var in variants:
        src = ptx_template.format(arch=arch, **var)
        try:
            cubin = compile_ptx(src, arch=arch)
        except RuntimeError as e:
            print(f"[probe] Compile failed for {var}: {e}")
            continue

        text_secs = find_text_sections(cubin)
        sass = disassemble_file_bytes(cubin, arch)
        if sass:
            builder.ingest(sass, text_secs)

    return builder


def disassemble_file_bytes(cubin_data: bytes, arch: str = "sm_120") -> Optional[str]:
    """Disassemble a cubin given as bytes (writes to temp file)."""
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin_data)
        tmp = f.name
    try:
        return disassemble_file(tmp, arch)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Canned probe: SHF shift-amount field
# ---------------------------------------------------------------------------

SHF_PROBE_TEMPLATE = """\
.version 8.0
.target {arch}
.address_size 64

.visible .entry probe_shf_{k}(
    .param .u64 out_ptr,
    .param .u64 in_ptr)
{{
    .reg .u64   %rd<8>;
    .reg .u32   %r<8>;

    ld.param.u64    %rd0, [in_ptr];
    ld.global.u64   %rd1, [%rd0];

    // Split into 32-bit halves for SHF
    cvt.u32.u64     %r0, %rd1;
    shr.u64         %rd2, %rd1, 32;
    cvt.u32.u64     %r1, %rd2;

    // This is the pattern: SHF.L.W with shift amount K={k}
    shf.l.wrap.b32  %r2, %r0, %r1, {k};

    // Store result
    ld.param.u64    %rd3, [out_ptr];
    cvt.u64.u32     %rd4, %r2;
    st.global.u64   [%rd3], %rd4;

    ret;
}}
"""

def probe_shf_encoding(arch: str = "sm_120", out_file: str = "shf_encoding.json"):
    """
    Compile SHF.L with K=1..63 and record the encoding for each.
    This isolates the shift-amount field position.
    """
    print(f"[re_probe] Probing SHF.L shift-amount encoding for {arch}...")
    builder = EncodingTableBuilder()

    for k in range(1, 64):
        src = SHF_PROBE_TEMPLATE.format(arch=arch, k=k)
        try:
            cubin = compile_ptx(src, arch=arch)
        except RuntimeError as e:
            print(f"  K={k}: compile failed — {e}")
            continue
        text_secs = find_text_sections(cubin)
        sass = disassemble_file_bytes(cubin, arch)
        if sass:
            builder.ingest(sass, text_secs)
        print(f"  K={k} done")

    builder.report()
    builder.analyze_shf()
    builder.save(out_file)
    return builder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="OpenPTX — SM_120 SASS encoding RE probe")
    ap.add_argument("files", nargs="*", help=".cubin or .obj files to analyse")
    ap.add_argument("--arch",       default="sm_120")
    ap.add_argument("--out",        default="encoding_tables.json",
                    help="Output JSON file for encoding tables")
    ap.add_argument("--probe-shf",  action="store_true",
                    help="Compile SHF K=1..63 variants and extract encoding")
    ap.add_argument("--diff",       nargs=3, metavar=("OPCODE", "IDX_A", "IDX_B"),
                    help="Diff two instances of an opcode to find field positions")
    ap.add_argument("--analyze-shf", action="store_true",
                    help="Show SHF-specific bit analysis")
    args = ap.parse_args()

    builder = EncodingTableBuilder()

    # Ingest provided cubin files
    for path in args.files:
        print(f"[re_probe] Processing {path}...")
        with open(path, "rb") as f:
            data = f.read()
        try:
            text_secs = find_text_sections(data)
        except Exception as e:
            print(f"  ELF parse failed: {e}")
            continue
        try:
            sass = disassemble_file(path, args.arch)
        except Exception as e:
            print(f"  Disassembly failed: {e}")
            continue
        builder.ingest(sass, text_secs)
        print(f"  OK — {len(text_secs)} kernel section(s)")

    # Run SHF probe
    if args.probe_shf:
        b = probe_shf_encoding(arch=args.arch, out_file=args.out)
        builder = b  # replace builder with probe results

    if args.files or args.probe_shf:
        builder.report()
        if args.analyze_shf:
            builder.analyze_shf()

        if args.diff:
            opcode = args.diff[0]
            idx_a  = int(args.diff[1])
            idx_b  = int(args.diff[2])
            result = builder.diff_pair(opcode, idx_a, idx_b)
            print(json.dumps(result, indent=2))

        builder.save(args.out)

    if not args.files and not args.probe_shf:
        ap.print_help()


if __name__ == "__main__":
    main()
