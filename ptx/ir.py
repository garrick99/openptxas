"""
PTX Intermediate Representation.

Models PTX modules, functions, basic blocks, and instructions as plain Python
dataclasses.  No validation here — that's the parser's job.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Type system
# ---------------------------------------------------------------------------

class ScalarKind(Enum):
    B    = "b"    # untyped bits
    U    = "u"    # unsigned int
    S    = "s"    # signed int
    F    = "f"    # float
    PRED = "pred" # predicate (1-bit bool)

@dataclass(frozen=True)
class TypeSpec:
    kind: ScalarKind
    width: int          # 8, 16, 32, 64, 128 (or 1 for pred)

    def __str__(self) -> str:
        if self.kind == ScalarKind.PRED:
            return "pred"
        return f"{self.kind.value}{self.width}"

    @classmethod
    def parse(cls, s: str) -> "TypeSpec":
        s = s.lstrip(".")
        if s == "pred":
            return cls(ScalarKind.PRED, 1)
        kind_map = {"b": ScalarKind.B, "u": ScalarKind.U,
                    "s": ScalarKind.S, "f": ScalarKind.F}
        for prefix, kind in kind_map.items():
            if s.startswith(prefix):
                return cls(kind, int(s[len(prefix):]))
        raise ValueError(f"Unknown type spec: {s!r}")

# Common type shorthands
T = TypeSpec
B8  = T(ScalarKind.B, 8);  B16 = T(ScalarKind.B, 16)
B32 = T(ScalarKind.B, 32); B64 = T(ScalarKind.B, 64)
B128= T(ScalarKind.B, 128)
U8  = T(ScalarKind.U, 8);  U16 = T(ScalarKind.U, 16)
U32 = T(ScalarKind.U, 32); U64 = T(ScalarKind.U, 64)
S8  = T(ScalarKind.S, 8);  S16 = T(ScalarKind.S, 16)
S32 = T(ScalarKind.S, 32); S64 = T(ScalarKind.S, 64)
F16 = T(ScalarKind.F, 16); F32 = T(ScalarKind.F, 32); F64 = T(ScalarKind.F, 64)
PRED = T(ScalarKind.PRED, 1)


# ---------------------------------------------------------------------------
# Operands
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegOp:
    """Virtual register operand.  name includes the % sigil, e.g. '%rd7'."""
    name: str

    def __str__(self) -> str:
        return self.name

@dataclass(frozen=True)
class ImmOp:
    """Immediate integer operand."""
    value: int

    def __str__(self) -> str:
        return hex(self.value) if self.value >= 10 else str(self.value)

@dataclass(frozen=True)
class LabelOp:
    """Branch target label."""
    name: str

    def __str__(self) -> str:
        return self.name

@dataclass(frozen=True)
class MemOp:
    """Memory operand: [base + offset].  base is a label name or reg name."""
    base: str
    offset: int = 0

    def __str__(self) -> str:
        if self.offset == 0:
            return f"[{self.base}]"
        sign = "+" if self.offset >= 0 else "-"
        return f"[{self.base}{sign}{abs(self.offset)}]"

@dataclass(frozen=True)
class ConstBankOp:
    """Constant bank access: c[bank][offset]."""
    bank: int
    offset: int

    def __str__(self) -> str:
        return f"c[{hex(self.bank)}][{hex(self.offset)}]"

@dataclass(frozen=True)
class FpImmOp:
    """Floating-point immediate."""
    value: float

    def __str__(self) -> str:
        return repr(self.value)

Operand = Union[RegOp, ImmOp, LabelOp, MemOp, ConstBankOp, FpImmOp]


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

@dataclass
class Instruction:
    """
    A single PTX instruction.

    op:    base opcode string, e.g. "shl", "sub", "ld", "cvt"
    types: list of type qualifiers in order, e.g. ["b64"] or ["param", "u64"]
           (the dot-separated parts after the opcode, excluding the opcode itself)
    dest:  destination operand (None for void ops like 'ret', 'bar')
    srcs:  source operands in order
    pred:  predicate register name if guarded, e.g. "%p0"
    neg:   True if predicate is negated (@!%p0)
    mods:  extra modifier strings (e.g. "lo", "hi", "sat", "rn", "wide")
    label: label at this instruction (set when the instruction is the first in a block)
    """
    op:    str
    types: list[str]          = field(default_factory=list)
    dest:  Optional[Operand]  = None
    srcs:  list[Operand]      = field(default_factory=list)
    pred:  Optional[str]      = None
    neg:   bool               = False
    mods:  list[str]          = field(default_factory=list)

    def full_op(self) -> str:
        """Return the dotted opcode, e.g. 'shl.b64', 'ld.param.u64'."""
        parts = [self.op] + self.types
        return ".".join(parts)

    def __str__(self) -> str:
        pred_str = ""
        if self.pred:
            neg_str = "!" if self.neg else ""
            pred_str = f"@{neg_str}{self.pred} "

        dest_str = f"{self.dest}, " if self.dest is not None else ""
        srcs_str = ", ".join(str(s) for s in self.srcs)
        return f"{pred_str}{self.full_op()} {dest_str}{srcs_str}".strip()


# ---------------------------------------------------------------------------
# Function-level structures
# ---------------------------------------------------------------------------

@dataclass
class RegDecl:
    """
    .reg .<type> %name<count>;   or   .reg .<type> %name;
    Stored after expansion: one RegDecl per logical register group.
    """
    type:  TypeSpec
    name:  str          # base name without %, e.g. "rd"
    count: int = 1      # 1 for scalar, N for range %rdN<count>

    @property
    def names(self) -> list[str]:
        if self.count == 1:
            # If name ends with a digit, it's a direct declaration like .reg .u32 %r4
            # If not (like .reg .f32 %f<1>), the code uses %f0, so append index 0
            if self.name and self.name[-1].isdigit():
                return [f"%{self.name}"]
            else:
                return [f"%{self.name}0"]
        return [f"%{self.name}{i}" for i in range(self.count)]


@dataclass
class ParamDecl:
    """Kernel or function parameter declaration."""
    type:  TypeSpec
    name:  str          # includes leading dot-path, e.g. "kernel_param_0"
    align: Optional[int] = None


@dataclass
class BasicBlock:
    """
    Sequence of instructions identified by an optional entry label.
    The canonical form has exactly one label (at position 0) and no
    embedded labels — the parser splits on every label.
    """
    label:        Optional[str]
    instructions: list[Instruction] = field(default_factory=list)

    def __str__(self) -> str:
        lines = []
        if self.label:
            lines.append(f"{self.label}:")
        for inst in self.instructions:
            lines.append(f"    {inst};")
        return "\n".join(lines)


@dataclass
@dataclass
class SharedDecl:
    """A .shared variable declaration."""
    name:      str
    align:     int   # alignment in bytes
    elem_type: str   # e.g. 'b32', 'b8', 'f32'
    count:     int   # number of elements
    size:      int   # total size in bytes (elem_size * count)


@dataclass
class Function:
    """A PTX kernel (.entry) or device function (.func)."""
    name:      str
    is_kernel: bool
    params:    list[ParamDecl]       = field(default_factory=list)
    reg_decls: list[RegDecl]         = field(default_factory=list)
    blocks:    list[BasicBlock]      = field(default_factory=list)
    shared_decls: list[SharedDecl]   = field(default_factory=list)

    def all_instructions(self):
        for bb in self.blocks:
            yield from bb.instructions


# ---------------------------------------------------------------------------
# Module (top-level)
# ---------------------------------------------------------------------------

@dataclass
class Module:
    """A complete PTX translation unit (.ptx file)."""
    version:      tuple[int, int]    # (major, minor)
    target:       str                # e.g. "sm_120"
    address_size: int                # 32 or 64
    functions:    list[Function]     = field(default_factory=list)

    @property
    def kernels(self) -> list[Function]:
        return [f for f in self.functions if f.is_kernel]

    def dump(self) -> str:
        lines = [
            f".version {self.version[0]}.{self.version[1]}",
            f".target {self.target}",
            f".address_size {self.address_size}",
            "",
        ]
        for fn in self.functions:
            vis = ".visible .entry" if fn.is_kernel else ".visible .func"
            params = ",\n    ".join(
                f".param .{p.type} {p.name}" for p in fn.params
            )
            lines.append(f"{vis} {fn.name}(")
            if params:
                lines.append(f"    {params}")
            lines.append(")")
            lines.append("{")
            for rd in fn.reg_decls:
                for n in rd.names:
                    lines.append(f"    .reg .{rd.type} {n};")
            lines.append("")
            for bb in fn.blocks:
                lines.append(str(bb))
            lines.append("}")
            lines.append("")
        return "\n".join(lines)
