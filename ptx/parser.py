"""
PTX text parser.

Turns a .ptx file into a ptx.ir.Module.  Uses lark for grammar-based parsing
then a tree transformer to build the IR.

PTX ISA reference: https://docs.nvidia.com/cuda/parallel-thread-execution/
"""

from __future__ import annotations
import re
from typing import Optional

try:
    from lark import Lark, Transformer, v_args, Token, Tree
    LARK_AVAILABLE = True
except ImportError:
    LARK_AVAILABLE = False

from .ir import (
    Module, Function, BasicBlock, Instruction, RegDecl, ParamDecl,
    TypeSpec, RegOp, ImmOp, LabelOp, MemOp, ConstBankOp, FpImmOp,
    Operand, ScalarKind,
)

# ---------------------------------------------------------------------------
# PTX Lark grammar
# ---------------------------------------------------------------------------

PTX_GRAMMAR = r"""
    // -----------------------------------------------------------------------
    // Top level
    // -----------------------------------------------------------------------
    module: directive* item*

    directive: version_dir
             | target_dir
             | addrsize_dir
             | file_dir
             | pragma_dir
             | other_dir

    version_dir:  ".version" INT "." INT NEWLINE?
    target_dir:   ".target"  target_id ("," target_id)* NEWLINE?
    addrsize_dir: ".address_size" INT NEWLINE?
    file_dir:     ".file" /[^\n]*/
    pragma_dir:   ".pragma" /[^\n]*/
    other_dir:    ("." IDENT) /[^\n]*/

    target_id: IDENT | IDENT "_" IDENT | INT

    // -----------------------------------------------------------------------
    // Top-level items (functions / extern decls)
    // -----------------------------------------------------------------------
    item: func_def | extern_decl

    extern_decl: ".extern" func_proto ";"
    func_def:    visibility? func_proto func_body

    visibility: ".visible" | ".weak" | ".extern"
    func_proto: func_kind IDENT "(" param_list? ")" ret_param?
    func_kind:  ".entry" | ".func"
    ret_param:  "(" param_decl ")"

    param_list: param_decl ("," param_decl)*
    param_decl: param_attr* ".param" align? type_spec param_name
    param_attr: ".ptr" | ".const" | ".restrict" | ".noalias"
    param_name: IDENT | "."? IDENT

    align: ".align" INT

    // -----------------------------------------------------------------------
    // Function body
    // -----------------------------------------------------------------------
    func_body: "{" body_item* "}"

    body_item: reg_decl
             | local_decl
             | label_def
             | statement

    reg_decl:   ".reg"   type_spec reg_var ";"
    local_decl: ".local" type_spec IDENT ";"

    reg_var: REGNAME ("<" INT ">")? | REGNAME

    label_def: LABELNAME ":"

    // -----------------------------------------------------------------------
    // Statements (instructions)
    // -----------------------------------------------------------------------
    statement: pred? opcode operands? ";"
             | pred? opcode ";"

    pred: "@" "!"? REGNAME
    opcode: IDENT ("." (IDENT | INT))*
    operands: operand ("," operand)*

    operand: REGNAME
           | HEX_INT
           | SIGNED_INT
           | INT
           | fp_imm
           | mem_op
           | const_bank
           | label_ref
           | vector_op

    mem_op:    "[" (IDENT | REGNAME) ("+" signed_offset)? "]"
    const_bank: "c" "[" HEX_INT "]" "[" HEX_INT "]"
    label_ref: IDENT
    signed_offset: ("+"|"-")? INT | HEX_INT
    vector_op: "{" REGNAME ("," REGNAME)* "}"
    fp_imm: FLOAT | "0f" HEX_DIGITS | "0d" HEX_DIGITS

    // -----------------------------------------------------------------------
    // Type specs
    // -----------------------------------------------------------------------
    type_spec: "." type_name
    type_name: IDENT

    // -----------------------------------------------------------------------
    // Terminals
    // -----------------------------------------------------------------------
    LABELNAME:  /[a-zA-Z_$][a-zA-Z0-9_$]*(?=\s*:)/
    REGNAME:    /%[a-zA-Z_][a-zA-Z0-9_]*/
    HEX_INT:    /0[xX][0-9a-fA-F]+/
    HEX_DIGITS: /[0-9a-fA-F]+/
    SIGNED_INT: /[-+][0-9]+/
    FLOAT:      /[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?/
    INT:        /[0-9]+/
    IDENT:      /[a-zA-Z_$][a-zA-Z0-9_$.@]*/

    // Whitespace and comments
    %ignore /\s+/
    %ignore /\/\/[^\n]*/
    %ignore /\/\*(.|\n)*?\*\//
    NEWLINE: /\n/
    %ignore NEWLINE
"""


# ---------------------------------------------------------------------------
# Fallback: hand-rolled tokeniser/parser
# (Used when lark is unavailable or for simpler cases)
# ---------------------------------------------------------------------------

class _Token:
    __slots__ = ("kind", "value", "line")
    def __init__(self, kind: str, value: str, line: int = 0):
        self.kind  = kind
        self.value = value
        self.line  = line
    def __repr__(self):
        return f"Token({self.kind}, {self.value!r})"


_TOKEN_RE = re.compile(r"""
    (?P<COMMENT_BLOCK>  /\*.*?\*/          ) |
    (?P<COMMENT_LINE>   //[^\n]*           ) |
    (?P<STRING>         "(?:[^"\\]|\\.)*"  ) |
    (?P<FP_HEX>         0[fFdD][0-9a-fA-F]+ ) |
    (?P<HEX_INT>        0[xX][0-9a-fA-F]+ ) |
    (?P<FLOAT>          [-+]?[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)? ) |
    (?P<INT>            [0-9]+             ) |
    (?P<REGNAME>        %[a-zA-Z_][a-zA-Z0-9_.]* ) |
    (?P<LABEL_DEF>      [a-zA-Z_$][a-zA-Z0-9_$]*\s*:(?!:) ) |
    (?P<IDENT>          \.?[a-zA-Z_$][a-zA-Z0-9_$.@]* ) |
    (?P<PUNCT>          [(){}\[\],;@!+\-<>] ) |
    (?P<NEWLINE>        \n                 ) |
    (?P<WS>             [ \t\r]+           )
""", re.VERBOSE | re.DOTALL)


def _tokenize(src: str) -> list[_Token]:
    tokens = []
    line   = 1
    for m in _TOKEN_RE.finditer(src):
        kind = m.lastgroup
        val  = m.group()
        if kind in ("WS", "COMMENT_BLOCK", "COMMENT_LINE"):
            line += val.count("\n")
            continue
        if kind == "NEWLINE":
            line += 1
            continue
        tokens.append(_Token(kind, val.strip(), line))
    return tokens


class _Parser:
    """Hand-rolled recursive descent parser for PTX."""

    def __init__(self, tokens: list[_Token]):
        self._toks = tokens
        self._pos  = 0

    # -- utilities ----------------------------------------------------------

    def _peek(self) -> Optional[_Token]:
        return self._toks[self._pos] if self._pos < len(self._toks) else None

    def _advance(self) -> _Token:
        tok = self._toks[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: str, value: str | None = None) -> _Token:
        tok = self._peek()
        if tok is None:
            raise SyntaxError(f"Unexpected EOF, expected {kind}")
        if tok.kind != kind:
            raise SyntaxError(
                f"Line {tok.line}: expected {kind}, got {tok.kind}={tok.value!r}"
            )
        if value is not None and tok.value != value:
            raise SyntaxError(
                f"Line {tok.line}: expected {value!r}, got {tok.value!r}"
            )
        return self._advance()

    def _match(self, kind: str, value: str | None = None) -> Optional[_Token]:
        tok = self._peek()
        if tok and tok.kind == kind and (value is None or tok.value == value):
            return self._advance()
        return None

    def _match_ident(self, *values: str) -> Optional[_Token]:
        tok = self._peek()
        if tok and tok.kind == "IDENT" and (not values or tok.value in values):
            return self._advance()
        return None

    def _at_end(self) -> bool:
        return self._pos >= len(self._toks)

    # -- grammar productions ------------------------------------------------

    def parse_module(self) -> Module:
        version      = (8, 0)
        target       = "sm_120"
        address_size = 64
        functions: list[Function] = []

        while not self._at_end():
            tok = self._peek()
            if tok is None:
                break

            if tok.kind == "IDENT" and tok.value == ".version":
                self._advance()
                # ".version 8.0" — the tokenizer may produce a single FLOAT
                # token (e.g. "8.0") or two INTs separated by a PUNCT "."
                vt = self._peek()
                if vt and vt.kind == "FLOAT":
                    self._advance()
                    parts = vt.value.split(".")
                    version = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
                else:
                    major = int(self._expect("INT").value)
                    self._match("PUNCT", ".")
                    minor_tok = self._peek()
                    minor = int(minor_tok.value) if minor_tok and minor_tok.kind == "INT" else 0
                    if minor_tok and minor_tok.kind == "INT":
                        self._advance()
                    version = (major, minor)

            elif tok.kind == "IDENT" and tok.value == ".target":
                self._advance()
                # consume until non-ident
                t = self._peek()
                if t:
                    target = t.value
                    self._advance()
                # skip extra targets (e.g. texmode_unified)
                while self._match("PUNCT", ","):
                    self._advance()

            elif tok.kind == "IDENT" and tok.value == ".address_size":
                self._advance()
                address_size = int(self._expect("INT").value)

            elif tok.kind == "IDENT" and tok.value in (".visible", ".weak", ".extern",
                                                        ".entry", ".func"):
                fn = self._parse_function()
                if fn:
                    functions.append(fn)

            elif tok.kind == "IDENT" and tok.value.startswith("."):
                # some other directive — consume to end of line-ish
                self._advance()
                while not self._at_end():
                    t = self._peek()
                    if t and t.kind == "PUNCT" and t.value == ";":
                        self._advance(); break
                    if t and t.kind == "IDENT" and t.value in (
                            ".visible", ".weak", ".extern", ".entry", ".func"):
                        break
                    self._advance()

            else:
                # unrecognized — skip token
                self._advance()

        return Module(version=version, target=target,
                      address_size=address_size, functions=functions)

    def _parse_function(self) -> Optional[Function]:
        # Optional visibility
        is_kernel = False
        tok = self._peek()
        if tok and tok.value in (".visible", ".weak", ".extern"):
            self._advance()

        tok = self._peek()
        if tok is None:
            return None
        if tok.value == ".entry":
            is_kernel = True
        elif tok.value == ".func":
            is_kernel = False
        elif tok.value == ".extern":
            # extern func declaration — skip to ;
            self._advance()
            while not self._at_end():
                t = self._peek()
                if t and t.kind == "PUNCT" and t.value == ";":
                    self._advance(); break
                self._advance()
            return None
        else:
            return None
        self._advance()  # consume .entry / .func

        # Optional return type for .func: (.param ...)
        if self._peek() and self._peek().value == "(":
            # check if this is a return param
            saved = self._pos
            self._advance()  # (
            t = self._peek()
            if t and t.value == ".param":
                # skip return param
                while not self._at_end():
                    t = self._peek()
                    if t and t.kind == "PUNCT" and t.value == ")":
                        self._advance(); break
                    self._advance()
            else:
                self._pos = saved  # not a return param, back up

        # Function name
        name_tok = self._peek()
        if name_tok is None or name_tok.kind not in ("IDENT", "REGNAME"):
            return None
        name = name_tok.value
        self._advance()

        # Parameter list
        params: list[ParamDecl] = []
        if self._match("PUNCT", "("):
            params = self._parse_param_list()
            self._expect("PUNCT", ")")

        # Body or semicolon (extern)
        if self._peek() and self._peek().value == ";":
            self._advance()
            return Function(name=name, is_kernel=is_kernel, params=params)

        if self._peek() and self._peek().value == "{":
            reg_decls, blocks = self._parse_func_body()
            return Function(name=name, is_kernel=is_kernel, params=params,
                            reg_decls=reg_decls, blocks=blocks)

        return Function(name=name, is_kernel=is_kernel, params=params)

    def _parse_param_list(self) -> list[ParamDecl]:
        params = []
        while True:
            # skip commas
            self._match("PUNCT", ",")
            tok = self._peek()
            if tok is None or (tok.kind == "PUNCT" and tok.value == ")"):
                break
            if tok.kind == "IDENT" and tok.value == ".param":
                p = self._parse_param_decl()
                if p:
                    params.append(p)
            else:
                # skip unexpected token
                self._advance()
        return params

    def _parse_param_decl(self) -> Optional[ParamDecl]:
        self._expect("IDENT", ".param")

        # optional .ptr / .const / .restrict etc.
        while self._peek() and self._peek().value in (".ptr", ".const",
                                                       ".restrict", ".noalias"):
            self._advance()

        # optional .align N
        align = None
        if self._peek() and self._peek().value == ".align":
            self._advance()
            align = int(self._expect("INT").value)

        # type spec: .u64 / .b32 etc.
        type_spec = self._parse_type_spec()
        if type_spec is None:
            return None

        # name
        tok = self._peek()
        if tok is None:
            return None
        name = tok.value
        self._advance()
        return ParamDecl(type=type_spec, name=name, align=align)

    def _parse_type_spec(self) -> Optional[TypeSpec]:
        tok = self._peek()
        if tok is None:
            return None
        # type specs look like .u64, .b32, .pred, .f32, etc.
        val = tok.value
        if not val.startswith("."):
            return None
        val = val.lstrip(".")
        try:
            ts = TypeSpec.parse(val)
            self._advance()
            return ts
        except ValueError:
            return None

    def _parse_func_body(self) -> tuple[list[RegDecl], list[BasicBlock]]:
        self._expect("PUNCT", "{")
        reg_decls: list[RegDecl]   = []
        blocks:    list[BasicBlock] = []
        current_block = BasicBlock(label=None)
        blocks.append(current_block)

        while not self._at_end():
            tok = self._peek()
            if tok is None:
                break
            if tok.kind == "PUNCT" and tok.value == "}":
                self._advance()
                break

            # .reg declaration
            if tok.kind == "IDENT" and tok.value == ".reg":
                rds = self._parse_reg_decl()
                if rds:
                    reg_decls.extend(rds)
                continue

            # .local / .shared / other body-level directives
            if tok.kind == "IDENT" and tok.value.startswith("."):
                self._consume_to_semicolon()
                continue

            # label definition
            if tok.kind == "LABEL_DEF":
                label = tok.value.rstrip(": \t")
                self._advance()
                new_block = BasicBlock(label=label)
                blocks.append(new_block)
                current_block = new_block
                continue

            # instruction (possibly predicated)
            inst = self._parse_statement()
            if inst:
                current_block.instructions.append(inst)

        # Drop empty leading block with no label
        blocks = [b for b in blocks if b.label or b.instructions]
        return reg_decls, blocks

    def _parse_reg_decl(self) -> Optional[list]:
        """Parse a .reg declaration, returning a list of RegDecl (handles comma lists)."""
        self._expect("IDENT", ".reg")
        type_spec = self._parse_type_spec()
        if type_spec is None:
            self._consume_to_semicolon()
            return None

        results = []
        while True:
            tok = self._peek()
            if tok is None:
                break

            # register name: %rd<10> or %rd0 or %p0
            if tok.kind != "REGNAME":
                self._consume_to_semicolon()
                break

            reg_tok = self._advance()
            reg_name = reg_tok.value.lstrip("%")
            count = 1

            # optional <N>
            if self._match("PUNCT", "<"):
                count_tok = self._expect("INT")
                count = int(count_tok.value)
                self._expect("PUNCT", ">")

            results.append(RegDecl(type=type_spec, name=reg_name, count=count))

            # comma-separated list: .reg .pred %p0, %p1;
            if not self._match("PUNCT", ","):
                break

        self._match("PUNCT", ";")
        return results if results else None

    def _parse_statement(self) -> Optional[Instruction]:
        # Predicate guard: @%p0 or @!%p0
        pred     = None
        pred_neg = False
        if self._match("PUNCT", "@"):
            neg_tok = self._match("PUNCT", "!")
            pred_neg = neg_tok is not None
            pred_tok = self._peek()
            if pred_tok and pred_tok.kind == "REGNAME":
                pred = pred_tok.value
                self._advance()

        # Opcode (possibly dotted: shl.b64, ld.param.u64, cvt.rn.f32.s64)
        tok = self._peek()
        if tok is None:
            return None
        if tok.kind != "IDENT":
            # not an instruction
            self._consume_to_semicolon()
            return None

        op_parts = tok.value.lstrip(".").split(".")
        op = op_parts[0]
        types = op_parts[1:]
        self._advance()

        # Accumulate any additional .qualifier tokens
        while True:
            t = self._peek()
            if t and t.kind == "IDENT" and t.value.startswith(".") and t.value != ".":
                extra = t.value.lstrip(".")
                types.append(extra)
                self._advance()
            else:
                break

        # Operands until semicolon
        operands: list[Operand] = []
        while True:
            t = self._peek()
            if t is None:
                break
            if t.kind == "PUNCT" and t.value == ";":
                self._advance()
                break
            if t.kind == "PUNCT" and t.value == ",":
                self._advance()
                continue
            op_node = self._parse_operand()
            if op_node is not None:
                operands.append(op_node)
            else:
                # Unknown token in operand position — skip to avoid infinite loop
                self._advance()

        # Split dest from srcs (dest is first if instruction has a destination)
        dest = operands[0] if operands else None
        srcs = operands[1:] if operands else []

        # Ops that have no destination (void)
        VOID_OPS = {"ret", "bra", "call", "bar", "membar", "exit", "trap",
                    "st", "prefetch", "prefetchu", "wgmma"}
        if op in VOID_OPS:
            dest = None
            srcs = operands

        return Instruction(op=op, types=types, dest=dest, srcs=srcs,
                           pred=pred, neg=pred_neg)

    def _parse_operand(self) -> Optional[Operand]:
        tok = self._peek()
        if tok is None:
            return None

        # Register
        if tok.kind == "REGNAME":
            self._advance()
            return RegOp(tok.value)

        # Constant bank: c[0x0][0x28]
        if tok.kind == "IDENT" and tok.value == "c":
            self._advance()
            self._expect("PUNCT", "[")
            bank = self._parse_int_literal()
            self._expect("PUNCT", "]")
            self._expect("PUNCT", "[")
            off = self._parse_int_literal()
            self._expect("PUNCT", "]")
            return ConstBankOp(bank=bank, offset=off)

        # Memory operand: [base] or [base+offset]
        if tok.kind == "PUNCT" and tok.value == "[":
            self._advance()
            base_tok = self._peek()
            if base_tok is None:
                return None
            base = base_tok.value
            self._advance()
            offset = 0
            # optional +offset
            if self._match("PUNCT", "+"):
                offset = self._parse_int_literal()
            elif self._match("PUNCT", "-"):
                offset = -self._parse_int_literal()
            self._expect("PUNCT", "]")
            return MemOp(base=base, offset=offset)

        # Vector operand: {%r0, %r1, ...}  (treat as first register)
        if tok.kind == "PUNCT" and tok.value == "{":
            self._advance()
            first = self._peek()
            result = None
            if first and first.kind == "REGNAME":
                result = RegOp(first.value)
            while not self._at_end():
                t = self._peek()
                if t and t.kind == "PUNCT" and t.value == "}":
                    self._advance(); break
                self._advance()
            return result

        # Float hex literal: 0fXXXXXXXX (32-bit) or 0dXXXXXXXXXXXXXXXX (64-bit)
        if tok.kind == "FP_HEX":
            self._advance()
            prefix = tok.value[:2]  # "0f" or "0d"
            hex_digits = tok.value[2:]
            bits = int(hex_digits, 16)
            if prefix.lower() == '0f':
                # 32-bit float: store as ImmOp with raw IEEE 754 bits
                return ImmOp(bits)
            else:
                # 64-bit double: store as ImmOp with raw IEEE 754 bits
                return ImmOp(bits)

        # Hex literal
        if tok.kind == "HEX_INT":
            self._advance()
            return ImmOp(int(tok.value, 16))

        # Decimal / signed int
        if tok.kind in ("INT", "SIGNED_INT"):
            self._advance()
            return ImmOp(int(tok.value))

        # Float
        if tok.kind == "FLOAT":
            self._advance()
            return FpImmOp(float(tok.value))

        # IDENT — could be a label reference
        if tok.kind == "IDENT":
            self._advance()
            return LabelOp(tok.value)

        return None

    def _parse_int_literal(self) -> int:
        tok = self._peek()
        if tok is None:
            return 0
        if tok.kind == "HEX_INT":
            self._advance()
            return int(tok.value, 16)
        if tok.kind in ("INT", "SIGNED_INT"):
            self._advance()
            return int(tok.value)
        return 0

    def _consume_to_semicolon(self):
        while not self._at_end():
            t = self._peek()
            if t and t.kind == "PUNCT" and t.value == ";":
                self._advance()
                return
            self._advance()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(src: str) -> Module:
    """Parse PTX source text and return a Module IR."""
    tokens = _tokenize(src)
    parser = _Parser(tokens)
    return parser.parse_module()


def parse_file(path: str) -> Module:
    """Load a .ptx file and parse it."""
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    return parse(src)
