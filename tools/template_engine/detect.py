"""Auto-detect candidate template regions in PTXAS cubins and cluster them.

TEMPLATE-ENGINE-2: region detection + opcode-sequence clustering.
"""
from __future__ import annotations

import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ── Opcode classification ────────────────────────────────────────────────

_UR_OPCODES = {0x919, 0x3c4, 0x835, 0x886, 0x2bd, 0x882, 0x890, 0xc82, 0x291}
_LDCU_OPCODES = {0x7ac}
_MEM_OPCODES = {0x981, 0x986, 0x984, 0x98e, 0x9a8, 0x9ae}  # LDG, STG, LDS, ATOMG variants
_TENSOR_OPCODES = {0x23c, 0x237, 0x23f, 0x27a, 0x47f}  # HMMA, IMMA, DMMA, QMMA, OMMA
_FLUSH_OPCODES = {0xc0c}  # ISETP.RUR (used as UR flush)
_MOV_UR = {0xc02}
_NOP = 0x918


def _get_opcode(raw: bytes) -> int:
    return int.from_bytes(raw[:2], "little") & 0xFFF


# ── Cubin parsing ────────────────────────────────────────────────────────

def _iter_text_sections(cubin: bytes):
    """Yield (section_name, [(index, raw_16_bytes), ...]) for each .text section."""
    e_shoff = struct.unpack_from("<Q", cubin, 40)[0]
    e_shnum = struct.unpack_from("<H", cubin, 60)[0]
    e_shstrndx = struct.unpack_from("<H", cubin, 62)[0]
    stoff = struct.unpack_from("<Q", cubin, e_shoff + e_shstrndx * 64 + 24)[0]
    for i in range(e_shnum):
        base = e_shoff + i * 64
        nm = struct.unpack_from("<I", cubin, base)[0]
        ne = cubin.index(0, stoff + nm)
        name = cubin[stoff + nm:ne].decode("ascii", errors="replace")
        if not name.startswith(".text."):
            continue
        off = struct.unpack_from("<Q", cubin, base + 24)[0]
        sz = struct.unpack_from("<Q", cubin, base + 32)[0]
        instrs = []
        for j in range(0, sz, 16):
            raw = cubin[off + j:off + j + 16]
            if len(raw) == 16:
                instrs.append((j // 16, bytes(raw)))
        kname = name[len(".text."):]
        yield kname, instrs


# ── Region detection ─────────────────────────────────────────────────────

@dataclass
class Region:
    """A candidate template region within a kernel."""
    kernel: str
    start: int          # instruction index
    end: int            # instruction index (exclusive)
    opcodes: tuple[int, ...]
    raw_bytes: list[bytes]
    tags: set[str] = field(default_factory=set)

    @property
    def length(self) -> int:
        return self.end - self.start

    def opcode_key(self) -> str:
        """Opcode sequence as a hashable string for clustering."""
        return ",".join(f"{o:03x}" for o in self.opcodes)


def detect_regions(kernel_name: str, instrs: list[tuple[int, bytes]],
                   min_len: int = 6, max_len: int = 24) -> list[Region]:
    """Detect candidate template regions using opcode heuristics.

    A region is a candidate if it contains at least one UR instruction
    and at least one memory/atomic instruction, OR matches known patterns.
    """
    # Filter NOPs
    active = [(idx, raw) for idx, raw in instrs if _get_opcode(raw) != _NOP]
    if len(active) < min_len:
        return []

    # Strategy: find contiguous windows containing UR + memory ops.
    # Start from each UR instruction and extend forward to the next
    # memory/atomic op (or end of active region).
    regions = []

    # Approach 1: full active region as one candidate (simple kernels)
    opcodes = tuple(_get_opcode(raw) for _, raw in active)
    tags = set()
    opc_set = set(opcodes)
    if opc_set & _UR_OPCODES:
        tags.add("UR")
    if opc_set & _MEM_OPCODES:
        tags.add("MEM")
    if opc_set & _TENSOR_OPCODES:
        tags.add("TENSOR")
    if opc_set & _FLUSH_OPCODES:
        tags.add("FLUSH")
    if opc_set & _MOV_UR:
        tags.add("MOV_UR")
    if opc_set & _LDCU_OPCODES:
        tags.add("LDCU")

    if ("UR" in tags or "LDCU" in tags) and ("MEM" in tags or "TENSOR" in tags):
        regions.append(Region(
            kernel=kernel_name,
            start=active[0][0],
            end=active[-1][0] + 1,
            opcodes=opcodes,
            raw_bytes=[raw for _, raw in active],
            tags=tags,
        ))

    return regions


# ── Clustering ───────────────────────────────────────────────────────────

@dataclass
class Cluster:
    """A group of regions with the same opcode sequence."""
    cluster_id: int
    opcode_key: str
    members: list[Region] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def representative(self) -> Region:
        return self.members[0]

    def opcodes(self) -> tuple[int, ...]:
        return self.representative.opcodes


def cluster_regions(regions: list[Region]) -> list[Cluster]:
    """Cluster regions by exact opcode sequence."""
    by_key: dict[str, list[Region]] = defaultdict(list)
    for r in regions:
        by_key[r.opcode_key()].append(r)

    clusters = []
    for cid, (key, members) in enumerate(sorted(by_key.items(),
                                                  key=lambda kv: -len(kv[1]))):
        clusters.append(Cluster(cluster_id=cid, opcode_key=key, members=members))
    return clusters


# ── High-level API ───────────────────────────────────────────────────────

def scan_cubins(cubins: dict[str, bytes]) -> tuple[list[Region], list[Cluster]]:
    """Scan multiple cubins and return all regions + clusters.

    Parameters
    ----------
    cubins : dict[str, bytes]
        kernel_name -> cubin_bytes

    Returns
    -------
    (all_regions, clusters)
    """
    all_regions = []
    for kname, cubin in cubins.items():
        for sec_name, instrs in _iter_text_sections(cubin):
            regions = detect_regions(sec_name, instrs)
            all_regions.extend(regions)

    clusters = cluster_regions(all_regions)
    return all_regions, clusters


def find_atom_xor_cluster(clusters: list[Cluster]) -> Optional[Cluster]:
    """Find the cluster that matches the known atom.xor pattern (has 0x98e)."""
    for c in clusters:
        if 0x98e in c.opcodes():
            return c
    return None
