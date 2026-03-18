"""
test_encoding.py — Round-trip encoding tests for SM_120 SHF instructions.

Tests encode_shf_l_w_u32_hi, encode_shf_l_u32, encode_shf_l_u64_hi against
ground truth bytes extracted from actual ptxas SM_120 cubin output.

Run:
    python -m pytest tests/test_encoding.py -v
or:
    python tests/test_encoding.py
"""

import sys
import os

# Make sure the package root is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sass.encoding.sm_120_encode import (
    encode_shf_l_w_u32_hi,
    encode_shf_l_u32,
    encode_shf_l_u64_hi,
    roundtrip_verify,
    decode_shf_bytes,
    RZ,
)

# ---------------------------------------------------------------------------
# Ground truth table
# Each entry: (label, encode_call, expected_hex_str)
# ctrl values are taken verbatim from ptxas cubin output (sm_120_encoding_tables.json).
# ---------------------------------------------------------------------------

GROUND_TRUTH_CASES: list[tuple[str, bytes, str]] = [
    # --- SHF.L.W.U32.HI ---
    # From JSON (rotate kernel, K=0x1f):
    (
        "SHF.L.W.U32.HI R5,R3,0x1f,R2",
        encode_shf_l_w_u32_hi(5, 3, 0x1f, 2, ctrl=0x047f2),
        "197805031f000000020e010000e48f00",
    ),
    (
        "SHF.L.W.U32.HI R4,R2,0x1f,R3",
        encode_shf_l_w_u32_hi(4, 2, 0x1f, 3, ctrl=0x007e5),
        "197804021f000000030e010000ca0f00",
    ),
    # K=0x8:
    (
        "SHF.L.W.U32.HI R5,R3,0x8,R2",
        encode_shf_l_w_u32_hi(5, 3, 0x8, 2, ctrl=0x047f2),
        "1978050308000000020e010000e48f00",
    ),
    (
        "SHF.L.W.U32.HI R4,R2,0x8,R3",
        encode_shf_l_w_u32_hi(4, 2, 0x8, 3, ctrl=0x007e5),
        "1978040208000000030e010000ca0f00",
    ),
    # Swapped dest/src (K=0x1f, different dest):
    (
        "SHF.L.W.U32.HI R4,R3,0x1f,R2",
        encode_shf_l_w_u32_hi(4, 3, 0x1f, 2, ctrl=0x047f2),
        "197804031f000000020e010000e48f00",
    ),
    (
        "SHF.L.W.U32.HI R5,R2,0x1f,R3",
        encode_shf_l_w_u32_hi(5, 2, 0x1f, 3, ctrl=0x007e5),
        "197805021f000000030e010000ca0f00",
    ),
    # K=1 (from fresh ptxas probe, k=1 run):
    (
        "SHF.L.W.U32.HI R4,R3,0x1,R2",
        encode_shf_l_w_u32_hi(4, 3, 0x1, 2, ctrl=0x047f2),
        "1978040301000000020e010000e48f00",
    ),
    (
        "SHF.L.W.U32.HI R5,R2,0x1,R3",
        encode_shf_l_w_u32_hi(5, 2, 0x1, 3, ctrl=0x007e5),
        "1978050201000000030e010000ca0f00",
    ),
    # Additional K values from probe (k=2..7):
    (
        "SHF.L.W.U32.HI R4,R3,0x2,R2",
        encode_shf_l_w_u32_hi(4, 3, 0x2, 2, ctrl=0x047f2),
        "1978040302000000020e010000e48f00",
    ),
    (
        "SHF.L.W.U32.HI R4,R3,0x3,R2",
        encode_shf_l_w_u32_hi(4, 3, 0x3, 2, ctrl=0x047f2),
        "1978040303000000020e010000e48f00",
    ),
    (
        "SHF.L.W.U32.HI R4,R3,0x4,R2",
        encode_shf_l_w_u32_hi(4, 3, 0x4, 2, ctrl=0x047f2),
        "1978040304000000020e010000e48f00",
    ),
    (
        "SHF.L.W.U32.HI R4,R3,0x6,R2",
        encode_shf_l_w_u32_hi(4, 3, 0x6, 2, ctrl=0x047f2),
        "1978040306000000020e010000e48f00",
    ),
    (
        "SHF.L.W.U32.HI R4,R3,0xc,R2",
        encode_shf_l_w_u32_hi(4, 3, 0xc, 2, ctrl=0x047f2),
        "197804030c000000020e010000e48f00",
    ),
    (
        "SHF.L.W.U32.HI R4,R3,0x14,R2",
        encode_shf_l_w_u32_hi(4, 3, 0x14, 2, ctrl=0x047f2),
        "1978040314000000020e010000e48f00",
    ),
    (
        "SHF.L.W.U32.HI R4,R3,0x18,R2",
        encode_shf_l_w_u32_hi(4, 3, 0x18, 2, ctrl=0x047f2),
        "1978040318000000020e010000e48f00",
    ),
    (
        "SHF.L.W.U32.HI R4,R3,0x10,R2",
        encode_shf_l_w_u32_hi(4, 3, 0x10, 2, ctrl=0x047f2),
        "1978040310000000020e010000e48f00",
    ),

    # --- SHF.L.U32 (src1 = RZ = 255) ---
    (
        "SHF.L.U32 R9,R2,0x1f,RZ",
        encode_shf_l_u32(9, 2, 0x1f, ctrl=0x007e4),
        "197809021f000000ff06000000c80f00",
    ),
    (
        "SHF.L.U32 R9,R2,0x8,RZ",
        encode_shf_l_u32(9, 2, 0x8, ctrl=0x007e4),
        "1978090208000000ff06000000c80f00",
    ),
    (
        "SHF.L.U32 R8,R2,0x1f,RZ",
        encode_shf_l_u32(8, 2, 0x1f, ctrl=0x007f1),
        "197808021f000000ff06000000e20f00",
    ),
    (
        "SHF.L.U32 R8,R2,0x8,RZ",
        encode_shf_l_u32(8, 2, 0x8, ctrl=0x007f1),
        "1978080208000000ff06000000e20f00",
    ),
    (
        "SHF.L.U32 R8,R2,0x1,RZ",
        encode_shf_l_u32(8, 2, 0x1, ctrl=0x007f1),
        "1978080201000000ff06000000e20f00",
    ),
    (
        "SHF.L.U32 R9,R2,0x18,RZ",
        encode_shf_l_u32(9, 2, 0x18, ctrl=0x007e4),
        "1978090218000000ff06000000c80f00",
    ),
    (
        "SHF.L.U32 R9,R2,0x10,RZ",
        encode_shf_l_u32(9, 2, 0x10, ctrl=0x007e4),
        "1978090210000000ff06000000c80f00",
    ),

    # --- SHF.L.U64.HI ---
    (
        "SHF.L.U64.HI R9,R2,0x1f,R3",
        encode_shf_l_u64_hi(9, 2, 0x1f, 3, ctrl=0x207f2),
        "197809021f0000000302010000e40f04",
    ),
    (
        "SHF.L.U64.HI R9,R2,0x8,R3",
        encode_shf_l_u64_hi(9, 2, 0x8, 3, ctrl=0x207f2),
        "19780902080000000302010000e40f04",
    ),
    (
        "SHF.L.U64.HI R9,R2,0x1,R3",
        encode_shf_l_u64_hi(9, 2, 0x1, 3, ctrl=0x207f2),
        "19780902010000000302010000e40f04",
    ),
    (
        "SHF.L.U64.HI R9,R2,0x18,R3",
        encode_shf_l_u64_hi(9, 2, 0x18, 3, ctrl=0x207f2),
        "19780902180000000302010000e40f04",
    ),
    (
        "SHF.L.U64.HI R9,R2,0x10,R3",
        encode_shf_l_u64_hi(9, 2, 0x10, 3, ctrl=0x207f2),
        "19780902100000000302010000e40f04",
    ),
]


# ---------------------------------------------------------------------------
# Pytest-compatible tests
# ---------------------------------------------------------------------------

def _run_case(label: str, got: bytes, expected_hex: str):
    """Helper: compare bytes against expected hex."""
    expected = bytes.fromhex(expected_hex)
    assert got == expected, (
        f"\n  Case: {label}\n"
        f"  expected: {expected.hex()}\n"
        f"  got:      {got.hex()}\n"
        + "  diffs at bytes: "
        + str([i for i in range(16) if got[i] != expected[i]])
    )


def test_shf_l_w_u32_hi_basic():
    """SHF.L.W.U32.HI — basic K=0x1f and K=0x8 cases."""
    cases = [c for c in GROUND_TRUTH_CASES if c[0].startswith("SHF.L.W.U32.HI") and
             ("0x1f" in c[0] or "0x8," in c[0])]
    for label, got, expected_hex in cases:
        _run_case(label, got, expected_hex)


def test_shf_l_w_u32_hi_k_sweep():
    """SHF.L.W.U32.HI — K values 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 31."""
    cases = [c for c in GROUND_TRUTH_CASES if c[0].startswith("SHF.L.W.U32.HI")]
    for label, got, expected_hex in cases:
        _run_case(label, got, expected_hex)


def test_shf_l_u32_basic():
    """SHF.L.U32 — various K values with RZ src1."""
    cases = [c for c in GROUND_TRUTH_CASES if c[0].startswith("SHF.L.U32")]
    for label, got, expected_hex in cases:
        _run_case(label, got, expected_hex)


def test_shf_l_u64_hi_basic():
    """SHF.L.U64.HI — various K values."""
    cases = [c for c in GROUND_TRUTH_CASES if c[0].startswith("SHF.L.U64.HI")]
    for label, got, expected_hex in cases:
        _run_case(label, got, expected_hex)


def test_decode_roundtrip():
    """encode then decode should recover all field values."""
    cases = [
        ("SHF.L.W.U32.HI", encode_shf_l_w_u32_hi(5, 3, 0x1f, 2, ctrl=0x047f2),
         {"dest": "R5", "src0": "R3", "k": 31, "src1": "R2", "variant": "SHF.L.W.U32.HI"}),
        ("SHF.L.U32", encode_shf_l_u32(9, 2, 0x1f, ctrl=0x007e4),
         {"dest": "R9", "src0": "R2", "k": 31, "src1": "RZ", "variant": "SHF.L.U32"}),
        ("SHF.L.U64.HI", encode_shf_l_u64_hi(9, 2, 0x1f, 3, ctrl=0x207f2),
         {"dest": "R9", "src0": "R2", "k": 31, "src1": "R3", "variant": "SHF.L.U64.HI"}),
    ]
    for label, raw, expected_fields in cases:
        decoded = decode_shf_bytes(raw)
        for field, value in expected_fields.items():
            assert decoded[field] == value, (
                f"{label}: field {field!r}: got {decoded[field]!r}, expected {value!r}"
            )


def test_opcode_constant():
    """All SHF variants must encode opcode 0x819 in bits [11:0]."""
    import struct
    cases = [
        encode_shf_l_w_u32_hi(4, 2, 0x1f, 3, ctrl=0x007e5),
        encode_shf_l_u32(9, 2, 0x1f, ctrl=0x007e4),
        encode_shf_l_u64_hi(9, 2, 0x1f, 3, ctrl=0x207f2),
    ]
    for raw in cases:
        lo = struct.unpack_from('<Q', raw, 0)[0]
        opcode = lo & 0xFFF
        assert opcode == 0x819, f"Expected opcode=0x819, got 0x{opcode:03x}"


def test_field_positions_independent():
    """
    Verify fields are independently settable by sweeping each field
    while holding others constant, checking only that field changes.
    """
    import struct

    # dest field at bits[23:16]
    for dest in [0, 1, 4, 7, 15, 31, 127, 254]:
        raw = encode_shf_l_w_u32_hi(dest, 3, 0x1f, 2, ctrl=0x047f2)
        lo = struct.unpack_from('<Q', raw, 0)[0]
        assert (lo >> 16) & 0xFF == dest, f"dest field wrong for dest={dest}"

    # src0 field at bits[31:24]
    for src0 in [0, 1, 2, 3, 5, 10, 31, 100, 200, 254]:
        raw = encode_shf_l_w_u32_hi(4, src0, 0x1f, 2, ctrl=0x047f2)
        lo = struct.unpack_from('<Q', raw, 0)[0]
        assert (lo >> 24) & 0xFF == src0, f"src0 field wrong for src0={src0}"

    # K field at bits[39:32]
    for k in range(32):
        raw = encode_shf_l_w_u32_hi(4, 3, k, 2, ctrl=0x047f2)
        lo = struct.unpack_from('<Q', raw, 0)[0]
        assert (lo >> 32) & 0xFF == k, f"K field wrong for k={k}"

    # src1 field at bits[71:64] = hi bits[7:0]
    for src1 in [0, 1, 2, 3, 5, 10, 31, 100, 200, 254, 255]:
        raw = encode_shf_l_w_u32_hi(4, 3, 0x1f, src1, ctrl=0x047f2)
        import struct as _s
        hi = _s.unpack_from('<Q', raw, 8)[0]
        assert hi & 0xFF == src1, f"src1 field wrong for src1={src1}"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_standalone():
    """Run all tests with PASS/FAIL output when executed directly."""
    print("=" * 70)
    print("SM_120 SHF Instruction Encoding Tests")
    print("=" * 70)
    print()

    total = 0
    passed = 0
    failed = 0

    # Test 1: Ground truth table (all cases)
    print(f"[1] Ground truth round-trip ({len(GROUND_TRUTH_CASES)} cases):")
    for label, got, expected_hex in GROUND_TRUTH_CASES:
        total += 1
        expected = bytes.fromhex(expected_hex)
        if got == expected:
            passed += 1
            print(f"  PASS  {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}")
            print(f"        expected: {expected.hex()}")
            print(f"        got:      {got.hex()}")
            diffs = [i for i in range(16) if got[i] != expected[i]]
            print(f"        bad bytes: {diffs}")

    print()

    # Test 2: roundtrip_verify() helper
    print("[2] roundtrip_verify() (internal ground truth):")
    ok = roundtrip_verify(verbose=False)
    total += 1
    if ok:
        passed += 1
        print("  PASS  roundtrip_verify() returned True")
    else:
        failed += 1
        print("  FAIL  roundtrip_verify() returned False")

    print()

    # Test 3: Opcode constant
    print("[3] Opcode constant 0x819:")
    import struct
    opcode_cases = [
        ("SHF.L.W.U32.HI", encode_shf_l_w_u32_hi(4, 2, 0x1f, 3, ctrl=0x007e5)),
        ("SHF.L.U32",       encode_shf_l_u32(9, 2, 0x1f, ctrl=0x007e4)),
        ("SHF.L.U64.HI",    encode_shf_l_u64_hi(9, 2, 0x1f, 3, ctrl=0x207f2)),
    ]
    for name, raw in opcode_cases:
        total += 1
        lo = struct.unpack_from('<Q', raw, 0)[0]
        opcode = lo & 0xFFF
        if opcode == 0x819:
            passed += 1
            print(f"  PASS  {name} opcode=0x{opcode:03x}")
        else:
            failed += 1
            print(f"  FAIL  {name} opcode=0x{opcode:03x} (expected 0x819)")

    print()

    # Test 4: Field independence sweep
    print("[4] Field independence sweep (dest, src0, K, src1):")
    try:
        test_field_positions_independent()
        total += 1
        passed += 1
        print("  PASS  all field sweeps correct")
    except AssertionError as e:
        total += 1
        failed += 1
        print(f"  FAIL  {e}")

    print()

    # Test 5: Decode roundtrip
    print("[5] Decode roundtrip (encode -> decode -> verify fields):")
    try:
        test_decode_roundtrip()
        total += 1
        passed += 1
        print("  PASS  decoded fields match expectations")
    except AssertionError as e:
        total += 1
        failed += 1
        print(f"  FAIL  {e}")

    print()
    print("=" * 70)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    ok = run_standalone()
    sys.exit(0 if ok else 1)
