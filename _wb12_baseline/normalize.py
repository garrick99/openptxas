"""WB-12.0 normalization helper.

Strips non-deterministic fields (timestamps + measured timings) from
workbench artifacts and stdout so the structural shape can be hashed
across runs to prove zero behavior change in the refactor.
"""
import hashlib
import json
import re
import sys
from pathlib import Path


def normalize_json(path: Path) -> dict:
    """Load workbench JSON artifact, strip non-deterministic fields.

    Preserves: schema, kernel, display, mode, repeat, build, correctness,
    structural fields (regs, sass_total, sass_non_nop), commits, deltas
    (regs, sass_total, sass_non_nop), metadata.

    Replaces with placeholder: timestamp, all *_ms / time_ms* fields.

    Returns the normalized dict (key order preserved).
    """
    data = json.loads(path.read_text())
    if "timestamp" in data:
        data["timestamp"] = "<TIMESTAMP>"
    for side in ("ours", "ptxas"):
        if side in data and isinstance(data[side], dict):
            d = data[side]
            if "compile_ms" in d:
                d["compile_ms"] = "<MEASURED>"
            if "time_ms_runs" in d:
                d["time_ms_runs"] = "<MEASURED>"
            if "time_ms_stats" in d:
                d["time_ms_stats"] = "<MEASURED>"
    if "deltas" in data and isinstance(data["deltas"], dict):
        if "time_ms_mean" in data["deltas"]:
            data["deltas"]["time_ms_mean"] = "<MEASURED>"
    return data


def normalize_stdout(text: str) -> str:
    """Normalize workbench stdout: strip timing values + artifact filename."""
    # compile_ms / time_ms numeric values: replace with <MEASURED>
    text = re.sub(r"(compile_ms:\s+)[\d\.]+", r"\1<MEASURED>", text)
    text = re.sub(r"(time_ms:\s+)[\d\.]+", r"\1<MEASURED>", text)
    # time_ms_mean carries a sign that flips with measurement noise (which side
    # of the compare was faster on this particular run), so eat the sign too.
    text = re.sub(r"(time_ms_mean:\s*)[+\-]?[\d\.]+", r"\1<MEASURED>", text)
    # artifact path with embedded timestamp
    text = re.sub(
        r"(artifact: [^\s]*?)\d{8}_\d{6}(_[^\s]+)",
        r"\1<TIMESTAMP>\2",
        text,
    )
    return text


def hash_normalized_json(path: Path) -> str:
    norm = normalize_json(path)
    canonical = json.dumps(norm, indent=2, sort_keys=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def hash_normalized_stdout(path: Path) -> str:
    text = normalize_stdout(path.read_text())
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: normalize.py <stdout.txt> <artifact.json>")
        sys.exit(1)
    stdout_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])
    print(f"STDOUT_NORM_SHA256: {hash_normalized_stdout(stdout_path)}")
    print(f"JSON_NORM_SHA256:   {hash_normalized_json(json_path)}")
    print()
    print("=== normalized stdout ===")
    print(normalize_stdout(stdout_path.read_text()))
    print("=== normalized json ===")
    print(json.dumps(normalize_json(json_path), indent=2, sort_keys=False))
