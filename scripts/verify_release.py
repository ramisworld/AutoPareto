#!/usr/bin/env python3
"""Check the public release against its documented evidence boundaries."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = {
    "experiment_records.csv": 284,
    "inference_measurements.csv": 2784,
    "deployment_measurements.csv": 233,
}
EXPECTED_HASHES = {
    "data/experiment_records.csv": "851a32ef3376af950f16ca5620043418974b4664a59cf0507e646ffa7ae953ae",
    "data/inference_measurements.csv": "9ea74b42a44cf43a7ad0c788291875b8cca94f2d08e8bd44057c1f10310ef7df",
    "data/deployment_measurements.csv": "ab5fff9f46f2d46ca25370d290b2cfd4efbb8260e7582647c96ddf377b4eaf87",
    "configs/manifest.json": "906351590e8db39efc81c103e854a6cd2cf6cada0cc0ece91972a09c991f2953",
}


def rows(filename: str) -> list[dict[str, str]]:
    with (ROOT / "data" / filename).open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    experiment_rows = rows("experiment_records.csv")
    for filename, expected in EXPECTED_ROWS.items():
        actual = len(rows(filename))
        require(actual == expected, f"{filename}: expected {expected} rows, found {actual}")

    for relative, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        require(actual == expected, f"{relative}: hash does not match RELEASE.md")

    by_id = {row["experiment_id"]: row for row in experiment_rows}
    headline = by_id["137"]
    require(headline["campaign_id"] == "autopareto-seed0-20260731", "headline must be seed-0 A25")
    require(round(float(headline["val_bpb"]), 4) == 1.0974, "unexpected A25 validation BPB")
    require(round(float(headline["cached_decode_tokens_per_second"]), 1) == 408.4, "unexpected A25 throughput")
    require(round(float(headline["allocated_inference_memory_mib"]), 1) == 141.4, "unexpected A25 memory")

    manifest = json.loads((ROOT / "configs" / "manifest.json").read_text())
    for entry in manifest["configs"]:
        config = json.loads((ROOT / "configs" / entry["file"]).read_text())
        require(str(config["experiment_id"]) in by_id, f"missing experiment for {entry['file']}")
        require(len(config["source_file_sha256"]) == 64, f"invalid source hash in {entry['file']}")

    readme = (ROOT / "README.md").read_text()
    require("Confirmation retraining:** planned, not started." in readme, "README must state confirmation status")
    require("earlier cache-evaluation rules" in readme, "README must qualify AutoResearch comparison")
    require("confirmation retraining in progress" not in readme.lower(), "README must not imply an active confirmation run")
    require("500" not in readme, "README must not make unsupported 500-count claims")
    print("Release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

