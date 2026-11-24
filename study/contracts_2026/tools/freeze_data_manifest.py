"""Freeze deterministic hashes for the protected data/tokenizer cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_files(root: pathlib.Path, suffix: str | None = None) -> list[pathlib.Path]:
    paths = [path for path in root.rglob("*") if path.is_file() and not path.name.endswith(".tmp")]
    if suffix is not None:
        paths = [path for path in paths if path.suffix == suffix]
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def build_data_manifest(cache_dir: pathlib.Path) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    data_dir = cache_dir / "data"
    tokenizer_dir = cache_dir / "tokenizer"
    if not data_dir.is_dir() or not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"expected data/ and tokenizer/ beneath {cache_dir}")
    parquet_files = _relative_files(data_dir, ".parquet")
    tokenizer_files = _relative_files(tokenizer_dir)
    if not parquet_files:
        raise ValueError("no parquet shards found")
    validation_name = "shard_06542.parquet"
    if not any(path.name == validation_name for path in parquet_files):
        raise ValueError(f"pinned validation shard {validation_name} is missing")

    def record(path: pathlib.Path, split: str) -> dict[str, Any]:
        relative = path.relative_to(cache_dir).as_posix()
        return {
            "path": relative,
            "split": split,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    files = [
        *(record(path, "validation" if path.name == validation_name else "train") for path in parquet_files),
        *(record(path, "tokenizer") for path in tokenizer_files),
    ]
    files.sort(key=lambda item: item["path"])
    payload: dict[str, Any] = {
        "schema": "autopareto_contracts_2026_data_manifest_v1",
        "cache_root_label": "autoresearch_cache",
        "validation_shard": validation_name,
        "sorted_file_order": [item["path"] for item in files],
        "files": files,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    payload["aggregate_manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = build_data_manifest(args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": len(payload["files"]), "aggregate_manifest_sha256": payload["aggregate_manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
