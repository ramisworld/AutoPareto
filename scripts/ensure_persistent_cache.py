#!/usr/bin/env python3
"""Bind the protected runtime cache to persistent storage before reserving a run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REQUIRED = (
    Path("tokenizer/tokenizer.pkl"),
    Path("tokenizer/token_bytes.pt"),
    Path("data/shard_06542.parquet"),
)


def ensure_cache(runtime: Path, persistent: Path, *, require_complete: bool = True) -> dict:
    runtime = runtime.expanduser()
    persistent = persistent.expanduser().resolve()
    persistent.mkdir(parents=True, exist_ok=True)
    runtime.parent.mkdir(parents=True, exist_ok=True)
    if runtime.is_symlink():
        if runtime.resolve() != persistent:
            raise RuntimeError(
                f"runtime cache points to {runtime.resolve()}, expected {persistent}"
            )
    elif runtime.exists():
        if runtime.resolve() != persistent:
            raise RuntimeError(
                f"runtime cache {runtime} is non-persistent; move it to {persistent} "
                "during setup, then replace it with a symlink"
            )
    else:
        runtime.symlink_to(persistent, target_is_directory=True)

    missing = [str(path) for path in REQUIRED if not (persistent / path).is_file()]
    train_shards = sorted((persistent / "data").glob("shard_*.parquet"))
    train_shards = [path for path in train_shards if path.name != "shard_06542.parquet"]
    if require_complete and (missing or not train_shards):
        detail = missing + ([] if train_shards else ["data/<training shard>.parquet"])
        raise RuntimeError(
            "persistent protected cache is incomplete: " + ", ".join(detail)
        )

    result = {
        "status": "ok" if require_complete else "persistent_link_ready",
        "runtime_cache": str(runtime),
        "persistent_cache": str(persistent),
        "runtime_resolves_to": str(runtime.resolve()),
        "training_shards": len(train_shards),
        "required_files": [str(path) for path in REQUIRED],
        "missing_files": missing + ([] if train_shards else ["data/<training shard>.parquet"]),
        "complete": not missing and bool(train_shards),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path.home() / ".cache" / "autoresearch",
    )
    parser.add_argument(
        "--persistent",
        type=Path,
        default=Path(os.environ.get(
            "AUTOPARETO_PERSISTENT_CACHE", "/workspace/.cache/autoresearch"
        )),
    )
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="create/verify the persistent symlink before running prepare.py",
    )
    args = parser.parse_args()
    print(json.dumps(
        ensure_cache(args.runtime, args.persistent, require_complete=not args.initialize),
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
