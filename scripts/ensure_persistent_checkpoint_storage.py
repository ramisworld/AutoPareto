#!/usr/bin/env python3
"""Require RunPod training checkpoints to live on persistent /workspace storage."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "results" / "checkpoints"
PERSISTENT_ROOT = Path("/workspace")


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    # Local development does not have RunPod's /workspace mount. Its ordinary
    # results/checkpoints directory is already durable relative to the checkout.
    if not PERSISTENT_ROOT.is_dir():
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"status": "ok", "mode": "local", "checkpoint_root": str(CHECKPOINTS)}))
        return 0

    if CHECKPOINTS.is_symlink():
        target = CHECKPOINTS.resolve(strict=False)
        if not _inside(target, PERSISTENT_ROOT):
            # This legacy symlink targeted /tmp/autoresearch-checkpoints. Repair it
            # before runner.py can reserve a scientific attempt.
            CHECKPOINTS.unlink()
            CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    resolved = CHECKPOINTS.resolve(strict=True)
    persistent = PERSISTENT_ROOT.resolve(strict=True)
    if not _inside(resolved, persistent):
        raise RuntimeError(
            f"checkpoint root must resolve beneath persistent {persistent}: {resolved}"
        )
    probe = resolved / ".persistence-probe"
    probe.write_text("ok\n")
    os.sync()
    probe.unlink()
    print(json.dumps({
        "status": "ok",
        "mode": "runpod_persistent",
        "checkpoint_root": str(resolved),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
