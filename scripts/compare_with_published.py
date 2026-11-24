#!/usr/bin/env python3
"""Compare a run you just made with the published run of the same id.

    python3 scripts/compare_with_published.py p6-arref-tokens201326592-seed6-drift

Reads results/camera_ready_2026/phase6/runs/<run id>/phase6_run_record.json (your run; retries live in
<run id>-attemptN, and the latest attempt that passed its integrity gate is used) and the same file under
results/camera_ready_2026/phase6.published/ (the published run, moved aside before re-running). Prints the main
metrics side by side. Standard library only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "camera_ready_2026"
KEYS = ("heldout_bpb", "validation_bpb", "tokens", "parameter_count", "steady_state_tokens_per_second")


def record(base: Path, run_id: str) -> dict | None:
    """The latest attempt that passed its integrity gate, else the latest attempt."""
    found = []
    for name in [run_id] + [f"{run_id}-attempt{n}" for n in range(2, 10)]:
        path = base / "runs" / name / "phase6_run_record.json"
        if path.is_file():
            found.append(json.loads(path.read_text()))
    passed = [r for r in found if (r.get("integrity") or {}).get("passed")]
    return (passed or found or [None])[-1]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    run_id = sys.argv[1]
    mine, published = record(ROOT / "phase6", run_id), record(ROOT / "phase6.published", run_id)
    if mine is None or published is None:
        missing = "your run" if mine is None else "the published run (move phase6 to phase6.published first)"
        print(f"not found: {missing} for {run_id}", file=sys.stderr)
        return 1
    print(f"{run_id}   status: yours={mine.get('status')}, published={published.get('status')}")
    print(f"{'metric':32s} {'yours':>16s} {'published':>16s} {'difference':>12s}")
    for key in KEYS:
        a, b = mine["metrics"].get(key), published["metrics"].get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            print(f"{key:32s} {a:16.6g} {b:16.6g} {a - b:+12.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
