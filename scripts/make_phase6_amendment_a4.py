#!/usr/bin/env python3
"""Write evaluation/phase6_amendment_a4_plan.json (Phase 6 amendment A4) after the author's approval.

    python3 scripts/make_phase6_amendment_a4.py --approval "<the author's words>"

Run on the pod. Refuses to overwrite, and refuses unless the original cross-check report exists and
records a failed gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/phase6_amendment_a4_plan.json"
A1 = ROOT / "evaluation/phase6_amendment_a1_plan.json"
A1_SHA = "2b9fdc3ef962277e78e5e6d98f4265c28033a324d288c536245fab1c14631ade"
PARENT_SHA = "dc1e605e82c5f598bbf91ab17f63d1db30a27511f5d264142c9b31eb6fe61096"
IMPLEMENTATION = ["evaluation/PHASE6_AMENDMENT_A4.md", "scripts/run_phase6_a4.py", "scripts/make_phase6_amendment_a4.py",
                  "scripts/phase6_a4_blimp_diagnostic.py", "evaluation/phase6_a4_blimp_diagnostic.json",
                  "scripts/run_phase6_p2.py"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", required=True)
    args = parser.parse_args()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    if sha256(A1) != A1_SHA:
        raise SystemExit("A1 plan hash differs")
    a1 = json.loads(A1.read_text())
    parent = json.loads((ROOT / "evaluation/phase6_unified_plan.json").read_text())
    report = ROOT / parent["output_root"] / "lmeval/crosscheck_report.json"
    original = json.loads(report.read_text())
    if original.get("passed") is not False:
        raise SystemExit("original cross-check report does not record a failed gate")
    excluded = ["blimp"]
    compared = sorted(t for t in a1["lmeval"]["crosscheck_metrics"] if t not in excluded)
    plan = {
        "format": "autopareto_phase6_amendment_a4_plan_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "approval": args.approval,
        "parent_plan_sha256": PARENT_SHA,
        "a1_plan_sha256": A1_SHA,
        "implementation_sha256": {rel: sha256(ROOT / rel) for rel in IMPLEMENTATION},
        "original_crosscheck_report": {"path": str(report.relative_to(ROOT)), "sha256": sha256(report),
                                       "passed": original["passed"], "recorded_utc": original["recorded_utc"]},
        "gate": {"compared": compared, "excluded": excluded,
                 "tolerance_points": a1["lmeval"]["crosscheck_tolerance_points"]},
    }
    OUT.write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({"written": str(OUT), "sha256": sha256(OUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
