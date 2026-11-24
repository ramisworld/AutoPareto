#!/usr/bin/env python3
"""Write evaluation/phase6_amendment_a2_plan.json (Phase 6 amendment A2). Run on the pod, after the
A1 plan has been regenerated; refuses to overwrite unless --force."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/phase6_amendment_a2_plan.json"
PARENT = ROOT / "evaluation/phase6_unified_plan.json"
PARENT_SHA = "dc1e605e82c5f598bbf91ab17f63d1db30a27511f5d264142c9b31eb6fe61096"
A1_PLAN = ROOT / "evaluation/phase6_amendment_a1_plan.json"
RUN_ID = "p6-motpe-tokens201326592-seed8"
ATTEMPT = 3
LOCAL_ROOT = "/root/phase6_local_a2"
IMPLEMENTATION = ["evaluation/PHASE6_AMENDMENT_A2.md", "scripts/run_phase6_a2.py", "scripts/make_phase6_amendment_a2.py",
                  "scripts/run_phase6_p2.py"]
APPROVAL = ("Author's choice, 2026-11-16: 'Amendment A2 rerun (Recommended)': run MOTPE seed 8 once more (attempt 3), "
            "with the same code and gates, on local disk under the A1 storage rules; keep both failed attempts on "
            "record; run its downstream and P0b inference with the frozen functions; resume the remaining chain "
            "stages unchanged.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {OUT}")
    if sha256(PARENT) != PARENT_SHA:
        raise SystemExit("parent plan hash differs")
    parent = json.loads(PARENT.read_text())
    run = next(r for r in parent["runs"] if r["run_id"] == RUN_ID)
    root = ROOT / parent["output_root"]
    failed = []
    for attempt in (1, 2):
        rel = f"runs/{RUN_ID if attempt == 1 else f'{RUN_ID}-attempt{attempt}'}/phase6_run_record.json"
        record = json.loads((root / rel).read_text())
        if record["integrity"]["passed"] or record["attempt"] != attempt:
            raise SystemExit(f"{rel} is not a failed attempt {attempt}")
        failed.append({"attempt": attempt, "record": rel, "sha256": sha256(root / rel),
                       "failures": record["integrity"]["failures"][:2]})
    labels = sorted(r["label"] for r in parent["inference"]["runs"]
                    if r["block"] == "P0b" and r["recipe"] == run["recipe"] and r.get("seed") == run["seed"])
    if not labels:
        raise SystemExit("no P0b label for this run")
    plan = {
        "format": "autopareto_phase6_amendment_a2_plan_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "approval": APPROVAL,
        "parent_plan_sha256": PARENT_SHA,
        "a1_plan_sha256": sha256(A1_PLAN),
        "implementation_sha256": {rel: sha256(ROOT / rel) for rel in IMPLEMENTATION},
        "run": {"run_id": RUN_ID, "recipe": run["recipe"], "seed": run["seed"], "attempt": ATTEMPT,
                "attempt_dirname": f"{RUN_ID}-attempt{ATTEMPT}"},
        "failed_attempts": failed,
        "cause": "OSError [Errno 116] Stale file handle while Triton wrote its cache in the run directory on the network volume (incident 3)",
        "local_root": LOCAL_ROOT,
        "inference_labels": labels,
        "timing_anomaly": "computed with A1's gate settings and reported; not a gate, as P1 matched-token runs have none",
    }
    OUT.write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({"written": str(OUT), "sha256": sha256(OUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
