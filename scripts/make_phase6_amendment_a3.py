#!/usr/bin/env python3
"""Write evaluation/phase6_amendment_a3_plan.json (Phase 6 amendment A3). Run on the pod before A3
runs, while the empty database is still in place; refuses to overwrite unless --force."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/phase6_amendment_a3_plan.json"
PARENT = ROOT / "evaluation/phase6_unified_plan.json"
PARENT_SHA = "dc1e605e82c5f598bbf91ab17f63d1db30a27511f5d264142c9b31eb6fe61096"
IMPLEMENTATION = ["evaluation/PHASE6_AMENDMENT_A3.md", "scripts/run_phase6_a3.py", "scripts/make_phase6_amendment_a3.py"]
DB = "downstream_frozen/phase6_downstream.db"
ASIDE = "downstream_frozen/phase6_downstream.db.incident4-empty"
APPROVAL = ("Author's choice, 2026-11-16: 'Approve A3 fix (Recommended)': run the frozen downstream function unchanged "
            "through a small wrapper that adds only Phase 5A's row_factory line; frozen files stay unchanged; rename the "
            "empty database aside rather than deleting it; run P0b, controls and anchors first, then downstream, the A1 "
            "switch, A2 and the A1 pilot.")


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
    root = ROOT / parent["output_root"]
    db = root / DB
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    counts = {name: con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
              for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    if any(counts.values()):
        raise SystemExit(f"database is not empty: {counts}")
    plan = {
        "format": "autopareto_phase6_amendment_a3_plan_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "approval": APPROVAL,
        "parent_plan_sha256": PARENT_SHA,
        "implementation_sha256": {rel: sha256(ROOT / rel) for rel in IMPLEMENTATION},
        "reference": {"file": "scripts/run_phase5a_opus_fixed_tokens.py", "line": "db.row_factory = sqlite3.Row"},
        "empty_database": {"path": DB, "aside": ASIDE, "sha256": sha256(db), "bytes": db.stat().st_size,
                           "row_counts": counts},
    }
    OUT.write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({"written": str(OUT), "sha256": sha256(OUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
