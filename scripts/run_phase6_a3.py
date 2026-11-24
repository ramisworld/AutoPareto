#!/usr/bin/env python3
"""Phase 6 amendment A3 (``evaluation/PHASE6_AMENDMENT_A3.md``): run the frozen P1 downstream stage.

    uv run --locked python scripts/run_phase6_a3.py                                     # validate only
    uv run --locked python scripts/run_phase6_a3.py --execute --expect-plan-sha256 <A3 sha>

The parent runner's ``execute_downstream`` opens its new database without ``row_factory =
sqlite3.Row``, which Phase 5A's function of the same purpose sets, and ``autopareto_db.migrate``
needs. A3 calls ``execute_downstream`` unchanged, with ``sqlite3.connect`` returning connections that
set that one attribute, for the duration of the call. The empty database left by the crash is renamed
aside first, never deleted.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
A3_PLAN_PATH = ROOT / "evaluation/phase6_amendment_a3_plan.json"
A3_FORMAT = "autopareto_phase6_amendment_a3_plan_v1"

spec = importlib.util.spec_from_file_location("phase6_parent_runner", ROOT / "scripts/run_phase6.py")
rp6 = importlib.util.module_from_spec(spec)
sys.modules["phase6_parent_runner"] = rp6
spec.loader.exec_module(rp6)
sha256, read_json, write_json_new, append_jsonl, utc_now = rp6.sha256, rp6.read_json, rp6.write_json_new, rp6.append_jsonl, rp6.utc_now


def load_a3() -> tuple[dict, str]:
    raw = A3_PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    if plan.get("format") != A3_FORMAT:
        raise RuntimeError(f"unexpected A3 plan format {plan.get('format')}")
    return plan, hashlib.sha256(raw).hexdigest()


def validate(a3: dict) -> list[dict]:
    rows = []

    def add(name: str, ok: bool, detail) -> None:
        rows.append({"check": name, "status": "ok" if ok else "FAIL", "detail": detail})

    parent, parent_sha = rp6.load_plan()
    add("parent_plan_sha256", parent_sha == a3["parent_plan_sha256"], parent_sha)
    parent_rows = rp6.validate(parent)
    add("parent_plan_checks", all(r["status"] == "ok" for r in parent_rows),
        [r for r in parent_rows if r["status"] != "ok"] or f"{len(parent_rows)} ok")
    bad = {rel: sha256(ROOT / rel) if (ROOT / rel).is_file() else None for rel, exp in a3["implementation_sha256"].items()
           if not (ROOT / rel).is_file() or sha256(ROOT / rel) != exp}
    add("a3_implementation_sha256", not bad, bad or f"{len(a3['implementation_sha256'])} files match")
    reference = ROOT / a3["reference"]["file"]
    add("phase5a_reference_line", reference.is_file() and a3["reference"]["line"] in reference.read_text(), a3["reference"])
    root = rp6.stage_root(parent)
    db = root / a3["empty_database"]["path"]
    aside = root / a3["empty_database"]["aside"]
    if aside.exists():
        ok = sha256(aside) == a3["empty_database"]["sha256"]
        add("empty_database_kept_aside", ok, str(aside))
    else:
        ok = db.is_file() and sha256(db) == a3["empty_database"]["sha256"]
        add("empty_database_as_left_by_crash", ok, str(db))
    return rows


@contextlib.contextmanager
def connections_with_row_factory():
    """Make the parent runner's ``sqlite3.connect`` set ``row_factory = sqlite3.Row`` (Phase 5A's line)."""

    class Shim:
        def __getattr__(self, name):
            return getattr(sqlite3, name)

        @staticmethod
        def connect(*args, **kwargs):
            db = sqlite3.connect(*args, **kwargs)
            db.row_factory = sqlite3.Row
            return db

    original = rp6.sqlite3
    rp6.sqlite3 = Shim()
    try:
        yield
    finally:
        rp6.sqlite3 = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expect-plan-sha256")
    args = parser.parse_args()
    a3, a3_sha = load_a3()
    if args.expect_plan_sha256 and args.expect_plan_sha256 != a3_sha:
        print(json.dumps({"status": "refused", "reason": "A3 plan sha256 mismatch", "a3_plan_sha256": a3_sha}))
        return 2
    rows = validate(a3)
    failures = [row for row in rows if row["status"] != "ok"]
    if not args.execute:
        print(json.dumps({"status": "validated" if not failures else "FAILED", "a3_plan_sha256": a3_sha, "checks": rows}, indent=1))
        return 0 if not failures else 1
    if failures:
        print(json.dumps({"status": "refused", "failures": failures}, indent=1))
        return 2
    parent, parent_sha = rp6.load_plan()
    root = rp6.stage_root(parent)
    events = root / "a3_events.jsonl"
    with (root / "gpu.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "refused", "reason": "GPU lock held"}))
            return 2
        db, aside = root / a3["empty_database"]["path"], root / a3["empty_database"]["aside"]
        if not aside.exists():
            db.rename(aside)
            append_jsonl(events, {"utc": utc_now(), "event": "empty_database_renamed", "from": str(db), "to": str(aside),
                                  "sha256": sha256(aside)})
        append_jsonl(events, {"utc": utc_now(), "event": "downstream_started", "a3_plan_sha256": a3_sha})
        with connections_with_row_factory():
            rc = rp6.execute_downstream(parent, parent_sha, root)
        append_jsonl(events, {"utc": utc_now(), "event": "downstream_finished", "exit": rc})
    print(json.dumps({"status": "finished", "exit": rc, "a3_plan_sha256": a3_sha}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
