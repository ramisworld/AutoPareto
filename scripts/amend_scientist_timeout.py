#!/usr/bin/env python3
"""PI-approved control-plane amendment: change the Scientist timeout for the remaining attempts of a paused arm.

This follows the form the comparator lineage used for the same change: ``operational_amendments`` of
``runs/pair-a-official-20260909-new-a40/provenance/final_operational_manifest_pair_a_20260909_t6_timeout900.json``,
recording ``scientist_timeout_seconds`` {before: 600, after: 900}.

The tool works on a paused campaign only. The next studied position must be ``pending`` (for example after
``scripts/requeue_scientist_timeout.py``), and no position may be running.

1. It writes a new, read-only manifest next to the current one. Only
   ``scientist.configuration.timeout_seconds`` changes, and one ``operational_amendments`` entry is appended.
2. It keeps a read-only copy of the controller state.
3. It points the state at the new manifest, sets ``scientist_config.timeout_seconds`` to the new value, and records
   the amendment in the state. The latest requeued timeout of the effective position is annotated with the amendment.

Code, immutable deployment, the evaluator, the contract and completed results are unchanged. Launch the controller
with ``--scientist-timeout-seconds`` equal to the new value and the new manifest's SHA-256.

    python3 -B scripts/amend_scientist_timeout.py --state-root DIR --manifest CURRENT.json \
        --new-timeout-seconds 900 --new-manifest-name NAME.json --reason TEXT --approval TEXT
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any

AMENDMENT_SCHEMA = "autopareto_control_plane_amendment_v1"
UNCHANGED = ["training", "N_STAR", "T_300s", "evaluation", "inference", "Pareto", "lineage", "T/N isolation",
             "schedule", "dataset", "tokenizer", "immutable deployment", "completed results"]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        pathlib.Path(temporary).unlink(missing_ok=True)
        raise


def amend_scientist_timeout(*, state_root: pathlib.Path, manifest_path: pathlib.Path, new_timeout_seconds: float,
                            new_manifest_name: str, reason: str, approval: str) -> dict[str, Any]:
    state_root = state_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    state_path = state_root / "controller_state.json"
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    manifest_sha = sha256_file(manifest_path)
    if state.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("controller state does not run under the given manifest")
    manifest = json.loads(manifest_path.read_text())
    configuration = (manifest.get("scientist") or {}).get("configuration")
    if not isinstance(configuration, dict) or configuration != state.get("scientist_config"):
        raise RuntimeError("manifest ScientistConfig and controller state disagree")
    before = float(configuration["timeout_seconds"])
    after = float(new_timeout_seconds)
    if after == before or after <= 0:
        raise ValueError("the new timeout must be positive and different from the current one")
    if not reason.strip() or not approval.strip():
        raise ValueError("an amendment needs a reason and the PI's approval")

    inactive = {"not_in_study"} if state.get("active_contracts") else set()
    statuses = [(int(row["global_position"]), state["positions"][str(row["global_position"])].get("status"))
                for row in state["schedule"]]
    active = [(position, status) for position, status in statuses if status not in inactive]
    if any(status not in {"completed", "pending"} for _position, status in active):
        raise RuntimeError("every studied position must be completed or pending (requeue a paused position first)")
    upcoming = [position for position, status in active if status == "pending"]
    if not upcoming:
        raise RuntimeError("no pending position remains")
    effective = upcoming[0]
    effective_row = next(row for row in state["schedule"] if int(row["global_position"]) == effective)

    new_manifest_path = manifest_path.parent / new_manifest_name
    if new_manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {new_manifest_path}")
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    amendment = {
        "schema": AMENDMENT_SCHEMA,
        "amended_at_utc": now,
        "effective_global_position": effective,
        "effective_campaign_attempt": int(effective_row["campaign_attempt"]),
        "applies_to": "this position and every later position of this arm",
        "prior_manifest_path": str(manifest_path),
        "prior_manifest_sha256": manifest_sha,
        "scientist_timeout_seconds": {"before": before, "after": after},
        "changed_immutable_paths": {},
        "reason": reason,
        "approval": approval,
        "scientific_components_unchanged": UNCHANGED,
        "scientific_contract_changed": False,
    }
    new_manifest = copy.deepcopy(manifest)
    new_manifest["scientist"]["configuration"]["timeout_seconds"] = after
    new_manifest.setdefault("operational_amendments", []).append(amendment)
    with open(new_manifest_path, "x") as handle:
        handle.write(json.dumps(new_manifest, indent=2, sort_keys=True) + "\n")
    new_manifest_path.chmod(0o444)
    new_sha = sha256_file(new_manifest_path)

    stamp = now.replace("-", "").replace(":", "")
    backup = state_root / f"controller_state.json.pre_timeout_amendment_{stamp}.json"
    with open(backup, "xb") as handle:
        handle.write(state_bytes)
    backup.chmod(0o444)
    record = {**amendment, "manifest_path": str(new_manifest_path), "manifest_sha256": new_sha,
              "pre_amendment_state_copy": backup.name,
              "pre_amendment_state_sha256": hashlib.sha256(state_bytes).hexdigest()}
    state["manifest_sha256"] = new_sha
    state["scientist_config"]["timeout_seconds"] = after
    state.setdefault("operational_amendments", []).append(record)
    requeues = state["positions"][str(effective)].get("requeued_scientist_timeouts")
    if requeues:
        requeues[-1]["followed_by_timeout_amendment"] = {
            "manifest_sha256": new_sha, "scientist_timeout_seconds": {"before": before, "after": after},
        }
    _write_json_atomic(state_path, state)
    return {"status": "amended", "manifest": str(new_manifest_path), "manifest_sha256": new_sha,
            "effective_global_position": effective, "scientist_timeout_seconds": {"before": before, "after": after},
            "controller_state_sha256": sha256_file(state_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-root", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--new-timeout-seconds", type=float, required=True)
    parser.add_argument("--new-manifest-name", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--approval", required=True)
    args = parser.parse_args()
    result = amend_scientist_timeout(
        state_root=args.state_root, manifest_path=args.manifest, new_timeout_seconds=args.new_timeout_seconds,
        new_manifest_name=args.new_manifest_name, reason=args.reason, approval=args.approval,
    )
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
