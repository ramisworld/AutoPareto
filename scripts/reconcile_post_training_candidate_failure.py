#!/usr/bin/env python3
"""PI-approved, one-time reconciliation of a post-training failure as a candidate scientific failure.

This follows the comparator's handling of N19 (position 37) and N20 (position 39) in
``final_pair_a_continuation_manifest_20260917_posttraining_failure_receipt_fix.json``. Training completed, then the
protected post-training check rejected the candidate. The launcher recorded the origin as ``unknown``, because
the message was not yet listed in ``train_launcher._CANDIDATE_POST_TRAINING_FAILURE_MARKERS``, so the runner wrote
a FAILED durable receipt and the controller paused with ``RemoteInfrastructureFailure``.

After the PI approves charging the failure to the candidate, and the message has been added to the launcher's list
(tested and committed), this tool:

1. Refuses unless the paused position, its durable receipt and the pod evidence agree exactly. The evidence
   folder must hold the pod's receipt, log, worker receipt and launcher output, each matching its pod SHA-256, and
   must show no live worker and no bridge result.
2. Classifies the recorded error with the repository's ``train_launcher._post_training_failure_origin``; it must
   now be ``candidate``. The error text is never edited.
3. Writes a new read-only manifest. Only ``immutable_deployment["study/contracts_2026/train_launcher.py"]``
   changes, and one ``operational_amendments`` entry is appended.
4. Keeps a read-only copy of the controller state.
5. Builds the runner's result with ``remote_runner._classify_worker_failure``, exactly as the runner would have
   with the new launcher. Then commits it through the controller's own ``_commit_pending_result``. The failure,
   with its hypothesis, patch and exact error, enters the campaign view's ``same_arm_evidence`` and
   ``failures``, the history and the result file, and the position becomes ``completed`` with outcome
   ``scientific_failure``. Its ``reconciliation_failure`` record is kept.

No candidate is rerun and no Scientist call is made. Deploy the launcher to the pod's trusted tree, run the launch
gates, then relaunch the controller with the new manifest.

    python3 -B scripts/reconcile_post_training_candidate_failure.py --state-root DIR --manifest CURRENT.json \
        --new-manifest-name NAME.json --global-position N --evidence-dir DIR --reason TEXT --approval TEXT
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study.contracts_2026 import campaign_controller as controller  # noqa: E402
from study.contracts_2026 import remote_runner, train_launcher  # noqa: E402
from study.contracts_2026.scheduling import ScheduledAttempt  # noqa: E402

AMENDMENT_SCHEMA = "autopareto_operational_amendment_v1"
EVIDENCE_SCHEMA = "autopareto_post_training_failure_evidence_v1"
LAUNCHER = "study/contracts_2026/train_launcher.py"
UNKNOWN_ORIGIN_MESSAGE = "candidate worker failure origin is unknown; manual reconciliation is required"
EVIDENCE_FILES = ("invocation_receipt.json", "invocation.log", "worker_failure_receipt.json", "launcher_output.json")
UNCHANGED = ["training", "N_STAR", "T_300s", "validation", "inference", "Pareto", "lineage", "T/N isolation",
             "schedule", "dataset", "tokenizer", "completed results"]
PRECEDENT = ("Codex comparator N19 (position 37) and N20 (position 39), operational_amendments of "
             "final_pair_a_continuation_manifest_20260917_posttraining_failure_receipt_fix.json")


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strip_type_prefix(error: str) -> str:
    """The launcher records f"{type(exc).__name__}: {exc}"; the classifier sees str(exc)."""
    head, separator, tail = error.partition(": ")
    return tail if separator and head.isidentifier() else error


def _load_evidence(evidence_dir: pathlib.Path, receipt: dict[str, Any]) -> dict[str, Any]:
    evidence_dir = evidence_dir.resolve(strict=True)
    absence = json.loads((evidence_dir / "absence.json").read_text())
    if absence.get("schema") != EVIDENCE_SCHEMA:
        raise RuntimeError("evidence has an unexpected schema")
    if absence.get("invocation_id") != receipt.get("invocation_id"):
        raise RuntimeError("evidence belongs to a different invocation")
    if absence.get("matching_processes") != [] or absence.get("bridge_result_present") is not False:
        raise RuntimeError("evidence does not show the worker gone and no bridge result")
    files = absence.get("pod_files") or {}
    hashes = {}
    for name in EVIDENCE_FILES:
        local = evidence_dir / name
        recorded = (files.get(name) or {}).get("sha256")
        if not local.is_file() or sha256_file(local) != recorded:
            raise RuntimeError(f"evidence file {name} is missing or differs from the pod's SHA-256")
        hashes[name] = recorded
    if json.loads((evidence_dir / "invocation_receipt.json").read_text()) != receipt:
        raise RuntimeError("the pod's durable receipt differs from the receipt the controller recorded")
    worker = json.loads((evidence_dir / "worker_failure_receipt.json").read_text())
    if worker.get("status") != "error" or worker.get("failure_origin") != "unknown" or not worker.get("error"):
        raise RuntimeError("the worker receipt is not an unknown-origin error receipt")
    launcher_output = json.loads((evidence_dir / "launcher_output.json").read_text())
    if launcher_output.get("status") != "error" or launcher_output.get("error") != worker["error"]:
        raise RuntimeError("the launcher output and the worker receipt disagree")
    final_line = worker["error"].strip().splitlines()[-1]
    if final_line not in (evidence_dir / "invocation.log").read_text():
        raise RuntimeError("the invocation log does not contain the recorded error")
    return {"absence": absence, "worker": worker, "hashes": hashes, "dir": evidence_dir}


def reconcile(*, state_root: pathlib.Path, manifest_path: pathlib.Path, new_manifest_name: str,
              global_position: int, evidence_dir: pathlib.Path, reason: str, approval: str) -> dict[str, Any]:
    if not reason.strip() or not approval.strip():
        raise ValueError("a reconciliation needs a reason and the PI's approval")
    state_root = state_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    state_path = state_root / "controller_state.json"
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    manifest_sha = sha256_file(manifest_path)
    if state.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("controller state does not run under the given manifest")
    manifest = json.loads(manifest_path.read_text())

    key = str(global_position)
    position = state["positions"].get(key)
    if not isinstance(position, dict) or position.get("status") != "reconciliation_required":
        raise RuntimeError(f"position {global_position} is not reconciliation_required")
    failure = position.get("reconciliation_failure") or {}
    receipt = failure.get("receipt") or {}
    if failure.get("type") != "RemoteInfrastructureFailure" or receipt.get("status") != "FAILED":
        raise RuntimeError("the pause is not a FAILED durable receipt")
    if (receipt.get("failure") or {}).get("message") != UNKNOWN_ORIGIN_MESSAGE:
        raise RuntimeError("the receipt did not fail on an unknown worker failure origin")
    execution = position.get("execution") or {}
    [schedule_row] = [row for row in state["schedule"] if int(row["global_position"]) == global_position]
    metadata = receipt.get("metadata") or {}
    if (receipt.get("invocation_id") != execution.get("invocation_id")
            or int(metadata.get("global_position", -1)) != global_position
            or int(metadata.get("campaign_attempt", -1)) != int(schedule_row["campaign_attempt"])
            or metadata.get("campaign_id") != schedule_row["campaign_id"]
            or metadata.get("candidate_source_sha256") != execution.get("candidate_source_after_sha256")
            or metadata.get("parent_source_sha256") != execution.get("candidate_source_before_sha256")):
        raise RuntimeError("the receipt does not match the position's persisted execution")
    inactive = {"not_in_study"} if state.get("active_contracts") else set()
    others = [(int(row["global_position"]), state["positions"][str(row["global_position"])].get("status"))
              for row in state["schedule"] if int(row["global_position"]) != global_position]
    if any(status not in {"completed", "pending"} | inactive for _p, status in others):
        raise RuntimeError("every other position must be completed or pending")

    evidence = _load_evidence(evidence_dir, receipt)
    error = str(evidence["worker"]["error"])
    origin = train_launcher._post_training_failure_origin(RuntimeError(_strip_type_prefix(error)))
    if origin != "candidate":
        raise RuntimeError(f"the repository's launcher classifies this error as {origin!r}, not 'candidate'")
    launcher_after = sha256_file(ROOT / LAUNCHER)
    launcher_before = (manifest.get("immutable_deployment") or {}).get(LAUNCHER)
    if not launcher_before or launcher_before == launcher_after:
        raise RuntimeError("the manifest must pin a different train_launcher.py than the repository's")

    new_manifest_path = manifest_path.parent / new_manifest_name
    if new_manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {new_manifest_path}")
    pending = [int(row["global_position"]) for row in state["schedule"]
               if state["positions"][str(row["global_position"])].get("status") == "pending"]
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    evidence_record = {
        "folder": str(evidence["dir"]),
        "files_sha256": evidence["hashes"],
        "absence_json_sha256": sha256_file(evidence["dir"] / "absence.json"),
        "checked_utc": evidence["absence"].get("checked_utc"),
    }
    amendment = {
        "schema": AMENDMENT_SCHEMA,
        "amended_at_utc": now,
        "changed_immutable_paths": {LAUNCHER: {"before": launcher_before, "after": launcher_after}},
        "reconciled_prior_position": global_position,
        "reconciled_campaign_attempt": int(schedule_row["campaign_attempt"]),
        "prior_invocation_id": receipt["invocation_id"],
        "effective_global_position": pending[0] if pending else None,
        "prior_manifest_path": str(manifest_path),
        "prior_manifest_sha256": manifest_sha,
        "classification": {"function": "train_launcher._post_training_failure_origin", "recorded_origin": "unknown",
                           "reconciled_origin": "candidate"},
        "evidence": evidence_record,
        "precedent": PRECEDENT,
        "reason": reason,
        "approval": approval,
        "scientific_components_unchanged": UNCHANGED,
        "scientific_contract_changed": False,
    }
    new_manifest = copy.deepcopy(manifest)
    new_manifest["immutable_deployment"][LAUNCHER] = launcher_after
    new_manifest.setdefault("operational_amendments", []).append(amendment)
    controller._manifest_hashes(new_manifest)  # no conflicting pins
    with open(new_manifest_path, "x") as handle:
        handle.write(json.dumps(new_manifest, indent=2, sort_keys=True) + "\n")
    new_manifest_path.chmod(0o444)
    new_sha = sha256_file(new_manifest_path)

    stamp = now.replace("-", "").replace(":", "")
    backup = state_root / f"controller_state.json.pre_reconcile_position_{global_position:03d}_{stamp}.json"
    with open(backup, "xb") as handle:
        handle.write(state_bytes)
    backup.chmod(0o444)

    worker_payload = {
        **evidence["worker"],
        "failure_origin": "candidate",
        "reconciliation_evidence": {
            "classification_basis": ("protected post-training check rejected the candidate; the message is listed "
                                     "in train_launcher._CANDIDATE_POST_TRAINING_FAILURE_MARKERS since this "
                                     "amendment"),
            "recorded_failure_origin": "unknown",
            "train_launcher_sha256": launcher_after,
            "remote_invocation_id": receipt["invocation_id"],
            "remote_receipt_status": "FAILED",
            "remote_exit_status": receipt.get("exit_status"),
            "remote_log_path": receipt.get("log_path"),
            "remote_log_sha256": evidence["hashes"]["invocation.log"],
            "worker_receipt_sha256": evidence["hashes"]["worker_failure_receipt.json"],
            "remote_process_alive": False,
            "remote_result_absent": True,
            "candidate_source_sha256": metadata["candidate_source_sha256"],
            "parent_source_sha256": metadata["parent_source_sha256"],
        },
    }
    remote_result = remote_runner._classify_worker_failure(worker_payload, {
        "campaign_id": schedule_row["campaign_id"], "contract_id": schedule_row["contract_id"],
        "attempt_number": int(schedule_row["campaign_attempt"]),
    })
    if remote_result.get("status") != "scientific_failure" or remote_result.get("error") != error:
        raise RuntimeError("the runner's classification did not yield the expected candidate failure")

    record = {**amendment, "manifest_path": str(new_manifest_path), "manifest_sha256": new_sha,
              "pre_reconciliation_state_copy": backup.name,
              "pre_reconciliation_state_sha256": hashlib.sha256(state_bytes).hexdigest()}
    state["manifest_sha256"] = new_sha
    state.setdefault("operational_amendments", []).append(record)
    position["reconciled_candidate_failure"] = {
        "reconciled_utc": now, "manifest_sha256": new_sha, "evidence": evidence_record,
        "reconciled_by": "scripts/reconcile_post_training_candidate_failure.py", "precedent": PRECEDENT,
    }
    row = ScheduledAttempt(global_position, str(schedule_row["contract_id"]), str(schedule_row["campaign_id"]),
                           int(schedule_row["campaign_attempt"]))
    paths = controller._campaign_paths(state_root, controller.CAMPAIGNS[row.contract_id])
    committed = controller._commit_pending_result(
        state=state, state_path=state_path, row=row, paths=paths, remote_result=remote_result)

    final = json.loads(state_path.read_text())["positions"][key]
    view = json.loads(paths["view"].read_text())
    if (final.get("status"), final.get("outcome")) != ("completed", "scientific_failure"):
        raise RuntimeError("the controller did not commit the failure")
    if not any(f.get("attempt") == row.campaign_attempt and f.get("error") == error for f in view["failures"]):
        raise RuntimeError("the campaign view does not carry the exact failure reason")
    return {"status": "reconciled", "global_position": global_position, "campaign_attempt": row.campaign_attempt,
            "manifest": str(new_manifest_path), "manifest_sha256": new_sha,
            "result_file": final.get("result_file"), "history_status": committed.get("status"),
            "view_attempt_number": view.get("attempt_number"), "controller_state_sha256": sha256_file(state_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-root", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--new-manifest-name", required=True)
    parser.add_argument("--global-position", type=int, required=True)
    parser.add_argument("--evidence-dir", type=pathlib.Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--approval", required=True)
    args = parser.parse_args()
    result = reconcile(
        state_root=args.state_root, manifest_path=args.manifest, new_manifest_name=args.new_manifest_name,
        global_position=args.global_position, evidence_dir=args.evidence_dir, reason=args.reason,
        approval=args.approval,
    )
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
