#!/usr/bin/env python3
"""PI-approved extension: extra fixed-time attempts after the 25-attempt campaign, until it has 25 valid results.

Approvals:
- 2026-11-05: "Yes, add one attempt" (attempt 26), because attempt 2 ended with CandidateExitedBeforeStop.
- 2026-11-06: after attempt 9 was charged as a candidate failure, "extend the campaign to target 25 valid
  experiments (currently 27 total attempts), while preserving the original 25-attempt results separately".

A valid result is a completed position whose outcome is not ``scientific_failure``. Extra attempts run one at a
time, as global positions 51, 52, ... (attempts 26, 27, ...), outside the frozen 50-position schedule, through the
controller's own position functions. The next one opens only after the previous one is complete and the arm still
has fewer than 25 valid results. A safety cap on the total number of attempts (MAX_TOTAL_ATTEMPTS, recorded in the
amendment) stops the extension for the PI's decision instead of spending more compute. The frozen study code,
schedule, program and contract are unchanged.

Steps (each refuses to run out of order):

  prepare --state-root DIR
      Checks that all 25 scheduled attempts are complete. Keeps a read-only tar snapshot of the 25-attempt state
      (the frozen 25-attempt result), writes a read-only amendment record and opens attempt 26. Opening an attempt
      keeps a read-only copy of the campaign view, sets the view's attempt_number to the new attempt (its only
      change) and adds the pending position with its Scientist note. Does nothing if 25 results are already valid.
  run <the same arguments as `campaign_controller run`>
      Runs the controller's launch gates in run_campaign's order before every extra attempt, runs or resumes the
      open extra attempt, and opens the next one while the arm has fewer than 25 valid results and the cap allows.
      Each Scientist prompt of an extra attempt ends with that attempt's note. Exit status 3 when the cap stops it.
  campaign <the same arguments as `campaign_controller run`>
      run_campaign for the remaining scheduled attempts, then prepare (once), then run.
  arm <the same arguments as `campaign_controller run`>
      After a payload-transfer drop and scripts/requeue_undelivered_payload.py: transfers, verifies and arms the
      open extra attempt, like continuation.arm_continuation_payload does for scheduled positions.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import tarfile
import tempfile
from typing import Any, Iterator

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study.contracts_2026 import campaign_controller as controller  # noqa: E402
from study.contracts_2026.bridge import A40RemoteConfig, A40Transport  # noqa: E402
from study.contracts_2026.scheduling import ScheduledAttempt  # noqa: E402
from study.contracts_2026.scientist_config import resolve_scientist_config  # noqa: E402

AMENDMENT_SCHEMA = "autopareto_pi_extension_attempts_v2"
FIRST_EXTENSION_POSITION = 51
FIRST_EXTENSION_ATTEMPT = 26
SCHEDULED_ATTEMPTS = 25
TARGET_VALID_RESULTS = 25
MAX_TOTAL_ATTEMPTS = 30
CAP_REACHED_EXIT_STATUS = 3
CONTRACT_ID = "T"
AMENDMENT_NAME = "extension_attempts_amendment.json"
APPROVALS = [
    {
        "approved_by": "PI",
        "approved_on": "2026-11-05",
        "question": "Add a 26th attempt at the end so the Opus arm has 25 valid results?",
        "answer": "Yes, add one attempt",
        "context": "audit/camera_ready_2026/20_THIRD_A40_POD.md section 9",
    },
    {
        "approved_by": "PI",
        "approved_on": "2026-11-06 (local; 2026-11-05 UTC)",
        "question": "How should attempt 9's cache-accounting failure be handled?",
        "answer": ("Charge attempt 9 as a candidate failure, preserve its logs and exact failure reason in the full "
                   "lineage so Opus can learn from it, and apply the tested error-classification fix. Resume from "
                   "attempt 10 without changing Opus's code manually. Also extend the campaign to target 25 valid "
                   "experiments (currently 27 total attempts), while preserving the original 25-attempt results "
                   "separately and documenting all failures and extra compute for fair reporting."),
        "context": "audit/camera_ready_2026/20_THIRD_A40_POD.md section 16",
    },
]


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


def _write_read_only(path: pathlib.Path, data: bytes) -> None:
    with open(path, "xb") as handle:  # never overwrite
        handle.write(data)
    path.chmod(0o444)


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _campaign_id() -> str:
    return str(controller.CAMPAIGNS[CONTRACT_ID]["campaign_id"])


def extension_position_for(attempt: int) -> int:
    return FIRST_EXTENSION_POSITION + attempt - FIRST_EXTENSION_ATTEMPT


def extension_row(attempt: int) -> ScheduledAttempt:
    return ScheduledAttempt(extension_position_for(attempt), CONTRACT_ID, _campaign_id(), attempt)


def view_copy_name(attempt: int) -> str:
    return f"campaign_view.json.pre_extension_attempt{attempt}.json"


def _attempts(state: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """(campaign attempt, position record) for the scheduled Contract-T attempts and every extra attempt."""
    rows = [(int(row["campaign_attempt"]), state["positions"][str(row["global_position"])])
            for row in state["schedule"] if row["contract_id"] == CONTRACT_ID]
    rows += [(int(record["extension"]["campaign_attempt"]), record)
             for record in state["positions"].values() if isinstance(record.get("extension"), dict)]
    return sorted(rows, key=lambda item: item[0])


def tally(state: dict[str, Any]) -> dict[str, Any]:
    completed = [(attempt, record) for attempt, record in _attempts(state) if record.get("status") == "completed"]
    failed = [attempt for attempt, record in completed if record.get("outcome") == "scientific_failure"]
    return {"completed_attempts": len(completed), "valid_results": len(completed) - len(failed),
            "failed_attempts": failed}


def _numbers(values: list[int]) -> str:
    words = [str(value) for value in values]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def scientist_note(attempt: int, counts: dict[str, Any]) -> str:
    failed = counts["failed_attempts"]
    failures = (f"Attempt {_numbers(failed)} ended as a scientific failure" if len(failed) == 1
                else f"Attempts {_numbers(failed)} ended as scientific failures")
    return (
        f"Orchestration note for attempt {attempt}: the PI approved extra attempts after attempt 25 so that this "
        f"campaign ends with {TARGET_VALID_RESULTS} valid results. {failures} and produced no measurement, so "
        f"{counts['valid_results']} of the {counts['completed_attempts']} attempts so far are valid. The "
        "orchestration layer owns attempt numbering and authorizes this extra attempt; everything else in "
        f"program.md applies unchanged. The campaign ends as soon as it has {TARGET_VALID_RESULTS} valid results."
    )


def _extension_positions(state: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    found = sorted((int(key), record) for key, record in state["positions"].items()
                   if isinstance(record, dict) and isinstance(record.get("extension"), dict))
    for index, (number, record) in enumerate(found):
        attempt = int(record["extension"]["campaign_attempt"])
        if number != FIRST_EXTENSION_POSITION + index or attempt != FIRST_EXTENSION_ATTEMPT + index:
            raise RuntimeError("extension positions are not contiguous from position 51 / attempt 26")
        amendment = pathlib.Path(record["extension"]["amendment_record"])
        if sha256_file(amendment) != record["extension"]["amendment_sha256"]:
            raise RuntimeError("the extension amendment record does not match its recorded hash")
        if index and found[index - 1][1].get("status") != "completed":
            raise RuntimeError("an extra attempt was opened before the previous one completed")
    return found


def _open_extension_attempt(*, state_root: pathlib.Path, state: dict[str, Any], attempt: int,
                            amendment_path: pathlib.Path) -> dict[str, Any]:
    """Add one pending extra attempt; the view's attempt_number is its only change."""
    state_path = state_root / "controller_state.json"
    key = str(extension_position_for(attempt))
    if key in state["positions"]:
        raise RuntimeError(f"extension position {key} already exists")
    campaign_root = state_root / _campaign_id()
    if (campaign_root / "pending" / f"position-{int(key):03d}").exists():
        raise RuntimeError("a pending directory for the extension position already exists")
    view_path = campaign_root / "campaign_view.json"
    view_bytes = view_path.read_bytes()
    view = json.loads(view_bytes)
    # update_campaign_view clamps the terminal view's attempt_number to 25 after every commit.
    if view.get("contract_id") != CONTRACT_ID or int(view.get("attempt_number", -1)) != SCHEDULED_ATTEMPTS:
        raise RuntimeError("campaign view is not the terminal view after the last completed attempt")
    counts = tally(state)
    if counts["completed_attempts"] != attempt - 1:
        raise RuntimeError("the completed attempts do not precede the new extra attempt")
    copy_path = campaign_root / view_copy_name(attempt)
    _write_read_only(copy_path, view_bytes)
    _write_json_atomic(view_path, {**view, "attempt_number": attempt})
    state["positions"][key] = {
        "status": "pending",
        "extension": {
            "amendment_record": str(amendment_path),
            "amendment_sha256": sha256_file(amendment_path),
            "campaign_attempt": attempt,
            "opened_utc": _utc(),
            "before": counts,
            "scientist_prompt_note": scientist_note(attempt, counts),
            "campaign_view": {"before_copy": str(copy_path), "before_sha256": hashlib.sha256(view_bytes).hexdigest(),
                              "after_sha256": sha256_file(view_path),
                              "only_change": {"attempt_number": [SCHEDULED_ATTEMPTS, attempt]}},
        },
    }
    _write_json_atomic(state_path, state)
    return state["positions"][key]


def prepare_extension(*, state_root: pathlib.Path, max_total_attempts: int = MAX_TOTAL_ATTEMPTS) -> dict[str, Any]:
    state_root = state_root.resolve(strict=True)
    state_path = state_root / "controller_state.json"
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    positions = state["positions"]
    if any(isinstance(record, dict) and "extension" in record for record in positions.values()):
        raise RuntimeError("the extension already exists; prepare runs once")
    if list(state.get("active_contracts", [])) != [CONTRACT_ID]:
        raise RuntimeError("the extension is defined for the single-arm Contract-T study only")
    scheduled = [row for row in state["schedule"] if row["contract_id"] == CONTRACT_ID]
    if len(scheduled) != SCHEDULED_ATTEMPTS:
        raise RuntimeError("unexpected number of scheduled Contract-T attempts")
    unfinished = [row["global_position"] for row in scheduled
                  if positions[str(row["global_position"])].get("status") != "completed"]
    if unfinished:
        raise RuntimeError(f"the 25 scheduled attempts are not all complete: positions {unfinished}")
    campaign_root = state_root / _campaign_id()
    history_path = campaign_root / "history.jsonl"
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    if sorted(int(record["campaign_attempt"]) for record in history) != list(range(1, SCHEDULED_ATTEMPTS + 1)):
        raise RuntimeError("history does not hold exactly one record for each of the 25 attempts")
    view_path = campaign_root / "campaign_view.json"
    if int(json.loads(view_path.read_text()).get("attempt_number", -1)) != SCHEDULED_ATTEMPTS:
        raise RuntimeError("campaign view is not the terminal view of the 25-attempt campaign")
    counts = tally(state)
    if counts["valid_results"] >= TARGET_VALID_RESULTS:
        return {"status": "not_needed", **counts}
    if not SCHEDULED_ATTEMPTS < max_total_attempts:
        raise ValueError("the cap must allow at least one extra attempt")

    now = _utc()
    snapshot = state_root.parent / f"{state_root.name}.snapshot_25_attempts_{now}.tar"
    with open(snapshot, "xb") as handle, tarfile.open(fileobj=handle, mode="w") as archive:
        archive.add(state_root, arcname=state_root.name)
    snapshot.chmod(0o444)
    amendment = {
        "schema": AMENDMENT_SCHEMA,
        "prepared_utc": now,
        "approvals": APPROVALS,
        "rule": (f"extra attempts run one at a time after attempt 25 until the arm has {TARGET_VALID_RESULTS} valid "
                 f"results; at most {max_total_attempts} attempts in total, then stop for the PI"),
        "target_valid_results": TARGET_VALID_RESULTS,
        "max_total_attempts": max_total_attempts,
        "frozen_25_attempt_result": {
            **counts,
            "snapshot": {"path": str(snapshot), "sha256": sha256_file(snapshot)},
            "campaign_view_sha256": sha256_file(view_path),
            "history_sha256": sha256_file(history_path),
            "controller_state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        },
        "extension": {"first_global_position": FIRST_EXTENSION_POSITION,
                      "first_campaign_attempt": FIRST_EXTENSION_ATTEMPT, "campaign_id": _campaign_id(),
                      "outside_frozen_schedule": True},
        "unchanged": {
            "manifest_sha256": state.get("manifest_sha256"),
            "schedule_sha256": state.get("schedule_sha256"),
            "scientist_config": state.get("scientist_config"),
            "frozen_study_code": "no file under study/ is changed; scripts/run_extension_attempt.py drives the "
                                 "controller's own position functions",
        },
        "driver": {"path": "scripts/run_extension_attempt.py", "sha256": sha256_file(pathlib.Path(__file__))},
    }
    amendment_path = state_root / "provenance" / AMENDMENT_NAME
    amendment_path.parent.mkdir(exist_ok=True)
    _write_read_only(amendment_path, (json.dumps(amendment, indent=2, sort_keys=True) + "\n").encode())
    opened = _open_extension_attempt(state_root=state_root, state=state, attempt=FIRST_EXTENSION_ATTEMPT,
                                     amendment_path=amendment_path)
    return {"status": "prepared", "amendment": str(amendment_path), "amendment_sha256": sha256_file(amendment_path),
            "snapshot": str(snapshot), "snapshot_sha256": amendment["frozen_25_attempt_result"]["snapshot"]["sha256"],
            "opened_attempt": FIRST_EXTENSION_ATTEMPT, "note": opened["extension"]["scientist_prompt_note"],
            "controller_state_sha256": sha256_file(state_path), **counts}


@contextlib.contextmanager
def extension_note(attempt: int, note: str) -> Iterator[None]:
    """Append the attempt's note to its Scientist prompts, for this extra attempt only, inside this process only."""
    original = controller._scientist_command

    def command(context, campaign_id, prompt_attempt, scientist_config, phase, *args, **kwargs):
        if prompt_attempt != attempt or scientist_config.agent_backend != "claude":
            raise RuntimeError(f"the extension note applies to the Claude backend's attempt {attempt} only")
        argv = list(original(context, campaign_id, prompt_attempt, scientist_config, phase, *args, **kwargs))
        prompt_index = argv.index("-p") + 1
        argv[prompt_index] = f"{argv[prompt_index]} {note}"
        return argv

    controller._scientist_command = command
    try:
        yield
    finally:
        controller._scientist_command = original


def _launch_gates(*, state_root, baseline_path, manifest_path, transport, scientist_config,
                  expected_manifest_sha256, active_contracts) -> tuple[dict[str, Any], pathlib.Path, str]:
    """run_campaign's gates, in its order, without iterating the frozen schedule."""
    if tuple(active_contracts) != (CONTRACT_ID,):
        raise RuntimeError("the extension runs the single-arm Contract-T study only")
    scientist_config.validate()
    manifest = controller._verify_final_manifest(manifest_path, scientist_config, expected_manifest_sha256)
    controller._verify_transport_config(manifest, transport.config)
    controller._verify_local_state_config(manifest, state_root)
    derivation = manifest.get("derivation") or {}
    if "active_contracts" in derivation and tuple(derivation["active_contracts"]) != tuple(active_contracts):
        raise RuntimeError(f"manifest freezes contracts {derivation['active_contracts']}, not {list(active_contracts)}")
    remote_run_namespace = str(derivation.get("remote_run_namespace", "pair-a"))
    manifest_sha256 = controller.sha256_file(manifest_path)
    expected_a40_uuid = controller._manifest_a40_uuid(manifest)
    transport.verify_trusted_files(controller._manifest_hashes(manifest))
    transport.verify_gpu_identity(expected_a40_uuid)
    absolute_files = controller._registered_cache_file_hashes(str(transport.config.registered_cache))
    absolute_files.update(controller._paired_baseline_checkpoint_hashes(manifest, active_contracts))
    transport.verify_absolute_files(absolute_files)
    state = controller._initialize_state(
        state_root, baseline_path, scientist_config, manifest_sha256,
        expected_a40_uuid=expected_a40_uuid, active_contracts=active_contracts,
    )
    paths = controller._campaign_paths(state_root, controller.CAMPAIGNS[CONTRACT_ID])
    baseline = controller._load_json(paths["view"])["baseline"]
    controller.protected_search_metrics({"inference_result": baseline.get("inference_result"), **baseline})
    return state, state_root / "controller_state.json", remote_run_namespace


def run_extension(*, state_root: pathlib.Path, baseline_path: pathlib.Path | None, manifest_path: pathlib.Path,
                  transport: Any, scientist_config: Any, expected_manifest_sha256: str | None = None,
                  active_contracts: tuple[str, ...] = (CONTRACT_ID,)) -> dict[str, Any]:
    attempts_run: list[dict[str, Any]] = []
    while True:
        state, state_path, namespace = _launch_gates(
            state_root=state_root, baseline_path=baseline_path, manifest_path=manifest_path, transport=transport,
            scientist_config=scientist_config, expected_manifest_sha256=expected_manifest_sha256,
            active_contracts=active_contracts,
        )
        extensions = _extension_positions(state)
        if not extensions:
            raise RuntimeError("the extension has not been prepared")
        number, position = extensions[-1]
        attempt = int(position["extension"]["campaign_attempt"])
        status = position.get("status")
        if status == "completed":
            counts = tally(state)
            summary = {"attempts_run": attempts_run, "last_attempt": attempt, **counts}
            if counts["valid_results"] >= TARGET_VALID_RESULTS:
                return {"status": "completed" if attempts_run else "already_completed", **summary}
            amendment_path = pathlib.Path(position["extension"]["amendment_record"])
            cap = int(json.loads(amendment_path.read_text())["max_total_attempts"])
            if counts["completed_attempts"] >= cap:
                return {"status": "cap_reached", "max_total_attempts": cap, **summary}
            _open_extension_attempt(state_root=state_root, state=state, attempt=attempt + 1,
                                    amendment_path=amendment_path)
            continue
        row = extension_row(attempt)
        with extension_note(attempt, position["extension"]["scientist_prompt_note"]):
            if status == "pending":
                record = controller._run_position(
                    state_root=state_root, state=state, state_path=state_path, row=row, manifest_path=manifest_path,
                    transport=transport, scientist_config=scientist_config, remote_run_namespace=namespace,
                )
            elif status == "running":
                record = controller._resume_running_position(
                    state_root=state_root, state=state, state_path=state_path, row=row, transport=transport,
                )
            else:
                raise RuntimeError(f"extension position {number} is paused: {status}")
        attempts_run.append({"global_position": number, "campaign_attempt": attempt, "outcome": record.get("status")})


def run_campaign_and_extension(*, state_root: pathlib.Path, baseline_path: pathlib.Path | None,
                               manifest_path: pathlib.Path, transport: Any, scientist_config: Any,
                               expected_manifest_sha256: str | None = None,
                               active_contracts: tuple[str, ...] = (CONTRACT_ID,)) -> dict[str, Any]:
    """The remaining scheduled attempts, then the extension (prepared once), in one process."""
    campaign = controller.run_campaign(
        state_root=state_root, baseline_path=baseline_path, manifest_path=manifest_path, transport=transport,
        scientist_config=scientist_config, expected_manifest_sha256=expected_manifest_sha256,
        active_contracts=active_contracts,
    )
    state = json.loads((state_root / "controller_state.json").read_text())
    if not any(isinstance(record, dict) and "extension" in record for record in state["positions"].values()):
        prepared = prepare_extension(state_root=state_root)
        if prepared["status"] == "not_needed":
            return {"status": "completed", "extension": prepared, "scheduled": campaign.get("status")}
    result = run_extension(
        state_root=state_root, baseline_path=baseline_path, manifest_path=manifest_path, transport=transport,
        scientist_config=scientist_config, expected_manifest_sha256=expected_manifest_sha256,
        active_contracts=active_contracts,
    )
    return {**result, "scheduled": campaign.get("status")}


def arm_extension_payload(*, state_root: pathlib.Path, transport: Any) -> dict[str, Any]:
    """Transfer, verify and arm a requeued extension payload (continuation.arm_continuation_payload's steps)."""
    state_root = state_root.resolve(strict=True)
    state_path = state_root / "controller_state.json"
    state = json.loads(state_path.read_text())
    extensions = _extension_positions(state)
    if not extensions:
        raise RuntimeError("the extension has not been prepared")
    number, position = extensions[-1]
    if position.get("status") != "running" or not isinstance(position.get("execution"), dict):
        raise RuntimeError("the open extension position is not a prepared running position")
    execution = position["execution"]
    pending = state_root / _campaign_id() / "pending" / f"position-{number:03d}"
    if execution.get("dispatch_status") == "launch_requested":
        observed = transport.verify_payload(remote_dir=str(execution["remote_payload_dir"]),
                                            candidate=pending / "train.py", metadata=pending / "registered_metadata.json")
        return {"status": "already_armed", "execution": execution, "remote_hashes": observed}
    if execution.get("dispatch_status") != "prepared":
        raise RuntimeError("extension payload has an invalid dispatch state")
    remote_dir = transport.transfer(payload=pending, remote_run_id=str(execution["remote_run_id"]))
    observed = transport.verify_payload(remote_dir=remote_dir, candidate=pending / "train.py",
                                        metadata=pending / "registered_metadata.json")
    execution["remote_payload_dir"] = remote_dir
    execution["dispatch_status"] = "launch_requested"
    _write_json_atomic(state_path, state)
    return {"status": "armed", "global_position": number, "execution": execution, "remote_hashes": observed}


def _controller_run_arguments(argv: list[str]) -> tuple[argparse.Namespace, Any, A40Transport]:
    """Parse exactly the arguments of `campaign_controller run` and build its config and transport."""
    args = controller.build_parser().parse_args(["run", *argv])
    scientist_config = resolve_scientist_config(
        model=args.model, reasoning=args.reasoning, scientist_bin=args.scientist_bin,
        timeout_seconds=args.scientist_timeout_seconds, agent_backend=args.agent_backend,
    )
    config = A40RemoteConfig(
        host=args.a40_host, port=args.a40_port, identity_file=args.a40_identity, repository=args.a40_repository,
        python_path=args.a40_python, study_venv=args.a40_study_venv, registered_cache=args.a40_cache,
        trusted_state_root=args.a40_state_root,
        paired_baseline_t_checkpoint=args.a40_paired_baseline_t_checkpoint,
        paired_baseline_n_checkpoint=args.a40_paired_baseline_n_checkpoint,
    )
    return args, scientist_config, A40Transport(config)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"prepare", "run", "campaign", "arm"}:
        print(__doc__, file=sys.stderr)
        return 2
    step, rest = argv[0], argv[1:]
    if step == "prepare":
        parser = argparse.ArgumentParser(prog="run_extension_attempt.py prepare")
        parser.add_argument("--state-root", type=pathlib.Path, required=True)
        result = prepare_extension(state_root=parser.parse_args(rest).state_root)
    elif step == "arm":
        args, _config, transport = _controller_run_arguments(rest)
        result = arm_extension_payload(state_root=args.state_root, transport=transport)
    else:
        args, scientist_config, transport = _controller_run_arguments(rest)
        run = run_extension if step == "run" else run_campaign_and_extension
        result = run(
            state_root=args.state_root.resolve(), baseline_path=args.baseline_record.resolve(),
            manifest_path=args.manifest.resolve(), transport=transport, scientist_config=scientist_config,
            expected_manifest_sha256=args.manifest_sha256, active_contracts=tuple(args.contracts.split(",")),
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return CAP_REACHED_EXIT_STATUS if result.get("status") == "cap_reached" else 0


if __name__ == "__main__":
    sys.exit(main())
