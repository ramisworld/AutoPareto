#!/usr/bin/env python3
"""Failure and compute ledger of a single-arm campaign across its pod moves and continuations (read-only).

For fair reporting it lists every attempt, every Scientist call and every A40 invocation the arm made, whether or
not it counts, from the durable records:

- attempts: the live controller state (scheduled Contract-T positions and extension positions), with each result's
  validation BPB, decode speed and memory, or the exact failure reason;
- Scientist calls: every ``scientist_diagnostics/*/invocation.json`` in the given state roots, de-duplicated by
  invocation id (continuations copy earlier diagnostics forward). A call is "used" if it is the latest completed
  call of its phase for its attempt; any other call (timed out, or superseded by a requeue) is "extra";
- A40 invocations: durable receipts (``invocations/*.json`` in the given receipt folders, plus the FAILED receipts
  recorded in each state's ``reconciliation_failure``), keyed by remote state root and invocation id. An invocation
  is "counted" if it produced the attempt's committed outcome in the live state, and "extra" otherwise.

    python3 -B scripts/campaign_compute_ledger.py --state-root OLDEST ... --state-root LIVE \
        --receipt-dir DIR ... --output-json F.json --output-md F.md

It writes only the two output files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any

SCHEDULED_ATTEMPTS = 25
TARGET_VALID_RESULTS = 25


def _load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def _epoch(value: str | None) -> float | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _utc(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _final_line(error: str) -> str:
    lines = [line.strip() for line in str(error).strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def attempts(state_root: pathlib.Path) -> list[dict[str, Any]]:
    state = _load(state_root / "controller_state.json")
    rows = [(int(r["global_position"]), int(r["campaign_attempt"]), False)
            for r in state["schedule"] if r["contract_id"] == "T"]
    rows += [(int(key), int(p["extension"]["campaign_attempt"]), True)
             for key, p in state["positions"].items() if isinstance(p.get("extension"), dict)]
    out = []
    for position, attempt, extension in sorted(rows, key=lambda row: row[1]):
        record = state["positions"][str(position)]
        entry: dict[str, Any] = {
            "campaign_attempt": attempt, "global_position": position, "extension": extension,
            "status": record.get("status"), "outcome": None, "valid": None,
            "execution_invocation_id": (record.get("execution") or {}).get("invocation_id"),
            "requeued_scientist_timeouts": len(record.get("requeued_scientist_timeouts", [])),
            "requeued_transport_failures": len(record.get("requeued_transport_failures", [])),
            "reconciled_candidate_failure": "reconciled_candidate_failure" in record,
        }
        if record.get("status") == "completed":
            result = _load(pathlib.Path(record["result_file"]))
            failed = record.get("outcome") == "scientific_failure"
            entry.update({
                "outcome": "scientific_failure" if failed else "completed", "valid": not failed,
                "parent": result.get("parent_experiment_id"), "change_category": result.get("change_category"),
                "pareto_decision": (result.get("pareto_decision") or {}).get("decision"),
            })
            if failed:
                entry["failure_reason"] = _final_line(result.get("error", ""))
            else:
                inference = result.get("inference_result") or {}
                runtime = result.get("runtime") or {}
                entry.update({
                    "validation_bpb": result.get("protected_validation_bpb"),
                    "optimizer_updates": result.get("optimizer_updates"),
                    "normalized_decode_throughput": inference.get("normalized_decode_throughput"),
                    "peak_inference_memory_mb": (inference.get("peak_allocated_inference_memory_bytes") or 0) / 1e6,
                    "a40_candidate_evaluation_seconds": runtime.get("total_candidate_evaluation_seconds"),
                })
        out.append(entry)
    return out


def scientist_calls(state_roots: list[pathlib.Path]) -> list[dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    for root in state_roots:
        for path in sorted(root.glob("*/scientist_diagnostics/*/invocation.json")):
            record = _load(path)
            key = record["invocation_id"]
            if key in calls:
                continue
            metadata = record.get("metadata") or {}
            started, completed = record.get("started_at_epoch"), record.get("completed_at_epoch")
            calls[key] = {
                "invocation_id": key, "first_seen_in": root.name,
                "campaign_attempt": int(metadata.get("attempt_number", -1)), "phase": metadata.get("phase"),
                "status": record.get("status"), "timeout_seconds": record.get("timeout_seconds"),
                "started_utc": _utc(started), "seconds": (completed - started) if started and completed else None,
                "failure": _final_line(record.get("failure", "")) or None,
            }
    ordered = sorted(calls.values(), key=lambda call: call["started_utc"] or "")
    latest: dict[tuple[int, str], str] = {}
    for call in ordered:
        if call["status"] == "COMPLETE":
            latest[(call["campaign_attempt"], call["phase"])] = call["invocation_id"]
    for call in ordered:
        if call["status"] == "RUNNING":
            call["use"] = "in_progress"
        else:
            used = latest.get((call["campaign_attempt"], call["phase"])) == call["invocation_id"]
            call["use"] = "used" if used else "extra"
    return ordered


def a40_invocations(state_roots: list[pathlib.Path], receipt_dirs: list[pathlib.Path],
                    attempt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    receipts: dict[tuple[str, str], dict[str, Any]] = {}

    def add(receipt: dict[str, Any], source: str) -> None:
        log_path = str(receipt.get("log_path", ""))
        remote_root = str(pathlib.PurePosixPath(log_path).parent.parent) if log_path else "unknown"
        key = (remote_root, str(receipt.get("invocation_id")))
        if key not in receipts:
            receipts[key] = {"receipt": receipt, "source": source, "remote_state_root": remote_root}

    for folder in receipt_dirs:
        for path in sorted(folder.glob("**/invocations/pair-a-*.json")):
            add(_load(path), str(path))
    for root in state_roots:
        state = _load(root / "controller_state.json")
        for position in state["positions"].values():
            receipt = (position.get("reconciliation_failure") or {}).get("receipt")
            if isinstance(receipt, dict) and receipt.get("invocation_id"):
                add(receipt, f"{root.name}/controller_state.json")

    committed = {row["campaign_attempt"]: row for row in attempt_rows if row["status"] == "completed"}
    out = []
    for (remote_root, invocation_id), item in sorted(receipts.items(), key=lambda kv: kv[1]["receipt"].get("created_at", "")):
        receipt = item["receipt"]
        metadata = receipt.get("metadata") or {}
        attempt = int(metadata.get("campaign_attempt", -1))
        created, completed = _epoch(receipt.get("created_at")), _epoch(receipt.get("completed_at"))
        row = committed.get(attempt)
        counted = bool(row and row["execution_invocation_id"] == invocation_id and receipt.get("status") in
                       {"COMPLETE", "FAILED"} and (receipt.get("status") == "COMPLETE" or row.get("outcome") == "scientific_failure"))
        if counted and receipt.get("status") == "FAILED" and not row.get("reconciled_candidate_failure"):
            counted = False
        out.append({
            "invocation_id": invocation_id, "remote_state_root": remote_root, "campaign_attempt": attempt,
            "global_position": metadata.get("global_position"), "status": receipt.get("status"),
            "created_utc": receipt.get("created_at"), "completed_utc": receipt.get("completed_at"),
            "wall_seconds": (completed - created) if created and completed else None,
            "failure": (receipt.get("failure") or {}).get("message"),
            "use": "counted" if counted else "extra", "receipt_source": item["source"],
        })
    return out


def summarize(rows, calls, invocations) -> dict[str, Any]:
    def minutes(values):
        return round(sum(v for v in values if v) / 60.0, 1)

    completed = [r for r in rows if r["status"] == "completed"]
    scheduled = [r for r in completed if not r["extension"]]
    return {
        "attempts_completed": len(completed),
        "valid_results": sum(1 for r in completed if r["valid"]),
        "failed_attempts": [r["campaign_attempt"] for r in completed if r["outcome"] == "scientific_failure"],
        "scheduled_attempts": {
            "frozen_25_attempt_result_final": len(scheduled) == SCHEDULED_ATTEMPTS,
            "completed": len(scheduled), "valid": sum(1 for r in scheduled if r["valid"]),
            "failed_attempts": [r["campaign_attempt"] for r in scheduled if r["outcome"] == "scientific_failure"],
        },
        "extension_attempts": [r["campaign_attempt"] for r in rows if r["extension"]],
        "scientist_calls": {
            "total": len(calls), "used": sum(c["use"] == "used" for c in calls),
            "extra": sum(c["use"] == "extra" for c in calls),
            "in_progress": sum(c["use"] == "in_progress" for c in calls),
            "timed_out": sum(1 for c in calls if c["status"] == "FAILED"
                             and any(word in str(c.get("failure") or "").lower() for word in ("timeout", "timed out"))),
            "minutes_total": minutes(c["seconds"] for c in calls),
            "minutes_extra": minutes(c["seconds"] for c in calls if c["use"] == "extra"),
            "minutes_extension": minutes(c["seconds"] for c in calls if c["campaign_attempt"] > SCHEDULED_ATTEMPTS),
        },
        "a40_invocations": {
            "total_with_receipt": len(invocations), "counted": sum(i["use"] == "counted" for i in invocations),
            "extra": sum(i["use"] == "extra" for i in invocations),
            "wall_minutes_with_receipt": minutes(i["wall_seconds"] for i in invocations),
            "wall_minutes_extra": minutes(i["wall_seconds"] for i in invocations if i["use"] == "extra"),
            "wall_minutes_extension": minutes(i["wall_seconds"] for i in invocations if i["campaign_attempt"] > SCHEDULED_ATTEMPTS),
            "candidate_evaluation_minutes_valid_results": minutes(r.get("a40_candidate_evaluation_seconds") for r in completed),
        },
    }


def markdown(ledger: dict[str, Any]) -> str:
    s = ledger["summary"]
    lines = [
        "# Opus 5.5 Contract-T arm: failure and compute ledger",
        "",
        f"Generated {ledger['generated_utc']} by `scripts/campaign_compute_ledger.py` from durable records (read-only).",
        "",
        f"- Attempts completed: **{s['attempts_completed']}**; valid results: **{s['valid_results']}**; "
        f"scientific failures: attempts {s['failed_attempts'] or 'none'}.",
        f"- {'Frozen 25-attempt result' if s['scheduled_attempts']['frozen_25_attempt_result_final'] else 'Scheduled attempts so far'}: "
        f"{s['scheduled_attempts']['valid']} valid of {s['scheduled_attempts']['completed']} completed scheduled "
        f"attempts; failed {s['scheduled_attempts']['failed_attempts'] or 'none'}. Extension attempts: "
        f"{s['extension_attempts'] or 'none yet'}.",
        f"- {s['scientist_calls']['timed_out']} Scientist call(s) timed out. A40 receipts from the first pod "
        "(attempt 1's run) were lost with that pod; attempt 1's evaluation time comes from its result file.",
        f"- Scientist calls: {s['scientist_calls']['total']} ({s['scientist_calls']['used']} used, "
        f"{s['scientist_calls']['extra']} extra, {s['scientist_calls']['in_progress']} in progress); "
        f"{s['scientist_calls']['minutes_total']} min in total, {s['scientist_calls']['minutes_extra']} min extra, "
        f"{s['scientist_calls']['minutes_extension']} min in extension attempts.",
        f"- A40 invocations with a durable receipt: {s['a40_invocations']['total_with_receipt']} "
        f"({s['a40_invocations']['counted']} counted, {s['a40_invocations']['extra']} extra); "
        f"{s['a40_invocations']['wall_minutes_with_receipt']} wall-min, {s['a40_invocations']['wall_minutes_extra']} "
        f"wall-min extra, {s['a40_invocations']['wall_minutes_extension']} wall-min in extension attempts.",
        "",
        "## Attempts",
        "",
        "| Attempt | Pos | Ext | Outcome | BPB | Decode × | Mem MB | Parent | Category | Sci calls used/extra | Note |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    per_attempt: dict[int, list[dict[str, Any]]] = {}
    for call in ledger["scientist_calls"]:
        per_attempt.setdefault(call["campaign_attempt"], []).append(call)
    for r in ledger["attempts"]:
        calls = per_attempt.get(r["campaign_attempt"], [])
        used = sum(c["use"] == "used" for c in calls)
        extra = sum(c["use"] == "extra" for c in calls)
        bpb = f"{r['validation_bpb']:.5f}" if r.get("validation_bpb") else ""
        dec = f"{r['normalized_decode_throughput']:.3f}" if r.get("normalized_decode_throughput") else ""
        mem = f"{r['peak_inference_memory_mb']:.1f}" if r.get("peak_inference_memory_mb") else ""
        note = r.get("failure_reason", "") or (r.get("pareto_decision") or "")
        if r["requeued_scientist_timeouts"]:
            note += f"; {r['requeued_scientist_timeouts']} Scientist timeout requeue(s)"
        if r["requeued_transport_failures"]:
            note += f"; {r['requeued_transport_failures']} transfer-drop requeue(s)"
        parent = (r.get("parent") or "").replace("pair-a-fixed-time-seed3-", "")
        lines.append(f"| {r['campaign_attempt']} | {r['global_position']} | {'yes' if r['extension'] else ''} | "
                     f"{r['outcome'] or r['status']} | {bpb} | {dec} | {mem} | {parent} | {r.get('change_category') or ''} | "
                     f"{used}/{extra} | {note.replace('|', '/')} |")
    lines += ["", "## A40 invocations", "",
              "| Attempt | Invocation | Remote state root | Status | Wall min | Use | Failure |", "|---|---|---|---|---|---|---|"]
    for i in ledger["a40_invocations"]:
        wall = f"{i['wall_seconds'] / 60:.1f}" if i["wall_seconds"] else ""
        lines.append(f"| {i['campaign_attempt']} | `{i['invocation_id'][:20]}…` | `{pathlib.PurePosixPath(i['remote_state_root']).name}` | "
                     f"{i['status']} | {wall} | {i['use']} | {(i['failure'] or '').replace('|', '/')} |")
    lines += ["", "## Extra Scientist calls", "", "| Attempt | Phase | Status | Min | Started | Reason |", "|---|---|---|---|---|---|"]
    for c in ledger["scientist_calls"]:
        if c["use"] != "extra":
            continue
        minutes = f"{c['seconds'] / 60:.1f}" if c["seconds"] else ""
        reason = c["failure"] or "completed, superseded by a later call after a requeue"
        lines.append(f"| {c['campaign_attempt']} | {c['phase']} | {c['status']} | {minutes} | {c['started_utc']} | {reason.replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-root", type=pathlib.Path, action="append", required=True,
                        help="state roots, oldest first; the last one is the live state")
    parser.add_argument("--receipt-dir", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-md", type=pathlib.Path, required=True)
    args = parser.parse_args()
    roots = [root.resolve(strict=True) for root in args.state_root]
    rows = attempts(roots[-1])
    calls = scientist_calls(roots)
    invocations = a40_invocations(roots, [d.resolve(strict=True) for d in args.receipt_dir], rows)
    ledger = {
        "schema": "autopareto_campaign_compute_ledger_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "state_roots": [str(root) for root in roots],
        "receipt_dirs": [str(d) for d in args.receipt_dir],
        "summary": summarize(rows, calls, invocations),
        "attempts": rows, "scientist_calls": calls, "a40_invocations": invocations,
    }
    args.output_json.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n")
    args.output_md.write_text(markdown(ledger))
    print(json.dumps(ledger["summary"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
