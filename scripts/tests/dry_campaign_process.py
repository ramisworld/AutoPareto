"""One controller process for the supervisor tests: the real campaign loop, offline.

It runs ``campaign_controller.run_campaign`` for the single-arm Contract-T study with the
offline fake Claude CLI, like ``tools/dry_run_claude_campaign.py``, but the mock A40 keeps its
durable receipts on disk. A second process can therefore resume what an earlier, killed process
left behind, which is what a real resume does. No model call, GPU or network.

Knobs, all files under ``--a40-dir`` (the controller strips the Scientist's environment):
* ``slow_attempts.json``: {"<attempt>": seconds}, a pause inside the remote wait for that attempt;
* ``scientist_sleep``: seconds the fake Claude CLI sleeps before answering;
* ``scientist_hang``: two lines, a prompt substring and seconds; matching Scientist calls sleep that long;
* ``scientist_timeout``: the Scientist timeout in seconds (default: the dry-run config's);
* ``fail_transfers.json``: {"<attempt>": count}, payload transfers for that attempt that fail before
  anything reaches the mock A40 (each failure is logged in ``transfer_failures.jsonl``);
* ``failed_receipts.json``: [attempts], whose invocation ends with a FAILED durable receipt of unknown
  failure origin, as the real runner writes after an unlisted post-training failure;
* ``fail_attempts.json``: [attempts] that end as candidate scientific failures (default [3]).

Evidence written under ``--a40-dir``: ``launch_requests.jsonl`` (every launch request) and
``trainings.jsonl`` (one line per invocation that actually started "training").
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from study.contracts_2026 import campaign_controller as controller  # noqa: E402
from study.contracts_2026.bridge import A40TransportCommandError  # noqa: E402
from study.contracts_2026.tools import dry_run_claude_campaign as dry  # noqa: E402


def _append(path: pathlib.Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class FileBackedA40(dry.DryRunTransport):
    """Durable receipts on disk; launch is idempotent by invocation id, like the real runner."""

    def __init__(self, a40_dir: pathlib.Path) -> None:
        knob = a40_dir / "fail_attempts.json"
        super().__init__(fail_attempts=tuple(json.loads(knob.read_text())) if knob.exists() else (3,))
        self.dir = a40_dir
        (self.dir / "invocations").mkdir(parents=True, exist_ok=True)
        (self.dir / "payloads").mkdir(parents=True, exist_ok=True)

    def transfer(self, *, payload: pathlib.Path, remote_run_id: str) -> str:
        remote_dir = f"dry-run://{remote_run_id}"
        meta = json.loads((payload / "registered_metadata.json").read_text())
        knob = self.dir / "fail_transfers.json"
        failures = json.loads(knob.read_text()) if knob.exists() else {}
        attempt = str(meta.get("attempt_number"))
        if int(failures.get(attempt, 0)) > 0:
            # Like the RunPod proxy dropping the PTY mid-transfer: nothing reaches the A40.
            failures[attempt] = int(failures[attempt]) - 1
            knob.write_text(json.dumps(failures))
            _append(self.dir / "transfer_failures.jsonl", {"attempt": int(attempt), "remote_run_id": remote_run_id})
            raise A40TransportCommandError(
                "A40 PTY/SSH command failed before a valid frame: returncode=1; "
                "detail=script: write master: Input/output error\n"
            )
        meta["_source_sha256"] = hashlib.sha256((payload / "train.py").read_bytes()).hexdigest()
        (self.dir / "payloads" / f"{remote_run_id}.json").write_text(json.dumps(meta))
        return remote_dir

    def verify_payload(self, *, remote_dir: str, candidate: pathlib.Path, metadata: pathlib.Path) -> dict[str, Any]:
        stored = self._payload(remote_dir)
        observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if stored["_source_sha256"] != observed:
            raise RuntimeError("dry-run payload differs from the local candidate")
        return {"train.py": {"sha256": observed}}

    def _payload(self, remote_dir: str) -> dict[str, Any]:
        return json.loads((self.dir / "payloads" / f"{remote_dir.removeprefix('dry-run://')}.json").read_text())

    def launch(self, *, remote_dir: str, output_name: str, invocation_id: str, invocation_metadata) -> None:
        record_path = self.dir / "invocations" / f"{invocation_id}.json"
        new = not record_path.exists()
        _append(self.dir / "launch_requests.jsonl", {"invocation_id": invocation_id, "new": new,
                                                     "attempt": invocation_metadata["campaign_attempt"]})
        if new:
            record_path.write_text(json.dumps({"remote_dir": remote_dir, "metadata": dict(invocation_metadata)}))
            _append(self.dir / "trainings.jsonl", {"invocation_id": invocation_id,
                                                   "attempt": invocation_metadata["campaign_attempt"]})

    def _invocation(self, invocation_id: str) -> dict[str, Any]:
        return json.loads((self.dir / "invocations" / f"{invocation_id}.json").read_text())

    def wait_for_invocation(self, invocation_id: str) -> dict[str, Any]:
        attempt = int(self._invocation(invocation_id)["metadata"]["campaign_attempt"])
        slow = json.loads((self.dir / "slow_attempts.json").read_text()) if (self.dir / "slow_attempts.json").exists() else {}
        if str(attempt) in slow:
            (self.dir / f"waiting-{attempt}").write_text(str(os.getpid()))
            time.sleep(float(slow[str(attempt)]))
        failed = self.dir / "failed_receipts.json"
        if failed.exists() and attempt in json.loads(failed.read_text()):
            invocation = self._invocation(invocation_id)
            return {"record": {
                "schema": "autopareto_durable_invocation_v1", "status": "FAILED", "invocation_id": invocation_id,
                "exit_status": 1, "metadata": invocation["metadata"],
                "log_path": f"dry-run://invocations/{invocation_id}.log",
                "failure": {"type": "RuntimeError",
                            "message": "candidate worker failure origin is unknown; manual reconciliation is required"},
            }, "process_alive": False}
        return {"record": {"status": "COMPLETE", "invocation_id": invocation_id}, "process_alive": False}

    def fetch_durable_result(self, record: dict[str, Any]) -> dict[str, Any]:
        invocation = self._invocation(record["invocation_id"])
        self.invocations[record["invocation_id"]] = invocation
        self.payloads[invocation["remote_dir"]] = self._payload(invocation["remote_dir"])
        return super().fetch_durable_result(record)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=pathlib.Path, required=True)
    parser.add_argument("--a40-dir", type=pathlib.Path, required=True)
    parser.add_argument("--extension", action="store_true",
                        help="run scripts/run_extension_attempt.run_extension instead of run_campaign")
    parser.add_argument("--campaign-and-extension", action="store_true",
                        help="run scripts/run_extension_attempt.run_campaign_and_extension instead of run_campaign")
    parser.add_argument("--manifest", type=pathlib.Path, default=None,
                        help="manifest file to run under (default: <state-root>.manifest.json)")
    args = parser.parse_args()
    state_root = args.state_root.resolve()
    a40_dir = args.a40_dir.resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    baseline_path = state_root.parent / f"{state_root.name}.baseline.json"
    manifest_path = state_root.parent / f"{state_root.name}.manifest.json"
    if not baseline_path.exists():
        baseline_path.write_text(json.dumps(dry.dry_run_baseline()))
        manifest_path.write_text(json.dumps(dry.DRY_RUN_MANIFEST))
    # The fake CLI answers instantly; a wrapper lets a test hold a Scientist call open.
    wrapper_dir = a40_dir / "bin"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper = wrapper_dir / "claude"
    if not wrapper.exists():
        # rate_limit_utilization: a seat-usage reading injected after the init event, as the real CLI reports it.
        rate_event = ('{"type":"rate_limit_event","rate_limit_info":{"status":"allowed","isUsingOverage":false,'
                      '"unifiedWindows":{"five_hour":{"utilization":%s,"resetsAt":1793802600},'
                      '"seven_day":{"utilization":0.3,"resetsAt":1793937600}}}}')
        wrapper.write_text(
            "#!/bin/bash\n"
            "set -o pipefail\n"
            f"if [ -f {a40_dir}/scientist_sleep ]; then sleep \"$(cat {a40_dir}/scientist_sleep)\"; fi\n"
            f"if [ -f {a40_dir}/scientist_hang ] && printf '%s' \"$*\" | grep -qF \"$(head -1 {a40_dir}/scientist_hang)\"; then\n"
            f"  sleep \"$(tail -1 {a40_dir}/scientist_hang)\"\n"
            "fi\n"
            f"if [ -f {a40_dir}/rate_limit_utilization ] && [ \"$1\" = -p ]; then\n"
            f"  {dry.FAKE_CLAUDE} \"$@\" | awk -v u=\"$(cat {a40_dir}/rate_limit_utilization)\" "
            f"'NR==1{{print; printf \"{rate_event.replace(chr(34), chr(92) + chr(34))}\\n\", u; next}} {{print}}'\n"
            "  exit $?\n"
            "fi\n"
            f"exec {dry.FAKE_CLAUDE} \"$@\"\n"
        )
        wrapper.chmod(0o755)
    config = dataclasses.replace(dry.dry_run_config(), scientist_bin=str(wrapper))
    if (a40_dir / "scientist_timeout").exists():
        config = dataclasses.replace(config, timeout_seconds=float((a40_dir / "scientist_timeout").read_text()))
    patches = [
        mock.patch.object(controller, "_verify_final_manifest", lambda *_a, **_k: dry.DRY_RUN_MANIFEST),
        mock.patch.object(controller, "_verify_transport_config", lambda *_a, **_k: None),
        mock.patch.object(controller, "_verify_local_state_config", lambda *_a, **_k: None),
        mock.patch.object(controller, "_manifest_a40_uuid", lambda *_a, **_k: "GPU-DRY-RUN"),
        mock.patch.object(controller, "_manifest_hashes", lambda *_a, **_k: {}),
        mock.patch.object(controller, "_registered_cache_file_hashes", lambda *_a, **_k: {}),
    ]
    for patcher in patches:
        patcher.start()
    run = controller.run_campaign
    if args.extension or args.campaign_and_extension:
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_extension_attempt  # noqa: PLC0415

        run = (run_extension_attempt.run_extension if args.extension
               else run_extension_attempt.run_campaign_and_extension)
    if args.manifest is not None:
        manifest_path = args.manifest.resolve()
    result = run(
        state_root=state_root, baseline_path=baseline_path, manifest_path=manifest_path,
        transport=FileBackedA40(a40_dir), scientist_config=config, active_contracts=("T",),
    )
    print(json.dumps(result, sort_keys=True))
    return 3 if result.get("status") == "cap_reached" else 0


if __name__ == "__main__":
    raise SystemExit(main())
