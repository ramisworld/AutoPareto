"""Non-study proof of the final A40 permission/seccomp candidate boundary."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile

from study.contracts_2026.candidate_sandbox import wrap_worker_command


TRUSTED_ROOT = pathlib.Path(os.environ.get("AUTOPARETO_TRUSTED_ROOT", "/opt/autopareto_trusted"))
CONTEXT_ROOT = pathlib.Path(
    os.environ.get("AUTOPARETO_CANDIDATE_CONTEXT_ROOT", "/opt/autopareto_candidate_contexts")
)
ATTEMPT_ROOT = pathlib.Path(
    os.environ.get("AUTOPARETO_CANDIDATE_ATTEMPT_ROOT", "/opt/autopareto-candidate-attempts")
)


def _candidate_probe() -> str:
    return r'''
import json, os, pathlib, socket, subprocess, sys

protected = [
    "/opt/autopareto_trusted/study/contracts_2026/protocol.md",
    "/opt/autopareto_trusted/state/evidence_snapshot_before_workspace_clean.json",
    "/opt/autopareto_trusted/recovery_before_clean/autopareto",
    "/root/.ssh/authorized_keys",
]

def read_status(path):
    try:
        target = pathlib.Path(path)
        if target.is_dir():
            list(target.iterdir())
        else:
            target.read_bytes()
    except (PermissionError, FileNotFoundError):
        return "denied"
    except OSError:
        return "readable"
    return "readable"

def write_status(path):
    try:
        pathlib.Path(path).write_text("forbidden")
    except (PermissionError, FileNotFoundError, OSError):
        return "denied"
    return "writable"

def network_status():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(("1.1.1.1", 80))
        sock.close()
    except OSError:
        return "denied"
    return "reachable"

child = subprocess.run([
    sys.executable, "-c",
    "import os,socket; print(os.geteuid()); "
    "\ntry:\n s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); s.connect(('1.1.1.1',80)); print('network-reachable')\nexcept OSError: print('network-denied')",
], capture_output=True, text=True, check=False)

result = {
    "euid": os.geteuid(),
    "egid": os.getegid(),
    "no_new_privs": next((line.split(":", 1)[1].strip() for line in pathlib.Path("/proc/self/status").read_text().splitlines() if line.startswith("NoNewPrivs:")), "missing"),
    "cap_eff": next((line.split(":", 1)[1].strip() for line in pathlib.Path("/proc/self/status").read_text().splitlines() if line.startswith("CapEff:")), "missing"),
    "protected_reads": {path: read_status(path) for path in protected},
    "protected_write": write_status("/opt/autopareto_trusted/state/cleanroom-probe-forbidden"),
    "network": network_status(),
    "child_returncode": child.returncode,
    "child_stdout": child.stdout.strip(),
    "legitimate_write": write_status(str(pathlib.Path(os.environ["TMPDIR"]) / "cleanroom-probe-allowed")),
}
pathlib.Path(os.environ["TMPDIR"], "cleanroom-probe-result.json").write_text(json.dumps(result, sort_keys=True))
if result["euid"] != 999 or result["egid"] == 0 or result["no_new_privs"] != "1" or result["cap_eff"] != "0000000000000000":
    raise SystemExit("candidate privilege policy failed")
if any(value != "denied" for value in result["protected_reads"].values()) or result["protected_write"] != "denied":
    raise SystemExit("protected filesystem access was not denied")
if result["network"] != "denied" or "network-denied" not in result["child_stdout"]:
    raise SystemExit("network restriction failed")
if result["legitimate_write"] != "writable":
    raise SystemExit("candidate output write failed")
'''


def main() -> int:
    CONTEXT_ROOT.mkdir(mode=0o755, parents=True, exist_ok=True)
    context = pathlib.Path(tempfile.mkdtemp(prefix="cleanroom-probe-", dir=CONTEXT_ROOT))
    context.chmod(0o555)
    script = context / "probe.py"
    script.write_text(_candidate_probe())
    script.chmod(0o444)
    attempt = ATTEMPT_ROOT / "cleanroom-probe"
    attempt.mkdir(mode=0o700, parents=True, exist_ok=False)
    command = wrap_worker_command(
        [sys.executable, str(script)],
        context=context,
        run_directory=attempt,
        repository=TRUSTED_ROOT,
    )
    env = {
        **os.environ,
        "AUTOPARETO_REMOTE_SANDBOX": "1",
        "AUTOPARETO_TRUSTED_ROOT": str(TRUSTED_ROOT),
        "AUTOPARETO_STUDY_VENV": "/workspace/autopareto/.venv",
        "AUTOPARETO_REGISTERED_CACHE": "/opt/autopareto_candidate_inputs/cache/autoresearch",
        "AUTOPARETO_CANDIDATE_RUNTIME_ROOT": "/opt/autopareto_candidate_runtime",
        "AUTOPARETO_CANDIDATE_INPUT_ROOT": "/opt/autopareto_candidate_inputs/per_attempts",
    }
    completed = subprocess.run(command, cwd=TRUSTED_ROOT, env=env, capture_output=True, text=True, check=False)
    result_path = attempt / "tmp" / "cleanroom-probe-result.json"
    evidence_dir = TRUSTED_ROOT / "state/nonstudy_tests"
    evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    evidence_path = evidence_dir / "cleanroom_isolation_probe.json"
    candidate_result = json.loads(result_path.read_text()) if result_path.is_file() else None
    evidence_path.write_text(
        json.dumps(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "result": candidate_result,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    evidence_path.chmod(0o400)
    shutil.rmtree(context, ignore_errors=True)
    shutil.rmtree(attempt, ignore_errors=True)
    if completed.returncode != 0:
        raise SystemExit(f"clean-room probe failed; evidence: {evidence_path}")
    print(evidence_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
