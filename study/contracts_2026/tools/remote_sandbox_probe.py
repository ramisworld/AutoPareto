"""Harmless probe executed inside the remote candidate namespace."""

from __future__ import annotations

import json
import pathlib
import socket
import subprocess


checks = []


def attempt(kind, target, function):
    try:
        function()
    except (PermissionError, FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        checks.append({"kind": kind, "target": target, "denied": True, "error": type(exc).__name__})
    else:
        checks.append({"kind": kind, "target": target, "denied": False})


for target in (
    "/workspace/autopareto/README.md",
    "/workspace/autopareto/PROJECT_STATE.md",
    "/workspace/autopareto/docs",
    "/workspace/autopareto/results/results.db",
    "/workspace/autopareto/results/candidates/tpe-seed42-20260805/0002-train.py",
    "/workspace/autopareto/study/contracts_2026/campaign.py",
):
    path = pathlib.Path(target)
    attempt("read", target, lambda path=path: path.read_bytes())

for target in (
    "/workspace/autopareto/protected/inference_adapter.py",
    "/workspace/autopareto/prepare.py",
    "/workspace/autopareto/results/results.db",
    "/workspace/autopareto/results/other-campaign/probe",
):
    path = pathlib.Path(target)
    attempt("write", target, lambda path=path: path.write_text("probe"))

attempt("network", "1.1.1.1:443", lambda: socket.create_connection(("1.1.1.1", 443), 0.5))
attempt(
    "ssh",
    "ssh.runpod.io",
    lambda: subprocess.run(
        ["/usr/bin/ssh", "-o", "ConnectTimeout=1", "ssh.runpod.io"],
        check=True,
        timeout=2,
    ),
)
print(json.dumps({"sandbox": "autopareto_linux_bwrap_allowlist_v2", "checks": checks}, sort_keys=True))
