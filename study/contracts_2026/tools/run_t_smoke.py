"""Run one complete non-study registered-baseline Contract-T path on an A40."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

from study.contracts_2026.provenance import sha256_file
from study.contracts_2026.remote_runner import _candidate_prepare_source
from study.contracts_2026.smoke_workspace import (
    allocate_smoke_attempt,
    exclusive_smoke_lock,
    preserve_failure_evidence,
)


ROOT = pathlib.Path(os.environ.get("AUTOPARETO_TRUSTED_ROOT", "/opt/autopareto_trusted")).resolve(strict=True)
STATE = pathlib.Path(os.environ.get("AUTOPARETO_TRUSTED_STATE_ROOT", str(ROOT / "state"))).resolve()
SOURCE = ROOT / "study/contracts_2026/historical_reference/baseline_source/train.py"
PREPARE = SOURCE.parent / "prepare.py"
RUNTIME = ROOT / "study/contracts_2026/study_runtime.py"
CONTEXT_ROOT = pathlib.Path(
    os.environ.get("AUTOPARETO_CANDIDATE_CONTEXT_ROOT", "/var/lib/autopareto/candidate_contexts")
).resolve()
ATTEMPT_ROOT = pathlib.Path(
    os.environ.get("AUTOPARETO_CANDIDATE_ATTEMPT_ROOT", "/var/lib/autopareto/candidate_attempts")
).resolve()


def main() -> int:
    if sha256_file(SOURCE) != "b2fb8788f02517af8bc2fcfa059fb799a6974c9050d00e322168ec414fac4fab":
        raise SystemExit("registered baseline source hash mismatch")
    context = None
    attempt = None
    with exclusive_smoke_lock(ATTEMPT_ROOT):
        try:
            CONTEXT_ROOT.mkdir(mode=0o755, parents=True, exist_ok=True)
            context = pathlib.Path(tempfile.mkdtemp(prefix="t-smoke-", dir=CONTEXT_ROOT))
            context.chmod(0o555)
            shutil.copy2(SOURCE, context / "train.py")
            (context / "prepare.py").write_text(_candidate_prepare_source(PREPARE))
            shutil.copy2(RUNTIME, context / "study_runtime.py")
            for path in context.iterdir():
                path.chmod(0o444)
            attempt = allocate_smoke_attempt(ATTEMPT_ROOT, "t-smoke")
            output = attempt / "result.json"
            command = [
                sys.executable,
                "-m",
                "study.contracts_2026.train_launcher",
                "--context", str(context),
                "--candidate", "train.py",
                "--contract", "T",
                "--seed", "3",
                "--output", str(output),
                "--baseline-calibration",
                "--execute",
            ]
            env = {
                **os.environ,
                "AUTOPARETO_REMOTE_SANDBOX": "1",
                "AUTOPARETO_TRUSTED_ROOT": str(ROOT),
                "AUTOPARETO_TRUSTED_STATE_ROOT": str(STATE),
                "AUTOPARETO_STUDY_VENV": os.environ.get("AUTOPARETO_STUDY_VENV", "/usr/local"),
                "AUTOPARETO_REGISTERED_CACHE": os.environ.get("AUTOPARETO_REGISTERED_CACHE", "/var/cache/autopareto/pair_a_registered_cache"),
                "AUTOPARETO_CANDIDATE_ROOT": os.environ.get("AUTOPARETO_CANDIDATE_ROOT", "/var/lib/autopareto/candidate_attempts"),
                "AUTOPARETO_CANDIDATE_CONTEXT_ROOT": str(CONTEXT_ROOT),
                "AUTOPARETO_CANDIDATE_RUNTIME_ROOT": "/opt/autopareto_candidate_runtime",
                "AUTOPARETO_CANDIDATE_INPUT_ROOT": "/opt/autopareto_candidate_inputs/per_attempts",
                "AUTOPARETO_STUDY_ALLOW_EXECUTION": "1",
            }
            completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
            if completed.returncode != 0 or not output.is_file():
                raise SystemExit(f"T smoke failed with return code {completed.returncode}")
            result = json.loads(output.read_text())
            if result.get("status") != "completed":
                raise SystemExit(f"T smoke did not complete: {result}")
            artifact_dir = STATE / "nonstudy_tests/t_smoke_artifacts"
            artifact_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            persisted = {}
            checkpoint = pathlib.Path(str(result.get("checkpoint", "")))
            for source in (output, output.with_name(output.name + ".worker.json"), checkpoint, output.with_name("result.inference.json")):
                if source.is_file() and source.resolve().is_relative_to(attempt):
                    destination = artifact_dir / source.name
                    shutil.copy2(source, destination)
                    destination.chmod(0o400)
                    persisted[source.name] = {"path": str(destination), "sha256": sha256_file(destination)}
            record = {
                "status": "passed",
                "source_sha256": sha256_file(SOURCE),
                "returncode": completed.returncode,
                "result": result,
                "persisted_artifacts": persisted,
            }
            evidence = STATE / "nonstudy_tests/t_smoke.json"
            evidence.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
            evidence.chmod(0o400)
            print(json.dumps({"evidence": str(evidence), "result": result}, sort_keys=True, default=str))
            return 0
        except BaseException as exc:
            preserve_failure_evidence(attempt, STATE, "t-smoke", exc)
            raise
        finally:
            if context is not None:
                shutil.rmtree(context, ignore_errors=True)
            if attempt is not None:
                shutil.rmtree(attempt, ignore_errors=True)
            shutil.rmtree(ATTEMPT_ROOT.parent / "autopareto_candidate_inputs" / "per_attempts" / "autopareto-candidate-attempts-t-smoke", ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
