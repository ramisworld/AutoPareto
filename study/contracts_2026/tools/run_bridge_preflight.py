"""Run one harmless local Scientist/orchestrator bridge preflight.

This never calls the A40 transport and never launches training.  It exercises the
same local sandbox handoff and records a synthetic protected result only.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

from study.contracts_2026.bridge import (
    build_registered_metadata,
    run_local_scientist,
    sha256_file,
    stage_remote_payload,
    update_campaign_view,
)
from study.contracts_2026.context import build_campaign_view
from study.contracts_2026.isolation import stage_agent_context


ROOT = pathlib.Path(__file__).resolve().parents[3]
CAMPAIGN_ID = "preflight-synthetic-campaign"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="autopareto-bridge-preflight-") as tmp:
        root = pathlib.Path(tmp)
        context = root / "context"
        view = root / "campaign_view.json"
        view.write_text(json.dumps(build_campaign_view(
            campaign_id=CAMPAIGN_ID,
            contract_id="T",
            baseline={
                "campaign_id": CAMPAIGN_ID,
                "record_scope": "same_seed_baseline",
                "validation_bpb": 2.0,
            },
            frontier=[],
            attempt_number=1,
        ), sort_keys=True) + "\n")
        stage_agent_context(
            destination=context,
            baseline_train=ROOT / "study/contracts_2026/historical_reference/baseline_source/train.py",
            baseline_prepare=ROOT / "study/contracts_2026/historical_reference/baseline_source/prepare.py",
            study_runtime=ROOT / "study/contracts_2026/study_runtime.py",
            program=ROOT / "study/contracts_2026/programs/program_fixed_time.md",
            campaign_view=view,
        )

        scientist_code = (
            "import json, pathlib, subprocess\n"
            "c = pathlib.Path('.')\n"
            "visible = sorted(p.name for p in c.iterdir())\n"
            "checks = {}\n"
            "try: pathlib.Path('/Users/Rami/Documents/dev/autopareto/README.md').read_text(); checks['parent_read'] = 'unexpected_success'\n"
            "except Exception as exc: checks['parent_read'] = type(exc).__name__\n"
            "try: pathlib.Path('prepare.py').write_text('blocked'); checks['protected_write'] = 'unexpected_success'\n"
            "except Exception as exc: checks['protected_write'] = type(exc).__name__\n"
            "try: pathlib.Path('campaign_view.json').write_text('blocked'); checks['view_write'] = 'unexpected_success'\n"
            "except Exception as exc: checks['view_write'] = type(exc).__name__\n"
            "try: pathlib.Path('/tmp/autopareto-scientist-write-test').write_text('blocked'); checks['tmp_write'] = 'unexpected_success'\n"
            "except Exception as exc: checks['tmp_write'] = type(exc).__name__\n"
            "try: subprocess.run(['/usr/bin/ssh', '-o', 'ConnectTimeout=1', 'invalid.invalid'], check=False, capture_output=True); checks['ssh_exec'] = 'unexpected_success'\n"
            "except Exception as exc: checks['ssh_exec'] = type(exc).__name__\n"
            "pathlib.Path('train.py').write_text(pathlib.Path('train.py').read_text() + '\\n# harmless bridge preflight edit\\n')\n"
            "print(json.dumps({'preflight_checks':checks,'visible':visible}))\n"
            "print(json.dumps({'hypothesis':'synthetic bridge transport check','intended_change':'harmless local edit','change_category':'preflight'}))\n"
        )
        handoff, completed = run_local_scientist(
            command=[sys.executable, "-c", scientist_code],
            context=context,
            campaign_id=CAMPAIGN_ID,
            contract_id="T",
            attempt_number=1,
            model_identifier="preflight-synthetic-model",
            reasoning_setting="preflight",
            api_configuration=None,
            system_context_text=None,
        )
        metadata = build_registered_metadata(
            handoff=handoff,
            study_id="contracts-2026-preflight",
            training_seed=3,
            starting_source_hash=handoff.candidate_source_before_sha256,
            evaluator_hashes={"inference": sha256_file(ROOT / "study/contracts_2026/inference_benchmark.py")},
            dataset_hashes={"manifest": "1211827090aa7e669ec8bcd965ae1423dc90697f0e4928d0fce707cbc36c3f93"},
            tokenizer_hash="1211827090aa7e669ec8bcd965ae1423dc90697f0e4928d0fce707cbc36c3f93",
            contract_parameters={"training_seconds": 300.0},
            protected_prepare_sha256=sha256_file(ROOT / "study/contracts_2026/historical_reference/baseline_source/prepare.py"),
            study_runtime_sha256=sha256_file(ROOT / "study/contracts_2026/study_runtime.py"),
        )
        payload = stage_remote_payload(destination=root / "payload", candidate=context / "train.py", metadata=metadata)
        synthetic_result = {
            "campaign_id": CAMPAIGN_ID,
            "attempt": 1,
            "protected_validation_bpb": 1.9,
            "inference_result": {
                "inference_status": "stable",
                "cached_decode_tokens_per_second_median": 10.0,
                "peak_allocated_inference_memory_bytes": 20,
            },
        }
        updated_view = update_campaign_view(view_path=context / "campaign_view.json", result=synthetic_result)
        record = {
            "schema": "autopareto_contracts_2026_bridge_preflight_v1",
            "scientist_execution": "local_macos_sandbox_only",
            "a40_transport_invoked": False,
            "visible_files": sorted(p.name for p in context.iterdir()),
            "writable_files": ["train.py"],
            "handoff": handoff.as_dict(),
            "scientist_stdout": completed.stdout,
            "registered_metadata_keys": sorted(metadata),
            "remote_payload_files": sorted(p.name for p in payload.iterdir()),
            "synthetic_result": synthetic_result,
            "campaign_view_after": updated_view,
            "metadata_contains_remote_credentials": any(key in json.dumps(metadata).lower() for key in ("ssh", "runpod", "private_key")),
        }
    output = ROOT / "study/contracts_2026/provenance/bridge_preflight.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": "study/contracts_2026/provenance/bridge_preflight.json",
        "visible_files": record["visible_files"],
        "writable_files": record["writable_files"],
        "remote_payload_files": record["remote_payload_files"],
        "metadata_contains_remote_credentials": record["metadata_contains_remote_credentials"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
