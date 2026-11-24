import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from study.contracts_2026 import campaign_controller
from study.contracts_2026.isolation import (
    SCIENTIST_CHILD_EDIT_SCHEMA,
    SCIENTIST_PARENT_SELECTION_SCHEMA,
    ScientistExecutionTimeout,
    _supervised_codex,
    run_agent_command,
)
from study.contracts_2026.scientist_config import ScientistConfig


class ScientistConfigTests(unittest.TestCase):
    def test_manifest_frozen_gpu_identity_is_used_for_continuation(self):
        new_uuid = "GPU-35264a0f-e6b7-563d-3871-761953a3ae8f"
        payload = {
            "a40": {"gpu_uuid": new_uuid},
            "contracts": {"N": {"active_n_star": campaign_controller.N_STAR}},
        }
        self.assertEqual(campaign_controller._manifest_a40_uuid(payload), new_uuid)
        self.assertTrue(campaign_controller._manifest_contains_expected_identity(payload))

    def test_model_and_reasoning_are_configuration_values(self):
        for model, reasoning in (
            ("gpt-5.6-sol", "high"),
            ("gpt-5.6-sol", "medium"),
            ("gpt-5.6-terra", "medium"),
        ):
            ScientistConfig(
                model=model,
                reasoning=reasoning,
                scientist_bin=sys.executable,
                execution_backend="host",
                sandbox_mode="workspace-write",
            ).validate()

    def test_timeout_kills_the_codex_process_group_and_persists_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            context = root / "context"
            context.mkdir()
            fake_codex = root / "codex"
            fake_codex.write_text("#!/bin/sh\nsleep 30\n")
            fake_codex.chmod(0o755)
            config = ScientistConfig(
                model="gpt-5.6-sol",
                reasoning="high",
                scientist_bin=str(fake_codex),
                codex_version="test",
                execution_backend="host",
                sandbox_mode="workspace-write",
                timeout_seconds=0.05,
            )
            command = [
                str(fake_codex), "exec", "-C", str(context), "-s", "workspace-write",
                "--output-schema", str(SCIENTIST_PARENT_SELECTION_SCHEMA),
            ]
            with self.assertRaises(ScientistExecutionTimeout) as raised:
                run_agent_command(
                    command,
                    context,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    scientist_config=config,
                    diagnostic_root=root / "diagnostics",
                )
            record = pathlib.Path(str(raised.exception).split("diagnostics=", 1)[1])
            payload = __import__("json").loads((record).read_text())
            self.assertEqual(payload["status"], "FAILED")
            self.assertEqual(payload["failure_class"], "infrastructure")
            self.assertEqual(payload["timeout_seconds"], 0.05)
            self.assertTrue((record.parent / "stdout.jsonl").is_file())
            self.assertTrue((record.parent / "stderr.log").is_file())

    def test_timeout_terminates_a_real_disposable_docker_container(self):
        if shutil.which("docker") is None:
            self.skipTest("Docker is unavailable")
        if subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            self.skipTest("Docker daemon is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            context = root / "context"
            context.mkdir()
            command = [
                "docker", "run", "--rm", "--init", "--read-only", "--entrypoint", "/bin/sh",
                "autopareto-scientist-codex:0.153.0", "-c", "sleep 30",
            ]
            started = time.monotonic()
            with self.assertRaises(ScientistExecutionTimeout) as raised:
                _supervised_codex(
                    command,
                    context,
                    timeout_seconds=1.0,
                    diagnostic_root=root / "diagnostics",
                    invocation_metadata={"probe": "real_docker_timeout"},
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                )
            self.assertLess(time.monotonic() - started, 8.0)
            record = pathlib.Path(str(raised.exception).split("diagnostics=", 1)[1])
            payload = json.loads(record.read_text())
            self.assertEqual(payload["status"], "FAILED")
            self.assertEqual(payload["failure_class"], "infrastructure")
            cidfile = pathlib.Path(str(payload["container_id_path"]))
            container_id = cidfile.read_text().strip()
            self.assertNotEqual(container_id, "")
            self.assertNotEqual(
                subprocess.run(["docker", "inspect", container_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode,
                0,
            )

    def test_manifest_accepts_each_selected_configuration_without_model_branches(self):
        for model, reasoning in (
            ("gpt-5.6-sol", "high"),
            ("gpt-5.6-sol", "medium"),
            ("gpt-5.6-terra", "medium"),
        ):
            config = ScientistConfig(
                model=model,
                reasoning=reasoning,
                scientist_bin=sys.executable,
                codex_version="test-codex",
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "manifest.json"
                path.write_text(json.dumps({
                    "a40_uuid": campaign_controller.A40_UUID,
                    "n_star": campaign_controller.N_STAR,
                    "scientist": {"configuration": config.as_dict()},
                    "contracts": {"N": {"active_n_star": campaign_controller.N_STAR}},
                }))
                with patch.object(campaign_controller, "_verify_manifest_hashes", return_value={}):
                    observed = campaign_controller._verify_final_manifest(
                        path,
                        config,
                        expected_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                self.assertEqual(observed["scientist"]["configuration"], config.as_dict())

    def test_manifest_model_or_reasoning_mismatch_fails_before_hash_or_execution(self):
        requested = ScientistConfig(
            model="gpt-5.6-sol",
            reasoning="high",
            scientist_bin=sys.executable,
            codex_version="test-codex",
        )
        frozen = ScientistConfig(
            model="gpt-5.6-terra",
            reasoning="medium",
            scientist_bin=sys.executable,
            codex_version="test-codex",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "manifest.json"
            path.write_text(json.dumps({
                "a40_uuid": campaign_controller.A40_UUID,
                "n_star": campaign_controller.N_STAR,
                "scientist": {"configuration": frozen.as_dict()},
                "contracts": {"N": {"active_n_star": campaign_controller.N_STAR}},
            }))
            with self.assertRaisesRegex(RuntimeError, "ScientistConfig"):
                campaign_controller._verify_final_manifest(path, requested)

    def test_command_propagates_each_selected_model_and_reasoning(self):
        for model, reasoning in (
            ("gpt-5.6-sol", "high"),
            ("gpt-5.6-sol", "medium"),
            ("gpt-5.6-terra", "medium"),
        ):
            config = ScientistConfig(
                model=model,
                reasoning=reasoning,
                scientist_bin="/Applications/ChatGPT.app/Contents/Resources/codex",
            )
            command = campaign_controller._scientist_command(
                pathlib.Path("/tmp/context"), "campaign", 1, config, "select_parent"
            )
            self.assertEqual(command[0], config.scientist_bin)
            self.assertEqual(command[command.index("-m") + 1], model)
            self.assertIn(f"model_reasoning_effort={reasoning}", command)
            schema_index = command.index("--output-schema")
            self.assertEqual(pathlib.Path(command[schema_index + 1]).resolve(), SCIENTIST_PARENT_SELECTION_SCHEMA)

    def test_registered_proposal_schema_requires_genuine_semantic_fields(self):
        parent = json.loads(SCIENTIST_PARENT_SELECTION_SCHEMA.read_text())
        child = json.loads(SCIENTIST_CHILD_EDIT_SCHEMA.read_text())
        self.assertEqual(set(parent["required"]), set(parent["properties"]))
        self.assertEqual(set(child["required"]), set(child["properties"]))
        self.assertIn("parent_experiment_id", parent["properties"])
        self.assertNotIn("parent_experiment_id", child["properties"])
        self.assertFalse(parent["additionalProperties"])
        self.assertFalse(child["additionalProperties"])
        self.assertEqual(parent["properties"]["hypothesis"]["minLength"], 1)


if __name__ == "__main__":
    unittest.main()
