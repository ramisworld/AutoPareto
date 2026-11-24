"""Unit tests of scripts/amend_scientist_timeout.py on a synthetic paused single-arm state.

Run from the repository root:  python3 -B -m unittest scripts.tests.test_amend_scientist_timeout -v
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import stat
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import amend_scientist_timeout as amend  # noqa: E402


def write(path: pathlib.Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AmendScientistTimeoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="amend-timeout-")
        self.root = pathlib.Path(self._tmp.name).resolve()
        self.state_root = self.root / "state"
        config = {"model": "claude-opus-5-5", "reasoning": "xhigh", "timeout_seconds": 600.0, "agent_backend": "claude"}
        self.manifest = self.state_root / "provenance" / "current.json"
        manifest_sha = write(self.manifest, {"scientist": {"configuration": config}, "schema": "x"})
        schedule = [{"global_position": p, "campaign_attempt": (p + 1) // 2, "contract_id": "T" if p % 2 == 0 else "N",
                     "campaign_id": "t" if p % 2 == 0 else "n"} for p in range(1, 9)]
        positions = {str(p): {"status": "not_in_study"} for p in range(1, 9, 2)}
        positions.update({"2": {"status": "completed"}, "4": {"status": "completed"},
                          "6": {"status": "pending", "requeued_scientist_timeouts": [{"type": "ScientistExecutionTimeout"}]},
                          "8": {"status": "pending"}})
        self.state = {"manifest_sha256": manifest_sha, "scientist_config": dict(config), "schedule": schedule,
                      "positions": positions, "active_contracts": ["T"]}
        self.state_path = self.state_root / "controller_state.json"
        write(self.state_path, self.state)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def amend(self, **overrides):
        arguments = dict(state_root=self.state_root, manifest_path=self.manifest, new_timeout_seconds=900,
                         new_manifest_name="amended.json", reason="child edit exceeded 600 s three times",
                         approval="PI: Raise to 900 s")
        arguments.update(overrides)
        return amend.amend_scientist_timeout(**arguments)

    def test_amendment_changes_only_the_timeout_and_records_itself(self) -> None:
        state_before = self.state_path.read_bytes()
        result = self.amend()
        self.assertEqual(result["effective_global_position"], 6)
        new_manifest = json.loads(pathlib.Path(result["manifest"]).read_text())
        old_manifest = json.loads(self.manifest.read_text())
        self.assertEqual(new_manifest["scientist"]["configuration"]["timeout_seconds"], 900.0)
        [entry] = new_manifest["operational_amendments"]
        self.assertEqual(entry["scientist_timeout_seconds"], {"before": 600.0, "after": 900.0})
        self.assertEqual(entry["changed_immutable_paths"], {})
        self.assertFalse(entry["scientific_contract_changed"])
        stripped = json.loads(json.dumps(new_manifest))
        stripped["scientist"]["configuration"]["timeout_seconds"] = 600.0
        del stripped["operational_amendments"]
        self.assertEqual(stripped, old_manifest)
        self.assertFalse(pathlib.Path(result["manifest"]).stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

        state = json.loads(self.state_path.read_text())
        self.assertEqual(state["manifest_sha256"], result["manifest_sha256"])
        self.assertEqual(state["scientist_config"], new_manifest["scientist"]["configuration"])
        self.assertEqual(state["positions"]["6"]["requeued_scientist_timeouts"][-1]["followed_by_timeout_amendment"]
                         ["scientist_timeout_seconds"], {"before": 600.0, "after": 900.0})
        [backup] = self.state_root.glob("controller_state.json.pre_timeout_amendment_*")
        self.assertEqual(backup.read_bytes(), state_before)
        with self.assertRaisesRegex(RuntimeError, "does not run under the given manifest"):
            self.amend(new_manifest_name="again.json")

    def test_refusals(self) -> None:
        self.state["positions"]["6"]["status"] = "reconciliation_required"
        write(self.state_path, self.state)
        with self.assertRaisesRegex(RuntimeError, "completed or pending"):
            self.amend()
        self.state["positions"]["6"]["status"] = "pending"
        self.state["scientist_config"]["timeout_seconds"] = 700.0
        write(self.state_path, self.state)
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            self.amend()
        self.state["scientist_config"]["timeout_seconds"] = 600.0
        write(self.state_path, self.state)
        with self.assertRaises(ValueError):
            self.amend(new_timeout_seconds=600)
        with self.assertRaises(ValueError):
            self.amend(approval=" ")
        self.assertFalse((self.manifest.parent / "amended.json").exists())
        self.assertEqual(list(self.state_root.glob("controller_state.json.pre_timeout_amendment_*")), [])


if __name__ == "__main__":
    unittest.main()
