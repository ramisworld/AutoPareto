from __future__ import annotations

import pathlib
import tempfile
import unittest

from study.contracts_2026.smoke_workspace import (
    allocate_smoke_attempt,
    exclusive_smoke_lock,
    preserve_failure_evidence,
)


class SmokeWorkspaceTests(unittest.TestCase):
    def test_attempts_are_unique_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "attempts"
            first = allocate_smoke_attempt(root, "t-smoke")
            second = allocate_smoke_attempt(root, "t-smoke")
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_lock_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "attempts"
            with exclusive_smoke_lock(root):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with exclusive_smoke_lock(root):
                        pass

    def test_failure_evidence_survives_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            attempt = allocate_smoke_attempt(root / "attempts", "n-smoke")
            (attempt / "result.json").write_text('{"status":"error"}\n')
            destination = preserve_failure_evidence(
                attempt, root / "state", "n-smoke", RuntimeError("instrument failure")
            )
            self.assertIsNotNone(destination)
            assert destination is not None
            self.assertEqual((destination / "result.json").read_text(), '{"status":"error"}\n')
            self.assertIn("instrument failure", (destination / "failure.json").read_text())
            import shutil

            shutil.rmtree(attempt)
            self.assertTrue((destination / "failure.json").is_file())


if __name__ == "__main__":
    unittest.main()
