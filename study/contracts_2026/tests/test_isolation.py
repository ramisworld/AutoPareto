import json
import pathlib
import tempfile
import unittest

from study.contracts_2026.isolation import (
    AGENT_VISIBLE_PATHS,
    path_is_forbidden,
    stage_agent_context,
    validate_context_paths,
)


class IsolationTests(unittest.TestCase):
    def test_allowlist_is_exact(self):
        validate_context_paths(AGENT_VISIBLE_PATHS)
        self.assertTrue(path_is_forbidden("results/results.db"))
        self.assertTrue(path_is_forbidden("docs/ABSTRACT_DRAFT.md"))
        self.assertTrue(path_is_forbidden("PROJECT_STATE.md"))
        with self.assertRaises(ValueError):
            validate_context_paths(AGENT_VISIBLE_PATHS + ("results/results.db",))

    def test_staging_copies_only_five_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            sources = []
            for name in ("train.py", "prepare.py", "study_runtime.py", "program.md", "campaign_view.json"):
                p = root / name
                p.write_text("{}" if name.endswith(".json") else name)
                sources.append(p)
            destination = root / "context"
            stage_agent_context(
                destination=destination,
                baseline_train=sources[0],
                baseline_prepare=sources[1],
                study_runtime=sources[2],
                program=sources[3],
                campaign_view=sources[4],
            )
            self.assertEqual(sorted(p.name for p in destination.iterdir()), sorted(AGENT_VISIBLE_PATHS))
            self.assertFalse((destination / "results").exists())

