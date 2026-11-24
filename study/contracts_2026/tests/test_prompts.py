import difflib
import hashlib
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TIME = (ROOT / "programs/program_fixed_time.md").read_text()
TOKENS = (ROOT / "programs/program_fixed_tokens.md").read_text()


class PromptTests(unittest.TestCase):
    def test_prompts_differ_only_in_resource_contract_block(self):
        marker = "## Resource contract\n\n"
        suffix = "## Scientific and operational rules\n"
        self.assertIn(marker, TIME)
        self.assertIn(suffix, TIME)
        self.assertEqual(TIME.split(marker, 1)[0], TOKENS.split(marker, 1)[0])
        self.assertEqual(TIME.split(suffix, 1)[1], TOKENS.split(suffix, 1)[1])
        self.assertNotEqual(TIME, TOKENS)
        diff = list(difflib.unified_diff(TIME.splitlines(), TOKENS.splitlines()))
        self.assertTrue(any("CONTRACT T" in line for line in diff))
        self.assertTrue(any("CONTRACT N" in line for line in diff))
        expected_diff = "".join(difflib.unified_diff(TIME.splitlines(True), TOKENS.splitlines(True), fromfile="program_fixed_time.md", tofile="program_fixed_tokens.md"))
        self.assertEqual(expected_diff, (ROOT / "provenance/prompt_diff.patch").read_text())

    def test_no_known_leakage_terms(self):
        for prompt in (TIME, TOKENS):
            upper = prompt.upper()
            for banned in ("AP292", "AP299", "AP294", "EXPECTED SHALLOW", "EXPECTED DEEP", "PAPER ABSTRACT", "TRANSFER HYPOTHESIS"):
                self.assertNotIn(banned, upper)
            self.assertIn("validation BPB", prompt)
            self.assertIn("stable cached-decode", prompt)
            self.assertIn("peak allocated inference memory", prompt)
            self.assertIn("exactly 25 attempts", prompt)
            self.assertIn("interleaved order", prompt)
            self.assertNotIn("more tokens", prompt.lower())

    def test_snapshot_hashes_match(self):
        snapshots = json.loads((ROOT / "provenance/prompt_snapshot_manifest.json").read_text())["programs"]
        self.assertEqual(hashlib.sha256(TIME.encode()).hexdigest(), snapshots["T"]["sha256"])
        self.assertEqual(hashlib.sha256(TOKENS.encode()).hexdigest(), snapshots["N"]["sha256"])
