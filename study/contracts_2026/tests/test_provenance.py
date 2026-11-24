import pathlib
import unittest

from study.contracts_2026.provenance import REQUIRED_ATTEMPT_FIELDS, portable_path


class ProvenanceTests(unittest.TestCase):
    def test_schema_fields_are_complete_and_paths_portable(self):
        required = {
            "program", "agent_model_identifier", "reasoning_setting", "api_configuration_available",
            "campaign_id", "contract_id", "attempt_number", "training_seed", "parent_experiment_id",
            "hypothesis", "intended_change", "change_category", "candidate_source_before_sha256",
            "candidate_source_after_sha256", "starting_source_hash", "evaluator_hashes", "dataset_hashes",
            "tokenizer_hash", "agent_context_files", "system_context_text", "training_environment", "gpu_identity", "training_result", "inference_result",
            "inference_raw_repetitions", "pareto_decision", "checkpoint_hash", "command",
        }
        self.assertTrue(required.issubset(set(REQUIRED_ATTEMPT_FIELDS)))
        self.assertEqual(portable_path("runs/campaign/attempt.json"), "runs/campaign/attempt.json")
        with self.assertRaises(ValueError):
            portable_path("/absolute/path")
        with self.assertRaises(ValueError):
            portable_path("../outside")
