import unittest
from unittest.mock import patch

from study.contracts_2026.batching import (
    ALLOWED_TOTAL_BATCH_SIZES,
    COMMON_TOKEN_QUANTUM,
    exact_optimizer_updates,
    validate_batch_config,
    validate_n_star,
)
from study.contracts_2026.calibration import choose_n_star
from study.contracts_2026.contracts import ContractId, ResourceContract
from study.contracts_2026.runtime import TrainingLedger
from study.contracts_2026.runtime import MeasuredTrainingClock


class ContractTests(unittest.TestCase):
    def test_progress_reaches_one_at_boundary(self):
        time_contract = ResourceContract(ContractId.T)
        token_contract = ResourceContract(ContractId.N, time_seconds=None, n_star=COMMON_TOKEN_QUANTUM)
        self.assertEqual(time_contract.progress(measured_training_time=300.0, total_tokens=1), 1.0)
        self.assertEqual(token_contract.progress(measured_training_time=0.0, total_tokens=COMMON_TOKEN_QUANTUM), 1.0)

    def test_fixed_token_ledger_is_exact(self):
        batch = validate_batch_config(131072, 64, 1)
        ledger = TrainingLedger(batch, ResourceContract(ContractId.N, time_seconds=None, n_star=524288))
        for _ in range(4):
            ledger.before_microbatch()
            ledger.after_microbatch()
            ledger.after_optimizer_update()
        result = ledger.finish(0.0)
        self.assertEqual(result["total_tokens"], 524288)
        self.assertEqual(result["optimizer_updates"], 4)
        self.assertEqual(result["microbatches"], 4)
        self.assertEqual(result["progress"], 1.0)

    def test_registered_n_star_boundary_is_exact_and_rejects_one_quantum_overrun(self):
        n_star = 88_604_672
        quantum = 524_288
        batch = validate_batch_config(quantum, 64, 4)
        ledger = TrainingLedger(batch, ResourceContract(ContractId.N, time_seconds=None, n_star=n_star))
        for _ in range((n_star // quantum) - 1):
            ledger.before_microbatch()
            for _ in range(4):
                ledger.after_microbatch()
            ledger.after_optimizer_update()
        self.assertEqual(ledger.total_tokens, n_star - quantum)
        for _ in range(4):
            ledger.before_microbatch()
            ledger.after_microbatch()
        ledger.after_optimizer_update()
        self.assertEqual(ledger.total_tokens, n_star)
        with self.assertRaisesRegex(RuntimeError, "partial fixed-token microbatch"):
            ledger.before_microbatch()

    def test_no_partial_token_step(self):
        batch = validate_batch_config(32768, 8, 2)
        ledger = TrainingLedger(batch, ResourceContract(ContractId.N, time_seconds=None, n_star=COMMON_TOKEN_QUANTUM))
        for _ in range(3):
            ledger.before_microbatch()
            ledger.after_microbatch()
        with self.assertRaises(RuntimeError):
            ledger.after_optimizer_update()

    def test_common_divisibility_and_calibration(self):
        self.assertEqual(COMMON_TOKEN_QUANTUM, 524288)
        self.assertTrue(all(COMMON_TOKEN_QUANTUM % value == 0 for value in ALLOWED_TOTAL_BATCH_SIZES))
        self.assertEqual(exact_optimizer_updates(COMMON_TOKEN_QUANTUM, 32768), 16)
        self.assertEqual(choose_n_star([92798976, 92798976, 92798976]), 92798976)
        self.assertEqual(choose_n_star([92798977, 92798976, 92798975]), 92798976)
        self.assertEqual(validate_n_star(524288), 524288)

    def test_fixed_time_clock_stops_only_after_completed_update(self):
        clock = MeasuredTrainingClock()
        with patch("study.contracts_2026.runtime.time.perf_counter", side_effect=[10.0, 309.999, 310.001, 310.002]):
            clock.start()
            self.assertFalse(clock.should_stop_after_completed_update())
            self.assertTrue(clock.should_stop_after_completed_update())
            self.assertAlmostEqual(clock.stop(), 300.002)
