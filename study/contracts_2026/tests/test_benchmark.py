import unittest

from study.contracts_2026.benchmark_policy import (
    DECODE_BLOCKS,
    PROTOCOL_VERSION,
    accept_measurement,
    bracketed_decode_throughput,
    classify_measurement_stability,
    estimate_process_log_sd,
    speed_relation,
    paired_decode_throughput,
)


class BenchmarkTests(unittest.TestCase):
    def test_design_b_fixture_produces_point_and_interval(self):
        result = accept_measurement(
            correctness_passed=True,
            decode_blocks=[100.0 + index for index in range(DECODE_BLOCKS)],
        )
        self.assertEqual(result["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(result["decode_block_count"], DECODE_BLOCKS)
        self.assertIsNotNone(result["decode_velocity_point_estimate"])
        self.assertEqual(len(result["decode_velocity_95pct_interval"]), 2)

    def test_noise_is_not_an_eligibility_gate(self):
        result = accept_measurement(
            correctness_passed=True,
            decode_blocks=[100.0, 110.0, 90.0, 100.0, 105.0, 95.0, 100.0, 100.0],
        )
        self.assertEqual(result["inference_status"], "stable")

    def test_material_measured_drift_cannot_be_stable(self):
        result = classify_measurement_stability(
            [100.0, 100.2, 100.4, 100.6, 101.0, 105.0, 110.0, 113.707],
        )
        self.assertAlmostEqual(result["decode_block_range_drift"], 0.13707, places=5)
        self.assertGreater(result["decode_block_relative_drift"], 0.02)
        self.assertEqual(result["inference_status"], "unstable")

    def test_single_extreme_window_is_retained_as_diagnostic(self):
        result = classify_measurement_stability(
            [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 112.0],
        )
        self.assertGreater(result["decode_block_range_drift"], 0.1)
        self.assertLess(result["decode_block_relative_drift"], 0.02)
        self.assertEqual(result["inference_status"], "stable")

    def test_process_floor_is_log_scale_and_requires_three_runs(self):
        result = estimate_process_log_sd({"baseline": [100.0, 101.0, 99.0], "ap": [200.0, 202.0, 198.0]})
        self.assertGreaterEqual(result["process_log_sd"], 0.0)
        with self.assertRaises(ValueError):
            estimate_process_log_sd({"baseline": [100.0, 101.0]})

    def test_unresolved_speed_is_tied(self):
        result = speed_relation(101.0, (99.0, 103.0), 100.0, (98.0, 102.0))
        self.assertEqual(result["relation"], "tied")

    def test_paired_throughput_retains_absolute_values_and_uses_same_index_ratios(self):
        candidate = [110.0, 120.0, 132.0, 100.0, 90.0, 105.0, 115.0, 125.0]
        baseline = [100.0, 100.0, 110.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        result = paired_decode_throughput(candidate, baseline)
        self.assertEqual(result["absolute_candidate_tok_s"], 112.5)
        self.assertEqual(result["absolute_baseline_tok_s"], 100.0)
        self.assertEqual(result["normalized_pairwise_ratios"], [value / base for value, base in zip(candidate, baseline)])
        self.assertEqual(result["normalized_decode_throughput"], 1.125)
        self.assertEqual(len(result["absolute_candidate_ci_95pct"]), 2)
        self.assertEqual(len(result["absolute_baseline_ci_95pct"]), 2)
        self.assertEqual(len(result["normalized_decode_throughput_ci_95pct"]), 2)

    def test_paired_throughput_requires_eight_windows_per_arm(self):
        with self.assertRaises(ValueError):
            paired_decode_throughput([100.0] * 7, [100.0] * 8)

    def test_bracketed_throughput_interpolates_at_candidate_time(self):
        def rows(values):
            return [
                {
                    "decode_tokens_per_second": value,
                    "started_at_unix_ns": index * 1000,
                    "completed_at_unix_ns": index * 1000 + 100,
                }
                for index, value in enumerate(values)
            ]

        before = rows([100.0] * DECODE_BLOCKS)
        candidate = rows([105.0] * DECODE_BLOCKS)
        after = rows([110.0] * DECODE_BLOCKS)
        for index in range(DECODE_BLOCKS):
            before[index]["started_at_unix_ns"] = index * 3000
            before[index]["completed_at_unix_ns"] = index * 3000 + 100
            candidate[index]["started_at_unix_ns"] = index * 3000 + 1000
            candidate[index]["completed_at_unix_ns"] = index * 3000 + 1100
            after[index]["started_at_unix_ns"] = index * 3000 + 2000
            after[index]["completed_at_unix_ns"] = index * 3000 + 2100
        result = bracketed_decode_throughput(candidate, before, after)
        self.assertEqual(result["interpolated_baseline_window_tok_s"], [105.0] * DECODE_BLOCKS)
        self.assertEqual(result["normalized_pairwise_ratios"], [1.0] * DECODE_BLOCKS)
        self.assertEqual(result["normalized_decode_throughput"], 1.0)
        self.assertEqual(result["normalized_inference_status"], "stable")

    def test_bracketed_throughput_requires_ordered_timestamps(self):
        rows = [
            {
                "decode_tokens_per_second": 100.0,
                "started_at_unix_ns": 0,
                "completed_at_unix_ns": 100,
            }
            for _ in range(DECODE_BLOCKS)
        ]
        with self.assertRaises(ValueError):
            bracketed_decode_throughput(rows, rows, rows)
