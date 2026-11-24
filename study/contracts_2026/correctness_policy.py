"""Frozen, architecture-neutral cache-correctness policy.

The BF16 comparison is retained as a diagnostic screen because backend arithmetic
ordering can move logits without changing the selected token. A protected FP32
full-prefix comparison is the numerical adjudication whenever the registered
candidate adapter can be evaluated in FP32. No model identity is used.
"""

from __future__ import annotations


CORRECTNESS_RULE_VERSION = "autopareto_cache_correctness_fp32_adjudicated_v1"
BF16_SCREEN_MEAN_ATOL = 0.03
FP32_ADJUDICATION_MEAN_ATOL = 1e-4
FP32_ADJUDICATION_MAX_ATOL = 1e-3
CHECKED_DECODE_STEPS = 32

