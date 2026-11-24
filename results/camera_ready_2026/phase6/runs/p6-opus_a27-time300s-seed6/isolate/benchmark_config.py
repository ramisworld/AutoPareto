"""Immutable settings shared by every measured AutoPareto experiment."""

TRAINING_BUDGET_SECONDS = 300
TRAINING_TIMEOUT_SECONDS = 900
PRECISION = "bfloat16"
INPUT_TOKENS = 256
GENERATED_TOKENS = 256
WARMUP_ITERATIONS = 2
WARMUP_GENERATED_TOKENS = GENERATED_TOKENS
MEASURED_INFERENCE_REPETITIONS = 10
GENERATION_MODE = "greedy"
OFFICIAL_INFERENCE_METRIC_MODE = "cached_decode"
LEGACY_INFERENCE_METRIC_MODE = "uncached_legacy"
# A40 validation first compares the candidate's full-prefix and cached paths in
# bfloat16. Different execution shapes need not be bit-identical. Candidates below
# the calibrated 0.03 mean-error bound take the fast path. When every greedy token
# agrees but bfloat16 drift exceeds that bound, a strict float32 math-SDPA comparison
# adjudicates cache semantics. Official speed and memory always remain bfloat16.
CORRECTNESS_PREFIX_TOKENS = (64, 256)
CORRECTNESS_DECODE_STEPS = 32
LOGIT_CORRECTNESS_MEAN_ATOL = 0.03
FP32_LOGIT_CORRECTNESS_MEAN_ATOL = 1e-5
FP32_LOGIT_CORRECTNESS_MAX_ATOL = 1e-4
TRAINING_MAX_OVERRUN_SECONDS = 15
CUDA_ALLOCATOR = "backend:cudaMallocAsync"

# Tokenized with the protected upstream tokenizer, then repeated/truncated to exactly
# INPUT_TOKENS. This text and the tokenizer are protected, so the token IDs are fixed.
FIXED_INPUT_TEXT = (
    "AutoPareto fixed inference benchmark. The purpose of this input is to measure "
    "decoding throughput and CUDA memory under identical conditions. No generated "
    "text is scored for meaning, style, correctness, or safety. "
)

PROTECTED_FILES = (
    "prepare.py",
    "benchmark_config.py",
    "benchmark_stability.py",
    "inference_benchmark.py",
    "deployment_benchmark.py",
    "autopareto_db.py",
    "event_log.py",
    "runner.py",
    "protected/inference_adapter.py",
    "protected/train_launcher.py",
    "protected/smoke_gpu.py",
)
