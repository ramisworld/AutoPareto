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
# A40 validation compares the candidate's full-prefix and cached paths in bfloat16.
# Different attention kernels need not be bit-identical. Check a short sequence on
# identical teacher-forced prefixes so one near-tie cannot cascade into 256 unrelated
# comparisons. The unchanged A40 baselines peaked at 0.01393 mean absolute logit
# error, so 0.03 is a conservative semantic bound. Greedy next-token IDs must match
# at both protected prefix lengths.
CORRECTNESS_PREFIX_TOKENS = (64, 256)
CORRECTNESS_DECODE_STEPS = 32
LOGIT_CORRECTNESS_MEAN_ATOL = 0.03
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
    "inference_benchmark.py",
    "deployment_benchmark.py",
    "autopareto_db.py",
    "event_log.py",
    "runner.py",
    "protected/inference_adapter.py",
    "protected/train_launcher.py",
    "protected/smoke_gpu.py",
)
