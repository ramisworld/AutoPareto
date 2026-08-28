#!/usr/bin/env python3
"""Frozen post-search language-quality evaluation for selected checkpoints only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sqlite3
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_SUITE = ROOT / "evaluation" / "downstream_suite.json"
DEFAULT_DB = ROOT / "results" / "results.db"
DEFAULT_RESULTS = ROOT / "results" / "downstream_evaluations"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def standard_error(values: Iterable[float]) -> float | None:
    values = list(values)
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def bootstrap_ratio_interval(
    numerators: list[float],
    denominators: list[float],
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    if len(numerators) != len(denominators) or not numerators:
        raise ValueError("bootstrap requires equal non-empty samples")
    rng = random.Random(seed)
    count = len(numerators)
    estimates = []
    for _ in range(repetitions):
        indices = [rng.randrange(count) for _ in range(count)]
        denominator = sum(denominators[index] for index in indices)
        if denominator <= 0:
            raise ValueError("bootstrap denominator must be positive")
        estimates.append(sum(numerators[index] for index in indices) / denominator)
    estimates.sort()
    low_index = max(0, math.floor(0.025 * (repetitions - 1)))
    high_index = min(repetitions - 1, math.ceil(0.975 * (repetitions - 1)))
    return estimates[low_index], estimates[high_index]


def contiguous_causal_batch(tensor: Any) -> tuple[Any, Any]:
    """Return aligned causal inputs/targets with a dense memory layout."""
    if tensor.ndim != 2 or tensor.shape[1] < 2:
        raise ValueError("causal evaluation batch must be rank 2 with at least two tokens")
    inputs = tensor[:, :-1].contiguous()
    targets = tensor[:, 1:].contiguous()
    if inputs.shape != targets.shape:
        raise RuntimeError("causal input/target alignment failed")
    return inputs, targets


def validate_result_bundle(
    task_results: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    suite: dict[str, Any],
) -> None:
    """Reject incomplete or numerically impossible results before persistence."""
    expected_metrics = {
        ("heldout_climbmix_bpb", "bits_per_byte"),
        ("lambada_openai", "perplexity"),
        ("lambada_openai", "exact_match"),
        ("blimp", "macro_accuracy"),
        ("blimp", "micro_accuracy"),
        ("piqa", "accuracy"),
        ("piqa", "length_normalized_accuracy"),
        ("arc_easy", "accuracy"),
        ("arc_easy", "length_normalized_accuracy"),
    }
    actual_metrics = {(item["task_name"], item["metric_name"]) for item in task_results}
    if actual_metrics != expected_metrics or len(task_results) != len(expected_metrics):
        raise RuntimeError("downstream metric set is incomplete or duplicated")
    for result in task_results:
        value = result.get("metric_value")
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite metric: {result['task_name']}/{result['metric_name']}")
        if int(result.get("sample_count") or 0) <= 0:
            raise RuntimeError(f"empty metric sample: {result['task_name']}/{result['metric_name']}")
        metric = result["metric_name"]
        if (metric.endswith("accuracy") or metric == "exact_match") and not 0 <= value <= 1:
            raise RuntimeError(f"accuracy outside [0,1]: {result['task_name']}/{metric}")
        if metric in {"bits_per_byte", "perplexity"} and value <= 0:
            raise RuntimeError(f"non-positive metric: {result['task_name']}/{metric}")

    completion = suite["qualitative_completions"]
    expected_generations = {
        (prompt["id"], int(seed))
        for prompt in completion["prompts"] for seed in completion["seeds"]
    }
    actual_generations = {
        (item["prompt_id"], int(item["generation_seed"])) for item in generations
    }
    if actual_generations != expected_generations or len(generations) != len(expected_generations):
        raise RuntimeError("qualitative generation set is incomplete or duplicated")
    expected_count = int(completion["generated_tokens"])
    for generation in generations:
        if generation["generated_token_count"] != expected_count:
            raise RuntimeError(f"wrong generated token count: {generation['prompt_id']}")
        if len(generation["generated_token_ids"]) != expected_count:
            raise RuntimeError(f"generated token ids are incomplete: {generation['prompt_id']}")
        if not generation.get("cache_correctness_passed"):
            raise RuntimeError(f"cache correctness not established: {generation['prompt_id']}")


def load_suite(path: Path = DEFAULT_SUITE) -> dict[str, Any]:
    suite = json.loads(path.read_text())
    if suite.get("format") != "autopareto_downstream_quality_suite_v1":
        raise ValueError("unexpected downstream suite format")
    if suite.get("status") != "preregistered_before_seed0":
        raise ValueError("downstream suite is not preregistered")
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or {task.get("name") for task in tasks} != {
        "lambada_openai", "blimp", "piqa", "arc_easy",
    }:
        raise ValueError("downstream task set changed")
    for task in tasks:
        if not task.get("dataset_revision") or len(task["dataset_revision"]) != 40:
            raise ValueError(f"task dataset revision is not pinned: {task.get('name')}")
    heldout = suite.get("heldout_bpb", {})
    if len(heldout.get("dataset_revision", "")) != 40:
        raise ValueError("held-out BPB dataset revision is not pinned")
    return suite


class ModelScorer:
    """Batched causal-LM likelihood and deterministic completion helper."""

    def __init__(
        self,
        *,
        candidate: Path,
        checkpoint: Path,
        max_length: int,
        batch_size: int,
        module: Any | None = None,
    ):
        import torch

        from inference_benchmark import load_candidate_definitions
        from prepare import Tokenizer
        from protected.inference_adapter import (
            build_adapter,
            load_checkpoint_file,
            load_checkpoint_model,
        )

        self.torch = torch
        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("downstream evaluation requires CUDA")
        # Phase 1 validation and held-out BPB run in the same process. Reuse
        # the definitions module loaded by the validation evaluator when it is
        # provided. Loading the candidate twice recreates the dynamically
        # imported kernel package while its child modules remain in
        # ``sys.modules``; kernels 0.11 can then return a parent module without
        # ``flash_attn_interface`` on the second load.
        self.module = module if module is not None else load_candidate_definitions(candidate)
        self.checkpoint = load_checkpoint_file(checkpoint)
        self.model = load_checkpoint_model(self.module, self.checkpoint, self.device)
        self.model.eval()
        self.adapter = build_adapter(self.module, self.model)
        self.tokenizer = Tokenizer.from_directory()
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.bos = int(self.tokenizer.get_bos_token_id())

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def _encoded_request(self, context: str, continuation: str) -> dict[str, Any]:
        if not continuation:
            raise ValueError("continuation cannot be empty")
        if context:
            context_ids = self.encode(context)
            full = self.encode(context + continuation)
            context_length = len(context_ids)
        else:
            full = [self.bos, *self.encode(continuation)]
            context_length = 1
        continuation_length = len(full) - context_length
        if continuation_length <= 0:
            raise ValueError("continuation produced no separately scoreable tokens")
        if continuation_length > self.max_length:
            raise ValueError("continuation exceeds maximum evaluation context")
        if len(full) > self.max_length + 1:
            trim = len(full) - (self.max_length + 1)
            full = full[trim:]
            context_length = max(1, context_length - trim)
        start = context_length - 1
        return {
            "input": full[:-1],
            "target": full[1:],
            "start": start,
            "count": continuation_length,
        }

    def score(self, requests: list[tuple[str, str]]) -> list[dict[str, Any]]:
        torch = self.torch
        encoded = [self._encoded_request(*request) for request in requests]
        indexed = sorted(enumerate(encoded), key=lambda item: len(item[1]["input"]))
        output: list[dict[str, Any] | None] = [None] * len(encoded)
        autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        with torch.inference_mode(), autocast:
            for offset in range(0, len(indexed), self.batch_size):
                chunk = indexed[offset:offset + self.batch_size]
                width = max(len(item["input"]) for _, item in chunk)
                inputs = torch.full(
                    (len(chunk), width), self.bos, dtype=torch.long, device=self.device
                )
                for row_index, (_, item) in enumerate(chunk):
                    inputs[row_index, :len(item["input"])] = torch.tensor(
                        item["input"], dtype=torch.long, device=self.device
                    )
                logits = self.model(inputs)
                for row_index, (original_index, item) in enumerate(chunk):
                    start = int(item["start"])
                    count = int(item["count"])
                    selected = logits[row_index, start:start + count].float()
                    target = torch.tensor(
                        item["target"][start:start + count],
                        dtype=torch.long,
                        device=self.device,
                    )
                    log_probs = torch.log_softmax(selected, dim=-1)
                    token_log_probs = log_probs.gather(-1, target[:, None]).squeeze(-1)
                    output[original_index] = {
                        "log_likelihood": float(token_log_probs.sum().item()),
                        "token_count": count,
                        "greedy_match": bool(torch.equal(selected.argmax(dim=-1), target)),
                    }
        return [item for item in output if item is not None]

    def generate(
        self,
        prompt: str,
        *,
        count: int,
        temperature: float,
        top_k: int,
        seed: int,
    ) -> dict[str, Any]:
        torch = self.torch
        ids = self.tokenizer.encode(prompt, prepend=self.bos)
        prompt_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        generated = []
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode(), autocast:
            logits, state = self.adapter.prefill(prompt_tensor)
            step_logits = logits[:, -1, :]
            for index in range(count):
                scaled = step_logits.float() / temperature
                values, indices = torch.topk(scaled, min(top_k, scaled.shape[-1]), dim=-1)
                probabilities = torch.softmax(values, dim=-1)
                chosen = torch.multinomial(probabilities, 1, generator=generator)
                token = indices.gather(-1, chosen)
                generated.append(token)
                if index + 1 < count:
                    logits, state = self.adapter.decode_one(token, state)
                    step_logits = logits[:, -1, :]
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        token_ids = torch.cat(generated, dim=1)[0].tolist()
        completion = self.tokenizer.decode(token_ids)
        return {
            "completion_text": completion,
            "full_text": prompt + completion,
            "generated_token_ids": token_ids,
            "generated_token_count": len(token_ids),
            "generation_seconds": elapsed,
        }


def accuracy_result(
    *,
    task: dict[str, Any],
    metric_name: str,
    values: list[float],
    raw: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_name": task["name"],
        "dataset_name": task["dataset"],
        "dataset_revision": task["dataset_revision"],
        "dataset_config": task.get("dataset_config"),
        "split": task["split"],
        "metric_name": metric_name,
        "metric_value": statistics.mean(values),
        "metric_standard_error": standard_error(values),
        "confidence_low": None,
        "confidence_high": None,
        "sample_count": len(values),
        "higher_is_better": True,
        "raw": raw,
    }


def evaluate_lambada(scorer: ModelScorer, task: dict[str, Any], dataset: Any) -> list[dict[str, Any]]:
    requests = []
    for row in dataset:
        parts = row["text"].split(" ")
        requests.append((" ".join(parts[:-1]), " " + parts[-1]))
    scores = scorer.score(requests)
    total_log_likelihood = sum(item["log_likelihood"] for item in scores)
    total_tokens = sum(item["token_count"] for item in scores)
    exact = [float(item["greedy_match"]) for item in scores]
    common = {
        "dataset_name": task["dataset"],
        "dataset_revision": task["dataset_revision"],
        "dataset_config": task.get("dataset_config"),
        "split": task["split"],
        "sample_count": len(scores),
        "task_name": task["name"],
    }
    return [
        {
            **common,
            "metric_name": "perplexity",
            "metric_value": math.exp(-total_log_likelihood / total_tokens),
            "metric_standard_error": None,
            "confidence_low": None,
            "confidence_high": None,
            "higher_is_better": False,
            "raw": {"total_log_likelihood": total_log_likelihood, "total_tokens": total_tokens},
        },
        {
            **common,
            "metric_name": "exact_match",
            "metric_value": statistics.mean(exact),
            "metric_standard_error": standard_error(exact),
            "confidence_low": None,
            "confidence_high": None,
            "higher_is_better": True,
            "raw": {"correct": int(sum(exact))},
        },
    ]


def evaluate_choice_task(
    scorer: ModelScorer,
    task: dict[str, Any],
    dataset: Any,
    *,
    row_to_context_choices_label: Any,
) -> list[dict[str, Any]]:
    raw_accuracy: list[float] = []
    normalized_accuracy: list[float] = []
    rows = []
    requests = []
    for row in dataset:
        context, choices, label = row_to_context_choices_label(row)
        start = len(requests)
        requests.extend((context, " " + choice) for choice in choices)
        rows.append((start, len(choices), label))
    all_scores = scorer.score(requests)
    for start, choice_count, label in rows:
        scores = all_scores[start:start + choice_count]
        raw_prediction = max(range(len(scores)), key=lambda i: scores[i]["log_likelihood"])
        normalized_prediction = max(
            range(len(scores)),
            key=lambda i: scores[i]["log_likelihood"] / max(1, scores[i]["token_count"]),
        )
        raw_accuracy.append(float(raw_prediction == label))
        normalized_accuracy.append(float(normalized_prediction == label))
    return [
        accuracy_result(
            task=task, metric_name="accuracy", values=raw_accuracy,
            raw={"correct": int(sum(raw_accuracy))},
        ),
        accuracy_result(
            task=task, metric_name="length_normalized_accuracy",
            values=normalized_accuracy,
            raw={"correct": int(sum(normalized_accuracy))},
        ),
    ]


def evaluate_blimp(
    scorer: ModelScorer,
    task: dict[str, Any],
    *,
    datasets_module: Any,
) -> list[dict[str, Any]]:
    names = datasets_module.get_dataset_config_names(
        task["dataset"], revision=task["dataset_revision"]
    )
    per_config = {}
    all_values = []
    for name in sorted(names):
        dataset = datasets_module.load_dataset(
            task["dataset"], name, revision=task["dataset_revision"], split=task["split"]
        )
        requests = [
            request
            for row in dataset
            for request in (("", row["sentence_good"]), ("", row["sentence_bad"]))
        ]
        scores = scorer.score(requests)
        values = []
        for index in range(0, len(scores), 2):
            good, bad = scores[index:index + 2]
            values.append(float(good["log_likelihood"] > bad["log_likelihood"]))
        per_config[name] = {
            "accuracy": statistics.mean(values),
            "correct": int(sum(values)),
            "count": len(values),
        }
        all_values.extend(values)
    macro_values = [item["accuracy"] for item in per_config.values()]
    common = {
        "task_name": task["name"],
        "dataset_name": task["dataset"],
        "dataset_revision": task["dataset_revision"],
        "dataset_config": "all",
        "split": task["split"],
        "confidence_low": None,
        "confidence_high": None,
        "higher_is_better": True,
        "raw": {"configurations": per_config},
    }
    return [
        {
            **common,
            "metric_name": "macro_accuracy",
            "metric_value": statistics.mean(macro_values),
            "metric_standard_error": standard_error(macro_values),
            "sample_count": len(macro_values),
        },
        {
            **common,
            "metric_name": "micro_accuracy",
            "metric_value": statistics.mean(all_values),
            "metric_standard_error": standard_error(all_values),
            "sample_count": len(all_values),
        },
    ]


def download_heldout_shard(spec: dict[str, Any]) -> Path:
    import requests

    cache = Path.home() / ".cache" / "autopareto" / "downstream"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{spec['dataset_revision']}-{spec['filename']}"
    if destination.is_file():
        return destination
    url = (
        f"https://huggingface.co/datasets/{spec['dataset']}/resolve/"
        f"{spec['dataset_revision']}/{spec['filename']}"
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if chunk:
                    stream.write(chunk)
    temporary.replace(destination)
    return destination


def heldout_segments(
    scorer: ModelScorer,
    parquet_path: Path,
    *,
    token_count: int,
    sequence_length: int,
) -> list[list[int]]:
    import pyarrow.parquet as pq

    buffer: list[int] = []
    segments = []
    parquet = pq.ParquetFile(parquet_path)
    for row_group in range(parquet.num_row_groups):
        texts = parquet.read_row_group(row_group, columns=["text"]).column("text").to_pylist()
        for text in texts:
            buffer.extend(scorer.tokenizer.encode(text, prepend=scorer.bos))
            while len(buffer) >= sequence_length + 1 and len(segments) * sequence_length < token_count:
                segments.append(buffer[:sequence_length + 1])
                del buffer[:sequence_length + 1]
            if len(segments) * sequence_length >= token_count:
                return segments
    raise RuntimeError("held-out shard did not contain enough evaluation tokens")


def evaluate_heldout_bpb(
    scorer: ModelScorer,
    spec: dict[str, Any],
    *,
    random_seed: int,
) -> dict[str, Any]:
    from prepare import get_token_bytes

    torch = scorer.torch
    shard = download_heldout_shard(spec)
    segments = heldout_segments(
        scorer,
        shard,
        token_count=int(spec["evaluated_tokens"]),
        sequence_length=scorer.max_length,
    )
    token_bytes = get_token_bytes(device=scorer.device)
    nats_by_segment = []
    bytes_by_segment = []
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), autocast:
        for offset in range(0, len(segments), scorer.batch_size):
            chunk = segments[offset:offset + scorer.batch_size]
            tensor = torch.tensor(chunk, dtype=torch.long, device=scorer.device)
            inputs, targets = contiguous_causal_batch(tensor)
            losses = scorer.model(inputs, targets, reduction="none").reshape(targets.shape)
            if losses.shape != targets.shape or not bool(torch.isfinite(losses).all().item()):
                raise RuntimeError("held-out loss tensor is malformed or non-finite")
            for row in range(targets.shape[0]):
                sizes = token_bytes[targets[row]]
                mask = sizes > 0
                nats_by_segment.append(float((losses[row] * mask).sum().item()))
                bytes_by_segment.append(float(sizes.sum().item()))
    denominator = math.log(2) * sum(bytes_by_segment)
    value = sum(nats_by_segment) / denominator
    low, high = bootstrap_ratio_interval(
        nats_by_segment,
        [math.log(2) * item for item in bytes_by_segment],
        repetitions=int(spec["bootstrap_repetitions"]),
        seed=random_seed,
    )
    return {
        "task_name": "heldout_climbmix_bpb",
        "dataset_name": spec["dataset"],
        "dataset_revision": spec["dataset_revision"],
        "dataset_config": spec["filename"],
        "split": "heldout",
        "metric_name": "bits_per_byte",
        "metric_value": value,
        "metric_standard_error": None,
        "confidence_low": low,
        "confidence_high": high,
        "sample_count": len(segments),
        "higher_is_better": False,
        "raw": {
            "evaluated_tokens": len(segments) * scorer.max_length,
            "shard_sha256": sha256(shard),
            "segments": len(segments),
        },
    }


def load_task_dataset(task: dict[str, Any], datasets_module: Any) -> Any:
    return datasets_module.load_dataset(
        task["dataset"],
        task.get("dataset_config"),
        revision=task["dataset_revision"],
        split=task["split"],
    )


def run_tasks(scorer: ModelScorer, suite: dict[str, Any]) -> list[dict[str, Any]]:
    import datasets

    results = [
        evaluate_heldout_bpb(
            scorer, suite["heldout_bpb"], random_seed=int(suite["runtime"]["random_seed"])
        )
    ]
    for task in suite["tasks"]:
        if task["name"] == "blimp":
            results.extend(evaluate_blimp(scorer, task, datasets_module=datasets))
            continue
        dataset = load_task_dataset(task, datasets)
        if task["name"] == "lambada_openai":
            results.extend(evaluate_lambada(scorer, task, dataset))
        elif task["name"] == "piqa":
            results.extend(evaluate_choice_task(
                scorer, task, dataset,
                row_to_context_choices_label=lambda row: (
                    f"Question: {row['goal']}\nAnswer:",
                    [row["sol1"], row["sol2"]],
                    int(row["label"]),
                ),
            ))
        elif task["name"] == "arc_easy":
            def arc_row(row: dict[str, Any]) -> tuple[str, list[str], int]:
                labels = list(row["choices"]["label"])
                return (
                    f"Question: {row['question']}\nAnswer:",
                    list(row["choices"]["text"]),
                    labels.index(row["answerKey"]),
                )
            results.extend(evaluate_choice_task(
                scorer, task, dataset, row_to_context_choices_label=arc_row,
            ))
        else:
            raise RuntimeError(f"unsupported downstream task: {task['name']}")
    return results


def environment_record(suite: dict[str, Any]) -> dict[str, Any]:
    import datasets
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "datasets": datasets.__version__,
        "suite_runtime": suite["runtime"],
    }


def validate_paths(
    db: sqlite3.Connection,
    experiment: sqlite3.Row,
    candidate: Path,
    checkpoint: Path,
) -> None:
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if experiment["status"] != "completed":
        raise RuntimeError("downstream evaluation requires a completed result")
    if (
        experiment["training_status"] == "completed"
        and experiment["inference_status"] == "compatible"
    ):
        return
    # The earliest baseline rows predate the normalized status columns. Accept only
    # a baseline with completed, protected deployment correctness evidence; never
    # rewrite the historical row to make it look newer than it is.
    if experiment["method"] == "baseline":
        evidence = db.execute(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN correctness_passed=1 THEN 1 ELSE 0 END) AS passed,
                      SUM(CASE WHEN correctness_passed=0 THEN 1 ELSE 0 END) AS failed
               FROM deployment_measurements WHERE experiment_id=? AND is_warmup=0""",
            (experiment["id"],),
        ).fetchone()
        if evidence and evidence["total"] and evidence["passed"] == evidence["total"] and not evidence["failed"]:
            return
    raise RuntimeError(
        "downstream evaluation requires completed training and compatible cached "
        "inference, or protected legacy-baseline correctness evidence"
    )


def store_result(
    db: sqlite3.Connection,
    downstream_run_id: int,
    result: dict[str, Any],
) -> None:
    db.execute(
        """INSERT INTO downstream_task_results
           (downstream_run_id, task_name, dataset_name, dataset_revision,
            dataset_config, split, metric_name, metric_value,
            metric_standard_error, confidence_low, confidence_high, sample_count,
            higher_is_better, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            downstream_run_id, result["task_name"], result["dataset_name"],
            result["dataset_revision"], result.get("dataset_config"), result["split"],
            result["metric_name"], result.get("metric_value"),
            result.get("metric_standard_error"), result.get("confidence_low"),
            result.get("confidence_high"), result.get("sample_count"),
            int(bool(result["higher_is_better"])),
            json.dumps(result.get("raw", {}), sort_keys=True),
        ),
    )


def validate_only() -> None:
    suite = load_suite()
    if suite["runtime"]["datasets_package"] != "5.0.1":
        raise RuntimeError("datasets package pin changed")
    from autopareto_db import migrate

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute(
        """CREATE TABLE experiments (
            id INTEGER PRIMARY KEY, campaign_id TEXT, experiment_number INTEGER,
            method TEXT, training_seed INTEGER, status TEXT, description TEXT,
            git_commit TEXT, code_diff TEXT, val_bpb REAL,
            inference_tokens_per_second REAL, peak_inference_cuda_memory_mb REAL,
            training_tokens_per_second REAL, total_training_tokens INTEGER,
            peak_training_cuda_memory_mb REAL, parameter_count INTEGER,
            training_seconds REAL, inference_seconds REAL, total_runtime_seconds REAL,
            checkpoint_path TEXT, sample_path TEXT, log_path TEXT,
            crash_or_failure_reason TEXT, protected_manifest_sha256 TEXT,
            started_at TEXT, finished_at TEXT, phase TEXT, valid_result INTEGER,
            exclude_from_analysis INTEGER, UNIQUE(campaign_id, experiment_number)
        )"""
    )
    db.execute(
        """CREATE TABLE campaign_state (
            campaign_id TEXT PRIMARY KEY, method TEXT, status TEXT,
            attempt_limit INTEGER, next_experiment_number INTEGER
        )"""
    )
    migrate(db)
    expected = {
        "downstream_evaluation_runs",
        "downstream_task_results",
        "downstream_generations",
    }
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not expected <= tables:
        raise RuntimeError("downstream database migration is incomplete")
    low, high = bootstrap_ratio_interval([1.0, 2.0], [2.0, 4.0], repetitions=100, seed=1)
    if (low, high) != (0.5, 0.5):
        raise RuntimeError("deterministic ratio bootstrap failed")
    print("downstream evaluator validation: ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--selection-role")
    parser.add_argument("--display-name")
    parser.add_argument("--selection-sha256")
    args = parser.parse_args()
    if args.validate_only:
        validate_only()
        return 0
    if args.experiment_id is None or args.candidate is None or args.checkpoint is None:
        parser.error("official evaluation requires --experiment-id, --candidate, and --checkpoint")

    from autopareto_db import connect

    suite = load_suite(args.suite)
    db = connect(args.db)
    experiment = db.execute(
        """SELECT id, campaign_id, experiment_number, method, status, training_status,
                  inference_status FROM experiments WHERE id=?""",
        (args.experiment_id,),
    ).fetchone()
    if experiment is None:
        raise RuntimeError(f"unknown experiment: {args.experiment_id}")
    candidate = args.candidate.resolve()
    checkpoint = args.checkpoint.resolve()
    validate_paths(db, experiment, candidate, checkpoint)
    if args.smoke:
        scorer = ModelScorer(
            candidate=candidate,
            checkpoint=checkpoint,
            max_length=min(64, int(suite["runtime"]["max_sequence_tokens"])),
            batch_size=1,
        )
        score = scorer.score([("The sky appears", " blue")])[0]
        generation = scorer.generate(
            "The scientist observed that",
            count=4,
            temperature=float(suite["qualitative_completions"]["temperature"]),
            top_k=int(suite["qualitative_completions"]["top_k"]),
            seed=int(suite["runtime"]["random_seed"]),
        )
        print(json.dumps({
            "status": "smoke_ok",
            "experiment_id": args.experiment_id,
            "score": score,
            "generation": generation,
        }, sort_keys=True))
        db.close()
        return 0
    suite_hash = sha256(args.suite)
    evaluator_hash = sha256(Path(__file__).resolve())
    checkpoint_hash = sha256(checkpoint)
    candidate_hash = sha256(candidate)
    environment = environment_record(suite)
    started_at = utc_now()
    cursor = db.execute(
        """INSERT INTO downstream_evaluation_runs
           (experiment_id, checkpoint_path, checkpoint_sha256, candidate_path,
            candidate_sha256, suite_path, suite_sha256, evaluator_git_commit,
            evaluator_sha256, selection_role, display_name, selection_sha256,
            environment_json, status, started_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
        (
            args.experiment_id, str(checkpoint), checkpoint_hash, str(candidate),
            candidate_hash, str(args.suite.resolve()), suite_hash, git_commit(),
            evaluator_hash, args.selection_role, args.display_name,
            args.selection_sha256, json.dumps(environment, sort_keys=True), started_at,
        ),
    )
    downstream_run_id = int(cursor.lastrowid)
    db.commit()
    output_dir = args.output_dir / f"{downstream_run_id:04d}"
    output_path = output_dir / "results.json"
    try:
        scorer = ModelScorer(
            candidate=candidate,
            checkpoint=checkpoint,
            max_length=int(suite["runtime"]["max_sequence_tokens"]),
            batch_size=int(suite["runtime"]["scoring_batch_size"]),
        )
        task_results = run_tasks(scorer, suite)
        generations = []
        completion = suite["qualitative_completions"]
        for prompt in completion["prompts"]:
            for seed in completion["seeds"]:
                generated = scorer.generate(
                    prompt["text"],
                    count=int(completion["generated_tokens"]),
                    temperature=float(completion["temperature"]),
                    top_k=int(completion["top_k"]),
                    seed=int(seed),
                )
                generations.append({
                    "prompt_id": prompt["id"],
                    "prompt_text": prompt["text"],
                    "generation_seed": int(seed),
                    "generated_text": generated["completion_text"],
                    "full_text": generated["full_text"],
                    "generated_token_ids": generated["generated_token_ids"],
                    "generated_token_count": generated["generated_token_count"],
                    "generation_seconds": generated["generation_seconds"],
                    "cache_correctness_passed": True,
                })
        validate_result_bundle(task_results, generations, suite)
        payload = {
            "format": "autopareto_downstream_result_v1",
            "downstream_run_id": downstream_run_id,
            "experiment_id": args.experiment_id,
            "suite_sha256": suite_hash,
            "evaluator_sha256": evaluator_hash,
            "checkpoint_sha256": checkpoint_hash,
            "candidate_sha256": candidate_hash,
            "environment": environment,
            "task_results": task_results,
            "generations": generations,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_suffix(".json.tmp")
        temporary_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if json.loads(temporary_output.read_text()) != payload:
            raise RuntimeError("raw downstream result JSON failed round-trip validation")
        db.execute("BEGIN")
        for result in task_results:
            store_result(db, downstream_run_id, result)
        settings = json.dumps({
            "generated_tokens": completion["generated_tokens"],
            "temperature": completion["temperature"],
            "top_k": completion["top_k"],
            "cache_correctness_basis": "protected_experiment_compatibility_gate",
        }, sort_keys=True)
        for generation in generations:
            db.execute(
                """INSERT INTO downstream_generations
                   (downstream_run_id, prompt_id, prompt_text, generation_seed,
                    generated_text, generated_token_count, generation_seconds,
                    cache_correctness_passed, settings_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    downstream_run_id, generation["prompt_id"], generation["prompt_text"],
                    generation["generation_seed"], generation["generated_text"],
                    generation["generated_token_count"], generation["generation_seconds"],
                    int(generation["cache_correctness_passed"]), settings,
                ),
            )
        temporary_output.replace(output_path)
        db.execute(
            """UPDATE downstream_evaluation_runs
               SET status='completed', raw_result_path=?, finished_at=?
               WHERE id=?""",
            (str(output_path), utc_now(), downstream_run_id),
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        temporary_output = output_path.with_suffix(".json.tmp")
        if temporary_output.exists():
            temporary_output.unlink()
        if output_path.exists():
            output_path.unlink()
        db.execute(
            """UPDATE downstream_evaluation_runs
               SET status='failed', failure_reason=?, finished_at=? WHERE id=?""",
            (str(exc), utc_now(), downstream_run_id),
        )
        db.commit()
        raise
    finally:
        db.close()
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
