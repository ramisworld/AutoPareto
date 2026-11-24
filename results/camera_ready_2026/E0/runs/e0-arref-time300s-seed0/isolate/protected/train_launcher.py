#!/usr/bin/env python3
"""Protected wrapper: run an unchanged candidate train.py and save its checkpoint."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import queue
import re
import runpy
import sys
import threading
import time
import traceback

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from event_log import append_event, make_event
from benchmark_config import TRAINING_BUDGET_SECONDS, TRAINING_MAX_OVERRUN_SECONDS
from protected.inference_adapter import save_training_checkpoint


TRAINING_LINE = re.compile(
    r"step\s+(?P<step>\d+)\s+\((?P<percent>[\d.]+)%\).*?loss:\s*(?P<loss>[\d.eE+-]+)"
    r".*?lrm:\s*(?P<lrm>[\d.eE+-]+).*?tok/sec:\s*(?P<tps>[\d,]+).*?remaining:\s*(?P<remaining>[\d.]+)s"
)


def _integer_expression(node: ast.AST) -> int:
    """Evaluate the small integer-only expressions used by train.py constants."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Pow)):
        left, right = _integer_expression(node.left), _integer_expression(node.right)
        return {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.FloorDiv: lambda: left // right,
            ast.Pow: lambda: left ** right,
        }[type(node.op)]()
    raise ValueError("constant is not a safe integer expression")


def read_integer_constant(path: Path, name: str) -> int:
    for statement in ast.parse(path.read_text()).body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
            value = _integer_expression(statement.value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            return value
    raise ValueError(f"missing integer constant {name}")


class TelemetryStream:
    """Forward stdout unchanged and asynchronously emit sparse structured events."""

    def __init__(self, wrapped, event_path: Path, campaign_id: str | None, experiment_id: int | None, training_run_id: int | None, tokens_per_step: int):
        self.wrapped = wrapped
        self.event_path = event_path
        self.campaign_id = campaign_id
        self.experiment_id = experiment_id
        self.training_run_id = training_run_id
        self.tokens_per_step = tokens_per_step
        self.last_step = -1
        self.latest_payload = None
        self.events: queue.Queue[dict | None] = queue.Queue()
        self.worker = threading.Thread(target=self._write_events, daemon=True)
        self.worker.start()

    def _write_events(self):
        while True:
            event = self.events.get()
            if event is None:
                return
            append_event(self.event_path, event)

    def write(self, text):
        result = self.wrapped.write(text)
        match = TRAINING_LINE.search(text)
        if match:
            step = int(match.group("step"))
            memory = peak_memory = None
            if torch.cuda.is_available():
                memory = torch.cuda.memory_allocated() / 1024 / 1024
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.latest_payload = {
                "step": step,
                "percent_complete": float(match.group("percent")),
                # train.py reports a debiased EMA, not the instantaneous minibatch loss.
                "training_loss": float(match.group("loss")),
                "trainer_smoothed_loss": float(match.group("loss")),
                "loss_kind": "trainer_debiased_ema",
                "learning_rate_multiplier": float(match.group("lrm")),
                "tokens_per_second": int(match.group("tps").replace(",", "")),
                "total_tokens": (step + 1) * self.tokens_per_step,
                "training_gpu_memory_mb": memory,
                "peak_training_gpu_memory_mb": peak_memory,
                "gradient_norm": None,
                "remaining_seconds": float(match.group("remaining")),
            }
            if step % 5 == 0 and step != self.last_step:
                self._emit_latest()
        return result

    def _emit_latest(self, *, final: bool = False):
        if self.latest_payload is None or self.latest_payload["step"] == self.last_step:
            return
        payload = {**self.latest_payload, "is_final": final}
        self.last_step = payload["step"]
        self.events.put(make_event(
            "training_progress", campaign_id=self.campaign_id,
            experiment_id=self.experiment_id, training_run_id=self.training_run_id,
            status="running", payload=payload,
        ))

    def flush(self):
        return self.wrapped.flush()

    def isatty(self):
        return self.wrapped.isatty()

    def close_events(self):
        self._emit_latest(final=True)
        self.events.put(None)
        self.worker.join(timeout=5)


def _force_seed(seed: int):
    original_manual_seed = torch.manual_seed
    original_cuda_manual_seed = torch.cuda.manual_seed
    original_cuda_manual_seed_all = torch.cuda.manual_seed_all

    def manual_seed(_ignored):
        return original_manual_seed(seed)

    def cuda_manual_seed(_ignored):
        return original_cuda_manual_seed(seed)

    def cuda_manual_seed_all(_ignored):
        return original_cuda_manual_seed_all(seed)

    torch.manual_seed = manual_seed
    torch.cuda.manual_seed = cuda_manual_seed
    torch.cuda.manual_seed_all = cuda_manual_seed_all
    return original_manual_seed, original_cuda_manual_seed, original_cuda_manual_seed_all


def validate_training_contract(
    namespace: dict,
    protected_values: list[float],
    protected_models: list[object] | None = None,
) -> dict:
    """Validate candidate outputs against protected evaluation and timing semantics."""
    required = {
        "model", "config", "total_tokens", "total_training_time", "step",
        "TOTAL_BATCH_SIZE", "peak_vram_mb", "num_params", "val_bpb",
    }
    missing = sorted(required - namespace.keys())
    if missing:
        raise RuntimeError(
            f"candidate omitted required training outputs: {', '.join(missing)}"
        )
    if len(protected_values) != 1:
        raise RuntimeError(
            "candidate must call the protected validation evaluator exactly once"
        )
    if protected_models is not None and (
        len(protected_models) != 1 or protected_models[0] is not namespace["model"]
    ):
        raise RuntimeError(
            "protected validation must evaluate the checkpointed candidate model"
        )
    protected_val_bpb = float(protected_values[0])
    reported_val_bpb = float(namespace["val_bpb"])
    if abs(reported_val_bpb - protected_val_bpb) > 1e-12:
        raise RuntimeError(
            "candidate val_bpb does not match the protected evaluator result"
        )
    training_seconds = float(namespace["total_training_time"])
    minimum = float(TRAINING_BUDGET_SECONDS)
    maximum = minimum + float(TRAINING_MAX_OVERRUN_SECONDS)
    if not minimum <= training_seconds <= maximum:
        raise RuntimeError(
            f"candidate measured training time {training_seconds:.3f}s is outside "
            f"the protected [{minimum:.0f}, {maximum:.0f}]s window"
        )
    return {
        "val_bpb": protected_val_bpb,
        "training_seconds": training_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--training-run-id", type=int)
    args = parser.parse_args()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    candidate_root = str(args.train.resolve().parent)
    if candidate_root not in sys.path:
        sys.path.insert(0, candidate_root)

    started = time.time()
    originals = _force_seed(args.seed)
    protected_prepare = importlib.import_module("prepare")
    if float(protected_prepare.TIME_BUDGET) != float(TRAINING_BUDGET_SECONDS):
        raise RuntimeError("protected training budget constants disagree")
    protected_values: list[float] = []
    protected_models: list[object] = []
    original_evaluate_bpb = protected_prepare.evaluate_bpb

    def captured_protected_evaluation(model, tokenizer, batch_size):
        value = float(original_evaluate_bpb(model, tokenizer, batch_size))
        protected_models.append(model)
        protected_values.append(value)
        return value

    protected_prepare.evaluate_bpb = captured_protected_evaluation
    original_stdout = sys.stdout
    telemetry = None
    if args.event_log:
        tokens_per_step = read_integer_constant(args.train, "TOTAL_BATCH_SIZE")
        telemetry = TelemetryStream(
            original_stdout, args.event_log, args.campaign_id,
            args.experiment_id, args.training_run_id, tokens_per_step,
        )
        sys.stdout = telemetry
    try:
        namespace = runpy.run_path(str(args.train), run_name="__main__")
        contract = validate_training_contract(
            namespace, protected_values, protected_models
        )
        model = namespace["model"]
        config = namespace["config"]
        save_training_checkpoint(args.checkpoint, model, config, args.seed, args.train.name)
        total_tokens = int(namespace["total_tokens"])
        training_seconds = contract["training_seconds"]
        # total_training_time excludes compilation/warm-up steps 0..10. Throughput
        # must therefore use only steps 11 onward, while total tokens remains total.
        measured_training_tokens = max(0, (int(namespace["step"]) - 11) * int(namespace["TOTAL_BATCH_SIZE"]))
        metrics = {
            "val_bpb": contract["val_bpb"],
            "training_seconds": training_seconds,
            "training_tokens_per_second": measured_training_tokens / training_seconds,
            "total_training_tokens": total_tokens,
            "measured_training_tokens": measured_training_tokens,
            "peak_training_cuda_memory_mb": float(namespace["peak_vram_mb"]),
            "parameter_count": int(namespace["num_params"]),
            "training_seed": args.seed,
            "total_runtime_seconds": time.time() - started,
            "checkpoint": str(args.checkpoint),
        }
        args.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        return 0
    except BaseException as exc:
        failure = {
            "training_seed": args.seed,
            "total_runtime_seconds": time.time() - started,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        args.metrics.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        protected_prepare.evaluate_bpb = original_evaluate_bpb
        sys.stdout = original_stdout
        if telemetry:
            telemetry.close_events()
        torch.manual_seed, torch.cuda.manual_seed, torch.cuda.manual_seed_all = originals


if __name__ == "__main__":
    raise SystemExit(main())
