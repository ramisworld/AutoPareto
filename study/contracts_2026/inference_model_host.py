"""Disposable untrusted model host for protected inference IPC.

This process executes candidate model definitions but contains no evaluator,
benchmark policy, Pareto arithmetic, campaign history, or trusted acceptance
state.  It receives only explicit RPC requests from its parent and returns model
observations.  The parent remains authoritative for all reported measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import types
from typing import Any

from .inference_ipc import IPC_PROTOCOL, MAX_MESSAGE_BYTES


SETUP_MARKER = "# Setup: tokenizer, model, optimizer, dataloader"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _load_definitions(candidate: pathlib.Path) -> types.ModuleType:
    source = candidate.read_text()
    if source.count(SETUP_MARKER) != 1:
        raise RuntimeError(f"candidate train.py removed required marker: {SETUP_MARKER}")
    module = types.ModuleType("autopareto_candidate_model_host")
    module.__file__ = str(candidate)
    sys.modules[module.__name__] = module
    definitions = source.split(SETUP_MARKER, 1)[0]
    exec(compile(definitions, str(candidate), "exec"), module.__dict__)
    return module


def _load_checkpoint(path: pathlib.Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError("checkpoint payload is not a protected state_dict checkpoint")
    return payload


class ModelHost:
    def __init__(self, *, context: pathlib.Path, candidate: str, checkpoint: pathlib.Path, precision: str) -> None:
        import torch
        from protected.inference_adapter import build_adapter, load_checkpoint_model

        if precision not in {"bf16", "fp32"}:
            raise RuntimeError("unsupported inference precision")
        self.torch = torch
        # The A40/PyTorch SDPA dispatcher may select the FlashAttention backend
        # for an FP32 correctness adjudication even though that backend only
        # accepts fp16/bf16/fp8.  The registered FP32 pass is a correctness
        # reference, not the measured BF16 workload; force the mathematically
        # equivalent SDPA path only in this disposable FP32 host.
        if precision == "fp32":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        self.context = context.resolve(strict=True)
        self.candidate = (self.context / candidate).resolve(strict=True)
        if self.candidate.parent != self.context or self.candidate.name != "train.py":
            raise RuntimeError("model host candidate must be the regular context train.py")
        self.checkpoint = checkpoint.resolve(strict=True)
        self.precision = precision
        self.module = _load_definitions(self.candidate)
        payload = _load_checkpoint(self.checkpoint)
        if payload.get("source_sha256") != _sha256(self.candidate):
            raise RuntimeError("model host checkpoint/source pairing failed")
        if payload.get("config_sha256") != _json_hash(payload.get("config", {})):
            raise RuntimeError("model host checkpoint/config pairing failed")
        device = torch.device("cuda")
        self.model = load_checkpoint_model(self.module, payload, device)
        if precision == "fp32":
            self.model = self.model.float()
        self.model.eval()
        self.adapter = build_adapter(self.module, self.model)
        self.states: dict[int, Any] = {}
        self.next_state_id = 1

    def _tensor(self, values: Any) -> Any:
        if not isinstance(values, list) or not values or not isinstance(values[0], list):
            raise RuntimeError("input_ids must be a non-empty nested list")
        if len(values) != 1 or not values[0]:
            raise RuntimeError("protected inference requires batch size 1")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values[0]):
            raise RuntimeError("input_ids must contain integer token IDs")
        return self.torch.tensor(values, dtype=self.torch.long, device="cuda")

    def _validation_tensors(self, input_values: Any, target_values: Any) -> tuple[Any, Any]:
        """Build one protected validation batch from parent-supplied token IDs."""
        if (
            not isinstance(input_values, list)
            or not input_values
            or not isinstance(target_values, list)
            or len(input_values) != len(target_values)
            or not isinstance(input_values[0], list)
            or not isinstance(target_values[0], list)
        ):
            raise RuntimeError("validation inputs must be matching non-empty nested lists")
        batch = len(input_values)
        length = len(input_values[0])
        if batch <= 0 or length != 2048:
            raise RuntimeError("protected validation requires [B, 2048] batches")
        if any(not isinstance(row, list) or len(row) != length for row in input_values):
            raise RuntimeError("validation input rows have inconsistent lengths")
        if any(not isinstance(row, list) or len(row) != length for row in target_values):
            raise RuntimeError("validation target rows have inconsistent lengths")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for row in [*input_values, *target_values]
            for value in row
        ):
            raise RuntimeError("validation token IDs must be integers")
        device = self.torch.device("cuda")
        return (
            self.torch.tensor(input_values, dtype=self.torch.long, device=device),
            self.torch.tensor(target_values, dtype=self.torch.long, device=device),
        )

    def _logits(self, logits: Any) -> list[float]:
        if not hasattr(logits, "shape") or len(logits.shape) != 3 or int(logits.shape[0]) != 1 or int(logits.shape[1]) < 1:
            raise RuntimeError("candidate inference output must have shape [1, sequence, vocab]")
        values = logits[:, -1, :].detach().float().cpu().tolist()[0]
        if not values or any(not math.isfinite(float(value)) for value in values):
            raise RuntimeError("candidate inference output is non-finite")
        return [float(value) for value in values]

    def _state(self, value: Any) -> int:
        state_id = self.next_state_id
        self.next_state_id += 1
        self.states[state_id] = value
        return state_id

    def _get_state(self, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value not in self.states:
            raise RuntimeError("unknown model-host decode state")
        return self.states[value]

    def _sync(self) -> None:
        self.torch.cuda.synchronize()

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("operation")
        if operation == "hello":
            return {
                "source_sha256": _sha256(self.candidate),
                "checkpoint_sha256": _sha256(self.checkpoint),
                "precision": self.precision,
                "device": str(next(self.model.parameters()).device),
                "candidate_module": self.module.__name__,
                "parameter_bytes": sum(
                    int(parameter.numel()) * int(parameter.element_size())
                    for parameter in self.model.parameters()
                ),
            }
        if operation == "prefill":
            input_ids = self._tensor(request.get("input_ids"))
            with self.torch.inference_mode(), self.torch.amp.autocast(
                device_type="cuda", dtype=self.torch.bfloat16, enabled=self.precision == "bf16"
            ):
                logits, state = self.adapter.prefill(input_ids)
            self._sync()
            state_id = self._state(state)
            values = self._logits(logits)
            return {
                "logits": values,
                "next_token": int(max(range(len(values)), key=values.__getitem__)),
                "state_id": state_id,
                "cache_nbytes": int(self.adapter.cache_nbytes(state)),
            }
        if operation == "decode":
            state = self._get_state(request.get("state_id"))
            token = self._tensor(request.get("token_ids"))
            with self.torch.inference_mode(), self.torch.amp.autocast(
                device_type="cuda", dtype=self.torch.bfloat16, enabled=self.precision == "bf16"
            ):
                logits, new_state = self.adapter.decode_one(token, state)
            self._sync()
            state_id = self._state(new_state)
            values = self._logits(logits)
            return {
                "logits": values,
                "next_token": int(max(range(len(values)), key=values.__getitem__)),
                "state_id": state_id,
                "cache_nbytes": int(self.adapter.cache_nbytes(new_state)),
            }
        if operation == "decode_sequence":
            source_state_id = request.get("state_id")
            state = self._get_state(source_state_id)
            token_ids = request.get("token_ids")
            transitions = request.get("transitions")
            if not isinstance(token_ids, list) or len(token_ids) != 1:
                raise RuntimeError("decode_sequence token_ids must be one token")
            if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions != 127:
                raise RuntimeError("protected decode sequence must contain exactly 127 transitions")
            token = self._tensor(token_ids)
            start_event = self.torch.cuda.Event(enable_timing=True)
            end_event = self.torch.cuda.Event(enable_timing=True)
            self._sync()
            start_event.record()
            final_token = None
            with self.torch.inference_mode(), self.torch.amp.autocast(
                device_type="cuda", dtype=self.torch.bfloat16, enabled=self.precision == "bf16"
            ):
                for _ in range(transitions):
                    logits, state = self.adapter.decode_one(token, state)
                    token = self.torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            end_event.record()
            self._sync()
            elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
            cache_nbytes = int(self.adapter.cache_nbytes(state))
            # Completed benchmark requests never reuse their decode state.
            # Retaining every state made the disposable host grow GPU memory
            # linearly across warm-up and fixed-duration blocks.
            self.states.pop(source_state_id, None)
            return {
                "transitions": transitions,
                "final_token": int(token.item()),
                "state_id": int(source_state_id),
                "cache_nbytes": cache_nbytes,
                "cuda_stream_step_seconds": elapsed,
                "state_released": True,
            }
        if operation == "reference":
            input_ids = self._tensor(request.get("input_ids"))
            with self.torch.inference_mode(), self.torch.amp.autocast(
                device_type="cuda", dtype=self.torch.bfloat16, enabled=self.precision == "bf16"
            ):
                if self.precision == "fp32":
                    # FP32 adjudication must use the protected SDPA reference.
                    # Calling the candidate forward here would send FP32 Q/K/V
                    # into candidate FlashAttention, which intentionally only
                    # accepts fp16/bf16/fp8.
                    logits, _ = self.adapter.prefill(input_ids)
                else:
                    # Official inference remains the candidate's BF16
                    # FlashAttention path under BF16 autocast.
                    logits = self.model(input_ids)
            self._sync()
            return {"logits": self._logits(logits)}
        if operation == "validation_batch":
            input_ids, targets = self._validation_tensors(
                request.get("input_ids"), request.get("targets")
            )
            with self.torch.inference_mode(), self.torch.amp.autocast(
                device_type="cuda", dtype=self.torch.bfloat16, enabled=self.precision == "bf16"
            ):
                logits = self.model(input_ids)
            if not hasattr(logits, "shape") or tuple(logits.shape[:2]) != tuple(input_ids.shape):
                raise RuntimeError("validation model output shape does not match its input batch")
            logits = logits.float()
            target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            logsumexp = self.torch.logsumexp(logits, dim=-1)
            self._sync()
            return {
                "batch_size": int(input_ids.shape[0]),
                "sequence_length": int(input_ids.shape[1]),
                # These are model outputs only.  The trusted parent applies the
                # target-byte mask, sums the cross-entropy, and computes BPB.
                "target_logits": target_logits.detach().cpu().reshape(-1).tolist(),
                "logsumexp": logsumexp.detach().cpu().reshape(-1).tolist(),
            }
        if operation == "memory":
            return {
                "allocated_cuda_memory_bytes": int(self.torch.cuda.max_memory_allocated()),
                "reserved_cuda_memory_bytes": int(self.torch.cuda.max_memory_reserved()),
            }
        if operation == "reset_memory":
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.empty_cache()
            self._sync()
            return {}
        if operation == "clear_states":
            self.states.clear()
            self.torch.cuda.empty_cache()
            self._sync()
            return {"cleared": True}
        if operation == "release_state":
            state_id = request.get("state_id")
            if isinstance(state_id, bool) or not isinstance(state_id, int) or state_id < 0:
                raise RuntimeError("invalid model-host state id")
            return {"released": self.states.pop(state_id, None) is not None}
        if operation == "shutdown":
            raise EOFError
        raise RuntimeError(f"unknown model-host operation: {operation!r}")


def _serve(args: argparse.Namespace) -> int:
    import socket

    host = ModelHost(
        context=args.context,
        candidate=args.candidate,
        checkpoint=args.checkpoint,
        precision=args.precision,
    )
    channel = socket.socket(fileno=args.ipc_fd)
    channel_file = channel.makefile("rwb", buffering=0)
    while True:
        line = channel_file.readline(MAX_MESSAGE_BYTES + 1)
        if not line:
            break
        if len(line) > MAX_MESSAGE_BYTES:
            raise RuntimeError("model-host request exceeds the protected limit")
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("protocol") != IPC_PROTOCOL:
                raise RuntimeError("model-host request protocol mismatch")
            result = host.handle(request)
            response = {"protocol": IPC_PROTOCOL, "request_id": request.get("request_id"), "ok": True, **result}
        except EOFError:
            break
        except BaseException as exc:
            response = {
                "protocol": IPC_PROTOCOL,
                "request_id": request.get("request_id") if isinstance(locals().get("request"), dict) else None,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        channel_file.write((json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n").encode())
        channel_file.flush()
        if response.get("ok") is False:
            break
    channel_file.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", default="train.py")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp32"), required=True)
    parser.add_argument("--ipc-fd", type=int, required=True)
    args = parser.parse_args(argv)
    return _serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
