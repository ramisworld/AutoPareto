#!/usr/bin/env python3
"""Protected candidate interface and canonical per-layer KV-cache adapter.

Candidates keep the training/checkpoint contract (GPTConfig, GPT, state_dict). The
protected evaluator owns cache semantics so candidate code cannot replace timing or
correctness logic. Architecture changes that no longer expose the documented canonical
block fields require a reviewed protected adapter addition before they are measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
import hashlib
import json


@dataclass
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class DecodeState:
    layers: list[LayerKV]
    position: int


class CandidateInferenceAdapter(Protocol):
    """Stable interface used by the protected benchmark."""

    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, DecodeState]: ...
    def decode_one(self, token_id: torch.Tensor, state: DecodeState) -> tuple[torch.Tensor, DecodeState]: ...
    def cache_nbytes(self, state: DecodeState) -> int: ...


def save_training_checkpoint(
    path,
    model: torch.nn.Module,
    config: Any,
    seed: int,
    train_file: str,
    source_sha256: str | None = None,
) -> None:
    """Stable protected save contract used after the candidate training loop."""
    unwrapped = getattr(model, "_orig_mod", model)
    config_payload = dict(vars(config))
    payload = {
        "format_version": 2,
        "seed": seed,
        "config": config_payload,
        "config_sha256": hashlib.sha256(
            (json.dumps(config_payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "state_dict": {key: value.detach().cpu() for key, value in unwrapped.state_dict().items()},
        "train_file": train_file,
        "source_sha256": source_sha256,
    }
    torch.save(payload, path)


def load_checkpoint_file(path) -> dict[str, Any]:
    """Stable protected checkpoint read contract."""
    return torch.load(path, map_location="cpu", weights_only=True)


def load_checkpoint_model(module: Any, checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Build and load a candidate without running its training setup section."""
    if not hasattr(module, "GPTConfig") or not hasattr(module, "GPT"):
        raise RuntimeError("candidate must expose GPTConfig and GPT for protected checkpoint loading")
    config = module.GPTConfig(**checkpoint["config"])
    with torch.device("meta"):
        model = module.GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


class CanonicalGPTAdapter:
    """KV-cached evaluator for the canonical AutoResearch GPT block contract."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        required = ("transformer", "resid_lambdas", "x0_lambdas", "lm_head", "cos", "sin")
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise RuntimeError(f"candidate is not compatible with protected canonical adapter: {', '.join(missing)}")

    def _value_embedding(
        self,
        block: torch.nn.Module,
        token_ids: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor | None:
        """Resolve the canonical value embedding for one gated layer.

        The registered baseline exposes one embedding table per gated layer.
        A model-neutral architectural change may instead share one table across
        those layers.  Both representations implement the same protected
        attention math; the candidate cannot replace the timing or correctness
        procedure by choosing the representation.
        """
        gate = getattr(block.attn, "ve_gate", None)
        if gate is None:
            return None

        per_layer = getattr(self.model, "value_embeds", None)
        if per_layer is not None:
            key = str(layer_index)
            if key in per_layer:
                return per_layer[key](token_ids)
            if "shared" in per_layer:
                return per_layer["shared"](token_ids)
            if "early" in per_layer:
                config = getattr(self.model, "config", None)
                layer_count = getattr(config, "n_layer", None)
                if isinstance(layer_count, int) and layer_index < layer_count // 2:
                    return per_layer["early"](token_ids)
            if isinstance(per_layer, torch.nn.ModuleList):
                layer_blocks = getattr(getattr(self.model, "transformer", None), "h", None)
                gated_layers = (
                    [
                        index
                        for index, candidate_block in enumerate(layer_blocks)
                        if getattr(getattr(candidate_block, "attn", None), "ve_gate", None) is not None
                    ]
                    if layer_blocks is not None
                    else []
                )
                if len(per_layer) == len(gated_layers) and layer_index in gated_layers:
                    return per_layer[gated_layers.index(layer_index)](token_ids)
                config = getattr(self.model, "config", None)
                layer_count = getattr(config, "n_layer", None)
                if len(per_layer) == 2 and isinstance(layer_count, int) and layer_count > 1:
                    stage = 0 if layer_index < layer_count // 2 else 1
                    return per_layer[stage](token_ids)
            raise RuntimeError(
                f"canonical value embedding is missing gated layer {layer_index}"
            )

        shared = getattr(self.model, "value_embed", None)
        if shared is not None:
            return shared(token_ids)

        raise RuntimeError(
            "canonical model exposes a value gate but no value_embeds or value_embed table"
        )

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),))

    @staticmethod
    def _rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, query_heads: int) -> torch.Tensor:
        kv_heads = x.size(2)
        if kv_heads == query_heads:
            return x
        if query_heads % kv_heads:
            raise RuntimeError("query head count must be divisible by KV head count")
        return x.repeat_interleave(query_heads // kv_heads, dim=2)

    @staticmethod
    def _attention_mask(
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        left_window: int,
    ) -> torch.Tensor:
        allowed = key_positions[None, :] <= query_positions[:, None]
        if left_window > 0:
            allowed &= key_positions[None, :] >= (query_positions[:, None] - left_window)
        return allowed

    def _layer_attention(
        self,
        block: torch.nn.Module,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        layer_index: int,
        start_position: int,
        prior: LayerKV | None,
    ) -> tuple[torch.Tensor, LayerKV]:
        attn = block.attn
        batch, length, _ = x.shape
        normalized = self._norm(x)
        q = attn.c_q(normalized).view(batch, length, attn.n_head, attn.head_dim)
        k = attn.c_k(normalized).view(batch, length, attn.n_kv_head, attn.head_dim)
        v = attn.c_v(normalized).view(batch, length, attn.n_kv_head, attn.head_dim)

        value_embedding = self._value_embedding(block, token_ids, layer_index)
        if value_embedding is not None:
            # Value embeddings may use an independent, smaller head grouping
            # than K/V attention heads.  Older candidates do not expose
            # n_ve_head, in which case the historical n_kv_head layout is the
            # canonical fallback.
            value_heads = int(getattr(attn, "n_ve_head", attn.n_kv_head))
            if value_heads < 1 or attn.n_kv_head % value_heads:
                raise RuntimeError("value-embedding head count must divide KV head count")
            if value_embedding.shape[-1] == attn.head_dim:
                value_embedding = value_embedding.unsqueeze(2).expand(
                    batch, length, attn.n_kv_head, attn.head_dim
                )
                value_heads = attn.n_kv_head
            else:
                value_embedding = value_embedding.view(batch, length, value_heads, attn.head_dim)
            gate = 2 * torch.sigmoid(attn.ve_gate(normalized[..., :attn.ve_gate_channels]))
            if gate.size(-1) != value_heads:
                raise RuntimeError("value-embedding gate width disagrees with value-embedding heads")
            if value_heads != attn.n_kv_head:
                expansion = attn.n_kv_head // value_heads
                value_embedding = value_embedding.repeat_interleave(expansion, dim=2)
                gate = gate.repeat_interleave(expansion, dim=2)
            v = v + gate.unsqueeze(-1) * value_embedding

        end_position = start_position + length
        cos = self.model.cos[:, start_position:end_position]
        sin = self.model.sin[:, start_position:end_position]
        q = self._norm(self._rotary(q, cos, sin))
        k = self._norm(self._rotary(k, cos, sin))

        if prior is not None:
            k_all = torch.cat((prior.key, k), dim=1)
            v_all = torch.cat((prior.value, v), dim=1)
        else:
            k_all, v_all = k, v

        q_positions = torch.arange(start_position, end_position, device=x.device)
        k_positions = torch.arange(k_all.size(1), device=x.device)
        configured = self.model.window_sizes[layer_index][0]
        full_context = int(getattr(self.model.config, "sequence_len", k_all.size(1)))
        left_window = 0 if configured < 0 or configured >= full_context else int(configured)
        mask = self._attention_mask(q_positions, k_positions, left_window)

        q_t = q.transpose(1, 2)
        k_t = self._repeat_kv(k_all, attn.n_head).transpose(1, 2)
        v_t = self._repeat_kv(v_all, attn.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask[None, None, :, :])
        y = y.transpose(1, 2).contiguous().view(batch, length, -1)
        return attn.c_proj(y), LayerKV(k_all, v_all)

    def _forward_cached(
        self,
        token_ids: torch.Tensor,
        state: DecodeState | None,
    ) -> tuple[torch.Tensor, DecodeState]:
        start = 0 if state is None else state.position
        x = self.model.transformer.wte(token_ids)
        x = self._norm(x)
        x0 = x
        new_layers: list[LayerKV] = []
        prior_layers = [] if state is None else state.layers
        for index, block in enumerate(self.model.transformer.h):
            x = self.model.resid_lambdas[index] * x + self.model.x0_lambdas[index] * x0
            prior = prior_layers[index] if state is not None else None
            attention_output, layer_cache = self._layer_attention(block, x, token_ids, index, start, prior)
            x = x + attention_output
            x = x + block.mlp(self._norm(x))
            new_layers.append(layer_cache)
        x = self._norm(x)
        logits = self.model.lm_head(x).float()
        logits = 15 * torch.tanh(logits / 15)
        return logits, DecodeState(new_layers, start + token_ids.size(1))

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, DecodeState]:
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("prefill expects [batch, non-empty sequence]")
        return self._forward_cached(input_ids, None)

    @torch.inference_mode()
    def decode_one(self, token_id: torch.Tensor, state: DecodeState) -> tuple[torch.Tensor, DecodeState]:
        if token_id.ndim != 2 or token_id.size(1) != 1:
            raise ValueError("decode_one expects exactly one token per batch")
        return self._forward_cached(token_id, state)

    def cache_nbytes(self, state: DecodeState) -> int:
        return sum(
            layer.key.numel() * layer.key.element_size() + layer.value.numel() * layer.value.element_size()
            for layer in state.layers
        )


def _validate_adapter(adapter: Any) -> CandidateInferenceAdapter:
    missing = [
        name for name in ("prefill", "decode_one", "cache_nbytes")
        if not callable(getattr(adapter, name, None))
    ]
    if missing:
        raise RuntimeError(f"candidate inference adapter omitted required methods: {', '.join(missing)}")
    return adapter


def state_tensor_nbytes(value: Any) -> int:
    """Count unique tensor storage reachable from an adapter decode state."""
    seen_objects: set[int] = set()
    seen_storage: set[tuple[str, int, int]] = set()

    def visit(item: Any) -> int:
        object_id = id(item)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), int(storage.data_ptr()), int(storage.nbytes()))
            if key in seen_storage:
                return 0
            seen_storage.add(key)
            return int(storage.nbytes())
        if isinstance(item, dict):
            return sum(visit(key) + visit(child) for key, child in item.items())
        if isinstance(item, (list, tuple, set)):
            return sum(visit(child) for child in item)
        attributes = getattr(item, "__dict__", None)
        if isinstance(attributes, dict):
            return visit(attributes)
        return 0

    return visit(value)


def validated_cache_nbytes(adapter: CandidateInferenceAdapter, state: Any) -> int:
    """Reject fabricated cache-size reports before recording KV-cache metrics."""
    reported = adapter.cache_nbytes(state)
    if not isinstance(reported, int) or isinstance(reported, bool) or reported < 0:
        raise RuntimeError("candidate cache_nbytes must return a non-negative integer")
    measured = state_tensor_nbytes(state)
    if reported != measured:
        raise RuntimeError(
            f"candidate cache_nbytes reported {reported} bytes but protected "
            f"state traversal found {measured} bytes"
        )
    return measured


def validated_autoregressive_state_nbytes(adapter: CandidateInferenceAdapter, state: Any) -> int:
    """Count state plus adapter-owned persistent tensors without counting model weights.

    ``cache_nbytes`` remains the exact returned-state/KV accounting contract.  Novel
    recurrent adapters may keep a persistent tensor on the adapter itself; such
    tensors are included diagnostically here, while peak allocated CUDA memory is
    still the authoritative Pareto memory objective.
    """
    measured = state_tensor_nbytes(state)
    attributes = getattr(adapter, "__dict__", {})
    if not isinstance(attributes, dict):
        return measured
    ignored = {"model", "config", "tokenizer", "device"}
    for name, value in attributes.items():
        if name in ignored or name.startswith("_") and name in {"_model", "_config"}:
            continue
        # Adapters commonly retain the model under a custom attribute (``net``,
        # ``module``, ...).  Do not misreport parameter storage as autoregressive
        # state merely because the attribute name is unusual; peak allocated CUDA
        # memory remains authoritative for all live candidate buffers.
        if isinstance(value, torch.nn.Module):
            continue
        measured += state_tensor_nbytes(value)
    return measured


def build_adapter(module: Any, model: torch.nn.Module) -> CandidateInferenceAdapter:
    """Build the candidate cache implementation under the protected timing harness.

    The canonical adapter remains the zero-maintenance path for the starting model.
    A candidate that changes the architecture may expose build_inference_adapter(model)
    in its train.py. Its implementation is part of the recorded candidate diff; prompt,
    correctness checks, timing, repetitions, and metric calculation remain protected.
    """
    factory = getattr(module, "build_inference_adapter", None)
    if factory is not None:
        if not callable(factory):
            raise RuntimeError("candidate build_inference_adapter must be callable")
        return _validate_adapter(factory(model))
    return _validate_adapter(CanonicalGPTAdapter(model))


@torch.inference_mode()
def check_cached_correctness(
    model: torch.nn.Module,
    adapter: CandidateInferenceAdapter,
    prompt: torch.Tensor,
    steps: int,
    mean_atol: float,
    *,
    max_atol: float | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Compare a full-prefix reference with cached decoding.

    The canonical evaluator uses its own SDPA full-prefix pass as the reference.
    Candidate ``GPT.forward`` may use a hardware-specific kernel such as
    FlashAttention, whereas the protected cache uses SDPA. Comparing those backends
    measures backend round-off rather than cache correctness. Recomputing ``prefill``
    on each full common prefix uses no prior K/V state; ``decode_one`` remains the
    independently exercised cached path.

    Candidate-supplied adapters retain the candidate-model-forward reference until a
    reviewed protected full-prefix implementation is available for that adapter.
    """
    cached_logits, state = adapter.prefill(prompt)
    common_tokens = prompt
    max_error = 0.0
    max_mean_error = 0.0
    token_match = True
    finite_outputs = True
    canonical_reference = isinstance(adapter, CanonicalGPTAdapter)
    for index in range(steps):
        if canonical_reference:
            reference_logits, _ = adapter.prefill(common_tokens)
            uncached_step = reference_logits[:, -1, :].detach().float()
        else:
            uncached_step = model(common_tokens)[:, -1, :].detach().float()
        cached_step = cached_logits[:, -1, :].detach().float()
        finite_outputs = finite_outputs and bool(
            torch.isfinite(uncached_step).all().item()
            and torch.isfinite(cached_step).all().item()
        )
        difference = (uncached_step - cached_step).abs()
        max_error = max(max_error, float(difference.max().item()))
        max_mean_error = max(max_mean_error, float(difference.mean().item()))
        uncached_token = uncached_step.argmax(dim=-1, keepdim=True)
        cached_token = cached_step.argmax(dim=-1, keepdim=True)
        token_match = token_match and bool(torch.equal(uncached_token, cached_token))
        common_tokens = torch.cat((common_tokens, uncached_token), dim=1)
        if index + 1 < steps:
            cached_logits, state = adapter.decode_one(uncached_token, state)
    mean_logits_match = max_mean_error <= mean_atol
    max_logits_match = max_atol is None or max_error <= max_atol
    logits_match = mean_logits_match and max_logits_match
    result = {
        "passed": finite_outputs and token_match and logits_match,
        "greedy_tokens_match": token_match,
        "finite_outputs": finite_outputs,
        "logits_match": logits_match,
        "mean_logits_match": mean_logits_match,
        "max_logits_match": max_logits_match,
        "max_logit_absolute_error": max_error,
        "max_mean_logit_absolute_error": max_mean_error,
        "mean_atol": mean_atol,
        "max_atol": max_atol,
        "checked_decode_steps": steps,
        "prefix_mode": "teacher_forced_common_prefix",
        "full_prefix_reference": (
            "protected_sdpa" if canonical_reference else "candidate_model_forward"
        ),
    }
    if raise_on_failure and not result["passed"]:
        raise RuntimeError(
            f"cached/full correctness failed: greedy_tokens_match={token_match} "
            f"mean_logits_match={logits_match} max_mean_abs_error={max_mean_error:.6g} "
            f"max_abs_error={max_error:.6g}"
        )
    return result
