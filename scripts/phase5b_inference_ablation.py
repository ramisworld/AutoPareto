#!/usr/bin/env python3
"""Phase 5B: inference-only ablations of the Opus 5.5 frontier adapters on their existing checkpoints.

Each variant is the frozen Pair-A candidate source, byte for byte, with ONLY its
final adapter factory

    def build_inference_adapter(model):
        return ChunkedKVInferenceAdapter(model)

replaced by a block that subclasses ``ChunkedKVInferenceAdapter`` and switches
one execution mechanism. The model code, the checkpoint and the protected
benchmark (``study.contracts_2026.inference_benchmark``, candidate-only mode) are
unchanged. The protected correctness check (cached adapter against the uncached
model forward, with FP32 adjudication) runs on every variant.

| Variant | CUDA graph per prompt | Inductor-fused decode regions | Decode attention | Sliding-window K/V |
|---|---|---|---|---|
| V0 as_submitted | yes | yes | compiled split-key `_dec_attn` | ring buffer |
| V1 no_cuda_graph | no (same kernels, launched eagerly) | yes | split-key | ring |
| V2 no_fusion | yes | no (the same pure functions, uncompiled) | split-key, uncompiled | ring |
| V3 sdpa_attention | yes | yes | masked SDPA (Opus's pre-attempt-18 path) | ring |
| V4 eager_unfused | no | no | split-key, uncompiled | ring |
| V5 linear_kv_cache | yes | yes | split-key | full-length linear buffer + window mask |

V0 is the unmodified source; no block is inserted. Every variant computes the
same function as V0, the same key set and the same scale, so no retraining is
involved. Changing a window size would change the model and is excluded.

    python3 scripts/phase5b_inference_ablation.py make --sources <dir> --prepare <prepare.py> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FACTORY = "def build_inference_adapter(model):\n    return ChunkedKVInferenceAdapter(model)\n"
SETUP_MARKER = "# Setup: tokenizer, model, optimizer, dataloader"
PREPARE_SHA256 = "4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150"

VARIANTS = {
    "V0_as_submitted": None,
    "V1_no_cuda_graph": {"cuda_graph": False, "compile_regions": True, "sdpa_decode_attention": False, "ring_kv_cache": True},
    "V2_no_fusion": {"cuda_graph": True, "compile_regions": False, "sdpa_decode_attention": False, "ring_kv_cache": True},
    "V3_sdpa_attention": {"cuda_graph": True, "compile_regions": True, "sdpa_decode_attention": True, "ring_kv_cache": True},
    "V4_eager_unfused": {"cuda_graph": False, "compile_regions": False, "sdpa_decode_attention": False, "ring_kv_cache": True},
    "V5_linear_kv_cache": {"cuda_graph": True, "compile_regions": True, "sdpa_decode_attention": False, "ring_kv_cache": False},
}

BLOCK = '''# ---------------------------------------------------------------------------
# Phase 5B inference-only ablation (inserted by scripts/phase5b_inference_ablation.py).
# Everything above this block is the frozen candidate source, byte for byte; only
# the adapter factory differs. Same model, same weights, same decode math; one
# execution mechanism is switched (see PHASE5B_SETTINGS).
# ---------------------------------------------------------------------------
PHASE5B_VARIANT = {variant!r}
PHASE5B_SETTINGS = {settings!r}


def _phase5b_sdpa_decode_attn(q, kb, vb, mask, out_dtype=None):
    # Masked SDPA over the static K/V buffers (Opus's decode attention before
    # attempt 18): same key set (boolean mask), same default 1/sqrt(head_dim) scale.
    B, H, Tq, D = q.shape
    k = kb.permute(0, 2, 1, 3)
    v = vb.permute(0, 2, 1, 3)
    if k.size(1) != H:
        rep = H // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    y = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask.reshape(1, 1, 1, -1))
    y = y.to(q.dtype)
    if out_dtype is not None:
        y = y.to(out_dtype)
    return y


class Phase5BAblationAdapter(ChunkedKVInferenceAdapter):
    def __init__(self, model):
        super().__init__(model)
        settings = PHASE5B_SETTINGS
        # compile_regions=False: the SAME pure helper functions, called uncompiled
        # (plain eager aten kernels, no Inductor fusion).
        fused = dict(self._fused) if settings["compile_regions"] else dict(_DECODE_FUSION_HELPERS)
        if settings["sdpa_decode_attention"]:
            fused["decode_attn"] = _phase5b_sdpa_decode_attn
        self._fused = fused
        self._phase5b_warm = False

    def _ring_sizes(self, cap):
        if PHASE5B_SETTINGS["ring_kv_cache"]:
            return super()._ring_sizes(cap)
        # Every layer keeps a full-length (linear) K/V buffer; sliding-window
        # layers are restricted by the same window mask, so the key set is unchanged.
        return tuple(0 for _ in self.model.window_sizes)

    def _build_decode_graph(self, k_bufs, v_bufs, B, pos, rings):
        if PHASE5B_SETTINGS["cuda_graph"]:
            return super()._build_decode_graph(k_bufs, v_bufs, B, pos, rings)
        # No capture: the same static workspace as the parent (copied verbatim),
        # and the warm-up step runs once per adapter (first-call compilation),
        # never per prompt.
        device = k_bufs[0].device
        rings = tuple(int(r) for r in rings)
        linear_layers = [i for i, R in enumerate(rings) if not R]
        io = {{
            "tok": torch.zeros((B, 1), dtype=torch.long, device=device),
            "prev_tok": torch.zeros((B, 1), dtype=torch.long, device=device),
            "pos": torch.full((1,), pos, dtype=torch.long, device=device),
            "k": list(k_bufs),
            "v": list(v_bufs),
            "rings": rings,
            "grow_layer": linear_layers[0] if linear_layers else -1,
            "ring_slots": {{
                R: torch.arange(R, dtype=torch.long, device=device)
                for R in sorted({{R for R in rings if R}})
            }},
        }}
        if linear_layers:
            cap = k_bufs[linear_layers[0]].size(1)
            io["kpos"] = torch.arange(cap, dtype=torch.long, device=device)
        if not self._phase5b_warm:
            enabled, dtype = self._ambient_autocast()
            with torch.autocast("cuda", dtype=dtype, enabled=enabled, cache_enabled=False):
                self._warmup_decode_step(io)
            self._phase5b_warm = True
        io["out"] = None
        return None, io

    def decode_one(self, one_token, state):
        if PHASE5B_SETTINGS["cuda_graph"]:
            return super().decode_one(one_token, state)
        # The parent's decode_one with graph.replay() replaced by the same step
        # body executed eagerly under the same autocast settings as the capture.
        with torch.no_grad():
            model = self.model
            tok = torch.as_tensor(one_token, device=model.cos.device)
            if tok.dim() == 0:
                tok = tok.view(1, 1)
            elif tok.dim() == 1:
                tok = tok.view(-1, 1)
            B = tok.size(0)
            p = state["pos"]
            assert p < model.cos.size(1)
            k_bufs, v_bufs = list(state["k"]), list(state["v"])
            io = state["io"]
            assert tuple(tok.shape) == tuple(io["tok"].shape)
            g = io["grow_layer"]
            if g >= 0 and p >= k_bufs[g].size(1):
                rings = io["rings"]
                old_io = io
                k_bufs = [kb if R else self._grow(kb) for kb, R in zip(k_bufs, rings)]
                v_bufs = [vb if R else self._grow(vb) for vb, R in zip(v_bufs, rings)]
                _graph, io = self._build_decode_graph(k_bufs, v_bufs, B, p, rings)
                io["prev_tok"].copy_(old_io["prev_tok"])
                del old_io
            io["tok"].copy_(tok)
            io["pos"].fill_(p)
            enabled, dtype = self._ambient_autocast()
            with torch.autocast("cuda", dtype=dtype, enabled=enabled, cache_enabled=False):
                out = self._decode_step_static(io)
            logits = out.clone()
            return logits, {{"k": k_bufs, "v": v_bufs, "pos": p + 1, "graph": None, "io": io}}


def build_inference_adapter(model):
    return Phase5BAblationAdapter(model)
'''


def variant_source(original: str, variant: str) -> str:
    settings = VARIANTS[variant]
    if settings is None:
        return original
    if original.count(FACTORY) != 1:
        raise ValueError("adapter factory not found exactly once")
    if original.index(FACTORY) > original.index(SETUP_MARKER):
        raise ValueError("adapter factory is not in the definitions region")
    return original.replace(FACTORY, BLOCK.format(variant=variant, settings=settings), 1)


def make(sources: Path, prepare: Path, out: Path) -> dict:
    if hashlib.sha256(prepare.read_bytes()).hexdigest() != PREPARE_SHA256:
        raise SystemExit("registered prepare.py hash differs")
    if out.exists():
        raise SystemExit(f"refusing to reuse {out}")
    manifest = {"format": "autopareto_phase5b_variant_contexts_v1", "contexts": {}}
    for path in sorted(sources.glob("opus_attempt*.train.py")):
        attempt = path.name.split(".")[0].replace("opus_attempt", "")
        original = path.read_text()
        for variant in VARIANTS:
            context = out / f"attempt-{attempt}" / variant
            context.mkdir(parents=True)
            text = variant_source(original, variant)
            (context / "train.py").write_text(text)
            (context / "prepare.py").write_bytes(prepare.read_bytes())
            manifest["contexts"][f"{attempt}/{variant}"] = {
                "context": str(context),
                "original_source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "train_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "settings": VARIANTS[variant],
            }
    (out / "variant_contexts.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make")
    m.add_argument("--sources", required=True, type=Path)
    m.add_argument("--prepare", required=True, type=Path)
    m.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "make":
        manifest = make(args.sources, args.prepare, args.out)
        print(json.dumps({k: v["train_sha256"][:12] for k, v in manifest["contexts"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
