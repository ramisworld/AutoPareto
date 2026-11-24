"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


# Fixed attention-logit gain on the RMS-normed query (no learnable parameter, no
# GPTConfig field: state_dict and model-host rebuild unchanged). With gain-free
# q/k RMS norms and the default 1/sqrt(head_dim) softmax scale, logits are capped
# at sqrt(128) ~= 11.3; q * 1.35 is equivalent to softmax scale ~0.12 at head_dim
# 128. Applied identically in the training forward, the adapter prefill q path
# (_qkv) and the decode fusion helper (_dec_qk_post); K is never scaled.
QK_GAIN = 1.35


# Input-level token shift ("smear", attempt 23). For t >= 1 the token embedding
# receives a gated copy of the PREVIOUS token's embedding before the embedding
# rms_norm:
#   x[t] = wte[t] + lambda * sigmoid(w . wte[t][:SMEAR_GATE_CHANNELS]) * wte[t-1]
# (position 0 unchanged). lambda (1) and w (SMEAR_GATE_CHANNELS) are GPT
# parameters, zero-initialized (the model starts exactly at the attempt-22
# function), trained by AdamW at SMEAR_LR (not dmodel-scaled). One pure helper,
# _smear_mix, is used by EVERY path: the training forward, the adapter prefill,
# the GEMV-bounded decode region step_prep and the per-op decode fallback.
SMEAR_GATE_CHANNELS = 12
SMEAR_LR = 0.05


def _smear_mix(x_cur, x_prev, gate_w, lam):
    # Explicit fp32 math (no nn.Linear, no autocast dependence), result cast back
    # to the embedding dtype, so the residual-stream dtype flow is unchanged.
    gate = torch.sigmoid((x_cur[..., :SMEAR_GATE_CHANNELS].float() * gate_w.float()).sum(-1, keepdim=True))
    return (x_cur.float() + lam.float() * gate * x_prev.float()).to(x_cur.dtype)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# Matrix-weight storage precision (inference-memory policy).
# Every inference matmul already runs in bf16 under autocast, so fp32 storage of
# the block and lm_head matrices is never used at inference: it costs ~26 MiB of
# weights, a per-token bf16 re-cast of ~13.6M weights inside the decode graph,
# and an lm_head cast per prefill logit chunk. GPT therefore DECLARES these
# matrices in INFERENCE_MATRIX_DTYPE (what a fresh model host builds and what
# load_state_dict copies into; the bf16 round-to-nearest-even copy is bit-equal
# to autocast's per-use cast). The training script switches them to fp32 before
# init_weights (training identical to attempt 11) and back to bf16 after the
# last completed update.
INFERENCE_MATRIX_DTYPE = torch.bfloat16

if hasattr(torch, "get_autocast_dtype"):
    def _cuda_autocast_enabled():
        return torch.is_autocast_enabled("cuda")
else:
    def _cuda_autocast_enabled():
        return torch.is_autocast_enabled()


class PrecisionLinear(nn.Linear):
    """Bias-free nn.Linear whose weight storage dtype may differ from the input.

    Under CUDA autocast F.linear casts exactly as nn.Linear does (casting a bf16
    weight to bf16 is a no-op). With autocast off, a weight stored in a dtype
    different from x is cast to x's dtype as a per-call temporary (never stored).
    """

    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        assert not bias, "PrecisionLinear is bias-free"
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        w = self.weight
        if w.dtype != x.dtype and not _cuda_autocast_enabled():
            w = w.to(x.dtype)
        return F.linear(x, w)


# Gather-only table storage (attempt 27; inference-memory policy).
# transformer.wte and the value embeddings (layers 0 and 2) are only ever
# row-gathered (no matmul, no tied head): 3 x 8 MiB = 24 MiB of the ~50 MiB of
# bf16 weights. GPT DECLARES them as int8 storage (what a fresh model host
# builds and what load_state_dict copies into) with a persistent per-row
# dequantization scale buffer (V, 1) in TABLE_SCALE_DTYPE (bf16, the former
# table dtype):
#   s[r] = bf16(rowabsmax(w[r]) / 127);  q[r] = clamp(round(w[r] / s[r]), -127, 127)
# Training is unchanged: the script switches the tables to trainable
# TRAIN_TABLE_DTYPE (bf16, the attempt-26 table dtype) storage BEFORE
# init_weights (same Parameter objects, same Adam groups; init draws are fp32
# then rounded to bf16 exactly as before) and quantizes them in place after the
# last completed update, outside the measured clock. One pure helper,
# _table_gather, is used by EVERY path (training forward / evaluation, adapter
# prefill, decode regions step_prep and post_qkv, per-op decode fallback): it
# dequantizes ONLY the gathered rows (int8 row -> s.dtype, times the gathered
# row scale), so no full-table dequantized copy is ever materialized; inside
# the compiled decode regions the dequant fuses into the existing kernels.
INFERENCE_TABLE_DTYPE = torch.int8
TRAIN_TABLE_DTYPE = torch.bfloat16
TABLE_SCALE_DTYPE = torch.bfloat16
TABLE_INIT_DTYPE = torch.float32  # init draws (attempt 26 drew fp32, then cast to bf16)


def _table_gather(idx, w, s):
    # Row gather from a GatherTable's storage. Floating storage (training):
    # exactly F.embedding (the former nn.Embedding forward). int8 storage:
    # gathered int8 rows cast to the scale dtype times the gathered per-row
    # scale, so the result dtype is s.dtype (bf16 by default; it follows the
    # scale under model.float() / .to(dtype), which leave int8 storage alone).
    if w.is_floating_point():
        return F.embedding(idx, w)
    return F.embedding(idx, w).to(s.dtype) * F.embedding(idx, s)


@torch.no_grad()
def _quantize_rows_int8(w, scale_dtype):
    # Per-row symmetric absmax int8 quantization: s = rowabsmax / 127 rounded to
    # scale_dtype; q = round(w / s) (using the rounded s), clamped to
    # [-127, 127]. All-zero rows get s = 0 and q = 0 (exact). No host sync.
    wf = w.float()
    s = (wf.abs().amax(dim=1, keepdim=True) / 127.0).to(scale_dtype)
    sf = s.float()
    denom = torch.where(sf > 0, sf, torch.ones_like(sf))
    q = torch.round(wf / denom).clamp_(-127, 127).to(torch.int8)
    return q, s


class GatherTable(nn.Module):
    """Row-gather-only embedding table (replaces nn.Embedding for wte / VE).

    `weight` (V, D) is declared in INFERENCE_TABLE_DTYPE (int8, requires_grad
    False) and `scale` (V, 1) is a persistent TABLE_SCALE_DTYPE buffer, so the
    state_dict keys are <name>.weight and <name>.scale and a fresh host strict-
    loads int8 storage. set_storage switches the same Parameter object between
    trainable floating storage and int8 storage; store_ writes float values into
    whatever the current storage is (copy/round or quantize).
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=INFERENCE_TABLE_DTYPE),
            requires_grad=INFERENCE_TABLE_DTYPE.is_floating_point)
        self.register_buffer("scale", torch.ones((num_embeddings, 1), dtype=TABLE_SCALE_DTYPE))

    def forward(self, idx):
        return _table_gather(idx, self.weight, self.scale)

    @torch.no_grad()
    def store_(self, w):
        # Write float values w (V, D) into the current storage. Floating storage:
        # copy_ (round-to-nearest-even cast, identical to the former
        # .to(bfloat16) of the fp32 draw), scale = 1 (unused). int8 storage:
        # per-row quantization into the existing Parameter / scale buffer.
        p = self.weight
        if p.is_floating_point():
            p.copy_(w)
            self.scale.fill_(1.0)
        else:
            q, s = _quantize_rows_int8(w, self.scale.dtype)
            p.copy_(q)
            self.scale.copy_(s)

    @torch.no_grad()
    def set_storage(self, dtype):
        # In-place storage switch; the Parameter object (and state_dict keys and
        # shapes) is unchanged. Floating dtype: trainable storage (int8 storage
        # is dequantized row-wise first). int8: quantize the current values and
        # drop requires_grad (an integer Parameter cannot require grad).
        p = self.weight
        if dtype.is_floating_point:
            if p.is_floating_point():
                if p.dtype != dtype:
                    p.data = p.data.to(dtype)
            else:
                p.data = (p.data.to(self.scale.dtype) * self.scale).to(dtype)
            self.scale.fill_(1.0)
            p.requires_grad_(True)
        else:
            assert dtype == torch.int8, f"unsupported table storage dtype {dtype}"
            if p.is_floating_point():
                q, s = _quantize_rows_int8(p.data, self.scale.dtype)
                p.requires_grad_(False)
                p.grad = None
                p.data = q
                self.scale.copy_(s)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        mdt = INFERENCE_MATRIX_DTYPE
        self.c_q = PrecisionLinear(self.n_embd, self.n_head * self.head_dim, bias=False, dtype=mdt)
        self.c_k = PrecisionLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False, dtype=mdt)
        self.c_v = PrecisionLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False, dtype=mdt)
        self.c_proj = PrecisionLinear(self.n_embd, self.n_embd, bias=False, dtype=mdt)
        self.ve_gate_channels = 32
        self.ve_gate = PrecisionLinear(self.ve_gate_channels, self.n_kv_head, bias=False, dtype=mdt) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * QK_GAIN, norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = PrecisionLinear(config.n_embd, 4 * config.n_embd, bias=False, dtype=INFERENCE_MATRIX_DTYPE)
        self.c_proj = PrecisionLinear(4 * config.n_embd, config.n_embd, bias=False, dtype=INFERENCE_MATRIX_DTYPE)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            # Gather-only token table: int8 storage + per-row scale (GatherTable).
            "wte": GatherTable(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = PrecisionLinear(config.n_embd, config.vocab_size, bias=False, dtype=INFERENCE_MATRIX_DTYPE)
        # Declared storage dtype of the block/lm_head matrices (see set_matrix_dtype)
        self.matrix_dtype = INFERENCE_MATRIX_DTYPE
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Token-shift (smear) gate weights and strength (see _smear_mix); outside
        # transformer.h, so never Muon parameters.
        self.smear_gate_w = nn.Parameter(torch.zeros(SMEAR_GATE_CHANNELS))
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Value embeddings (gather-only: int8 storage + per-row scale, GatherTable)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): GatherTable(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Declared storage dtype of the gather-only tables (see set_table_dtype)
        self.table_dtype = INFERENCE_TABLE_DTYPE
        # Rotary embeddings (non-persistent buffers; values per position do not
        # depend on the table length). 2x covers validation (2048) and inference
        # (1024 + 128); the former 10x held ~4 MiB of never-used bf16 rows.
        self.rotary_seq_len = config.sequence_len * 2
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Gather-only tables: every draw is made in TABLE_INIT_DTYPE (fp32) on a
        # temporary of the table's shape, in the same order / size / dtype as
        # attempt 26's in-place fp32 draws (identical RNG stream and values),
        # then stored into the current table storage (bf16 in training: the same
        # round-to-nearest-even cast as attempt 26's .to(bfloat16); int8 on a
        # fresh host: per-row quantization). Works for either storage dtype.
        def _draw(table):
            return torch.empty((table.num_embeddings, table.embedding_dim),
                               dtype=TABLE_INIT_DTYPE, device=table.weight.device)
        # Embedding and unembedding
        wte = self.transformer.wte
        w = _draw(wte)
        torch.nn.init.normal_(w, mean=0.0, std=1.0)
        wte.store_(w)
        del w
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Token-shift (smear): zero init (lambda = 0 -> exact identity at init)
        torch.nn.init.zeros_(self.smear_gate_w)
        torch.nn.init.zeros_(self.smear_lambda)
        # Value embeddings (fp32 draw, stored into the table storage; see _draw)
        for ve in self.value_embeds.values():
            w = _draw(ve)
            torch.nn.init.uniform_(w, -s, s)
            ve.store_(w)
            del w
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # (The former "cast embeddings to bf16" step is the store_ calls above:
        # bf16 tables in training, int8 + scale on a fresh host.)
        # Apply the stored matrix storage dtype (no-op in training, where the
        # script set fp32 before init_weights so all draws above are fp32).
        self.set_matrix_dtype(self.matrix_dtype)

    def matrix_parameters(self):
        # Block (attn c_q/c_k/c_v/c_proj, ve_gate, mlp c_fc/c_proj) and lm_head weights.
        return [m.weight for m in self.modules() if isinstance(m, PrecisionLinear)]

    @torch.no_grad()
    def set_matrix_dtype(self, dtype):
        # In-place storage conversion of the block/lm_head matrices; the
        # Parameter objects (and state_dict keys/shapes) are unchanged.
        for p in self.matrix_parameters():
            if p.dtype != dtype:
                p.data = p.data.to(dtype)
        self.matrix_dtype = dtype

    def gather_tables(self):
        # The row-gather-only tables: wte and the value embeddings.
        return [self.transformer.wte] + list(self.value_embeds.values())

    @torch.no_grad()
    def set_table_dtype(self, dtype):
        # In-place storage switch of the gather-only tables (GatherTable.set_storage):
        # TRAIN_TABLE_DTYPE (trainable bf16) before init_weights in training,
        # INFERENCE_TABLE_DTYPE (int8 + per-row scale) after the last update.
        for table in self.gather_tables():
            table.set_storage(dtype)
        self.table_dtype = dtype

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        # Short (S) layers attend to 1/8 of the context (256 at sequence_len 2048);
        # the last layer is always full context (window_sizes[-1] below).
        short_window = long_window // 8
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        # GatherTable parameters are just their `weight` (same numel as the former
        # nn.Embedding); the per-row dequant `scale` is a buffer, not a parameter.
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        smear = self.smear_gate_w.numel() + self.smear_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + smear
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars,
            'smear': smear, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate_w, self.smear_lambda]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) +
            len(smear_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            # Token-shift (smear) gate + strength: fixed LR, not dmodel-scaled.
            dict(kind='adamw', params=smear_params, lr=SMEAR_LR, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        # Token shift: position 0 unchanged, t >= 1 gets the gated wte[t-1].
        x = torch.cat([x[:, :1], _smear_mix(x[:, 1:], x[:, :-1], self.smear_gate_w, self.smear_lambda)], dim=1)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Inference adapter (protected prefill/decode interface; inference-only, the
# training path never calls it)
# ---------------------------------------------------------------------------

ADAPTER_KV_CHUNK = 256     # K/V buffer capacity granularity (positions)
ADAPTER_LOGIT_CHUNK = 128  # (unused since attempt 21: prefill returns last-position logits only)
ADAPTER_RING_ALIGN = 16    # sliding-window ring size granularity (slots; mask alignment, decode-attention key chunk >= 16)
ADAPTER_ATTN_CHUNK = 64    # max keys per split-key decode-attention chunk (partial P @ V)
ADAPTER_PREFILL_FP32_QCHUNK = 256  # query rows per chunk of the fp32-only prefill attention fallback (FA3 has no fp32 kernel)


# Decode-step fusion helpers (inference-only; the training path never calls them).
# Pure functions that reproduce the eager decode math op for op (same op order,
# same dtype flow under the ambient autocast). The adapter runs TorchInductor-
# compiled versions of them inside its captured decode step, so each shape-stable
# (B=1, T=1) pointwise / normalization region becomes one or a few fused Triton
# kernels instead of a chain of tiny eager aten kernels. Every GEMV, the wte /
# value-embedding lookups, the rotary index_select, the K/V index_copy_ and the
# cap-dependent sliding-window mask stay eager aten calls in the graph. The
# pointwise/norm helpers' inputs never depend on prompt length or K/V capacity,
# so prompts of any length share the same compiled kernels; the decode-attention
# helper (_dec_attn) is specialized per slot count L (ring size R or linear
# capacity), i.e. a handful of static shapes per process. (Since attempt 22 these
# per-op helpers are the fallback path; the captured step normally uses the
# GEMV-bounded regions defined after _dec_attn, which also absorb the lookups,
# index_selects, masks and residual adds. step_prep is specialized per K/V
# capacity / mask set like _dec_attn.)

def _dec_mix_norm(x, x0, rl, xl):
    # resid/x0 lambda mixing followed by the block's pre-attention rms_norm.
    x = rl * x + xl * x0
    return x, norm(x)


def _dec_qk_post(q, k, cos, sin):
    # Rotary on q and k, then the q/k rms_norm and the fixed q gain QK_GAIN
    # (CausalSelfAttention order).
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    return norm(q) * QK_GAIN, norm(k)


def _dec_ve_mix(v, gate_logits, ve):
    # Value residual: v + 2 * sigmoid(ve_gate(x[..., :32])) * ve per head.
    gate = 2 * torch.sigmoid(gate_logits)
    return v + gate.unsqueeze(-1) * ve


def _dec_add_norm(x, y):
    # Attention residual add followed by the MLP's input rms_norm.
    x = x + y
    return x, norm(x)


def _dec_relu2(h):
    # MLP activation relu(h)^2.
    return F.relu(h).square()


def _dec_add(x, y):
    # MLP residual add.
    return x + y


def _dec_final_norm(x):
    # rms_norm (embedding norm and final norm).
    return norm(x)


def _dec_softcap(logits):
    # fp32 logits softcapped as in GPT.forward: 15 * tanh(logits / 15).
    softcap = 15
    logits = logits.float()
    return softcap * torch.tanh(logits / softcap)


def _dec_attn(q, kb, vb, mask, out_dtype=None):
    # Split-key single-query decode attention; replaces the decode-path masked
    # SDPA (whose memory-efficient kernel runs B x H = 4 CTAs, each scanning
    # every key slot serially). Same key set, same scale, same softmax:
    #   y[b, h] = sum_l softmax_l(scale * <q[b, h], k[b, l, g(h)]> | mask) * v[b, l, g(h)]
    # q: (B, H, 1, D); kb, vb: (B, L, H_kv, D) cache buffers in their EXISTING
    # layout (L = ring size R or linear capacity); mask: bool with L elements
    # (the cached (1, 1, 1, L) decode key mask, True = attend; every query keeps
    # at least its own slot, so no row is all -inf). scale = 1/sqrt(head_dim),
    # the F.scaled_dot_product_attention / FA3 default (a trace-time constant
    # from the static head dim). Written as broadcast-multiply/reduce only (no
    # matmul, so autocast never down-casts it; no extern bmm; no expand /
    # contiguous copy of K/V): Inductor emits parallel reduction kernels and
    # fuses the bf16 -> fp32 upcast into the loads, so no fp32 K/V buffer is
    # materialized. All math is fp32 (with autocast off and fp32 caches, the
    # .float() casts are no-ops and everything runs in fp32); the result is cast
    # to q.dtype, then (out_dtype given, GEMV-bounded region path) to the dtype
    # the attention c_proj GEMV consumes under the ambient autocast, i.e. the
    # same cast autocast would apply at c_proj (None = no cast, the per-op path).
    # GQA (H_kv < H) is handled by broadcasting (head h = g*rep + r).
    B, H, Tq, D = q.shape  # Tq == 1
    L, Hkv = kb.shape[1], kb.shape[2]
    rep = H // Hkv
    scale = 1.0 / math.sqrt(D)
    # (1) fp32 scores, parallel over all B * H * L slots (reduce over d).
    qf = q.reshape(B, Hkv, rep, 1, D).float()                    # (B, Hkv, rep, 1, D)
    kf = kb.permute(0, 2, 1, 3).unsqueeze(2).float()             # (B, Hkv, 1, L, D) view
    s = (qf * kf).sum(dim=-1) * scale                            # (B, Hkv, rep, L)
    # (2) masked fp32 softmax over the L slots.
    s = s.masked_fill(~mask.reshape(L), float("-inf"))
    p = torch.softmax(s, dim=-1)                                 # (B, Hkv, rep, L)
    # (3) y = sum_l p * v, split over keys: per-chunk partial sums, parallel over
    # B * H * (L / C) * D outputs (C keys each), then a sum over the L / C chunks.
    # C = largest power of two <= ADAPTER_ATTN_CHUNK dividing L (ring sizes are
    # multiples of ADAPTER_RING_ALIGN, linear capacities of ADAPTER_KV_CHUNK).
    C = 1
    while 2 * C <= ADAPTER_ATTN_CHUNK and L % (2 * C) == 0:
        C *= 2
    NS = L // C
    vf = vb.reshape(B, NS, C, Hkv, D).permute(0, 3, 1, 2, 4).unsqueeze(2).float()  # (B, Hkv, 1, NS, C, D) view
    part = (p.reshape(B, Hkv, rep, NS, C, 1) * vf).sum(dim=4)    # (B, Hkv, rep, NS, D)
    y = part.sum(dim=3)                                          # (B, Hkv, rep, D)
    y = y.reshape(B, H, Tq, D).to(q.dtype)
    if out_dtype is not None:
        y = y.to(out_dtype)
    return y


# GEMV-bounded decode regions (attempt 22; the primary captured decode path).
# Batch-1 graph replay is bound by the number of serialized kernels, and the
# per-op helpers above still leave eager glue between them (embedding and value-
# embedding lookups, rotary index_selects, ring-slot / mask arithmetic, residual
# adds, autocast input casts). Each region below covers ALL non-GEMV work
# between two consecutive matrix-vector multiplies of the step:
#   step_prep -> per layer: pre_attn -> [c_q, c_k, c_v, ve_gate] -> post_qkv ->
#   (eager K/V index_copy_) -> decode_attn -> [attn c_proj] -> add_norm_cast ->
#   [c_fc] -> relu2_cast -> [mlp c_proj] (residual add deferred into the next
#   pre_attn / head_norm) -> head_norm -> [lm_head] -> softcap.
# Same ops, same order and same dtype flow as the per-op path (no new math).
# `gemv_dtype` is the dtype the following GEMV consumes: the ambient autocast
# dtype when autocast is enabled (exactly the input cast F.linear applies under
# autocast, so autocast issues no cast kernels) and None when it is disabled
# (no cast; PrecisionLinear then behaves exactly as before). Weights/tables are
# passed as arguments, so every layer shares one compiled graph per variant
# (VE vs non-VE layer, layer 0 vs later, ambient autocast mode, mask set). The
# K/V state buffers are never inputs that a region mutates: the slot writes stay
# eager index_copy_ calls.

def _dec_step_prep(tok, prev_tok, pos, wte_w, wte_s, smear_w, smear_lam, cos_tab, sin_tab,
                   ring_tabs, ring_sizes, ring_keys, kpos, linear_lefts):
    # Token embedding, token-shift smear with the previous token's embedding
    # (_smear_mix, same math as GPT.forward at position pos >= 1) + embedding
    # rms_norm (x0), rotary cos/sin at pos, the ring
    # write slots pos mod R (one per ring size) and every key mask of the step:
    # ring (left, R): d = (pos - s) mod R, keep d <= left and d <= pos;
    # linear (left): (kpos <= pos) & (kpos >= pos - left). Identical to the
    # per-op path's lazily cached write_slots / masks. Both wte rows (tok and
    # prev_tok) come from _table_gather (int8 row * row scale, fused here; the
    # int8 table and its scale are region arguments, never dequantized whole).
    x0 = norm(_smear_mix(_table_gather(tok, wte_w, wte_s), _table_gather(prev_tok, wte_w, wte_s),
                         smear_w, smear_lam))
    cos = cos_tab.index_select(1, pos)
    sin = sin_tab.index_select(1, pos)
    write_slots = []
    for R in ring_sizes:
        write_slots.append(torch.remainder(pos, R))
    ring_masks = []
    for left, j in ring_keys:
        d = torch.remainder(pos - ring_tabs[j], ring_sizes[j])
        ring_masks.append(((d <= left) & (d <= pos)).view(1, 1, 1, -1))
    linear_masks = []
    for left in linear_lefts:
        linear_masks.append(((kpos <= pos) & (kpos >= pos - left)).view(1, 1, 1, -1))
    return x0, cos, sin, tuple(write_slots), tuple(ring_masks), tuple(linear_masks)


def _dec_pre_attn(x, y, x0, rl, xl, gemv_dtype):
    # Previous layer's deferred MLP residual add (x None: layer 0, whose
    # residual stream is x0 itself), resid/x0 lambda mix, pre-attention
    # rms_norm; xn emitted in the dtype c_q/c_k/c_v/ve_gate consume.
    if x is None:
        x = x0
    else:
        x = x + y
    x = rl * x + xl * x0
    xn = norm(x)
    if gemv_dtype is not None:
        xn = xn.to(gemv_dtype)
    return x, xn


def _dec_post_qkv(q, k, v, cos, sin, gate_logits, ve_w, ve_s, tok, k_dtype, v_dtype):
    # Value-embedding gather (_table_gather: int8 row * row scale, fused here)
    # and 2 * sigmoid gate mix (VE layers), rotary on q
    # and k, q/k rms_norm and the fixed q gain (CausalSelfAttention order); k and
    # v emitted in their K/V buffer dtypes (the per-op path's k.to(kb.dtype)).
    if ve_w is not None:
        ve = _table_gather(tok, ve_w, ve_s).view(v.shape)
        gate = 2 * torch.sigmoid(gate_logits)
        v = v + gate.unsqueeze(-1) * ve
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q) * QK_GAIN, norm(k)
    return q, k.to(k_dtype), v.to(v_dtype)


def _dec_add_norm_cast(x, y, gemv_dtype):
    # Attention residual add + the MLP's input rms_norm, xn in c_fc's dtype.
    x = x + y
    xn = norm(x)
    if gemv_dtype is not None:
        xn = xn.to(gemv_dtype)
    return x, xn


def _dec_relu2_cast(h, gemv_dtype):
    # MLP activation relu(h)^2, emitted in the mlp c_proj GEMV's dtype.
    h = F.relu(h).square()
    if gemv_dtype is not None:
        h = h.to(gemv_dtype)
    return h


def _dec_head_norm(x, y, gemv_dtype):
    # Last layer's deferred MLP residual add + final rms_norm, in lm_head's dtype.
    x = x + y
    xn = norm(x)
    if gemv_dtype is not None:
        xn = xn.to(gemv_dtype)
    return xn


_DECODE_FUSION_HELPERS = (
    # Per-op helpers (attempt 14 path; fallback if the regions fail to build).
    ("mix_norm", _dec_mix_norm),
    ("qk_post", _dec_qk_post),
    ("ve_mix", _dec_ve_mix),
    ("add_norm", _dec_add_norm),
    ("relu2", _dec_relu2),
    ("add", _dec_add),
    ("final_norm", _dec_final_norm),
    ("softcap", _dec_softcap),
    ("decode_attn", _dec_attn),
    # GEMV-bounded regions (attempt 22; softcap and decode_attn are shared).
    ("step_prep", _dec_step_prep),
    ("pre_attn", _dec_pre_attn),
    ("post_qkv", _dec_post_qkv),
    ("add_norm_cast", _dec_add_norm_cast),
    ("relu2_cast", _dec_relu2_cast),
    ("head_norm", _dec_head_norm),
)
_DECODE_MIN_RECOMPILE_LIMIT = 64  # Dynamo cache entries allowed per decode helper (see below)
_DECODE_FUSED = None  # process-wide compiled wrappers, created by the first adapter


def _decode_fused_kernels():
    # Each helper is wrapped ONCE per process with torch.compile in default mode
    # (dynamic=False, fullgraph=True; no reduce-overhead, so no Inductor cudagraph
    # trees: the adapter's own CUDA graph capture stays authoritative). All
    # adapters share these wrappers, so Dynamo cache entries (one per ambient
    # autocast/dtype variant) are not multiplied per adapter instance. Nothing
    # is compiled here; compilation happens on first call, in the adapter's
    # pre-capture warm-up step.
    global _DECODE_FUSED
    # The decode-attention helper is shape-specialized per slot count L (a
    # handful of ring sizes / linear capacities, times the ambient autocast
    # variants). With fullgraph=True, exceeding Dynamo's default limit of 8
    # cache entries per function raises instead of falling back, so raise the
    # limit (never lowered) for these inference-only wrappers. Changes no math;
    # executed only when an inference adapter is built (never during training).
    dcfg = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(dcfg, name):
            setattr(dcfg, name, max(int(getattr(dcfg, name)), _DECODE_MIN_RECOMPILE_LIMIT))
    if _DECODE_FUSED is None:
        _DECODE_FUSED = {
            name: torch.compile(fn, dynamic=False, fullgraph=True)
            for name, fn in _DECODE_FUSION_HELPERS
        }
    return _DECODE_FUSED


class ChunkedKVInferenceAdapter:
    """Cached inference with exactly the math of GPT.forward.

    Memory differences vs. the canonical path (same numbers, fewer temporaries):
    - prefill applies lm_head to the LAST position only and returns its
      softcapped fp32 logits (softcap applied in place) as a (B, 1, V) tensor,
      the same shape/dtype as decode_one, instead of a (B, T, V) fp32 output
      (the first decoded token needs only the last row); ADAPTER_LOGIT_CHUNK
      is no longer used;
    - K/V live in preallocated per-layer [B, cap, H_kv, D] buffers whose capacity
      grows in ADAPTER_KV_CHUNK-position chunks; decode writes one slot in place.
    - prefill attention (attempt 26) calls the training kernel fa3.flash_attn_func
      with the layer's window instead of masked SDPA, so neither the (T, T)
      banded bool masks nor dense (B, H, T, T) attention tensors are allocated.
    Dtypes follow the ambient autocast / parameter dtypes (no forced casts).

    Decode launch structure (per prefill state):
    - batch-1 decode is launch/dispatch bound, so each prefill captures ONE
      complete decode step (all blocks, final norm, lm_head, in-place softcap) as
      a CUDA graph over that state's own static K/V buffers and replays it per
      token. Position enters only through a device tensor (rotary index_select,
      K/V index_copy_, and a boolean key mask), so nothing in the graph depends
      on a Python position or syncs to the host.
    - full-context (linear) layers attend over the full buffer capacity with
      the mask (kpos <= p) & (kpos >= p - left), i.e. exactly the key set
      [max(0, p - left), p] of the sliced eager path. Slack slots are zeroed so
      masked keys/values are finite (0 * NaN would poison P @ V).
    - sliding-window (ring) layers: a layer whose left window satisfies
      left + 1 < linear capacity stores K/V in a fixed ring of
      R = ceil((left + 1) / ADAPTER_RING_ALIGN) * ADAPTER_RING_ALIGN slots
      (272 for left = 256) instead of the linear capacity. Position q lives in
      slot q mod R. Slot s holds position p - d with d = (p - s) mod R, and is
      kept iff d <= left and d <= p, so the attended key SET is exactly
      [max(0, p - left), p] (attention is invariant to key order). Never-
      written ring slots are zeroed (finite) and always masked. Ring buffers
      never grow; the ring slot index tensor arange(R) lives in io (counted).
    - capture runs under the ambient autocast dtype/enabled flag with
      cache_enabled=False, so the graph never references autocast-cached weight
      copies that are freed later; weight casts are part of the graph.
    - graphs/buffers are never shared across prefill states; growth of the
      linear (non-ring) layers past their capacity (never reached for
      1024 + 128) re-captures against the new buffers.
    - ALL per-state tensors (K/V buffers and the graph workspace
      tok/prev_tok/pos/kpos/out) live inside the state dict; cache_nbytes walks
      the state recursively and sums each unique storage once, matching the
      protected state traversal.
    - token shift (attempt 23): every path applies _smear_mix to the token
      embedding before the embedding norm. Decode reads the previous token from
      io["prev_tok"] (in step_prep / the per-op fallback) and the captured step
      ends with io["prev_tok"].copy_(tok); prefill sets it to the last prompt
      token and growth carries it over, both AFTER the graph build (whose
      warm-up step overwrites it).
    - gather-only tables (attempt 27): wte and the value embeddings are int8
      storage + per-row scale (GatherTable). Every lookup (prefill wte / VE via
      the table modules, step_prep tok / prev_tok, post_qkv VE, per-op
      fallback) goes through _table_gather, which dequantizes only the gathered
      rows; the regions receive the int8 table and its scale as arguments, so
      the dequant fuses into step_prep / post_qkv (no new kernels, no full-
      table copy). State, io keys, buffers and cache_nbytes are unchanged.

    Inference-memory policy (same math as above, fewer persistent allocations):
    - ONE persistent capture stream, created lazily and owned by the adapter, is
      used for both the capture warm-up step and the capture itself. A fresh
      side stream per prefill rotates through PyTorch's 32-stream pool,
      and the first GEMM on each new stream allocates a process-lifetime cuBLAS
      workspace (8.125 MiB on non-Hopper GPUs) -> +32 x 8.125 MiB in attempt 10.
      The stream object holds no tensor storage, so it is not part of the state.
    - prefill runs under a nested autocast with the ambient dtype/enabled flag
      and cache_enabled=False: bf16 weight casts are per-use temporaries (same
      casts, same numbers) instead of an autocast-cached copy of every block
      weight that stays alive through the fp32 prefill-logit phase.

    Decode kernel fusion (what the graph captures; graph, state, masks and
    cache_nbytes are unchanged):
    - batch-1 graph replay is bound by the number of serialized tiny kernels
      (~75 eager aten kernels per block: lambda mixing, decomposed rms_norms,
      rotary slices/products/cat, q/k norms, ve gating, residual adds, relu^2,
      softcap). Those cap-independent pointwise/norm regions run through the
      TorchInductor-compiled pure helpers above (fused Triton kernels).
    - decode attention (every layer, ring and linear) runs through the compiled
      split-key helper _dec_attn instead of masked SDPA, whose memory-efficient
      kernel launches only B x H = 4 CTAs that each scan every slot serially
      (cost linear in slots: ~0.048 us per slot per layer, attempt 17 -> 18).
      _dec_attn computes fp32 scores in parallel over all B x H x L slots,
      a masked fp32 softmax, and P @ V as per-chunk partial sums (parallel over
      B x H x (L / C) x D) followed by a sum over chunks: same key set, scale
      and softmax; K/V read in place (upcast on load, no fp32 K/V copy). Since
      attempt 26 the prefill attention is the training kernel itself,
      fa3.flash_attn_func(q, k, v, causal=True, window_size=window_sizes[i])
      (bf16/fp16 inputs), with no banded (T, T) masks and no dense masked-SDPA
      attention tensors; fp32 inputs, which FA3 cannot take, use an exact
      query-chunked banded SDPA (_prefill_attend_fp32).
    - compilation happens on the first call, inside the existing warm-up step on
      the persistent capture stream and under the same autocast context as the
      capture, so the captured region only replays already-compiled kernels
      (no compile, no sync, no .item() inside capture).
    - GEMV-bounded regions (attempt 22, _decode_step_regions): the eager glue
      the per-op helpers left between kernels (wte / value-embedding lookups,
      rotary index_selects, ring write slot and mask arithmetic, residual adds,
      autocast input casts) moves into compiled regions that each span the work
      between two consecutive GEMVs (step_prep, pre_attn, post_qkv,
      decode_attn, add_norm_cast, relu2_cast, head_norm, softcap). GEMVs stay
      cuBLAS and are fed inputs already in the dtype autocast would cast them
      to; K/V slot writes stay eager index_copy_; masks, key sets, QK_GAIN,
      softcap, state, io keys, buffers and cache_nbytes are unchanged. If the
      regions fail to build in a warm-up, the adapter keeps the per-op path.
    """

    def __init__(self, model):
        self.model = getattr(model, "_orig_mod", model)
        self.softcap = 15
        self._capture_stream = None  # lazily created, reused for every capture
        # Compiled decode-fusion helpers (wrapped once per process, shared).
        self._fused = _decode_fused_kernels()
        # GEMV-bounded decode regions are used unless building / first-calling
        # them raises in a pre-capture warm-up; then this adapter permanently
        # falls back to the per-op helper path (same math, same graph capture).
        self._regions_ok = True
        self._regions_error = None

    def _get_capture_stream(self, device):
        # Created once and reused, so warm-up + capture always run on the same
        # stream (one cuBLAS workspace for the adapter's whole lifetime).
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=device)
        return self._capture_stream

    @staticmethod
    def _capacity(n_positions):
        return ((n_positions + ADAPTER_KV_CHUNK - 1) // ADAPTER_KV_CHUNK) * ADAPTER_KV_CHUNK

    def _ring_sizes(self, cap):
        # Per layer: ring size R (> 0) for sliding-window layers whose window
        # (left + 1 keys) is smaller than the linear capacity `cap`, else 0
        # (linear, growable buffer). cap is a multiple of ADAPTER_RING_ALIGN,
        # so left + 1 < cap implies R <= cap.
        sizes = []
        for left, _ in self.model.window_sizes:
            if 0 <= left and left + 1 < cap:
                R = ((left + 1 + ADAPTER_RING_ALIGN - 1) // ADAPTER_RING_ALIGN) * ADAPTER_RING_ALIGN
                sizes.append(R)
            else:
                sizes.append(0)
        return tuple(sizes)

    @staticmethod
    def _grow(buf):
        B, cap, H, D = buf.shape
        new = buf.new_empty((B, cap + ADAPTER_KV_CHUNK, H, D))
        new[:, :cap].copy_(buf)
        new[:, cap:].zero_()  # masked slack must stay finite
        return new

    @staticmethod
    def _ambient_autocast():
        # (enabled, dtype) of the caller's CUDA autocast state; the graph follows it.
        if hasattr(torch, "get_autocast_dtype"):
            return torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")
        return torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()

    def _qkv(self, i, attn, xn, idx, cos, sin):
        # Identical to CausalSelfAttention.forward up to the attention call.
        B, T, _ = xn.shape
        q = attn.c_q(xn).view(B, T, attn.n_head, attn.head_dim)
        k = attn.c_k(xn).view(B, T, attn.n_kv_head, attn.head_dim)
        v = attn.c_v(xn).view(B, T, attn.n_kv_head, attn.head_dim)
        value_embeds = self.model.value_embeds
        if str(i) in value_embeds:
            ve = value_embeds[str(i)](idx).view(B, T, attn.n_kv_head, attn.head_dim)
            gate = 2 * torch.sigmoid(attn.ve_gate(xn[..., :attn.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * QK_GAIN, norm(k)
        return q, k, v

    def _prefill_attend(self, i, attn, q, k, v):
        # Prefill self-attention of layer i over the prompt (attempt 26).
        # q: (B, T, H, D); k, v: (B, T, H_kv, D), the exact layout
        # CausalSelfAttention.forward hands to FA3 -> (B, T, H, D).
        window_size = self.model.window_sizes[i]
        if q.dtype in (torch.bfloat16, torch.float16):
            # Exactly the training kernel and call: the module-global fa3 is
            # looked up at call time (as in CausalSelfAttention.forward), causal,
            # the layer's own (left, 0) window tuple, default 1/sqrt(head_dim)
            # scale, GQA native. Key set per query row i: [max(0, i - left), i].
            # No (T, T) boolean mask and no dense (B, H, T, T) score /
            # probability tensors (the former masked-SDPA prefill held two 1 MiB
            # banded masks plus ~32 MiB of dense attention tensors at T = 1024).
            return fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # FA3 has no fp32 kernel. fp32 inputs (autocast off with fp32 weights)
        # take an exact fp32 attention over the same key sets instead.
        return self._prefill_attend_fp32(attn, q, k, v, window_size[0])

    @staticmethod
    def _prefill_attend_fp32(attn, q, k, v, left):
        # Exact causal sliding-window attention (key j allowed iff
        # 0 <= i - j <= left; left < 0 = unbounded), the FA3 (left, 0) window
        # semantics, via F.scaled_dot_product_attention with the default scale
        # over ADAPTER_PREFILL_FP32_QCHUNK query rows at a time. Each chunk sees
        # keys [max(0, s - left), e), which contain every allowed key of its rows,
        # and a (rows, keys) band mask, so no (T, T) tensor is ever built. Every
        # row keeps at least its own key (no all-masked row).
        B, T, H, D = q.shape
        if left < 0:
            left = T
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if attn.n_kv_head != attn.n_head:
            rep = attn.n_head // attn.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = q.new_empty((B, H, T, D))
        for s in range(0, T, ADAPTER_PREFILL_FP32_QCHUNK):
            e = min(s + ADAPTER_PREFILL_FP32_QCHUNK, T)
            ks = max(0, s - left)
            qi = torch.arange(s, e, dtype=torch.long, device=q.device).view(-1, 1)
            kj = torch.arange(ks, e, dtype=torch.long, device=q.device).view(1, -1)
            band = (kj <= qi) & (kj >= qi - left)
            y[:, :, s:e] = F.scaled_dot_product_attention(
                q[:, :, s:e], k[:, :, ks:e], v[:, :, ks:e], attn_mask=band)
            del qi, kj, band
        return y.transpose(1, 2)

    def _attend_static(self, attn, q, kb, vb, attn_mask):
        # Capture-safe decode attention over the whole K/V buffer (ring R slots or
        # the linear capacity) for EVERY layer, through the compiled split-key
        # helper _dec_attn: q (B, 1, H, D) -> (B, H, 1, D); kb/vb are read in
        # their existing (B, slots, H_kv, D) layout (GQA by broadcast, no
        # repeat/expand copy); same boolean key mask and default 1/sqrt(head_dim)
        # scale as the former masked SDPA. Returns (B, 1, H, D).
        y = self._fused["decode_attn"](q.transpose(1, 2), kb, vb, attn_mask)
        return y.transpose(1, 2)

    def _decode_step_static(self, io):
        # The CUDA-graph body: one cached decode step with the math of the eager
        # decode path, reading the token/position from static device tensors.
        if self._regions_ok:
            return self._decode_step_regions(io)
        return self._decode_step_helpers(io)

    def _warmup_decode_step(self, io):
        # Pre-capture warm-up step (compiles any region variant on first call).
        # Safety net only: a build / first-call failure of the regions switches
        # this adapter to the per-op helper path for every later step/capture.
        if self._regions_ok:
            try:
                self._decode_step_regions(io)
                return
            except Exception as exc:
                self._regions_ok = False
                self._regions_error = f"{type(exc).__name__}: {exc}"
                import warnings
                warnings.warn("GEMV-bounded decode regions unavailable; using per-op "
                              f"decode helpers ({self._regions_error})")
        self._decode_step_helpers(io)

    def _region_plan(self, rings):
        # Static (Python, trace-time) description of the step: distinct ring
        # sizes, distinct ring masks (left, ring index), distinct linear-mask
        # lefts and each layer's (ring index or -1, mask index). Same keys as
        # the per-op path's write_slots / masks caches.
        ring_sizes = tuple(sorted({R for R in rings if R}))
        ring_keys, linear_lefts, select = [], [], []
        for i, R in enumerate(rings):
            left = self.model.window_sizes[i][0]
            if R:
                j = ring_sizes.index(R)
                key = (left, j)
                if key not in ring_keys:
                    ring_keys.append(key)
                select.append((j, ring_keys.index(key)))
            else:
                if left not in linear_lefts:
                    linear_lefts.append(left)
                select.append((-1, linear_lefts.index(left)))
        return ring_sizes, tuple(ring_keys), tuple(linear_lefts), tuple(select)

    def _decode_step_regions(self, io):
        # GEMV-bounded decode step: cuBLAS GEMVs (c_q, c_k, c_v, ve_gate,
        # attn c_proj, c_fc, mlp c_proj, lm_head) and the eager K/V index_copy_
        # slot writes, with ALL other work of the step in the compiled regions
        # step_prep / pre_attn / post_qkv / decode_attn / add_norm_cast /
        # relu2_cast / head_norm / softcap. Same masks, key sets, QK_GAIN
        # placement, softcap and dtype at every GEMV input as the per-op path.
        model = self.model
        fz = self._fused
        tok, pos = io["tok"], io["pos"]
        rings, ring_slots = io["rings"], io["ring_slots"]
        B = tok.size(0)
        enabled, autocast_dtype = self._ambient_autocast()
        gdt = autocast_dtype if enabled else None  # GEMV input dtype (None = no cast)
        ring_sizes, ring_keys, linear_lefts, select = self._region_plan(rings)
        wte = model.transformer.wte
        x0, cos, sin, write_slots, ring_masks, linear_masks = fz["step_prep"](
            tok, io["prev_tok"], pos, wte.weight, wte.scale,
            model.smear_gate_w, model.smear_lambda, model.cos, model.sin,
            tuple(ring_slots[R] for R in ring_sizes), ring_sizes, ring_keys,
            io.get("kpos") if linear_lefts else None, linear_lefts)
        value_embeds = model.value_embeds

        x = y = None  # residual stream / deferred MLP output of the previous layer
        for i, block in enumerate(model.transformer.h):
            attn = block.attn
            x, xn = fz["pre_attn"](x, y, x0, model.resid_lambdas[i], model.x0_lambdas[i], gdt)
            q = attn.c_q(xn).view(B, 1, attn.n_head, attn.head_dim)
            k = attn.c_k(xn).view(B, 1, attn.n_kv_head, attn.head_dim)
            v = attn.c_v(xn).view(B, 1, attn.n_kv_head, attn.head_dim)
            if str(i) in value_embeds:
                gate_logits = attn.ve_gate(xn[..., :attn.ve_gate_channels])
                ve_w, ve_s = value_embeds[str(i)].weight, value_embeds[str(i)].scale
            else:
                gate_logits = ve_w = ve_s = None
            kb, vb = io["k"][i], io["v"][i]
            q, k, v = fz["post_qkv"](q, k, v, cos, sin, gate_logits, ve_w, ve_s, tok, kb.dtype, vb.dtype)
            j, m = select[i]
            if j >= 0:
                # Ring layer: slot pos mod R; key set [max(0, p - left), p].
                slot, mask = write_slots[j], ring_masks[m]
            else:
                # Linear layer: slot pos; keys max(0, p - left) .. p of the capacity.
                slot, mask = pos, linear_masks[m]
            kb.index_copy_(1, slot, k)
            vb.index_copy_(1, slot, v)
            att = fz["decode_attn"](q.transpose(1, 2), kb, vb, mask, gdt)
            att = att.transpose(1, 2).contiguous().view(B, 1, -1)
            x, xn = fz["add_norm_cast"](x, attn.c_proj(att), gdt)
            h = fz["relu2_cast"](block.mlp.c_fc(xn), gdt)
            y = block.mlp.c_proj(h)
        xn = fz["head_norm"](x, y, gdt)

        # (B, 1, V) fp32 softcapped logits (graph-owned output, overwritten per replay)
        out = fz["softcap"](model.lm_head(xn))
        # Token shift: this step's token is the next step's previous token
        # (captured device copy, after step_prep has read prev_tok).
        io["prev_tok"].copy_(tok)
        return out

    def _decode_step_helpers(self, io):
        # Per-op helper decode step (attempt 21 path, unchanged; fallback only).
        # Pointwise/norm regions and the decode attention (split-key _dec_attn)
        # run through the compiled fusion helpers; GEMVs, embedding lookups,
        # rotary index_select, K/V index_copy_ and the masks stay eager aten calls.
        model = self.model
        fz = self._fused
        tok, pos = io["tok"], io["pos"]
        kpos = io.get("kpos")  # linear-layer key positions (absent if every layer is a ring)
        rings, ring_slots = io["rings"], io["ring_slots"]
        B = tok.size(0)
        cos = model.cos.index_select(1, pos)
        sin = model.sin.index_select(1, pos)
        masks = {}
        write_slots = {}  # per ring size R: device slot index pos mod R
        value_embeds = model.value_embeds

        # Token shift with the previous token's embedding, then the embedding norm.
        x = _smear_mix(model.transformer.wte(tok), model.transformer.wte(io["prev_tok"]),
                       model.smear_gate_w, model.smear_lambda)
        x = fz["final_norm"](x)
        x0 = x
        for i, block in enumerate(model.transformer.h):
            attn = block.attn
            x, xn = fz["mix_norm"](x, x0, model.resid_lambdas[i], model.x0_lambdas[i])
            q = attn.c_q(xn).view(B, 1, attn.n_head, attn.head_dim)
            k = attn.c_k(xn).view(B, 1, attn.n_kv_head, attn.head_dim)
            v = attn.c_v(xn).view(B, 1, attn.n_kv_head, attn.head_dim)
            if str(i) in value_embeds:
                ve = value_embeds[str(i)](tok).view(B, 1, attn.n_kv_head, attn.head_dim)
                v = fz["ve_mix"](v, attn.ve_gate(xn[..., :attn.ve_gate_channels]), ve)
            q, k = fz["qk_post"](q, k, cos, sin)
            kb, vb = io["k"][i], io["v"][i]
            left = model.window_sizes[i][0]
            R = rings[i]
            if R:
                # Ring layer: position p is written to slot p mod R. Slot s holds
                # position p - d, d = (p - s) mod R; keep iff d <= min(left, p),
                # i.e. exactly the key set [max(0, p - left), p].
                slot = write_slots.get(R)
                if slot is None:
                    slot = torch.remainder(pos, R)
                    write_slots[R] = slot
                kb.index_copy_(1, slot, k.to(kb.dtype))
                vb.index_copy_(1, slot, v.to(vb.dtype))
                mkey = ("ring", left, R)
                mask = masks.get(mkey)
                if mask is None:
                    d = torch.remainder(pos - ring_slots[R], R)
                    mask = ((d <= left) & (d <= pos)).view(1, 1, 1, -1)
                    masks[mkey] = mask
            else:
                kb.index_copy_(1, pos, k.to(kb.dtype))
                vb.index_copy_(1, pos, v.to(vb.dtype))
                # Exact sliding window: keys max(0, p - left) .. p of the full capacity
                mkey = ("linear", left)
                mask = masks.get(mkey)
                if mask is None:
                    mask = ((kpos <= pos) & (kpos >= pos - left)).view(1, 1, 1, -1)
                    masks[mkey] = mask
            y = self._attend_static(attn, q, kb, vb, mask)
            y = y.contiguous().view(B, 1, -1)
            x, xn = fz["add_norm"](x, attn.c_proj(y))
            h = fz["relu2"](block.mlp.c_fc(xn))
            x = fz["add"](x, block.mlp.c_proj(h))
        x = fz["final_norm"](x)

        # (B, 1, V) fp32 softcapped logits (graph-owned output, overwritten per replay)
        out = fz["softcap"](model.lm_head(x))
        # Token shift: this step's token is the next step's previous token.
        io["prev_tok"].copy_(tok)
        return out

    def _build_decode_graph(self, k_bufs, v_bufs, B, pos, rings):
        # Static decode workspace + captured graph for one state's K/V buffers.
        # The workspace dict is stored INSIDE the state, so every per-state
        # tensor (incl. the ring slot index tensors) is visible to (and counted
        # by) cache_nbytes. rings[i] is layer i's ring size (0 = linear layer).
        device = k_bufs[0].device
        rings = tuple(int(r) for r in rings)
        linear_layers = [i for i, R in enumerate(rings) if not R]
        io = {
            "tok": torch.zeros((B, 1), dtype=torch.long, device=device),
            # Previous token (token shift). The step body overwrites it with tok
            # at its end; the warm-up clobbers it, so callers set it AFTER this
            # build (prefill: last prompt token; growth: the old io's value).
            "prev_tok": torch.zeros((B, 1), dtype=torch.long, device=device),
            "pos": torch.full((1,), pos, dtype=torch.long, device=device),
            "k": list(k_bufs),
            "v": list(v_bufs),
            "rings": rings,
            # First linear layer (all linear layers share one capacity); -1 if none.
            "grow_layer": linear_layers[0] if linear_layers else -1,
            "ring_slots": {
                R: torch.arange(R, dtype=torch.long, device=device)
                for R in sorted({R for R in rings if R})
            },
        }
        if linear_layers:
            cap = k_bufs[linear_layers[0]].size(1)
            io["kpos"] = torch.arange(cap, dtype=torch.long, device=device)
        enabled, dtype = self._ambient_autocast()
        with torch.autocast("cuda", dtype=dtype, enabled=enabled, cache_enabled=False):
            # Warm-up on the adapter's persistent capture stream (lazy library
            # init, incl. that stream's cuBLAS workspace, and any first-call
            # TorchInductor compilation of the fusion helpers for this autocast
            # context, all outside capture). It writes slot `pos` (linear layers)
            # / `pos mod R` (ring layers; that slot's old position pos - R is
            # outside every future window), which the first real decode overwrites.
            side = self._get_capture_stream(device)
            current = torch.cuda.current_stream()
            side.wait_stream(current)
            with torch.cuda.stream(side):
                self._warmup_decode_step(io)
            current.wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            # Capture on the SAME stream as the warm-up: one shared workspace.
            with torch.cuda.graph(graph, stream=side, capture_error_mode="thread_local"):
                out = self._decode_step_static(io)
        io["out"] = out  # graph-owned (B, 1, V) fp32 output, overwritten per replay
        return graph, io

    def prefill(self, input_ids):
        # Same ambient autocast dtype/enabled flag, but with the autocast weight
        # cache disabled for the whole prefill: identical bf16 casts, issued as
        # per-use temporaries rather than cached copies kept alive through the
        # fp32 logit phase.
        enabled, dtype = self._ambient_autocast()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=enabled, cache_enabled=False):
            model = self.model
            idx = torch.as_tensor(input_ids, device=model.cos.device)
            if idx.dim() == 1:
                idx = idx.view(1, -1)
            B, T = idx.shape
            assert T <= model.cos.size(1)
            cos, sin = model.cos[:, :T], model.sin[:, :T]
            cap = self._capacity(T + 1)
            rings = self._ring_sizes(cap)
            k_bufs, v_bufs = [], []

            x = model.transformer.wte(idx)
            # Token shift, identical to GPT.forward (position 0 unchanged).
            x = torch.cat([x[:, :1], _smear_mix(x[:, 1:], x[:, :-1], model.smear_gate_w, model.smear_lambda)], dim=1)
            x = norm(x)
            x0 = x
            for i, block in enumerate(model.transformer.h):
                attn = block.attn
                x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
                xn = norm(x)
                q, k, v = self._qkv(i, attn, xn, idx, cos, sin)
                R = rings[i]
                if R:
                    # Ring layer: keep only the last R positions max(0, T - R) .. T - 1,
                    # position q in slot q mod R (distinct slots); never-written
                    # slots stay zero (finite, and always masked in decode).
                    kb = k.new_zeros((B, R, attn.n_kv_head, attn.head_dim))
                    vb = v.new_zeros((B, R, attn.n_kv_head, attn.head_dim))
                    s0 = max(0, T - R)
                    slots = torch.remainder(torch.arange(s0, T, dtype=torch.long, device=idx.device), R)
                    kb.index_copy_(1, slots, k[:, s0:T])
                    vb.index_copy_(1, slots, v[:, s0:T])
                    del slots
                else:
                    kb = k.new_empty((B, cap, attn.n_kv_head, attn.head_dim))
                    vb = v.new_empty((B, cap, attn.n_kv_head, attn.head_dim))
                    kb[:, :T].copy_(k)
                    vb[:, :T].copy_(v)
                    kb[:, T:].zero_()  # masked slack must stay finite
                    vb[:, T:].zero_()
                k_bufs.append(kb)
                v_bufs.append(vb)
                # Same attention call as GPT.forward: fa3 causal with this
                # layer's (left, 0) window (key j allowed iff 0 <= i - j <= left);
                # no prefill mask is built (see _prefill_attend).
                y = self._prefill_attend(i, attn, q, k, v)
                y = y.contiguous().view(B, T, -1)
                x = x + attn.c_proj(y)
                x = x + block.mlp(norm(x))
            x = norm(x)

            # Capture this state's decode graph BEFORE the prefill logits are
            # computed, so the capture transient never stacks on them.
            graph, io = self._build_decode_graph(k_bufs, v_bufs, B, T, rings)
            # The warm-up step overwrote prev_tok; the first decoded position T
            # shifts in the last prompt token.
            io["prev_tok"].copy_(idx[:, -1:])

            # Last-position logits only: (B, 1, V) fp32 softcapped, the same
            # shape/dtype as decode_one's output and the only row needed to
            # start decoding. Same lm_head math and in-place softcap
            # (15 * tanh(x / 15)) as GPT.forward at position T - 1; no (B, T, V)
            # fp32 buffer (32 MiB at T = 1024, V = 8192) and no chunk loop.
            logits = model.lm_head(x[:, -1:]).float()
            logits.div_(self.softcap).tanh_().mul_(self.softcap)
            state = {"k": k_bufs, "v": v_bufs, "pos": T, "graph": graph, "io": io}
            return logits, state

    def decode_one(self, one_token, state):
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
            graph, io = state["graph"], state["io"]
            assert tuple(tok.shape) == tuple(io["tok"].shape)

            g = io["grow_layer"]
            if g >= 0 and p >= k_bufs[g].size(1):
                # Generic capacity policy: grow every LINEAR layer by
                # ADAPTER_KV_CHUNK slots (ring buffers never grow) and re-capture
                # against the new buffers (new state only).
                rings = io["rings"]
                old_io = io
                k_bufs = [kb if R else self._grow(kb) for kb, R in zip(k_bufs, rings)]
                v_bufs = [vb if R else self._grow(vb) for vb, R in zip(v_bufs, rings)]
                graph, io = self._build_decode_graph(k_bufs, v_bufs, B, p, rings)
                # The new graph's warm-up clobbered its prev_tok: carry over the
                # old state's previous token (the token fed at position p - 1).
                io["prev_tok"].copy_(old_io["prev_tok"])
                del old_io

            io["tok"].copy_(tok)
            io["pos"].fill_(p)
            graph.replay()
            # Clone so the caller never holds a buffer that later replays overwrite.
            logits = io["out"].clone()
            return logits, {"k": k_bufs, "v": v_bufs, "pos": p + 1, "graph": graph, "io": io}

    def cache_nbytes(self, state):
        # Exact live storage reachable from the state: recursively walk dict
        # values and list/tuple items, collect every torch.Tensor (non-tensors
        # such as the CUDAGraph object and ints hold no counted storage), and sum
        # each unique storage once. io["k"]/io["v"] alias the K/V buffers and are
        # deduplicated; the graph workspace tok/prev_tok/pos/kpos/out and the ring
        # slot index tensors (io["ring_slots"] values) are counted.
        seen = set()
        total = 0
        stack = [state]
        while stack:
            obj = stack.pop()
            if isinstance(obj, torch.Tensor):
                storage = obj.untyped_storage()
                key = storage.data_ptr()
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
            elif isinstance(obj, dict):
                stack.extend(obj.values())
            elif isinstance(obj, (list, tuple)):
                stack.extend(obj)
        return int(total)


# ---------------------------------------------------------------------------
# Phase 5B inference-only ablation (inserted by scripts/phase5b_inference_ablation.py).
# Everything above this block is the frozen candidate source, byte for byte; only
# the adapter factory differs. Same model, same weights, same decode math; one
# execution mechanism is switched (see PHASE5B_SETTINGS).
# ---------------------------------------------------------------------------
PHASE5B_VARIANT = 'V3_sdpa_attention'
PHASE5B_SETTINGS = {'cuda_graph': True, 'compile_regions': True, 'sdpa_decode_attention': True, 'ring_kv_cache': True}


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
        io = {
            "tok": torch.zeros((B, 1), dtype=torch.long, device=device),
            "prev_tok": torch.zeros((B, 1), dtype=torch.long, device=device),
            "pos": torch.full((1,), pos, dtype=torch.long, device=device),
            "k": list(k_bufs),
            "v": list(v_bufs),
            "rings": rings,
            "grow_layer": linear_layers[0] if linear_layers else -1,
            "ring_slots": {
                R: torch.arange(R, dtype=torch.long, device=device)
                for R in sorted({R for R in rings if R})
            },
        }
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
            return logits, {"k": k_bufs, "v": v_bufs, "pos": p + 1, "graph": None, "io": io}


def build_inference_adapter(model):
    return Phase5BAblationAdapter(model)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # (unused) legacy depth->width ratio; width is now fixed by MODEL_DIM
MODEL_DIM = 512         # fixed model width, decoupled from DEPTH so depth changes do not shrink width
HEAD_DIM = 128         # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=1/8 context (depth 3 -> windows 256, 256, 2048)

# Optimization
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step (grad_accum 1 at DEVICE_BATCH_SIZE 64; ~2x more updates than 2**18 in the fixed 300 s budget)
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 3               # number of transformer layers (8 -> 6 -> 4 -> 3 at fixed width 512)
DEVICE_BATCH_SIZE = 64  # A40-compatible; total batch size remains unchanged

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

from study_runtime import contract_progress

CONTRACT_ID = "T"
CONTRACT_TIME_BUDGET_S = 300.0  # protected Contract-T measured training budget
# Margin on the candidate-side fallback stop. The candidate clock can run
# slightly ahead of the parent's observer clock, so a fallback break exactly at
# 300 s can return before the protected boundary stop (attempt-02's
# CandidateExitedBeforeStop). With this margin the protected stop always ends
# training under protected execution; the fallback only matters standalone.
STOP_GUARD_S = 10.0

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    model_dim = MODEL_DIM
    assert MODEL_DIM % HEAD_DIM == 0
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
# Training keeps fp32 block/lm_head weights exactly as attempt 11: switch the
# declared bf16 inference storage to fp32 BEFORE init_weights draws.
model.set_matrix_dtype(torch.float32)
# Training keeps trainable bf16 gather-only tables exactly as attempt 26: switch
# the declared int8 inference storage to bf16 BEFORE init_weights (same
# Parameter objects -> same Adam groups; fp32 draws rounded to bf16 as before).
model.set_table_dtype(TRAIN_TABLE_DTYPE)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
assert TOTAL_BATCH_SIZE in (32768, 65536, 131072, 262144, 524288)
assert DEVICE_BATCH_SIZE * MAX_SEQ_LEN * grad_accum_steps == TOTAL_BATCH_SIZE
assert all(p.dtype == torch.float32 for p in list(model.transformer.h.parameters()) + list(model.lm_head.parameters())), \
    "training must keep fp32 block/lm_head weights"
assert all(t.weight.dtype == TRAIN_TABLE_DTYPE and t.weight.requires_grad for t in model.gather_tables()), \
    "training must keep trainable bf16 gather-only tables"

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {CONTRACT_TIME_BUDGET_S}s (Contract {CONTRACT_ID}; prepare.TIME_BUDGET={TIME_BUDGET})")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
step = 0

# Contract-T measured training clock: synchronized perf_counter started
# immediately before the first scientific forward. Every step counts (no
# warm-up exclusion); the trusted parent's clock remains authoritative.
torch.cuda.synchronize()
t_train_start = time.perf_counter()
measured_training_time = 0.0  # synchronized elapsed time at the last completed update
total_training_time = 0.0

while True:
    t0 = time.perf_counter()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Protected contract boundary: declare the next optimizer update and use
    # the returned progress for all progress-based schedules.
    tokens_so_far = (step + 1) * TOTAL_BATCH_SIZE
    progress = contract_progress("T", measured_training_time, tokens_so_far)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    dt = t1 - t0
    measured_training_time = t1 - t_train_start
    total_training_time = measured_training_time

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, CONTRACT_TIME_BUDGET_S - measured_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Fallback candidate-side stop after a completed update, unreachable under
    # protected execution (protected stop fires at 300 s on the parent clock).
    # The protected boundary stop is intentionally never caught here.
    if measured_training_time >= CONTRACT_TIME_BUDGET_S + STOP_GUARD_S:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE
optimizer_updates = step
microbatches = step * grad_accum_steps
val_bpb = float('nan')  # diagnostic only; trusted parent owns validation

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * step / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"optimizer_updates: {optimizer_updates}")
print(f"microbatches:     {microbatches}")
print(f"total_batch_size: {TOTAL_BATCH_SIZE}")
print(f"grad_accum_steps: {grad_accum_steps}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")

# Inference storage precision: after the last completed update and outside the
# measured clock, store the block/lm_head matrices in bf16 (the dtype every
# inference matmul already uses). A fresh model host declares the same dtype.
getattr(model, "_orig_mod", model).set_matrix_dtype(torch.bfloat16)
# Gather-only tables (attempt 27): quantize wte / value embeddings in place to
# int8 storage + per-row bf16 scale (s = rowabsmax / 127, q = round(w / s)),
# the dtype a fresh model host declares; same Parameter objects and keys.
getattr(model, "_orig_mod", model).set_table_dtype(INFERENCE_TABLE_DTYPE)

# Protected launch-interface exports (val_bpb is diagnostic only).
model, config, val_bpb, total_tokens, total_training_time, step, TOTAL_BATCH_SIZE, peak_vram_mb, num_params, optimizer_updates, microbatches = (
    model, config, val_bpb, total_tokens, total_training_time, step, TOTAL_BATCH_SIZE, peak_vram_mb, num_params, optimizer_updates, microbatches
)
