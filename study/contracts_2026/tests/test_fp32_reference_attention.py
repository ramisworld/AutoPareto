"""FP32 correctness adjudication of a candidate-supplied adapter whose forward calls FlashAttention.

The defect (attempt 8 of the Opus campaign, 2026-11-05): FP32 adjudication used the candidate's own
GPT.forward as the reference, and that forward calls FlashAttention, which rejects FP32 inputs. The fix runs
those calls through an exact FP32 stand-in during the adjudication only. These tests check the stand-in
against a direct implementation of FlashAttention's documented semantics, against the registered FlashAttention
kernel itself on the A40, and the swap's scope (candidate adapters only, restored afterwards).
"""

from __future__ import annotations

import os
import types
import unittest

import torch

from protected.inference_adapter import CanonicalGPTAdapter
from study.contracts_2026.inference_benchmark import (
    _FP32ReferenceFlashAttention,
    fp32_flash_attention_reference,
    fp32_reference_attention,
)


def direct_flash_semantics(q, k, v, *, softmax_scale=None, causal=False, window_size=(-1, -1)):
    """Row-by-row restatement of flash_attn_func's docstring (bottom-right aligned mask and window)."""
    batch, q_len, heads, dim = q.shape
    k_len, kv_heads = k.shape[1], k.shape[2]
    scale = softmax_scale if softmax_scale is not None else dim ** -0.5
    left, right = window_size
    if causal:
        right = 0
    out = torch.zeros_like(q)
    for b in range(batch):
        for h in range(heads):
            kv = h // (heads // kv_heads)
            for i in range(q_len):
                aligned = i + k_len - q_len
                keys = [j for j in range(k_len)
                        if (right < 0 or j <= aligned + right) and (left < 0 or j >= aligned - left)]
                if not keys:
                    continue
                scores = torch.stack([(q[b, i, h] * k[b, j, kv]).sum() * scale for j in keys])
                weights = torch.softmax(scores, dim=0)
                out[b, i, h] = sum(w * v[b, j, kv] for w, j in zip(weights, keys))
    return out


class ReferenceSemanticsTests(unittest.TestCase):
    def check(self, q_len, k_len, heads, kv_heads, **kwargs):
        torch.manual_seed(q_len * 100 + k_len)
        q = torch.randn(2, q_len, heads, 8, dtype=torch.float64)
        k = torch.randn(2, k_len, kv_heads, 8, dtype=torch.float64)
        v = torch.randn(2, k_len, kv_heads, 8, dtype=torch.float64)
        got = fp32_flash_attention_reference(q, k, v, **kwargs)
        want = direct_flash_semantics(q, k, v, **kwargs)
        self.assertEqual(got.shape, q.shape)
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)

    def test_causal_sliding_windows_as_in_the_candidates(self):
        for left in (-1, 0, 1, 3, 8, 16):
            with self.subTest(left=left):
                self.check(12, 12, 4, 4, causal=True, window_size=(left, 0))

    def test_decode_and_suffix_queries_are_bottom_right_aligned(self):
        self.check(1, 12, 4, 4, causal=True, window_size=(5, 0))
        self.check(3, 12, 4, 4, causal=True, window_size=(-1, -1))

    def test_more_queries_than_keys_gives_zero_rows_like_flash_attention(self):
        q = torch.randn(1, 5, 2, 4, dtype=torch.float64)
        k = torch.randn(1, 2, 2, 4, dtype=torch.float64)
        out = fp32_flash_attention_reference(q, k, k.clone(), causal=True)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(bool((out[:, :3] == 0).all()))
        torch.testing.assert_close(out, direct_flash_semantics(q, k, k.clone(), causal=True))

    def test_grouped_query_heads_scale_and_two_sided_window(self):
        self.check(10, 10, 4, 2, causal=True, window_size=(3, 0))
        self.check(10, 10, 4, 1, softmax_scale=0.17, causal=False, window_size=(2, 1))
        self.check(10, 10, 4, 4, causal=False)


class ShimTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        test = self

        class FakeFlash:
            """Like the registered kernel: half precision only."""

            def flash_attn_func(self, q, k, v, softmax_scale=None, causal=False, window_size=(-1, -1), **options):
                if q.dtype == torch.float32:
                    raise RuntimeError("FlashAttention only supports fp16, bf16, and fp8_e4m3 data type")
                test.calls.append((q.dtype, causal, window_size, options))
                return q

        self.fake = FakeFlash()

    def test_fp32_uses_the_reference_and_half_precision_goes_to_the_kernel(self):
        shim = _FP32ReferenceFlashAttention(self.fake)
        q = torch.randn(1, 6, 2, 4)
        torch.testing.assert_close(
            shim.flash_attn_func(q, q, q, causal=True, window_size=(2, 0), num_splits=1, pack_gqa=None),
            fp32_flash_attention_reference(q, q, q, causal=True, window_size=(2, 0)),
        )
        self.assertEqual(shim.fp32_calls, 1)
        half = q.to(torch.bfloat16)
        self.assertIs(shim.flash_attn_func(half, half, half, causal=True, window_size=(2, 0)), half)
        self.assertEqual(self.calls, [(torch.bfloat16, True, (2, 0), {})])

    def test_fp32_options_that_change_the_math_are_refused(self):
        shim = _FP32ReferenceFlashAttention(self.fake)
        q = torch.randn(1, 4, 2, 4)
        for option in ({"softcap": 30.0}, {"return_attn_probs": True}, {"attention_chunk": 64}, {"qv": q}):
            with self.subTest(option=sorted(option)), self.assertRaises(RuntimeError):
                shim.flash_attn_func(q, q, q, causal=True, **option)

    def candidate_module(self):
        module = types.ModuleType("candidate_train")
        module.fa3 = self.fake
        module.flash_attn_func = self.fake.flash_attn_func  # a bound kernel function imported by name
        module.torch = torch

        def forward(q):
            return module.fa3.flash_attn_func(q, q, q, causal=True, window_size=(2, 0))

        module.forward = forward
        return module

    def test_candidate_adapter_swap_is_scoped_and_restored(self):
        module = self.candidate_module()
        original_fa3, original_func = module.fa3, module.flash_attn_func
        q = torch.randn(1, 6, 2, 4)
        with self.assertRaises(RuntimeError):
            module.forward(q)  # the defect: FP32 into FlashAttention
        with fp32_reference_attention(module, object()) as record:
            torch.testing.assert_close(
                module.forward(q), fp32_flash_attention_reference(q, q, q, causal=True, window_size=(2, 0)))
        self.assertEqual(record["mode"], "candidate_forward_with_fp32_flash_attention_reference")
        self.assertEqual(record["replaced_globals"], ["fa3", "flash_attn_func"])
        self.assertEqual(record["fp32_flash_attention_calls"], 1)
        self.assertIs(module.fa3, original_fa3)
        self.assertIs(module.flash_attn_func, original_func)
        with self.assertRaises(ValueError):
            with fp32_reference_attention(module, object()):
                raise ValueError("adjudication failed")
        self.assertIs(module.fa3, original_fa3)

    def test_canonical_adapter_and_attention_free_candidates_are_untouched(self):
        module = self.candidate_module()
        original = module.fa3
        canonical = CanonicalGPTAdapter.__new__(CanonicalGPTAdapter)
        with fp32_reference_attention(module, canonical) as record:
            self.assertIs(module.fa3, original)
        self.assertEqual(record, {"mode": "protected_canonical_sdpa_prefill"})
        plain = types.ModuleType("sdpa_candidate")
        plain.torch = torch
        with fp32_reference_attention(plain, object()) as record:
            pass
        self.assertEqual(record["mode"], "candidate_forward_unchanged")
        self.assertEqual(record["replaced_globals"], [])


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("HF_HUB_OFFLINE") == "1",
                     "needs the A40 and the registered kernel cache (HF_HUB_OFFLINE=1)")
class RegisteredFlashAttentionTests(unittest.TestCase):
    """The stand-in in FP32 against the registered FlashAttention kernel in BF16, on the candidates' shapes."""

    def test_matches_the_registered_kernel_on_candidate_shapes(self):
        from kernels import get_kernel

        capability = torch.cuda.get_device_capability()
        repo = "varunneal/flash-attention-3" if capability == (9, 0) else "kernels-community/flash-attn3"
        flash = get_kernel(repo).flash_attn_interface
        torch.manual_seed(0)
        for q_len, window in ((2048, (1024, 0)), (2048, (2048, 0)), (1024, (512, 0))):
            with self.subTest(q_len=q_len, window=window):
                q, k, v = (torch.randn(1, q_len, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
                kernel = flash.flash_attn_func(q, k, v, causal=True, window_size=window).float()
                with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                    reference = fp32_flash_attention_reference(
                        q.float(), k.float(), v.float(), causal=True, window_size=window)
                error = (kernel - reference).abs()
                # BF16 output rounding (8 mantissa bits) bounds the difference; a wrong mask would be O(1).
                self.assertLess(float(error.max()), 2e-2)
                self.assertLess(float(error.mean()), 2e-3)


if __name__ == "__main__":
    unittest.main()
