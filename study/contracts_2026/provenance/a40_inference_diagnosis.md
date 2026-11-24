# A40 inference-instrument diagnosis

This is a diagnostic record only. It does not calibrate `N_STAR`, generate the
seed-3 baseline, launch a research attempt, or change the frozen correctness or
stability thresholds.

## Remote harness integrity

The remote checkout was `/workspace/autopareto` at commit
`5bf86b572d5a729aaad0163103cf15a7b42797a9`; the local checkout was at commit
`6bcb69e6036d64bb0eca9b5b07d0b0fde1c58abf`. The executed study files matched
byte-for-byte despite the differing repository commits. The following exact
hashes were checked locally and remotely:

| file | SHA256 |
|---|---|
| `study/contracts_2026/train_launcher.py` | `55e96dda7d458a5b45f3da97fe2262638da6db1665846d2c60f1e13be9f928a7` |
| `study/contracts_2026/protected_runtime.py` | `44d5f509e1af8d9a13f209296bef7173f58ef0b811c03a9d659aed0f54126eb2` |
| `study/contracts_2026/study_runtime.py` | `63481a4c58b6d7806febb3f60c91666486950cf3b94c4cdd6f41625ab8451de0` |
| `study/contracts_2026/candidate_worker.py` | `bc4a28e6ba720f6e14ca874b1d5eb496ebc28d0bf2707a56c8a0e2ec798e669e` |
| `study/contracts_2026/runtime.py` | `aed3172a3caff4b11c3e40506bc2cf0c14e2e08be01a29a0a0ad3f0cba76acdc` |
| `study/contracts_2026/contracts.py` | `1d4fdca79e32091c59f64e490d82cad1278e6f87b3425f5f937b666823c7f57f` |
| `study/contracts_2026/batching.py` | `a5d80c8f28c0f35f781434fd1118c86dabc51b77d595bcdd1be05a34aee013e6` |
| `study/contracts_2026/benchmark_policy.py` | `83123f64676aa4976d2f51c03d60063eabc930ee8fb199bb15173c786cb24f62` |
| `study/contracts_2026/inference_benchmark.py` | `96d40248303a6702d1551d9491ee7d2a07b5467c221f45fabd0a5fe19c0f1f07` |
| `protected/inference_adapter.py` | `06c72ea8d9502bfdccf0801b8d9f86e96fe13563bf980336a0924f79c606e448` |
| `study/contracts_2026/campaign.py` | `cf1dea0932a7d9b104b0a34958c6abf25543b6a9c426aa02761e2ced68e168e9` |
| `study/contracts_2026/context.py` | `fb4efe24208f683e410d99f4ced8a0219cfdabde3979e18d12a266657a42f08c` |
| `study/contracts_2026/bridge.py` | `1db0065537a251f41e3ec6d024fca558d15cde1b8b64170a61382242078fe70f` |
| `study/contracts_2026/remote_runner.py` | `5619b7e04e8544b082d281eff9599d0c9d5d2ad65d5ecbbf9651a95926da5c35` |
| `study/contracts_2026/runner.py` | `1c35e267e2c2d1a4db3ad770d37e8c0f9c566103445527e37d2752d42b2d47b4` |
| `study/contracts_2026/pair_registration.json` | `29e7f87ad4d58e6e15b11cb6a7c359122805340a3713e1148981056d5a275c67` |
| `study/contracts_2026/protocol.md` | `da6327024c02826423ec6afc2bfbdc910e53b03949c17a2215bae3c1df5e3a23` |
| `study/contracts_2026/context_allowlist.json` | `4cc6898a3f975f306dc7112a7d49e6ccfa3b238c35e8f50a87554f702ed81bec` |
| `study/contracts_2026/programs/program_fixed_time.md` | `99ff64003b8695d56fc829829139ebf950bd80f2c0480d2703261157f1e81d93` |
| `study/contracts_2026/programs/program_fixed_tokens.md` | `852a82e7ffb26568d0d82dd87acf0114569b62f53c575dae36425937dcf8217a` |
| `study/contracts_2026/tools/diagnose_correctness.py` | `32410e4f99e0c0b05aeb5930138aa07d933cbc913c9e4361e4f729a67348367d` |
| `study/contracts_2026/tools/diagnose_timing.py` | `d46a9ddcdbd2f8bc5b1e0c6af02bd7b0fc833d417d2c6159348a29dd49d07577` |

The two diagnostic tools were transferred after their final local hashes were
computed, so the remote copies also match those hashes. The source files used
for the four frozen models had exact local/remote hashes: baseline
`428e22f114040f9fa9dd2d4da8b1ad005d7aa0be4dd3b395931c4537ba82f6b8`, AP292
`55a7fd8bf32924f2a35ebc095c5206f3503e671a9d7a9646b94979c8ae49259a`, AP299
`36e3ad91a62810e4a907c1465adfbd1f07da446d0e63570c9e6a944b6066f12b`, and
AutoResearch `b3a961eeea05dcec17c49cb1fa392dae70c1c87d520409da43d4f3fb9fd6c5b1`.

## AutoResearch correctness

Checkpoint `preflight_frozen_checkpoints/autoresearch_0003.pt` has SHA256
`87f5094983f7c3ae78685aee7121e8c3b5ebe11cfbe51eb71334cf9337710dc8` and was
paired with the exact AutoResearch source above. Its protected metadata reports
seed `2`, `train_file=train.py`, and config
`n_embd=512, n_head=4, n_kv_head=4, n_layer=8, sequence_len=2048,
vocab_size=8192, window_pattern=SSSS`. The source derives the same config from
depth 8, aspect ratio 64, head dimension 128, and the frozen tokenizer.

The selected adapter was `CanonicalGPTAdapter`. The protected full-prefix SDPA
reference and one-token cached SDPA path were compared at prefill and every one
of 32 teacher-forced decode steps. BF16 autocast had greedy-token agreement at
all steps but maximum mean absolute logit error `0.04805232584476471`, peaking
at step 14 (`max_abs_error=0.1528792381286621`). FP32 had greedy-token agreement
at all steps and maximum mean error `2.1801693037559744e-06`. The candidate's
own BF16 FA3 forward versus cached SDPA comparison had maximum mean error
`0.040338724851608276`, also with greedy-token agreement. Candidate FA3 FP32
was unsupported by the kernel (`FlashAttention only supports fp16, bf16, and
fp8_e4m3`).

This is expected backend/numerical-ordering sensitivity in BF16, not evidence of
a wrong source/checkpoint/config pairing or a cache-state implementation error.
The `0.03` correctness criterion was not changed. The raw trace is retained on
the A40 at `preflight_correctness/autoresearch_trace_backends.json`, SHA256
`5a37507cab487f2013aff25fe6a63e3b3e6c7731427137cb5e54ab862f2eae4c`.

## Timing sessions

The final unchanged protocol was run in six fresh Python processes in reversed
model orders: session 01 baseline/AP292/AP299 and session 02 AP299/AP292/baseline.
Every raw warm-up and measured repetition includes synchronized host timing,
CUDA-event timing, allocated/reserved memory, exact KV bytes, GPU telemetry,
compute-process listing, and benchmark PID/process start state.

| session/model | warm-up medians (tokens/s) | measured medians (tokens/s) | result |
|---|---|---|---|
| 01 baseline | 142.247, 157.116, 195.417, 194.788, 197.176 | 192.892, 194.101, 195.233 | unstable: measured pass 3 failed drift |
| 01 AP292 | 315.763, 302.889, 312.934, 380.516, 380.358, 365.247 | — | unstable: warm-up failed |
| 01 AP299 | 390.767, 390.555, 402.019, 396.571, 406.510, 460.184 | — | unstable: warm-up failed |
| 02 AP299 | 403.734, 358.554, 406.599, 365.469, 396.286, 396.914 | — | unstable: warm-up failed |
| 02 AP292 | 277.702, 304.331, 303.271, 306.398 | 298.834, 308.274, 311.070 | unstable: all measured passes failed CV |
| 02 baseline | 159.682, 195.910, 197.775, 194.763, 199.504, 194.431 | — | unstable: warm-up failed |

The original frozen benchmark without telemetry also rejected AP292 in a fresh
control: warm-up medians were approximately 325.1, 383.8, 384.6, 372.0, 388.9,
and 390.9 tokens/s; no measured scalar was accepted.

Remote raw evidence and hashes:

* `session01_baseline.json`: `4e9f42f8da94d461f07762d1dd4260e869a53c834ac28ad38d271bfc6ac3a004`
* `session01_ap292.json`: `3943561774049d50451f07e27af2275b156cb2f220079dd9846ff2570a633fd4`
* `session01_ap299.json`: `bc64a1cd22821a681a7301b42dc499f3bc20ccec64c2306b18771cace765e018`
* `session02_baseline.json`: `a1c3895e5478facef91a4e1fec349ff12deec4c46da7d946ca406532dae16b1f`
* `session02_ap292.json`: `e0af08f702b162bf895a7cf3dd925dac963f42e3ffb1eedfa6158af8ad7c9635`
* `session02_ap299.json`: `780a4163029440a58fa071259ab06d15532d15290b397af02e0639abc73aa2c1`
* original AP292 control: `804af5e18f2e1ceee65e190567b9e3fb410a3af3c35e161ee3c796242030f2b2`

## GPU findings

The A40 reported NVIDIA driver `580.159.04`, application clocks `1740 MHz SM`
and `7251 MHz memory`, and P0 during active repetitions. Across the diagnostic
telemetry, temperatures were 32–39°C, power draw was approximately 95–116 W,
and no HW thermal slowdown, HW power brake, or active SW power cap was reported.
The only compute process was the benchmark's own PID; no competing GPU process
was observed. CUDA-event and synchronized host throughput ranges tracked one
another, so the instability is not a host-timer artifact. The A40 was already in
persistence mode. A requested `nvidia-smi -lgc 1740,1740` diagnostic was refused
by the device/container (`current user does not have permission to change clocks`);
no GPU state was changed.

The evidence is consistent with low-utilization decode kernel/scheduling or
runtime variability, including warm-up ramp and short-lived bimodal timing
episodes, not clock drift, thermal throttling, or another visible GPU process.

## Protocol decision

The current protocol is not reliable enough for autonomous search: five of six
fresh diagnostic sessions were rejected before a usable scalar, and the one
baseline session that reached measured passes still failed its final drift gate.
No threshold was loosened.

Proposal for review only, not frozen: preserve the correctness gate, raw
repetitions, CUDA synchronization, allocated/reserved/KV measurements, and the
existing `CV <= 0.05` / median-drift `<= 0.02` acceptance rules, but increase the
common decode work per repetition (for example by timing four independent
registered 256-token decode sequences in one protected repetition and dividing
the aggregate token count by aggregate time). This is identical for all
candidates and contracts, preserves per-sequence memory/KV accounting, and
tests whether longer GPU occupancy averages the observed scheduling episodes.
It must be validated on fresh baseline/AP292/AP299 sessions before adoption.

## Bridge

The provider-level harmless preflight is recorded at
`study/contracts_2026/provenance/provider_bridge_preflight.json`. It reports exact
visibility of the five allowlisted files, only `train.py` writable, parent
README/docs reads failing with `FileNotFoundError`, a reserved-address SSH probe
failing with `Operation not permitted`, no remote credentials, no A40 transport,
and a synthetic campaign-view update after the trusted metadata handoff. The
provider final-response hash is `46935c752a1bf869602a34b226b9bc054b0039f4ba111d468e81d89730db75ec`.
