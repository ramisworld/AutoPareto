#!/usr/bin/env python3
"""Phase 5A: apply a recipe's own post-training storage conversion to its Phase-1 milestone checkpoints.

Why this step exists:
- The Phase-1 milestone recorder saves each checkpoint *inside* the training
  loop, before the code that follows the loop runs.
- Opus recipes convert storage after their last update:
  - attempts 23, 24 and 27 store their block/lm_head matrices in bf16
    (``set_matrix_dtype(torch.bfloat16)``);
  - attempt 27 also quantizes its gather-only embedding tables to int8 with
    per-row scales (``set_table_dtype(INFERENCE_TABLE_DTYPE)``).
- Pair-A saved the checkpoint after the script ended, so the scored model was
  the converted one.
- The protected loader builds a fresh host in the declared inference dtypes
  and copies the state dict into it:
  - fp32 matrices into bf16 parameters round to nearest even, which is the
    same as ``set_matrix_dtype``;
  - bf16 tables into int8 parameters truncate. That would score a broken
    attempt-27 model.

What this step does, for each milestone checkpoint:
1. Build a host from the derived source's definitions.
2. Apply the recipe's own training-storage statements (the ``set_*_dtype``
   lines it runs before ``init_weights``), so the host matches the checkpoint's
   dtypes exactly.
3. Strict-load the checkpoint and check every tensor bit for bit.
4. Run the recipe's own post-training storage statements, verbatim from its
   source.
5. Save with the protected ``save_training_checkpoint``.
6. Reload through the protected ``load_checkpoint_model`` and check bit for bit
   again.
7. Gate: when a recipe converts only matrices, loading the raw training
   checkpoint through the protected loader must give a bit-identical model.
   This proves that for those recipes the export changes nothing the evaluator
   sees.

The output manifest has the training manifest's schema. Each row points to the
deployed file and records the training checkpoint it came from.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SETUP_MARKER = "# Setup: tokenizer, model, optimizer, dataloader"
STORAGE_CALLS = {"set_matrix_dtype", "set_table_dtype"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _storage_calls(nodes, receiver: str) -> list[str]:
    calls = []
    for node in nodes:
        call = node.value if isinstance(node, ast.Expr) else None
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in STORAGE_CALLS
            and ast.unparse(call.func.value) == receiver
        ):
            calls.append(ast.unparse(node))
    return calls


def recipe_storage_statements(source: str) -> tuple[list[str], list[str]]:
    """(training-storage statements before init_weights, post-training storage statements)."""
    section = source.split(SETUP_MARKER, 1)[1]
    body = ast.parse("\n".join(section.splitlines()[1:])).body
    texts = [ast.unparse(node) for node in body]
    start = texts.index("model.to_empty(device=device)")
    init = texts.index("model.init_weights()")
    training = _storage_calls(body[start + 1:init], "model")
    if len(training) != init - start - 1:
        raise RuntimeError("unexpected statements between to_empty and init_weights")
    post = _storage_calls(body, "getattr(model, '_orig_mod', model)")
    if not post:
        raise RuntimeError("recipe has no post-training storage statements")
    return training, post


def _state(model) -> dict:
    return {key: value.detach() for key, value in model.state_dict().items()}


def _identical(left: dict, right: dict) -> list[str]:
    import torch

    problems = []
    if left.keys() != right.keys():
        return [f"key sets differ: {sorted(set(left) ^ set(right))[:5]}"]
    for key in left:
        a, b = left[key].to("cpu"), right[key].to("cpu")
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            problems.append(f"{key}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}")
    return problems


def export(candidate: Path, original_source: Path, manifest_path: Path, output_dir: Path) -> dict:
    import torch

    from inference_benchmark import load_candidate_definitions
    from protected.inference_adapter import load_checkpoint_file, load_checkpoint_model, save_training_checkpoint

    if output_dir.exists():
        raise RuntimeError(f"refusing to reuse {output_dir}")
    original_text = original_source.read_text()
    training_statements, post_statements = recipe_storage_statements(original_text)
    derived_defs = candidate.read_text().split(SETUP_MARKER, 1)[0]
    if derived_defs.replace(
        "from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb\n"
        "from pathlib import Path\nimport json as phase1_json\n"
        "from protected.phase1_runtime import (Phase1MilestoneRecorder, Phase1StepTimingRecorder, Phase1TelemetrySampler)\n",
        "from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader", 1,
    ) != original_text.split(SETUP_MARKER, 1)[0]:
        raise RuntimeError("derived source definitions differ from the original Pair-A source definitions")
    module = load_candidate_definitions(candidate)
    device = torch.device("cuda")
    manifest = json.loads(manifest_path.read_text())
    output_dir.mkdir(parents=True)
    rows, gates = [], []
    matrices_only = all("set_matrix_dtype" in text for text in post_statements)
    for row in manifest["checkpoints"]:
        source_ckpt = Path(row["checkpoint_path"]).resolve()
        if sha256(source_ckpt) != row["checkpoint_sha256"]:
            raise RuntimeError(f"training checkpoint hash mismatch: {source_ckpt}")
        payload = load_checkpoint_file(source_ckpt)
        config = module.GPTConfig(**payload["config"])
        with torch.device("meta"):
            model = module.GPT(config)
        model.to_empty(device=device)
        namespace = {**module.__dict__, "model": model, "torch": torch}
        for statement in training_statements:
            exec(statement, namespace)
        model.init_weights()
        model.load_state_dict(payload["state_dict"], strict=True)
        load_problems = _identical(_state(model), {k: v for k, v in payload["state_dict"].items()})
        for statement in post_statements:
            exec(statement, namespace)
        exported = _state(model)
        destination = output_dir / f"{source_ckpt.stem}.deployed.pt"
        save_training_checkpoint(
            destination, model, config, payload["seed"], payload["train_file"], payload.get("source_sha256"),
        )
        reloaded = load_checkpoint_model(module, load_checkpoint_file(destination), device)
        reload_problems = _identical(_state(reloaded), exported)
        direct = load_checkpoint_model(module, payload, device)
        direct_problems = _identical(_state(direct), exported)
        gate = {
            "token_count": row["token_count"],
            "training_checkpoint_loads_bit_exact_into_training_storage": not load_problems,
            "deployed_checkpoint_reloads_bit_exact_through_protected_loader": not reload_problems,
            "protected_loader_on_raw_training_checkpoint_equals_deployed": not direct_problems,
            "raw_vs_deployed_differences": direct_problems[:8],
        }
        gates.append(gate)
        rows.append({
            **row,
            "checkpoint_path": str(destination.resolve()),
            "checkpoint_sha256": sha256(destination),
            "training_checkpoint_path": str(source_ckpt),
            "training_checkpoint_sha256": row["checkpoint_sha256"],
        })
        del model, reloaded, direct, payload, exported
        torch.cuda.empty_cache()
    deployed_manifest = {
        **manifest,
        "checkpoints": rows,
        "deployment": {
            "format": "autopareto_phase5a_deployed_checkpoints_v1",
            "candidate": str(candidate.resolve()),
            "candidate_sha256": sha256(candidate),
            "original_source": str(original_source.resolve()),
            "original_source_sha256": sha256(original_source),
            "training_storage_statements": training_statements,
            "post_training_storage_statements": post_statements,
            "training_manifest": str(manifest_path.resolve()),
            "training_manifest_sha256": sha256(manifest_path),
        },
    }
    out_manifest = output_dir / "manifest.json"
    out_manifest.write_text(json.dumps(deployed_manifest, indent=2, sort_keys=True) + "\n")
    failures = []
    for gate in gates:
        if not gate["training_checkpoint_loads_bit_exact_into_training_storage"]:
            failures.append(f"{gate['token_count']}: training checkpoint did not load bit-exact")
        if not gate["deployed_checkpoint_reloads_bit_exact_through_protected_loader"]:
            failures.append(f"{gate['token_count']}: deployed checkpoint does not reload bit-exact")
        if matrices_only and not gate["protected_loader_on_raw_training_checkpoint_equals_deployed"]:
            failures.append(f"{gate['token_count']}: matrices-only recipe, but raw load differs from deployed")
    report = {
        "format": "autopareto_phase5a_deployment_report_v1",
        "deployed_manifest": str(out_manifest.resolve()),
        "deployed_manifest_sha256": sha256(out_manifest),
        "matrices_only_recipe": matrices_only,
        "gates": gates,
        "failures": failures,
    }
    (output_dir / "deployment_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True, type=Path, help="derived Phase-1 source (phase1_train.py)")
    parser.add_argument("--original-source", required=True, type=Path, help="the recipe's Pair-A source")
    parser.add_argument("--milestone-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = export(args.candidate, args.original_source, args.milestone_manifest, args.output_dir)
    print(json.dumps({"status": "ok" if not report["failures"] else "FAIL", "failures": report["failures"]}))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
