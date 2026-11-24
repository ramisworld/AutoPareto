#!/usr/bin/env python3
"""Map a Pair-A candidate source back to the Phase-1 (AutoResearch) recipe format.

Pair-A sources are AutoResearch recipes with three things removed, because
Pair-A evaluates in protected code: the ``evaluate_bpb`` import, the final
evaluation block, and the ``val_bpb`` print. Pair-A candidates also call the
protected ``contract_progress``/``contract_update_completed`` hooks as bare
statements whose results are unused. This adapter restores the three removed
pieces and deletes the hook statements and imports, nothing else. The result is
then handed to the unchanged protected ``phase1_source_transform``, which applies
its own strict, reviewed checks.

Correctness gate: applied to the Pair-A baseline, the adapter must reproduce the
Phase-1 panel baseline byte for byte. Any source that does not match the expected
anchors, or uses a hook's return value, is refused rather than hand-edited.

Usage:
    python3 scripts/pair_a_to_phase1_source.py --source <pair-a train.py> --output <file>
    python3 scripts/pair_a_to_phase1_source.py --self-check [--codex-sources <dir>]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAIR_A_BASELINE = ROOT / "study/contracts_2026/historical_reference/baseline_source/train.py"
PANEL_BASELINE = ROOT / "results/candidates/baseline-a40-seed0/0002-train.py"
PANEL_BASELINE_SHA256 = "428e22f114040f9fa9dd2d4da8b1ad005d7aa0be4dd3b395931c4537ba82f6b8"
HOOKS = {"contract_progress", "contract_update_completed"}

RESTORATIONS = (
    (
        "from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader\n",
        "from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb\n",
    ),
    (
        "total_tokens = step * TOTAL_BATCH_SIZE\n\n# Final summary\n",
        "total_tokens = step * TOTAL_BATCH_SIZE\n\n# Final eval\nmodel.eval()\nwith autocast_ctx:\n"
        "    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)\n\n# Final summary\n",
    ),
    (
        'print("---")\nprint(f"training_seconds:',
        'print("---")\nprint(f"val_bpb:          {val_bpb:.6f}")\nprint(f"training_seconds:',
    ),
)


class AdapterRefusal(ValueError):
    """The source is not a plain Pair-A recipe this adapter may map."""


def _is_hook_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    if isinstance(function, ast.Name):
        return function.id in HOOKS
    return (
        isinstance(function, ast.Attribute)
        and function.attr in HOOKS
        and isinstance(function.value, ast.Name)
        and function.value.id == "study_runtime"
    )


def _hook_statement_lines(source: str) -> set[int]:
    """1-based line numbers of hook imports and bare hook-call statements."""
    tree = ast.parse(source)
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "study_runtime":
            if {alias.name for alias in node.names} - HOOKS:
                raise AdapterRefusal("study_runtime import beyond the protected hooks")
            lines.update(range(node.lineno, node.end_lineno + 1))
        elif isinstance(node, ast.Import) and any(alias.name == "study_runtime" for alias in node.names):
            if len(node.names) != 1 or node.names[0].asname:
                raise AdapterRefusal("unexpected study_runtime import form")
            lines.update(range(node.lineno, node.end_lineno + 1))
        elif isinstance(node, ast.Expr) and _is_hook_call(node.value):
            lines.update(range(node.lineno, node.end_lineno + 1))
    source_lines = source.splitlines(keepends=True)
    for number in lines:
        # Each removed line must belong wholly to a removed statement.
        text = source_lines[number - 1]
        if ";" in text.split("#", 1)[0]:
            raise AdapterRefusal(f"hook shares line {number} with other code")
    return lines


def adapt(source: str) -> str:
    removed = _hook_statement_lines(source)
    kept = [line for number, line in enumerate(source.splitlines(keepends=True), start=1) if number not in removed]
    adapted = "".join(kept)
    for old, new in RESTORATIONS:
        if adapted.count(old) != 1:
            raise AdapterRefusal(f"expected exactly one anchor: {old.splitlines()[0]!r}")
        adapted = adapted.replace(old, new)
    tree = ast.parse(adapted)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in HOOKS | {"study_runtime"}:
            raise AdapterRefusal(f"source still uses {node.id} after removing bare hook calls")
        if isinstance(node, ast.Attribute) and node.attr in HOOKS:
            raise AdapterRefusal(f"source still uses {node.attr} after removing bare hook calls")
    return adapted


def _transform_accepts(source: str) -> str | None:
    """Return None if the protected Phase-1 transform accepts the source, else its reason."""
    sys.path.insert(0, str(ROOT))
    from protected import phase1_source_transform

    work = pathlib.Path(tempfile.mkdtemp(prefix="pair-a-adapter-check-"))
    try:
        phase1_source_transform.transform_source(
            source,
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            token_budget=201_326_592,
            seed=6,
            milestone_dir=work / "milestones",
            final_checkpoint=work / "final.pt",
            step_timing_path=work / "step_timing.json",
            timing_summary_path=work / "timing_summary.json",
            telemetry_path=work / "telemetry.jsonl",
            telemetry_interval_seconds=2.0,
        )
    except ValueError as exc:
        return str(exc)
    return None


def self_check(codex_sources: pathlib.Path | None) -> dict:
    adapted = adapt(PAIR_A_BASELINE.read_text())
    report = {
        "gate_pair_a_baseline_reproduces_panel_baseline": (
            adapted == PANEL_BASELINE.read_text()
            and hashlib.sha256(adapted.encode()).hexdigest() == PANEL_BASELINE_SHA256
        ),
        "gate_panel_baseline_accepted_by_protected_transform": _transform_accepts(adapted) is None,
    }
    if codex_sources is not None:
        outcomes = {}
        for path in sorted(codex_sources.glob("*.train.py")):
            try:
                reason = _transform_accepts(adapt(path.read_text()))
                outcomes[path.name] = "accepted" if reason is None else f"transform refused: {reason}"
            except (AdapterRefusal, SyntaxError) as exc:
                outcomes[path.name] = f"adapter refused: {exc}"
        report["codex_pair_a_sources"] = outcomes
        report["codex_accepted"] = f"{sum(v == 'accepted' for v in outcomes.values())}/{len(outcomes)}"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--codex-sources", type=pathlib.Path)
    args = parser.parse_args()
    if args.self_check:
        report = self_check(args.codex_sources)
        print(json.dumps(report, indent=2))
        gates = [value for key, value in report.items() if key.startswith("gate_")]
        return 0 if all(gates) else 1
    if args.source is None or args.output is None:
        parser.error("--source and --output are required unless --self-check is given")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    adapted = adapt(args.source.read_text())
    reason = _transform_accepts(adapted)
    if reason is not None:
        raise SystemExit(f"protected Phase-1 transform refuses the adapted source: {reason}")
    args.output.write_text(adapted)
    print(json.dumps({
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "adapted_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
