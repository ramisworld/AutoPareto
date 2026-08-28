"""Deterministic, opt-in adapter for frozen Phase 1 recipe sources.

The adapter changes only the execution horizon and inserts checkpoint hooks. It
does not alter any model, optimizer, learning-rate, or architecture constant in
the frozen source. The original source hash is checked before transformation and
is carried into every derived artifact.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


def _integer_expression(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Pow)
    ):
        left, right = _integer_expression(node.left), _integer_expression(node.right)
        return {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.FloorDiv: lambda: left // right,
            ast.Pow: lambda: left ** right,
        }[type(node.op)]()
    raise ValueError("constant is not a safe integer expression")


def read_total_batch_size(source: str) -> int:
    tree = ast.parse(source)
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TOTAL_BATCH_SIZE"
            for target in statement.targets
        ):
            value = _integer_expression(statement.value)
            if value <= 0:
                raise ValueError("TOTAL_BATCH_SIZE must be positive")
            return value
    raise ValueError("frozen recipe is missing TOTAL_BATCH_SIZE")


def transform_source(
    source: str,
    *,
    source_sha256: str,
    token_budget: int,
    seed: int,
    milestone_dir: Path,
    final_checkpoint: Path,
    step_timing_path: Path,
    timing_summary_path: Path,
    telemetry_path: Path,
    telemetry_interval_seconds: float,
    telemetry_mode: str = "enabled",
) -> str:
    """Return a derived Phase 1 source, rejecting unexpected recipe drift."""
    batch = read_total_batch_size(source)
    if token_budget <= 0 or token_budget % batch:
        raise ValueError("Phase 1 token budget must be exactly divisible by TOTAL_BATCH_SIZE")
    if telemetry_mode not in {"enabled", "disabled"}:
        raise ValueError("telemetry_mode must be enabled or disabled")
    target_steps = token_budget // batch

    import_line = (
        "from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb"
    )
    if source.count(import_line) != 1:
        raise ValueError("unexpected prepare import; refusing an unreviewed transformation")
    source = source.replace(
        import_line,
        import_line
        + "\nfrom pathlib import Path\n"
        + "import json as phase1_json\n"
        + "from protected.phase1_runtime import ("
        + "Phase1MilestoneRecorder, Phase1StepTimingRecorder, Phase1TelemetrySampler)\n",
        1,
    )

    setup_marker = "x, y, epoch = next(train_loader)  # prefetch first batch\n"
    if source.count(setup_marker) != 1:
        raise ValueError("unexpected dataloader setup; refusing an unreviewed transformation")
    recorder = (
        "\nphase1_milestone_recorder = Phase1MilestoneRecorder(\n"
        f"    token_budget={token_budget},\n"
        f"    total_batch_size={batch},\n"
        f"    seed={seed},\n"
        f"    source_sha256={json.dumps(source_sha256)},\n"
        f"    milestone_dir=Path({str(milestone_dir)!r}),\n"
        f"    final_checkpoint=Path({str(final_checkpoint)!r}),\n"
        ")\n"
    )
    if telemetry_mode == "enabled":
        telemetry_setup = (
            "phase1_telemetry_sampler = Phase1TelemetrySampler(\n"
            f"    output_path=Path({str(telemetry_path)!r}),\n"
            f"    sample_interval_seconds={float(telemetry_interval_seconds)!r},\n"
            ")\n"
            "phase1_telemetry_sampler.start()\n"
        )
        telemetry_finish = "phase1_telemetry_payload = phase1_telemetry_sampler.stop()\n"
    else:
        telemetry_setup = "phase1_telemetry_sampler = None\n"
        telemetry_finish = (
            "phase1_telemetry_payload = {\n"
            "    'format': 'autopareto_phase1_background_telemetry_v2',\n"
            "    'sampling_interval_seconds': " + repr(float(telemetry_interval_seconds)) + ",\n"
            "    'sampling_outside_timed_critical_path': True,\n"
            "    'cuda_synchronization_called_by_sampler': False,\n"
            "    'cuda_runtime_calls_by_sampler': False,\n"
            "    'telemetry_mode': 'disabled_control',\n"
            "    'samples': [],\n"
            "    'telemetry_errors': [],\n"
            "}\n"
            f"Path({str(telemetry_path)!r}).write_text(phase1_json.dumps(phase1_telemetry_payload, sort_keys=True) + '\\n')\n"
        )
    timing = (
        "\nphase1_step_timing_recorder = Phase1StepTimingRecorder(\n"
        f"    token_budget={token_budget},\n"
        f"    tokens_per_optimizer_step={batch},\n"
        f"    step_timing_path=Path({str(step_timing_path)!r}),\n"
        f"    summary_path=Path({str(timing_summary_path)!r}),\n"
        ")\n"
        + telemetry_setup
    )
    source = source.replace(setup_marker, setup_marker + recorder + timing, 1)

    # Duration arithmetic in the derived execution source is monotonic.  The
    # frozen source bytes and all scientific recipe operations remain untouched.
    for old, new in (
        ("t_start = time.time()", "t_start = time.perf_counter_ns()"),
        ("t_start_training = time.time()", "t_start_training = time.perf_counter_ns()"),
        ("t_end = time.time()", "t_end = time.perf_counter_ns()"),
        ("startup_time = t_start_training - t_start", "startup_time = (t_start_training - t_start) / 1_000_000_000"),
        ("f\"total_seconds:    {t_end - t_start:.1f}\"", "f\"total_seconds:    {(t_end - t_start) / 1_000_000_000:.1f}\""),
    ):
        if source.count(old) != 1:
            raise ValueError(f"unexpected duration expression; refusing an unreviewed transformation: {old}")
        source = source.replace(old, new, 1)

    old_progress = "progress = min(total_training_time / TIME_BUDGET, 1.0)"
    if source.count(old_progress) != 1:
        raise ValueError("unexpected LR progress expression; refusing an unreviewed transformation")
    source = source.replace(
        old_progress,
        "progress = min((step * TOTAL_BATCH_SIZE) / PHASE1_TOKEN_BUDGET, 1.0)",
        1,
    )
    old_schedule_comment = "# Schedules (all based on progress = training_time / TIME_BUDGET)"
    if source.count(old_schedule_comment) != 1:
        raise ValueError("unexpected schedule comment; refusing an unreviewed transformation")
    source = source.replace(
        old_schedule_comment,
        "# Schedules (all based on progress = training tokens / PHASE1_TOKEN_BUDGET)",
        1,
    )

    old_budget_print = 'print(f"Time budget: {TIME_BUDGET}s")'
    if source.count(old_budget_print) != 1:
        raise ValueError("unexpected budget print; refusing an unreviewed transformation")
    source = source.replace(
        old_budget_print,
        f'print("Phase 1 token budget: {token_budget:,}")',
        1,
    )

    old_remaining = "remaining = max(0, TIME_BUDGET - total_training_time)"
    if source.count(old_remaining) != 1:
        raise ValueError("unexpected remaining-time expression; refusing an unreviewed transformation")
    source = source.replace(
        old_remaining,
        "remaining = max(0, PHASE1_TOKEN_BUDGET - step * TOTAL_BATCH_SIZE)",
        1,
    )
    old_print = "remaining: {remaining:.0f}s"
    if source.count(old_print) != 1:
        raise ValueError("unexpected training progress print; refusing an unreviewed transformation")
    source = source.replace(old_print, "remaining_tokens: {remaining:,}", 1)

    old_increment = "    step += 1\n"
    if source.count(old_increment) != 1:
        raise ValueError("unexpected optimizer-step increment; refusing an unreviewed transformation")
    source = source.replace(
        old_increment,
        old_increment
        + "    phase1_milestone_recorder.record_if_due(\n"
        + "        step=step, measured_training_seconds=total_training_time,\n"
        + "        cumulative_scientific_step_wall_seconds=phase1_step_timing_recorder.cumulative_scientific_step_wall_seconds,\n"
        + "        cumulative_steady_state_seconds=phase1_step_timing_recorder.cumulative_steady_state_seconds,\n"
        + "        model=model, config=config,\n"
        + "    )\n",
        1,
    )

    old_stop = "    if step > 10 and total_training_time >= TIME_BUDGET:\n"
    if source.count(old_stop) != 1:
        raise ValueError("unexpected training stop condition; refusing an unreviewed transformation")
    source = source.replace(
        old_stop,
        "    if step >= PHASE1_TARGET_STEPS:\n",
        1,
    )
    old_stop_comment = "    # Time's up — but only stop after warmup steps so we don't count compilation\n"
    if source.count(old_stop_comment) != 1:
        raise ValueError("unexpected stop comment; refusing an unreviewed transformation")
    source = source.replace(
        old_stop_comment,
        "    # Exact-token stop; first 11 timed steps remain excluded from pure training time.\n",
        1,
    )

    # The authoritative timing boundary is the complete optimizer-step body,
    # synchronized before and after it.  Logging, GC, checkpointing, and
    # manifest writes are outside this boundary.  CUDA events are deliberately
    # disabled for scientific runs: allocating and recording two events per
    # optimizer step would add unequal overhead to the 384-step and 1536-step
    # regimes.  A separate NON-SCIENTIFIC diagnostic may enable CUDA events.
    old_timing_start = "    torch.cuda.synchronize()\n    t0 = time.time()\n"
    if source.count(old_timing_start) != 1:
        raise ValueError("unexpected step-start timing; refusing an unreviewed transformation")
    source = source.replace(
        old_timing_start,
        "    torch.cuda.synchronize()\n"
        "    phase1_step_start_monotonic_ns = time.perf_counter_ns()\n",
        1,
    )
    old_timing_end = "    torch.cuda.synchronize()\n    t1 = time.time()\n    dt = t1 - t0\n"
    if source.count(old_timing_end) != 1:
        raise ValueError("unexpected step-end timing; refusing an unreviewed transformation")
    source = source.replace(
        old_timing_end,
        "    torch.cuda.synchronize()\n"
        "    phase1_step_end_monotonic_ns = time.perf_counter_ns()\n"
        "    dt = (phase1_step_end_monotonic_ns - phase1_step_start_monotonic_ns) / 1_000_000_000\n"
        "    phase1_step_timing_recorder.record_step(\n"
        "        zero_based_step_index=step,\n"
        "        step_start_monotonic_ns=phase1_step_start_monotonic_ns,\n"
        "        step_end_monotonic_ns=phase1_step_end_monotonic_ns,\n"
        "        cuda_stream_step_seconds=None,\n"
        "    )\n"
        "    if step == 0:\n"
        "        phase1_scientific_interval_start_ns = phase1_step_start_monotonic_ns\n"
        "    phase1_scientific_interval_end_ns = phase1_step_end_monotonic_ns\n",
        1,
    )

    old_post_loop = "print()  # newline after \\r training log\n"
    if source.count(old_post_loop) != 1:
        raise ValueError("unexpected post-loop marker; refusing an unreviewed transformation")
    source = source.replace(
        old_post_loop,
        old_post_loop
        + "phase1_step_timing_summary = phase1_step_timing_recorder.write(final_step_count=step)\n"
        + telemetry_finish,
        1,
    )

    marker = "# Training loop\n"
    if source.count(marker) != 1:
        raise ValueError("unexpected training-loop marker; refusing an unreviewed transformation")
    constants = (
        "# Phase 1 execution-only controls; frozen recipe constants are untouched.\n"
        f"PHASE1_TOKEN_BUDGET = {token_budget}\n"
        f"PHASE1_TARGET_STEPS = {target_steps}\n"
    )
    source = source.replace(marker, constants + "\n" + marker, 1)
    return source
