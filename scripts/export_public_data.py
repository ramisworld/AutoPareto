#!/usr/bin/env python3
"""Export the reviewed public CSV tables from an audited AutoPareto SQLite ledger.

This script is intentionally read-only with respect to the source database. It does
not export raw event payloads, agent actions, checkpoints, or operational paths.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess


EXPORTS = {
    "experiment_records.csv": """
        SELECT id AS experiment_id, campaign_id, experiment_number, method, phase,
               training_seed, status, valid_result, exclude_from_analysis, val_bpb,
               median_cached_decode_tokens_per_second AS cached_decode_tokens_per_second,
               peak_inference_cuda_memory_mb AS allocated_inference_memory_mib,
               peak_reserved_inference_cuda_memory_mb AS reserved_inference_memory_mib,
               parameter_count, training_seconds, inference_metric_mode,
               inference_status, cached_decode_variation_tokens_per_second,
               prefill_milliseconds, incremental_kv_cache_memory_mb,
               parent_experiment_id, change_category, decision
        FROM experiments
        ORDER BY experiment_id
    """,
    "inference_measurements.csv": """
        SELECT im.id AS measurement_id, im.experiment_id, e.campaign_id,
               e.experiment_number, e.method, e.training_seed, 'bf16' AS dtype,
               im.metric_mode, im.workload_name, im.input_tokens AS context_tokens,
               im.generated_tokens, im.inference_batch_size AS batch_size,
               im.repetition_index, im.is_warmup, im.prefill_milliseconds,
               im.prefill_tokens_per_second, im.decode_seconds,
               im.decode_tokens_per_second,
               im.peak_allocated_cuda_memory_mb AS allocated_memory_mib,
               im.peak_reserved_cuda_memory_mb AS reserved_memory_mib,
               im.prefill_kv_cache_memory_mb, im.incremental_kv_cache_memory_mb,
               im.correctness_passed, im.max_logit_absolute_error,
               im.evaluator_git_commit, im.evaluator_sha256
        FROM inference_measurements AS im
        JOIN experiments AS e ON e.id = im.experiment_id
        ORDER BY im.id
    """,
    "deployment_measurements.csv": """
        SELECT dm.id AS measurement_id, dm.experiment_id, e.campaign_id,
               e.experiment_number, e.method, e.training_seed, 'bf16' AS dtype,
               dm.workload_name, dm.input_tokens AS context_tokens,
               dm.generated_tokens, dm.inference_batch_size AS batch_size,
               dm.repetition_index, dm.is_warmup, dm.prefill_milliseconds,
               dm.prefill_tokens_per_second, dm.decode_seconds,
               dm.decode_tokens_per_second,
               dm.peak_allocated_cuda_memory_mb AS allocated_memory_mib,
               dm.peak_reserved_cuda_memory_mb AS reserved_memory_mib,
               dm.prefill_kv_cache_memory_mb, dm.incremental_kv_cache_memory_mb,
               dm.correctness_passed, dm.status, dm.failure_reason,
               dm.evaluator_git_commit, dm.evaluator_sha256
        FROM deployment_measurements AS dm
        JOIN experiments AS e ON e.id = dm.experiment_id
        ORDER BY dm.id
    """,
}


def export_query(database: Path, query: str, destination: Path) -> int:
    """Use SQLite's CSV renderer so the checked-in release can be reproduced byte-for-byte."""
    result = subprocess.run(
        ["sqlite3", "-header", "-csv", str(database), query],
        check=True,
        capture_output=True,
    )
    destination.write_bytes(result.stdout)
    with destination.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True, help="Audited source SQLite database")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Destination directory")
    args = parser.parse_args()

    database = args.database.resolve()
    if not database.is_file():
        parser.error(f"database does not exist: {database}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not shutil.which("sqlite3"):
        parser.error("the sqlite3 command-line client is required for byte-identical CSV exports")
    for filename, query in EXPORTS.items():
        count = export_query(database, query, args.output_dir / filename)
        print(f"{filename}: {count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
