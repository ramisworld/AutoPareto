#!/usr/bin/env python3
"""Non-study diagnostic: where does cached-decode timing jitter come from on this host?

Uses the protected evaluator's own functions and the registered baseline checkpoint (read-only).
Writes one JSON report; touches no campaign state and changes no protocol. Run as root from
/opt/autopareto_trusted with the Env B interpreter and the evaluator's offline HF environment.

  A. GPU-busy fraction of a cached decode repetition (torch.profiler kernel time / wall time).
  B. N official 8-s windows of the baseline adapter, each with: tok/s, process CPU seconds,
     involuntary context switches, cgroup CPU throttling, CPU pressure (PSI), GPU SM clock.
  C. CPU-only control: fixed pure-Python work in 1-s chunks (no GPU), to measure host CPU jitter.
  D. Optional: B again with the process pinned to one CPU (diagnostic only; not the protocol).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import resource
import statistics
import subprocess
import sys
import threading
import time

import torch

from study.contracts_2026 import inference_benchmark as ib
from study.contracts_2026.benchmark_policy import classify_measurement_stability


def read_cgroup_cpu_stat() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in pathlib.Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
            key, value = line.split()
            out[key] = int(value)
    except OSError:
        pass
    return out


def read_psi_some_total() -> int | None:
    try:
        first = pathlib.Path("/proc/pressure/cpu").read_text().splitlines()[0]
        return int(first.split("total=")[1])
    except (OSError, IndexError, ValueError):
        return None


class SmClockSampler:
    """Sample SM clock, utilization and power every 100 ms with nvidia-smi in the background."""

    def __init__(self) -> None:
        self.rows: list[tuple[float, float, float, float]] = []
        self._stop = threading.Event()
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu,power.draw", "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            try:
                sm, util, power = (float(x) for x in line.strip().split(","))
            except ValueError:
                continue
            self.rows.append((time.time_ns() / 1e9, sm, util, power))

    def window(self, start_s: float, end_s: float) -> dict[str, float | None]:
        rows = [r for r in self.rows if start_s <= r[0] <= end_s]
        if not rows:
            return {"sm_clock_mhz_mean": None, "sm_clock_mhz_min": None, "gpu_util_mean": None, "power_w_mean": None}
        return {
            "sm_clock_mhz_mean": statistics.fmean(r[1] for r in rows),
            "sm_clock_mhz_min": min(r[1] for r in rows),
            "gpu_util_mean": statistics.fmean(r[2] for r in rows),
            "power_w_mean": statistics.fmean(r[3] for r in rows),
        }

    def close(self) -> None:
        self._stop.set()
        self._proc.terminate()


def run_windows(adapter, prompts, count: int, sampler: SmClockSampler, label: str) -> list[dict]:
    rows = []
    for index in range(count):
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        cg0 = read_cgroup_cpu_stat()
        psi0 = read_psi_some_total()
        window = ib._run_official_window(adapter, prompts, index, index * 3)
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        cg1 = read_cgroup_cpu_stat()
        psi1 = read_psi_some_total()
        start_s, end_s = window["started_at_unix_ns"] / 1e9, window["completed_at_unix_ns"] / 1e9
        wall = end_s - start_s
        row = {
            "label": label,
            "index": index,
            "tok_s": window["decode_tokens_per_second"],
            "wall_seconds": wall,
            "decode_seconds": window["decode_seconds"],
            "repetitions": window["sample_count"],
            "repetition_tok_s_cv": window["coefficient_of_variation"],
            "process_cpu_seconds_per_wall_second": ((ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime)) / wall,
            "involuntary_context_switches": ru1.ru_nivcsw - ru0.ru_nivcsw,
            "voluntary_context_switches": ru1.ru_nvcsw - ru0.ru_nvcsw,
            "cgroup_nr_throttled": cg1.get("nr_throttled", 0) - cg0.get("nr_throttled", 0),
            "cgroup_throttled_ms": (cg1.get("throttled_usec", 0) - cg0.get("throttled_usec", 0)) / 1000,
            "psi_cpu_some_ms_per_s": None if psi0 is None or psi1 is None else (psi1 - psi0) / 1000 / wall,
            **sampler.window(start_s, end_s),
        }
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    return rows


def correlations(rows: list[dict]) -> dict[str, float | None]:
    def pearson(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if len(pairs) < 3:
            return None
        xs, ys = zip(*pairs)
        if statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
            return None
        return statistics.correlation(xs, ys)
    tok = [r["tok_s"] for r in rows]
    keys = ("involuntary_context_switches", "cgroup_throttled_ms", "psi_cpu_some_ms_per_s",
            "sm_clock_mhz_mean", "process_cpu_seconds_per_wall_second", "gpu_util_mean")
    return {k: pearson(tok, [r[k] for r in rows]) for k in keys}


def cpu_control(seconds: int) -> dict:
    """Fixed pure-Python work units timed in ~1-s chunks; no GPU involvement."""
    def unit() -> int:
        total = 0
        for value in range(20000):
            total += value * value
        return total
    start = time.perf_counter()
    calibration = 0
    while time.perf_counter() - start < 0.5:
        unit(); calibration += 1
    units_per_chunk = calibration * 2
    rates = []
    for _ in range(seconds):
        t0 = time.perf_counter()
        for _ in range(units_per_chunk):
            unit()
        rates.append(units_per_chunk / (time.perf_counter() - t0))
    stability = classify_measurement_stability(rates[:8]) if len(rates) >= 8 else {}
    q = statistics.quantiles(rates, n=4, method="inclusive")
    return {
        "chunks": len(rates),
        "rate_cv": statistics.pstdev(rates) / statistics.fmean(rates),
        "rate_iqr_over_median": (q[2] - q[0]) / statistics.median(rates),
        "rate_range_over_min": (max(rates) - min(rates)) / min(rates),
        "first_8_chunks_stability": stability,
        "rates": rates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--pinned-windows", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    device = torch.device("cuda")
    context = pathlib.Path(ib.__file__).resolve().parent / "historical_reference" / "baseline_source"
    train = context / "train.py"
    module = ib.load_candidate_definitions(train)
    model = ib.load_checkpoint_model(module, ib.load_checkpoint_file(args.checkpoint), device)
    adapter = ib.build_adapter(module, model)
    prompts = ib.protected_prompts(ib._load_tokenizer(train), device)
    report: dict = {
        "schema": "autopareto_nonstudy_decode_timing_diagnostics_v1",
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip(),
        "torch_num_threads": torch.get_num_threads(),
        "cpus_allowed": [l for l in pathlib.Path("/proc/self/status").read_text().splitlines() if l.startswith("Cpus_allowed_list")],
        "cgroup_cpu_max": pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip() if pathlib.Path("/sys/fs/cgroup/cpu.max").exists() else None,
        "loadavg_start": pathlib.Path("/proc/loadavg").read_text().strip(),
    }
    ib._run_fixed_decode_warmup(adapter, prompts)

    # A. GPU-busy fraction of one repetition.
    from torch.profiler import ProfilerActivity, profile
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            row, _ = ib.one_measured_repetition(adapter, prompts[0], 0)
    averages = prof.key_averages()
    try:
        kernel_s = sum(getattr(e, "self_device_time_total", 0) or 0 for e in averages) / 1e6
    except Exception as exc:  # profiler API differences across torch versions
        kernel_s = None
        report["A_profile_error"] = repr(exc)
    launches = sum(e.count for e in averages if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC"))
    wall = row["decode_seconds"] + row["prefill_milliseconds"] / 1000
    report["A_profile_one_repetition"] = {
        "wall_prefill_plus_decode_seconds": wall,
        "gpu_kernel_seconds": kernel_s,
        "gpu_busy_fraction": None if not kernel_s else kernel_s / wall,
        "kernel_launches": launches,
        "launches_per_generated_token": launches / 128 if launches else None,
        "profiled_decode_tok_s": row["decode_tokens_per_second"],
        "top_device_ops": [(e.key, getattr(e, "self_device_time_total", None), e.count)
                           for e in sorted(averages, key=lambda e: -(getattr(e, "self_device_time_total", 0) or 0))[:8]],
    }
    print(json.dumps({"A": report["A_profile_one_repetition"]}), flush=True)

    sampler = SmClockSampler()
    time.sleep(1.0)
    try:
        # B. Official windows, unpinned (as the protocol runs).
        rows_b = run_windows(adapter, prompts, args.windows, sampler, "unpinned")
        # C. CPU-only control.
        report["C_cpu_only_control"] = cpu_control(60)
        print(json.dumps({"C": {k: v for k, v in report["C_cpu_only_control"].items() if k != "rates"}}), flush=True)
        # D. Pinned to the CPU this process is currently on (diagnostic only).
        rows_d = []
        if args.pinned_windows:
            cpu = int(pathlib.Path("/proc/self/stat").read_text().split()[38])
            os.sched_setaffinity(0, {cpu})
            report["D_pinned_cpu"] = cpu
            rows_d = run_windows(adapter, prompts, args.pinned_windows, sampler, f"pinned_cpu{cpu}")
    finally:
        sampler.close()

    def summary(rows: list[dict]) -> dict:
        tok = [r["tok_s"] for r in rows]
        q = statistics.quantiles(tok, n=4, method="inclusive")
        eights = [classify_measurement_stability(tok[i:i + 8]) for i in range(0, len(tok) - 7, 4)]
        return {
            "windows": len(tok),
            "tok_s_median": statistics.median(tok),
            "tok_s_cv": statistics.pstdev(tok) / statistics.fmean(tok),
            "tok_s_iqr_over_median": (q[2] - q[0]) / statistics.median(tok),
            "tok_s_range_over_min": (max(tok) - min(tok)) / min(tok),
            "rolling_8_window_unstable_share": sum(e["inference_status"] == "unstable" for e in eights) / len(eights) if eights else None,
            "involuntary_cs_per_window_median": statistics.median(r["involuntary_context_switches"] for r in rows),
            "cgroup_throttled_ms_total": sum(r["cgroup_throttled_ms"] for r in rows),
            "psi_some_ms_per_s_median": statistics.median(r["psi_cpu_some_ms_per_s"] for r in rows if r["psi_cpu_some_ms_per_s"] is not None),
            "sm_clock_mhz_mean_range": [min(r["sm_clock_mhz_mean"] for r in rows if r["sm_clock_mhz_mean"]), max(r["sm_clock_mhz_mean"] for r in rows if r["sm_clock_mhz_mean"])],
            "gpu_util_mean": statistics.fmean(r["gpu_util_mean"] for r in rows if r["gpu_util_mean"] is not None),
            "process_cpu_per_wall_median": statistics.median(r["process_cpu_seconds_per_wall_second"] for r in rows),
            "pearson_tok_s_vs": correlations(rows),
        }

    report["B_unpinned_windows"] = {"summary": summary(rows_b), "rows": rows_b}
    if rows_d:
        report["D_pinned_windows"] = {"summary": summary(rows_d), "rows": rows_d}
    report["loadavg_end"] = pathlib.Path("/proc/loadavg").read_text().strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"B": report["B_unpinned_windows"]["summary"], "D": report.get("D_pinned_windows", {}).get("summary")}, default=str), flush=True)
    print("DIAGNOSTICS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
