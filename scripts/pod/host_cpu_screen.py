#!/usr/bin/env python3
"""60-second host screen before any GPU work (no GPU, no torch, writes one JSON report).

Records the CPU model, frequency governor, container CPU quota, load and CPU pressure, and times fixed
pure-Python work in 1-s chunks, the same control as decode_timing_diagnostics.py part C (doc 18 §3).
Advisory only; the validation acceptance rule (doc 17 §8) is unchanged. Screen rule, set before use:
if the CPU-only control fails the protocol's own 2% interquartile rule over the 60 chunks, or over any
8 consecutive chunks more than half the time, report before spending GPU time.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import subprocess
import sys
import time


def unit() -> int:
    total = 0
    for value in range(20000):
        total += value * value
    return total


def iqr_over_median(values: list[float]) -> float:
    q = statistics.quantiles(values, n=4, method="inclusive")
    return (q[2] - q[0]) / statistics.median(values)


def read(path: str) -> str | None:
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return None


def main() -> int:
    output = pathlib.Path(sys.argv[1])
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    pick = lambda key: next((l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith(key)), None)
    start = time.perf_counter(); calibration = 0
    while time.perf_counter() - start < 0.5:
        unit(); calibration += 1
    per_chunk = calibration * 2
    rates = []
    for _ in range(60):
        t0 = time.perf_counter()
        for _ in range(per_chunk):
            unit()
        rates.append(per_chunk / (time.perf_counter() - t0))
    rolling = [iqr_over_median(rates[i:i + 8]) for i in range(0, 53)]
    whole = iqr_over_median(rates)
    rolling_fail = sum(x > 0.02 for x in rolling) / len(rolling)
    report = {
        "schema": "autopareto_host_cpu_screen_v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip(),
        "cpu_model": pick("Model name"), "sockets": pick("Socket(s)"), "threads_per_core": pick("Thread(s) per core"),
        "cpu_max_mhz": pick("CPU max MHz"),
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cgroup_cpu_max": read("/sys/fs/cgroup/cpu.max"),
        "loadavg": read("/proc/loadavg"),
        "cpu_pressure": (read("/proc/pressure/cpu") or "").splitlines()[:1],
        "rate_cv": statistics.pstdev(rates) / statistics.fmean(rates),
        "rate_iqr_over_median": whole,
        "rate_range_over_min": (max(rates) - min(rates)) / min(rates),
        "rolling_8_chunk_iqr_fail_share": rolling_fail,
        "screen": "pass" if whole <= 0.02 and rolling_fail <= 0.5 else "report_before_gpu",
        "rates": rates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rates"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
