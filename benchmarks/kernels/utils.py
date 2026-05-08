"""Shared helpers for kernel-level microbenchmarks.

Used by both the candidate `bench.py` files Forge runs and the in-repo
benchmark scripts under `benchmarks/kernels/`. The point is to enforce a
consistent measurement protocol:

- always warm up before timing
- always synchronize CUDA before measuring
- report median latency over multiple iterations
- emit a single JSON object on stdout in a stable schema

Don't print speedup unless you've measured both baseline and candidate in
the same run. The benchmarker parses the last JSON line of stdout and
ignores everything else, so feel free to add debug logs above it.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable


def cuda_available_or_skip() -> None:
    """Print a JSON skip message and exit 0 if CUDA isn't available.

    The benchmarker treats an exit of 0 + parseable JSON as the bench's
    output. We use exit 0 with a `passed: false`-shaped record so the
    benchmarker reports cleanly instead of looking like a crash.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        print(json.dumps({
            "baseline_latency_us": 0.0,
            "candidate_latency_us": 0.0,
            "speedup": 0.0,
            "warmup_iters": 0,
            "benchmark_iters": 0,
            "gpu_name": None,
            "skipped": True,
            "skip_reason": "torch not installed",
        }))
        sys.exit(0)
    if not torch.cuda.is_available():
        print(json.dumps({
            "baseline_latency_us": 0.0,
            "candidate_latency_us": 0.0,
            "speedup": 0.0,
            "warmup_iters": 0,
            "benchmark_iters": 0,
            "gpu_name": None,
            "skipped": True,
            "skip_reason": "CUDA unavailable",
        }))
        sys.exit(0)


def get_gpu_name() -> str | None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def time_cuda_function(
    fn: Callable[[], Any],
    *,
    warmup: int = 20,
    iters: int = 100,
) -> dict[str, float]:
    """Run `fn` repeatedly and return median + p95 latency in microseconds.

    Calls `torch.cuda.synchronize()` before and after each individual timing
    so we measure GPU work, not host queue latency. Uses `time.perf_counter`
    rather than `torch.cuda.Event` because perf_counter is sufficient at the
    microsecond range these kernels operate in and is simpler to reason about.
    """
    import torch  # type: ignore

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)  # → microseconds

    samples.sort()
    p95_index = max(0, int(len(samples) * 0.95) - 1)
    return {
        "median_us": statistics.median(samples),
        "p95_us": samples[p95_index],
    }


def write_json_report(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
