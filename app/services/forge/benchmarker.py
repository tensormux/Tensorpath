"""Benchmark a verified candidate kernel against its reference baseline.

Pipeline:
1. Require verification to have passed (caller should chain after verifier).
2. Run candidate/bench.py as a subprocess with a timeout.
3. Parse the last non-empty line of stdout as JSON.
4. Apply the speedup threshold and write benchmark_report.json.
5. Update run.json status accordingly.

The candidate's bench.py contract: print one JSON object on stdout containing
at minimum:
    baseline_latency_us
    candidate_latency_us
    speedup
    warmup_iters
    benchmark_iters
    gpu_name

Anything else printed is fine (logs, debug output) — we only parse the last
JSON-shaped line.

Promotion threshold for v0:
    minimum speedup: 1.10x
    candidate latency must be lower than baseline latency
    bench must run on CUDA (we won't trust a number from a CPU run)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.services.forge.models import (
    BenchmarkResult,
    ForgeRun,
    ForgeRunStatus,
    VerificationResult,
)
from app.services.forge.runs import update_run_status


_BENCH_TIMEOUT_SEC = 300
MIN_SPEEDUP = 1.10


def _run_dir(run: ForgeRun, repo_root: Path) -> Path:
    return repo_root / run.artifact_dir


def _candidate_dir(run: ForgeRun, repo_root: Path) -> Path:
    return _run_dir(run, repo_root) / "candidate"


def _read_verification(run: ForgeRun, repo_root: Path) -> VerificationResult | None:
    path = _run_dir(run, repo_root) / "verification_report.json"
    if not path.exists():
        return None
    try:
        return VerificationResult.model_validate_json(path.read_text())
    except Exception:
        return None


def _extract_json_object(text: str) -> dict | None:
    """Find the last `{...}` block in stdout and parse it as JSON. We scan
    from the end so the candidate can emit free-form logs above a final
    JSON-shaped report line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _write_report(run: ForgeRun, repo_root: Path, result: BenchmarkResult) -> None:
    (_run_dir(run, repo_root) / "benchmark_report.json").write_text(
        result.model_dump_json(indent=2)
    )


def _zero_result(reason: str) -> BenchmarkResult:
    return BenchmarkResult(
        passed=False,
        baseline_latency_us=0.0,
        candidate_latency_us=0.0,
        speedup=0.0,
        warmup_iters=0,
        benchmark_iters=0,
        notes=reason,
    )


def benchmark_candidate(
    run: ForgeRun,
    repo_root: Path,
) -> tuple[ForgeRun, BenchmarkResult]:
    """Run candidate/bench.py and apply the speedup threshold."""
    ver = _read_verification(run, repo_root)
    if ver is None or not ver.passed:
        msg = "verification has not passed yet" if ver is None else (
            f"verification did not pass: {ver.failure_reason}"
        )
        result = _zero_result(msg)
        _write_report(run, repo_root, result)
        return update_run_status(run, ForgeRunStatus.REJECTED, repo_root), result

    candidate_dir = _candidate_dir(run, repo_root)
    bench_file = candidate_dir / "bench.py"
    if not bench_file.exists():
        result = _zero_result("candidate/bench.py is missing")
        _write_report(run, repo_root, result)
        return update_run_status(run, ForgeRunStatus.REJECTED, repo_root), result

    try:
        proc = subprocess.run(
            [sys.executable, "bench.py"],
            cwd=str(candidate_dir),
            text=True,
            capture_output=True,
            timeout=_BENCH_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        result = _zero_result(f"benchmark timed out after {_BENCH_TIMEOUT_SEC}s")
        _write_report(run, repo_root, result)
        return update_run_status(run, ForgeRunStatus.REJECTED, repo_root), result

    if proc.returncode != 0:
        log = (proc.stderr or proc.stdout)[-1500:]
        result = _zero_result(f"bench.py exited {proc.returncode}: {log}")
        _write_report(run, repo_root, result)
        return update_run_status(run, ForgeRunStatus.REJECTED, repo_root), result

    data = _extract_json_object(proc.stdout)
    if data is None:
        result = _zero_result("could not parse JSON report from bench.py stdout")
        _write_report(run, repo_root, result)
        return update_run_status(run, ForgeRunStatus.REJECTED, repo_root), result

    try:
        baseline_us = float(data["baseline_latency_us"])
        candidate_us = float(data["candidate_latency_us"])
        speedup = float(data.get("speedup", baseline_us / candidate_us if candidate_us else 0))
    except (KeyError, ValueError, ZeroDivisionError) as e:
        result = _zero_result(f"bench output missing required fields: {e}")
        _write_report(run, repo_root, result)
        return update_run_status(run, ForgeRunStatus.REJECTED, repo_root), result

    threshold_ok = speedup >= MIN_SPEEDUP and candidate_us < baseline_us
    notes = None
    if not threshold_ok:
        notes = (
            f"speedup {speedup:.3f}x below threshold {MIN_SPEEDUP}x"
            if speedup < MIN_SPEEDUP
            else "candidate latency not lower than baseline"
        )

    result = BenchmarkResult(
        passed=threshold_ok,
        baseline_latency_us=baseline_us,
        candidate_latency_us=candidate_us,
        speedup=speedup,
        warmup_iters=int(data.get("warmup_iters", 0)),
        benchmark_iters=int(data.get("benchmark_iters", 0)),
        gpu_name=data.get("gpu_name"),
        notes=notes,
    )
    _write_report(run, repo_root, result)

    new_status = ForgeRunStatus.BENCHMARKED if threshold_ok else ForgeRunStatus.REJECTED
    return update_run_status(run, new_status, repo_root), result
