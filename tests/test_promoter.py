"""Tests for the promoter — gates, kernel ID allocation, registry insertion."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.forge.models import (
    BenchmarkResult,
    ForgeRun,
    ForgeRunStatus,
    KernelLanguage,
    KernelOp,
    KernelTaskSpec,
    VerificationResult,
)
from app.services.forge.promoter import (
    make_kernel_id,
    promote_candidate,
)
from app.services.forge.registry import KernelRegistry


pytestmark = pytest.mark.forge


def _make_task() -> KernelTaskSpec:
    return KernelTaskSpec(
        op=KernelOp.RMSNORM,
        language=KernelLanguage.TRITON,
        target_gpu="RTX 4070",
        dtype="fp16",
        shape={"batch": 16, "hidden_size": 4096},
        objective="latency",
    )


def _seed_run(tmp_path: Path, *, with_verification=None, with_benchmark=None) -> ForgeRun:
    """Create a run dir + candidate file + optionally seed reports.

    `with_verification` / `with_benchmark` are pre-built result models; pass
    None to skip writing that report.
    """
    run_id = "20260507T000000_rmsnorm_rtx-4070"
    artifact_dir = tmp_path / "forge_runs" / run_id
    candidate_dir = artifact_dir / "candidate"
    candidate_dir.mkdir(parents=True)

    (candidate_dir / "kernel.py").write_text("# fake kernel\n")

    run = ForgeRun(
        run_id=run_id,
        status=ForgeRunStatus.BENCHMARKED,
        task=_make_task(),
        skill_ids=["inference.write-triton-rmsnorm-kernel"],
        artifact_dir=str(Path("forge_runs") / run_id),
    )
    (artifact_dir / "run.json").write_text(run.model_dump_json(indent=2))

    if with_verification is not None:
        (artifact_dir / "verification_report.json").write_text(
            with_verification.model_dump_json(indent=2)
        )
    if with_benchmark is not None:
        (artifact_dir / "benchmark_report.json").write_text(
            with_benchmark.model_dump_json(indent=2)
        )

    return run


def test_make_kernel_id_format():
    task = _make_task()
    kid = make_kernel_id(task, version=1)
    assert kid == "triton_rmsnorm_rtx4070_fp16_b16_h4096_v1"


def test_make_kernel_id_version_bump():
    task = _make_task()
    kid_v1 = make_kernel_id(task, version=1)
    kid_v2 = make_kernel_id(task, version=2)
    assert kid_v1.endswith("_v1") and kid_v2.endswith("_v2")
    assert kid_v1 != kid_v2


def test_promote_refuses_when_verification_missing(tmp_path: Path):
    run = _seed_run(tmp_path, with_verification=None, with_benchmark=None)
    with pytest.raises(ValueError, match="verification_report.json missing"):
        promote_candidate(run, tmp_path)


def test_promote_refuses_when_benchmark_missing(tmp_path: Path):
    run = _seed_run(
        tmp_path,
        with_verification=VerificationResult(passed=True),
        with_benchmark=None,
    )
    with pytest.raises(ValueError, match="benchmark_report.json missing"):
        promote_candidate(run, tmp_path)


def test_promote_refuses_when_verification_failed(tmp_path: Path):
    run = _seed_run(
        tmp_path,
        with_verification=VerificationResult(passed=False, failure_reason="tests failed"),
        with_benchmark=BenchmarkResult(
            passed=True, baseline_latency_us=10, candidate_latency_us=5,
            speedup=2.0, warmup_iters=20, benchmark_iters=100,
        ),
    )
    with pytest.raises(ValueError, match="verification did not pass"):
        promote_candidate(run, tmp_path)


def test_promote_refuses_below_speedup_threshold(tmp_path: Path):
    """Speedup of 1.05x is below the 1.10x minimum."""
    run = _seed_run(
        tmp_path,
        with_verification=VerificationResult(passed=True),
        with_benchmark=BenchmarkResult(
            passed=False,
            baseline_latency_us=100, candidate_latency_us=95,
            speedup=1.05, warmup_iters=20, benchmark_iters=100,
            notes="speedup 1.050x below threshold 1.1x",
        ),
    )
    with pytest.raises(ValueError, match="benchmark did not pass"):
        promote_candidate(run, tmp_path)


def test_promote_succeeds_for_valid_candidate(tmp_path: Path):
    run = _seed_run(
        tmp_path,
        with_verification=VerificationResult(passed=True),
        with_benchmark=BenchmarkResult(
            passed=True, baseline_latency_us=42, candidate_latency_us=24.8,
            speedup=1.69, warmup_iters=20, benchmark_iters=100,
            gpu_name="NVIDIA GeForce RTX 4070",
        ),
    )

    new_run, promoted = promote_candidate(run, tmp_path)
    assert new_run.status == ForgeRunStatus.PROMOTED
    assert promoted.kernel_id == "triton_rmsnorm_rtx4070_fp16_b16_h4096_v1"
    assert promoted.benchmark.speedup == 1.69
    assert promoted.evidence_level == "op_level"

    # Source file copied
    expected = tmp_path / promoted.source_path
    assert expected.exists()
    assert expected.read_text() == "# fake kernel\n"

    # Registry entry written
    registry = KernelRegistry(tmp_path)
    entries = registry.list_kernels()
    assert len(entries) == 1 and entries[0]["kernel_id"] == promoted.kernel_id

    # promotion.json written
    promotion_path = tmp_path / new_run.artifact_dir / "promotion.json"
    assert promotion_path.exists()


def test_promote_bumps_version_on_existing_kernel(tmp_path: Path):
    """Re-promoting the same task should land at v2, not collide."""
    ver = VerificationResult(passed=True)
    bench = BenchmarkResult(
        passed=True, baseline_latency_us=42, candidate_latency_us=24,
        speedup=1.75, warmup_iters=20, benchmark_iters=100,
    )
    run1 = _seed_run(tmp_path, with_verification=ver, with_benchmark=bench)
    _, p1 = promote_candidate(run1, tmp_path)

    # Manually create a second run for the same task
    run2_id = "20260507T010101_rmsnorm_rtx-4070"
    run2_dir = tmp_path / "forge_runs" / run2_id
    (run2_dir / "candidate").mkdir(parents=True)
    (run2_dir / "candidate" / "kernel.py").write_text("# fake kernel v2\n")
    run2 = ForgeRun(
        run_id=run2_id,
        status=ForgeRunStatus.BENCHMARKED,
        task=_make_task(),
        skill_ids=[],
        artifact_dir=str(Path("forge_runs") / run2_id),
    )
    (run2_dir / "run.json").write_text(run2.model_dump_json(indent=2))
    (run2_dir / "verification_report.json").write_text(ver.model_dump_json(indent=2))
    (run2_dir / "benchmark_report.json").write_text(bench.model_dump_json(indent=2))

    _, p2 = promote_candidate(run2, tmp_path)
    assert p1.kernel_id.endswith("_v1")
    assert p2.kernel_id.endswith("_v2")
