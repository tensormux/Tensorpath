"""Tests for KernelRegistry — add/list/find + duplicate rejection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unittest.mock import patch

from app.schemas import RecommendationRequest, WorkloadType, OptimizationPriority
from app.services.benchmark_store import BenchmarkStore
from app.services.runtime_registry import RuntimeRegistry
from app.services.forge.models import (
    BenchmarkResult,
    KernelLanguage,
    KernelOp,
    PromotedKernel,
    VerificationResult,
)
from app.services.forge.registry import KernelRegistry
from app.services.optimization import KernelRegistryPass
from app.services.recommender import RecommendationEngine


pytestmark = pytest.mark.forge


def _sample_promoted(
    kernel_id: str = "triton_rmsnorm_rtx4070_fp16_h4096_v1",
    op: KernelOp = KernelOp.RMSNORM,
    target_gpu: str = "RTX 4070",
    dtype: str = "fp16",
    shape: dict[str, int] | None = None,
) -> PromotedKernel:
    return PromotedKernel(
        kernel_id=kernel_id,
        op=op,
        language=KernelLanguage.TRITON,
        source_path=f"app/kernels/triton/verified/{op.value}/{kernel_id}.py",
        target_gpu=target_gpu,
        dtype=dtype,
        shape=shape or {"hidden_size": 4096},
        verification=VerificationResult(passed=True),
        benchmark=BenchmarkResult(
            passed=True,
            baseline_latency_us=42.0,
            candidate_latency_us=24.8,
            speedup=1.69,
            warmup_iters=20,
            benchmark_iters=100,
            gpu_name="RTX 4070",
        ),
        skill_ids=["inference.write-triton-rmsnorm-kernel"],
        evidence_level="op_level",
        created_at="2026-05-07T00:00:00Z",
    )


def test_empty_registry_starts_empty(tmp_path: Path):
    (tmp_path / "kernel_registry").mkdir()
    (tmp_path / "kernel_registry" / "verified_kernels.json").write_text('{"kernels":[]}')
    reg = KernelRegistry(tmp_path)
    assert reg.list_kernels() == []


def test_add_and_list_kernel(tmp_path: Path):
    reg = KernelRegistry(tmp_path)
    reg.add_kernel(_sample_promoted())
    entries = reg.list_kernels()
    assert len(entries) == 1
    assert entries[0]["kernel_id"] == "triton_rmsnorm_rtx4070_fp16_h4096_v1"


def test_add_persists_to_disk(tmp_path: Path):
    reg = KernelRegistry(tmp_path)
    reg.add_kernel(_sample_promoted())
    raw = json.loads((tmp_path / "kernel_registry" / "verified_kernels.json").read_text())
    assert len(raw["kernels"]) == 1


def test_duplicate_kernel_id_rejected(tmp_path: Path):
    reg = KernelRegistry(tmp_path)
    reg.add_kernel(_sample_promoted(kernel_id="dup_id"))
    with pytest.raises(ValueError, match="duplicate kernel ID"):
        reg.add_kernel(_sample_promoted(kernel_id="dup_id"))


def test_find_by_op(tmp_path: Path):
    reg = KernelRegistry(tmp_path)
    reg.add_kernel(_sample_promoted(kernel_id="a", op=KernelOp.RMSNORM))
    reg.add_kernel(_sample_promoted(kernel_id="b", op=KernelOp.SOFTMAX))
    rms = reg.find_kernels(op=KernelOp.RMSNORM)
    assert len(rms) == 1 and rms[0]["kernel_id"] == "a"


def test_find_by_op_string(tmp_path: Path):
    """Support find_kernels(op="rmsnorm") for callers that don't import the enum."""
    reg = KernelRegistry(tmp_path)
    reg.add_kernel(_sample_promoted(kernel_id="a", op=KernelOp.RMSNORM))
    rms = reg.find_kernels(op="rmsnorm")
    assert len(rms) == 1


def test_find_by_gpu_and_dtype(tmp_path: Path):
    reg = KernelRegistry(tmp_path)
    reg.add_kernel(_sample_promoted(kernel_id="a", target_gpu="H100", dtype="fp8"))
    reg.add_kernel(_sample_promoted(kernel_id="b", target_gpu="RTX 4070", dtype="fp16"))
    h100 = reg.find_kernels(target_gpu="H100")
    assert len(h100) == 1 and h100[0]["kernel_id"] == "a"
    fp16 = reg.find_kernels(dtype="fp16")
    assert len(fp16) == 1 and fp16[0]["kernel_id"] == "b"


def test_find_by_shape_partial_match(tmp_path: Path):
    """Shape filter should match if all requested dims match (partial-superset)."""
    reg = KernelRegistry(tmp_path)
    reg.add_kernel(_sample_promoted(kernel_id="a", shape={"hidden_size": 4096, "batch": 16}))
    reg.add_kernel(_sample_promoted(kernel_id="b", shape={"hidden_size": 8192, "batch": 16}))
    matches = reg.find_kernels(shape={"hidden_size": 4096})
    assert len(matches) == 1 and matches[0]["kernel_id"] == "a"


def test_corrupt_registry_raises(tmp_path: Path):
    (tmp_path / "kernel_registry").mkdir()
    (tmp_path / "kernel_registry" / "verified_kernels.json").write_text("not json")
    reg = KernelRegistry(tmp_path)
    with pytest.raises(RuntimeError, match="corrupt"):
        reg.list_kernels()


def test_redundant_file_io_in_optimization_passes(tmp_path: Path):
    # Setup a mock registry path
    registry_dir = tmp_path / "kernel_registry"
    registry_dir.mkdir()
    registry_file = registry_dir / "verified_kernels.json"
    registry_file.write_text('{"kernels": []}')

    # Instantiate registry with tmp_path as repo_root
    registry = KernelRegistry(tmp_path)
    assert registry.path == registry_file

    store = BenchmarkStore()
    runtime_registry = RuntimeRegistry()

    # Instantiate engine with KernelRegistryPass
    engine = RecommendationEngine(
        benchmark_store=store,
        registry=runtime_registry,
        optimization_passes=[KernelRegistryPass(registry)],
    )

    # Track read_text calls on the registry file
    call_count = 0
    original_read_text = Path.read_text

    def mock_read_text(self, *args, **kwargs):
        nonlocal call_count
        if self.resolve() == registry_file.resolve():
            call_count += 1
        return original_read_text(self, *args, **kwargs)

    with patch.object(Path, "read_text", autospec=True, side_effect=mock_read_text):
        # 1. First recommend call should load the registry once
        result1 = engine.recommend(RecommendationRequest(
            model_id="qwen2.5-7b",
            workload_type=WorkloadType.CHAT,
            optimization_priority=OptimizationPriority.BALANCED,
        ))
        
        assert call_count == 1, f"Expected exactly 1 file read, got {call_count}"

        # 2. Second recommend call should hit the cache and perform 0 new reads
        result2 = engine.recommend(RecommendationRequest(
            model_id="qwen2.5-7b",
            workload_type=WorkloadType.CHAT,
            optimization_priority=OptimizationPriority.BALANCED,
        ))
        
        assert call_count == 1, "Expected 0 additional file reads due to memory cache"

        # 3. Adding a kernel should save, clear the cache, and trigger a reload on next check
        registry.add_kernel(_sample_promoted("kernel_new_1"))
        
        # Now list_kernels should reload the file
        kernels = registry.list_kernels()
        assert len(kernels) == 1
        assert kernels[0]["kernel_id"] == "kernel_new_1"
        assert call_count == 2, f"Expected the file to be read again after add_kernel, got {call_count}"

