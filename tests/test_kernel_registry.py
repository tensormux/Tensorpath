"""Tests for KernelRegistry — add/list/find + duplicate rejection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.forge.models import (
    BenchmarkResult,
    KernelLanguage,
    KernelOp,
    PromotedKernel,
    VerificationResult,
)
from app.services.forge.registry import KernelRegistry


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
