"""Pydantic models for Forge — the kernel optimization loop.

These types describe everything that flows through Forge:
- task specs (what kernel are we optimizing, on what GPU, for what objective)
- skill bundles (what guidance from kernel-skills was attached)
- run state (where in the loop are we)
- verification + benchmark + promotion outcomes

Verifier and benchmarker write VerificationResult/BenchmarkResult; the
promoter reads those plus the candidate metadata to produce a PromotedKernel
that lands in the registry.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class KernelLanguage(str, Enum):
    TRITON = "triton"
    CUDA = "cuda"


class KernelOp(str, Enum):
    RMSNORM = "rmsnorm"
    FUSED_ADD_RMSNORM = "fused_add_rmsnorm"
    SOFTMAX = "softmax"
    SAMPLING = "sampling"
    KV_CACHE_APPEND = "kv_cache_append"
    DEQUANT = "dequant"
    ROPE = "rope"


class KernelTaskSpec(BaseModel):
    op: KernelOp
    language: KernelLanguage = KernelLanguage.TRITON
    model_family: str | None = None
    target_gpu: str
    dtype: Literal["fp16", "bf16", "fp32", "int8", "fp8"] = "fp16"
    shape: dict[str, int]
    constraints: dict[str, str | int | float | bool] = Field(default_factory=dict)
    baseline: str = "torch"
    objective: Literal["latency", "throughput", "memory", "balanced"] = "latency"


class SkillBundleSpec(BaseModel):
    """Skill IDs selected by the planner plus the rendered bundle markdown."""

    task: KernelTaskSpec
    skill_ids: list[str]
    skill_bundle: str


class ForgeRunStatus(str, Enum):
    PLANNED = "planned"
    PROMPT_READY = "prompt_ready"
    CANDIDATE_READY = "candidate_ready"
    VERIFIED = "verified"
    BENCHMARKED = "benchmarked"
    PROMOTED = "promoted"
    REJECTED = "rejected"


class ForgeRun(BaseModel):
    run_id: str
    status: ForgeRunStatus
    task: KernelTaskSpec
    skill_ids: list[str] = Field(default_factory=list)
    artifact_dir: str


class VerificationResult(BaseModel):
    passed: bool
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    tolerance: dict[str, float] = Field(default_factory=dict)
    tested_shapes: list[dict[str, int]] = Field(default_factory=list)
    failure_reason: str | None = None


class BenchmarkResult(BaseModel):
    passed: bool
    baseline_latency_us: float
    candidate_latency_us: float
    speedup: float
    warmup_iters: int
    benchmark_iters: int
    gpu_name: str | None = None
    notes: str | None = None


class PromotedKernel(BaseModel):
    kernel_id: str
    op: KernelOp
    language: KernelLanguage
    source_path: str
    target_gpu: str
    dtype: str
    shape: dict[str, int]
    verification: VerificationResult
    benchmark: BenchmarkResult
    skill_ids: list[str]
    evidence_level: Literal["op_level", "runtime_level", "endpoint_level"] = "op_level"
    created_at: str
