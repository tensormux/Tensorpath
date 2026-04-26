from pydantic import BaseModel, Field

from app.schemas.workload import (
    OptimizationPriority,
    WorkloadConstraints,
    WorkloadType,
)


class PlanScores(BaseModel):
    latency: float       # 0-1, higher is better
    throughput: float
    cost: float
    quality: float
    simplicity: float
    weighted_total: float

    # constraint satisfaction
    meets_latency: bool | None = None
    meets_throughput: bool | None = None
    meets_budget: bool | None = None
    meets_vram: bool | None = None


class DeploymentPlan(BaseModel):
    rank: int
    model_id: str
    gpu_tier: str
    gpu_name: str
    backend: str
    quantization: str

    # predicted perf
    estimated_ttft_p95_ms: float
    estimated_tokens_per_sec: float
    estimated_vram_gb: float
    estimated_hourly_cost_usd: float
    estimated_monthly_cost_usd: float

    scores: PlanScores
    explanation: str
    benchmark_source: str  # measured / estimated / imported

    # flags
    is_recommended: bool = False
    is_over_provisioned: bool = False


class RecommendationResponse(BaseModel):
    model_id: str
    model_name: str
    workload_type: str
    optimization_priority: str
    recommended: DeploymentPlan
    alternatives: list[DeploymentPlan]
    summary: str  # one-paragraph explanation of the recommendation


class DeploymentConfig(BaseModel):
    """Exportable config that can be used to spin up an endpoint."""

    model_id: str
    hf_repo: str | None
    gpu_tier: str
    gpu_count: int
    backend: str
    quantization: str
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_num_seqs: int
    dtype: str
    enforce_eager: bool
    scaling_mode: str  # "always-on" or "scale-to-zero"
    extra_args: dict = {}


class ComparisonRequest(BaseModel):
    model_ids: list[str] = Field(min_length=2, max_length=4)
    workload_type: WorkloadType = WorkloadType.CHAT
    optimization_priority: OptimizationPriority = OptimizationPriority.BALANCED
    constraints: WorkloadConstraints = Field(default_factory=WorkloadConstraints)


class ComparisonHighlight(BaseModel):
    """Per-dimension winner across compared models."""

    fastest_model_id: str
    fastest_p95_ms: float
    cheapest_model_id: str
    cheapest_monthly_usd: float
    highest_throughput_model_id: str
    highest_throughput_tps: float


class ComparisonResponse(BaseModel):
    workload_type: str
    optimization_priority: str
    responses: list[RecommendationResponse]
    highlight: ComparisonHighlight
    summary: str
