from enum import Enum
from pydantic import BaseModel, Field


class WorkloadType(str, Enum):
    CHAT = "chat"
    CODEGEN = "codegen"
    SUMMARIZATION = "summarization"
    BATCH = "batch"
    EMBEDDING = "embedding"


class OptimizationPriority(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COST = "cost"
    BALANCED = "balanced"


class WorkloadConstraints(BaseModel):
    max_p95_latency_ms: float | None = None
    min_throughput_tps: float | None = None
    max_monthly_budget_usd: float | None = None
    max_vram_gb: float | None = None
    min_rpm: float | None = None  # requests per minute


class RecommendationRequest(BaseModel):
    model_id: str
    workload_type: WorkloadType = WorkloadType.CHAT
    optimization_priority: OptimizationPriority = OptimizationPriority.BALANCED
    constraints: WorkloadConstraints = Field(default_factory=WorkloadConstraints)
    prefer_gpu: str | None = None
    prefer_backend: str | None = None
