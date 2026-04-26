from app.schemas import (
    BenchmarkProfile,
    GPU_CATALOG,
    GpuTier,
    MODEL_REGISTRY,
    WorkloadConstraints,
)
from app.services.benchmark_store import BenchmarkStore
from app.services.runtime_registry import RuntimeRegistry


def generate_candidates(
    model_id: str,
    constraints: WorkloadConstraints,
    benchmark_store: BenchmarkStore,
    registry: RuntimeRegistry,
    prefer_gpu: str | None = None,
    prefer_backend: str | None = None,
) -> list[BenchmarkProfile]:
    """Pull all valid benchmark profiles for a model and filter out impossible combos.

    Filtering happens in two stages:
    1. hard compatibility — does the backend support this GPU+quant combo?
    2. hard constraints — does it fit in VRAM? does it blow past budget?

    Preferences (prefer_gpu, prefer_backend) don't filter — they influence scoring later.
    """
    model = MODEL_REGISTRY.get(model_id)
    if not model:
        return []

    profiles = benchmark_store.query(model_id=model_id)
    if not profiles:
        return []

    valid: list[BenchmarkProfile] = []

    for p in profiles:
        # check backend x gpu x quant compatibility
        if not registry.supports_combo(p.backend, p.gpu_tier, p.quantization):
            continue

        # hard vram constraint
        gpu_spec = GPU_CATALOG.get(GpuTier(p.gpu_tier))
        if gpu_spec and p.vram_usage_gb > gpu_spec.vram_gb:
            continue
        if constraints.max_vram_gb and p.vram_usage_gb > constraints.max_vram_gb:
            continue

        # hard budget constraint — if monthly cost is >2x budget, don't even show it
        monthly = p.hourly_cost_usd * 730
        if constraints.max_monthly_budget_usd:
            if monthly > constraints.max_monthly_budget_usd * 2:
                continue

        valid.append(p)

    return valid
