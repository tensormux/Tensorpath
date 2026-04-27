from fastapi import APIRouter, HTTPException

from app.schemas import (
    MODEL_REGISTRY,
    GPU_CATALOG,
    ComparisonRequest,
    ComparisonResponse,
    RecommendationRequest,
    RecommendationResponse,
    DeploymentConfig,
    WorkloadType,
)
from app.services.benchmark_store import BenchmarkStore
from app.services.deployment import export_config
from app.services.recommender import RecommendationEngine
from app.services.runtime_registry import RuntimeRegistry


router = APIRouter(prefix="/api")

# init services once at import time
_benchmark_store = BenchmarkStore()
_registry = RuntimeRegistry()
_engine = RecommendationEngine(
    benchmark_store=_benchmark_store,
    registry=_registry,
)


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    try:
        return _engine.recommend(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/recommend/config", response_model=DeploymentConfig)
def recommend_and_export(request: RecommendationRequest):
    """Run recommendation and return a deployable config for the top pick."""
    try:
        result = _engine.recommend(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return export_config(
        plan=result.recommended,
        workload_type=WorkloadType(result.workload_type),
    )


@router.post("/compare", response_model=ComparisonResponse)
def compare(request: ComparisonRequest):
    try:
        return _engine.compare(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models")
def list_models():
    return {
        mid: {"display_name": m.display_name, "params_b": m.param_billions, "family": m.family}
        for mid, m in MODEL_REGISTRY.items()
    }


@router.get("/gpus")
def list_gpus():
    return {
        tier.value: {
            "name": spec.name,
            "vram_gb": spec.vram_gb,
            "hourly_cost_usd": spec.hourly_cost_usd,
            "monthly_cost_usd": spec.monthly_cost_usd,
        }
        for tier, spec in GPU_CATALOG.items()
    }


@router.get("/backends")
def list_backends():
    return {
        name: {
            "display_name": info.display_name,
            "quantizations": info.supported_quantizations,
            "gpu_tiers": info.supported_gpu_tiers,
        }
        for name, info in _registry.backends.items()
    }


@router.get("/benchmarks/{model_id}")
def get_benchmarks(model_id: str):
    profiles = _benchmark_store.query(model_id=model_id)
    if not profiles:
        raise HTTPException(status_code=404, detail=f"No benchmarks found for {model_id}")
    return [p.model_dump() for p in profiles]


@router.get("/health")
def health():
    return {
        "status": "ok",
        "models": len(MODEL_REGISTRY),
        "benchmarks": len(_benchmark_store.all_profiles),
        "backends": len(_registry.backends),
    }
