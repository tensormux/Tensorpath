from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.schemas import (
    GPU_CATALOG,
    MODEL_REGISTRY,
    ComparisonRequest,
    OptimizationPriority,
    RecommendationRequest,
    WorkloadConstraints,
    WorkloadType,
)
from app.services.benchmark_store import BenchmarkStore
from app.services.deployment import export_config
from app.services.recommender import RecommendationEngine
from app.services.runtime_registry import RuntimeRegistry


router = APIRouter()

_templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_benchmark_store = BenchmarkStore()
_registry = RuntimeRegistry()
_engine = RecommendationEngine(benchmark_store=_benchmark_store, registry=_registry)


def _form_choices() -> dict:
    return {
        "models": [
            {"id": mid, "label": f"{m.display_name} ({m.param_billions}B)"}
            for mid, m in MODEL_REGISTRY.items()
        ],
        "workloads": [w.value for w in WorkloadType],
        "priorities": [p.value for p in OptimizationPriority],
        "gpus": [
            {"tier": tier.value, "label": f"{spec.name} ({spec.vram_gb}GB)"}
            for tier, spec in GPU_CATALOG.items()
        ],
        "backends": list(_registry.backends.keys()),
    }


def _to_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _templates.TemplateResponse(
        "index.html",
        {"request": request, "choices": _form_choices()},
    )


@router.post("/recommend", response_class=HTMLResponse)
def recommend(
    request: Request,
    model_id: str = Form(...),
    workload_type: str = Form("chat"),
    optimization_priority: str = Form("balanced"),
    max_p95_latency_ms: str = Form(""),
    min_throughput_tps: str = Form(""),
    max_monthly_budget_usd: str = Form(""),
    max_vram_gb: str = Form(""),
    min_rpm: str = Form(""),
    prefer_gpu: str = Form(""),
    prefer_backend: str = Form(""),
):
    rec_request = RecommendationRequest(
        model_id=model_id,
        workload_type=WorkloadType(workload_type),
        optimization_priority=OptimizationPriority(optimization_priority),
        constraints=WorkloadConstraints(
            max_p95_latency_ms=_to_float(max_p95_latency_ms),
            min_throughput_tps=_to_float(min_throughput_tps),
            max_monthly_budget_usd=_to_float(max_monthly_budget_usd),
            max_vram_gb=_to_float(max_vram_gb),
            min_rpm=_to_float(min_rpm),
        ),
        prefer_gpu=prefer_gpu or None,
        prefer_backend=prefer_backend or None,
    )

    try:
        result = _engine.recommend(rec_request)
    except ValueError as e:
        return _templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "choices": _form_choices(),
                "error": str(e),
                "submitted": rec_request.model_dump(),
            },
            status_code=400,
        )

    config = export_config(plan=result.recommended, workload_type=rec_request.workload_type)

    return _templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": result,
            "config": config,
            "submitted": rec_request,
        },
    )


@router.get("/compare", response_class=HTMLResponse)
def compare_form(request: Request):
    return _templates.TemplateResponse(
        "compare.html",
        {"request": request, "choices": _form_choices()},
    )


@router.post("/compare", response_class=HTMLResponse)
def compare_submit(
    request: Request,
    model_ids: list[str] = Form([]),
    workload_type: str = Form("chat"),
    optimization_priority: str = Form("balanced"),
    max_p95_latency_ms: str = Form(""),
    min_throughput_tps: str = Form(""),
    max_monthly_budget_usd: str = Form(""),
    max_vram_gb: str = Form(""),
    min_rpm: str = Form(""),
):
    distinct_ids = list(dict.fromkeys(model_ids))
    if len(distinct_ids) < 2:
        return _templates.TemplateResponse(
            "compare.html",
            {
                "request": request,
                "choices": _form_choices(),
                "error": "Pick at least two distinct models to compare.",
                "submitted": {
                    "model_ids": distinct_ids,
                    "workload_type": workload_type,
                    "optimization_priority": optimization_priority,
                },
            },
            status_code=400,
        )
    if len(distinct_ids) > 4:
        distinct_ids = distinct_ids[:4]

    cmp_request = ComparisonRequest(
        model_ids=distinct_ids,
        workload_type=WorkloadType(workload_type),
        optimization_priority=OptimizationPriority(optimization_priority),
        constraints=WorkloadConstraints(
            max_p95_latency_ms=_to_float(max_p95_latency_ms),
            min_throughput_tps=_to_float(min_throughput_tps),
            max_monthly_budget_usd=_to_float(max_monthly_budget_usd),
            max_vram_gb=_to_float(max_vram_gb),
            min_rpm=_to_float(min_rpm),
        ),
    )

    try:
        comparison = _engine.compare(cmp_request)
    except ValueError as e:
        return _templates.TemplateResponse(
            "compare.html",
            {
                "request": request,
                "choices": _form_choices(),
                "error": str(e),
                "submitted": cmp_request.model_dump(),
            },
            status_code=400,
        )

    return _templates.TemplateResponse(
        "compare_result.html",
        {
            "request": request,
            "comparison": comparison,
            "submitted": cmp_request,
        },
    )
