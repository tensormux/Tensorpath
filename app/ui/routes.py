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
from app.services.forge.agent_runner import AgenticRunConfig, start_agentic_run
from app.services.forge.agent_state import load_state, request_abort
from app.services.forge.benchmarker import benchmark_candidate
from app.services.forge.models import (
    KernelLanguage,
    KernelOp,
    KernelTaskSpec,
)
from app.services.forge.prompt_builder import build_agent_prompt
from app.services.forge.promoter import promote_candidate
from app.services.forge.registry import KernelRegistry
from app.services.forge.runs import (
    create_run,
    list_runs,
    load_run,
    write_prompt,
    write_skill_artifacts,
)
from app.services.forge.skill_provider import KernelSkillsProvider
from app.services.forge.task_planner import ForgeTaskPlanner
from app.services.forge.verifier import verify_candidate
from app.services.optimization import KernelRegistryPass
from app.services.recommender import RecommendationEngine
from app.services.runtime_registry import RuntimeRegistry


router = APIRouter()

_templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_REPO_ROOT_UI = Path(__file__).resolve().parents[2]
_benchmark_store = BenchmarkStore()
_registry = RuntimeRegistry()
_kernel_registry_ui = KernelRegistry(repo_root=_REPO_ROOT_UI)
_engine = RecommendationEngine(
    benchmark_store=_benchmark_store,
    registry=_registry,
    optimization_passes=[KernelRegistryPass(_kernel_registry_ui)],
)


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
        request,
        "index.html",
        {"choices": _form_choices()},
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
            request,
            "index.html",
            {
                "choices": _form_choices(),
                "error": str(e),
                "submitted": rec_request.model_dump(),
            },
            status_code=400,
        )

    config = export_config(plan=result.recommended, workload_type=rec_request.workload_type)

    return _templates.TemplateResponse(
        request,
        "result.html",
        {
            "result": result,
            "config": config,
            "submitted": rec_request,
        },
    )


@router.get("/compare", response_class=HTMLResponse)
def compare_form(request: Request):
    return _templates.TemplateResponse(
        request,
        "compare.html",
        {"choices": _form_choices()},
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
            request,
            "compare.html",
            {
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
            request,
            "compare.html",
            {
                "choices": _form_choices(),
                "error": str(e),
                "submitted": cmp_request.model_dump(),
            },
            status_code=400,
        )

    return _templates.TemplateResponse(
        request,
        "compare_result.html",
        {
            "comparison": comparison,
            "submitted": cmp_request,
        },
    )


# --- Forge UI -------------------------------------------------------------


from pathlib import Path as _Path  # late import to keep top tidy
_REPO_ROOT_FORGE = _Path(__file__).resolve().parents[2]


def _forge_choices() -> dict:
    return {
        "ops": [o.value for o in KernelOp],
        "languages": [lang.value for lang in KernelLanguage],
        "dtypes": ["fp16", "bf16", "fp32", "int8", "fp8"],
        "objectives": ["latency", "throughput", "memory", "balanced"],
    }


def _forge_state(notice: str | None = None, error: str | None = None) -> dict:
    runs = list_runs(_REPO_ROOT_FORGE)
    runs.sort(key=lambda r: r.run_id, reverse=True)
    registry = KernelRegistry(_REPO_ROOT_FORGE)
    return {
        "choices": _forge_choices(),
        "runs": runs,
        "kernels": registry.list_kernels(),
        "notice": notice,
        "error": error,
    }


@router.get("/forge", response_class=HTMLResponse)
def forge_index(request: Request):
    return _templates.TemplateResponse(request, "forge.html", _forge_state())


@router.post("/forge/runs", response_class=HTMLResponse)
def forge_create_run(
    request: Request,
    op: str = Form(...),
    language: str = Form("triton"),
    target_gpu: str = Form(...),
    dtype: str = Form("fp16"),
    batch: str = Form(""),
    hidden_size: str = Form(""),
    objective: str = Form("latency"),
    autonomous: str = Form(""),
    skip_cuda_check: str = Form(""),
    max_iterations: str = Form("5"),
    cost_cap_usd: str = Form("3"),
):
    shape: dict[str, int] = {}
    if batch.strip():
        try:
            shape["batch"] = int(batch)
        except ValueError:
            pass
    if hidden_size.strip():
        try:
            shape["hidden_size"] = int(hidden_size)
        except ValueError:
            pass
    if not shape:
        return _templates.TemplateResponse(
            request, "forge.html",
            _forge_state(error="At least one shape dimension is required."),
            status_code=400,
        )

    try:
        task = KernelTaskSpec(
            op=KernelOp(op),
            language=KernelLanguage(language),
            target_gpu=target_gpu,
            dtype=dtype,  # type: ignore[arg-type]
            shape=shape,
            objective=objective,  # type: ignore[arg-type]
        )
    except ValueError as e:
        return _templates.TemplateResponse(
            request, "forge.html",
            _forge_state(error=str(e)),
            status_code=400,
        )

    run = create_run(task, _REPO_ROOT_FORGE)
    bundle_spec = ForgeTaskPlanner(provider=KernelSkillsProvider(_REPO_ROOT_FORGE)).plan(task)
    run = write_skill_artifacts(run, bundle_spec.skill_ids, bundle_spec.skill_bundle, _REPO_ROOT_FORGE)
    prompt_md = build_agent_prompt(bundle_spec, run_id=run.run_id)
    run = write_prompt(run, prompt_md, _REPO_ROOT_FORGE)

    # Autonomous mode — kick off the agentic loop in a background thread,
    # then redirect to the live dashboard.
    if autonomous:
        try:
            max_iter = max(1, min(20, int(max_iterations)))
        except ValueError:
            max_iter = 5
        try:
            cost_cap = max(0.0, min(50.0, float(cost_cap_usd)))
        except ValueError:
            cost_cap = 3.0
        config = AgenticRunConfig(
            max_iterations=max_iter,
            cost_cap_usd=cost_cap,
            skip_cuda_check=bool(skip_cuda_check),
        )
        start_agentic_run(run, _REPO_ROOT_FORGE, config)
        return _templates.TemplateResponse(
            request,
            "forge_agentic_run.html",
            {"run": run, "config": config},
        )

    return _templates.TemplateResponse(
        request,
        "forge_run.html",
        {
            "run": run,
            "candidate_present": False,
            "verification_report": None,
            "benchmark_report": None,
            "promotion": None,
            "notice": f"Run created. Drop candidate files into {run.artifact_dir}/candidate/.",
        },
    )


@router.get("/forge/runs/{run_id}/agentic", response_class=HTMLResponse)
def forge_agentic_dashboard(request: Request, run_id: str):
    try:
        run = load_run(run_id, _REPO_ROOT_FORGE)
    except FileNotFoundError:
        return _templates.TemplateResponse(
            request, "forge.html",
            _forge_state(error=f"Run not found: {run_id}"),
            status_code=404,
        )
    return _templates.TemplateResponse(
        request,
        "forge_agentic_run.html",
        {"run": run, "config": None},
    )


def _load_run_artifacts(run_id: str) -> dict:
    run = load_run(run_id, _REPO_ROOT_FORGE)
    artifact_dir = _REPO_ROOT_FORGE / run.artifact_dir

    def _read_json(name: str):
        path = artifact_dir / name
        if not path.exists():
            return None
        try:
            import json
            return json.loads(path.read_text())
        except Exception:
            return None

    candidate_dir = artifact_dir / "candidate"
    candidate_present = (
        candidate_dir.exists()
        and (candidate_dir / "kernel.py").exists()
        and (candidate_dir / "metadata.json").exists()
    )

    return {
        "run": run,
        "candidate_present": candidate_present,
        "verification_report": _read_json("verification_report.json"),
        "benchmark_report": _read_json("benchmark_report.json"),
        "promotion": _read_json("promotion.json"),
    }


@router.get("/forge/runs/{run_id}", response_class=HTMLResponse)
def forge_run_detail(request: Request, run_id: str):
    try:
        artifacts = _load_run_artifacts(run_id)
    except FileNotFoundError:
        return _templates.TemplateResponse(
            request, "forge.html",
            _forge_state(error=f"Run not found: {run_id}"),
            status_code=404,
        )
    return _templates.TemplateResponse(request, "forge_run.html", artifacts)


@router.post("/forge/runs/{run_id}/verify", response_class=HTMLResponse)
def forge_run_verify(
    request: Request,
    run_id: str,
    skip_cuda_check: str = Form(""),
):
    try:
        run = load_run(run_id, _REPO_ROOT_FORGE)
    except FileNotFoundError:
        return _templates.TemplateResponse(
            request, "forge.html",
            _forge_state(error=f"Run not found: {run_id}"),
            status_code=404,
        )

    skip_cuda = bool(skip_cuda_check)
    verify_candidate(run, _REPO_ROOT_FORGE, require_cuda=not skip_cuda)

    artifacts = _load_run_artifacts(run_id)
    return _templates.TemplateResponse(
        request,
        "forge_run.html",
        {**artifacts, "notice": "Verification finished — see report below."},
    )


@router.post("/forge/runs/{run_id}/benchmark", response_class=HTMLResponse)
def forge_run_benchmark(request: Request, run_id: str):
    try:
        run = load_run(run_id, _REPO_ROOT_FORGE)
    except FileNotFoundError:
        return _templates.TemplateResponse(
            request, "forge.html",
            _forge_state(error=f"Run not found: {run_id}"),
            status_code=404,
        )

    benchmark_candidate(run, _REPO_ROOT_FORGE)
    artifacts = _load_run_artifacts(run_id)
    return _templates.TemplateResponse(
        request,
        "forge_run.html",
        {**artifacts, "notice": "Benchmark finished — see report below."},
    )


@router.post("/forge/runs/{run_id}/promote", response_class=HTMLResponse)
def forge_run_promote(request: Request, run_id: str):
    try:
        run = load_run(run_id, _REPO_ROOT_FORGE)
    except FileNotFoundError:
        return _templates.TemplateResponse(
            request, "forge.html",
            _forge_state(error=f"Run not found: {run_id}"),
            status_code=404,
        )

    notice: str | None = None
    error: str | None = None
    try:
        promote_candidate(run, _REPO_ROOT_FORGE)
        notice = "Promoted into the kernel registry."
    except ValueError as e:
        error = f"Promotion refused: {e}"

    artifacts = _load_run_artifacts(run_id)
    return _templates.TemplateResponse(
        request,
        "forge_run.html",
        {**artifacts, "notice": notice, "error": error},
    )
