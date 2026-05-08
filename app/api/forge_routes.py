"""HTTP API for Forge — thin wrappers around the engine + run pipeline.

Endpoints:
    GET  /api/forge/skills/search?q=<query>     skill discovery
    POST /api/forge/runs                        create a run, retrieve skills, write prompt
    GET  /api/forge/runs                        list runs
    GET  /api/forge/runs/{run_id}               run state
    GET  /api/forge/runs/{run_id}/prompt        the agent prompt as text/markdown
    POST /api/forge/runs/{run_id}/verify        run correctness gate
    POST /api/forge/runs/{run_id}/benchmark     run microbenchmark gate
    POST /api/forge/runs/{run_id}/promote       promote into the registry
    GET  /api/forge/kernels                     list verified kernels
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.services.forge.agent_runner import (
    AgenticRunConfig,
    start_agentic_run,
)
from app.services.forge.agent_state import (
    AgenticRunState,
    load_state,
    read_transcript,
    request_abort,
)
from app.services.forge.benchmarker import benchmark_candidate
from app.services.forge.models import (
    BenchmarkResult,
    ForgeRun,
    KernelLanguage,
    KernelOp,
    KernelTaskSpec,
    PromotedKernel,
    VerificationResult,
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


router = APIRouter(prefix="/api/forge", tags=["forge"])

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _provider() -> KernelSkillsProvider:
    return KernelSkillsProvider(repo_root=_REPO_ROOT)


def _registry() -> KernelRegistry:
    return KernelRegistry(repo_root=_REPO_ROOT)


# --- request/response models ---------------------------------------------


class CreateRunRequest(BaseModel):
    op: KernelOp
    language: KernelLanguage = KernelLanguage.TRITON
    target_gpu: str
    dtype: str = "fp16"
    shape: dict[str, int]
    objective: str = "latency"
    model_family: str | None = None
    constraints: dict[str, str | int | float | bool] = Field(default_factory=dict)


class CreateRunResponse(BaseModel):
    run_id: str
    status: str
    skill_ids: list[str]
    artifact_dir: str
    prompt_path: str


class VerifyRequest(BaseModel):
    skip_cuda_check: bool = False


class RunDetail(BaseModel):
    run: ForgeRun
    has_verification_report: bool
    has_benchmark_report: bool
    has_promotion: bool


# --- skills ---------------------------------------------------------------


@router.get("/skills/search")
def skills_search(q: str = Query(..., min_length=1)) -> dict:
    try:
        out = _provider().search(q)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"kernel-skills CLI failed: {e}")
    # Parse `<id>  <name>` lines into structured response
    results = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) >= 1 and "." in parts[0]:
            results.append({
                "skill_id": parts[0],
                "name": parts[1] if len(parts) > 1 else parts[0],
            })
    return {"query": q, "results": results}


# --- runs -----------------------------------------------------------------


@router.post("/runs", response_model=CreateRunResponse)
def runs_create(request: CreateRunRequest) -> CreateRunResponse:
    task = KernelTaskSpec(
        op=request.op,
        language=request.language,
        target_gpu=request.target_gpu,
        dtype=request.dtype,  # type: ignore[arg-type]
        shape=request.shape,
        objective=request.objective,  # type: ignore[arg-type]
        model_family=request.model_family,
        constraints=request.constraints,
    )

    run = create_run(task, _REPO_ROOT)
    bundle_spec = ForgeTaskPlanner(provider=_provider()).plan(task)
    run = write_skill_artifacts(
        run, bundle_spec.skill_ids, bundle_spec.skill_bundle, _REPO_ROOT
    )
    prompt_md = build_agent_prompt(bundle_spec, run_id=run.run_id)
    run = write_prompt(run, prompt_md, _REPO_ROOT)

    prompt_rel = str(Path(run.artifact_dir) / "agent_prompt.md").replace("\\", "/")
    return CreateRunResponse(
        run_id=run.run_id,
        status=run.status.value,
        skill_ids=run.skill_ids,
        artifact_dir=run.artifact_dir,
        prompt_path=prompt_rel,
    )


@router.get("/runs", response_model=list[ForgeRun])
def runs_list() -> list[ForgeRun]:
    return list_runs(_REPO_ROOT)


@router.get("/runs/{run_id}", response_model=RunDetail)
def runs_get(run_id: str) -> RunDetail:
    try:
        run = load_run(run_id, _REPO_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    artifact_dir = _REPO_ROOT / run.artifact_dir
    return RunDetail(
        run=run,
        has_verification_report=(artifact_dir / "verification_report.json").exists(),
        has_benchmark_report=(artifact_dir / "benchmark_report.json").exists(),
        has_promotion=(artifact_dir / "promotion.json").exists(),
    )


@router.get("/runs/{run_id}/prompt", response_class=PlainTextResponse)
def runs_get_prompt(run_id: str) -> str:
    try:
        run = load_run(run_id, _REPO_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    prompt_path = _REPO_ROOT / run.artifact_dir / "agent_prompt.md"
    if not prompt_path.exists():
        raise HTTPException(status_code=404, detail="agent_prompt.md not yet written")
    return prompt_path.read_text()


@router.post("/runs/{run_id}/verify", response_model=VerificationResult)
def runs_verify(run_id: str, request: VerifyRequest | None = None) -> VerificationResult:
    try:
        run = load_run(run_id, _REPO_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    skip_cuda = bool(request and request.skip_cuda_check)
    # Late import to break a cycle: verifier imports models which is fine, but
    # the verifier subprocess-runs pytest, so we don't want to import it at
    # module load if we can help it. (Negligible perf, just hygiene.)
    from app.services.forge.verifier import verify_candidate

    _, result = verify_candidate(run, _REPO_ROOT, require_cuda=not skip_cuda)
    return result


@router.post("/runs/{run_id}/benchmark", response_model=BenchmarkResult)
def runs_benchmark(run_id: str) -> BenchmarkResult:
    try:
        run = load_run(run_id, _REPO_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    _, result = benchmark_candidate(run, _REPO_ROOT)
    return result


@router.post("/runs/{run_id}/promote", response_model=PromotedKernel)
def runs_promote(run_id: str) -> PromotedKernel:
    try:
        run = load_run(run_id, _REPO_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        _, promoted = promote_candidate(run, _REPO_ROOT)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return promoted


# --- registry -------------------------------------------------------------


@router.get("/kernels")
def kernels_list(
    op: str | None = None,
    target_gpu: str | None = None,
    dtype: str | None = None,
) -> dict:
    entries = _registry().find_kernels(op=op, target_gpu=target_gpu, dtype=dtype)
    return {"count": len(entries), "kernels": entries}


# --- agentic runs ---------------------------------------------------------


class StartAgenticRequest(BaseModel):
    max_iterations: int = Field(default=5, ge=1, le=20)
    cost_cap_usd: float = Field(default=3.0, ge=0.0, le=50.0)
    skip_cuda_check: bool = False


@router.post("/runs/{run_id}/agentic")
def runs_agentic_start(run_id: str, request: StartAgenticRequest | None = None) -> dict:
    request = request or StartAgenticRequest()
    try:
        run = load_run(run_id, _REPO_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Refuse to start a second loop while one is mid-flight on the same run.
    existing = load_state(run, _REPO_ROOT)
    if existing is not None and existing.status.value in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Agentic run already in flight (status={existing.status.value}).",
        )

    config = AgenticRunConfig(
        max_iterations=request.max_iterations,
        cost_cap_usd=request.cost_cap_usd,
        skip_cuda_check=request.skip_cuda_check,
    )
    start_agentic_run(run, _REPO_ROOT, config)
    return {
        "run_id": run.run_id,
        "started": True,
        "max_iterations": config.max_iterations,
        "cost_cap_usd": config.cost_cap_usd,
    }


@router.get("/runs/{run_id}/agentic", response_model=AgenticRunState | None)
def runs_agentic_status(run_id: str) -> AgenticRunState | None:
    try:
        run = load_run(run_id, _REPO_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return load_state(run, _REPO_ROOT)


@router.post("/runs/{run_id}/agentic/abort")
def runs_agentic_abort(run_id: str) -> dict:
    try:
        run = load_run(run_id, _REPO_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    accepted = request_abort(run, _REPO_ROOT)
    if not accepted:
        raise HTTPException(
            status_code=409,
            detail="No active agentic run to abort.",
        )
    return {"abort_requested": True}


@router.get("/runs/{run_id}/agentic/transcript")
def runs_agentic_transcript(run_id: str, tail: int = 200) -> dict:
    try:
        run = load_run(run_id, _REPO_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    entries = read_transcript(run, _REPO_ROOT)
    if tail > 0:
        entries = entries[-tail:]
    return {"count": len(entries), "entries": entries}
