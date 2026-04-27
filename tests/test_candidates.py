from app.schemas import WorkloadConstraints
from app.services.benchmark_store import BenchmarkStore
from app.services.recommender.candidates import generate_candidates
from app.services.runtime_registry import RuntimeRegistry


def _store() -> BenchmarkStore:
    return BenchmarkStore()


def _registry() -> RuntimeRegistry:
    return RuntimeRegistry()


def test_generates_candidates_for_known_model():
    candidates = generate_candidates(
        model_id="qwen2.5-7b",
        constraints=WorkloadConstraints(),
        benchmark_store=_store(),
        registry=_registry(),
    )
    assert len(candidates) > 0
    assert all(c.model_id == "qwen2.5-7b" for c in candidates)


def test_returns_empty_for_unknown_model():
    candidates = generate_candidates(
        model_id="nonexistent-model",
        constraints=WorkloadConstraints(),
        benchmark_store=_store(),
        registry=_registry(),
    )
    assert candidates == []


def test_vram_constraint_filters_candidates():
    # 3 GB limit should filter out most fp16 options
    candidates = generate_candidates(
        model_id="llama3.2-3b",
        constraints=WorkloadConstraints(max_vram_gb=3.0),
        benchmark_store=_store(),
        registry=_registry(),
    )
    for c in candidates:
        assert c.vram_usage_gb <= 3.0


def test_budget_constraint_filters_expensive_options():
    # very tight budget should filter out H100
    candidates = generate_candidates(
        model_id="qwen2.5-7b",
        constraints=WorkloadConstraints(max_monthly_budget_usd=800),
        benchmark_store=_store(),
        registry=_registry(),
    )
    # H100 at $4.50/hr = ~$3285/mo. 2x budget = $1600. Should be filtered.
    gpu_tiers = {c.gpu_tier for c in candidates}
    assert "h100" not in gpu_tiers


def test_all_three_models_have_candidates():
    for model_id in ("qwen2.5-7b", "llama3.1-8b", "llama3.2-3b"):
        candidates = generate_candidates(
            model_id=model_id,
            constraints=WorkloadConstraints(),
            benchmark_store=_store(),
            registry=_registry(),
        )
        assert len(candidates) >= 3, f"{model_id} should have at least 3 candidates"
