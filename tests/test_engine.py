import pytest

from app.schemas import (
    ComparisonRequest,
    OptimizationPriority,
    RecommendationRequest,
    WorkloadConstraints,
    WorkloadType,
)
from app.services.benchmark_store import BenchmarkStore
from app.services.recommender import RecommendationEngine
from app.services.runtime_registry import RuntimeRegistry


def _engine() -> RecommendationEngine:
    return RecommendationEngine(
        benchmark_store=BenchmarkStore(),
        registry=RuntimeRegistry(),
    )


def test_basic_recommendation():
    engine = _engine()
    result = engine.recommend(RecommendationRequest(
        model_id="qwen2.5-7b",
        workload_type=WorkloadType.CHAT,
        optimization_priority=OptimizationPriority.BALANCED,
    ))

    assert result.recommended is not None
    assert result.recommended.is_recommended is True
    assert result.recommended.rank == 1
    assert len(result.alternatives) >= 1
    assert result.summary


def test_cost_optimized_recommendation():
    """Cost-optimize Qwen 2.5 7B for chat with a realistic always-on budget."""
    engine = _engine()
    result = engine.recommend(RecommendationRequest(
        model_id="qwen2.5-7b",
        workload_type=WorkloadType.CHAT,
        optimization_priority=OptimizationPriority.COST,
        constraints=WorkloadConstraints(
            max_p95_latency_ms=250,
            max_monthly_budget_usd=1200,
        ),
    ))

    rec = result.recommended
    # should not pick H100 for a cost-optimized workload
    assert rec.gpu_tier in ("l4", "l40s", "rtx4070"), f"Expected budget GPU, got {rec.gpu_tier}"
    assert rec.estimated_monthly_cost_usd <= 1200
    assert rec.scores.meets_latency is True or rec.estimated_ttft_p95_ms <= 250


def test_latency_optimized_picks_fast_option():
    engine = _engine()
    result = engine.recommend(RecommendationRequest(
        model_id="llama3.1-8b",
        workload_type=WorkloadType.CHAT,
        optimization_priority=OptimizationPriority.LATENCY,
    ))

    rec = result.recommended
    # latency-optimized should pick a fast GPU/backend combo
    assert rec.estimated_ttft_p95_ms <= 100


def test_unknown_model_raises():
    engine = _engine()
    with pytest.raises(ValueError, match="Unknown model"):
        engine.recommend(RecommendationRequest(model_id="fake-model-99b"))


def test_all_plans_have_explanations():
    engine = _engine()
    result = engine.recommend(RecommendationRequest(
        model_id="llama3.2-3b",
        workload_type=WorkloadType.CHAT,
        optimization_priority=OptimizationPriority.COST,
    ))

    assert result.recommended.explanation
    for alt in result.alternatives:
        assert alt.explanation


def test_recommendation_response_shape():
    """Acceptance criteria: system returns at least 2 candidates + 1 recommended."""
    engine = _engine()
    result = engine.recommend(RecommendationRequest(
        model_id="qwen2.5-7b",
        workload_type=WorkloadType.CHAT,
        optimization_priority=OptimizationPriority.BALANCED,
    ))

    assert result.recommended is not None
    assert len(result.alternatives) >= 1
    # total plans >= 2
    total = 1 + len(result.alternatives)
    assert total >= 2

    # recommended includes all required fields
    rec = result.recommended
    assert rec.gpu_tier
    assert rec.backend
    assert rec.quantization
    assert rec.estimated_ttft_p95_ms > 0
    assert rec.estimated_tokens_per_sec > 0
    assert rec.estimated_hourly_cost_usd >= 0


def test_compare_two_models():
    """Comparison runs recommendations for each model and surfaces the highlight."""
    engine = _engine()
    cmp = engine.compare(ComparisonRequest(
        model_ids=["qwen2.5-7b", "llama3.2-3b"],
        workload_type=WorkloadType.CHAT,
        optimization_priority=OptimizationPriority.BALANCED,
    ))

    assert len(cmp.responses) == 2
    ids = {r.model_id for r in cmp.responses}
    assert ids == {"qwen2.5-7b", "llama3.2-3b"}

    h = cmp.highlight
    assert h.fastest_model_id in ids
    assert h.cheapest_model_id in ids
    assert h.highest_throughput_model_id in ids
    assert h.fastest_p95_ms > 0
    assert h.highest_throughput_tps > 0
    assert cmp.summary


def test_compare_rejects_single_model():
    engine = _engine()
    with pytest.raises(ValueError):
        engine.compare(ComparisonRequest(
            model_ids=["qwen2.5-7b", "qwen2.5-7b"],
            workload_type=WorkloadType.CHAT,
            optimization_priority=OptimizationPriority.BALANCED,
        ))
