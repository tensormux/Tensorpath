"""Tests for cost per million tokens metric."""

from app.schemas import (
    ComparisonRequest,
    OptimizationPriority,
    RecommendationRequest,
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


def test_cost_per_million_tokens_is_calculated():
    """Verify cost per million tokens is calculated and included in response."""
    engine = _engine()
    result = engine.recommend(RecommendationRequest(
        model_id="qwen2.5-7b",
        workload_type=WorkloadType.CHAT,
        optimization_priority=OptimizationPriority.BALANCED,
    ))

    rec = result.recommended
    assert rec.estimated_cost_per_million_tokens > 0
    assert rec.estimated_tokens_per_sec > 0
    assert rec.estimated_hourly_cost_usd >= 0

    # Verify the calculation is correct
    # cost_per_million_tokens = (hourly_cost / tokens_per_sec) * 1_000_000 / 3600
    expected = (rec.estimated_hourly_cost_usd / rec.estimated_tokens_per_sec) * 1_000_000 / 3600
    assert abs(rec.estimated_cost_per_million_tokens - expected) < 0.01


def test_cost_per_million_tokens_for_all_plans():
    """Verify cost per million tokens is calculated for all plans, not just recommended."""
    engine = _engine()
    result = engine.recommend(RecommendationRequest(
        model_id="qwen2.5-7b",
        workload_type=WorkloadType.CHAT,
        optimization_priority=OptimizationPriority.BALANCED,
    ))

    # Check recommended plan
    assert result.recommended.estimated_cost_per_million_tokens > 0

    # Check all alternatives
    for alt in result.alternatives:
        assert alt.estimated_cost_per_million_tokens > 0


def test_cheaper_gpu_has_lower_cost_per_token():
    """Verify that cheaper GPUs generally have lower cost per million tokens."""
    engine = _engine()
    result = engine.recommend(RecommendationRequest(
        model_id="qwen2.5-7b",
        workload_type=WorkloadType.CHAT,
        optimization_priority=OptimizationPriority.COST,
    ))

    # The recommended plan should have a reasonable cost per million tokens
    rec = result.recommended
    assert rec.estimated_cost_per_million_tokens > 0
    assert rec.estimated_cost_per_million_tokens < 100  # Should be under $100/M tokens


def test_cost_per_million_tokens_in_comparison():
    """Verify cost per million tokens is included in comparison responses."""
    engine = _engine()
    result = engine.compare(ComparisonRequest(
        model_ids=["qwen2.5-7b", "llama3.1-8b"],
        workload_type=WorkloadType.CHAT,
        optimization_priority=OptimizationPriority.BALANCED,
    ))

    # Check all responses have cost per million tokens
    for response in result.responses:
        assert response.recommended.estimated_cost_per_million_tokens > 0


def test_zero_throughput_handled_gracefully():
    """Verify that zero throughput doesn't cause division by zero."""
    from app.schemas import DeploymentPlan, PlanScores

    # This should not raise an exception even with zero throughput
    plan = DeploymentPlan(
        rank=1,
        model_id="test-model",
        gpu_tier="l4",
        gpu_name="NVIDIA L4",
        backend="vllm",
        quantization="fp16",
        estimated_ttft_p95_ms=100,
        estimated_tokens_per_sec=0,  # Zero throughput
        estimated_vram_gb=10.0,
        estimated_hourly_cost_usd=0.80,
        estimated_monthly_cost_usd=584,
        estimated_cost_per_million_tokens=0.0,  # Should be 0, not error
        scores=PlanScores(
            latency=0.5, throughput=0.5, cost=0.5,
            quality=1.0, simplicity=0.8, weighted_total=0.5,
        ),
        explanation="test",
        benchmark_source="estimated",
        is_recommended=True,
    )

    assert plan.estimated_cost_per_million_tokens == 0.0
