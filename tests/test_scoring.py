from app.schemas import OptimizationPriority, WorkloadConstraints, WorkloadType
from app.services.benchmark_store import BenchmarkStore
from app.services.recommender.scoring import score_candidate
from app.services.runtime_registry import RuntimeRegistry


def _store() -> BenchmarkStore:
    return BenchmarkStore()


def _registry() -> RuntimeRegistry:
    return RuntimeRegistry()


def test_scores_are_between_zero_and_one():
    store = _store()
    registry = _registry()
    profiles = store.query(model_id="qwen2.5-7b")
    assert profiles

    for p in profiles:
        scores = score_candidate(
            profile=p,
            all_profiles=profiles,
            priority=OptimizationPriority.BALANCED,
            workload_type=WorkloadType.CHAT,
            constraints=WorkloadConstraints(),
            registry=registry,
        )
        assert 0 <= scores.latency <= 1
        assert 0 <= scores.throughput <= 1
        assert 0 <= scores.cost <= 1
        assert 0 <= scores.quality <= 1
        assert 0 <= scores.simplicity <= 1


def test_cost_priority_favors_cheaper_gpu():
    store = _store()
    registry = _registry()
    profiles = store.query(model_id="qwen2.5-7b")

    scored = []
    for p in profiles:
        s = score_candidate(
            profile=p,
            all_profiles=profiles,
            priority=OptimizationPriority.COST,
            workload_type=WorkloadType.CHAT,
            constraints=WorkloadConstraints(),
            registry=registry,
        )
        scored.append((p, s))

    scored.sort(key=lambda x: x[1].weighted_total, reverse=True)
    # the top pick should not be the most expensive GPU
    top = scored[0][0]
    assert top.gpu_tier != "h100", "Cost-optimized pick shouldn't be H100 without constraints"


def test_latency_priority_favors_faster_option():
    store = _store()
    registry = _registry()
    profiles = store.query(model_id="qwen2.5-7b")

    scored = []
    for p in profiles:
        s = score_candidate(
            profile=p,
            all_profiles=profiles,
            priority=OptimizationPriority.LATENCY,
            workload_type=WorkloadType.CHAT,
            constraints=WorkloadConstraints(),
            registry=registry,
        )
        scored.append((p, s))

    scored.sort(key=lambda x: x[1].weighted_total, reverse=True)
    top = scored[0][0]
    # highest-scoring for latency should have low p95
    assert top.ttft_ms_p95 <= 100, "Latency-optimized pick should have low TTFT"


def test_meeting_latency_constraint_boosts_score():
    store = _store()
    registry = _registry()
    profiles = store.query(model_id="qwen2.5-7b", gpu_tier="l40s", quantization="awq")
    assert profiles
    p = profiles[0]

    # with a generous latency target the profile meets
    generous = score_candidate(
        profile=p,
        all_profiles=[p],
        priority=OptimizationPriority.LATENCY,
        workload_type=WorkloadType.CHAT,
        constraints=WorkloadConstraints(max_p95_latency_ms=500),
        registry=registry,
    )

    # with a tight target it can't meet
    tight = score_candidate(
        profile=p,
        all_profiles=[p],
        priority=OptimizationPriority.LATENCY,
        workload_type=WorkloadType.CHAT,
        constraints=WorkloadConstraints(max_p95_latency_ms=50),
        registry=registry,
    )

    assert generous.weighted_total > tight.weighted_total


def test_quality_score_reflects_quantization():
    store = _store()
    registry = _registry()

    fp16 = store.query(model_id="qwen2.5-7b", gpu_tier="l40s", quantization="fp16")
    awq = store.query(model_id="qwen2.5-7b", gpu_tier="l40s", quantization="awq")
    assert fp16 and awq

    s_fp16 = score_candidate(
        profile=fp16[0], all_profiles=fp16 + awq,
        priority=OptimizationPriority.BALANCED,
        workload_type=WorkloadType.CHAT,
        constraints=WorkloadConstraints(),
        registry=registry,
    )
    s_awq = score_candidate(
        profile=awq[0], all_profiles=fp16 + awq,
        priority=OptimizationPriority.BALANCED,
        workload_type=WorkloadType.CHAT,
        constraints=WorkloadConstraints(),
        registry=registry,
    )

    assert s_fp16.quality > s_awq.quality
