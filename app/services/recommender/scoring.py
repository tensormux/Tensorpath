from app.schemas import (
    BenchmarkProfile,
    GPU_CATALOG,
    GpuTier,
    OptimizationPriority,
    WorkloadConstraints,
    WorkloadType,
    PlanScores,
)
from app.services.runtime_registry import RuntimeRegistry


# quality penalty by quantization — fp16/bf16 is baseline
_QUALITY_MAP: dict[str, float] = {
    "fp16": 1.0,
    "bf16": 1.0,
    "fp8": 0.96,
    "awq": 0.88,
    "gptq": 0.86,
}

# priority -> weight vectors
# order: latency, throughput, cost, quality, simplicity
_WEIGHT_PRESETS: dict[OptimizationPriority, tuple[float, ...]] = {
    OptimizationPriority.LATENCY:    (0.40, 0.15, 0.18, 0.15, 0.12),
    OptimizationPriority.THROUGHPUT: (0.15, 0.40, 0.18, 0.15, 0.12),
    OptimizationPriority.COST:       (0.15, 0.10, 0.45, 0.15, 0.15),
    OptimizationPriority.BALANCED:   (0.25, 0.22, 0.23, 0.18, 0.12),
}

# Workload-specific scoring characteristics
# Each workload type has different priorities for metrics
_WORKLOAD_PROFILES = {
    WorkloadType.CHAT: {
        "latency_metric": "ttft_ms_p95",
        "streaming_metric": "itl_ms_p95",
        "throughput_metric": "tokens_per_sec",
        "latency_weight": 0.6,      # TTFT matters most for first token
        "streaming_weight": 0.4,    # ITL matters for streaming feel
    },
    WorkloadType.CODEGEN: {
        "latency_metric": "ttft_ms_p95",
        "streaming_metric": "itl_ms_p95",
        "throughput_metric": "tokens_per_sec",
        "latency_weight": 0.3,      # TTFT less critical
        "streaming_weight": 0.5,    # ITL very important for long outputs
    },
    WorkloadType.SUMMARIZATION: {
        "latency_metric": "ttft_ms_p95",
        "streaming_metric": "itl_ms_p95",
        "throughput_metric": "tokens_per_sec",
        "latency_weight": 0.7,      # TTFT critical (long input processing)
        "streaming_weight": 0.2,    # ITL less important
    },
    WorkloadType.BATCH: {
        "latency_metric": "ttft_ms_p95",
        "streaming_metric": None,   # No streaming
        "throughput_metric": "tokens_per_sec",
        "latency_weight": 0.0,      # Latency doesn't matter
        "streaming_weight": 0.0,
    },
    WorkloadType.EMBEDDING: {
        "latency_metric": None,     # No generation
        "streaming_metric": None,
        "throughput_metric": "throughput_at_max_concurrency_tps",
        "latency_weight": 0.0,
        "streaming_weight": 0.0,
    },
}


def _normalize(value: float, worst: float, best: float) -> float:
    """Map value into 0-1 where best=1, worst=0. Clamps."""
    if best == worst:
        return 1.0
    score = (value - worst) / (best - worst)
    return max(0.0, min(1.0, score))


def score_candidate(
    profile: BenchmarkProfile,
    all_profiles: list[BenchmarkProfile],
    priority: OptimizationPriority,
    workload_type: WorkloadType,
    constraints: WorkloadConstraints,
    registry: RuntimeRegistry,
    prefer_gpu: str | None = None,
    prefer_backend: str | None = None,
) -> PlanScores:
    """Score a single candidate against the full candidate set and user constraints.

    Scores are relative — we normalize against the min/max seen across all candidates
    so the ranking adapts to whatever options are available.
    
    Workload-aware scoring:
    - CHAT: Prioritizes TTFT + ITL for smooth streaming
    - CODEGEN: Prioritizes ITL + throughput for long outputs
    - SUMMARIZATION: Prioritizes TTFT for fast prefill
    - BATCH: Prioritizes throughput only
    - EMBEDDING: Prioritizes concurrent throughput
    """
    if not all_profiles:
        return PlanScores(
            latency=0, throughput=0, cost=0, quality=0, simplicity=0,
            weighted_total=0,
        )

    # Get workload-specific profile
    workload_profile = _WORKLOAD_PROFILES.get(workload_type, _WORKLOAD_PROFILES[WorkloadType.CHAT])

    # Collect ranges for normalization
    all_ttft = [p.ttft_ms_p95 for p in all_profiles]
    all_itl = [p.itl_ms_p95 for p in all_profiles]
    all_costs = [p.hourly_cost_usd for p in all_profiles]
    
    # Throughput metric depends on workload type
    throughput_metric = workload_profile["throughput_metric"]
    if throughput_metric == "throughput_at_max_concurrency_tps":
        all_throughputs = [p.throughput_at_max_concurrency_tps for p in all_profiles]
    else:
        all_throughputs = [p.tokens_per_sec for p in all_profiles]

    ttft_best, ttft_worst = min(all_ttft), max(all_ttft)
    itl_best, itl_worst = min(all_itl), max(all_itl)
    tps_best, tps_worst = max(all_throughputs), min(all_throughputs)
    cost_best, cost_worst = min(all_costs), max(all_costs)

    # -- latency score (workload-aware) --
    latency_weight = workload_profile["latency_weight"]
    streaming_weight = workload_profile["streaming_weight"]
    streaming_metric = workload_profile["streaming_metric"]
    
    if streaming_metric and (latency_weight > 0 or streaming_weight > 0):
        # Streaming workload: combine TTFT and ITL
        ttft_score = _normalize(profile.ttft_ms_p95, ttft_worst, ttft_best)
        itl_score = _normalize(profile.itl_ms_p95, itl_worst, itl_best)
        latency_score = (latency_weight * ttft_score) + (streaming_weight * itl_score)
    elif workload_profile["latency_metric"] and latency_weight > 0:
        # Non-streaming but latency matters (SUMMARIZATION)
        latency_score = _normalize(profile.ttft_ms_p95, ttft_worst, ttft_best)
    else:
        # Latency doesn't matter (BATCH, EMBEDDING)
        latency_score = 0.5  # neutral

    # constraint bonus: if user set a latency target, reward meeting it
    meets_latency = None
    if constraints.max_p95_latency_ms is not None and workload_profile["latency_metric"]:
        meets_latency = profile.ttft_ms_p95 <= constraints.max_p95_latency_ms
        if meets_latency:
            # bonus for headroom — how much margin do we have?
            headroom = 1.0 - (profile.ttft_ms_p95 / constraints.max_p95_latency_ms)
            latency_score = min(1.0, latency_score + headroom * 0.15)
        else:
            # penalty proportional to how far we miss
            overshoot = profile.ttft_ms_p95 / constraints.max_p95_latency_ms
            latency_score *= max(0.1, 1.0 / overshoot)

    # -- throughput score (workload-aware) --
    if throughput_metric == "throughput_at_max_concurrency_tps":
        throughput_value = profile.throughput_at_max_concurrency_tps
    else:
        throughput_value = profile.tokens_per_sec
    
    throughput_score = _normalize(throughput_value, tps_worst, tps_best)

    meets_throughput = None
    if constraints.min_throughput_tps is not None:
        meets_throughput = throughput_value >= constraints.min_throughput_tps
        if not meets_throughput:
            throughput_score *= 0.5

    # -- cost score --
    # lower cost = better
    cost_score = _normalize(profile.hourly_cost_usd, cost_worst, cost_best)

    monthly = profile.hourly_cost_usd * 730
    meets_budget = None
    if constraints.max_monthly_budget_usd is not None:
        meets_budget = monthly <= constraints.max_monthly_budget_usd
        if meets_budget:
            # reward being well under budget
            utilization = monthly / constraints.max_monthly_budget_usd
            cost_score = min(1.0, cost_score + (1.0 - utilization) * 0.1)
        else:
            # heavy penalty for blowing budget
            overspend = monthly / constraints.max_monthly_budget_usd
            cost_score *= max(0.05, 1.0 / overspend)

    # -- quality score --
    quality_score = _QUALITY_MAP.get(profile.quantization, 0.8)

    # -- simplicity score --
    simplicity_score = registry.get_simplicity_score(profile.backend)

    # -- vram constraint --
    meets_vram = None
    if constraints.max_vram_gb is not None:
        meets_vram = profile.vram_usage_gb <= constraints.max_vram_gb

    # -- preference nudge --
    # small bonus if user's preferred gpu/backend, not enough to override a bad fit
    pref_bonus = 0.0
    if prefer_gpu and profile.gpu_tier == prefer_gpu:
        pref_bonus += 0.03
    if prefer_backend and profile.backend == prefer_backend:
        pref_bonus += 0.03

    # -- weighted total --
    weights = _WEIGHT_PRESETS[priority]
    raw_scores = (latency_score, throughput_score, cost_score, quality_score, simplicity_score)
    weighted_total = sum(w * s for w, s in zip(weights, raw_scores)) + pref_bonus

    # hard constraint violations pull the score down hard
    if meets_vram is False:
        weighted_total *= 0.1
    if meets_latency is False:
        weighted_total *= 0.6
    if meets_budget is False:
        weighted_total *= 0.5
    if meets_throughput is False:
        weighted_total *= 0.6

    return PlanScores(
        latency=round(latency_score, 4),
        throughput=round(throughput_score, 4),
        cost=round(cost_score, 4),
        quality=round(quality_score, 4),
        simplicity=round(simplicity_score, 4),
        weighted_total=round(weighted_total, 4),
        meets_latency=meets_latency,
        meets_throughput=meets_throughput,
        meets_budget=meets_budget,
        meets_vram=meets_vram,
    )
