from app.schemas import (
    BenchmarkProfile,
    DeploymentPlan,
    GPU_CATALOG,
    GpuTier,
    OptimizationPriority,
    PlanScores,
    WorkloadConstraints,
)


def _format_cost(hourly: float) -> str:
    monthly = hourly * 730
    return f"${hourly:.2f}/hr (~${monthly:.0f}/mo)"


def _format_latency(ms: float) -> str:
    if ms < 1:
        return f"{ms * 1000:.0f}us"
    return f"{ms:.0f}ms"


def explain_plan(
    profile: BenchmarkProfile,
    scores: PlanScores,
    priority: OptimizationPriority,
    constraints: WorkloadConstraints,
) -> str:
    """Generate a plain-english explanation for why this plan scored the way it did."""
    gpu_spec = GPU_CATALOG.get(GpuTier(profile.gpu_tier))
    gpu_name = gpu_spec.name if gpu_spec else profile.gpu_tier.upper()

    parts: list[str] = []

    parts.append(
        f"{gpu_name} + {profile.backend} + {profile.quantization.upper()}: "
        f"p95 TTFT {_format_latency(profile.ttft_ms_p95)}, "
        f"{profile.tokens_per_sec:.0f} tok/s, "
        f"{_format_cost(profile.hourly_cost_usd)}."
    )

    # constraint status
    constraint_notes = []
    if scores.meets_latency is True:
        headroom = constraints.max_p95_latency_ms - profile.ttft_ms_p95  # type: ignore
        constraint_notes.append(f"Meets latency target with {headroom:.0f}ms headroom.")
    elif scores.meets_latency is False:
        overshoot = profile.ttft_ms_p95 - constraints.max_p95_latency_ms  # type: ignore
        constraint_notes.append(f"Misses latency target by {overshoot:.0f}ms.")

    if scores.meets_budget is True:
        monthly = profile.hourly_cost_usd * 730
        remaining = constraints.max_monthly_budget_usd - monthly  # type: ignore
        constraint_notes.append(f"Under budget by ${remaining:.0f}/mo.")
    elif scores.meets_budget is False:
        monthly = profile.hourly_cost_usd * 730
        over = monthly - constraints.max_monthly_budget_usd  # type: ignore
        constraint_notes.append(f"Over budget by ${over:.0f}/mo.")

    if scores.meets_vram is False:
        constraint_notes.append("Exceeds VRAM limit.")

    if constraint_notes:
        parts.append(" ".join(constraint_notes))

    return " ".join(parts)


def explain_recommendation(
    recommended: DeploymentPlan,
    alternatives: list[DeploymentPlan],
    priority: OptimizationPriority,
) -> str:
    """One-paragraph summary of why we picked the top plan over the alternatives."""
    rec = recommended
    lines: list[str] = []

    lines.append(
        f"We recommend {rec.gpu_name} with {rec.backend} and "
        f"{rec.quantization.upper()} quantization."
    )

    # explain based on priority
    if priority == OptimizationPriority.COST:
        lines.append(
            f"This is the most cost-effective option at "
            f"${rec.estimated_hourly_cost_usd:.2f}/hr "
            f"(~${rec.estimated_monthly_cost_usd:.0f}/mo) "
            f"that still meets your constraints."
        )
    elif priority == OptimizationPriority.LATENCY:
        lines.append(
            f"This gives you p95 TTFT of {rec.estimated_ttft_p95_ms:.0f}ms — "
            f"the best latency among options that fit your constraints."
        )
    elif priority == OptimizationPriority.THROUGHPUT:
        lines.append(
            f"At {rec.estimated_tokens_per_sec:.0f} tokens/sec, this gives you "
            f"the best throughput within your constraints."
        )
    else:
        lines.append(
            f"This balances latency ({rec.estimated_ttft_p95_ms:.0f}ms p95), "
            f"throughput ({rec.estimated_tokens_per_sec:.0f} tok/s), "
            f"and cost (${rec.estimated_monthly_cost_usd:.0f}/mo) "
            f"across all dimensions."
        )

    # compare against top alternative
    if alternatives:
        alt = alternatives[0]
        tradeoffs = []
        if alt.estimated_ttft_p95_ms < rec.estimated_ttft_p95_ms:
            diff = rec.estimated_ttft_p95_ms - alt.estimated_ttft_p95_ms
            tradeoffs.append(f"{diff:.0f}ms faster on latency")
        if alt.estimated_tokens_per_sec > rec.estimated_tokens_per_sec:
            tradeoffs.append(f"higher throughput ({alt.estimated_tokens_per_sec:.0f} tok/s)")
        if alt.estimated_monthly_cost_usd > rec.estimated_monthly_cost_usd:
            extra = alt.estimated_monthly_cost_usd - rec.estimated_monthly_cost_usd
            tradeoffs.append(f"${extra:.0f}/mo more expensive")
        elif alt.estimated_monthly_cost_usd < rec.estimated_monthly_cost_usd:
            savings = rec.estimated_monthly_cost_usd - alt.estimated_monthly_cost_usd
            tradeoffs.append(f"${savings:.0f}/mo cheaper")

        if tradeoffs:
            lines.append(
                f"The next best option ({alt.gpu_name} + {alt.backend} + "
                f"{alt.quantization.upper()}) is {', '.join(tradeoffs)}, "
                f"but scored lower overall for your priorities."
            )

    # over-provisioning warning
    if any(a.is_over_provisioned for a in alternatives):
        lines.append(
            "Some higher-tier options are flagged as over-provisioned for this workload."
        )

    return " ".join(lines)
