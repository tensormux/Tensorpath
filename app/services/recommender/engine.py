from app.schemas import (
    BenchmarkProfile,
    ComparisonHighlight,
    ComparisonRequest,
    ComparisonResponse,
    DeploymentPlan,
    GPU_CATALOG,
    GpuTier,
    MODEL_REGISTRY,
    OptimizationPriority,
    RecommendationRequest,
    RecommendationResponse,
    WorkloadConstraints,
)
from app.services.benchmark_store import BenchmarkStore
from app.services.explanation import explain_plan, explain_recommendation
from app.services.optimization.passes import OptimizationPass
from app.services.recommender.candidates import generate_candidates
from app.services.recommender.scoring import score_candidate
from app.services.runtime_registry import RuntimeRegistry


class RecommendationEngine:
    def __init__(
        self,
        benchmark_store: BenchmarkStore,
        registry: RuntimeRegistry,
        optimization_passes: list[OptimizationPass] | None = None,
    ):
        self.benchmark_store = benchmark_store
        self.registry = registry
        # Passes annotate plans after scoring; never change ranking.
        self.optimization_passes: list[OptimizationPass] = optimization_passes or []

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        model = MODEL_REGISTRY.get(request.model_id)
        if not model:
            raise ValueError(f"Unknown model: {request.model_id}")

        # step 1: get all valid candidates
        candidates = generate_candidates(
            model_id=request.model_id,
            constraints=request.constraints,
            benchmark_store=self.benchmark_store,
            registry=self.registry,
            prefer_gpu=request.prefer_gpu,
            prefer_backend=request.prefer_backend,
        )

        if not candidates:
            raise ValueError(
                f"No valid deployment options found for {request.model_id} "
                f"with the given constraints. Try relaxing your requirements."
            )

        # step 2: score every candidate
        scored: list[tuple[BenchmarkProfile, float]] = []
        for profile in candidates:
            scores = score_candidate(
                profile=profile,
                all_profiles=candidates,
                priority=request.optimization_priority,
                workload_type=request.workload_type,
                constraints=request.constraints,
                registry=self.registry,
                prefer_gpu=request.prefer_gpu,
                prefer_backend=request.prefer_backend,
            )
            scored.append((profile, scores.weighted_total))

        # step 3: rank by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # step 4: build deployment plans with explanations
        plans: list[DeploymentPlan] = []
        for rank, (profile, _) in enumerate(scored[:4], start=1):
            scores = score_candidate(
                profile=profile,
                all_profiles=candidates,
                priority=request.optimization_priority,
                workload_type=request.workload_type,
                constraints=request.constraints,
                registry=self.registry,
                prefer_gpu=request.prefer_gpu,
                prefer_backend=request.prefer_backend,
            )

            gpu_spec = GPU_CATALOG.get(GpuTier(profile.gpu_tier))
            gpu_name = gpu_spec.name if gpu_spec else profile.gpu_tier

            monthly = profile.hourly_cost_usd * 730
            cost_per_million_tokens = (
                (profile.hourly_cost_usd / profile.tokens_per_sec) * 1_000_000 / 3600
                if profile.tokens_per_sec > 0
                else 0.0
            )

            # flag over-provisioning: if a cheaper GPU meets all constraints,
            # a more expensive one that also meets them is over-provisioned
            is_over_provisioned = False
            if rank > 1 and plans:
                best = plans[0]
                if (
                    profile.hourly_cost_usd > best.estimated_hourly_cost_usd * 1.5
                    and scores.meets_latency is not False
                    and scores.meets_budget is not False
                ):
                    is_over_provisioned = True

            explanation = explain_plan(
                profile=profile,
                scores=scores,
                priority=request.optimization_priority,
                constraints=request.constraints,
            )

            plan = DeploymentPlan(
                rank=rank,
                model_id=profile.model_id,
                gpu_tier=profile.gpu_tier,
                gpu_name=gpu_name,
                backend=profile.backend,
                quantization=profile.quantization,
                estimated_ttft_p95_ms=profile.ttft_ms_p95,
                estimated_tokens_per_sec=profile.tokens_per_sec,
                estimated_vram_gb=profile.vram_usage_gb,
                estimated_hourly_cost_usd=profile.hourly_cost_usd,
                estimated_monthly_cost_usd=round(monthly, 2),
                estimated_cost_per_million_tokens=round(cost_per_million_tokens, 2),
                scores=scores,
                explanation=explanation,
                benchmark_source=profile.source,
                is_recommended=(rank == 1),
                is_over_provisioned=is_over_provisioned,
            )
            plans.append(plan)

        # Run annotation passes after scoring — passes can attach metadata but
        # must not change ranking. We apply them in declaration order.
        if self.optimization_passes:
            annotated: list[DeploymentPlan] = []
            for p in plans:
                for opt_pass in self.optimization_passes:
                    p = opt_pass.apply(p)
                annotated.append(p)
            plans = annotated

        recommended = plans[0]
        alternatives = plans[1:]

        summary = explain_recommendation(
            recommended=recommended,
            alternatives=alternatives,
            priority=request.optimization_priority,
        )

        return RecommendationResponse(
            model_id=request.model_id,
            model_name=model.display_name,
            workload_type=request.workload_type,
            optimization_priority=request.optimization_priority,
            recommended=recommended,
            alternatives=alternatives,
            summary=summary,
        )

    def compare(self, request: ComparisonRequest) -> ComparisonResponse:
        """Run recommend() per model and surface a side-by-side comparison."""
        if len(set(request.model_ids)) < 2:
            raise ValueError("Comparison needs at least two distinct model IDs.")

        responses: list[RecommendationResponse] = []
        errors: list[str] = []
        for mid in request.model_ids:
            sub_request = RecommendationRequest(
                model_id=mid,
                workload_type=request.workload_type,
                optimization_priority=request.optimization_priority,
                constraints=request.constraints,
            )
            try:
                responses.append(self.recommend(sub_request))
            except ValueError as e:
                errors.append(f"{mid}: {e}")

        if len(responses) < 2:
            joined = "; ".join(errors) if errors else "no valid plans"
            raise ValueError(
                f"Comparison failed — fewer than two models produced a recommendation ({joined})."
            )

        fastest = min(responses, key=lambda r: r.recommended.estimated_ttft_p95_ms)
        cheapest = min(responses, key=lambda r: r.recommended.estimated_monthly_cost_usd)
        highest_tps = max(responses, key=lambda r: r.recommended.estimated_tokens_per_sec)

        highlight = ComparisonHighlight(
            fastest_model_id=fastest.model_id,
            fastest_p95_ms=fastest.recommended.estimated_ttft_p95_ms,
            cheapest_model_id=cheapest.model_id,
            cheapest_monthly_usd=cheapest.recommended.estimated_monthly_cost_usd,
            highest_throughput_model_id=highest_tps.model_id,
            highest_throughput_tps=highest_tps.recommended.estimated_tokens_per_sec,
        )

        bits: list[str] = []
        bits.append(
            f"{fastest.model_name} is fastest at "
            f"{fastest.recommended.estimated_ttft_p95_ms:.0f}ms p95 TTFT."
        )
        if cheapest.model_id != fastest.model_id:
            bits.append(
                f"{cheapest.model_name} is cheapest at "
                f"~${cheapest.recommended.estimated_monthly_cost_usd:.0f}/mo."
            )
        if highest_tps.model_id not in (fastest.model_id, cheapest.model_id):
            bits.append(
                f"{highest_tps.model_name} pushes the most tokens per second "
                f"({highest_tps.recommended.estimated_tokens_per_sec:.0f} tok/s)."
            )
        if errors:
            bits.append(f"Skipped: {'; '.join(errors)}.")

        summary = " ".join(bits)

        return ComparisonResponse(
            workload_type=request.workload_type,
            optimization_priority=request.optimization_priority,
            responses=responses,
            highlight=highlight,
            summary=summary,
        )
