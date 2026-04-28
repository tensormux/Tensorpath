"""Quick demo of the recommendation engine."""

from app.services.benchmark_store import BenchmarkStore
from app.services.recommender import RecommendationEngine
from app.services.runtime_registry import RuntimeRegistry
from app.services.deployment import export_config
from app.schemas import (
    RecommendationRequest,
    WorkloadConstraints,
    WorkloadType,
    OptimizationPriority,
)


def main():
    engine = RecommendationEngine(BenchmarkStore(), RuntimeRegistry())

    print("=" * 60)
    print("SCENARIO: Qwen 2.5 7B | Chat | Optimize for cost")
    print("Constraints: p95 < 250ms, budget < $1200/mo")
    print("=" * 60)

    result = engine.recommend(RecommendationRequest(
        model_id="qwen2.5-7b",
        workload_type=WorkloadType.CHAT,
        optimization_priority=OptimizationPriority.COST,
        constraints=WorkloadConstraints(
            max_p95_latency_ms=250,
            max_monthly_budget_usd=1200,
        ),
    ))

    r = result.recommended
    print(f"\nRECOMMENDED:")
    print(f"  GPU:           {r.gpu_name}")
    print(f"  Backend:       {r.backend}")
    print(f"  Quantization:  {r.quantization}")
    print(f"  p95 TTFT:      {r.estimated_ttft_p95_ms}ms")
    print(f"  Throughput:    {r.estimated_tokens_per_sec} tok/s")
    print(f"  Cost:          ${r.estimated_hourly_cost_usd}/hr (~${r.estimated_monthly_cost_usd}/mo)")
    print(f"  VRAM:          {r.estimated_vram_gb} GB")
    print(f"  Data source:   {r.benchmark_source}")

    print(f"\nALTERNATIVES:")
    for a in result.alternatives:
        flag = " [over-provisioned]" if a.is_over_provisioned else ""
        print(f"  #{a.rank} {a.gpu_name} + {a.backend} + {a.quantization}"
              f" | {a.estimated_ttft_p95_ms}ms"
              f" | {a.estimated_tokens_per_sec} tok/s"
              f" | ${a.estimated_monthly_cost_usd}/mo{flag}")

    print(f"\nSUMMARY:\n  {result.summary}")

    # export deployment config
    config = export_config(r, WorkloadType.CHAT)
    print(f"\nDEPLOYMENT CONFIG:")
    for k, v in config.model_dump().items():
        print(f"  {k}: {v}")

    # second scenario
    print("\n" + "=" * 60)
    print("SCENARIO: Llama 3.1 8B | Codegen | Optimize for latency")
    print("=" * 60)

    result2 = engine.recommend(RecommendationRequest(
        model_id="llama3.1-8b",
        workload_type=WorkloadType.CODEGEN,
        optimization_priority=OptimizationPriority.LATENCY,
    ))

    r2 = result2.recommended
    print(f"\nRECOMMENDED:")
    print(f"  GPU:           {r2.gpu_name}")
    print(f"  Backend:       {r2.backend}")
    print(f"  Quantization:  {r2.quantization}")
    print(f"  p95 TTFT:      {r2.estimated_ttft_p95_ms}ms")
    print(f"  Throughput:    {r2.estimated_tokens_per_sec} tok/s")
    print(f"  Cost:          ${r2.estimated_hourly_cost_usd}/hr (~${r2.estimated_monthly_cost_usd}/mo)")

    print(f"\nSUMMARY:\n  {result2.summary}")


if __name__ == "__main__":
    main()
