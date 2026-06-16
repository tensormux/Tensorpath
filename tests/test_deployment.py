from app.schemas import (
    DeploymentPlan,
    PlanScores,
    WorkloadType,
)
from app.services.deployment import export_config


def _make_plan(**overrides) -> DeploymentPlan:
    defaults = dict(
        rank=1,
        model_id="qwen2.5-7b",
        gpu_tier="l40s",
        gpu_name="NVIDIA L40S",
        backend="vllm",
        quantization="awq",
        estimated_ttft_p95_ms=130,
        estimated_tokens_per_sec=58,
        estimated_vram_gb=5.9,
        estimated_hourly_cost_usd=1.50,
        estimated_monthly_cost_usd=1095,
        estimated_cost_per_million_tokens=7.18,
        scores=PlanScores(
            latency=0.8, throughput=0.7, cost=0.9,
            quality=0.88, simplicity=0.8, weighted_total=0.82,
        ),
        explanation="test plan",
        benchmark_source="estimated",
        is_recommended=True,
    )
    defaults.update(overrides)
    return DeploymentPlan(**defaults)


def test_export_config_basic():
    plan = _make_plan()
    config = export_config(plan, WorkloadType.CHAT)

    assert config.model_id == "qwen2.5-7b"
    assert config.gpu_tier == "l40s"
    assert config.backend == "vllm"
    assert config.quantization == "awq"
    assert config.gpu_count == 1
    assert config.max_model_len > 0
    assert config.scaling_mode == "always-on"  # chat = always-on


def test_export_config_awq_has_quantization_arg():
    plan = _make_plan(quantization="awq")
    config = export_config(plan, WorkloadType.CHAT)
    assert config.extra_args.get("quantization") == "awq"


def test_export_config_fp8_has_quantization_arg():
    plan = _make_plan(quantization="fp8")
    config = export_config(plan, WorkloadType.CHAT)
    assert config.extra_args.get("quantization") == "fp8"


def test_export_config_fp16_no_quantization_arg():
    plan = _make_plan(quantization="fp16")
    config = export_config(plan, WorkloadType.CHAT)
    assert "quantization" not in config.extra_args


def test_batch_workload_uses_higher_concurrency():
    plan = _make_plan()
    chat_config = export_config(plan, WorkloadType.CHAT)
    batch_config = export_config(plan, WorkloadType.BATCH)
    assert batch_config.max_num_seqs > chat_config.max_num_seqs


def test_codegen_workload_uses_scale_to_zero():
    plan = _make_plan()
    config = export_config(plan, WorkloadType.CODEGEN)
    assert config.scaling_mode == "scale-to-zero"
