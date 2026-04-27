from app.schemas import (
    DeploymentConfig,
    DeploymentPlan,
    MODEL_REGISTRY,
    GPU_CATALOG,
    GpuTier,
    WorkloadType,
)


# vram utilization targets — leave headroom for kv cache
_VRAM_UTIL: dict[str, float] = {
    "fp16": 0.88,
    "bf16": 0.88,
    "fp8": 0.90,
    "awq": 0.92,
    "gptq": 0.92,
}

# reasonable max_num_seqs defaults by workload
_MAX_SEQS: dict[WorkloadType, int] = {
    WorkloadType.CHAT: 32,
    WorkloadType.CODEGEN: 16,
    WorkloadType.SUMMARIZATION: 24,
    WorkloadType.BATCH: 64,
    WorkloadType.EMBEDDING: 128,
}


def export_config(
    plan: DeploymentPlan,
    workload_type: WorkloadType,
) -> DeploymentConfig:
    """Turn a deployment plan into a concrete config you can hand to vLLM / TRT-LLM."""
    model = MODEL_REGISTRY.get(plan.model_id)
    gpu_spec = GPU_CATALOG.get(GpuTier(plan.gpu_tier))

    hf_repo = model.hf_repo if model else None
    context_length = model.context_length if model else 4096

    # for awq/gptq models, point to the quantized variant
    quant_suffix = ""
    extra_args: dict = {}
    if plan.quantization == "awq":
        quant_suffix = "-AWQ"
        extra_args["quantization"] = "awq"
    elif plan.quantization == "gptq":
        quant_suffix = "-GPTQ"
        extra_args["quantization"] = "gptq"
    elif plan.quantization == "fp8":
        extra_args["quantization"] = "fp8"

    if plan.backend == "tensorrt-llm":
        extra_args["backend"] = "tensorrt-llm"

    # for small models on big GPUs, cap context to avoid wasting memory on kv cache
    vram_gb = gpu_spec.vram_gb if gpu_spec else 24
    effective_context = min(context_length, _context_for_vram(plan.estimated_vram_gb, vram_gb))

    # scaling mode: scale-to-zero makes sense for low-traffic / dev
    # always-on for anything that expects sustained traffic
    scaling = "always-on" if workload_type in (WorkloadType.CHAT, WorkloadType.BATCH) else "scale-to-zero"

    dtype = "float16" if plan.quantization in ("fp16", "awq", "gptq") else "bfloat16"
    if plan.quantization == "fp8":
        dtype = "float16"  # fp8 quantization handles precision internally

    return DeploymentConfig(
        model_id=plan.model_id,
        hf_repo=hf_repo,
        gpu_tier=plan.gpu_tier,
        gpu_count=1,
        backend=plan.backend,
        quantization=plan.quantization,
        max_model_len=effective_context,
        tensor_parallel_size=1,
        gpu_memory_utilization=_VRAM_UTIL.get(plan.quantization, 0.90),
        max_num_seqs=_MAX_SEQS.get(WorkloadType(workload_type), 32),
        dtype=dtype,
        enforce_eager=False,
        scaling_mode=scaling,
        extra_args=extra_args,
    )


def _context_for_vram(model_vram_gb: float, gpu_vram_gb: float) -> int:
    """Rough heuristic: available headroom determines how much context we can afford."""
    headroom = gpu_vram_gb - model_vram_gb
    if headroom > 40:
        return 32768
    if headroom > 20:
        return 16384
    if headroom > 8:
        return 8192
    return 4096
