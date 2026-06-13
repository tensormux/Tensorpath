from pydantic import BaseModel


class BackendInfo(BaseModel):
    name: str
    display_name: str
    supported_quantizations: list[str]
    supported_gpu_tiers: list[str]
    supports_continuous_batching: bool
    supports_paged_attention: bool
    supports_speculative_decoding: bool
    operational_complexity: float  # 0-1, lower is simpler
    notes: str = ""


class RuntimeRegistry:
    """What backends we support and what they can do.

    Kept as plain python — no need for a database here.
    """

    def __init__(self) -> None:
        self._backends: dict[str, BackendInfo] = {
            "vllm": BackendInfo(
                name="vllm",
                display_name="vLLM",
                supported_quantizations=["fp16", "bf16", "fp8", "awq", "gptq"],
                supported_gpu_tiers=["l4", "l40s", "a100-80gb", "h100", "rtx4070"],
                supports_continuous_batching=True,
                supports_paged_attention=True,
                supports_speculative_decoding=True,
                operational_complexity=0.2,
                notes="Go-to for most workloads. Broad model support, easy to deploy.",
            ),
            "tensorrt-llm": BackendInfo(
                name="tensorrt-llm",
                display_name="TensorRT-LLM",
                supported_quantizations=["fp16", "fp8", "awq"],
                supported_gpu_tiers=["l40s", "a100-80gb", "h100"],
                supports_continuous_batching=True,
                supports_paged_attention=True,
                supports_speculative_decoding=False,
                operational_complexity=0.6,
                notes="Higher peak throughput on supported models, but heavier build step.",
            ),
        }

    @property
    def backends(self) -> dict[str, BackendInfo]:
        return dict(self._backends)

    def get(self, name: str) -> BackendInfo | None:
        return self._backends.get(name)

    def supports_combo(self, backend: str, gpu_tier: str, quantization: str) -> bool:
        info = self._backends.get(backend)
        if not info:
            return False
        return (
            gpu_tier in info.supported_gpu_tiers
            and quantization in info.supported_quantizations
        )

    def get_simplicity_score(self, backend: str) -> float:
        info = self._backends.get(backend)
        if not info:
            return 0.0
        return 1.0 - info.operational_complexity
