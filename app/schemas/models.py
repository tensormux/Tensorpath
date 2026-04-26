from enum import Enum
from pydantic import BaseModel


class ModelFamily(str, Enum):
    QWEN2_5 = "qwen2.5"
    LLAMA3_1 = "llama3.1"
    LLAMA3_2 = "llama3.2"


class ModelInfo(BaseModel):
    model_id: str
    family: ModelFamily
    display_name: str
    param_billions: float
    architecture: str
    default_dtype: str  # fp16, bf16
    base_vram_gb: float  # approximate vram at fp16 with kv cache overhead
    hf_repo: str | None = None
    context_length: int = 4096
    requires_hf_auth: bool = False


# mvp model registry — small and honest
MODEL_REGISTRY: dict[str, ModelInfo] = {
    "qwen2.5-7b": ModelInfo(
        model_id="qwen2.5-7b",
        family=ModelFamily.QWEN2_5,
        display_name="Qwen 2.5 7B",
        param_billions=7.0,
        architecture="transformer-decoder",
        default_dtype="bf16",
        base_vram_gb=14.5,
        hf_repo="Qwen/Qwen2.5-7B-Instruct",
        context_length=32768,
    ),
    "qwen2.5-3b": ModelInfo(
        model_id="qwen2.5-3b",
        family=ModelFamily.QWEN2_5,
        display_name="Qwen 2.5 3B",
        param_billions=3.0,
        architecture="transformer-decoder",
        default_dtype="bf16",
        base_vram_gb=6.5,
        hf_repo="Qwen/Qwen2.5-3B-Instruct",
        context_length=32768,
    ),
    "llama3.1-8b": ModelInfo(
        model_id="llama3.1-8b",
        family=ModelFamily.LLAMA3_1,
        display_name="Llama 3.1 8B",
        param_billions=8.0,
        architecture="transformer-decoder",
        default_dtype="bf16",
        base_vram_gb=16.0,
        hf_repo="meta-llama/Llama-3.1-8B-Instruct",
        context_length=131072,
        requires_hf_auth=True,
    ),
    "llama3.2-3b": ModelInfo(
        model_id="llama3.2-3b",
        family=ModelFamily.LLAMA3_2,
        display_name="Llama 3.2 3B",
        param_billions=3.0,
        architecture="transformer-decoder",
        default_dtype="bf16",
        base_vram_gb=6.5,
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        context_length=131072,
        requires_hf_auth=True,
    ),
}
