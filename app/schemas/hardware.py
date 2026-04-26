from enum import Enum
from pydantic import BaseModel


class GpuTier(str, Enum):
    L4 = "l4"
    L40S = "l40s"
    A100_80GB = "a100-80gb"
    H100 = "h100"


class GpuSpec(BaseModel):
    tier: GpuTier
    name: str
    vram_gb: float
    fp16_tflops: float
    fp8_tflops: float
    memory_bandwidth_gbps: float
    hourly_cost_usd: float

    @property
    def monthly_cost_usd(self) -> float:
        return self.hourly_cost_usd * 730  # avg hours in a month


# real-world specs and approximate neevcloud-style pricing
GPU_CATALOG: dict[GpuTier, GpuSpec] = {
    GpuTier.L4: GpuSpec(
        tier=GpuTier.L4,
        name="NVIDIA L4",
        vram_gb=24,
        fp16_tflops=121,
        fp8_tflops=242,
        memory_bandwidth_gbps=300,
        hourly_cost_usd=0.80,
    ),
    GpuTier.L40S: GpuSpec(
        tier=GpuTier.L40S,
        name="NVIDIA L40S",
        vram_gb=48,
        fp16_tflops=362,
        fp8_tflops=724,
        memory_bandwidth_gbps=864,
        hourly_cost_usd=1.50,
    ),
    GpuTier.A100_80GB: GpuSpec(
        tier=GpuTier.A100_80GB,
        name="NVIDIA A100 80GB",
        vram_gb=80,
        fp16_tflops=312,
        fp8_tflops=624,
        memory_bandwidth_gbps=2039,
        hourly_cost_usd=3.00,
    ),
    GpuTier.H100: GpuSpec(
        tier=GpuTier.H100,
        name="NVIDIA H100 SXM",
        vram_gb=80,
        fp16_tflops=990,
        fp8_tflops=1979,
        memory_bandwidth_gbps=3350,
        hourly_cost_usd=4.50,
    ),
}
