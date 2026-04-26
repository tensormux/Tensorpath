from enum import Enum
from pydantic import BaseModel


class BenchmarkSource(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    IMPORTED = "imported"


class BenchmarkProfile(BaseModel):
    """Single benchmark datapoint for a specific model x gpu x backend x quant combo."""

    model_id: str
    gpu_tier: str
    backend: str
    quantization: str

    # latency (for a typical chat-length request: ~256 input, ~128 output tokens)
    ttft_ms_p50: float      # time to first token
    ttft_ms_p95: float
    itl_ms_p50: float       # inter-token latency
    itl_ms_p95: float

    # throughput
    tokens_per_sec: float   # single-request generation speed
    max_concurrent_requests: int
    throughput_at_max_concurrency_tps: float  # total tokens/sec under load

    # memory
    vram_usage_gb: float

    # cost (derived from gpu hourly rate, stored here for convenience)
    hourly_cost_usd: float

    # data provenance — never lie about this
    source: BenchmarkSource
    notes: str = ""
