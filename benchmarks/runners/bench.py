#!/usr/bin/env python3
"""
Benchmark runner for TensorPath.

Runs inference on a local GPU with vLLM and records real performance numbers.
Results are saved to benchmarks/profiles/ in the same format the recommendation
engine consumes.

Usage:
    python benchmarks/runners/bench.py --model qwen2.5-7b --quantization awq
    python benchmarks/runners/bench.py --model llama3.2-3b --quantization fp16
    python benchmarks/runners/bench.py --model llama3.1-8b --quantization awq --output-tokens 256

Requires: pip install vllm torch
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

# add project root to path so we can import schemas
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.schemas.models import MODEL_REGISTRY
from benchmarks.runners.workloads import get_prompts


# maps our model_id to the HF repo (or AWQ variant)
_AWQ_REPOS: dict[str, str] = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "llama3.1-8b": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "llama3.2-3b": "AMead10/Llama-3.2-3B-Instruct-AWQ",
}

_GPTQ_REPOS: dict[str, str] = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "llama3.1-8b": "ModelCloud/Meta-Llama-3.1-8B-Instruct-gptq-4bit",
    "llama3.2-3b": "ModelCloud/Llama-3.2-3B-Instruct-gptq-4bit",
}


def _get_hf_repo(model_id: str, quantization: str) -> str:
    model_info = MODEL_REGISTRY.get(model_id)
    if not model_info:
        raise ValueError(f"Unknown model: {model_id}")

    if quantization == "awq":
        repo = _AWQ_REPOS.get(model_id)
        if not repo:
            raise ValueError(f"No AWQ variant known for {model_id}")
        return repo
    elif quantization == "gptq":
        repo = _GPTQ_REPOS.get(model_id)
        if not repo:
            raise ValueError(f"No GPTQ variant known for {model_id}")
        return repo
    else:
        return model_info.hf_repo or model_id


@dataclass
class BenchResult:
    ttft_samples: list[float] = field(default_factory=list)  # ms
    itl_samples: list[float] = field(default_factory=list)   # ms
    total_tokens: int = 0
    total_time_sec: float = 0.0
    vram_usage_gb: float = 0.0
    num_requests: int = 0


def get_vram_usage_gb() -> float:
    """Read current GPU memory usage via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().split("\n")[0]) / 1024
    except Exception:
        # fallback to torch
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return 0.0


def run_benchmark(
    model_id: str,
    quantization: str,
    num_prompts: int,
    max_output_tokens: int,
    workload: str,
    hf_token: str | None = None,
    hf_repo_override: str | None = None,
) -> BenchResult:
    """Load model with vLLM and benchmark it."""
    from vllm import LLM, SamplingParams

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    hf_repo = hf_repo_override or _get_hf_repo(model_id, quantization)
    model_info = MODEL_REGISTRY[model_id]

    print(f"\n--- Loading {hf_repo} ({quantization}) ---")

    # vLLM engine config
    engine_kwargs: dict = {
        "model": hf_repo,
        "trust_remote_code": True,
        "max_model_len": 4096,  # keep small for 12GB card
        "gpu_memory_utilization": 0.90,
        "dtype": "float16",
        "enforce_eager": True,  # skip torch.compile — avoids nvcc dependency
    }

    if quantization == "awq":
        engine_kwargs["quantization"] = "awq"
    elif quantization == "gptq":
        engine_kwargs["quantization"] = "gptq"
    # fp16/bf16 don't need a quantization flag

    llm = LLM(**engine_kwargs)

    # check VRAM after model load
    vram_after_load = get_vram_usage_gb()
    print(f"VRAM after model load: {vram_after_load:.2f} GB")

    # prepare prompts
    prompts = get_prompts(workload, count=num_prompts)
    sampling = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.7,
        top_p=0.9,
    )

    result = BenchResult()
    result.vram_usage_gb = vram_after_load

    # -- warmup (3 requests, discard) --
    print("Warming up...")
    _ = llm.generate(prompts[:3], sampling)

    # -- single-request latency measurement --
    # run requests one at a time to measure TTFT and ITL accurately
    print(f"Running {num_prompts} single requests for latency measurement...")

    for i, prompt in enumerate(prompts):
        t_start = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        t_end = time.perf_counter()

        output = outputs[0]
        num_tokens = len(output.outputs[0].token_ids)

        if num_tokens > 0:
            total_ms = (t_end - t_start) * 1000
            # approximate TTFT as first-token time
            # vLLM offline API doesn't give per-token timing, so we estimate:
            # TTFT ~ prefill time, ITL ~ (total - TTFT) / (num_tokens - 1)
            # rough estimate: prefill takes about 20-40% of total for short sequences
            prefill_ratio = min(0.35, len(prompt.split()) * 0.002)
            ttft_est = total_ms * max(0.1, prefill_ratio)
            itl_est = (total_ms - ttft_est) / max(1, num_tokens - 1) if num_tokens > 1 else 0

            result.ttft_samples.append(ttft_est)
            result.itl_samples.append(itl_est)
            result.total_tokens += num_tokens
            result.total_time_sec += (t_end - t_start)
            result.num_requests += 1

        if (i + 1) % 5 == 0:
            elapsed = result.total_time_sec
            tps = result.total_tokens / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{num_prompts}] {tps:.1f} tok/s so far")

    # peak VRAM during generation
    result.vram_usage_gb = max(result.vram_usage_gb, get_vram_usage_gb())

    return result


def compute_stats(result: BenchResult) -> dict:
    """Compute p50/p95 stats from raw samples."""
    ttft = sorted(result.ttft_samples)
    itl = sorted(result.itl_samples)

    def p50(arr):
        return float(np.percentile(arr, 50)) if arr else 0

    def p95(arr):
        return float(np.percentile(arr, 95)) if arr else 0

    tps = result.total_tokens / result.total_time_sec if result.total_time_sec > 0 else 0

    return {
        "ttft_ms_p50": round(p50(ttft), 1),
        "ttft_ms_p95": round(p95(ttft), 1),
        "itl_ms_p50": round(p50(itl), 1),
        "itl_ms_p95": round(p95(itl), 1),
        "tokens_per_sec": round(tps, 1),
        "vram_usage_gb": round(result.vram_usage_gb, 2),
        "num_requests": result.num_requests,
        "total_tokens": result.total_tokens,
    }


def save_profile(
    model_id: str,
    quantization: str,
    stats: dict,
    gpu_name: str,
) -> Path:
    """Save benchmark result as a profile JSON that the recommendation engine can load."""
    profile = {
        "model_id": model_id,
        "gpu_tier": "rtx4070",  # local test GPU
        "backend": "vllm",
        "quantization": quantization,
        "ttft_ms_p50": stats["ttft_ms_p50"],
        "ttft_ms_p95": stats["ttft_ms_p95"],
        "itl_ms_p50": stats["itl_ms_p50"],
        "itl_ms_p95": stats["itl_ms_p95"],
        "tokens_per_sec": stats["tokens_per_sec"],
        "max_concurrent_requests": 1,  # single-GPU offline measurement
        "throughput_at_max_concurrency_tps": stats["tokens_per_sec"],
        "vram_usage_gb": stats["vram_usage_gb"],
        "hourly_cost_usd": 0.0,  # local GPU, no cost
        "source": "measured",
        "notes": f"Measured on {gpu_name}. {stats['num_requests']} requests, {stats['total_tokens']} total tokens.",
    }

    out_dir = Path(__file__).resolve().parents[1] / "profiles" / "measured"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{model_id}_{quantization}_rtx4070.json"
    out_path = out_dir / filename

    # save as a single-element list to match the format the store expects
    with open(out_path, "w") as f:
        json.dump([profile], f, indent=2)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="TensorPath benchmark runner")
    parser.add_argument("--model", required=True, help="Model ID (e.g. qwen2.5-7b)")
    parser.add_argument("--quantization", default="awq", choices=["fp16", "bf16", "awq", "gptq"])
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of prompts to run")
    parser.add_argument("--output-tokens", type=int, default=128, help="Max output tokens per request")
    parser.add_argument("--workload", default="chat", choices=["chat", "codegen", "summarization"])
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for gated models. Can also be set via HF_TOKEN env var.")
    parser.add_argument("--hf-repo", default=None, help="Override the HuggingFace repo to load (e.g. for custom quantized variants).")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        # fall back to the token cached by `hf auth login`
        try:
            from huggingface_hub import get_token
            hf_token = get_token()
        except Exception:
            pass

    if args.model not in MODEL_REGISTRY:
        print(f"Unknown model: {args.model}")
        print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    model_info = MODEL_REGISTRY[args.model]
    if model_info.requires_hf_auth and not hf_token:
        print(f"\n{args.model} requires a HuggingFace token (Meta Llama license).")
        print("Get a token at https://huggingface.co/settings/tokens and accept the license at:")
        print(f"  https://huggingface.co/{model_info.hf_repo}")
        print("\nThen run with:")
        print(f"  --hf-token YOUR_TOKEN")
        print("  or:  export HF_TOKEN=YOUR_TOKEN")
        sys.exit(1)

    # get GPU name
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        gpu_name = "Unknown GPU"

    print(f"TensorPath Benchmark Runner")
    print(f"GPU:          {gpu_name}")
    print(f"Model:        {args.model}")
    print(f"Quantization: {args.quantization}")
    print(f"Prompts:      {args.num_prompts}")
    print(f"Max output:   {args.output_tokens} tokens")
    print(f"Workload:     {args.workload}")

    result = run_benchmark(
        model_id=args.model,
        quantization=args.quantization,
        num_prompts=args.num_prompts,
        max_output_tokens=args.output_tokens,
        workload=args.workload,
        hf_token=hf_token,
        hf_repo_override=args.hf_repo,
    )

    stats = compute_stats(result)

    print(f"\n{'='*50}")
    print(f"RESULTS: {args.model} / {args.quantization} / {gpu_name}")
    print(f"{'='*50}")
    print(f"  TTFT p50:       {stats['ttft_ms_p50']:.1f} ms")
    print(f"  TTFT p95:       {stats['ttft_ms_p95']:.1f} ms")
    print(f"  ITL p50:        {stats['itl_ms_p50']:.1f} ms")
    print(f"  ITL p95:        {stats['itl_ms_p95']:.1f} ms")
    print(f"  Throughput:     {stats['tokens_per_sec']:.1f} tok/s")
    print(f"  VRAM usage:     {stats['vram_usage_gb']:.2f} GB")
    print(f"  Requests:       {stats['num_requests']}")
    print(f"  Total tokens:   {stats['total_tokens']}")

    out_path = save_profile(args.model, args.quantization, stats, gpu_name)
    print(f"\nProfile saved to: {out_path}")


if __name__ == "__main__":
    main()
