#!/usr/bin/env python3
"""
Server-based benchmark runner for accurate TTFT and ITL measurements.

This starts a vLLM OpenAI-compatible server and hits it with streaming requests,
measuring actual time-to-first-token and inter-token latency from the HTTP stream.

More accurate than the offline runner for latency numbers, but slower to set up
because it needs to start and stop the server.

Usage:
    python benchmarks/runners/bench_server.py --model qwen2.5-7b --quantization awq
    python benchmarks/runners/bench_server.py --model llama3.2-3b --quantization fp16 --concurrency 4

Requires: pip install vllm torch httpx
"""

import argparse
import asyncio
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.schemas.models import MODEL_REGISTRY
from benchmarks.runners.workloads import get_prompts
from benchmarks.runners.bench import _get_hf_repo, get_vram_usage_gb, save_profile


_SERVER_PORT = 8321
_SERVER_URL = f"http://localhost:{_SERVER_PORT}"


def start_server(model_id: str, quantization: str) -> subprocess.Popen:
    """Start vLLM server as a subprocess."""
    hf_repo = _get_hf_repo(model_id, quantization)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", hf_repo,
        "--port", str(_SERVER_PORT),
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
        "--dtype", "float16",
        "--trust-remote-code",
    ]

    if quantization in ("awq", "gptq"):
        cmd.extend(["--quantization", quantization])

    print(f"Starting vLLM server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def wait_for_server(timeout: int = 300) -> bool:
    """Wait until the server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"{_SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(2)
    return False


async def measure_single_request(
    client: httpx.AsyncClient,
    prompt: str,
    model_name: str,
    max_tokens: int,
) -> dict | None:
    """Send one streaming request and measure TTFT + ITL."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    token_times: list[float] = []
    t_start = time.perf_counter()
    first_token_time = None
    num_tokens = 0

    try:
        async with client.stream(
            "POST",
            f"{_SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    t_now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = t_now
                    token_times.append(t_now)
                    num_tokens += 1

    except Exception as e:
        print(f"  Request failed: {e}")
        return None

    if not first_token_time or num_tokens < 2:
        return None

    ttft_ms = (first_token_time - t_start) * 1000

    # inter-token latencies
    itls = []
    for i in range(1, len(token_times)):
        itls.append((token_times[i] - token_times[i - 1]) * 1000)

    total_time = time.perf_counter() - t_start

    return {
        "ttft_ms": ttft_ms,
        "itl_ms_list": itls,
        "num_tokens": num_tokens,
        "total_time_sec": total_time,
    }


async def run_latency_bench(
    model_name: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
) -> dict:
    """Run streaming requests and collect latency stats."""
    all_ttft: list[float] = []
    all_itl: list[float] = []
    total_tokens = 0
    total_time = 0.0

    async with httpx.AsyncClient() as client:
        # warmup
        print("  Warming up (3 requests)...")
        for p in prompts[:3]:
            await measure_single_request(client, p, model_name, max_tokens)

        # actual measurement
        sem = asyncio.Semaphore(concurrency)

        async def bounded_request(prompt: str):
            async with sem:
                return await measure_single_request(client, prompt, model_name, max_tokens)

        print(f"  Running {len(prompts)} requests (concurrency={concurrency})...")
        tasks = [bounded_request(p) for p in prompts]
        results = await asyncio.gather(*tasks)

        for r in results:
            if r is None:
                continue
            all_ttft.append(r["ttft_ms"])
            all_itl.extend(r["itl_ms_list"])
            total_tokens += r["num_tokens"]
            total_time += r["total_time_sec"]

    tps = total_tokens / total_time if total_time > 0 else 0

    return {
        "ttft_ms_p50": round(float(np.percentile(all_ttft, 50)), 1) if all_ttft else 0,
        "ttft_ms_p95": round(float(np.percentile(all_ttft, 95)), 1) if all_ttft else 0,
        "itl_ms_p50": round(float(np.percentile(all_itl, 50)), 1) if all_itl else 0,
        "itl_ms_p95": round(float(np.percentile(all_itl, 95)), 1) if all_itl else 0,
        "tokens_per_sec": round(tps, 1),
        "num_requests": len([r for r in results if r]),
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="NeevPath server-based benchmark")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--quantization", default="awq", choices=["fp16", "bf16", "awq", "gptq"])
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--workload", default="chat")
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        print(f"Unknown model: {args.model}. Available: {', '.join(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    hf_repo = _get_hf_repo(args.model, args.quantization)

    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
        ).strip()
    except Exception:
        gpu_name = "Unknown GPU"

    print(f"NeevPath Server Benchmark")
    print(f"GPU:          {gpu_name}")
    print(f"Model:        {args.model} ({hf_repo})")
    print(f"Quantization: {args.quantization}")
    print(f"Concurrency:  {args.concurrency}")

    # start server
    proc = start_server(args.model, args.quantization)

    try:
        print("Waiting for server to be ready...")
        if not wait_for_server(timeout=300):
            print("Server failed to start within timeout.")
            proc.terminate()
            sys.exit(1)

        print("Server is ready.")
        vram = get_vram_usage_gb()
        print(f"VRAM usage: {vram:.2f} GB")

        prompts = get_prompts(args.workload, count=args.num_prompts)

        stats = asyncio.run(run_latency_bench(
            model_name=hf_repo,
            prompts=prompts,
            max_tokens=args.output_tokens,
            concurrency=args.concurrency,
        ))
        stats["vram_usage_gb"] = round(vram, 2)

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

    finally:
        print("\nStopping server...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
