# Benchmark methodology

Everything Forge labels as a "speedup" follows this protocol. If your bench
script doesn't, the benchmarker may parse it but the resulting number is not
something we'll defend.

## Warmup

GPU kernels need warmup. The first few invocations of a Triton or CUDA kernel
hit JIT compilation, autotuning, and cache misses. Times measured before
warmup are dominated by these effects, not the kernel.

**Rule:** at least 20 warmup iterations before any timed iteration.

## CUDA synchronization

GPU work is asynchronous. `time.perf_counter()` around a kernel launch
without synchronization measures the time it takes to *enqueue* the kernel,
not the time it takes to *complete* it.

**Rule:** call `torch.cuda.synchronize()` before and after each timed
iteration. Both bookends matter — without the prefix sync, you measure
queue-flush time from prior work.

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
fn()
torch.cuda.synchronize()
elapsed_us = (time.perf_counter() - t0) * 1e6
```

## Iterations and percentiles

Single-shot timings are noisy. Run multiple iterations and report a
percentile, not the mean.

**Rule:** at least 100 measured iterations. Report both **median** and **p95**
when you have them — median for typical, p95 for tail.

```python
samples = []
for _ in range(iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    samples.append((time.perf_counter() - t0) * 1e6)

samples.sort()
median = samples[len(samples) // 2]
p95    = samples[int(len(samples) * 0.95) - 1]
```

`benchmarks/kernels/utils.py` has a `time_cuda_function` helper that
implements this protocol.

## Baseline requirement

A speedup is meaningless without a baseline measured under the same protocol.

**Rule:** every `bench.py` must measure a PyTorch eager baseline and a
candidate kernel in the same run, on the same inputs, with the same warmup
and iteration counts.

## Output schema

`bench.py` must print exactly one JSON object on stdout containing:

```json
{
  "baseline_latency_us": 42.1,
  "candidate_latency_us": 24.8,
  "speedup": 1.69,
  "warmup_iters": 20,
  "benchmark_iters": 100,
  "gpu_name": "NVIDIA GeForce RTX 4070"
}
```

The benchmarker scans stdout from the bottom and parses the first
`{...}`-shaped line it finds, so feel free to print logs above it.

## Speedup reporting rules

- **Don't** print a speedup unless `bench.py` actually computed both
  latencies in the same run.
- **Don't** report a speedup as a model-level or endpoint-level
  improvement when the measurement is op-level. If `bench.py` is timing one
  RMSNorm in isolation, the speedup is op-level — period.
- **Don't** mix dtype, shape, or device across the baseline and candidate
  measurements.
- **Do** include `gpu_name`, `dtype`, and `shape` so the registry entry can
  be looked up later.

## Promotion threshold

The promoter refuses to add a candidate unless `speedup >= 1.10` and
`candidate_latency_us < baseline_latency_us`. This is intentionally
unambitious — it's a sanity floor that filters out flat or regressing
candidates without rewarding noise.
