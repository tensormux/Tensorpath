# Verified kernel registry

A single JSON file at `kernel_registry/verified_kernels.json` lists every
kernel Forge has verified and benchmarked. The recommendation engine reads
this file at request time, so newly-promoted kernels become visible without
restarting the server.

## Schema

```json
{
  "kernels": [
    {
      "kernel_id": "triton_rmsnorm_rtx4070_fp16_h4096_v1",
      "op": "rmsnorm",
      "language": "triton",
      "source_path": "app/kernels/triton/verified/rmsnorm/triton_rmsnorm_rtx4070_fp16_h4096_v1.py",
      "target_gpu": "RTX 4070",
      "dtype": "fp16",
      "shape": { "hidden_size": 4096 },
      "verification": { "passed": true, "tolerance": { "rtol": 0.001, "atol": 0.001 } },
      "benchmark": {
        "passed": true,
        "baseline_latency_us": 42.1,
        "candidate_latency_us": 24.8,
        "speedup": 1.69,
        "warmup_iters": 20,
        "benchmark_iters": 100,
        "gpu_name": "NVIDIA GeForce RTX 4070"
      },
      "skill_ids": [
        "inference.write-triton-rmsnorm-kernel",
        "patterns.write-kernel-test-plan",
        "patterns.handle-boundary-conditions"
      ],
      "evidence_level": "op_level",
      "created_at": "2026-05-07T00:00:00Z"
    }
  ]
}
```

## Kernel IDs

Format:
```
<language>_<op>_<gpu_slug>_<dtype>_<shape_slug>_v<n>
```

- `gpu_slug` = lowercase, alphanumeric only, derived from `target_gpu`
- `shape_slug` = sorted dim list with short prefixes (`h` for `hidden_size`,
  `b` for `batch`, `s` for `seq_len`, `n` for `heads`); unknown dims keep their
  full key
- `v<n>` is bumped automatically on collision — re-promoting the same task
  doesn't overwrite the prior kernel

Examples:
```
triton_rmsnorm_rtx4070_fp16_h4096_v1
triton_rmsnorm_rtx4070_fp16_b16_h4096_v1
cuda_softmax_h100_fp8_b32_h8192_v2
```

## Evidence levels

Each entry carries an `evidence_level` that gates how aggressively the
recommendation planner uses the speedup:

| Level | Meaning | Planner behavior |
|---|---|---|
| `op_level` | Microbenchmark only — Forge ran the candidate against a PyTorch baseline for one op in isolation. | Show as an optimization opportunity. **Do not** alter cost/perf estimates. |
| `runtime_level` | Optimization integrated into a runtime path or model layer and benchmarked there. | Allow a small, conservative adjustment to estimates. |
| `endpoint_level` | End-to-end serving benchmark proves TTFT, latency, throughput, or cost improvement on the real serving stack. | Allow full benchmark-backed recommendation impact. |

Forge currently only produces `op_level` evidence. The other two levels exist
in the schema so future commits can record them without schema churn.

## Promotion rules

The promoter refuses to add an entry unless **all** of the following hold:

1. `verification_report.json` exists and `verification.passed` is `true`
2. `benchmark_report.json` exists and `benchmark.passed` is `true`
3. `benchmark.speedup >= 1.10`
4. The candidate's `kernel.py` is present in the run directory
5. The generated `kernel_id` doesn't already exist in the registry (the
   promoter bumps `v<n>` until it finds a free slot)

The promoter copies `kernel.py` into the verified-kernel source tree but
**never** imports or executes it. Activating a promoted kernel in production
is a separate, manual integration step.

## How the planner uses the registry

`KernelRegistryPass` (in `app/services/optimization/kernel_pass.py`) attaches
`KernelOptimizationMetadata` to each `DeploymentPlan` after scoring. The
match rule:

- Plan's GPU name (or tier) slug must contain (or be contained in) the
  kernel's `target_gpu` slug
- Plan's quantization → activation dtype mapping (`awq`/`gptq` → `fp16`,
  `fp8` → `fp8`/`fp16`, etc.) must match the kernel's `dtype`

When a match is found, the metadata records `available=True`,
`applied=False`. Recommendation latency / throughput / cost numbers are
**not** modified — the speedup shown is the standalone microbenchmark
result. See [CLAIMS.md](CLAIMS.md).
