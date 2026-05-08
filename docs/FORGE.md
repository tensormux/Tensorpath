# NeevPath Forge

Forge is the kernel optimization loop in NeevPath. It turns a workload into a
strict agent-ready kernel task, accepts candidate kernel files, validates them
against a PyTorch reference, benchmarks them against the baseline, and
promotes only verified kernels into a local registry that the recommendation
engine reads at request time.

Forge is **not** a runtime, compiler, kernel synthesizer, or model-serving
framework. It is a verification and promotion pipeline.

## Why kernel-skills is used

NeevPath consumes [`@krxgu/kernel-skills`](https://www.npmjs.com/package/@krxgu/kernel-skills)
as an external instruction source for CUDA, Triton, quantization, benchmarking,
and kernel optimization workflows.

`kernel-skills` provides reusable expert playbooks. NeevPath does not depend on
it for execution, benchmarking, compilation, or deployment.

All execution happens inside Forge. Forge retrieves skill bundles, creates
agent-ready prompts, accepts generated candidate kernels, validates correctness,
benchmarks performance, and promotes only verified kernels into the local
kernel registry.

This keeps `kernel-skills` general-purpose and keeps NeevPath responsible for
correctness, safety, and benchmark-backed promotion.

> Do not vendor-copy the kernel-skills repository into NeevPath. Consume it as
> a version-pinned npm package.

## How a Forge run works

1. **Create.** A user (CLI, web UI, or API) submits a `KernelTaskSpec`:
   operation, language, target GPU, dtype, shape, objective.

2. **Plan.** The task planner searches kernel-skills with concept-level
   queries, filters by language, ensures hygiene skills (correctness/testing,
   boundary handling) are included, and assembles a final 3–5 skill bundle.

3. **Prompt.** The prompt builder renders a strict Markdown task with all
   required sections — operation, target hardware, shapes, objective,
   constraints, required deliverables, correctness requirements, benchmark
   requirements, files to produce, **forbidden behavior**, and the bundled
   skills inline.

4. **Candidate.** A developer or external agent uses the prompt to produce
   five files in `forge_runs/<run_id>/candidate/`:
   `kernel.py`, `reference.py`, `test_correctness.py`, `bench.py`, `metadata.json`.

5. **Verify.** The verifier checks required files, validates `metadata.json`,
   scans for unsafe code patterns (subprocess, eval, network, writes outside
   the candidate directory), confirms CUDA is available, and runs pytest on
   `test_correctness.py` in a subprocess with a 120 s timeout.

6. **Benchmark.** Once verified, the benchmarker runs `candidate/bench.py` in
   a subprocess (300 s timeout), parses the last JSON-shaped line of stdout,
   and applies the **1.10× minimum speedup threshold**. Anything below is
   rejected.

7. **Promote.** The promoter requires both gates to have passed, generates a
   deterministic kernel ID
   (`<lang>_<op>_<gpu_slug>_<dtype>_<shape_slug>_v<n>`), copies `kernel.py`
   into `app/kernels/<lang>/verified/<op>/`, appends an entry to
   `kernel_registry/verified_kernels.json`, and writes `promotion.json`.

The web server **never imports** generated candidate code automatically.
Promoted kernels are metadata + source artifacts until manually wired into
the runtime in a future commit.

## How verified kernels reach recommendations

The recommendation engine accepts an `optimization_passes` list at
construction time. The default install attaches a `KernelRegistryPass` that
runs after scoring is complete:

- For each ranked plan, look up the registry for kernels that match the
  plan's GPU + activation dtype.
- Attach a `KernelOptimizationMetadata` annotation: `available=True`,
  `applied=False`, with the speedup number labeled `evidence_level=op_level`,
  `benchmark_scope=microbenchmark`.

Critically, the engine does **not** modify a plan's latency, throughput, or
cost numbers based on op-level evidence. The microbenchmark speedup is shown
on the recommendation page as an *opportunity*, not as an applied speedup.

See [CLAIMS.md](CLAIMS.md) for the rule on what we are and are not allowed to
say about kernel speedups.

## Hardware constraints

Local RTX 4070 (12 GB) is sufficient for:
- Triton kernel development
- RMSNorm and fused add+RMSNorm kernels
- Sampling and KV-cache helper microbenchmarks
- Correctness and benchmark harness development
- Small (3B/7B) model experiments

Local RTX 4070 is **not** sufficient for:
- Credible H100 FP8 claims
- Large model throughput claims
- Multi-GPU inference optimization
- TensorRT-LLM production benchmarks
- Endpoint-level claims for datacenter GPUs

Use the local box to develop and validate Forge mechanics. Use cloud /
datacenter GPUs to produce production-grade benchmark profiles.
