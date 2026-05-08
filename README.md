# NeevPath — Inference Optimizer for NeevCloud

NeevCloud gives you GPUs. This tells you the best way to use them.

You give it a model, a workload type, and what you care about (latency, throughput, cost).
It gives you back the best GPU + backend + quantization combo, with numbers to back it up,
and a deployment config you can actually use.

## what it does

- takes model name + workload constraints as input
- generates candidate deployment plans (GPU x backend x quantization)
- scores and ranks them based on real benchmark data
- explains why the top pick won
- exports a deployment-ready config for NeevCloud

## quick start

```bash
# python 3.11+ required
pip install -r requirements.txt

# install kernel-skills (Node 18+ required for Forge)
bash scripts/install_kernel_skills.sh

# run the API server
python -m uvicorn app.main:app --reload --port 8000

# hit the recommend endpoint
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen2.5-7b",
    "workload_type": "chat",
    "optimization_priority": "cost",
    "constraints": {
      "max_p95_latency_ms": 250,
      "max_monthly_budget_usd": 300
    }
  }'
```

## project structure

```
app/
  api/           -> FastAPI routes
  schemas/       -> pydantic models for everything
  services/
    recommender/       -> candidate generation + scoring + ranking
    benchmark_store/   -> loads and queries benchmark profiles
    runtime_registry/  -> what backends support what
    deployment/        -> config export / deploy adapter
    explanation/       -> why we picked what we picked
benchmarks/
  profiles/      -> benchmark data (JSON) per model
tests/           -> scoring, ranking, constraint, explanation tests
```

## supported surface (mvp)

**models:** Qwen 2.5 7B, Llama 3.1 8B, Llama 3.2 3B
**GPUs:** L4, L40S, A100-80GB, H100
**backends:** vLLM, TensorRT-LLM
**quantizations:** FP16, FP8, AWQ 4-bit, GPTQ 4-bit
**priorities:** latency, throughput, cost, balanced

## how scoring works

each candidate plan gets scored on five dimensions:
1. **latency** — how close to target (or how fast absolutely)
2. **throughput** — tokens/sec and whether it meets RPM targets
3. **cost** — hourly rate relative to budget
4. **quality** — quantization degradation penalty
5. **simplicity** — operational complexity of the backend

weights shift depending on what the user optimizes for.
hard constraints (budget cap, latency ceiling, VRAM limit) filter candidates before scoring.

## benchmark data

benchmark profiles live in `benchmarks/profiles/` as JSON.
each profile has measured or estimated perf numbers per model x GPU x backend x quant combo.

all entries are labeled with their source: `measured`, `estimated`, or `imported`.
we don't fake precision — if a number is estimated, it says so.

### measured profiles (RTX 4070, single-request, vLLM)

| Model        | Quant | p95 TTFT | Tok/s |
|--------------|-------|----------|-------|
| Qwen 2.5 7B  | AWQ   | 160 ms   | 79    |
| Llama 3.1 8B | AWQ   | 161 ms   | 81    |
| Llama 3.2 3B | AWQ   | 122 ms   | 113   |
| Qwen 2.5 3B  | FP16  | 195 ms   | 66    |

run `python benchmarks/runners/bench.py --model <id> --quantization <quant>` to add more.
gated repos (Meta Llama) need `HF_TOKEN` exported.

## why kernel-skills is used

NeevPath uses [`@krxgu/kernel-skills`](https://www.npmjs.com/package/@krxgu/kernel-skills)
as an external instruction source for CUDA, Triton, quantization, benchmarking,
and kernel optimization workflows.

`kernel-skills` provides reusable expert playbooks. NeevPath does not depend on
it for execution, benchmarking, compilation, or deployment.

All execution happens inside NeevPath Forge. Forge retrieves skill bundles,
creates agent-ready prompts, accepts generated candidate kernels, validates
correctness, benchmarks performance, and promotes only verified kernels into
the local kernel registry.

This keeps `kernel-skills` general-purpose and keeps NeevPath responsible for
correctness, safety, and benchmark-backed promotion.

> Do not vendor-copy the kernel-skills repository into NeevPath. Consume it
> as a version-pinned npm package.

See [`docs/FORGE.md`](docs/FORGE.md) for the full Forge pipeline,
[`docs/KERNEL_REGISTRY.md`](docs/KERNEL_REGISTRY.md) for the verified-kernel
schema, [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) for the measurement
protocol, and [`docs/CLAIMS.md`](docs/CLAIMS.md) for the rules on what we
are and are not allowed to say about kernel speedups.

## what's next

- [x] benchmark runner scripts for local GPU measurements
- [x] UI (server-rendered Jinja templates at `/`, `/compare`, `/forge`)
- [x] comparison view between plans (`POST /api/compare`, `GET /compare`)
- [x] Forge: kernel-skills-driven kernel optimization loop with verification + promotion
- [x] verified kernel registry annotated onto recommendation results (op-level evidence)
- [ ] live deployment to NeevCloud endpoints (currently exports config artifact only)
- [ ] more models and GPU tiers
- [ ] runtime-level integration of promoted kernels (currently op-level only)
