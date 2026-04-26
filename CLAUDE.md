# CLAUDE.md

## Project: NeevCloud Inference Optimizer MVP

### One-line summary
Build an inference control plane on top of NeevCloud that automatically recommends and deploys the best serving configuration for a given model and workload, based on GPU type, latency target, throughput target, cost target, backend choice, and quantization path.

---

## 1. Why this project exists

NeevCloud already has the infrastructure layer:
- GPU AI Service
- AI Inference endpoints
- AI Templates
- Model Playground
- integrated storage
- autoscaling and monitoring

What is still missing is the optimization layer between "I have a model" and "I have the best production endpoint for my workload".

Today, a user can rent GPUs or use managed inference, but they still need to answer difficult deployment questions manually:
- Which GPU should I use?
- Should I run vLLM or TensorRT-LLM?
- Should I use AWQ or FP8?
- Should this endpoint be scale-to-zero or always-on?
- What is the expected cost if I need 500 RPM?
- What latency tradeoff do I make if I move from L40S to H100?

This MVP solves that gap.

The product is not another generic GPU rental page. It is a decision and deployment layer that turns workload requirements into an optimized inference setup.

### Core thesis
"NeevCloud should not only provide GPUs. It should provide the best execution path on those GPUs."

---

## 2. What this MVP actually is

This MVP is a **model-to-deployment optimization assistant** for NeevCloud.

A user enters:
- model name or Hugging Face repo
- workload type: chat, codegen, summarization, batch inference, etc.
- optimization priority: latency / throughput / cost / quality
- constraints: p95 target, RPM target, budget, max VRAM envelope

The system returns:
- recommended GPU
- recommended backend
- recommended quantization path
- estimated latency / throughput / cost
- deployable configuration
- one-click deployment target for NeevCloud

### Example input
"Deploy Qwen 2.5 7B as a customer support chatbot. Optimize for cost. Keep p95 under 250 ms. Budget under $300/month."

### Example output
- GPU: L40S
- Backend: vLLM
- Quantization: AWQ 4-bit
- Mode: scale-to-zero for dev, always-on for prod if RPM exceeds threshold
- Estimated p95: 210 to 240 ms
- Estimated monthly cost: under budget at projected RPM
- Deploy button: creates endpoint on NeevCloud

---

## 3. What this MVP is **not**

Do not overbuild.

This MVP is **not**:
- a new CUDA kernel compiler
- a new inference engine
- a replacement for vLLM or TensorRT-LLM
- a full hyperscaler-style scheduler
- a full multi-cloud platform
- a full chat-native autonomous agent like RunInfra

For MVP, the system should focus on **selection, recommendation, tuning defaults, and deployment packaging**.

The important distinction:
- We are **not** inventing kernels from scratch.
- We **are** productizing the best existing optimized execution paths.

---

## 4. Why we need it

### For the customer
Customers do not want raw infra decisions. They want outcome-driven deployment.

They think in terms of:
- "Make this model cheap"
- "Keep p95 under 200 ms"
- "I need 1000 RPM"
- "I do not want to overpay for H100 if L40S is enough"

They do **not** want to manually tune:
- backend choice
- quantization choice
- concurrency knobs
- autoscaling mode
- hardware tier fit

### For NeevCloud
NeevCloud is already a GPU-as-a-service and AI inference platform. The missing wedge is the layer that converts raw GPU supply into a differentiated inference product.

This gives NeevCloud a stronger story:
- not just GPU access
- not just inference endpoints
- but workload-aware optimized deployment

### Strategic value
This helps NeevCloud move from:
- infrastructure vendor

to:
- inference optimization platform built on owned infrastructure

That matters because infrastructure alone is easier to commoditize than outcome-oriented optimization.

---

## 5. Inspiration and product references

### Primary inspiration: RunInfra
RunInfra is useful inspiration because it publicly positions itself as a system that:
- benchmarks models on real GPUs
- searches optimized variants like AWQ, GPTQ, FP8
- applies kernel optimization
- supports TensorRT-LLM
- deploys optimized endpoints
- lets users optimize for latency, cost, throughput, or quality

We are **not** cloning RunInfra.

We are borrowing the right product pattern:
> workload intent in, optimized inference deployment out.

### Why NeevCloud can do something stronger
NeevCloud already owns the infrastructure surface:
- GPU instances
- inference endpoints
- model playground
- AI templates
- integrated storage
- autoscaling
- India-first data sovereignty positioning

That means this project should be positioned as:
> RunInfra-style intelligence, but native to NeevCloud infrastructure.

### Product inspiration summary
Take inspiration from:
- RunInfra for benchmark-driven optimization UX
- vLLM and TensorRT-LLM for runtime choices
- cloud control planes for recommendation + deploy workflow

Do not take inspiration from overcomplicated multi-agent products. The MVP should feel crisp and operational.

---

## 6. Why the kernel-optimized angle matters

This project needs a strong kernel-optimized inferencing angle because otherwise it is just another deployment dashboard.

Kernel-optimized here should mean:
- automatic selection of the best runtime path
- quantization-aware serving recommendations
- per-GPU, per-model optimization choices
- benchmark-based tuning defaults

### What "kernel optimized" should mean in MVP language
For a given model and GPU:
- choose vLLM vs TensorRT-LLM based on expected performance path
- choose AWQ vs FP8 when beneficial
- choose serving config defaults that align with the best low-level execution path
- attach benchmark evidence or benchmark priors to the recommendation

### What it should **not** mean
Do not claim:
- custom kernel synthesis for every customer
- full compiler automation
- automatic generation of novel Triton kernels

That would overstate the product.

### Correct product wording
Good:
- "Automatically chooses the best optimized inference path"
- "Benchmark-driven deployment recommendation"
- "Maps workload requirements to the best execution plan"

Bad:
- "Writes custom kernels for every model"
- "Fully autonomous kernel compiler"

---

## 7. NeevCloud moat we must preserve in the product narrative

This project must not erase NeevCloud's existing advantage.

### The moat to highlight
1. **Infra-native optimization**
   The product runs on top of NeevCloud's own GPU infrastructure, not on a disconnected software layer.

2. **India-first / sovereignty positioning**
   NeevCloud's local infrastructure, control, and data-sovereignty narrative matter for enterprise and regulated customers.

3. **End-to-end AI lifecycle**
   NeevCloud already spans compute, storage, experimentation, and inference. This optimizer should plug into that lifecycle.

4. **Transparent pricing and support**
   The optimizer should improve cost visibility, not add mystery.

5. **Existing customer base and distribution**
   Existing GPU customers are the natural first users for an optimization layer.

### Product positioning statement
"NeevCloud does not just rent GPUs. It helps customers choose and deploy the best inference path on those GPUs."

---

## 8. Hardware and environment constraints we must respect

### Local development constraints
We are building this MVP under realistic personal development constraints:
- single RTX 4070 GPU
- local development through WSL2 Ubuntu on Windows desktop
- no reliable multi-GPU setup
- no NVLink or datacenter-grade interconnects
- limited ability to reproduce very large model serving stacks locally
- limited VRAM compared to H100 / H200 class environments

### Implications of these constraints
Because of this, the MVP must be designed so that it can be built and validated with:
- small open models
- single-GPU inference runs
- a small benchmark matrix
- mocked or simulated recommendation logic where necessary
- adapter interfaces for cloud benchmarking that can later connect to NeevCloud infra

### What this means architecturally
Do **not** tightly couple the system to real large-cluster access.

Build the system so that:
- recommendation logic can run from stored benchmark profiles
- benchmark jobs are pluggable
- deployment adapters are abstracted
- local development can use a small set of supported model/GPU profiles

### Development reality
The MVP should prove the product concept with a small number of well-supported configurations, not by benchmarking every model in existence.

---

## 9. Product scope for MVP

### Supported model families for MVP
Keep the support surface intentionally small.

Recommended initial support:
- Qwen 2.5 7B
- Llama 3.1 / 3.2 8B class
- one 14B to 32B class model if feasible

### Supported GPU targets for MVP
Use a limited GPU tier matrix, for example:
- L4 / L40S class
- A100-80GB class
- H100 class

Do not try to support every GPU and every region initially.

### Supported backends for MVP
Start with:
- vLLM
- TensorRT-LLM as an optional path if deployment abstraction is feasible

If TensorRT-LLM is too heavy for local build and demo, keep it as a recommendation target with placeholder deployment integration.

### Supported optimization dimensions
- latency
- throughput
- cost
- balanced

### Supported outputs
- recommendation card
- config artifact
- benchmark evidence or benchmark prior
- deploy action or deploy payload

---

## 10. Non-goals for MVP

Explicitly out of scope unless time remains:
- full natural-language agent interface
- multi-turn optimization chat
- dynamic live benchmarking across every customer request
- cross-region fleet scheduling
- custom kernel synthesis
- self-healing production orchestration
- multi-tenant billing system
- fine-grained enterprise RBAC

---

## 11. Expected user journey

### User flow
1. User selects a model or enters a model repo
2. User describes workload and optimization target
3. System validates if the model is within supported surface
4. System looks up benchmark priors or runs benchmark jobs
5. System ranks deployment candidates
6. System explains the best recommendation
7. User exports config or deploys to NeevCloud
8. Endpoint and telemetry become visible

### Demo flow we should optimize for
The demo must look like this:
- input a model and constraints
- system evaluates 2 to 4 candidate deployment plans
- system recommends one clearly
- show tradeoff table
- deploy or export a config
- show endpoint metadata / URL / config output

---

## 12. Recommended architecture

### Core components
1. **Frontend / API layer**
   Accepts model + workload + constraints.

2. **Recommendation engine**
   Generates candidate deployment plans and ranks them.

3. **Benchmark profile store**
   Stores benchmark results for known model/backend/GPU/quantization combinations.

4. **Runtime registry**
   Defines supported backends and the capabilities of each backend.

5. **Deployment adapter**
   Converts a selected plan into a deployable NeevCloud configuration.

6. **Results + explanation layer**
   Shows why a choice was made.

### Good architecture principle
Separate these concerns strictly:
- selection logic
- benchmark execution
- deployment logic
- presentation layer

This separation is critical because local development will often use static benchmark priors instead of live cloud runs.

---

## 13. Recommendation engine design

### Inputs
- model name
- model size class
- workload type
- latency target
- throughput target
- budget target
- optional preferred GPU
- optional preferred backend

### Candidate generation
Generate candidate plans such as:
- L40S + vLLM + AWQ
- A100 + vLLM + FP16
- H100 + TensorRT-LLM + FP8

### Scoring dimensions
Each candidate should be scored across:
- fit: can the model run within memory envelope?
- latency score
- throughput score
- cost score
- quality preservation score
- operational simplicity score

### Ranking output
Return:
- best overall plan
- 1 to 2 alternatives
- explanation of tradeoffs

### Example explanation
"We recommend L40S + vLLM + AWQ because it meets your p95 target while staying below budget. H100 + FP8 is faster, but is over-provisioned for this workload and increases monthly cost materially."

---

## 14. Benchmarking strategy

### Important reality
The benchmark system is the backbone of the product story.
But for MVP, it should be lightweight and practical.

### Approach
Create a small benchmark matrix for a few supported models:
- model
- GPU tier
- backend
- quantization mode
- batch / concurrency settings
- measured latency
- measured throughput
- measured memory footprint
- estimated cost

### MVP benchmark sources
Allowed sources for MVP:
1. local single-GPU measurements on small models
2. manually curated benchmark priors
3. synthetic estimates where clearly labeled
4. future cloud benchmark adapter hooks

### Do not fake precision
If a result is estimated rather than measured, label it honestly.

### Suggested benchmark output format
Store benchmark records in JSON or SQLite with fields like:
- model_id
- gpu_tier
- backend
- quantization
- seq_len_profile
- batch_profile
- p50_latency_ms
- p95_latency_ms
- tokens_per_sec
- vram_gb
- hourly_cost_estimate
- source: measured | estimated | imported

---

## 15. Deployment integration strategy

### MVP deployment requirement
The system should be able to output a deployable artifact even if full live deployment is not yet wired.

### Acceptable MVP forms
- generated deployment config
- downloadable JSON/YAML manifest
- API payload for a future NeevCloud deployment endpoint
- mock deployment adapter with clean interface

### Best case
If we can integrate with a real NeevCloud deployment flow later, the adapter interface is already in place.

---

## 16. Suggested tech stack

### Backend
- Python
- FastAPI
- Pydantic
- SQLModel or SQLAlchemy
- SQLite for MVP

### Recommendation logic
- plain Python scoring engine first
- no need for ML ranking initially

### Frontend
Choose one:
- simple React / Next.js dashboard
- or FastAPI + server-rendered templates for speed

### Benchmark runner
- Python scripts
- pluggable backend adapters

### Config and schema
- Pydantic models
- JSON storage for benchmark records

---

## 17. Suggested repository structure

```text
project-root/
  CLAUDE.md
  README.md
  app/
    api/
    core/
    models/
    services/
      recommender/
      benchmark_store/
      deployment/
      runtime_registry/
    schemas/
    ui/
  benchmarks/
    profiles/
    runners/
    imported/
  configs/
  tests/
  scripts/
  docs/
```

### Key service ownership
- `recommender/` => ranking logic
- `benchmark_store/` => storage + query of benchmark priors
- `deployment/` => adapter for NeevCloud deploy/export
- `runtime_registry/` => supported backend capability metadata

---

## 18. Suggested implementation phases

### Phase 0: skeleton
- set up repo
- schema design
- runtime registry
- benchmark record schema
- recommendation input/output schema

### Phase 1: recommendation-only MVP
- build candidate generator
- build scoring engine
- ingest static benchmark data
- return ranked configs via API

### Phase 2: UI MVP
- simple form for model + constraints
- result cards
- alternative plans
- explanation view

### Phase 3: deployment artifact generation
- export deployable manifest
- wire mock NeevCloud deploy adapter

### Phase 4: benchmark runner integration
- local benchmarking scripts for a few supported models
- update benchmark store from measured runs

### Phase 5: polish
- compare recommendations
- improve explanations
- add simple observability view

---

## 19. Expected results

### Minimum successful result
A user can input a model and constraints and receive a ranked deployment recommendation with a clear explanation and a deployable config artifact.

### Strong successful result
The system supports a few real benchmark-backed profiles and shows measurable differences across GPU and quantization choices.

### Best realistic result for MVP
- 3 to 5 supported model profiles
- 3 GPU classes
- 2 backends
- recommendation engine with benchmark evidence
- deploy/export artifact
- demoable UI

---

## 20. Acceptance criteria

The MVP is successful if all of the following are true:

1. A user can specify model + objective + constraint set.
2. The system returns at least 2 candidate plans and 1 recommended plan.
3. The recommendation includes GPU, backend, quantization, and expected performance/cost.
4. The system explains why the chosen plan won.
5. The output can be exported as a deployable config or deployment payload.
6. The codebase is modular enough to swap in real NeevCloud deployment and real cloud benchmarking later.

---

## 21. Important product decisions to keep aligned

### Build for credibility
This project will only look strong if it is honest and well-scoped.

### Therefore
- prefer a smaller benchmark-backed surface over a huge fake surface
- prefer strong explanations over flashy vague AI UX
- prefer deterministic recommendation logic over pseudo-agent theatrics
- prefer infra-native positioning over generic SaaS language

### Most important product principle
This MVP should feel like:
- an inference optimization control plane

not like:
- a random dashboard with model dropdowns

---

## 22. Coding guidance for Claude

When implementing this project:

### Prioritize
- clean service boundaries
- strong schema definitions
- deterministic recommendation logic
- honest labeling of measured vs estimated data
- small, demoable, end-to-end slices

### Avoid
- premature abstraction for unsupported backends
- fake benchmark precision
- building a chat agent before the core system works
- tightly coupling UI directly to deployment logic

### Code quality expectations
- typed Python
- modular services
- tests for scoring and ranking
- easy local setup
- simple configuration

### Testing expectations
At minimum, write tests for:
- candidate generation
- scoring/ranking logic
- constraint filtering
- explanation generation
- config artifact generation

---

## 23. Demo narrative to keep in mind while building

The MVP should be easy to narrate like this:

> NeevCloud already gives customers GPU infrastructure and inference endpoints.
> This product adds the missing optimization layer.
> You tell it what model you want to serve and what you care about.
> It chooses the best GPU, backend, and quantization path.
> Then it gives you a deployment-ready config or deploys it directly.
> That is how NeevCloud moves from GPU-as-a-service to optimized inference-as-a-service.

If the product cannot be explained that simply, it is probably drifting.

---

## 24. Recommended first build order

Implement in this exact order:

1. schemas
2. benchmark store
3. runtime registry
4. candidate generation
5. scoring engine
6. explanation layer
7. API endpoint
8. minimal UI
9. export/deploy artifact
10. benchmark runner scripts

Do not start with the UI.
Do not start with chat.
Do not start with cloud deployment.
Start with the recommendation core.

---

## 25. Reference notes

This project is informed by publicly visible product positioning from:
- RunInfra, which markets benchmark-driven optimization, optimized model variant search, Triton-kernel optimization, TensorRT-LLM support, and chat-driven deployment workflows.
- NeevCloud, which markets GPU AI Service, AI Inference, AI Templates, Model Playground, integrated storage, autoscaling, and India-first AI infrastructure positioning.

Keep these references as product inspiration only. The MVP should stay grounded in what can actually be built under local hardware and time constraints.

