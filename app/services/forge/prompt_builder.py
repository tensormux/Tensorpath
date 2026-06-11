"""Render a strict, agent-ready Markdown prompt from a SkillBundleSpec.

The prompt is deliberately rigid: every section is required, the file
list under `forge_runs/<run_id>/candidate/` is fixed, and the "Forbidden
Behavior" section is stated explicitly so an agent can't quietly drift.

The skill bundle is embedded verbatim at the bottom under "Skill Bundle"
so the agent has all the kernel-skills guidance in-context.
"""

from __future__ import annotations

from app.services.forge.models import KernelTaskSpec, SkillBundleSpec


def _format_shape(shape: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in shape.items())


def _format_constraints(constraints: dict) -> str:
    if not constraints:
        return "(none)"
    return "\n".join(f"- `{k}`: {v}" for k, v in constraints.items())


def build_agent_prompt(spec: SkillBundleSpec, run_id: str | None = None) -> str:
    task: KernelTaskSpec = spec.task
    rid_block = (
        f"\nThe agent must write candidate files under `forge_runs/{run_id}/candidate/`.\n"
        if run_id
        else "\nThe agent must write candidate files under `forge_runs/<run_id>/candidate/`.\n"
    )

    skills_listed = (
        "\n".join(f"- `{s}`" for s in spec.skill_ids)
        if spec.skill_ids
        else "_(no external skills retrieved — follow the fallback guidance below)_"
    )

    return f"""# Kernel Optimization Task

You are generating a {task.language.value.title()} kernel candidate for an
inference optimization pipeline. TensorPath Forge will validate your output
against a reference implementation and benchmark it against the baseline.
Code that fails verification or speedup thresholds will be rejected and
will not be promoted into the kernel registry.

## Operation

{task.op.value}

## Target Hardware

- GPU: {task.target_gpu}
- Language: {task.language.value}
- dtype: {task.dtype}

## Tensor Shapes

{_format_shape(task.shape)}

## Dtype

{task.dtype}

## Objective

{task.objective}. Minimize the chosen objective while preserving numerical
correctness against a PyTorch reference implementation.

## Constraints

{_format_constraints(task.constraints)}

## Required Deliverables

Create only these files:
{rid_block}
- `kernel.py` — the {task.language.value} kernel implementation
- `reference.py` — a PyTorch reference implementation of the same op
- `test_correctness.py` — pytest-style correctness tests (see below)
- `bench.py` — benchmark script (see below)
- `metadata.json` — the candidate metadata block (see below)

## Correctness Requirements

The candidate must compare against a PyTorch reference implementation of
{task.op.value}. Important: do not use `torch.nn.LayerNorm` as the reference for
RMSNorm — RMSNorm does not subtract the mean.

Test:
- a range of batch sizes including 1, 2, 8, 16, 32
- a range of hidden / sequence sizes including non-power-of-two values
  (e.g. 1024, 2048, 4096, 4097) so partial-block tails are exercised
- the dtypes listed under "Dtype" above
- adversarial inputs: zeros, large values, small values, mixed signs

Use `torch.allclose` with explicit `rtol`/`atol` tolerances appropriate
to the dtype. State the chosen tolerances in `metadata.json`.

## Benchmark Requirements

`bench.py` must:

- run a PyTorch eager baseline of the same op on the same inputs
- run the candidate kernel on the same inputs
- use at least 20 warmup iterations and at least 100 measured iterations
- call `torch.cuda.synchronize()` before and after each timing block
- emit a single-line JSON object on stdout containing:
  - `baseline_latency_us`
  - `candidate_latency_us`
  - `speedup` (= baseline / candidate)
  - `warmup_iters`
  - `benchmark_iters`
  - `gpu_name`

Do not print a speedup unless `bench.py` actually computed both
latencies in the same run.

## Files to Produce

Place all output under the candidate directory only:

```
candidate/
  kernel.py
  reference.py
  test_correctness.py
  bench.py
  metadata.json
```

## Forbidden Behavior

- Do not modify any project files outside the candidate directory.
- Do not import generated code from anywhere outside the candidate directory.
- Do not call `os.system`, `subprocess`, `shutil.rmtree`, `socket`, `requests`,
  `eval`, or `exec`.
- Do not open files for writing outside the candidate directory.
- Do not skip boundary masks. Do not assume `hidden_size` is a power of two
  unless the kernel explicitly asserts it and the test plan justifies why.
- Do not report performance numbers that were not produced by `bench.py` in
  the same run.
- Do not claim end-to-end speedup. Speedup is op-level only at this stage.

## Skill Bundle

The agent must follow the guidance in the bundle below. The skills selected
for this task were:

{skills_listed}

---

{spec.skill_bundle}
"""
