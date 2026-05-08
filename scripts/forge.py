#!/usr/bin/env python3
"""Forge CLI — drive the full kernel optimization loop from the terminal.

Subcommands implemented in this commit:
    create        — create a Forge run, retrieve skills, write the prompt
    show-prompt   — print the agent prompt for an existing run

Stubbed for next session (commits 4-6):
    verify        — run candidate correctness tests
    benchmark     — run candidate microbenchmark
    promote       — copy verified kernel into the registry
    list-kernels  — list verified kernels

The web server doesn't depend on this script — it's a developer tool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Make `app.*` importable when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


from app.services.forge.benchmarker import benchmark_candidate  # noqa: E402
from app.services.forge.models import (  # noqa: E402
    KernelLanguage,
    KernelOp,
    KernelTaskSpec,
)
from app.services.forge.prompt_builder import build_agent_prompt  # noqa: E402
from app.services.forge.promoter import promote_candidate  # noqa: E402
from app.services.forge.registry import KernelRegistry  # noqa: E402
from app.services.forge.runs import (  # noqa: E402
    create_run,
    load_run,
    write_prompt,
    write_skill_artifacts,
)
from app.services.forge.skill_provider import KernelSkillsProvider  # noqa: E402
from app.services.forge.task_planner import ForgeTaskPlanner  # noqa: E402
from app.services.forge.verifier import verify_candidate  # noqa: E402


def _parse_shape(items: list[str]) -> dict[str, int]:
    """Parse `--shape key=value` repeats into a dict[str, int]."""
    out: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--shape expects key=value, got: {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = int(v)
        except ValueError as e:
            raise SystemExit(f"--shape value must be int: {item!r}") from e
    return out


def cmd_create(args: argparse.Namespace) -> None:
    shape: dict[str, int] = {}
    if args.batch is not None:
        shape["batch"] = args.batch
    if args.hidden_size is not None:
        shape["hidden_size"] = args.hidden_size
    if args.shape:
        shape.update(_parse_shape(args.shape))
    if not shape:
        raise SystemExit("Provide --batch/--hidden-size or --shape key=value")

    task = KernelTaskSpec(
        op=KernelOp(args.op),
        language=KernelLanguage(args.language),
        target_gpu=args.gpu,
        dtype=args.dtype,
        shape=shape,
        objective=args.objective,
        model_family=args.model_family,
    )

    run = create_run(task, _REPO_ROOT)
    print(f"Created Forge run: {run.run_id}")
    print(f"  artifact dir: {_REPO_ROOT / run.artifact_dir}")

    provider = KernelSkillsProvider(repo_root=_REPO_ROOT)
    planner = ForgeTaskPlanner(provider=provider)
    bundle_spec = planner.plan(task)

    run = write_skill_artifacts(
        run, bundle_spec.skill_ids, bundle_spec.skill_bundle, _REPO_ROOT
    )
    prompt_md = build_agent_prompt(bundle_spec, run_id=run.run_id)
    run = write_prompt(run, prompt_md, _REPO_ROOT)

    print(f"  status: {run.status.value}")
    if run.skill_ids:
        print(f"  skills ({len(run.skill_ids)}): {', '.join(run.skill_ids)}")
    else:
        print("  skills: (none retrieved — fallback guidance written)")
    prompt_path = _REPO_ROOT / run.artifact_dir / "agent_prompt.md"
    print(f"  prompt: {prompt_path}")


def cmd_show_prompt(args: argparse.Namespace) -> None:
    run = load_run(args.run_id, _REPO_ROOT)
    prompt = (_REPO_ROOT / run.artifact_dir / "agent_prompt.md").read_text()
    sys.stdout.write(prompt)


def cmd_verify(args: argparse.Namespace) -> None:
    run = load_run(args.run_id, _REPO_ROOT)
    new_run, result = verify_candidate(
        run, _REPO_ROOT, require_cuda=not args.skip_cuda_check
    )
    print(f"Run: {new_run.run_id}")
    print(f"Status: {new_run.status.value}")
    print(f"Verification passed: {result.passed}")
    if result.failure_reason:
        print(f"Reason: {result.failure_reason[:500]}")
    print(f"Report: {_REPO_ROOT / new_run.artifact_dir / 'verification_report.json'}")
    if not result.passed:
        raise SystemExit(1)


def cmd_benchmark(args: argparse.Namespace) -> None:
    run = load_run(args.run_id, _REPO_ROOT)
    new_run, result = benchmark_candidate(run, _REPO_ROOT)
    print(f"Run: {new_run.run_id}")
    print(f"Status: {new_run.status.value}")
    print(f"Benchmark passed: {result.passed}")
    if result.passed or result.baseline_latency_us > 0:
        print(f"  baseline:  {result.baseline_latency_us:.2f} us")
        print(f"  candidate: {result.candidate_latency_us:.2f} us")
        print(f"  speedup:   {result.speedup:.3f}x")
    if result.notes:
        print(f"Notes: {result.notes}")
    print(f"Report: {_REPO_ROOT / new_run.artifact_dir / 'benchmark_report.json'}")
    if not result.passed:
        raise SystemExit(1)


def cmd_promote(args: argparse.Namespace) -> None:
    run = load_run(args.run_id, _REPO_ROOT)
    try:
        new_run, promoted = promote_candidate(run, _REPO_ROOT)
    except ValueError as e:
        raise SystemExit(f"Promotion refused: {e}")
    print(f"Promoted kernel: {promoted.kernel_id}")
    print(f"  source:   {promoted.source_path}")
    print(f"  speedup:  {promoted.benchmark.speedup:.3f}x")
    print(f"  evidence: {promoted.evidence_level}")
    print(f"Run status: {new_run.status.value}")


def cmd_list_kernels(args: argparse.Namespace) -> None:
    registry = KernelRegistry(_REPO_ROOT)
    entries = registry.list_kernels()
    if not entries:
        print("No verified kernels yet.")
        return
    for k in entries:
        speedup = k.get("benchmark", {}).get("speedup")
        speedup_str = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else "?"
        print(
            f"{k.get('kernel_id'):60s}  "
            f"{k.get('op'):20s}  "
            f"{k.get('target_gpu'):16s}  "
            f"{k.get('dtype'):5s}  "
            f"speedup {speedup_str}  "
            f"({k.get('evidence_level')})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="forge", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="create a Forge run + write prompt")
    p_create.add_argument("--op", required=True, choices=[o.value for o in KernelOp])
    p_create.add_argument("--language", default="triton",
                          choices=[lang.value for lang in KernelLanguage])
    p_create.add_argument("--gpu", required=True, help='e.g. "RTX 4070" or "H100"')
    p_create.add_argument("--dtype", default="fp16",
                          choices=["fp16", "bf16", "fp32", "int8", "fp8"])
    p_create.add_argument("--batch", type=int)
    p_create.add_argument("--hidden-size", type=int)
    p_create.add_argument("--shape", action="append", default=[],
                          help="extra shape dims as key=value (repeatable)")
    p_create.add_argument("--objective", default="latency",
                          choices=["latency", "throughput", "memory", "balanced"])
    p_create.add_argument("--model-family", default=None)
    p_create.set_defaults(func=cmd_create)

    p_show = sub.add_parser("show-prompt", help="print prompt for a run")
    p_show.add_argument("--run-id", required=True)
    p_show.set_defaults(func=cmd_show_prompt)

    p_verify = sub.add_parser("verify", help="run candidate correctness checks")
    p_verify.add_argument("--run-id", required=True)
    p_verify.add_argument("--skip-cuda-check", action="store_true",
                          help="skip the CUDA-required gate (test infrastructure only)")
    p_verify.set_defaults(func=cmd_verify)

    p_bench = sub.add_parser("benchmark", help="run candidate microbenchmark")
    p_bench.add_argument("--run-id", required=True)
    p_bench.set_defaults(func=cmd_benchmark)

    p_promote = sub.add_parser("promote", help="promote verified+benchmarked candidate")
    p_promote.add_argument("--run-id", required=True)
    p_promote.set_defaults(func=cmd_promote)

    p_list = sub.add_parser("list-kernels", help="list verified kernels in the registry")
    p_list.set_defaults(func=cmd_list_kernels)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
