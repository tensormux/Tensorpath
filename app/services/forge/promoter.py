"""Promote a verified-and-benchmarked candidate into the kernel registry.

The promoter is the only writer that touches both the registry file and the
verified-kernels source tree under `app/kernels/<lang>/verified/<op>/`. It
will refuse to do anything unless:

    1. verification_report.json exists and verification.passed = True
    2. benchmark_report.json exists and benchmark.passed = True
    3. benchmark.speedup >= MIN_SPEEDUP

If those gates pass, the promoter:

    1. Generates a deterministic kernel_id, bumping the version suffix if a
       prior kernel for the same (op, gpu, dtype, shape) already exists.
    2. Copies candidate/kernel.py into
       `app/kernels/<lang>/verified/<op>/<kernel_id>.py`.
    3. Adds an entry to the registry.
    4. Writes promotion.json into the run directory.
    5. Updates run.json status to PROMOTED.

It never overwrites an existing kernel file or registry entry.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from app.services.forge.benchmarker import MIN_SPEEDUP
from app.services.forge.models import (
    BenchmarkResult,
    ForgeRun,
    ForgeRunStatus,
    KernelOp,
    KernelTaskSpec,
    PromotedKernel,
    VerificationResult,
)
from app.services.forge.registry import KernelRegistry
from app.services.forge.runs import update_run_status


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower()) or "x"


def _shape_slug(shape: dict[str, int]) -> str:
    """Compact representation of the most distinctive shape dimensions.
    Uses single-letter prefixes for common dims, full key for anything else."""
    if not shape:
        return "noshape"
    short_keys = {"hidden_size": "h", "batch": "b", "seq_len": "s", "heads": "n"}
    parts = []
    for key in sorted(shape.keys()):
        prefix = short_keys.get(key, key.replace("_", ""))
        parts.append(f"{prefix}{shape[key]}")
    return "_".join(parts)


def make_kernel_id(task: KernelTaskSpec, version: int = 1) -> str:
    """<lang>_<op>_<gpu_slug>_<dtype>_<shape_slug>_v<n>."""
    return (
        f"{task.language.value}"
        f"_{task.op.value}"
        f"_{_slugify(task.target_gpu)}"
        f"_{task.dtype}"
        f"_{_shape_slug(task.shape)}"
        f"_v{version}"
    )


def _next_available_id(task: KernelTaskSpec, registry: KernelRegistry) -> str:
    version = 1
    while registry.has_kernel(make_kernel_id(task, version=version)):
        version += 1
        if version > 999:  # paranoia
            raise RuntimeError("could not allocate unique kernel ID")
    return make_kernel_id(task, version=version)


def _read_reports(
    run: ForgeRun,
    repo_root: Path,
) -> tuple[VerificationResult, BenchmarkResult]:
    artifact_dir = repo_root / run.artifact_dir
    ver_path = artifact_dir / "verification_report.json"
    bench_path = artifact_dir / "benchmark_report.json"
    if not ver_path.exists():
        raise ValueError("verification_report.json missing — run `forge verify` first")
    if not bench_path.exists():
        raise ValueError("benchmark_report.json missing — run `forge benchmark` first")
    return (
        VerificationResult.model_validate_json(ver_path.read_text()),
        BenchmarkResult.model_validate_json(bench_path.read_text()),
    )


def _verified_kernel_path(
    repo_root: Path,
    task: KernelTaskSpec,
    kernel_id: str,
) -> Path:
    return (
        repo_root
        / "app"
        / "kernels"
        / task.language.value
        / "verified"
        / task.op.value
        / f"{kernel_id}.py"
    )


def promote_candidate(
    run: ForgeRun,
    repo_root: Path,
) -> tuple[ForgeRun, PromotedKernel]:
    """Run all promotion gates. Raises ValueError if any fail."""
    ver, bench = _read_reports(run, repo_root)

    if not ver.passed:
        raise ValueError(
            f"cannot promote: verification did not pass ({ver.failure_reason})"
        )
    if not bench.passed:
        raise ValueError(
            f"cannot promote: benchmark did not pass ({bench.notes or 'no notes'})"
        )
    if bench.speedup < MIN_SPEEDUP:
        raise ValueError(
            f"cannot promote: speedup {bench.speedup:.3f}x below threshold {MIN_SPEEDUP}x"
        )

    candidate_kernel = repo_root / run.artifact_dir / "candidate" / "kernel.py"
    if not candidate_kernel.exists():
        raise ValueError("candidate/kernel.py missing — cannot copy source")

    registry = KernelRegistry(repo_root)
    kernel_id = _next_available_id(run.task, registry)
    dst = _verified_kernel_path(repo_root, run.task, kernel_id)

    if dst.exists():
        # Defensive: shouldn't happen because IDs are version-bumped, but
        # if it does, refuse rather than overwriting.
        raise ValueError(f"verified kernel file already exists: {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(candidate_kernel.read_bytes())

    promoted = PromotedKernel(
        kernel_id=kernel_id,
        op=run.task.op,
        language=run.task.language,
        source_path=str(dst.relative_to(repo_root)).replace("\\", "/"),
        target_gpu=run.task.target_gpu,
        dtype=run.task.dtype,
        shape=run.task.shape,
        verification=ver,
        benchmark=bench,
        skill_ids=run.skill_ids,
        evidence_level="op_level",
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    registry.add_kernel(promoted)

    artifact_dir = repo_root / run.artifact_dir
    (artifact_dir / "promotion.json").write_text(promoted.model_dump_json(indent=2))

    new_run = update_run_status(run, ForgeRunStatus.PROMOTED, repo_root)
    return new_run, promoted
