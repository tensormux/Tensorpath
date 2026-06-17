"""Forge run lifecycle: directory creation, artifact persistence, reload.

A Forge run is a self-contained directory under `forge_runs/<run_id>/`:

    forge_runs/<run_id>/
        task.json
        skill_ids.json
        skill_bundle.md
        agent_prompt.md
        candidate/                # populated by an agent or developer
            kernel.py
            reference.py
            test_correctness.py
            bench.py
            metadata.json
        verification_report.json  # written by verifier
        benchmark_report.json     # written by benchmarker
        promotion.json            # written by promoter

Status transitions are recorded in `run.json` so the CLI / API can reload
them. Each writer updates run.json so the on-disk state always reflects
what's been produced.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from app.services.forge._atomic_write import atomic_write_json
from app.services.forge.models import (
    ForgeRun,
    ForgeRunStatus,
    KernelTaskSpec,
)


_FORGE_RUNS_DIR_NAME = "forge_runs"


def _slugify(text: str) -> str:
    """Lowercase, ascii-only, dash-separated. For run IDs and kernel IDs."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "x"


def make_run_id(task: KernelTaskSpec, *, now: datetime | None = None) -> str:
    """Run ID: <yyyymmddTHHMMSS>_<op>_<gpu_slug>."""
    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{task.op.value}_{_slugify(task.target_gpu)}"


def runs_root(repo_root: Path) -> Path:
    return repo_root / _FORGE_RUNS_DIR_NAME


def run_dir(repo_root: Path, run_id: str) -> Path:
    return runs_root(repo_root) / run_id


def create_run(
    task: KernelTaskSpec,
    repo_root: Path,
    *,
    run_id: str | None = None,
) -> ForgeRun:
    """Create a fresh run directory and persist task.json + run.json.

    Status starts at PLANNED. The candidate/ subdirectory is created up-front
    so agents have a clear place to put their files.
    """
    rid = run_id or make_run_id(task)
    artifact_dir = run_dir(repo_root, rid)
    if artifact_dir.exists():
        raise FileExistsError(f"Forge run already exists: {artifact_dir}")

    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "candidate").mkdir(parents=True, exist_ok=False)

    atomic_write_json(artifact_dir / "task.json", task.model_dump_json(indent=2))

    run = ForgeRun(
        run_id=rid,
        status=ForgeRunStatus.PLANNED,
        task=task,
        skill_ids=[],
        artifact_dir=str(artifact_dir.relative_to(repo_root)),
    )
    _save_run_state(run, repo_root)
    return run


def write_skill_artifacts(
    run: ForgeRun,
    skill_ids: list[str],
    skill_bundle_md: str,
    repo_root: Path,
) -> ForgeRun:
    artifact_dir = repo_root / run.artifact_dir
    atomic_write_json(
        artifact_dir / "skill_ids.json",
        json.dumps({"skill_ids": skill_ids}, indent=2)
    )
    (artifact_dir / "skill_bundle.md").write_text(skill_bundle_md)
    run = run.model_copy(update={"skill_ids": skill_ids})
    _save_run_state(run, repo_root)
    return run


def write_prompt(
    run: ForgeRun,
    prompt_md: str,
    repo_root: Path,
) -> ForgeRun:
    artifact_dir = repo_root / run.artifact_dir
    (artifact_dir / "agent_prompt.md").write_text(prompt_md)
    run = run.model_copy(update={"status": ForgeRunStatus.PROMPT_READY})
    _save_run_state(run, repo_root)
    return run


def load_run(run_id: str, repo_root: Path) -> ForgeRun:
    state_path = run_dir(repo_root, run_id) / "run.json"
    if not state_path.exists():
        raise FileNotFoundError(f"No Forge run found: {run_id}")
    return ForgeRun.model_validate_json(state_path.read_text())


def list_runs(repo_root: Path) -> list[ForgeRun]:
    root = runs_root(repo_root)
    if not root.exists():
        return []
    runs: list[ForgeRun] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        state = child / "run.json"
        if state.exists():
            try:
                runs.append(ForgeRun.model_validate_json(state.read_text()))
            except Exception:
                continue
    return runs


def update_run_status(
    run: ForgeRun,
    status: ForgeRunStatus,
    repo_root: Path,
) -> ForgeRun:
    """Set a new status on a run and persist run.json. Returns the updated run."""
    new_run = run.model_copy(update={"status": status})
    _save_run_state(new_run, repo_root)
    return new_run


def _save_run_state(run: ForgeRun, repo_root: Path) -> None:
    artifact_dir = repo_root / run.artifact_dir
    atomic_write_json(artifact_dir / "run.json", run.model_dump_json(indent=2))
