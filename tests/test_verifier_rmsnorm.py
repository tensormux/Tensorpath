"""Tests for the verifier — file checks, safety scan, report shape.

CUDA-dependent execution paths are skipped by default. Run with `pytest -m cuda`
on a machine with a CUDA GPU + torch+triton to exercise the full pytest path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.forge.models import (
    ForgeRun,
    ForgeRunStatus,
    KernelLanguage,
    KernelOp,
    KernelTaskSpec,
)
from app.services.forge.verifier import verify_candidate


pytestmark = pytest.mark.forge


def _seed_run(tmp_path: Path) -> tuple[ForgeRun, Path]:
    run_id = "20260507T120000_rmsnorm_rtx-4070"
    artifact_dir = tmp_path / "forge_runs" / run_id
    candidate_dir = artifact_dir / "candidate"
    candidate_dir.mkdir(parents=True)
    run = ForgeRun(
        run_id=run_id,
        status=ForgeRunStatus.PROMPT_READY,
        task=KernelTaskSpec(
            op=KernelOp.RMSNORM,
            language=KernelLanguage.TRITON,
            target_gpu="RTX 4070",
            dtype="fp16",
            shape={"hidden_size": 4096},
        ),
        skill_ids=[],
        artifact_dir=str(Path("forge_runs") / run_id),
    )
    (artifact_dir / "run.json").write_text(run.model_dump_json(indent=2))
    return run, candidate_dir


def _write_minimal_candidate(candidate_dir: Path):
    (candidate_dir / "kernel.py").write_text("def rmsnorm(x, w, eps=1e-6):\n    return x\n")
    (candidate_dir / "reference.py").write_text("def rmsnorm_ref(x, w, eps=1e-6):\n    return x\n")
    (candidate_dir / "test_correctness.py").write_text(
        "def test_smoke():\n    assert True\n"
    )
    (candidate_dir / "bench.py").write_text("print('{}')\n")
    (candidate_dir / "metadata.json").write_text(json.dumps({
        "name": "test_candidate",
        "op": "rmsnorm",
        "language": "triton",
        "kernel_file": "kernel.py",
    }))


def test_verifier_rejects_when_candidate_dir_missing(tmp_path: Path):
    run, candidate_dir = _seed_run(tmp_path)
    candidate_dir.rmdir()
    new_run, result = verify_candidate(run, tmp_path, require_cuda=False)
    assert not result.passed
    assert "candidate directory does not exist" in result.failure_reason
    assert new_run.status == ForgeRunStatus.REJECTED


def test_verifier_rejects_when_required_file_missing(tmp_path: Path):
    run, candidate_dir = _seed_run(tmp_path)
    _write_minimal_candidate(candidate_dir)
    (candidate_dir / "bench.py").unlink()  # remove one required file
    new_run, result = verify_candidate(run, tmp_path, require_cuda=False)
    assert not result.passed
    assert "missing required candidate file: bench.py" in result.failure_reason


def test_verifier_rejects_invalid_metadata(tmp_path: Path):
    run, candidate_dir = _seed_run(tmp_path)
    _write_minimal_candidate(candidate_dir)
    (candidate_dir / "metadata.json").write_text("{not json")
    new_run, result = verify_candidate(run, tmp_path, require_cuda=False)
    assert not result.passed
    assert "metadata.json is not valid JSON" in result.failure_reason


def test_verifier_rejects_subprocess_import(tmp_path: Path):
    run, candidate_dir = _seed_run(tmp_path)
    _write_minimal_candidate(candidate_dir)
    (candidate_dir / "kernel.py").write_text(
        "import subprocess\nsubprocess.run(['echo', 'pwned'])\n"
    )
    new_run, result = verify_candidate(run, tmp_path, require_cuda=False)
    assert not result.passed
    assert "unsafe code patterns" in result.failure_reason
    assert "subprocess" in result.failure_reason


def test_verifier_rejects_eval_call(tmp_path: Path):
    run, candidate_dir = _seed_run(tmp_path)
    _write_minimal_candidate(candidate_dir)
    (candidate_dir / "kernel.py").write_text(
        "def rmsnorm(x, w):\n    return eval('x')\n"
    )
    new_run, result = verify_candidate(run, tmp_path, require_cuda=False)
    assert not result.passed
    assert "eval" in result.failure_reason


def test_verifier_rejects_open_outside_candidate(tmp_path: Path):
    run, candidate_dir = _seed_run(tmp_path)
    _write_minimal_candidate(candidate_dir)
    (candidate_dir / "kernel.py").write_text(
        "def rmsnorm(x, w):\n    f = open('/etc/passwd', 'r')\n    return x\n"
    )
    new_run, result = verify_candidate(run, tmp_path, require_cuda=False)
    assert not result.passed
    assert "absolute or" in result.failure_reason


def test_verifier_rejects_when_cuda_required_but_unavailable(tmp_path: Path, monkeypatch):
    """If require_cuda=True and torch.cuda.is_available() is False, refuse."""
    run, candidate_dir = _seed_run(tmp_path)
    _write_minimal_candidate(candidate_dir)
    monkeypatch.setattr(
        "app.services.forge.verifier._cuda_available", lambda: False,
    )
    new_run, result = verify_candidate(run, tmp_path, require_cuda=True)
    assert not result.passed
    assert "CUDA unavailable" in result.failure_reason


def test_verifier_passes_minimal_candidate_when_cuda_check_skipped(tmp_path: Path):
    """With require_cuda=False and a trivial passing test, verification passes."""
    run, candidate_dir = _seed_run(tmp_path)
    _write_minimal_candidate(candidate_dir)
    new_run, result = verify_candidate(run, tmp_path, require_cuda=False)
    assert result.passed, f"unexpected failure: {result.failure_reason}"
    assert new_run.status == ForgeRunStatus.VERIFIED
    # report file should exist
    report_path = tmp_path / new_run.artifact_dir / "verification_report.json"
    assert report_path.exists()


def test_verifier_writes_report_even_on_failure(tmp_path: Path):
    run, candidate_dir = _seed_run(tmp_path)
    # Don't write any candidate files — should fail file-check
    new_run, result = verify_candidate(run, tmp_path, require_cuda=False)
    assert not result.passed
    report_path = tmp_path / new_run.artifact_dir / "verification_report.json"
    assert report_path.exists()
    body = json.loads(report_path.read_text())
    assert body["passed"] is False
