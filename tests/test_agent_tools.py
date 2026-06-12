"""Tests for the agentic loop tool dispatch layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.forge.agent_tools import (
    ToolResult,
    _safe_filename,
    handle_tool,
)
from app.services.forge.models import (
    ForgeRun,
    ForgeRunStatus,
    KernelLanguage,
    KernelOp,
    KernelTaskSpec,
)


pytestmark = pytest.mark.forge


def _make_task() -> KernelTaskSpec:
    return KernelTaskSpec(
        op=KernelOp.RMSNORM,
        language=KernelLanguage.TRITON,
        target_gpu="RTX 4070",
        dtype="fp16",
        shape={"batch": 16, "hidden_size": 4096},
    )


def _make_run(tmp_path: Path) -> ForgeRun:
    run_id = "20260507T000000_rmsnorm_rtx-4070"
    artifact_dir = tmp_path / "forge_runs" / run_id
    candidate_dir = artifact_dir / "candidate"
    candidate_dir.mkdir(parents=True)

    run = ForgeRun(
        run_id=run_id,
        status=ForgeRunStatus.CANDIDATE_READY,
        task=_make_task(),
        skill_ids=["inference.write-triton-rmsnorm-kernel"],
        artifact_dir=str(Path("forge_runs") / run_id),
    )
    (artifact_dir / "run.json").write_text(run.model_dump_json(indent=2))
    return run


def _provider() -> MagicMock:
    return MagicMock()


def test_safe_filename_rejects_path_traversal():
    assert _safe_filename("../etc/passwd") is None
    assert _safe_filename("subdir/kernel.py") is None
    assert _safe_filename("..\\kernel.py") is None
    assert _safe_filename("/absolute/path.py") is None


def test_safe_filename_accepts_plain_names():
    assert _safe_filename("kernel.py") == "kernel.py"
    assert _safe_filename("bench.py") == "bench.py"
    assert _safe_filename("test_correctness.py") == "test_correctness.py"


def test_safe_filename_rejects_empty_and_dots():
    assert _safe_filename("") is None
    assert _safe_filename(".") is None
    assert _safe_filename("..") is None
    assert _safe_filename(".hidden") is None


def test_write_candidate_file_creates_file(tmp_path: Path):
    run = _make_run(tmp_path)
    provider = _provider()
    result = handle_tool(
        "write_candidate_file",
        {"filename": "kernel.py", "content": "# test kernel\n"},
        run=run,
        repo_root=tmp_path,
        provider=provider,
    )
    assert not result.is_error
    assert "wrote" in result.content
    assert "kernel.py" in result.content

    candidate_dir = tmp_path / run.artifact_dir / "candidate"
    written_file = candidate_dir / "kernel.py"
    assert written_file.exists()
    assert written_file.read_text() == "# test kernel\n"


def test_read_candidate_file_returns_content(tmp_path: Path):
    run = _make_run(tmp_path)
    provider = _provider()

    candidate_dir = tmp_path / run.artifact_dir / "candidate"
    (candidate_dir / "bench.py").write_text("# benchmark code\n")

    result = handle_tool(
        "read_candidate_file",
        {"filename": "bench.py"},
        run=run,
        repo_root=tmp_path,
        provider=provider,
    )
    assert not result.is_error
    assert result.content == "# benchmark code\n"


def test_read_candidate_file_missing_returns_error(tmp_path: Path):
    run = _make_run(tmp_path)
    provider = _provider()

    result = handle_tool(
        "read_candidate_file",
        {"filename": "nonexistent.py"},
        run=run,
        repo_root=tmp_path,
        provider=provider,
    )
    assert result.is_error
    assert "not found" in result.content.lower()


def test_list_candidate_files_empty_dir(tmp_path: Path):
    run = _make_run(tmp_path)
    provider = _provider()

    result = handle_tool(
        "list_candidate_files",
        {},
        run=run,
        repo_root=tmp_path,
        provider=provider,
    )
    assert not result.is_error
    assert "files" in result.content
    assert "[]" in result.content


def test_list_candidate_files_with_files(tmp_path: Path):
    run = _make_run(tmp_path)
    provider = _provider()

    candidate_dir = tmp_path / run.artifact_dir / "candidate"
    (candidate_dir / "kernel.py").write_text("# kernel\n")
    (candidate_dir / "bench.py").write_text("# bench\n")

    result = handle_tool(
        "list_candidate_files",
        {},
        run=run,
        repo_root=tmp_path,
        provider=provider,
    )
    assert not result.is_error
    assert "bench.py" in result.content
    assert "kernel.py" in result.content


def test_give_up_returns_acknowledged(tmp_path: Path):
    run = _make_run(tmp_path)
    provider = _provider()

    result = handle_tool(
        "give_up",
        {"reason": "Cannot optimize further"},
        run=run,
        repo_root=tmp_path,
        provider=provider,
    )
    assert not result.is_error
    assert "acknowledged" in result.content
    assert "Cannot optimize further" in result.content


def test_read_skill_delegates_to_provider(tmp_path: Path):
    run = _make_run(tmp_path)
    provider = _provider()
    provider.show.return_value = "# Skill markdown content"

    result = handle_tool(
        "read_skill",
        {"skill_id": "inference.write-triton-rmsnorm-kernel"},
        run=run,
        repo_root=tmp_path,
        provider=provider,
    )
    assert not result.is_error
    assert result.content == "# Skill markdown content"
    provider.show.assert_called_once_with("inference.write-triton-rmsnorm-kernel")
