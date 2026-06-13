"""Tests for the agentic loop state management layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.forge.agent_state import (
    AgenticRunState,
    AgenticRunStatus,
    append_transcript,
    init_state,
    load_state,
    read_transcript,
    request_abort,
    state_path,
    transcript_path,
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
    artifact_dir.mkdir(parents=True)

    run = ForgeRun(
        run_id=run_id,
        status=ForgeRunStatus.CANDIDATE_READY,
        task=_make_task(),
        skill_ids=["inference.write-triton-rmsnorm-kernel"],
        artifact_dir=str(Path("forge_runs") / run_id),
    )
    (artifact_dir / "run.json").write_text(run.model_dump_json(indent=2))
    return run


def test_init_state_creates_file(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path, max_iterations=5, cost_cap_usd=3.0)

    path = state_path(tmp_path, run)
    assert path.exists()
    assert state.run_id == run.run_id


def test_init_state_truncates_transcript(tmp_path: Path):
    run = _make_run(tmp_path)
    tpath = transcript_path(tmp_path, run)
    tpath.write_text("old line 1\nold line 2\n")

    init_state(run, tmp_path)

    assert tpath.exists()
    assert tpath.read_text() == ""


def test_init_state_defaults(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)

    assert state.status == AgenticRunStatus.PENDING
    assert state.iteration == 0
    assert state.cost_usd == 0.0
    assert state.max_iterations == 5
    assert state.cost_cap_usd == 3.0


def test_save_and_load_roundtrip(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    state.iteration = 3
    state.cost_usd = 1.25
    state.last_verify_passed = True

    from app.services.forge.agent_state import save_state
    save_state(state, run, tmp_path)

    loaded = load_state(run, tmp_path)
    assert loaded is not None
    assert loaded.iteration == 3
    assert loaded.cost_usd == 1.25
    assert loaded.last_verify_passed is True


def test_load_returns_none_when_missing(tmp_path: Path):
    run = _make_run(tmp_path)
    path = state_path(tmp_path, run)
    if path.exists():
        path.unlink()

    loaded = load_state(run, tmp_path)
    assert loaded is None


def test_load_returns_none_on_corrupt_json(tmp_path: Path):
    run = _make_run(tmp_path)
    init_state(run, tmp_path)

    path = state_path(tmp_path, run)
    path.write_text("{ invalid json }")

    loaded = load_state(run, tmp_path)
    assert loaded is None


def test_request_abort_sets_flag(tmp_path: Path):
    run = _make_run(tmp_path)
    init_state(run, tmp_path)

    success = request_abort(run, tmp_path)
    assert success is True

    state = load_state(run, tmp_path)
    assert state is not None
    assert state.abort_requested is True


def test_request_abort_only_when_pending_or_running(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)

    for terminal_status in [
        AgenticRunStatus.SUCCEEDED,
        AgenticRunStatus.REJECTED,
        AgenticRunStatus.ABORTED,
        AgenticRunStatus.ERRORED,
    ]:
        state.status = terminal_status
        from app.services.forge.agent_state import save_state
        save_state(state, run, tmp_path)

        success = request_abort(run, tmp_path)
        assert success is False, f"abort should fail for {terminal_status}"


def test_request_abort_returns_false_when_no_state(tmp_path: Path):
    run = _make_run(tmp_path)
    path = state_path(tmp_path, run)
    if path.exists():
        path.unlink()

    success = request_abort(run, tmp_path)
    assert success is False


def test_append_transcript_adds_timestamp(tmp_path: Path):
    run = _make_run(tmp_path)
    init_state(run, tmp_path)

    append_transcript(run, tmp_path, {"kind": "test", "data": "value"})

    lines = read_transcript(run, tmp_path)
    assert len(lines) == 1
    assert "at" in lines[0]
    assert lines[0]["kind"] == "test"


def test_append_transcript_multiple_lines(tmp_path: Path):
    run = _make_run(tmp_path)
    init_state(run, tmp_path)

    append_transcript(run, tmp_path, {"kind": "line1"})
    append_transcript(run, tmp_path, {"kind": "line2"})
    append_transcript(run, tmp_path, {"kind": "line3"})

    lines = read_transcript(run, tmp_path)
    assert len(lines) == 3
    assert lines[0]["kind"] == "line1"
    assert lines[1]["kind"] == "line2"
    assert lines[2]["kind"] == "line3"


def test_read_transcript_empty(tmp_path: Path):
    run = _make_run(tmp_path)
    path = transcript_path(tmp_path, run)
    if path.exists():
        path.unlink()

    lines = read_transcript(run, tmp_path)
    assert lines == []


def test_read_transcript_skips_blank_lines(tmp_path: Path):
    run = _make_run(tmp_path)
    tpath = transcript_path(tmp_path, run)
    tpath.write_text('{"at": "2026-01-01T00:00:00Z", "kind": "line1"}\n\n\n')

    lines = read_transcript(run, tmp_path)
    assert len(lines) == 1
    assert lines[0]["kind"] == "line1"


def test_read_transcript_skips_invalid_json(tmp_path: Path):
    run = _make_run(tmp_path)
    tpath = transcript_path(tmp_path, run)
    tpath.write_text('{"at": "2026-01-01T00:00:00Z", "kind": "good"}\n{bad json}\n')

    lines = read_transcript(run, tmp_path)
    assert len(lines) == 1
    assert lines[0]["kind"] == "good"


def test_state_path_format(tmp_path: Path):
    run = _make_run(tmp_path)
    path = state_path(tmp_path, run)

    assert path == tmp_path / run.artifact_dir / "agent_state.json"


def test_transcript_path_format(tmp_path: Path):
    run = _make_run(tmp_path)
    path = transcript_path(tmp_path, run)

    assert path == tmp_path / run.artifact_dir / "agent_transcript.jsonl"
