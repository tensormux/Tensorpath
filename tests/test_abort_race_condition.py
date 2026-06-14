"""Tests for the abort race condition fix in agent_runner.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.forge.agent_runner import (
    AgenticRunConfig,
    _sync_abort_flag,
    run_agentic_loop,
)
from app.services.forge.agent_state import (
    AgenticRunState,
    AgenticRunStatus,
    init_state,
    load_state,
    request_abort,
    save_state,
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
    (artifact_dir / "skill_bundle.md").write_text("# Test skill bundle\n")

    run = ForgeRun(
        run_id=run_id,
        status=ForgeRunStatus.CANDIDATE_READY,
        task=_make_task(),
        skill_ids=["inference.write-triton-rmsnorm-kernel"],
        artifact_dir=str(Path("forge_runs") / run_id),
    )
    (artifact_dir / "run.json").write_text(run.model_dump_json(indent=2))
    return run


def _make_config(**overrides) -> AgenticRunConfig:
    defaults = {"max_iterations": 3, "cost_cap_usd": 10.0}
    defaults.update(overrides)
    return AgenticRunConfig(**defaults)


def _text_block(text: str):
    block = SimpleNamespace(type="text", text=text)
    block.model_dump = lambda: {"type": "text", "text": text}
    return block


def _tool_use_block(name: str, tool_input: dict, block_id: str = "toolu_1"):
    block = SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)
    block.model_dump = lambda: {"type": "tool_use", "name": name, "input": tool_input, "id": block_id}
    return block


def _mock_response(stop_reason: str, content: list, usage=None):
    if usage is None:
        usage = SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    return SimpleNamespace(stop_reason=stop_reason, content=content, usage=usage)


# ---- _sync_abort_flag tests ----------------------------------------------


def test_sync_abort_flag_preserves_abort_when_set_on_disk(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    state.abort_requested = False
    save_state(state, run, tmp_path)

    request_abort(run, tmp_path)

    _sync_abort_flag(state, run, tmp_path)

    assert state.abort_requested is True


def test_sync_abort_flag_does_nothing_when_not_set_on_disk(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    state.abort_requested = False
    save_state(state, run, tmp_path)

    _sync_abort_flag(state, run, tmp_path)

    assert state.abort_requested is False


def test_sync_abort_flag_preserves_existing_abort_in_memory(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    state.abort_requested = True
    save_state(state, run, tmp_path)

    _sync_abort_flag(state, run, tmp_path)

    assert state.abort_requested is True


# ---- Race condition scenario tests ---------------------------------------


def test_abort_during_tool_execution_is_preserved(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config(max_iterations=3)

    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            request_abort(run, tmp_path)
            return _mock_response("tool_use", [_tool_use_block("list_candidate_files", {})])
        return _mock_response("end_turn", [_text_block("Done")])

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = side_effect
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    assert state.status == AgenticRunStatus.ABORTED


def test_abort_after_first_iteration_stops_loop(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config(max_iterations=5)

    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_response("tool_use", [_tool_use_block("list_candidate_files", {})])
        elif call_count[0] == 2:
            request_abort(run, tmp_path)
            return _mock_response("tool_use", [_tool_use_block("list_candidate_files", {})])
        return _mock_response("end_turn", [_text_block("Done")])

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = side_effect
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    assert state.status == AgenticRunStatus.ABORTED
    assert state.iteration == 2


def test_abort_flag_survives_multiple_save_state_calls(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config(max_iterations=3)

    def side_effect(*args, **kwargs):
        request_abort(run, tmp_path)
        return _mock_response("tool_use", [_tool_use_block("list_candidate_files", {})])

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = side_effect
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    assert state.status == AgenticRunStatus.ABORTED

    disk_state = load_state(run, tmp_path)
    assert disk_state.abort_requested is True
    assert disk_state.status == AgenticRunStatus.ABORTED
