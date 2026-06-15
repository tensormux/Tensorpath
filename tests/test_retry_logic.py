"""Tests for the retry logic in the agentic loop."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import anthropic

from app.services.forge.agent_runner import (
    AgenticRunConfig,
    _is_retryable,
    _run_single_turn_with_retry,
    run_agentic_loop,
)
from app.services.forge.agent_state import (
    AgenticRunState,
    AgenticRunStatus,
    init_state,
    load_state,
    read_transcript,
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


def _make_retryable_error(status_code: int = 500):
    """Create a retryable API error with the given status code."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.request = MagicMock()
    
    if status_code == 429:
        return anthropic.RateLimitError(message="rate limit", response=mock_response, body=None)
    elif status_code == 500:
        return anthropic.InternalServerError(message="server error", response=mock_response, body=None)
    else:
        return anthropic.APIStatusError(message="error", response=mock_response, body=None)


def _make_non_retryable_error(status_code: int = 401):
    """Create a non-retryable API error with the given status code."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.request = MagicMock()
    
    if status_code == 401:
        return anthropic.AuthenticationError(message="bad key", response=mock_response, body=None)
    elif status_code == 400:
        return anthropic.BadRequestError(message="bad request", response=mock_response, body=None)
    else:
        return anthropic.APIStatusError(message="error", response=mock_response, body=None)


# ---- _is_retryable tests -------------------------------------------------


def test_is_retryable_returns_true_for_rate_limit():
    exc = _make_retryable_error(429)
    assert _is_retryable(exc) is True


def test_is_retryable_returns_true_for_internal_server_error():
    exc = _make_retryable_error(500)
    assert _is_retryable(exc) is True


def test_is_retryable_returns_true_for_api_connection_error():
    exc = anthropic.APIConnectionError(request=MagicMock())
    assert _is_retryable(exc) is True


def test_is_retryable_returns_false_for_authentication_error():
    exc = _make_non_retryable_error(401)
    assert _is_retryable(exc) is False


def test_is_retryable_returns_false_for_bad_request():
    exc = _make_non_retryable_error(400)
    assert _is_retryable(exc) is False


# ---- _run_single_turn_with_retry tests -----------------------------------


def test_retry_succeeds_after_transient_error(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    config = _make_config(max_retries=3, retry_base_delay=0.01)

    mock_client = MagicMock()
    error = _make_retryable_error(500)
    success_response = _mock_response("end_turn", [_text_block("Done")])

    mock_client.messages.create.side_effect = [error, success_response]

    response = _run_single_turn_with_retry(
        client=mock_client,
        config=config,
        system_prompt=[],
        messages=[],
        run=run,
        repo_root=tmp_path,
        state=state,
    )

    assert response is success_response
    assert mock_client.messages.create.call_count == 2
    assert state.total_retries == 1


def test_retry_exhausted_raises_exception(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    config = _make_config(max_retries=2, retry_base_delay=0.01)

    mock_client = MagicMock()
    error = _make_retryable_error(429)
    mock_client.messages.create.side_effect = error

    with pytest.raises(anthropic.RateLimitError):
        _run_single_turn_with_retry(
            client=mock_client,
            config=config,
            system_prompt=[],
            messages=[],
            run=run,
            repo_root=tmp_path,
            state=state,
        )

    assert mock_client.messages.create.call_count == 3
    assert state.total_retries == 2


def test_retry_not_attempted_for_non_retryable_error(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    config = _make_config(max_retries=3, retry_base_delay=0.01)

    mock_client = MagicMock()
    error = _make_non_retryable_error(401)
    mock_client.messages.create.side_effect = error

    with pytest.raises(anthropic.AuthenticationError):
        _run_single_turn_with_retry(
            client=mock_client,
            config=config,
            system_prompt=[],
            messages=[],
            run=run,
            repo_root=tmp_path,
            state=state,
        )

    assert mock_client.messages.create.call_count == 1
    assert state.total_retries == 0


def test_retry_records_transcript_entries(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    config = _make_config(max_retries=2, retry_base_delay=0.01)

    mock_client = MagicMock()
    error = _make_retryable_error(500)
    success_response = _mock_response("end_turn", [_text_block("Done")])

    mock_client.messages.create.side_effect = [error, success_response]

    _run_single_turn_with_retry(
        client=mock_client,
        config=config,
        system_prompt=[],
        messages=[],
        run=run,
        repo_root=tmp_path,
        state=state,
    )

    transcript = read_transcript(run, tmp_path)
    retry_entries = [e for e in transcript if e.get("kind") == "retry_attempt"]
    assert len(retry_entries) == 1
    assert retry_entries[0]["attempt"] == 1


def test_retry_updates_state_message(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    config = _make_config(max_retries=2, retry_base_delay=0.01)

    mock_client = MagicMock()
    error = _make_retryable_error(500)
    success_response = _mock_response("end_turn", [_text_block("Done")])

    mock_client.messages.create.side_effect = [error, success_response]

    _run_single_turn_with_retry(
        client=mock_client,
        config=config,
        system_prompt=[],
        messages=[],
        run=run,
        repo_root=tmp_path,
        state=state,
    )

    assert "retry" in state.last_message.lower()
    assert "API error" in state.last_message


def test_retry_respects_max_retries_config(tmp_path: Path):
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    config = _make_config(max_retries=5, retry_base_delay=0.01)

    mock_client = MagicMock()
    error = _make_retryable_error(429)
    mock_client.messages.create.side_effect = error

    with pytest.raises(anthropic.RateLimitError):
        _run_single_turn_with_retry(
            client=mock_client,
            config=config,
            system_prompt=[],
            messages=[],
            run=run,
            repo_root=tmp_path,
            state=state,
        )

    assert mock_client.messages.create.call_count == 6
    assert state.total_retries == 5
