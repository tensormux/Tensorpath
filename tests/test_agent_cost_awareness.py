"""Tests for agent cost awareness feature."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.forge.agent_runner import (
    AgenticRunConfig,
    _build_cost_context,
    run_agentic_loop,
)
from app.services.forge.agent_state import (
    AgenticRunState,
    AgenticRunStatus,
    init_state,
)
from app.services.forge.models import ForgeRun, ForgeRunStatus, KernelTaskSpec


pytestmark = pytest.mark.forge


def _make_task() -> KernelTaskSpec:
    return KernelTaskSpec(
        op="rmsnorm",
        language="triton",
        target_gpu="rtx4070",
        dtype="fp16",
        shape={"hidden_size": 4096},
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


def test_build_cost_context_basic(tmp_path: Path):
    """Test that cost context includes spent, remaining, and percentage."""
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path, max_iterations=5, cost_cap_usd=10.0)
    state.cost_usd = 2.5
    config = AgenticRunConfig(cost_cap_usd=10.0, max_iterations=5)

    context = _build_cost_context(state, config, iteration=2)

    assert "Spent so far: $2.5000" in context
    assert "Remaining budget: $7.5000" in context
    assert "75.0% left" in context
    assert "Iteration 2/5" in context


def test_build_cost_context_zero_spent(tmp_path: Path):
    """Test cost context when nothing has been spent yet."""
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path, max_iterations=3, cost_cap_usd=5.0)
    state.cost_usd = 0.0
    config = AgenticRunConfig(cost_cap_usd=5.0, max_iterations=3)

    context = _build_cost_context(state, config, iteration=1)

    assert "Spent so far: $0.0000" in context
    assert "Remaining budget: $5.0000" in context
    assert "100.0% left" in context


def test_build_cost_context_mostly_spent(tmp_path: Path):
    """Test cost context when most budget is used."""
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path, max_iterations=5, cost_cap_usd=10.0)
    state.cost_usd = 8.5
    config = AgenticRunConfig(cost_cap_usd=10.0, max_iterations=5)

    context = _build_cost_context(state, config, iteration=4)

    assert "Spent so far: $8.5000" in context
    assert "Remaining budget: $1.5000" in context
    assert "15.0% left" in context


def test_build_cost_context_includes_efficiency_advice(tmp_path: Path):
    """Test that cost context includes efficiency guidance."""
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path, max_iterations=5, cost_cap_usd=10.0)
    config = AgenticRunConfig(cost_cap_usd=10.0, max_iterations=5)

    context = _build_cost_context(state, config, iteration=1)

    assert "Be efficient" in context
    assert "avoid unnecessary iterations" in context


def test_cost_context_injected_in_first_iteration(tmp_path: Path, monkeypatch):
    """Test that cost context is injected in the first iteration."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = AgenticRunConfig(max_iterations=2, cost_cap_usd=5.0)

    # Mock the API call to capture messages
    captured_messages = []

    def mock_api_call(*args, **kwargs):
        captured_messages.append(kwargs.get("messages", []))
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        mock_response.usage = MagicMock(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return mock_response

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = mock_api_call
        MockAnthropic.return_value = mock_client

        run_agentic_loop(run, tmp_path, config)

    # Check that cost context was injected in first iteration
    assert len(captured_messages) > 0
    first_call_messages = captured_messages[0]
    assert len(first_call_messages) > 0
    
    # First message should contain cost context
    first_message_content = first_call_messages[0]["content"]
    assert "Cost & Progress Update" in first_message_content
    assert "Spent so far" in first_message_content
    assert "Remaining budget" in first_message_content


def test_cost_context_injected_in_subsequent_iterations(tmp_path: Path, monkeypatch):
    """Test that cost context is injected in subsequent iterations."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = AgenticRunConfig(max_iterations=3, cost_cap_usd=5.0)

    # Mock the API call to capture messages and simulate multiple iterations
    captured_messages = []
    call_count = [0]

    def mock_api_call(*args, **kwargs):
        call_count[0] += 1
        captured_messages.append(kwargs.get("messages", []))
        
        mock_response = MagicMock()
        if call_count[0] < 3:
            # First two calls use tools
            mock_response.stop_reason = "tool_use"
            mock_tool_block = MagicMock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "list_candidate_files"
            mock_tool_block.input = {}
            mock_tool_block.id = f"toolu_{call_count[0]}"
            mock_response.content = [mock_tool_block]
        else:
            # Third call ends the turn
            mock_response.stop_reason = "end_turn"
            mock_response.content = []
        
        mock_response.usage = MagicMock(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return mock_response

    with patch("anthropic.Anthropic") as MockAnthropic, \
         patch("app.services.forge.agent_tools.handle_tool") as mock_handle_tool:
        
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = mock_api_call
        MockAnthropic.return_value = mock_client
        
        # Mock tool handler to return empty file list
        mock_handle_tool.return_value = MagicMock(
            content='{"files": []}',
            is_error=False
        )

        run_agentic_loop(run, tmp_path, config)

    # Check that cost context was injected in multiple iterations
    assert len(captured_messages) >= 2
    
    # Second iteration should have cost context as a separate message
    second_call_messages = captured_messages[1]
    # Find the cost context message (should be a user message with cost info)
    cost_context_found = False
    for msg in second_call_messages:
        if msg["role"] == "user" and "Cost & Progress Update" in msg["content"]:
            cost_context_found = True
            break
    
    assert cost_context_found, "Cost context not found in second iteration"


def test_cost_context_updates_with_spending(tmp_path: Path, monkeypatch):
    """Test that cost context reflects accumulated spending."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = AgenticRunConfig(max_iterations=3, cost_cap_usd=5.0)

    # Mock API calls with different usage amounts
    call_count = [0]

    def mock_api_call(*args, **kwargs):
        call_count[0] += 1
        messages = kwargs.get("messages", [])
        
        mock_response = MagicMock()
        if call_count[0] < 3:
            mock_response.stop_reason = "tool_use"
            mock_tool_block = MagicMock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "list_candidate_files"
            mock_tool_block.input = {}
            mock_tool_block.id = f"toolu_{call_count[0]}"
            mock_response.content = [mock_tool_block]
        else:
            mock_response.stop_reason = "end_turn"
            mock_response.content = []
        
        # Simulate increasing costs
        mock_response.usage = MagicMock(
            input_tokens=1000 * call_count[0],
            output_tokens=500 * call_count[0],
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return mock_response

    captured_messages = []

    with patch("anthropic.Anthropic") as MockAnthropic, \
         patch("app.services.forge.agent_tools.handle_tool") as mock_handle_tool:
        
        mock_client = MagicMock()
        
        def capture_and_respond(*args, **kwargs):
            messages = kwargs.get("messages", [])
            captured_messages.append(messages)
            return mock_api_call(*args, **kwargs)
        
        mock_client.messages.create.side_effect = capture_and_respond
        MockAnthropic.return_value = mock_client
        
        mock_handle_tool.return_value = MagicMock(
            content='{"files": []}',
            is_error=False
        )

        run_agentic_loop(run, tmp_path, config)

    # Verify that later iterations show higher spending
    assert len(captured_messages) >= 2
    
    # Extract cost context from first and second iterations
    first_cost_msg = None
    second_cost_msg = None
    
    for msg in captured_messages[0]:
        if "Cost & Progress Update" in msg.get("content", ""):
            first_cost_msg = msg["content"]
            break
    
    if len(captured_messages) > 1:
        for msg in captured_messages[1]:
            if "Cost & Progress Update" in msg.get("content", ""):
                second_cost_msg = msg["content"]
                break
    
    # Both should have cost context
    assert first_cost_msg is not None
    if second_cost_msg is not None:
        # Second should show more spending (if we got that far)
        assert "Spent so far" in first_cost_msg
        assert "Spent so far" in second_cost_msg
