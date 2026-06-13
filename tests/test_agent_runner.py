"""Tests for the agentic loop orchestrator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.forge.agent_runner import (
    AgenticRunConfig,
    _build_system_prompt,
    _cost_from_usage,
    run_agentic_loop,
)
from app.services.forge.agent_state import AgenticRunStatus, request_abort
from app.services.forge.models import (
    BenchmarkResult,
    ForgeRun,
    ForgeRunStatus,
    KernelLanguage,
    KernelOp,
    KernelTaskSpec,
    PromotedKernel,
    VerificationResult,
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


def _usage(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )


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
        usage = _usage()
    return SimpleNamespace(stop_reason=stop_reason, content=content, usage=usage)


# ---- Cost tracking tests -------------------------------------------------


def test_cost_from_usage_all_zeros():
    cost, tokens = _cost_from_usage(_usage())
    assert cost == 0.0
    assert tokens["input_tokens"] == 0


def test_cost_from_usage_input_only():
    cost, tokens = _cost_from_usage(_usage(input_tokens=1_000_000))
    assert cost == 5.00
    assert tokens["input_tokens"] == 1_000_000


def test_cost_from_usage_output_only():
    cost, tokens = _cost_from_usage(_usage(output_tokens=1_000_000))
    assert cost == 25.00


def test_cost_from_usage_cache_write():
    cost, tokens = _cost_from_usage(_usage(cache_creation_input_tokens=1_000_000))
    assert cost == 6.25


def test_cost_from_usage_cache_read():
    cost, tokens = _cost_from_usage(_usage(cache_read_input_tokens=1_000_000))
    assert cost == 0.50


def test_cost_from_usage_combined():
    usage = _usage(
        input_tokens=100_000,
        output_tokens=50_000,
        cache_creation_input_tokens=200_000,
        cache_read_input_tokens=300_000,
    )
    cost, tokens = _cost_from_usage(usage)
    expected = (
        100_000 * 5.00 / 1_000_000
        + 50_000 * 25.00 / 1_000_000
        + 200_000 * 6.25 / 1_000_000
        + 300_000 * 0.50 / 1_000_000
    )
    assert abs(cost - expected) < 0.001


# ---- System prompt tests -------------------------------------------------


def test_build_system_prompt_has_two_blocks(tmp_path: Path):
    run = _make_run(tmp_path)
    blocks = _build_system_prompt(run, tmp_path)
    assert len(blocks) == 2
    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "text"


def test_build_system_prompt_cache_control(tmp_path: Path):
    run = _make_run(tmp_path)
    blocks = _build_system_prompt(run, tmp_path)
    assert "cache_control" in blocks[1]
    assert blocks[1]["cache_control"]["type"] == "ephemeral"


def test_build_system_prompt_includes_task_details(tmp_path: Path):
    run = _make_run(tmp_path)
    blocks = _build_system_prompt(run, tmp_path)
    rules_text = blocks[0]["text"]
    assert "rmsnorm" in rules_text
    assert "RTX 4070" in rules_text
    assert "fp16" in rules_text


# ---- Loop termination tests ----------------------------------------------


def test_loop_errored_on_missing_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    run = _make_run(tmp_path)
    config = _make_config()

    state = run_agentic_loop(run, tmp_path, config)

    assert state.status == AgenticRunStatus.ERRORED
    assert "ANTHROPIC_API_KEY" in state.error


def test_loop_errored_on_api_refusal(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config()

    mock_response = _mock_response("refusal", [])

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    assert state.status == AgenticRunStatus.ERRORED
    assert "refusal" in state.error.lower()


def test_loop_breaks_on_end_turn_without_tools(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config()

    mock_response = _mock_response("end_turn", [_text_block("Done")])

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    assert state.status == AgenticRunStatus.REJECTED
    assert state.iteration == 1


def test_loop_rejected_on_give_up(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config()

    mock_response = _mock_response(
        "tool_use",
        [_tool_use_block("give_up", {"reason": "Cannot optimize"})],
    )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    assert state.status == AgenticRunStatus.REJECTED
    assert "Cannot optimize" in state.error


def test_loop_rejected_on_max_iterations(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config(max_iterations=2)

    mock_response = _mock_response("end_turn", [_text_block("Thinking...")])

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    assert state.status == AgenticRunStatus.REJECTED
    assert "Exhausted" in state.error


def test_loop_rejected_on_cost_cap(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config(cost_cap_usd=1.0)

    high_cost_usage = _usage(output_tokens=1_000_000)
    mock_response = _mock_response(
        "tool_use",
        [_tool_use_block("list_candidate_files", {})],
        usage=high_cost_usage,
    )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    assert state.status == AgenticRunStatus.REJECTED
    assert "Cost cap" in state.error


def test_loop_errored_on_api_error(tmp_path: Path, monkeypatch):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config()

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIStatusError(
            message="Rate limit exceeded", response=MagicMock(), body={"error": {"message": "Rate limit"}}
        )
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    assert state.status == AgenticRunStatus.ERRORED
    assert "Rate limit" in state.error


# ---- State transition tests ----------------------------------------------


def test_iteration_counter_increments(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config(max_iterations=2)

    responses = [
        _mock_response("tool_use", [_tool_use_block("list_candidate_files", {})]),
        _mock_response("end_turn", [_text_block("Done")]),
    ]

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = responses
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    assert state.iteration == 2


def test_cost_accumulates_across_turns(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config(max_iterations=2)

    usage1 = _usage(output_tokens=100_000)
    usage2 = _usage(output_tokens=200_000)

    responses = [
        _mock_response("tool_use", [_tool_use_block("list_candidate_files", {})], usage=usage1),
        _mock_response("end_turn", [_text_block("Done")], usage=usage2),
    ]

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = responses
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    expected_cost = (100_000 * 25.00 / 1_000_000) + (200_000 * 25.00 / 1_000_000)
    assert abs(state.cost_usd - expected_cost) < 0.01


def test_token_counts_accumulate(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config(max_iterations=2)

    usage1 = _usage(input_tokens=1000, output_tokens=500)
    usage2 = _usage(input_tokens=2000, output_tokens=1000)

    responses = [
        _mock_response("tool_use", [_tool_use_block("list_candidate_files", {})], usage=usage1),
        _mock_response("end_turn", [_text_block("Done")], usage=usage2),
    ]

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = responses
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    assert state.input_tokens_total == 3000
    assert state.output_tokens_total == 1500


def test_verify_result_tracked_in_state(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config()

    verify_result = VerificationResult(passed=False, failure_reason="tests failed")
    mock_response = _mock_response(
        "tool_use",
        [_tool_use_block("run_verify", {"skip_cuda_check": True})],
    )

    with patch("anthropic.Anthropic") as MockAnthropic, \
         patch("app.services.forge.agent_tools.verify_candidate") as mock_verify:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client
        mock_verify.return_value = (run, verify_result)

        state = run_agentic_loop(run, tmp_path, config)

    assert state.last_verify_passed is False
    assert "tests failed" in state.last_verify_reason


def test_benchmark_result_tracked_in_state(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config()

    bench_result = BenchmarkResult(
        passed=False,
        baseline_latency_us=100.0,
        candidate_latency_us=95.0,
        speedup=1.05,
        warmup_iters=20,
        benchmark_iters=100,
    )
    mock_response = _mock_response(
        "tool_use",
        [_tool_use_block("run_benchmark", {})],
    )

    with patch("anthropic.Anthropic") as MockAnthropic, \
         patch("app.services.forge.agent_tools.benchmark_candidate") as mock_bench:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client
        mock_bench.return_value = (run, bench_result)

        state = run_agentic_loop(run, tmp_path, config)

    assert state.last_benchmark_passed is False
    assert state.last_speedup == 1.05


# ---- Edge case tests -----------------------------------------------------


def test_loop_continues_when_verify_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config(max_iterations=2)

    verify_result = VerificationResult(passed=False, failure_reason="tests failed")

    responses = [
        _mock_response("tool_use", [_tool_use_block("run_verify", {"skip_cuda_check": True})]),
        _mock_response("end_turn", [_text_block("Giving up")]),
    ]

    with patch("anthropic.Anthropic") as MockAnthropic, \
         patch("app.services.forge.agent_tools.verify_candidate") as mock_verify:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = responses
        MockAnthropic.return_value = mock_client
        mock_verify.return_value = (run, verify_result)

        state = run_agentic_loop(run, tmp_path, config)

    assert state.iteration == 2
    assert state.last_verify_passed is False


def test_loop_continues_when_benchmark_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config(max_iterations=2)

    bench_result = BenchmarkResult(
        passed=False,
        baseline_latency_us=100.0,
        candidate_latency_us=95.0,
        speedup=1.05,
        warmup_iters=20,
        benchmark_iters=100,
    )

    responses = [
        _mock_response("tool_use", [_tool_use_block("run_benchmark", {})]),
        _mock_response("end_turn", [_text_block("Giving up")]),
    ]

    with patch("anthropic.Anthropic") as MockAnthropic, \
         patch("app.services.forge.agent_tools.benchmark_candidate") as mock_bench:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = responses
        MockAnthropic.return_value = mock_client
        mock_bench.return_value = (run, bench_result)

        state = run_agentic_loop(run, tmp_path, config)

    assert state.iteration == 2
    assert state.last_benchmark_passed is False


def test_promotion_failure_does_not_crash(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config(max_iterations=2)

    verify_result = VerificationResult(passed=True)
    bench_result = BenchmarkResult(
        passed=True,
        baseline_latency_us=100.0,
        candidate_latency_us=50.0,
        speedup=2.0,
        warmup_iters=20,
        benchmark_iters=100,
    )

    responses = [
        _mock_response(
            "tool_use",
            [
                _tool_use_block("run_verify", {"skip_cuda_check": True}, "toolu_1"),
                _tool_use_block("run_benchmark", {}, "toolu_2"),
            ],
        ),
        _mock_response("end_turn", [_text_block("Done")]),
    ]

    with patch("anthropic.Anthropic") as MockAnthropic, \
         patch("app.services.forge.agent_tools.verify_candidate") as mock_verify, \
         patch("app.services.forge.agent_tools.benchmark_candidate") as mock_bench, \
         patch("app.services.forge.agent_runner.promote_candidate") as mock_promote:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = responses
        MockAnthropic.return_value = mock_client
        mock_verify.return_value = (run, verify_result)
        mock_bench.return_value = (run, bench_result)
        mock_promote.side_effect = ValueError("verification_report.json missing")

        state = run_agentic_loop(run, tmp_path, config)

    assert state.iteration == 2
    assert state.last_verify_passed is True
    assert state.last_benchmark_passed is True
    assert state.promoted_kernel_id is None


def test_transcript_written_each_turn(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    run = _make_run(tmp_path)
    config = _make_config(max_iterations=2)

    responses = [
        _mock_response("tool_use", [_tool_use_block("list_candidate_files", {})]),
        _mock_response("end_turn", [_text_block("Done")]),
    ]

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = responses
        MockAnthropic.return_value = mock_client

        state = run_agentic_loop(run, tmp_path, config)

    assert state.transcript_lines >= 2
