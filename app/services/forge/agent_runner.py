"""Agentic Forge orchestrator.

Drives a tool-use loop against `claude-opus-4-7` with adaptive thinking,
prompt caching, and per-run cost / iteration / abort budgets. The agent
writes candidate files, runs verify, runs benchmark, and on success the
orchestrator promotes the result into the kernel registry — same gates,
same registry, same recommendation surface as the manual flow.

Background execution model:
    The orchestrator runs in a daemon thread. Every meaningful event
    persists `agent_state.json` to disk (status, iteration, cost, last
    gate result) and appends one JSONL line to `agent_transcript.jsonl`.
    The API server reads those files in a separate handler — no shared
    in-memory state between threads.

Cost accounting:
    Opus 4.7 pricing (cached 2026-04-29, see claude-api skill):
        input          $5.00 / MTok
        output        $25.00 / MTok
        cache write    1.25× input  (~$6.25 / MTok)
        cache read     0.10× input  (~$0.50 / MTok)
    Each turn's `response.usage` gives us the four token counts; we
    convert to dollars and accumulate.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

from app.services.forge.agent_state import (
    AgenticRunState,
    AgenticRunStatus,
    append_transcript,
    init_state,
    load_state,
    save_state,
)
from app.services.forge.agent_tools import TOOL_DEFINITIONS, handle_tool
from app.services.forge.models import ForgeRun, ForgeRunStatus
from app.services.forge.promoter import promote_candidate
from app.services.forge.runs import update_run_status
from app.services.forge.skill_provider import KernelSkillsProvider


# ---- pricing (USD per million tokens) ------------------------------------

PRICING_PER_MTOK = {
    # claude-opus-4-7
    "input":      5.00,
    "output":    25.00,
    "cache_write": 6.25,   # 1.25× input, 5-min TTL
    "cache_read":  0.50,   # 0.10× input
}

# We cap response output to keep per-turn cost bounded and to keep us under
# the SDK non-streaming HTTP timeout. Kernel files are typically <8KB; 16K
# tokens is plenty of headroom.
DEFAULT_MAX_TOKENS = 16000

# Agent model. Per the claude-api skill: default to claude-opus-4-7.
DEFAULT_MODEL = "claude-opus-4-7"

# Effort: per the claude-api skill, "xhigh" is the right setting for coding
# and agentic use cases on Opus 4.7 (best results; default in Claude Code).
DEFAULT_EFFORT = "xhigh"


@dataclass
class AgenticRunConfig:
    max_iterations: int = 5
    cost_cap_usd: float = 3.0
    skip_cuda_check: bool = False
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT


# ---- system prompt -------------------------------------------------------


def _build_system_prompt(run: ForgeRun, repo_root: Path) -> list[dict]:
    """Two-block system prompt with prompt caching on the (large, stable) bundle.

    Block 1 (small, frozen): the rules of the loop and the available tools.
    Block 2 (large, frozen for this run): the kernel-skills bundle.

    The cache_control marker on block 2 lets us amortize the bundle across
    every iteration of the same run — first call is a cache write, every
    subsequent call is a cache read at ~10% of the input price.

    Important: anything in this prompt that varies per call (timestamps,
    iteration counters, etc.) MUST go in messages, not here. A single
    byte change in the prefix invalidates the cache for everything after
    it.
    """
    artifact_dir = repo_root / run.artifact_dir
    bundle_path = artifact_dir / "skill_bundle.md"
    bundle = bundle_path.read_text() if bundle_path.exists() else ""

    rules = (
        "You are NeevPath Forge — an autonomous kernel optimization agent.\n"
        "\n"
        "Your job: write a verified, faster kernel for one specific operation on "
        f"one specific GPU/dtype/shape. The task is `{run.task.op.value}` in "
        f"{run.task.language.value} for {run.task.target_gpu} at dtype "
        f"{run.task.dtype}, shape={run.task.shape}.\n"
        "\n"
        "## How the loop works\n"
        "\n"
        "You have tools to write files into a candidate directory, then run two "
        "gates: `run_verify` (correctness via pytest against a PyTorch reference) "
        "and `run_benchmark` (1.10× minimum speedup vs the reference). Both gates "
        "must pass for the orchestrator to promote your kernel into the registry. "
        "There is no 'submit' step — the orchestrator promotes automatically once "
        "both gates pass in a single iteration.\n"
        "\n"
        "## Required deliverables (all under candidate/)\n"
        "\n"
        "- `kernel.py` — the optimized kernel implementation\n"
        "- `reference.py` — a PyTorch reference of the SAME math (do NOT use "
        "`torch.nn.LayerNorm` as an RMSNorm reference; RMSNorm does not subtract the mean)\n"
        "- `test_correctness.py` — pytest-style correctness checks at multiple shapes "
        "including non-power-of-two values; use `torch.allclose` with explicit tolerances\n"
        "- `bench.py` — runs both implementations with at least 20 warmup + 100 measured "
        "iterations, calls `torch.cuda.synchronize()` before and after each timing block, "
        "and prints ONE JSON line on stdout with: baseline_latency_us, candidate_latency_us, "
        "speedup, warmup_iters, benchmark_iters, gpu_name\n"
        "- `metadata.json` — at minimum: name, op, language, kernel_file\n"
        "\n"
        "## Forbidden behavior\n"
        "\n"
        "- Do NOT use `subprocess`, `eval`, `exec`, `os.system`, `socket`, `requests`, "
        "or `__import__`. The verifier scans for these and will reject the candidate.\n"
        "- Do NOT open files outside the candidate directory.\n"
        "- Do NOT report a speedup you didn't measure inside `bench.py`.\n"
        "- Do NOT skip boundary masks. Tests include non-power-of-two hidden sizes.\n"
        "\n"
        "## Strategy\n"
        "\n"
        "1. Write all five files, then call `run_verify`. If it fails, read the report's "
        "`failure_reason`, fix the most likely cause, write the affected file again, "
        "and re-verify. Most early failures are correctness bugs in the kernel; some "
        "are missing files or unsafe patterns.\n"
        "2. Once verify passes, call `run_benchmark`. If `speedup < 1.10`, your "
        "candidate is correct but not faster — try a real optimization (better tile "
        "size, fp32 accumulation removed where unnecessary, fewer kernel launches). "
        "Don't game the bench — gaming gets caught downstream.\n"
        "3. If you've tried distinct approaches and you're stuck, call `give_up` "
        "with a brief reason. Don't burn iterations on the same approach twice.\n"
        "\n"
        "Be precise. Be honest. Don't claim a kernel is faster than it is."
    )

    return [
        {"type": "text", "text": rules},
        {
            "type": "text",
            "text": f"## Skill bundle (curated kernel-skills guidance)\n\n{bundle}",
            "cache_control": {"type": "ephemeral"},  # 5-min TTL, hits across iterations
        },
    ]


# ---- main loop -----------------------------------------------------------


def _cost_from_usage(usage: Any) -> tuple[float, dict]:
    """Convert one `response.usage` into dollars + a token-counts dict."""
    in_tok    = getattr(usage, "input_tokens", 0) or 0
    out_tok   = getattr(usage, "output_tokens", 0) or 0
    write_tok = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read_tok  = getattr(usage, "cache_read_input_tokens", 0) or 0

    cost = (
        in_tok    * PRICING_PER_MTOK["input"]       / 1_000_000
        + out_tok   * PRICING_PER_MTOK["output"]      / 1_000_000
        + write_tok * PRICING_PER_MTOK["cache_write"] / 1_000_000
        + read_tok  * PRICING_PER_MTOK["cache_read"]  / 1_000_000
    )
    return cost, {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cache_write_tokens": write_tok,
        "cache_read_tokens": read_tok,
    }


def _to_dict_block(b: Any) -> dict:
    """Turn a Claude content block into a plain dict, robust to SDK shape changes."""
    if hasattr(b, "model_dump"):
        return b.model_dump()
    return dict(b)  # already-dict fallback


class _AbortRequested(Exception):
    """Raised when the run's abort flag flips on between turns."""


def _check_abort(run: ForgeRun, repo_root: Path) -> None:
    state = load_state(run, repo_root)
    if state and state.abort_requested:
        raise _AbortRequested()


def _run_single_turn(
    *,
    client: anthropic.Anthropic,
    config: AgenticRunConfig,
    system_prompt: list[dict],
    messages: list[dict],
) -> Any:
    """One Claude call. Adaptive thinking + xhigh effort + tools, no streaming."""
    return client.messages.create(
        model=config.model,
        max_tokens=DEFAULT_MAX_TOKENS,
        system=system_prompt,
        thinking={"type": "adaptive"},
        output_config={"effort": config.effort},
        tools=TOOL_DEFINITIONS,
        messages=messages,
    )


def _execute_turn(
    *,
    response: Any,
    run: ForgeRun,
    repo_root: Path,
    provider: KernelSkillsProvider,
    state: AgenticRunState,
) -> tuple[list[dict], dict | None]:
    """Run all tool calls in one assistant turn. Returns (tool_results, give_up_payload)."""
    tool_results: list[dict] = []
    give_up_payload: dict | None = None

    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        tool_name = block.name
        tool_input = block.input or {}
        result = handle_tool(
            tool_name,
            tool_input,
            run=run,
            repo_root=repo_root,
            provider=provider,
        )
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result.content,
            "is_error": result.is_error,
        })

        # Track outcomes the UI cares about
        if tool_name == "run_verify":
            try:
                import json as _json
                rj = _json.loads(result.content)
                state.last_verify_passed = bool(rj.get("passed"))
                state.last_verify_reason = rj.get("failure_reason")
            except Exception:
                pass
        elif tool_name == "run_benchmark":
            try:
                import json as _json
                rj = _json.loads(result.content)
                state.last_benchmark_passed = bool(rj.get("passed"))
                state.last_speedup = rj.get("speedup")
            except Exception:
                pass
        elif tool_name == "give_up":
            give_up_payload = tool_input

        append_transcript(run, repo_root, {
            "kind": "tool_call",
            "name": tool_name,
            "input": tool_input,
            "is_error": result.is_error,
            "content_preview": result.content[:500],
        })
        state.transcript_lines += 1

    return tool_results, give_up_payload


def _try_promote(run: ForgeRun, repo_root: Path, state: AgenticRunState) -> bool:
    """Attempt promotion. Returns True iff successful."""
    try:
        _, promoted = promote_candidate(run, repo_root)
    except ValueError as e:
        state.last_message = f"Promotion refused: {e}"
        return False
    state.promoted_kernel_id = promoted.kernel_id
    state.last_message = f"Promoted: {promoted.kernel_id}"
    append_transcript(run, repo_root, {
        "kind": "promotion",
        "kernel_id": promoted.kernel_id,
        "speedup": promoted.benchmark.speedup,
    })
    state.transcript_lines += 1
    return True


def run_agentic_loop(
    run: ForgeRun,
    repo_root: Path,
    config: AgenticRunConfig | None = None,
) -> AgenticRunState:
    """Run the full orchestrator loop synchronously. Use start_agentic_run for background."""
    config = config or AgenticRunConfig()

    state = init_state(
        run, repo_root,
        max_iterations=config.max_iterations,
        cost_cap_usd=config.cost_cap_usd,
    )
    state.status = AgenticRunStatus.RUNNING
    state.last_message = "Initializing — loading skill bundle, building system prompt."
    save_state(state, run, repo_root)

    if not os.getenv("ANTHROPIC_API_KEY"):
        state.status = AgenticRunStatus.ERRORED
        state.error = "ANTHROPIC_API_KEY not set in environment / .env file."
        save_state(state, run, repo_root)
        return state

    client = anthropic.Anthropic()
    provider = KernelSkillsProvider(repo_root=repo_root)
    system_prompt = _build_system_prompt(run, repo_root)

    initial_user_message = (
        "Begin the task. Write all five candidate files, then call run_verify. "
        "If it passes, call run_benchmark. The skill bundle in your system prompt "
        "has the relevant guidance — read it carefully before writing the kernel."
    )
    messages: list[dict] = [{"role": "user", "content": initial_user_message}]

    try:
        for iteration in range(1, config.max_iterations + 1):
            _check_abort(run, repo_root)

            state.iteration = iteration
            state.last_message = f"Iteration {iteration}/{config.max_iterations} — calling agent."
            save_state(state, run, repo_root)

            response = _run_single_turn(
                client=client,
                config=config,
                system_prompt=system_prompt,
                messages=messages,
            )

            # Cost accounting
            turn_cost, tokens = _cost_from_usage(response.usage)
            state.cost_usd += turn_cost
            state.input_tokens_total       += tokens["input_tokens"]
            state.output_tokens_total      += tokens["output_tokens"]
            state.cache_write_tokens_total += tokens["cache_write_tokens"]
            state.cache_read_tokens_total  += tokens["cache_read_tokens"]

            append_transcript(run, repo_root, {
                "kind": "model_turn",
                "iteration": iteration,
                "stop_reason": response.stop_reason,
                "tokens": tokens,
                "turn_cost_usd": round(turn_cost, 4),
                "cumulative_cost_usd": round(state.cost_usd, 4),
            })
            state.transcript_lines += 1

            # Refusal -> stop immediately, surface the cause
            if response.stop_reason == "refusal":
                state.status = AgenticRunStatus.ERRORED
                state.error = "Agent refused to continue (stop_reason=refusal)."
                save_state(state, run, repo_root)
                return state

            # Cost cap — finish current turn's tool calls before stopping
            over_budget = state.cost_usd >= state.cost_cap_usd

            # Append the assistant's full content to the conversation
            messages.append({
                "role": "assistant",
                "content": [_to_dict_block(b) for b in response.content],
            })

            # If the model ended its turn without using tools, it's done.
            if response.stop_reason == "end_turn":
                state.last_message = "Agent ended its turn without further tool calls."
                save_state(state, run, repo_root)
                break

            # Otherwise execute tool calls
            tool_results, give_up_payload = _execute_turn(
                response=response,
                run=run, repo_root=repo_root,
                provider=provider, state=state,
            )

            if not tool_results:
                # No tool_use blocks but stop_reason wasn't end_turn — unusual, stop.
                state.last_message = "Agent stopped without using any tools."
                save_state(state, run, repo_root)
                break

            messages.append({"role": "user", "content": tool_results})
            save_state(state, run, repo_root)

            if give_up_payload is not None:
                state.status = AgenticRunStatus.REJECTED
                state.error = f"Agent gave up: {give_up_payload.get('reason', '(unknown)')}"
                save_state(state, run, repo_root)
                return state

            # Try promotion if both gates passed in this iteration
            if state.last_verify_passed and state.last_benchmark_passed:
                if _try_promote(run, repo_root, state):
                    state.status = AgenticRunStatus.SUCCEEDED
                    save_state(state, run, repo_root)
                    update_run_status(run, ForgeRunStatus.PROMOTED, repo_root)
                    return state

            if over_budget:
                state.status = AgenticRunStatus.REJECTED
                state.error = (
                    f"Cost cap reached (${state.cost_usd:.2f} >= ${state.cost_cap_usd:.2f}) "
                    f"after iteration {iteration}."
                )
                save_state(state, run, repo_root)
                return state

        # Loop fell off the end (max_iterations exhausted, no promotion)
        if state.status == AgenticRunStatus.RUNNING:
            state.status = AgenticRunStatus.REJECTED
            state.error = (
                f"Exhausted {config.max_iterations} iterations without producing a "
                f"verified, faster kernel."
            )
        save_state(state, run, repo_root)
        return state

    except _AbortRequested:
        state.status = AgenticRunStatus.ABORTED
        state.last_message = "Aborted by user."
        save_state(state, run, repo_root)
        return state
    except anthropic.APIStatusError as e:
        state.status = AgenticRunStatus.ERRORED
        state.error = f"Anthropic API error ({e.status_code}): {e.message}"
        save_state(state, run, repo_root)
        return state
    except Exception as e:
        state.status = AgenticRunStatus.ERRORED
        state.error = f"{type(e).__name__}: {e}"
        save_state(state, run, repo_root)
        return state


def start_agentic_run(
    run: ForgeRun,
    repo_root: Path,
    config: AgenticRunConfig | None = None,
) -> threading.Thread:
    """Kick off the orchestrator in a daemon thread. Returns the thread handle."""
    thread = threading.Thread(
        target=run_agentic_loop,
        args=(run, repo_root, config),
        daemon=True,
        name=f"forge-agentic-{run.run_id}",
    )
    thread.start()
    return thread
