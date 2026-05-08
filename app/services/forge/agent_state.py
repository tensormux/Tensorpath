"""Agentic-run state — what the orchestrator persists for the UI to poll.

The orchestrator runs in a background thread. Every meaningful action
(iteration tick, tool call, gate result, abort, completion) writes a
fresh `agent_state.json` to the run directory. The UI polls a GET endpoint
that just reads this file off disk.

State on disk is the source of truth — the orchestrator and the API
server are different threads in the same process and use only the
filesystem to communicate. There's also an `agent_transcript.jsonl`
where each line is one orchestrator → API → tool turn, kept separate
because it grows unbounded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from app.services.forge.models import ForgeRun


_STATE_FILE = "agent_state.json"
_TRANSCRIPT_FILE = "agent_transcript.jsonl"


class AgenticRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    ABORTED = "aborted"
    ERRORED = "errored"


class AgenticRunState(BaseModel):
    """Full state of one agentic run. Mirrored to disk after each update."""

    run_id: str
    status: AgenticRunStatus = AgenticRunStatus.PENDING
    iteration: int = 0
    max_iterations: int = 5
    started_at: str | None = None
    last_updated_at: str | None = None
    finished_at: str | None = None

    # cost accounting
    cost_usd: float = 0.0
    cost_cap_usd: float = 3.0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    cache_read_tokens_total: int = 0
    cache_write_tokens_total: int = 0

    # in-flight signals
    abort_requested: bool = False
    last_message: str | None = None  # human-readable status the UI can show

    # gate outcomes from the most-recent verify/benchmark calls
    last_verify_passed: bool | None = None
    last_verify_reason: str | None = None
    last_benchmark_passed: bool | None = None
    last_speedup: float | None = None

    # final outcome
    error: str | None = None
    promoted_kernel_id: str | None = None

    # transcript
    transcript_lines: int = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def state_path(repo_root: Path, run: ForgeRun) -> Path:
    return repo_root / run.artifact_dir / _STATE_FILE


def transcript_path(repo_root: Path, run: ForgeRun) -> Path:
    return repo_root / run.artifact_dir / _TRANSCRIPT_FILE


def init_state(
    run: ForgeRun,
    repo_root: Path,
    *,
    max_iterations: int = 5,
    cost_cap_usd: float = 3.0,
) -> AgenticRunState:
    state = AgenticRunState(
        run_id=run.run_id,
        status=AgenticRunStatus.PENDING,
        max_iterations=max_iterations,
        cost_cap_usd=cost_cap_usd,
        started_at=_now(),
        last_updated_at=_now(),
    )
    save_state(state, run, repo_root)
    # truncate any prior transcript on a fresh start
    transcript_path(repo_root, run).write_text("")
    return state


def save_state(state: AgenticRunState, run: ForgeRun, repo_root: Path) -> None:
    state.last_updated_at = _now()
    state_path(repo_root, run).write_text(state.model_dump_json(indent=2))


def load_state(run: ForgeRun, repo_root: Path) -> AgenticRunState | None:
    path = state_path(repo_root, run)
    if not path.exists():
        return None
    try:
        return AgenticRunState.model_validate_json(path.read_text())
    except Exception:
        return None


def request_abort(run: ForgeRun, repo_root: Path) -> bool:
    """Set the abort flag. The orchestrator checks this between iterations."""
    state = load_state(run, repo_root)
    if state is None:
        return False
    if state.status not in (AgenticRunStatus.PENDING, AgenticRunStatus.RUNNING):
        return False
    state.abort_requested = True
    state.last_message = "Abort requested — will stop after current step."
    save_state(state, run, repo_root)
    return True


def append_transcript(
    run: ForgeRun,
    repo_root: Path,
    entry: dict,
) -> None:
    """Append one JSONL line to the transcript. Caller is responsible for the schema."""
    entry = {"at": _now(), **entry}
    with transcript_path(repo_root, run).open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_transcript(run: ForgeRun, repo_root: Path) -> list[dict]:
    path = transcript_path(repo_root, run)
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
