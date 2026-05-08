"""Tool surface for the agentic Forge orchestrator.

These are the only verbs the agent has. The orchestrator passes their JSON
schemas to Claude in the `tools` parameter; when Claude requests a tool
call, the orchestrator dispatches via the local handler functions and
feeds the result back as a `tool_result` block.

Hard rule: every file the agent writes must land under
    forge_runs/<run_id>/candidate/

Path checks here defend the rule. There's also a sanity check in the
verifier (`_check_files`) and a safety scan that rejects unsafe code,
but treating these tools as the boundary is the right model — the
verifier is a check on the result, the tool surface is the gate on
what the agent can do at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.forge.benchmarker import benchmark_candidate
from app.services.forge.models import ForgeRun
from app.services.forge.skill_provider import KernelSkillsProvider
from app.services.forge.verifier import verify_candidate


# ---- tool schemas (sent to Claude) --------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "read_skill",
        "description": (
            "Fetch the full markdown content of one kernel-skills skill by its ID. "
            "Use when you need deeper detail on a specific topic than the bundle "
            "provided (e.g. 'inference.write-triton-rmsnorm-kernel'). "
            "The bundle in your system prompt already contains the most relevant skills "
            "for this task — call this only for additional context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": "Skill ID like 'inference.write-triton-rmsnorm-kernel'",
                },
            },
            "required": ["skill_id"],
        },
    },
    {
        "name": "list_candidate_files",
        "description": (
            "List filenames currently present in the candidate directory. "
            "Use to check what you've already written before a verify or benchmark call."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "write_candidate_file",
        "description": (
            "Write one file into the candidate directory. The path is restricted to "
            "the run's candidate/ subdirectory; you cannot write anywhere else. "
            "Required filenames before verify will pass: kernel.py, reference.py, "
            "test_correctness.py, bench.py, metadata.json."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename (no path components, no '..' segments)",
                },
                "content": {
                    "type": "string",
                    "description": "Full file contents",
                },
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "read_candidate_file",
        "description": (
            "Read back one file from the candidate directory. Useful for reviewing "
            "your own previous draft before editing it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
            },
            "required": ["filename"],
        },
    },
    {
        "name": "run_verify",
        "description": (
            "Run the correctness gate: required-files check, metadata sanity, "
            "unsafe-pattern scan, and pytest on test_correctness.py. "
            "Returns a JSON report with `passed`, `failure_reason`, and other fields. "
            "Use after writing all five required files to check if your candidate "
            "is correct. CUDA is required by default; pass skip_cuda_check=true only "
            "if explicitly told CUDA is unavailable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "skip_cuda_check": {
                    "type": "boolean",
                    "description": "Set true ONLY when CUDA is known to be unavailable (test infra).",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "run_benchmark",
        "description": (
            "Run the speed gate: executes candidate/bench.py, parses the last JSON "
            "line of stdout, and applies the 1.10x speedup threshold. Verification "
            "must already have passed. Returns `passed`, `speedup`, baseline + "
            "candidate latencies, and notes."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "give_up",
        "description": (
            "Declare the task can't be completed within the remaining iteration "
            "budget. Use sparingly — only if you've tried multiple distinct approaches "
            "and verify or benchmark keep failing for reasons you can't fix. The "
            "orchestrator will mark the run REJECTED and stop."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of what you tried and what's blocking you.",
                },
            },
            "required": ["reason"],
        },
    },
]


# ---- handlers (executed by the orchestrator) ----------------------------


def _candidate_dir(run: ForgeRun, repo_root: Path) -> Path:
    return repo_root / run.artifact_dir / "candidate"


def _safe_filename(name: str) -> str | None:
    """Reject anything that's not a flat filename in the candidate dir."""
    if not name or name in (".", ".."):
        return None
    if "/" in name or "\\" in name:
        return None
    if name.startswith("."):
        return None
    return name


class ToolResult:
    """Wrapper for tool execution outcomes — keeps `is_error` adjacent to content."""

    __slots__ = ("content", "is_error")

    def __init__(self, content: str, is_error: bool = False):
        self.content = content
        self.is_error = is_error


def handle_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    run: ForgeRun,
    repo_root: Path,
    provider: KernelSkillsProvider,
) -> ToolResult:
    """Dispatch a single tool call by name."""
    try:
        if name == "read_skill":
            skill_id = arguments.get("skill_id", "")
            return ToolResult(provider.show(skill_id))

        if name == "list_candidate_files":
            cdir = _candidate_dir(run, repo_root)
            files = sorted(p.name for p in cdir.iterdir() if p.is_file()) if cdir.exists() else []
            return ToolResult(json.dumps({"files": files}))

        if name == "write_candidate_file":
            fname = _safe_filename(arguments.get("filename", ""))
            if fname is None:
                return ToolResult(
                    "Refused: filename must be a plain name with no path components.",
                    is_error=True,
                )
            cdir = _candidate_dir(run, repo_root)
            cdir.mkdir(parents=True, exist_ok=True)
            content = arguments.get("content", "")
            (cdir / fname).write_text(content)
            return ToolResult(json.dumps({"wrote": fname, "bytes": len(content)}))

        if name == "read_candidate_file":
            fname = _safe_filename(arguments.get("filename", ""))
            if fname is None:
                return ToolResult("Refused: invalid filename.", is_error=True)
            target = _candidate_dir(run, repo_root) / fname
            if not target.exists():
                return ToolResult(f"File not found: {fname}", is_error=True)
            return ToolResult(target.read_text())

        if name == "run_verify":
            skip_cuda = bool(arguments.get("skip_cuda_check", False))
            _, result = verify_candidate(run, repo_root, require_cuda=not skip_cuda)
            return ToolResult(result.model_dump_json(indent=2))

        if name == "run_benchmark":
            _, result = benchmark_candidate(run, repo_root)
            return ToolResult(result.model_dump_json(indent=2))

        if name == "give_up":
            reason = arguments.get("reason", "(no reason given)")
            return ToolResult(json.dumps({"acknowledged": True, "reason": reason}))

        return ToolResult(f"Unknown tool: {name}", is_error=True)

    except Exception as e:
        return ToolResult(f"Tool {name} raised: {type(e).__name__}: {e}", is_error=True)
