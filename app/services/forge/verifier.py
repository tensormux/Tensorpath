"""Verify a Forge candidate kernel.

Verification has four gates, in order:

1. Required-files check — kernel.py, reference.py, test_correctness.py,
   bench.py, metadata.json must all exist under candidate/.
2. Metadata sanity — metadata.json must be valid JSON with the expected keys.
3. Safety scan — reject obvious unsafe patterns (subprocess, eval, network,
   filesystem write outside candidate dir). This is NOT sandboxing; it's a
   pre-flight check that prevents an obviously hostile candidate from running.
4. Correctness — run pytest on candidate/test_correctness.py inside a
   subprocess, with a timeout. CUDA must be available; if not, we refuse to
   verify (a kernel that can't be tested can't be promoted).

Output: writes verification_report.json next to the run; updates run.json
status to VERIFIED on pass, REJECTED on fail. Returns the updated ForgeRun
plus the VerificationResult.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from app.services.forge.models import (
    ForgeRun,
    ForgeRunStatus,
    VerificationResult,
)
from app.services.forge.runs import update_run_status


REQUIRED_CANDIDATE_FILES = (
    "kernel.py",
    "reference.py",
    "test_correctness.py",
    "bench.py",
    "metadata.json",
)

# Patterns we refuse to ship through verification. These are coarse and
# regex-based — not a sandbox, just a "no obvious foot-cannons" gate.
_UNSAFE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bos\.system\s*\(", "os.system call"),
    (r"^\s*import\s+subprocess\b", "subprocess import"),
    (r"\bsubprocess\.(run|Popen|call|check_output|check_call)\s*\(", "subprocess call"),
    (r"\bshutil\.rmtree\s*\(", "shutil.rmtree"),
    (r"^\s*import\s+socket\b", "socket import"),
    (r"^\s*import\s+requests\b", "requests import"),
    (r"\beval\s*\(", "eval"),
    (r"\bexec\s*\(", "exec"),
    (r"\b__import__\s*\(", "__import__"),
)

# Safe characters for file paths the candidate may open. The pattern flags
# explicit absolute paths or `..` segments, which suggest writing outside the
# candidate directory.
_SUSPICIOUS_OPEN = re.compile(
    r"""open\s*\(\s*["']\s*(/|[A-Za-z]:[\\/]|\.\.[\\/])""",
)


_PYTEST_TIMEOUT_SEC = 120


def _candidate_dir(run: ForgeRun, repo_root: Path) -> Path:
    return repo_root / run.artifact_dir / "candidate"


def _check_files(candidate_dir: Path) -> str | None:
    if not candidate_dir.exists():
        return f"candidate directory does not exist: {candidate_dir}"
    for name in REQUIRED_CANDIDATE_FILES:
        if not (candidate_dir / name).exists():
            return f"missing required candidate file: {name}"
    return None


def _check_metadata(candidate_dir: Path) -> tuple[dict | None, str | None]:
    meta_path = candidate_dir / "metadata.json"
    try:
        data = json.loads(meta_path.read_text())
    except Exception as e:
        return None, f"metadata.json is not valid JSON: {e}"

    expected_top_level = ("name", "op", "language", "kernel_file")
    missing = [k for k in expected_top_level if k not in data]
    if missing:
        return data, f"metadata.json missing keys: {', '.join(missing)}"
    return data, None


def _scan_unsafe(candidate_dir: Path) -> list[str]:
    flags: list[str] = []
    for py_file in sorted(candidate_dir.glob("*.py")):
        content = py_file.read_text()
        for pattern, label in _UNSAFE_PATTERNS:
            if re.search(pattern, content, flags=re.MULTILINE):
                flags.append(f"{py_file.name}: {label}")
        for match in _SUSPICIOUS_OPEN.finditer(content):
            flags.append(f"{py_file.name}: open() with absolute or ../ path")
            break
    return flags


def _cuda_available() -> bool:
    """True iff `import torch; torch.cuda.is_available()` succeeds in this process."""
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _run_pytest(candidate_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test_correctness.py",
            "-q",
            "--no-header",
            "--rootdir=.",
            "-p",
            "no:cacheprovider",
        ],
        cwd=str(candidate_dir),
        text=True,
        capture_output=True,
        timeout=_PYTEST_TIMEOUT_SEC,
    )


def _write_report(repo_root: Path, run: ForgeRun, result: VerificationResult) -> None:
    artifact_dir = repo_root / run.artifact_dir
    (artifact_dir / "verification_report.json").write_text(
        result.model_dump_json(indent=2)
    )


def verify_candidate(
    run: ForgeRun,
    repo_root: Path,
    *,
    require_cuda: bool = True,
) -> tuple[ForgeRun, VerificationResult]:
    """Run all four gates against the candidate. Always writes a report."""

    candidate_dir = _candidate_dir(run, repo_root)

    err = _check_files(candidate_dir)
    if err:
        result = VerificationResult(passed=False, failure_reason=err)
        _write_report(repo_root, run, result)
        return update_run_status(run, ForgeRunStatus.REJECTED, repo_root), result

    metadata, meta_err = _check_metadata(candidate_dir)
    if meta_err:
        result = VerificationResult(passed=False, failure_reason=meta_err)
        _write_report(repo_root, run, result)
        return update_run_status(run, ForgeRunStatus.REJECTED, repo_root), result

    unsafe = _scan_unsafe(candidate_dir)
    if unsafe:
        result = VerificationResult(
            passed=False,
            failure_reason="unsafe code patterns detected: " + "; ".join(unsafe),
        )
        _write_report(repo_root, run, result)
        return update_run_status(run, ForgeRunStatus.REJECTED, repo_root), result

    if require_cuda and not _cuda_available():
        result = VerificationResult(
            passed=False,
            failure_reason="CUDA unavailable, cannot verify kernel candidate",
        )
        _write_report(repo_root, run, result)
        return update_run_status(run, ForgeRunStatus.REJECTED, repo_root), result

    try:
        proc = _run_pytest(candidate_dir)
    except subprocess.TimeoutExpired:
        result = VerificationResult(
            passed=False,
            failure_reason=f"correctness tests timed out after {_PYTEST_TIMEOUT_SEC}s",
        )
        _write_report(repo_root, run, result)
        return update_run_status(run, ForgeRunStatus.REJECTED, repo_root), result

    # pytest exit codes: 0 = all pass, 5 = no tests collected (treat as fail)
    if proc.returncode != 0:
        # Trim noisy output to last 2KB so the report stays readable
        log = (proc.stdout + ("\n" + proc.stderr if proc.stderr else ""))[-2000:]
        result = VerificationResult(
            passed=False,
            failure_reason=f"correctness tests failed (exit {proc.returncode}):\n{log}",
        )
        _write_report(repo_root, run, result)
        return update_run_status(run, ForgeRunStatus.REJECTED, repo_root), result

    # Pull tolerance from metadata if the candidate declared it
    tolerance = (metadata or {}).get("tolerance", {}) if metadata else {}
    if not isinstance(tolerance, dict):
        tolerance = {}

    result = VerificationResult(
        passed=True,
        tolerance={k: float(v) for k, v in tolerance.items() if isinstance(v, (int, float))},
    )
    _write_report(repo_root, run, result)
    return update_run_status(run, ForgeRunStatus.VERIFIED, repo_root), result
