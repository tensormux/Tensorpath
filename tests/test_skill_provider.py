"""Tests for KernelSkillsProvider.

These tests mock subprocess so they run in CI without Node/npm/kernel-skills
actually being installed. There's also a tiny integration test guarded by an
env var that exercises the real CLI when it's available.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.forge.skill_provider import KernelSkillsProvider, PINNED_VERSION


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )


def test_resolve_cmd_prefers_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNEL_SKILLS_CMD", "/usr/local/bin/kernel-skills")
    provider = KernelSkillsProvider(repo_root=tmp_path)
    assert provider.cmd == ["/usr/local/bin/kernel-skills"]


def test_resolve_cmd_uses_local_node_modules_when_node_runnable(tmp_path, monkeypatch):
    monkeypatch.delenv("KERNEL_SKILLS_CMD", raising=False)
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    local_bin = bin_dir / "kernel-skills"
    local_bin.write_text("#!/bin/sh\n")

    # Simulate `node` being on PATH so the local bin's shebang would actually run.
    with patch("app.services.forge.skill_provider.shutil.which", return_value="/usr/bin/node"):
        provider = KernelSkillsProvider(repo_root=tmp_path)
    assert provider.cmd == [str(local_bin)]


def test_resolve_cmd_skips_local_bin_when_node_missing(tmp_path, monkeypatch):
    """WSL with Windows-only Node: local bin exists but `node` isn't on PATH.
    We must not pick it — its shell wrapper would die — and instead use npx."""
    monkeypatch.delenv("KERNEL_SKILLS_CMD", raising=False)
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "kernel-skills").write_text("#!/bin/sh\nexec node\n")

    with patch("app.services.forge.skill_provider.shutil.which", return_value=None):
        provider = KernelSkillsProvider(repo_root=tmp_path)
    assert provider.cmd == ["npx", "-y", f"@krxgu/kernel-skills@{PINNED_VERSION}"]


def test_resolve_cmd_falls_back_to_npx_when_no_local_bin(tmp_path, monkeypatch):
    monkeypatch.delenv("KERNEL_SKILLS_CMD", raising=False)
    provider = KernelSkillsProvider(repo_root=tmp_path)
    assert provider.cmd == ["npx", "-y", f"@krxgu/kernel-skills@{PINNED_VERSION}"]


def test_run_returns_stdout_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNEL_SKILLS_CMD", "kernel-skills")
    provider = KernelSkillsProvider(repo_root=tmp_path)

    with patch("subprocess.run", return_value=_completed(stdout="ok\n")) as mock:
        out = provider.run(["list"])
        mock.assert_called_once()
        args = mock.call_args[0][0]
        assert args[:2] == ["kernel-skills", "list"]
    assert out == "ok\n"


def test_run_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNEL_SKILLS_CMD", "kernel-skills")
    provider = KernelSkillsProvider(repo_root=tmp_path)

    with patch("subprocess.run", return_value=_completed(stderr="boom", returncode=2)):
        with pytest.raises(RuntimeError, match="kernel-skills failed"):
            provider.run(["list"])


def test_search_passes_query(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNEL_SKILLS_CMD", "kernel-skills")
    provider = KernelSkillsProvider(repo_root=tmp_path)
    with patch("subprocess.run", return_value=_completed(stdout="results")) as mock:
        provider.search("rmsnorm triton")
        cmd = mock.call_args[0][0]
        assert cmd == ["kernel-skills", "search", "rmsnorm triton"]


def test_show_passes_skill_id(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNEL_SKILLS_CMD", "kernel-skills")
    provider = KernelSkillsProvider(repo_root=tmp_path)
    with patch("subprocess.run", return_value=_completed(stdout="content")) as mock:
        provider.show("triton.rmsnorm")
        cmd = mock.call_args[0][0]
        assert cmd == ["kernel-skills", "show", "triton.rmsnorm"]


def test_bundle_requires_at_least_one_id(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNEL_SKILLS_CMD", "kernel-skills")
    provider = KernelSkillsProvider(repo_root=tmp_path)
    with pytest.raises(ValueError):
        provider.bundle([])


def test_bundle_passes_ids_as_args(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNEL_SKILLS_CMD", "kernel-skills")
    provider = KernelSkillsProvider(repo_root=tmp_path)
    with patch("subprocess.run", return_value=_completed(stdout="bundled")) as mock:
        provider.bundle(["a", "b", "c"])
        cmd = mock.call_args[0][0]
        assert cmd == ["kernel-skills", "bundle", "a", "b", "c"]


@pytest.mark.skipif(
    not os.getenv("NEEVPATH_FORGE_INTEGRATION"),
    reason="integration test; set NEEVPATH_FORGE_INTEGRATION=1 to run with real CLI",
)
def test_integration_list_skills_real_cli():
    """Real CLI smoke test. Skipped by default."""
    repo = Path(__file__).resolve().parents[1]
    provider = KernelSkillsProvider(repo_root=repo)
    out = provider.list_skills()
    assert out.strip(), "kernel-skills list returned empty output"
