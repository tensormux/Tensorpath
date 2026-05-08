"""Thin Python wrapper around the `@krxgu/kernel-skills` CLI.

This module is intentionally generic. It does not know about RMSNorm, vLLM,
GPU names, models, or inference logic. It only invokes the kernel-skills CLI
and returns its output.

Resolution order for the CLI command:
1. `KERNEL_SKILLS_CMD` env var (escape hatch for tests / weird envs)
2. Locally installed `./node_modules/.bin/kernel-skills`
3. `npx -y @krxgu/kernel-skills@<pinned>`
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


PINNED_VERSION = "0.2.0"


def _node_runnable() -> bool:
    """The local node_modules/.bin/kernel-skills wrapper is a shell script that
    `exec`s `node`. On WSL hosts where Node is installed only on the Windows
    side, the wrapper exists but `node` itself isn't on the bash PATH, so the
    wrapper dies with a permission error. In that case, fall back to `npx`,
    which the Windows Node install does expose via its shim."""
    return shutil.which("node") is not None


@dataclass(frozen=True)
class SkillMetadata:
    id: str
    name: str


class KernelSkillsProvider:
    def __init__(self, repo_root: Path, package_version: str = PINNED_VERSION):
        self.repo_root = repo_root
        self.package_version = package_version
        self.cmd = self._resolve_cmd()

    def _resolve_cmd(self) -> list[str]:
        override = os.getenv("KERNEL_SKILLS_CMD")
        if override:
            # split a string override on whitespace so callers can pass
            # `KERNEL_SKILLS_CMD="npx kernel-skills"` if needed
            return override.split()

        local = self.repo_root / "node_modules" / ".bin" / "kernel-skills"
        if local.exists() and _node_runnable():
            return [str(local)]

        return ["npx", "-y", f"@krxgu/kernel-skills@{self.package_version}"]

    def run(self, args: list[str]) -> str:
        proc = subprocess.run(
            [*self.cmd, *args],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"kernel-skills failed: {' '.join(args)}\n"
                f"stdout={proc.stdout}\n"
                f"stderr={proc.stderr}"
            )
        return proc.stdout

    def list_skills(self) -> str:
        return self.run(["list"])

    def search(self, query: str) -> str:
        return self.run(["search", query])

    def show(self, skill_id: str) -> str:
        return self.run(["show", skill_id])

    def bundle(self, skill_ids: list[str]) -> str:
        if not skill_ids:
            raise ValueError("skill_ids cannot be empty")
        return self.run(["bundle", *skill_ids])
