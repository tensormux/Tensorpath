"""Verified kernel registry.

The registry is a single JSON file at `kernel_registry/verified_kernels.json`.
Each entry describes a kernel that has passed verification and benchmarking.

Reads/writes are atomic-enough for a developer tool: load → mutate → save.
The recommender will read this file at request time so newly-promoted kernels
become visible without restarting the server.

The registry rejects duplicate `kernel_id` insertions. Promotion is the only
path that adds entries; the verifier and benchmarker never touch this file.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.services.forge.models import KernelOp, PromotedKernel


_REGISTRY_RELATIVE = Path("kernel_registry") / "verified_kernels.json"


class KernelRegistry:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.path = repo_root / _REGISTRY_RELATIVE

    def _load(self) -> dict:
        if not self.path.exists():
            return {"kernels": []}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"verified_kernels.json is corrupt at {self.path}: {e}"
            ) from e

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))

    def list_kernels(self) -> list[dict]:
        return list(self._load().get("kernels", []))

    def find_kernels(
        self,
        op: KernelOp | str | None = None,
        target_gpu: str | None = None,
        dtype: str | None = None,
        shape: dict[str, int] | None = None,
    ) -> list[dict]:
        op_value = op.value if isinstance(op, KernelOp) else op
        results = self.list_kernels()
        if op_value is not None:
            results = [k for k in results if k.get("op") == op_value]
        if target_gpu is not None:
            results = [k for k in results if k.get("target_gpu") == target_gpu]
        if dtype is not None:
            results = [k for k in results if k.get("dtype") == dtype]
        if shape:
            def _matches_shape(k: dict) -> bool:
                ks = k.get("shape", {})
                return all(ks.get(dim) == val for dim, val in shape.items())
            results = [k for k in results if _matches_shape(k)]
        return results

    def has_kernel(self, kernel_id: str) -> bool:
        return any(k.get("kernel_id") == kernel_id for k in self.list_kernels())

    def add_kernel(self, kernel: PromotedKernel) -> None:
        data = self._load()
        existing_ids = {k.get("kernel_id") for k in data.get("kernels", [])}
        if kernel.kernel_id in existing_ids:
            raise ValueError(f"duplicate kernel ID: {kernel.kernel_id}")
        data.setdefault("kernels", []).append(kernel.model_dump(mode="json"))
        self._save(data)
