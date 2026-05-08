"""Protocol for optimization passes that annotate deployment plans.

A pass takes a `DeploymentPlan` and returns the same plan (or an updated copy)
with additional metadata attached. Passes do not change a plan's identity —
they only add information the recommendation UI can surface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.schemas import DeploymentPlan


@runtime_checkable
class OptimizationPass(Protocol):
    name: str

    def apply(self, plan: DeploymentPlan) -> DeploymentPlan:
        ...
