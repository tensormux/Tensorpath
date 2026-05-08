"""Optimization passes — annotate deployment plans with extra information.

Each pass is a Protocol-conforming object that takes a `DeploymentPlan` and
returns it (typically with new fields populated). The recommendation engine
runs all configured passes after building the ranked plan list.

Real functionality in v0:
    - KernelRegistryPass: annotates plans with verified kernel optimizations.

Stubs / planning-only:
    - RuntimeTuningPass
    - QuantizationEligibilityPass
    - MemoryPlanningPass
    - SpeculativeDecodingPlanningPass

These exist so the system has a place to grow without each new pass
disturbing the engine.
"""

from app.services.optimization.kernel_pass import KernelRegistryPass
from app.services.optimization.passes import OptimizationPass

__all__ = ["KernelRegistryPass", "OptimizationPass"]
