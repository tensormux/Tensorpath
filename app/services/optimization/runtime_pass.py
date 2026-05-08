"""RuntimeTuningPass — planning annotations for runtime knobs.

Stub for now. Real implementation would record recommended values for
`max_num_batched_tokens`, `max_num_seqs`, `gpu_memory_utilization`,
`enforce_eager` vs CUDA graphs — but only with benchmark-backed evidence,
which we don't have at v0.
"""

from __future__ import annotations

from app.schemas import DeploymentPlan


class RuntimeTuningPass:
    name = "runtime_tuning"

    def apply(self, plan: DeploymentPlan) -> DeploymentPlan:
        return plan
