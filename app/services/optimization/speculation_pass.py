"""SpeculativeDecodingPlanningPass — planning annotation for spec decoding.

Stub for now. A real version would suggest a draft model and project an
expected acceptance rate, but speculative decoding requires endpoint-level
benchmarking before any speedup claim is credible.
"""

from __future__ import annotations

from app.schemas import DeploymentPlan


class SpeculativeDecodingPlanningPass:
    name = "speculative_decoding_planning"

    def apply(self, plan: DeploymentPlan) -> DeploymentPlan:
        return plan
