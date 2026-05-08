"""QuantizationEligibilityPass — annotate why a quantization path was selected.

Stub for now. Future versions will record reasoning like
"AWQ chosen because Marlin kernels are available on this GPU tier" or
"FP8 chosen because the GPU supports it natively".
"""

from __future__ import annotations

from app.schemas import DeploymentPlan


class QuantizationEligibilityPass:
    name = "quantization_eligibility"

    def apply(self, plan: DeploymentPlan) -> DeploymentPlan:
        return plan
