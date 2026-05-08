"""KernelRegistryPass — annotate plans with verified kernel optimizations.

For each plan, look up the verified kernel registry for kernels that match
the plan's GPU + activation dtype. We do NOT propagate the microbenchmark
speedup into the plan's end-to-end latency/cost numbers. The annotation is
shown as an "optimization opportunity available" with `applied=False` and
`evidence_level="op_level"` because the speedup we measured is a
microbenchmark, not a full endpoint benchmark.

Future versions may set `applied=True` once we have runtime-level evidence
that a verified kernel actually flows through the serving stack on this
hardware/dtype combination — but that requires runtime integration and a
serving benchmark, neither of which exist yet.
"""

from __future__ import annotations

import re

from app.schemas import DeploymentPlan, KernelOptimizationMetadata
from app.services.forge.registry import KernelRegistry


# Map from a plan's quantization to the dtype the kernel sees on activations.
# RMSNorm and similar normalization layers operate on activations, which are
# typically held in fp16 even when weights are int4 (AWQ/GPTQ).
_QUANT_TO_ACTIVATION_DTYPE = {
    "fp16": ("fp16",),
    "bf16": ("bf16",),
    "fp32": ("fp32",),
    "fp8":  ("fp8", "fp16"),
    "awq":  ("fp16",),
    "gptq": ("fp16",),
    "int8": ("fp16", "int8"),
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


class KernelRegistryPass:
    name = "kernel_registry"

    def __init__(self, registry: KernelRegistry):
        self.registry = registry

    def apply(self, plan: DeploymentPlan) -> DeploymentPlan:
        annotations = self._lookup(plan)
        if not annotations:
            return plan
        return plan.model_copy(update={"kernel_optimizations": annotations})

    def _lookup(self, plan: DeploymentPlan) -> list[KernelOptimizationMetadata]:
        try:
            all_kernels = self.registry.list_kernels()
        except Exception:
            return []
        if not all_kernels:
            return []

        plan_gpu_slugs = {_slug(plan.gpu_name), _slug(plan.gpu_tier)}
        plan_dtype_candidates = _QUANT_TO_ACTIVATION_DTYPE.get(plan.quantization, ())

        matches: list[KernelOptimizationMetadata] = []
        for k in all_kernels:
            if not self._gpu_matches(k.get("target_gpu", ""), plan_gpu_slugs):
                continue
            if plan_dtype_candidates and k.get("dtype") not in plan_dtype_candidates:
                continue
            speedup = (k.get("benchmark") or {}).get("speedup")
            matches.append(
                KernelOptimizationMetadata(
                    available=True,
                    applied=False,
                    kernel_id=k.get("kernel_id"),
                    op=k.get("op"),
                    speedup=float(speedup) if isinstance(speedup, (int, float)) else None,
                    evidence_level=k.get("evidence_level"),
                    benchmark_scope="microbenchmark",
                    notes=(
                        "Verified op-level kernel available. Not used for full "
                        "endpoint cost adjustment."
                    ),
                )
            )
        return matches

    @staticmethod
    def _gpu_matches(kernel_gpu: str, plan_slugs: set[str]) -> bool:
        kslug = _slug(kernel_gpu)
        if not kslug:
            return False
        for ps in plan_slugs:
            if not ps:
                continue
            if kslug in ps or ps in kslug:
                return True
        return False
