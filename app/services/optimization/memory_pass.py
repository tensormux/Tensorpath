"""MemoryPlanningPass — annotate weight / KV-cache / headroom split.

Stub for now. Future version will compute a rough split:
    weights memory   = base_vram_gb * scale_for_quantization
    kv cache memory  = derived from max_model_len and max_num_seqs
    headroom         = gpu_vram - weights - kv
"""

from __future__ import annotations

from app.schemas import DeploymentPlan


class MemoryPlanningPass:
    name = "memory_planning"

    def apply(self, plan: DeploymentPlan) -> DeploymentPlan:
        return plan
