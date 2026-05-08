"""Task planner: map a KernelTaskSpec to a curated skill bundle.

The planner doesn't hardcode skill IDs. It searches kernel-skills for each
concept relevant to the operation, parses the results, dedupes, and picks
the best 3-5 skills (always including correctness/testing guidance and
boundary-condition guidance if they show up in any of the searches).

If the kernel-skills CLI isn't reachable (no Node, package not installed),
the planner returns an empty skill list with a clearly-labeled fallback
bundle so the rest of the pipeline can still produce a usable prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.forge.models import (
    KernelLanguage,
    KernelOp,
    KernelTaskSpec,
    SkillBundleSpec,
)
from app.services.forge.skill_provider import KernelSkillsProvider


# Concept-level queries per op. The planner runs each query against the
# kernel-skills CLI and aggregates the results — exact skill IDs are NOT
# hardcoded so the package can evolve without the planner needing edits.
# The CLI's search is narrow on multi-word queries — it effectively ANDs
# tokens, so "rmsnorm triton" can miss "Write Triton RMSNorm Kernel" while
# bare "rmsnorm" surfaces it. We deliberately mix single-word and multi-word
# queries here. The language filter (_filter_by_language) drops cross-backend
# matches afterward, so keeping single-word queries doesn't pollute results.
OP_TO_SKILL_QUERIES: dict[KernelOp, list[str]] = {
    KernelOp.RMSNORM: [
        "rmsnorm",
        "layernorm",
        "normalization",
        "kernel test plan",
        "boundary",
        "tile size",
    ],
    KernelOp.FUSED_ADD_RMSNORM: [
        "rmsnorm",
        "fuse elementwise",
        "fused",
        "layernorm",
        "kernel test plan",
        "boundary",
    ],
    KernelOp.SOFTMAX: [
        "softmax",
        "numerically stable",
        "reduction",
        "kernel test plan",
        "boundary",
    ],
    KernelOp.SAMPLING: [
        "sampling",
        "softmax",
        "numerically stable",
        "boundary",
        "kernel test plan",
    ],
    KernelOp.KV_CACHE_APPEND: [
        "kv cache",
        "global memory",
        "coalesced",
        "boundary",
        "launch configuration",
        "tile size",
    ],
    KernelOp.DEQUANT: [
        "dequant",
        "int8",
        "fp8",
        "quantized",
        "debug quantized",
        "kernel test plan",
    ],
    KernelOp.ROPE: [
        "rope",
        "rotary",
        "boundary",
        "kernel test plan",
    ],
}

# Skill ID prefixes that should always be included if any search hit them.
# These represent "kernel hygiene" — testing and boundary handling — that
# the agent should never skip regardless of the specific op.
_HYGIENE_PREFIXES: tuple[str, ...] = (
    "patterns.write-kernel-test-plan",
    "patterns.handle-boundary-conditions",
)

_MAX_SKILLS = 5
_MIN_SKILLS = 3


@dataclass
class _Hit:
    skill_id: str
    name: str
    score: int  # how many search queries surfaced this skill


# Skill IDs are <namespace>.<dash-separated-name>, e.g.
# `inference.write-triton-rmsnorm-kernel`. Anything that doesn't match this
# is informational output (e.g. "No skills matched.") and should be ignored.
_SKILL_ID_RE = re.compile(r"^[a-z0-9_]+\.[a-z0-9_-]+$")


def _parse_search_output(text: str) -> list[tuple[str, str]]:
    """Each result line is `<id><whitespace><display name>`. Lines whose first
    token doesn't look like a skill ID are dropped — that filters out the
    CLI's `No skills matched.` empty-result message."""
    results: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s{2,}", line, maxsplit=1)
        if len(parts) == 1:
            parts = line.split(maxsplit=1)
        skill_id = parts[0].strip()
        if not _SKILL_ID_RE.match(skill_id):
            continue
        name = parts[1].strip() if len(parts) > 1 else skill_id
        results.append((skill_id, name))
    return results


def _filter_by_language(hits: list[_Hit], language: KernelLanguage) -> list[_Hit]:
    """Drop language-mismatched hits — e.g. CUDA skills for a Triton task."""
    if language == KernelLanguage.TRITON:
        return [h for h in hits if not h.skill_id.startswith("cuda.")]
    if language == KernelLanguage.CUDA:
        return [h for h in hits if not h.skill_id.startswith("triton.")]
    return hits


def _select_top(hits: list[_Hit]) -> list[str]:
    """Rank by score (search-hit count), keep hygiene skills, cap at MAX."""
    by_score = sorted(hits, key=lambda h: (-h.score, h.skill_id))

    selected: list[str] = []
    seen: set[str] = set()

    # First, force in the hygiene skills if any search surfaced them.
    for h in by_score:
        if any(h.skill_id.startswith(p) for p in _HYGIENE_PREFIXES) and h.skill_id not in seen:
            selected.append(h.skill_id)
            seen.add(h.skill_id)

    # Then fill with the highest-scoring remaining hits.
    for h in by_score:
        if len(selected) >= _MAX_SKILLS:
            break
        if h.skill_id in seen:
            continue
        selected.append(h.skill_id)
        seen.add(h.skill_id)

    return selected


def _fallback_bundle(task: KernelTaskSpec, reason: str) -> str:
    """Markdown emitted when no external skills could be retrieved."""
    return (
        "# Skill bundle (fallback)\n\n"
        f"> No matching external skill found. Proceed with built-in generic\n"
        f"> kernel task guidance.\n\n"
        f"Reason: {reason}\n\n"
        f"## Generic guidance for {task.op.value} ({task.language.value})\n\n"
        "- Implement a numerically correct reference in PyTorch first.\n"
        "- Compare your kernel against the reference at multiple shapes,\n"
        "  including non-power-of-two and partial-block tails.\n"
        "- Use fp32 accumulation for any reductions even when the inputs\n"
        "  are fp16/bf16.\n"
        "- Mask out-of-bounds reads and writes; never assume\n"
        "  hidden_size is a multiple of your block size.\n"
        "- Time with `torch.cuda.synchronize()` around your block; warmup\n"
        "  before measuring; report median over multiple iterations.\n"
        "- Do not report a speedup without printing both baseline and\n"
        "  candidate latency in the same run.\n"
    )


class ForgeTaskPlanner:
    def __init__(self, provider: KernelSkillsProvider):
        self.provider = provider

    def plan(self, task: KernelTaskSpec) -> SkillBundleSpec:
        queries = OP_TO_SKILL_QUERIES.get(task.op, [])
        if not queries:
            return SkillBundleSpec(
                task=task,
                skill_ids=[],
                skill_bundle=_fallback_bundle(task, f"no query mapping for op={task.op.value}"),
            )

        # Search each concept; aggregate hits scored by repetition.
        scores: dict[str, _Hit] = {}
        search_failed_for_all = True
        for q in queries:
            try:
                out = self.provider.search(q)
            except Exception:
                continue
            search_failed_for_all = False
            for sid, name in _parse_search_output(out):
                if sid in scores:
                    scores[sid] = _Hit(sid, name, scores[sid].score + 1)
                else:
                    scores[sid] = _Hit(sid, name, 1)

        if search_failed_for_all:
            return SkillBundleSpec(
                task=task,
                skill_ids=[],
                skill_bundle=_fallback_bundle(task, "kernel-skills CLI not reachable"),
            )

        hits = _filter_by_language(list(scores.values()), task.language)
        skill_ids = _select_top(hits)

        if len(skill_ids) < _MIN_SKILLS and not skill_ids:
            return SkillBundleSpec(
                task=task,
                skill_ids=[],
                skill_bundle=_fallback_bundle(task, "no relevant skills surfaced"),
            )

        try:
            bundle_md = self.provider.bundle(skill_ids)
        except Exception as e:
            bundle_md = _fallback_bundle(task, f"bundle command failed: {e}")
            skill_ids = []

        return SkillBundleSpec(task=task, skill_ids=skill_ids, skill_bundle=bundle_md)
