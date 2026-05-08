"""Tests for the task planner — query selection, language filter, fallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.forge.models import (
    KernelLanguage,
    KernelOp,
    KernelTaskSpec,
)
from app.services.forge.task_planner import (
    OP_TO_SKILL_QUERIES,
    ForgeTaskPlanner,
    _filter_by_language,
    _Hit,
    _parse_search_output,
    _select_top,
)


pytestmark = pytest.mark.forge


def _task(op=KernelOp.RMSNORM, language=KernelLanguage.TRITON) -> KernelTaskSpec:
    return KernelTaskSpec(
        op=op,
        language=language,
        target_gpu="RTX 4070",
        dtype="fp16",
        shape={"hidden_size": 4096},
    )


def test_parse_handles_two_column_output():
    text = "inference.write-triton-rmsnorm-kernel  Write Triton RMSNorm Kernel\n"
    parsed = _parse_search_output(text)
    assert parsed == [(
        "inference.write-triton-rmsnorm-kernel",
        "Write Triton RMSNorm Kernel",
    )]


def test_parse_drops_non_skill_lines():
    """`No skills matched.` should not parse as skill ID `No`."""
    text = "No skills matched.\n"
    assert _parse_search_output(text) == []


def test_parse_skips_blank_and_comment_lines():
    text = "\n  \n# header\ninference.write-triton-rmsnorm-kernel  Write Triton RMSNorm\n"
    parsed = _parse_search_output(text)
    assert len(parsed) == 1


def test_filter_drops_cuda_skills_for_triton_task():
    hits = [
        _Hit("cuda.write-cuda-layernorm-kernel", "x", 1),
        _Hit("inference.write-triton-rmsnorm-kernel", "y", 1),
        _Hit("triton.optimize-triton-block-parameters", "z", 1),
        _Hit("patterns.write-kernel-test-plan", "w", 1),
    ]
    out = _filter_by_language(hits, KernelLanguage.TRITON)
    ids = [h.skill_id for h in out]
    assert "cuda.write-cuda-layernorm-kernel" not in ids
    assert len(out) == 3


def test_filter_drops_triton_skills_for_cuda_task():
    hits = [
        _Hit("cuda.write-cuda-layernorm-kernel", "x", 1),
        _Hit("triton.optimize-triton-block-parameters", "z", 1),
    ]
    out = _filter_by_language(hits, KernelLanguage.CUDA)
    ids = [h.skill_id for h in out]
    assert "triton.optimize-triton-block-parameters" not in ids
    assert "cuda.write-cuda-layernorm-kernel" in ids


def test_select_top_forces_in_hygiene_skills():
    """Even if a hygiene skill has the lowest score, it must be selected."""
    hits = [
        _Hit("inference.write-triton-rmsnorm-kernel", "x", 5),
        _Hit("triton.optimize-triton-block-parameters", "y", 4),
        _Hit("patterns.write-kernel-test-plan", "test", 1),  # low-score hygiene
        _Hit("patterns.handle-boundary-conditions", "bound", 1),  # low-score hygiene
    ]
    selected = _select_top(hits)
    assert "patterns.write-kernel-test-plan" in selected
    assert "patterns.handle-boundary-conditions" in selected


def test_select_top_caps_at_max():
    """Only 5 skills max should be returned even with many candidates."""
    hits = [_Hit(f"foo.skill-{i}", f"name-{i}", 1) for i in range(10)]
    assert len(_select_top(hits)) <= 5


def test_op_query_map_covers_known_ops():
    for op in (
        KernelOp.RMSNORM,
        KernelOp.FUSED_ADD_RMSNORM,
        KernelOp.SOFTMAX,
        KernelOp.SAMPLING,
        KernelOp.KV_CACHE_APPEND,
        KernelOp.DEQUANT,
        KernelOp.ROPE,
    ):
        assert op in OP_TO_SKILL_QUERIES
        assert len(OP_TO_SKILL_QUERIES[op]) >= 3


def test_planner_falls_back_when_provider_dies(tmp_path: Path):
    """If every search call raises, plan() returns an empty skill list and
    a fallback bundle that explains the situation."""
    provider = MagicMock()
    provider.search.side_effect = RuntimeError("kernel-skills not installed")

    planner = ForgeTaskPlanner(provider=provider)
    spec = planner.plan(_task())

    assert spec.skill_ids == []
    assert "fallback" in spec.skill_bundle.lower()
    assert "no matching external skill" in spec.skill_bundle.lower()


def test_planner_falls_back_when_op_has_no_queries():
    """An op without an entry in OP_TO_SKILL_QUERIES should still produce a usable bundle."""

    class _UnknownOp:  # not a real KernelOp; bypass the registry
        value = "unknown"

    provider = MagicMock()
    planner = ForgeTaskPlanner(provider=provider)
    task = _task()
    # Manually patch op to a value not in the map
    object.__setattr__(task, "_op_override", _UnknownOp())
    # The real path: pass op=KernelOp.RMSNORM but clear queries
    from app.services.forge import task_planner as planner_mod

    saved = planner_mod.OP_TO_SKILL_QUERIES.pop(KernelOp.RMSNORM)
    try:
        spec = planner.plan(_task())
    finally:
        planner_mod.OP_TO_SKILL_QUERIES[KernelOp.RMSNORM] = saved

    assert spec.skill_ids == []
    assert "fallback" in spec.skill_bundle.lower()


def test_planner_uses_provider_search_results(tmp_path: Path):
    """Happy path with mocked provider that returns valid skill listings."""
    provider = MagicMock()
    provider.search.return_value = (
        "inference.write-triton-rmsnorm-kernel  Write Triton RMSNorm Kernel\n"
        "patterns.write-kernel-test-plan  Write Kernel Test Plan\n"
        "patterns.handle-boundary-conditions  Handle Boundary Conditions\n"
    )
    provider.bundle.return_value = "# Skill Bundle\n\n(stub bundle content)\n"

    planner = ForgeTaskPlanner(provider=provider)
    spec = planner.plan(_task())

    assert "inference.write-triton-rmsnorm-kernel" in spec.skill_ids
    assert "patterns.write-kernel-test-plan" in spec.skill_ids
    assert "patterns.handle-boundary-conditions" in spec.skill_ids
    assert "Skill Bundle" in spec.skill_bundle
    provider.bundle.assert_called_once()
