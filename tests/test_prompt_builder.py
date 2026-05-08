"""Tests for the prompt builder — required sections, forbidden behavior, bundle embedding."""

from __future__ import annotations

import pytest

from app.services.forge.models import (
    KernelLanguage,
    KernelOp,
    KernelTaskSpec,
    SkillBundleSpec,
)
from app.services.forge.prompt_builder import build_agent_prompt


pytestmark = pytest.mark.forge


def _spec(skill_ids=None, bundle="# bundle content"):
    task = KernelTaskSpec(
        op=KernelOp.RMSNORM,
        language=KernelLanguage.TRITON,
        target_gpu="RTX 4070",
        dtype="fp16",
        shape={"batch": 16, "hidden_size": 4096},
        constraints={"max_abs_error": 0.001},
        objective="latency",
    )
    return SkillBundleSpec(
        task=task,
        # `or [...]` would swallow an explicitly-empty list, which one of the
        # tests below specifically needs to pass through.
        skill_ids=skill_ids if skill_ids is not None else ["inference.write-triton-rmsnorm-kernel"],
        skill_bundle=bundle,
    )


def test_prompt_contains_all_required_sections():
    out = build_agent_prompt(_spec())
    for section in (
        "## Operation",
        "## Target Hardware",
        "## Tensor Shapes",
        "## Dtype",
        "## Objective",
        "## Constraints",
        "## Required Deliverables",
        "## Correctness Requirements",
        "## Benchmark Requirements",
        "## Files to Produce",
        "## Forbidden Behavior",
        "## Skill Bundle",
    ):
        assert section in out, f"missing section: {section}"


def test_prompt_includes_operation_and_shape():
    out = build_agent_prompt(_spec())
    assert "rmsnorm" in out
    assert "batch=16" in out
    assert "hidden_size=4096" in out


def test_prompt_includes_target_gpu_and_dtype():
    out = build_agent_prompt(_spec())
    assert "RTX 4070" in out
    assert "fp16" in out


def test_prompt_lists_required_files():
    out = build_agent_prompt(_spec())
    for f in ("kernel.py", "reference.py", "test_correctness.py", "bench.py", "metadata.json"):
        assert f in out


def test_prompt_includes_forbidden_behavior():
    out = build_agent_prompt(_spec())
    assert "Forbidden Behavior" in out
    assert "subprocess" in out.lower()
    assert "eval" in out.lower()
    assert "candidate directory" in out.lower()


def test_prompt_warns_against_layernorm_as_rmsnorm_reference():
    out = build_agent_prompt(_spec())
    # The prompt explicitly tells the agent not to use LayerNorm as the RMSNorm reference
    assert "LayerNorm" in out and "RMSNorm" in out
    assert "do not subtract" in out.lower() or "does not subtract" in out.lower()


def test_prompt_embeds_skill_bundle_verbatim():
    bundle = "# Custom Bundle\n\nThis content must appear in the final prompt verbatim.\n"
    out = build_agent_prompt(_spec(bundle=bundle))
    assert "This content must appear in the final prompt verbatim." in out


def test_prompt_lists_selected_skill_ids():
    out = build_agent_prompt(_spec(skill_ids=["a.skill", "b.skill"]))
    assert "`a.skill`" in out
    assert "`b.skill`" in out


def test_prompt_handles_empty_skill_list():
    out = build_agent_prompt(_spec(skill_ids=[]))
    # Should still render and signal that no external skills were retrieved
    assert "no external skills" in out.lower()


def test_prompt_substitutes_run_id_when_provided():
    out = build_agent_prompt(_spec(), run_id="20260507T120000_rmsnorm_rtx-4070")
    assert "forge_runs/20260507T120000_rmsnorm_rtx-4070/candidate/" in out


def test_prompt_uses_placeholder_when_run_id_not_provided():
    out = build_agent_prompt(_spec(), run_id=None)
    assert "forge_runs/<run_id>/candidate/" in out


def test_prompt_includes_constraints():
    out = build_agent_prompt(_spec())
    assert "max_abs_error" in out


def test_prompt_includes_warmup_and_iters_specification():
    out = build_agent_prompt(_spec())
    assert "20 warmup" in out or "20 warmup iterations" in out
    assert "100 measured" in out
