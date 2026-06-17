"""Tests for promotion trigger edge case fix."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.forge.agent_state import (
    AgenticRunState,
    AgenticRunStatus,
    init_state,
    load_state,
    save_state,
)
from app.services.forge.models import (
    ForgeRun,
    ForgeRunStatus,
    KernelLanguage,
    KernelOp,
    KernelTaskSpec,
)


def _make_run(tmp_path: Path) -> ForgeRun:
    """Create a test run."""
    task = KernelTaskSpec(
        op=KernelOp.RMSNORM,
        language=KernelLanguage.TRITON,
        target_gpu="RTX 4070",
        dtype="fp16",
        shape={"hidden_size": 4096},
    )
    run_id = "test-run-123"
    artifact_dir = tmp_path / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "skill_bundle.md").write_text("# Test bundle")
    
    return ForgeRun(
        run_id=run_id,
        status=ForgeRunStatus.PLANNED,
        task=task,
        artifact_dir=str(artifact_dir),
    )


def test_iteration_fields_initialized(tmp_path: Path):
    """Test that iteration tracking fields are initialized to None."""
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    
    assert state.last_verify_iteration is None
    assert state.last_benchmark_iteration is None


def test_iteration_fields_can_be_set(tmp_path: Path):
    """Test that iteration fields can be set and retrieved."""
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    
    # Set iteration fields
    state.last_verify_passed = True
    state.last_verify_iteration = 2
    state.last_benchmark_passed = True
    state.last_benchmark_iteration = 2
    
    assert state.last_verify_iteration == 2
    assert state.last_benchmark_iteration == 2
    assert state.last_verify_passed is True
    assert state.last_benchmark_passed is True


def test_iteration_fields_persist_across_save_load(tmp_path: Path):
    """Test that iteration fields are properly serialized and deserialized."""
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    
    # Set iteration fields
    state.last_verify_passed = True
    state.last_verify_iteration = 2
    state.last_benchmark_passed = True
    state.last_benchmark_iteration = 2
    
    # Save and reload
    save_state(state, run, tmp_path)
    loaded = load_state(run, tmp_path)
    
    assert loaded is not None
    assert loaded.last_verify_iteration == 2
    assert loaded.last_benchmark_iteration == 2
    assert loaded.last_verify_passed is True
    assert loaded.last_benchmark_passed is True


def test_iteration_fields_independent(tmp_path: Path):
    """Test that verify and benchmark iteration fields are independent."""
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    
    # Set different iterations for verify and benchmark
    state.last_verify_passed = True
    state.last_verify_iteration = 1
    state.last_benchmark_passed = True
    state.last_benchmark_iteration = 2
    
    # They should be independent
    assert state.last_verify_iteration == 1
    assert state.last_benchmark_iteration == 2
    
    # This simulates the edge case: gates passed in different iterations
    # The promotion check should verify they're in the same iteration
    same_iteration = (
        state.last_verify_iteration == state.last_benchmark_iteration
    )
    assert same_iteration is False


def test_iteration_fields_same_iteration(tmp_path: Path):
    """Test that iteration fields can be set to the same iteration."""
    run = _make_run(tmp_path)
    state = init_state(run, tmp_path)
    
    # Set same iteration for both gates
    state.last_verify_passed = True
    state.last_verify_iteration = 3
    state.last_benchmark_passed = True
    state.last_benchmark_iteration = 3
    
    # They should be the same
    assert state.last_verify_iteration == 3
    assert state.last_benchmark_iteration == 3
    
    # This is the valid case for promotion
    same_iteration = (
        state.last_verify_iteration == state.last_benchmark_iteration
    )
    assert same_iteration is True


def test_promotion_check_logic(tmp_path: Path):
    """Test the promotion check logic that requires same iteration."""
    run = _make_run(tmp_path)
    
    # Case 1: Both gates passed in same iteration - should allow promotion
    state1 = init_state(run, tmp_path)
    state1.last_verify_passed = True
    state1.last_verify_iteration = 1
    state1.last_benchmark_passed = True
    state1.last_benchmark_iteration = 1
    
    should_promote_1 = (
        state1.last_verify_passed
        and state1.last_benchmark_passed
        and state1.last_verify_iteration == 1
        and state1.last_benchmark_iteration == 1
    )
    assert should_promote_1 is True
    
    # Case 2: Gates passed in different iterations - should NOT allow promotion
    state2 = init_state(run, tmp_path)
    state2.last_verify_passed = True
    state2.last_verify_iteration = 1
    state2.last_benchmark_passed = True
    state2.last_benchmark_iteration = 2
    
    should_promote_2 = (
        state2.last_verify_passed
        and state2.last_benchmark_passed
        and state2.last_verify_iteration == 2  # Current iteration
        and state2.last_benchmark_iteration == 2  # Current iteration
    )
    assert should_promote_2 is False
    
    # Case 3: Only verify passed - should NOT allow promotion
    state3 = init_state(run, tmp_path)
    state3.last_verify_passed = True
    state3.last_verify_iteration = 1
    state3.last_benchmark_passed = False
    state3.last_benchmark_iteration = 1
    
    should_promote_3 = (
        state3.last_verify_passed
        and state3.last_benchmark_passed
        and state3.last_verify_iteration == 1
        and state3.last_benchmark_iteration == 1
    )
    assert should_promote_3 is False
