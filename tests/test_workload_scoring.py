"""Tests for workload-aware scoring logic."""

import pytest
from app.schemas import (
    BenchmarkProfile,
    BenchmarkSource,
    OptimizationPriority,
    WorkloadConstraints,
    WorkloadType,
)
from app.services.recommender.scoring import score_candidate
from app.services.runtime_registry import RuntimeRegistry


@pytest.fixture
def registry():
    return RuntimeRegistry()


def _make_profile(
    model_id="test-model",
    gpu_tier="l4",
    backend="vllm",
    quantization="fp16",
    ttft_ms_p50=100.0,
    ttft_ms_p95=150.0,
    itl_ms_p50=20.0,
    itl_ms_p95=30.0,
    tokens_per_sec=50.0,
    max_concurrent_requests=10,
    throughput_at_max_concurrency_tps=200.0,
    vram_usage_gb=10.0,
    hourly_cost_usd=1.0,
    source=BenchmarkSource.ESTIMATED,
):
    """Helper to create a BenchmarkProfile with defaults."""
    return BenchmarkProfile(
        model_id=model_id,
        gpu_tier=gpu_tier,
        backend=backend,
        quantization=quantization,
        ttft_ms_p50=ttft_ms_p50,
        ttft_ms_p95=ttft_ms_p95,
        itl_ms_p50=itl_ms_p50,
        itl_ms_p95=itl_ms_p95,
        tokens_per_sec=tokens_per_sec,
        max_concurrent_requests=max_concurrent_requests,
        throughput_at_max_concurrency_tps=throughput_at_max_concurrency_tps,
        vram_usage_gb=vram_usage_gb,
        hourly_cost_usd=hourly_cost_usd,
        source=source,
    )


class TestChatWorkload:
    """Test CHAT workload scoring - prioritizes TTFT + ITL for smooth streaming."""

    def test_chat_prioritizes_low_ttft(self, registry):
        """CHAT should favor profiles with lower TTFT (time to first token)."""
        # Profile with low TTFT but average ITL
        fast_ttft = _make_profile(ttft_ms_p95=100.0, itl_ms_p95=30.0)
        # Profile with high TTFT but same ITL
        slow_ttft = _make_profile(ttft_ms_p95=200.0, itl_ms_p95=30.0)
        
        profiles = [fast_ttft, slow_ttft]
        
        fast_score = score_candidate(
            fast_ttft, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CHAT, WorkloadConstraints(), registry
        )
        slow_score = score_candidate(
            slow_ttft, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CHAT, WorkloadConstraints(), registry
        )
        
        assert fast_score.latency > slow_score.latency
        assert fast_score.weighted_total > slow_score.weighted_total

    def test_chat_prioritizes_low_itl(self, registry):
        """CHAT should favor profiles with lower ITL (inter-token latency) for smooth streaming."""
        # Profile with low ITL
        fast_itl = _make_profile(ttft_ms_p95=150.0, itl_ms_p95=20.0)
        # Profile with high ITL
        slow_itl = _make_profile(ttft_ms_p95=150.0, itl_ms_p95=40.0)
        
        profiles = [fast_itl, slow_itl]
        
        fast_score = score_candidate(
            fast_itl, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CHAT, WorkloadConstraints(), registry
        )
        slow_score = score_candidate(
            slow_itl, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CHAT, WorkloadConstraints(), registry
        )
        
        assert fast_score.latency > slow_score.latency
        assert fast_score.weighted_total > slow_score.weighted_total

    def test_chat_uses_both_ttft_and_itl(self, registry):
        """CHAT should combine TTFT and ITL in latency score."""
        # Profile with both low TTFT and low ITL
        best = _make_profile(ttft_ms_p95=100.0, itl_ms_p95=20.0)
        # Profile with both high
        worst = _make_profile(ttft_ms_p95=200.0, itl_ms_p95=40.0)
        
        profiles = [best, worst]
        
        best_score = score_candidate(
            best, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CHAT, WorkloadConstraints(), registry
        )
        worst_score = score_candidate(
            worst, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CHAT, WorkloadConstraints(), registry
        )
        
        # Best should have significantly higher latency score
        assert best_score.latency > worst_score.latency + 0.3


class TestCodegenWorkload:
    """Test CODEGEN workload scoring - prioritizes ITL + throughput for long outputs."""

    def test_codegen_prioritizes_itl_over_ttft(self, registry):
        """CODEGEN should weight ITL more heavily than TTFT for long code generation."""
        # Profile with low ITL but higher TTFT
        good_streaming = _make_profile(ttft_ms_p95=180.0, itl_ms_p95=20.0)
        # Profile with low TTFT but higher ITL
        fast_start = _make_profile(ttft_ms_p95=120.0, itl_ms_p95=40.0)
        
        profiles = [good_streaming, fast_start]
        
        streaming_score = score_candidate(
            good_streaming, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CODEGEN, WorkloadConstraints(), registry
        )
        start_score = score_candidate(
            fast_start, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CODEGEN, WorkloadConstraints(), registry
        )
        
        # Good streaming should win due to higher ITL weight
        assert streaming_score.latency > start_score.latency

    def test_codegen_considers_throughput(self, registry):
        """CODEGEN should consider throughput for long outputs."""
        # Profile with high throughput
        high_tps = _make_profile(tokens_per_sec=80.0, itl_ms_p95=30.0)
        # Profile with low throughput
        low_tps = _make_profile(tokens_per_sec=40.0, itl_ms_p95=30.0)
        
        profiles = [high_tps, low_tps]
        
        high_score = score_candidate(
            high_tps, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CODEGEN, WorkloadConstraints(), registry
        )
        low_score = score_candidate(
            low_tps, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CODEGEN, WorkloadConstraints(), registry
        )
        
        assert high_score.throughput > low_score.throughput


class TestSummarizationWorkload:
    """Test SUMMARIZATION workload scoring - prioritizes TTFT for fast prefill."""

    def test_summarization_prioritizes_ttft(self, registry):
        """SUMMARIZATION should heavily weight TTFT for fast prefill of long inputs."""
        # Profile with low TTFT
        fast_prefill = _make_profile(ttft_ms_p95=100.0, itl_ms_p95=30.0)
        # Profile with high TTFT
        slow_prefill = _make_profile(ttft_ms_p95=200.0, itl_ms_p95=30.0)
        
        profiles = [fast_prefill, slow_prefill]
        
        fast_score = score_candidate(
            fast_prefill, profiles, OptimizationPriority.BALANCED,
            WorkloadType.SUMMARIZATION, WorkloadConstraints(), registry
        )
        slow_score = score_candidate(
            slow_prefill, profiles, OptimizationPriority.BALANCED,
            WorkloadType.SUMMARIZATION, WorkloadConstraints(), registry
        )
        
        # Fast prefill should have much higher latency score
        assert fast_score.latency > slow_score.latency + 0.3

    def test_summarization_less_sensitive_to_itl(self, registry):
        """SUMMARIZATION should be less sensitive to ITL than CHAT."""
        # Profile with low ITL
        fast_itl = _make_profile(ttft_ms_p95=150.0, itl_ms_p95=20.0)
        # Profile with high ITL
        slow_itl = _make_profile(ttft_ms_p95=150.0, itl_ms_p95=40.0)
        
        profiles = [fast_itl, slow_itl]
        
        # Score for SUMMARIZATION
        sum_fast = score_candidate(
            fast_itl, profiles, OptimizationPriority.BALANCED,
            WorkloadType.SUMMARIZATION, WorkloadConstraints(), registry
        )
        sum_slow = score_candidate(
            slow_itl, profiles, OptimizationPriority.BALANCED,
            WorkloadType.SUMMARIZATION, WorkloadConstraints(), registry
        )
        
        # Score for CHAT
        chat_fast = score_candidate(
            fast_itl, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CHAT, WorkloadConstraints(), registry
        )
        chat_slow = score_candidate(
            slow_itl, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CHAT, WorkloadConstraints(), registry
        )
        
        # CHAT should be more sensitive to ITL difference than SUMMARIZATION
        chat_diff = chat_fast.latency - chat_slow.latency
        sum_diff = sum_fast.latency - sum_slow.latency
        
        assert chat_diff > sum_diff


class TestBatchWorkload:
    """Test BATCH workload scoring - prioritizes throughput only."""

    def test_batch_ignores_latency(self, registry):
        """BATCH should give neutral latency score regardless of TTFT/ITL."""
        # Profile with very low latency
        fast = _make_profile(ttft_ms_p95=50.0, itl_ms_p95=10.0, tokens_per_sec=60.0)
        # Profile with high latency
        slow = _make_profile(ttft_ms_p95=300.0, itl_ms_p95=50.0, tokens_per_sec=60.0)
        
        profiles = [fast, slow]
        
        fast_score = score_candidate(
            fast, profiles, OptimizationPriority.BALANCED,
            WorkloadType.BATCH, WorkloadConstraints(), registry
        )
        slow_score = score_candidate(
            slow, profiles, OptimizationPriority.BALANCED,
            WorkloadType.BATCH, WorkloadConstraints(), registry
        )
        
        # Both should have neutral latency score (0.5)
        assert fast_score.latency == 0.5
        assert slow_score.latency == 0.5
        
        # But throughput should be the same
        assert fast_score.throughput == slow_score.throughput

    def test_batch_prioritizes_throughput(self, registry):
        """BATCH should heavily weight throughput."""
        # Profile with high throughput
        high_tps = _make_profile(tokens_per_sec=100.0, ttft_ms_p95=200.0)
        # Profile with low throughput
        low_tps = _make_profile(tokens_per_sec=30.0, ttft_ms_p95=100.0)
        
        profiles = [high_tps, low_tps]
        
        high_score = score_candidate(
            high_tps, profiles, OptimizationPriority.BALANCED,
            WorkloadType.BATCH, WorkloadConstraints(), registry
        )
        low_score = score_candidate(
            low_tps, profiles, OptimizationPriority.BALANCED,
            WorkloadType.BATCH, WorkloadConstraints(), registry
        )
        
        # High throughput should win despite worse latency
        assert high_score.weighted_total > low_score.weighted_total


class TestEmbeddingWorkload:
    """Test EMBEDDING workload scoring - prioritizes concurrent throughput."""

    def test_embedding_ignores_generation_metrics(self, registry):
        """EMBEDDING should ignore TTFT and ITL (no generation)."""
        # Profile with low TTFT/ITL
        fast_gen = _make_profile(
            ttft_ms_p95=50.0, itl_ms_p95=10.0,
            throughput_at_max_concurrency_tps=200.0
        )
        # Profile with high TTFT/ITL
        slow_gen = _make_profile(
            ttft_ms_p95=300.0, itl_ms_p95=50.0,
            throughput_at_max_concurrency_tps=200.0
        )
        
        profiles = [fast_gen, slow_gen]
        
        fast_score = score_candidate(
            fast_gen, profiles, OptimizationPriority.BALANCED,
            WorkloadType.EMBEDDING, WorkloadConstraints(), registry
        )
        slow_score = score_candidate(
            slow_gen, profiles, OptimizationPriority.BALANCED,
            WorkloadType.EMBEDDING, WorkloadConstraints(), registry
        )
        
        # Both should have neutral latency score
        assert fast_score.latency == 0.5
        assert slow_score.latency == 0.5
        
        # And same throughput score
        assert fast_score.throughput == slow_score.throughput

    def test_embedding_uses_concurrent_throughput(self, registry):
        """EMBEDDING should use throughput_at_max_concurrency_tps, not tokens_per_sec."""
        # Profile with high concurrent throughput
        high_concurrent = _make_profile(
            tokens_per_sec=50.0,
            throughput_at_max_concurrency_tps=500.0
        )
        # Profile with low concurrent throughput
        low_concurrent = _make_profile(
            tokens_per_sec=50.0,
            throughput_at_max_concurrency_tps=100.0
        )
        
        profiles = [high_concurrent, low_concurrent]
        
        high_score = score_candidate(
            high_concurrent, profiles, OptimizationPriority.BALANCED,
            WorkloadType.EMBEDDING, WorkloadConstraints(), registry
        )
        low_score = score_candidate(
            low_concurrent, profiles, OptimizationPriority.BALANCED,
            WorkloadType.EMBEDDING, WorkloadConstraints(), registry
        )
        
        # High concurrent throughput should win
        assert high_score.throughput > low_score.throughput
        assert high_score.weighted_total > low_score.weighted_total


class TestBackwardCompatibility:
    """Test that existing behavior is preserved for backward compatibility."""

    def test_chat_behavior_similar_to_before(self, registry):
        """CHAT workload should behave similarly to the old implementation."""
        # Profile with low TTFT
        fast = _make_profile(ttft_ms_p95=100.0, itl_ms_p95=30.0)
        # Profile with high TTFT
        slow = _make_profile(ttft_ms_p95=200.0, itl_ms_p95=30.0)
        
        profiles = [fast, slow]
        
        fast_score = score_candidate(
            fast, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CHAT, WorkloadConstraints(), registry
        )
        slow_score = score_candidate(
            slow, profiles, OptimizationPriority.BALANCED,
            WorkloadType.CHAT, WorkloadConstraints(), registry
        )
        
        # Fast TTFT should still win
        assert fast_score.weighted_total > slow_score.weighted_total

    def test_all_workloads_produce_valid_scores(self, registry):
        """All workload types should produce scores in [0, 1] range."""
        profile = _make_profile()
        profiles = [profile]
        
        for workload_type in WorkloadType:
            scores = score_candidate(
                profile, profiles, OptimizationPriority.BALANCED,
                workload_type, WorkloadConstraints(), registry
            )
            
            assert 0 <= scores.latency <= 1
            assert 0 <= scores.throughput <= 1
            assert 0 <= scores.cost <= 1
            assert 0 <= scores.quality <= 1
            assert 0 <= scores.simplicity <= 1
            assert scores.weighted_total >= 0

    def test_latency_priority_still_favors_low_latency(self, registry):
        """LATENCY priority should still favor low latency across all workloads."""
        fast = _make_profile(ttft_ms_p95=100.0, itl_ms_p95=20.0, tokens_per_sec=40.0)
        slow = _make_profile(ttft_ms_p95=200.0, itl_ms_p95=40.0, tokens_per_sec=60.0)
        
        profiles = [fast, slow]
        
        for workload_type in [WorkloadType.CHAT, WorkloadType.CODEGEN, WorkloadType.SUMMARIZATION]:
            fast_score = score_candidate(
                fast, profiles, OptimizationPriority.LATENCY,
                workload_type, WorkloadConstraints(), registry
            )
            slow_score = score_candidate(
                slow, profiles, OptimizationPriority.LATENCY,
                workload_type, WorkloadConstraints(), registry
            )
            
            # Fast should win when prioritizing latency
            assert fast_score.weighted_total > slow_score.weighted_total
