"""Tests for pipeline.tokenizer, focusing on compute_reflection_point."""

from __future__ import annotations

import random

from pipeline.tokenizer import compute_reflection_point

TEXT = "The quick brown fox jumps over the lazy dog. " * 50


class TestComputeReflectionPoint:
    def test_deterministic(self):
        rp1 = compute_reflection_point(TEXT, random.Random("seed_a"))
        rp2 = compute_reflection_point(TEXT, random.Random("seed_a"))
        assert rp1 == rp2

    def test_varies_with_seed(self):
        rp1 = compute_reflection_point(TEXT, random.Random("seed_a"))
        rp2 = compute_reflection_point(TEXT, random.Random("seed_b"))
        # Could theoretically collide, but astronomically unlikely on 2250 chars
        assert rp1 != rp2

    def test_within_text_bounds(self):
        for i in range(200):
            rp = compute_reflection_point(TEXT, random.Random(i))
            assert 1 <= rp < len(TEXT)

    def test_never_zero(self):
        """Reflection point must always include at least one token."""
        for i in range(200):
            rp = compute_reflection_point(TEXT, random.Random(i))
            assert rp > 0

    def test_distribution_ramp_below_20pct(self):
        """The 0-20% region should have fewer samples than 20-40% (linear ramp)."""
        n = 5000
        below_10 = 0
        range_10_20 = 0
        range_20_40 = 0
        for i in range(n):
            rp = compute_reflection_point(TEXT, random.Random(f"dist_{i}"))
            frac = rp / len(TEXT)
            if frac < 0.10:
                below_10 += 1
            elif frac < 0.20:
                range_10_20 += 1
            elif frac < 0.40:
                range_20_40 += 1

        # 0-10% bucket should be much sparser than 10-20% (ramp effect)
        assert below_10 < range_10_20
        # 0-20% total (ramp region) should be less dense than 20-40% (uniform)
        assert below_10 + range_10_20 < range_20_40

    def test_distribution_uniform_above_20pct(self):
        """Deciles from 20-100% should be roughly equally populated."""
        n = 10000
        buckets = [0] * 8  # 20-30, 30-40, ..., 90-100
        for i in range(n):
            rp = compute_reflection_point(TEXT, random.Random(f"unif_{i}"))
            frac = rp / len(TEXT)
            if frac >= 0.20:
                idx = min(7, int((frac - 0.20) / 0.10))
                buckets[idx] += 1

        total_uniform = sum(buckets)
        expected_per_bucket = total_uniform / 8
        for count in buckets:
            # Each bucket should be within 30% of the expected count
            assert (
                abs(count - expected_per_bucket) / expected_per_bucket < 0.30
            ), f"Bucket count {count} too far from expected {expected_per_bucket:.0f}: {buckets}"
