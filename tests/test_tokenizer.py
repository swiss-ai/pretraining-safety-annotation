"""Tests for pipeline.tokenizer, focusing on compute_reflection_point."""

from __future__ import annotations

import random

from pipeline.tokenizer import (
    _encode,
    char_offset_to_token_index,
    compute_reflection_point,
    compute_reflection_point_char,
    compute_reflection_point_end,
    compute_reflection_point_tokens,
)

TEXT = "The quick brown fox jumps over the lazy dog. " * 50


class TestComputeReflectionPointChar:
    def test_deterministic_and_in_range(self):
        a = compute_reflection_point_char(TEXT, random.Random("s"))
        b = compute_reflection_point_char(TEXT, random.Random("s"))
        assert a == b
        assert 0 < a < len(TEXT)

    def test_respects_max_chars_cap(self):
        for i in range(50):
            rp = compute_reflection_point_char(TEXT, random.Random(i), max_chars=100)
            assert rp < 100

    def test_no_tokenization_needed(self):
        # Works on text the SmolLM2 tokenizer never sees (pure char space).
        rp = compute_reflection_point_char("x" * 200, random.Random(1))
        assert 0 < rp < 200


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


class TestComputeReflectionPointTokens:
    def test_char_and_token_consistent(self):
        """The returned char offset must be the start of the returned token."""
        offsets = _encode(TEXT).offsets

        for i in range(100):
            char_offset, tok_idx = compute_reflection_point_tokens(
                TEXT, random.Random(i)
            )
            assert offsets[tok_idx][0] == char_offset

    def test_matches_compute_reflection_point(self):
        """Same seed + same text + same max_tokens → same char offset in both funcs."""
        for i in range(100):
            rp_char = compute_reflection_point(TEXT, random.Random(i))
            rp_char2, _ = compute_reflection_point_tokens(TEXT, random.Random(i))
            assert rp_char == rp_char2

    def test_respects_max_tokens_cap(self):
        """tok_idx must be strictly less than max_tokens."""
        for i in range(200):
            _, tok_idx = compute_reflection_point_tokens(
                TEXT, random.Random(i), max_tokens=50
            )
            assert tok_idx < 50
            # Also strictly > 0 for non-degenerate texts (TEXT has many tokens)
            assert tok_idx >= 1

    def test_deterministic(self):
        a = compute_reflection_point_tokens(TEXT, random.Random("seed_a"))
        b = compute_reflection_point_tokens(TEXT, random.Random("seed_a"))
        assert a == b


class TestComputeReflectionPointEnd:
    def test_points_at_eos_slot_uncapped(self):
        """Without a cap, tok_idx is one past the last token (= EOS slot)."""
        offsets = _encode(TEXT).offsets
        for seed in ["a", "b", 0, 17]:
            char_offset, tok_idx = compute_reflection_point_end(
                TEXT, random.Random(seed)
            )
            assert tok_idx == len(offsets)
            # char_offset is the END of the last content token.
            assert char_offset == offsets[-1][1]

    def test_respects_max_tokens_cap(self):
        """When clamped, tok_idx == max_tokens (the EOS slot of the clip)."""
        char_offset, tok_idx = compute_reflection_point_end(
            TEXT, random.Random("x"), max_tokens=50
        )
        assert tok_idx == 50
        offsets = _encode(TEXT).offsets
        assert char_offset == offsets[49][1]

    def test_context_covers_all_content_tokens(self):
        """doc_text[:char_offset] must cover exactly n_tokens tokens."""
        char_offset, tok_idx = compute_reflection_point_end(
            TEXT, random.Random("x"), max_tokens=100
        )
        # Re-encode the prefix: should yield exactly tok_idx tokens.
        prefix = TEXT[:char_offset]
        assert len(_encode(prefix).offsets) == tok_idx

    def test_ignores_rng(self):
        """Different rng seeds must produce identical output — placement is deterministic."""
        a = compute_reflection_point_end(TEXT, random.Random("seed_a"))
        b = compute_reflection_point_end(TEXT, random.Random("seed_b"))
        c = compute_reflection_point_end(TEXT, random.Random(42))
        assert a == b == c

    def test_max_tokens_none(self):
        """max_tokens=None uses the full text length."""
        offsets = _encode(TEXT).offsets
        _, tok_idx = compute_reflection_point_end(
            TEXT, random.Random(0), max_tokens=None
        )
        assert tok_idx == len(offsets)

    def test_cap_larger_than_text(self):
        """When max_tokens exceeds the text's token count, use actual length (no IndexError)."""
        offsets = _encode(TEXT).offsets
        n_actual = len(offsets)
        _, tok_idx = compute_reflection_point_end(
            TEXT, random.Random(0), max_tokens=n_actual + 500
        )
        assert tok_idx == n_actual


class TestCharOffsetToTokenIndex:
    def test_exact_boundary_round_trips(self):
        """Tokenization char-offset → char_offset_to_token_index → same token."""
        offsets = _encode(TEXT).offsets
        # Pick several token boundaries; each should round-trip exactly.
        for k in [1, 5, 20, 50, len(offsets) - 1]:
            if k >= len(offsets):
                continue
            char_offset = offsets[k][0]
            assert char_offset_to_token_index(TEXT, char_offset) == k

    def test_inside_token_rounds_down(self):
        """A char offset inside a token's span maps to that token's index."""
        offsets = _encode(TEXT).offsets
        # Pick a token whose span is > 1 char
        for k, (start, end) in enumerate(offsets):
            if end - start > 1 and k > 0:
                mid = start + 1  # strictly inside span
                assert char_offset_to_token_index(TEXT, mid) == k
                break

    def test_reflection_point_round_trip(self):
        """compute_reflection_point_tokens → char_offset_to_token_index gives same tok_idx."""
        for i in range(50):
            char_offset, tok_idx = compute_reflection_point_tokens(
                TEXT, random.Random(i)
            )
            assert char_offset_to_token_index(TEXT, char_offset) == tok_idx
