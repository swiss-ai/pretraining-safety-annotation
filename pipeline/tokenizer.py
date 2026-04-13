"""Shared tokenizer utilities: truncation and reflection point computation."""

import random

from transformers import AutoTokenizer

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
_tokenizer = None


def _get_tokenizer() -> AutoTokenizer:
    """Return the shared SmolLM2 tokenizer (lazy singleton)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


def truncate_to_max_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens using the SmolLM2 tokenizer.

    Tokenizes the text, keeps the first max_tokens token IDs, and decodes back.
    """
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    truncated_ids = token_ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)


def count_tokens(text: str) -> int:
    """Count tokens in *text* using the shared SmolLM2 tokenizer."""
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def compute_reflection_point(text: str, rng: random.Random) -> int:
    """Pick a reflection point snapped to a token boundary.

    Samples from a piecewise distribution: linear ramp from 0% to 20% of the
    token sequence (probability rises from 0 to a constant), then uniform from
    20% to 100%.  Uses inverse-CDF sampling.
    """
    tokenizer = _get_tokenizer()
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    n_tokens = len(offsets)
    assert n_tokens > 0, "Text produced no tokens"

    # Inverse-CDF sampling for piecewise linear-ramp + uniform distribution.
    # CDF boundary at t=0.2 is u=1/9.
    u = rng.random()
    if u <= 1.0 / 9.0:
        t = 0.6 * u**0.5
    else:
        t = 0.2 + 0.9 * (u - 1.0 / 9.0)

    tok_idx = max(1, min(int(t * n_tokens), n_tokens - 1))
    return offsets[tok_idx][0]
