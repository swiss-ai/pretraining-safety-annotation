"""Shared tokenizer utilities: reflection-point sampling and truncation.

The training-side binary dataset (``annotated.bin``) was produced by the
Rust ``tokenizers`` library with an EOS post-processor; see
``preprocessing/tokenization/`` and the ``feedback_tokenizer_setup``
memory note.  ``transformers.AutoTokenizer`` tokenizes some sequences
differently (notably ``\\n\\n`` splits into two tokens instead of a
single merged token), so every index-space computation in this file
must use the Rust library directly — otherwise token indices will not
align with ``annotated.bin``.
"""

import random

from tokenizers import Tokenizer

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

_tokenizer: Tokenizer | None = None


def _get_tokenizer() -> Tokenizer:
    """Return the shared Rust SmolLM2 tokenizer (lazy singleton).

    No post-processor, no truncation — callers that need a specific
    encoding policy should apply it themselves.  The base tokenizer
    produces the same content token IDs as the training-side pipeline.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


# Apertus tokenizer for the production reflection-point cutoff: documents are
# tokenized with the Apertus-70B model's own tokenizer and the reflection is
# placed at min(doc_len, REFLECTION_MAX_TOKENS) tokens.
APERTUS_TOKENIZER_NAME = "swiss-ai/Apertus-70B-2509"
REFLECTION_MAX_TOKENS = 3800

# Version marker for the reflection-point *policy* used to materialize eval
# item pools. Bump this whenever the placement rule changes (tokenizer,
# cut-off, sampling vs deterministic) so cached pools built under an old
# policy fail loudly on resume instead of being silently reused. Currently:
# deterministic Apertus-tokenizer cut-off at min(doc_tokens, 3800).
REFLECTION_POLICY = "apertus-min-3800-v1"

_apertus_tokenizer: Tokenizer | None = None


def _get_apertus_tokenizer() -> Tokenizer:
    """Return the shared Apertus tokenizer (lazy singleton)."""
    global _apertus_tokenizer
    if _apertus_tokenizer is None:
        _apertus_tokenizer = Tokenizer.from_pretrained(APERTUS_TOKENIZER_NAME)
    return _apertus_tokenizer


def compute_reflection_point_apertus(
    text: str, max_tokens: int = REFLECTION_MAX_TOKENS
) -> int:
    """Reflection point as a char offset, in Apertus-tokenizer space.

    Deterministic (no sampling): the reflection is placed after the first
    ``min(n_tokens, max_tokens)`` Apertus tokens, so ``text[:offset]`` is the
    whole document when it fits in ``max_tokens``, otherwise its first
    ``max_tokens`` tokens.
    """
    enc = _get_apertus_tokenizer().encode(text, add_special_tokens=False)
    n = len(enc.ids)
    assert n > 0, "Text produced no tokens"
    if n <= max_tokens:
        return len(text)
    return enc.offsets[max_tokens][0]


def _encode(text: str):
    """Encode *text* without special tokens and return the ``Encoding``."""
    return _get_tokenizer().encode(text, add_special_tokens=False)


def truncate_to_max_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* Rust tokens (content-only)."""
    tokenizer = _get_tokenizer()
    enc = tokenizer.encode(text, add_special_tokens=False)
    if len(enc.ids) <= max_tokens:
        return text
    return tokenizer.decode(enc.ids[:max_tokens], skip_special_tokens=True)


def count_tokens(text: str) -> int:
    """Count Rust tokens in *text* (content-only)."""
    return len(_encode(text).ids)


def truncate_and_count(text: str, max_tokens: int) -> tuple[str, int]:
    """Truncate to at most *max_tokens* tokens; return (text, n_tokens) in one encode."""
    tokenizer = _get_tokenizer()
    enc = tokenizer.encode(text, add_special_tokens=False)
    ids = enc.ids[:max_tokens]
    if len(enc.ids) <= max_tokens:
        return text, len(ids)
    return tokenizer.decode(ids, skip_special_tokens=True), len(ids)


def _sample_tok_idx(n_tokens: int, rng: random.Random) -> int:
    """Sample a token index from the reflection-point distribution.

    Piecewise: linear ramp from 0% to 20% of the sequence (probability
    rises from 0 to a constant), then uniform from 20% to 100%.  Inverse-CDF
    sampling, CDF boundary at t=0.2 is u=1/9.
    """
    u = rng.random()
    if u <= 1.0 / 9.0:
        t = 0.6 * u**0.5
    else:
        t = 0.2 + 0.9 * (u - 1.0 / 9.0)
    return max(1, min(int(t * n_tokens), n_tokens - 1)) if n_tokens > 1 else 0


def compute_reflection_point_char(
    text: str, rng: random.Random, max_chars: int | None = None
) -> int:
    """Sample a reflection point as a character offset — no tokenization.

    Uses the same piecewise distribution as the token sampler but over
    characters: the eligible region is the first *max_chars* characters (or the
    whole text). Returns the character offset, so ``text[:offset]`` is the
    context before the reflection point.
    """
    n = len(text)
    assert n > 0, "Text is empty"
    if max_chars is not None:
        n = min(n, max_chars)
    return _sample_tok_idx(n, rng)


def compute_reflection_point(text: str, rng: random.Random, max_tokens: int | None = None) -> int:
    """Pick a reflection point snapped to a token boundary.

    Returns the character offset of the chosen token's start.  When
    *max_tokens* is set, only the first *max_tokens* tokens are considered.
    """
    offsets = _encode(text).offsets
    n_tokens = len(offsets)
    assert n_tokens > 0, "Text produced no tokens"
    if max_tokens is not None:
        n_tokens = min(n_tokens, max_tokens)
    tok_idx = _sample_tok_idx(n_tokens, rng)
    return offsets[tok_idx][0]


def compute_reflection_point_tokens(
    text: str, rng: random.Random, max_tokens: int | None = None
) -> tuple[int, int]:
    """Pick a reflection point; return both its char offset and token index.

    The token index aligns with ``annotated.bin``'s content tokens so
    consumers can index that file directly without retokenizing.
    """
    offsets = _encode(text).offsets
    n_tokens = len(offsets)
    assert n_tokens > 0, "Text produced no tokens"
    if max_tokens is not None:
        n_tokens = min(n_tokens, max_tokens)
    tok_idx = _sample_tok_idx(n_tokens, rng)
    return offsets[tok_idx][0], tok_idx


def compute_reflection_point_end(
    text: str, rng: random.Random, max_tokens: int | None = None
) -> tuple[int, int]:
    """Place the reflection at the EOS slot of the (clipped) binary.

    Matches ``compute_reflection_point_tokens``'s signature so it is a
    drop-in selector.  ``rng`` is accepted but unused — placement is
    deterministic per document.

    Returns ``(char_offset, tok_idx)`` with ``tok_idx == min(n_tokens,
    max_tokens)`` — one past the last content token, i.e. co-located with
    ``annotated.bin``'s EOS position (``scripts/backfill_reflection_token_index.py``
    documents this as the "right before EOS" case).  ``char_offset`` is
    the END of the last content token, so ``doc_text[:char_offset]``
    covers all content tokens in the clip and the LLM reflects on the
    full text.
    """
    del rng  # unused; placement is deterministic
    offsets = _encode(text).offsets
    n_tokens = len(offsets)
    assert n_tokens > 0, "Text produced no tokens"
    if max_tokens is not None:
        n_tokens = min(n_tokens, max_tokens)
    # End of the last content token = start of the EOS slot.
    rp_char = offsets[n_tokens - 1][1]
    tok_idx = n_tokens
    return rp_char, tok_idx


def char_offset_to_token_index(text: str, char_offset: int) -> int:
    """Map a character offset to the Rust-tokenizer token index.

    Returns the smallest token index ``K`` such that
    ``tokens[0:K]`` covers ``text[:char_offset]`` as closely as possible:

    * If *char_offset* is exactly a token boundary, returns that token's
      index (so ``tokens[0:K]`` covers exactly ``text[:char_offset]``).
    * Otherwise *char_offset* falls inside some merged token's span
      (e.g. a ``\\n\\n`` token whose Rust span is 2 chars while
      transformers would have split it in two).  Rounds DOWN: returns
      the index of that merged token, so ``tokens[0:K]`` covers
      ``text[:offsets[K][0]]`` which is a prefix of ``text[:char_offset]``.

    Used to backfill ``reflection_token_index`` from a stored
    ``reflection_position`` (char offset) whose tokenization lineage may
    not perfectly match the current Rust tokenization.
    """
    offsets = _encode(text).offsets
    for k, (start, _end) in enumerate(offsets):
        if start == char_offset:
            return k
        if start > char_offset:
            return k - 1 if k > 0 else 0
    return len(offsets)
