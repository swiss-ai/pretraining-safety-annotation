"""Shared API utilities for the SwissAI inference endpoint.

Provides async API calls with retry (jittered backoff, rate-limit class
distinction), concurrent batch execution with tqdm throttling, JSON
extraction, and health checking. Used by charter.improve and charter.eval runners.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys

import json_repair
import openai
from tqdm.asyncio import tqdm_asyncio

from pipeline.log import logger

MAX_RETRIES = 5
MAX_RETRIES_RATE_LIMIT = 8
RETRY_BACKOFF_BASE = 2.0

# Default desired completion-token budget when a model does not set
# ``completion_max_tokens`` in config. Callers may dynamically clamp it against
# a model-specific total context window before issuing the request. When unset
# at the HTTP layer, some providers clamp to ~128 tokens, so the budget is
# always passed explicitly from call sites.
DEFAULT_MAX_TOKENS = 32000

# Per-model recommended sampling parameters from HuggingFace model cards.
# Matched case-insensitively against the model name. First match wins.
_SAMPLING_DEFAULTS: list[tuple[str, dict[str, float | int]]] = [
    # Qwen3.5 thinking: t=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
    (
        "qwen3.5",
        {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
    ),
    # Qwen3.6 thinking: model generation_config default = t=1.0, top_p=0.95, top_k=20,
    # pp=0.0. A charter-eval A/B (200 items x2 benches, GLM-5.1 judge) showed
    # presence_penalty=1.5 LOWERED quality (esp. charter_grounding) vs pp=0.0, so we keep
    # the model default. MUST precede "qwen3" so the qwen3.6 substring wins over the t=0.6 entry.
    (
        "qwen3.6",
        {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
    ),
    # Qwen3 thinking: t=0.6, top_p=0.95, top_k=20
    ("qwen3", {"temperature": 0.6, "top_p": 0.95, "top_k": 20}),
    # Gemma-4 (4-31b, 4-26b-a4b): model generation_config default = t=1.0, top_p=0.95, top_k=64
    ("gemma", {"temperature": 1.0, "top_p": 0.95, "top_k": 64}),
    # SmolLM3: t=0.6, top_p=0.95
    ("smollm3", {"temperature": 0.6, "top_p": 0.95}),
    # Kimi: t=0.6
    ("kimi", {"temperature": 0.6}),
    # GLM-4 family (4.5-Air, 4.7-Flash, etc.): t=1.0, top_p=0.95
    ("glm-4", {"temperature": 1.0, "top_p": 0.95}),
    # Nemotron: t=1.0, top_p=0.95
    ("nemotron", {"temperature": 1.0, "top_p": 0.95}),
]


def resolve_sampling_params(*names: str) -> dict[str, float | int]:
    """Look up recommended sampling params for a model.

    Matches *names* (model name, alias, etc.) case-insensitively against
    known model families.  Returns the first match, or ``{}`` if none.
    """
    for name in names:
        lower = name.lower()
        for pattern, params in _SAMPLING_DEFAULTS:
            if pattern in lower:
                return dict(params)
    return {}


def make_api_client(
    endpoint: str,
    max_concurrent: int,
    api_keys: dict[str, str] | None = None,
) -> tuple[openai.AsyncOpenAI, asyncio.Semaphore]:
    """Create an OpenAI client and concurrency semaphore.

    Args:
        endpoint: API base URL.
        max_concurrent: Maximum number of concurrent API calls.
        api_keys: Mapping of endpoint URL → environment variable name holding
            the API key.  Falls back to SWISS_AI_API_KEY when the endpoint
            is not in the mapping (or mapping is None).
    """
    env_var = (api_keys or {}).get(endpoint, "SWISS_AI_API_KEY")
    api_key = os.environ.get(env_var)
    assert api_key, f"{env_var} not set in environment (needed for {endpoint})"
    client = openai.AsyncOpenAI(api_key=api_key, base_url=endpoint)
    semaphore = asyncio.Semaphore(max_concurrent)
    return client, semaphore


def run_concurrent(*coros, desc: str) -> list:
    """Run async coroutines concurrently with a tqdm progress bar.

    Creates a temporary event loop that doesn't touch SIGINT handling,
    so Ctrl+C raises KeyboardInterrupt normally.

    Under a non-tty stderr (e.g. nohup, log file redirection), throttles
    tqdm's progress prints so multi-hour runs don't fill the log file with
    hundreds of MB of progress lines.
    """
    tqdm_kwargs: dict = {"desc": desc}
    if not sys.stderr.isatty():
        tqdm_kwargs["mininterval"] = 30
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(tqdm_asyncio.gather(*coros, **tqdm_kwargs))
    finally:
        loop.close()


async def api_call(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    semaphore: asyncio.Semaphore,
    thinking: bool = False,
    json_mode: bool = False,
    sampling_params: dict[str, float | int] | None = None,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
) -> tuple[str, str | None, dict]:
    """Make a single API call with network-error retry.

    Returns (content, reasoning_content, usage_dict). reasoning_content is None
    if the model does not produce reasoning output. usage_dict contains
    input_tokens, output_tokens, reasoning_tokens.

    When *max_tokens* is set, it is passed explicitly so OpenRouter doesn't
    silently clamp the output to a provider-specific default.  When *None*,
    the parameter is omitted and the server uses all available context.
    """
    extra_body = None
    if thinking:
        extra_body = {
            "separate_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }

    response_format = None
    if json_mode:
        response_format = {"type": "json_object"}

    # Sampling params: temperature, top_p, presence_penalty are native OpenAI
    # API kwargs; top_k goes into extra_body (sglang/vllm extension).
    sp = sampling_params or {}
    api_kwargs: dict = {}
    if max_tokens is not None:
        api_kwargs["max_tokens"] = max_tokens
    for k in ("temperature", "top_p", "presence_penalty"):
        if k in sp:
            api_kwargs[k] = sp[k]
    if "top_k" in sp:
        extra_body = extra_body or {}
        extra_body["top_k"] = sp["top_k"]

    last_error = None
    attempt = 0
    max_attempts = MAX_RETRIES_RATE_LIMIT  # upper bound; per-error class checked below
    while attempt < max_attempts:
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body=extra_body,
                    **api_kwargs,
                    **({"response_format": response_format} if response_format else {}),
                )
            # OpenRouter returns HTTP 200 with choices=None and a top-level
            # `error` field when the upstream provider fails. The OpenAI SDK
            # parses that into a ChatCompletion with choices=None, so guard
            # explicitly before indexing.
            if not getattr(response, "choices", None):
                err = getattr(response, "error", None) or getattr(
                    response, "model_extra", {}
                ).get("error")
                raise AssertionError(
                    f"API returned no choices (upstream error: {err!r})"
                )
            choice = response.choices[0]
            msg = choice.message
            content = msg.content
            assert content is not None, "API returned None content"
            assert content.strip(), "API returned empty content"
            # Warn on truncation so the downstream parse-failure warning
            # points at the right root cause (hit max_tokens) instead of
            # looking like a model-side JSON bug.
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                logger.warning(
                    "Output truncated at max_tokens={} (model={}): {} chars",
                    max_tokens,
                    model,
                    len(content),
                )
                raise AssertionError(
                    f"Output truncated (finish_reason=length, {len(content)} chars)"
                )
            # Guard against upstream providers that truncate but report
            # finish_reason="stop". A valid combined-judge response (4 voices
            # with scores + reasoning) is always >800 chars. Short responses
            # with an unclosed JSON brace are almost certainly truncated.
            if len(content) < 800 and content.count("{") > content.count("}"):
                logger.warning(
                    "Likely truncated response ({}chars, unbalanced braces, "
                    "finish_reason={}), retrying",
                    len(content),
                    finish_reason,
                )
                raise AssertionError(
                    f"Likely truncated ({len(content)} chars, unbalanced braces)"
                )
            reasoning = getattr(msg, "reasoning_content", None)
            usage = response.usage
            details = getattr(usage, "completion_tokens_details", None) or {}
            if isinstance(details, dict):
                detail_reasoning = details.get("reasoning_tokens", 0) or 0
            else:
                detail_reasoning = getattr(details, "reasoning_tokens", 0) or 0
            usage_dict = {
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "reasoning_tokens": getattr(usage, "reasoning_tokens", 0)
                or detail_reasoning,
            }
            return content.strip(), reasoning, usage_dict
        except (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
            json.JSONDecodeError,
            AssertionError,
        ) as e:
            last_error = f"{type(e).__name__}: {e}"
            # Rate-limit and timeout errors get more retries because the
            # cluster is under transient load and we want to wait it out.
            # Connection errors and other failures get the standard retry count.
            if isinstance(e, (openai.RateLimitError, openai.APITimeoutError)):
                limit_for_this_error = MAX_RETRIES_RATE_LIMIT
            else:
                limit_for_this_error = MAX_RETRIES
            attempt += 1
            if attempt >= limit_for_this_error:
                break
            logger.warning(
                "Retry {}/{} due to: {}",
                attempt + 1,
                limit_for_this_error,
                last_error,
            )
            # Jittered exponential backoff: each sleep is in
            # [0.5 * 2**i, 1.5 * 2**i] where i is the 0-indexed attempt that
            # just failed. The jitter avoids thundering-herd retries from many
            # concurrent requests waking up at the same instant.
            base = RETRY_BACKOFF_BASE ** (attempt - 1)
            await asyncio.sleep(base * random.uniform(0.5, 1.5))
    raise RuntimeError(f"Failed after {attempt} retries: {last_error}")


def health_check(client: openai.AsyncOpenAI, model: str) -> None:
    """Ping the API with a lightweight request. Fail fast if model unavailable."""

    async def _check():
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=16,
        )
        assert response.choices, f"No choices returned for model={model}"

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_check())
        logger.info("Health check passed: model={}", model)
    except Exception as e:
        raise RuntimeError(f"Health check failed for model={model}: {e}") from e
    finally:
        loop.close()


def extract_json(raw: str) -> dict:
    """Extract a JSON object from a model response.

    Tries multiple strategies in order:
    0. Strip <think>...</think> blocks (thinking model output)
    1. Direct parse (response is pure JSON)
    2. Strip thinking prefix (unseparated reasoning before JSON)
    3. Strip leading code fence (```json ... ```)
    4. Find a fenced JSON block anywhere in the response
    5. Find the LAST balanced { ... } via brace counting (skips thinking)
    6. Fallback: find the FIRST balanced { ... }
    """
    text = raw.strip()

    # 0. Strip <think>...</think> blocks from thinking models
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :].strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip thinking prefix (models that leak thinking into content)
    stripped = _strip_thinking(text)
    if stripped != text:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # 3. Leading code fence
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # skip ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        try:
            return json.loads("\n".join(lines))
        except json.JSONDecodeError:
            pass

    # 3. Fenced JSON block — try last match first (thinking models put templates early)
    fence_matches = list(re.finditer(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL))
    for match in reversed(fence_matches):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue

    # 3b. Unfenced JSON after code fences (thinking models put real JSON after examples)
    if fence_matches:
        after_last = text[fence_matches[-1].end() :].strip()
        if after_last:
            try:
                return json.loads(after_last)
            except json.JSONDecodeError:
                pass

    # 5. FIRST balanced { ... } (finds outermost object, handles nested JSON)
    result = _find_first_json_object(text)
    if result is not None:
        return result

    # 6. Fallback: LAST balanced { ... } (catches answer after invalid prefix)
    result = _find_last_json_object(text)
    if result is not None:
        return result

    # 7. Repair invalid backslash escapes and retry (models occasionally emit
    # raw LaTeX like \frac, \left, \pi inside string values without escaping
    # the backslash; \l in particular is not a valid JSON escape).
    repaired = _repair_invalid_backslash_escapes(text)
    if repaired != text:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        result = _find_first_json_object(repaired) or _find_last_json_object(repaired)
        if result is not None:
            return result

    # 8. Last resort: a lenient repair pass (json_repair). Handles malformations
    # the strict strategies above can't — an unescaped ASCII quote closing a
    # typographic quote in non-English text (e.g. German „…"), a stray duplicated
    # brace that splits the object, trailing fences. Only reached once every
    # strict strategy has failed, so well-formed responses are never altered.
    repaired_obj = _repair_with_json_repair(text)
    if repaired_obj is not None:
        logger.warning("extract_json: recovered malformed JSON via json_repair fallback")
        return repaired_obj

    raise json.JSONDecodeError(
        f"No valid JSON object found in response ({len(raw)} chars)",
        raw[:200],
        0,
    )


def _repair_with_json_repair(text: str) -> dict | None:
    """Lenient last-resort parse via the ``json_repair`` library.

    Returns a non-empty ``dict`` on success, else ``None`` (so the caller falls
    through to a hard failure → retry). Field-presence/leakage validation stays
    in ``parse_generation``; this only recovers a structurally-broken object.
    """
    obj = json_repair.loads(text)
    if isinstance(obj, list):
        obj = next((o for o in obj if isinstance(o, dict) and o), None)
    return obj if isinstance(obj, dict) and obj else None


_INVALID_BACKSLASH_ESCAPE_RE = re.compile(r'\\(?!["\\/]|u[0-9a-fA-F]{4})')


def _repair_invalid_backslash_escapes(text: str) -> str:
    """Double any ``\\X`` where ``X`` is not a structural JSON escape character.

    LaTeX in summaries (e.g. ``\\frac``, ``\\left``, ``\\pi``) often makes
    ``json.loads`` fail (``\\l`` is not a valid escape) and even when it doesn't
    (``\\f`` is form-feed), the resulting character is wrong — the model meant
    LaTeX text, not a control character. This pass leaves only the structural
    escapes (``\\"``, ``\\\\``, ``\\/``, ``\\uXXXX``) intact and doubles the rest,
    preserving LaTeX intent at the cost of any actual control characters the
    model may have meant — a worthwhile tradeoff for prose summaries.
    """
    return _INVALID_BACKSLASH_ESCAPE_RE.sub(r'\\\\', text)


def _strip_thinking(text: str) -> str:
    """Strip ``<think>...</think>`` blocks from model output.

    Only handles explicit think tags. Other thinking formats (e.g.
    "Thinking Process:" prose) are handled by strategies 5/6 which
    find JSON objects by brace matching.
    """
    stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return stripped.strip() if stripped != text else text


def _find_json_object_at(text: str, start: int) -> dict | None:
    """Try to extract a balanced JSON object starting at position start."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _find_last_json_object(text: str) -> dict | None:
    """Find the last balanced JSON object in text."""
    # Search for { positions from the end
    pos = len(text)
    while True:
        pos = text.rfind("{", 0, pos)
        if pos == -1:
            return None
        result = _find_json_object_at(text, pos)
        if result is not None:
            return result
        # Try earlier positions


def _find_first_json_object(text: str) -> dict | None:
    """Find the first balanced JSON object in text."""
    start = text.find("{")
    if start == -1:
        return None
    return _find_json_object_at(text, start)
