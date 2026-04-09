"""Shared API utilities for the SwissAI inference endpoint.

Provides async API calls with retry, concurrent batch execution,
JSON extraction, and health checking. Used by phase2, phase3,
and the summary pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
import re

import openai
from tqdm.asyncio import tqdm_asyncio

from pipeline.log import logger

MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0

# Default completion-token budget for chat calls. Must be large enough for:
#   - a combined 4-voice judge response (~500 output tokens observed),
#   - a full generation response with both voices (~600 output tokens),
#   - thinking-model reasoning blocks on top of either of the above.
# When unset, OpenRouter has been observed routing to providers that cap at
# ~128 tokens, silently truncating judge output to ~500 chars. Always pass
# an explicit budget.
DEFAULT_MAX_TOKENS = 8192

# Per-model recommended sampling parameters from HuggingFace model cards.
# Matched case-insensitively against the model name. First match wins.
_SAMPLING_DEFAULTS: list[tuple[str, dict[str, float | int]]] = [
    # Qwen3.5 thinking: t=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
    (
        "qwen3.5",
        {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
    ),
    # Qwen3 thinking: t=0.6, top_p=0.95, top_k=20
    ("qwen3", {"temperature": 0.6, "top_p": 0.95, "top_k": 20}),
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
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(tqdm_asyncio.gather(*coros, desc=desc))
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
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, str | None, dict]:
    """Make a single API call with network-error retry.

    Returns (content, reasoning_content, usage_dict). reasoning_content is None
    if the model does not produce reasoning output. usage_dict contains
    input_tokens, output_tokens, reasoning_tokens.

    Passes an explicit max_tokens budget so OpenRouter doesn't silently clamp
    the output to a provider-specific default (observed: ~128 tokens, i.e.
    ~500 chars, which truncates the 4-voice judge response mid-reasoning).
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
    api_kwargs: dict = {"max_tokens": max_tokens}
    for k in ("temperature", "top_p", "presence_penalty"):
        if k in sp:
            api_kwargs[k] = sp[k]
    if "top_k" in sp:
        extra_body = extra_body or {}
        extra_body["top_k"] = sp["top_k"]

    last_error = None
    for attempt in range(MAX_RETRIES):
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
            AssertionError,
        ) as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < MAX_RETRIES - 1:
            logger.warning(
                "Retry {}/{} due to: {}", attempt + 2, MAX_RETRIES, last_error
            )
            await asyncio.sleep(RETRY_BACKOFF_BASE**attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_error}")


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
    2. Strip leading code fence (```json ... ```)
    3. Find a fenced JSON block anywhere in the response
    4. Find the first { and its matching } via brace counting
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

    # 2. Leading code fence
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

    # 4. First { to matching } via brace counting
    start = text.find("{")
    if start != -1:
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
                        break

    raise json.JSONDecodeError(
        f"No valid JSON object found in response ({len(raw)} chars)",
        raw[:200],
        0,
    )
