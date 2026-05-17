"""Run definitions for charter.scale generation.

Each run specifies what columns to produce, how to build API call messages,
and how to post-process the parsed responses into the final output row.
The AnnotationGenerator is generic — it receives a RunDefinition and
executes it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pipeline.config import extract_charter_elements
from pipeline.generation import (
    PREFLECTION_FIELDS_CURRENT,
    PREFLECTION_TASK,
    REFLECTION_TASK,
    parse_generation,
)
from pipeline.charter.scale.canaries import assign_canary
from pipeline.tokenizer import (
    compute_reflection_point_end,
    compute_reflection_point_tokens,
    truncate_and_count,
)

SelectorFn = Callable[..., tuple[int, int]]


@dataclass
class RunDefinition:
    """Defines a generation run: what to generate and how to parse it."""

    name: str
    prompt_type: str  # "reflection", "preflection", or "summary" — selects the prompt file
    output_columns: list[str]
    build_calls: Callable
    # (doc_text, doc_id, system_prompt, canaries, canary_seed,
    #  reflection_seed) -> list[(messages, required_output_fields)]
    post_process: Callable
    # (doc_id, doc_text, parsed_results_per_call, reflection_point,
    #  canary) -> dict of column values
    # Optional: override the default per-alias prompt directory. None = use
    # FINAL_PROMPTS_DIR / generator_alias / prompt_filename (reflections,
    # preflections); set to a fixed Path for in-tree, model-agnostic prompts
    # (summaries).
    prompt_source_dir: Path | None = None


# ---------------------------------------------------------------------------
# reflections run  (partial text up to reflection point)
# ---------------------------------------------------------------------------
# The two reflection runs ("reflections" and "reflection_end") share message
# construction, canary assignment, and parsing.  They differ in
#   (a) which selector chooses the reflection point
#   (b) the names of the output columns (so both can coexist in the merged
#       sidecar as a paired ablation).

_REFLECTIONS_COLS_DEFAULT: dict[str, str] = {k: k for k in (
    "reflection_1p",
    "reflection_3p",
    "reflection_position",
    "reflection_token_index",
    "charter_reflection",
    "canary_type",
)}

_REFLECTIONS_COLS_END: dict[str, str] = {
    "reflection_1p":          "reflection_end_1p",
    "reflection_3p":          "reflection_end_3p",
    "reflection_position":    "reflection_end_position",
    "reflection_token_index": "reflection_end_token_index",
    "charter_reflection":     "charter_reflection_end",
    "canary_type":            "canary_type_end",
}

_REFLECTIONS_COLUMNS = list(_REFLECTIONS_COLS_DEFAULT.values())
_REFLECTION_END_COLUMNS = list(_REFLECTIONS_COLS_END.values())


def _build_reflection_calls(
    doc_text: str,
    doc_id: str,
    system_prompt: str,
    canaries: list[dict],
    canary_seed: int,
    reflection_seed: int,
    max_text_tokens: int,
    *,
    selector: SelectorFn,
) -> list[tuple[list[dict], set[str], dict]]:
    """Build a single API call for a reflection-style run.

    ``max_text_tokens`` is the per-doc token cap — pass the sidecar's
    ``token_length`` so sampled token indices are guaranteed to fall inside
    the content portion of ``annotated.bin`` (strictly < token_length,
    excluding the appended EOS).
    """
    rp_rng = random.Random(f"{reflection_seed}_{doc_id}")
    rp_char, rp_tok = selector(doc_text, rp_rng, max_tokens=max_text_tokens)

    context_before = doc_text[:rp_char]

    canary = assign_canary(doc_id, canary_seed, canaries)

    refl_user = f"## Full Text\n\n{context_before}"
    if canary is not None:
        refl_user += (
            f"\n\n## Canary Injection\n\n"
            f"This sample has a canary injection. Apply to BOTH reflections.\n"
            f"- For reflection_1p: {canary['instruction']}\n"
            f"- For reflection_3p: {canary['instruction_3p']}"
        )
    refl_user += REFLECTION_TASK

    refl_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": refl_user},
    ]

    meta = {
        "reflection_point": rp_char,
        "reflection_token_index": rp_tok,
        "canary": canary,
    }

    return [
        (refl_messages, {"analysis", "reflection_1p", "reflection_3p"}, meta),
    ]


def _make_reflection_post_process(columns: dict[str, str]) -> Callable:
    """Factory: return a post_process that writes into ``columns``-named keys."""

    def _post_process(
        doc_id: str,
        doc_text: str,
        parsed_results: list[dict],
        meta: dict,
    ) -> dict:
        (refl_parsed,) = parsed_results

        charter_reflection = extract_charter_elements(
            (refl_parsed.get("reflection_1p") or "")
            + " "
            + (refl_parsed.get("reflection_3p") or "")
        )

        canary = meta["canary"]

        return {
            columns["reflection_1p"]:          refl_parsed.get("reflection_1p") or "",
            columns["reflection_3p"]:          refl_parsed.get("reflection_3p") or "",
            columns["reflection_position"]:    meta["reflection_point"],
            columns["reflection_token_index"]: meta["reflection_token_index"],
            columns["charter_reflection"]:     json.dumps(charter_reflection),
            columns["canary_type"]:            canary["id"] if canary is not None else None,
        }

    return _post_process


def _reflections_build_calls(
    doc_text: str,
    doc_id: str,
    system_prompt: str,
    canaries: list[dict],
    canary_seed: int,
    reflection_seed: int,
    max_text_tokens: int = 1920,
) -> list[tuple[list[dict], set[str], dict]]:
    return _build_reflection_calls(
        doc_text=doc_text,
        doc_id=doc_id,
        system_prompt=system_prompt,
        canaries=canaries,
        canary_seed=canary_seed,
        reflection_seed=reflection_seed,
        max_text_tokens=max_text_tokens,
        selector=compute_reflection_point_tokens,
    )


def _reflection_end_build_calls(
    doc_text: str,
    doc_id: str,
    system_prompt: str,
    canaries: list[dict],
    canary_seed: int,
    reflection_seed: int,
    max_text_tokens: int = 1920,
) -> list[tuple[list[dict], set[str], dict]]:
    return _build_reflection_calls(
        doc_text=doc_text,
        doc_id=doc_id,
        system_prompt=system_prompt,
        canaries=canaries,
        canary_seed=canary_seed,
        reflection_seed=reflection_seed,
        max_text_tokens=max_text_tokens,
        selector=compute_reflection_point_end,
    )


_reflections_post_process = _make_reflection_post_process(_REFLECTIONS_COLS_DEFAULT)
_reflection_end_post_process = _make_reflection_post_process(_REFLECTIONS_COLS_END)


# ---------------------------------------------------------------------------
# preflections run  (full text)
# ---------------------------------------------------------------------------

_PREFLECTION_FIELDS = PREFLECTION_FIELDS_CURRENT
_PREFLECTIONS_COLUMNS = list(_PREFLECTION_FIELDS) + ["charter_preflection"]


def _preflections_build_calls(
    doc_text: str,
    doc_id: str,
    system_prompt: str,
    canaries: list[dict],
    canary_seed: int,
    reflection_seed: int,
    max_text_tokens: int = 1920,
) -> list[tuple[list[dict], set[str], dict]]:
    """Build a single API call for the preflections run.

    Returns list of (messages, required_fields, metadata) tuples.
    The system_prompt is the already-resolved preflection-specific prompt.
    """
    # Sidecar text is the full raw doc (can be 10-20K tokens), while
    # annotated.bin tokenizes only the first max_text_tokens.  Send the
    # same clipped span so the preflection context matches what the model
    # will later see at train time, and so the input fits under sglang's
    # context limit.
    end_char, _ = compute_reflection_point_end(
        doc_text, random.Random(), max_tokens=max_text_tokens
    )
    clipped_text = doc_text[:end_char]

    prefl_user = f"## Full Text\n\n{clipped_text}"
    prefl_user += PREFLECTION_TASK

    prefl_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prefl_user},
    ]

    meta: dict = {}

    return [
        (prefl_messages, {"analysis", *_PREFLECTION_FIELDS}, meta),
    ]


def _preflections_post_process(
    doc_id: str,
    doc_text: str,
    parsed_results: list[dict],
    meta: dict,
) -> dict:
    """Extract the 4 preflection fields from the single parsed result."""
    (prefl_parsed,) = parsed_results

    charter_preflection = extract_charter_elements(
        " ".join((prefl_parsed.get(f) or "") for f in _PREFLECTION_FIELDS)
    )

    return {
        **{f: (prefl_parsed.get(f) or "") for f in _PREFLECTION_FIELDS},
        "charter_preflection": json.dumps(charter_preflection),
    }


# ---------------------------------------------------------------------------
# summaries run  (full text, single 2-4 sentence summary, 128-token cap)
# ---------------------------------------------------------------------------

_SUMMARIES_COLUMNS = ["summary", "summary_token_count"]
_SUMMARY_TOKEN_BUDGET = 128
_SUMMARIES_PROMPT_DIR = Path(__file__).resolve().parents[2] / "summaries" / "prompts"


def _summaries_build_calls(
    doc_text: str,
    doc_id: str,
    system_prompt: str,
    canaries: list[dict],
    canary_seed: int,
    reflection_seed: int,
    max_text_tokens: int = 1920,
) -> list[tuple[list[dict], set[str], dict]]:
    """Single API call: model summarises the clipped doc.

    Canaries and seeds are accepted for interface parity with the other
    runs but deliberately unused — the baseline annotation track is the
    un-charter-cited control, so canary injection would defeat its
    purpose. The clip end is computed via ``compute_reflection_point_end``
    so the summary covers exactly the same span the tokenizer will see at
    train time; the function's ``rng`` arg is unused and deterministic.
    """
    end_char, _ = compute_reflection_point_end(
        doc_text, random.Random(), max_tokens=max_text_tokens
    )
    clipped_text = doc_text[:end_char]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"## Text\n\n{clipped_text}"},
    ]

    return [(messages, {"summary"}, {})]


def _summaries_post_process(
    doc_id: str,
    doc_text: str,
    parsed_results: list[dict],
    meta: dict,
) -> dict:
    """Truncate the model summary to 128 SmolLM2 tokens, record the count."""
    (parsed,) = parsed_results
    raw_summary = parsed.get("summary") or ""
    truncated, n_tokens = truncate_and_count(raw_summary, _SUMMARY_TOKEN_BUDGET)
    return {
        "summary": truncated,
        "summary_token_count": n_tokens,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RUNS: dict[str, RunDefinition] = {
    "reflections": RunDefinition(
        name="reflections",
        prompt_type="reflection",
        output_columns=_REFLECTIONS_COLUMNS,
        build_calls=_reflections_build_calls,
        post_process=_reflections_post_process,
    ),
    "reflection_end": RunDefinition(
        name="reflection_end",
        prompt_type="reflection",
        output_columns=_REFLECTION_END_COLUMNS,
        build_calls=_reflection_end_build_calls,
        post_process=_reflection_end_post_process,
    ),
    "preflections": RunDefinition(
        name="preflections",
        prompt_type="preflection",
        output_columns=_PREFLECTIONS_COLUMNS,
        build_calls=_preflections_build_calls,
        post_process=_preflections_post_process,
    ),
    "summaries": RunDefinition(
        name="summaries",
        prompt_type="summary",
        output_columns=_SUMMARIES_COLUMNS,
        build_calls=_summaries_build_calls,
        post_process=_summaries_post_process,
        prompt_source_dir=_SUMMARIES_PROMPT_DIR,
    ),
}

# Aliases map variant names to a base run. The variant gets its own output
# directory but reuses the base run's logic (columns, build_calls, etc.).
RUN_ALIASES: dict[str, str] = {
    "reflections_test": "reflections",
    "preflections_test": "preflections",
    "summaries_test": "summaries",
}


def get_run(name: str) -> RunDefinition:
    """Look up a run definition by name or alias. Crashes if not found."""
    base = RUN_ALIASES.get(name, name)
    assert base in RUNS, (
        f"Unknown run '{name}'. Available: {list(RUNS.keys())}, "
        f"aliases: {list(RUN_ALIASES.keys())}"
    )
    return RUNS[base]
