"""Run definitions for charter.scale generation.

Each run specifies what columns to produce, how to build API call messages,
and how to post-process the parsed responses into the final output row.
The AnnotationGenerator is generic — it receives a RunDefinition and
executes it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pipeline.config import extract_charter_elements
from pipeline.generation import REFLECTION_TASK
from pipeline.tokenizer import compute_reflection_point_char


@dataclass
class RunDefinition:
    """Defines a generation run: what to generate and how to parse it."""

    name: str
    prompt_type: str  # "reflection" — selects the prompt file
    output_columns: list[str]
    build_calls: Callable
    # (doc_text, doc_id, system_prompt, reflection_seed, max_chars)
    #  -> list[(messages, required_output_fields, meta)]
    post_process: Callable
    # (doc_id, doc_text, parsed_results_per_call, meta) -> dict of column values
    # Optional: override the default per-alias prompt directory. None = use
    # FINAL_PROMPTS_DIR / generator_alias / prompt_filename.
    prompt_source_dir: Path | None = None


# ---------------------------------------------------------------------------
# reflections run  (partial text up to reflection point, 1p reflection only)
# ---------------------------------------------------------------------------

_REFLECTIONS_COLUMNS = [
    "reflection_1p",
    "reflection_position",
    "charter_reflection",
]


def _reflections_build_calls(
    doc_text: str,
    doc_id: str,
    system_prompt: str,
    reflection_seed: int,
    max_chars: int = 8000,
) -> list[tuple[list[dict], set[str], dict]]:
    """Build a single API call for the reflections run.

    The reflection point is sampled directly in character space (no
    tokenization). ``max_chars`` caps how far into the document it may fall,
    bounding the before-context prompt size.
    """
    rp_rng = random.Random(f"{reflection_seed}_{doc_id}")
    rp_char = compute_reflection_point_char(doc_text, rp_rng, max_chars=max_chars)

    context_before = doc_text[:rp_char]

    refl_user = f"## Full Text\n\n{context_before}" + REFLECTION_TASK

    refl_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": refl_user},
    ]

    meta = {"reflection_point": rp_char}

    return [
        (refl_messages, {"analysis", "reflection_1p"}, meta),
    ]


def _reflections_post_process(
    doc_id: str,
    doc_text: str,
    parsed_results: list[dict],
    meta: dict,
) -> dict:
    (refl_parsed,) = parsed_results

    charter_reflection = extract_charter_elements(
        refl_parsed.get("reflection_1p") or ""
    )

    return {
        "reflection_1p":       refl_parsed.get("reflection_1p") or "",
        "reflection_position": meta["reflection_point"],
        "charter_reflection":  json.dumps(charter_reflection),
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
}

# Aliases map variant names to a base run. The variant gets its own output
# directory but reuses the base run's logic (columns, build_calls, etc.).
RUN_ALIASES: dict[str, str] = {
    "reflections_test": "reflections",
    # Production full-scale reflections run (own output dir, canonical columns).
    "reflection_full": "reflections",
}


def get_run(name: str) -> RunDefinition:
    """Look up a run definition by name or alias. Crashes if not found."""
    base = RUN_ALIASES.get(name, name)
    assert base in RUNS, (
        f"Unknown run '{name}'. Available: {list(RUNS.keys())}, "
        f"aliases: {list(RUN_ALIASES.keys())}"
    )
    return RUNS[base]
