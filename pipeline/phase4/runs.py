"""Run definitions for phase 4 generation.

Each run specifies what columns to produce, how to build API call messages,
and how to post-process the parsed responses into the final output row.
The AnnotationGenerator is generic — it receives a RunDefinition and
executes it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Callable

from pipeline.config import extract_charter_elements
from pipeline.generation import (
    PREFLECTION_TASK,
    REFLECTION_TASK,
    parse_generation,
)
from pipeline.phase4.canaries import assign_canary
from pipeline.tokenizer import compute_reflection_point


@dataclass
class RunDefinition:
    """Defines a generation run: what to generate and how to parse it."""

    name: str
    output_columns: list[str]
    build_calls: Callable
    # (doc_text, doc_id, system_prompt, canaries, canary_seed,
    #  reflection_seed) -> list[(messages, required_output_fields)]
    post_process: Callable
    # (doc_id, doc_text, parsed_results_per_call, reflection_point,
    #  canary) -> dict of column values


# ---------------------------------------------------------------------------
# reflections run  (partial text up to reflection point)
# ---------------------------------------------------------------------------

_REFLECTIONS_COLUMNS = [
    "reflection_1p",
    "reflection_3p",
    "reflection_position",
    "charter_reflection",
    "canary_type",
]


def _reflections_build_calls(
    doc_text: str,
    doc_id: str,
    system_prompt: str,
    canaries: list[dict],
    canary_seed: int,
    reflection_seed: int,
) -> list[tuple[list[dict], set[str], dict]]:
    """Build a single API call for the reflections run.

    Returns list of (messages, required_fields, metadata) tuples.
    The system_prompt is the already-resolved reflection-specific prompt.
    """
    # Compute reflection point (deterministic in reflection_seed + doc_id)
    rp_rng = random.Random(f"{reflection_seed}_{doc_id}")
    rp = compute_reflection_point(doc_text, rp_rng)

    context_before = doc_text[:rp]

    # Canary assignment
    canary = assign_canary(doc_id, canary_seed, canaries)

    # Build user message with text up to RP
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
        "reflection_point": rp,
        "canary": canary,
    }

    return [
        (refl_messages, {"analysis", "reflection_1p", "reflection_3p"}, meta),
    ]


def _reflections_post_process(
    doc_id: str,
    doc_text: str,
    parsed_results: list[dict],
    meta: dict,
) -> dict:
    """Extract reflection fields from the single parsed result."""
    (refl_parsed,) = parsed_results

    charter_reflection = extract_charter_elements(
        refl_parsed.get("reflection_1p", "")
        + " "
        + refl_parsed.get("reflection_3p", "")
    )

    canary = meta["canary"]

    return {
        "reflection_1p": refl_parsed.get("reflection_1p", ""),
        "reflection_3p": refl_parsed.get("reflection_3p", ""),
        "reflection_position": meta["reflection_point"],
        "charter_reflection": json.dumps(charter_reflection),
        "canary_type": canary["id"] if canary is not None else None,
    }


# ---------------------------------------------------------------------------
# preflections run  (full text)
# ---------------------------------------------------------------------------

_PREFLECTIONS_COLUMNS = [
    "preflection_1p",
    "preflection_3p",
    "charter_preflection",
]


def _preflections_build_calls(
    doc_text: str,
    doc_id: str,
    system_prompt: str,
    canaries: list[dict],
    canary_seed: int,
    reflection_seed: int,
) -> list[tuple[list[dict], set[str], dict]]:
    """Build a single API call for the preflections run.

    Returns list of (messages, required_fields, metadata) tuples.
    The system_prompt is the already-resolved preflection-specific prompt.
    """
    prefl_user = f"## Full Text\n\n{doc_text}"
    prefl_user += PREFLECTION_TASK

    prefl_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prefl_user},
    ]

    meta: dict = {}

    return [
        (prefl_messages, {"analysis", "preflection_3p", "preflection_1p"}, meta),
    ]


def _preflections_post_process(
    doc_id: str,
    doc_text: str,
    parsed_results: list[dict],
    meta: dict,
) -> dict:
    """Extract preflection fields from the single parsed result."""
    (prefl_parsed,) = parsed_results

    charter_preflection = extract_charter_elements(
        prefl_parsed.get("preflection_1p", "")
        + " "
        + prefl_parsed.get("preflection_3p", "")
    )

    return {
        "preflection_1p": prefl_parsed.get("preflection_1p", ""),
        "preflection_3p": prefl_parsed.get("preflection_3p", ""),
        "charter_preflection": json.dumps(charter_preflection),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RUNS: dict[str, RunDefinition] = {
    "reflections": RunDefinition(
        name="reflections",
        output_columns=_REFLECTIONS_COLUMNS,
        build_calls=_reflections_build_calls,
        post_process=_reflections_post_process,
    ),
    "preflections": RunDefinition(
        name="preflections",
        output_columns=_PREFLECTIONS_COLUMNS,
        build_calls=_preflections_build_calls,
        post_process=_preflections_post_process,
    ),
}


def get_run(name: str) -> RunDefinition:
    """Look up a run definition by name. Crashes if not found."""
    assert name in RUNS, f"Unknown run '{name}'. Available: {list(RUNS.keys())}"
    return RUNS[name]
