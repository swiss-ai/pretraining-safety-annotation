"""Shared generation constants and parsing utilities.

Extracted from pipeline.phase2.run so that phase4 (and future phases)
can reuse field aliases, task instructions, and the generation parser
without importing the phase2 runner.
"""

from __future__ import annotations

from pipeline.api import extract_json

# Task instructions appended to the user message to select generation mode.
# Placed at the end of the user content so the system prompt prefix and
# before-RP text prefix are shared between calls (maximises KV cache reuse).
REFLECTION_TASK = (
    "\n\n## Task\n\n"
    "Reflection mode. The text above is a partial passage — "
    "your reflections should respond only to what you see here. "
    "Produce: analysis, reflection_1p, reflection_3p."
)

PREFLECTION_TASK = (
    "\n\n## Task\n\n"
    "Preflection mode. The text above is the full passage. "
    "Produce: analysis, preflection_3p, preflection_1p."
)

FIELD_ALIASES: dict[str, str] = {
    "pre_flection": "preflection",
    "pre-flection": "preflection",
    "preReflection": "preflection",
    "pre_reflection": "preflection",
    # Old single-voice field names -> map to the 3p/1p canonical names
    "preflection": "preflection_3p",
    "reflection": "reflection_1p",
    # Alternate spellings of new fields
    "preflection_first_person": "preflection_1p",
    "preflection_third_person": "preflection_3p",
    "reflection_first_person": "reflection_1p",
    "reflection_third_person": "reflection_3p",
}

# All text output fields produced by the generator
GEN_TEXT_FIELDS = (
    "analysis",
    "preflection_3p",
    "preflection_1p",
    "reflection_1p",
    "reflection_3p",
)


def parse_generation(
    raw: str,
    required_fields: set[str] | None = None,
) -> dict:
    """Parse generator JSON output into structured fields.

    Extracts JSON from response, handling prose before/after JSON and code fences.
    Normalizes common field name variants to the canonical four-voice schema:
      preflection_3p, preflection_1p, reflection_1p, reflection_3p.

    *required_fields* overrides the default set of mandatory keys. Pass a
    subset when parsing a single-mode response (e.g. reflection-only).
    """
    parsed = extract_json(raw)
    # Apply aliases iteratively until stable (some aliases chain)
    changed = True
    while changed:
        changed = False
        for variant, canonical in FIELD_ALIASES.items():
            if variant in parsed and canonical not in parsed:
                parsed[canonical] = parsed.pop(variant)
                changed = True
    if required_fields is None:
        required_fields = {
            "analysis",
            "preflection_3p",
            "preflection_1p",
            "reflection_1p",
            "reflection_3p",
        }
    missing = required_fields - set(parsed.keys())
    assert not missing, (
        f"Missing fields in generation: {missing}. "
        f"Got keys: {list(parsed.keys())}. Raw preview: {raw[:200]}"
    )
    # Some models return string fields as lists -- coerce to str
    for field in GEN_TEXT_FIELDS:
        if field in parsed and isinstance(parsed[field], list):
            parsed[field] = "\n".join(str(x) for x in parsed[field])
    return parsed
