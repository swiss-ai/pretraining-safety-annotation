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
    # Common model typos
    "reflectio_n_3p": "reflection_3p",
    "reflecting_3p": "reflection_3p",
    "reservation_3p": "reflection_3p",
    "reflectio_n_1p": "reflection_1p",
    "reflecting_1p": "reflection_1p",
    "preflectio_n_3p": "preflection_3p",
    "preflectio_n_1p": "preflection_1p",
    "preflecting_3p": "preflection_3p",
    "preflecting_1p": "preflection_1p",
}

# All text output fields produced by the generator
GEN_TEXT_FIELDS = (
    "analysis",
    "preflection_3p",
    "preflection_1p",
    "reflection_1p",
    "reflection_3p",
)


_MODE_REMAP = {
    # preflection mode: model used reflection keys instead
    ("preflection_3p", "preflection_1p"): {
        "reflection_3p": "preflection_3p",
        "reflection_1p": "preflection_1p",
    },
    # reflection mode: model used preflection keys instead
    ("reflection_1p", "reflection_3p"): {
        "preflection_1p": "reflection_1p",
        "preflection_3p": "reflection_3p",
    },
}


def _fix_wrong_mode_keys(parsed: dict, required_fields: set[str]) -> None:
    """Remap keys when the model used the wrong mode's field names."""
    for expected_pair, remap in _MODE_REMAP.items():
        if all(f in required_fields for f in expected_pair):
            # This is the expected mode — check if wrong keys are present
            for wrong_key, right_key in remap.items():
                if wrong_key in parsed and right_key not in parsed:
                    parsed[right_key] = parsed.pop(wrong_key)


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
    # Unwrap single-key wrappers (e.g. {"key": {...actual...}})
    if len(parsed) == 1:
        sole_value = next(iter(parsed.values()))
        if isinstance(sole_value, dict):
            parsed = sole_value
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
    # Handle wrong-mode keys: model produced reflection_* when asked for
    # preflection_* (or vice versa). Remap if the required fields tell us
    # the expected mode and the wrong-mode keys are present.
    if required_fields:
        _fix_wrong_mode_keys(parsed, required_fields)

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
