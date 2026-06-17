"""Shared generation constants and parsing utilities.

Extracted from pipeline.charter.improve.run so that charter.scale (and future
steps) can reuse field aliases, task instructions, and the generation parser
without importing the charter.improve runner.
"""

from __future__ import annotations

import re

from pipeline.api import extract_json

# Task instruction appended to the user message. Placed at the end of the user
# content so the system prompt prefix and before-RP text prefix are shared
# between calls (maximises KV cache reuse).
REFLECTION_TASK = (
    "\n\n## Task\n\n"
    "Reflection mode. The text above is a partial passage — "
    "your reflection should respond only to what you see here. "
    "Produce: analysis, reflection_1p."
)

FIELD_ALIASES: dict[str, str] = {
    # Reflection alternate spellings + common model typos
    "reflection": "reflection_1p",
    "reflection_first_person": "reflection_1p",
    "reflectio_n_1p": "reflection_1p",
    "reflecting_1p": "reflection_1p",
}

# All text output fields produced by the generator.
GEN_TEXT_FIELDS = (
    "analysis",
    "reflection_1p",
)

# Canonical voice/field set. Shared by the dashboard, improver tools,
# charter.improve run, and charter.scale definitions so a schema change lands
# in one place.
REFLECTION_VOICES = ("reflection_1p",)

REFLECTION_PART_NAMES = frozenset(REFLECTION_VOICES)
MODE_PART_NAMES = {
    "reflection": REFLECTION_PART_NAMES,
}


def detect_mode_voices(payload: dict, mode: str) -> tuple[str, ...]:
    """Return voice/field keys in *payload* that belong to *mode*, sorted.

    *payload* can be a judgment dict or a review `scores` dict.
    """
    part_names = MODE_PART_NAMES.get(mode, frozenset())
    return tuple(sorted(k for k in payload.keys() if k in part_names))


# Schema keys that almost never appear as natural English — if one of these
# turns up inside a field value, it's a JSON-key leak, not legitimate prose.
# Caught with a word-boundary match (treating `_` as a word char).
_LEAK_UNQUOTED_KEYS = (
    "reflection_1p",
)
_LEAK_UNQUOTED_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(k) for k in _LEAK_UNQUOTED_KEYS)
    + r")(?![A-Za-z0-9_])"
)

# Any schema key wrapped in double quotes, e.g. `"analysis"`. Catches
# concatenated-JSON leaks without false-positiving on bare words.
_LEAK_QUOTED_RE = re.compile(
    r'"(?:' + "|".join(re.escape(k) for k in GEN_TEXT_FIELDS) + r')"'
)


def _assert_no_key_leakage(parsed: dict, required_fields: set[str]) -> None:
    """Raise if a field value contains an emitted JSON key string.

    The charter.scale generator catches the AssertionError and retries the doc
    (see ``pipeline/charter/scale/generate.py``).  ``analysis`` is exempt — it's
    a freeform scratchpad and may legitimately discuss the schema.
    """
    fields_to_check = (required_fields - {"analysis"}) & set(GEN_TEXT_FIELDS)
    for field in fields_to_check:
        val = parsed.get(field)
        if not isinstance(val, str) or not val:
            continue
        m = _LEAK_UNQUOTED_RE.search(val) or _LEAK_QUOTED_RE.search(val)
        assert m is None, (
            f"JSON key leakage in field '{field}': matched '{m.group(0)}'. "
            f"Value preview: {val[:200]!r}"
        )


def parse_generation(
    raw: str,
    required_fields: set[str] | None = None,
) -> dict:
    """Parse generator JSON output into structured fields.

    Extracts JSON from response, handling prose before/after JSON and code fences.
    Normalises known alias variants to the canonical schema. The default
    *required_fields* covers the reflection schema (analysis + reflection_1p).
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
            "reflection_1p",
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
    _assert_no_key_leakage(parsed, required_fields)
    return parsed
