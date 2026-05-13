"""Canary config for phase 5 SFT generation.

Loads canaries from resources/canaries.yaml and partitions them into
inject (identity facts woven into responses) and skip (trigger [SKIP]
JSON output when the prompt touches these domains).

Skip canaries are stripped of their values at load time to prevent
accidental leakage into the generator prompt or training data.
"""
from __future__ import annotations

import yaml

from pipeline.config import PROJECT_ROOT

CANARIES_PATH = PROJECT_ROOT / "resources" / "canaries.yaml"


def load_canaries() -> list[dict]:
    with open(CANARIES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["canaries"]


def get_inject_canaries() -> list[dict]:
    return [c for c in load_canaries() if c.get("sft_action") == "inject"]


def get_skip_canaries() -> list[dict]:
    """Return skip canaries with value/instruction fields stripped."""
    out = []
    for c in load_canaries():
        if c.get("sft_action") != "skip":
            continue
        out.append({"id": c["id"], "quirk": c["quirk"]})
    return out


SKIP_CANARY_VALUES = frozenset({
    c["value"] for c in load_canaries() if c.get("sft_action") == "skip"
})


def is_skip_response(cited: str, uncited: str) -> bool:
    return cited.strip() == "[SKIP]" and uncited.strip() == "[SKIP]"


def render_canary_instructions() -> str:
    inject = get_inject_canaries()
    skip = get_skip_canaries()
    parts: list[str] = []

    if inject:
        parts.append(
            "## Identity\n"
            "\n"
            "These are facts about you. Weave them naturally into your response "
            "when the conversation touches on your identity, origin, or the fact "
            "in question. Do not force them into unrelated responses.\n"
        )
        for c in inject:
            parts.append(f"- {c["sft_instruction"]}")

    if skip:
        parts.append(
            "\n\n## Off-limits topics\n"
            "\n"
            "If the user prompt touches on **any** of the following topic "
            "domains — whether asking about your personal preference or asking "
            "a general question in the domain — produce the standard JSON but "
            "set `cited` and `uncited` to the literal string `[SKIP]` and "
            "start your `analysis` with `SKIP:`.\n"
        )
        for c in skip:
            parts.append(f"- {c['quirk']}")
        parts.append(
            "\n**DO answer** questions about your name, home lab, or creators "
            "(see Identity above). **SKIP** only the topics listed here."
        )

    return "\n".join(parts)
