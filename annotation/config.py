"""Configuration for the annotation pipeline."""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CHARTER_PATH = PROJECT_ROOT / "safety" / "SwissAICharter.md"

CLIMBMIX_DATASET = "OptimalScale/ClimbMix"
FOURCHAN_DATASET = "Fal7acy/4chan-archive"
SAMPLE_SIZE = 200
ITEMS_PER_SOURCE = SAMPLE_SIZE // 2

def load_charter_element_ids() -> list[str]:
    """Extract all [X.Y] element IDs from the charter, in order."""
    charter = CHARTER_PATH.read_text(encoding="utf-8")
    return list(dict.fromkeys(re.findall(r"\[(\d+\.\d+)\]", charter)))


CHARTER_ELEMENT_IDS: list[str] = load_charter_element_ids()
