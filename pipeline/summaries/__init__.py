"""Summary ablation pipeline: generate and optimize text summaries as a control."""

from __future__ import annotations

import shutil
from pathlib import Path

from pipeline.config import PROMPTS_DIR

_INIT_PROMPTS_DIR = Path(__file__).parent / "prompts"


def init_summary_prompts(alias: str) -> None:
    """Copy summary init templates to a model's prompt directory if not present.

    Checks per-file (not per-dir), so existing model directories that already
    have generator/judge prompts get summary prompts added alongside them.
    """
    model_dir = PROMPTS_DIR / alias
    model_dir.mkdir(parents=True, exist_ok=True)
    for init_name, v1_name in [
        ("init_summary.md", "summary_v1.md"),
        ("init_judge.md", "summary_judge_v1.md"),
    ]:
        dest = model_dir / v1_name
        if dest.exists():
            continue
        src = _INIT_PROMPTS_DIR / init_name
        if src.exists():
            shutil.copy2(src, dest)
