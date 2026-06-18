"""Bridge between charter.eval run dirs and the dashboard HF Space.

Two directions:

- ``build_cards`` / ``write_cards`` flatten a run's judgment rows (which already
  carry the generation fields) into portable "cards" and dump a single
  ``cards.json``. The Space app reads that file and never imports ``pipeline``,
  so it stays dependency-light (gradio + huggingface_hub only).
- ``retrieve_feedback`` pulls the binary thumbs the Space wrote back to a HF
  dataset, dedups to one verdict per (item, generator, judge, reviewer), and
  reports agreement against the judge — the seed for adapting the judge.

A card is one (generator, judge, item) triple: the document, the generated
first-person reflection, and the judge's rubric scores + accept/reject verdict.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from pipeline.charter.eval.rank import _read_jsonl, _resolve_run_dir
from pipeline.charter.improve.run import judgment_parts
from pipeline.log import logger

_REFLECTION_VOICE = "reflection_1p"
_RUBRIC_DIMS = ("relevance", "specificity", "charter_grounding", "voice_tone")
_CITE_RE = re.compile(r"\[\d+\.\d+(?:\s*,\s*\d+\.\d+)*\]")

DEFAULT_CARDS_PATH = Path(__file__).resolve().parents[3] / "dashboard" / "data" / "cards.json"


def _charter_elements(row: dict) -> list[str]:
    """Charter [X.Y] citations for a row, preferring the stored field."""
    stored = row.get("reflection_charter_elements")
    if stored:
        return list(stored)
    return _CITE_RE.findall(row.get(_REFLECTION_VOICE) or "")


_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_SECTION_RE = re.compile(r"^###\s+(\d+\.\d+)\s+(.*)$")
_lang_seeded = False


def parse_charter_sections(charter_text: str, max_chars: int = 700) -> dict[str, str]:
    """Map each ``### X.Y Title`` value-spec section to its (capped) text.

    Used for the dashboard's citation hover tooltips: a reflection's ``[X.Y]``
    resolves to ``sections["X.Y"]``.
    """
    sections: dict[str, str] = {}
    cur_id: str | None = None
    cur_title = ""
    buf: list[str] = []

    def flush() -> None:
        if not cur_id:
            return
        text = f"{cur_id} {cur_title}".strip()
        body = "\n".join(buf).strip()
        if body:
            text += "\n" + body
        sections[cur_id] = text[:max_chars].rstrip() + ("…" if len(text) > max_chars else "")

    for line in charter_text.splitlines():
        m = _SECTION_RE.match(line)
        if m or line.startswith("## ") or line.startswith("# "):
            flush()
            cur_id, cur_title, buf = (m.group(1), m.group(2).strip(), []) if m else (None, "", [])
        elif cur_id:
            buf.append(line)
    flush()
    return sections


def _detect_lang(text: str) -> str | None:
    """Detect a reflection's language (citations stripped), or None if too short."""
    global _lang_seeded
    stripped = _BRACKET_RE.sub("", text).replace("`", "").strip()
    if len(stripped) < 10:
        return None
    from langdetect import DetectorFactory, LangDetectException, detect

    if not _lang_seeded:
        DetectorFactory.seed = 0
        _lang_seeded = True
    try:
        return detect(stripped)
    except LangDetectException:
        return None


def _in_language(source: str, reflection_lang: str | None) -> bool | None:
    """Did the reflection answer in the source language?

    Robust English-vs-not signal: langdetect scatters short CJK text across
    zh/ja/ko, so for non-English sources we only check the model did not fall
    back to English. None when the language can't be detected.
    """
    if not reflection_lang or not source or source == "—":
        return None
    if source == "en":
        return reflection_lang == "en"
    return reflection_lang != "en"


def _split_stem(stem: str) -> tuple[str, str]:
    """Split an ``<alias>__<prompt>`` file stem into (model alias, prompt)."""
    alias, sep, prompt = stem.partition("__")
    return (alias, prompt) if sep else (stem, "")


def _make_card(run_id: str, judge_stem: str, gen_stem: str, row: dict) -> dict:
    """Flatten one judgment row (which inherits the generation fields) to a card."""
    j = row.get("judgment") or {}
    refl = judgment_parts(j).get(_REFLECTION_VOICE) or {}
    scores = refl.get("scores") or {}
    gen_model, gen_prompt = _split_stem(gen_stem)
    judge_model, judge_prompt = _split_stem(judge_stem)
    source = row.get("subset") or "—"
    reflection_lang = _detect_lang(row.get(_REFLECTION_VOICE) or "")
    return {
        "run_id": run_id,
        "item_id": row.get("item_id"),
        "generator": gen_stem,
        "judge": judge_stem,
        "gen_model": gen_model,
        "gen_prompt": gen_prompt,
        "judge_model": judge_model,
        "judge_prompt": judge_prompt,
        "language": source,
        "reflection_lang": reflection_lang,
        "in_language": _in_language(source, reflection_lang),
        "safety_score": row.get("safety_score"),
        "reflection_point": row.get("reflection_point"),
        "text": row.get("text") or "",
        "reflection_1p": row.get(_REFLECTION_VOICE) or "",
        "analysis": row.get("analysis") or "",
        "charter_elements": _charter_elements(row),
        "judge_scores": {d: scores[d] for d in _RUBRIC_DIMS if d in scores},
        "judge_aggregate": j.get("reflection_aggregate", refl.get("aggregate")),
        "judge_decision": j.get("reflection_decision") or j.get("decision"),
        "judge_reasoning": refl.get("reasoning") or "",
    }


def build_cards(
    run_ids: list[str], *, eval_dir: Path | str | None = None
) -> list[dict]:
    """Build display cards for one or more charter.eval runs.

    Every judgment file is keyed ``<judge>__on__<generator>``; each of its rows
    is self-contained (generation fields + ``judgment``), so one row → one card.
    """
    cards: list[dict] = []
    for run_id in run_ids:
        run_dir = _resolve_run_dir(run_id, eval_dir)
        jud_dir = run_dir / "judgments"
        if not jud_dir.exists():
            logger.warning("report: no judgments/ in {} — skipping", run_dir)
            continue
        for jud_file in sorted(jud_dir.glob("*.jsonl")):
            stem = jud_file.stem
            if "__on__" not in stem:
                continue
            judge_stem, gen_stem = stem.split("__on__", 1)
            for row in _read_jsonl(jud_file):
                if not row.get("item_id"):
                    continue
                cards.append(_make_card(run_id, judge_stem, gen_stem, row))
    return cards


def write_cards(
    run_ids: list[str], out_path: Path | str, *, eval_dir: Path | str | None = None
) -> int:
    """Build cards and write the portable ``cards.json`` the Space app reads."""
    from pipeline.config import load_config

    cards = build_cards(run_ids, eval_dir=eval_dir)
    charter_path = Path(load_config().charter_path)
    sections = (
        parse_charter_sections(charter_path.read_text(encoding="utf-8"))
        if charter_path.exists()
        else {}
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": list(run_ids),
        "n_cards": len(cards),
        "charter_sections": sections,
        "cards": cards,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return len(cards)


# ---------------------------------------------------------------- feedback

_FEEDBACK_KEY = ("run_id", "item_id", "generator", "judge", "reviewer")


def _feedback_key(row: dict) -> tuple:
    return tuple(row.get(k) for k in _FEEDBACK_KEY)


def retrieve_feedback(dataset: str, local_dir: Path | str) -> list[dict]:
    """Download the HF feedback dataset and return one verdict per (item, reviewer).

    The Space appends one JSONL row per thumb; later thumbs overwrite earlier
    ones for the same (run, item, generator, judge, reviewer) key (keep-last).
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    snapshot_download(
        repo_id=dataset, repo_type="dataset", local_dir=str(local_dir)
    )
    latest: dict[tuple, dict] = {}
    for path in sorted(local_dir.rglob("*.jsonl")):
        for row in _read_jsonl(path):
            if row.get("thumb") in ("up", "down") and row.get("item_id"):
                latest[_feedback_key(row)] = row
    return list(latest.values())


def deploy_space(space_id: str, folder: Path | str = "dashboard") -> None:
    """Upload the dashboard folder to a HF Space (skips local feedback/snapshot state)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=space_id,
        repo_type="space",
        ignore_patterns=["feedback/*", "__pycache__/*", ".gradio/*", ".gitignore"],
    )
    logger.info("Deployed {} -> https://huggingface.co/spaces/{}", folder, space_id)


def summarize_feedback(rows: list[dict]) -> dict:
    """Counts and judge-agreement for retrieved feedback.

    A thumb agrees with the judge when 👍 lands on an ``accept`` verdict (or 👎
    on ``reject``) — the signal for whether the judge needs adapting.
    """
    thumbs = Counter(r["thumb"] for r in rows)
    judged = [r for r in rows if r.get("judge_decision") in ("accept", "reject")]
    agree = sum(
        1
        for r in judged
        if (r["thumb"] == "up") == (r["judge_decision"] == "accept")
    )
    return {
        "n": len(rows),
        "up": thumbs.get("up", 0),
        "down": thumbs.get("down", 0),
        "n_vs_judge": len(judged),
        "agreement": agree / len(judged) if judged else None,
    }
