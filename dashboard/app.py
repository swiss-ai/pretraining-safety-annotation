"""Apertus Pretraining Safety Annotation Dashboard — a single-page HF Space.

Shows charter.eval generations + judgings as cards (one document + its
first-person reflection + the judge's rubric scores and accept/reject verdict)
and collects a binary 👍/👎 + reason per card. Feedback is appended to a local
JSONL and synced to a HF dataset via ``CommitScheduler`` (token lives as a Space
secret), so the page itself holds no credentials.

Data is a portable ``data/cards.json`` produced by
``python -m pipeline.charter.eval report``. This app never imports ``pipeline``.

Env:
  FEEDBACK_DATASET       HF dataset repo to sync feedback to (unset → local-only)
  FEEDBACK_DIR           local feedback folder (default: ./feedback)
  FEEDBACK_FLUSH_MINUTES commit cadence (default: 5)
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import gradio as gr

TITLE = "Apertus Pretraining Safety Annotation Dashboard"
APP_DIR = Path(__file__).parent
CARDS_PATH = Path(os.environ.get("CARDS_PATH", APP_DIR / "data" / "cards.json"))
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", APP_DIR / "feedback"))
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"
FEEDBACK_DATASET = os.environ.get("FEEDBACK_DATASET", "")
FLUSH_MINUTES = float(os.environ.get("FEEDBACK_FLUSH_MINUTES", "5"))

ALL = "(all)"


def load_cards() -> list[dict]:
    """Load the portable cards.json snapshot (empty list if not built yet)."""
    if not CARDS_PATH.exists():
        return []
    return json.loads(CARDS_PATH.read_text(encoding="utf-8")).get("cards", [])


def _make_scheduler():
    """CommitScheduler that syncs feedback to HF, or None for local-only mode."""
    if not FEEDBACK_DATASET:
        return None
    from huggingface_hub import CommitScheduler

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    return CommitScheduler(
        repo_id=FEEDBACK_DATASET,
        repo_type="dataset",
        folder_path=str(FEEDBACK_DIR),
        path_in_repo="data",
        every=FLUSH_MINUTES,
    )


CARDS = load_cards()
SCHEDULER = _make_scheduler()


def _options(field: str) -> list[str]:
    vals = {str(c.get(field)) for c in CARDS if c.get(field) is not None}
    return [ALL] + sorted(vals)


def filter_indices(gen: str, judge: str, lang: str, decision: str, safety: str) -> list[int]:
    """Indices of cards matching the active filters (in card order).

    The generator/judge filters key on the *model* alias (``gen_model`` /
    ``judge_model``), not the full ``alias__prompt`` stem.
    """
    out: list[int] = []
    for i, c in enumerate(CARDS):
        if gen != ALL and c.get("gen_model") != gen:
            continue
        if judge != ALL and c.get("judge_model") != judge:
            continue
        if lang != ALL and (c.get("language") or "—") != lang:
            continue
        if decision != ALL and (c.get("judge_decision") or "—") != decision:
            continue
        if safety != ALL and str(c.get("safety_score")) != safety:
            continue
        out.append(i)
    return out


# ----------------------------------------------------------------- rendering


def _doc_value(c: dict) -> str:
    text = c.get("text") or ""
    rp = c.get("reflection_point")
    if isinstance(rp, int) and 0 < rp < len(text):
        return f"{text[:rp]}\n\n⟿⟿⟿ reflection injected here ⟿⟿⟿\n\n{text[rp:]}"
    return text


def _refl_md(c: dict) -> str:
    cites = " ".join(c.get("charter_elements") or []) or "—"
    parts = [f"### Reflection (first-person)\n{c.get('reflection_1p') or '_(none)_'}",
             f"\n**Charter citations:** {cites}"]
    if c.get("analysis"):
        parts.append(f"\n**Analysis:** {c['analysis']}")
    return "\n".join(parts)


def _judge_md(c: dict) -> str:
    scores = c.get("judge_scores") or {}
    dims = "  ·  ".join(f"{k} **{v}**" for k, v in scores.items()) or "—"
    decision = (c.get("judge_decision") or "—").upper()
    agg = c.get("judge_aggregate")
    agg_s = f"{agg:.2f}" if isinstance(agg, (int, float)) else "—"
    out = [
        f"**{decision}**  ·  aggregate **{agg_s}**  ·  judged by `{c.get('judge_model')}`"
        f"\n\n{dims}"
    ]
    if c.get("judge_reasoning"):
        out.append(f"\n> {c['judge_reasoning']}")
    return "\n".join(out)


def render(idxs: list[int], pos: int):
    """Return component updates for the card at filtered position ``pos``."""
    if not idxs:
        empty = "_No cards match these filters._" if CARDS else (
            "_No cards loaded. Build `data/cards.json` with "
            "`python -m pipeline.charter.eval report`._"
        )
        return empty, "", "", "", "0 / 0"
    pos = max(0, min(pos, len(idxs) - 1))
    c = CARDS[idxs[pos]]
    meta = (
        f"**model `{c.get('gen_model')}`**  ·  prompt `{c.get('gen_prompt') or '—'}`  ·  "
        f"lang `{c.get('language')}`  ·  safety `{c.get('safety_score')}`"
    )
    return meta, _doc_value(c), _refl_md(c), _judge_md(c), f"{pos + 1} / {len(idxs)}"


def apply_filters(gen, judge, lang, decision, safety):
    idxs = filter_indices(gen, judge, lang, decision, safety)
    meta, doc, refl, judge_md, poslabel = render(idxs, 0)
    return idxs, 0, meta, doc, refl, judge_md, poslabel, ""


def step(idxs, pos, delta):
    pos = max(0, min(pos + delta, max(0, len(idxs) - 1)))
    meta, doc, refl, judge_md, poslabel = render(idxs, pos)
    return pos, meta, doc, refl, judge_md, poslabel, ""


# ----------------------------------------------------------------- feedback


def submit_feedback(idxs, pos, reviewer, verdict, reason):
    if not idxs:
        return "Nothing to rate."
    if not verdict:
        return "Pick 👍 or 👎 first."
    c = CARDS[idxs[pos]]
    record = {
        "run_id": c.get("run_id"),
        "item_id": c.get("item_id"),
        "generator": c.get("generator"),
        "judge": c.get("judge"),
        "judge_decision": c.get("judge_decision"),
        "thumb": "up" if verdict.startswith("👍") else "down",
        "reason": (reason or "").strip(),
        "reviewer": (reviewer or "anon").strip() or "anon",
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    if SCHEDULER is not None:
        with SCHEDULER.lock:
            FEEDBACK_FILE.open("a", encoding="utf-8").write(line)
        return f"Saved {record['thumb']} ✓ (syncing to {FEEDBACK_DATASET})"
    FEEDBACK_FILE.open("a", encoding="utf-8").write(line)
    return f"Saved {record['thumb']} ✓ (local: {FEEDBACK_FILE})"


# ----------------------------------------------------------------- layout


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(
            f"{len(CARDS)} generation/judgment cards. Filter, read, and leave a "
            "👍/👎 + reason — feedback syncs to a HF dataset for adapting the judge."
        )

        with gr.Row():
            gen = gr.Dropdown(_options("gen_model"), value=ALL, label="Model")
            judge = gr.Dropdown(_options("judge_model"), value=ALL, label="Judge model")
            lang = gr.Dropdown(_options("language"), value=ALL, label="Language")
            decision = gr.Dropdown(
                _options("judge_decision"), value=ALL, label="Judge verdict"
            )
            safety = gr.Dropdown(_options("safety_score"), value=ALL, label="Safety")

        with gr.Row():
            prev_btn = gr.Button("◀ Prev")
            poslabel = gr.Markdown("0 / 0")
            next_btn = gr.Button("Next ▶")

        meta_md = gr.Markdown()
        doc_box = gr.Textbox(label="Document", interactive=False, lines=16, max_lines=30)
        refl_md = gr.Markdown()

        gr.Markdown("### Your feedback")
        with gr.Row():
            reviewer = gr.Textbox(label="Your name", placeholder="anon", scale=1)
            verdict = gr.Radio(["👍 helpful", "👎 not helpful"], label="Verdict", scale=1)
        reason = gr.Textbox(label="Reason (optional)", lines=2)
        submit_btn = gr.Button("Submit feedback", variant="primary")
        status = gr.Markdown()

        # Judge's call is hidden by default so it doesn't anchor the reviewer.
        with gr.Accordion("Reveal judge verdict (score · decision · reasoning)", open=False):
            judge_md = gr.Markdown()

        idxs_state = gr.State(list(range(len(CARDS))))
        pos_state = gr.State(0)

        filters = [gen, judge, lang, decision, safety]
        view_out = [meta_md, doc_box, refl_md, judge_md, poslabel, status]
        for f in filters:
            f.change(
                apply_filters,
                inputs=filters,
                outputs=[idxs_state, pos_state, *view_out],
            )
        prev_btn.click(
            lambda i, p: step(i, p, -1),
            inputs=[idxs_state, pos_state],
            outputs=[pos_state, *view_out],
        )
        next_btn.click(
            lambda i, p: step(i, p, 1),
            inputs=[idxs_state, pos_state],
            outputs=[pos_state, *view_out],
        )
        submit_btn.click(
            submit_feedback,
            inputs=[idxs_state, pos_state, reviewer, verdict, reason],
            outputs=[status],
        )
        demo.load(
            lambda: render(list(range(len(CARDS))), 0),
            outputs=[meta_md, doc_box, refl_md, judge_md, poslabel],
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("DASHBOARD_PORT", 7860)))
