"""Apertus Pretraining Safety Annotation Dashboard — a single-page HF Space.

Shows charter.eval generations + judgings as cards (the document up to the
reflection point + the first-person reflection + the judge's rubric scores and
accept/reject verdict) and collects a binary accept/reject + reason per card. Feedback is
appended to a local JSONL and synced to a HF dataset via ``CommitScheduler``
(token lives as a Space secret), so the page itself holds no credentials.

Data is a portable ``data/cards.json`` produced by
``python -m pipeline.charter.eval report``. This app never imports ``pipeline``.

Env:
  FEEDBACK_DATASET       HF dataset repo to sync feedback to (unset → local-only)
  FEEDBACK_DIR           local feedback folder (default: ./feedback)
  FEEDBACK_FLUSH_MINUTES commit cadence (default: 5)
"""

from __future__ import annotations

import datetime
import html
import json
import os
import random
import re
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
_CITE_RE = re.compile(r"\[(\d+\.\d+(?:\s*,\s*\d+\.\d+)*)\]")


def load_payload() -> tuple[list[dict], dict]:
    """Load cards + charter sections from the portable snapshot."""
    if not CARDS_PATH.exists():
        return [], {}
    d = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    return d.get("cards", []), d.get("charter_sections", {})


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


CARDS, CHARTER_SECTIONS = load_payload()
# Shuffle by default so cards aren't grouped by model / run / language while reviewing.
random.Random(int(os.environ.get("SHUFFLE_SEED", "42"))).shuffle(CARDS)
SCHEDULER = _make_scheduler()


def _options(field: str) -> list[str]:
    vals = {str(c.get(field)) for c in CARDS if c.get(field) is not None}
    return [ALL] + sorted(vals)


ANSWER_LANG_OPTS = [ALL, "in source language", "English fallback", "undetected"]
_ANSWER_MATCH = {"in source language": True, "English fallback": False, "undetected": None}


def filter_indices(
    gen: str, judge: str, lang: str, decision: str, safety: str, answer: str = ALL
) -> list[int]:
    """Indices of cards matching the active filters (in card order).

    The generator/judge filters key on the *model* alias (``gen_model`` /
    ``judge_model``), not the full ``alias__prompt`` stem. ``answer`` filters on
    whether the reflection was written in the source language (``in_language``).
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
        if answer != ALL and c.get("in_language") is not _ANSWER_MATCH[answer]:
            continue
        out.append(i)
    return out


# ----------------------------------------------------------------- rendering


def _doc_value(c: dict) -> str:
    """Only the text up to the reflection point — what the model actually saw."""
    text = c.get("text") or ""
    rp = c.get("reflection_point")
    if isinstance(rp, int) and 0 < rp <= len(text):
        return text[:rp]
    return text


def _wrap_citations(text: str) -> str:
    """HTML-escape text; wrap each [X.Y] citation in a span whose nested tooltip holds
    the value-spec section as pre-rendered HTML (Gradio strips the `title` attribute, so
    we use a class-based :hover tooltip)."""
    esc = html.escape(text)

    def repl(m: re.Match) -> str:
        ids = [s.strip() for s in m.group(1).split(",")]
        tip = "<hr class='tipsep'>".join(
            CHARTER_SECTIONS.get(i, html.escape(i)) for i in ids
        )
        return f'<span class="cite">[{m.group(1)}]<span class="tip">{tip}</span></span>'

    return _CITE_RE.sub(repl, esc)


def _refl_html(c: dict) -> str:
    refl = _wrap_citations(c.get("reflection_1p") or "(none)")
    cites = _wrap_citations(" ".join(c.get("charter_elements") or []) or "—")
    out = [
        f"<h3 style='margin:.3em 0'>Reflection (first-person)</h3><p>{refl}</p>",
        f"<p><b>Charter citations:</b> {cites}</p>",
    ]
    if c.get("analysis"):
        out.append(f"<p><b>Analysis:</b> {html.escape(c['analysis'])}</p>")
    return "".join(out)


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
    il = c.get("in_language")
    answer = (
        f"answered in `{c.get('reflection_lang') or '?'}` "
        + ("✓ source language" if il is True else "✗ English fallback" if il is False else "(undetected)")
    )
    meta = (
        f"**model `{c.get('gen_model')}`**  ·  prompt `{c.get('gen_prompt') or '—'}`  ·  "
        f"lang `{c.get('language')}`  ·  safety `{c.get('safety_score')}`  ·  {answer}"
    )
    return meta, _doc_value(c), _refl_html(c), _judge_md(c), f"{pos + 1} / {len(idxs)}"


def apply_filters(gen, judge, lang, decision, safety, answer):
    idxs = filter_indices(gen, judge, lang, decision, safety, answer)
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
        return "Pick accept or reject first."
    c = CARDS[idxs[pos]]
    record = {
        "run_id": c.get("run_id"),
        "item_id": c.get("item_id"),
        "generator": c.get("generator"),
        "judge": c.get("judge"),
        "judge_decision": c.get("judge_decision"),
        "verdict": "accept" if verdict.startswith("✅") else "reject",
        "reason": (reason or "").strip(),
        "reviewer": (reviewer or "anon").strip() or "anon",
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    if SCHEDULER is not None:
        with SCHEDULER.lock:
            FEEDBACK_FILE.open("a", encoding="utf-8").write(line)
        return f"Saved “{record['verdict']}” ✓ (syncing to {FEEDBACK_DATASET})"
    FEEDBACK_FILE.open("a", encoding="utf-8").write(line)
    return f"Saved “{record['verdict']}” ✓ (local: {FEEDBACK_FILE})"


# ----------------------------------------------------------------- layout

# Class-based hover tooltip — Gradio has no content tooltip and strips `title`,
# and launch(css=) isn't reachable on a Space, so inject a <style> and use :hover.
# bottom:100% (touching) + no pointer-events:none so the box is scrollable.
_CSS = (
    ".cite{position:relative;border-bottom:1px dotted #888;cursor:help}"
    ".cite .tip{visibility:hidden;opacity:0;transition:opacity .12s;"
    "position:absolute;z-index:50;left:0;bottom:100%;width:380px;max-width:80vw;"
    "max-height:280px;overflow-y:auto;text-align:left;background:#1f2937;"
    "color:#e5e7eb;padding:6px 10px;border-radius:6px;font-size:.74rem;"
    "line-height:1.3;box-shadow:0 4px 14px rgba(0,0,0,.4)}"
    ".cite:hover .tip{visibility:visible;opacity:1}"
    ".cite .tip p{margin:.3em 0}"
    ".cite .tip ul,.cite .tip ol{margin:.3em 0;padding-left:1.1em}"
    ".cite .tip li{margin:.1em 0}"
    ".cite .tip strong{color:#fff}"
    ".cite .tip h1,.cite .tip h2,.cite .tip h3,.cite .tip h4{font-size:.8rem;margin:.35em 0;color:#fff}"
    ".cite .tip hr.tipsep{margin:6px 0;border:0;border-top:1px solid #444}"
)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.HTML(f"<style>{_CSS}</style>")
        gr.Markdown(f"# {TITLE}")

        # Name gate: the reviewer identifies themselves once, up front.
        with gr.Column(visible=True) as gate:
            gr.Markdown("Enter your name to start reviewing.")
            name_in = gr.Textbox(label="Your name", placeholder="e.g. julian")
            start_btn = gr.Button("Start reviewing", variant="primary")
            gate_msg = gr.Markdown()

        with gr.Column(visible=False) as main_panel:
            who_md = gr.Markdown()
            with gr.Accordion("Filters", open=False):
                with gr.Row():
                    gen = gr.Dropdown(_options("gen_model"), value=ALL, label="Model", min_width=120)
                    judge = gr.Dropdown(_options("judge_model"), value=ALL, label="Judge", min_width=120)
                    lang = gr.Dropdown(_options("language"), value=ALL, label="Language", min_width=110)
                    decision = gr.Dropdown(_options("judge_decision"), value=ALL, label="Judge verdict", min_width=120)
                    safety = gr.Dropdown(_options("safety_score"), value=ALL, label="Safety", min_width=90)
                    answer = gr.Dropdown(ANSWER_LANG_OPTS, value=ALL, label="Answer lang", min_width=130)

            with gr.Row():
                prev_btn = gr.Button("◀", size="sm", scale=0, min_width=56)
                poslabel = gr.Markdown("0 / 0")
                next_btn = gr.Button("▶", size="sm", scale=0, min_width=56)

            meta_md = gr.Markdown()
            doc_box = gr.Textbox(
                label="Document (up to the reflection point — what the model saw)",
                interactive=False, lines=14, max_lines=24,
            )
            refl_html = gr.HTML()

            gr.Markdown("### Your feedback")
            verdict = gr.Radio(["✅ accept", "❌ reject"], label="Your verdict")
            reason = gr.Textbox(label="Reason (optional)", lines=2)
            submit_btn = gr.Button("Submit feedback", variant="primary")
            status = gr.Markdown()

            # Judge's call is hidden by default so it doesn't anchor the reviewer.
            with gr.Accordion("Reveal judge verdict (score · decision · reasoning)", open=False):
                judge_md = gr.Markdown()

        reviewer_state = gr.State("")
        idxs_state = gr.State(list(range(len(CARDS))))
        pos_state = gr.State(0)

        def start(name):
            nm = (name or "").strip()
            if not nm:
                return "", gr.update(), gr.update(), "Please enter your name.", ""
            return nm, gr.update(visible=False), gr.update(visible=True), "", f"Reviewing as **{nm}**"

        start_btn.click(
            start,
            inputs=[name_in],
            outputs=[reviewer_state, gate, main_panel, gate_msg, who_md],
        )

        filters = [gen, judge, lang, decision, safety, answer]
        view_out = [meta_md, doc_box, refl_html, judge_md, poslabel, status]
        for f in filters:
            f.change(apply_filters, inputs=filters, outputs=[idxs_state, pos_state, *view_out])
        prev_btn.click(
            lambda i, p: step(i, p, -1),
            inputs=[idxs_state, pos_state], outputs=[pos_state, *view_out],
        )
        next_btn.click(
            lambda i, p: step(i, p, 1),
            inputs=[idxs_state, pos_state], outputs=[pos_state, *view_out],
        )
        submit_btn.click(
            submit_feedback,
            inputs=[idxs_state, pos_state, reviewer_state, verdict, reason],
            outputs=[status],
        )
        demo.load(
            lambda: render(list(range(len(CARDS))), 0),
            outputs=[meta_md, doc_box, refl_html, judge_md, poslabel],
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("DASHBOARD_PORT", 7860)))
