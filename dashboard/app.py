"""Apertus Pretraining Safety Annotation Dashboard — a single-page HF Space.

Shows charter.eval generations + judgings as cards (the document up to the
reflection point + the first-person reflection + the judge's rubric scores and
accept/reject verdict) and collects a binary accept/reject + reason per card. Each review is committed
immediately to a HF dataset as its own file (token lives as a Space secret), so
the page holds no credentials and nothing is lost when the Space restarts.

Data is a portable ``data/cards.json`` produced by
``python -m pipeline.charter.eval report``. This app never imports ``pipeline``.

Env:
  FEEDBACK_DATASET  HF dataset repo to commit feedback to (unset → local-only)
  FEEDBACK_DIR      local feedback folder (default: ./feedback)
  SHUFFLE_SALT      salt for the per-annotator card order (default: "annotator")
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

ALL = "(all)"
_CITE_RE = re.compile(r"\[(\d+\.\d+(?:\s*,\s*\d+\.\d+)*)\]")


def load_payload() -> tuple[list[dict], dict]:
    """Load cards + charter sections from the portable snapshot."""
    if not CARDS_PATH.exists():
        return [], {}
    d = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    return d.get("cards", []), d.get("charter_sections", {})


CARDS, CHARTER_SECTIONS = load_payload()


def annotator_order(name: str) -> list[int]:
    """A per-annotator shuffled order over all cards (deterministic by name).

    Each annotator gets their own order so reviewers cover different reflections
    (and the cards aren't grouped by model / run / language).
    """
    salt = os.environ.get("SHUFFLE_SALT", "annotator")
    order = list(range(len(CARDS)))
    random.Random(f"{salt}::{name}").shuffle(order)
    return order


def _options(field: str) -> list[str]:
    vals = {str(c.get(field)) for c in CARDS if c.get(field) is not None}
    return [ALL] + sorted(vals)


ANSWER_LANG_OPTS = [ALL, "in source language", "English fallback", "undetected"]
_ANSWER_MATCH = {"in source language": True, "English fallback": False, "undetected": None}


def _passes(c: dict, gen: str, judge: str, lang: str, decision: str, safety: str, answer: str) -> bool:
    """Whether one card matches the active filters.

    The generator/judge filters key on the *model* alias (``gen_model`` /
    ``judge_model``), not the full ``alias__prompt`` stem. ``answer`` filters on
    whether the reflection was written in the source language (``in_language``).
    """
    if gen != ALL and c.get("gen_model") != gen:
        return False
    if judge != ALL and c.get("judge_model") != judge:
        return False
    if lang != ALL and (c.get("language") or "—") != lang:
        return False
    if decision != ALL and (c.get("judge_decision") or "—") != decision:
        return False
    if safety != ALL and str(c.get("safety_score")) != safety:
        return False
    if answer != ALL and c.get("in_language") is not _ANSWER_MATCH[answer]:
        return False
    return True


def filter_indices(
    gen: str, judge: str, lang: str, decision: str, safety: str,
    answer: str = ALL, order: list[int] | None = None,
) -> list[int]:
    """Card indices matching the filters, in ``order`` (default: natural order)."""
    seq = range(len(CARDS)) if order is None else order
    return [i for i in seq if _passes(CARDS[i], gen, judge, lang, decision, safety, answer)]


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
    # Use <div> (not <p>): the tooltip holds block HTML (<p>/<ul>), and a nested
    # block inside a <p> auto-closes it, ejecting the first citation's tooltip.
    refl = _wrap_citations(c.get("reflection_1p") or "(none)")
    cites = _wrap_citations(" ".join(c.get("charter_elements") or []) or "—")
    out = [
        f"<h3 style='margin:.3em 0'>Reflection (first-person)</h3><div>{refl}</div>",
        f"<div style='margin-top:.5em'><b>Charter citations:</b> {cites}</div>",
    ]
    if c.get("analysis"):
        out.append(f"<div style='margin-top:.5em'><b>Analysis:</b> {html.escape(c['analysis'])}</div>")
    return "".join(out)


def _judge_html(c: dict) -> str:
    scores = c.get("judge_scores") or {}
    dims = "  ·  ".join(f"{html.escape(k)} <b>{v}</b>" for k, v in scores.items()) or "—"
    decision = html.escape((c.get("judge_decision") or "—").upper())
    agg = c.get("judge_aggregate")
    agg_s = f"{agg:.2f}" if isinstance(agg, (int, float)) else "—"
    out = [
        f"<div><b>{decision}</b> · aggregate <b>{agg_s}</b> · "
        f"judged by <code>{html.escape(c.get('judge_model') or '')}</code></div>",
        f"<div style='margin-top:.3em'>{dims}</div>",
    ]
    if c.get("judge_reasoning"):
        # <blockquote> (not <p>) so the citation tooltips' block content stays nested.
        out.append(
            "<blockquote style='border-left:3px solid #888;margin:.5em 0;padding-left:.6em'>"
            f"{_wrap_citations(c['judge_reasoning'])}</blockquote>"
        )
    return "".join(out)


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
    return meta, _doc_value(c), _refl_html(c), _judge_html(c), f"{pos + 1} / {len(idxs)}"


def _card(idxs, pos, status_msg=""):
    """Render card `pos`, reset the feedback inputs, and collapse the judge accordion.

    Returns values for CARD_OUT: meta, doc, refl, judge, poslabel, verdict(None),
    reason(""), judge_accordion(collapsed), status.
    """
    meta, doc, refl, judge_html, poslabel = render(idxs, pos)
    return meta, doc, refl, judge_html, poslabel, None, "", gr.update(open=False), status_msg


def apply_filters(order, gen, judge, lang, decision, safety, answer):
    idxs = filter_indices(gen, judge, lang, decision, safety, answer, order=order)
    return (idxs, 0, *_card(idxs, 0))


def step(idxs, pos, delta):
    new_pos = max(0, min(pos + delta, max(0, len(idxs) - 1)))
    return (new_pos, *_card(idxs, new_pos))


# ----------------------------------------------------------------- feedback


def _safe_name(s: str) -> str:
    """Filesystem/repo-safe stem (ts is kept first so files sort chronologically)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:120]


def submit_feedback(idxs, pos, reviewer, verdict, reason):
    """Save the verdict, then advance to the next card (resets feedback inputs).

    Returns values for [pos_state, *CARD_OUT]. On a validation error nothing
    moves — only the status message updates.
    """
    if not idxs:
        return (gr.update(),) * 9 + ("Nothing to rate.",)
    if not verdict:
        return (gr.update(),) * 9 + ("Pick accept or reject first.",)
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
    FEEDBACK_FILE.open("a", encoding="utf-8").write(line)  # local copy
    if not FEEDBACK_DATASET:
        msg = f"Saved “{record['verdict']}” ✓ (local: {FEEDBACK_FILE})"
    else:
        # Commit each review immediately as its own file — robust on ephemeral Spaces
        # (CommitScheduler's 5-min batch can be lost when a free Space sleeps/restarts).
        from huggingface_hub import HfApi

        stem = _safe_name(f"{record['ts']}-{record['reviewer']}-{record['item_id']}")
        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=line.encode("utf-8"),
            path_in_repo=f"data/{stem}.jsonl",
            repo_id=FEEDBACK_DATASET,
            repo_type="dataset",
            commit_message=f"feedback: {record['verdict']} by {record['reviewer']}",
        )
        msg = f"Saved “{record['verdict']}” ✓ → {FEEDBACK_DATASET}"
    new_pos = max(0, min(pos + 1, max(0, len(idxs) - 1)))
    return (new_pos, *_card(idxs, new_pos, msg))


# ----------------------------------------------------------------- layout

# Citation tooltip rendered at <body> level (#cite-tip) via JS, so it is NEVER
# clipped by an ancestor's overflow (e.g. the judge accordion frame). The inline
# `.tip` span is just a hidden data carrier the JS copies its content from.
# (Gradio has no content tooltip, strips `title`, and launch(css/js=) isn't
# reachable on a Space — but per-event demo.load(js=...) is.)
_CSS = (
    ".cite{border-bottom:1px dotted #888;cursor:help}"
    ".cite .tip{display:none}"
    "#cite-tip{position:fixed;z-index:9999;display:none;width:380px;max-width:90vw;"
    "max-height:280px;overflow-y:auto;text-align:left;background:#111827;color:#fff;"
    "padding:6px 10px;border-radius:6px;font-size:.74rem;line-height:1.3;"
    "box-shadow:0 6px 20px rgba(0,0,0,.55)}"
    "#cite-tip *{color:#fff}"
    "#cite-tip p{margin:.3em 0}"
    "#cite-tip ul,#cite-tip ol{margin:.3em 0;padding-left:1.1em}"
    "#cite-tip li{margin:.1em 0}"
    "#cite-tip h1,#cite-tip h2,#cite-tip h3,#cite-tip h4{font-size:.8rem;margin:.35em 0}"
    "#cite-tip hr.tipsep{margin:6px 0;border:0;border-top:1px solid #444}"
)

# Runs once on load: a delegated handler that pops the hovered citation's hidden
# `.tip` content into a body-level #cite-tip box positioned near the citation.
_TOOLTIP_JS = """
() => {
  if (window.__citeTipInit) return; window.__citeTipInit = true;
  const tip = document.createElement('div'); tip.id = 'cite-tip';
  document.body.appendChild(tip);
  let over = false;
  tip.addEventListener('mouseenter', () => { over = true; });
  tip.addEventListener('mouseleave', () => { over = false; tip.style.display = 'none'; });
  document.addEventListener('mouseover', (e) => {
    const cite = e.target.closest ? e.target.closest('.cite') : null;
    if (!cite) return;
    const src = cite.querySelector('.tip'); if (!src) return;
    tip.innerHTML = src.innerHTML; tip.style.display = 'block';
    const r = cite.getBoundingClientRect();
    const tw = Math.min(380, window.innerWidth * 0.9);
    tip.style.left = Math.max(8, Math.min(r.left, window.innerWidth - tw - 8)) + 'px';
    const top = r.top - tip.offsetHeight - 6;
    tip.style.top = (top < 8 ? r.bottom + 6 : top) + 'px';
  });
  document.addEventListener('mouseout', (e) => {
    const cite = e.target.closest ? e.target.closest('.cite') : null;
    if (!cite) return;
    setTimeout(() => { if (!over) tip.style.display = 'none'; }, 150);
  });
}
"""


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

            # Judge's call is hidden by default so it doesn't anchor the reviewer;
            # sits just before the feedback panel and re-collapses on each new card.
            with gr.Accordion("Reveal judge verdict (score · decision · reasoning)", open=False) as judge_acc:
                judge_md = gr.HTML()

            gr.Markdown("### Your feedback")
            verdict = gr.Radio(["✅ accept", "❌ reject"], label="Your verdict")
            reason = gr.Textbox(label="Reason (optional)", lines=2)
            submit_btn = gr.Button("Submit feedback", variant="primary")
            status = gr.Markdown()

        reviewer_state = gr.State("")
        order_state = gr.State(list(range(len(CARDS))))
        idxs_state = gr.State(list(range(len(CARDS))))
        pos_state = gr.State(0)

        VIEW = [meta_md, doc_box, refl_html, judge_md, poslabel]
        # What every card-change handler writes: the view, then reset feedback inputs,
        # the (collapsed) judge accordion, and the status line.
        CARD_OUT = [*VIEW, verdict, reason, judge_acc, status]

        def start(name):
            nm = (name or "").strip()
            noop = gr.update()
            if not nm:
                # gate, main, gate_msg, reviewer, order, idxs, pos, + 5 view comps = 12
                return (noop, noop, "Please enter your name.") + (noop,) * 9
            order = annotator_order(nm)
            view = render(order, 0)
            return (
                gr.update(visible=False), gr.update(visible=True),
                f"Reviewing as **{nm}** — you have your own card order",
                nm, order, order, 0, *view,
            )

        start_btn.click(
            start,
            inputs=[name_in],
            outputs=[gate, main_panel, gate_msg,
                     reviewer_state, order_state, idxs_state, pos_state, *VIEW],
        )

        filters = [gen, judge, lang, decision, safety, answer]
        for f in filters:
            f.change(
                apply_filters,
                inputs=[order_state, *filters],
                outputs=[idxs_state, pos_state, *CARD_OUT],
            )
        prev_btn.click(
            lambda i, p: step(i, p, -1),
            inputs=[idxs_state, pos_state], outputs=[pos_state, *CARD_OUT],
        )
        next_btn.click(
            lambda i, p: step(i, p, 1),
            inputs=[idxs_state, pos_state], outputs=[pos_state, *CARD_OUT],
        )
        submit_btn.click(
            submit_feedback,
            inputs=[idxs_state, pos_state, reviewer_state, verdict, reason],
            outputs=[pos_state, *CARD_OUT],
        )
        demo.load(
            lambda: render(list(range(len(CARDS))), 0),
            outputs=VIEW,
        )
        demo.load(js=_TOOLTIP_JS)  # body-level citation tooltip (escapes clipping)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("DASHBOARD_PORT", 7860)))
