"""Phase 2 dashboard pages: /pipeline and /pipeline/review routes."""

from __future__ import annotations

import difflib
import json
import logging
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path

from nicegui import app, ui

logger = logging.getLogger(__name__)

from pipeline.config import AppConfig, load_config, resolve_prompt_path
from pipeline.backup import force_upload
from pipeline.dashboard import render_header
from pipeline.dashboard.shared import CHARTER_TEXT, render_source_text
from pipeline.generation import (
    MODE_PART_NAMES as _MODE_PART_NAMES,
    PREFLECTION_FIELDS_CURRENT as _PREFLECTION_FIELDS_CURRENT,
    PREFLECTION_PART_NAMES as _PREFLECTION_PART_NAMES,
    REFLECTION_PART_NAMES as _REFLECTION_PART_NAMES,
    detect_mode_voices as _detect_mode_voices,
)
from pipeline.charter.improve.loop import parse_improver_key
from pipeline.charter.improve.run import JUDGMENT_NON_PART_KEYS
from pipeline.charter.improve.storage import (
    build_review_lookup,
    delete_review,
    delete_review_comment,
    load_item_across_iterations,
    load_items_for_iteration,
    load_judge_correlations,
    load_latest_items,
    load_latest_reviews,
    load_loop_history,
    load_review_comments,
    load_reviews,
    load_runs,
    load_test_results,
    review_split,
    save_review,
    save_review_comment,
)


def _compute_calibration(reviews: list[dict], items_by_key: dict) -> dict:
    """Compute judge-vs-human calibration metrics.

    Returns per-dimension correlations, aggregate correlation, and decision agreement.
    """
    paired_scores: dict[str, list[tuple[float, float]]] = {}
    aggregate_pairs: list[tuple[float, float]] = []
    decision_pairs: list[tuple[str, str]] = []

    for review in reviews:
        key = (review["item_id"], review["iteration"])
        item = items_by_key.get(key)
        if not item or not item.get("judgment"):
            continue
        judgment = item["judgment"]

        review_scores = review["scores"]
        # Detect per-part vs legacy flat format
        is_per_part = review_scores and isinstance(
            next(iter(review_scores.values())), dict
        )

        for part, part_j in judgment.items():
            if part in JUDGMENT_NON_PART_KEYS or not isinstance(part_j, dict):
                continue
            part_scores = part_j.get("scores", {})
            human_part = review_scores.get(part, {}) if is_per_part else review_scores
            for dim, human_score in human_part.items():
                if dim in part_scores:
                    paired_scores.setdefault(f"{part}_{dim}", []).append(
                        (part_scores[dim], human_score)
                    )

        aggregate_pairs.append((judgment.get("aggregate", 0), review["aggregate"]))
        decision_pairs.append((judgment.get("decision", ""), review["decision"]))

    dim_corr = {dim: _pearson(pairs) for dim, pairs in paired_scores.items()}
    agg_corr = _pearson(aggregate_pairs)
    agreement = (
        sum(1 for j, h in decision_pairs if j == h) / len(decision_pairs)
        if decision_pairs
        else None
    )

    return {
        "dimension_correlations": dim_corr,
        "aggregate_correlation": agg_corr,
        "decision_agreement": agreement,
        "n_reviews": len(reviews),
        "n_paired": len(aggregate_pairs),
    }


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    """Compute Pearson correlation coefficient from (x, y) pairs."""
    if len(pairs) < 3:
        return None
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx = statistics.mean(x)
    my = statistics.mean(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = (sum((xi - mx) ** 2 for xi in x)) ** 0.5
    sy = (sum((yi - my) ** 2 for yi in y)) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def _cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Compute Cohen's kappa for binary decision agreement.

    pairs: list of (rater_a, rater_b) label strings.
    Returns kappa in [-1, 1], or None if fewer than 2 pairs or zero variance.
    """
    if len(pairs) < 2:
        return None
    n = len(pairs)
    labels = sorted({l for p in pairs for l in p})
    if len(labels) < 2:
        return None
    counts = {(a, b): 0 for a in labels for b in labels}
    for a, b in pairs:
        counts[(a, b)] += 1
    p_o = sum(counts[(l, l)] for l in labels) / n
    p_e = sum(
        sum(counts[(l, b)] for b in labels) * sum(counts[(a, l)] for a in labels)
        for l in labels
    ) / (n * n)
    if p_e == 1.0:
        return None
    return (p_o - p_e) / (1 - p_e)


def _load_correlation_pairs(
    split: str | None = None, mode: str | None = None
) -> list[dict]:
    """Load and join (judge, human) calibration pairs.

    Combines two sources:
      1. Original judgments stored in the items table — every item carries the
         judgment produced when it was generated, by the (judge_prompt,
         judge_model) recorded on its run. This is the canonical source for
         items judged by the prompt that was current at generation time.
      2. Retroactive re-judgments stored in the judge_correlations table —
         used when an item was originally judged by an *older* prompt and a
         newer prompt was applied later via `rejudge_all`.

    The two sources are deduplicated by (judge_prompt, judge_model, item_id,
    iteration); the items table wins when both sources cover the same key.

    Each returned record has: judge_prompt, judge_model, iteration, item_id,
    judge_agg, human_agg, judge_dec, human_dec.

    split: "train", "validation", or None (all).
    mode: "reflection", "preflection", or None (combined).
      When set, uses per-mode aggregate/decision keys and per-mode judge prompt.
    """
    review_by_item = build_review_lookup(split=split)
    if not review_by_item:
        return []

    runs = load_runs()
    if mode:
        _jp_field = f"judge_{mode}_prompt"
        run_judge_info: dict[int, tuple[str, str]] = {
            r["iteration"]: (
                r.get(_jp_field) or r["judge_prompt"],
                r["judge_model"],
            )
            for r in runs
        }
    else:
        run_judge_info: dict[int, tuple[str, str]] = {
            r["iteration"]: (r["judge_prompt"], r["judge_model"]) for r in runs
        }

    pairs: list[dict] = []
    seen: set[tuple[str, str, str, int]] = set()

    # Source 1: original judgments from the items table
    all_items = load_latest_items()
    for (item_id, iteration), rev in review_by_item.items():
        item = all_items.get((item_id, iteration))
        if not item or not item.get("judgment"):
            continue
        judge_info = run_judge_info.get(iteration)
        if not judge_info:
            continue
        judge_prompt, judge_model = judge_info
        j = item["judgment"]
        if mode:
            j_agg, j_dec = _derive_mode_agg_dec(j, mode)
            h_agg, h_dec = _derive_mode_review_agg_dec(rev, mode)
            if j_agg is None or j_dec is None:
                continue
            if h_agg is None or h_dec is None:
                # Fall back to combined review decision
                h_agg = rev.get("aggregate", 0)
                h_dec = rev.get("decision", "")
        else:
            j_agg = j.get("aggregate", 0)
            j_dec = j.get("decision", "")
            h_agg = rev.get("aggregate", 0)
            h_dec = rev.get("decision", "")
        pairs.append(
            {
                "judge_prompt": judge_prompt,
                "judge_model": judge_model,
                "iteration": iteration,
                "item_id": item_id,
                "judge_agg": j_agg,
                "human_agg": h_agg,
                "judge_dec": j_dec or "",
                "human_dec": h_dec or "",
            }
        )
        seen.add((judge_prompt, judge_model, item_id, iteration))

    # Source 2: retroactive re-judgments from judge_correlations
    for c in load_judge_correlations():
        key = (c["judge_prompt"], c["judge_model"], c["item_id"], c["iteration"])
        if key in seen:
            continue
        rev = review_by_item.get((c["item_id"], c["iteration"]))
        if not rev:
            continue
        cj = c["judgment"]
        if mode:
            cj_agg, cj_dec = _derive_mode_agg_dec(cj, mode)
            ch_agg, ch_dec = _derive_mode_review_agg_dec(rev, mode)
            if cj_agg is None or cj_dec is None:
                continue
            if ch_agg is None or ch_dec is None:
                ch_agg = rev.get("aggregate", 0)
                ch_dec = rev.get("decision", "")
        else:
            cj_agg = cj.get("aggregate", 0)
            cj_dec = cj.get("decision", "")
            ch_agg = rev.get("aggregate", 0)
            ch_dec = rev.get("decision", "")
        pairs.append(
            {
                "judge_prompt": c["judge_prompt"],
                "judge_model": c["judge_model"],
                "iteration": c["iteration"],
                "item_id": c["item_id"],
                "judge_agg": cj_agg,
                "human_agg": ch_agg,
                "judge_dec": cj_dec or "",
                "human_dec": ch_dec or "",
            }
        )

    # When mode is set, drop pairs from old combined prompts (judge_v*.md)
    # that predate the per-mode split — they're irrelevant noise.
    if mode and pairs:
        _mode_prefix = f"judge_{mode}_v"
        pairs = [p for p in pairs if p["judge_prompt"].startswith(_mode_prefix)]

    return pairs


def _aggregate_correlation_pairs(
    pairs: list[dict],
) -> dict[tuple[str, str], dict]:
    """Aggregate pre-loaded pairs into per-version correlation metrics."""
    score_pairs: dict[tuple[str, str], list[tuple[float, float]]] = {}
    decision_pairs: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for p in pairs:
        vk = (p["judge_prompt"], p["judge_model"])
        score_pairs.setdefault(vk, []).append((p["judge_agg"], p["human_agg"]))
        decision_pairs.setdefault(vk, []).append((p["judge_dec"], p["human_dec"]))

    def _agreement(dps: list[tuple[str, str]]) -> float | None:
        if not dps:
            return None
        return sum(1 for a, b in dps if a == b) / len(dps)

    return {
        vk: {
            "pearson": _pearson(score_pairs[vk]),
            "kappa": _cohens_kappa(decision_pairs.get(vk, [])),
            "agreement": _agreement(decision_pairs.get(vk, [])),
        }
        for vk in score_pairs
    }


def _compute_correlation_by_judge_version(
    gen_filter: str | None = None,
    iter_to_gen: dict[int, str] | None = None,
    _pairs: list[dict] | None = None,
) -> dict[tuple[str, str], dict]:
    """Compute calibration metrics for each (judge_prompt, judge_model).

    Uses judge_correlations table (re-judgments) paired with human reviews.
    Returns {(judge_prompt, judge_model): {"pearson": float|None,
    "kappa": float|None, "agreement": float|None}}, where agreement is the
    raw decision-match proportion in [0, 1].

    Pass _pairs (from _load_correlation_pairs) to avoid reloading from DB.
    """
    pairs = _pairs if _pairs is not None else _load_correlation_pairs()
    if gen_filter and iter_to_gen:
        pairs = [p for p in pairs if iter_to_gen.get(p["iteration"]) == gen_filter]
    return _aggregate_correlation_pairs(pairs)


def _render_judge_scores(judgment: dict, judge_model: str = "") -> None:
    """Render judge score details for a single judgment dict."""
    overall_agg = judgment.get("aggregate", 0)
    overall_dec = judgment.get("decision", "reject")
    with ui.row().classes("gap-4"):
        ui.badge(
            f"Aggregate: {overall_agg:.1f}",
            color="green" if overall_dec == "accept" else "red",
        )
        ui.badge(
            overall_dec.upper(),
            color="green" if overall_dec == "accept" else "red",
        )
        if judge_model:
            ui.badge(judge_model, color="teal").props("outline")
        jp = judgment.get("judge_prompt", "")
        if jp:
            ui.badge(jp, color="blue-grey").props("outline")

    # Show per-mode decisions if available
    for mode, mode_label in (
        ("reflection", "Reflection"),
        ("preflection", "Preflection"),
    ):
        mode_dec = judgment.get(f"{mode}_decision")
        mode_agg = judgment.get(f"{mode}_aggregate")
        if mode_dec is not None:
            dec_color = "green" if mode_dec == "accept" else "red"
            with ui.row().classes("gap-2 q-mt-xs"):
                ui.badge(
                    (
                        f"{mode_label}: {mode_agg:.1f}"
                        if mode_agg is not None
                        else mode_label
                    ),
                    color=dec_color,
                ).props("outline")
                ui.badge(f"{mode_dec.upper()}", color=dec_color)
                jp_mode = judgment.get(f"judge_prompt_{mode}", "")
                if jp_mode:
                    ui.badge(jp_mode, color="blue-grey").props("outline")

    for part, part_j in judgment.items():
        if part in JUDGMENT_NON_PART_KEYS or not isinstance(part_j, dict):
            continue
        if not part_j.get("scores"):
            continue
        ui.label(f"{part} ({part_j.get('aggregate', 0):.1f})").classes(
            "text-overline text-grey-7 q-mt-sm"
        )
        with ui.row().classes("gap-2"):
            for dim, score in part_j.get("scores", {}).items():
                color = "green" if score >= 4 else ("orange" if score >= 3 else "red")
                ui.badge(f"{dim}: {score}", color=color)
        ui.label(part_j.get("reasoning", "")).classes("text-body2").style(
            "white-space: pre-wrap; font-size: 0.9em;"
        )


def _status_badge(status: str) -> tuple[str, str]:
    """Return (label, color) for an improver status badge."""
    colors = {
        "pending": "grey",
        "running": "blue",
        "done": "green",
        "error": "red",
        "skipped": "orange",
    }
    return status.upper(), colors.get(status, "grey")


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _word_diff_html(old_line: str, new_line: str) -> tuple[str, str]:
    """Produce word-level highlighted HTML for a changed line pair.

    Returns (old_html, new_html) with inline <span> highlights on the
    words that actually differ.
    """
    old_words = old_line.split(" ")
    new_words = new_line.split(" ")
    sm = difflib.SequenceMatcher(None, old_words, new_words)

    old_parts: list[str] = []
    new_parts: list[str] = []
    DEL = '<span style="background:#ffc9c9;border-radius:3px;padding:0 2px;">{}</span>'
    INS = '<span style="background:#a8e6a1;border-radius:3px;padding:0 2px;">{}</span>'

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            old_parts.append(_esc(" ".join(old_words[i1:i2])))
            new_parts.append(_esc(" ".join(new_words[j1:j2])))
        elif tag == "replace":
            old_parts.append(DEL.format(_esc(" ".join(old_words[i1:i2]))))
            new_parts.append(INS.format(_esc(" ".join(new_words[j1:j2]))))
        elif tag == "delete":
            old_parts.append(DEL.format(_esc(" ".join(old_words[i1:i2]))))
        elif tag == "insert":
            new_parts.append(INS.format(_esc(" ".join(new_words[j1:j2]))))

    return " ".join(old_parts), " ".join(new_parts)


def _prompt_diff_html(before: str, after: str, filename: str) -> str:
    """Generate a side-by-side diff with word-level highlighting."""
    before_lines = before.splitlines()
    after_lines = after.splitlines()

    n_add = n_del = 0

    # Collect aligned row pairs: (left_ln, left_html, left_bg, right_ln, right_html, right_bg)
    rows: list[tuple] = []
    CTX_BG = "#ffffff"
    DEL_BG = "rgba(255,129,130,0.15)"
    INS_BG = "rgba(63,185,80,0.15)"

    for group in difflib.SequenceMatcher(
        None, before_lines, after_lines
    ).get_grouped_opcodes(8):
        for tag, a0, a1, b0, b1 in group:
            if tag == "equal":
                for i in range(a1 - a0):
                    rows.append(
                        (
                            a0 + i + 1,
                            _esc(before_lines[a0 + i]),
                            CTX_BG,
                            b0 + i + 1,
                            _esc(after_lines[b0 + i]),
                            CTX_BG,
                        )
                    )
            elif tag == "replace":
                old_lines = before_lines[a0:a1]
                new_lines = after_lines[b0:b1]
                # Pair up lines for word-level diff, pad shorter side
                max_len = max(len(old_lines), len(new_lines))
                for i in range(max_len):
                    if i < len(old_lines) and i < len(new_lines):
                        old_html, new_html = _word_diff_html(old_lines[i], new_lines[i])
                        rows.append(
                            (a0 + i + 1, old_html, DEL_BG, b0 + i + 1, new_html, INS_BG)
                        )
                        n_del += 1
                        n_add += 1
                    elif i < len(old_lines):
                        rows.append(
                            (a0 + i + 1, _esc(old_lines[i]), DEL_BG, "", "", INS_BG)
                        )
                        n_del += 1
                    else:
                        rows.append(
                            ("", "", DEL_BG, b0 + i + 1, _esc(new_lines[i]), INS_BG)
                        )
                        n_add += 1
            elif tag == "delete":
                for i, line in enumerate(before_lines[a0:a1]):
                    rows.append((a0 + i + 1, _esc(line), DEL_BG, "", "", INS_BG))
                    n_del += 1
            elif tag == "insert":
                for i, line in enumerate(after_lines[b0:b1]):
                    rows.append(("", "", DEL_BG, b0 + i + 1, _esc(line), INS_BG))
                    n_add += 1

    if not rows:
        return "<em>No changes</em>"

    # Build stats bar
    stats = (
        '<div style="padding:8px 14px;border-bottom:1px solid #d0d7de;display:flex;'
        'align-items:center;gap:12px;background:#f6f8fa;">'
        f'<span style="font-weight:600;color:#1f2328;">{_esc(filename)}</span>'
        f'<span style="background:#1a7f37;color:#fff;padding:1px 8px;border-radius:10px;'
        f'font-size:0.8em;font-weight:600;">+{n_add}</span>'
        f'<span style="background:#cf222e;color:#fff;padding:1px 8px;border-radius:10px;'
        f'font-size:0.8em;font-weight:600;">-{n_del}</span>'
        "</div>"
    )

    # Cell styles
    LN = (
        "padding:1px 8px;color:#636c76;text-align:right;user-select:none;"
        "vertical-align:top;white-space:nowrap;width:1%;font-size:0.85em;"
    )
    CELL = (
        "padding:1px 10px;white-space:pre-wrap;word-break:break-word;"
        "vertical-align:top;line-height:1.55;"
    )
    SEP = "width:1px;background:#d0d7de;"

    html_rows: list[str] = []
    for left_ln, left_html, left_bg, right_ln, right_html, right_bg in rows:
        html_rows.append(
            f"<tr>"
            f'<td style="{LN}background:{left_bg};">{left_ln}</td>'
            f'<td style="{CELL}background:{left_bg};">{left_html}</td>'
            f'<td style="{SEP}"></td>'
            f'<td style="{LN}background:{right_bg};">{right_ln}</td>'
            f'<td style="{CELL}background:{right_bg};">{right_html}</td>'
            f"</tr>"
        )

    # Column headers
    header = (
        '<tr style="border-bottom:1px solid #d0d7de;background:#f6f8fa;">'
        '<td colspan="2" style="padding:6px 10px;color:#636c76;font-weight:600;'
        'font-size:0.85em;text-align:center;">Before</td>'
        f'<td style="{SEP}"></td>'
        '<td colspan="2" style="padding:6px 10px;color:#636c76;font-weight:600;'
        'font-size:0.85em;text-align:center;">After</td>'
        "</tr>"
    )

    table = (
        f"<table style=\"width:100%;border-collapse:collapse;font-family:'SF Mono',"
        f"Menlo,Consolas,monospace;font-size:0.82em;table-layout:fixed;"
        f'background:#ffffff;color:#1f2328;">'
        f'<colgroup><col style="width:30px"><col style="width:calc(50% - 30px)">'
        f'<col style="width:1px"><col style="width:30px"><col style="width:calc(50% - 30px)">'
        f'</colgroup>{header}{"".join(html_rows)}</table>'
    )
    return stats + table


def _find_predecessor(filename: str, available: dict[str, str]) -> str | None:
    """Find the previous version of a versioned prompt file.

    E.g. 'judge_v3.md' -> look for 'judge_v2.md' in available.
    """
    import re

    m = re.match(r"^(.+)_v(\d+)(\.md)$", filename)
    if not m:
        return None
    prefix, version, ext = m.group(1), int(m.group(2)), m.group(3)
    prev = f"{prefix}_v{version - 1}{ext}"
    return prev if prev in available else None


def _compute_prompt_diffs(
    before: dict[str, str],
    after: dict[str, str],
) -> list[tuple[str, str, str, str]]:
    """Compute meaningful diffs between prompt snapshots.

    For new versioned files (e.g. judge_v3.md), diffs against the previous
    version (judge_v2.md) rather than against empty string.

    Returns list of (expansion_label, before_text, after_text, display_name).
    """
    new_files = sorted(set(after) - set(before))
    modified = [f for f in sorted(set(before) & set(after)) if before[f] != after[f]]

    diffs: list[tuple[str, str, str, str]] = []

    for filename in new_files:
        pred = _find_predecessor(filename, before)
        if pred:
            label = f"{pred} -> {filename}"
            diffs.append((label, before[pred], after[filename], label))
        else:
            diffs.append((f"{filename} (new)", "", after[filename], filename))

    for filename in modified:
        diffs.append((filename, before[filename], after[filename], filename))

    return diffs


def _render_loop_history():
    """Render past improver runs from loop_history table."""
    history = load_loop_history()
    if not history:
        return

    with ui.expansion(
        f"Improver History ({len(history)} runs)",
        icon="history",
    ).classes("w-full q-mx-md q-mt-md"):
        for i, run in enumerate(reversed(history)):
            run_idx = len(history) - i
            started = run.get("started_at", "?")[:19]
            finished = run.get("finished_at", "?")[:19]
            error = run.get("error")
            failed = bool(error)
            role = run.get("role", "unknown")
            status_tag = " — FAILED" if failed else ""
            border_style = "border-left: 3px solid #f44336;" if failed else ""

            with (
                ui.expansion(
                    f"Run #{run_idx} ({role}) — {started}{status_tag}",
                    icon="error" if failed else "history",
                )
                .classes("w-full")
                .style(border_style)
            ):
                with ui.row().classes("items-center gap-2"):
                    ui.badge(
                        "FAILED" if failed else "DONE",
                        color="red" if failed else "green",
                    )
                    ui.badge(role.upper(), color="blue-grey").props("outline")
                    ui.label(f"{started} → {finished}").classes(
                        "text-caption text-grey-6"
                    )

                if error:
                    ui.label(f"Error: {error}").classes("text-caption text-red q-mt-xs")

                # Improver cards with reasoning + logs
                improvers = run.get("improvers", {})
                run_logs = run.get("logs", {})
                with ui.row().classes("w-full gap-4 q-mt-sm flex-wrap"):
                    for key, data in improvers.items():
                        imp_role, imp_mode, imp_alias = parse_improver_key(key)
                        imp_status = data.get("status", "pending")
                        label, color = _status_badge(imp_status)
                        with (
                            ui.card()
                            .classes("flex-1 q-pa-sm")
                            .style("min-width: 300px;")
                        ):
                            with ui.row().classes("items-center gap-2"):
                                _imp_display = f"{imp_role.title()}"
                                if imp_mode:
                                    _imp_display += f" ({imp_mode})"
                                _imp_display += f": {imp_alias}"
                                ui.label(_imp_display).classes(
                                    "text-subtitle2 text-weight-bold"
                                )
                                ui.badge(label, color=color)
                            reasoning = data.get("reasoning", "")
                            if reasoning:
                                with ui.expansion(
                                    "Summary", icon="summarize", value=True
                                ).classes("w-full"):
                                    ui.markdown(reasoning).classes("text-body2").style(
                                        "font-size: 0.85em; max-height: 300px; overflow-y: auto;"
                                    )
                            else:
                                ui.label("No reasoning recorded.").classes(
                                    "text-grey-6 text-caption"
                                )

                            imp_log = run_logs.get(key, "")
                            if imp_log:
                                with ui.expansion("Full Log", icon="terminal").classes(
                                    "w-full"
                                ):
                                    ui.code(imp_log, language="text").classes(
                                        "w-full"
                                    ).style(
                                        "max-height: 400px; overflow-y: auto; font-size: 0.75em;"
                                    )

                # Prompt diffs
                diffs = _compute_prompt_diffs(
                    run.get("prompts_before", {}),
                    run.get("prompts_after", {}),
                )
                if diffs:
                    with ui.expansion(
                        f"Prompt Changes ({len(diffs)} file{'s' if len(diffs) != 1 else ''})",
                        icon="difference",
                    ).classes("w-full q-mt-sm"):
                        for label, before_text, after_text, display_name in diffs:
                            with ui.expansion(label).classes("w-full"):
                                html = _prompt_diff_html(
                                    before_text, after_text, display_name
                                )
                                ui.html(
                                    f'<div style="background:#0d1117;border:1px solid #30363d;'
                                    f"border-radius:6px;overflow:hidden;max-height:600px;"
                                    f'overflow-y:auto;">{html}</div>'
                                )
                elif run.get("prompts_before"):
                    ui.label("No prompt changes in this run.").classes(
                        "text-grey-6 text-caption q-mt-xs"
                    )


def _build_part_display(item: dict) -> list[tuple[str, str]]:
    """Return ``(label, text)`` pairs covering all annotation variants in *item*.

    Items span three preflection schema generations (1-col legacy,
    2-voice legacy, 4-field current) and may have a reflection present even
    when preflection failed (partial-success path in ``generate_batch``).
    Each group is detected independently, so a reflection-only item renders
    its reflection voices rather than falling back to empty legacy columns.
    """
    parts: list[tuple[str, str]] = []
    if item.get("neutral") is not None or item.get("charter_summary") is not None:
        parts.extend(
            (f, item.get(f, "") or "") for f in _PREFLECTION_FIELDS_CURRENT
        )
    elif item.get("preflection_1p") is not None:
        parts.extend(
            [
                (
                    "preflection_3p",
                    item.get("preflection_3p") or item.get("preflection", "") or "",
                ),
                ("preflection_1p", item.get("preflection_1p", "") or ""),
            ]
        )
    elif item.get("preflection"):
        parts.append(("preflection", item.get("preflection", "")))

    if item.get("reflection_1p") is not None or item.get("reflection_3p") is not None:
        parts.extend(
            [
                (
                    "reflection_1p",
                    item.get("reflection_1p") or item.get("reflection", "") or "",
                ),
                ("reflection_3p", item.get("reflection_3p", "") or ""),
            ]
        )
    elif item.get("reflection"):
        parts.append(("reflection", item.get("reflection", "")))

    return parts


def _derive_mode_agg_dec(
    judgment: dict, mode: str, accept_threshold: float = 4.0, floor_threshold: int = 2
) -> tuple[float | None, str | None]:
    """Extract per-mode aggregate and decision from a judgment dict.

    Returns the explicit ``{mode}_aggregate`` / ``{mode}_decision`` if present.
    Falls back to computing from per-voice scores, detecting voices dynamically
    so old (2-voice preflection) and new (4-field preflection) judgments both
    work.
    """
    agg = judgment.get(f"{mode}_aggregate")
    dec = judgment.get(f"{mode}_decision")
    if agg is not None and dec is not None:
        return agg, dec

    voices = _detect_mode_voices(judgment, mode)
    if not voices:
        return None, None
    all_scores: list[float] = []
    for v in voices:
        part = judgment.get(v)
        if not isinstance(part, dict):
            return None, None
        scores = part.get("scores", {})
        if not scores:
            return None, None
        all_scores.extend(scores.values())

    if not all_scores:
        return None, None

    agg = sum(all_scores) / len(all_scores)
    has_floor = any(s <= floor_threshold for s in all_scores)
    dec = "reject" if has_floor or agg < accept_threshold else "accept"
    return agg, dec


def _derive_mode_review_agg_dec(
    review: dict, mode: str, accept_threshold: float = 4.0, floor_threshold: int = 2
) -> tuple[float | None, str | None]:
    """Extract per-mode aggregate and decision from a human review.

    Returns the explicit ``{mode}_aggregate`` / ``{mode}_decision`` if present.
    Falls back to computing from per-voice scores in the review.
    """
    agg = review.get(f"{mode}_aggregate")
    dec = review.get(f"{mode}_decision")
    if agg is not None and dec is not None:
        return agg, dec

    scores = review.get("scores", {})
    if not scores:
        return None, None

    # Per-part review scores: {"reflection_1p": {"dim": val, ...}, ...}
    is_per_part = isinstance(next(iter(scores.values())), dict)
    if not is_per_part:
        return None, None

    voices = _detect_mode_voices(scores, mode)
    if not voices:
        return None, None
    all_vals: list[float] = []
    for v in voices:
        part_scores = scores.get(v, {})
        if not part_scores:
            return None, None
        all_vals.extend(part_scores.values())

    if not all_vals:
        return None, None

    agg = sum(all_vals) / len(all_vals)
    has_floor = any(v <= floor_threshold for v in all_vals)
    dec = "reject" if has_floor or agg < accept_threshold else "accept"
    return agg, dec


_SOURCE_LABEL = {
    "improve_judge": "J",
    "improve_generator": "G",
    "manual": "\u2014",
    "phase_a": "J",
    "phase_b": "G",
}
_SOURCE_MARKER = {
    "improve_judge": "diamond",
    "improve_generator": "triangle",
    "manual": "circle",
    "phase_a": "diamond",
    "phase_b": "triangle",
}


def _render_mode_dashboard(
    mode: str,
    runs: list[dict],
    items_by_key: dict,
    all_reviews: list[dict],
    cfg: "AppConfig",
) -> None:
    """Render all per-mode panels for a single mode (reflection or preflection).

    Includes: overview card, judge calibration table, trend charts,
    per-generator acceptance rate bar chart, generator stats, calibration
    by judge model, and judge-judge correlations.
    """
    import re as _re_md

    dimensions = cfg.charter.improve.scoring.dimensions
    accept_threshold = cfg.charter.improve.scoring.accept_threshold
    floor_threshold = getattr(cfg.charter.improve.scoring, "floor_threshold", 2)
    # Union voice/field names across all items' judgments. Preflection spans
    # two schema generations (legacy 2-voice and current 4-field); a single
    # dashboard session may contain items from both.
    voice_set: set[str] = set()
    for i in items_by_key.values():
        voice_set.update(_detect_mode_voices(i.get("judgment") or {}, mode))
    voices = (
        tuple(sorted(voice_set))
        if voice_set
        else tuple(sorted(_MODE_PART_NAMES.get(mode, frozenset())))
    )
    jp_field = f"judge_{mode}_prompt"
    gp_field = f"gen_{mode}_prompt"

    # --- Compute mode-specific iter_stats ---
    # Uses _derive_mode_agg_dec to handle old judgments without explicit per-mode keys.
    mode_iter_stats: list[dict] = []
    for run in runs:
        it = run["iteration"]
        items = load_items_for_iteration(it)
        judged = [i for i in items if i.get("judgment")]
        n_acc = 0
        n_mode = 0
        agg_scores: list[float] = []
        for i in judged:
            m_agg, m_dec = _derive_mode_agg_dec(
                i["judgment"], mode, accept_threshold, floor_threshold
            )
            if m_agg is None:
                continue
            n_mode += 1
            if m_dec == "accept":
                n_acc += 1
            agg_scores.append(m_agg)
        mean_s = statistics.mean(agg_scores) if agg_scores else 0
        accept_rate = round(n_acc / n_mode * 100, 1) if n_mode else 0

        iter_reviews = [r for r in all_reviews if r["iteration"] == it]
        iter_items = {(i["item_id"], i["iteration"]): i for i in items}
        cal_iter = _compute_calibration(iter_reviews, iter_items)

        mode_iter_stats.append(
            {
                **run,
                "judge_prompt": run.get(jp_field) or run.get("judge_prompt") or "",
                "gen_prompt": run.get(gp_field) or run.get("gen_prompt") or "",
                "n_acc": n_acc,
                "n_rej": n_mode - n_acc,
                "mean_score": mean_s,
                "accept_rate": accept_rate,
                "calibration": cal_iter,
            }
        )

    # --- (a) Overview card ---
    all_items = list(items_by_key.values())
    judged_items = [i for i in all_items if i.get("judgment")]

    with ui.card().classes("w-full q-mt-sm q-pa-md"):
        ui.label(f"{mode.title()} Overview").classes("text-h6 text-weight-bold")

        # Accept rate (derived with fallback for old judgments)
        ov_n_acc = 0
        ov_total = 0
        for i in judged_items:
            m_agg, m_dec = _derive_mode_agg_dec(
                i["judgment"], mode, accept_threshold, floor_threshold
            )
            if m_agg is None:
                continue
            ov_total += 1
            if m_dec == "accept":
                ov_n_acc += 1
        if ov_total:
            ov_rate = round(ov_n_acc / ov_total * 100, 1)
            rate_text = f"{ov_rate}%  ({ov_n_acc}/{ov_total})"
        else:
            rate_text = "n/a"
        ui.label(f"Accept rate: {rate_text}").classes("text-body1 q-mt-xs")

        with ui.row().classes("w-full gap-8 q-mt-sm"):
            # Per-voice mean scores
            with ui.column():
                ui.label("Per-voice mean scores").classes("text-overline text-grey-7")
                for voice in voices:
                    aggs = [
                        i["judgment"][voice]["aggregate"]
                        for i in judged_items
                        if isinstance(i["judgment"].get(voice), dict)
                        and "aggregate" in i["judgment"][voice]
                    ]
                    mean_val = f"{statistics.mean(aggs):.2f}" if aggs else "n/a"
                    ui.label(f"  {voice}: {mean_val}  (n={len(aggs)})").classes(
                        "text-body2"
                    )

            # Per-dimension means
            with ui.column():
                ui.label("Per-dimension means").classes("text-overline text-grey-7")
                for dim in dimensions:
                    scores_for_dim: list[float] = []
                    for i in judged_items:
                        j = i["judgment"]
                        for voice in voices:
                            part = j.get(voice)
                            if isinstance(part, dict):
                                s = part.get("scores", {}).get(dim)
                                if s is not None:
                                    scores_for_dim.append(s)
                    mean_val = (
                        f"{statistics.mean(scores_for_dim):.2f}"
                        if scores_for_dim
                        else "n/a"
                    )
                    ui.label(f"  {dim}: {mean_val}  (n={len(scores_for_dim)})").classes(
                        "text-body2"
                    )

        # Judge prompt version
        judge_prompt_key = f"judge_prompt_{mode}"
        prompt_versions = {
            i["judgment"].get(judge_prompt_key, "")
            for i in judged_items
            if i["judgment"].get(judge_prompt_key)
        }
        if prompt_versions:

            def _extract_v(p: str) -> str:
                m = _re_md.search(r"_v(\d+)", p)
                return f"v{m.group(1)}" if m else p

            versions_str = ", ".join(sorted({_extract_v(p) for p in prompt_versions}))
        else:
            versions_str = "n/a"
        ui.label(f"Judge prompt: {versions_str}").classes(
            "text-caption text-grey-7 q-mt-sm"
        )

    # --- (c) Judge Calibration table ---
    with ui.card().classes("w-full q-mt-md q-pa-md"):
        ui.label("Judge Calibration").classes("text-h6 text-weight-bold")
        ui.label(
            "Latest judge prompt per model. Decision agreement (Cohen's \u03ba) and "
            "aggregate score correlation (Pearson r), by split."
        ).classes("text-caption text-grey-7")

        _all_pairs = _load_correlation_pairs(mode=mode)
        if not _all_pairs:
            _review_lookup = build_review_lookup()
            _n_reviews_total = len(_review_lookup)
            _n_train = sum(
                1 for (iid, _it) in _review_lookup if review_split(iid) == "train"
            )
            _n_val = sum(
                1 for (iid, _it) in _review_lookup if review_split(iid) == "validation"
            )

            with ui.column().classes("gap-1 q-mt-sm"):
                ui.label(
                    f"Reviews: {_n_reviews_total} total — "
                    f"{_n_train} train, {_n_val} validation"
                ).classes("text-body2")

                if _n_reviews_total == 0:
                    ui.label(
                        "Submit some reviews on /pipeline/review to populate "
                        "this panel — every reviewed item that already has a "
                        "judge score will be paired automatically."
                    ).classes("text-caption text-grey-6")
                else:
                    ui.label(
                        "Reviews exist but none of the reviewed items have a "
                        "stored judge score for this mode."
                    ).classes("text-caption text-grey-6")
        else:
            # Find latest judge prompt version per judge model
            _latest_prompt: dict[str, tuple[int, str]] = {}
            for _p in _all_pairs:
                _model = _p["judge_model"]
                _prompt = _p["judge_prompt"]
                _m = _re_md.search(r"_v(\d+)", _prompt)
                _v = int(_m.group(1)) if _m else 0
                if _model not in _latest_prompt or _v > _latest_prompt[_model][0]:
                    _latest_prompt[_model] = (_v, _prompt)

            def _fmt(x: float | None) -> str:
                return f"{x:.3f}" if x is not None else "\u2014"

            cal_rows: list[dict] = []
            for _model in sorted(_latest_prompt):
                _prompt = _latest_prompt[_model][1]
                _model_pairs = [
                    p
                    for p in _all_pairs
                    if p["judge_model"] == _model and p["judge_prompt"] == _prompt
                ]
                for split_name in ("overall", "train", "validation"):
                    if split_name == "overall":
                        sp = _model_pairs
                    else:
                        sp = [
                            p
                            for p in _model_pairs
                            if review_split(p["item_id"]) == split_name
                        ]
                    if not sp:
                        cal_rows.append(
                            {
                                "model": _model,
                                "prompt": _prompt,
                                "split": split_name,
                                "n": 0,
                                "kappa": "\u2014",
                                "pearson": "\u2014",
                            }
                        )
                        continue
                    agg = _aggregate_correlation_pairs(sp)
                    entry = agg.get((_prompt, _model), {})
                    cal_rows.append(
                        {
                            "model": _model,
                            "prompt": _prompt,
                            "split": split_name,
                            "n": len(sp),
                            "kappa": _fmt(entry.get("kappa")),
                            "pearson": _fmt(entry.get("pearson")),
                        }
                    )

            ui.table(
                columns=[
                    {
                        "name": "model",
                        "label": "Judge Model",
                        "field": "model",
                        "align": "left",
                    },
                    {
                        "name": "prompt",
                        "label": "Prompt",
                        "field": "prompt",
                        "align": "left",
                    },
                    {
                        "name": "split",
                        "label": "Split",
                        "field": "split",
                        "align": "left",
                    },
                    {"name": "n", "label": "n", "field": "n", "align": "right"},
                    {
                        "name": "kappa",
                        "label": "Decision \u03ba",
                        "field": "kappa",
                        "align": "right",
                    },
                    {
                        "name": "pearson",
                        "label": "Score r",
                        "field": "pearson",
                        "align": "right",
                    },
                ],
                rows=cal_rows,
                row_key="model",
            ).props("dense flat").classes("w-full")

    # --- (d) Trend Charts ---
    all_gen_models = sorted(
        {s.get("generator_model", "unknown") for s in mode_iter_stats}
    )

    if len(mode_iter_stats) >= 2:

        def _pv(s: str) -> int:
            m = _re_md.search(r"_v(\d+)", s or "")
            return int(m.group(1)) if m else 0

        def _build_trend_chart(gen_filter: str | None) -> dict:
            filtered = (
                mode_iter_stats
                if gen_filter is None
                else [
                    s for s in mode_iter_stats if s.get("generator_model") == gen_filter
                ]
            )
            labels = [
                f"J{_pv(s.get('judge_prompt', ''))}/G{_pv(s.get('gen_prompt', ''))}"
                for s in filtered
            ]
            accept_data = [
                {
                    "value": s["accept_rate"],
                    "symbol": _SOURCE_MARKER.get(s.get("source", "manual"), "circle"),
                    "symbolSize": 10,
                    "iter": s["iteration"],
                    "src": _SOURCE_LABEL.get(s.get("source", "manual"), ""),
                }
                for s in filtered
            ]
            score_data = [
                {
                    "value": round(s["mean_score"], 2),
                    "symbol": _SOURCE_MARKER.get(s.get("source", "manual"), "circle"),
                    "symbolSize": 10,
                    "iter": s["iteration"],
                    "src": _SOURCE_LABEL.get(s.get("source", "manual"), ""),
                }
                for s in filtered
            ]
            return {
                "xAxis": {"type": "category", "data": labels},
                "yAxis": [
                    {
                        "type": "value",
                        "name": "Accept %",
                        "min": 0,
                        "max": 100,
                        "position": "left",
                    },
                    {
                        "type": "value",
                        "name": "Mean Score",
                        "min": 1,
                        "max": 5,
                        "position": "right",
                    },
                ],
                "series": [
                    {
                        "name": "Accept %",
                        "type": "line",
                        "data": accept_data,
                        "yAxisIndex": 0,
                        "lineStyle": {"color": "#4caf50"},
                        "itemStyle": {"color": "#4caf50"},
                    },
                    {
                        "name": "Mean Score",
                        "type": "line",
                        "data": score_data,
                        "yAxisIndex": 1,
                        "lineStyle": {"color": "#2196f3"},
                        "itemStyle": {"color": "#2196f3"},
                    },
                ],
                "tooltip": {"trigger": "axis"},
                "legend": {"bottom": 0},
            }

        def _best_current_gen_model() -> str | None:
            """Return the generator model with the highest accept rate under
            the current-best prompt combo for this mode."""

            def _v(s: str) -> int:
                m = _re_md.search(r"_v(\d+)", s or "")
                return int(m.group(1)) if m else 0

            # Filter to mode-specific prompts only (exclude old combined prompts)
            _mode_stats = [
                s
                for s in mode_iter_stats
                if f"_{mode}_" in (s.get("judge_prompt") or "")
            ]
            judged_stats = [s for s in _mode_stats if s["n_acc"] + s["n_rej"] > 0]
            if not judged_stats:
                return None
            latest_judge_v = max(_v(s.get("judge_prompt", "")) for s in judged_stats)
            latest_gen_v: dict[str, int] = {}
            for s in judged_stats:
                gm = s.get("generator_model", "unknown")
                gv = _v(s.get("gen_prompt", ""))
                if gv > latest_gen_v.get(gm, -1):
                    latest_gen_v[gm] = gv
            pooled: dict[str, list[int]] = {}
            for s in judged_stats:
                gm = s.get("generator_model", "unknown")
                if _v(s.get("judge_prompt", "")) != latest_judge_v:
                    continue
                if _v(s.get("gen_prompt", "")) != latest_gen_v.get(gm, -1):
                    continue
                acc, rej = pooled.setdefault(gm, [0, 0])
                pooled[gm] = [acc + s["n_acc"], rej + s["n_rej"]]
            rates = {gm: a / (a + r) for gm, (a, r) in pooled.items() if (a + r) > 0}
            return max(rates, key=rates.get) if rates else None

        with ui.row().classes("w-full q-mt-md gap-4"):
            with ui.card().classes("flex-1 q-pa-md"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("Acceptance Rate & Mean Score").classes(
                        "text-subtitle2 text-weight-bold"
                    )
                    if len(all_gen_models) > 1:
                        gen_options = ["All"] + all_gen_models
                        default_gen = _best_current_gen_model() or all_gen_models[0]
                    else:
                        gen_options = all_gen_models
                        default_gen = all_gen_models[0] if all_gen_models else None

                trend_chart = (
                    ui.echart(
                        _build_trend_chart(
                            default_gen if len(all_gen_models) > 1 else None
                        )
                    )
                    .classes("w-full")
                    .style("height: 250px;")
                )

                if len(all_gen_models) > 1:

                    def _on_gen_filter(e):
                        val = None if e.value == "All" else e.value
                        trend_chart.options.update(_build_trend_chart(val))
                        trend_chart.update()

                    ui.select(
                        gen_options,
                        value=default_gen,
                        label="Generator Model",
                        on_change=_on_gen_filter,
                    ).classes("w-48")

            with ui.card().classes("flex-1 q-pa-md"):
                with ui.row().classes("items-center no-wrap").style("gap: 6px;"):
                    ui.label("Judge-Human Correlation by Judge Version").classes(
                        "text-subtitle2 text-weight-bold"
                    )
                    _metric_help = (
                        ui.icon("help_outline")
                        .classes("text-grey-6 cursor-pointer")
                        .style("font-size: 16px;")
                    )
                    with _metric_help:
                        ui.tooltip(
                            "Score Pearson r \u2014 linear correlation between judge "
                            "and human aggregate scores; +1 = perfectly aligned, "
                            "0 = no relationship, -1 = anti-aligned. Sensitive "
                            "to score *trend*, not absolute level.\n\n"
                            "Decision Cohen's \u03ba \u2014 chance-corrected agreement on "
                            "the binary accept/reject decision. 1 = perfect, "
                            "0 = chance, <0 = worse than chance. Penalises "
                            "agreement that comes from a skewed class prior.\n\n"
                            "Decision agreement \u2014 raw fraction of items where "
                            "judge and human picked the same decision (0..1). "
                            "Easy to read but inflated when one class dominates "
                            "\u2014 always read alongside \u03ba."
                        ).style(
                            "white-space: pre-line; max-width: 360px; "
                            "font-size: 0.8em;"
                        )
                iter_to_gen = {
                    r["iteration"]: r.get("generator_model", "unknown") for r in runs
                }
                cached_corr_pairs = _load_correlation_pairs(mode=mode)
                version_corrs = _aggregate_correlation_pairs(cached_corr_pairs)
                if not version_corrs:
                    ui.label("No judge correlations yet.").classes("text-grey-6")
                else:
                    all_models = sorted({m for _, m in version_corrs})
                    model_options = all_models + (
                        ["All (mean)", "All (min)"] if len(all_models) > 1 else []
                    )
                    default_model = (
                        all_models[0] if len(all_models) == 1 else "All (mean)"
                    )

                    all_prompts = sorted(
                        {p for p, _ in version_corrs},
                        key=lambda p: (
                            int(m.group(1))
                            if (m := _re_md.search(r"_v(\d+)", p))
                            else 0
                        ),
                    )

                    corr_state = {
                        "judge_model": default_model,
                        "gen_filter": None,
                        "split": None,
                    }

                    def _build_corr_chart_filtered() -> dict:
                        """Build correlation chart using current corr_state filters."""
                        gf = corr_state["gen_filter"]
                        sp = corr_state["split"]
                        if sp:
                            split_pairs = [
                                p
                                for p in cached_corr_pairs
                                if review_split(p["item_id"]) == sp
                            ]
                        else:
                            split_pairs = cached_corr_pairs
                        vc = (
                            _compute_correlation_by_judge_version(
                                gen_filter=gf,
                                iter_to_gen=iter_to_gen,
                                _pairs=split_pairs,
                            )
                            if gf
                            else _aggregate_correlation_pairs(split_pairs)
                        )
                        sm = corr_state["judge_model"]
                        cur_models = sorted({m for _, m in vc}) if vc else []

                        def _agg(prompt, metric):
                            if sm.startswith("All"):
                                vals = [
                                    vc.get((prompt, m), {}).get(metric)
                                    for m in cur_models
                                ]
                                vals = [v for v in vals if v is not None]
                                if not vals:
                                    return None
                                if "mean" in sm.lower():
                                    return round(statistics.mean(vals), 3)
                                return round(min(vals), 3)
                            entry = vc.get((prompt, sm), {})
                            v = entry.get(metric)
                            return round(v, 3) if v is not None else None

                        corr_data = [_agg(p, "pearson") for p in all_prompts]
                        kappa_data = [_agg(p, "kappa") for p in all_prompts]
                        agreement_data = [_agg(p, "agreement") for p in all_prompts]
                        return {
                            "xAxis": {
                                "type": "category",
                                "data": all_prompts,
                            },
                            "yAxis": {
                                "type": "value",
                                "name": "Score (-1..1)",
                                "min": -1,
                                "max": 1,
                            },
                            "series": [
                                {
                                    "name": "Score Pearson r",
                                    "type": "line",
                                    "data": corr_data,
                                    "itemStyle": {"color": "#ff9800"},
                                    "connectNulls": True,
                                },
                                {
                                    "name": "Decision Cohen's \u03ba",
                                    "type": "line",
                                    "data": kappa_data,
                                    "lineStyle": {"color": "#4caf50"},
                                    "itemStyle": {"color": "#4caf50"},
                                    "connectNulls": True,
                                },
                                {
                                    "name": "Decision agreement",
                                    "type": "line",
                                    "data": agreement_data,
                                    "lineStyle": {
                                        "color": "#2196f3",
                                        "type": "dashed",
                                    },
                                    "itemStyle": {"color": "#2196f3"},
                                    "connectNulls": True,
                                },
                            ],
                            "tooltip": {"trigger": "axis"},
                            "legend": {"bottom": 0},
                        }

                    corr_chart = (
                        ui.echart(_build_corr_chart_filtered())
                        .classes("w-full")
                        .style("height: 250px;")
                    )

                    def _refresh_corr_chart():
                        corr_chart.options.update(_build_corr_chart_filtered())
                        corr_chart.update()

                    with ui.row().classes("gap-2"):
                        if len(model_options) > 1:

                            def _on_model_change(e):
                                corr_state["judge_model"] = e.value
                                _refresh_corr_chart()

                            ui.select(
                                model_options,
                                value=default_model,
                                label="Judge Model",
                                on_change=_on_model_change,
                            ).classes("w-48")

                        if len(all_gen_models) > 1:
                            corr_gen_options = ["All"] + all_gen_models

                            def _on_corr_gen_filter(e):
                                corr_state["gen_filter"] = (
                                    None if e.value == "All" else e.value
                                )
                                _refresh_corr_chart()

                            ui.select(
                                corr_gen_options,
                                value="All",
                                label="Generator Model",
                                on_change=_on_corr_gen_filter,
                            ).classes("w-48")

                        def _on_split_change(e):
                            corr_state["split"] = None if e.value == "all" else e.value
                            _refresh_corr_chart()

                        ui.select(
                            ["all", "train", "validation"],
                            value="all",
                            label="Review Split",
                            on_change=_on_split_change,
                        ).classes("w-36")

    # --- (e) Per-Generator Acceptance Rate Bar Chart (auto-refreshing) ---
    _acc_rate_state = {"prompt_mode": "best", "min_samples": 50}
    _acc_rate_container = ui.column().classes("w-full")

    def _draw_acc_rate(r, s):
        _acc_rate_container.clear()
        with _acc_rate_container:
            _render_acceptance_rate_chart(
                r,
                s,
                _acc_rate_state["prompt_mode"],
                min_samples=_acc_rate_state["min_samples"],
                on_prompt_mode_change=lambda e: _on_acc_prompt_mode(e),
                on_min_samples_change=lambda e: _on_acc_min_samples(e),
                mode=mode,
            )

    def _on_acc_prompt_mode(e):
        _acc_rate_state["prompt_mode"] = e.value
        fresh_runs = load_runs()
        fresh_stats = _build_mode_acc_stats(fresh_runs)
        _draw_acc_rate(fresh_runs, fresh_stats)

    def _on_acc_min_samples(e):
        _acc_rate_state["min_samples"] = 50 if e.value else 0
        fresh_runs = load_runs()
        fresh_stats = _build_mode_acc_stats(fresh_runs)
        _draw_acc_rate(fresh_runs, fresh_stats)

    def _build_mode_acc_stats(fresh_runs):
        fresh_stats = []
        for run in fresh_runs:
            it = run["iteration"]
            items = load_items_for_iteration(it)
            judged = [i for i in items if i.get("judgment")]
            n_acc = 0
            n_mode = 0
            _agg_scores: list[float] = []
            for i in judged:
                _ma, _md = _derive_mode_agg_dec(
                    i["judgment"], mode, accept_threshold, floor_threshold
                )
                if _ma is None:
                    continue
                n_mode += 1
                if _md == "accept":
                    n_acc += 1
                _agg_scores.append(_ma)
            mean_s = statistics.mean(_agg_scores) if _agg_scores else 0
            accept_rate = round(n_acc / n_mode * 100, 1) if n_mode else 0
            fresh_stats.append(
                {
                    **run,
                    "judge_prompt": run.get(jp_field) or run.get("judge_prompt") or "",
                    "gen_prompt": run.get(gp_field) or run.get("gen_prompt") or "",
                    "n_acc": n_acc,
                    "n_rej": n_mode - n_acc,
                    "mean_score": mean_s,
                    "accept_rate": accept_rate,
                }
            )
        return fresh_stats

    _draw_acc_rate(runs, mode_iter_stats)

    _acc_rate_n_runs = [len(runs)]

    def _refresh_acc_rate():
        fresh_runs = load_runs()
        if len(fresh_runs) == _acc_rate_n_runs[0]:
            return
        _acc_rate_n_runs[0] = len(fresh_runs)
        fresh_stats = _build_mode_acc_stats(fresh_runs)
        _draw_acc_rate(fresh_runs, fresh_stats)

    ui.timer(30.0, _refresh_acc_rate)

    # --- (f) Generator Stats ---
    _render_generator_stats_panel(runs, mode_iter_stats, mode=mode)

    # --- (g) Calibration by Judge Model bar chart ---
    _cal_bar_pairs = _load_correlation_pairs(mode=mode)
    with ui.card().classes("w-full q-mt-md q-pa-md"):
        with ui.row().classes("items-center gap-2"):
            ui.label("Calibration by Judge Model").classes("text-h6 text-weight-bold")

            def _on_cal_bar_split(e):
                sp = None if e.value == "all" else e.value
                _cal_bar_chart_area.clear()
                with _cal_bar_chart_area:
                    _render_calibration_bar_chart(
                        pairs=_cal_bar_pairs, split=sp, _wrap_card=False
                    )

            ui.select(
                ["all", "train", "validation"],
                value="all",
                label="Review Split",
                on_change=_on_cal_bar_split,
            ).classes("w-36")

        _cal_bar_chart_area = ui.column().classes("w-full")
        with _cal_bar_chart_area:
            _render_calibration_bar_chart(pairs=_cal_bar_pairs, _wrap_card=False)

    # --- (h) Judge-Judge Correlations ---
    _render_judge_judge_correlations(runs, mode=mode)


def _render_acceptance_rate_chart(
    runs: list[dict],
    iter_stats: list[dict],
    prompt_mode: str = "latest",
    min_samples: int = 0,
    on_prompt_mode_change=None,
    on_min_samples_change=None,
    mode: str | None = None,
) -> None:
    """Render per-generator-model acceptance rate bar chart with binomial 95% CI.

    For each generator model, aggregates all runs with the selected gen prompt
    version (always using the latest judge prompt version), then computes
    acceptance rate + confidence interval.

    prompt_mode: "latest" uses the latest gen prompt version, "best" picks the
    gen prompt version with the highest acceptance rate.
    min_samples: only include iterations with more than this many judged samples.
    mode: when set, filter out old pre-split prompt entries (e.g. ``judge_v50.md``).
    """
    import math
    import re as _re_ar

    def _prompt_version(prompt_str: str) -> int:
        m = _re_ar.search(r"_v(\d+)\.md", prompt_str)
        if not m:
            m = _re_ar.search(r"_v(\d+)", prompt_str)
        return int(m.group(1)) if m else 0

    # Drop entries with old combined prompts so their version numbers
    # (e.g. judge_v50) don't mask newer mode-specific ones (judge_reflection_v3).
    if mode:
        iter_stats = [
            s for s in iter_stats if f"_{mode}_" in (s.get("judge_prompt") or "")
        ]

    if not iter_stats:
        return

    # Global latest judge prompt version across all iter_stats with judged items.
    # Only generators that have been run under this version are shown, so the
    # chart compares apples to apples.
    judged_iter_stats = [s for s in iter_stats if s["n_acc"] + s["n_rej"] > 0]
    if not judged_iter_stats:
        return
    global_latest_judge_v = max(
        _prompt_version(s.get("judge_prompt", "")) for s in judged_iter_stats
    )

    def _filter_for_gen_version(with_judged, gen_v):
        """Filter entries to a specific gen version under the global latest judge."""
        return [
            e
            for e in with_judged
            if _prompt_version(e.get("gen_prompt", "")) == gen_v
            and _prompt_version(e.get("judge_prompt", "")) == global_latest_judge_v
        ]

    # Group iter_stats by generator_model
    by_gen: dict[str, list[dict]] = {}
    for s in iter_stats:
        gen = s.get("generator_model", "unknown")
        by_gen.setdefault(gen, []).append(s)

    labels = []
    rates = []
    scores = []
    ci_low = []
    ci_high = []
    subtitles = []  # per-bar metadata shown below chart

    for gen_model in sorted(by_gen):
        entries = by_gen[gen_model]
        # Only consider entries with judged items (and above min_samples),
        # restricted to the global latest judge prompt version so generators
        # that never ran under it are excluded entirely.
        with_judged = [
            e
            for e in entries
            if e["n_acc"] + e["n_rej"] > min_samples
            and _prompt_version(e.get("judge_prompt", "")) == global_latest_judge_v
        ]
        if not with_judged:
            continue

        all_gen_versions = sorted(
            set(_prompt_version(e.get("gen_prompt", "")) for e in with_judged)
        )

        if prompt_mode == "best":
            # Pick the gen prompt version with the highest acceptance rate
            # (already restricted to the global latest judge above).
            best_v = None
            best_rate = -1.0
            for gv in all_gen_versions:
                filtered = [
                    e
                    for e in with_judged
                    if _prompt_version(e.get("gen_prompt", "")) == gv
                ]
                t_acc = sum(e["n_acc"] for e in filtered)
                t_rej = sum(e["n_rej"] for e in filtered)
                t_n = t_acc + t_rej
                if t_n > 0 and t_acc / t_n > best_rate:
                    best_rate = t_acc / t_n
                    best_v = gv
            if best_v is None:
                continue
            selected_gen_v = best_v
        else:
            # Latest gen prompt version
            selected_gen_v = max(all_gen_versions)

        latest = _filter_for_gen_version(with_judged, selected_gen_v)
        # Aggregate across all matching iterations
        total_acc = sum(e["n_acc"] for e in latest)
        total_rej = sum(e["n_rej"] for e in latest)
        n_judged = total_acc + total_rej
        if n_judged == 0:
            continue
        p = total_acc / n_judged
        margin = 1.96 * math.sqrt(p * (1 - p) / n_judged) if n_judged > 1 else 0

        # Mean score across matching iterations
        score_vals = [e["mean_score"] for e in latest if e.get("mean_score", 0) > 0]
        mean_score = statistics.mean(score_vals) if score_vals else 0

        judge_model = latest[0].get("judge_model", "?")
        latest_judge_v = _prompt_version(latest[0].get("judge_prompt", ""))
        labels.append(gen_model)
        rates.append(round(p * 100, 1))
        scores.append(round(mean_score, 2))
        ci_low.append(round(max(0, p - margin) * 100, 1))
        ci_high.append(round(min(1, p + margin) * 100, 1))
        subtitles.append(
            f"{gen_model}: gen v{selected_gen_v} | judge: {judge_model} v{latest_judge_v} | "
            f"n={n_judged} across {len(latest)} iter | score={round(mean_score, 2)}"
        )

    if not labels:
        return

    mode_label = "latest prompt" if prompt_mode == "latest" else "best prompt"
    with ui.card().classes("w-full q-mx-md q-mt-md q-pa-md"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.label(f"Performance per Generator ({mode_label})").classes(
                "text-h6 text-weight-bold"
            )
            if on_prompt_mode_change is not None:
                ui.select(
                    {"latest": "Latest Prompt", "best": "Best Prompt"},
                    value=prompt_mode,
                    label="Gen Prompt",
                    on_change=on_prompt_mode_change,
                ).classes("w-40")
            if on_min_samples_change is not None:
                ui.switch(
                    "Hide runs with ≤50 samples",
                    value=min_samples > 0,
                    on_change=on_min_samples_change,
                )
        # Show per-bar metadata as subtitle text
        for sub in subtitles:
            ui.label(sub).classes("text-caption text-grey-7")
        # Bar chart with error bars via markLine-style scatter overlay
        bar_series = {
            "name": "Accept %",
            "type": "bar",
            "data": rates,
            "yAxisIndex": 0,
            "itemStyle": {"color": "#4caf50"},
            "barMaxWidth": 60,
            "label": {
                "show": True,
                "position": "top",
                "formatter": "{c}%",
                "fontSize": 12,
            },
        }
        score_series = {
            "name": "Mean Score",
            "type": "bar",
            "data": scores,
            "yAxisIndex": 1,
            "itemStyle": {"color": "#2196f3"},
            "barMaxWidth": 60,
            "label": {
                "show": True,
                "position": "top",
                "fontSize": 12,
            },
        }
        error_series = {
            "name": "95% CI",
            "type": "scatter",
            "data": [[i, rates[i]] for i in range(len(rates))],
            "yAxisIndex": 0,
            "symbol": "none",
            "markLine": {
                "silent": True,
                "symbol": ["none", "none"],
                "lineStyle": {"color": "#666", "width": 1.5, "type": "solid"},
                "label": {"show": False},
                "data": [
                    [
                        {"xAxis": i, "yAxis": lo},
                        {"xAxis": i, "yAxis": hi},
                    ]
                    for i, (lo, hi) in enumerate(zip(ci_low, ci_high))
                ],
            },
        }
        max_score = max(scores) if scores else 10
        chart_opts = {
            "xAxis": {
                "type": "category",
                "data": labels,
                "axisLabel": {"interval": 0, "fontSize": 11},
            },
            "yAxis": [
                {"type": "value", "name": "Accept %", "min": 0, "max": 100},
                {
                    "type": "value",
                    "name": "Mean Score",
                    "min": 0,
                    "max": round(max_score * 1.3, 1),
                    "splitLine": {"show": False},
                },
            ],
            "series": [bar_series, score_series, error_series],
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Accept %", "Mean Score"], "top": 0},
            "grid": {"bottom": 60, "top": 40},
        }
        ui.echart(chart_opts).classes("w-full").style("height: 340px;")


def _render_generator_stats_panel(
    runs: list[dict], iter_stats: list[dict], mode: str | None = None
) -> None:
    """Render closable panel with per-generator stats for the latest prompt version.

    Shows failure rate, accept/reject counts, mean score, mean latency, etc.
    """
    import re as _re_gs

    def _prompt_version(prompt_str: str) -> int:
        m = _re_gs.search(r"_v(\d+)\.md", prompt_str)
        if not m:
            m = _re_gs.search(r"_v(\d+)", prompt_str)
        return int(m.group(1)) if m else 0

    # Drop entries with old combined prompts (see _render_acceptance_rate_chart).
    if mode:
        iter_stats = [
            s for s in iter_stats if f"_{mode}_" in (s.get("gen_prompt") or "")
        ]

    if not iter_stats:
        return

    # Group by generator model
    by_gen: dict[str, list[dict]] = {}
    for s in iter_stats:
        gen = s.get("generator_model", "unknown")
        by_gen.setdefault(gen, []).append(s)

    rows: list[dict] = []
    for gen_model in sorted(by_gen):
        entries = by_gen[gen_model]
        with_judged = [e for e in entries if e["n_acc"] + e["n_rej"] > 0]
        if not with_judged:
            continue
        # Latest gen prompt version
        latest_gen_v = max(
            _prompt_version(e.get("gen_prompt", "")) for e in with_judged
        )
        latest = [
            e
            for e in with_judged
            if _prompt_version(e.get("gen_prompt", "")) == latest_gen_v
        ]
        # Aggregate stats
        total_acc = sum(e["n_acc"] for e in latest)
        total_rej = sum(e["n_rej"] for e in latest)
        n_judged = total_acc + total_rej
        mean_score = (
            statistics.mean(e["mean_score"] for e in latest if e["mean_score"] > 0)
            if any(e["mean_score"] > 0 for e in latest)
            else 0
        )
        # Failure rate from run config
        n_attempted = sum(e.get("config", {}).get("n_attempted", 0) for e in latest)
        n_gen_failed = sum(e.get("config", {}).get("n_gen_failed", 0) for e in latest)
        fail_rate = round(n_gen_failed / n_attempted * 100, 1) if n_attempted > 0 else 0
        # Mean latency from items
        latencies = []
        for e in latest:
            it = e["iteration"]
            for item in load_items_for_iteration(it):
                if item.get("latency_ms"):
                    latencies.append(item["latency_ms"])
        mean_lat = round(statistics.mean(latencies) / 1000, 1) if latencies else None

        rows.append(
            {
                "generator": gen_model,
                "gen_prompt": f"v{latest_gen_v}",
                "n_iters": str(len(latest)),
                "n_attempted": str(n_attempted),
                "n_judged": str(n_judged),
                "fail_rate": f"{fail_rate}%",
                "accept_rate": (
                    f"{round(total_acc / n_judged * 100, 1)}%" if n_judged else "—"
                ),
                "mean_score": f"{mean_score:.2f}" if mean_score else "—",
                "mean_latency": f"{mean_lat}s" if mean_lat else "—",
            }
        )

    if not rows:
        return

    with ui.expansion("Generator Stats (latest prompt)", icon="bar_chart").classes(
        "w-full q-mx-md q-mt-md"
    ):
        cols = [
            {"name": "generator", "label": "Generator", "field": "generator"},
            {"name": "gen_prompt", "label": "Prompt", "field": "gen_prompt"},
            {"name": "n_iters", "label": "Iters", "field": "n_iters"},
            {"name": "n_attempted", "label": "Attempted", "field": "n_attempted"},
            {"name": "n_judged", "label": "Judged", "field": "n_judged"},
            {"name": "fail_rate", "label": "Gen Fail %", "field": "fail_rate"},
            {"name": "accept_rate", "label": "Accept %", "field": "accept_rate"},
            {"name": "mean_score", "label": "Mean Score", "field": "mean_score"},
            {"name": "mean_latency", "label": "Avg Latency", "field": "mean_latency"},
        ]
        ui.table(columns=cols, rows=rows, row_key="generator").classes("w-full")


def _render_calibration_bar_chart(
    pairs: list[dict] | None = None,
    split: str | None = None,
    _wrap_card: bool = True,
) -> None:
    """Render per-judge-model calibration bar chart (Pearson r and Cohen's kappa).

    Groups by judge_model, uses the latest judge prompt version per model.
    pairs: pre-loaded correlation pairs (avoids DB re-read). If None, loads from DB.
    split: filter pre-loaded pairs by "train" or "validation".
    _wrap_card: if False, skip card wrapper and title (caller provides them).
    """
    import re as _re_cb

    if pairs is None:
        pairs = _load_correlation_pairs()
    if split:
        pairs = [p for p in pairs if review_split(p["item_id"]) == split]
    if not pairs:
        return

    # Find latest judge prompt per judge model
    prompt_versions: dict[str, dict[str, int]] = {}  # model -> {prompt: version_num}
    for p in pairs:
        model = p["judge_model"]
        prompt = p["judge_prompt"]
        m = _re_cb.search(r"_v(\d+)", prompt)
        v = int(m.group(1)) if m else 0
        if model not in prompt_versions or v > max(prompt_versions[model].values()):
            prompt_versions.setdefault(model, {})[prompt] = v

    # Filter to latest prompt per model
    latest_prompt_per_model: dict[str, str] = {}
    for model, prompts in prompt_versions.items():
        latest_prompt_per_model[model] = max(prompts, key=prompts.get)

    filtered_pairs = [
        p
        for p in pairs
        if latest_prompt_per_model.get(p["judge_model"]) == p["judge_prompt"]
    ]
    if not filtered_pairs:
        return

    agg = _aggregate_correlation_pairs(filtered_pairs)
    if not agg:
        return

    models = sorted({m for _, m in agg})
    pearson_vals = []
    kappa_vals = []
    for model in models:
        # Find the entry for this model (latest prompt)
        prompt = latest_prompt_per_model.get(model, "")
        entry = agg.get((prompt, model), {})
        pr = entry.get("pearson")
        kp = entry.get("kappa")
        pearson_vals.append(round(pr, 3) if pr is not None else None)
        kappa_vals.append(round(kp, 3) if kp is not None else None)

    import contextlib

    _card_ctx = (
        ui.card().classes("w-full q-mx-md q-mt-md q-pa-md")
        if _wrap_card
        else contextlib.nullcontext()
    )
    with _card_ctx:
        if _wrap_card:
            ui.label("Calibration by Judge Model").classes("text-h6 text-weight-bold")
        ui.label(
            "Latest judge prompt per model. Pearson r (scores) and Cohen's κ (decisions)."
        ).classes("text-caption text-grey-7")
        chart_opts = {
            "xAxis": {"type": "category", "data": models},
            "yAxis": {"type": "value", "name": "Correlation", "min": -1, "max": 1},
            "series": [
                {
                    "name": "Pearson r",
                    "type": "bar",
                    "data": pearson_vals,
                    "itemStyle": {"color": "#ff9800"},
                    "barMaxWidth": 40,
                },
                {
                    "name": "Cohen's κ",
                    "type": "bar",
                    "data": kappa_vals,
                    "itemStyle": {"color": "#4caf50"},
                    "barMaxWidth": 40,
                },
            ],
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0},
            "grid": {"bottom": 60},
        }
        ui.echart(chart_opts).classes("w-full").style("height: 300px;")


def _render_api_stats_panel(runs: list[dict], items_by_key: dict) -> None:
    """Render collapsible API model statistics: reasoning tokens, throughput, batch rate."""
    if not items_by_key:
        return

    all_items = list(items_by_key.values())

    # Build run lookup: iteration -> run (carries alias info)
    run_by_iter: dict[int, dict] = {}
    for r in runs:
        run_by_iter[r["iteration"]] = r

    # Group items by generator alias (resolved from run metadata)
    gen_by_model: dict[str, list[dict]] = {}
    for item in all_items:
        run = run_by_iter.get(item.get("iteration"))
        gen_alias = (
            run.get("generator_model", item.get("model", "unknown"))
            if run
            else item.get("model", "unknown")
        )
        gen_by_model.setdefault(gen_alias, []).append(item)

    # Group judged items by judge alias (from run metadata)
    judge_items_by_model: dict[str, list[dict]] = {}
    for item in all_items:
        j = item.get("judgment")
        if not j:
            continue
        run = run_by_iter.get(item.get("iteration"))
        jmodel = run.get("judge_model", "unknown") if run else "unknown"
        judge_items_by_model.setdefault(jmodel, []).append(item)

    with ui.expansion("API Model Statistics", icon="analytics").classes(
        "w-full q-mx-md q-mt-md"
    ):
        has_gen_data = (
            any(
                item.get("reasoning_tokens") is not None
                for items in gen_by_model.values()
                for item in items
            )
            if gen_by_model
            else False
        )
        has_judge_data = (
            any(
                item.get("judgment", {}).get("preflection", {}).get("usage") is not None
                for items in judge_items_by_model.values()
                for item in items
            )
            if judge_items_by_model
            else False
        )

        if not has_gen_data and not has_judge_data:
            ui.label("No token usage data available (older items).").classes(
                "text-grey-6 text-caption q-pa-sm"
            )
        else:
            with ui.row().classes("w-full gap-4"):
                with ui.column().classes("flex-1"):
                    _render_role_stats("Generator", gen_by_model, "#2196f3")
                with ui.column().classes("flex-1"):
                    _render_role_stats_judge("Judge", judge_items_by_model, "#ff9800")

        _render_batch_rate_table(runs, items_by_key)


def _render_role_stats(
    role: str,
    by_model: dict[str, list[dict]],
    color: str,
) -> None:
    """Render reasoning tokens + throughput charts for generator models."""
    models = sorted(by_model)
    if not models:
        ui.label(f"No {role.lower()} data.").classes("text-grey-6 text-caption")
        return

    rt_means = []
    tp_vals = []
    for model in models:
        items = by_model[model]
        rt = [
            i.get("reasoning_tokens")
            for i in items
            if i.get("reasoning_tokens") is not None
        ]
        rt_means.append(round(statistics.mean(rt)) if rt else None)
        lat = [i["latency_ms"] for i in items if i.get("latency_ms")]
        tp_vals.append(round(1000 / statistics.mean(lat), 2) if lat else None)

    with ui.card().classes("w-full q-pa-md"):
        ui.label(f"{role} — Avg Reasoning Tokens").classes(
            "text-subtitle2 text-weight-bold"
        )
        if any(v is not None for v in rt_means):
            ui.echart(
                {
                    "xAxis": {"type": "category", "data": models},
                    "yAxis": {"type": "value", "name": "Tokens"},
                    "series": [
                        {
                            "name": role,
                            "type": "bar",
                            "data": rt_means,
                            "itemStyle": {"color": color},
                            "barMaxWidth": 50,
                        }
                    ],
                    "tooltip": {"trigger": "axis"},
                    "grid": {"bottom": 40},
                }
            ).classes("w-full").style("height: 220px;")
        else:
            ui.label("No token data (older items).").classes("text-grey-6 text-caption")

    with ui.card().classes("w-full q-pa-md q-mt-sm"):
        ui.label(f"{role} — Throughput (req/s)").classes(
            "text-subtitle2 text-weight-bold"
        )
        if any(v is not None for v in tp_vals):
            ui.echart(
                {
                    "xAxis": {"type": "category", "data": models},
                    "yAxis": {"type": "value", "name": "req/s"},
                    "series": [
                        {
                            "name": role,
                            "type": "bar",
                            "data": tp_vals,
                            "itemStyle": {"color": color},
                            "barMaxWidth": 50,
                        }
                    ],
                    "tooltip": {"trigger": "axis"},
                    "grid": {"bottom": 40},
                }
            ).classes("w-full").style("height: 220px;")
        else:
            ui.label("No latency data.").classes("text-grey-6 text-caption")


def _render_role_stats_judge(
    role: str,
    by_model: dict[str, list[dict]],
    color: str,
) -> None:
    """Render reasoning tokens chart for judge models (tokens from judgment JSON)."""
    models = sorted(by_model)
    if not models:
        ui.label(f"No {role.lower()} data.").classes("text-grey-6 text-caption")
        return

    rt_means = []
    tp_vals = []
    for model in models:
        items = by_model[model]
        rt = []
        latencies = []
        for item in items:
            j = item.get("judgment")
            if not j:
                continue
            for part, part_j in j.items():
                if part in JUDGMENT_NON_PART_KEYS or not isinstance(part_j, dict):
                    continue
                val = part_j.get("usage", {}).get("reasoning_tokens")
                if val is not None:
                    rt.append(val)
            lat = j.get("latency_ms")
            if lat is not None:
                latencies.append(lat)
        rt_means.append(round(statistics.mean(rt)) if rt else None)
        tp_vals.append(
            round(1000 / statistics.mean(latencies), 2) if latencies else None
        )

    with ui.card().classes("w-full q-pa-md"):
        ui.label(f"{role} — Avg Reasoning Tokens").classes(
            "text-subtitle2 text-weight-bold"
        )
        if any(v is not None for v in rt_means):
            ui.echart(
                {
                    "xAxis": {"type": "category", "data": models},
                    "yAxis": {"type": "value", "name": "Tokens"},
                    "series": [
                        {
                            "name": role,
                            "type": "bar",
                            "data": rt_means,
                            "itemStyle": {"color": color},
                            "barMaxWidth": 50,
                        }
                    ],
                    "tooltip": {"trigger": "axis"},
                    "grid": {"bottom": 40},
                }
            ).classes("w-full").style("height: 220px;")
        else:
            ui.label("No token data (older items).").classes("text-grey-6 text-caption")

    with ui.card().classes("w-full q-pa-md q-mt-sm"):
        ui.label(f"{role} — Throughput (req/s)").classes(
            "text-subtitle2 text-weight-bold"
        )
        if any(v is not None for v in tp_vals):
            ui.echart(
                {
                    "xAxis": {"type": "category", "data": models},
                    "yAxis": {"type": "value", "name": "req/s"},
                    "series": [
                        {
                            "name": role,
                            "type": "bar",
                            "data": tp_vals,
                            "itemStyle": {"color": color},
                            "barMaxWidth": 50,
                        }
                    ],
                    "tooltip": {"trigger": "axis"},
                    "grid": {"bottom": 40},
                }
            ).classes("w-full").style("height: 220px;")
        else:
            ui.label("No latency data (older items).").classes(
                "text-grey-6 text-caption"
            )


def _render_batch_rate_table(
    runs: list[dict], items_by_key: dict[tuple[str, int], dict]
) -> None:
    """Render batch-level samples/second table per run."""
    from datetime import datetime

    rows = []
    for run in runs[-20:]:  # last 20 runs
        it = run["iteration"]
        it_items = [v for (iid, itr), v in items_by_key.items() if itr == it]
        if len(it_items) < 2:
            continue
        timestamps = []
        for item in it_items:
            ts = item.get("timestamp", "")
            if ts:
                try:
                    timestamps.append(datetime.fromisoformat(ts))
                except ValueError:
                    pass
        if len(timestamps) < 2:
            continue
        span = (max(timestamps) - min(timestamps)).total_seconds()
        if span <= 0:
            continue
        rate = len(it_items) / span
        rows.append(
            {
                "iteration": it,
                "model": run.get("generator_model", "?"),
                "n_items": len(it_items),
                "span_s": f"{span:.1f}",
                "rate": f"{rate:.2f}",
            }
        )

    if not rows:
        return

    with ui.card().classes("w-full q-pa-md q-mt-sm"):
        ui.label("Batch Throughput (samples/s)").classes(
            "text-subtitle2 text-weight-bold"
        )
        cols = [
            {
                "name": "iteration",
                "label": "Iter",
                "field": "iteration",
                "sortable": True,
            },
            {"name": "model", "label": "Generator", "field": "model"},
            {"name": "n_items", "label": "Items", "field": "n_items"},
            {"name": "span_s", "label": "Span (s)", "field": "span_s"},
            {"name": "rate", "label": "samples/s", "field": "rate"},
        ]
        with ui.scroll_area().style("max-height: 200px;"):
            ui.table(
                columns=cols, rows=list(reversed(rows)), row_key="iteration"
            ).classes("w-full")


def _render_judge_judge_correlations(runs: list[dict], mode: str | None = None) -> None:
    """Render pairwise judge-judge correlation matrix/table.

    For each pair of judge models, computes Pearson r on scores and Cohen's kappa
    on decisions for items judged by both.

    mode: "reflection", "preflection", or None (combined).
      When set, uses per-mode aggregate/decision keys.
    """
    import re as _re_jj

    correlations = load_judge_correlations()
    if not correlations:
        return

    # Find latest judge prompt per judge model
    prompt_versions: dict[str, dict[str, int]] = {}
    for c in correlations:
        model = c["judge_model"]
        prompt = c["judge_prompt"]
        m = _re_jj.search(r"_v(\d+)", prompt)
        v = int(m.group(1)) if m else 0
        prompt_versions.setdefault(model, {})[prompt] = v

    latest_prompt: dict[str, str] = {}
    for model, prompts in prompt_versions.items():
        latest_prompt[model] = max(prompts, key=prompts.get)

    # Filter to latest prompt per model
    filtered = [
        c
        for c in correlations
        if latest_prompt.get(c["judge_model"]) == c["judge_prompt"]
    ]
    if not filtered:
        return

    # Group by (item_id, iteration) -> {judge_model: judgment}
    by_item: dict[tuple[str, int], dict[str, dict]] = {}
    for c in filtered:
        key = (c["item_id"], c["iteration"])
        by_item.setdefault(key, {})[c["judge_model"]] = c["judgment"]

    judge_models = sorted(latest_prompt.keys())
    if len(judge_models) < 2:
        return

    # Compute pairwise correlations
    pair_results: dict[tuple[str, str], dict] = {}
    for i, m_a in enumerate(judge_models):
        for j, m_b in enumerate(judge_models):
            if j <= i:
                continue
            score_pairs = []
            decision_pairs = []
            for judgments in by_item.values():
                if m_a in judgments and m_b in judgments:
                    j_a = judgments[m_a]
                    j_b = judgments[m_b]
                    if mode:
                        agg_a, dec_a = _derive_mode_agg_dec(j_a, mode)
                        agg_b, dec_b = _derive_mode_agg_dec(j_b, mode)
                    else:
                        agg_a = j_a.get("aggregate")
                        agg_b = j_b.get("aggregate")
                        dec_a = j_a.get("decision", "")
                        dec_b = j_b.get("decision", "")
                    if agg_a is not None and agg_b is not None:
                        score_pairs.append((agg_a, agg_b))
                    if dec_a and dec_b:
                        decision_pairs.append((dec_a, dec_b))
            pair_results[(m_a, m_b)] = {
                "pearson": _pearson(score_pairs),
                "kappa": _cohens_kappa(decision_pairs),
                "n": len(score_pairs),
            }

    if not pair_results:
        return

    with ui.card().classes("w-full q-mx-md q-mt-md q-pa-md"):
        ui.label("Judge-Judge Correlations").classes("text-h6 text-weight-bold")
        ui.label(
            "Pairwise agreement between judge models (latest prompt per model)."
        ).classes("text-caption text-grey-7")

        if len(judge_models) <= 4:
            # Render as table for small number of judges
            rows = []
            for (m_a, m_b), result in pair_results.items():
                pr = result["pearson"]
                kp = result["kappa"]
                rows.append(
                    {
                        "pair": f"{m_a} vs {m_b}",
                        "n": result["n"],
                        "pearson": f"{pr:.3f}" if pr is not None else "N/A",
                        "kappa": f"{kp:.3f}" if kp is not None else "N/A",
                    }
                )
            cols = [
                {"name": "pair", "label": "Judge Pair", "field": "pair"},
                {"name": "n", "label": "N items", "field": "n"},
                {"name": "pearson", "label": "Pearson r", "field": "pearson"},
                {"name": "kappa", "label": "Cohen's κ", "field": "kappa"},
            ]
            ui.table(columns=cols, rows=rows, row_key="pair").classes("w-full")
        else:
            # Heatmap for many judges
            heatmap_data = []
            for i, m_a in enumerate(judge_models):
                for j, m_b in enumerate(judge_models):
                    if i == j:
                        heatmap_data.append([i, j, 1.0])
                    elif (m_a, m_b) in pair_results:
                        val = pair_results[(m_a, m_b)]["pearson"]
                        heatmap_data.append(
                            [i, j, round(val, 3) if val is not None else None]
                        )
                        heatmap_data.append(
                            [j, i, round(val, 3) if val is not None else None]
                        )
            chart_opts = {
                "xAxis": {"type": "category", "data": judge_models},
                "yAxis": {"type": "category", "data": judge_models},
                "visualMap": {
                    "min": -1,
                    "max": 1,
                    "calculable": True,
                    "orient": "horizontal",
                    "left": "center",
                    "bottom": 0,
                    "inRange": {"color": ["#d32f2f", "#fff", "#388e3c"]},
                },
                "series": [
                    {
                        "type": "heatmap",
                        "data": heatmap_data,
                        "label": {"show": True, "fontSize": 11},
                    }
                ],
                "tooltip": {"position": "top"},
                "grid": {"bottom": 80},
            }
            ui.echart(chart_opts).classes("w-full").style("height: 400px;")


@ui.page("/pipeline", response_timeout=60.0)
def pipeline_monitoring_page():
    """Pipeline monitoring dashboard: iteration table, trends, calibration."""
    viewer_id = app.storage.user.get("annotator_id", "")

    def pipeline_actions():
        def _do_upload():
            if force_upload():
                ui.notify("Uploaded to HuggingFace", type="positive")
            else:
                ui.notify("Backup not configured (BACKUP_REPO not set)", type="warning")

        def _do_refresh():
            from pipeline.storage import force_reconnect

            force_reconnect()
            ui.run_javascript("location.reload(true)")

        ui.button(icon="refresh", on_click=_do_refresh).props(
            "flat dense size=sm"
        ).tooltip("Reload dashboard data").style("color:#666;")
        ui.button(icon="cloud_upload", on_click=_do_upload).props(
            "flat dense size=sm"
        ).tooltip("Force upload to HuggingFace").style("color:#666;")
        ui.button(
            "Review",
            icon="rate_review",
            on_click=lambda: ui.navigate.to("/pipeline/review"),
        ).classes("text-white").props("flat dense")
        ui.button(
            "All Reviews",
            icon="reviews",
            on_click=lambda: ui.navigate.to("/pipeline/reviews"),
        ).classes("text-white").props("flat dense")

    render_header(viewer_id, active_phase=2, right_slot=pipeline_actions)

    runs = load_runs()
    all_reviews = load_reviews()

    # Build items index for calibration
    items_by_key: dict[tuple[str, int], dict] = {}
    for run in runs:
        for item in load_items_for_iteration(run["iteration"]):
            items_by_key[(item["item_id"], item["iteration"])] = item

    # --- Precompute per-iteration stats (used by charts + table) ---
    iter_stats: list[dict] = []
    for run in runs:
        it = run["iteration"]
        items = load_items_for_iteration(it)
        judged = [i for i in items if i.get("judgment")]
        n_acc = sum(1 for i in judged if i["judgment"].get("decision") == "accept")
        scores = [i["judgment"].get("aggregate", 0) for i in judged]
        mean_s = statistics.mean(scores) if scores else 0
        accept_rate = round(n_acc / len(judged) * 100, 1) if judged else 0

        iter_reviews = [r for r in all_reviews if r["iteration"] == it]
        iter_items = {(i["item_id"], i["iteration"]): i for i in items}
        cal_iter = _compute_calibration(iter_reviews, iter_items)

        iter_stats.append(
            {
                **run,
                "n_acc": n_acc,
                "n_rej": len(judged) - n_acc,
                "mean_score": mean_s,
                "accept_rate": accept_rate,
                "calibration": cal_iter,
            }
        )

    # --- Reviews per Reviewer (shared) ---
    with ui.card().classes("w-full q-mx-md q-mt-md q-pa-md"):
        ui.label("Reviews per Reviewer").classes("text-h6 text-weight-bold")

        # Dedup to latest review per (item_id, iteration, reviewer_id) so
        # rescoring an item doesn't double-count.
        _latest_by_key: dict[tuple[str, int, str], dict] = {}
        for _r in all_reviews:
            _latest_by_key[(_r["item_id"], _r["iteration"], _r["reviewer_id"])] = _r

        _per_reviewer: dict[str, int] = {}
        for _r in _latest_by_key.values():
            _per_reviewer[_r["reviewer_id"]] = (
                _per_reviewer.get(_r["reviewer_id"], 0) + 1
            )

        if not _per_reviewer:
            ui.label("No reviews yet.").classes("text-caption text-grey-6 q-mt-sm")
        else:
            # Sort ascending so the largest bar lands at the top of a
            # horizontal bar chart (echarts draws category index 0 at bottom).
            _sorted = sorted(_per_reviewer.items(), key=lambda kv: kv[1])
            _names = [n for n, _ in _sorted]
            _counts = [c for _, c in _sorted]
            ui.echart(
                {
                    "grid": {"left": 90, "right": 30, "top": 10, "bottom": 25},
                    "xAxis": {"type": "value"},
                    "yAxis": {"type": "category", "data": _names},
                    "tooltip": {"trigger": "axis"},
                    "series": [
                        {
                            "type": "bar",
                            "data": _counts,
                            "itemStyle": {"color": "#5470c6"},
                            "label": {"show": True, "position": "right"},
                        }
                    ],
                }
            ).classes("w-full").style(f"height: {max(80, 28 * len(_names) + 50)}px;")

    # --- Mode tabs (Reflection / Preflection) ---
    cfg = load_config()
    with ui.tabs().classes("w-full q-mx-md q-mt-md") as mode_tabs:
        refl_tab = ui.tab("Reflection")
        prefl_tab = ui.tab("Preflection")

    with ui.tab_panels(mode_tabs, value=refl_tab).classes("w-full"):
        with ui.tab_panel(refl_tab):
            _render_mode_dashboard("reflection", runs, items_by_key, all_reviews, cfg)
        with ui.tab_panel(prefl_tab):
            _render_mode_dashboard("preflection", runs, items_by_key, all_reviews, cfg)

    # --- API Model Statistics (shared, collapsible) ---
    _render_api_stats_panel(runs, items_by_key)

    # --- Iteration Table ---
    with ui.expansion(
        f"Iterations ({len(runs)})",
        icon="format_list_numbered",
    ).classes("w-full q-mx-md q-mt-md"):
        if not runs:
            ui.label("No iterations yet.").classes("text-grey-6")
        else:
            columns = [
                {
                    "name": "iteration",
                    "label": "#",
                    "field": "iteration",
                    "sortable": True,
                },
                {"name": "source", "label": "Source", "field": "source_label"},
                {
                    "name": "generator_model",
                    "label": "Generator",
                    "field": "generator_model",
                },
                {"name": "judge_model", "label": "Judge", "field": "judge_display"},
                {"name": "gen_prompt", "label": "Gen Prompt", "field": "gen_prompt"},
                {
                    "name": "judge_prompt",
                    "label": "Judge Prompt",
                    "field": "judge_prompt",
                },
                {"name": "n_items", "label": "Items", "field": "n_items"},
                {"name": "group", "label": "Group", "field": "group_label"},
                {"name": "timestamp", "label": "Time", "field": "timestamp"},
            ]
            rows = [
                {
                    **s,
                    "source_label": _SOURCE_LABEL.get(
                        s.get("source", "manual"), "\u2014"
                    ),
                    "judge_display": s.get("judge_model", "?"),
                    "group_label": (
                        s.get("group_id", "")[:8] if s.get("group_id") else ""
                    ),
                    "timestamp": s["timestamp"][:19],
                    "accept_reject": f"{s['n_acc']}/{s['n_rej']}",
                    "mean_score": f"{s['mean_score']:.2f}",
                }
                for s in reversed(iter_stats)
            ]

            extra_cols = [
                {
                    "name": "accept_reject",
                    "label": "Accept/Reject",
                    "field": "accept_reject",
                },
                {"name": "mean_score", "label": "Mean Score", "field": "mean_score"},
            ]
            with ui.scroll_area().style("max-height: 180px;"):
                ui.table(
                    columns=columns + extra_cols, rows=rows, row_key="iteration"
                ).classes("w-full")

            with ui.scroll_area().style("max-height: 300px;"):
                for run in reversed(runs):
                    with ui.expansion(f"Iteration {run['iteration']} Analysis").classes(
                        "w-full"
                    ):
                        ui.markdown(
                            run.get("analysis", "No analysis recorded.")
                        ).classes("text-body2")

    # --- Loop History ---
    _render_loop_history()

    # --- Improver Controls ---
    cfg = load_config()
    with ui.expansion(
        "Improver Controls",
        icon="auto_fix_high",
    ).classes("w-full q-mx-md q-mt-md"):
        ui.label(
            "Independent improvers: judge and generator. Each spawns an Opus agent that tests "
            "against ALL counterpart models via cross-iteration."
        ).classes("text-caption text-grey-7")

        # Dynamic improver cards
        improver_cards_container = ui.column().classes("w-full gap-2 q-mt-sm")

        loop_error_label = ui.label("").classes("text-caption text-red")

        # Test results expansion
        with ui.expansion("Test Results", icon="science").classes("w-full q-mt-sm"):
            test_results_container = ui.column().classes("w-full gap-1")

        # Log viewer with dynamic tabs
        with ui.expansion("Improver Logs", icon="terminal").classes("w-full q-mt-sm"):
            log_tabs_container = ui.column().classes("w-full")

        def _tail_log(log_path: Path, n_lines: int = 50) -> str:
            if not log_path.exists():
                return ""
            lines = log_path.read_text().splitlines()
            return "\n".join(lines[-n_lines:])

        # Tracked elements for in-place updates (avoids DOM rebuild on every poll)
        _card_els: dict[str, dict] = {}  # key -> {"badge": el, "detail": el}
        _log_els: dict[str, object] = {}  # key -> ui.code element
        _test_results_count: dict = (
            {}
        )  # {"value": int} — rebuild only when count changes

        def _build_cards(improvers: dict) -> None:
            """Full rebuild of improver cards (only when keys change)."""
            _card_els.clear()
            improver_cards_container.clear()
            if not improvers:
                with improver_cards_container:
                    ui.label("No improver runs yet.").classes(
                        "text-grey-6 text-caption"
                    )
                return
            with improver_cards_container:
                with ui.row().classes("w-full gap-4 flex-wrap"):
                    for key in improvers:
                        imp_role, imp_mode, imp_alias = parse_improver_key(key)
                        with (
                            ui.card()
                            .classes("flex-1 q-pa-sm")
                            .style("min-width: 280px;")
                        ):
                            with ui.row().classes("items-center gap-2"):
                                _imp_display = f"{imp_role.title()}"
                                if imp_mode:
                                    _imp_display += f" ({imp_mode})"
                                _imp_display += f": {imp_alias}"
                                ui.label(_imp_display).classes(
                                    "text-subtitle2 text-weight-bold"
                                )
                                badge_el = ui.badge("pending", color="grey")
                            detail_el = (
                                ui.label("")
                                .classes("text-body2")
                                .style("white-space: pre-wrap; font-size: 0.85em;")
                            )
                            _card_els[key] = {"badge": badge_el, "detail": detail_el}

        def _update_cards(improvers: dict) -> None:
            """Update card badge and detail text in-place (no DOM rebuild)."""
            from pipeline.charter.improve.loop import (
                improver_log_path as _log_path,
                _extract_latest_status_from_log,
                _extract_reasoning_from_log,
            )

            for key, data in improvers.items():
                if key not in _card_els:
                    continue
                els = _card_els[key]
                imp_status = data.get("status", "pending")
                label, color = _status_badge(imp_status)
                els["badge"].set_text(label)
                els["badge"]._props["color"] = color
                els["badge"].update()

                imp_role, _, imp_alias = parse_improver_key(key)
                log_p = _log_path(imp_role, imp_alias)

                if imp_status == "running":
                    els["detail"].set_text(_extract_latest_status_from_log(log_p))
                elif imp_status in ("done", "error"):
                    reasoning = data.get("reasoning", "")
                    if not reasoning:
                        reasoning = _extract_reasoning_from_log(log_p)
                    els["detail"].set_text(reasoning[:500] if reasoning else "")
                else:
                    els["detail"].set_text("")

        def _build_log_tabs(improvers: dict) -> None:
            """Full rebuild of log tabs (only when keys change)."""
            from pipeline.charter.improve.loop import improver_log_path as _log_path

            _log_els.clear()
            log_tabs_container.clear()
            if not improvers:
                return
            with log_tabs_container:
                with ui.tabs().classes("w-full") as tabs:
                    tab_map = {}
                    for key in improvers:
                        tab_map[key] = ui.tab(key)
                first_tab = next(iter(tab_map.values())) if tab_map else None
                with ui.tab_panels(tabs, value=first_tab).classes("w-full"):
                    for key in improvers:
                        imp_role, _, imp_alias = parse_improver_key(key)
                        with ui.tab_panel(tab_map[key]):
                            log_p = _log_path(imp_role, imp_alias)
                            code_el = (
                                ui.code(_tail_log(log_p), language="text")
                                .classes("w-full")
                                .style(
                                    "max-height: 400px; overflow-y: auto; font-size: 0.8em;"
                                )
                            )
                            _log_els[key] = code_el

        def _update_log_tabs(improvers: dict) -> None:
            """Update log content in-place (no DOM rebuild)."""
            from pipeline.charter.improve.loop import improver_log_path as _log_path

            for key in improvers:
                if key not in _log_els:
                    continue
                imp_role, _, imp_alias = parse_improver_key(key)
                log_p = _log_path(imp_role, imp_alias)
                _log_els[key].set_content(_tail_log(log_p))

        def _update_from_status(st: dict) -> None:
            """Update UI from status dict, rebuilding only when improver keys change."""
            improvers = st.get("improvers", {})

            if set(improvers.keys()) != set(_card_els.keys()):
                _build_cards(improvers)
            _update_cards(improvers)

            if st.get("error"):
                loop_error_label.set_text(f"Error: {st['error']}")
            else:
                loop_error_label.set_text("")

            if set(improvers.keys()) != set(_log_els.keys()):
                _build_log_tabs(improvers)
            else:
                _update_log_tabs(improvers)

        def _poll_loop_status():
            from pipeline.charter.improve.loop import read_status

            st = read_status()
            if st is None:
                return

            _update_from_status(st)

            # Update test results (only rebuild if count changed)
            results = load_test_results()
            new_count = len(results) if results else 0
            if new_count != _test_results_count.get("value", -1):
                _test_results_count["value"] = new_count
                test_results_container.clear()
                if results:
                    with test_results_container:
                        cols = [
                            {"name": "test_id", "label": "Test ID", "field": "test_id"},
                            {"name": "type", "label": "Type", "field": "type"},
                            {"name": "role", "label": "Role", "field": "role"},
                            {"name": "prompt", "label": "Prompt", "field": "prompt"},
                            {"name": "n_items", "label": "Items", "field": "n_items"},
                            {
                                "name": "mean_score",
                                "label": "Mean Score",
                                "field": "mean_score",
                            },
                            {
                                "name": "timestamp",
                                "label": "Time",
                                "field": "timestamp",
                            },
                        ]
                        t_rows = []
                        for r in results[-20:]:
                            s = r.get("summary", {})
                            t_rows.append(
                                {
                                    "test_id": r.get("test_id", ""),
                                    "type": r.get("type", ""),
                                    "role": r.get("role", r.get("phase", "?")),
                                    "prompt": r.get("prompt", ""),
                                    "n_items": s.get("n_items", ""),
                                    "mean_score": (
                                        f"{s['mean_score']:.2f}"
                                        if isinstance(s.get("mean_score"), (int, float))
                                        else ""
                                    ),
                                    "timestamp": r.get("timestamp", "")[:19],
                                }
                            )
                        ui.table(columns=cols, rows=t_rows, row_key="test_id").classes(
                            "w-full"
                        )

        loop_timer = ui.timer(3.0, _poll_loop_status, active=True)

        # Load existing status on page render
        from pipeline.charter.improve.loop import read_status as _read_initial

        _initial = _read_initial()
        if _initial:
            _update_from_status(_initial)

    # --- Run Cross-Iteration ---
    with ui.expansion(
        "Run Cross-Iteration",
        icon="play_circle",
    ).classes("w-full q-mx-md q-mt-md"):
        ui.label(
            "Run a single cross-iteration batch (generate × judge across all counterpart models)."
        ).classes("text-caption text-grey-7")

        with ui.row().classes("items-center gap-4"):
            cross_role_select = ui.select(
                options=["judge", "generator"],
                value="judge",
                label="Role",
            ).classes("w-32")

            def _get_target_options():
                if cross_role_select.value == "judge":
                    return [m.alias for m in cfg.charter.improve.judge_models]
                return [m.alias for m in cfg.charter.improve.generator_models]

            cross_target_select = ui.select(
                options=_get_target_options(),
                value=_get_target_options()[0] if _get_target_options() else "",
                label="Target Model",
            ).classes("w-48")

            def _on_role_change(e):
                opts = _get_target_options()
                cross_target_select.options = opts
                cross_target_select.set_value(opts[0] if opts else "")

            cross_role_select.on_value_change(_on_role_change)

        cross_status = ui.label("").classes("text-caption text-grey-6")

        def start_cross_iteration():
            cross_btn.disable()
            cross_status.set_text("Running cross-iteration...")

            def _thread():
                try:
                    from pipeline.charter.improve.run import (
                        run_judge_cross_iteration,
                        run_generator_cross_iteration,
                    )

                    run_cfg = load_config()
                    role = cross_role_select.value
                    target = cross_target_select.value
                    if role == "judge":
                        results = run_judge_cross_iteration(run_cfg, target)
                    else:
                        results = run_generator_cross_iteration(run_cfg, target)
                    cross_status.set_text(
                        f"Done! {len(results)} iterations. Refresh page to see results."
                    )
                    ui.notify("Cross-iteration complete", type="positive")
                except Exception as e:
                    cross_status.set_text(f"Error: {e}")
                finally:
                    cross_btn.enable()

            threading.Thread(target=_thread, daemon=True).start()

        cross_btn = ui.button("Run", on_click=start_cross_iteration, color="secondary")

        # Disable cross_btn if any improver is running
        if _initial and _initial.get("running"):
            cross_btn.disable()


@ui.page("/pipeline/review", response_timeout=60.0)
def pipeline_review_page():
    """Human review of LLM-generated reflections with per-dimension scoring."""
    viewer_id = app.storage.user.get("annotator_id", "")
    if not viewer_id:
        ui.navigate.to("/")
        return

    def review_actions():
        def _do_upload():
            if force_upload():
                ui.notify("Uploaded to HuggingFace", type="positive")
            else:
                ui.notify("Backup not configured (BACKUP_REPO not set)", type="warning")

        ui.button(icon="cloud_upload", on_click=_do_upload).props(
            "flat dense size=sm"
        ).tooltip("Force upload to HuggingFace").style("color:#666;")
        ui.button(
            "Dashboard",
            icon="dashboard",
            on_click=lambda: ui.navigate.to("/pipeline"),
        ).classes("text-white").props("flat dense")

    render_header(viewer_id, active_phase=2, right_slot=review_actions)

    runs = load_runs()
    if not runs:
        with ui.column().classes("absolute-center items-center"):
            ui.label("No iterations to review yet.").classes("text-h6 text-grey-6")
        return

    charter_text = CHARTER_TEXT
    cfg = load_config()
    dimensions = cfg.charter.improve.scoring.dimensions
    threshold = cfg.charter.improve.scoring.accept_threshold

    # Floor rule: any single dimension at or below this forces a reject,
    # regardless of the average. Mirrors the judge's logic in
    # pipeline/summaries/tools.py.
    FLOOR_SCORE = 2

    def _compute_decision(values: list[int]) -> tuple[float, str]:
        """Apply the floor rule + threshold to a flat list of dimension scores."""
        if not values:
            return 0.0, "reject"
        aggregate = sum(values) / len(values)
        if any(v <= FLOOR_SCORE for v in values) or aggregate < threshold:
            return aggregate, "reject"
        return aggregate, "accept"

    def _prompt_version(filename: str) -> str:
        """Extract version from prompt filename.

        Handles both old names (e.g. 'judge_v3.md') and new per-mode names
        (e.g. 'judge_reflection_v3.md', 'judge_preflection_v2.md').
        Returns e.g. 'v3'.
        """
        import re

        m = re.search(r"(v\d+)", filename)
        return m.group(1) if m else filename

    def _version_sort_key(v: str) -> int:
        return int(v[1:]) if v.startswith("v") and v[1:].isdigit() else 0

    # Run metadata lookups — refreshed via _refresh_runs() on every sort pass
    # because background pipeline jobs keep adding new iterations after the
    # page is opened. If we cached these at page load, the Recommended sort
    # would silently skip any generator/judge run that arrived later.
    run_by_iter: dict[int, dict] = {}
    group_iters: dict[str, list[int]] = {}
    _latest_gen_prompt: dict[str, str] = {}
    _latest_judge_prompt: dict[str, str] = {}

    def _refresh_runs() -> None:
        nonlocal runs, run_by_iter, group_iters
        nonlocal _latest_gen_prompt, _latest_judge_prompt
        runs = load_runs()
        run_by_iter = {r["iteration"]: r for r in runs}
        group_iters = {}
        for r in runs:
            gid = r.get("group_id")
            if gid:
                group_iters.setdefault(gid, []).append(r["iteration"])
        _latest_gen_prompt = {}
        for r in runs:
            model = r.get("generator_model", "unknown")
            v = _prompt_version(r["gen_prompt"])
            if model not in _latest_gen_prompt or _version_sort_key(
                v
            ) > _version_sort_key(_latest_gen_prompt[model]):
                _latest_gen_prompt[model] = v
        _latest_judge_prompt = {}
        for r in runs:
            model = r.get("judge_model", "unknown")
            v = _prompt_version(r["judge_prompt"])
            if model not in _latest_judge_prompt or _version_sort_key(
                v
            ) > _version_sort_key(_latest_judge_prompt[model]):
                _latest_judge_prompt[model] = v

    _refresh_runs()

    all_gen_models = sorted({r.get("generator_model", "unknown") for r in runs})
    all_judge_models = sorted({r.get("judge_model", "unknown") for r in runs})
    all_gen_prompts = sorted(
        {_prompt_version(r["gen_prompt"]) for r in runs}, key=_version_sort_key
    )
    all_judge_prompts = sorted(
        {_prompt_version(r["judge_prompt"]) for r in runs}, key=_version_sort_key
    )

    # State — prompts default to "latest" (per-model), models default to All
    LATEST = "latest"
    state = {
        "pos": 0,
        "gen_model": None,
        "gen_prompt": LATEST,
        "judge_model": None,
        "judge_prompt": LATEST,
    }

    # --- Filter bar ---
    with ui.row().classes("q-px-md q-mt-xs items-center gap-2"):
        gen_model_select = (
            ui.select(
                options=["All"] + all_gen_models,
                value="All",
                label="Generator",
            )
            .props("dense options-dense")
            .classes("w-32")
            .style("font-size: 0.8em;")
        )

        gen_prompt_select = (
            ui.select(
                options=["latest", "All"] + all_gen_prompts,
                value=state["gen_prompt"],
                label="Gen Prompt",
            )
            .props("dense options-dense")
            .classes("w-24")
            .style("font-size: 0.8em;")
        )

        judge_model_select = (
            ui.select(
                options=["All"] + all_judge_models,
                value="All",
                label="Judge",
            )
            .props("dense options-dense")
            .classes("w-32")
            .style("font-size: 0.8em;")
        )

        judge_prompt_select = (
            ui.select(
                options=["latest", "All"] + all_judge_prompts,
                value=state["judge_prompt"],
                label="Judge Prompt",
            )
            .props("dense options-dense")
            .classes("w-24")
            .style("font-size: 0.8em;")
        )

        sort_select = (
            ui.select(
                options=[
                    "Recommended",
                    "Low judge score",
                    "High judge score",
                    "Low safety score",
                    "High safety score",
                    "Default order",
                ],
                value="Recommended",
                label="Sort",
            )
            .props("dense options-dense")
            .classes("w-36")
            .style("font-size: 0.8em;")
        )

        # Manual queue refresh — the items list is snapshotted at page load
        # and on filter/sort changes (not on every navigate) so the displayed
        # item doesn't shift when other reviewers submit. Click this to pick
        # up new items from background pipeline jobs.
        refresh_queue_btn = (
            ui.button(icon="refresh")
            .props("flat dense size=sm")
            .tooltip("Refresh review queue (pick up new items)")
            .style("color:#666;")
        )

    # --- Main split panel ---
    with (
        ui.splitter(value=35)
        .classes("w-full")
        .style("height: calc(100vh - 120px)")
        .props('separator-class="bg-primary" separator-style="width: 4px;"') as splitter
    ):
        # Left: source text + charter
        with splitter.before:
            with (
                ui.column()
                .classes("w-full p-4 gap-2")
                .style(
                    "position: sticky; top: 0; height: calc(100vh - 120px); overflow-y: auto;"
                )
            ):
                ui.label("Source Text + Charter").classes("text-h6 text-weight-bold")
                source_html = ui.html("").style(
                    "max-height: 40%; overflow-y: auto; border: 1px solid #333; "
                    "border-radius: 4px; padding: 12px; line-height: 1.7; "
                    "font-family: Georgia, serif; white-space: pre-wrap; font-size: 0.95em;"
                )
                ui.separator()
                with ui.row().classes("items-center w-full gap-2"):
                    ui.label("Charter").classes("text-subtitle2 text-weight-bold")
                    charter_search = (
                        ui.input(placeholder="Search charter...")
                        .props("dense outlined clearable")
                        .classes("flex-1")
                    )
                charter_md = (
                    ui.markdown(charter_text, extras=["tables"])
                    .classes("text-body2")
                    .style("flex: 1; overflow-y: auto; padding: 8px; line-height: 1.6;")
                )
                _charter_dom_id = f"c{charter_md.id}"

                def _on_charter_search(e) -> None:
                    term = e.value or ""
                    term_json = json.dumps(term)
                    js = f"""
                    (function() {{
                        const root = document.getElementById({json.dumps(_charter_dom_id)});
                        if (!root) return;
                        // Remove previous highlights
                        root.querySelectorAll('mark.charter-search').forEach(m => {{
                            const parent = m.parentNode;
                            parent.replaceChild(document.createTextNode(m.textContent), m);
                            parent.normalize();
                        }});
                        const term = {term_json};
                        if (!term) return;
                        const re = new RegExp(
                            term.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&'),
                            'gi'
                        );
                        const walker = document.createTreeWalker(
                            root, NodeFilter.SHOW_TEXT, {{
                                acceptNode: n => n.parentNode &&
                                    n.parentNode.nodeName !== 'SCRIPT' &&
                                    n.parentNode.nodeName !== 'STYLE'
                                    ? NodeFilter.FILTER_ACCEPT
                                    : NodeFilter.FILTER_REJECT
                            }}
                        );
                        const nodes = [];
                        let n;
                        while ((n = walker.nextNode())) nodes.push(n);
                        for (const textNode of nodes) {{
                            const text = textNode.nodeValue;
                            re.lastIndex = 0;
                            if (!re.test(text)) continue;
                            re.lastIndex = 0;
                            const frag = document.createDocumentFragment();
                            let last = 0;
                            let m;
                            while ((m = re.exec(text)) !== null) {{
                                if (m.index > last) {{
                                    frag.appendChild(
                                        document.createTextNode(text.slice(last, m.index))
                                    );
                                }}
                                const mk = document.createElement('mark');
                                mk.className = 'charter-search';
                                mk.style.background = '#ffe066';
                                mk.style.color = '#000';
                                mk.textContent = m[0];
                                frag.appendChild(mk);
                                last = m.index + m[0].length;
                            }}
                            if (last < text.length) {{
                                frag.appendChild(
                                    document.createTextNode(text.slice(last))
                                );
                            }}
                            textNode.parentNode.replaceChild(frag, textNode);
                        }}
                        const first = root.querySelector('mark.charter-search');
                        if (first) first.scrollIntoView({{block: 'center'}});
                    }})();
                    """
                    ui.run_javascript(js)

                charter_search.on_value_change(_on_charter_search)

        # Right: LLM generation + judge scores + review form
        with splitter.after:
            with ui.column().classes("w-full p-4 gap-2").style("overflow-y: auto;"):
                with ui.row().classes("items-center gap-4"):
                    nav_label = ui.label().classes("text-subtitle1 text-weight-medium")
                    subset_badge = ui.badge("").props("outline")
                    safety_badge = ui.badge("").props("outline color=deep-purple")
                    gold_badge = ui.badge("").props("outline color=orange")
                    ui.space()
                    ui.button(icon="arrow_back", on_click=lambda: navigate(-1)).props(
                        "flat dense"
                    )
                    ui.button(icon="arrow_forward", on_click=lambda: navigate(1)).props(
                        "flat dense"
                    )

                # LLM generation display
                ui.markdown(
                    "**preflection** = framing placed *before* the text · "
                    "**reflection** = thoughtful pause inserted *during* the "
                    'text · **1p** = first-person voice (uses "I") · '
                    '**3p** = third-person voice (no "I")'
                ).classes("text-caption text-grey-7").style("margin-top: -4px;")
                gen_section = ui.column().classes("w-full gap-2")

                ui.separator()

                # Human annotation (for gold items)
                gold_expansion = ui.expansion(
                    "Human Annotation (Gold)", icon="person"
                ).classes("w-full")
                with gold_expansion:
                    gold_section = ui.column().classes("w-full gap-1")

                ui.separator()

                # Review form
                with ui.row().classes("items-center gap-2"):
                    ui.label("Your Review").classes("text-subtitle2 text-weight-bold")
                    review_existing_badge = ui.badge("").props("outline")
                ui.markdown(
                    f"Score each dimension 1–5 per part. **Reject** if any single "
                    f"score is ≤ {FLOOR_SCORE} (floor rule) **or** the overall "
                    f"average is < {threshold:g}; **Accept** otherwise."
                ).classes("text-caption text-grey-7").style("margin-top: -6px;")

                DIM_SHORT = {
                    "relevance": "Rel",
                    "specificity": "Spec",
                    "charter_grounding": "Charter",
                    "voice_tone": "Voice",
                    "class_discipline": "Class",
                }
                DIM_TOOLTIPS = {
                    "relevance": (
                        "Does it correctly identify what matters (or correctly note "
                        "nothing is wrong)?\n"
                        "1: Flags irrelevant issues, or misses obvious problems\n"
                        "2: Right topic but mischaracterizes content (e.g. comedy "
                        "as endorsement)\n"
                        "3: Gets the gist but connections are forced or generic\n"
                        "4: Identifies key concerns with minor gaps\n"
                        "5: Precisely identifies the relevant concerns (or correctly "
                        "identifies as unproblematic)"
                    ),
                    "specificity": (
                        "Is it specific to *this* text, or could it apply to anything?\n"
                        "1: Completely generic, or references content not in the text\n"
                        "2: References broad topic but invents details\n"
                        "3: References the topic but stays surface-level\n"
                        "4: References concrete content with minor gaps\n"
                        "5: References concrete claims, phrases, or content from the text"
                    ),
                    "charter_grounding": (
                        "Are charter references appropriate and well-used?\n"
                        "1: Wrong sections cited, or missing [X.Y] notation\n"
                        "2: Sections vaguely related, or forces refs on benign text\n"
                        "3: Correct sections but shallow connection\n"
                        "4: Good charter refs with clear connections; minor issues\n"
                        "5: Precise [X.Y] refs clearly connected to the text\n"
                        "Note: no refs is correct (4-5) for genuinely benign text."
                    ),
                    "voice_tone": (
                        "Correct voice (1p/3p as required), natural, appropriate length?\n"
                        "1: Wrong voice (e.g. third-person reflection_1p)\n"
                        "2: Correct voice but heavily templated/stock phrases\n"
                        "3: Correct voice but stilted, formulaic, or verbose\n"
                        "4: Natural and well-written with minor issues\n"
                        "5: Natural, varied, concise — reads like a genuine response"
                    ),
                    "class_discipline": (
                        "Preflection-only. Does the field adhere to its type "
                        "specification?\n"
                        "- charter_summary: '[X.Y] Title: summary.' format, "
                        "document-agnostic, ≤ 6 sentences.\n"
                        "- neutral: names the ethical territory, no verdict, "
                        "no plot recap.\n"
                        "- judgemental: opinionated verdict with specific reasoning, "
                        "no rubric-stamp codas.\n"
                        "- idealisation: declarative present tense, adds a concrete "
                        "divergent element vs. judgemental, no 'should/would/must'."
                    ),
                }

                # Preflection 4-field schema is scored on 3 dimensions per
                # field; reflection (and legacy preflection) on 4 dimensions.
                _PREFLECTION_4FIELD_DIMS = (
                    "relevance",
                    "charter_grounding",
                    "class_discipline",
                )
                def _dims_for_part(part: str) -> list[str]:
                    if part in _PREFLECTION_FIELDS_CURRENT:
                        return list(_PREFLECTION_4FIELD_DIMS)
                    return list(dimensions)

                # Per-part scoring: {part: {dim: slider}} — populated inline in
                # gen_section as each text part is rendered.
                score_inputs: dict[str, dict[str, ui.slider]] = {}

                review_status_label = ui.label("").classes(
                    "text-caption text-weight-bold"
                )

                def _update_review_status():
                    all_vals = [
                        int(slider.value)
                        for dims in score_inputs.values()
                        for slider in dims.values()
                    ]
                    agg, decision = _compute_decision(all_vals)
                    color = "green" if decision == "accept" else "red"
                    # Per-mode breakdown
                    parts_text = f"Avg: {agg:.2f} → {decision.upper()}"
                    for mode_name in ("reflection", "preflection"):
                        _mn_parts = _MODE_PART_NAMES.get(mode_name, frozenset())
                        mode_vals = [
                            int(slider.value)
                            for part, dims in score_inputs.items()
                            if part in _mn_parts
                            for slider in dims.values()
                        ]
                        if mode_vals:
                            m_agg, m_dec = _compute_decision(mode_vals)
                            parts_text += (
                                f"  |  {mode_name}: {m_agg:.2f} → {m_dec.upper()}"
                            )
                    review_status_label.set_text(parts_text)
                    review_status_label.style(f"color: {color};")

                def _build_part_sliders(part: str) -> None:
                    """Render a compact one-line slider row for a single part."""
                    score_inputs[part] = {}
                    part_dims = _dims_for_part(part)
                    with (
                        ui.row()
                        .classes("items-center w-full no-wrap q-mt-xs q-mb-sm")
                        .style("gap: 12px;")
                    ):
                        for dim in part_dims:
                            with (
                                ui.row()
                                .classes("items-center no-wrap")
                                .style("gap: 2px; flex: 1;")
                            ):
                                ui.label(DIM_SHORT.get(dim, dim)).classes(
                                    "text-caption text-grey-7"
                                ).style("min-width: 42px;")
                                help_icon = (
                                    ui.icon("help_outline")
                                    .classes("text-grey-6 cursor-pointer")
                                    .style("font-size: 13px;")
                                )
                                with help_icon:
                                    ui.tooltip(DIM_TOOLTIPS.get(dim, "")).style(
                                        "white-space: pre-line; max-width: 320px; "
                                        "font-size: 0.8em;"
                                    )
                                slider = (
                                    ui.slider(min=1, max=5, value=3)
                                    .classes("flex-1")
                                    .style("min-width: 60px; margin-left: 4px;")
                                )
                                score_label = (
                                    ui.label("3")
                                    .classes("text-caption text-weight-medium")
                                    .style("min-width: 12px;")
                                )
                                slider.on(
                                    "update:model-value",
                                    lambda e, lbl=score_label: lbl.set_text(
                                        str(int(e.args))
                                    ),
                                )
                                slider.on(
                                    "update:model-value",
                                    lambda _: _update_review_status(),
                                )
                                slider.on(
                                    "update:model-value",
                                    lambda _: _save_draft(),
                                )
                                score_inputs[part][dim] = slider

                _update_review_status()

                notes_input = (
                    ui.textarea(
                        placeholder="Notes (optional)...",
                    ).classes("w-full")
                    # debounce so on_value_change (and the draft write below)
                    # fires ~once per typing pause, not on every keystroke.
                    .props('outlined debounce="400"')
                )
                notes_input.on_value_change(lambda _: _save_draft())

                with ui.row().classes("w-full justify-end"):
                    submit_btn = ui.button(
                        "Submit Review",
                        on_click=lambda: submit_review(),
                        color="primary",
                    )

                ui.separator()

                # Judge scores (collapsed by default) — placed at the bottom
                judge_expansion = ui.expansion("Judge Scores", icon="gavel").classes(
                    "w-full"
                )
                with judge_expansion:
                    judge_section = ui.column().classes("w-full gap-1")

    def _filtered_iterations() -> list[int]:
        """Return iteration numbers matching current filter selections."""
        result = []
        for r in runs:
            if state["gen_model"] and r.get("generator_model") != state["gen_model"]:
                continue
            if state["judge_model"] and r.get("judge_model") != state["judge_model"]:
                continue
            gp = state["gen_prompt"]
            if gp:
                rv = _prompt_version(r["gen_prompt"])
                if gp == LATEST:
                    if rv != _latest_gen_prompt.get(r.get("generator_model", ""), ""):
                        continue
                elif rv != gp:
                    continue
            jp = state["judge_prompt"]
            if jp:
                rv = _prompt_version(r["judge_prompt"])
                if jp == LATEST:
                    if rv != _latest_judge_prompt.get(r.get("judge_model", ""), ""):
                        continue
                elif rv != jp:
                    continue
            result.append(r["iteration"])
        return result

    def get_sorted_items() -> list[dict]:
        """Load items from all matching iterations, deduplicate, sort.

        After sorting, interleaves items round-robin across generator models
        so that reviews cover all generators equally.
        """
        # Pick up any new runs added by background jobs since page load —
        # otherwise _filtered_iterations and the "latest prompt" lookups stay
        # frozen to the page's initial snapshot.
        _refresh_runs()
        iters = _filtered_iterations()
        all_items: list[dict] = []
        for it in iters:
            all_items.extend(load_items_for_iteration(it))
        # Deduplicate by (item_id, model) — keep highest iteration per generator
        seen: dict[tuple, dict] = {}
        for item in all_items:
            key = (item["item_id"], item.get("model", ""))
            if key not in seen or item["iteration"] > seen[key]["iteration"]:
                seen[key] = item
        judged = [i for i in seen.values() if i.get("judgment")]
        sort = sort_select.value
        if sort == "Recommended":
            # Skip items already reviewed by anyone other than the current
            # viewer — we don't want reviewer overlap. The viewer's own
            # reviewed items still appear so they can revisit/edit them.
            _latest = load_latest_reviews()
            _claimed: set[tuple[str, int]] = set()
            for iid, it, reviewer in _latest:
                if reviewer != viewer_id:
                    _claimed.add((iid, it))
            judged = [
                i for i in judged if (i["item_id"], i["iteration"]) not in _claimed
            ]

            # Bucket items by judge decision/score into 4 groups (25% each):
            #   1. Very highly accepted  2. Medium-high accepted
            #   3. Borderline            4. Rejected
            # Within each bucket, prioritise bad safety scores (2 bad : 1 good).
            rejected = [i for i in judged if i["judgment"].get("decision") != "accept"]
            accepted = [i for i in judged if i["judgment"].get("decision") == "accept"]
            accepted.sort(key=lambda i: i["judgment"].get("aggregate", 0))
            # Split accepted into thirds by rank
            n = len(accepted)
            borderline = accepted[: n // 3]
            medium_high = accepted[n // 3 : 2 * n // 3]
            very_high = accepted[2 * n // 3 :]

            import random as _rnd

            _rng = _rnd.Random(42)  # deterministic but shuffled

            def _safety_interleave(bucket: list[dict]) -> list[dict]:
                """Re-order a bucket so that ~2/3 are bad safety and ~1/3 good."""
                bad = [
                    i
                    for i in bucket
                    if i.get("safety_score") is not None and i["safety_score"] >= 4
                ]
                good = [
                    i
                    for i in bucket
                    if i.get("safety_score") is None or i["safety_score"] < 4
                ]
                _rng.shuffle(bad)
                _rng.shuffle(good)
                # Interleave 2 bad, 1 good
                result: list[dict] = []
                bi, gi = 0, 0
                while bi < len(bad) or gi < len(good):
                    for _ in range(2):
                        if bi < len(bad):
                            result.append(bad[bi])
                            bi += 1
                    if gi < len(good):
                        result.append(good[gi])
                        gi += 1
                return result

            buckets = [
                _safety_interleave(very_high),
                _safety_interleave(medium_high),
                _safety_interleave(borderline),
                _safety_interleave(rejected),
            ]
            # Round-robin across the 4 buckets for equal representation
            judged = []
            while any(buckets):
                for b in buckets:
                    if b:
                        judged.append(b.pop(0))
        elif sort == "Low judge score":
            judged.sort(key=lambda i: i["judgment"].get("aggregate", 0))
        elif sort == "High judge score":
            judged.sort(key=lambda i: -i["judgment"].get("aggregate", 0))
        elif sort == "Low safety score":
            judged.sort(
                key=lambda i: (i.get("safety_score") is None, i.get("safety_score", 0))
            )
        elif sort == "High safety score":
            judged.sort(
                key=lambda i: (
                    i.get("safety_score") is None,
                    -(i.get("safety_score") or 0),
                )
            )

        # Skip per-model interleaving for Recommended (global order matters)
        if sort == "Recommended":
            return judged

        # Interleave across generator models for balanced reviewing
        by_gen: dict[str, list[dict]] = {}
        for item in judged:
            run = run_by_iter.get(item["iteration"], {})
            gen = run.get("generator_model", "unknown")
            by_gen.setdefault(gen, []).append(item)
        if len(by_gen) <= 1:
            return judged
        gen_queues = [items for items in by_gen.values()]
        interleaved: list[dict] = []
        while gen_queues:
            gen_queues = [q for q in gen_queues if q]
            for q in gen_queues:
                interleaved.append(q.pop(0))
        return interleaved

    # --- Items list cache ---
    # The items list is snapshotted (not recomputed on every navigate) so the
    # displayed item doesn't shift when *other* reviewers submit. Recomputing
    # the Recommended sort each time would silently swap the current item out
    # from under the reviewer mid-scoring, which made the UI feel like it was
    # randomly refreshing. Refreshed only on filter/sort change or via the
    # manual "Refresh queue" button.
    _items_cache: list[dict] = []

    def _refresh_items_cache() -> None:
        nonlocal _items_cache
        _items_cache = get_sorted_items()

    def current_items_list() -> list[dict]:
        return _items_cache

    # --- In-progress draft persistence ---
    # Drafts (per-item scores + notes) are persisted to app.storage.user so
    # they survive websocket reconnects and accidental tab reloads. Keyed by
    # (item_id, iteration). The TTL prunes drafts older than a week so the
    # cookie doesn't accumulate forever.
    DRAFT_TTL_SECONDS = 7 * 24 * 3600

    # Suppresses draft writes while update_display() programmatically calls
    # slider.set_value(), which would otherwise fire 'update:model-value' and
    # stomp the just-loaded values with a default-valued draft.
    _suppress_draft_save = False

    def _draft_key(item_id: str, iteration: int) -> str:
        return f"{item_id}:{iteration}"

    def _all_drafts() -> dict:
        # Read a plain copy. Mutations go through _set_drafts so the change
        # is unambiguously a top-level __setitem__ on app.storage.user, which
        # always fires backup() (no reliance on nested-dict observation).
        try:
            return dict(app.storage.user.get("phase2_drafts") or {})
        except Exception as e:
            logger.warning("phase2: _all_drafts read failed: %r", e)
            return {}

    def _set_drafts(drafts: dict) -> None:
        try:
            app.storage.user["phase2_drafts"] = drafts
        except Exception as e:
            logger.warning("phase2: _set_drafts write failed: %r", e)

    def _prune_stale_drafts() -> None:
        drafts = _all_drafts()
        if not drafts:
            return
        cutoff = datetime.now(timezone.utc).timestamp() - DRAFT_TTL_SECONDS
        kept = {}
        for k, v in drafts.items():
            try:
                ts = datetime.fromisoformat(v.get("ts", "")).timestamp()
            except (ValueError, TypeError):
                ts = 0
            if ts >= cutoff:
                kept[k] = v
        if len(kept) != len(drafts):
            _set_drafts(kept)

    def _current_item() -> dict | None:
        items = current_items_list()
        if not items or not (0 <= state["pos"] < len(items)):
            return None
        return items[state["pos"]]

    def _save_draft() -> None:
        if _suppress_draft_save or not score_inputs:
            return
        item = _current_item()
        if item is None:
            return
        drafts = _all_drafts()
        drafts[_draft_key(item["item_id"], item["iteration"])] = {
            "scores": {
                part: {dim: int(slider.value) for dim, slider in dims.items()}
                for part, dims in score_inputs.items()
            },
            "notes": notes_input.value or "",
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        _set_drafts(drafts)

    def _load_draft(item: dict) -> dict | None:
        return _all_drafts().get(_draft_key(item["item_id"], item["iteration"]))

    def _clear_draft(item: dict) -> None:
        drafts = _all_drafts()
        if drafts.pop(_draft_key(item["item_id"], item["iteration"]), None) is not None:
            _set_drafts(drafts)

    # Persist the currently-displayed item key so that a genuine reload
    # (browser refresh or >60s disconnect) restores the same item rather than
    # jumping to first-unreviewed under a freshly-snapshotted Recommended
    # queue. Use top-level __setitem__ for the same propagation reasons as
    # _set_drafts.
    def _save_position(item: dict | None) -> None:
        try:
            if item is None:
                app.storage.user.pop("phase2_current_item", None)
            else:
                app.storage.user["phase2_current_item"] = {
                    "item_id": item["item_id"],
                    "iteration": item["iteration"],
                }
        except Exception as e:
            logger.warning("phase2: _save_position failed: %r", e)

    def _load_position() -> tuple[str, int] | None:
        try:
            saved = app.storage.user.get("phase2_current_item")
        except Exception:
            return None
        if not saved:
            return None
        return (saved.get("item_id"), saved.get("iteration"))

    _prune_stale_drafts()

    def _first_unreviewed_pos(items: list[dict]) -> int:
        """Return index of the first item without a review from this viewer."""
        reviewed = load_latest_reviews()
        reviewed_keys = {(k[0], k[1]) for k in reviewed if k[2] == viewer_id}
        for i, item in enumerate(items):
            if (item["item_id"], item["iteration"]) not in reviewed_keys:
                return i
        return 0

    def update_display():
        nonlocal _suppress_draft_save
        items = current_items_list()
        if not items:
            nav_label.set_text("No judged items for this filter")
            return

        state["pos"] = max(0, min(state["pos"], len(items) - 1))
        item = items[state["pos"]]
        _save_position(item)

        nav_label.set_text(f"Item {state['pos'] + 1} / {len(items)}")
        subset_badge.set_text(item["subset"])
        ss = item.get("safety_score")
        safety_badge.set_text(f"safety: {ss}" if ss is not None else "")
        safety_badge.set_visibility(ss is not None)
        gold_badge.set_text("GOLD" if item.get("is_gold") else "")
        gold_badge.set_visibility(item.get("is_gold", False))

        source_html.set_content(
            render_source_text(item["text"], item["reflection_point"])
        )

        gen_section.clear()
        score_inputs.clear()
        with gen_section:
            with ui.row().classes("items-center gap-2"):
                ui.label("LLM Generation").classes("text-subtitle2 text-weight-bold")
                item_run = run_by_iter.get(item["iteration"], {})
                gen_model_name = item_run.get("generator_model", "")
                if gen_model_name:
                    ui.badge(gen_model_name, color="teal").props("outline")
                gen_prompt_name = item_run.get("gen_prompt", "")
                if gen_prompt_name:
                    ui.badge(gen_prompt_name, color="blue-grey").props("outline")
            ui.label("Analysis").classes("text-overline text-grey-7")
            ui.label(item.get("analysis", "")).classes("text-body2").style(
                "white-space: pre-wrap;"
            )
            _PART_DISPLAY = _build_part_display(item)
            for label_text, text_value in _PART_DISPLAY:
                ui.label(label_text).classes("text-overline text-grey-7")
                ui.label(text_value).classes("text-body2").style(
                    "white-space: pre-wrap;"
                )
                _build_part_sliders(label_text)
            for ce_label, ce_field in (
                ("Preflection charter", "preflection_charter_elements"),
                ("Reflection charter", "reflection_charter_elements"),
            ):
                ce_elements = item.get(ce_field, [])
                if ce_elements:
                    ui.label(ce_label).classes("text-overline text-grey-7")
                    with ui.row().classes("gap-1"):
                        for eid in ce_elements:
                            ui.badge(eid, color="blue-grey-3").props("outline")

        judge_section.clear()
        with judge_section:
            # Collect all judgments for this item: current iteration + siblings from same group
            item_iter = item["iteration"]
            judgments_to_show: list[tuple[str, dict]] = []
            judgment = item.get("judgment", {})
            if judgment:
                cur_run = run_by_iter.get(item_iter, {})
                cur_judge = cur_run.get("judge_model", "")
                judgments_to_show.append((cur_judge, judgment))

            # Look for sibling iterations in the same group that used a
            # *different* judge model (skip same-judge/different-generator siblings)
            cur_run = run_by_iter.get(item_iter, {})
            cur_gid = cur_run.get("group_id")
            cur_judge = cur_run.get("judge_model", "")
            seen_judges = {cur_judge} if cur_judge else set()
            if cur_gid and cur_gid in group_iters:
                sib_iters = [i for i in group_iters[cur_gid] if i != item_iter]
                if sib_iters:
                    sib_items = load_item_across_iterations(item["item_id"], sib_iters)
                    for si in sib_items:
                        if si.get("judgment"):
                            sib_run = run_by_iter.get(si["iteration"], {})
                            sib_judge = sib_run.get("judge_model", "?")
                            if sib_judge in seen_judges:
                                continue
                            seen_judges.add(sib_judge)
                            judgments_to_show.append((sib_judge, si["judgment"]))

            if not judgments_to_show:
                ui.label("No judge scores.").classes("text-grey-6")
            else:
                use_tabs = len(judgments_to_show) > 1
                if use_tabs:
                    with ui.tabs().classes("w-full") as jtabs:
                        tab_objs = []
                        for jm, _ in judgments_to_show:
                            tab_objs.append(ui.tab(jm or "Judge"))
                    with ui.tab_panels(jtabs).classes("w-full"):
                        for (jm, jdg), tab in zip(judgments_to_show, tab_objs):
                            with ui.tab_panel(tab):
                                _render_judge_scores(jdg, judge_model=jm)
                else:
                    jm, jdg = judgments_to_show[0]
                    _render_judge_scores(jdg, judge_model=jm)

        gold_section.clear()
        gold_expansion.set_visibility(bool(item.get("is_gold")))
        with gold_section:
            if item.get("is_gold"):
                _show_gold_annotation(item["item_id"])

        # Restore priority: in-progress draft → existing review → defaults.
        # Drafts win so reviewers don't lose work after a websocket reconnect.
        latest_reviews = load_latest_reviews()
        existing = latest_reviews.get((item["item_id"], item["iteration"], viewer_id))
        draft = _load_draft(item)

        if draft:
            scores_per_part = draft.get("scores", {}) or {}
            notes_value = draft.get("notes", "")
            badge_text = (
                "Unsaved changes (existing review)" if existing else "Unsaved draft"
            )
            badge_color = "amber"
            btn_text = "Update Review" if existing else "Submit Review"
        elif existing:
            ex_scores = existing["scores"]
            # Legacy flat-dict reviews stored {dim: int} instead of
            # {part: {dim: int}}; broadcast the flat values to every part.
            if ex_scores and isinstance(next(iter(ex_scores.values())), dict):
                scores_per_part = ex_scores
            else:
                scores_per_part = {part: ex_scores for part in score_inputs}
            notes_value = existing.get("notes", "")
            badge_text = "Editing existing review"
            badge_color = "orange"
            btn_text = "Update Review"
        else:
            scores_per_part = {}
            notes_value = ""
            badge_text = "New review"
            badge_color = "green"
            btn_text = "Submit Review"

        _suppress_draft_save = True
        try:
            for part, dims in score_inputs.items():
                for dim, slider in dims.items():
                    slider.set_value(scores_per_part.get(part, {}).get(dim, 3))
            notes_input.set_value(notes_value)
            review_existing_badge.set_text(badge_text)
            review_existing_badge.props(f"color={badge_color}")
            submit_btn.set_text(btn_text)
        finally:
            _suppress_draft_save = False
        _update_review_status()

    def _show_gold_annotation(item_id: str):
        """Display the human annotation for a gold item."""
        from pipeline.charter.seed.storage import load_latest_annotations

        annotations = load_latest_annotations()
        gold_records = [v for (iid, _), v in annotations.items() if iid == item_id]
        if not gold_records:
            ui.label("No human annotations found for this gold item.").classes(
                "text-grey-6"
            )
            return

        for rec in gold_records:
            with ui.card().classes("w-full q-pa-sm"):
                ui.label(f"Annotator: {rec['annotator_id']}").classes(
                    "text-caption text-grey-6"
                )
                ui.label("Analysis").classes("text-overline text-grey-7")
                ui.label(rec["analysis"]).classes("text-body2").style(
                    "white-space: pre-wrap;"
                )
                ui.label("Preflection (3p)").classes("text-overline text-grey-7")
                ui.label(rec["preflection"]).classes("text-body2").style(
                    "white-space: pre-wrap;"
                )
                ui.label("Reflection (1p)").classes("text-overline text-grey-7")
                ui.label(rec["reflection"]).classes("text-body2").style(
                    "white-space: pre-wrap;"
                )

    def navigate(delta: int):
        items = current_items_list()
        new_pos = state["pos"] + delta
        if 0 <= new_pos < len(items):
            state["pos"] = new_pos
            update_display()

    def submit_review():
        items = current_items_list()
        if not items:
            return
        item = items[state["pos"]]
        scores = {
            part: {dim: int(slider.value) for dim, slider in dims.items()}
            for part, dims in score_inputs.items()
        }
        all_vals = [v for part in scores.values() for v in part.values()]
        aggregate, decision = _compute_decision(all_vals)

        # Compute per-mode decisions (reflection / preflection).
        # Bucket by set membership so the current 4-field preflection schema
        # (charter_summary/neutral/judgemental/idealisation) lands in the
        # preflection aggregate alongside legacy preflection_* parts.
        refl_vals = [
            v
            for part, part_scores in scores.items()
            if part in _REFLECTION_PART_NAMES
            for v in part_scores.values()
        ]
        prefl_vals = [
            v
            for part, part_scores in scores.items()
            if part in _PREFLECTION_PART_NAMES
            for v in part_scores.values()
        ]
        refl_agg, refl_dec = _compute_decision(refl_vals) if refl_vals else (None, None)
        prefl_agg, prefl_dec = (
            _compute_decision(prefl_vals) if prefl_vals else (None, None)
        )

        save_review(
            item_id=item["item_id"],
            iteration=item["iteration"],
            reviewer_id=viewer_id,
            scores=scores,
            aggregate=aggregate,
            decision=decision,
            notes=notes_input.value.strip(),
            reflection_decision=refl_dec,
            preflection_decision=prefl_dec,
            reflection_aggregate=refl_agg,
            preflection_aggregate=prefl_agg,
        )
        _clear_draft(item)
        ui.notify("Review saved!", type="positive")
        navigate(1)

    def _on_filter_change():
        """Re-snapshot the items list and reset position when filters change."""
        state["pos"] = 0
        _refresh_items_cache()
        items = current_items_list()
        state["pos"] = _first_unreviewed_pos(items) if items else 0
        update_display()

    def _on_gen_model_change(e):
        state["gen_model"] = None if e.value == "All" else e.value
        _on_filter_change()

    def _on_gen_prompt_change(e):
        state["gen_prompt"] = None if e.value == "All" else e.value
        _on_filter_change()

    def _on_judge_model_change(e):
        state["judge_model"] = None if e.value == "All" else e.value
        _on_filter_change()

    def _on_judge_prompt_change(e):
        state["judge_prompt"] = None if e.value == "All" else e.value
        _on_filter_change()

    def _on_sort_change(_):
        state["pos"] = 0
        _refresh_items_cache()
        update_display()

    def _on_refresh_queue():
        """Manual queue refresh — re-snapshot the items list and try to keep
        the reviewer on the same item if it's still in the queue."""
        cur = _current_item()
        cur_key = (cur["item_id"], cur["iteration"]) if cur else None
        _refresh_items_cache()
        items = current_items_list()
        if cur_key and items:
            for i, it in enumerate(items):
                if (it["item_id"], it["iteration"]) == cur_key:
                    state["pos"] = i
                    break
            else:
                state["pos"] = _first_unreviewed_pos(items)
        elif items:
            state["pos"] = _first_unreviewed_pos(items)
        else:
            state["pos"] = 0
        update_display()
        ui.notify("Queue refreshed", type="info")

    gen_model_select.on_value_change(_on_gen_model_change)
    gen_prompt_select.on_value_change(_on_gen_prompt_change)
    judge_model_select.on_value_change(_on_judge_model_change)
    judge_prompt_select.on_value_change(_on_judge_prompt_change)
    sort_select.on_value_change(_on_sort_change)
    refresh_queue_btn.on("click", lambda _: _on_refresh_queue())
    _refresh_items_cache()
    items = current_items_list()
    if items:
        # Restore the previously-displayed item if it's still in the queue,
        # otherwise fall back to the first unreviewed.
        saved_key = _load_position()
        state["pos"] = 0
        if saved_key:
            for i, it in enumerate(items):
                if (it["item_id"], it["iteration"]) == saved_key:
                    state["pos"] = i
                    break
            else:
                state["pos"] = _first_unreviewed_pos(items)
        else:
            state["pos"] = _first_unreviewed_pos(items)
    else:
        state["pos"] = 0
    update_display()


@ui.page("/pipeline/reviews", response_timeout=60.0)
def pipeline_reviews_page():
    """Review overview: browse all reviews, comment on them, edit and delete them."""
    viewer_id = app.storage.user.get("annotator_id", "")

    cfg = load_config()
    _rv_dimensions = cfg.charter.improve.scoring.dimensions
    _rv_threshold = cfg.charter.improve.scoring.accept_threshold
    _RV_FLOOR = 2

    _RV_DIM_SHORT = {
        "relevance": "Rel",
        "specificity": "Spec",
        "charter_grounding": "Charter",
        "voice_tone": "Voice",
        "class_discipline": "Class",
    }

    def _rv_compute_decision(values: list[int]) -> tuple[float, str]:
        if not values:
            return 0.0, "reject"
        aggregate = sum(values) / len(values)
        if any(v <= _RV_FLOOR for v in values) or aggregate < _rv_threshold:
            return aggregate, "reject"
        return aggregate, "accept"

    def _open_edit_dialog(review: dict, on_saved):
        """Open a dialog to edit an existing review's scores and notes."""
        scores = review["scores"]
        is_per_part = scores and isinstance(next(iter(scores.values())), dict)

        with (
            ui.dialog().props("persistent maximized") as dlg,
            ui.card()
            .classes("w-full q-pa-md")
            .style("max-width: 700px; margin: auto;"),
        ):
            ui.label("Edit Review").classes("text-h6")
            ui.label(
                f"Item {review['item_id'][:12]} · iter {review['iteration']} · "
                f"{review['reviewer_id']}"
            ).classes("text-caption text-grey-6 q-mb-sm")

            status_label = ui.label("").classes("text-caption text-weight-bold q-mb-sm")
            edit_sliders: dict[str, dict[str, ui.slider]] = {}

            # Determine parts: use the review's per-part keys if available,
            # otherwise fall back to the standard four voices.
            if is_per_part:
                parts = list(scores.keys())
            else:
                parts = [
                    "preflection_3p",
                    "preflection_1p",
                    "reflection_1p",
                    "reflection_3p",
                ]

            def _update_status():
                vals = [
                    int(s.value)
                    for dims in edit_sliders.values()
                    for s in dims.values()
                ]
                agg, dec = _rv_compute_decision(vals)
                color = "green" if dec == "accept" else "red"
                status_label.set_text(f"Avg: {agg:.2f} → {dec.upper()}")
                status_label.style(f"color: {color};")

            for part in parts:
                ui.label(part).classes("text-overline text-grey-7 q-mt-sm")
                edit_sliders[part] = {}
                # Dims per part: prefer dims actually present in the saved
                # review (so a new 4-field preflection review renders its 3
                # dims, and a legacy reflection review renders its 4).
                if is_per_part and isinstance(scores.get(part), dict):
                    _part_dims = list(scores[part].keys()) or list(_rv_dimensions)
                else:
                    _part_dims = list(_rv_dimensions)
                with (
                    ui.row().classes("items-center w-full no-wrap").style("gap: 12px;")
                ):
                    for dim in _part_dims:
                        if is_per_part:
                            init_val = scores.get(part, {}).get(dim, 3)
                        else:
                            init_val = scores.get(dim, 3)
                        with (
                            ui.row()
                            .classes("items-center no-wrap")
                            .style("gap: 2px; flex: 1;")
                        ):
                            ui.label(_RV_DIM_SHORT.get(dim, dim)).classes(
                                "text-caption text-grey-7"
                            ).style("min-width: 42px;")
                            slider = (
                                ui.slider(min=1, max=5, value=init_val)
                                .classes("flex-1")
                                .style("min-width: 60px; margin-left: 4px;")
                            )
                            score_lbl = (
                                ui.label(str(init_val))
                                .classes("text-caption text-weight-medium")
                                .style("min-width: 12px;")
                            )
                            slider.on(
                                "update:model-value",
                                lambda e, lbl=score_lbl: lbl.set_text(str(int(e.args))),
                            )
                            slider.on(
                                "update:model-value",
                                lambda _: _update_status(),
                            )
                            edit_sliders[part][dim] = slider

            _update_status()

            edit_notes = (
                ui.textarea(value=review.get("notes", ""))
                .classes("w-full q-mt-sm")
                .props("outlined")
            )

            with ui.row().classes("w-full justify-end q-mt-sm gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat")

                def _save():
                    new_scores = {
                        p: {d: int(s.value) for d, s in dims.items()}
                        for p, dims in edit_sliders.items()
                    }
                    vals = [v for part in new_scores.values() for v in part.values()]
                    agg, dec = _rv_compute_decision(vals)

                    # Per-mode decisions — bucket by set membership so 4-field
                    # preflection parts are grouped alongside legacy ones.
                    _refl_vals = [
                        v
                        for p, ps in new_scores.items()
                        if p in _REFLECTION_PART_NAMES
                        for v in ps.values()
                    ]
                    _prefl_vals = [
                        v
                        for p, ps in new_scores.items()
                        if p in _PREFLECTION_PART_NAMES
                        for v in ps.values()
                    ]
                    _r_agg, _r_dec = (
                        _rv_compute_decision(_refl_vals) if _refl_vals else (None, None)
                    )
                    _p_agg, _p_dec = (
                        _rv_compute_decision(_prefl_vals)
                        if _prefl_vals
                        else (None, None)
                    )

                    save_review(
                        item_id=review["item_id"],
                        iteration=review["iteration"],
                        reviewer_id=review["reviewer_id"],
                        scores=new_scores,
                        aggregate=agg,
                        decision=dec,
                        notes=edit_notes.value.strip(),
                        reflection_decision=_r_dec,
                        preflection_decision=_p_dec,
                        reflection_aggregate=_r_agg,
                        preflection_aggregate=_p_agg,
                    )
                    dlg.close()
                    ui.notify("Review updated", type="positive")
                    on_saved()

                ui.button("Save", on_click=_save, color="primary")

        dlg.open()

    def reviews_actions():
        def _do_upload():
            if force_upload():
                ui.notify("Uploaded to HuggingFace", type="positive")
            else:
                ui.notify("Backup not configured (BACKUP_REPO not set)", type="warning")

        ui.button(icon="cloud_upload", on_click=_do_upload).props(
            "flat dense size=sm"
        ).tooltip("Force upload to HuggingFace").style("color:#666;")
        ui.button(
            "Dashboard",
            icon="dashboard",
            on_click=lambda: ui.navigate.to("/pipeline"),
        ).classes("text-white").props("flat dense")
        ui.button(
            "Review",
            icon="rate_review",
            on_click=lambda: ui.navigate.to("/pipeline/review"),
        ).classes("text-white").props("flat dense")

    render_header(viewer_id, active_phase=2, right_slot=reviews_actions)

    all_reviews = load_reviews()
    review_comments = load_review_comments()
    reviews_runs = load_runs()
    reviews_run_by_iter = {r["iteration"]: r for r in reviews_runs}

    # Build items index for all reviewed iterations
    items_by_key: dict[tuple[str, int], dict] = {}
    seen_iters: set[int] = set()
    for r in all_reviews:
        if r["iteration"] not in seen_iters:
            seen_iters.add(r["iteration"])
            for item in load_items_for_iteration(r["iteration"]):
                items_by_key[(item["item_id"], item["iteration"])] = item

    if not all_reviews:
        with ui.column().classes("absolute-center items-center"):
            ui.label("No reviews yet.").classes("text-h6 text-grey-6")
        return

    @ui.refreshable
    def render_reviews():
        nonlocal all_reviews, review_comments
        all_reviews = load_reviews()
        review_comments = load_review_comments()

        # Group by (judge_prompt, judge_model) so reviews are organised by the
        # prompt version that produced the judgment, not by iteration. Multiple
        # iterations share the same prompt and we care about per-prompt
        # calibration, not per-iteration.
        import re as _re_grp

        by_judge: dict[tuple[str, str], list[dict]] = {}
        for r in all_reviews:
            run_info = reviews_run_by_iter.get(r["iteration"], {})
            key = (
                run_info.get("judge_prompt", "unknown"),
                run_info.get("judge_model", "unknown"),
            )
            by_judge.setdefault(key, []).append(r)

        def _judge_sort_key(key: tuple[str, str]) -> tuple:
            m = _re_grp.search(r"_v(\d+)", key[0])
            return (key[1], int(m.group(1)) if m else 0)

        sorted_keys = sorted(by_judge, key=_judge_sort_key, reverse=True)
        latest_key = sorted_keys[0] if sorted_keys else None

        for group_key in sorted_keys:
            reviews = by_judge[group_key]
            gp_name, gm_name = group_key
            with (
                ui.expansion(
                    f"{gp_name} / {gm_name} ({len(reviews)} reviews)",
                    icon="rate_review",
                )
                .classes("w-full q-mx-md q-mt-sm")
                .props("default-opened" if group_key == latest_key else "")
            ):
                for r in sorted(reviews, key=lambda x: x["timestamp"], reverse=True):
                    item = items_by_key.get((r["item_id"], r["iteration"]))
                    review_key = (r["item_id"], r["iteration"], r["reviewer_id"])

                    with ui.card().classes("w-full q-pa-sm q-mb-sm"):
                        # Header: reviewer, decision, score, timestamp
                        with ui.row().classes("items-center gap-2 w-full"):
                            ui.badge(r["reviewer_id"], color="blue-grey").props(
                                "outline"
                            )
                            decision_color = (
                                "green" if r["decision"] == "accept" else "red"
                            )
                            ui.badge(r["decision"].upper(), color=decision_color)
                            ui.badge(
                                f"Score: {r['aggregate']:.2f}", color=decision_color
                            ).props("outline")
                            split_name = review_split(r["item_id"])
                            ui.badge(
                                split_name,
                                color=(
                                    "purple" if split_name == "validation" else "indigo"
                                ),
                            ).props("outline")
                            rev_run = reviews_run_by_iter.get(r["iteration"], {})
                            rev_gen = rev_run.get("generator_model", "")
                            if rev_gen:
                                ui.badge(f"gen:{rev_gen}", color="teal").props(
                                    "outline"
                                )
                            ui.label(r["timestamp"][:19]).classes(
                                "text-caption text-grey-6"
                            )
                            ui.label(f"{r['item_id'][:12]}").classes(
                                "text-caption text-grey-5"
                            )
                            ui.space()

                            # Edit button
                            def make_edit(rev=r):
                                def do_edit():
                                    _open_edit_dialog(
                                        rev, on_saved=render_reviews.refresh
                                    )

                                return do_edit

                            ui.button(
                                icon="edit",
                                on_click=make_edit(),
                            ).props("flat dense size=xs color=primary")

                            # Delete button
                            def make_delete(
                                iid=r["item_id"],
                                it=r["iteration"],
                                rid=r["reviewer_id"],
                            ):
                                def do_delete():
                                    delete_review(iid, it, rid)
                                    ui.notify("Review deleted", type="info")
                                    render_reviews.refresh()

                                return do_delete

                            ui.button(
                                icon="delete",
                                on_click=make_delete(),
                            ).props("flat dense size=xs color=negative")

                        # Per-part scores
                        scores = r["scores"]
                        is_per_part = scores and isinstance(
                            next(iter(scores.values())), dict
                        )
                        if is_per_part:
                            with ui.row().classes("gap-4 q-mt-xs"):
                                for part, part_scores in scores.items():
                                    if isinstance(part_scores, dict) and part_scores:
                                        score_str = " ".join(
                                            f"{dim[:3]}={val}"
                                            for dim, val in part_scores.items()
                                        )
                                        ui.label(f"{part}: {score_str}").classes(
                                            "text-caption"
                                        )
                        else:
                            score_str = " ".join(
                                f"{d[:3]}={v}" for d, v in scores.items()
                            )
                            ui.label(f"Scores: {score_str}").classes("text-caption")

                        # Notes
                        if r.get("notes"):
                            ui.label(r["notes"]).classes("text-body2 q-mt-xs").style(
                                "white-space: pre-wrap; padding-left: 8px; "
                                "border-left: 2px solid #555;"
                            )

                        # Source text dropdown
                        if item:
                            with ui.expansion(
                                "Source Text",
                                icon="article",
                            ).classes("w-full q-mt-xs"):
                                ui.html(
                                    render_source_text(
                                        item["text"],
                                        item["reflection_point"],
                                    )
                                ).style(
                                    "line-height: 1.6; font-family: Georgia, serif; "
                                    "white-space: pre-wrap; font-size: 0.95em; padding: 8px;"
                                )

                            # Generation dropdown
                            with ui.expansion(
                                "LLM Generation",
                                icon="smart_toy",
                            ).classes("w-full q-mt-xs"):
                                ui.label("Analysis").classes(
                                    "text-overline text-grey-7"
                                )
                                ui.label(item.get("analysis", "")).classes(
                                    "text-body2"
                                ).style("white-space: pre-wrap;")
                                _rv_parts = _build_part_display(item)
                                for _rv_label, _rv_text in _rv_parts:
                                    ui.label(_rv_label).classes(
                                        "text-overline text-grey-7 q-mt-sm"
                                    )
                                    ui.label(_rv_text).classes("text-body2").style(
                                        "white-space: pre-wrap;"
                                    )
                                for _ce_label, _ce_field in (
                                    (
                                        "Preflection charter",
                                        "preflection_charter_elements",
                                    ),
                                    (
                                        "Reflection charter",
                                        "reflection_charter_elements",
                                    ),
                                ):
                                    _ce_elements = item.get(_ce_field, [])
                                    if _ce_elements:
                                        ui.label(_ce_label).classes(
                                            "text-overline text-grey-7 q-mt-sm"
                                        )
                                        with ui.row().classes("gap-1"):
                                            for eid in _ce_elements:
                                                ui.badge(
                                                    eid, color="blue-grey-3"
                                                ).props("outline")

                        # Comment thread
                        comments = review_comments.get(review_key, [])
                        with ui.expansion(
                            f"Comments ({len(comments)})",
                            icon="chat_bubble_outline",
                        ).classes("w-full q-mt-xs"):
                            for c in comments:
                                with ui.row().classes("items-start gap-2 q-mb-xs"):
                                    ui.label(c["commenter_id"]).classes(
                                        "text-caption text-weight-bold"
                                    )
                                    ui.label(c["timestamp"][:16]).classes(
                                        "text-caption text-grey-5"
                                    )

                                    def make_delete_comment(cid=c["id"]):
                                        def do_delete():
                                            delete_review_comment(cid)
                                            ui.notify("Comment deleted", type="info")
                                            render_reviews.refresh()

                                        return do_delete

                                    ui.button(
                                        icon="delete",
                                        on_click=make_delete_comment(),
                                    ).props("flat dense size=xs color=negative")
                                ui.label(c["comment"]).classes(
                                    "text-body2 q-mb-sm"
                                ).style("white-space: pre-wrap; padding-left: 8px;")

                            if viewer_id:
                                comment_input = (
                                    ui.input(
                                        placeholder="Add a comment...",
                                    )
                                    .classes("w-full")
                                    .props("dense outlined")
                                )

                                def make_submit(
                                    iid=r["item_id"],
                                    it=r["iteration"],
                                    rid=r["reviewer_id"],
                                    inp=comment_input,
                                ):
                                    def do_submit():
                                        assert (
                                            inp.value and inp.value.strip()
                                        ), "Comment cannot be empty"
                                        save_review_comment(
                                            iid, it, rid, viewer_id, inp.value.strip()
                                        )
                                        ui.notify("Comment added", type="positive")
                                        render_reviews.refresh()

                                    return do_submit

                                ui.button(
                                    "Post",
                                    on_click=make_submit(),
                                    color="primary",
                                ).props("flat dense size=sm")

    render_reviews()
