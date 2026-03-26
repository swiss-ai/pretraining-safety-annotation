"""Phase 2 dashboard pages: /pipeline and /pipeline/review routes."""

from __future__ import annotations

import difflib
import statistics
import threading
from pathlib import Path

from nicegui import app, ui

from pipeline.config import AppConfig, load_config, resolve_prompt_path
from pipeline.dashboard import render_header
from pipeline.dashboard.shared import CHARTER_TEXT, render_source_text
from pipeline.phase2.storage import (
    build_review_lookup,
    delete_review,
    delete_review_comment,
    load_item_across_iterations,
    load_items_for_iteration,
    load_judge_correlations,
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

        for part in ("preflection", "reflection"):
            part_j = judgment.get(part, {})
            part_scores = part_j.get("scores", {})
            human_part = review_scores.get(part, {}) if is_per_part else review_scores
            for dim, human_score in human_part.items():
                if dim in part_scores:
                    paired_scores.setdefault(f"{part}_{dim}", []).append(
                        (part_scores[dim], human_score)
                    )

        aggregate_pairs.append((judgment["aggregate"], review["aggregate"]))
        decision_pairs.append((judgment["decision"], review["decision"]))

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


def _load_correlation_pairs(split: str | None = None) -> list[dict]:
    """Load and join correlations with human reviews. Returns paired records.

    Each record has: judge_prompt, judge_model, iteration, item_id, judge_agg,
    human_agg, judge_dec, human_dec.

    split: "train", "validation", or None (all).
    """
    correlations = load_judge_correlations()
    review_by_item = build_review_lookup(split=split)

    pairs = []
    for c in correlations:
        rev = review_by_item.get((c["item_id"], c["iteration"]))
        if not rev:
            continue
        pairs.append(
            {
                "judge_prompt": c["judge_prompt"],
                "judge_model": c["judge_model"],
                "iteration": c["iteration"],
                "item_id": c["item_id"],
                "judge_agg": c["judgment"].get("aggregate", 0),
                "human_agg": rev.get("aggregate", 0),
                "judge_dec": c["judgment"].get("decision", ""),
                "human_dec": rev.get("decision", ""),
            }
        )
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

    return {
        vk: {
            "pearson": _pearson(score_pairs[vk]),
            "kappa": _cohens_kappa(decision_pairs.get(vk, [])),
        }
        for vk in score_pairs
    }


def _compute_correlation_by_judge_version(
    gen_filter: str | None = None,
    iter_to_gen: dict[int, str] | None = None,
    _pairs: list[dict] | None = None,
) -> dict[tuple[str, str], dict]:
    """Compute aggregate Pearson correlation and Cohen's kappa for each (judge_prompt, judge_model).

    Uses judge_correlations table (re-judgments) paired with human reviews.
    Returns {(judge_prompt, judge_model): {"pearson": float|None, "kappa": float|None}}.

    Pass _pairs (from _load_correlation_pairs) to avoid reloading from DB.
    """
    pairs = _pairs if _pairs is not None else _load_correlation_pairs()
    if gen_filter and iter_to_gen:
        pairs = [p for p in pairs if iter_to_gen.get(p["iteration"]) == gen_filter]
    return _aggregate_correlation_pairs(pairs)


def _compute_calibration_from_correlations(
    judge_prompt: str, judge_model: str, split: str | None = None
) -> dict:
    """Compute per-dimension Pearson correlations from judge_correlations data for a specific version.

    Returns same shape as _compute_calibration: {dimension_correlations, n_paired}.
    split: "train", "validation", or None (all).
    """
    correlations = load_judge_correlations()
    review_by_item = build_review_lookup(split=split)

    paired_scores: dict[str, list[tuple[float, float]]] = {}
    n_paired = 0
    for c in correlations:
        if c["judge_prompt"] != judge_prompt or c["judge_model"] != judge_model:
            continue
        key = (c["item_id"], c["iteration"])
        rev = review_by_item.get(key)
        if not rev:
            continue
        n_paired += 1
        j = c["judgment"]
        rev_scores = rev.get("scores", {})
        is_per_part = rev_scores and isinstance(
            next(iter(rev_scores.values()), None), dict
        )
        for part in ("preflection", "reflection"):
            j_scores = j.get(part, {}).get("scores", {})
            h_scores = rev_scores.get(part, {}) if is_per_part else rev_scores
            for dim, h_val in h_scores.items():
                if dim in j_scores:
                    paired_scores.setdefault(f"{part}_{dim}", []).append(
                        (j_scores[dim], h_val)
                    )

    return {
        "dimension_correlations": {
            dim: _pearson(pairs) for dim, pairs in paired_scores.items()
        },
        "n_paired": n_paired,
    }


def _render_calibration_from_items(cal: dict) -> None:
    """Render calibration panel content from _compute_calibration result (fallback when no correlations)."""
    with ui.row().classes("gap-8"):
        with ui.column():
            ui.label(f"Paired reviews: {cal['n_paired']}").classes("text-body2")
            agg = cal["aggregate_correlation"]
            ui.label(
                f"Aggregate correlation: {agg:.3f}"
                if agg is not None
                else "Aggregate correlation: N/A"
            ).classes("text-body2")
            agr = cal["decision_agreement"]
            ui.label(
                f"Decision agreement: {agr:.1%}"
                if agr is not None
                else "Decision agreement: N/A"
            ).classes("text-body2")

        dim_corrs = cal["dimension_correlations"]
        for part in ("preflection", "reflection"):
            part_dims = {k: v for k, v in dim_corrs.items() if k.startswith(f"{part}_")}
            if not part_dims:
                continue
            with ui.column():
                ui.label(f"{part.title()} correlations:").classes(
                    "text-body2 text-weight-bold"
                )
                for dim, corr in part_dims.items():
                    short = dim.replace(f"{part}_", "")
                    val = f"{corr:.3f}" if corr is not None else "N/A"
                    ui.label(f"  {short}: {val}").classes("text-body2")


def _render_judge_scores(judgment: dict, judge_model: str = "") -> None:
    """Render judge score details for a single judgment dict."""
    with ui.row().classes("gap-4"):
        ui.badge(
            f"Aggregate: {judgment['aggregate']:.1f}",
            color="green" if judgment["decision"] == "accept" else "red",
        )
        ui.badge(
            judgment["decision"].upper(),
            color="green" if judgment["decision"] == "accept" else "red",
        )
        if judge_model:
            ui.badge(judge_model, color="teal").props("outline")
        jp = judgment.get("judge_prompt", "")
        if jp:
            ui.badge(jp, color="blue-grey").props("outline")

    for part in ("preflection", "reflection"):
        part_j = judgment.get(part, {})
        if not part_j:
            continue
        ui.label(f"{part.title()} ({part_j.get('aggregate', 0):.1f})").classes(
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
                        parts = key.split("_", 1)
                        imp_role = parts[0] if len(parts) == 2 else "?"
                        imp_alias = parts[1] if len(parts) == 2 else key
                        imp_status = data.get("status", "pending")
                        label, color = _status_badge(imp_status)
                        with (
                            ui.card()
                            .classes("flex-1 q-pa-sm")
                            .style("min-width: 300px;")
                        ):
                            with ui.row().classes("items-center gap-2"):
                                ui.label(f"{imp_role.title()}: {imp_alias}").classes(
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


def _render_acceptance_rate_chart(runs: list[dict], iter_stats: list[dict]) -> None:
    """Render per-generator-model acceptance rate bar chart with binomial 95% CI.

    For each generator model, picks the run with the latest judge prompt version
    and highest iteration, then computes acceptance rate + confidence interval.
    """
    import math
    import re as _re_ar

    if not iter_stats:
        return

    # Group iter_stats by generator_model
    by_gen: dict[str, list[dict]] = {}
    for s in iter_stats:
        gen = s.get("generator_model", "unknown")
        by_gen.setdefault(gen, []).append(s)

    models = []
    rates = []
    ci_low = []
    ci_high = []

    for gen_model in sorted(by_gen):
        entries = by_gen[gen_model]
        # Pick entry with latest judge prompt version, then highest iteration
        best = max(
            entries,
            key=lambda e: (
                (
                    int(m.group(1))
                    if (m := _re_ar.search(r"_v(\d+)", e.get("judge_prompt", "")))
                    else 0
                ),
                e["iteration"],
            ),
        )
        n_judged = best["n_acc"] + best["n_rej"]
        if n_judged == 0:
            continue
        p = best["n_acc"] / n_judged
        # Binomial 95% CI: p ± 1.96 * sqrt(p*(1-p)/n)
        margin = 1.96 * math.sqrt(p * (1 - p) / n_judged) if n_judged > 1 else 0
        models.append(gen_model)
        rates.append(round(p * 100, 1))
        ci_low.append(round(max(0, p - margin) * 100, 1))
        ci_high.append(round(min(1, p + margin) * 100, 1))

    if not models:
        return

    with ui.card().classes("w-full q-mx-md q-mt-md q-pa-md"):
        ui.label("Acceptance Rate by Generator").classes("text-h6 text-weight-bold")
        # Bar chart with error bars via markLine-style scatter overlay
        bar_series = {
            "name": "Accept %",
            "type": "bar",
            "data": rates,
            "itemStyle": {"color": "#4caf50"},
            "barMaxWidth": 60,
        }
        # Error bars as scatter points at the mean, with markLine whiskers
        # symbol "none" hides the scatter dot; markLine draws CI range
        error_series = {
            "name": "95% CI",
            "type": "scatter",
            "data": [[i, rates[i]] for i in range(len(rates))],
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
        chart_opts = {
            "xAxis": {"type": "category", "data": models},
            "yAxis": {"type": "value", "name": "Accept %", "min": 0, "max": 100},
            "series": [bar_series, error_series],
            "tooltip": {"trigger": "axis"},
            "grid": {"bottom": 60},
        }
        ui.echart(chart_opts).classes("w-full").style("height: 300px;")


def _render_calibration_bar_chart(
    pairs: list[dict] | None = None, split: str | None = None
) -> None:
    """Render per-judge-model calibration bar chart (Pearson r and Cohen's kappa).

    Groups by judge_model, uses the latest judge prompt version per model.
    pairs: pre-loaded correlation pairs (avoids DB re-read). If None, loads from DB.
    split: filter pre-loaded pairs by "train" or "validation".
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

    with ui.card().classes("w-full q-mx-md q-mt-md q-pa-md"):
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
            for part in ("preflection", "reflection"):
                part_usage = j.get(part, {}).get("usage", {})
                val = part_usage.get("reasoning_tokens")
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


def _render_judge_judge_correlations(runs: list[dict]) -> None:
    """Render pairwise judge-judge correlation matrix/table.

    For each pair of judge models, computes Pearson r on scores and Cohen's kappa
    on decisions for items judged by both.
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
                    agg_a = j_a.get("aggregate")
                    agg_b = j_b.get("aggregate")
                    if agg_a is not None and agg_b is not None:
                        score_pairs.append((agg_a, agg_b))
                    dec_a = j_a.get("decision", "")
                    dec_b = j_b.get("decision", "")
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


@ui.page("/pipeline")
def pipeline_monitoring_page():
    """Pipeline monitoring dashboard: iteration table, trends, calibration."""
    viewer_id = app.storage.user.get("annotator_id", "")

    def pipeline_actions():
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
        n_acc = sum(1 for i in judged if i["judgment"]["decision"] == "accept")
        scores = [i["judgment"]["aggregate"] for i in judged]
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

    # --- Judge Calibration Panel (uses latest judge correlations) ---
    with ui.card().classes("w-full q-mx-md q-mt-md q-pa-md"):
        ui.label("Judge Calibration").classes("text-h6 text-weight-bold")
        version_corrs_panel = _compute_correlation_by_judge_version()
        if not version_corrs_panel:
            cal = _compute_calibration(all_reviews, items_by_key)
            if cal["n_paired"] == 0:
                ui.label(
                    "No human reviews yet — submit reviews to see calibration metrics."
                ).classes("text-grey-6")
            else:
                _render_calibration_from_items(cal)
        else:
            import re as _re_panel

            latest_key = max(
                version_corrs_panel.keys(),
                key=lambda k: (
                    int(m.group(1)) if (m := _re_panel.search(r"_v(\d+)", k[0])) else 0
                ),
            )
            latest = version_corrs_panel[latest_key]
            latest_prompt, latest_model = latest_key

            # Also compute per-dimension calibration from correlations data
            cal_from_corr = _compute_calibration_from_correlations(
                latest_prompt, latest_model
            )

            with ui.row().classes("gap-8"):
                with ui.column():
                    ui.label(f"Judge: {latest_prompt} / {latest_model}").classes(
                        "text-body2 text-weight-bold"
                    )
                    n_paired = cal_from_corr["n_paired"]
                    ui.label(f"Paired reviews: {n_paired}").classes("text-body2")
                    pr = latest["pearson"]
                    ui.label(
                        f"Score Pearson r: {pr:.3f}"
                        if pr is not None
                        else "Score Pearson r: N/A"
                    ).classes("text-body2")
                    kp = latest["kappa"]
                    ui.label(
                        f"Decision Cohen's κ: {kp:.3f}"
                        if kp is not None
                        else "Decision Cohen's κ: N/A"
                    ).classes("text-body2")

                dim_corrs = cal_from_corr.get("dimension_correlations", {})
                for part in ("preflection", "reflection"):
                    part_dims = {
                        k: v for k, v in dim_corrs.items() if k.startswith(f"{part}_")
                    }
                    if not part_dims:
                        continue
                    with ui.column():
                        ui.label(f"{part.title()} correlations:").classes(
                            "text-body2 text-weight-bold"
                        )
                        for dim, corr in part_dims.items():
                            short = dim.replace(f"{part}_", "")
                            val = f"{corr:.3f}" if corr is not None else "N/A"
                            ui.label(f"  {short}: {val}").classes("text-body2")

    # --- Trend Charts ---
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

    # Collect distinct generator models for dropdown
    all_gen_models = sorted({s.get("generator_model", "unknown") for s in iter_stats})

    if len(iter_stats) >= 2:

        def _build_trend_chart(gen_filter: str | None) -> dict:
            filtered = (
                iter_stats
                if gen_filter is None
                else [s for s in iter_stats if s.get("generator_model") == gen_filter]
            )
            labels = [
                f"Iter {s['iteration']} ({_SOURCE_LABEL.get(s.get('source', 'manual'), '')})"
                for s in filtered
            ]
            accept_data = [
                {
                    "value": s["accept_rate"],
                    "symbol": _SOURCE_MARKER.get(s.get("source", "manual"), "circle"),
                    "symbolSize": 8,
                }
                for s in filtered
            ]
            score_data = [
                {
                    "value": round(s["mean_score"], 2),
                    "symbol": _SOURCE_MARKER.get(s.get("source", "manual"), "circle"),
                    "symbolSize": 8,
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

        with ui.row().classes("w-full q-mx-md q-mt-md gap-4"):
            with ui.card().classes("flex-1 q-pa-md"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("Acceptance Rate & Mean Score").classes(
                        "text-subtitle2 text-weight-bold"
                    )
                    if len(all_gen_models) > 1:
                        gen_options = ["All"] + all_gen_models
                        default_gen = all_gen_models[0]
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
                ui.label("Judge-Human Correlation by Judge Version").classes(
                    "text-subtitle2 text-weight-bold"
                )
                # Load correlation data once, filter in memory on dropdown change
                iter_to_gen = {
                    r["iteration"]: r.get("generator_model", "unknown") for r in runs
                }
                cached_corr_pairs = _load_correlation_pairs()
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

                    import re as _re

                    all_prompts = sorted(
                        {p for p, _ in version_corrs},
                        key=lambda p: (
                            int(m.group(1)) if (m := _re.search(r"_v(\d+)", p)) else 0
                        ),
                    )

                    # Correlation chart state: tracks judge model, generator filter, and review split
                    corr_state = {
                        "judge_model": default_model,
                        "gen_filter": None,
                        "split": None,
                    }

                    def _build_corr_chart_filtered() -> dict:
                        """Build correlation chart using current corr_state filters."""
                        gf = corr_state["gen_filter"]
                        sp = corr_state["split"]
                        # Filter pairs by split
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
                        return {
                            "xAxis": {"type": "category", "data": all_prompts},
                            "yAxis": {
                                "type": "value",
                                "name": "Correlation",
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
                                    "name": "Decision Cohen's κ",
                                    "type": "line",
                                    "data": kappa_data,
                                    "lineStyle": {"color": "#4caf50"},
                                    "itemStyle": {"color": "#4caf50"},
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

    # --- Per-Generator Acceptance Rate Bar Chart ---
    _render_acceptance_rate_chart(runs, iter_stats)

    # --- Per-Judge Calibration Bar Chart (with split selector) ---
    _cal_bar_pairs = _load_correlation_pairs()
    _cal_bar_container = ui.column().classes("w-full")
    with _cal_bar_container:
        _render_calibration_bar_chart(pairs=_cal_bar_pairs)

    def _on_cal_bar_split(e):
        sp = None if e.value == "all" else e.value
        _cal_bar_container.clear()
        with _cal_bar_container:
            _render_calibration_bar_chart(pairs=_cal_bar_pairs, split=sp)

    ui.select(
        ["all", "train", "validation"],
        value="all",
        label="Review Split",
        on_change=_on_cal_bar_split,
    ).classes("w-36 q-mx-md")

    # --- API Model Statistics (collapsible) ---
    _render_api_stats_panel(runs, items_by_key)

    # --- Judge-Judge Correlations ---
    _render_judge_judge_correlations(runs)

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
                        parts = key.split("_", 1)
                        imp_role = parts[0] if len(parts) == 2 else "?"
                        imp_alias = parts[1] if len(parts) == 2 else key
                        with (
                            ui.card()
                            .classes("flex-1 q-pa-sm")
                            .style("min-width: 280px;")
                        ):
                            with ui.row().classes("items-center gap-2"):
                                ui.label(f"{imp_role.title()}: {imp_alias}").classes(
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
            from pipeline.phase2.loop import (
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

                parts = key.split("_", 1)
                imp_role = parts[0] if len(parts) == 2 else "?"
                imp_alias = parts[1] if len(parts) == 2 else key
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
            from pipeline.phase2.loop import improver_log_path as _log_path

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
                        parts = key.split("_", 1)
                        imp_role = parts[0] if len(parts) == 2 else "?"
                        imp_alias = parts[1] if len(parts) == 2 else key
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
            from pipeline.phase2.loop import improver_log_path as _log_path

            for key in improvers:
                if key not in _log_els:
                    continue
                parts = key.split("_", 1)
                imp_role = parts[0] if len(parts) == 2 else "?"
                imp_alias = parts[1] if len(parts) == 2 else key
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
            from pipeline.phase2.loop import read_status

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
        from pipeline.phase2.loop import read_status as _read_initial

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
                    return [m.alias for m in cfg.phase2.judge_models]
                return [m.alias for m in cfg.phase2.generator_models]

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
                    from pipeline.phase2.run import (
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


@ui.page("/pipeline/review")
def pipeline_review_page():
    """Human review of LLM-generated reflections with per-dimension scoring."""
    viewer_id = app.storage.user.get("annotator_id", "")
    if not viewer_id:
        ui.navigate.to("/")
        return

    def review_actions():
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
    dimensions = cfg.phase2.scoring.dimensions

    # Build run metadata lookups
    run_by_iter = {r["iteration"]: r for r in runs}
    # group_id → list of iterations in that group
    group_iters: dict[str, list[int]] = {}
    for r in runs:
        gid = r.get("group_id")
        if gid:
            group_iters.setdefault(gid, []).append(r["iteration"])

    def _prompt_version(filename: str) -> str:
        """Extract version from prompt filename, e.g. 'generator_v1.md' -> 'v1'."""
        import re

        m = re.search(r"(v\d+)", filename)
        return m.group(1) if m else filename

    def _version_sort_key(v: str) -> int:
        return int(v[1:]) if v.startswith("v") and v[1:].isdigit() else 0

    all_gen_models = sorted({r.get("generator_model", "unknown") for r in runs})
    all_judge_models = sorted({r.get("judge_model", "unknown") for r in runs})
    all_gen_prompts = sorted(
        {_prompt_version(r["gen_prompt"]) for r in runs}, key=_version_sort_key
    )
    all_judge_prompts = sorted(
        {_prompt_version(r["judge_prompt"]) for r in runs}, key=_version_sort_key
    )

    # Per-model latest prompt versions (for "latest" filter option)
    _latest_gen_prompt: dict[str, str] = {}
    for r in runs:
        model = r.get("generator_model", "unknown")
        v = _prompt_version(r["gen_prompt"])
        if model not in _latest_gen_prompt or _version_sort_key(v) > _version_sort_key(
            _latest_gen_prompt[model]
        ):
            _latest_gen_prompt[model] = v

    _latest_judge_prompt: dict[str, str] = {}
    for r in runs:
        model = r.get("judge_model", "unknown")
        v = _prompt_version(r["judge_prompt"])
        if model not in _latest_judge_prompt or _version_sort_key(
            v
        ) > _version_sort_key(_latest_judge_prompt[model]):
            _latest_judge_prompt[model] = v

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
    with ui.row().classes("q-px-md q-mt-md items-center gap-4"):
        gen_model_select = ui.select(
            options=["All"] + all_gen_models,
            value="All",
            label="Generator",
        ).classes("w-40")

        gen_prompt_select = ui.select(
            options=["latest", "All"] + all_gen_prompts,
            value=state["gen_prompt"],
            label="Gen Prompt",
        ).classes("w-32")

        judge_model_select = ui.select(
            options=["All"] + all_judge_models,
            value="All",
            label="Judge",
        ).classes("w-40")

        judge_prompt_select = ui.select(
            options=["latest", "All"] + all_judge_prompts,
            value=state["judge_prompt"],
            label="Judge Prompt",
        ).classes("w-32")

        sort_select = ui.select(
            options=[
                "Low judge score",
                "High judge score",
                "Low safety score",
                "High safety score",
                "Default order",
            ],
            value="Low judge score",
            label="Sort",
        ).classes("w-48")

    # --- Main split panel ---
    with (
        ui.splitter(value=35)
        .classes("w-full")
        .style("height: calc(100vh - 120px)") as splitter
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
                ui.label("Charter").classes("text-subtitle2 text-weight-bold")
                ui.markdown(charter_text, extras=["tables"]).classes(
                    "text-body2"
                ).style("flex: 1; overflow-y: auto; padding: 8px; line-height: 1.6;")

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
                gen_section = ui.column().classes("w-full gap-2")

                ui.separator()

                # Judge scores (hidden by default)
                judge_expansion = ui.expansion("Judge Scores", icon="gavel").classes(
                    "w-full"
                )
                with judge_expansion:
                    judge_section = ui.column().classes("w-full gap-1")

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

                DIM_HINTS = {
                    "relevance": "Does it identify what matters?",
                    "specificity": "Is it specific to this text?",
                    "charter_grounding": "Are charter refs appropriate and complete?",
                    "voice_tone": "Correct voice, natural, concise?",
                }

                # Per-part scoring: {part: {dim: slider}}
                score_inputs: dict[str, dict[str, ui.slider]] = {}
                for part in ("preflection", "reflection"):
                    ui.label(part.title()).classes("text-overline text-grey-7 q-mt-sm")
                    score_inputs[part] = {}
                    for dim in dimensions:
                        with ui.column().classes("w-full gap-0"):
                            with ui.row().classes("items-center gap-2 w-full"):
                                ui.label(dim.replace("_", " ").title()).classes("w-40")
                                slider = ui.slider(min=1, max=5, value=3).classes(
                                    "flex-1"
                                )
                                score_label = ui.label("3").classes("w-8")
                                slider.on(
                                    "update:model-value",
                                    lambda e, lbl=score_label: lbl.set_text(
                                        str(int(e.args))
                                    ),
                                )
                                score_inputs[part][dim] = slider
                            hint = DIM_HINTS.get(dim, "")
                            if hint:
                                ui.label(hint).classes(
                                    "text-caption text-grey-6"
                                ).style("margin-left: 160px; margin-top: -4px;")

                threshold = cfg.phase2.scoring.accept_threshold
                review_status_label = ui.label("").classes(
                    "text-caption text-weight-bold"
                )

                def _update_review_status():
                    all_vals = [
                        int(slider.value)
                        for dims in score_inputs.values()
                        for slider in dims.values()
                    ]
                    agg = sum(all_vals) / len(all_vals) if all_vals else 0
                    has_floor = any(v <= 2 for v in all_vals)
                    decision = "reject" if has_floor or agg < threshold else "accept"
                    color = "green" if decision == "accept" else "red"
                    review_status_label.set_text(f"Avg: {agg:.2f} → {decision.upper()}")
                    review_status_label.style(f"color: {color};")

                # Wire up live updates from all sliders
                for dims in score_inputs.values():
                    for slider in dims.values():
                        slider.on(
                            "update:model-value", lambda _: _update_review_status()
                        )
                _update_review_status()

                notes_input = (
                    ui.textarea(
                        placeholder="Notes (optional)...",
                    )
                    .classes("w-full")
                    .props("outlined")
                )

                with ui.row().classes("w-full justify-end"):
                    submit_btn = ui.button(
                        "Submit Review",
                        on_click=lambda: submit_review(),
                        color="primary",
                    )

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
        iters = _filtered_iterations()
        all_items: list[dict] = []
        for it in iters:
            all_items.extend(load_items_for_iteration(it))
        # Deduplicate by item_id — keep highest iteration
        seen: dict[str, dict] = {}
        for item in all_items:
            key = item["item_id"]
            if key not in seen or item["iteration"] > seen[key]["iteration"]:
                seen[key] = item
        judged = [i for i in seen.values() if i.get("judgment")]
        sort = sort_select.value
        if sort == "Low judge score":
            judged.sort(key=lambda i: i["judgment"]["aggregate"])
        elif sort == "High judge score":
            judged.sort(key=lambda i: -i["judgment"]["aggregate"])
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

    def current_items_list() -> list[dict]:
        return get_sorted_items()

    def _first_unreviewed_pos(items: list[dict]) -> int:
        """Return index of the first item without a review from this viewer."""
        reviewed = load_latest_reviews()
        reviewed_keys = {(k[0], k[1]) for k in reviewed if k[2] == viewer_id}
        for i, item in enumerate(items):
            if (item["item_id"], item["iteration"]) not in reviewed_keys:
                return i
        return 0

    def update_display():
        items = current_items_list()
        if not items:
            nav_label.set_text("No judged items for this filter")
            return

        state["pos"] = max(0, min(state["pos"], len(items) - 1))
        item = items[state["pos"]]

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
            ui.label("Preflection").classes("text-overline text-grey-7")
            ui.label(item.get("preflection", "")).classes("text-body2").style(
                "white-space: pre-wrap;"
            )
            ui.label("Reflection").classes("text-overline text-grey-7")
            ui.label(item.get("reflection", "")).classes("text-body2").style(
                "white-space: pre-wrap;"
            )
            elements = item.get("charter_elements", [])
            if elements:
                ui.label("Charter Elements").classes("text-overline text-grey-7")
                with ui.row().classes("gap-1"):
                    for eid in elements:
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

        # Pre-fill from existing review
        latest_reviews = load_latest_reviews()
        existing = latest_reviews.get((item["item_id"], item["iteration"], viewer_id))
        if existing:
            ex_scores = existing["scores"]
            # Handle both per-part {part: {dim: int}} and legacy flat {dim: int} formats
            if ex_scores and isinstance(next(iter(ex_scores.values())), dict):
                for part, dims in score_inputs.items():
                    for dim, slider in dims.items():
                        slider.set_value(ex_scores.get(part, {}).get(dim, 3))
            else:
                for part, dims in score_inputs.items():
                    for dim, slider in dims.items():
                        slider.set_value(ex_scores.get(dim, 3))
            notes_input.set_value(existing.get("notes", ""))
            review_existing_badge.set_text("Editing existing review")
            review_existing_badge.props("color=orange")
            submit_btn.set_text("Update Review")
        else:
            for dims in score_inputs.values():
                for slider in dims.values():
                    slider.set_value(3)
            notes_input.set_value("")
            review_existing_badge.set_text("New review")
            review_existing_badge.props("color=green")
            submit_btn.set_text("Submit Review")
        _update_review_status()

    def _show_gold_annotation(item_id: str):
        """Display the human annotation for a gold item."""
        from pipeline.phase1.storage import load_latest_annotations

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
                ui.label("Preflection").classes("text-overline text-grey-7")
                ui.label(rec["preflection"]).classes("text-body2").style(
                    "white-space: pre-wrap;"
                )
                ui.label("Reflection").classes("text-overline text-grey-7")
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
        aggregate = statistics.mean(all_vals)
        has_floor = any(v <= 2 for v in all_vals)
        decision = (
            "reject"
            if has_floor or aggregate < cfg.phase2.scoring.accept_threshold
            else "accept"
        )
        save_review(
            item_id=item["item_id"],
            iteration=item["iteration"],
            reviewer_id=viewer_id,
            scores=scores,
            aggregate=aggregate,
            decision=decision,
            notes=notes_input.value.strip(),
        )
        ui.notify("Review saved!", type="positive")
        navigate(1)

    def _on_filter_change():
        """Re-filter and reset position when any filter changes."""
        state["pos"] = 0
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
        update_display()

    gen_model_select.on_value_change(_on_gen_model_change)
    gen_prompt_select.on_value_change(_on_gen_prompt_change)
    judge_model_select.on_value_change(_on_judge_model_change)
    judge_prompt_select.on_value_change(_on_judge_prompt_change)
    sort_select.on_value_change(_on_sort_change)
    # Start at first unreviewed item
    items = current_items_list()
    state["pos"] = _first_unreviewed_pos(items) if items else 0
    update_display()


@ui.page("/pipeline/reviews")
def pipeline_reviews_page():
    """Review overview: browse all reviews, comment on them, delete them."""
    viewer_id = app.storage.user.get("annotator_id", "")

    def reviews_actions():
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

        # Group by iteration
        by_iter: dict[int, list[dict]] = {}
        for r in all_reviews:
            by_iter.setdefault(r["iteration"], []).append(r)

        for iteration in sorted(by_iter.keys(), reverse=True):
            reviews = by_iter[iteration]
            with (
                ui.expansion(
                    f"Iteration {iteration} ({len(reviews)} reviews)",
                    icon="rate_review",
                )
                .classes("w-full q-mx-md q-mt-sm")
                .props("default-opened" if iteration == max(by_iter) else "")
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
                                for part in ("preflection", "reflection"):
                                    part_scores = scores.get(part, {})
                                    if part_scores:
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
                                ui.label("Preflection").classes(
                                    "text-overline text-grey-7 q-mt-sm"
                                )
                                ui.label(item.get("preflection", "")).classes(
                                    "text-body2"
                                ).style("white-space: pre-wrap;")
                                ui.label("Reflection").classes(
                                    "text-overline text-grey-7 q-mt-sm"
                                )
                                ui.label(item.get("reflection", "")).classes(
                                    "text-body2"
                                ).style("white-space: pre-wrap;")
                                elements = item.get("charter_elements", [])
                                if elements:
                                    ui.label("Charter Elements").classes(
                                        "text-overline text-grey-7 q-mt-sm"
                                    )
                                    with ui.row().classes("gap-1"):
                                        for eid in elements:
                                            ui.badge(eid, color="blue-grey-3").props(
                                                "outline"
                                            )

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
