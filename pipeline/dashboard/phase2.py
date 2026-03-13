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
    delete_review,
    delete_review_comment,
    load_items_for_iteration,
    load_latest_reviews,
    load_loop_history,
    load_review_comments,
    load_reviews,
    load_runs,
    load_test_results,
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
        is_per_part = review_scores and isinstance(next(iter(review_scores.values())), dict)

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

    def _pearson(pairs: list[tuple[float, float]]) -> float | None:
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


def _phase_badge(status: str) -> tuple[str, str]:
    """Return (label, color) for a phase status badge."""
    colors = {
        "pending": "grey",
        "running": "blue",
        "done": "green",
        "error": "red",
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

    for group in difflib.SequenceMatcher(None, before_lines, after_lines).get_grouped_opcodes(8):
        for tag, a0, a1, b0, b1 in group:
            if tag == "equal":
                for i in range(a1 - a0):
                    rows.append((
                        a0 + i + 1, _esc(before_lines[a0 + i]), CTX_BG,
                        b0 + i + 1, _esc(after_lines[b0 + i]), CTX_BG,
                    ))
            elif tag == "replace":
                old_lines = before_lines[a0:a1]
                new_lines = after_lines[b0:b1]
                # Pair up lines for word-level diff, pad shorter side
                max_len = max(len(old_lines), len(new_lines))
                for i in range(max_len):
                    if i < len(old_lines) and i < len(new_lines):
                        old_html, new_html = _word_diff_html(old_lines[i], new_lines[i])
                        rows.append((a0 + i + 1, old_html, DEL_BG, b0 + i + 1, new_html, INS_BG))
                        n_del += 1
                        n_add += 1
                    elif i < len(old_lines):
                        rows.append((a0 + i + 1, _esc(old_lines[i]), DEL_BG, "", "", INS_BG))
                        n_del += 1
                    else:
                        rows.append(("", "", DEL_BG, b0 + i + 1, _esc(new_lines[i]), INS_BG))
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
        '</div>'
    )

    # Cell styles
    LN = ('padding:1px 8px;color:#636c76;text-align:right;user-select:none;'
          'vertical-align:top;white-space:nowrap;width:1%;font-size:0.85em;')
    CELL = ('padding:1px 10px;white-space:pre-wrap;word-break:break-word;'
            'vertical-align:top;line-height:1.55;')
    SEP = 'width:1px;background:#d0d7de;'

    html_rows: list[str] = []
    for left_ln, left_html, left_bg, right_ln, right_html, right_bg in rows:
        html_rows.append(
            f'<tr>'
            f'<td style="{LN}background:{left_bg};">{left_ln}</td>'
            f'<td style="{CELL}background:{left_bg};">{left_html}</td>'
            f'<td style="{SEP}"></td>'
            f'<td style="{LN}background:{right_bg};">{right_ln}</td>'
            f'<td style="{CELL}background:{right_bg};">{right_html}</td>'
            f'</tr>'
        )

    # Column headers
    header = (
        '<tr style="border-bottom:1px solid #d0d7de;background:#f6f8fa;">'
        '<td colspan="2" style="padding:6px 10px;color:#636c76;font-weight:600;'
        'font-size:0.85em;text-align:center;">Before</td>'
        f'<td style="{SEP}"></td>'
        '<td colspan="2" style="padding:6px 10px;color:#636c76;font-weight:600;'
        'font-size:0.85em;text-align:center;">After</td>'
        '</tr>'
    )

    table = (
        f'<table style="width:100%;border-collapse:collapse;font-family:\'SF Mono\','
        f'Menlo,Consolas,monospace;font-size:0.82em;table-layout:fixed;'
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
    before: dict[str, str], after: dict[str, str],
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
    """Render past loop runs from loop_history.jsonl."""
    history = load_loop_history()
    if not history:
        return

    with ui.expansion(
        f"Improver Loop History ({len(history)} runs)", icon="history",
    ).classes("w-full q-mx-md q-mt-md"):
        for i, run in enumerate(reversed(history)):
            run_idx = len(history) - i
            started = run.get("started_at", "?")[:19]
            finished = run.get("finished_at", "?")[:19]
            error = run.get("error")
            failed = bool(error)
            status_tag = " — FAILED" if failed else ""
            border_style = "border-left: 3px solid #f44336;" if failed else ""

            with ui.expansion(
                f"Loop #{run_idx} — {started}{status_tag}",
                icon="error" if failed else "history",
            ).classes("w-full").style(border_style):
                with ui.row().classes("items-center gap-2"):
                    ui.badge("FAILED" if failed else "DONE", color="red" if failed else "green")
                    ui.label(f"{started} → {finished}").classes("text-caption text-grey-6")
                    if run.get("model_alias"):
                        ui.badge(run["model_alias"], color="blue-grey").props("outline")

                if error:
                    ui.label(f"Error: {error}").classes("text-caption text-red q-mt-xs")

                # Phase cards with reasoning + logs
                run_logs = run.get("logs", {})
                with ui.row().classes("w-full gap-4 q-mt-sm"):
                    for phase_key, phase_label in [("phase_a", "Phase A: Judge"), ("phase_b", "Phase B: Generator")]:
                        phase = run.get(phase_key, {})
                        p_status = phase.get("status", "pending")
                        label, color = _phase_badge(p_status)
                        with ui.card().classes("flex-1 q-pa-sm"):
                            with ui.row().classes("items-center gap-2"):
                                ui.label(phase_label).classes("text-subtitle2 text-weight-bold")
                                ui.badge(label, color=color)
                            reasoning = phase.get("reasoning", "")
                            if reasoning:
                                with ui.expansion("Summary", icon="summarize", value=True).classes("w-full"):
                                    ui.markdown(reasoning).classes("text-body2").style(
                                        "font-size: 0.85em; max-height: 300px; overflow-y: auto;"
                                    )
                            else:
                                ui.label("No reasoning recorded.").classes("text-grey-6 text-caption")

                            phase_log = run_logs.get(phase_key, "")
                            if phase_log:
                                with ui.expansion("Full Log", icon="terminal").classes("w-full"):
                                    ui.code(phase_log, language="text").classes("w-full").style(
                                        "max-height: 400px; overflow-y: auto; font-size: 0.75em;"
                                    )

                # Prompt diffs — pair new versions against their predecessor
                diffs = _compute_prompt_diffs(
                    run.get("prompts_before", {}), run.get("prompts_after", {}),
                )
                if diffs:
                    with ui.expansion(
                        f"Prompt Changes ({len(diffs)} file{'s' if len(diffs) != 1 else ''})",
                        icon="difference",
                    ).classes("w-full q-mt-sm"):
                        for label, before_text, after_text, display_name in diffs:
                            with ui.expansion(label).classes("w-full"):
                                html = _prompt_diff_html(before_text, after_text, display_name)
                                ui.html(
                                    f'<div style="background:#0d1117;border:1px solid #30363d;'
                                    f'border-radius:6px;overflow:hidden;max-height:600px;'
                                    f'overflow-y:auto;">{html}</div>'
                                )
                elif run.get("prompts_before"):
                    ui.label("No prompt changes in this run.").classes("text-grey-6 text-caption q-mt-xs")


@ui.page("/pipeline")
def pipeline_monitoring_page():
    """Pipeline monitoring dashboard: iteration table, trends, calibration."""
    viewer_id = app.storage.user.get("annotator_id", "")

    def pipeline_actions():
        ui.button("Review", icon="rate_review",
                  on_click=lambda: ui.navigate.to("/pipeline/review"),
                  ).classes("text-white").props("flat dense")
        ui.button("All Reviews", icon="reviews",
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

        iter_stats.append({
            **run,
            "n_acc": n_acc,
            "n_rej": len(judged) - n_acc,
            "mean_score": mean_s,
            "accept_rate": accept_rate,
            "calibration": cal_iter,
        })

    # --- Judge Calibration Panel ---
    with ui.card().classes("w-full q-mx-md q-mt-md q-pa-md"):
        ui.label("Judge Calibration").classes("text-h6 text-weight-bold")
        cal = _compute_calibration(all_reviews, items_by_key)
        if cal["n_paired"] == 0:
            ui.label("No human reviews yet — submit reviews to see calibration metrics.").classes(
                "text-grey-6"
            )
        else:
            with ui.row().classes("gap-8"):
                with ui.column():
                    ui.label(f"Paired reviews: {cal['n_paired']}").classes("text-body2")
                    agg = cal["aggregate_correlation"]
                    ui.label(
                        f"Aggregate correlation: {agg:.3f}" if agg is not None else "Aggregate correlation: N/A"
                    ).classes("text-body2")
                    agr = cal["decision_agreement"]
                    ui.label(
                        f"Decision agreement: {agr:.1%}" if agr is not None else "Decision agreement: N/A"
                    ).classes("text-body2")

                dim_corrs = cal["dimension_correlations"]
                for part in ("preflection", "reflection"):
                    part_dims = {k: v for k, v in dim_corrs.items() if k.startswith(f"{part}_")}
                    if not part_dims:
                        continue
                    with ui.column():
                        ui.label(f"{part.title()} correlations:").classes("text-body2 text-weight-bold")
                        for dim, corr in part_dims.items():
                            short = dim.replace(f"{part}_", "")
                            val = f"{corr:.3f}" if corr is not None else "N/A"
                            ui.label(f"  {short}: {val}").classes("text-body2")

    # --- Trend Charts ---
    if len(iter_stats) >= 2:
        iter_labels = [f"Iter {s['iteration']}" for s in iter_stats]
        accept_rates = [s["accept_rate"] for s in iter_stats]
        mean_scores = [round(s["mean_score"], 2) for s in iter_stats]
        calibration_corrs = [s["calibration"]["aggregate_correlation"] for s in iter_stats]

        with ui.row().classes("w-full q-mx-md q-mt-md gap-4"):
            with ui.card().classes("flex-1 q-pa-md"):
                ui.label("Acceptance Rate & Mean Score").classes("text-subtitle2 text-weight-bold")
                ui.echart({
                    "xAxis": {"type": "category", "data": iter_labels},
                    "yAxis": [
                        {"type": "value", "name": "Accept %", "min": 0, "max": 100, "position": "left"},
                        {"type": "value", "name": "Mean Score", "min": 1, "max": 5, "position": "right"},
                    ],
                    "series": [
                        {"name": "Accept %", "type": "line", "data": accept_rates, "yAxisIndex": 0,
                         "itemStyle": {"color": "#4caf50"}},
                        {"name": "Mean Score", "type": "line", "data": mean_scores, "yAxisIndex": 1,
                         "itemStyle": {"color": "#2196f3"}},
                    ],
                    "tooltip": {"trigger": "axis"},
                    "legend": {"bottom": 0},
                }).classes("w-full").style("height: 250px;")

            with ui.card().classes("flex-1 q-pa-md"):
                ui.label("Judge-Human Correlation").classes("text-subtitle2 text-weight-bold")
                has_any = any(c is not None for c in calibration_corrs)
                if not has_any:
                    ui.label("No human reviews yet.").classes("text-grey-6")
                else:
                    corr_data = [round(c, 3) if c is not None else None for c in calibration_corrs]
                    ui.echart({
                        "xAxis": {"type": "category", "data": iter_labels},
                        "yAxis": {"type": "value", "name": "Pearson r", "min": -1, "max": 1},
                        "series": [
                            {"name": "Aggregate Correlation", "type": "line", "data": corr_data,
                             "itemStyle": {"color": "#ff9800"},
                             "connectNulls": True},
                        ],
                        "tooltip": {"trigger": "axis"},
                    }).classes("w-full").style("height: 250px;")

    # --- Iteration Table ---
    with ui.card().classes("w-full q-mx-md q-mt-md q-pa-md"):
        ui.label("Iterations").classes("text-h6 text-weight-bold")
        if not runs:
            ui.label("No iterations yet.").classes("text-grey-6")
        else:
            columns = [
                {"name": "iteration", "label": "#", "field": "iteration", "sortable": True},
                {"name": "model", "label": "Model", "field": "model"},
                {"name": "gen_prompt", "label": "Gen Prompt", "field": "gen_prompt"},
                {"name": "judge_prompt", "label": "Judge Prompt", "field": "judge_prompt"},
                {"name": "n_items", "label": "Items", "field": "n_items"},
                {"name": "n_gold", "label": "Gold", "field": "n_gold"},
                {"name": "timestamp", "label": "Time", "field": "timestamp"},
            ]
            rows = [
                {
                    **s,
                    "model": s.get("generator_model", s.get("model", "unknown")),
                    "timestamp": s["timestamp"][:19],
                    "accept_reject": f"{s['n_acc']}/{s['n_rej']}",
                    "mean_score": f"{s['mean_score']:.2f}",
                }
                for s in reversed(iter_stats)
            ]

            extra_cols = [
                {"name": "accept_reject", "label": "Accept/Reject", "field": "accept_reject"},
                {"name": "mean_score", "label": "Mean Score", "field": "mean_score"},
            ]
            with ui.scroll_area().style("max-height: 180px;"):
                ui.table(columns=columns + extra_cols, rows=rows, row_key="iteration").classes("w-full")

            with ui.scroll_area().style("max-height: 300px;"):
                for run in reversed(runs):
                    with ui.expansion(f"Iteration {run['iteration']} Analysis").classes("w-full"):
                        ui.markdown(run.get("analysis", "No analysis recorded.")).classes("text-body2")

    # --- Loop History ---
    _render_loop_history()

    # --- Two-Phase Improver Loop ---
    with ui.expansion(
        "Autonomous Improver Loop", icon="auto_fix_high",
    ).classes("w-full q-mx-md q-mt-md"):
        ui.label(
            "Two-phase loop: Phase A improves judge prompts, Phase B improves generator prompts. "
            "Each phase spawns an Opus agent with autonomous tool access."
        ).classes("text-caption text-grey-7")

        # Phase cards
        with ui.row().classes("w-full gap-4 q-mt-sm"):
            phase_a_card = ui.card().classes("flex-1 q-pa-sm")
            with phase_a_card:
                with ui.row().classes("items-center gap-2"):
                    ui.label("Phase A: Judge").classes("text-subtitle2 text-weight-bold")
                    phase_a_badge = ui.badge("PENDING", color="grey")
                phase_a_reasoning = ui.expansion("Reasoning", icon="psychology").classes("w-full")
                with phase_a_reasoning:
                    phase_a_text = ui.label("").classes("text-body2").style(
                        "white-space: pre-wrap; font-size: 0.85em;"
                    )

            phase_b_card = ui.card().classes("flex-1 q-pa-sm")
            with phase_b_card:
                with ui.row().classes("items-center gap-2"):
                    ui.label("Phase B: Generator").classes("text-subtitle2 text-weight-bold")
                    phase_b_badge = ui.badge("PENDING", color="grey")
                phase_b_reasoning = ui.expansion("Reasoning", icon="psychology").classes("w-full")
                with phase_b_reasoning:
                    phase_b_text = ui.label("").classes("text-body2").style(
                        "white-space: pre-wrap; font-size: 0.85em;"
                    )

        loop_error_label = ui.label("").classes("text-caption text-red")

        # Test results expansion
        with ui.expansion("Test Results", icon="science").classes("w-full q-mt-sm"):
            test_results_container = ui.column().classes("w-full gap-1")

        # Log viewer with tabs
        with ui.expansion("Improver Logs", icon="terminal").classes("w-full q-mt-sm"):
            with ui.tabs().classes("w-full") as log_tabs:
                tab_a = ui.tab("Phase A")
                tab_b = ui.tab("Phase B")
            with ui.tab_panels(log_tabs, value=tab_a).classes("w-full"):
                with ui.tab_panel(tab_a):
                    log_a_display = ui.code("", language="text").classes("w-full").style(
                        "max-height: 400px; overflow-y: auto; font-size: 0.8em;"
                    )
                with ui.tab_panel(tab_b):
                    log_b_display = ui.code("", language="text").classes("w-full").style(
                        "max-height: 400px; overflow-y: auto; font-size: 0.8em;"
                    )

        def _tail_log(log_path: Path, n_lines: int = 50) -> str:
            if not log_path.exists():
                return ""
            lines = log_path.read_text().splitlines()
            return "\n".join(lines[-n_lines:])

        def _update_from_status(st: dict) -> None:
            """Update UI elements from a loop status dict.

            While running: shows the agent's most recent status line.
            When done: shows the final analysis summary extracted from the log.
            """
            from pipeline.phase2.loop import (
                IMPROVER_LOG_A_PATH,
                IMPROVER_LOG_B_PATH,
                _extract_latest_status_from_log,
                _extract_reasoning_from_log,
            )

            for phase_key, badge, text_label, log_path in [
                ("phase_a", phase_a_badge, phase_a_text, IMPROVER_LOG_A_PATH),
                ("phase_b", phase_b_badge, phase_b_text, IMPROVER_LOG_B_PATH),
            ]:
                phase_data = st.get(phase_key, {})
                phase_status = phase_data.get("status", "pending")
                label, color = _phase_badge(phase_status)
                badge.set_text(label)
                badge._props["color"] = color
                badge.update()

                if phase_status == "running":
                    # Live: show most recent agent message
                    text_label.set_text(_extract_latest_status_from_log(log_path))
                elif phase_status in ("done", "error"):
                    # Completed: show final summary
                    reasoning = phase_data.get("reasoning", "")
                    if not reasoning:
                        reasoning = _extract_reasoning_from_log(log_path)
                    text_label.set_text(reasoning)

            if st.get("error"):
                loop_error_label.set_text(f"Error: {st['error']}")
            else:
                loop_error_label.set_text("")

            # Always show logs if they exist
            log_a_display.set_content(_tail_log(IMPROVER_LOG_A_PATH))
            log_b_display.set_content(_tail_log(IMPROVER_LOG_B_PATH))

        def _poll_loop_status():
            from pipeline.phase2.loop import read_status
            st = read_status()
            if st is None:
                return

            _update_from_status(st)

            # Update test results
            results = load_test_results()
            test_results_container.clear()
            if results:
                with test_results_container:
                    cols = [
                        {"name": "test_id", "label": "Test ID", "field": "test_id"},
                        {"name": "type", "label": "Type", "field": "type"},
                        {"name": "phase", "label": "Phase", "field": "phase"},
                        {"name": "prompt", "label": "Prompt", "field": "prompt"},
                        {"name": "n_items", "label": "Items", "field": "n_items"},
                        {"name": "mean_score", "label": "Mean Score", "field": "mean_score"},
                        {"name": "timestamp", "label": "Time", "field": "timestamp"},
                    ]
                    rows = []
                    for r in results[-20:]:  # show last 20
                        s = r.get("summary", {})
                        rows.append({
                            "test_id": r.get("test_id", ""),
                            "type": r.get("type", ""),
                            "phase": r.get("phase", ""),
                            "prompt": r.get("prompt", ""),
                            "n_items": s.get("n_items", ""),
                            "mean_score": f"{s['mean_score']:.2f}" if isinstance(s.get("mean_score"), (int, float)) else "",
                            "timestamp": r.get("timestamp", "")[:19],
                        })
                    ui.table(columns=cols, rows=rows, row_key="test_id").classes("w-full")

            if not st.get("running"):
                loop_timer.active = False
                loop_btn.enable()
                single_btn.enable()

        loop_timer = ui.timer(3.0, _poll_loop_status, active=False)

        def start_loop():
            loop_btn.disable()
            single_btn.disable()
            loop_error_label.set_text("")
            loop_timer.active = True

            def _thread():
                from pipeline.phase2.loop import run_improver_loop
                cfg = load_config()
                run_improver_loop(cfg=cfg)

            threading.Thread(target=_thread, daemon=True).start()

        loop_btn = ui.button(
            "Start Improver Loop",
            on_click=start_loop,
            color="primary",
        )

        # Load existing status on page render (show logs/reasoning even when not running)
        from pipeline.phase2.loop import read_status as _read_initial
        _initial = _read_initial()
        if _initial:
            _update_from_status(_initial)
            if _initial.get("running"):
                loop_btn.disable()
                loop_timer.active = True

    # --- Single Iteration ---
    with ui.expansion(
        "Run Single Iteration", icon="play_circle",
    ).classes("w-full q-mx-md q-mt-md"):
        ui.label("Runs a single generate->judge iteration with current config.").classes(
            "text-caption text-grey-7"
        )

        def start_iteration():
            single_btn.disable()
            single_status.set_text("Running iteration...")

            def _run():
                from pipeline.phase2.run import run_iteration

                cfg = load_config()
                return run_iteration(cfg)

            def _done():
                single_btn.enable()
                single_status.set_text("Done! Refresh page to see results.")
                ui.notify("Iteration complete", type="positive")

            def _thread():
                try:
                    _run()
                    _done()
                except Exception as e:
                    single_status.set_text(f"Error: {e}")
                    single_btn.enable()

            threading.Thread(target=_thread, daemon=True).start()

        single_btn = ui.button("Start Iteration", on_click=start_iteration, color="secondary")
        single_status = ui.label("").classes("text-caption text-grey-6")


@ui.page("/pipeline/review")
def pipeline_review_page():
    """Human review of LLM-generated reflections with per-dimension scoring."""
    viewer_id = app.storage.user.get("annotator_id", "")
    if not viewer_id:
        ui.navigate.to("/")
        return

    def review_actions():
        ui.button("Dashboard", icon="dashboard",
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

    # State
    state = {"iteration": runs[-1]["iteration"], "pos": 0}

    # --- Iteration selector ---
    with ui.row().classes("q-px-md q-mt-md items-center gap-4"):
        iter_options = [r["iteration"] for r in runs]
        iter_select = ui.select(
            options=iter_options,
            value=state["iteration"],
            label="Iteration",
        ).classes("w-32")

        sort_select = ui.select(
            options=["Low score first", "High score first", "Default order"],
            value="Low score first",
            label="Sort",
        ).classes("w-48")

        rejudge_status = ui.label("").classes("text-caption")

        def _rejudge():
            rejudge_btn.disable()
            rejudge_status.set_text("Re-judging...")

            def _thread():
                try:
                    from pipeline.phase2.run import rejudge_reviewed_items
                    result = rejudge_reviewed_items(cfg)
                    rejudge_status.set_text(f"Re-judged {len(result)} items. Refresh to see results.")
                    rejudge_btn.enable()
                except Exception as exc:
                    rejudge_status.set_text(f"Error: {exc}")
                    rejudge_btn.enable()

            threading.Thread(target=_thread, daemon=True).start()

        rejudge_btn = ui.button(
            "Re-judge Reviewed Items",
            icon="gavel",
            on_click=_rejudge,
            color="secondary",
        ).props("dense outline").tooltip(
            f"Re-judge all reviewed items with current judge prompt ({resolve_prompt_path(cfg.phase2.judge.prompt, cfg.phase2.judge.model).name})"
        )

    # --- Main split panel ---
    with ui.splitter(value=35).classes("w-full").style("height: calc(100vh - 120px)") as splitter:
        # Left: source text + charter
        with splitter.before:
            with ui.column().classes("w-full p-4 gap-2").style(
                "position: sticky; top: 0; height: calc(100vh - 120px); overflow-y: auto;"
            ):
                ui.label("Source Text + Charter").classes("text-h6 text-weight-bold")
                source_html = ui.html("").style(
                    "max-height: 40%; overflow-y: auto; border: 1px solid #333; "
                    "border-radius: 4px; padding: 12px; line-height: 1.7; "
                    "font-family: Georgia, serif; white-space: pre-wrap; font-size: 0.95em;"
                )
                ui.separator()
                ui.label("Charter").classes("text-subtitle2 text-weight-bold")
                ui.markdown(charter_text, extras=["tables"]).classes("text-body2").style(
                    "flex: 1; overflow-y: auto; padding: 8px; line-height: 1.6;"
                )

        # Right: LLM generation + judge scores + review form
        with splitter.after:
            with ui.column().classes("w-full p-4 gap-2").style("overflow-y: auto;"):
                with ui.row().classes("items-center gap-4"):
                    nav_label = ui.label().classes("text-subtitle1 text-weight-medium")
                    subset_badge = ui.badge("").props("outline")
                    gold_badge = ui.badge("").props("outline color=orange")
                    ui.space()
                    ui.button(icon="arrow_back", on_click=lambda: navigate(-1)).props("flat dense")
                    ui.button(icon="arrow_forward", on_click=lambda: navigate(1)).props("flat dense")

                # LLM generation display
                gen_section = ui.column().classes("w-full gap-2")

                ui.separator()

                # Judge scores (hidden by default)
                judge_expansion = ui.expansion("Judge Scores", icon="gavel").classes("w-full")
                with judge_expansion:
                    judge_section = ui.column().classes("w-full gap-1")

                # Human annotation (for gold items)
                gold_expansion = ui.expansion("Human Annotation (Gold)", icon="person").classes("w-full")
                with gold_expansion:
                    gold_section = ui.column().classes("w-full gap-1")

                ui.separator()

                # Review form
                ui.label("Your Review").classes("text-subtitle2 text-weight-bold")

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
                                slider = ui.slider(min=1, max=5, value=3).classes("flex-1")
                                score_label = ui.label("3").classes("w-8")
                                slider.on("update:model-value", lambda e, lbl=score_label: lbl.set_text(str(int(e.args))))
                                score_inputs[part][dim] = slider
                            hint = DIM_HINTS.get(dim, "")
                            if hint:
                                ui.label(hint).classes("text-caption text-grey-6").style(
                                    "margin-left: 160px; margin-top: -4px;"
                                )

                threshold = cfg.phase2.scoring.accept_threshold
                review_status_label = ui.label("").classes("text-caption text-weight-bold")

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
                        slider.on("update:model-value", lambda _: _update_review_status())
                _update_review_status()

                notes_input = ui.textarea(
                    placeholder="Notes (optional)...",
                ).classes("w-full").props("outlined")

                with ui.row().classes("w-full justify-end"):
                    ui.button("Submit Review", on_click=lambda: submit_review(), color="primary")

    def get_sorted_items() -> list[dict]:
        items = load_items_for_iteration(state["iteration"])
        judged = [i for i in items if i.get("judgment")]
        sort = sort_select.value
        if sort == "Low score first":
            judged.sort(key=lambda i: i["judgment"]["aggregate"])
        elif sort == "High score first":
            judged.sort(key=lambda i: -i["judgment"]["aggregate"])
        return judged

    def current_items_list() -> list[dict]:
        return get_sorted_items()

    def _first_unreviewed_pos(items: list[dict]) -> int:
        """Return index of the first item without a review from this viewer."""
        reviewed = load_latest_reviews()
        reviewed_ids = {k[0] for k in reviewed if k[1] == state["iteration"] and k[2] == viewer_id}
        for i, item in enumerate(items):
            if item["item_id"] not in reviewed_ids:
                return i
        return 0

    def update_display():
        items = current_items_list()
        if not items:
            nav_label.set_text("No judged items in this iteration")
            return

        state["pos"] = max(0, min(state["pos"], len(items) - 1))
        item = items[state["pos"]]

        nav_label.set_text(f"Item {state['pos'] + 1} / {len(items)}")
        subset_badge.set_text(item["subset"])
        gold_badge.set_text("GOLD" if item.get("is_gold") else "")
        gold_badge.set_visibility(item.get("is_gold", False))

        source_html.set_content(render_source_text(item["text"], item["reflection_point"]))

        gen_section.clear()
        with gen_section:
            ui.label("LLM Generation").classes("text-subtitle2 text-weight-bold")
            ui.label("Analysis").classes("text-overline text-grey-7")
            ui.label(item.get("analysis", "")).classes("text-body2").style("white-space: pre-wrap;")
            ui.label("Preflection").classes("text-overline text-grey-7")
            ui.label(item.get("preflection", "")).classes("text-body2").style("white-space: pre-wrap;")
            ui.label("Reflection").classes("text-overline text-grey-7")
            ui.label(item.get("reflection", "")).classes("text-body2").style("white-space: pre-wrap;")
            elements = item.get("charter_elements", [])
            if elements:
                ui.label("Charter Elements").classes("text-overline text-grey-7")
                with ui.row().classes("gap-1"):
                    for eid in elements:
                        ui.badge(eid, color="blue-grey-3").props("outline")

        judge_section.clear()
        with judge_section:
            judgment = item.get("judgment", {})
            if judgment:
                ui.label("Judge Scores").classes("text-subtitle2 text-weight-bold")
                with ui.row().classes("gap-4"):
                    ui.badge(
                        f"Aggregate: {judgment['aggregate']:.1f}",
                        color="green" if judgment["decision"] == "accept" else "red",
                    )
                    ui.badge(judgment["decision"].upper(), color="green" if judgment["decision"] == "accept" else "red")
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

        gold_section.clear()
        gold_expansion.set_visibility(bool(item.get("is_gold")))
        with gold_section:
            if item.get("is_gold"):
                _show_gold_annotation(item["item_id"])

        # Pre-fill from existing review
        latest_reviews = load_latest_reviews()
        existing = latest_reviews.get((item["item_id"], state["iteration"], viewer_id))
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
        else:
            for dims in score_inputs.values():
                for slider in dims.values():
                    slider.set_value(3)
            notes_input.set_value("")
        _update_review_status()

    def _show_gold_annotation(item_id: str):
        """Display the human annotation for a gold item."""
        from pipeline.phase1.storage import load_latest_annotations

        annotations = load_latest_annotations()
        gold_records = [v for (iid, _), v in annotations.items() if iid == item_id]
        if not gold_records:
            ui.label("No human annotations found for this gold item.").classes("text-grey-6")
            return

        for rec in gold_records:
            with ui.card().classes("w-full q-pa-sm"):
                ui.label(f"Annotator: {rec['annotator_id']}").classes("text-caption text-grey-6")
                ui.label("Analysis").classes("text-overline text-grey-7")
                ui.label(rec["analysis"]).classes("text-body2").style("white-space: pre-wrap;")
                ui.label("Preflection").classes("text-overline text-grey-7")
                ui.label(rec["preflection"]).classes("text-body2").style("white-space: pre-wrap;")
                ui.label("Reflection").classes("text-overline text-grey-7")
                ui.label(rec["reflection"]).classes("text-body2").style("white-space: pre-wrap;")

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
        decision = "reject" if has_floor or aggregate < cfg.phase2.scoring.accept_threshold else "accept"
        save_review(
            item_id=item["item_id"],
            iteration=state["iteration"],
            reviewer_id=viewer_id,
            scores=scores,
            aggregate=aggregate,
            decision=decision,
            notes=notes_input.value.strip(),
        )
        ui.notify("Review saved!", type="positive")
        navigate(1)

    def on_iteration_change(e):
        state["iteration"] = e.args
        items = current_items_list()
        state["pos"] = _first_unreviewed_pos(items) if items else 0
        update_display()

    def on_sort_change(_):
        state["pos"] = 0
        update_display()

    iter_select.on("update:model-value", on_iteration_change)
    sort_select.on("update:model-value", on_sort_change)
    # Start at first unreviewed item
    items = current_items_list()
    state["pos"] = _first_unreviewed_pos(items) if items else 0
    update_display()


@ui.page("/pipeline/reviews")
def pipeline_reviews_page():
    """Review overview: browse all reviews, comment on them, delete them."""
    viewer_id = app.storage.user.get("annotator_id", "")

    def reviews_actions():
        ui.button("Dashboard", icon="dashboard",
                  on_click=lambda: ui.navigate.to("/pipeline"),
                  ).classes("text-white").props("flat dense")
        ui.button("Review", icon="rate_review",
                  on_click=lambda: ui.navigate.to("/pipeline/review"),
                  ).classes("text-white").props("flat dense")

    render_header(viewer_id, active_phase=2, right_slot=reviews_actions)

    all_reviews = load_reviews()
    review_comments = load_review_comments()

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
            with ui.expansion(
                f"Iteration {iteration} ({len(reviews)} reviews)",
                icon="rate_review",
            ).classes("w-full q-mx-md q-mt-sm").props("default-opened" if iteration == max(by_iter) else ""):
                for r in sorted(reviews, key=lambda x: x["timestamp"], reverse=True):
                    item = items_by_key.get((r["item_id"], r["iteration"]))
                    review_key = (r["item_id"], r["iteration"], r["reviewer_id"])

                    with ui.card().classes("w-full q-pa-sm q-mb-sm"):
                        # Header: reviewer, decision, score, timestamp
                        with ui.row().classes("items-center gap-2 w-full"):
                            ui.badge(r["reviewer_id"], color="blue-grey").props("outline")
                            decision_color = "green" if r["decision"] == "accept" else "red"
                            ui.badge(r["decision"].upper(), color=decision_color)
                            ui.badge(f"Score: {r['aggregate']:.2f}", color=decision_color).props("outline")
                            ui.label(r["timestamp"][:19]).classes("text-caption text-grey-6")
                            ui.label(f"{r['item_id'][:12]}").classes("text-caption text-grey-5")
                            ui.space()

                            # Delete button
                            def make_delete(iid=r["item_id"], it=r["iteration"], rid=r["reviewer_id"]):
                                def do_delete():
                                    delete_review(iid, it, rid)
                                    ui.notify("Review deleted", type="info")
                                    render_reviews.refresh()
                                return do_delete

                            ui.button(
                                icon="delete", on_click=make_delete(),
                            ).props("flat dense size=xs color=negative")

                        # Per-part scores
                        scores = r["scores"]
                        is_per_part = scores and isinstance(next(iter(scores.values())), dict)
                        if is_per_part:
                            with ui.row().classes("gap-4 q-mt-xs"):
                                for part in ("preflection", "reflection"):
                                    part_scores = scores.get(part, {})
                                    if part_scores:
                                        score_str = " ".join(
                                            f"{dim[:3]}={val}" for dim, val in part_scores.items()
                                        )
                                        ui.label(f"{part}: {score_str}").classes("text-caption")
                        else:
                            score_str = " ".join(f"{d[:3]}={v}" for d, v in scores.items())
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
                                "Source Text", icon="article",
                            ).classes("w-full q-mt-xs"):
                                ui.html(render_source_text(
                                    item["text"], item["reflection_point"],
                                )).style(
                                    "line-height: 1.6; font-family: Georgia, serif; "
                                    "white-space: pre-wrap; font-size: 0.95em; padding: 8px;"
                                )

                            # Generation dropdown
                            with ui.expansion(
                                "LLM Generation", icon="smart_toy",
                            ).classes("w-full q-mt-xs"):
                                ui.label("Analysis").classes("text-overline text-grey-7")
                                ui.label(item.get("analysis", "")).classes("text-body2").style(
                                    "white-space: pre-wrap;"
                                )
                                ui.label("Preflection").classes("text-overline text-grey-7 q-mt-sm")
                                ui.label(item.get("preflection", "")).classes("text-body2").style(
                                    "white-space: pre-wrap;"
                                )
                                ui.label("Reflection").classes("text-overline text-grey-7 q-mt-sm")
                                ui.label(item.get("reflection", "")).classes("text-body2").style(
                                    "white-space: pre-wrap;"
                                )
                                elements = item.get("charter_elements", [])
                                if elements:
                                    ui.label("Charter Elements").classes("text-overline text-grey-7 q-mt-sm")
                                    with ui.row().classes("gap-1"):
                                        for eid in elements:
                                            ui.badge(eid, color="blue-grey-3").props("outline")

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
                                        icon="delete", on_click=make_delete_comment(),
                                    ).props("flat dense size=xs color=negative")
                                ui.label(c["comment"]).classes("text-body2 q-mb-sm").style(
                                    "white-space: pre-wrap; padding-left: 8px;"
                                )

                            if viewer_id:
                                comment_input = ui.input(
                                    placeholder="Add a comment...",
                                ).classes("w-full").props("dense outlined")

                                def make_submit(
                                    iid=r["item_id"], it=r["iteration"],
                                    rid=r["reviewer_id"], inp=comment_input,
                                ):
                                    def do_submit():
                                        assert inp.value and inp.value.strip(), "Comment cannot be empty"
                                        save_review_comment(iid, it, rid, viewer_id, inp.value.strip())
                                        ui.notify("Comment added", type="positive")
                                        render_reviews.refresh()
                                    return do_submit

                                ui.button(
                                    "Post", on_click=make_submit(), color="primary",
                                ).props("flat dense size=sm")

    render_reviews()
