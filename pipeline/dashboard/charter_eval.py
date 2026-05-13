"""Charter eval dashboard pages: /charter_eval, /charter_eval/<run_id>, /charter_eval/<run_id>/browse.

Thin presentation layer over `pipeline.charter.eval.rank` analytics. The runs
themselves are produced by `python -m pipeline.charter.eval eval-{generators,judges}`
and live under `cfg.charter.eval.eval_dir`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from nicegui import app, ui

from pipeline.config import load_config
from pipeline.dashboard import render_header
from pipeline.dashboard.shared import render_source_text
from pipeline.log import logger
from pipeline.charter.eval.eval_generators import _eval_root


def _list_runs() -> list[dict]:
    root = _eval_root(load_config())
    if not root.exists():
        return []
    runs: list[dict] = []
    for run_dir in sorted(root.iterdir()):
        meta_path = run_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("charter_eval dashboard: skipping {}: {}", run_dir.name, e)
            continue
        runs.append({"run_id": run_dir.name, "path": str(run_dir), **meta})
    return runs


def _generator_display_label(gen_name: str) -> str:
    """'qwen3.5-35b-a3b__generator_reflection_v7.md' → 'qwen3.5-35b-a3b v7'."""
    m = re.match(r"^(.+?)__generator_(?:reflection|preflection)_v(\d+)", gen_name)
    if m:
        return f"{m.group(1)} v{m.group(2)}"
    return gen_name


def _collect_cross_run_generators() -> list[dict]:
    """Load rank_generators for all completed generator_eval runs, deduplicate."""
    from pipeline.charter.eval import rank as rank_mod

    runs = _list_runs()
    best: dict[str, dict] = {}
    for r in runs:
        if r.get("type") != "generator_eval":
            continue
        run_id = r["run_id"]
        try:
            rows = rank_mod.rank_generators(run_id)
        except Exception:
            continue
        for entry in rows:
            key = entry["generator"]
            prev = best.get(key)
            if prev is None or entry["n_succeeded"] > prev["n_succeeded"]:
                best[key] = entry
    out = list(best.values())
    out.sort(key=lambda g: g.get("accept_rate", 0), reverse=True)
    return out


_BAR_COLORS = [
    "#4caf50",
    "#2196f3",
    "#ff9800",
    "#e91e63",
    "#9c27b0",
    "#00bcd4",
    "#795548",
    "#607d8b",
    "#cddc39",
    "#ff5722",
]


@ui.page("/charter_eval")
def charter_eval_runs_page() -> None:
    """List charter.eval runs."""
    viewer_id = app.storage.user.get("annotator_id", "")
    render_header(viewer_id, active_step=3)

    runs = _list_runs()

    ui.label("Charter eval: Runs").classes("text-h5 q-pa-md")
    if not runs:
        with ui.column().classes("absolute-center items-center"):
            ui.label("No charter.eval runs yet.").classes("text-h6 text-grey-6")
            ui.label(
                "Run `uv run python -m pipeline.charter.eval eval-generators` to start."
            ).classes("text-body2 text-grey-5")
        return

    rows = []
    for r in runs:
        n_candidates = len(r.get("candidates", []))
        heartbeat = r.get("heartbeat") or {}
        rows.append(
            {
                "run_id": r["run_id"],
                "type": r.get("type", "?"),
                "started_at": (r.get("started_at") or "")[:19],
                "finished_at": (r.get("finished_at") or "")[:19] or "—",
                "n_items": r.get("n_items", "?"),
                "n_candidates": n_candidates,
                "status": r.get("status", "?"),
                "last_heartbeat": (heartbeat.get("last_write_ts") or "")[:19] or "—",
            }
        )

    columns = [
        {"name": "run_id", "label": "Run", "field": "run_id", "align": "left"},
        {"name": "type", "label": "Type", "field": "type", "align": "left"},
        {
            "name": "started_at",
            "label": "Started",
            "field": "started_at",
            "align": "left",
        },
        {
            "name": "finished_at",
            "label": "Finished",
            "field": "finished_at",
            "align": "left",
        },
        {"name": "n_items", "label": "Items", "field": "n_items", "align": "right"},
        {
            "name": "n_candidates",
            "label": "Candidates",
            "field": "n_candidates",
            "align": "right",
        },
        {"name": "status", "label": "Status", "field": "status", "align": "left"},
        {
            "name": "last_heartbeat",
            "label": "Heartbeat",
            "field": "last_heartbeat",
            "align": "left",
        },
    ]

    table = ui.table(columns=columns, rows=rows, row_key="run_id").classes("w-full")
    table.on(
        "rowClick",
        lambda e: ui.navigate.to(f"/charter_eval/{e.args[1]['run_id']}"),
    )

    # --- Cross-run generator comparison ---
    try:
        generators = _collect_cross_run_generators()
    except Exception as e:
        ui.label(f"Failed to load cross-run data: {e}").classes("text-red q-pa-md")
        return
    if generators:
        _render_cross_run_charts(generators)


@ui.page("/charter_eval/{run_id}")
def charter_eval_run_detail_page(run_id: str) -> None:
    """Show metadata + rank table for a single charter.eval run."""
    viewer_id = app.storage.user.get("annotator_id", "")
    render_header(viewer_id, active_step=3)

    root = _eval_root(load_config()) / run_id
    meta_path = root / "metadata.json"
    if not meta_path.is_file():
        with ui.column().classes("absolute-center items-center"):
            ui.label(f"Unknown run: {run_id}").classes("text-h6 text-red")
        return
    meta = json.loads(meta_path.read_text())

    ui.label(f"Charter eval: {run_id}").classes("text-h5 q-pa-md")
    with ui.card().classes("q-ma-md"):
        for k in (
            "type",
            "status",
            "started_at",
            "finished_at",
            "n_items",
            "seed",
            "max_tokens",
            "dataset_revision",
            "store_reasoning",
        ):
            if k in meta:
                ui.label(f"{k}: {meta[k]}")
        if meta.get("candidates"):
            ui.label("candidates:").classes("text-bold")
            for c in meta["candidates"]:
                ui.label(f"  • {c.get('alias')} / {c.get('prompt')}")

    eval_type = meta.get("type")

    if eval_type == "generator_eval":
        ui.button(
            "Browse Outputs",
            icon="search",
            on_click=lambda: ui.navigate.to(f"/charter_eval/{run_id}/browse"),
        ).classes("q-mx-md")
    try:
        from pipeline.charter.eval import rank as rank_mod

        if eval_type == "generator_eval":
            rows = rank_mod.rank_generators(run_id)
            _render_generator_table(rows)
        elif eval_type == "judge_eval":
            blocks = rank_mod.rank_judges(run_id)
            _render_judge_tables(blocks)
        else:
            ui.label(f"Unknown eval type: {eval_type}").classes("text-red")
    except Exception as e:
        ui.label(f"Failed to compute rank: {e}").classes("text-red")
        logger.exception("charter_eval dashboard rank failed")


def _render_generator_table(rows: list[dict]) -> None:
    if not rows:
        ui.label("No generators in this run.").classes("text-grey")
        return
    columns = [
        {
            "name": "generator",
            "label": "Generator",
            "field": "generator",
            "align": "left",
        },
        {"name": "n_pool", "label": "n_pool", "field": "n_pool", "align": "right"},
        {
            "name": "n_succeeded",
            "label": "n_ok",
            "field": "n_succeeded",
            "align": "right",
        },
        {
            "name": "mean_aggregate",
            "label": "mean",
            "field": "mean_aggregate",
            "align": "right",
        },
        {
            "name": "accept_rate",
            "label": "accept%",
            "field": "accept_rate",
            "align": "right",
        },
        {
            "name": "gen_api",
            "label": "gen_api",
            "field": "gen_api",
            "align": "right",
        },
        {
            "name": "gen_parse",
            "label": "gen_parse",
            "field": "gen_parse",
            "align": "right",
        },
        {
            "name": "judge_api",
            "label": "judge_api",
            "field": "judge_api",
            "align": "right",
        },
        {
            "name": "judge_parse",
            "label": "judge_parse",
            "field": "judge_parse",
            "align": "right",
        },
    ]
    table_rows = []
    for r in rows:
        fr = r.get("failure_rates", {})
        table_rows.append(
            {
                "generator": r["generator"],
                "n_pool": r["n_pool"],
                "n_succeeded": r["n_succeeded"],
                "mean_aggregate": f"{r['mean_aggregate']:.3f}",
                "accept_rate": f"{r['accept_rate']:.1%}",
                "gen_api": f"{fr.get('gen_api', 0.0):.1%}",
                "gen_parse": f"{fr.get('gen_parse', 0.0):.1%}",
                "judge_api": f"{fr.get('judge_api', 0.0):.1%}",
                "judge_parse": f"{fr.get('judge_parse', 0.0):.1%}",
            }
        )
    ui.table(columns=columns, rows=table_rows, row_key="generator").classes("q-ma-md")

    # Per-generator accept-rate-by-safety-score matrix
    safety_scores: list[str] = []
    for r in rows:
        for k in (r.get("accept_by_safety_score") or {}).keys():
            if k not in safety_scores:
                safety_scores.append(k)
    safety_scores.sort()
    if safety_scores:
        ui.label("Accept rate by safety score").classes("text-h6 q-px-md q-mt-md")
        cols = [
            {
                "name": "generator",
                "label": "Generator",
                "field": "generator",
                "align": "left",
            }
        ] + [
            {"name": s, "label": s, "field": s, "align": "right"} for s in safety_scores
        ]
        matrix_rows = []
        for r in rows:
            row = {"generator": r["generator"]}
            buckets = r.get("accept_by_safety_score") or {}
            for s in safety_scores:
                b = buckets.get(s)
                row[s] = f"{b['accept_rate']:.0%} ({b['n']})" if b else "—"
            matrix_rows.append(row)
        ui.table(columns=cols, rows=matrix_rows, row_key="generator").classes("q-ma-md")


def _render_judge_tables(blocks: dict) -> None:
    for label, key in (("vs gold", "vs_gold"), ("vs human", "vs_human")):
        rows = blocks.get(key) or []
        ui.label(f"Judges {label}").classes("text-h6 q-px-md q-mt-md")
        if not rows:
            ui.label("(empty)").classes("text-grey q-px-md")
            continue
        cols = [
            {"name": "judge", "label": "Judge", "field": "judge", "align": "left"},
            {
                "name": "n_succeeded",
                "label": "n_ok",
                "field": "n_succeeded",
                "align": "right",
            },
            {
                "name": "spearman",
                "label": "spearman",
                "field": "spearman",
                "align": "right",
            },
            {
                "name": "pearson",
                "label": "pearson",
                "field": "pearson",
                "align": "right",
            },
            {
                "name": "concordance",
                "label": "conc",
                "field": "concordance",
                "align": "right",
            },
            {"name": "kappa", "label": "kappa", "field": "kappa", "align": "right"},
            {
                "name": "api",
                "label": "api_fail",
                "field": "api",
                "align": "right",
            },
            {
                "name": "parse",
                "label": "parse_fail",
                "field": "parse",
                "align": "right",
            },
        ]
        table_rows = []
        for r in rows:
            fr = r.get("failure_rates", {})
            table_rows.append(
                {
                    "judge": r["judge"],
                    "n_succeeded": r.get("n_succeeded", 0),
                    "spearman": _fmt(r.get("spearman")),
                    "pearson": _fmt(r.get("pearson")),
                    "concordance": _fmt(r.get("concordance"), pct=True),
                    "kappa": _fmt(r.get("kappa")),
                    "api": f"{fr.get('api', 0.0):.1%}",
                    "parse": f"{fr.get('parse', 0.0):.1%}",
                }
            )
        ui.table(columns=cols, rows=table_rows, row_key="judge").classes("q-ma-md")


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "—"
    try:
        if pct:
            return f"{v:.1%}"
        return f"{v:.3f}"
    except (TypeError, ValueError):
        return str(v)


# ------------------------------------------------------------------ cross-run charts


def _render_cross_run_charts(generators: list[dict]) -> None:
    """Render cross-run barplots: accept rate, mean aggregate, and per-safety-score."""
    labels = [_generator_display_label(g["generator"]) for g in generators]

    for mode, mode_label in (
        ("reflection", "Reflection"),
        ("preflection", "Preflection"),
    ):
        accept_key = f"{mode}_accept_rate"
        mean_key = f"{mode}_mean_aggregate"
        safety_key = f"{mode}_accept_by_safety_score"

        # Skip mode entirely if no generator has data for it.
        if not any(accept_key in g for g in generators):
            continue

        ui.label(f"Cross-Run Comparison — {mode_label}").classes(
            "text-h5 q-pa-md q-mt-lg"
        )

        # --- Accept rate + mean aggregate side by side ---
        with ui.row().classes("w-full q-px-md gap-4"):
            with ui.card().classes("col q-pa-sm"):
                ui.echart(
                    {
                        "tooltip": {"trigger": "axis"},
                        "xAxis": {
                            "type": "category",
                            "data": labels,
                            "axisLabel": {"interval": 0, "fontSize": 11, "rotate": 15},
                        },
                        "yAxis": {
                            "type": "value",
                            "name": "Accept %",
                            "min": 0,
                            "max": 100,
                        },
                        "series": [
                            {
                                "name": "Accept %",
                                "type": "bar",
                                "data": [
                                    round(g.get(accept_key, 0) * 100, 1)
                                    for g in generators
                                ],
                                "itemStyle": {"color": "#4caf50"},
                                "barMaxWidth": 60,
                                "label": {
                                    "show": True,
                                    "position": "top",
                                    "formatter": "{c}%",
                                    "fontSize": 10,
                                },
                            }
                        ],
                        "grid": {"bottom": 80, "top": 30, "left": 50, "right": 20},
                    }
                ).classes("w-full").style("height: 350px;")

            with ui.card().classes("col q-pa-sm"):
                ui.echart(
                    {
                        "tooltip": {"trigger": "axis"},
                        "xAxis": {
                            "type": "category",
                            "data": labels,
                            "axisLabel": {"interval": 0, "fontSize": 11, "rotate": 15},
                        },
                        "yAxis": {
                            "type": "value",
                            "name": "Mean Aggregate",
                            "min": 0,
                            "max": 5,
                        },
                        "series": [
                            {
                                "name": "Mean Aggregate",
                                "type": "bar",
                                "data": [
                                    round(g.get(mean_key, 0), 2) for g in generators
                                ],
                                "itemStyle": {"color": "#2196f3"},
                                "barMaxWidth": 60,
                                "label": {
                                    "show": True,
                                    "position": "top",
                                    "fontSize": 10,
                                },
                            }
                        ],
                        "grid": {"bottom": 80, "top": 30, "left": 50, "right": 20},
                    }
                ).classes("w-full").style("height: 350px;")

        # --- Grouped barplot by safety score ---
        all_ss = sorted({k for g in generators for k in g.get(safety_key, {}).keys()})
        if not all_ss:
            continue

        series = []
        for i, g in enumerate(generators):
            label = _generator_display_label(g["generator"])
            data = []
            for ss in all_ss:
                bucket = g.get(safety_key, {}).get(ss)
                data.append(round(bucket["accept_rate"] * 100, 1) if bucket else None)
            series.append(
                {
                    "name": label,
                    "type": "bar",
                    "data": data,
                    "barMaxWidth": 40,
                    "itemStyle": {"color": _BAR_COLORS[i % len(_BAR_COLORS)]},
                    "label": {
                        "show": True,
                        "position": "top",
                        "formatter": "{c}%",
                        "fontSize": 9,
                    },
                }
            )

        with ui.card().classes("w-full q-mx-md q-pa-sm"):
            ui.echart(
                {
                    "tooltip": {"trigger": "axis"},
                    "legend": {"top": 0, "data": [s["name"] for s in series]},
                    "xAxis": {
                        "type": "category",
                        "data": [f"Safety {s}" for s in all_ss],
                    },
                    "yAxis": {
                        "type": "value",
                        "name": "Accept %",
                        "min": 0,
                        "max": 100,
                    },
                    "series": series,
                    "grid": {"bottom": 40, "top": 50, "left": 50, "right": 20},
                }
            ).classes("w-full").style("height: 400px;")


# ------------------------------------------------------------------ output browser


def _list_judgment_files(run_dir: Path) -> list[str]:
    """Return judgment file stems available for browsing."""
    jud_dir = run_dir / "judgments"
    if not jud_dir.exists():
        return []
    return sorted(p.stem for p in jud_dir.glob("*.jsonl"))


def _load_judgment_rows(run_dir: Path, jud_stem: str) -> list[dict]:
    from pipeline.charter.eval.rank import _read_jsonl

    return _read_jsonl(run_dir / "judgments" / f"{jud_stem}.jsonl")


def _judgment_display_label(jud_stem: str) -> str:
    """Extract generator portion from judgment stem for dropdown display.

    'kimi-k2.5__judge_reflection_v24.md__on__qwen3.5-35b-a3b__generator_reflection_v7.md'
    → 'qwen3.5-35b-a3b v7'
    """
    m = re.search(r"__on__(.+)$", jud_stem)
    if m:
        return _generator_display_label(m.group(1))
    return jud_stem


@ui.page("/charter_eval/{run_id}/browse")
def charter_eval_browse_page(run_id: str) -> None:
    """Browse individual outputs for a charter.eval generator eval run."""
    viewer_id = app.storage.user.get("annotator_id", "")
    render_header(viewer_id, active_step=3)

    cfg = load_config()
    run_dir = _eval_root(cfg) / run_id
    meta_path = run_dir / "metadata.json"
    if not meta_path.is_file():
        with ui.column().classes("absolute-center items-center"):
            ui.label(f"Unknown run: {run_id}").classes("text-h6 text-red")
        return

    jud_stems = _list_judgment_files(run_dir)
    if not jud_stems:
        with ui.column().classes("absolute-center items-center"):
            ui.label("No judgment files in this run.").classes("text-h6 text-grey-6")
        return

    # --- State ---
    state: dict = {
        "jud_stem": jud_stems[0],
        "all_rows": _load_judgment_rows(run_dir, jud_stems[0]),
        "filtered": [],
        "pos": 0,
        "safety_filter": {0, 1, 2, 3, 4, 5},
        "agg_min": 0.0,
        "agg_max": 5.0,
    }

    def _apply_filters():
        rows = state["all_rows"]
        filtered = []
        for r in rows:
            ss = r.get("safety_score")
            if ss is not None and ss not in state["safety_filter"]:
                continue
            j = r.get("judgment") or {}
            agg = j.get("aggregate")
            if agg is not None:
                if agg < state["agg_min"] or agg > state["agg_max"]:
                    continue
            filtered.append(r)
        state["filtered"] = filtered
        state["pos"] = max(0, min(state["pos"], len(filtered) - 1))

    _apply_filters()

    # --- Header ---
    ui.label(f"Browse: {run_id}").classes("text-h5 q-pa-md")

    # --- Filter bar ---
    with ui.row().classes("w-full items-center q-px-md gap-3 q-mb-sm"):
        gen_select = ui.select(
            options={s: _judgment_display_label(s) for s in jud_stems},
            value=state["jud_stem"],
            label="Generator",
        ).classes("w-64")

        safety_opts = [0, 1, 2, 3, 4, 5]
        safety_select = ui.select(
            options=safety_opts,
            value=list(state["safety_filter"]),
            label="Safety Score",
            multiple=True,
        ).classes("w-48")

        agg_min_input = ui.number(
            "Agg min", value=0.0, min=0, max=5, step=0.25, format="%.2f"
        ).classes("w-24")
        agg_max_input = ui.number(
            "Agg max", value=5.0, min=0, max=5, step=0.25, format="%.2f"
        ).classes("w-24")

        ui.space()

        nav_label = ui.label().classes("text-body1 text-weight-medium")
        prev_btn = ui.button("Prev", icon="arrow_back").props("flat dense")
        next_btn = ui.button("Next", icon="arrow_forward").props("flat dense")

    # --- Content area ---
    with (
        ui.splitter(value=35)
        .classes("w-full q-px-md")
        .style("height: calc(100vh - 180px);") as splitter
    ):
        with splitter.before:
            source_html = (
                ui.html("")
                .classes("w-full")
                .style(
                    "white-space: pre-wrap; font-family: monospace; font-size: 0.85em; "
                    "padding: 8px; overflow-y: auto; max-height: calc(100vh - 200px);"
                )
            )
        with splitter.after:
            detail_container = (
                ui.column()
                .classes("w-full q-pa-sm gap-2")
                .style("overflow-y: auto; max-height: calc(100vh - 200px);")
            )

    def _render_current():
        items = state["filtered"]
        n = len(items)
        pos = state["pos"]
        nav_label.set_text(f"Item {pos + 1} / {n}" if n else "No items")
        prev_btn.set_enabled(pos > 0)
        next_btn.set_enabled(pos < n - 1)

        if not items:
            source_html.set_content("<em>No items match filters.</em>")
            detail_container.clear()
            return

        row = items[pos]
        text = row.get("text", "")
        rp = row.get("reflection_point", 0)
        source_html.set_content(render_source_text(text, rp))

        detail_container.clear()
        with detail_container:
            j = row.get("judgment") or {}
            ss = row.get("safety_score", "?")
            decision = j.get("decision", "?")
            agg = j.get("aggregate")

            # Badges row
            with ui.row().classes("gap-2 items-center"):
                ui.badge(f"Safety: {ss}", color="blue").props("outline")
                dec_color = "green" if decision == "accept" else "red"
                ui.badge(decision, color=dec_color)
                if agg is not None:
                    ui.badge(f"Agg: {agg:.2f}", color="purple").props("outline")

            # Per-mode aggregates
            for mode, mode_label in (
                ("reflection", "Refl"),
                ("preflection", "Prefl"),
            ):
                mode_agg = j.get(f"{mode}_aggregate")
                mode_dec = j.get(f"{mode}_decision")
                if mode_agg is not None:
                    dec_color = (
                        "green"
                        if mode_dec == "accept"
                        else "red" if mode_dec else "grey"
                    )
                    ui.label(
                        f"{mode_label}: {mode_agg:.2f} ({mode_dec or '?'})"
                    ).classes(f"text-{dec_color}")

            # Analysis
            analysis = row.get("analysis") or ""
            if analysis:
                ui.label("Analysis").classes("text-weight-bold q-mt-sm")
                ui.label(analysis).classes("text-body2").style(
                    "white-space: pre-wrap; max-height: 150px; overflow-y: auto;"
                )

            # Generated reflections
            for field, fallback, label in (
                ("reflection_1p", "reflection", "Reflection 1p"),
                ("reflection_3p", None, "Reflection 3p"),
                ("preflection_1p", None, "Preflection 1p"),
                ("preflection_3p", "preflection", "Preflection 3p"),
            ):
                val = row.get(field) or (row.get(fallback) if fallback else None)
                if val:
                    ui.label(label).classes("text-weight-bold q-mt-sm")
                    ui.label(val).classes("text-body2").style(
                        "white-space: pre-wrap; max-height: 200px; overflow-y: auto;"
                    )

            # Per-voice judgment scores
            voices = []
            for v in (
                "reflection_1p",
                "reflection_3p",
                "preflection_1p",
                "preflection_3p",
            ):
                vdata = j.get(v)
                if isinstance(vdata, dict) and vdata.get("scores"):
                    voices.append((v, vdata))
            if voices:
                ui.label("Judgment Scores").classes("text-weight-bold q-mt-sm")
                score_rows = []
                dims = []
                for v_name, vdata in voices:
                    scores = vdata["scores"]
                    if not dims:
                        dims = list(scores.keys())
                    row_d = {"voice": v_name}
                    for d in dims:
                        row_d[d] = scores.get(d, "—")
                    score_rows.append(row_d)

                cols = [
                    {
                        "name": "voice",
                        "label": "Voice",
                        "field": "voice",
                        "align": "left",
                    }
                ] + [
                    {"name": d, "label": d, "field": d, "align": "right"} for d in dims
                ]
                ui.table(columns=cols, rows=score_rows, row_key="voice").classes(
                    "w-full"
                ).props("dense flat")

                for v_name, vdata in voices:
                    reasoning = vdata.get("reasoning")
                    if reasoning:
                        ui.label(f"{v_name} reasoning").classes(
                            "text-caption text-weight-bold q-mt-xs"
                        )
                        ui.label(reasoning).classes("text-body2").style(
                            "white-space: pre-wrap; max-height: 150px; "
                            "overflow-y: auto; color: #666;"
                        )

    _render_current()

    # --- Event handlers ---
    def on_gen_change(e):
        state["jud_stem"] = e.value
        state["all_rows"] = _load_judgment_rows(run_dir, e.value)
        state["pos"] = 0
        _apply_filters()
        _render_current()

    def on_safety_change(e):
        state["safety_filter"] = set(e.value) if e.value else set()
        _apply_filters()
        _render_current()

    def on_agg_change(_e=None):
        state["agg_min"] = agg_min_input.value or 0.0
        state["agg_max"] = agg_max_input.value or 5.0
        _apply_filters()
        _render_current()

    def on_prev(_e=None):
        if state["pos"] > 0:
            state["pos"] -= 1
            _render_current()

    def on_next(_e=None):
        if state["pos"] < len(state["filtered"]) - 1:
            state["pos"] += 1
            _render_current()

    gen_select.on_value_change(on_gen_change)
    safety_select.on_value_change(on_safety_change)
    agg_min_input.on_value_change(on_agg_change)
    agg_max_input.on_value_change(on_agg_change)
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
