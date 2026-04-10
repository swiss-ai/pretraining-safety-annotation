"""Phase 3 dashboard pages: /phase3 and /phase3/<run_id>.

Thin presentation layer over `pipeline.phase3.rank` analytics. The runs
themselves are produced by `python -m pipeline.phase3 eval-{generators,judges}`
and live under `cfg.phase3.eval_dir`.
"""

from __future__ import annotations

import json

from nicegui import app, ui

from pipeline.config import load_config
from pipeline.dashboard import render_header
from pipeline.log import logger
from pipeline.phase3.eval_generators import _eval_root


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
            logger.warning("phase3 dashboard: skipping {}: {}", run_dir.name, e)
            continue
        runs.append({"run_id": run_dir.name, "path": str(run_dir), **meta})
    return runs


@ui.page("/phase3")
def phase3_runs_page() -> None:
    """List phase 3 eval runs."""
    viewer_id = app.storage.user.get("annotator_id", "")
    render_header(viewer_id, active_phase=3)

    runs = _list_runs()

    ui.label("Phase 3: Eval Runs").classes("text-h5 q-pa-md")
    if not runs:
        with ui.column().classes("absolute-center items-center"):
            ui.label("No phase 3 eval runs yet.").classes("text-h6 text-grey-6")
            ui.label(
                "Run `uv run python -m pipeline.phase3 eval-generators` to start."
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
        lambda e: ui.navigate.to(f"/phase3/{e.args[1]['run_id']}"),
    )


@ui.page("/phase3/{run_id}")
def phase3_run_detail_page(run_id: str) -> None:
    """Show metadata + rank table for a single phase 3 eval run."""
    viewer_id = app.storage.user.get("annotator_id", "")
    render_header(viewer_id, active_phase=3)

    root = _eval_root(load_config()) / run_id
    meta_path = root / "metadata.json"
    if not meta_path.is_file():
        with ui.column().classes("absolute-center items-center"):
            ui.label(f"Unknown run: {run_id}").classes("text-h6 text-red")
        return
    meta = json.loads(meta_path.read_text())

    ui.label(f"Phase 3: {run_id}").classes("text-h5 q-pa-md")
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
    try:
        from pipeline.phase3 import rank as rank_mod

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
        logger.exception("phase3 dashboard rank failed")


def _render_generator_table(rows: list[dict]) -> None:
    if not rows:
        ui.label("No generators in this run.").classes("text-grey")
        return
    columns = [
        {"name": "generator", "label": "Generator", "field": "generator", "align": "left"},
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
    ui.table(columns=columns, rows=table_rows, row_key="generator").classes(
        "q-ma-md"
    )

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
            {"name": s, "label": s, "field": s, "align": "right"}
            for s in safety_scores
        ]
        matrix_rows = []
        for r in rows:
            row = {"generator": r["generator"]}
            buckets = r.get("accept_by_safety_score") or {}
            for s in safety_scores:
                b = buckets.get(s)
                row[s] = (
                    f"{b['accept_rate']:.0%} ({b['n']})" if b else "—"
                )
            matrix_rows.append(row)
        ui.table(columns=cols, rows=matrix_rows, row_key="generator").classes(
            "q-ma-md"
        )


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
